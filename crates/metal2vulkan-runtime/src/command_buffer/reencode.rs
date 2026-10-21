//! A committed command buffer's encoders encoded again at a rebind, before its first submission
//! (the logs are `replay.rs`'s).
//!
//! - **Marked under the rebind's lock.** Its commit notes the command buffer, with the keys of
//!   the buffers over host pages its replayable encoders named, while its encoders' holds still
//!   refuse a rebind of them (`Rebinds::wait_for_replay`). A rebind of one of them marks it
//!   (`reencode`) under the device's lock of those buffers, the lock its first submission reads
//!   the mark under (`State::first_submission`): that submission came before the rebind, and
//!   keeps the old pages (Metal's table 3), or it waits, held at its place in the scheduler's
//!   order, until the mark is gone.
//! - **Encoded again in the rebind's call.** The rebind then encodes again, on its own thread
//!   (the program's call, under no lock of the runtime's), each encoder of the command buffer a
//!   command of which named a buffer that lies elsewhere now than it took it, from its log, into
//!   new `VkCommandBuffer`s of the command buffer's pool, and puts what it recorded in the
//!   encoder's place: its recordings, uses, executions, keys and what its commands took (held
//!   weakly, as the commit holds them). The first recording is never submitted. No GPU work is
//!   waited for. Rebinds on two threads take turns (`reencoding`); a rebind during another's
//!   turn marks the command buffer again, and its own turn encodes what moved since. A host step
//!   of the command buffer, which records into its pool on the scheduler's thread, takes the
//!   same turns: Vulkan synchronizes a pool's recordings externally.
//! - **Work recorded on the GPU alone, recorded again.** Work its commit recorded on the GPU
//!   alone, for good (a GPU-written execution's preprocess and its dispatches or draws,
//!   `gpu_execute.rs`, `render/executions/gpu.rs`), resolved there what its commands named and
//!   the resident set its preprocess searches: the live buffers and what the encoders made
//!   resident, as they lay then. An encoder that left such work keeps its log as one that left
//!   none does, and its encoding again records that work at once, on the GPU alone, in segments
//!   of the same parts (`EncoderCore::finish_replay`), its own commands (a render pass's
//!   shadowed textures' copies and its other steps among them) recorded again around them. Its
//!   commit notes the command buffer for any rebind too (`Rebinds::wait_for_rerecord`): each
//!   rebind before the first submission records the work of the encoders it does not encode
//!   again in its place, as the commit did, over where the buffers lie now: its recording, what
//!   its commands took and what its host step needs. The commit notes it before it records that
//!   work, with its lock free: a rebind meanwhile marks it, and leaves the mark to the commit,
//!   which records the work again once it ended. Work a host step recorded again since
//!   (with the barriers between its commands, `gpu_execute::Rerecord`, or as a route outgrew
//!   it, `DeferredPass::outgrown`) keeps that step's recording.
//! - **What cannot be encoded again stays as before.** An encoder whose log is gone (a command
//!   predication's command buffer, failed at the commit) or whose second
//!   encoding is refused or records otherwise (`EncoderCore::finish_replay`), its work among
//!   them where it no longer runs on the GPU alone, keeps its first recording, which the first
//!   submission finds rebound: `CommandBufferError::EncodedUseRebound`, as before. So does work
//!   on the GPU alone the rebind no longer records there.

use super::rebound_use::Unlogged;
use super::{lock, CommandBuffer, CommandBufferShared, State};
use crate::buffer::HostPages;
use crate::deferred::{Deferred, StepInputs};
use crate::device::functions::Vk;
use crate::encoder::{RecordedSegment, Replayed, ReplayedPart};
use crate::replay::Replay;
use crate::residency::Residency;
use crate::resource::ResourceKey;
use crate::used_resources::EncoderKind;
use std::sync::atomic::Ordering;
use std::sync::Arc;

impl CommandBufferShared {
    /// A rebind since its commit, of a buffer one of its replayable encoders named, has them
    /// to encode again ([`reencode`](Self::reencode)) before its first submission: marked under
    /// the device's lock of its buffers over host pages (`Rebinds::replace`).
    pub(crate) fn mark_reencode(&self) {
        self.reencode.fetch_add(1, Ordering::AcqRel);
    }

    /// Whether its first submission waits for its encoders encoded again (the module docs).
    pub(crate) fn reencode_pending(&self) -> bool {
        self.reencode.load(Ordering::Acquire) > 0
    }

    /// Encode again each of its encoders that a command of named a buffer over host pages that
    /// lies elsewhere now than the command took it, and record again the work its commit
    /// recorded on the GPU alone (the module docs), on the rebind's thread, with no lock of the
    /// runtime's held; then its first submission may go, and the scheduler looks again. Nothing
    /// here fails the rebind: an encoder it cannot encode again, or work it cannot record on the
    /// GPU alone again, keeps its first recording, which its first submission fails.
    pub(crate) fn reencode(self: &Arc<Self>) {
        let turn = lock(&self.reencoding);
        let marked = self.reencode.load(Ordering::Acquire);
        if marked == 0 {
            return;
        }
        // Its commit records its work on the GPU alone with the lock free: the mark stays, and
        // the commit records that work again once it ended.
        if lock(&self.state).committing {
            return;
        }
        let device = &self.device;
        let moved: Vec<(usize, Arc<Replay>)> = {
            let state = lock(&self.state);
            match state.submitted {
                true => Vec::new(),
                false => {
                    let submitting = device.shared().rebinds.submitting();
                    state
                        .ended
                        .iter()
                        .enumerate()
                        .filter_map(|(at, record)| {
                            let replay = record.replay.as_ref()?;
                            submitting
                                .rebound(&record.taken)
                                .then(|| (at, Arc::clone(replay)))
                        })
                        .collect()
                }
            }
        };
        let handle = CommandBuffer(Arc::clone(self));
        // The encoders encoded again, whose segments the encoding again recorded.
        let mut spliced = Vec::new();
        for (at, replay) in moved {
            let residency = lock(&self.state).residency_before(at, replay.kind());
            let run = replay.run(&handle, residency);
            let counts = &device.shared().counts;
            let (replayed, residency) = match run {
                Ok(replayed) => replayed,
                Err(error) => {
                    counts.reencode_refused();
                    lock(&self.state).ended[at].refused = Some(error.to_string());
                    continue;
                }
            };
            // What a refused splice held goes after the lock.
            let vk = device.shared().vk();
            let spliced_now = lock(&self.state).splice_replay(vk, at, replayed, residency);
            match spliced_now {
                Ok(()) => {
                    counts.encoder_reencoded();
                    spliced.push(at);
                }
                Err(unspliced) => {
                    counts.reencode_refused();
                    let refusal = "its place holds other parts or work than its encoding again's";
                    lock(&self.state).ended[at].refused = Some(refusal.to_owned());
                    drop(unspliced);
                }
            }
        }
        // Its work recorded on the GPU alone in encoders not encoded again, each segment by its
        // part and its place there, which hold through this turn: only the first submission
        // places segments, and it waits for the turn; a host step, which records one, takes the
        // turn too.
        let alone: Vec<((usize, usize), Deferred)> = {
            let state = lock(&self.state);
            match state.submitted {
                true => Vec::new(),
                false => (state.recorded.iter().enumerate())
                    .filter(|(_, recorded)| !spliced.contains(&recorded.encoder))
                    .flat_map(|(part, recorded)| {
                        let segments = recorded.segments.iter().enumerate();
                        segments
                            .filter(|(_, segment)| segment.gpu_alone && segment.recorded.is_some())
                            .map(move |(at, segment)| ((part, at), segment.work.clone()))
                    })
                    .collect(),
            }
        };
        for (place, work) in alone {
            // Its notices were announced as it committed (`replay::is_replaying`), and its
            // commands take what they use without holding it.
            let again = crate::replay::replaying(|| work.record_on_gpu(&handle));
            let counts = &device.shared().counts;
            let Ok(Some((recorded, inputs))) = again else {
                counts.reencode_refused();
                continue;
            };
            let vk = device.shared().vk();
            let put = lock(&self.state).splice_segment(vk, place, recorded, inputs);
            match put {
                Ok(()) => counts.segment_recorded_again(),
                Err(unspliced) => {
                    counts.reencode_refused();
                    drop(unspliced);
                }
            }
        }
        drop(handle);
        // Its first submission may go, unless a rebind marked it meanwhile: that rebind's turn
        // comes next and encodes what moved since.
        let _ = self
            .reencode
            .compare_exchange(marked, 0, Ordering::AcqRel, Ordering::Acquire);
        drop(turn);
        device.shared().scheduler.wake();
    }
}

impl State {
    /// The residency an ended encoder `at` of `kind` started with, as the command buffer's
    /// encoders ended before it leave it (`CommandBuffer::inherited_residency`): none for a blit
    /// encoder, which makes nothing resident.
    pub(super) fn residency_before(&self, at: usize, kind: EncoderKind) -> Residency {
        if kind == EncoderKind::Blit {
            return Residency::default();
        }
        let mut joined = Residency::default();
        for record in self.ended.iter().take(at) {
            if let Some(residency) = &record.residency {
                joined.join_encoder(residency, record.kind == EncoderKind::Compute);
            }
        }
        // The textures made in its heaps since, as the commit's sync made them resident.
        drop(joined.sync_heaps());
        joined.shadows(kind == EncoderKind::Render)
    }

    /// Put `replayed`, ended encoder `at` encoded again, and what it made resident in its place,
    /// before the first submission: each of its parts' recordings, uses and segments (recorded
    /// on the GPU alone), its executions (the later encoders' and segments' places among them
    /// moved by the difference), its keys, and what its commands and its segments' took (held
    /// weakly until the first submission). What its first recording retained stays retained,
    /// and so does what its first segments recorded and what their host steps needed; its
    /// `VkCommandBuffer`s go with the pool unsubmitted. Refused, changing nothing and giving
    /// `replayed` back, where its place holds other parts than the encoding again's, or a segment
    /// but a pending dispatch's (which goes, the encoding again having recorded it at once) or
    /// work its commit recorded on the GPU alone, or the command buffer was submitted. Its
    /// replays hold what its commands took with `device`, the device's functions
    /// (`Replays::hold`).
    pub(super) fn splice_replay(
        &mut self,
        device: &Vk,
        at: usize,
        replayed: Replayed,
        residency: Option<Residency>,
    ) -> Result<(), Box<(Replayed, Option<Residency>)>> {
        let parts: Vec<usize> = (self.recorded.iter().enumerate())
            .filter(|(_, part)| part.encoder == at)
            .map(|(index, _)| index)
            .collect();
        let replaces = |part: &usize| {
            let recorded = &self.recorded[*part];
            recorded.sparse.is_empty()
                && (recorded.segments.iter()).all(|segment| segment.pends() || segment.gpu_alone)
        };
        let fits = !self.submitted
            && !parts.is_empty()
            && parts.len() == replayed.parts.len()
            && parts.iter().all(replaces)
            && !self.unrunnable.contains(&at)
            && self
                .ended
                .get(at)
                .is_some_and(|record| record.replay.is_some());
        if !fits {
            return Err(Box::new((replayed, residency)));
        }
        let Replayed {
            parts: mut replayed_parts,
            held,
            retained,
            executions,
            keys,
            mut taken,
            heaps,
        } = replayed;
        // Held weakly until the first submission, as the commit held what the first recording
        // and its segments took, and by its replays until they are replayed: what they held last
        // goes here, as it does without them (`Replays::hold`, with `device`, the device's
        // functions).
        let mut wait = |pages: &mut Vec<Arc<HostPages>>| {
            taken.wait(pages);
            let segments = (replayed_parts.iter_mut())
                .flat_map(|part| part.segments.iter_mut())
                .filter_map(|segment| segment.recorded.as_mut());
            for done in segments {
                done.taken.wait(pages);
            }
        };
        match self.replays.clone() {
            Some(replays) => drop(replays.hold(device, wait)),
            None => wait(&mut Vec::new()),
        }
        let first = self.ended[at].executions.clone();
        for (&part, replayed) in parts.iter().zip(replayed_parts) {
            let ReplayedPart {
                raws,
                summary,
                segments,
            } = replayed;
            let recorded = &mut self.recorded[part];
            recorded.raws = raws;
            recorded.summary = summary;
            // Its first segments go: the encoding again recorded its pending dispatches at once,
            // and its work on the GPU alone in segments of its own. What the commit recorded of
            // one stays retained, never submitted, as the first recording does.
            for segment in recorded.segments.drain(..) {
                if let Some(mut before) = segment.recorded {
                    self.retained.extend(before.retained.into_entry());
                    before.taken.retain_into(&mut self.retained);
                }
                self.retained.push(Box::new((segment.work, segment.inputs)));
            }
            for mut segment in segments {
                segment.executions_at += first.start;
                recorded.segments.push(segment);
            }
        }
        // Its executions in its place; an encoder that met a detached buffer at its first
        // encoding may have recorded fewer.
        let count = executions.len();
        self.executions.splice(first.clone(), executions);
        if count != first.len() {
            let moved = |place: usize| place - first.end + first.start + count;
            for later in self.ended[at + 1..].iter_mut() {
                if later.replay.is_some() {
                    later.executions = moved(later.executions.start)..moved(later.executions.end);
                }
            }
            let last = parts.last().copied().unwrap_or_default();
            let segments = self.recorded[last + 1..]
                .iter_mut()
                .flat_map(|later| later.segments.iter_mut());
            for segment in segments {
                segment.executions_at = moved(segment.executions_at);
            }
        }
        let record = &mut self.ended[at];
        record.executions = first.start..first.start + count;
        record.keys = keys;
        record.held = held;
        record.taken = taken;
        if let Some(residency) = residency {
            if let Some(first) = record.residency.replace(residency) {
                self.retained.push(Box::new(first));
            }
        }
        self.retained.extend(retained.into_entry());
        self.retained.push(Box::new(heaps));
        Ok(())
    }

    /// Put `recorded`, segment `at` of part `part` recorded again on the GPU alone, in its place
    /// before the first submission, with `inputs`, what its host step needs now: what its
    /// commands took held weakly until the first submission, as the commit holds them (by its
    /// replays too, with `device`, the device's functions: `Replays::hold`). What the first
    /// recording retained, its executions and its host step's inputs stay retained; its
    /// `VkCommandBuffer`s go with the pool unsubmitted, and what its commands took, held weakly,
    /// goes. Refused, changing nothing and giving `recorded` and `inputs` back, where the place
    /// holds no such segment, or the command buffer was submitted.
    pub(super) fn splice_segment(
        &mut self,
        device: &Vk,
        (part, at): (usize, usize),
        mut recorded: RecordedSegment,
        inputs: StepInputs,
    ) -> Result<(), Box<(RecordedSegment, StepInputs)>> {
        let fits = !self.submitted
            && (self.recorded.get(part))
                .and_then(|recorded| recorded.segments.get(at))
                .is_some_and(|segment| segment.gpu_alone && segment.recorded.is_some());
        if !fits {
            return Err(Box::new((recorded, inputs)));
        }
        match self.replays.clone() {
            Some(replays) => drop(replays.hold(device, |pages| recorded.taken.wait(pages))),
            None => recorded.taken.wait(&mut Vec::new()),
        }
        let segment = &mut self.recorded[part].segments[at];
        let first = segment.recorded.replace(recorded);
        let before = segment.inputs.replace(inputs);
        if let Some(first) = first {
            let RecordedSegment {
                retained,
                executions,
                ..
            } = first;
            self.retained.extend(retained.into_entry());
            self.retained.push(Box::new(executions));
        }
        if let Some(before) = before {
            self.retained.push(Box::new(before));
        }
        Ok(())
    }

    /// As it commits: the logs of encoders no rebind encodes again go, for the caller to let go
    /// of without this lock: every log where an encoder used command predication, and those of
    /// encoders the commit's failure fails (they run nothing). The keys the encoders whose logs
    /// stay named of buffers over host pages, sorted, each once (`Rebinds::wait_for_replay`).
    pub(super) fn keep_replays(&mut self) -> (Vec<Arc<Replay>>, Box<[ResourceKey]>) {
        let mut gone = Vec::new();
        let mut keys = Vec::new();
        let predicated = self.predicated;
        for (at, record) in self.ended.iter_mut().enumerate() {
            let fails = self
                .failed_at_commit
                .as_ref()
                .is_some_and(|failed| failed.fails(at));
            if predicated || fails {
                if let Some(log) = record.replay.take() {
                    gone.push(log);
                    record.unlogged = Some(match predicated {
                        true => Unlogged::Predicated,
                        false => Unlogged::Fails,
                    });
                }
                continue;
            }
            if record.replay.is_some() {
                keys.extend(record.taken.named_pins());
            }
        }
        keys.sort_unstable();
        keys.dedup();
        (gone, keys.into_boxed_slice())
    }
}
