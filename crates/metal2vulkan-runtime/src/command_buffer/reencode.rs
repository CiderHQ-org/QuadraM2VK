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
//! - **What cannot be encoded again stays as before.** An encoder whose log is gone (a command
//!   predication's command buffer, failed at the commit) or whose second
//!   encoding is refused or records otherwise (`EncoderCore::finish_replay`) keeps its first
//!   recording, which the first submission finds rebound: `CommandBufferError::EncodedUseRebound`,
//!   as before. So does any encoder of a kind that logs nothing (a render encoder's).

use super::{lock, CommandBuffer, CommandBufferShared, State};
use crate::device::functions::Vk;
use crate::encoder::{Replayed, Segment};
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
    /// lies elsewhere now than the command took it (the module docs), on the rebind's thread,
    /// with no lock of the runtime's held; then its first submission may go, and the scheduler
    /// looks again. Nothing here fails the rebind: an encoder it cannot encode again keeps its
    /// first recording, which its first submission fails.
    pub(crate) fn reencode(self: &Arc<Self>) {
        let turn = lock(&self.reencoding);
        let marked = self.reencode.load(Ordering::Acquire);
        if marked == 0 {
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
        for (at, replay) in moved {
            let residency = lock(&self.state).residency_before(at, replay.kind());
            let run = replay.run(&handle, residency);
            let counts = &device.shared().counts;
            let Ok((replayed, residency)) = run else {
                counts.reencode_refused();
                continue;
            };
            // What a refused splice held goes after the lock.
            let vk = device.shared().vk();
            let spliced = lock(&self.state).splice_replay(vk, at, replayed, residency);
            match spliced {
                Ok(()) => counts.encoder_reencoded(),
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
    /// before the first submission: its one part's recordings and uses, its executions (the
    /// later encoders' and segments' places among them moved by the difference), its keys, and
    /// what its commands took (held weakly until the first submission). What its first
    /// recording retained stays retained; its `VkCommandBuffer`s go with the pool unsubmitted.
    /// Refused, changing nothing and giving `replayed` back, where its place is not one part
    /// without segments but pending dispatches' (which go, the encoding again having recorded them
    /// at once), or the command buffer was submitted. Its replays hold what its commands took
    /// with `device`, the device's functions (`Replays::hold`).
    pub(super) fn splice_replay(
        &mut self,
        device: &Vk,
        at: usize,
        replayed: Replayed,
        residency: Option<Residency>,
    ) -> Result<(), Box<(Replayed, Option<Residency>)>> {
        let mut parts = self
            .recorded
            .iter()
            .enumerate()
            .filter(|(_, part)| part.encoder == at)
            .map(|(index, _)| index);
        let part = match (parts.next(), parts.next()) {
            (Some(part), None) => part,
            _ => return Err(Box::new((replayed, residency))),
        };
        let fits = !self.submitted
            && !self.unrunnable.contains(&at)
            && self.recorded[part].segments.iter().all(Segment::pends)
            && self.recorded[part].sparse.is_empty()
            && self
                .ended
                .get(at)
                .is_some_and(|record| record.replay.is_some());
        if !fits {
            return Err(Box::new((replayed, residency)));
        }
        let Replayed {
            raws,
            summary,
            held,
            retained,
            executions,
            keys,
            mut taken,
            heaps,
        } = replayed;
        // Held weakly until the first submission, as the commit held what the first recording
        // took, and by its replays until they are replayed: what they held last goes here, as
        // it does without them (`Replays::hold`, with `device`, the device's functions).
        match self.replays.clone() {
            Some(replays) => drop(replays.hold(device, |pages| taken.wait(pages))),
            None => taken.wait(&mut Vec::new()),
        }
        let recorded = &mut self.recorded[part];
        recorded.raws = raws;
        recorded.summary = summary;
        // Its pending dispatches' segments go: the encoding again recorded those dispatches at
        // once. What the commit recorded of one stays retained, never submitted, as the first
        // recording does.
        for segment in recorded.segments.drain(..) {
            if let Some(mut before) = segment.recorded {
                self.retained.extend(before.retained.into_entry());
                before.taken.retain_into(&mut self.retained);
            }
            self.retained.push(Box::new(segment.work));
        }
        // Its executions in its place; an encoder that met a detached buffer at its first
        // encoding may have recorded fewer.
        let first = self.ended[at].executions.clone();
        let count = executions.len();
        self.executions.splice(first.clone(), executions);
        if count != first.len() {
            let moved = |place: usize| place - first.end + first.start + count;
            for later in self.ended[at + 1..].iter_mut() {
                if later.replay.is_some() {
                    later.executions = moved(later.executions.start)..moved(later.executions.end);
                }
            }
            let segments = self.recorded[part + 1..]
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
                gone.extend(record.replay.take());
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
