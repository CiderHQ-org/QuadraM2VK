//! Submitting one batch of a command buffer, on the scheduler's thread, in its order, under its
//! lock.
//!
//! One `vkQueueSubmit` of two `VkSubmitInfo`s on the queue's `VkQueue`:
//! 1. The work: the batch's waits, Vulkan waits for its events' signal indices, for the caller's
//!    binary semaphores and the tracker's waits for other `VkQueue`s' timelines, then its encoders'
//!    `VkCommandBuffer`s (an encoder's segments among its own), each encoder after the barrier the
//!    tracker planned for it. What completed work wrote is made visible by a wait or a barrier,
//!    see below.
//! 2. The end: the barrier that makes every write visible to the host ([`Barrier::to_host`]),
//!    then the signals: the events' kept signal indices, the caller's binary semaphores and the
//!    queue's timeline value. When an event's previous kept signal is pending on another
//!    `VkQueue`, this part waits for it, so the event's semaphore is signaled in index order
//!    without holding back the work.
//!
//! **What completed work wrote.** The tracker forgets a batch once the host saw its queue's
//! timeline pass it, and the device's own submissions outside the batches (`device/submit.rs`,
//! `device/staging.rs`) it never tracks: their writes are ordered before a later batch by their
//! completion alone. The work waits for each such timeline, each `VkQueue`'s and the device's two
//! own, at the value the host saw it reach, where a batch of its `VkQueue` did not wait for that
//! value already: a wait that is satisfied as it is submitted. The signal made the writes
//! available ("The first access scope includes all memory access performed by the device",
//! semaphore signaling), and the wait makes them visible: the submission's "second
//! synchronization scope of each semaphore wait operation defined by this command includes every
//! command in the same batch that the wait operation is defined in, and all commands that occur
//! later in submission order", and the wait's "second access scope includes all memory access
//! performed by the device" (`vkQueueSubmit`, semaphore waiting). So one wait per value serves
//! every later batch of its `VkQueue`. A batch with nothing planned before its first encoder then
//! needs no command buffer before it, a timed command buffer's first batch included where its
//! first encoder wrote the start (**GPU times**, below): on an RTX 5080 and a Radeon 780M, nearly
//! every batch of a frame of draws, and of command buffers committed from several threads. A wait costs CPU in `vkQueueSubmit`: about
//! 1.1 us on ms01's Iris Xe (ANV), 1.6 us on a Radeon 780M (RADV) and 0.5 us on an RTX 5080,
//! where a command buffer of barriers cost about 12 us of the Iris Xe's GPU. Counted:
//! `WorkCounts::completed_waits`, `WorkCounts::bare_starts` (the batches with work that began
//! with no command buffer of barriers).
//!
//! A driver measured to stall its submissions on semaphore operations
//! (`DriverFact::SemaphoreOperationsStallSubmits`: NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU
//! and the RTX 3090, about 2 ms of CPU in one `vkQueueSubmit` in about every thousand waits and
//! signals) takes no such wait: each batch with work starts instead with a command buffer of
//! barriers whose first is [`Barrier::after_completed`], which makes what completed batches
//! wrote visible to every later command, and the staging queue's copies signal no timeline.
//! Where every batch waited, about twice as many of its submissions took that long as where
//! batches only signal.
//!
//! Each barrier goes in a command buffer of barriers alone, which the scheduler records once per
//! list of barriers and submits again once the batch that last carried it completed
//! ([`kept_barriers`](super::kept_barriers)); one that also writes a command buffer's timestamps
//! once per list and slot, submitted again only by a later holding of the slot; one that moves
//! the caller's images is recorded by its batch.
//!
//! **The end in the work's last command buffer.** An encoder's last `VkCommandBuffer` stays
//! recording after the encoder ended, as its command buffer's tail, until an encoder after it
//! joins its batch, its batch ends at encode time, or the command buffer commits
//! (`CommandBufferShared::close_tail`). Where its batch ended there, the end goes into it before
//! it ends: the barrier to the host, then, in the last batch of a timed command buffer, the end
//! timestamp, as the end's own command buffer holds them. Only where the batch signals no event
//! (a kept signal may need the end after a wait for another `VkQueue`'s, above), leaves its
//! submission no segment or sparse update (recorded at the submission, among or before the
//! work's command buffers), and names no texture over the caller's image that moves between
//! layouts (its move back is planned at the submission, before the end). The batch's submission
//! then has no end of its own, and the signals join the work's `VkSubmitInfo`: "The first
//! synchronization scope of each semaphore signal operation defined by this command includes
//! every command in the same batch that the signal operation is defined in, and all commands that
//! occur earlier in submission order" (`vkQueueSubmit`). Where that command buffer does not go
//! to the queue as recorded (its encoder runs nothing, or a rebind encoded it again), the end is
//! a command buffer of its own, as every other batch's. Each command buffer a submission holds
//! costs GPU time of its own: a probe of one 1x1 copy took 42.6 us of GPU time as three primary
//! command buffers (a start, the copy, an end) and 25.6 us as one on ms01's Iris Xe (ANV), 19.6
//! and 13.6 cold on a Radeon 780M (RADV), 15.4 to 17.4 and 7.9 to 9.0 on an RTX 5080. Counted:
//! `WorkCounts::finales_folded`, `WorkCounts::finales_apart`.
//!
//! A batch with no work is one `VkSubmitInfo` of every wait, the end and the signals: a wait
//! orders only its own `VkSubmitInfo`'s commands, and a signal follows commands, not waits
//! (`vkQueueSubmit`'s synchronization scopes), so a signal after a wait needs both in one, with
//! the end's barrier, a command, between. Where it has no wait, no image to move back and no
//! timestamp to write either (an empty command buffer's batch, one of signals alone), its end
//! records nothing and the `VkSubmitInfo` holds no `VkCommandBuffer`: none of its commands wrote
//! anything, every batch before it made its own writes visible to the host before its signals,
//! and its signals follow every command submitted before them. ANV on i915 lists every
//! `VkDeviceMemory` of the device, and its own buffer objects, in each kernel submission of
//! commands, and two objects in one of none (Mesa 26.2.4, `i915/anv_batch_chain.c`).
//!
//! **The caller's images** ([`Device::new_texture_with_vk_image`]). A batch whose encoders use a
//! wrapped image that is not at rest in `GENERAL` moves it there before its first encoder and back
//! to its home layout in the end, before the signals. The moves are planned with the encoders, as
//! writes of the whole image outside them
//! ([`WrappedImage::transition_use`](crate::texture::wrapped::WrappedImage::transition_use)),
//! so the tracker orders them against every other use of the image on any queue, and each is
//! recorded as an image memory barrier from all commands before to all commands after.
//!
//! **New resources' first work** (`texture/first_move.rs`). Every batch first does the first GPU
//! work of the resources made since the device's last submission: each new image's move to
//! `GENERAL`, and the zeros a new texture, stand-in or buffer reads. It runs before every command
//! of the batch, whether a use names the resource or not (a command may reach a texture through its
//! resource ID, a buffer through its address), and the batch holds what it worked on until it
//! completes. On a device of several `VkQueue`s every batch on another one waits for it until it
//! completed. It is a `VkSubmitInfo` of its own, first in the batch's `vkQueueSubmit`, waiting for
//! what the work waits for but a binary semaphore: MoltenVK 1.4.1 runs each `VkSubmitInfo` as a
//! Metal command buffer of its own (`MVKQueue::submit`), and lost a copy into a new image
//! recorded in the same one as the image's move (EVs STL-p4a-mvk4, STL-p4b-mvk4); Mesa's drivers
//! merge it with the work, as it signals nothing (`vk_queue_submits_merge`), so it costs no
//! kernel submission of its own. A failed `vkQueueSubmit` made none of it, or lost the device.
//!
//! [`Device::new_texture_with_vk_image`]: crate::Device::new_texture_with_vk_image
//!
//! The submission of the batch that ends a command buffer also signals a fence of the device's
//! where its driver reads the command buffer's objects after its value
//! ([`fences`](super::fences)), at which the command buffer then finishes.
//!
//! **GPU times** (`gpu_time.rs`). In a timed command buffer, the start timestamp is written at
//! the top of the pipe, before every barrier and encoder of its first batch: by its first
//! encoder's first command buffer, which wrote it into its slot's third query as it opened, where
//! the batch plans nothing before that encoder (no barrier, no image moved, and no
//! [`Barrier::after_completed`] where the driver stalls on semaphore operations) and the command
//! buffer goes to the queue as recorded first (`WorkCounts::prefixed_starts`); else by the
//! batch's first command buffer, a command buffer of barriers, which resets the slot's first two
//! queries first. Where the batch plans a barrier before its first encoder for earlier work (the
//! encoder's own, or the one before the images it moves), the start is written after that
//! barrier instead, at the bottom of the pipe: once every command submitted before it on the
//! queue completed. Vulkan (`vkCmdWriteTimestamp`): its "first synchronization scope includes
//! all commands that occur earlier in submission order", and `VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT`
//! "is equivalent to `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` with `VkAccessFlags` set to `0` when
//! specified in the first synchronization scope". So a command buffer whose first encoder depends
//! on work still running starts after that work ended, as Metal's does: on an M3 Max under macOS
//! 27.0.1, a fill committed behind a running fill of the same bytes started 0.5 to 0.9 us after
//! the first's `GPUEndTime` (10 of 10), and one of other bytes while the first ran (probe
//! `gpustart`). The end of its last batch writes the end timestamp at the bottom of the pipe,
//! after its barriers. The scheduler's host times as it begins the first batch and ends the last
//! submission are the kernel times.
//!
//! The tracker keeps the plan, and the events their kept signals, only once the submission is
//! made; a batch that is not submitted changes neither.
//!
//! The submission is a section of the exit guard ([`exit_guard`]). What it did is published
//! ([`CommandBufferShared::publish`]: the transition to `Scheduled`, which runs the scheduled
//! handlers and ends waits for scheduling, and the queue waiter's watch) only after that
//! section ended. A section ends by registering the exit hook again when its calls may have
//! registered destructors, so an exit the program makes once it learned of the submission runs
//! the hook before them.
//!
//! The submission of a command buffer's last batch leaves its submission handlers to the submitting
//! thread, once the scheduler's lock is free; its transition to `Scheduled` waits for them.

use super::completion::Submitted;
use super::kept_barriers::{KeptBarriers, SlotStamps};
use super::{lock, recording_error, submission_error, CommandBufferShared, State};
use crate::deferred::Deferred;
use crate::encoder::RecordedSegment;
use crate::error::CommandBufferError;
use crate::error::{internal, Error};
use crate::event_timeline::Signal;
use crate::exit_guard::{self, Call};
use crate::gpu_time;
use crate::release;
use crate::resource::ResourceKey;
use crate::retained::Retained;
use crate::scheduler;
use crate::sparse::bind::{bind_batch, BindFailure, Bound};
use crate::sparse::encoder::SparseUpdate;
use crate::texture::decoded_twin::{SecondDecode, TwinWrite};
use crate::texture::wrapped::Transition;
use crate::texture::Texture;
use crate::tracker::{Barrier, EncoderUses, Point, Summary};
use ash::vk;
use metal2vulkan::bind::ReplayFaults;
use std::sync::Arc;

/// The lists a batch's submission fills and empties, kept in the scheduler's state from batch to
/// batch with their room, so that a submission allocates none of them once they have room for
/// what it puts in them ([`CommandBufferShared::submit_batch`]).
#[derive(Default)]
pub(crate) struct BatchLists {
    /// What each timeline whose writes the tracker forgets completed through as the host saw
    /// it, each `VkQueue`'s, then the device's own: none where it is at 0.
    reached: Vec<Option<(vk::Semaphore, u64)>>,
    /// The work's waits.
    waits: BatchWaits,
    /// Of `reached`, the timelines (by place) and values the work waits for.
    waited: Vec<(usize, u64)>,
}

/// A `VkSubmitInfo`'s waits: the semaphores, their values and their stages.
#[derive(Default)]
struct BatchWaits {
    semaphores: Vec<vk::Semaphore>,
    values: Vec<u64>,
    stages: Vec<vk::PipelineStageFlags>,
}

/// A batch submitted, which nothing has seen yet: for the scheduler to
/// [`publish`](CommandBufferShared::publish).
#[must_use = "a batch not published never finishes its command buffer"]
pub(crate) struct Submission {
    /// Its `VkQueue`.
    queue: usize,
    batch: Submitted,
    /// It ends a command buffer with submission handlers, or with a switch of storage forms riding
    /// on it, which the submitting thread runs or ends once the scheduler's lock is free
    /// ([`Submission::runs_handlers`]).
    handlers: bool,
}

impl Submission {
    /// Whether the submitting thread runs its command buffer's submission handlers
    /// ([`CommandBufferShared::run_submission_handlers`]) once it holds no lock, after it ended
    /// the switch of storage forms riding on it: its transition to `Scheduled` waits for them.
    pub(crate) fn runs_handlers(&self) -> bool {
        self.handlers
    }
}

impl CommandBufferShared {
    /// Submit batch `index` of the command buffer (`last` when it ends it) after the Vulkan waits
    /// `event_waits` for its events, against the scheduler's tracker and timeline values. The
    /// batch is for the caller to [`publish`](Self::publish) once its section ended.
    /// `Ok(None)`, nothing submitted or changed, while its first submission waits for encoders
    /// of it a rebind encodes again (`reencode.rs`): the caller holds the batch back at its place.
    pub(crate) fn submit_batch(
        self: &Arc<Self>,
        index: usize,
        last: bool,
        event_waits: &[(vk::Semaphore, u64)],
        scheduler: &mut scheduler::State,
    ) -> Result<Option<Submission>, CommandBufferError> {
        let device = &self.device;
        if device.is_lost() {
            return Err(CommandBufferError::DeviceLost);
        }
        let shared = device.shared();
        let raw = shared.vk();
        let queue = self.queue.vk_queue();
        let mut state = lock(&self.state);
        // A host step that failed fails the command buffer here; else the batch's segments go in
        // their places.
        if let Some(error) = state.step_failure.take() {
            return Err(error);
        }
        // Its first submission resolves what its commands reach as Metal's does, before any of
        // its batches goes to the `VkQueue` (`first_submission.rs`).
        if !state.submitted {
            let first = state
                .first_submission(device, self)
                .map_err(|error| recording_error(device, error))?;
            if !first {
                return Ok(None);
            }
        }
        // Each of Metal's submissions it found rebound, or holding a detached buffer, fails as
        // its own batch is submitted, after the batch's checks: one a check failed keeps the
        // check's code.
        let rebound = state.failing_at(device, index);
        let range = state.batch_range(index);
        // A batch that begins with resource state encoders binds their updates first, after its
        // waits, which its work then waits for through the bind.
        let updates: Vec<SparseUpdate> = state.recorded[range.clone()]
            .iter()
            .flat_map(|recorded| recorded.sparse.iter().cloned())
            .collect();
        // Where the updates leave work on emulated textures or tile maps and bind nothing, the
        // values the work waits for in the bind's place: every queue's last.
        let mut after_all = None;
        let bound = if updates.is_empty() {
            None
        } else {
            let binary = &state.batches.all()[index].binary_waits;
            let waits: Vec<(vk::Semaphore, u64)> = event_waits
                .iter()
                .copied()
                .chain(binary.iter().map(|&semaphore| (semaphore, 0)))
                .collect();
            let Bound {
                unbound,
                signalled,
                mappings,
            } = bind_batch(device, queue, &updates, &waits, &scheduler.values)
                .map_err(|failure| bind_error(device, failure))?;
            state.retained.push(Box::new(unbound));
            if !mappings.is_empty() {
                // The work's first command buffer, in the first resource state encoder's place,
                // which the tracker sees write what it writes.
                let uses = mappings
                    .uses()
                    .map_err(|error| recording_error(device, error))?;
                let raw = self.record_raw(&mut state, |state, buffer| {
                    // SAFETY: the command buffer is recording; the textures are live (the
                    // command buffer holds them), in `GENERAL`; the work waits for the bind, or
                    // for every queue's last value.
                    unsafe { mappings.record(device, &mut state.uploads, buffer) }
                })?;
                let first = &mut state.recorded[range.start];
                first.raws.push(raw);
                first.summary.join(Summary::outside_encoders(uses));
                if !signalled {
                    after_all = Some(scheduler.values.clone());
                }
            }
            signalled.then(|| {
                scheduler.values[queue] += 1;
                scheduler.values[queue]
            })
        };
        // The scrubs of emulated textures' writes, as mapped now, then every segment in its place.
        self.record_scrubs(&mut state, range.clone())?;
        state
            .place(range.clone())
            .map_err(|error| recording_error(device, error))?;
        // No work of an encoder that recorded nothing for a command (a buffer it named was
        // detached), nor of one whose submission failed rebound or detached, goes to the queue.
        state
            .runs_nothing_unrunnable(range.clone())
            .map_err(|error| recording_error(device, error))?;
        // The decodes into ASTC carriers' second decoded twins: those queued, before its work;
        // its blits' writes', after each part.
        self.record_second_decodes(&mut state, range.clone())?;
        let at = Point {
            queue,
            value: scheduler.values[queue] + 1,
        };
        // What each timeline whose writes the tracker forgets completed through as the host saw
        // it, each `VkQueue`'s, then the device's own (the module docs): read once, for the
        // tracker and for the work's waits alike.
        let BatchLists {
            reached,
            waits,
            waited,
        } = &mut scheduler.lists;
        reached.clear();
        for (slot, vk_queue) in shared.queues.iter().enumerate() {
            let value = vk_queue.completed_value();
            scheduler.tracker.completed(slot, value);
            reached.push((value > 0).then_some((vk_queue.timeline, value)));
        }
        reached.extend(device.own_completed());
        if index == 0 {
            state.times.kernel_start = gpu_time::host_nanos();
        }
        let slot = state
            .times
            .slot
            .as_ref()
            .map(|slot| (slot.queries(), slot.first(), slot.lease()));
        // The start timestamp goes into the first command buffer of the first batch, a barrier
        // one (with the barrier planned before its first encoder, if any), or into its end when
        // it has no work, unless its first encoder wrote it (below).
        let mut start_stamp = slot.filter(|_| index == 0);
        let batches = state.batches.all();
        let (start, end) = (range.start, range.end);
        let binary_waits = batches[index].binary_waits.clone();
        let binary_signals = batches[index].binary_signals.clone();
        let mut kept = Vec::new();
        let mut order_waits = Vec::new();
        for (event, value) in &batches[index].signals {
            if let Signal::Kept {
                index: signal,
                after,
            } = event.signal(*value, queue)
            {
                if let Some(after) = after {
                    order_waits.push((event.semaphore(), after));
                }
                kept.push((Arc::clone(event), signal, *value));
            }
        }
        let images = shared.interop.wrapped.among(
            state.recorded[start..end]
                .iter()
                .flat_map(|recorded| recorded.summary.keys()),
        );
        // The textures its encoders made for their own commands (`Device::new_planned_texture`),
        // which it holds until it completes.
        let own: Vec<Texture> = lock(&self.own_textures).drain(..).collect();
        if !own.is_empty() {
            state.retained.push(Box::new(own));
        }
        let encoders = &state.recorded[start..end];
        let mut entries = Vec::new();
        let mut exits = Vec::new();
        let (mut entry_uses, mut exit_uses) = (Vec::new(), Vec::new());
        for image in &images {
            let (entry, exit) = image.transitions();
            if let Some(entry) = entry {
                entries.push(entry);
                entry_uses.push(image.transition_use());
            }
            if let Some(exit) = exit {
                exits.push(exit);
                exit_uses.push(image.transition_use());
            }
        }
        // The new resources' first work, before every command (`texture/first_move.rs`), given
        // back if the submission is not made. A batch's commands may reach every texture
        // unnamed, so it takes every move, those of images bound to host pages among them, whose
        // pages it then holds. The moves that keep the bytes their images are bound to the host
        // makes first, once every submission before completed (`Device::keep_on_host`); one
        // registered meanwhile too.
        let firsts = &shared.textures.first_moves;
        let first_work = loop {
            (device.keep_on_host(|_| true)).map_err(|error| recording_error(device, error))?;
            let work = firsts.take(|_| true);
            if !work.has_keeps() {
                break work;
            }
        };
        let entry = (!entries.is_empty()).then(|| Summary::outside_encoders(entry_uses));
        let exit = (!exits.is_empty()).then(|| Summary::outside_encoders(exit_uses));
        let plan = scheduler.tracker.plan(
            entry
                .iter()
                .chain(encoders.iter().map(|recorded| &recorded.summary))
                .chain(&exit),
            at,
        );
        // One barrier per summary planned: the entry's, the encoders', the exit's.
        let mut barriers = plan.barriers();
        let mut entry_barrier = None;
        if let (Some(_), Some((&first, rest))) = (&entry, barriers.split_first()) {
            entry_barrier = first;
            barriers = rest;
        }
        let mut exit_barrier = None;
        if let (Some(_), Some((&last, rest))) = (&exit, barriers.split_last()) {
            exit_barrier = last;
            barriers = rest;
        }
        // Where the driver's submissions stall on semaphore operations, the first command buffer
        // of barriers makes what completed work wrote visible, in place of the work's waits (the
        // module docs).
        let by_barrier = shared.completed_by_barrier;
        let mut completed_barrier = by_barrier.then(Barrier::after_completed);
        // The first encoder's prefix wrote the start where the batch plans nothing before that
        // encoder (no barrier, no image moved, no barrier for completed work) and its first
        // command buffer goes to the queue as recorded (`gpu_time.rs`): then no command buffer of
        // barriers starts the batch for the timestamp's sake.
        let prefix = state.prefix;
        let prefixed = start_stamp.is_some()
            && completed_barrier.is_none()
            && entry.is_none()
            && barriers.first().is_some_and(Option::is_none)
            && prefix.is_some_and(|raw| {
                state
                    .recorded
                    .get(start)
                    .and_then(|first| first.raws.first())
                    == Some(&raw)
            });
        if prefixed {
            start_stamp = None;
        }
        if index == 0 {
            state.times.prefixed = prefixed;
        }
        let mut work = Vec::with_capacity(2 * (end - start) + 1);
        // Whether a command buffer of barriers goes before the first encoder.
        let mut started = false;
        // A barrier planned before the first encoder, for earlier work that may still run: the
        // start goes after the last such barrier, once every command before it completed (the
        // module docs).
        let encoder_behind = barriers.first().is_some_and(Option::is_some);
        if entry.is_some() {
            started = true;
            let globals: Vec<Barrier> = completed_barrier
                .take()
                .into_iter()
                .chain(entry_barrier)
                .collect();
            let stamps = Stamps {
                start: start_stamp.filter(|_| !encoder_behind),
                after: entry_barrier.is_some(),
                end: None,
            };
            if stamps.start.is_some() {
                start_stamp = None;
            }
            work.push(self.record_barriers(&mut state, &globals, &entries, stamps)?);
        }
        for (offset, barrier) in barriers.iter().enumerate() {
            let globals: Vec<Barrier> = completed_barrier
                .take()
                .into_iter()
                .chain(*barrier)
                .collect();
            if !globals.is_empty() || start_stamp.is_some() {
                started |= offset == 0;
                let stamps = Stamps {
                    start: start_stamp.take(),
                    after: barrier.is_some(),
                    end: None,
                };
                let kept = &mut scheduler.kept;
                work.push(self.barrier_buffer(&mut state, kept, at, &globals, &[], stamps)?);
            }
            work.extend(&state.recorded[start + offset].raws);
        }
        // The first work goes in a `VkSubmitInfo` of its own just before the batch's, in its
        // `vkQueueSubmit` (the module docs). Host calls' staged copies go at the batch's start.
        let lead = match first_work.has_first() {
            true => Some(self.record_raw(&mut state, |_, buffer| {
                // SAFETY: the command buffer is recording, outside a render pass; it is
                // submitted just before the batch, before any later one, and every batch on
                // another `VkQueue` waits for the batch (`FirstMoves::waits`).
                unsafe { first_work.record(raw, buffer, shared.queue_flags_of(queue)) };
                Ok(())
            })?),
            false => None,
        };
        if first_work.has_staged() {
            let staged = self.record_raw(&mut state, |_, buffer| {
                // SAFETY: the command buffer is recording, outside a render pass; every image's
                // move is in the first work's `VkSubmitInfo` or a submission before.
                unsafe { first_work.record_staged(raw, buffer) };
                Ok(())
            })?;
            work.insert(0, staged);
        }
        let BatchWaits {
            semaphores: wait_semaphores,
            values: wait_values,
            stages: wait_stages,
        } = waits;
        wait_semaphores.clear();
        wait_values.clear();
        wait_stages.clear();
        if let Some(value) = bound {
            // The bind waited for the batch's waits; the work waits for the bind.
            wait_semaphores.push(shared.queues[queue].timeline);
            wait_values.push(value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        } else {
            for &(semaphore, value) in event_waits {
                wait_semaphores.push(semaphore);
                wait_values.push(value);
                wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            for &semaphore in &binary_waits {
                // A binary semaphore's value is ignored.
                wait_semaphores.push(semaphore);
                wait_values.push(0);
                wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            for (slot, &value) in shared.queues.iter().zip(after_all.iter().flatten()) {
                if value > 0 {
                    wait_semaphores.push(slot.timeline);
                    wait_values.push(value);
                    wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
                }
            }
        }
        for wait in plan.waits() {
            wait_semaphores.push(shared.queues[wait.at.queue].timeline);
            wait_values.push(wait.at.value);
            wait_stages.push(wait.legacy_stages());
        }
        // Every batch on another `VkQueue` that did new resources' first work, until it
        // completed: any command may reach those resources unnamed.
        let completed = |point: Point| shared.queues[point.queue].completed_value() >= point.value;
        for point in firsts.waits(queue, completed) {
            wait_semaphores.push(shared.queues[point.queue].timeline);
            wait_values.push(point.value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        }
        // And the internal submission on the first `VkQueue` that did some, until it completed.
        if let Some((timeline, value)) = firsts.internal_wait().filter(|_| queue != 0) {
            wait_semaphores.push(timeline);
            wait_values.push(value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        }
        // And the last binding of a no-copy buffer's dma-bufs, on every `VkQueue`, the first
        // too (a binding is ordered with no later submission), until it completed.
        if let Some((timeline, value)) = device.binding_wait() {
            wait_semaphores.push(timeline);
            wait_values.push(value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        }
        // The first work waits for what the batch's commands wait for (a heap's fill another
        // queue did comes before a move of an image placed in it; an event the program waits
        // for before reusing a heap's memory), but a binary semaphore, which one wait consumes;
        // the batch's commands wait again (a timeline's wait may repeat).
        let (mut lead_semaphores, mut lead_values, mut lead_stages) =
            (Vec::new(), Vec::new(), Vec::new());
        for ((&semaphore, &value), &stage) in wait_semaphores
            .iter()
            .zip(wait_values.iter())
            .zip(wait_stages.iter())
            .filter(|((semaphore, _), _)| !binary_waits.contains(semaphore))
        {
            lead_semaphores.push(semaphore);
            lead_values.push(value);
            lead_stages.push(stage);
        }
        let mut order_semaphores: Vec<vk::Semaphore> =
            order_waits.iter().map(|&(s, _)| s).collect();
        let mut order_values: Vec<u64> = order_waits.iter().map(|&(_, value)| value).collect();
        let mut order_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; order_waits.len()];
        // No work: every wait goes with the end, which it then orders.
        let single = work.is_empty();
        if single {
            order_semaphores.append(wait_semaphores);
            order_values.append(wait_values);
            order_stages.append(wait_stages);
        }
        // The work waits for what completed, where its `VkQueue` did not already and no barrier
        // made it visible (the module docs); the waits are taken as made once the submission is.
        waited.clear();
        if !single && !by_barrier {
            for (source, &timeline) in reached.iter().enumerate() {
                let Some((semaphore, value)) = timeline else {
                    continue;
                };
                if value > scheduler.waited[queue][source] {
                    wait_semaphores.push(semaphore);
                    wait_values.push(value);
                    wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
                    waited.push((source, value));
                }
            }
        }
        // The end went into the work's last command buffer as the batch ended (the module docs),
        // where that command buffer goes to the queue as recorded: then the batch has no end of
        // its own. One that does not go (its encoder runs nothing, or was encoded again) leaves
        // the end to its own command buffer, as every other batch's.
        let folded = match state.fold(index).filter(|(raw, _)| work.contains(raw)) {
            None => false,
            Some((raw, stamped)) => {
                let ends = work.last() == Some(&raw)
                    && exit_barrier.is_none()
                    && exits.is_empty()
                    && order_semaphores.is_empty()
                    && start_stamp.is_none()
                    && stamped == (last && slot.is_some());
                debug_assert!(ends, "a batch's end recorded where the batch does not end");
                if !ends {
                    let error = internal("a batch's end recorded where the batch does not end");
                    return Err(recording_error(device, error));
                }
                true
            }
        };
        // The end, where the batch has something to end. A batch with no work, no wait, no image
        // to move back and no timestamp records none: none of its commands wrote anything, every
        // batch before it on any `VkQueue` made its own writes visible to the host before its
        // signals, and its signals follow every command submitted before them. A wait needs it:
        // a wait orders the commands of its batch, and a signal follows those, not the wait.
        let stamps = Stamps {
            start: start_stamp.take(),
            after: false,
            end: slot.filter(|_| last),
        };
        let ends = !single
            || !order_semaphores.is_empty()
            || exit_barrier.is_some()
            || !exits.is_empty()
            || stamps.start.is_some()
            || stamps.end.is_some();
        let finale: Vec<vk::CommandBuffer> = match ends && !folded {
            true => {
                let finale_barriers: Vec<Barrier> = exit_barrier
                    .into_iter()
                    .chain([Barrier::to_host()])
                    .collect();
                let kept = &mut scheduler.kept;
                vec![self.barrier_buffer(&mut state, kept, at, &finale_barriers, &exits, stamps)?]
            }
            false => Vec::new(),
        };
        let signal_semaphores: Vec<vk::Semaphore> = kept
            .iter()
            .map(|(event, _, _)| event.semaphore())
            .chain(binary_signals.iter().copied())
            .chain([shared.queues[queue].timeline])
            .collect();
        let signal_values: Vec<u64> = kept
            .iter()
            .map(|&(_, signal, _)| signal)
            .chain(binary_signals.iter().map(|_| 0))
            .chain([at.value])
            .collect();
        // A batch whose end its work holds signals with its work.
        let (work_signals, work_signal_values) = match folded {
            true => (&signal_semaphores[..], &signal_values[..]),
            false => (&[][..], &[][..]),
        };
        let mut work_timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(wait_values)
            .signal_semaphore_values(work_signal_values);
        let first = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(&work)
            .signal_semaphores(work_signals)
            .push_next(&mut work_timeline);
        // The fence the last batch signals where the driver reads a command buffer's objects
        // after its value, and every batch where the queue's failures are relayed
        // (`device/relay.rs`): taken before the submission, given back if it fails. Elsewhere
        // the batch takes none (`fences::ends_at_fence`).
        let fence = if (last || shared.flights(queue).is_some())
            && super::fences::ends_at_fence(device, queue)
        {
            // SAFETY: the device's fences, inside the scheduler's section.
            Some(
                unsafe { shared.fences.take(raw) }
                    .map_err(|error| submission_error(device, error))?,
            )
        } else {
            None
        };
        let mut end_timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&order_values)
            .signal_semaphore_values(&signal_values);
        let second = vk::SubmitInfo::default()
            .wait_semaphores(&order_semaphores)
            .wait_dst_stage_mask(&order_stages)
            .command_buffers(&finale)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut end_timeline);
        let infos = [first, second];
        let infos = match (single, folded) {
            (true, _) => &infos[1..],
            (false, true) => &infos[..1],
            (false, false) => &infos[..],
        };
        let lead_buffers: Vec<vk::CommandBuffer> = lead.into_iter().collect();
        let mut lead_timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&lead_values);
        let lead_info = vk::SubmitInfo::default()
            .wait_semaphores(&lead_semaphores)
            .wait_dst_stage_mask(&lead_stages)
            .command_buffers(&lead_buffers)
            .push_next(&mut lead_timeline);
        // The first work's `VkSubmitInfo` leads the batch's, in one `vkQueueSubmit`.
        let submits: Vec<vk::SubmitInfo<'_>> = (!lead_buffers.is_empty())
            .then_some(lead_info)
            .into_iter()
            .chain(infos.iter().copied())
            .collect();
        let buffers = (submits.iter())
            .map(|info| info.command_buffer_count as usize)
            .sum();
        // From here its heap set versions may be pending: nothing writes them any more, and its
        // first batch takes the ticket a texture that goes from now on waits for
        // (`heap_sets.rs`). One committed and not yet submitted holds no gone texture.
        if let Some(ticket) = state.heaps.submit(device) {
            state.retained.push(Box::new(ticket));
        }
        // SAFETY: every command buffer is executable and submitted once; every wait is for a
        // signal already submitted (the scheduler's order, and for a binary semaphore the
        // caller's contract of `encode_wait_for_vk_semaphore`); every signal of a timeline is
        // above its value and any earlier one (the queue's values rise, and an event's kept
        // indices rise and wait for the previous one on another queue), and every binary one is
        // of an unsignaled semaphore with no signal pending (`encode_signal_vk_semaphore`); every
        // transition is of a live image of the caller's in the layout it is in (its contract,
        // `new_texture_with_vk_image`); the fence is unsignaled and in no pending submission;
        // the `VkQueue` is used under its lock.
        let submitted = shared.submit_logged(
            (queue, Some(at.value)),
            fence,
            &self.relayed,
            |vk_queue| unsafe {
                shared.counts.queue_submitted(buffers);
                raw.queue_submit(vk_queue, &submits, fence.unwrap_or(vk::Fence::null()))
            },
        );
        exit_guard::called(Call::QueueSubmit);
        if let Err(error) = submitted {
            if let Some(fence) = fence {
                // SAFETY: the failed submission is not pending; inside the section. Released,
                // not kept: Vulkan leaves a fence's state after a failed submission unsaid.
                unsafe { release::object(raw, fence) };
            }
            // A `vkQueueSubmit` that fails leaves what its submissions reference as it was, or
            // loses the device: the first work goes back, still to be done, as `first_work`
            // drops. Where the device is lost, its command buffers, of this command buffer's pool
            // that goes as the command buffer fails, go once the driver's own completion of what
            // the call ran did (`Device::drain_queues`).
            let error = submission_error(device, error);
            if matches!(error, CommandBufferError::DeviceLost) {
                device.drain_queues();
            }
            return Err(error);
        }
        if last {
            state.times.kernel_end = gpu_time::host_nanos();
        }
        if !single {
            match folded {
                true => shared.counts.finale_folded(),
                false => shared.counts.finale_apart(),
            }
        }
        for &(source, value) in waited.iter() {
            scheduler.waited[queue][source] = value;
        }
        shared.counts.completed_waited(waited.len());
        if !single && !started {
            shared.counts.bare_start();
        }
        if prefixed {
            shared.counts.prefixed_start();
        }
        // Its submission handlers run on this thread once the scheduler's lock is free, and a
        // switch of storage forms riding on it ends there first (`scheduler/mod.rs`); its
        // transition to `Scheduled`, posted as it is published, waits for them.
        let handlers = last && (!state.submission.is_empty() || state.storage_form.is_some());
        state.submitting = handlers;
        scheduler.tracker.apply(plan);
        let several = shared.queues.len() > 1;
        let held = first_work.submitted(several.then_some(at));
        if !held.is_empty() {
            state.retained.push(Box::new(held));
        }
        scheduler.values[queue] = at.value;
        for image in &images {
            image.submitted();
            // One that rests in `GENERAL` from now on (an encode-time image after its first
            // batch) is moved by no later batch.
            if image.settled() {
                shared.interop.wrapped.forget(image);
            }
        }
        for (event, signal, value) in &kept {
            event.keep(*signal, *value, queue);
        }
        // Its commands may reach every texture through its resource ID where one bound a heap
        // set: the stale-reach degrade's trigger decides (`detach.rs`).
        let reaches = state.heaps.bound();
        drop(state);
        if rebound {
            device.degrade_encoded_use_rebound();
        }
        device.reach_stale_detached(reaches);
        Ok(Some(Submission {
            queue,
            batch: Submitted {
                value: at.value,
                buffer: self.hold(),
                last,
                fence,
                signals: kept
                    .into_iter()
                    .map(|(event, signal, _)| (event, signal))
                    .collect(),
            },
            handlers,
        }))
    }

    /// Publish a batch [`submit_batch`](Self::submit_batch) submitted, after the section that
    /// submitted it ended: when it ends the command buffer, its transition to `Scheduled`, then
    /// its queue waiter's watch for it. From the watch on, the command buffer may complete
    /// and run its release on its queue's handler thread before the caller lets go of anything
    /// it took, so the caller holds nothing of what its commands or encoders took
    /// (`scheduler::Spent`).
    pub(crate) fn publish(self: &Arc<Self>, submission: Submission) {
        let Submission { queue, batch, .. } = submission;
        if batch.last {
            // Before a completion turn can see it finish, so the queue's context runs its
            // transition to `Scheduled` first.
            self.post_scheduled();
        }
        self.device.shared().completion.watch(queue, batch);
    }

    /// A command buffer of this command buffer's pool holding what `record` records into it,
    /// with the state at hand.
    fn record_raw(
        &self,
        state: &mut State,
        record: impl FnOnce(&mut State, vk::CommandBuffer) -> Result<(), Error>,
    ) -> Result<vk::CommandBuffer, CommandBufferError> {
        let device = &self.device;
        let raw = device.shared().vk();
        let buffer = self
            .allocate_raw(state)
            .map_err(|error| recording_error(device, error))?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from the pool, under the state lock, and
        // only this call records it.
        let begun = unsafe { raw.begin_command_buffer(buffer, &begin) };
        exit_guard::called(Call::BeginCommandBuffer);
        begun.map_err(|error| submission_error(device, error))?;
        device.shared().counts.command_buffer_begun();
        record(state, buffer).map_err(|error| recording_error(device, error))?;
        // SAFETY: as above.
        let ended = unsafe { self.end_raw(buffer) };
        exit_guard::called(Call::EndCommandBuffer);
        ended.map_err(|error| submission_error(device, error))?;
        Ok(buffer)
    }

    /// Record the scrub segments of encoders `range` (`Deferred::SparseScrub`), which their batch's
    /// submission records with the mappings it applied.
    fn record_scrubs(
        &self,
        state: &mut State,
        range: std::ops::Range<usize>,
    ) -> Result<(), CommandBufferError> {
        let mut scrubs = Vec::new();
        for (encoder, recorded) in state.recorded[range.clone()].iter().enumerate() {
            for (at, segment) in recorded.segments.iter().enumerate() {
                if let (Deferred::SparseScrub(scrub), None) = (&segment.work, &segment.recorded) {
                    scrubs.push((range.start + encoder, at, Arc::clone(scrub)));
                }
            }
        }
        let device = &self.device;
        for (encoder, at, scrub) in scrubs {
            let raw = self.record_raw(state, |_, buffer| {
                // SAFETY: the command buffer is recording; the texture is live (its encoder's
                // command buffer holds it), in `GENERAL`, and its batch's updates applied.
                unsafe { scrub.record(device.shared().vk(), buffer) }
            })?;
            state.recorded[encoder].segments[at].recorded = Some(RecordedSegment {
                raws: vec![raw],
                summary: EncoderUses::new(true).finish(),
                retained: Retained::default(),
                executions: Vec::new(),
                keys: Default::default(),
                taken: Default::default(),
                resident_misses: 0,
                faults: ReplayFaults::default(),
            });
        }
        Ok(())
    }

    /// Record the decodes into ASTC carriers' second decoded twins (`texture/decoded_twin.rs`) of
    /// the batch of encoders `range`, a batch with work: those queued (as each twin was made,
    /// and after host calls wrote its carrier), in one command buffer before the first part's
    /// own, their uses joined to its summary, so the barrier and waits planned before that part
    /// order them after the earlier writes of their images; but those of an image a part of the
    /// batch wrote without a decode (the image had no second twin as the write was encoded), which
    /// stay queued for a later batch: a twin made since reads the carrier as this batch leaves it.
    /// Then after each part's own command buffers, the decodes its writes made as they were
    /// encoded, their uses joined to its summary. Nothing, and no command buffer, where there is
    /// none (counted: `WorkCounts::second_twin_checks`, `WorkCounts::second_twin_decodes`).
    fn record_second_decodes(
        &self,
        state: &mut State,
        range: std::ops::Range<usize>,
    ) -> Result<(), CommandBufferError> {
        let device = &self.device;
        let counts = &device.shared().counts;
        if range.is_empty() {
            return Ok(());
        }
        let held: Vec<ResourceKey> = state.recorded[range.clone()]
            .iter()
            .flat_map(|part| &part.twin_writes)
            .filter_map(TwinWrite::without_decode)
            .collect();
        let queued = device.take_second_decodes(|key| held.contains(&key));
        if !queued.is_empty() {
            let raw = self.record_raw(state, |_, buffer| {
                for decode in &queued {
                    // SAFETY: the command buffer is recording, outside a render pass, on the
                    // device's queue family; the decode is retained until the command buffer
                    // completes.
                    unsafe { decode.record(device.shared().vk(), buffer) };
                }
                Ok(())
            })?;
            let uses = queued.iter().flat_map(|decode| decode.uses()).cloned();
            let first = &mut state.recorded[range.start];
            first.raws.insert(0, raw);
            first
                .summary
                .join(Summary::outside_encoders(uses.collect()));
            counts.second_twin_decodes(queued.len());
            state.retained.push(Box::new(queued));
        }
        for part in range {
            let mut checked = 0;
            let mut decodes: Vec<SecondDecode> = Vec::new();
            for write in state.recorded[part].twin_writes.drain(..) {
                checked += 1;
                decodes.extend(write.into_decode());
            }
            counts.second_twin_checks(checked);
            if decodes.is_empty() {
                continue;
            }
            let raw = self.record_raw(state, |_, buffer| {
                for decode in &decodes {
                    // SAFETY: as above.
                    unsafe { decode.record(device.shared().vk(), buffer) };
                }
                Ok(())
            })?;
            let uses = decodes.iter().flat_map(|decode| decode.uses()).cloned();
            let recorded = &mut state.recorded[part];
            recorded.raws.push(raw);
            recorded
                .summary
                .join(Summary::outside_encoders(uses.collect()));
            counts.second_twin_decodes(decodes.len());
            state.retained.push(Box::new(decodes));
        }
        Ok(())
    }

    /// A command buffer holding only `barriers`, in order, then `transitions` of the caller's
    /// images, between the `stamps` it writes, for the batch that signals `at`: where it moves no
    /// image, one the scheduler keeps (`kept_barriers.rs`; one that writes timestamps, kept for
    /// their slot) on a queue whose command buffers end at their timeline's value
    /// (`fences::ends_at_fence`), and else, or past the kept ones' bounds, one recorded into this
    /// command buffer's pool ([`record_barriers`](Self::record_barriers)).
    fn barrier_buffer(
        &self,
        state: &mut State,
        kept: &mut KeptBarriers,
        at: Point,
        barriers: &[Barrier],
        transitions: &[Transition],
        stamps: Stamps,
    ) -> Result<vk::CommandBuffer, CommandBufferError> {
        let device = &self.device;
        if transitions.is_empty() && !super::fences::ends_at_fence(device, at.queue) {
            let taken = kept
                .take(device, at.queue, barriers, stamps.kept(), at.value)
                .map_err(|error| recording_error(device, error))?;
            if let Some(buffer) = taken {
                return Ok(buffer);
            }
        }
        self.record_barriers(state, barriers, transitions, stamps)
    }

    /// A command buffer of this command buffer's pool holding only `barriers`, in order, then
    /// `transitions` of the caller's images, from all commands before to all commands after,
    /// between the `stamps` it writes.
    fn record_barriers(
        &self,
        state: &mut State,
        barriers: &[Barrier],
        transitions: &[Transition],
        stamps: Stamps,
    ) -> Result<vk::CommandBuffer, CommandBufferError> {
        let device = &self.device;
        let raw = device.shared().vk();
        let buffer = self
            .allocate_raw(state)
            .map_err(|error| recording_error(device, error))?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from the pool, under the state lock, and
        // only this call records it.
        unsafe {
            let begun = raw.begin_command_buffer(buffer, &begin);
            exit_guard::called(Call::BeginCommandBuffer);
            begun.map_err(|error| submission_error(device, error))?;
            device.shared().counts.command_buffer_begun();
            if let Some((queries, first, _)) = stamps.start {
                raw.cmd_reset_query_pool(buffer, queries, first, 2);
                exit_guard::called(Call::CmdResetQueryPool);
                if !stamps.after {
                    let top = vk::PipelineStageFlags::TOP_OF_PIPE;
                    raw.cmd_write_timestamp(buffer, top, queries, first);
                    exit_guard::called(Call::CmdWriteTimestamp);
                }
            }
            for barrier in barriers {
                barrier.record(raw, buffer);
                exit_guard::called(Call::CmdPipelineBarrier);
            }
            if !transitions.is_empty() {
                let family = device.shared().queue_flags_of(self.queue.vk_queue());
                let images: Vec<vk::ImageMemoryBarrier<'_>> = (transitions.iter())
                    .map(|transition| transition.barrier(family))
                    .collect();
                let all = vk::PipelineStageFlags::ALL_COMMANDS;
                raw.cmd_pipeline_barrier(
                    buffer,
                    all,
                    all,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &images,
                );
                exit_guard::called(Call::CmdPipelineBarrier);
            }
            if let Some((queries, first, _)) = stamps.start.filter(|_| stamps.after) {
                let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                raw.cmd_write_timestamp(buffer, bottom, queries, first);
                exit_guard::called(Call::CmdWriteTimestamp);
            }
            if let Some((queries, first, _)) = stamps.end {
                let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                raw.cmd_write_timestamp(buffer, bottom, queries, first + 1);
                exit_guard::called(Call::CmdWriteTimestamp);
            }
            let ended = self.end_raw(buffer);
            exit_guard::called(Call::EndCommandBuffer);
            ended.map_err(|error| submission_error(device, error))?;
        }
        Ok(buffer)
    }
}

/// The command buffer's failure for a batch's sparse binding that failed: a heap filled inside a
/// device unit, the device's memory, or the bind itself.
fn bind_error(device: &crate::Device, failure: BindFailure) -> CommandBufferError {
    match failure {
        BindFailure::Split => CommandBufferError::SparseHeapSplit,
        BindFailure::Memory(error) => recording_error(device, error),
        BindFailure::Vulkan(error) => submission_error(device, error),
    }
}

/// The timestamps a barrier command buffer writes into its command buffer's slot of the device's
/// timestamp pools, each the slot's pool, its first query and the holding's lease
/// (`gpu_time::Slot::lease`): the start (after resetting the slot) before its barriers at the top
/// of the pipe, or with `after` after them at the bottom of the pipe; the end after them.
#[derive(Clone, Copy)]
struct Stamps {
    start: Option<(vk::QueryPool, u32, u64)>,
    after: bool,
    end: Option<(vk::QueryPool, u32, u64)>,
}

impl Stamps {
    /// What a kept command buffer writes for these (`kept_barriers.rs`): `None` where they are
    /// none.
    fn kept(self) -> Option<SlotStamps> {
        let (queries, first, lease) = self.start.or(self.end)?;
        Some(SlotStamps {
            queries,
            first,
            lease,
            start: self.start.is_some(),
            after: self.after,
            end: self.end.is_some(),
        })
    }
}
