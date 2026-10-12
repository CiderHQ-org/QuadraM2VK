//! Submitting one batch of a command buffer, on the scheduler's thread, in its order, under its
//! lock.
//!
//! One `vkQueueSubmit` of two `VkSubmitInfo`s on the queue's `VkQueue`:
//! 1. The work: the batch's waits, Vulkan waits for its events' signal indices, for the caller's
//!    binary semaphores and the tracker's waits for other `VkQueue`s' timelines, then its encoders'
//!    `VkCommandBuffer`s (an encoder's segments among its own), each encoder after the barrier the
//!    tracker planned for it. The first barrier also makes what completed batches wrote visible
//!    ([`Barrier::after_completed`]).
//! 2. The end: the barrier that makes every write visible to the host ([`Barrier::to_host`]),
//!    then the signals: the events' kept signal indices, the caller's binary semaphores and the
//!    queue's timeline value. When an event's previous kept signal is pending on another
//!    `VkQueue`, this part waits for it, so the event's semaphore is signaled in index order
//!    without holding back the work.
//!
//! A batch with no work is one `VkSubmitInfo` of every wait, the end and the signals: a wait
//! orders only its own `VkSubmitInfo`'s commands, so a signal after a wait with nothing between
//! needs both in one.
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
//! completed.
//!
//! [`Device::new_texture_with_vk_image`]: crate::Device::new_texture_with_vk_image
//!
//! The submission of the batch that ends a command buffer also signals a fence of the device's
//! where its driver reads the command buffer's objects after its value
//! ([`fences`](super::fences)), at which the command buffer then finishes.
//!
//! **GPU times** (`gpu_time.rs`). In a timed command buffer, the first command buffer of its
//! first batch resets its slot of the device's timestamp pools and writes the start timestamp at
//! the top of the pipe, before every barrier and encoder; the end of its last batch writes the
//! end timestamp at the bottom of the pipe, after its barriers. The scheduler's host times as it begins the first
//! batch and ends the last submission are the kernel times.
//!
//! The tracker keeps the plan, and the events their kept signals, only once the submission is
//! made; a batch that is not submitted changes neither.
//!
//! The submission is a section of the exit guard ([`exit_guard`]). What it did is published
//! ([`CommandBufferShared::publish`]: the transition to `Scheduled`, which runs the scheduled
//! handlers and ends waits for scheduling, and the completion thread's watch) only after that
//! section ended. A section ends by registering the exit hook again when its calls may have
//! registered destructors, so an exit the program makes once it learned of the submission runs
//! the hook before them.
//!
//! The submission of a command buffer's last batch leaves its submission handlers to the submitting
//! thread, once the scheduler's lock is free; its transition to `Scheduled` waits for them.

use super::completion::Submitted;
use super::{lock, recording_error, submission_error, CommandBufferShared, State};
use crate::deferred::Deferred;
use crate::encoder::RecordedSegment;
use crate::error::CommandBufferError;
use crate::error::Error;
use crate::event_timeline::Signal;
use crate::exit_guard::{self, Call};
use crate::gpu_time;
use crate::release;
use crate::replay::Replay;
use crate::scheduler;
use crate::sparse::bind::{bind_batch, BindFailure, Bound};
use crate::sparse::encoder::SparseUpdate;
use crate::texture::wrapped::Transition;
use crate::texture::Texture;
use crate::tracker::{Barrier, EncoderUses, Point, Summary};
use ash::vk;
use metal2vulkan::bind::ReplayFaults;
use std::sync::Arc;

/// A batch submitted, which nothing has seen yet: for the scheduler to
/// [`publish`](CommandBufferShared::publish).
#[must_use = "a batch not published never finishes its command buffer"]
pub(crate) struct Submission {
    /// Its `VkQueue`.
    queue: usize,
    batch: Submitted,
    /// It ends a command buffer with submission handlers, which the submitting thread runs once
    /// the scheduler's lock is free ([`Submission::runs_handlers`]).
    handlers: bool,
    /// The logs its command buffer's first submission ended (`replay.rs`), which
    /// [`publish`](CommandBufferShared::publish) hands back for the scheduler to let go of
    /// without its lock: what they hold may be a no-copy buffer's last handle.
    logs: Vec<Arc<Replay>>,
}

impl Submission {
    /// Whether the submitting thread runs its command buffer's submission handlers
    /// ([`CommandBufferShared::run_submission_handlers`]) once it holds no lock: its transition
    /// to `Scheduled` waits for them.
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
        let first_batch = !state.submitted;
        if first_batch {
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
        let at = Point {
            queue,
            value: scheduler.values[queue] + 1,
        };
        for (slot, vk_queue) in shared.queues.iter().enumerate() {
            scheduler
                .tracker
                .completed(slot, vk_queue.completed_value());
        }
        if index == 0 {
            state.times.kernel_start = gpu_time::host_nanos();
        }
        let slot = state
            .times
            .slot
            .as_ref()
            .map(|slot| (slot.queries(), slot.first()));
        // The start timestamp goes into the first command buffer of the first batch, a barrier
        // one (its first barrier makes what completed batches wrote visible), or into its end
        // when it has no work.
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
        let mut lead = Some(Barrier::after_completed());
        let mut work = Vec::with_capacity(2 * (end - start) + 1);
        if entry.is_some() {
            let globals: Vec<Barrier> = lead.take().into_iter().chain(entry_barrier).collect();
            let stamps = Stamps {
                start: start_stamp.take(),
                end: None,
            };
            work.push(self.record_barriers(&mut state, &globals, &entries, stamps)?);
        }
        for (offset, barrier) in barriers.iter().enumerate() {
            let globals: Vec<Barrier> = lead.take().into_iter().chain(*barrier).collect();
            if !globals.is_empty() {
                let stamps = Stamps {
                    start: start_stamp.take(),
                    end: None,
                };
                work.push(self.record_barriers(&mut state, &globals, &[], stamps)?);
            }
            work.extend(&state.recorded[start + offset].raws);
        }
        let finale_barriers: Vec<Barrier> = exit_barrier
            .into_iter()
            .chain([Barrier::to_host()])
            .collect();
        let stamps = Stamps {
            start: start_stamp.take(),
            end: slot.filter(|_| last),
        };
        let finale = [self.record_barriers(&mut state, &finale_barriers, &exits, stamps)?];
        // The new resources' first work, before every command (`texture/first_move.rs`), given
        // back if the submission is not made. A move of an image bound to host pages goes only
        // with a batch whose encoders name its key, and so hold its pages.
        let firsts = &shared.textures.first_moves;
        let first_work = firsts.take(|key| {
            state.recorded[start..end].iter().any(|recorded| {
                recorded.summary.keys().any(|named| named == key)
                    || (state.ended.get(recorded.encoder))
                        .is_some_and(|ended| ended.keys.contains(&key))
            })
        });
        // The first work goes in a `vkQueueSubmit` of its own just before the batch's, under the
        // queue's lock: MoltenVK lost a copy into a new image submitted with the image's move.
        // Host calls' staged copies go at the batch's start.
        let lead = match first_work.has_first() {
            true => Some(self.record_raw(&mut state, |_, buffer| {
                // SAFETY: the command buffer is recording, outside a render pass; it is
                // submitted just before the batch, before any later one, and every batch on
                // another `VkQueue` waits for the batch (`FirstMoves::waits`).
                unsafe { first_work.record(raw, buffer) };
                Ok(())
            })?),
            false => None,
        };
        if first_work.has_staged() {
            let staged = self.record_raw(&mut state, |_, buffer| {
                // SAFETY: the command buffer is recording, outside a render pass; every image's
                // move is in the first work's submission or one before.
                unsafe { first_work.record_staged(raw, buffer) };
                Ok(())
            })?;
            work.insert(0, staged);
        }

        let mut wait_semaphores = Vec::new();
        let mut wait_values = Vec::new();
        let mut wait_stages = Vec::new();
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
            .zip(&wait_values)
            .zip(&wait_stages)
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
            order_semaphores.append(&mut wait_semaphores);
            order_values.append(&mut wait_values);
            order_stages.append(&mut wait_stages);
        }
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
        let mut work_timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
        let first = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&work)
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
        let infos = &infos[usize::from(single)..];
        let lead_buffers: Vec<vk::CommandBuffer> = lead.into_iter().collect();
        let mut lead_timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&lead_values);
        let lead_info = vk::SubmitInfo::default()
            .wait_semaphores(&lead_semaphores)
            .wait_dst_stage_mask(&lead_stages)
            .command_buffers(&lead_buffers)
            .push_next(&mut lead_timeline);
        let (mut lead_made, mut closing) = (false, None);
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
        // The first work's submission, with no fence, is the batch's lead: where the batch's
        // fails after it was made, a submission of no work signals a fence of its own once the
        // first work completed (the device's fences, inside the scheduler's section).
        let submitted = shared.submit_logged(queue, fence, &self.relayed, |vk_queue| unsafe {
            if !lead_buffers.is_empty() {
                raw.queue_submit(
                    vk_queue,
                    std::slice::from_ref(&lead_info),
                    vk::Fence::null(),
                )?;
                lead_made = true;
            }
            let made = raw.queue_submit(vk_queue, infos, fence.unwrap_or(vk::Fence::null()));
            if made.is_err() && lead_made {
                if let Ok(ending) = shared.fences.take(raw) {
                    match raw.queue_submit(vk_queue, &[], ending) {
                        Ok(()) => closing = Some(ending),
                        // SAFETY: the failed submission is not pending.
                        Err(_) => {
                            release::object(raw, ending);
                        }
                    }
                }
            }
            made
        });
        exit_guard::called(Call::QueueSubmit);
        if let Err(error) = submitted {
            if let Some(fence) = fence {
                // SAFETY: the failed submission is not pending; inside the section. Released,
                // not kept: Vulkan leaves a fence's state after a failed submission unsaid.
                unsafe { release::object(raw, fence) };
            }
            // The first work went: what it holds goes once it completed. Where nothing ends its
            // submission that a wait could see, the device is taken as lost.
            if lead_made {
                // The first work's command buffer is of this command buffer's pool, which goes
                // as the command buffer fails: where the device is lost, once the driver's own
                // completion of the first work ran (`Device::drain_queues`).
                let ended = match closing {
                    Some(ending) => {
                        // SAFETY: the fence was submitted just now; inside the section.
                        let waited = unsafe { raw.wait_for_fences(&[ending], true, u64::MAX) };
                        exit_guard::called(Call::WaitForFences);
                        // SAFETY: signaled or the device lost: not pending.
                        unsafe { release::object(raw, ending) };
                        waited.is_ok()
                    }
                    None => {
                        device.mark_lost();
                        false
                    }
                };
                if !ended {
                    device.drain_queues();
                }
                drop(first_work.submitted(None));
            }
            return Err(submission_error(device, error));
        }
        if last {
            state.times.kernel_end = gpu_time::host_nanos();
        }
        // Its submission handlers run on this thread once the scheduler's lock is free; its
        // transition to `Scheduled`, posted as it is published, waits for them.
        let handlers = last && !state.submission.is_empty();
        state.submitting = handlers;
        scheduler.tracker.apply(plan);
        let several = shared.queues.len() > 1;
        let held = first_work.submitted(several.then_some(at));
        if !held.is_empty() {
            state.retained.push(Box::new(held));
        }
        scheduler.values[queue] = at.value;
        shared.queues[queue].submitted(at.value);
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
        // No rebind encodes its encoders again once it was first submitted: their logs go with
        // this batch, for the scheduler to drop without its lock. A failure before here leaves
        // them to go with the command buffer.
        let logs: Vec<Arc<Replay>> = match first_batch {
            true => state
                .ended
                .iter_mut()
                .filter_map(|record| record.replay.take())
                .collect(),
            false => Vec::new(),
        };
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
            logs,
        }))
    }

    /// Publish a batch [`submit_batch`](Self::submit_batch) submitted, after the section that
    /// submitted it ended: when it ends the command buffer, its transition to `Scheduled`, then
    /// the completion thread's watch for it.
    /// The logs the command buffer's first submission ended come back, for the caller to let go
    /// of without its lock.
    pub(crate) fn publish(self: &Arc<Self>, submission: Submission) -> Vec<Arc<Replay>> {
        let Submission {
            queue, batch, logs, ..
        } = submission;
        if batch.last {
            // Before the completion thread can see it finish, so the queue's context runs its
            // transition to `Scheduled` first.
            self.post_scheduled();
        }
        self.device.shared().completion.watch(queue, batch);
        logs
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
        record(state, buffer).map_err(|error| recording_error(device, error))?;
        // SAFETY: as above.
        let ended = unsafe { raw.end_command_buffer(buffer) };
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
                raw,
                summary: EncoderUses::new(true).finish(),
                retained: Vec::new(),
                executions: Vec::new(),
                keys: Default::default(),
                taken: Default::default(),
                resident_misses: 0,
                faults: ReplayFaults::default(),
            });
        }
        Ok(())
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
            if let Some((queries, first)) = stamps.start {
                raw.cmd_reset_query_pool(buffer, queries, first, 2);
                exit_guard::called(Call::CmdResetQueryPool);
                let top = vk::PipelineStageFlags::TOP_OF_PIPE;
                raw.cmd_write_timestamp(buffer, top, queries, first);
                exit_guard::called(Call::CmdWriteTimestamp);
            }
            for barrier in barriers {
                barrier.record(raw, buffer);
                exit_guard::called(Call::CmdPipelineBarrier);
            }
            if !transitions.is_empty() {
                let images: Vec<vk::ImageMemoryBarrier<'_>> =
                    transitions.iter().map(Transition::barrier).collect();
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
            if let Some((queries, first)) = stamps.end {
                let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                raw.cmd_write_timestamp(buffer, bottom, queries, first + 1);
                exit_guard::called(Call::CmdWriteTimestamp);
            }
            let ended = raw.end_command_buffer(buffer);
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
/// timestamp pools, each the slot's pool and its first query: the start (after resetting the
/// slot) before its barriers, the end after them.
#[derive(Clone, Copy)]
struct Stamps {
    start: Option<(vk::QueryPool, u32)>,
    end: Option<(vk::QueryPool, u32)>,
}
