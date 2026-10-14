//! The device's own synchronous submissions: work the runtime does for itself outside any
//! command buffer (a host copy into or out of a texture, an explicit clear of a live buffer).
//! Each runs alone on the device's first `VkQueue`, and the caller waits for it, so it completed
//! (and the host saw it) before any batch that uses its result is submitted; every batch starts
//! with the barrier that makes such writes visible. Each first does the new resources' first GPU
//! work that waits (`texture/first_move.rs`), after the batches on other `VkQueue`s that did such
//! work: no new resource waits for one, so making one waits for no GPU work. That work goes in a
//! `VkSubmitInfo` of its own just before the work's, in its `vkQueueSubmit`, as a batch's does
//! (`command_buffer/submit.rs`): one kernel submission on Mesa's drivers.
//!
//! The one submission no one waits for is a no-copy buffer's sparse binding of its dma-bufs
//! (`dma_buf.rs`), on the same queue and timeline: it would wait behind the queue's work in
//! flight. Every submission after it, on every `VkQueue`, waits for it until the host saw it
//! complete, and the buffer's release waits for it.

use super::relay::By;
use super::{lock, Device};
use crate::error::{vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::release;
use crate::resource::ResourceKey;
use ash::vk;
use std::time::Duration;

/// How long one wait for an internal submission lasts before it is repeated.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The command pool, command buffers and fence of internal submissions, made on first use.
pub(super) struct Internal {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    /// The command buffer of a submission's first work, in a `VkSubmitInfo` of its own just
    /// before the work's.
    first: vk::CommandBuffer,
    fence: vk::Fence,
    /// The timeline each submission signals, and the value the last one signalled: a batch on
    /// another `VkQueue` waits for one that did first work (`texture/first_move.rs`).
    timeline: vk::Semaphore,
    signalled: u64,
}

impl Internal {
    fn new(device: &Device) -> Result<Internal, Error> {
        let raw = device.shared().vk();
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(device.shared().queue_family);
        // SAFETY: the device is live and `pool_info` valid.
        let pool = unsafe { raw.create_command_pool(&pool_info, None) }.map_err(vulkan)?;
        let made = (|| {
            let allocate = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(2);
            let buffers = device.allocate_command_buffers(&allocate)?;
            let (buffer, first) = (buffers[0], buffers[1]);
            // SAFETY: the device is live.
            let fence = unsafe { raw.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(vulkan)?;
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
            // SAFETY: the device is live and enables `timelineSemaphore`.
            let timeline = match unsafe { raw.create_semaphore(&info, None) } {
                Ok(timeline) => timeline,
                Err(error) => {
                    // SAFETY: the fence is unused.
                    unsafe { release::object(raw, fence) };
                    return Err(vulkan(error));
                }
            };
            Ok(Internal {
                pool,
                buffer,
                first,
                fence,
                timeline,
                signalled: 0,
            })
        })();
        if made.is_err() {
            // SAFETY: nothing allocated from the pool is in use.
            unsafe { release::object(raw, pool) };
        }
        made
    }

    /// Destroy the objects; the device is idle.
    pub(super) fn destroy(&self, device: &ash::Device) {
        // SAFETY: the device is idle and nothing else holds these objects.
        unsafe {
            release::object(device, self.fence);
            release::object(device, self.timeline);
            release::object(device, self.pool);
        }
    }
}

impl Device {
    /// Record `record` into the device's internal command buffer, submit it alone, and wait for
    /// it. Internal submissions are serialized by the internal lock.
    pub(crate) fn submit_internal(
        &self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), Error> {
        self.submit_internal_naming(|_| false, record)
    }

    /// [`submit_internal`](Self::submit_internal), its first work (`texture/first_move.rs`)
    /// taking the moves of images bound to host pages whose key `named` names too (a texture's
    /// `VkImage` handed out, `Texture::vk_image`).
    pub(crate) fn submit_internal_naming(
        &self,
        named: impl Fn(ResourceKey) -> bool,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let shared = self.shared();
        let raw = shared.vk();
        let mut internal = lock(&shared.internal);
        if internal.is_none() {
            *internal = Some(Internal::new(self)?);
        }
        let Some(objects) = internal.as_mut() else {
            unreachable!("made above");
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // The new resources' first work, given back if the submission is not made, after every
        // batch on another `VkQueue` that did some. Its place in the submissions' order is held
        // until it was made.
        let firsts = &shared.textures.first_moves;
        let taken = firsts.take(named);
        let waits = firsts.waits(0, |point| {
            shared.queues[point.queue].completed_value() >= point.value
        });
        // The first work goes in a command buffer and a `VkSubmitInfo` of its own, just before
        // the work's, in its `vkQueueSubmit`: MoltenVK runs each `VkSubmitInfo` as a Metal
        // command buffer of its own, and lost a copy into a new image recorded in the same one as
        // the image's move (EVs STL-p4a-mvk4, STL-p4b-mvk4).
        let lead = taken.has_first();
        if lead {
            // SAFETY: as below, for the first work's command buffer.
            unsafe { raw.begin_command_buffer(objects.first, &begin) }.map_err(vulkan)?;
            // SAFETY: recording, outside a render pass, for the queue family; it is submitted
            // just before the work's submission, which is waited for before anything that uses
            // the resources is submitted after it.
            unsafe { taken.record(raw, objects.first) };
            // SAFETY: as above.
            unsafe { raw.end_command_buffer(objects.first) }.map_err(vulkan)?;
        }
        // SAFETY: the command buffer is not pending (every submission is waited for below) and
        // the internal lock is held; beginning it resets it (the pool resets command buffers).
        unsafe { raw.begin_command_buffer(objects.buffer, &begin) }.map_err(vulkan)?;
        // Host calls' staged copies, after the first work's `VkSubmitInfo`, before the work.
        // SAFETY: recording, outside a render pass; every image's move is in the first work's
        // submission or one before.
        unsafe { taken.record_staged(raw, objects.buffer) };
        record(raw, objects.buffer);
        // SAFETY: as above.
        unsafe { raw.end_command_buffer(objects.buffer) }.map_err(vulkan)?;
        // SAFETY: the fence is unsignaled or signaled and not pending.
        unsafe { raw.reset_fences(&[objects.fence]) }.map_err(vulkan)?;
        let buffers = [objects.buffer];
        // And the last binding of a no-copy buffer's dma-bufs, until it completed.
        let waits: Vec<(vk::Semaphore, u64)> = (waits.iter())
            .map(|point| (shared.queues[point.queue].timeline, point.value))
            .chain(self.binding_wait())
            .collect();
        let wait_semaphores: Vec<vk::Semaphore> = waits.iter().map(|wait| wait.0).collect();
        let wait_values: Vec<u64> = waits.iter().map(|wait| wait.1).collect();
        let wait_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        // It signals its own timeline, which a batch on another `VkQueue` waits for where it did
        // first work.
        let value = objects.signalled + 1;
        let signal_semaphores = [objects.timeline];
        let signal_values = [value];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        let first_buffers = [objects.first];
        // The first work waits as the work's does (a heap's fill another queue did comes before
        // a move of an image placed in it), and the work after it waits again: a timeline's wait
        // may repeat.
        let mut lead_timeline =
            vk::TimelineSemaphoreSubmitInfo::default().wait_semaphore_values(&wait_values);
        let lead_submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&first_buffers)
            .push_next(&mut lead_timeline);
        let submits: Vec<vk::SubmitInfo<'_>> = lead
            .then_some(lead_submit)
            .into_iter()
            .chain([submit])
            .collect();
        // SAFETY: the queue is used under its lock; the command buffers are executable; every
        // signal of the timeline is above its value. Where the queue's failures are relayed, a
        // failure of this submission loses the device (`relay.rs`).
        let made = shared.use_queue(0, By::Runtime, |queue| unsafe {
            let buffers = submits
                .iter()
                .map(|info| info.command_buffer_count as usize);
            shared.counts.queue_submitted(buffers.sum());
            raw.queue_submit(queue, &submits, objects.fence)
        });
        // A `vkQueueSubmit` that fails leaves what its submissions reference as it was, or loses
        // the device: the work goes back as `taken` drops.
        if let Err(error) = made {
            return Err(self.failed(error));
        }
        objects.signalled = value;
        shared.counts.internal_submission();
        // What it works on, held until it completed.
        let _held = taken.submitted_internally(objects.timeline, value);
        loop {
            // SAFETY: the fence was submitted just now.
            match unsafe {
                raw.wait_for_fences(&[objects.fence], true, WAIT_SLICE.as_nanos() as u64)
            } {
                Ok(()) => {
                    firsts.internal_completed(value);
                    return Ok(());
                }
                Err(vk::Result::TIMEOUT) => continue,
                Err(error) => return Err(self.failed(error)),
            }
        }
    }

    /// Bind `binds` to `buffer`, a sparse buffer no command has used, with one
    /// `vkQueueBindSparse` on the device's first `VkQueue` that signals the internal timeline,
    /// and return without waiting for it: every submission made after it, on every `VkQueue`,
    /// waits for it until the host saw it complete ([`Device::binding_wait`]). The timeline and
    /// the value it signals, which the buffer's release waits for.
    pub(crate) fn bind_sparse_buffer(
        &self,
        buffer: vk::Buffer,
        binds: &[vk::SparseMemoryBind],
    ) -> Result<(vk::Semaphore, u64), Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let shared = self.shared();
        let raw = shared.vk();
        let mut internal = lock(&shared.internal);
        if internal.is_none() {
            *internal = Some(Internal::new(self)?);
        }
        let Some(objects) = internal.as_mut() else {
            unreachable!("made above");
        };
        let buffer_binds = [vk::SparseBufferMemoryBindInfo::default()
            .buffer(buffer)
            .binds(binds)];
        let value = objects.signalled + 1;
        let signal_semaphores = [objects.timeline];
        let signal_values = [value];
        let mut timeline =
            vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);
        let info = vk::BindSparseInfo::default()
            .buffer_binds(&buffer_binds)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        // SAFETY: the buffer is live, made with `SPARSE_BINDING`, unused; every bind lies in it
        // and in its memory at its alignment, of a type it takes (`dma_buf.rs`); the queue's
        // family binds sparse memory, and the queue is used under its lock; the signal is above
        // the timeline's value (the internal lock is held).
        shared.counts.queue_submitted(0);
        let bound = shared.use_queue(0, By::Runtime, |queue| unsafe {
            raw.queue_bind_sparse(queue, &[info], vk::Fence::null())
        });
        exit_guard::called(Call::QueueBindSparse);
        bound.map_err(|error| self.failed(error))?;
        objects.signalled = value;
        *lock(&shared.bound_at) = Some((objects.timeline, value));
        Ok((objects.timeline, value))
    }

    /// The last binding of a no-copy buffer's dma-bufs ([`Device::bind_sparse_buffer`]) while
    /// the host has not seen it complete: what a submission waits for, as the buffer may be
    /// among what it uses.
    pub(crate) fn binding_wait(&self) -> Option<(vk::Semaphore, u64)> {
        let shared = self.shared();
        let mut bound_at = lock(&shared.bound_at);
        let (timeline, value) = (*bound_at)?;
        // SAFETY: the timeline is the device's, live while the device is.
        let reached = unsafe { shared.vk().get_semaphore_counter_value(timeline) };
        if reached.is_ok_and(|reached| reached >= value) {
            *bound_at = None;
            return None;
        }
        Some((timeline, value))
    }

    /// The error of a failed submission or wait, recording a lost device.
    pub(crate) fn failed(&self, error: vk::Result) -> Error {
        if error == vk::Result::ERROR_DEVICE_LOST {
            self.mark_lost();
        }
        vulkan(error)
    }
}
