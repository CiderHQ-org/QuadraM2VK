//! A batch's barrier command buffers, recorded once and submitted again.
//!
//! Every batch with encoders submits a command buffer of barriers alone before its first encoder
//! (`Barrier::after_completed`, with the barrier the tracker planned for that encoder) and one
//! after its last (`Barrier::to_host`), and one before each later encoder the tracker planned a
//! barrier for (`submit.rs`): batch after batch, the same few lists of barriers. Each list's
//! command buffer is recorded once, kept by the scheduler, and submitted again by a later batch
//! once the batch that last carried it completed, its queue's timeline at that batch's value as
//! the host saw it: the same commands, in the same place of the same submission, and no command
//! buffer begun for them as a batch is submitted. ANV clears a command buffer's state as its pool
//! resets it and again as it begins it (Mesa 26.2.4, `anv_cmd_state_init`), for each of a
//! window server frame's 67 or more batches.
//!
//! - **Never pending twice.** A kept command buffer is recorded without
//!   `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT` and goes to one batch at a time: ANV chains a
//!   submission's command buffers into one batch buffer only where none has that flag
//!   (`anv_cmd_buffer_is_chainable`), and submits the rest one execbuffer each. The batch that
//!   takes one marks it with its own timeline value before it is submitted: a batch whose
//!   submission fails leaves that value to the queue's next batch, so the command buffer waits
//!   for that one instead.
//! - **Where.** On every `VkQueue` but one whose command buffers end at a fence
//!   (`fences::ends_at_fence`): MoltenVK reads a command buffer after its timeline signal and
//!   refuses one submitted again before its completed handler ran. A barrier command buffer that
//!   writes timestamps or moves the caller's images is recorded by its batch, as before.
//! - **Bounded.** At most [`LISTS`] lists of barriers per `VkQueue`, each with at most [`KEPT`]
//!   command buffers, as many as its batches in flight at once needed: past them a batch records
//!   its own into its command buffer's pool. They live in a command pool of the queue's family,
//!   made at the first, and go with the device once its queues are idle.
//! - **Inside the scheduler's section**: every call is in the exit guard's table.

use crate::error::{vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::release;
use crate::tracker::Barrier;
use crate::Device;
use ash::vk;
use std::collections::VecDeque;

/// The most lists of barriers a `VkQueue` keeps command buffers of.
pub(crate) const LISTS: usize = 8;

/// The most command buffers a `VkQueue` keeps of one list of barriers.
pub(crate) const KEPT: usize = 128;

/// The scheduler's kept barrier command buffers, by `VkQueue` (`scheduler::State::kept`).
#[derive(Default)]
pub(crate) struct KeptBarriers {
    queues: Vec<QueueKept>,
}

/// One `VkQueue`'s.
#[derive(Default)]
struct QueueKept {
    /// The pool they are made from; null until the first.
    pool: vk::CommandPool,
    lists: Vec<KeptList>,
}

/// The command buffers of one list of barriers, in the order of the batches that last took them.
struct KeptList {
    barriers: Vec<Barrier>,
    buffers: VecDeque<Kept>,
}

/// A kept command buffer, and the timeline value of the batch that last took it.
struct Kept {
    raw: vk::CommandBuffer,
    after: u64,
}

impl KeptBarriers {
    /// A command buffer of `device`'s `VkQueue` `queue` holding only `barriers` (each recorded
    /// as [`Barrier::record`] records it), for the batch that signals `at` on that queue's
    /// timeline: a kept one whose last batch completed, else one recorded now and kept. `None`
    /// past the bounds (the module docs), where the batch records its own.
    pub(crate) fn take(
        &mut self,
        device: &Device,
        queue: usize,
        barriers: &[Barrier],
        at: u64,
    ) -> Result<Option<vk::CommandBuffer>, Error> {
        let shared = device.shared();
        let completed = shared.queues[queue].completed_value();
        if self.queues.len() <= queue {
            self.queues.resize_with(queue + 1, QueueKept::default);
        }
        let kept = &mut self.queues[queue];
        let index = match kept.lists.iter().position(|list| list.barriers == barriers) {
            Some(index) => index,
            None if kept.lists.len() < LISTS => {
                kept.lists.push(KeptList {
                    barriers: barriers.to_vec(),
                    buffers: VecDeque::new(),
                });
                kept.lists.len() - 1
            }
            None => return Ok(None),
        };
        let list = &mut kept.lists[index];
        // The batches that took them did so in their timeline values' order: the first is the
        // one that may have completed.
        if list.buffers.front().is_some_and(|k| k.after <= completed) {
            if let Some(mut taken) = list.buffers.pop_front() {
                taken.after = at;
                let raw = taken.raw;
                list.buffers.push_back(taken);
                return Ok(Some(raw));
            }
        }
        if list.buffers.len() >= KEPT {
            return Ok(None);
        }
        let raw = shared.vk();
        if kept.pool == vk::CommandPool::null() {
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(shared.queue_family_of(queue));
            // SAFETY: the device is live and `info` valid.
            let pool = unsafe { raw.create_command_pool(&info, None) };
            exit_guard::called(Call::CreateCommandPool);
            kept.pool = pool.map_err(vulkan)?;
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(kept.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffer = device.allocate_command_buffers(&info)?[0];
        // Submitted again, one batch at a time: neither one-time nor simultaneous.
        let begin = vk::CommandBufferBeginInfo::default();
        // SAFETY: the command buffer was just allocated from the scheduler's pool, under its
        // lock, and only this call records it.
        unsafe {
            let begun = raw.begin_command_buffer(buffer, &begin);
            exit_guard::called(Call::BeginCommandBuffer);
            begun.map_err(vulkan)?;
            shared.counts.command_buffer_begun();
            for barrier in barriers {
                barrier.record(raw, buffer);
                exit_guard::called(Call::CmdPipelineBarrier);
            }
            let ended = raw.end_command_buffer(buffer);
            exit_guard::called(Call::EndCommandBuffer);
            ended.map_err(vulkan)?;
        }
        shared.counts.barrier_buffer_kept();
        let list = &mut self.queues[queue].lists[index];
        list.buffers.push_back(Kept {
            raw: buffer,
            after: at,
        });
        Ok(Some(buffer))
    }

    /// Release the pools, with every kept command buffer, as the device is torn down after its
    /// queues are idle.
    pub(crate) fn release(&mut self, raw: &ash::Device) {
        for kept in self.queues.drain(..) {
            if kept.pool != vk::CommandPool::null() {
                // SAFETY: the device's queues are idle: no kept command buffer is pending.
                unsafe { release::object(raw, kept.pool) };
            }
        }
    }
}
