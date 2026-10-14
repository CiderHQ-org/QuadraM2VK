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
//! - **Lists that went out of use give way.** A list met while [`LISTS`] are kept replaces the
//!   one taken least lately among those no batch took in the last [`STALE`] values of the queue's
//!   timeline and whose command buffers all completed, so the lists a program's first batches met
//!   do not hold the places for its life. Its command buffers are kept for the lists' later
//!   recordings, each recorded again from the start (the pool resets one at its begin,
//!   `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`). Lists all in use in turn, more than
//!   [`LISTS`] of them, replace none: the first kept stay kept and the rest are recorded by their
//!   batches, where replacing one at each use would record every list at each of its uses.
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

/// How many values of its queue's timeline a list no batch took stays kept for before a new list
/// may replace it: a few frames of a window server's (67 or more batches a frame).
pub(crate) const STALE: u64 = 256;

/// The scheduler's kept barrier command buffers, by `VkQueue` (`scheduler::State::kept`).
#[derive(Default)]
pub(crate) struct KeptBarriers {
    queues: Vec<QueueKept>,
}

/// One `VkQueue`'s.
#[derive(Default)]
pub(crate) struct QueueKept {
    /// The pool they are made from; null until the first.
    pool: vk::CommandPool,
    lists: Vec<KeptList>,
    /// The command buffers of the lists replaced, each completed, for later recordings.
    spare: Vec<vk::CommandBuffer>,
}

/// What a batch takes of a `VkQueue`'s kept command buffers ([`QueueKept::take`]).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Take {
    /// A kept command buffer of the list, whose last batch completed: the batch's now.
    Kept(vk::CommandBuffer),
    /// A command buffer to record for the list at this index, then kept
    /// ([`QueueKept::recorded`]): a spare one where there is one, else one allocated.
    Record(usize),
    /// None: past the bounds, the batch records its own.
    Past,
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

impl QueueKept {
    /// What the batch that signals `at` takes for `barriers`, its queue's timeline completed
    /// through `completed`.
    pub(crate) fn take(&mut self, barriers: &[Barrier], at: u64, completed: u64) -> Take {
        let index = match self.lists.iter().position(|list| list.barriers == barriers) {
            Some(index) => index,
            None if self.lists.len() < LISTS => {
                self.lists.push(KeptList {
                    barriers: barriers.to_vec(),
                    buffers: VecDeque::new(),
                });
                self.lists.len() - 1
            }
            None => match self.out_of_use(at, completed) {
                Some(index) => {
                    let list = &mut self.lists[index];
                    self.spare
                        .extend(list.buffers.drain(..).map(|kept| kept.raw));
                    list.barriers.clear();
                    list.barriers.extend_from_slice(barriers);
                    index
                }
                None => return Take::Past,
            },
        };
        let list = &mut self.lists[index];
        // The batches that took them did so in their timeline values' order: the first is the
        // one that may have completed.
        if list.buffers.front().is_some_and(|k| k.after <= completed) {
            if let Some(mut taken) = list.buffers.pop_front() {
                taken.after = at;
                let raw = taken.raw;
                list.buffers.push_back(taken);
                return Take::Kept(raw);
            }
        }
        if list.buffers.len() >= KEPT {
            return Take::Past;
        }
        Take::Record(index)
    }

    /// The list a new one replaces, its timeline at `at` and completed through `completed`: of
    /// those no batch took since `at - STALE` and whose command buffers all completed, the one
    /// taken least lately.
    fn out_of_use(&self, at: u64, completed: u64) -> Option<usize> {
        self.lists
            .iter()
            .enumerate()
            .filter_map(|(index, list)| {
                // The last batch that took one; the list's every buffer completed with it.
                let last = list.buffers.back().map_or(0, |kept| kept.after);
                (last <= completed && last.saturating_add(STALE) <= at).then_some((last, index))
            })
            .min()
            .map(|(_, index)| index)
    }

    /// A spare command buffer to record, where a replaced list left one.
    pub(crate) fn spare(&mut self) -> Option<vk::CommandBuffer> {
        self.spare.pop()
    }

    /// `raw`, recorded for the list at `list` ([`Take::Record`]), taken by the batch that signals
    /// `at`.
    pub(crate) fn recorded(&mut self, list: usize, raw: vk::CommandBuffer, at: u64) {
        self.lists[list].buffers.push_back(Kept { raw, after: at });
    }

    /// Each list's barriers and its command buffers in order, with the spare ones.
    #[cfg(test)]
    pub(crate) fn lists(
        &self,
    ) -> (
        Vec<(Vec<Barrier>, Vec<vk::CommandBuffer>)>,
        Vec<vk::CommandBuffer>,
    ) {
        let lists = self
            .lists
            .iter()
            .map(|list| {
                let buffers = list.buffers.iter().map(|kept| kept.raw).collect();
                (list.barriers.clone(), buffers)
            })
            .collect();
        (lists, self.spare.clone())
    }
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
        let list = match kept.take(barriers, at, completed) {
            Take::Kept(raw) => return Ok(Some(raw)),
            Take::Past => return Ok(None),
            Take::Record(list) => list,
        };
        let raw = shared.vk();
        if kept.pool == vk::CommandPool::null() {
            // Each command buffer is reset at its begin: a replaced list's are recorded again.
            let info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(shared.queue_family_of(queue));
            // SAFETY: the device is live and `info` valid.
            let pool = unsafe { raw.create_command_pool(&info, None) };
            exit_guard::called(Call::CreateCommandPool);
            kept.pool = pool.map_err(vulkan)?;
        }
        let buffer = match kept.spare() {
            Some(buffer) => buffer,
            None => {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(kept.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                device.allocate_command_buffers(&info)?[0]
            }
        };
        // Submitted again, one batch at a time: neither one-time nor simultaneous.
        let begin = vk::CommandBufferBeginInfo::default();
        // SAFETY: the command buffer is the scheduler pool's, used under its lock, and only this
        // call records it: just allocated, or a spare one, whose every batch completed (none is
        // pending), which its begin resets (the pool's `RESET_COMMAND_BUFFER`).
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
        self.queues[queue].recorded(list, buffer, at);
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
