//! A device's spare command pools and descriptor pools: a command buffer that completed gives back
//! the command pool its encoders recorded into, with the command buffers made from it, and the
//! descriptor pools its commands' sets came from (`descriptor_arena.rs`); the next command buffer
//! takes them where it would make its own. Made, used once and destroyed, a command pool cost 38
//! to 48 us and a descriptor pool 23 to 26 us on RADV (Radeon 780M), 7 to 8 and 3 on ANV, against
//! 0.2 and 0.02, and 1.3 and 0.1, reset (EV PERF-raw1).
//!
//! - **Command pools are kept, never released while the device lives**. Every command pool a
//!   command buffer gives back is kept, whatever its status, and every command buffer dropped with
//!   one gives it back too: the runtime calls `vkDestroyCommandPool` on a command buffer's pool
//!   only as the device is torn down, after its queues are idle.
//!   The spares stay bounded all the same: a command buffer makes a pool only where no spare of its
//!   queue's family is free, so the device holds at most as many as its command buffers held at
//!   once (Metal bounds those by each queue's `maxCommandBufferCount`), each holding what its
//!   largest command buffer recorded. Before, a pool given back past 8 spares, or one that made
//!   more than 64 command buffers, was released on the handler thread that completed its command
//!   buffer; on NVIDIA's driver 615.71.09 that `vkDestroyCommandPool` intermittently never returned
//!   (the driver spinning, with its own `[vkrt] Analysis` thread), so the command buffer never
//!   completed and its queue stalled.
//! - **Descriptor pools are bounded.** At most [`DESCRIPTOR_POOLS`] descriptor pools of at most
//!   [`DESCRIPTORS`] descriptors together; a descriptor pool given back past them releases the
//!   oldest spares first, so the spares follow the shapes in use.
//! - **A pool whose command buffers all ended is not reset; its command buffers are, as they are
//!   begun.** Its pools are made with `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`, and
//!   Vulkan: "If a command buffer is in the invalid, or executable state, and the command buffer
//!   was allocated from a command pool with the VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
//!   flag set, then vkBeginCommandBuffer implicitly resets the command buffer, behaving as if
//!   vkResetCommandBuffer had been called with VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT not
//!   set". A command buffer counts the command buffers of its pool it ended
//!   (`CommandBuffer::end_raw`, every `vkEndCommandBuffer` of one): where it completed and ended
//!   every one handed out, none is pending or recording, so each is executable or invalid. A
//!   pool given back otherwise may hold one still recording, which the validation layer forbids
//!   beginning (`VUID-vkBeginCommandBuffer-commandBuffer-00049`: "commandBuffer must not be in
//!   the recording or pending state"): by a command buffer that failed or was dropped
//!   unsubmitted (an encoder never ended, a recording an error cut short), or by one that
//!   completed with a recording it left unused, as its encoders may (a render pass instance's
//!   draws its pass replaced before any draw, a refusal past a begin, a segment's core dropped).
//!   Its taker resets that pool first, as every taker did before ([`CommandPool::recording`]),
//!   and counts it (`WorkCounts::spare_command_pools_reset`). ANV resets a command buffer at each begin whatever its pool
//!   (Mesa 26.2.4, `genX(BeginCommandBuffer)`), so a `vkResetCommandPool` before the begins reset
//!   each one twice: on ms01's Iris Xe, ready to record again, one command buffer took 8.0 to 20.5
//!   us of thread time begun alone against 12.1 to 25.1 us after the pool's reset, three 24.5 to
//!   50.3 against 36.6 to 63.2; the RTX 5080 (615.71.09) 0.13 to 0.24 us less; RADV (Radeon 780M)
//!   0.04 to 0.15 us less for one and 0.16 to 0.30 us more for three.
//! - **A pool given back holds a secondary command buffer.** One that made none makes one as it
//!   is given back, on the thread giving it back, inside a section of the exit guard. A pass
//!   whose first instance does not record inline begins its draws in a secondary command buffer
//!   as it opens (`render.rs`); taken from a pool that had served only passes recorded inline,
//!   it would be made there, on the thread that encodes: on the M3 Max (MoltenVK, macOS 27.0.1),
//!   0 to 5 of a run's 4020 pass opens made one, 16 allocations each against 14. A making that
//!   fails is left to that open.
//! - **Descriptor pools are reset by the taker**, on the thread that encodes: the program's own, or
//!   one running the program's code (a handler).
//! - **Taken outside sections alone.** A section of the exit guard (a host step's recording on the
//!   scheduler's thread) calls only the guard's table of entry points, which holds no reset of a
//!   pool, and takes no spare: a command buffer or command recorded there makes its
//!   pools as before; what it made is given back all the same. A command buffer that encoded
//!   nothing, whose submissions record only their barriers, in sections, takes its spare command
//!   pool at its commit, on the committing thread.
//! - **Given back once nothing of them is pending.** A command buffer that completed
//!   (`CommandBufferStatus::Completed`) gives back its command pool and its descriptor pools; one
//!   that failed, once its work ended, gives back its command pool and releases its descriptor
//!   pools; one dropped unsubmitted gives back its command pool.
//! - **Released with the device**, as its teardown begins (`DeviceShared`'s drop, after
//!   `vkDeviceWaitIdle`): every command buffer holds the device, so none is left to give one
//!   back.

use super::descriptor_arena::{Room, SetNeeds};
use super::device::{lock, DeviceShared};
use super::error::{vulkan, Error};
use super::exit_guard;
use super::release::{self, Released};
use ash::vk;
use std::collections::VecDeque;
use std::sync::Mutex;

/// The most spare descriptor pools a device keeps.
pub(super) const DESCRIPTOR_POOLS: usize = 32;

/// The most descriptors the spare descriptor pools hold together.
pub(super) const DESCRIPTORS: u64 = 1 << 18;

/// The index of `level`'s list in a [`CommandPool`]: primary 0, secondary 1.
fn level_index(level: vk::CommandBufferLevel) -> usize {
    usize::from(level == vk::CommandBufferLevel::SECONDARY)
}

/// A command pool and the command buffers made from it, by level: those a command buffer handed
/// out since it took the pool come first.
pub(super) struct CommandPool {
    pub(super) raw: vk::CommandPool,
    /// The queue family the pool was made for: its command buffers run on queues of it only.
    family: u32,
    made: [Vec<vk::CommandBuffer>; 2],
    handed: [usize; 2],
    /// One of its command buffers may still be recording: the command buffer that gave it back
    /// did not complete, or left one it was handed unended (the module docs), so its taker
    /// resets it.
    pub(super) recording: bool,
}

impl CommandPool {
    /// A new pool, made for queue family `family`, of no command buffer yet.
    pub(super) fn new(raw: vk::CommandPool, family: u32) -> CommandPool {
        CommandPool {
            raw,
            family,
            made: [Vec::new(), Vec::new()],
            handed: [0, 0],
            recording: false,
        }
    }

    /// A command buffer of `level` made from the pool before a command buffer last took it, not
    /// handed out since, in any state but pending (its begin resets it: the module docs); `None`
    /// once each was.
    pub(super) fn reuse(&mut self, level: vk::CommandBufferLevel) -> Option<vk::CommandBuffer> {
        let at = level_index(level);
        let buffer = self.made[at].get(self.handed[at]).copied()?;
        self.handed[at] += 1;
        Some(buffer)
    }

    /// The command buffers handed out since a command buffer last took it, of both levels.
    pub(super) fn handed_out(&self) -> usize {
        self.handed[0] + self.handed[1]
    }

    /// `buffer`, of `level`, was just made from the pool, and is handed out.
    pub(super) fn made(&mut self, level: vk::CommandBufferLevel, buffer: vk::CommandBuffer) {
        let at = level_index(level);
        self.made[at].push(buffer);
        self.handed[at] = self.made[at].len();
    }

    /// A command buffer took the pool, reset where it was [`recording`](Self::recording): none of
    /// the command buffers made from it is handed out, and none is recording.
    pub(super) fn taken(&mut self) {
        self.handed = [0, 0];
        self.recording = false;
    }

    /// Whether the pool made no secondary command buffer (the module docs).
    pub(super) fn wants_secondary(&self) -> bool {
        self.made[level_index(vk::CommandBufferLevel::SECONDARY)].is_empty()
    }

    /// `buffer`, a secondary command buffer just made from the pool, not handed out: the next
    /// command buffer that takes the pool hands it out first.
    pub(super) fn stock_secondary(&mut self, buffer: vk::CommandBuffer) {
        self.made[level_index(vk::CommandBufferLevel::SECONDARY)].push(buffer);
    }

    /// Make the pool a secondary command buffer where it [`wants_secondary`](Self::wants_secondary)
    /// one (the module docs): inside a section of the exit guard, on whatever thread gives the
    /// pool back; nothing where the guard is closing, or where the making fails.
    fn stock(&mut self, device: &DeviceShared) {
        if !self.wants_secondary() {
            return;
        }
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.raw)
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);
        if let Ok(made) = device.allocate_command_buffers(&info) {
            for buffer in made {
                self.stock_secondary(buffer);
            }
        }
    }
}

/// A descriptor pool, with the room it was made with.
pub(super) struct DescriptorPool {
    pub(super) raw: vk::DescriptorPool,
    pub(super) room: Room,
}

/// A device's spares.
#[derive(Default)]
pub(super) struct Spares {
    command_pools: Mutex<Vec<CommandPool>>,
    /// Oldest first.
    descriptor_pools: Mutex<VecDeque<DescriptorPool>>,
}

impl Spares {
    /// A spare command pool of `device` made for queue family `family`, none of whose command
    /// buffers is pending or recording (reset where one may be, the module docs); `None` when
    /// there is none, or this thread is in a section.
    pub(super) fn command_pool(
        &self,
        device: &DeviceShared,
        family: u32,
    ) -> Result<Option<CommandPool>, Error> {
        if exit_guard::in_section() {
            return Ok(None);
        }
        let Some(mut pool) = take_command_pool(&mut lock(&self.command_pools), family) else {
            return Ok(None);
        };
        if pool.recording {
            // SAFETY: the pool is the caller's alone now, and nothing of it is pending: the
            // command buffer that gave it back ended its work or was never submitted.
            let reset = unsafe {
                device
                    .vk()
                    .reset_command_pool(pool.raw, vk::CommandPoolResetFlags::empty())
            };
            if let Err(error) = reset {
                // SAFETY: as above.
                unsafe { release_in_section(device, pool.raw) };
                return Err(vulkan(error));
            }
            device.counts.spare_command_pool_reset();
        }
        pool.taken();
        device.counts.spare_command_pool_taken();
        Ok(Some(pool))
    }

    /// Keep `pool` of `device`, of a command buffer whose work ended or that was never
    /// submitted: nothing of it is pending, and nothing else holds it; `settled` where the
    /// command buffer completed and ended every command buffer of it handed out, so none is
    /// recording (the module docs). A pool that made no secondary command buffer makes one
    /// first. Never released here.
    pub(super) fn give_command_pool(
        &self,
        device: &DeviceShared,
        mut pool: CommandPool,
        settled: bool,
    ) {
        pool.recording = !settled;
        pool.stock(device);
        keep_command_pool(&mut lock(&self.command_pools), pool);
    }

    /// A spare descriptor pool of `device`, reset, with room for one set of `needs`, the newest
    /// such; `None` when there is none, or this thread is in a section.
    pub(super) fn descriptor_pool(
        &self,
        device: &DeviceShared,
        needs: &SetNeeds,
    ) -> Result<Option<DescriptorPool>, Error> {
        if exit_guard::in_section() {
            return Ok(None);
        }
        let pool = take_descriptor_pool(&mut lock(&self.descriptor_pools), needs);
        let Some(pool) = pool else {
            return Ok(None);
        };
        // SAFETY: the pool is the caller's alone now, and nothing uses its sets: a command buffer
        // that completed gave it back.
        let reset = unsafe {
            device
                .vk()
                .reset_descriptor_pool(pool.raw, vk::DescriptorPoolResetFlags::empty())
        };
        if let Err(error) = reset {
            // SAFETY: as above.
            unsafe { release_in_section(device, pool.raw) };
            return Err(vulkan(error));
        }
        Ok(Some(pool))
    }

    /// Keep `pool` of `device`, of a command buffer that completed, releasing the oldest spares
    /// past the bounds (or `pool` itself, when it alone is past them).
    pub(super) fn give_descriptor_pool(&self, device: &DeviceShared, pool: DescriptorPool) {
        let past = keep_descriptor_pool(&mut lock(&self.descriptor_pools), pool);
        for raw in past {
            // SAFETY: nothing uses the pool's sets (the caller's, or a spare's), and nothing else
            // holds it.
            unsafe { release_in_section(device, raw) };
        }
    }

    /// Release every spare of `device`: its teardown, when no command buffer is left.
    pub(super) fn release(&self, device: &DeviceShared) {
        for pool in lock(&self.command_pools).drain(..) {
            // SAFETY: no command buffer is left, so nothing of the pool is pending.
            unsafe { release_in_section(device, pool.raw) };
        }
        for pool in lock(&self.descriptor_pools).drain(..) {
            // SAFETY: as above.
            unsafe { release_in_section(device, pool.raw) };
        }
    }
}

/// Release `object` of `device` inside a section of the exit guard, which this thread may be
/// outside of (a command buffer completes on a handler thread): the device's functions are
/// reached inside it. Once the process is exiting, the object is left to its end.
///
/// # Safety
///
/// As [`release::object`]'s: nothing pending uses the object.
pub(super) unsafe fn release_in_section<T: Released>(device: &DeviceShared, object: T) {
    let Some(_busy) = exit_guard::enter() else {
        return;
    };
    // SAFETY: the caller's.
    unsafe { release::object(device.vk(), object) };
}

/// The newest of the spare command pools `pools` made for queue family `family`, taken out.
pub(super) fn take_command_pool(pools: &mut Vec<CommandPool>, family: u32) -> Option<CommandPool> {
    let at = pools.iter().rposition(|pool| pool.family == family)?;
    Some(pools.remove(at))
}

/// Keep `pool` among the spare command pools `pools`, newest last: every one, however many
/// command buffers it made and however many spares there are (the module docs).
pub(super) fn keep_command_pool(pools: &mut Vec<CommandPool>, pool: CommandPool) {
    pools.push(pool);
}

/// The newest of the spare descriptor pools `pools` (oldest first) with room for one set of
/// `needs`, taken out.
pub(super) fn take_descriptor_pool(
    pools: &mut VecDeque<DescriptorPool>,
    needs: &SetNeeds,
) -> Option<DescriptorPool> {
    let at = pools.iter().rposition(|pool| pool.room.fits(needs))?;
    pools.remove(at)
}

/// Keep `pool` among the spare descriptor pools `pools` (oldest first): the pools to release, the
/// oldest while `pools` holds more than [`DESCRIPTOR_POOLS`] or [`DESCRIPTORS`], or `pool` itself
/// when it alone holds more descriptors than that.
pub(super) fn keep_descriptor_pool(
    pools: &mut VecDeque<DescriptorPool>,
    pool: DescriptorPool,
) -> Vec<vk::DescriptorPool> {
    if pool.room.descriptors() > DESCRIPTORS {
        return vec![pool.raw];
    }
    pools.push_back(pool);
    let mut past = Vec::new();
    let mut held: u64 = pools.iter().map(|pool| pool.room.descriptors()).sum();
    while pools.len() > DESCRIPTOR_POOLS || held > DESCRIPTORS {
        let Some(oldest) = pools.pop_front() else {
            break;
        };
        held -= oldest.room.descriptors();
        past.push(oldest.raw);
    }
    past
}
