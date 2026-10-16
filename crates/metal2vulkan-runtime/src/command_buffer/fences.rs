//! The fences that end command buffers, where the driver needs them: the batch that ends a command
//! buffer signals one with its submission, and the completion turn finishes the command buffer
//! only once both its queue's timeline reached the batch's value and the fence is signaled
//! ([`completion`](super::completion)).
//!
//! - **Where both.** Vulkan makes the two equivalent: once a timeline semaphore's value is
//!   reached, the work submitted before its signal completed, and what it used may be released.
//!   A driver may still read some of it later. MoltenVK 1.4.1 signals a timeline semaphore from
//!   the GPU (an `MTLSharedEvent`), before the Metal command buffer's completed handlers run, and
//!   one of those handlers calls into every query pool the command buffer reset or began, without
//!   retaining it: a command buffer released at the value crashed there. It signals a fence from
//!   the last of those handlers. That is MoltenVK's behavior, not a guarantee of Vulkan's, and the
//!   device's driver defects name it (`RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal`,
//!   `driver_defects.rs`).
//! - **Only there**. Every other driver signals the fence with the value, so a command buffer there
//!   ends at its value alone and its commit takes no fence ([`ends_at_fence`]): NVIDIA's
//!   `vkCreateFence` on the commit path waited up to 1.9 s for other processes' work. A queue whose
//!   failures are relayed (`device/relay.rs`) still takes one for every batch, which its log of
//!   submissions in flight needs.
//! - **Pooled, and bounded.** A fence is taken when the last batch is submitted (made only when
//!   none is idle) and given back, reset, once a completion turn saw it signaled. At most
//!   [`IDLE`] wait idle; the ones beyond are released. So the fences live are the command buffers
//!   in flight plus at most [`IDLE`], whatever the number of command buffers committed.
//! - **The exit guard's doors.** Making one is the scheduler's (inside its section), waiting for
//!   and resetting them the queues' waiters' (inside their sections), and every release goes
//!   through `release.rs`; each entry point is in the guard's table and primed
//!   (`exit_guard`, `device/prime.rs`).

use crate::device::lock;
use crate::exit_guard::{self, Call};
use crate::release;
use crate::Device;
use ash::vk;
use metal2vulkan::error::RuntimeDriverDefect;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The most fences kept idle.
pub(crate) const IDLE: usize = 16;

/// Whether a submission to `device`'s `VkQueue` `queue` that ends a command buffer takes a fence:
/// where the driver reads a command buffer's objects after its signal, or the queue's failures
/// are relayed (the module docs).
pub(crate) fn ends_at_fence(device: &Device, queue: usize) -> bool {
    device.shared().flights(queue).is_some()
        || (device.properties().driver_defects)
            .contains(&RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal)
}

/// A device's idle fences, each unsignaled and in no pending submission.
#[derive(Default)]
pub(crate) struct Fences {
    idle: Mutex<Vec<vk::Fence>>,
    /// The fences made, none idle: `WorkCounts::fences_made`.
    made: AtomicU64,
}

impl Fences {
    /// An unsignaled fence of `device`: an idle one, else a new one.
    ///
    /// # Safety
    ///
    /// The fences are `device`'s; the caller is inside a section of the exit guard.
    pub(crate) unsafe fn take(&self, device: &ash::Device) -> Result<vk::Fence, vk::Result> {
        if let Some(fence) = self.take_idle() {
            return Ok(fence);
        }
        // SAFETY: the device is live (the caller holds it).
        let made = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) };
        exit_guard::called(Call::CreateFence);
        self.made.fetch_add(1, Ordering::Relaxed);
        made
    }

    /// How many fences [`take`](Self::take) made so far.
    pub(crate) fn made(&self) -> u64 {
        self.made.load(Ordering::Relaxed)
    }

    /// Give back `fences` of `device`, each unsignaled (reset, or never submitted) and in no
    /// pending submission: kept idle up to [`IDLE`], the rest released. `fences` is left empty,
    /// its room kept for the caller's next ones.
    ///
    /// # Safety
    ///
    /// As above.
    pub(crate) unsafe fn give(&self, device: &ash::Device, fences: &mut Vec<vk::Fence>) {
        self.keep(fences);
        for fence in fences.drain(..) {
            // SAFETY: the caller's: no pending submission uses it.
            unsafe { release::object(device, fence) };
        }
    }

    /// Release every idle fence: the device is being torn down.
    ///
    /// # Safety
    ///
    /// As above.
    pub(crate) unsafe fn release(&self, device: &ash::Device) {
        let idle: Vec<vk::Fence> = lock(&self.idle).drain(..).collect();
        for fence in idle {
            // SAFETY: the caller's.
            unsafe { release::object(device, fence) };
        }
    }

    /// An idle fence, if any.
    pub(crate) fn take_idle(&self) -> Option<vk::Fence> {
        lock(&self.idle).pop()
    }

    /// Keep `fences` idle up to [`IDLE`], the first of them first; the ones beyond stay in
    /// `fences`, for the caller to release.
    pub(crate) fn keep(&self, fences: &mut Vec<vk::Fence>) {
        let mut idle = lock(&self.idle);
        let kept = IDLE.saturating_sub(idle.len()).min(fences.len());
        idle.extend(fences.drain(..kept));
    }
}
