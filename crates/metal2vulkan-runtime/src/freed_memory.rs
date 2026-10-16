//! Freed memory that waits for the batches in flight, on a device whose driver fails pending work
//! when memory a device-address buffer lay in is freed
//! (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`: MoltenVK).
//!
//! - **The defect.** MoltenVK makes every `VkBuffer` of `SHADER_DEVICE_ADDRESS` usage resident
//!   (`useResource:`) in every encoder whose pipeline reads device addresses
//!   (`MVKDevice::encodeGPUAddressableBuffers`), in Metal command buffers that retain none of it,
//!   and lets go of the buffer's Metal memory as its `VkDeviceMemory` is freed: a Metal command
//!   buffer encoded before the free and not yet run fails (`Invalid Resource`, Metal's code 9).
//!   Vulkan allows the free, as no submitted command refers to the memory
//!   (VUID-vkFreeMemory-memory-00677). Destroying the buffer alone, its memory kept, fails
//!   nothing, and takes it out of the driver's list.
//! - **The wait.** There a block of the runtime's memory that goes (`memory.rs`) has its buffer of
//!   addresses destroyed at once, and its memory waits, parked, until each `VkQueue`'s timeline
//!   reaches the value of the last batch submitted there as the block went. Vulkan's semaphore
//!   signal operations "defined by vkQueueSubmit or vkQueueSubmit2 additionally include all
//!   commands that occur earlier in submission order" in their first synchronization scope, so
//!   at that value every submission there before it completed. A host import's deallocator and
//!   keep-alive, which go after its memory is freed (`buffer.rs`), wait with it, the deallocator's
//!   wait announced (`host-deallocator-delayed`).
//! - **Each batch noted.** A batch's `vkQueueSubmit` and the store of its value run under the
//!   gate's read lock, inside the queue's lock (`relay.rs`); a block's noting of the values, after
//!   its buffers were destroyed, under the write lock. So a batch whose encoding listed such a
//!   buffer, submitted before the buffer was destroyed, is noted, and a batch submitted after the
//!   noting lists none.
//! - **And the device's own submissions.** Those that run command buffers outside the batches
//!   (`device/submit.rs`) signal no queue's timeline: each takes a number before its
//!   `vkQueueSubmit` and gives it back once its submitter saw it complete ([`OwnSubmission`]),
//!   and a block waits too for every one numbered before it went. Its sparse binds and the
//!   staging queue's copies run no pipeline, so no encoder of theirs lists a buffer. The
//!   program's own submissions (`CommandQueue::with_vk_queue`) are not waited for: the runtime
//!   sees neither what they list nor when they complete.
//! - **Freed off the runtime's completion threads.** A queue's waiter that publishes a value while
//!   blocks wait posts their release to a handler thread (`Device::let_go_elsewhere`), which frees
//!   the blocks it reached and then lets go of what waited for them; a block whose values were
//!   reached as it went is freed at once, where it went. At the device's teardown, its queues
//!   idle, every block left is freed (`Memory::destroy`).

use crate::device::{lock, Device};
use crate::exit_guard;
use ash::vk;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock, RwLockReadGuard};

/// A device's record of its batches for the blocks that wait (the module docs).
pub(crate) struct Parking {
    /// Read around each batch's submission and the store of its value, written around a block's
    /// noting of the values.
    gate: RwLock<()>,
    /// Per `VkQueue`: the value of the last batch submitted there.
    submitted: Box<[AtomicU64]>,
    /// Per `VkQueue`: the value through which every batch submitted there completed, as the host
    /// saw.
    completed: Box<[AtomicU64]>,
    /// The blocks waiting (`Memory`'s, counted here so that a waiter reads it with no lock).
    waiting: AtomicUsize,
    /// A release is posted to a handler thread and has not begun: no other is posted.
    posted: AtomicBool,
    /// The device's own submissions: the next one's number, and those in flight.
    own: Mutex<Own>,
}

/// The device's own submissions ([`OwnSubmission`]).
#[derive(Default)]
struct Own {
    /// The next one's number.
    next: u64,
    /// The numbers of those whose submitter has not yet seen them complete.
    in_flight: BTreeSet<u64>,
}

/// What a block noted as it went: the value of each `VkQueue`'s last batch submitted, and the
/// number the device's next own submission takes.
pub(crate) struct Noted {
    values: Box<[u64]>,
    own: u64,
}

impl Parking {
    /// The record of a device with `queues` `VkQueue`s, none submitted to.
    pub(crate) fn new(queues: usize) -> Parking {
        Parking {
            gate: RwLock::new(()),
            submitted: (0..queues).map(|_| AtomicU64::new(0)).collect(),
            completed: (0..queues).map(|_| AtomicU64::new(0)).collect(),
            waiting: AtomicUsize::new(0),
            posted: AtomicBool::new(false),
            own: Mutex::new(Own::default()),
        }
    }

    /// The gate's read lock, held around a batch's `vkQueueSubmit` and the store of its value
    /// ([`submitted`](Self::submitted)), inside the queue's lock.
    pub(crate) fn submitting(&self) -> RwLockReadGuard<'_, ()> {
        self.gate
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// A batch signalling `value` was submitted to `VkQueue` `queue`, under
    /// [`submitting`](Self::submitting)'s lock.
    pub(crate) fn submitted(&self, queue: usize, value: u64) {
        self.submitted[queue].fetch_max(value, Ordering::SeqCst);
    }

    /// The host saw `VkQueue` `queue`'s timeline reach `value`.
    pub(crate) fn completed(&self, queue: usize, value: u64) {
        self.completed[queue].fetch_max(value, Ordering::SeqCst);
    }

    /// The value of each `VkQueue`'s last batch submitted, under the gate's write lock, and the
    /// device's next own submission's number: called once the block's buffers were destroyed.
    pub(crate) fn note(&self) -> Noted {
        let _gate = self
            .gate
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let values = (self.submitted.iter())
            .map(|value| value.load(Ordering::SeqCst))
            .collect();
        Noted {
            values,
            own: lock(&self.own).next,
        }
    }

    /// Whether every `VkQueue`'s timeline reached the value `noted` holds for it, and every own
    /// submission numbered before `noted`'s completed.
    pub(crate) fn reached(&self, noted: &Noted) -> bool {
        let batches = (self.completed.iter())
            .zip(&noted.values)
            .all(|(completed, noted)| completed.load(Ordering::SeqCst) >= *noted);
        batches
            && lock(&self.own)
                .in_flight
                .first()
                .is_none_or(|first| *first >= noted.own)
    }

    /// An own submission of `device`'s, from before its `vkQueueSubmit` until the guard drops,
    /// once its submitter saw it complete (or gave up on it).
    pub(crate) fn own_submission<'a>(&'a self, device: &'a Device) -> OwnSubmission<'a> {
        let mut own = lock(&self.own);
        let number = own.next;
        own.next += 1;
        own.in_flight.insert(number);
        OwnSubmission {
            parking: self,
            device,
            number,
        }
    }

    /// One more block waits.
    pub(crate) fn waits(&self) {
        self.waiting.fetch_add(1, Ordering::SeqCst);
    }

    /// `count` blocks waited and are freed.
    pub(crate) fn freed(&self, count: usize) {
        self.waiting.fetch_sub(count, Ordering::SeqCst);
    }

    /// After the host saw a value of a queue's: where blocks wait and no release is posted, post
    /// one to a handler thread of `device`'s, which frees those whose values were reached.
    pub(crate) fn after_completion(&self, device: &Device) {
        if self.waiting.load(Ordering::SeqCst) == 0 || self.posted.swap(true, Ordering::SeqCst) {
            return;
        }
        device.let_go_elsewhere(Box::new(Unpark(device.clone())));
    }
}

/// An own submission of the device's in flight ([`Parking::own_submission`]): as it drops, the
/// blocks that waited for it alone may go, on a handler thread.
pub(crate) struct OwnSubmission<'a> {
    parking: &'a Parking,
    device: &'a Device,
    number: u64,
}

impl Drop for OwnSubmission<'_> {
    fn drop(&mut self) {
        lock(&self.parking.own).in_flight.remove(&self.number);
        self.parking.after_completion(self.device);
    }
}

/// A block's memory waiting for the batches in flight as it went, and what goes after it.
pub(crate) struct Parked {
    pub(crate) noted: Noted,
    pub(crate) memory: vk::DeviceMemory,
    /// What waits for the memory to be freed (a host import's deallocators and keep-alive).
    pub(crate) after: Option<Box<dyn Send>>,
}

impl Parked {
    /// Free the memory: what waited for it, to let go of outside the exit guard's section the
    /// free runs in (it runs the program's code).
    ///
    /// # Safety
    ///
    /// The memory is `device`'s, nothing is placed in it, and every batch that may list it
    /// completed.
    pub(crate) unsafe fn release(self, device: &ash::Device) -> Option<Box<dyn Send>> {
        let Parked { memory, after, .. } = self;
        // SAFETY: the caller's.
        unsafe { crate::release::object(device, memory) };
        after
    }
}

/// The release a queue's waiter posts ([`Parking::after_completion`]): on a handler thread, the
/// blocks whose values were reached are freed, and what waited for them goes.
struct Unpark(Device);

impl Drop for Unpark {
    fn drop(&mut self) {
        let shared = self.0.shared();
        let Some(parking) = shared.parking.get() else {
            return;
        };
        // Cleared first: a value published from here on posts another release.
        parking.posted.store(false, Ordering::SeqCst);
        // The frees are a section of the exit guard: while the process exits, the blocks stay
        // to its end (`exit_guard`).
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        let reached = lock(&shared.memory).reached();
        let after: Vec<Box<dyn Send>> = (reached.into_iter())
            // SAFETY: its values were reached, so every batch that may list it completed.
            .filter_map(|parked| unsafe { parked.release(shared.vk()) })
            .collect();
        drop(busy);
        drop(after);
    }
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl Device {
    /// Let the memory's blocks that go from now on wait for the batches in flight, as where the
    /// driver has `FreedAddressMemoryFailsPendingWork`: the crate's tests' way to take that path
    /// on any device. Before the device's first batch.
    pub(crate) fn park_freed_memory(&self) {
        let shared = self.shared();
        let parking = shared
            .parking
            .get_or_init(|| std::sync::Arc::new(Parking::new(shared.queues.len())));
        lock(&shared.memory).park_by(std::sync::Arc::clone(parking));
    }
}
