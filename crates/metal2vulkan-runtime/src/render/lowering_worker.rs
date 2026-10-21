//! A device's lowering thread (`render/deferral.rs`): it lowers the batches of logged calls the
//! device's render encoders hand it, while the program goes on.
//!
//! - **Order.** It takes the slots it is handed in the order they were handed, and each slot's
//!   batches in the order the program logged them, one at a time: a slot's calls are lowered in
//!   their order whichever thread lowers them, since a drain on the program's thread waits for the
//!   batch the thread is lowering and takes the rest (`LoweringSlot::drain`).
//! - **The program's calls.** Each batch is lowered as the calls it holds
//!   (`exit_guard::lowering_program_calls`): the notices it announces, the device's maps it
//!   changes and the pipelines it waits for are those calls', which the program made in that
//!   order on its own thread.
//! - **Exit.** Each batch is lowered inside a section of the exit guard, which an exit waits for.
//!   Once the process exits it lowers none: the program's thread lowers what it was handed at its
//!   next drain, or not at all.
//! - **A panic** while lowering ends the thread: the slot it was lowering refuses every later
//!   drain with that panic, on the program's thread, as the call would have panicked under
//!   `RenderLoweringPolicy::AtCall`, and from then on the program's threads lower every slot's
//!   calls themselves.
//! - **Started** with the device's first encoder that logs its calls, at the user-interactive
//!   class (`exit_guard::spawn_user_interactive`), as the threads command buffers' work waits for;
//!   stopped and joined with the device (`DeviceShared`'s drop). No slot handed to it holds the
//!   device by then: an encoder holds its device until it ends or goes, and its slot lets go of
//!   its lowering then.

use super::deferral::LoweringSlot;
use crate::device::lock;
use crate::exit_guard;
use crate::recorder::take_leaving_room;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// The device's lowering thread (the module docs).
pub(crate) struct LoweringWorker {
    shared: Arc<WorkerShared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// What the thread and the encoders that hand it their slots share.
struct WorkerShared {
    queue: Mutex<Queue>,
    wake: Condvar,
    /// The thread ended by a panic (the module docs).
    ended: AtomicBool,
}

struct Queue {
    /// The slots handed to the thread, in order.
    slots: VecDeque<Arc<LoweringSlot>>,
    /// The thread waits for a slot.
    parked: bool,
    /// The device is torn down: the thread ends.
    stop: bool,
}

impl LoweringWorker {
    /// The device's lowering thread, started now; `None` where the host starts no thread, where
    /// the program's threads lower every logged call at their drains.
    pub(super) fn start() -> Option<LoweringWorker> {
        let shared = Arc::new(WorkerShared {
            queue: Mutex::new(Queue {
                slots: VecDeque::new(),
                parked: false,
                stop: false,
            }),
            wake: Condvar::new(),
            ended: AtomicBool::new(false),
        });
        let running = Arc::clone(&shared);
        let builder = std::thread::Builder::new().name("metal2vulkan-lowering".to_owned());
        let thread = exit_guard::spawn_user_interactive(builder, move || running.run()).ok()?;
        Some(LoweringWorker {
            shared,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Hand `slot`, which has batches handed and is in no queue, to the thread: whether it takes
    /// it (it has not ended).
    pub(super) fn hand(&self, slot: &Arc<LoweringSlot>) -> bool {
        let mut queue = lock(&self.shared.queue);
        if self.shared.ended.load(Ordering::Acquire) {
            return false;
        }
        queue.slots.push_back(Arc::clone(slot));
        if queue.parked {
            self.shared.wake.notify_one();
        }
        true
    }

    /// Whether the thread lowers what it is handed: it has not ended.
    pub(super) fn runs(&self) -> bool {
        !self.shared.ended.load(Ordering::Acquire)
    }

    /// End the thread and wait for it, as the device is torn down.
    pub(super) fn stop(&self) {
        lock(&self.shared.queue).stop = true;
        self.shared.wake.notify_one();
        if let Some(thread) = lock(&self.thread).take() {
            let _ = thread.join();
        }
    }
}

impl WorkerShared {
    /// The thread's loop: each slot handed lowered, until [`LoweringWorker::stop`].
    fn run(&self) {
        let mut running = Running {
            shared: self,
            slot: None,
        };
        loop {
            let slot = {
                let mut queue = lock(&self.queue);
                loop {
                    if let Some(slot) = take_leaving_room(&mut queue.slots) {
                        break slot;
                    }
                    if queue.stop {
                        return;
                    }
                    queue.parked = true;
                    queue =
                        (self.wake.wait(queue)).unwrap_or_else(std::sync::PoisonError::into_inner);
                    queue.parked = false;
                }
            };
            running.slot = Some(slot);
            if let Some(slot) = &running.slot {
                slot.lower_handed();
            }
            running.slot = None;
        }
    }
}

/// The thread's loop as it runs: a panic that unwinds it ends the thread (the module docs).
struct Running<'a> {
    shared: &'a WorkerShared,
    /// The slot it lowers now.
    slot: Option<Arc<LoweringSlot>>,
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        let waiting: Vec<Arc<LoweringSlot>> = {
            let mut queue = lock(&self.shared.queue);
            self.shared.ended.store(true, Ordering::Release);
            queue.slots.drain(..).collect()
        };
        if let Some(slot) = &self.slot {
            slot.lost_to_panic();
        }
        for slot in waiting {
            slot.left_by_thread();
        }
    }
}
