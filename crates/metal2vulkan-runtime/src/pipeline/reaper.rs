//! The device's reaper, for the release that never waits: one thread of the device's that lets go
//! of what a released render state's pool pieces give back as they end, so the state's release
//! returns at once and a pool thread is never the last holder of the device.
//!
//! - **Started by a release.** A state released while a piece of its own is running starts the
//!   thread ([`Reaper::start`], on the releasing thread), unless it runs; until then the device
//!   has none.
//! - **What a piece gave back.** Such a piece, as it ends, hands what it took ([`Reaps::reap`]),
//!   with the trail it opened before settling its making (`trails.rs`), which closes after it;
//!   the thread lets go of each, in turn, outside any section of the exit guard: each release of a
//!   Vulkan object it reaches runs in a section of its own (`release.rs`), refused once the
//!   process is exiting.
//! - **Holds nothing of the device.** The thread holds only its queue. What it lets go of may be
//!   the device's last holder: the teardown then runs here, and its [`Reaper::stop`] does not
//!   join this thread, which ends once its queue is empty. A teardown on any other thread joins
//!   it: nothing the thread has left to let go of holds the device then.

use crate::device::lock;
use crate::exit_guard;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// The device's reaper: its thread, once a release started it.
#[derive(Default)]
pub(crate) struct Reaper {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// Where the pieces of a released state hand what they gave back ([`Reaper::start`]): the
/// reaper's queue, which holds nothing of the device.
#[derive(Clone)]
pub(crate) struct Reaps(Arc<Shared>);

#[derive(Default)]
struct Shared {
    queue: Mutex<Queue>,
    /// Something was handed over, or the thread is asked to stop.
    work: Condvar,
}

#[derive(Default)]
struct Queue {
    held: VecDeque<Box<dyn Send>>,
    stopping: bool,
}

impl Reaper {
    /// Start the thread unless it runs, and where the pieces hand what they give back; `None`
    /// where the host cannot start a thread.
    pub(crate) fn start(&self) -> Option<Reaps> {
        let mut thread = lock(&self.thread);
        if thread.is_none() {
            let shared = Arc::clone(&self.shared);
            let handle = exit_guard::spawn(
                std::thread::Builder::new().name("metal2vulkan-reaper".into()),
                move || run(&shared),
            )
            .ok()?;
            *thread = Some(handle);
        }
        Some(Reaps(Arc::clone(&self.shared)))
    }

    /// The device's teardown: the thread ends once its queue is empty, and is joined unless this
    /// is it.
    pub(crate) fn stop(&self) {
        lock(&self.shared.queue).stopping = true;
        self.shared.work.notify_all();
        let mut thread = lock(&self.thread);
        let own = thread
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id());
        if !own {
            if let Some(handle) = thread.take() {
                let _ = handle.join();
            }
        }
    }
}

impl Reaps {
    /// Hand `held` to the thread, which lets go of it.
    pub(crate) fn reap(&self, held: Box<dyn Send>) {
        lock(&self.0.queue).held.push_back(held);
        self.0.work.notify_all();
    }
}

/// The thread's loop: let go of each value handed over, in turn, until stopped with none left.
fn run(shared: &Shared) {
    loop {
        let held = {
            let mut queue = lock(&shared.queue);
            loop {
                if let Some(held) = queue.held.pop_front() {
                    break held;
                }
                if queue.stopping {
                    return;
                }
                queue = shared
                    .work
                    .wait(queue)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        // Outside the queue's lock: it may be the device's last holder, whose teardown stops
        // this thread.
        drop(held);
    }
}
