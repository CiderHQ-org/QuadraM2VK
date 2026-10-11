//! The cache's background thread, one per device: the scan at device creation
//! ([`Store::sweep`]) and the scans that bring the directory back under its cap
//! ([`Store::evict`]), off every program's thread; nothing on a pipeline's or a command's path
//! waits for them. (The saves of the device's one driver cache it made, went with that cache; each
//! making now writes its own record, `driver.rs`.)

use super::store::Store;
use crate::device::lock;
use crate::error::{Error, MemoryKind};
use crate::exit_guard;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// The thread and what it is asked to do.
pub(super) struct Worker {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

struct Shared {
    work: Mutex<Work>,
    wake: Condvar,
    /// Set when the device goes: a running scan stops early.
    stop: AtomicBool,
}

#[derive(Default)]
struct Work {
    evict: bool,
    stopping: bool,
}

impl Worker {
    /// Start the thread over `store`.
    pub(super) fn start(store: Arc<Store>) -> Result<Worker, Error> {
        let shared = Arc::new(Shared {
            work: Mutex::new(Work::default()),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let thread = {
            let shared = Arc::clone(&shared);
            exit_guard::spawn(
                std::thread::Builder::new().name("metal2vulkan-shader-cache".into()),
                move || run(&shared, &store),
            )
            .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?
        };
        Ok(Worker {
            shared,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Ask for a scan that brings the directory under its cap.
    pub(super) fn evict(&self) {
        lock(&self.shared.work).evict = true;
        self.shared.wake.notify_all();
    }

    /// Stop the thread, and wait for it.
    pub(super) fn stop(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        lock(&self.shared.work).stopping = true;
        self.shared.wake.notify_all();
        if let Some(handle) = lock(&self.thread).take() {
            let _ = handle.join();
        }
    }
}

/// The thread's loop.
fn run(shared: &Shared, store: &Store) {
    store.sweep(&shared.stop);
    loop {
        {
            let mut work = lock(&shared.work);
            while !work.stopping && !work.evict {
                work = shared
                    .wake
                    .wait(work)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            if work.stopping {
                break;
            }
            work.evict = false;
        }
        store.evict(&shared.stop);
    }
}
