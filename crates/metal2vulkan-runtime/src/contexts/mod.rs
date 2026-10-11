//! The device's handler threads: each logical queue's serial context, run on a pool of threads, as
//! Metal runs its queues' handlers.
//!
//! - **What a context runs.** In its queue's order, each command buffer's transitions: to
//!   `Scheduled` with its scheduled handlers, and to `Completed` (or `Error`) with its completed
//!   handlers and the release of what it retained. The scheduler and the completion thread only
//!   post them ([`Contexts::post`]); they run no such handler. (A transition to `Scheduled` waits
//!   for its command buffer's submission handlers, which the thread that submitted its last batch
//!   runs.)
//! - **Metal's order** (measured on an M4, macOS 27.0.1, EV `RT4-metal-order`): while one of a
//!   queue's handlers blocks, the queue's later transitions wait, even of a command buffer whose
//!   work already ran; another queue's transitions and handlers go on, on another thread. Metal
//!   ran 256 queues' handlers blocked at once, on 256 threads, and reached no limit.
//! - **Threads.** One thread starts with the device and stays. Another starts when a context is
//!   ready and no thread is free, up to [`THREADS`]; one that found nothing to run for [`IDLE`]
//!   ends. The cap is the runtime's own, above the 256 measured on Metal. Each has a stack of
//!   [`STACK`] bytes.
//! - **Waits that would deadlock** ([`block`]). A handler's wait for another queue's command
//!   buffer blocks its thread. It is refused when it would close a cycle of such waits (across
//!   every device) or leave the device's pool with no thread that is not blocked so; then at least
//!   one thread can always run the contexts the blocked waits wait for.
//! - **Stopping.** The device's drop stops the threads once none runs an item (every item holds
//!   its command buffer, which holds the device). A drop on one of the threads (its item held the
//!   last reference) does not wait for itself.
//!
//! Locks: a pool's state, then its census; the waits' graph, then a census. No call made under
//! either takes the state again.

pub(crate) mod lines;

use super::command_buffer::own::{Context, Waits};
use super::device::lock;
use super::error::{DeadlockCause, Error, MemoryKind};
use super::exit_guard;
use lines::Lines;
use std::cell::RefCell;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// The most threads a device's pool runs at once: above the 256 handlers Metal was measured to
/// run blocked at once.
pub(crate) const THREADS: usize = 512;

/// Each thread's stack, in bytes.
pub(crate) const STACK: usize = 512 * 1024;

/// How long a thread other than the first finds nothing to run before it ends.
const IDLE: Duration = Duration::from_secs(2);

/// An item of a queue's context: run once, on one of the device's handler threads.
pub(crate) trait Run: Send + 'static {
    /// Run the item; dropping it lets go of what it holds.
    fn run(self);
}

/// One device's handler threads. A clone is another handle of the same pool, which holds nothing
/// of the device.
pub(crate) struct Contexts<I: Run> {
    shared: Arc<Shared<I>>,
}

impl<I: Run> Clone for Contexts<I> {
    fn clone(&self) -> Contexts<I> {
        Contexts {
            shared: Arc::clone(&self.shared),
        }
    }
}

struct Shared<I> {
    state: Mutex<State<I>>,
    /// Signaled when a line becomes ready or the pool stops.
    wake: Condvar,
    census: Arc<Census>,
}

struct State<I> {
    lines: Lines<I>,
    /// The threads that run no item: waiting for one, or starting.
    free: usize,
    stopping: bool,
}

/// What a pool counts, shared with the waits its threads block in.
pub(crate) struct Census {
    count: Mutex<Count>,
    /// Signaled when a thread ends.
    gone: Condvar,
}

struct Count {
    /// The threads alive.
    threads: usize,
    /// The threads blocked in a wait for another queue's command buffer.
    blocked: usize,
    /// The most threads: [`THREADS`], or the count reached when the host refused another.
    limit: usize,
}

thread_local! {
    /// The census of the pool this thread belongs to, on a handler thread.
    static POOL: RefCell<Option<Arc<Census>>> = const { RefCell::new(None) };
}

/// The waits of every handler thread for another queue's context.
static WAITS: Mutex<Waits> = Mutex::new(Waits::new());

impl<I: Run> Contexts<I> {
    /// A pool of at most `limit` threads (the device's: [`THREADS`]), with its first thread;
    /// refused when the host cannot start it.
    pub(crate) fn start(limit: usize) -> Result<Contexts<I>, Error> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                lines: Lines::default(),
                free: 1,
                stopping: false,
            }),
            wake: Condvar::new(),
            census: Arc::new(Census {
                count: Mutex::new(Count {
                    threads: 1,
                    blocked: 0,
                    limit,
                }),
                gone: Condvar::new(),
            }),
        });
        spawn(&shared, true).map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
        Ok(Contexts { shared })
    }

    /// Append `item` to the context of logical queue `queue`; start a thread when a context is
    /// ready that no free thread will take, below the cap.
    pub(crate) fn post(&self, queue: u64, item: I) {
        let mut state = lock(&self.shared.state);
        state.lines.push(queue, item);
        if state.lines.ready() > state.free {
            let start = {
                let mut count = lock(&self.shared.census.count);
                let start = count.threads < count.limit;
                if start {
                    count.threads += 1;
                }
                start
            };
            if start {
                if spawn(&self.shared, false).is_ok() {
                    state.free += 1;
                } else {
                    // The host's threads ran out: the pool keeps those it has, and so does the
                    // cap on blocked waits.
                    let mut count = lock(&self.shared.census.count);
                    count.threads -= 1;
                    count.limit = count.threads;
                }
            }
        }
        drop(state);
        self.shared.wake.notify_one();
    }

    /// Stop the threads and wait for them to end, but for this one when it is one of them. The
    /// device is being dropped, so no item is left.
    pub(crate) fn stop(&self) {
        lock(&self.shared.state).stopping = true;
        self.shared.wake.notify_all();
        let census = &self.shared.census;
        let own = POOL.with(|pool| {
            pool.borrow()
                .as_ref()
                .is_some_and(|pool| Arc::ptr_eq(pool, census))
        });
        let floor = usize::from(own);
        let mut count = lock(&census.count);
        while count.threads > floor {
            count = census
                .gone
                .wait(count)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// The threads alive and the threads blocked in a wait.
    #[cfg(test)]
    pub(crate) fn census(&self) -> (usize, usize) {
        let count = lock(&self.shared.census.count);
        (count.threads, count.blocked)
    }
}

/// Start one of `shared`'s threads (counted already); `first` stays until the pool stops.
fn spawn<I: Run>(shared: &Arc<Shared<I>>, first: bool) -> std::io::Result<()> {
    let shared = Arc::clone(shared);
    exit_guard::spawn(
        std::thread::Builder::new()
            .name("metal2vulkan-handlers".into())
            .stack_size(STACK),
        move || run(&shared, first),
    )
    .map(drop)
}

/// A thread's loop: run the ready contexts' items, one at a time.
fn run<I: Run>(shared: &Shared<I>, first: bool) {
    POOL.with(|pool| *pool.borrow_mut() = Some(Arc::clone(&shared.census)));
    let mut state = lock(&shared.state);
    loop {
        if let Some((queue, item)) = state.lines.take() {
            state.free -= 1;
            drop(state);
            item.run();
            state = lock(&shared.state);
            state.lines.done(queue);
            state.free += 1;
            continue;
        }
        if state.stopping {
            break;
        }
        let (next, waited) = shared
            .wake
            .wait_timeout(state, IDLE)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next;
        if waited.timed_out() && !first && !state.lines.is_ready() && !state.stopping {
            break;
        }
    }
    state.free -= 1;
    drop(state);
    POOL.with(|pool| *pool.borrow_mut() = None);
    let mut count = lock(&shared.census.count);
    count.threads -= 1;
    drop(count);
    shared.census.gone.notify_all();
}

/// A handler thread's wait for another queue's context, counted until dropped.
pub(crate) struct Blocked {
    census: Arc<Census>,
    from: Context,
    to: Context,
}

/// Count a wait of this handler thread, which runs `from`'s item, for `to`'s context. Refused,
/// with its cause: when it would close a cycle of such waits (`Cycle`), and when it would leave
/// this thread's pool with no thread that is not blocked so, or this is not a handler thread
/// (`HandlerThreadCap`).
pub(crate) fn block(from: Context, to: Context) -> Result<Blocked, DeadlockCause> {
    let census = POOL
        .with(|pool| pool.borrow().clone())
        .ok_or(DeadlockCause::HandlerThreadCap)?;
    let mut waits = lock(&WAITS);
    if waits.closes_cycle(from, to) {
        return Err(DeadlockCause::Cycle);
    }
    {
        let mut count = lock(&census.count);
        if count.blocked + 1 >= count.limit {
            return Err(DeadlockCause::HandlerThreadCap);
        }
        count.blocked += 1;
    }
    waits.insert(from, to);
    Ok(Blocked { census, from, to })
}

impl Drop for Blocked {
    fn drop(&mut self) {
        lock(&WAITS).remove(self.from, self.to);
        lock(&self.census.count).blocked -= 1;
    }
}
