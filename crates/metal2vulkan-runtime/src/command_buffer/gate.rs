//! When a command buffer's waiters return: once it finished (its completed handlers ran) and the
//! runtime's threads let go of every reference they took to it.
//!
//! - **Apart from the command buffer.** The [`Gate`] is not part of what it guards, so a thread
//!   lets go of its reference first and leaves the gate after: the gate never opens while a
//!   runtime thread still holds the command buffer, or anything it held for it.
//! - **Why.** A program that waits for its last command buffer and then drops its last handles
//!   destroys the device on its own thread, before the drop returns. Were a runtime thread to
//!   hold the last reference instead, the device would go on that thread, perhaps while the
//!   program exits, which drivers do not survive (RADV ended such a process with `SIGSEGV`).
//! - **What it does not cover.** A program that learns of the completion otherwise (from a
//!   handler, an event the command buffer signals, or its status) may drop its last handles while
//!   the runtime's thread still holds the command buffer; the device then goes on that thread.
//!   An exit meanwhile waits for that teardown, or skips it once begun (`exit_guard`).
//! - **Exit.** Once the process is exiting, the threads that would open a gate end, so a wait
//!   answers at once, and one already blocked within `exit_guard::WAKE`.

use crate::device::lock;
use crate::error::{DeadlockCause, TimedOut};
use crate::exit_guard;
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// A wait that cannot end: the process is exiting.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Exiting;

impl From<Exiting> for DeadlockCause {
    fn from(Exiting: Exiting) -> DeadlockCause {
        DeadlockCause::Exiting
    }
}

impl From<Exiting> for crate::error::Error {
    fn from(exiting: Exiting) -> crate::error::Error {
        DeadlockCause::from(exiting).into()
    }
}

/// One command buffer's gate.
#[derive(Debug, Default)]
pub(crate) struct Gate {
    state: Mutex<GateState>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct GateState {
    /// The command buffer finished: its completed handlers ran.
    finished: bool,
    /// The [`Hold`]s alive.
    held: usize,
}

impl GateState {
    fn open(&self) -> bool {
        self.finished && self.held == 0
    }
}

impl Gate {
    /// The command buffer finished; the gate opens once no [`Hold`] is left.
    pub(crate) fn finish(&self) {
        lock(&self.state).finished = true;
        self.changed.notify_all();
    }

    /// Whether the waiters may return.
    pub(crate) fn is_open(&self) -> bool {
        lock(&self.state).open()
    }

    /// Block until the gate opens. `Err` once the process is exiting and it is not open: the
    /// runtime's threads that would open it end (`exit_guard`).
    pub(crate) fn wait(&self) -> Result<(), Exiting> {
        self.wait_unless(exit_guard::closed)
    }

    /// [`Gate::wait`], with `exiting` saying whether the process is exiting (the exit guard's
    /// [`closed`](exit_guard::closed), or a test's own).
    pub(crate) fn wait_unless(&self, exiting: impl Fn() -> bool) -> Result<(), Exiting> {
        let mut state = lock(&self.state);
        while !state.open() {
            if exiting() {
                return Err(Exiting);
            }
            state = self
                .changed
                .wait_timeout(state, exit_guard::WAKE)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        Ok(())
    }

    /// Block until the gate opens, or until `deadline`; at once once the process is exiting, as
    /// [`Gate::wait`].
    pub(crate) fn wait_until(&self, deadline: Instant) -> Result<(), TimedOut> {
        let mut state = lock(&self.state);
        while !state.open() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() || exit_guard::closed() {
                return Err(TimedOut);
            }
            state = self
                .changed
                .wait_timeout(state, left.min(exit_guard::WAKE))
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        Ok(())
    }

    fn enter(&self) {
        lock(&self.state).held += 1;
    }

    fn leave(&self) {
        let mut state = lock(&self.state);
        state.held -= 1;
        let open = state.open();
        drop(state);
        if open {
            self.changed.notify_all();
        }
    }
}

/// A reference a runtime thread holds (the scheduler's, the completion thread's, a queue
/// context's transition), counted by a [`Gate`]. Its fields drop in declaration order: the
/// reference first, then the count, so the gate cannot open while the reference is alive.
pub(crate) struct Hold<T> {
    value: Arc<T>,
    _count: Count,
}

/// A [`Hold`]'s place in its gate's count.
struct Count(Arc<Gate>);

impl Drop for Count {
    fn drop(&mut self) {
        self.0.leave();
    }
}

impl<T> Hold<T> {
    /// A new reference to `value`, counted by `gate`.
    pub(crate) fn new(value: &Arc<T>, gate: &Arc<Gate>) -> Hold<T> {
        gate.enter();
        Hold {
            value: Arc::clone(value),
            _count: Count(Arc::clone(gate)),
        }
    }

    /// Whether `self` and `other` hold the same value.
    pub(crate) fn same(&self, other: &Hold<T>) -> bool {
        Arc::ptr_eq(&self.value, &other.value)
    }
}

impl<T> Deref for Hold<T> {
    type Target = Arc<T>;

    fn deref(&self) -> &Arc<T> {
        &self.value
    }
}
