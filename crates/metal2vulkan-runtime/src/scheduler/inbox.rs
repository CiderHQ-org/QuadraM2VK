//! The scheduler's lock with an inbox, as a pure structure: a commit that finds another thread
//! making a pass leaves its batches to it and returns, instead of waiting for the pass's
//! submissions.
//!
//! A pass takes the lock, takes in what was committed, and submits what is ready, one
//! `vkQueueSubmit` after another (about 80 us each on ANV with i915, whose execbuffer lists every
//! memory object of the device). A commit made while one runs waited for all of it.
//!
//! - **Inline where the lock is free.** A commit tries the lock: where it gets it, it takes in
//!   what the inbox holds and then its own commit, and makes the pass itself, as before.
//! - **Handed to the pass under way.** Where the lock is taken, it puts its commit in the inbox
//!   and returns: the thread making the pass takes it in at its next one, which it makes as soon
//!   as it let go of the lock (below). The inbox keeps commit order, and every pass takes it in
//!   before anything else is committed: so the order sees commits in commit order, and the
//!   batches go to the `VkQueue`s, and are planned, in the order a pass picks them, as before.
//! - **Nothing left behind.** The threads making a pass or trying for one are counted
//!   (`passers`). Each, once done and counted out, looks at the inbox again after a sequentially
//!   consistent fence, and makes another pass for what it holds unless another passer is still
//!   counted, which looks again in turn. A commit puts its own in before the same fence and
//!   look: whichever of the two fences comes first, the later look sees the other's write, so
//!   a commit is either seen by a passer's look or finds no passer, and passes itself. The lock's
//!   other holders (a wake, a decode count, the thread's look for work) make no pass and are not
//!   counted: a commit that finds only one of them waits for its short hold, and then passes.
//! - **The scheduler's thread looks again until its passes leave it work** ([`Inbox::pass`]).
//!   Its passes may leave it work to do without the lock first (a submission check, a host step),
//!   which the batches behind it wait for: it stops looking again at the first pass that leaves
//!   some, rather than pass on for the commits that keep coming while that work waits, and looks
//!   at the inbox itself ([`Inbox::holds_commits`]) before each piece of that work and before it
//!   sleeps. A commit is left to it only while it holds the lock for a pass, so one waits at most
//!   for that pass and the submission handlers the thread runs right after it, never for a wake.
//! - **Who runs the after-work.** A pass's thread lets go, outside the lock, of what its passes
//!   submitted (the holds, the submission handlers) in submission order, as the scheduler does
//!   for its own: a commit handed off runs its submission handlers on that thread.

use crate::device::lock;
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, TryLockError};

/// A state `S` that passes work on under its lock, and the commits `C` waiting to be taken in.
pub(crate) struct Inbox<S, C> {
    state: Mutex<S>,
    /// Commits not yet taken in, in commit order.
    commits: Mutex<Vec<C>>,
    /// `commits` holds one (read without its lock; written under it).
    posted: AtomicBool,
    /// The threads making a pass or trying the lock for one.
    passers: AtomicUsize,
}

impl<S, C> Inbox<S, C> {
    pub(crate) fn new(state: S) -> Inbox<S, C> {
        Inbox {
            state: Mutex::new(state),
            commits: Mutex::new(Vec::new()),
            posted: AtomicBool::new(false),
            passers: AtomicUsize::new(0),
        }
    }

    /// The state's lock, for what makes no pass: a read, a mark for the thread, a wait on it.
    pub(crate) fn state(&self) -> &Mutex<S> {
        &self.state
    }

    /// Whether a commit waits in the inbox, not yet taken in.
    pub(crate) fn holds_commits(&self) -> bool {
        self.posted.load(Ordering::SeqCst)
    }

    /// Commit `commit`: where the lock is free, make the pass on this thread, `pass` taking in
    /// what the inbox held and `commit` after it, in commit order; where another thread holds it,
    /// leave `commit` to that thread's pass. Then look at the inbox again ([`look_again`]). Whether
    /// `commit` was left to another thread.
    ///
    /// [`look_again`]: Self::look_again
    pub(crate) fn commit(&self, commit: C, mut pass: impl FnMut(&mut S, Vec<C>)) -> bool {
        let mut pass = |state: &mut S, commits: Vec<C>| {
            pass(state, commits);
            true
        };
        self.passers.fetch_add(1, Ordering::SeqCst);
        let state = match self.state.try_lock() {
            Ok(state) => Some(state),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        };
        let left = match state {
            Some(mut state) => {
                let mut commits = self.take();
                commits.push(commit);
                pass(&mut state, commits);
                false
            }
            None => {
                let mut commits = lock(&self.commits);
                commits.push(commit);
                self.posted.store(true, Ordering::SeqCst);
                true
            }
        };
        self.passers.fetch_sub(1, Ordering::SeqCst);
        let passed = self.look_again(&mut pass);
        left && !passed
    }

    /// A pass by a thread that waits for the lock (the scheduler's own), `pass` taking in what the
    /// inbox holds and answering whether to look again; then look at the inbox again
    /// ([`look_again`](Self::look_again)) while each pass answers so. Where one answers no, the
    /// inbox may hold commits left to this thread, which it looks for itself
    /// ([`holds_commits`](Self::holds_commits)) and passes for before it sleeps (the module docs).
    pub(crate) fn pass(&self, mut pass: impl FnMut(&mut S, Vec<C>) -> bool) {
        self.passers.fetch_add(1, Ordering::SeqCst);
        let mut state = lock(&self.state);
        let commits = self.take();
        let again = pass(&mut state, commits);
        drop(state);
        self.passers.fetch_sub(1, Ordering::SeqCst);
        if again {
            self.look_again(&mut pass);
        }
    }

    /// After a passer is counted out: while the inbox holds a commit and no other passer is
    /// counted, take the lock and pass for it (waiting only for a holder that makes no pass),
    /// until a pass answers not to look again. Whether this made a pass.
    fn look_again(&self, pass: &mut impl FnMut(&mut S, Vec<C>) -> bool) -> bool {
        let mut passed = false;
        loop {
            fence(Ordering::SeqCst);
            if !self.posted.load(Ordering::SeqCst) || self.passers.load(Ordering::SeqCst) > 0 {
                return passed;
            }
            self.passers.fetch_add(1, Ordering::SeqCst);
            let mut state = match self.state.try_lock() {
                Ok(state) => state,
                Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                // Another passer took it since: it looks again once done.
                Err(TryLockError::WouldBlock) if self.passers.load(Ordering::SeqCst) > 1 => {
                    self.passers.fetch_sub(1, Ordering::SeqCst);
                    continue;
                }
                // A short hold that makes no pass.
                Err(TryLockError::WouldBlock) => lock(&self.state),
            };
            let commits = self.take();
            let again = pass(&mut state, commits);
            drop(state);
            passed = true;
            self.passers.fetch_sub(1, Ordering::SeqCst);
            if !again {
                return passed;
            }
        }
    }

    /// The commits in the inbox, in commit order, taken out; called under the state's lock.
    fn take(&self) -> Vec<C> {
        if !self.posted.load(Ordering::SeqCst) {
            return Vec::new();
        }
        let mut commits = lock(&self.commits);
        self.posted.store(false, Ordering::SeqCst);
        commits.drain(..).collect()
    }
}
