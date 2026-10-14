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
//! - **A committing thread passes once for the others** ([`Inbox::commit`]). Its look again makes
//!   at most one pass ([`AGAIN`]): its own commit's, where it found the lock held for no pass, or
//!   else one for what other threads left to its pass. Where that leaves the inbox asked (a
//!   commit, or a nudge, left to the pass meanwhile) with no other passer counted, it answers so
//!   ([`Outcome::unattended`]), and its caller wakes the thread that makes passes itself (the
//!   scheduler's), as for a nudge that finds no passer: so a commit returns after at most two
//!   passes, and never passes on for the commits other threads keep leaving it. One so left waits
//!   for that wake, or, where the scheduler's thread is running a check or a host step, for the
//!   end of that piece of work, unless another commit takes it in first.
//! - **The scheduler's thread looks again until its passes leave it work** ([`Inbox::pass`]).
//!   Its passes may leave it work to do without the lock first (a submission check, a host step),
//!   which the batches behind it wait for: it stops looking again at the first pass that leaves
//!   some, and before each piece of that work, passes again for as long as the inbox holds
//!   commits ([`Inbox::holds_commits`], `scheduler/left.rs`), and looks before it sleeps. A commit
//!   left to it, while it holds the lock for a pass, so waits for that pass, the submission
//!   handlers the thread runs right after it, and the pass that takes it in; never for a wake or
//!   the thread's work.
//! - **Who runs the after-work.** A pass's thread lets go, outside the lock, of what its passes
//!   submitted (the holds, the submission handlers) in submission order, as the scheduler does
//!   for its own: a commit handed off runs its submission handlers on that thread.
//! - **A wake is handed off too** ([`Inbox::nudge`]). What a pass reads may change without a
//!   commit: an event's value rises, a batch completes (a check waits for its signal to land).
//!   Where a passer is counted, the waker leaves it a nudge and returns, as a commit does: the
//!   passer looks again for it once done, by the same fences, and each pass takes it in first.
//!   The completion thread so never waits for a pass's submissions before it finishes the batches
//!   it saw complete, whose completed handlers wait for it. A pass takes the nudge in by a
//!   sequentially consistent swap, which reads the waker's store where it takes that nudge, and
//!   so sees what the waker changed before it.

use crate::device::lock;
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, TryLockError};

/// The passes a committing thread's look again makes at most ([`Inbox::commit`]).
pub(crate) const AGAIN: usize = 1;

/// What became of a commit ([`Inbox::commit`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Outcome {
    /// It was left to another thread's pass.
    pub(crate) left: bool,
    /// The committing thread stopped looking again at [`AGAIN`] passes with the inbox asked and
    /// no other passer counted: the caller wakes the thread that makes passes itself, which takes
    /// in what it holds.
    pub(crate) unattended: bool,
}

/// A state `S` that passes work on under its lock, and the commits `C` waiting to be taken in.
pub(crate) struct Inbox<S, C> {
    state: Mutex<S>,
    /// Commits not yet taken in, in commit order.
    commits: Mutex<Vec<C>>,
    /// `commits` holds one (read without its lock; written under it).
    posted: AtomicBool,
    /// A waker asked for a pass since the last one took the inbox in ([`Inbox::nudge`]).
    nudged: AtomicBool,
    /// The threads making a pass or trying the lock for one.
    passers: AtomicUsize,
}

impl<S, C> Inbox<S, C> {
    pub(crate) fn new(state: S) -> Inbox<S, C> {
        Inbox {
            state: Mutex::new(state),
            commits: Mutex::new(Vec::new()),
            posted: AtomicBool::new(false),
            nudged: AtomicBool::new(false),
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

    /// Whether a waker asked for a pass that no pass took in yet.
    pub(crate) fn nudged(&self) -> bool {
        self.nudged.load(Ordering::SeqCst)
    }

    /// Ask for a pass: what a pass reads changed. Where a passer is counted, it looks again once
    /// done and passes for this, as for a commit, and this answers true. Else false: the caller
    /// wakes the thread that makes passes itself, under the lock, which only a short hold or a
    /// passer counted since (whose pass, made after this, reads the change) may have.
    pub(crate) fn nudge(&self) -> bool {
        self.nudged.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        self.passing()
    }

    /// Whether a passer is counted.
    fn passing(&self) -> bool {
        self.passers.load(Ordering::SeqCst) > 0
    }

    /// Wake for a pass: [`nudge`](Self::nudge), and where that finds no passer, `mark` the state
    /// under the lock, which then only a short hold or a passer counted since (whose pass, made
    /// after the nudge, reads the change) may have. Whether it marked the state, for the caller to
    /// wake the thread that makes passes itself: the waker never waits for a pass's submissions.
    pub(crate) fn wake(&self, mark: impl FnOnce(&mut S)) -> bool {
        if self.nudge() {
            return false;
        }
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            // A pass counted since: it took the nudge in, after the change.
            Err(TryLockError::WouldBlock) if self.passing() => return false,
            Err(TryLockError::WouldBlock) => lock(&self.state),
        };
        mark(&mut state);
        true
    }

    /// Commit `commit`: where the lock is free, make the pass on this thread, `pass` taking in
    /// what the inbox held and `commit` after it, in commit order; where another thread holds it,
    /// leave `commit` to that thread's pass. Then look at the inbox again ([`look_again`]), making
    /// at most [`AGAIN`] passes. Whether `commit` was left to another thread, and whether the
    /// inbox was left unattended (the module docs).
    ///
    /// [`look_again`]: Self::look_again
    pub(crate) fn commit(&self, commit: C, mut pass: impl FnMut(&mut S, Vec<C>)) -> Outcome {
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
        let looked = self.look_again(AGAIN, &mut pass);
        Outcome {
            left: left && !looked.passed,
            unattended: looked.unattended,
        }
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
            self.look_again(usize::MAX, &mut pass);
        }
    }

    /// After a passer is counted out: while the inbox holds a commit or a nudge and no other
    /// passer is counted, take the lock and pass for it (waiting only for a holder that makes no
    /// pass), until a pass answers not to look again, or `passes` passes were made. Whether this
    /// made a pass that took commits in, and whether it stopped at `passes` with the inbox still
    /// asked and no other passer counted.
    fn look_again(&self, passes: usize, pass: &mut impl FnMut(&mut S, Vec<C>) -> bool) -> Looked {
        let mut looked = Looked::default();
        let mut made = 0;
        loop {
            fence(Ordering::SeqCst);
            let asked = self.posted.load(Ordering::SeqCst) || self.nudged.load(Ordering::SeqCst);
            if !asked || self.passers.load(Ordering::SeqCst) > 0 {
                return looked;
            }
            if made == passes {
                looked.unattended = true;
                return looked;
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
            made += 1;
            let commits = self.take();
            looked.passed |= !commits.is_empty();
            let again = pass(&mut state, commits);
            drop(state);
            self.passers.fetch_sub(1, Ordering::SeqCst);
            if !again {
                return looked;
            }
        }
    }

    /// The commits in the inbox, in commit order, taken out, and its nudge with them; called under
    /// the state's lock, before the pass reads anything. The nudge is taken by a swap: where it
    /// reads a waker's store, it synchronizes with it, so the pass reads what the waker changed
    /// before its nudge; where it reads none, any later store of one stays for a later look.
    fn take(&self) -> Vec<C> {
        self.nudged.swap(false, Ordering::SeqCst);
        if !self.posted.load(Ordering::SeqCst) {
            return Vec::new();
        }
        let mut commits = lock(&self.commits);
        self.posted.store(false, Ordering::SeqCst);
        commits.drain(..).collect()
    }
}

/// What a look again came to ([`Inbox::look_again`]).
#[derive(Default)]
struct Looked {
    /// A pass of it took commits in.
    passed: bool,
    /// It stopped at its passes with the inbox asked and no other passer counted.
    unattended: bool,
}
