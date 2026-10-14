//! `scheduler/inbox.rs`: a commit that finds a pass under way on another thread returns at once
//! and its commit goes next, in commit order; one that finds the lock held for no pass passes
//! itself; the scheduler's pass looks again for a commit left to it until a pass answers it has
//! work of its own first; a committing thread passes once for what others left it, and leaves the
//! rest to the thread its caller wakes; commits from many threads racing passes, ones that stop
//! looking again, wakes and short holds are all passed, each thread's in its order, with no pass
//! made for them at the end; a nudge finding a pass under way returns at once and is passed for
//! once that pass ends; and wakers racing passes and commits have every change they made read by
//! a pass.

use crate::device::lock;
use crate::scheduler::inbox::{Inbox, Outcome, AGAIN};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar};
use std::thread;
use std::time::{Duration, Instant};

/// The longest a step of these tests may take: a commit that waits for the pass under way
/// instead of returning fails the test after it, rather than hanging it.
const WAIT: Duration = Duration::from_secs(30);

/// The state: the commits in the order the passes took them in.
type Log = Inbox<Vec<u32>, u32>;

/// A pass that only takes the commits in.
fn take_in(log: &mut Vec<u32>, commits: Vec<u32>) {
    log.extend(commits);
}

/// Commit `commit` from another thread while this one holds `inbox`'s lock for a pass: what
/// became of it. It is left to this thread's pass, so it makes none itself.
pub(super) fn commit_elsewhere<S: Send, C: Send>(inbox: &Inbox<S, C>, commit: C) -> Outcome {
    thread::scope(|scope| {
        let committer = scope.spawn(|| {
            inbox.commit(commit, |_, _| {
                panic!("a commit that found a pass under way passed itself")
            })
        });
        committer.join().unwrap()
    })
}

#[test]
fn a_commit_finding_a_pass_under_way_returns_at_once_and_goes_next_in_commit_order() {
    let inbox = Arc::new(Log::new(Vec::new()));
    let (entered, in_pass) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let passer = {
        let inbox = Arc::clone(&inbox);
        thread::spawn(move || {
            let mut passes = 0;
            inbox.commit(0, |log, commits| {
                take_in(log, commits);
                passes += 1;
                // The first pass stays under way, the lock held, until the test lets it end.
                if passes == 1 {
                    entered.send(()).unwrap();
                    released.recv_timeout(WAIT).unwrap();
                }
            })
        })
    };
    in_pass.recv_timeout(WAIT).unwrap();
    let (done, returned) = mpsc::channel();
    let committer = {
        let inbox = Arc::clone(&inbox);
        thread::spawn(move || {
            for commit in 1..=3 {
                done.send(inbox.commit(commit, take_in).left).unwrap();
            }
        })
    };
    for commit in 1..=3 {
        let left = returned.recv_timeout(WAIT).unwrap_or_else(|_| {
            panic!("commit {commit} waited for the pass under way instead of returning")
        });
        assert!(left, "commit {commit} was left to the pass under way");
    }
    committer.join().unwrap();
    assert!(inbox.holds_commits());
    release.send(()).unwrap();
    assert_eq!(
        passer.join().unwrap(),
        Outcome::default(),
        "the passer's own commit was its own, and it passed for the others"
    );
    // The passer took them in once its pass ended, in commit order, with no pass asked for.
    assert_eq!(*lock(inbox.state()), [0, 1, 2, 3]);
    assert!(!inbox.holds_commits());
}

#[test]
fn a_commit_finding_the_lock_held_for_no_pass_passes_itself() {
    let inbox = Arc::new(Log::new(Vec::new()));
    let held = lock(inbox.state());
    let committer = {
        let inbox = Arc::clone(&inbox);
        thread::spawn(move || inbox.commit(7, take_in))
    };
    // It found the lock taken and posted: no passer is counted, so it waits for the lock.
    let until = Instant::now() + WAIT;
    while !inbox.holds_commits() {
        assert!(Instant::now() < until, "the commit never posted");
        thread::yield_now();
    }
    drop(held);
    assert_eq!(
        committer.join().unwrap(),
        Outcome::default(),
        "it passed its commit itself"
    );
    assert_eq!(*lock(inbox.state()), [7]);
    assert!(!inbox.holds_commits());
}

#[test]
fn the_scheduler_s_pass_looks_again_until_a_pass_answers_it_has_work_first() {
    for again in [true, false] {
        let inbox = Arc::new(Log::new(Vec::new()));
        let (entered, in_pass) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let scheduler = {
            let inbox = Arc::clone(&inbox);
            thread::spawn(move || {
                let mut passes = 0;
                inbox.pass(|log, commits| {
                    take_in(log, commits);
                    passes += 1;
                    // The first pass stays under way until the test lets it end.
                    if passes == 1 {
                        entered.send(()).unwrap();
                        released.recv_timeout(WAIT).unwrap();
                    }
                    again
                });
                passes
            })
        };
        in_pass.recv_timeout(WAIT).unwrap();
        assert!(inbox.commit(1, take_in).left, "left to the pass under way");
        release.send(()).unwrap();
        let passes = scheduler.join().unwrap();
        if again {
            // It looked again and took the commit in.
            assert_eq!(passes, 2);
            assert_eq!(*lock(inbox.state()), [1]);
            assert!(!inbox.holds_commits());
        } else {
            // It stopped: the commit waits for its own look.
            assert_eq!(passes, 1);
            assert!(lock(inbox.state()).is_empty());
            assert!(inbox.holds_commits());
            inbox.pass(|log, commits| {
                take_in(log, commits);
                true
            });
            assert_eq!(*lock(inbox.state()), [1]);
            assert!(!inbox.holds_commits());
        }
    }
}

/// A committing thread makes its own pass and [`AGAIN`] more for what other threads left it while
/// it passed, however many they leave it meanwhile: the last one left waits in the inbox, which it
/// answers was left unattended, for its caller to wake the thread that makes passes itself.
#[test]
fn a_committing_thread_passes_once_for_the_others_and_leaves_the_rest_to_a_wake() {
    let inbox = Log::new(Vec::new());
    let mut passes = 0;
    let outcome = inbox.commit(0, |log, commits| {
        take_in(log, commits);
        passes += 1;
        // While each of its passes is under way, another thread's commit is left to it.
        if passes <= 4 {
            assert!(commit_elsewhere(&inbox, passes).left);
        }
    });
    assert_eq!(
        passes as usize,
        1 + AGAIN,
        "its own pass and {AGAIN} for the others'"
    );
    assert_eq!(
        outcome,
        Outcome {
            left: false,
            unattended: true
        }
    );
    assert_eq!(*lock(inbox.state()), [0, 1]);
    assert!(
        inbox.holds_commits(),
        "the last one left waits for the thread its caller wakes"
    );
    inbox.pass(|log, commits| {
        take_in(log, commits);
        true
    });
    assert_eq!(*lock(inbox.state()), [0, 1, 2]);
    assert!(!inbox.holds_commits());
}

/// The state of [`Model`]'s passes: the commits in the order they took them in, each waker's
/// value as the last pass read it, and a wake's mark.
#[derive(Default)]
struct Seen {
    log: Vec<u32>,
    values: Vec<u64>,
    dirty: bool,
    stop: bool,
}

/// The scheduler's thread and its wakes over an inbox, as `scheduler/mod.rs` makes them.
struct Model {
    inbox: Inbox<Seen, u32>,
    woken: Condvar,
}

impl Model {
    fn new(wakers: usize) -> Model {
        Model {
            inbox: Inbox::new(Seen {
                values: vec![0; wakers],
                ..Seen::default()
            }),
            woken: Condvar::new(),
        }
    }

    /// `Scheduler::commit`'s: commit, and wake the thread where the inbox was left unattended.
    fn commit(&self, commit: u32, pass: impl FnMut(&mut Seen, Vec<u32>)) -> Outcome {
        let outcome = self.inbox.commit(commit, pass);
        if outcome.unattended {
            self.wake();
        }
        outcome
    }

    /// `Scheduler::wake`'s: whether it was left to a pass under way.
    fn wake(&self) -> bool {
        if self.inbox.wake(|seen| seen.dirty = true) {
            self.woken.notify_all();
            return false;
        }
        true
    }

    /// The thread's loop: it waits for a wake's mark, a commit or a nudge left to it, with no
    /// timed look, then passes, its passes answering `again` of their count whether to look again
    /// (false: it has work of its own first, done at once, after which it looks at the inbox
    /// again). Once stopped, it ends when it has nothing to pass for.
    fn run(&self, mut pass: impl FnMut(&mut Seen, Vec<u32>), again: impl Fn(u64) -> bool) {
        let asked = |seen: &Seen| seen.dirty || self.inbox.holds_commits() || self.inbox.nudged();
        let mut passes = 0;
        loop {
            let mut seen = lock(self.inbox.state());
            while !asked(&seen) && !seen.stop {
                seen = self.woken.wait(seen).unwrap();
            }
            if !asked(&seen) {
                return;
            }
            drop(seen);
            self.inbox.pass(|seen, commits| {
                seen.dirty = false;
                pass(seen, commits);
                passes += 1;
                again(passes)
            });
        }
    }

    fn stop(&self) {
        lock(self.inbox.state()).stop = true;
        self.woken.notify_all();
    }
}

/// A pass that holds the lock a while, so that commits and wakes find it under way.
fn slow(seen: &mut Seen, commits: Vec<u32>) {
    for _ in 0..200 {
        std::hint::spin_loop();
    }
    seen.log.extend(commits);
}

/// Whether `done` holds of the model's state within [`WAIT`], read every millisecond.
fn soon(model: &Model, done: impl Fn(&Seen) -> bool) -> bool {
    let until = Instant::now() + WAIT;
    while !done(&lock(model.inbox.state())) {
        if Instant::now() > until {
            return false;
        }
        thread::sleep(Duration::from_millis(1));
    }
    true
}

/// Commits from several threads race the scheduler thread's passes, which answer in turn that
/// they have work first, and short holds: each is passed once, each thread's in its order, by a
/// pass made for it or by the thread its committer woke, with no pass made for them at the end.
#[test]
fn commits_racing_passes_wakes_and_short_holds_are_all_passed_each_thread_s_in_its_order() {
    const THREADS: u32 = 4;
    const COMMITS: u32 = 2000;
    let model = Model::new(0);
    let all = (THREADS * COMMITS) as usize;
    let (left, unattended, passed_all) = thread::scope(|scope| {
        // The scheduler's thread: every other pass answers it has work of its own first.
        let scheduler = scope.spawn(|| model.run(slow, |pass| pass % 2 == 0));
        let committers: Vec<_> = (0..THREADS)
            .map(|index| {
                let model = &model;
                scope.spawn(move || {
                    let outcomes: Vec<Outcome> = (0..COMMITS)
                        .map(|commit| model.commit(index * COMMITS + commit, slow))
                        .collect();
                    let count = |what: fn(&Outcome) -> bool| {
                        outcomes.iter().filter(|outcome| what(outcome)).count()
                    };
                    (count(|o| o.left), count(|o| o.unattended))
                })
            })
            .collect();
        // A decode count: short holds that make no pass.
        let marker = scope.spawn(|| {
            for _ in 0..COMMITS {
                drop(lock(model.inbox.state()));
                thread::yield_now();
            }
        });
        let counts: Vec<(usize, usize)> = (committers.into_iter())
            .map(|committer| committer.join().unwrap())
            .collect();
        marker.join().unwrap();
        // The thread passes for nothing on its own: a commit left unattended waits for its wake.
        let passed_all = soon(&model, |seen| seen.log.len() == all);
        model.stop();
        scheduler.join().unwrap();
        let (left, unattended) = (counts.into_iter()).fold((0, 0), |(l, u), (left, unattended)| {
            (l + left, u + unattended)
        });
        (left, unattended, passed_all)
    });
    let seen = lock(model.inbox.state());
    assert!(
        passed_all,
        "{} of {all} passed ({left} left to other threads, {unattended} left unattended)",
        seen.log.len()
    );
    eprintln!("INBOX-RACE commits: {left} left to other threads, {unattended} left unattended");
    assert!(!model.inbox.holds_commits());
    assert_eq!(seen.log.len(), all, "each once");
    for index in 0..THREADS {
        let own: Vec<u32> = (seen.log.iter().copied())
            .filter(|commit| commit / COMMITS == index)
            .collect();
        let ordered: Vec<u32> = (index * COMMITS..(index + 1) * COMMITS).collect();
        assert_eq!(own, ordered, "thread {index}'s commits in its order");
    }
}

/// A nudge (an event's value rose, a batch completed) that finds a pass under way returns at once,
/// left to it, and the passer passes again once done, with no commit; one that finds no passer
/// answers so, for its caller to wake the thread, and the next pass takes it in.
#[test]
fn a_nudge_finding_a_pass_under_way_is_passed_for_once_it_ends() {
    let inbox = Arc::new(Log::new(Vec::new()));
    let (entered, in_pass) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let passer = {
        let inbox = Arc::clone(&inbox);
        thread::spawn(move || {
            let mut passes = 0;
            inbox.commit(0, |log, commits| {
                take_in(log, commits);
                passes += 1;
                if passes == 1 {
                    entered.send(()).unwrap();
                    released.recv_timeout(WAIT).unwrap();
                }
            });
            passes
        })
    };
    in_pass.recv_timeout(WAIT).unwrap();
    assert!(inbox.nudge(), "the nudge was left to the pass under way");
    release.send(()).unwrap();
    assert_eq!(
        passer.join().unwrap(),
        2,
        "the passer passed again for the nudge"
    );
    assert!(!inbox.nudged());
    assert_eq!(*lock(inbox.state()), [0]);

    assert!(!inbox.nudge(), "no passer: the caller wakes the thread");
    assert!(inbox.nudged());
    inbox.pass(|log, commits| {
        take_in(log, commits);
        false
    });
    assert!(!inbox.nudged(), "the next pass took it in");
}

/// Wakers each raise a value of their own and wake for a pass (an event's value rising, a batch
/// completing), racing the scheduler thread's passes, committing threads' passes and their wakes:
/// once they are done, a pass read every waker's last value, made for its wake, with no pass made
/// for it at the end. A nudge taken in by a pass is read with what its waker changed before it.
#[test]
fn wakers_racing_passes_and_commits_have_every_change_read_by_a_pass() {
    const WAKERS: usize = 3;
    const RAISES: u64 = 3000;
    const THREADS: u32 = 2;
    const COMMITS: u32 = 2000;
    let model = Model::new(WAKERS);
    let values: Vec<AtomicU64> = (0..WAKERS).map(|_| AtomicU64::new(0)).collect();
    let read = |seen: &mut Seen, commits: Vec<u32>| {
        for _ in 0..100 {
            std::hint::spin_loop();
        }
        seen.log.extend(commits);
        // Raised with no ordering of their own: only the wake orders a raise before a pass.
        for (seen, value) in seen.values.iter_mut().zip(&values) {
            *seen = value.load(Ordering::Relaxed);
        }
    };
    let (read_all, left) = thread::scope(|scope| {
        // Every third pass answers it has work of its own first.
        let scheduler = scope.spawn(|| model.run(read, |pass| pass % 3 != 0));
        let committers: Vec<_> = (0..THREADS)
            .map(|index| {
                let model = &model;
                scope.spawn(move || {
                    for commit in 0..COMMITS {
                        model.commit(index * COMMITS + commit, read);
                    }
                })
            })
            .collect();
        let wakers: Vec<_> = (0..WAKERS)
            .map(|waker| {
                let (model, value) = (&model, &values[waker]);
                scope.spawn(move || {
                    let mut left = 0;
                    for raise in 1..=RAISES {
                        value.store(raise, Ordering::Relaxed);
                        left += u64::from(model.wake());
                        if raise % 8 == 0 {
                            thread::yield_now();
                        }
                    }
                    left
                })
            })
            .collect();
        for thread in committers {
            thread.join().unwrap();
        }
        let left: u64 = (wakers.into_iter())
            .map(|thread| thread.join().unwrap())
            .sum();
        // The last raises' wakes were passed for; the thread passes on its own for nothing.
        let read_all = soon(&model, |seen| seen.values == [RAISES; WAKERS]);
        model.stop();
        scheduler.join().unwrap();
        (read_all, left)
    });
    let seen = lock(model.inbox.state());
    let wakes = RAISES * WAKERS as u64;
    eprintln!("INBOX-RACE wakes: {left} of {wakes} left to a pass under way");
    assert!(
        read_all,
        "a wake was lost: the passes read {:?} of {RAISES}",
        seen.values
    );
    assert_eq!(seen.log.len(), (THREADS * COMMITS) as usize, "each once");
    assert!(!model.inbox.holds_commits() && !model.inbox.nudged());
}
