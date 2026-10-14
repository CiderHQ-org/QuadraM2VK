//! `scheduler/inbox.rs`: a commit that finds a pass under way on another thread returns at once
//! and its commit goes next, in commit order; one that finds the lock held for no pass passes
//! itself; the scheduler's pass looks again for a commit left to it until a pass answers it has
//! work of its own first; and commits from many threads racing passes, ones that stop looking
//! again, and short holds are all passed, each thread's in its order, with no pass made for them
//! at the end; and a nudge finding a pass under way returns at once and is passed for once that
//! pass ends.

use crate::device::lock;
use crate::scheduler::inbox::Inbox;
use std::sync::mpsc;
use std::sync::Arc;
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
                done.send(inbox.commit(commit, take_in)).unwrap();
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
    assert!(
        !passer.join().unwrap(),
        "the passer's own commit was its own"
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
    assert!(!committer.join().unwrap(), "it passed its commit itself");
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
        assert!(inbox.commit(1, take_in), "left to the pass under way");
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

#[test]
fn commits_racing_passes_and_short_holds_are_all_passed_each_thread_s_in_its_order() {
    const THREADS: u32 = 4;
    const COMMITS: u32 = 2000;
    let inbox = Arc::new(Log::new(Vec::new()));
    // A pass that holds the lock a while, so that commits find it under way.
    let slow = |log: &mut Vec<u32>, commits: Vec<u32>| {
        for _ in 0..200 {
            std::hint::spin_loop();
        }
        log.extend(commits);
    };
    let committers: Vec<_> = (0..THREADS)
        .map(|index| {
            let inbox = Arc::clone(&inbox);
            thread::spawn(move || {
                (0..COMMITS)
                    .filter(|&commit| inbox.commit(index * COMMITS + commit, slow))
                    .count()
            })
        })
        .collect();
    // The scheduler's thread: passes of its own, every other one answering it has work first,
    // which the thread's next pass follows, as its own look before it sleeps would; the last one
    // looks again.
    let scheduler = {
        let inbox = Arc::clone(&inbox);
        thread::spawn(move || {
            for round in 0..COMMITS {
                inbox.pass(|log, commits| {
                    slow(log, commits);
                    round % 2 == 0 || round + 1 == COMMITS
                });
                thread::yield_now();
            }
        })
    };
    // A wake, a decode count: short holds that make no pass.
    let marker = {
        let inbox = Arc::clone(&inbox);
        thread::spawn(move || {
            for _ in 0..COMMITS {
                drop(lock(inbox.state()));
                thread::yield_now();
            }
        })
    };
    let left: usize = committers.into_iter().map(|c| c.join().unwrap()).sum();
    scheduler.join().unwrap();
    marker.join().unwrap();
    // No pass made for them at the end: every commit was passed by then.
    assert!(!inbox.holds_commits(), "{left} left to other threads");
    let log = lock(inbox.state());
    assert_eq!(log.len(), (THREADS * COMMITS) as usize, "each once");
    for index in 0..THREADS {
        let own: Vec<u32> = (log.iter().copied())
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
