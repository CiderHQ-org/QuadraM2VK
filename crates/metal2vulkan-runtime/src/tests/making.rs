//! `pipeline/making.rs`: distinct keys made together, one key made once, a failed or panicking
//! making made again, the memo bounded, a making settled on another thread, and a making
//! waited for run first in the device's pool. Here, with the tests' own threads, which the
//! runtime's sources start only through the exit guard
//! (`exit_guard::threads_and_the_program_s_code_go_through_the_guard`).

use crate::error::{internal, Error};
use crate::pipeline::budget::Budget;
use crate::pipeline::making::{Begun, Making};
use crate::pipeline::pool::Pool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// How long a making waits for the others before the test calls the makings serial: a guard
/// against a hang, never reached when the makings overlap.
const GUARD: Duration = Duration::from_secs(60);

/// Makings that wait until `n` of them are in flight together: each returns whether they all
/// were, which only concurrent makings can be.
struct Rendezvous {
    n: usize,
    in_flight: Mutex<usize>,
    arrived: Condvar,
}

impl Rendezvous {
    fn make(&self) -> bool {
        let mut in_flight = self.in_flight.lock().unwrap();
        *in_flight += 1;
        self.arrived.notify_all();
        let (in_flight, timeout) = self
            .arrived
            .wait_timeout_while(in_flight, GUARD, |count| *count < self.n)
            .unwrap();
        drop(in_flight);
        !timeout.timed_out()
    }
}

/// `n` keys a thread each asks for at once are `n` makings in flight together: none waits for
/// another's driver call.
#[test]
fn distinct_keys_are_made_together() {
    const N: usize = 8;
    let memo: Arc<Making<usize, bool>> = Arc::new(Making::counted(32));
    let meeting = Arc::new(Rendezvous {
        n: N,
        in_flight: Mutex::new(0),
        arrived: Condvar::new(),
    });
    let threads: Vec<_> = (0..N)
        .map(|key| {
            let (memo, meeting) = (Arc::clone(&memo), Arc::clone(&meeting));
            std::thread::spawn(move || *memo.get_or_make(&key, || Ok(meeting.make())).unwrap())
        })
        .collect();
    let together: Vec<bool> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(
        together,
        vec![true; N],
        "every making met the others in flight"
    );
}

/// Many callers of one key make it once and share it, whatever their order.
#[test]
fn one_key_is_made_once() {
    const N: usize = 8;
    let memo: Arc<Making<u32, u32>> = Arc::new(Making::counted(32));
    let makings = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..N)
        .map(|_| {
            let (memo, makings) = (Arc::clone(&memo), Arc::clone(&makings));
            std::thread::spawn(move || {
                memo.get_or_make(&7, || {
                    makings.fetch_add(1, Ordering::SeqCst);
                    Ok(49)
                })
                .unwrap()
            })
        })
        .collect();
    let made: Vec<Arc<u32>> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(makings.load(Ordering::SeqCst), 1, "made once");
    assert!(made.iter().all(|m| Arc::ptr_eq(m, &made[0])), "one object");
}

/// A failed making's caller gets its error and the memo keeps nothing: the next caller makes
/// it again. A making that panics leaves the memo alike.
#[test]
fn a_failed_or_panicking_making_is_made_again() {
    let memo: Making<u32, u32> = Making::counted(4);
    let failed: Result<_, Error> = memo.get_or_make(&1, || Err(internal("refused")));
    assert_eq!(failed, Err(internal("refused")));
    assert_eq!(*memo.get_or_make(&1, || Ok(10)).unwrap(), 10);
    // Made now: no further making.
    assert_eq!(*memo.get_or_make(&1, || Ok(11)).unwrap(), 10);
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        memo.get_or_make(&2, || panic!("a making that panics"))
    }));
    assert!(panicked.is_err());
    assert_eq!(*memo.get_or_make(&2, || Ok(20)).unwrap(), 20);
}

/// The memo keeps at most its count, the least recently used going first; a key it let go is
/// made again.
#[test]
fn the_memo_is_bounded() {
    let memo: Making<u32, u32> = Making::counted(2);
    for key in 0..3 {
        memo.get_or_make(&key, || Ok(key)).unwrap();
    }
    assert_eq!(
        *memo.get_or_make(&0, || Ok(100)).unwrap(),
        100,
        "0 went first"
    );
    assert_eq!(*memo.get_or_make(&2, || Ok(200)).unwrap(), 2, "2 stayed");
}

/// A making begun here and settled on another thread reaches every caller of its key: one that
/// looked at it before, one that waits for it in `get_or_make`, its wakers once, each given the
/// object, and later callers, which find it made.
#[test]
fn a_making_settled_elsewhere_reaches_its_callers() {
    let memo: Arc<Making<u32, u32>> = Arc::new(Making::counted(4));
    let Begun::Yours(ticket) = memo.begin(&1) else {
        panic!("the first caller's making");
    };
    let looked = ticket.pending();
    assert!(looked.outcome().is_none(), "in flight");
    let Begun::Pending(other) = memo.begin(&1) else {
        panic!("the making in flight");
    };
    let (woke, wakes) = mpsc::channel();
    let early = woke.clone();
    // Each waker is given the making's outcome.
    other.when_finished(Box::new(move |made| {
        let given = made.as_ref().map(|made| **made).ok();
        early.send(("early", given)).unwrap();
    }));
    // A caller of the key waits for the making in flight and makes nothing itself.
    let waiter = {
        let memo = Arc::clone(&memo);
        std::thread::spawn(move || *memo.get_or_make(&1, || Ok(99)).unwrap())
    };
    let settler = std::thread::spawn(move || *ticket.settle(Ok(5)).unwrap());
    assert_eq!(settler.join().unwrap(), 5);
    assert_eq!(
        waiter.join().unwrap(),
        5,
        "the waiter took the settled object"
    );
    assert_eq!(wakes.recv_timeout(GUARD), Ok(("early", Some(5))));
    assert_eq!(*looked.wait().unwrap(), 5);
    assert_eq!(*other.outcome().unwrap().unwrap(), 5);
    // A waker registered once it finished runs at once.
    other.when_finished(Box::new(move |made| {
        let given = made.as_ref().map(|made| **made).ok();
        woke.send(("late", given)).unwrap();
    }));
    assert_eq!(wakes.try_recv(), Ok(("late", Some(5))));
    assert!(wakes.try_recv().is_err(), "each waker ran once");
    match memo.begin(&1) {
        Begun::Made(made) => assert_eq!(*made, 5),
        _ => panic!("a settled making is kept"),
    }
}

/// A ticket dropped unsettled fails its making, as a failed making does: its callers see the
/// error, the memo keeps nothing, and the next caller makes the key again.
#[test]
fn a_ticket_dropped_unsettled_fails_its_making() {
    let memo: Making<u32, u32> = Making::counted(4);
    let Begun::Yours(ticket) = memo.begin(&1) else {
        panic!("the first caller's making");
    };
    let looked = ticket.pending();
    let (woke, wakes) = mpsc::channel();
    looked.when_finished(Box::new(move |_| woke.send(()).unwrap()));
    drop(ticket);
    assert_eq!(wakes.try_recv(), Ok(()), "its wakers ran");
    assert_eq!(
        looked.outcome().map(|outcome| outcome.map(|made| *made)),
        Some(Err(internal(
            "a pipeline's making ended without its object"
        )))
    );
    assert!(
        matches!(memo.begin(&1), Begun::Yours(_)),
        "the memo kept nothing"
    );
    assert_eq!(*memo.get_or_make(&1, || Ok(3)).unwrap(), 3);
}

/// A making handed to the pool behind `MAKERS` pieces, every thread busy, goes first once a
/// caller waits for it, and runs before them once a thread is free; the rest keep their order.
/// The pool is within a budget of its own whose turns let `MAKERS` pieces nothing waits for
/// run at once: the process's lets fewer (all but one of its turns), and the other tests make in
/// it.
#[test]
fn a_making_waited_for_goes_before_the_pool_s_queued_pieces() {
    /// The turns of the test's budget, so the threads of the pool within it.
    const MAKERS: usize = 4;
    let pool = Pool::within(Arc::new(Budget::new(MAKERS, MAKERS)));
    // Pieces that hold every thread, one let go per message.
    let (release, gate) = mpsc::channel::<()>();
    let gate = Arc::new(Mutex::new(gate));
    let (started, busy) = mpsc::channel();
    for _ in 0..MAKERS {
        let (gate, started) = (Arc::clone(&gate), started.clone());
        pool.run(Box::new(move || {
            started.send(()).unwrap();
            gate.lock().unwrap().recv_timeout(GUARD).unwrap();
        }));
    }
    for _ in 0..MAKERS {
        busy.recv_timeout(GUARD).unwrap();
    }
    let (ran, order) = mpsc::channel();
    for at in 0..MAKERS {
        let ran = ran.clone();
        pool.run(Box::new(move || ran.send(at).unwrap()));
    }
    let memo: Making<u32, usize> = Making::counted(4);
    let Begun::Yours(ticket) = memo.begin(&1) else {
        panic!("the first caller's making");
    };
    let pending = ticket.pending();
    let place = pool.run(Box::new(move || {
        ran.send(MAKERS).unwrap();
        drop(ticket.settle(Ok(MAKERS)));
    }));
    pending.queued_at(place.clone());
    assert!(!place.is_first(), "queued last");
    let waiter = {
        let pending = pending.clone();
        std::thread::spawn(move || *pending.wait().unwrap())
    };
    let began = Instant::now();
    while !place.is_first() {
        assert!(began.elapsed() < GUARD, "the wait hastened its making");
        std::thread::yield_now();
    }
    // One thread let go runs every piece waiting, in the pool's order.
    release.send(()).unwrap();
    assert_eq!(
        order.recv_timeout(GUARD),
        Ok(MAKERS),
        "the making ran first"
    );
    assert_eq!(waiter.join().unwrap(), MAKERS);
    let rest: Vec<usize> = (0..MAKERS)
        .map(|_| order.recv_timeout(GUARD).unwrap())
        .collect();
    assert_eq!(rest, (0..MAKERS).collect::<Vec<_>>());
    for _ in 1..MAKERS {
        release.send(()).unwrap();
    }
}
