//! `command_buffer/gate.rs`: a command buffer's waiters return once it finished and every
//! runtime thread let go of it, and a hold lets go of its value before the gate opens.

use crate::command_buffer::gate::{Exiting, Gate, Hold};
use crate::error::{DeadlockCause, Error, TimedOut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A held value that records, when it goes, whether its gate was open then.
struct Probe {
    gate: Arc<Gate>,
    gone: Arc<AtomicBool>,
    gone_while_open: Arc<AtomicBool>,
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.gone_while_open
            .store(self.gate.is_open(), Ordering::SeqCst);
        self.gone.store(true, Ordering::SeqCst);
    }
}

/// A probe, the gate it records, and its two flags.
fn probe() -> (Arc<Probe>, Arc<Gate>, Arc<AtomicBool>, Arc<AtomicBool>) {
    let gate = Arc::new(Gate::default());
    let gone = Arc::new(AtomicBool::new(false));
    let gone_while_open = Arc::new(AtomicBool::new(false));
    let probe = Arc::new(Probe {
        gate: Arc::clone(&gate),
        gone: Arc::clone(&gone),
        gone_while_open: Arc::clone(&gone_while_open),
    });
    (probe, gate, gone, gone_while_open)
}

#[test]
fn the_gate_opens_once_finished_and_every_hold_is_gone() {
    let gate = Arc::new(Gate::default());
    let value = Arc::new(7_u32);
    let past = Instant::now();
    assert!(!gate.is_open());
    assert_eq!(gate.wait_until(past), Err(TimedOut));

    // Holds gone before the finish: the gate opens at the finish.
    drop(Hold::new(&value, &gate));
    assert!(!gate.is_open());

    let first = Hold::new(&value, &gate);
    let second = Hold::new(&value, &gate);
    assert!(first.same(&second));
    assert_eq!(**first, 7);
    gate.finish();
    assert!(!gate.is_open());
    assert_eq!(gate.wait_until(past), Err(TimedOut));
    drop(first);
    assert!(!gate.is_open());
    drop(second);
    assert!(gate.is_open());
    assert_eq!(gate.wait_until(past), Ok(()));
    assert_eq!(gate.wait(), Ok(()));

    let other = Arc::new(7_u32);
    let gate = Arc::new(Gate::default());
    let held = Hold::new(&value, &gate);
    assert!(!held.same(&Hold::new(&other, &gate)));
    gate.finish();
    assert!(!gate.is_open());
    drop(held);
    assert!(gate.is_open());
}

/// The last reference a runtime thread holds goes before the gate opens: what the program's
/// waiter then drops is the last of it.
#[test]
fn a_hold_lets_go_of_its_value_before_the_gate_opens() {
    let (value, gate, gone, gone_while_open) = probe();
    let held = Hold::new(&value, &gate);
    // The program dropped its own reference; the runtime's is the last.
    drop(value);
    gate.finish();
    assert!(!gone.load(Ordering::SeqCst));
    drop(held);
    assert!(gone.load(Ordering::SeqCst));
    assert!(!gone_while_open.load(Ordering::SeqCst));
    assert!(gate.is_open());
}

/// A waiter blocked on the gate returns only after the finish and the last hold, and then sees
/// the held value gone.
#[test]
fn a_waiter_returns_after_the_last_hold_is_gone() {
    let (value, gate, gone, gone_while_open) = probe();
    let held = Hold::new(&value, &gate);
    drop(value);
    let waiter = {
        let gate = Arc::clone(&gate);
        let gone = Arc::clone(&gone);
        std::thread::spawn(move || {
            assert_eq!(gate.wait(), Ok(()));
            gone.load(Ordering::SeqCst)
        })
    };
    gate.finish();
    assert_eq!(
        gate.wait_until(Instant::now() + Duration::from_millis(50)),
        Err(TimedOut)
    );
    assert!(!waiter.is_finished());
    drop(held);
    assert!(
        waiter.join().unwrap(),
        "the waiter returned before the hold went"
    );
    assert!(!gone_while_open.load(Ordering::SeqCst));
}

/// Once the process is exiting, a wait for a gate that is not open answers at once, and one
/// already blocked within `exit_guard::WAKE`: `WaitWouldDeadlock` of `DeadlockCause::Exiting`.
#[test]
fn a_wait_once_the_process_is_exiting_is_refused_for_that_cause() {
    let gate = Arc::new(Gate::default());
    assert_eq!(gate.wait_unless(|| true), Err(Exiting));
    let exiting = Arc::new(AtomicBool::new(false));
    let waiter = {
        let (gate, exiting) = (Arc::clone(&gate), Arc::clone(&exiting));
        std::thread::spawn(move || gate.wait_unless(|| exiting.load(Ordering::SeqCst)))
    };
    std::thread::sleep(Duration::from_millis(20));
    assert!(!waiter.is_finished());
    let start = Instant::now();
    exiting.store(true, Ordering::SeqCst);
    let answer = waiter.join().unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "the blocked wait answered late"
    );
    let refusal = Error::from(answer.unwrap_err());
    assert_eq!(refusal, Error::WaitWouldDeadlock(DeadlockCause::Exiting));
    // An open gate answers whatever the exit.
    gate.finish();
    assert_eq!(gate.wait_unless(|| true), Ok(()));
}

/// A gate is awaited only while a thread is blocked in `wait` or `wait_until`, whichever way the
/// wait ends: what the finishing thread does before its hold goes delays only such a thread.
#[test]
fn a_gate_is_awaited_while_a_thread_is_blocked_on_it() {
    let gate = Arc::new(Gate::default());
    let value = Arc::new(7_u32);
    assert!(!gate.awaited());
    assert_eq!(gate.wait_until(Instant::now()), Err(TimedOut));
    assert!(!gate.awaited(), "a wait that timed out is not counted");
    let held = Hold::new(&value, &gate);
    let waiters: Vec<_> = (0..2)
        .map(|n| {
            let gate = Arc::clone(&gate);
            std::thread::spawn(move || match n {
                0 => gate.wait().is_ok(),
                _ => gate
                    .wait_until(Instant::now() + Duration::from_secs(60))
                    .is_ok(),
            })
        })
        .collect();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !gate.awaited() {
        assert!(Instant::now() < deadline, "no waiter blocked");
        std::thread::yield_now();
    }
    gate.finish();
    assert!(
        gate.awaited(),
        "the gate is not open while the hold is alive"
    );
    drop(held);
    for waiter in waiters {
        assert!(waiter.join().unwrap());
    }
    assert!(!gate.awaited());
    assert_eq!(gate.wait(), Ok(()));
    assert!(!gate.awaited(), "a wait of an open gate is not counted");
}
