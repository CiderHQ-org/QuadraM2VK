//! `trails.rs`: a wait for the trails below a mark waits for those opened before the mark alone,
//! and returns as the last of them closes.

use crate::trails::{Ended, Trails};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

/// The longest a test waits for another thread.
const WAIT: Duration = Duration::from_secs(60);

/// A trail opened before the mark holds a wait at the mark; one opened after does not, and the
/// mark is the number the next trail takes.
#[test]
fn a_wait_waits_for_the_trails_opened_before_its_mark_alone() {
    let trails = Arc::new(Trails::default());
    assert_eq!(trails.mark(), 0);
    assert_eq!(
        trails.wait_below(0, Some(Instant::now())),
        Ok(()),
        "a mark of 0 waits for none"
    );
    let before = trails.open();
    let mark = trails.mark();
    assert_eq!(mark, 1);
    let after = trails.open();
    assert_eq!(trails.open_now(), 2);
    assert_eq!(
        trails.wait_below(mark, Some(Instant::now())),
        Err(Ended::TimedOut),
        "the trail opened before the mark holds the wait"
    );
    drop(before);
    assert_eq!(
        trails.wait_below(mark, Some(Instant::now())),
        Ok(()),
        "the trail opened after the mark does not"
    );
    drop(after);
    assert_eq!(trails.open_now(), 0);
}

/// A blocked wait returns once the last trail below its mark closes, on another thread.
#[test]
fn a_blocked_wait_returns_as_the_trail_closes() {
    let trails = Arc::new(Trails::default());
    let trail = trails.open();
    let mark = trails.mark();
    let (started, waiting) = mpsc::channel();
    let waiter = {
        let trails = Arc::clone(&trails);
        std::thread::spawn(move || {
            started.send(()).unwrap();
            trails.wait_below(mark, Some(Instant::now() + WAIT))
        })
    };
    waiting.recv_timeout(WAIT).unwrap();
    drop(trail);
    assert_eq!(waiter.join().unwrap(), Ok(()));
}
