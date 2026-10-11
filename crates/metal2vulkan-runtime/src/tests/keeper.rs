//! `device/keeper.rs`: one kept value per key, made once under one lock, and nothing kept when
//! making fails.

use crate::device::keeper::Kept;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

#[test]
fn a_key_keeps_the_first_value_made_and_makes_no_other() {
    let kept = Kept::new();
    let made = AtomicUsize::new(0);
    let make = || {
        made.fetch_add(1, Ordering::SeqCst);
        Some(())
    };
    assert!(kept.keep(1usize, make));
    assert!(kept.keep(1, make));
    assert_eq!(
        made.load(Ordering::SeqCst),
        1,
        "the second call made another"
    );
    assert_eq!(kept.len(), 1);
    assert!(kept.keep(2, make));
    assert_eq!(made.load(Ordering::SeqCst), 2);
    assert_eq!(kept.len(), 2);
}

#[test]
fn a_failed_make_keeps_nothing_and_the_next_call_tries_again() {
    let kept = Kept::new();
    assert!(!kept.keep(1usize, || None::<()>));
    assert_eq!(kept.len(), 0);
    assert!(kept.keep(1, || Some(())));
    assert_eq!(kept.len(), 1);
}

#[test]
fn concurrent_first_calls_make_one_value() {
    const THREADS: usize = 8;
    let kept = Arc::new(Kept::new());
    let made = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(THREADS));
    let threads: Vec<_> = (0..THREADS)
        .map(|_| {
            let (kept, made, start) = (Arc::clone(&kept), Arc::clone(&made), Arc::clone(&start));
            std::thread::spawn(move || {
                start.wait();
                kept.keep(7usize, || {
                    made.fetch_add(1, Ordering::SeqCst);
                    // Long enough for every other thread to reach the lock.
                    std::thread::sleep(Duration::from_millis(20));
                    Some(())
                })
            })
        })
        .collect();
    for thread in threads {
        assert!(thread.join().unwrap());
    }
    assert_eq!(
        made.load(Ordering::SeqCst),
        1,
        "concurrent first calls made more than one"
    );
    assert_eq!(kept.len(), 1);
}
