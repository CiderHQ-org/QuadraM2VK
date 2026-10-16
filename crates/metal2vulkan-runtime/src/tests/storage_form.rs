//! `texture/storage_form.rs`: a switch counts the storage forms it swapped in before its flag
//! drops, so a new command buffer that waited the switch out, reading the flag down, reads the
//! count, whichever thread ended the switch and however long that thread is held up after.

use crate::texture::storage_form::StorageForms;
use crate::work_counts::Counters;
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

/// The longest a step of this test may take: a close that never drops the flag fails the test
/// rather than hanging it.
const WAIT: Duration = Duration::from_secs(30);

/// The thread closing a switch (as the scheduler's thread does) is held right after the flag
/// dropped, until a new command buffer's gate read the flag down and the count: the count holds
/// the swap.
#[test]
fn a_switch_is_counted_before_its_flag_drops() {
    let (paused, held) = mpsc::channel();
    let (resume, resumed) = mpsc::channel::<()>();
    let resumed = Mutex::new(resumed);
    let forms = StorageForms::under_way(move || {
        paused.send(()).unwrap();
        let resumed = resumed.lock().unwrap();
        resumed.recv_timeout(WAIT).unwrap();
    });
    let counts = Counters::default();
    let seen = thread::scope(|scope| {
        let closer = scope.spawn(|| forms.close_after(&counts, 1));
        let started = Instant::now();
        while forms.switching() {
            assert!(started.elapsed() < WAIT, "the switch's flag never dropped");
            std::hint::spin_loop();
        }
        let seen = counts.storage_forms_taken();
        held.recv_timeout(WAIT).unwrap();
        resume.send(()).unwrap();
        closer.join().unwrap();
        seen
    });
    assert_eq!(
        seen, 1,
        "the count a new command buffer reads once the flag dropped"
    );
    assert_eq!(counts.storage_forms_taken(), 1);
}
