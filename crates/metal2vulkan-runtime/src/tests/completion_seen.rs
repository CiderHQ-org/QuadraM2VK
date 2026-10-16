//! `command_buffer/completion.rs`'s `seen`: a queue's waiter publishes a timeline
//! value only once a host wait returned for it, a fence's or the timeline's, never from the
//! counter's read alone.

use crate::command_buffer::completion::seen;

/// What `seen` publishes after `known`, with a wait that returned for `waited` and a read of
/// `read`, where a wait for the value read returns `reached`; and the values it waited for.
fn published(known: u64, waited: Option<u64>, read: u64, reached: bool) -> (u64, Vec<u64>) {
    let mut asked = Vec::new();
    let value = seen::<()>(known, waited, read, |value| {
        asked.push(value);
        Ok(reached)
    })
    .unwrap();
    (value, asked)
}

#[test]
fn a_value_only_the_read_shows_is_waited_for_before_it_is_published() {
    // The front's wait returned and the read shows its value: nothing more to wait for.
    assert_eq!(published(3, Some(4), 4, true), (4, vec![]));
    // Later batches completed meanwhile: their value is waited for, at once, then published.
    assert_eq!(published(3, Some(4), 6, true), (6, vec![6]));
    // The front's wait ran out as its signal landed: the read's value is waited for too.
    assert_eq!(published(3, None, 4, true), (4, vec![4]));
    // The wait ran out and nothing moved: nothing new, and no wait.
    assert_eq!(published(3, None, 3, true), (3, vec![]));
    // A fence's batch whose timeline the read shows late (lavapipe): the fence's wait stands.
    assert_eq!(published(3, Some(4), 3, true), (4, vec![]));
    // A wait for the value read that ran out publishes no more than a wait returned for.
    assert_eq!(published(3, Some(4), 6, false), (4, vec![6]));
    assert_eq!(published(3, None, 6, false), (3, vec![6]));
    // A failed wait is the poll's failure.
    assert_eq!(seen(3, None, 5, |_| Err("lost")), Err("lost"));
}
