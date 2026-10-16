//! An event's signal indices and Metal's values (`event_timeline.rs`): monotone values, dropped
//! signals, the order of kept signals across `VkQueue`s, the host's floor, the waits a batch is
//! submitted with, and a counter read observed only once a host wait returned for it.

use crate::event_timeline::{EventTimeline, Signal, WaitFor};

/// Submit a GPU signal of `value` on `queue`: what it became, kept when it is.
fn submit(timeline: &mut EventTimeline<usize>, value: u64, queue: usize) -> Signal {
    let signal = timeline.signal(value, queue);
    if let Signal::Kept { index, .. } = signal {
        timeline.keep(index, value, queue);
    }
    signal
}

#[test]
fn kept_signals_take_rising_indices() {
    let mut timeline = EventTimeline::default();
    assert_eq!(
        submit(&mut timeline, 3, 0),
        Signal::Kept {
            index: 1,
            after: None
        }
    );
    assert_eq!(
        submit(&mut timeline, 5, 0),
        Signal::Kept {
            index: 2,
            after: None
        }
    );
    assert_eq!(timeline.known(), 5);
    // Nothing reached yet.
    assert_eq!(timeline.value(), 0);
    assert!(timeline.observe(1));
    assert_eq!(timeline.value(), 3);
    assert!(timeline.observe(2));
    assert_eq!(timeline.value(), 5);
    assert_eq!(timeline.pending(), 0);
    // A later signal continues from the last index reached.
    assert_eq!(
        submit(&mut timeline, 6, 1),
        Signal::Kept {
            index: 3,
            after: None
        }
    );
}

#[test]
fn a_signal_not_above_what_is_known_is_dropped() {
    let mut timeline = EventTimeline::default();
    submit(&mut timeline, 7, 0);
    // Lower and equal: Metal's maximum makes them no-ops (measured).
    assert_eq!(submit(&mut timeline, 4, 1), Signal::Dropped);
    assert_eq!(submit(&mut timeline, 7, 0), Signal::Dropped);
    assert_eq!(timeline.pending(), 1);
    timeline.observe(1);
    assert_eq!(timeline.value(), 7);
}

#[test]
fn either_submission_order_of_seven_and_four_ends_at_seven() {
    // Seven first: the four is dropped.
    let mut first = EventTimeline::default();
    submit(&mut first, 7, 0);
    assert_eq!(submit(&mut first, 4, 1), Signal::Dropped);
    first.observe(1);
    assert_eq!(first.value(), 7);
    // Four first: the seven is kept after it, and waits for it on another queue.
    let mut second = EventTimeline::default();
    submit(&mut second, 4, 1);
    assert_eq!(
        submit(&mut second, 7, 0),
        Signal::Kept {
            index: 2,
            after: Some(1)
        }
    );
    second.observe(1);
    assert_eq!(second.value(), 4);
    second.observe(2);
    assert_eq!(second.value(), 7);
}

#[test]
fn a_kept_signal_waits_only_for_one_pending_on_another_queue() {
    let mut timeline = EventTimeline::default();
    submit(&mut timeline, 1, 0);
    // Same queue: Vulkan's signal operation order keeps them in order.
    assert_eq!(
        submit(&mut timeline, 2, 0),
        Signal::Kept {
            index: 2,
            after: None
        }
    );
    // Another queue, the previous one pending: wait for it.
    assert_eq!(
        submit(&mut timeline, 3, 1),
        Signal::Kept {
            index: 3,
            after: Some(2)
        }
    );
    // Once it was reached, no wait.
    timeline.observe(3);
    assert_eq!(
        submit(&mut timeline, 4, 0),
        Signal::Kept {
            index: 4,
            after: None
        }
    );
}

#[test]
fn waits_map_to_the_first_index_that_reaches_the_value() {
    let mut timeline = EventTimeline::default();
    submit(&mut timeline, 3, 0);
    submit(&mut timeline, 8, 0);
    assert_eq!(timeline.wait(0), Some(WaitFor::Reached));
    assert_eq!(timeline.wait(2), Some(WaitFor::Index(1)));
    assert_eq!(timeline.wait(3), Some(WaitFor::Index(1)));
    assert_eq!(timeline.wait(4), Some(WaitFor::Index(2)));
    assert_eq!(timeline.wait(8), Some(WaitFor::Index(2)));
    // Nothing submitted reaches 9: the batch is held.
    assert_eq!(timeline.wait(9), None);
    timeline.observe(1);
    assert_eq!(timeline.wait(3), Some(WaitFor::Reached));
}

#[test]
fn the_host_floor_takes_effect_at_once_and_never_on_the_semaphore() {
    let mut timeline = EventTimeline::default();
    submit(&mut timeline, 7, 0);
    // A host set below the pending GPU signal: the value is 5 now.
    assert!(timeline.set(5));
    assert_eq!(timeline.value(), 5);
    // Lower and equal sets change nothing (measured).
    assert!(!timeline.set(3));
    assert!(!timeline.set(5));
    // A wait the floor reached needs no Vulkan wait, though a GPU signal above it is pending.
    assert_eq!(timeline.wait(5), Some(WaitFor::Reached));
    assert_eq!(timeline.wait(6), Some(WaitFor::Index(1)));
    // The GPU's seven lands above the floor.
    assert!(timeline.observe(1));
    assert_eq!(timeline.value(), 7);
}

#[test]
fn a_wait_only_the_host_floor_satisfies_is_never_a_vulkan_wait() {
    let mut timeline = EventTimeline::default();
    assert_eq!(timeline.wait(9), None);
    timeline.set(9);
    assert_eq!(timeline.known(), 9);
    for value in [1, 8, 9] {
        assert_eq!(timeline.wait(value), Some(WaitFor::Reached));
    }
    assert_eq!(timeline.wait(10), None);
    // A GPU signal not above the floor is dropped, so no index ever stands for the floor.
    assert_eq!(submit(&mut timeline, 9, 0), Signal::Dropped);
    assert_eq!(
        submit(&mut timeline, 12, 0),
        Signal::Kept {
            index: 1,
            after: None
        }
    );
    assert_eq!(timeline.wait(9), Some(WaitFor::Reached));
    assert_eq!(timeline.wait(10), Some(WaitFor::Index(1)));
}

#[test]
fn a_host_set_above_a_pending_signal_leaves_its_waits_on_that_signal() {
    // The residual: a wait handed to Vulkan before the host set still waits for the GPU signal,
    // which was submitted before it.
    let mut timeline = EventTimeline::default();
    submit(&mut timeline, 7, 0);
    let handed = timeline.wait(5);
    assert_eq!(handed, Some(WaitFor::Index(1)));
    timeline.set(9);
    assert_eq!(timeline.value(), 9);
    // A wait submitted now needs nothing.
    assert_eq!(timeline.wait(5), Some(WaitFor::Reached));
    // The GPU's seven reaches index 1, and the value stays at the floor.
    assert!(!timeline.observe(1));
    assert_eq!(timeline.value(), 9);
}

#[test]
fn the_largest_value_is_kept() {
    let mut timeline = EventTimeline::default();
    assert_eq!(
        submit(&mut timeline, u64::MAX, 0),
        Signal::Kept {
            index: 1,
            after: None
        }
    );
    timeline.observe(1);
    assert_eq!(timeline.value(), u64::MAX);
    assert_eq!(submit(&mut timeline, u64::MAX, 0), Signal::Dropped);
    assert!(!timeline.set(u64::MAX));
}

/// A counter read past the index last seen raises the value only once a host wait returned for
/// it (`event.rs`): a read alone orders no host operation after the signal. A read the index
/// last seen covers asks for no wait, and a failed wait observes nothing.
#[test]
fn a_counter_read_is_observed_only_once_a_host_wait_returned_for_it() {
    let mut timeline = EventTimeline::default();
    submit(&mut timeline, 3, 0);
    submit(&mut timeline, 5, 0);
    let mut asked = Vec::new();
    let mut observe = |timeline: &mut EventTimeline<usize>, read: u64, returned: bool| {
        timeline.observe_read::<()>(read, |index| {
            asked.push(index);
            Ok(returned)
        })
    };
    // The read shows both signals; the wait for them does not return: nothing is observed.
    assert_eq!(observe(&mut timeline, 2, false), Ok(false));
    assert_eq!(timeline.value(), 0);
    // It returns: both are.
    assert_eq!(observe(&mut timeline, 2, true), Ok(true));
    assert_eq!(timeline.value(), 5);
    // A read the index last seen covers needs no wait.
    assert_eq!(observe(&mut timeline, 2, true), Ok(false));
    assert_eq!(asked, [2, 2]);
    // A failed wait (a lost device) observes nothing.
    submit(&mut timeline, 7, 0);
    assert_eq!(timeline.observe_read(3, |_| Err("lost")), Err("lost"));
    assert_eq!(timeline.value(), 5);
}
