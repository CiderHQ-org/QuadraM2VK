//! The device's heap (`heap_sets.rs`, residency part C): an image retired from it waits for the
//! command buffers that may read it, those submitted before it went (committed, for what a rebind
//! or a detach replaced).

use crate::heap_sets::Retirement;

/// With no command buffer out, a retired image goes at once.
#[test]
fn an_image_retired_with_no_command_buffer_out_goes_at_once() {
    let mut retirement = Retirement::default();
    assert_eq!(retirement.retire('a'), Some('a'));
}

/// An image waits for every ticket taken before it was retired, and for none taken after.
#[test]
fn an_image_waits_for_the_command_buffers_committed_before_it_went() {
    let mut retirement = Retirement::default();
    let one = retirement.take();
    assert_eq!(retirement.retire('a'), None, "one may read it");
    let two = retirement.take();
    assert_eq!(retirement.retire('b'), None, "one and two may read it");
    let three = retirement.take();
    assert_eq!(
        retirement.give(two),
        Vec::<char>::new(),
        "one may still read both"
    );
    assert_eq!(
        retirement.give(one),
        ['a', 'b'],
        "three was committed after both went"
    );
    assert_eq!(retirement.retire('c'), None, "three may read it");
    assert_eq!(retirement.give(three), ['c']);
    assert_eq!(retirement.retire('d'), Some('d'), "no ticket is out");
}

/// Two tickets of one epoch: the image waits for both.
#[test]
fn tickets_of_one_epoch_are_counted() {
    let mut retirement = Retirement::default();
    let first = retirement.take();
    let second = retirement.take();
    assert_eq!(first, second);
    assert_eq!(retirement.retire('a'), None);
    assert_eq!(retirement.give(first), Vec::<char>::new());
    assert_eq!(retirement.give(second), ['a']);
}
