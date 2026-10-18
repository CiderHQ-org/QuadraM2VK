//! `uniqueIdentifier`: the device's allocations of the identifiers from 2^36, and the plain count
//! past them.

use std::sync::Arc;

use crate::degradation::Degradation;
use crate::resource_ids::StateKind;
use crate::unique_identifier::{
    state_value_degradation, UniqueIdentifier, UniqueIdentifiers, BUFFERS_END, BUFFER_FIRST,
    COMPUTE_FIRST, COMPUTE_STEP, INDIRECT_BASE, PLAIN_FIRST, RENDER_STEP,
};

fn device() -> Arc<UniqueIdentifiers> {
    Arc::default()
}

/// Indirect command buffers lie as Metal's do: buffers of one 16-KiB page take 32 KiB each from
/// `0x18000`; one given back leaves a gap the next that fits takes, and a larger one goes past the
/// live ones.
#[test]
fn buffers_take_the_lowest_gap_that_holds_them_and_a_page() {
    let ids = device();
    let page = |ids: &Arc<UniqueIdentifiers>| UniqueIdentifier::buffer(ids, 0x4000).unwrap();
    let at = |offset: u64| INDIRECT_BASE + offset;
    let [a, b, c] = [page(&ids), page(&ids), page(&ids)];
    assert_eq!(
        [a.value(), b.value(), c.value()],
        [at(0x1_8000), at(0x2_0000), at(0x2_8000)]
    );
    assert_eq!(a.value(), INDIRECT_BASE + BUFFER_FIRST);
    drop(b);
    // Two pages and one more take 48 KiB, at a multiple of 32 KiB: 64 KiB, past `c`.
    let large = UniqueIdentifier::buffer(&ids, 0x8000).unwrap();
    assert_eq!(large.value(), at(0x3_0000));
    let gap = page(&ids);
    assert_eq!(gap.value(), at(0x2_0000), "the gap `b` left");
    let past = page(&ids);
    assert_eq!(past.value(), at(0x4_0000), "past `large`");
    // A footprint past the range is refused.
    assert!(UniqueIdentifier::buffer(&ids, BUFFERS_END).is_err());
    assert!(UniqueIdentifier::buffer(&ids, u64::MAX).is_err());
}

/// States made with `supportIndirectCommandBuffers` take their kind's next slot, the last given
/// back first, as Metal's do (the consumer's measurement on an M3 Max); compute and render
/// states and buffers never meet, and plain identifiers lie past all of them.
#[test]
fn indirect_states_take_the_last_released_identifier_first() {
    let ids = device();
    let compute = || UniqueIdentifier::state(&ids, StateKind::Compute, true).unwrap();
    let render = || UniqueIdentifier::state(&ids, StateKind::Render, true).unwrap();
    let [a, b, c] = [compute(), compute(), compute()];
    assert_eq!(b.value() - a.value(), COMPUTE_STEP);
    assert_eq!(c.value() - b.value(), COMPUTE_STEP);
    assert_eq!(a.value(), INDIRECT_BASE + COMPUTE_FIRST);
    let (a_value, c_value) = (a.value(), c.value());
    drop(a);
    drop(c);
    let again = compute();
    assert_eq!(again.value(), c_value, "the last released");
    let r = [render(), render()];
    assert_eq!(r[1].value() - r[0].value(), RENDER_STEP);
    assert!(r[0].value() >= INDIRECT_BASE + BUFFERS_END && r[0].value() > c_value);
    let first = compute();
    assert_eq!(
        first.value(),
        a_value,
        "render states take another allocation"
    );
    let plain = UniqueIdentifier::state(&ids, StateKind::Compute, false).unwrap();
    let other = UniqueIdentifier::state(&device(), StateKind::Render, false).unwrap();
    assert!(plain.value() >= PLAIN_FIRST && other.value() >= PLAIN_FIRST);
    assert!(r[1].value() < PLAIN_FIRST);
    assert_ne!(plain.value(), other.value(), "one count for every device");
}

/// Compute states take Metal's region below the buffers' range, 511 of them from `0xc0`, then go
/// past the buffers' range; none meets a buffer's.
#[test]
fn compute_states_go_past_the_buffers_once_metal_s_region_is_full() {
    let ids = device();
    let fit = (BUFFER_FIRST - COMPUTE_FIRST) / COMPUTE_STEP;
    assert_eq!(fit, 511);
    let held: Vec<UniqueIdentifier> = (0..=fit)
        .map(|_| UniqueIdentifier::state(&ids, StateKind::Compute, true).unwrap())
        .collect();
    let last_in = held[fit as usize - 1].value() - INDIRECT_BASE;
    assert!(last_in + COMPUTE_STEP <= BUFFER_FIRST, "{last_in:#x}");
    assert_eq!(held[fit as usize].value(), INDIRECT_BASE + BUFFERS_END);
    let buffer = UniqueIdentifier::buffer(&ids, 0x4000).unwrap();
    assert_eq!(buffer.value(), INDIRECT_BASE + BUFFER_FIRST);
}

/// Which identifiers are measured to be Metal's number: a buffer made before its device opened
/// an encoder; not one made after, nor any state, each with the kind its answer announces.
#[test]
fn only_buffers_made_before_an_encoder_are_metal_s_number() {
    let ids = device();
    let before = UniqueIdentifier::buffer(&ids, 0x4000).unwrap();
    assert_eq!(before.degradation(), None);
    ids.encoder_opened();
    let after = UniqueIdentifier::buffer(&ids, 0x4000).unwrap();
    let unmeasured = Some(Degradation::UnmeasuredUniqueIdentifier);
    assert_eq!(after.degradation(), unmeasured);
    assert_eq!(before.degradation(), None, "made before");
    let indirect = UniqueIdentifier::state(&ids, StateKind::Render, true).unwrap();
    assert_eq!(indirect.degradation(), unmeasured);
    let plain = UniqueIdentifier::state(&device(), StateKind::Compute, false).unwrap();
    let run_to_run = Some(Degradation::PlainStateUniqueIdentifier);
    assert_eq!(plain.degradation(), run_to_run);
    // A command's answer by its value: none for no state.
    assert_eq!(state_value_degradation(0), None);
    assert_eq!(state_value_degradation(indirect.value()), unmeasured);
    assert_eq!(state_value_degradation(plain.value()), run_to_run);
}
