//! The spare blocks of a command buffer's GPU pools (`gpu_pools.rs`): command buffers that keep as
//! many pools in flight at once make no block after the first round, a leased block goes to no
//! other lease, and a spare idle for [`QUIET`] goes as a later lease ends, the oldest first.

use super::super::gpu_pools::{Spares, QUIET};
use std::time::{Duration, Instant};

/// A lease from `spares`: the spare given back last, or block `made + 1`, made.
fn lease(spares: &mut Spares<u32>, made: &mut u32) -> u32 {
    spares.take().unwrap_or_else(|| {
        *made += 1;
        *made
    })
}

/// Two command buffers in flight at once, round after round, the first given back before the
/// second: the second round and every later one make and release nothing. (The single spare kept
/// before made one block and released one each round.)
#[test]
fn two_leases_at_once_make_blocks_in_the_first_round_only() {
    let mut spares = Spares::default();
    let mut made = 0;
    let start = Instant::now();
    for round in 0..8u32 {
        let now = start + Duration::from_millis(16) * round;
        let first = lease(&mut spares, &mut made);
        let second = lease(&mut spares, &mut made);
        assert_ne!(first, second, "a leased block goes to no other lease");
        assert_eq!(spares.give(first, now), Vec::<u32>::new(), "round {round}");
        assert_eq!(spares.give(second, now), Vec::<u32>::new(), "round {round}");
        assert_eq!(made, 2, "round {round}");
    }
    assert_eq!(spares.idle(), 2);
}

/// The spare given back last is the next taken, so a burst's extra blocks stay at the bottom and
/// go once they stayed idle for [`QUIET`], as a later lease ends; the blocks in use stay.
#[test]
fn spares_idle_past_the_quiet_period_go_as_a_later_lease_ends() {
    let mut spares = Spares::default();
    let mut made = 0;
    let start = Instant::now();
    // A burst of four leases at once.
    let burst: Vec<u32> = (0..4).map(|_| lease(&mut spares, &mut made)).collect();
    for &block in &burst {
        assert!(spares.give(block, start).is_empty());
    }
    assert_eq!(made, 4);
    // Then one lease at a time, every 16 ms: the block given back last serves each, and nothing
    // goes before the others stayed idle for the quiet period.
    let mut now = start;
    let mut released = Vec::new();
    while now < start + QUIET + Duration::from_millis(100) {
        now += Duration::from_millis(16);
        let block = lease(&mut spares, &mut made);
        assert_eq!(block, 4, "the spare given back last");
        let gone = spares.give(block, now);
        if now.duration_since(start) < QUIET {
            assert!(gone.is_empty(), "nothing goes before the quiet period");
        }
        released.extend(gone);
    }
    assert_eq!(made, 4, "no block made after the burst");
    assert_eq!(released, [1, 2, 3], "the idle ones, the oldest first");
    assert_eq!(spares.idle(), 1);
}

/// A device's spares all go as it is destroyed.
#[test]
fn every_spare_drains() {
    let mut spares = Spares::default();
    let now = Instant::now();
    for block in [1u32, 2, 3] {
        assert!(spares.give(block, now).is_empty());
    }
    assert_eq!(spares.drain(), [1, 2, 3]);
    assert_eq!(spares.take(), None);
}
