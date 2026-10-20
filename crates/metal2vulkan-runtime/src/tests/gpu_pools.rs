//! A queue's blocks of the GPU pools (`gpu_pools.rs`): the command buffers of one queue share its
//! block, made once, round after round; another queue has a block of its own; a block no lease
//! held for [`QUIET`] goes as a later lease ends, and a queue's goes with the queue.

use super::super::gpu_pools::{QueueBlocks, QUIET};
use std::time::{Duration, Instant};

/// A lease of `queue`'s block from `blocks` at `now`: the block (`made + 1` where the lease
/// makes it) and the leases holding it now.
fn lease(blocks: &mut QueueBlocks<u32>, queue: u64, made: &mut u32, now: Instant) -> (u32, usize) {
    if let Some(taken) = blocks.take(queue, |&block| block) {
        return taken;
    }
    *made += 1;
    let (block, leases, extra) = blocks.made(queue, *made, now, |&block| block);
    assert_eq!(extra, None, "no other lease made one meanwhile");
    (block, leases)
}

/// Two command buffers of one queue in flight at once, round after round, share one block, made
/// in the first round alone.
#[test]
fn command_buffers_of_one_queue_share_its_block() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let start = Instant::now();
    for round in 0..8u32 {
        let now = start + Duration::from_millis(16) * round;
        let first = lease(&mut blocks, 7, &mut made, now);
        let second = lease(&mut blocks, 7, &mut made, now);
        assert_eq!((first, second), ((1, 1), (1, 2)), "round {round}");
        assert!(blocks.give(7, now).is_empty(), "round {round}");
        assert!(blocks.give(7, now).is_empty(), "round {round}");
    }
    assert_eq!((made, blocks.blocks()), (1, 1));
}

/// Another queue's command buffers take a block of their own.
#[test]
fn another_queue_takes_its_own_block() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let now = Instant::now();
    assert_eq!(lease(&mut blocks, 1, &mut made, now), (1, 1));
    assert_eq!(lease(&mut blocks, 2, &mut made, now), (2, 1));
    assert_eq!(lease(&mut blocks, 1, &mut made, now), (1, 2));
    assert_eq!((made, blocks.blocks()), (2, 2));
}

/// A block two leases made at once is made twice: the second made is the one released.
#[test]
fn a_block_made_meanwhile_is_released() {
    let mut blocks = QueueBlocks::default();
    let now = Instant::now();
    assert_eq!(blocks.take(3, |&block: &u32| block), None);
    assert_eq!(blocks.take(3, |&block: &u32| block), None);
    assert_eq!(blocks.made(3, 10, now, |&block| block), (10, 1, None));
    assert_eq!(blocks.made(3, 11, now, |&block| block), (10, 2, Some(11)));
    assert_eq!(blocks.blocks(), 1);
}

/// A queue's block no lease held for the quiet period goes as a lease of another queue ends; a
/// block in use stays, however old, and one whose last lease ended less than the quiet period
/// before stays too.
#[test]
fn a_block_no_lease_held_for_the_quiet_period_goes() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let start = Instant::now();
    // Queue 1 leases once and stops; queue 2 keeps one lease held throughout; queue 3 leases on.
    lease(&mut blocks, 1, &mut made, start);
    assert!(blocks.give(1, start).is_empty());
    lease(&mut blocks, 2, &mut made, start);
    let mut now = start;
    let mut released = Vec::new();
    while now < start + QUIET + Duration::from_millis(100) {
        now += Duration::from_millis(16);
        lease(&mut blocks, 3, &mut made, now);
        let gone = blocks.give(3, now);
        if now.duration_since(start) < QUIET {
            assert!(gone.is_empty(), "nothing goes before the quiet period");
        }
        released.extend(gone);
    }
    assert_eq!(released, [1], "queue 1's block alone");
    assert_eq!(blocks.blocks(), 2);
    assert_eq!(made, 3);
}

/// A queue that goes takes its block with it; the device's teardown takes the rest.
#[test]
fn blocks_go_with_their_queue_and_with_the_device() {
    let mut blocks = QueueBlocks::default();
    let mut made = 0;
    let now = Instant::now();
    for queue in [1, 2, 3] {
        lease(&mut blocks, queue, &mut made, now);
        assert!(blocks.give(queue, now).is_empty());
    }
    assert_eq!(blocks.forget(2), Some(2));
    assert_eq!(blocks.forget(2), None);
    assert_eq!(blocks.drain(), [1, 3]);
    assert_eq!(blocks.blocks(), 0);
}
