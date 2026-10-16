//! `command_buffer/fences.rs`: the idle fences stay bounded, whatever the number of command
//! buffers, and fences are made only while more are in flight than wait idle.

use crate::command_buffer::fences::{Fences, IDLE};
use ash::vk;
use ash::vk::Handle;

/// The bookkeeping of a device's fences, with stand-in handles: what `take` would make, and
/// what `give` would release.
#[derive(Default)]
struct Ledger {
    fences: Fences,
    made: u64,
    released: Vec<vk::Fence>,
}

impl Ledger {
    /// A fence for a command buffer's last batch: an idle one, else a new one.
    fn take(&mut self) -> vk::Fence {
        self.fences.take_idle().unwrap_or_else(|| {
            self.made += 1;
            vk::Fence::from_raw(self.made)
        })
    }

    /// Fences given back as their command buffers finished.
    fn give(&mut self, mut fences: Vec<vk::Fence>) {
        self.fences.keep(&mut fences);
        self.released.extend(fences);
    }

    /// The fences idle now (taken back after counting).
    fn idle(&mut self) -> usize {
        let mut idle = Vec::new();
        while let Some(fence) = self.fences.take_idle() {
            idle.push(fence);
        }
        let count = idle.len();
        self.fences.keep(&mut idle);
        assert!(idle.is_empty());
        count
    }
}

/// Command buffers one after another, each finishing before the next is committed, reuse one
/// fence: 10 000 of them make one.
#[test]
fn sequential_command_buffers_reuse_one_fence() {
    let mut ledger = Ledger::default();
    for _ in 0..10_000 {
        let fence = ledger.take();
        ledger.give(vec![fence]);
    }
    assert_eq!(ledger.made, 1);
    assert_eq!(ledger.idle(), 1);
    assert!(ledger.released.is_empty());
}

/// A burst of command buffers in flight makes one fence each; once they finished, at most
/// [`IDLE`] stay idle and the rest are released, and later bursts no larger than [`IDLE`] make
/// none: the fences kept do not grow with the command buffers committed.
#[test]
fn the_idle_fences_are_bounded_whatever_the_command_buffers() {
    let mut ledger = Ledger::default();
    let burst: Vec<vk::Fence> = (0..100).map(|_| ledger.take()).collect();
    assert_eq!(ledger.made, 100);
    ledger.give(burst);
    assert_eq!(ledger.idle(), IDLE);
    assert_eq!(ledger.released.len(), 100 - IDLE);
    for round in 0..1_000 {
        let burst: Vec<vk::Fence> = (0..1 + round % IDLE).map(|_| ledger.take()).collect();
        ledger.give(burst);
        assert_eq!(ledger.idle(), IDLE);
    }
    assert_eq!(ledger.made, 100);
    assert_eq!(ledger.released.len(), 100 - IDLE);
    // Every fence made is idle or released, once each.
    let mut seen: Vec<u64> = ledger.released.iter().map(|fence| fence.as_raw()).collect();
    while let Some(fence) = ledger.fences.take_idle() {
        seen.push(fence.as_raw());
    }
    seen.sort_unstable();
    assert_eq!(seen, (1..=100).collect::<Vec<u64>>());
}
