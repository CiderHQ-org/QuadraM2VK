//! `command_buffer/kept_barriers.rs`: what a batch takes of a `VkQueue`'s kept barrier command
//! buffers (`QueueKept::take`) as the queue's timeline moves: a kept one only once the batch that
//! last took it completed, so none is pending twice; a list met twice in one batch, and a value a
//! failed submission left to the next batch; where a batch records its own, past the bounds; and
//! which list a new one replaces once the kept ones fill the places. The command buffers are
//! handles the test makes: no device records or submits them.

use crate::command_buffer::kept_barriers::{QueueKept, Take, KEPT, LISTS, STALE};
use crate::tracker::{Access, Barrier};
use ash::vk::{self, Handle};

/// The list of barriers numbered `n`: one barrier, into stage bit `n`.
fn list(n: u32) -> Vec<Barrier> {
    vec![Barrier {
        src: Access::NONE,
        dst: Access::new(
            vk::PipelineStageFlags2::from_raw(1 << n),
            vk::AccessFlags2::MEMORY_READ,
        ),
    }]
}

/// A queue's kept command buffers, and the handles the test records them as.
#[derive(Default)]
struct Queue {
    kept: QueueKept,
    recorded: u64,
}

/// What a batch got.
#[derive(Debug, PartialEq, Eq)]
enum Got {
    /// A kept command buffer, the `n`th recorded.
    Kept(u64),
    /// One recorded now, the `n`th, into the list at `list`.
    Recorded { n: u64, list: usize },
    /// None: the batch records its own.
    Own,
}

impl Queue {
    /// What the batch that signals `at` takes for `barriers`, the timeline completed through
    /// `completed`, as `KeptBarriers::take` does: one recorded is a spare one where there is one.
    fn take(&mut self, barriers: &[Barrier], at: u64, completed: u64) -> Got {
        match self.kept.take(barriers, at, completed) {
            Take::Kept(raw) => Got::Kept(raw.as_raw()),
            Take::Past => Got::Own,
            Take::Record(list) => {
                let raw = self.kept.spare().unwrap_or_else(|| {
                    self.recorded += 1;
                    vk::CommandBuffer::from_raw(self.recorded)
                });
                self.kept.recorded(list, raw, at);
                Got::Recorded {
                    n: raw.as_raw(),
                    list,
                }
            }
        }
    }
}

/// A batch that meets one list twice takes two command buffers, and the next batch takes the
/// same two once the first completed; one taken while they are pending records a third.
#[test]
fn a_list_met_twice_in_one_batch_takes_two_command_buffers() {
    let mut queue = Queue::default();
    let barriers = list(0);
    let recorded = |n| Got::Recorded { n, list: 0 };
    assert_eq!(queue.take(&barriers, 1, 0), recorded(1));
    assert_eq!(queue.take(&barriers, 1, 0), recorded(2));
    // Batch 2 while batch 1 runs: both pending.
    assert_eq!(queue.take(&barriers, 2, 0), recorded(3));
    // Batch 3 once batch 1 completed: its two, then a fourth (batch 2's is pending).
    assert_eq!(queue.take(&barriers, 3, 1), Got::Kept(1));
    assert_eq!(queue.take(&barriers, 3, 1), Got::Kept(2));
    assert_eq!(queue.take(&barriers, 3, 1), recorded(4));
    // Every batch completed: in the order the batches last took them.
    for n in [3, 1, 2, 4] {
        assert_eq!(queue.take(&barriers, 4, 3), Got::Kept(n));
    }
    assert_eq!(
        queue.kept.lists().0[0].1,
        [3, 1, 2, 4].map(vk::CommandBuffer::from_raw)
    );
}

/// A batch whose submission failed leaves its timeline value to the queue's next batch: the
/// command buffer it took waits for that batch, beside the one the next batch takes, and both
/// are taken again once it completed.
#[test]
fn a_failed_submission_leaves_its_command_buffer_to_the_next_batch_s_value() {
    let mut queue = Queue::default();
    let barriers = list(0);
    assert_eq!(queue.take(&barriers, 1, 0), Got::Recorded { n: 1, list: 0 });
    // Batch 1 completed. Batch 2 takes the kept one, and its submission fails.
    assert_eq!(queue.take(&barriers, 2, 1), Got::Kept(1));
    // The next batch signals 2 in its place: the one batch 2 took is not taken again.
    assert_eq!(queue.take(&barriers, 2, 1), Got::Recorded { n: 2, list: 0 });
    assert_eq!(queue.take(&barriers, 3, 1), Got::Recorded { n: 3, list: 0 });
    // Once 2 completed, both are taken again, and the third once 3 completed.
    assert_eq!(queue.take(&barriers, 4, 2), Got::Kept(1));
    assert_eq!(queue.take(&barriers, 4, 2), Got::Kept(2));
    assert_eq!(queue.take(&barriers, 4, 2), Got::Recorded { n: 4, list: 0 });
    assert_eq!(queue.take(&barriers, 5, 3), Got::Kept(3));
}

/// Past [`KEPT`] command buffers of a list all pending, and past [`LISTS`] lists all in use, a
/// batch records its own; the kept ones are taken again once they completed.
#[test]
fn past_the_bounds_a_batch_records_its_own() {
    let mut queue = Queue::default();
    let barriers = list(0);
    for n in 1..=KEPT as u64 {
        assert_eq!(queue.take(&barriers, 1, 0), Got::Recorded { n, list: 0 });
    }
    assert_eq!(queue.take(&barriers, 1, 0), Got::Own);
    assert_eq!(queue.take(&barriers, 2, 1), Got::Kept(1));

    let mut queue = Queue::default();
    for n in 0..LISTS as u32 {
        let got = queue.take(&list(n), 1, 0);
        assert_eq!(
            got,
            Got::Recorded {
                n: u64::from(n) + 1,
                list: n as usize
            }
        );
    }
    let extra = list(LISTS as u32);
    assert_eq!(queue.take(&extra, 1, 0), Got::Own);
    // Completed, but taken within the last `STALE` values: still in use.
    assert_eq!(queue.take(&extra, 2, 1), Got::Own);
    assert_eq!(queue.take(&extra, STALE, 1), Got::Own);
    assert_eq!(queue.take(&list(0), STALE, 1), Got::Kept(1));
}

/// Once [`LISTS`] lists are kept, a new list replaces the one taken least lately among those no
/// batch took in the last [`STALE`] values and whose command buffers all completed, and is
/// recorded into that one's command buffers (and kept from then on); lists still in use, or with
/// a command buffer pending, stay.
#[test]
fn a_list_out_of_use_gives_way_to_a_new_one() {
    let mut queue = Queue::default();
    // A program's first batches meet `LISTS` lists, one each, the first list twice.
    assert_eq!(queue.take(&list(0), 1, 0), Got::Recorded { n: 1, list: 0 });
    assert_eq!(queue.take(&list(0), 1, 0), Got::Recorded { n: 2, list: 0 });
    for n in 1..LISTS as u32 {
        let at = u64::from(n) + 1;
        let got = queue.take(&list(n), at, at - 1);
        assert_eq!(
            got,
            Got::Recorded {
                n: at + 1,
                list: n as usize
            }
        );
    }
    // Long after: lists 1 to 3 still in use, list 4's command buffer pending.
    let at = 1000;
    for n in 1..=3 {
        assert!(matches!(queue.take(&list(n), at, at - 1), Got::Kept(_)));
    }
    assert!(matches!(queue.take(&list(4), at + 1, at), Got::Kept(_)));
    // The frame's lists, each new: the first replaces list 0 (taken least lately, at 1), and is
    // recorded into its two command buffers in turn.
    let frame = |n| list(LISTS as u32 + n);
    assert_eq!(
        queue.take(&frame(0), at + 2, at),
        Got::Recorded { n: 2, list: 0 }
    );
    assert_eq!(
        queue.take(&frame(0), at + 2, at),
        Got::Recorded { n: 1, list: 0 }
    );
    let (lists, spare) = queue.kept.lists();
    assert_eq!(lists[0].0, frame(0));
    assert!(spare.is_empty(), "{spare:?}");
    // Then lists 5, 6 and 7, each into its own command buffer, never 1 to 3 (in use) or 4
    // (pending).
    for (n, list) in [(1, 5), (2, 6), (3, 7)] {
        let got = queue.take(&frame(n), at + 2, at);
        let n = list as u64 + 2;
        assert_eq!(got, Got::Recorded { n, list }, "frame list {n}");
    }
    assert_eq!(queue.take(&frame(4), at + 2, at), Got::Own);
    // Each is kept from then on.
    for n in 0..4 {
        assert!(matches!(
            queue.take(&frame(n), at + 3, at + 2),
            Got::Kept(_)
        ));
    }
}

/// More than [`LISTS`] lists in use in turn replace none, whichever comes first in a batch: the
/// first kept stay kept, and the rest are recorded by their batches, batch after batch.
#[test]
fn lists_in_use_in_turn_replace_none() {
    let mut queue = Queue::default();
    // The ones met last in each batch are met first: the kept ones are completed, not yet taken.
    let lists: Vec<u32> = (0..LISTS as u32 + 2).rev().collect();
    let mut kept = Vec::new();
    for round in 0..4 * STALE {
        let at = round + 1;
        for &n in &lists {
            let got = queue.take(&list(n), at, at - 1);
            match got {
                Got::Recorded { .. } if round == 0 => kept.push(n),
                Got::Kept(_) if round > 0 && kept.contains(&n) => {}
                Got::Own if !kept.contains(&n) => {}
                got => panic!("round {round}, list {n}: {got:?}, kept {kept:?}"),
            }
        }
    }
    assert_eq!(kept.len(), LISTS);
}
