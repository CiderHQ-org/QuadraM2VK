//! The scheduler's rule for the caller's binary semaphores, as a pure structure: no batch is
//! submitted with a wait for a binary semaphore whose signal by this device is not submitted.
//!
//! Vulkan requires a binary semaphore's signal to be submitted before any wait for it
//! (VUID-vkQueueSubmit-pWaitSemaphores-03238), and the runtime submits its batches only from the
//! scheduler, after `commit`. So a wait whose signal comes from the caller (an acquire, or the
//! caller's own submission) is ready at once: the caller submitted that signal before it committed
//! the waiting command buffer (`encode_wait_for_vk_semaphore`'s contract). A wait whose signal a
//! command buffer of this device encoded is ready once that signal is submitted: the scheduler
//! counts, per semaphore, the signals committed and the signals submitted, and a wait committed
//! after `n` signals of its semaphore waits until `n` are submitted.
//!
//! Each count leaves the table once no committed signal is pending and no held wait counts on it,
//! so the table holds only the semaphores of batches still held.

use std::collections::HashMap;
use std::hash::Hash;

/// The signals of one binary semaphore the device's command buffers committed and submitted, and
/// the held waits that count on them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    committed: u64,
    submitted: u64,
    waiting: u64,
}

/// What a held batch needs of the binary semaphores it waits for: for each, how many of its
/// signals must be submitted first.
pub(crate) type Needs<S> = Vec<(S, u64)>;

/// The device's binary semaphore signals, by semaphore.
#[derive(Debug)]
pub(crate) struct BinarySignals<S> {
    counts: HashMap<S, Counts>,
}

impl<S> Default for BinarySignals<S> {
    fn default() -> BinarySignals<S> {
        BinarySignals {
            counts: HashMap::new(),
        }
    }
}

impl<S: Copy + Eq + Hash> BinarySignals<S> {
    /// A batch committed now, waiting for `waits` and signaling `signals`: what its waits need.
    /// Its waits count every signal committed before it, not its own (a batch waits before it
    /// signals).
    pub(crate) fn commit(&mut self, waits: &[S], signals: &[S]) -> Needs<S> {
        let needs = waits
            .iter()
            .map(|&semaphore| {
                let counts = self.counts.entry(semaphore).or_default();
                counts.waiting += 1;
                (semaphore, counts.committed)
            })
            .collect();
        for &semaphore in signals {
            self.counts.entry(semaphore).or_default().committed += 1;
        }
        needs
    }

    /// Whether every signal `needs` counts on is submitted.
    pub(crate) fn ready(&self, needs: &[(S, u64)]) -> bool {
        needs.iter().all(|(semaphore, count)| {
            self.counts
                .get(semaphore)
                .is_none_or(|counts| counts.submitted >= *count)
        })
    }

    /// The batch whose waits needed `needs` and that signals `signals` was submitted.
    pub(crate) fn submitted(&mut self, needs: &[(S, u64)], signals: &[S]) {
        for &semaphore in signals {
            if let Some(counts) = self.counts.get_mut(&semaphore) {
                counts.submitted += 1;
            }
        }
        self.leave(needs);
        for semaphore in signals {
            self.prune(semaphore);
        }
    }

    /// The batch whose waits needed `needs` will never be submitted (its command buffer failed):
    /// its waits leave. Its signals stay committed and never submitted, so a later wait for one
    /// stays held, as it would wait on Vulkan for a signal that never comes.
    pub(crate) fn dropped(&mut self, needs: &[(S, u64)]) {
        self.leave(needs);
    }

    fn leave(&mut self, needs: &[(S, u64)]) {
        for (semaphore, _) in needs {
            if let Some(counts) = self.counts.get_mut(semaphore) {
                counts.waiting = counts.waiting.saturating_sub(1);
            }
            self.prune(semaphore);
        }
    }

    /// Forget `semaphore` once nothing pends on it: every committed signal submitted, and no held
    /// wait counting.
    fn prune(&mut self, semaphore: &S) {
        if self
            .counts
            .get(semaphore)
            .is_some_and(|counts| counts.committed == counts.submitted && counts.waiting == 0)
        {
            self.counts.remove(semaphore);
        }
    }

    /// How many semaphores the table holds.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.counts.len()
    }
}
