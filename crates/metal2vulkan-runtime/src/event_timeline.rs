//! What one event's timeline semaphore means in Metal's values, as a pure function of the signals
//! the scheduler submitted, the host's sets, and what the semaphore reached.
//!
//! **Metal's values.** An event's value only rises: a signal, from the GPU or the host, of a value
//! not above the current one changes nothing, and two signals land as their maximum whatever order
//! they execute in (measured on the M3 and the M4). A Vulkan timeline semaphore needs every signal
//! above the value before it, so the semaphore does not carry Metal's values:
//! - **Signal indices.** Each GPU signal the scheduler keeps raises the semaphore by one, and the
//!   timeline maps that index to its Metal value. A signal is kept only when its value is above
//!   every value submitted or set before it, in the scheduler's submission order, so kept values
//!   rise with their indices; a signal not above them is dropped, as Metal's maximum would make it
//!   a no-op. A kept signal that follows one still pending on another `VkQueue` waits for that one
//!   ([`Signal::after`]), so the semaphore's signals execute in index order; on one `VkQueue`,
//!   Vulkan's signal operation order already does.
//! - **The host's floor.** A host set never touches the semaphore: it takes effect at once as a
//!   floor the value never drops below. A GPU wait for a value the floor reached needs no Vulkan
//!   wait at all, so no batch ever waits for a host signal on a `VkQueue`.
//! - **What the scheduler knows.** The value every submitted signal and every host set reaches
//!   ([`EventTimeline::known`]). A wait for a known value is satisfied by work already submitted,
//!   or by the host: the scheduler submits only such waits.
//!
//! **The residual.** A wait already handed to Vulkan for a GPU signal's index is not released
//! early by a later host set above its value: it waits for that GPU signal, which was submitted
//! before it and so comes. Metal releases it at the host set.

use crate::command_buffer::completion::seen;
use std::collections::VecDeque;

/// One kept GPU signal the semaphore has not been seen to reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pending<Q> {
    index: u64,
    value: u64,
    /// The `VkQueue` of the batch that signals it.
    queue: Q,
}

/// How a batch waits for a Metal value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WaitFor {
    /// The value was reached (on the semaphore, or by the host's floor): no Vulkan wait.
    Reached,
    /// A Vulkan wait for the semaphore to reach this index, a submitted GPU signal's.
    Index(u64),
}

/// What a GPU signal of a value becomes when its batch is submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Signal {
    /// Not above what is known: Metal's maximum makes it a no-op.
    Dropped,
    /// Kept: the batch signals the semaphore to `index`, after it reached `after` when that is a
    /// signal pending on another `VkQueue`.
    Kept { index: u64, after: Option<u64> },
}

/// One event's map from its semaphore's signal indices to Metal's values.
#[derive(Debug)]
pub(super) struct EventTimeline<Q> {
    /// The highest index the semaphore was seen to reach.
    reached: u64,
    /// The Metal value of that index (0 before any).
    reached_value: u64,
    /// The kept GPU signals above `reached`, in index order (and so in value order).
    pending: VecDeque<Pending<Q>>,
    /// The highest value the host set.
    floor: u64,
}

impl<Q> Default for EventTimeline<Q> {
    fn default() -> EventTimeline<Q> {
        EventTimeline {
            reached: 0,
            reached_value: 0,
            pending: VecDeque::new(),
            floor: 0,
        }
    }
}

impl<Q: Copy + Eq> EventTimeline<Q> {
    /// The event's value now: the semaphore's last seen index's value, or the host's floor.
    pub(super) fn value(&self) -> u64 {
        self.reached_value.max(self.floor)
    }

    /// The value every submitted signal and host set reaches: what a wait may be submitted for.
    pub(super) fn known(&self) -> u64 {
        self.pending
            .back()
            .map_or(self.value(), |last| last.value.max(self.value()))
    }

    /// The semaphore reached `index`. Whether the value rose.
    pub(super) fn observe(&mut self, index: u64) -> bool {
        let before = self.value();
        while let Some(&front) = self.pending.front() {
            if front.index > index {
                break;
            }
            self.pending.pop_front();
            self.reached = front.index;
            self.reached_value = front.value;
        }
        self.value() > before
    }

    /// The semaphore's counter read `read`: observed only up to the index a host wait returned
    /// for (`reaches`, the answer of `command_buffer/completion.rs`'s `seen`), as a read alone
    /// orders no host operation after the signal; a read past the index last seen is waited for
    /// first. Whether the value rose.
    pub(super) fn observe_read<E>(
        &mut self,
        read: u64,
        reaches: impl FnOnce(u64) -> Result<bool, E>,
    ) -> Result<bool, E> {
        let index = seen(self.reached, None, read, reaches)?;
        Ok(self.observe(index))
    }

    /// The host set `value`. Whether the value rose.
    pub(super) fn set(&mut self, value: u64) -> bool {
        let before = self.value();
        self.floor = self.floor.max(value);
        self.value() > before
    }

    /// How a batch submitted now waits for `value`; `None` when no submitted signal and no host
    /// set reaches it yet, and the batch must be held.
    pub(super) fn wait(&self, value: u64) -> Option<WaitFor> {
        if value <= self.value() {
            return Some(WaitFor::Reached);
        }
        self.pending
            .iter()
            .find(|pending| pending.value >= value)
            .map(|pending| WaitFor::Index(pending.index))
    }

    /// What a GPU signal of `value` by a batch submitted now on `queue` becomes. Nothing changes
    /// until [`EventTimeline::keep`].
    pub(super) fn signal(&self, value: u64, queue: Q) -> Signal {
        if value <= self.known() {
            return Signal::Dropped;
        }
        match self.pending.back() {
            Some(last) => Signal::Kept {
                index: last.index + 1,
                after: (last.queue != queue).then_some(last.index),
            },
            None => Signal::Kept {
                index: self.reached + 1,
                after: None,
            },
        }
    }

    /// The batch of a kept signal (`index` from [`EventTimeline::signal`]) of `value` on `queue`
    /// was submitted.
    pub(super) fn keep(&mut self, index: u64, value: u64, queue: Q) {
        self.pending.push_back(Pending {
            index,
            value,
            queue,
        });
    }

    /// The kept signals the semaphore has not been seen to reach.
    #[cfg(test)]
    pub(super) fn pending(&self) -> usize {
        self.pending.len()
    }
}
