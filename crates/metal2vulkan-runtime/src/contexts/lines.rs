//! The queues' serial contexts as a pure structure: one first-in-first-out line of items per
//! logical queue, each line run by at most one thread at a time, and the lines waiting for a thread
//! in the order they became ready.
//!
//! A line is ready when it holds an item and no thread runs it. A thread takes the first ready
//! line's first item and runs it; until it is done with it, that line's later items wait, while
//! every other line can be taken by another thread. A line that is done and still holds items is
//! ready again, behind the lines that became ready meanwhile.

use std::collections::{HashMap, VecDeque};

/// The items not yet run, by logical queue.
#[derive(Debug)]
pub(crate) struct Lines<I> {
    /// Each queue's line while it holds an item or a thread runs it, by the queue's identity.
    lines: HashMap<u64, Line<I>>,
    /// The ready lines, in the order they became ready.
    ready: VecDeque<u64>,
}

#[derive(Debug)]
struct Line<I> {
    items: VecDeque<I>,
    /// A thread runs the line's first item.
    running: bool,
}

impl<I> Default for Lines<I> {
    fn default() -> Lines<I> {
        Lines {
            lines: HashMap::new(),
            ready: VecDeque::new(),
        }
    }
}

impl<I> Lines<I> {
    /// Append `item` to `queue`'s line.
    pub(crate) fn push(&mut self, queue: u64, item: I) {
        let line = self.lines.entry(queue).or_insert_with(|| Line {
            items: VecDeque::new(),
            running: false,
        });
        if line.items.is_empty() && !line.running {
            self.ready.push_back(queue);
        }
        line.items.push_back(item);
    }

    /// Take the first item of the first ready line: the line is running until
    /// [`Lines::done`]. `None` when no line is ready.
    pub(crate) fn take(&mut self) -> Option<(u64, I)> {
        let queue = self.ready.pop_front()?;
        let line = self.lines.get_mut(&queue)?;
        line.running = true;
        let item = line.items.pop_front()?;
        Some((queue, item))
    }

    /// The thread running `queue`'s line is done with the item it took: the line is ready again
    /// if it holds more, and is forgotten otherwise.
    pub(crate) fn done(&mut self, queue: u64) {
        let Some(line) = self.lines.get_mut(&queue) else {
            return;
        };
        line.running = false;
        if line.items.is_empty() {
            self.lines.remove(&queue);
        } else {
            self.ready.push_back(queue);
        }
    }

    /// Take `queue`'s line for the calling thread, where no thread runs it and `plain` holds for
    /// each item it holds: those items, in order, which the caller runs before anything it runs
    /// in the line itself; the line is running until [`Lines::done`]. `None` otherwise, changing
    /// nothing.
    pub(crate) fn take_plain(
        &mut self,
        queue: u64,
        plain: impl Fn(&I) -> bool,
    ) -> Option<VecDeque<I>> {
        if let Some(line) = self.lines.get(&queue) {
            if line.running || !line.items.iter().all(plain) {
                return None;
            }
        }
        let items = self
            .lines
            .remove(&queue)
            .map_or_else(VecDeque::new, |line| line.items);
        if !items.is_empty() {
            // A line that holds an item and runs on no thread is ready, once.
            self.ready.retain(|ready| *ready != queue);
        }
        self.lines.insert(
            queue,
            Line {
                items: VecDeque::new(),
                running: true,
            },
        );
        Some(items)
    }

    /// Take `queue`'s line for the calling thread where it holds no item and no thread runs it:
    /// the line is running until [`Lines::done`]. False otherwise, changing nothing.
    pub(crate) fn take_idle(&mut self, queue: u64) -> bool {
        if self.lines.contains_key(&queue) {
            return false;
        }
        self.lines.insert(
            queue,
            Line {
                items: VecDeque::new(),
                running: true,
            },
        );
        true
    }

    /// How many lines are ready.
    pub(crate) fn ready(&self) -> usize {
        self.ready.len()
    }

    /// Whether a line is ready.
    pub(crate) fn is_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    /// Whether no line holds an item or is running.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}
