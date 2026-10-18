//! Deferred command buffers' handles: each is the address of one entry of a table of this
//! process's, so it never equals a driver's command buffer (a dispatchable handle is a pointer to
//! the driver's or the loader's own memory), and the entry holds its log (`recorder`).
//!
//! - **Made** by an encoder's command buffer from its pool's place ([`make`]), for the command
//!   buffer's [`Replays`]; none once [`HANDLES`] are live together, where the command buffer takes
//!   a driver's command buffer instead, recorded as before.
//! - **Found** by each routed entry point with one subtraction and one load ([`log`]); a
//!   driver's command buffer is outside the table.
//! - **Let go of** with their command buffer's replays (`Replays::release`): the logs keep their
//!   room for the next ones.

use super::log::{Log, Stage};
use super::replays::Replays;
use crate::device::lock;
use ash::vk::{self, Handle};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

/// How many deferred command buffers may be live together in the process.
pub(super) const HANDLES: usize = 1 << 16;

/// Each handle's log; null while it is free.
static LOGS: [AtomicPtr<Log>; HANDLES] = [const { AtomicPtr::new(std::ptr::null_mut()) }; HANDLES];

/// The free handles and the logs kept for reuse.
struct Free {
    /// Freed handles, reused first.
    indices: Vec<u32>,
    /// The next handle never made.
    next: u32,
    /// Logs let go of, emptied, their room kept.
    logs: Vec<Kept>,
}

/// A log kept for reuse in the box a handle's entry points to: a handle made takes it with no
/// allocation.
struct Kept(Box<Log>);

static FREE: Mutex<Free> = Mutex::new(Free {
    indices: Vec::new(),
    next: 0,
    logs: Vec::new(),
});

/// The most logs [`FREE`] keeps for reuse.
const KEPT_LOGS: usize = 1024;

/// The handle of entry `index`.
pub(super) fn handle(index: u32) -> vk::CommandBuffer {
    vk::CommandBuffer::from_raw(std::ptr::from_ref(&LOGS[index as usize]) as u64)
}

/// The entry `raw` names, where it is a deferred command buffer's handle.
pub(super) fn index(raw: vk::CommandBuffer) -> Option<u32> {
    let base = LOGS.as_ptr() as u64;
    let offset = raw.as_raw().wrapping_sub(base);
    let size = std::mem::size_of::<AtomicPtr<Log>>() as u64;
    (offset < HANDLES as u64 * size && offset.is_multiple_of(size))
        .then_some((offset / size) as u32)
}

/// Whether `raw` is a deferred command buffer's handle.
pub(crate) fn is_deferred(raw: vk::CommandBuffer) -> bool {
    index(raw).is_some()
}

/// Log a command into `raw`'s log with `log`, where `raw` is a live deferred command buffer: its
/// answer. Where the command ended the log, its replays take it once `log`'s borrow of it ended,
/// so a replay, which may start at once, reads it with no other borrow of it live.
///
/// # Safety
///
/// The caller records `raw` alone, as Vulkan's external synchronization of a command buffer asks.
pub(super) unsafe fn record<R>(
    raw: vk::CommandBuffer,
    log: impl FnOnce(&mut Log) -> R,
) -> Option<R> {
    let index = index(raw)?;
    let entry = LOGS[index as usize].load(Ordering::Acquire);
    if entry.is_null() {
        return None;
    }
    // SAFETY: a live entry's log lives until its handle is let go of, which its command buffer
    // does only once nothing records it; only the caller records it, and no replay reads it
    // before it ended (below).
    let answer = log(unsafe { &mut *entry });
    // SAFETY: as above; read only, the borrow above having ended.
    let logged = unsafe { &*entry };
    if logged.stage == Stage::Ended {
        if let Some(replays) = logged.replays() {
            replays.ended(logged.index());
        }
    }
    Some(answer)
}

/// The log of `raw`, where it is a live deferred command buffer, to read.
///
/// # Safety
///
/// The log ended (only replays read it from then on, and only its release writes it).
pub(super) unsafe fn log<'a>(raw: vk::CommandBuffer) -> Option<&'a Log> {
    let index = index(raw)?;
    let log = LOGS[index as usize].load(Ordering::Acquire);
    // SAFETY: the caller's; a live entry's log lives until its handle is let go of.
    unsafe { log.as_ref() }
}

/// The log at `index`, live and ended, to read.
///
/// # Safety
///
/// As [`log`].
pub(super) unsafe fn log_at<'a>(index: u32) -> &'a Log {
    let log = LOGS[index as usize].load(Ordering::Acquire);
    // SAFETY: the caller's: the entry is live.
    unsafe { &*log }
}

/// A new deferred command buffer of `level` for `replays`, its index; `None` where every handle
/// is live.
pub(super) fn make(level: vk::CommandBufferLevel, replays: &Arc<Replays>) -> Option<u32> {
    let (index, kept) = {
        let mut free = lock(&FREE);
        let index = match free.indices.pop() {
            Some(index) => index,
            None if (free.next as usize) < HANDLES => {
                free.next += 1;
                free.next - 1
            }
            None => return None,
        };
        (index, free.logs.pop())
    };
    let mut log = match kept {
        Some(Kept(mut log)) => {
            log.clear(level);
            log
        }
        None => Box::new(Log::new(level)),
    };
    log.own(index, Arc::clone(replays));
    LOGS[index as usize].store(Box::into_raw(log), Ordering::Release);
    Some(index)
}

/// Let go of the handles `indices`: their logs go back for reuse.
///
/// # Safety
///
/// Nothing records or reads any of them, nor will: their command buffer let go of them.
pub(super) unsafe fn free(indices: &[u32]) {
    let mut free = lock(&FREE);
    for &index in indices {
        let log = LOGS[index as usize].swap(std::ptr::null_mut(), Ordering::AcqRel);
        if log.is_null() {
            continue;
        }
        // SAFETY: made by `Box::into_raw` in `make`, taken out of the table just now.
        let mut log = unsafe { Box::from_raw(log) };
        // Its hold on its replays goes: never their last, which their command buffer holds.
        log.disown();
        if free.logs.len() < KEPT_LOGS {
            free.logs.push(Kept(log));
        }
    }
    free.indices.extend_from_slice(indices);
}
