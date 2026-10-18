//! A command buffer's deferred command buffers and their replay, and the device's recorder thread
//! (`recorder`).

use super::handles;
use super::log::Log;
use super::thunks::{Command, CommandsReplayed};
use crate::buffer::{HeldByReplay, HostPages};
use crate::command_buffer::Item;
use crate::contexts::Contexts;
use crate::device::functions::{Functions, Vk};
use crate::device::lock;
use crate::exit_guard::{self, Call};
use crate::loader_word::LoaderWord;
use crate::spares::CommandPool;
use ash::vk::{self, Handle};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// What a device's replays share: its slot's commands, its loader word for the command buffers
/// they make, and the recorder thread's queue. The device's functions are each replay's caller's
/// to lend.
pub(crate) struct RecorderShared {
    slot: usize,
    loader_word: LoaderWord,
    queue: Mutex<Queue>,
    wake: Condvar,
    /// Where pages the replays held last go (`Device::elsewhere`), set once the device is.
    elsewhere: OnceLock<(Contexts<Item>, u64)>,
    /// Replays let go of, emptied, their room kept: a command buffer takes one with no
    /// allocation. Emptied as the recorder stops (each holds this).
    kept: Mutex<Vec<Arc<Replays>>>,
    /// Held through each [`settle_all`](Self::settle_all), so one finds the replays another
    /// took off the queue replayed.
    settling: Mutex<()>,
    /// Command buffers replayed by the recorder's thread, and by a thread that submits.
    pub(super) by_recorder: AtomicU64,
    pub(super) by_submitter: AtomicU64,
    /// Command pools the replays made, where their command buffer took no spare.
    pools_made: AtomicU64,
    /// Replays that recorded into their pool's primary begun ahead (`spares.rs`).
    begun_ahead_taken: AtomicU64,
    /// The commands the replays called, one count per command at its index
    /// ([`CommandsReplayed`]).
    commands: [AtomicU64; Command::COUNT],
}

/// What a device's replays did so far: the command buffers replayed by the recorder's thread and
/// by threads that submit, the command pools they made, those replays that recorded into their
/// pool's primary begun ahead, and the commands they called, by kind.
#[derive(Clone, Copy, Default)]
pub(crate) struct ReplayCounts {
    pub(crate) by_recorder: u64,
    pub(crate) by_submitter: u64,
    pub(crate) pools_made: u64,
    pub(crate) begun_ahead_taken: u64,
    pub(crate) commands: CommandsReplayed,
}

/// The replays the recorder's queue holds from its making without growing: more than a frame
/// commits, as the scheduler's inbox holds (`scheduler::inbox::ROOM`).
const QUEUE_ROOM: usize = crate::scheduler::inbox::ROOM;

/// The recorder thread's queue.
#[derive(Default)]
struct Queue {
    /// Committed command buffers' replays, in commit order. It starts with room for
    /// [`QUEUE_ROOM`], keeps its room, and the recorder's thread grows it ahead of the commits
    /// ([`take_leaving_room`]).
    replays: VecDeque<Arc<Replays>>,
    /// The replays the thread replays now, taken off `replays`.
    current: Option<Arc<Replays>>,
    /// The thread waits for a replay.
    parked: bool,
    stop: bool,
}

impl Queue {
    /// An empty queue, with room for [`QUEUE_ROOM`] replays.
    fn new() -> Queue {
        Queue {
            replays: VecDeque::with_capacity(QUEUE_ROOM),
            ..Queue::default()
        }
    }
}

/// How many replays a new recorder's queue holds without growing.
#[cfg(test)]
pub(crate) fn queue_room() -> usize {
    Queue::new().replays.capacity()
}

/// The most replays [`RecorderShared`] keeps for reuse.
const KEPT_REPLAYS: usize = 1024;

/// The first of `queue`, taken off, and `queue` left room for twice as many as waited, made on
/// the thread that takes: the recorder's, for its queue, so a commit's push
/// ([`RecorderShared::enqueue`]) grows the queue, an allocation on the committing thread, only
/// where more wait than [`QUEUE_ROOM`] and twice as many as the recorder's thread ever found, as
/// the scheduler's inbox leaves its commits room (`scheduler/inbox.rs`).
pub(crate) fn take_leaving_room<T>(queue: &mut VecDeque<T>) -> Option<T> {
    let waited = queue.len();
    let first = queue.pop_front()?;
    queue.reserve(waited + 1);
    Some(first)
}

/// One command buffer's deferred command buffers (the module docs of `recorder`).
pub(crate) struct Replays {
    state: Mutex<ReplayState>,
    recorder: Arc<RecorderShared>,
}

struct ReplayState {
    /// The queue family its command buffer runs on: its pool's.
    family: u32,
    /// Every handle made for it, let go of in its release.
    handles: Vec<u32>,
    /// Those ended and not yet replayed, in the order they ended. It has room for every handle
    /// made ([`Replays::make`]), so an end, the commit's included, never grows it.
    ended: VecDeque<u32>,
    /// Its command buffer committed: its ended command buffers may be replayed.
    open: bool,
    /// It waits in the recorder thread's queue.
    queued: bool,
    /// The pool its replays allocate from: a spare its command buffer took, or one made at the
    /// first replay.
    pool: Option<CommandPool>,
    /// The command buffers of the pool that replays ended.
    ended_real: u64,
    /// A replay's failure (`vkBeginCommandBuffer` or `vkEndCommandBuffer`): the submission of
    /// what it replayed fails with it.
    failure: Option<vk::Result>,
    /// The host pages what ended names, which its command buffer held only weakly from its
    /// commit (`rebind.rs`): held here until nothing ended waits for its replay.
    held: Vec<Arc<HostPages>>,
}

impl RecorderShared {
    /// A device's, through slot `slot`, with its `loader_word`.
    pub(crate) fn new(slot: usize, loader_word: LoaderWord) -> Arc<Self> {
        Arc::new(RecorderShared {
            slot,
            loader_word,
            queue: Mutex::new(Queue::new()),
            wake: Condvar::new(),
            elsewhere: OnceLock::new(),
            kept: Mutex::default(),
            settling: Mutex::default(),
            by_recorder: AtomicU64::new(0),
            by_submitter: AtomicU64::new(0),
            pools_made: AtomicU64::new(0),
            begun_ahead_taken: AtomicU64::new(0),
            commands: [const { AtomicU64::new(0) }; Command::COUNT],
        })
    }

    /// Where the pages the replays held last go from now on: `elsewhere`, the device's
    /// (`Device::elsewhere`).
    pub(crate) fn set_elsewhere(&self, elsewhere: (Contexts<Item>, u64)) {
        let _ = self.elsewhere.set(elsewhere);
    }

    /// Hand `replays` to the recorder thread, waking it where it waits.
    fn enqueue(&self, replays: Arc<Replays>) {
        let mut queue = lock(&self.queue);
        queue.replays.push_back(replays);
        if queue.parked {
            self.wake.notify_one();
        }
    }

    /// The recorder thread's loop: each queued command buffer's ended deferred command buffers
    /// replayed, inside a section, with the device's `functions`, until [`stop`](Self::stop).
    pub(crate) fn run(&self, functions: &Functions) {
        loop {
            let replays = {
                let mut queue = lock(&self.queue);
                loop {
                    if let Some(replays) = take_leaving_room(&mut queue.replays) {
                        queue.current = Some(Arc::clone(&replays));
                        break replays;
                    }
                    if queue.stop {
                        return;
                    }
                    queue.parked = true;
                    queue = self
                        .wake
                        .wait(queue)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    queue.parked = false;
                }
            };
            // Once the process exits, nothing is replayed here: a submission replays what it
            // submits itself, in its own section, or fails.
            if let Some(_busy) = exit_guard::enter() {
                let replayed = replays.drain(functions.get(), true);
                self.by_recorder.fetch_add(replayed, Ordering::Relaxed);
            }
            lock(&self.queue).current = None;
            // Released while it waited here (a submission replayed it first), it is kept now.
            replays.keep();
        }
    }

    /// Replay, on this thread, with `device`, the device's functions, everything ended in a
    /// committed command buffer the recorder's thread has yet to replay, waiting for the replay
    /// it is in, and for another thread's settle: what the release, rebind or detach of host
    /// pages that replays hold does first, so the replays let go of them before it
    /// (`HostPages::let_go_by_buffer`). Once the process exits, what ended is dropped
    /// unreplayed instead ([`Replays::replay_ended`]).
    pub(crate) fn settle_all(&self, device: &Vk) {
        let _settling = lock(&self.settling);
        let (queued, current) = {
            let mut queue = lock(&self.queue);
            let queued: Vec<_> = queue.replays.drain(..).collect();
            (queued, queue.current.clone())
        };
        let mut replayed = 0;
        for replays in queued {
            replayed += replays.drain(device, true);
        }
        // The thread's replay ended once this takes its lock: what it left, nothing as a rule.
        if let Some(replays) = current {
            replayed += replays.drain(device, false);
        }
        self.by_submitter.fetch_add(replayed, Ordering::Relaxed);
    }

    /// End the thread's loop once its queue is empty, and let go of the replays kept.
    pub(crate) fn stop(&self) {
        lock(&self.queue).stop = true;
        self.wake.notify_one();
        lock(&self.kept).clear();
    }

    /// A command buffer's replays, on queue family `family`, allocating from `pool` (a spare it
    /// took) or a pool the first replay makes: a kept one where there is one.
    pub(crate) fn replays(
        self: &Arc<Self>,
        family: u32,
        pool: Option<CommandPool>,
    ) -> Arc<Replays> {
        let kept = lock(&self.kept).pop();
        let Some(replays) = kept else {
            return Arc::new(Replays {
                state: Mutex::new(ReplayState {
                    family,
                    handles: Vec::new(),
                    ended: VecDeque::new(),
                    open: false,
                    queued: false,
                    pool,
                    ended_real: 0,
                    failure: None,
                    held: Vec::new(),
                }),
                recorder: Arc::clone(self),
            });
        };
        {
            let mut state = lock(&replays.state);
            state.family = family;
            state.pool = pool;
        }
        replays
    }

    /// What the replays did so far ([`ReplayCounts`]).
    pub(crate) fn counts(&self) -> ReplayCounts {
        ReplayCounts {
            by_recorder: self.by_recorder.load(Ordering::Relaxed),
            by_submitter: self.by_submitter.load(Ordering::Relaxed),
            pools_made: self.pools_made.load(Ordering::Relaxed),
            begun_ahead_taken: self.begun_ahead_taken.load(Ordering::Relaxed),
            commands: CommandsReplayed::read(&self.commands),
        }
    }
}

impl Replays {
    /// A new deferred command buffer of `level`; `None` where every handle is live. `ended` is
    /// given room for it here, as `handles` is: each handle ends once, so its end, which may be
    /// the commit's (`CommandBufferShared::close_tail`), allocates nothing. A kept replays' room
    /// stays from its last command buffer, so this grows nothing that one did not need.
    pub(crate) fn make(
        self: &Arc<Self>,
        level: vk::CommandBufferLevel,
    ) -> Option<vk::CommandBuffer> {
        let index = handles::make(level, self)?;
        let mut state = lock(&self.state);
        state.handles.push(index);
        let unended = state.handles.len().saturating_sub(state.ended.len());
        state.ended.reserve(unended);
        Some(handles::handle(index))
    }

    /// How many deferred command buffers were made for it, and how many may end without
    /// growing `ended`.
    #[cfg(test)]
    pub(crate) fn room_to_end(&self) -> (usize, usize) {
        let state = lock(&self.state);
        (state.handles.len(), state.ended.capacity())
    }

    /// Deferred command buffer `index` ended: replayed once its command buffer committed.
    pub(super) fn ended(self: &Arc<Self>, index: u32) {
        let mut state = lock(&self.state);
        state.ended.push_back(index);
        if state.open && !state.queued {
            state.queued = true;
            drop(state);
            self.recorder.enqueue(Arc::clone(self));
        }
    }

    /// Hold the host pages what ended names, which `fill` adds to what it holds, until nothing
    /// ended waits for its replay: an encoder encoded again before its command buffer's first
    /// submission (`reencode.rs`). As at the commit ([`commit`](Self::commit)), the pages it held
    /// last are the answer, for the caller to let go of where it lets go of them without a
    /// recorder. `fill` runs under its lock, which it must not take again.
    pub(crate) fn hold(
        self: &Arc<Self>,
        device: &Vk,
        fill: impl FnOnce(&mut Vec<Arc<HostPages>>),
    ) -> Vec<HostPages> {
        let mut state = lock(&self.state);
        Self::fill(&mut state, fill);
        self.settle(device, &mut state)
    }

    /// Its command buffer commits, its commands' holds on host pages `fill` adds to what it holds
    /// ([`hold`](Self::hold)): what ended is replayed, on the recorder thread, and what ends from
    /// now on; where nothing ended waits, what it holds goes now. `fill` runs under its lock,
    /// which it must not take again.
    ///
    /// Where the buffer of some of those pages let go of them (released, rebound or detached
    /// while an encoder took them, on any thread), the commit would let go of them last without
    /// a recorder: what ended replays now, on this thread, with `device`, the device's
    /// functions. The pages it held last are the answer, for the caller to let go of where the
    /// commit lets go of them without a recorder. Nothing as a rule.
    pub(crate) fn commit(
        self: &Arc<Self>,
        device: &Vk,
        fill: impl FnOnce(&mut Vec<Arc<HostPages>>),
    ) -> Vec<HostPages> {
        let mut state = lock(&self.state);
        Self::fill(&mut state, fill);
        state.open = true;
        self.settle(device, &mut state)
    }

    /// What `state` holds once its command buffer committed (nothing before), under its lock:
    /// queued for the recorder where something ended waits, then each page's buffer's let-go
    /// read, after its hold was counted and the replays queued, so a let-go after the read finds
    /// both (`HostPages::let_go_by_buffer`). Where one came before, everything ended replays now,
    /// on this thread, with `device`; where nothing ended waits, what it holds goes now. The
    /// pages it held last are the answer.
    fn settle(self: &Arc<Self>, device: &Vk, state: &mut ReplayState) -> Vec<HostPages> {
        let mut last = Vec::new();
        if !state.open {
            return last;
        }
        if !state.ended.is_empty() {
            if !state.queued {
                state.queued = true;
                self.recorder.enqueue(Arc::clone(self));
            }
            if state.held.iter().any(|pages| pages.buffer_let_go()) {
                let replayed = self.replay_ended(device, state);
                self.recorder
                    .by_submitter
                    .fetch_add(replayed, Ordering::Relaxed);
            }
        }
        if state.ended.is_empty() {
            self.let_go(state, Some(&mut last));
        }
        last
    }

    /// Add what `fill` adds to what `state` holds, counted in each page's holds.
    fn fill(state: &mut ReplayState, fill: impl FnOnce(&mut Vec<Arc<HostPages>>)) {
        let before = state.held.len();
        fill(&mut state.held);
        for pages in &state.held[before..] {
            pages.held_for_replay();
        }
    }

    /// Replay every ended command buffer not yet replayed, in the order they ended, waiting for
    /// a replay in progress on another thread: how many this call replayed. On the recorder's
    /// thread (`queued`, which it takes off), a submitting one, or one that settles, with
    /// `device`, the device's functions.
    fn drain(&self, device: &Vk, queued: bool) -> u64 {
        let mut state = lock(&self.state);
        if queued {
            state.queued = false;
        }
        let replayed = self.replay_ended(device, &mut state);
        // Nothing ended waits now: what it held for the replays goes.
        if state.open {
            self.let_go(&mut state, None);
        }
        replayed
    }

    /// Replay every ended command buffer of `state` not yet replayed, in the order they ended,
    /// inside a section, with `device`, the device's functions: how many. Once the process
    /// exits, outside a section, they are dropped unreplayed instead, and their submission fails
    /// ([`resolve`]), as a submission does then.
    fn replay_ended(&self, device: &Vk, state: &mut ReplayState) -> u64 {
        let Some(_busy) = exit_guard::enter() else {
            state.ended.clear();
            return 0;
        };
        let mut replayed = 0;
        while let Some(index) = state.ended.pop_front() {
            self.replay(device, state, index);
            replayed += 1;
        }
        replayed
    }

    /// Let go of what `state` holds, under its lock, running nothing of the program's. Where
    /// `last` is given (a commit's, [`settle`](Self::settle)), the pages held last go to it,
    /// for the caller to let go of. Elsewhere none is held last: the release, rebind or detach
    /// of pages held here replays first, and a commit after it replays itself
    /// (`HostPages::let_go_by_buffer`). Were one held last, it would go on the device's handler
    /// threads (`Device::elsewhere`), as its deallocator may run the program's code, which may
    /// not run on the recorder's thread nor under a submission's locks: announced there
    /// (`HeldByReplay`).
    fn let_go(&self, state: &mut ReplayState, mut last: Option<&mut Vec<HostPages>>) {
        for pages in state.held.drain(..) {
            let Some(pages) = HostPages::let_go_by_replay(pages) else {
                continue;
            };
            if let Some(last) = last.as_deref_mut() {
                last.push(pages);
                continue;
            }
            let last = Box::new(HeldByReplay(pages));
            match self.recorder.elsewhere.get() {
                Some((contexts, context)) => contexts.post(*context, Item::Release(last)),
                // Only before the device is made, which no command buffer is.
                None => drop(last),
            }
        }
    }

    /// Replay deferred command buffer `index` into a command buffer of the pool.
    fn replay(&self, device: &Vk, state: &mut ReplayState, index: u32) {
        // SAFETY: an ended log, which only replays read, under the replays' lock.
        let log = unsafe { handles::log_at(index) };
        let recorder = &self.recorder;
        let (real, begun) = match self.allocate(device, state, log) {
            Ok(real) => real,
            Err(error) => {
                state.failure.get_or_insert(error);
                return;
            }
        };
        let resolve = |raw: vk::CommandBuffer| -> Option<vk::CommandBuffer> {
            match handles::index(raw) {
                None => Some(raw),
                // SAFETY: a secondary the log executes ended before it (Vulkan's executable
                // state), so was replayed before it, published under this lock.
                Some(at) => unsafe { handles::log_at(at) }.real(),
            }
        };
        // SAFETY: `real` was just taken from the pool, which only replays record, under the
        // replays' lock; every object the log names is live until its command buffer completes,
        // which is after the replay (its submission waits for it) or after its release, which
        // takes this lock and replays nothing after.
        match unsafe { log.replay(recorder.slot, real, begun, &resolve, &recorder.commands) } {
            Ok(()) => {
                state.ended_real += 1;
                log.replayed(real);
            }
            Err(error) => {
                state.failure.get_or_insert(error);
            }
        }
    }

    /// A command buffer of `log`'s level from the pool, made on first use, and whether it is
    /// begun: the pool's primary begun ahead where `log` may take it (`Log::takes_begun`; the
    /// first encoder's, which an encoder of the command buffer that took the pool would take
    /// without a recorder, `spares.rs`).
    fn allocate(
        &self,
        raw: &Vk,
        state: &mut ReplayState,
        log: &Log,
    ) -> Result<(vk::CommandBuffer, bool), vk::Result> {
        let level = log.level;
        if state.pool.is_none() {
            let info = vk::CommandPoolCreateInfo::default()
                .flags(
                    vk::CommandPoolCreateFlags::TRANSIENT
                        | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                )
                .queue_family_index(state.family);
            // SAFETY: the device is live and `info` valid.
            let pool = unsafe { raw.create_command_pool(&info, None) }?;
            exit_guard::called(Call::CreateCommandPool);
            self.recorder.pools_made.fetch_add(1, Ordering::Relaxed);
            state.pool = Some(CommandPool::new(pool, state.family));
        }
        let Some(pool) = &mut state.pool else {
            return Err(vk::Result::ERROR_UNKNOWN);
        };
        if log.takes_begun() {
            if let Some(buffer) = pool.take_begun() {
                self.recorder
                    .begun_ahead_taken
                    .fetch_add(1, Ordering::Relaxed);
                return Ok((buffer, true));
            }
        }
        if let Some(buffer) = pool.reuse(level) {
            return Ok((buffer, false));
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool.raw)
            .level(level)
            .command_buffer_count(1);
        // SAFETY: the pool is this command buffer's, held under the replays' lock; the loader
        // reads the device's word first (`loader_word.rs`).
        let buffer = unsafe {
            self.recorder.loader_word.restore();
            raw.allocate_command_buffers(&info)
        }?[0];
        exit_guard::called(Call::AllocateCommandBuffers);
        pool.made(level, buffer);
        Ok((buffer, false))
    }

    /// Let go of everything: nothing ended is replayed after, a replay in progress ended first,
    /// every handle is free, and what it held goes; kept for another command buffer where
    /// nothing else holds it. The pool, with whether it is settled (every command buffer it
    /// handed out ended, `completed` where its command buffer completed), to give back to the
    /// device's spares.
    pub(crate) fn release(self: Arc<Self>, completed: bool) -> Option<(CommandPool, bool)> {
        let pool = {
            let mut state = lock(&self.state);
            state.ended.clear();
            state.open = false;
            self.let_go(&mut state, None);
            // SAFETY: nothing records these any more (their command buffer completed or is
            // dropped), and no replay reads them after this lock (`ended` is empty, `open`
            // false).
            unsafe { handles::free(&state.handles) };
            state.handles.clear();
            let settled = completed
                && state.failure.is_none()
                && state
                    .pool
                    .as_ref()
                    .is_some_and(|pool| pool.handed_out() as u64 == state.ended_real);
            state.ended_real = 0;
            state.failure = None;
            state.pool.take().map(|pool| (pool, settled))
        };
        self.keep();
        pool
    }

    /// Keep it for another command buffer where nothing else holds it, released
    /// ([`release`](Self::release)): where the recorder's queue holds it still, the recorder
    /// keeps it as it takes it.
    fn keep(self: Arc<Self>) {
        if Arc::strong_count(&self) == 1 {
            let recorder = Arc::clone(&self.recorder);
            let mut kept = lock(&recorder.kept);
            if kept.len() < KEPT_REPLAYS {
                kept.push(self);
            }
        }
    }
}

/// The command buffer to submit in place of `raw`: `raw` where it is a driver's; a deferred one's
/// replay, replayed now by this thread with `device`, the device's functions, where the
/// recorder's has not yet, waiting for a replay in progress. A replay's failure is the result.
///
/// # Safety
///
/// `raw` ended, and its command buffer committed; the caller is inside a section.
pub(crate) unsafe fn resolve(
    device: &Vk,
    raw: vk::CommandBuffer,
) -> Result<vk::CommandBuffer, vk::Result> {
    // SAFETY: an ended log (the caller's), read for what its replay published.
    let Some(log) = (unsafe { handles::log(raw) }) else {
        return Ok(raw);
    };
    if let Some(real) = log.real() {
        return Ok(real);
    }
    let Some(replays) = log.replays() else {
        return Err(vk::Result::ERROR_UNKNOWN);
    };
    let replayed = replays.drain(device, false);
    replays
        .recorder
        .by_submitter
        .fetch_add(replayed, Ordering::Relaxed);
    if let Some(real) = log.real() {
        return Ok(real);
    }
    Err(lock(&replays.state)
        .failure
        .unwrap_or(vk::Result::ERROR_UNKNOWN))
}

/// The real command buffer of a raw handle stored in a log's `real`.
pub(super) fn real_of(raw: &AtomicU64) -> Option<vk::CommandBuffer> {
    match raw.load(Ordering::Acquire) {
        0 => None,
        raw => Some(vk::CommandBuffer::from_raw(raw)),
    }
}
