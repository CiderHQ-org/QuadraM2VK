//! The device's scheduler: one thread per device that submits the committed command buffers'
//! batches to the device's `VkQueue`s.
//!
//! - **Handed to the thread** ([`Inbox`]). `commit` leaves its batches in the inbox and returns
//!   at once, as Metal's `commit` returns before its command buffer is scheduled: it makes no
//!   pass and submits nothing, so no driver call holds it. This thread takes them in, in commit
//!   order, at its next pass, and submits what is ready. The commit wakes the thread where it is
//!   not making a pass; where it is, the thread looks at the inbox again once its pass ends. The
//!   order sees every commit in commit order, and every batch is planned as it is submitted. This
//!   thread stops looking at the inbox again at a pass that leaves it a check or a host step, and
//!   before each one it runs passes again for as long as the inbox holds commits, and looks at it
//!   before it sleeps ([`left`]): a commit waits for the pass under way, the submission handlers
//!   that pass leaves, and the pass that takes it in, never for a check or a step.
//!
//! - **The order** ([`Order`]). Each logical queue's batches wait in commit order; the head
//!   committed first among those ready goes next. A batch is ready once every event it waits for
//!   is reached by a signal already submitted or by the host ([`EventTimeline`]), so no `VkQueue`
//!   ever holds a wait for a signal that is not submitted: logical queues sharing a `VkQueue`
//!   cannot deadlock through the order they wait for each other in.
//! - **Binary semaphores** ([`BinarySignals`]). A batch that waits for one of the caller's binary
//!   semaphores is ready once every signal of it that a command buffer of this device committed
//!   before is submitted; a signal from the caller was submitted before the commit.
//! - **Tracking** ([`Tracker`]). Each batch is planned as it is submitted, against the device's
//!   hazard state, in this order: the order Metal follows across queues. A batch held for its host
//!   step or its checks keeps its place among the tracked accesses: a later commit's batch on
//!   another queue that conflicts with it waits behind it, its own host step and checks too
//!   ([`Order::behind`]).
//! - **Many logical queues on few `VkQueue`s.** A logical queue keeps one `VkQueue` (round robin
//!   at creation); each `VkQueue` signals its own timeline semaphore with every batch, which is
//!   the [`Point`] the tracker and the completion turns know the batch by.
//! - **Scheduled.** A command buffer is scheduled once its last batch is submitted (its every
//!   wait was satisfied): this thread posts its transition to `Scheduled` to its queue's context ([`contexts`](super::contexts)), which
//!   runs its scheduled handlers. No submission runs a scheduled handler, so none holds back a
//!   submission.
//! - **Submission handlers** (a runtime extension). This thread runs a command buffer's
//!   submission handlers once it submitted its last batch, in submission order among the command
//!   buffers its passes submitted, once it let go of the lock and before it lets go of the command
//!   buffers, before its next pass, which they hold back. The command buffer's transition to
//!   `Scheduled` waits for them.
//! - **Host steps**. A batch that executes commands the GPU may write
//!   (`CommandBufferShared::needs_host_step`) is held, once otherwise ready, until every
//!   `VkQueue` completed what was submitted to it by then (its gate, a snapshot of the queues'
//!   values): only then are the records the work before it wrote known. The thread then records
//!   the batch's segments (`CommandBufferShared::host_step`) without its lock, each step a
//!   section of the exit guard, and submits the batch in its order. Meanwhile it submits other
//!   ready batches; the completion turns wake it as batches complete while a gate is pending.
//! - **Submission checks** (`command_buffer/checks.rs`). A batch of a command buffer with a check
//!   is held, once otherwise ready and its host step recorded, until the checks of Metal's
//!   submissions in it ran (`CommandBufferShared::run_checks`), which the thread runs as it runs
//!   host steps: without its lock, each pass a section of the exit guard. Its checks wait, besides,
//!   until every event value it waits for was reached on the device or by the host: a value only a
//!   submitted GPU signal reaches is held for until that signal landed, as Metal submits (and
//!   checks) only then. A completion turn sees each finished batch's event signals before it
//!   wakes the thread, as for a gate. A command buffer without a check costs one flag per batch.
//! - **Waits are bounded.** The thread wakes on a commit or a host signal, and at least every
//!   [`WAIT_SLICE`], so a stop is seen promptly.
//! - **Exit.** Each submission is a section of the exit guard ([`exit_guard`]); once the process
//!   is exiting, the thread ends, and what it holds stays held.
//! - **Let go without the lock.** A submitted batch's hold of its command buffer goes after the
//!   lock: it may be the last reference, and the command buffer's device may go with it, whose
//!   drop stops this thread.
//!
//! Locks are taken in one order: the scheduler's state, then its inbox's, then a command buffer's,
//! then an event's or a `VkQueue`'s, then the handler threads' state, then the queues' waiters'
//! list. No call made under the scheduler's lock takes it again.
//!
//! [`EventTimeline`]: super::event_timeline::EventTimeline
//! [`Point`]: super::tracker::Point

pub(super) mod binary;
pub(super) mod inbox;
pub(super) mod left;
pub(super) mod order;

use super::command_buffer::gate::Hold;
use super::command_buffer::kept_barriers::KeptBarriers;
use super::command_buffer::BatchLists;
use super::command_buffer::CommandBufferShared;
use super::device::lock;
use super::error::{Error, MemoryKind};
use super::event::EventShared;
use super::event_timeline::WaitFor;
use super::exit_guard;
use super::replay::SpareLogs;
use super::resource::ResourceKey;
use super::tracker::{self, Footprint, Tracker};
use ash::vk;
use binary::{BinarySignals, Needs};
use inbox::Inbox;
use left::{work_through, Kind, Left};
use order::{Behind, Order};
use std::collections::BTreeMap;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// How long the thread sleeps at most without looking for work.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// One device's scheduler thread.
pub(crate) struct Scheduler {
    shared: Arc<SchedulerShared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

struct SchedulerShared {
    /// The state, under the lock a pass holds, with the commits a pass under way takes in.
    inbox: Inbox<State, Committed>,
    wake: Condvar,
    /// A batch waits at its gate for work to complete: the completion turns wake the thread
    /// as batches complete.
    gated: AtomicBool,
    /// The device's cleared encoder logs (`replay.rs`): an encoder takes one, and its log comes
    /// back as the encoder ends, or in its command buffer's release.
    spare_logs: SpareLogs,
}

/// What the scheduler submits from, and the device state it submits against.
pub(crate) struct State {
    order: Order<Held>,
    /// The device's hazard state, in submission order.
    pub(crate) tracker: Tracker<ResourceKey>,
    /// Per `VkQueue`: the timeline value of the last batch submitted there.
    pub(crate) values: Vec<u64>,
    /// Per `VkQueue`, per timeline whose completed writes its batches wait for (each `VkQueue`'s,
    /// then the device's two own, `device/submit.rs`): the value a batch there last waited for,
    /// which made what completed through it visible to every later command there
    /// (`command_buffer/submit.rs`).
    pub(crate) waited: Vec<Vec<u64>>,
    /// The binary semaphore signals the device's command buffers committed and submitted.
    binary: BinarySignals<vk::Semaphore>,
    /// A commit or a host signal came since the thread last looked.
    dirty: bool,
    stopping: bool,
    /// The textures whose decoded twins have CPU decodes queued or running, each with how many
    /// (`texture/twin_decode.rs`): a batch whose footprint names one waits for them.
    decodes: BTreeMap<ResourceKey, usize>,
    /// How many of `decodes` are of untracked textures, whose uses no footprint names: every
    /// batch waits while one is left.
    untracked_decodes: usize,
    /// The batches' barrier command buffers, recorded once and submitted again
    /// (`command_buffer/kept_barriers.rs`).
    pub(crate) kept: KeptBarriers,
    /// The lists each batch's submission fills and empties, with their room
    /// (`command_buffer/submit.rs`).
    pub(crate) lists: BatchLists,
}

/// A commit not yet taken into the order: the command buffer, its logical queue, its batches.
struct Committed {
    buffer: Hold<CommandBufferShared>,
    queue: u64,
    batches: Vec<BatchSync>,
}

/// A committed batch not yet submitted.
struct Held {
    buffer: Hold<CommandBufferShared>,
    /// Its index among the command buffer's batches.
    index: usize,
    /// Whether it is the command buffer's last batch.
    last: bool,
    /// The events it waits for, with the values.
    waits: Vec<(Arc<EventShared>, u64)>,
    /// What its binary semaphore waits need submitted first.
    binary_needs: Needs<vk::Semaphore>,
    /// The binary semaphores it signals.
    binary_signals: Vec<vk::Semaphore>,
    /// It holds segments, so it may need a host step.
    segments: bool,
    /// For its host step: each `VkQueue`'s value of the last batch submitted there once the batch
    /// was otherwise ready, which the queue's completed value reaches before the step.
    gate: Option<Vec<u64>>,
    /// Its command buffer has a submission check, so it may wait for checks to run.
    checked: bool,
}

/// What a batch's host step waits for (`CommandBufferShared::step_waits`).
pub(crate) struct StepWaits {
    /// The work submitted before it to complete (its gate): the GPU may write what it reads.
    pub(crate) gpu: bool,
    /// A pipeline it records, still being made.
    pub(crate) pipelines: bool,
}

/// What one committed batch waits for and signals, as the scheduler orders it.
pub(crate) struct BatchSync {
    /// The events it waits for, with the values.
    pub(crate) waits: Vec<(Arc<EventShared>, u64)>,
    /// The caller's binary semaphores it waits for.
    pub(crate) binary_waits: Vec<vk::Semaphore>,
    /// The caller's binary semaphores it signals.
    pub(crate) binary_signals: Vec<vk::Semaphore>,
    /// It holds segments, so it may need a host step.
    pub(crate) segments: bool,
    /// Its command buffer has a submission check, so it may wait for checks to run.
    pub(crate) checked: bool,
}

impl Scheduler {
    /// Start the thread of a device with `queues` `VkQueue`s; refused when the host cannot start
    /// one.
    pub(crate) fn start(queues: usize) -> Result<Scheduler, Error> {
        let shared = Arc::new(SchedulerShared {
            inbox: Inbox::new(State {
                order: Order::default(),
                tracker: Tracker::default(),
                values: vec![0; queues],
                waited: vec![vec![0; queues + 2]; queues],
                binary: BinarySignals::default(),
                dirty: false,
                stopping: false,
                decodes: BTreeMap::new(),
                untracked_decodes: 0,
                kept: KeptBarriers::default(),
                lists: BatchLists::default(),
            }),
            wake: Condvar::new(),
            gated: AtomicBool::new(false),
            spare_logs: SpareLogs::default(),
        });
        let thread = {
            let shared = Arc::clone(&shared);
            exit_guard::spawn_user_interactive(
                std::thread::Builder::new().name("metal2vulkan-scheduler".into()),
                move || run(&shared),
            )
            .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?
        };
        Ok(Scheduler {
            shared,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Hold the batches of `buffer`, committed now on logical queue `queue`: what each batch
    /// waits for and signals, in order. They go to the thread's inbox, and `commit` returns at
    /// once, with nothing submitted: the thread takes them in at its next pass, woken for it
    /// where it is not making one (the module docs).
    pub(crate) fn commit(
        &self,
        buffer: &Arc<CommandBufferShared>,
        queue: u64,
        batches: Vec<BatchSync>,
    ) {
        let shared = &*self.shared;
        let committed = Committed {
            buffer: buffer.hold(),
            queue,
            batches,
        };
        buffer.device().shared().counts.commit_handed_off();
        if (shared.inbox).post(committed, |state| state.dirty = true) {
            shared.wake.notify_all();
        }
    }

    /// What makes the thread look for work again from any thread: a pipeline a held batch's
    /// host step waits for was made (`render/pending.rs`). It holds the thread's state weakly,
    /// so a making it waits on keeps nothing of the scheduler alive.
    pub(crate) fn waker(&self) -> impl Fn() + Clone + Send + 'static {
        let shared = Arc::downgrade(&self.shared);
        move || {
            if let Some(shared) = shared.upgrade() {
                lock(shared.inbox.state()).dirty = true;
                shared.wake.notify_all();
            }
        }
    }

    /// Whether every batch committed so far was submitted: none is held for an event, a binary
    /// semaphore, a host step or a check, or waits in the inbox (`texture/storage_form.rs`).
    pub(crate) fn holds_none(&self) -> bool {
        let inbox = &self.shared.inbox;
        lock(inbox.state()).order.is_empty() && !inbox.holds_commits()
    }

    /// A CPU decode into the decoded twin of the resource `key` was queued: every batch whose
    /// footprint names it waits at its submission until [`decode_finished`](Self::decode_finished)
    /// says so of every one queued; where it is not `tracked` (its uses not recorded), every
    /// batch. A batch already submitted is not waited for.
    pub(crate) fn decode_queued(&self, key: ResourceKey, tracked: bool) {
        let mut state = lock(self.shared.inbox.state());
        *state.decodes.entry(key).or_default() += 1;
        if !tracked {
            state.untracked_decodes += 1;
        }
    }

    /// One CPU decode into the decoded twin of `key` (`tracked` as it was queued) finished: the
    /// batches waiting for its decodes go once none is left.
    pub(crate) fn decode_finished(&self, key: ResourceKey, tracked: bool) {
        let mut state = lock(self.shared.inbox.state());
        if let Some(count) = state.decodes.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                state.decodes.remove(&key);
            }
        }
        if !tracked {
            state.untracked_decodes -= 1;
        }
        state.dirty = true;
        drop(state);
        self.shared.wake.notify_all();
    }

    /// Whether this thread is the scheduler's (a submission check runs on it).
    pub(crate) fn is_current(&self) -> bool {
        lock(&self.thread)
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id())
    }

    /// Look for work again: an event's value rose, a batch completed, or a commit's thread left
    /// what other threads committed to the thread ([`inbox::Outcome::unattended`]). Where a pass
    /// is under way, it passes again once done, and wakes the thread for a step or a check that
    /// finds; else the thread is woken here, under the lock, which then only a short hold or a
    /// pass made after this one's change has ([`Inbox::wake`]): the waker never waits for a
    /// pass's submissions.
    pub(crate) fn wake(&self) {
        if (self.shared.inbox).wake(|state| state.dirty = true) {
            self.shared.wake.notify_all();
        }
    }

    /// Batches completed, their queues' completed values and their event signals published: look
    /// for work again when a batch waits at its gate or for a GPU signal to land. The flag is read
    /// after a fence, as the thread sets it before it reads the completed values or the events,
    /// so one of the two sees the other.
    pub(crate) fn completions(&self) {
        fence(Ordering::SeqCst);
        if self.shared.gated.load(Ordering::SeqCst) {
            self.wake();
        }
    }

    /// The device's cleared encoder logs (`replay.rs`).
    pub(crate) fn spare_logs(&self) -> &SpareLogs {
        &self.shared.spare_logs
    }

    /// Release the kept barrier command buffers (`command_buffer/kept_barriers.rs`): the device is
    /// being torn down, its queues idle.
    pub(crate) fn release_kept(&self, raw: &ash::Device) {
        lock(self.shared.inbox.state()).kept.release(raw);
    }

    /// Ask the thread to end, and wait for it unless this is the thread. The device is being
    /// dropped, so nothing is held: every held command buffer holds its device.
    pub(crate) fn stop(&self) {
        lock(self.shared.inbox.state()).stopping = true;
        self.shared.wake.notify_all();
        let mut thread = lock(&self.thread);
        let own = thread
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id());
        if !own {
            if let Some(handle) = thread.take() {
                let _ = handle.join();
            }
        }
    }
}

/// The thread's loop. It ends when the device stops it, or once the process is exiting
/// ([`exit_guard`]): what is held then stays held, as Metal's pending work stays pending.
fn run(shared: &SchedulerShared) {
    exit_guard::serves_completion();
    // What its rounds let go of and the work they leave it, emptied after each waking and kept
    // with their room.
    let (mut passed, mut left) = (Passed::default(), Left::default());
    loop {
        let mut state = lock(shared.inbox.state());
        // A commit, or a nudge, left to this thread's pass may wait in the inbox (`inbox.rs`).
        while !state.dirty
            && !state.stopping
            && !shared.inbox.holds_commits()
            && !shared.inbox.nudged()
        {
            state = shared
                .wake
                .wait_timeout(state, WAIT_SLICE)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        if state.stopping {
            return;
        }
        // The pass takes the lock again, counted as a passer from before (`inbox.rs`): a commit
        // that finds it taken leaves its batches to this thread.
        drop(state);
        match work_through(
            &shared.inbox,
            &mut left,
            |left| round(shared, left, &mut passed),
            run_work,
        ) {
            None => return,
            Some(false) => {}
            // What the work did may let batches go.
            Some(true) => lock(shared.inbox.state()).dirty = true,
        }
    }
}

/// Run one piece of work the thread's passes left it, without the lock (a check runs the program's
/// code; a step records, and may make what a dispatch makes), a section of the exit guard; false,
/// with nothing run, once the process is exiting.
fn run_work(kind: Kind, work: Work) -> bool {
    let Some(busy) = exit_guard::enter() else {
        return false;
    };
    let (buffer, index) = &work;
    match kind {
        Kind::Step => {
            buffer.host_step(*index);
            for &call in exit_guard::HOST_STEP {
                exit_guard::called(call);
            }
        }
        Kind::Check => buffer.run_checks(*index),
    }
    drop(busy);
    // Out of the section: it may be the command buffer's last reference.
    drop(work);
    true
}

/// One round of the thread's: a pass, looking at the inbox again while the passes leave the
/// thread no work ([`Inbox::pass`]), what they leave added to `left` and what they let go of to
/// `passed` (empty, kept from round to round); then, without the lock, what they let go of: the
/// switches of storage forms settled ended and the submission handlers run (on this thread,
/// before its next pass; neither once the process is exiting, as the thread then runs nothing
/// more, their command buffers
/// staying `Committed`), the holds dropped (the last reference to a command buffer may be among
/// them, and with it its device, whose drop stops this thread). False once the process is
/// exiting.
fn round(shared: &SchedulerShared, left: &mut Left<Work>, passed: &mut Passed) -> bool {
    (shared.inbox).pass(|state, commits| {
        pass(state, commits, (&mut *passed, &mut *left), &shared.gated);
        !passed.exiting && left.is_empty()
    });
    if passed.exiting {
        passed.settled.clear();
        passed.handlers.clear();
    } else {
        settle(&mut passed.settled);
        run_submission_handlers(&mut passed.handlers);
    }
    passed.spent.clear();
    !passed.exiting
}

/// What passes let go of for their thread to finish without the lock ([`submit_ready`]'s
/// outputs), and whether the process is exiting.
#[derive(Default)]
struct Passed {
    spent: Spent,
    settled: Settled,
    handlers: Handlers,
    exiting: bool,
}

/// A pass of the thread's: take `commits` into the order, in commit order, then submit what is
/// ready ([`submit_ready`]) into `passed` and `left`.
fn pass(
    state: &mut State,
    commits: &mut Vec<Committed>,
    (passed, left): (&mut Passed, &mut Left<Work>),
    gated: &AtomicBool,
) {
    for Committed {
        buffer,
        queue,
        batches,
    } in commits.drain(..)
    {
        let count = batches.len();
        let State { order, binary, .. } = &mut *state;
        let held = batches.into_iter().enumerate().map(|(index, batch)| Held {
            buffer: buffer.hold(),
            index,
            last: index + 1 == count,
            binary_needs: binary.commit(&batch.binary_waits, &batch.binary_signals),
            waits: batch.waits,
            binary_signals: batch.binary_signals,
            segments: batch.segments,
            gate: None,
            checked: batch.checked,
        });
        order.push(queue, held);
        // The commit's own hold goes without the lock: its batches hold the command buffer now.
        passed.spent.push(buffer);
    }
    state.dirty = false;
    let exiting = submit_ready(
        state,
        (&mut passed.spent, &mut passed.settled),
        left,
        &mut passed.handlers,
        gated,
    )
    .is_err();
    passed.exiting |= exiting;
}

/// What a pass of [`submit_ready`] lets go of, for the caller to drop without the lock: the holds
/// of the command buffers its batches came from, the last of which may be a command buffer's last
/// reference. Nothing else: what a command buffer's commands and encoders took (what it retained,
/// its encoders' logs) goes in its release, on its queue's handler thread
/// (`CommandBufferShared::complete`), never here. A batch is published before its pass ends, so
/// its command buffer may complete, and run its release, before these holds go; a handle let go
/// of here after that would be the last, and a no-copy buffer's deallocator would run on this
/// thread, whose own hold keeps the command buffer's gate shut against the deallocator's wait for
/// it.
type Spent = Vec<Hold<CommandBufferShared>>;

/// Submit every batch that is ready, in order, until none is; each submission is a section of
/// the exit guard. A batch is published (or its failure posted) only after its section ended, so
/// the exit hook's registration at the section's end comes before anything the program can
/// learn of it (`exit_guard`). A batch's events are let go of before its command buffer
/// (`gate`), whose holds go to `spent`, for the caller to drop without the lock. A batch whose
/// host step may run now goes to `left` as a step (once), for the caller to record without the
/// lock; `gated` says whether one still waits at its gate. A batch whose submission checks may run
/// now goes to `left` as a check (once), for the caller to run without the lock. A command buffer whose
/// last batch was submitted, or that failed, and on which a fetched target's storage form rides
/// (`texture/storage_form.rs`) is settled and goes to `settled`, for the caller to announce
/// without the lock ([`settle`]). A command buffer whose last batch was submitted with
/// submission handlers goes to `handlers`, in submission order, for the caller to run them
/// without the lock ([`run_submission_handlers`]). `Err` once the process is exiting.
fn submit_ready(
    state: &mut State,
    (spent, settled): (&mut Spent, &mut Settled),
    left: &mut Left<Work>,
    handlers: &mut Handlers,
    gated: &AtomicBool,
) -> Result<(), Exiting> {
    loop {
        let busy = exit_guard::enter().ok_or(Exiting)?;
        // The heads that wait behind a batch committed before them that keeps its place: no step,
        // check or submission of theirs before it is submitted (`order.rs`).
        // Of those whose host step or checks may start (`gate_heads`, `check_heads`), and of the
        // ready ones in commit order until one goes (`Order::next`); `memo` holds what this pass
        // found.
        // With one line no head waits behind another line's batch.
        let mut memo = Behind::default();
        let behind: Vec<u64> = match state.order.several_lines() {
            false => Vec::new(),
            true => (state
                .order
                .heads_where(|held| held.segments || held.checked))
            .into_iter()
            .filter(|&sequence| is_behind(&state.order, &state.binary, &mut memo, sequence))
            .collect(),
        };
        let waiting = gate_heads(state, &behind, left, gated);
        let landing = check_heads(state, &behind, left, gated);
        let binary = &state.binary;
        let decodes = &state.decodes;
        let untracked_decodes = state.untracked_decodes;
        let behind = |order: &Order<Held>, sequence| is_behind(order, binary, &mut memo, sequence);
        let next = state.order.next(behind, |held| {
            if !binary.ready(&held.binary_needs) {
                return None;
            }
            // Its first submission waits while a rebind encodes encoders of it again.
            if held.index == 0 && held.buffer.reencode_pending() {
                return None;
            }
            // Held for a decoded twin's CPU decode it may read: any batch, while an untracked
            // texture's is left.
            if untracked_decodes > 0 || (!decodes.is_empty() && reads_decoding(held, decodes)) {
                return None;
            }
            // Held for its host step.
            if held.segments && held.buffer.needs_host_step(held.index) {
                return None;
            }
            // Held for its submission checks.
            if held.checked && held.buffer.needs_check(held.index) {
                return None;
            }
            event_waits(held)
        });
        let Some((queue, sequence, held, waits)) = next else {
            if !waiting && !landing {
                gated.store(false, Ordering::SeqCst);
            }
            return Ok(());
        };
        let submitted = held
            .buffer
            .submit_batch(held.index, held.last, &waits, state)
            .transpose();
        // A rebind marked it since it was found ready: back at its place, which it keeps, until
        // the rebind encoded its encoders again (`command_buffer/reencode.rs`), which wakes the
        // thread.
        let Some(submitted) = submitted else {
            drop(busy);
            state.order.put_back(queue, sequence, held);
            continue;
        };
        let Held {
            buffer,
            last,
            waits: events,
            binary_needs,
            binary_signals,
            ..
        } = held;
        drop(events);
        drop(busy);
        match submitted {
            Ok(submission) => {
                state.binary.submitted(&binary_needs, &binary_signals);
                if submission.runs_handlers() {
                    handlers.push(buffer.hold());
                }
                buffer.publish(submission);
                if last && buffer.settle_storage_forms() {
                    settled.push(buffer.hold());
                }
            }
            Err(error) => {
                // The rest of the command buffer never runs (dropped here, before it can
                // finish).
                state.binary.dropped(&binary_needs);
                let rest = state
                    .order
                    .drop_head(queue, |next| next.buffer.same(&buffer));
                for held in &rest {
                    state.binary.dropped(&held.binary_needs);
                }
                spent.extend(rest.into_iter().map(|held| held.buffer));
                buffer.failed(error, &state.values);
                if buffer.settle_storage_forms() {
                    settled.push(buffer.hold());
                }
            }
        }
        spent.push(buffer);
    }
}

/// The process is exiting ([`exit_guard`]): the thread submits no more.
struct Exiting;

/// Whether `held` may read a texture of `decodes`: its footprint names one, or it has none it can
/// know before its submission (sparse work, a segment whose uses its host step finds).
fn reads_decoding(held: &Held, decodes: &BTreeMap<ResourceKey, usize>) -> bool {
    held.buffer
        .footprint(held.index)
        .is_none_or(|footprint| footprint.names_any(|key| decodes.contains_key(key)))
}

/// End the switches of storage forms riding on the `settled` command buffers
/// (`texture/storage_form.rs`) on this thread, with the scheduler's lock free (the device's switch
/// lock is taken before it, never under it), before their transitions to `Scheduled` go on
/// ([`run_submission_handlers`]): the program sees each swap, and its announcement, by the time it
/// can see its command buffer scheduled. A thread that made a command buffer meanwhile may have
/// ended it first. Each end is a section of the exit guard (it makes Vulkan calls and announces):
/// once the process is exiting, the switch stays as it is.
fn settle(settled: &mut Settled) {
    for buffer in settled.iter() {
        let Some(busy) = exit_guard::enter() else {
            break;
        };
        buffer.device().finish_storage_forms();
        drop(busy);
    }
    settled.clear();
}

/// The command buffers whose storage forms the scheduler settled, to end without its lock.
type Settled = Vec<Hold<CommandBufferShared>>;

/// The command buffers whose last batch a pass submitted with submission handlers, in
/// submission order, for the submitting thread to run without the lock.
type Handlers = Vec<Hold<CommandBufferShared>>;

/// Run the submission handlers of `handlers`' command buffers on this thread, which holds none of
/// the runtime's locks, in submission order; each one's transition to `Scheduled` goes on after
/// its own (`CommandBufferShared::run_submission_handlers`), and after its switch of storage forms
/// ended ([`settle`], run first). Each hold goes after its handlers, outside every lock: it may be
/// its command buffer's last reference.
fn run_submission_handlers(handlers: &mut Handlers) {
    for buffer in handlers.drain(..) {
        buffer.run_submission_handlers();
    }
}

/// A batch for the thread to work on without its lock: its command buffer with the batch's index.
type Work = (Hold<CommandBufferShared>, usize);

/// Look at each line's head of a command buffer with a submission check that is otherwise
/// ready (its binary semaphores' signals submitted, its events' values reached, its host step
/// recorded, not `behind` an earlier commit) and waits for its checks: once every value it waits
/// for was reached on the device or by the host, not only by a submitted GPU signal (or the device
/// is lost), it goes to `left` as a check, unless there already. `gated` is set before the events
/// are read again, as [`Scheduler::completions`] says. Whether a head still waits for a GPU signal
/// to land.
fn check_heads(
    state: &mut State,
    behind: &[u64],
    left: &mut Left<Work>,
    gated: &AtomicBool,
) -> bool {
    let State { order, binary, .. } = state;
    let mut landing = false;
    for held in order.heads_mut(behind) {
        if !held.checked
            || !binary.ready(&held.binary_needs)
            || (held.segments && held.buffer.needs_host_step(held.index))
            || !held.buffer.needs_check(held.index)
            || left.holds(Kind::Check, |(buffer, index)| {
                buffer.same(&held.buffer) && *index == held.index
            })
        {
            continue;
        }
        match event_waits(held) {
            None => continue,
            Some(waits) if !waits.is_empty() && !held.buffer.device().is_lost() => {
                gated.store(true, Ordering::SeqCst);
                fence(Ordering::SeqCst);
                if event_waits(held).is_some_and(|waits| !waits.is_empty()) {
                    landing = true;
                    continue;
                }
            }
            Some(_) => {}
        }
        left.push(Kind::Check, (held.buffer.hold(), held.index));
    }
    landing
}

/// Look at each line's head that is otherwise ready (not `behind` an earlier commit either,
/// its gate then covers that commit's work) and needs a host step: its gate is taken (the queues'
/// submitted values, the first time), and once every `VkQueue`'s completed value reached it (or the
/// device is lost), its step goes to `left`, unless there already. `gated` is set before the
/// completed values are read, as [`Scheduler::completions`] says. Whether a head still waits at its
/// gate.
fn gate_heads(
    state: &mut State,
    behind: &[u64],
    left: &mut Left<Work>,
    gated: &AtomicBool,
) -> bool {
    let State {
        order,
        values,
        binary,
        ..
    } = state;
    let mut waiting = false;
    for held in order.heads_mut(behind) {
        if !held.segments
            || !binary.ready(&held.binary_needs)
            || event_waits(held).is_none()
            || !held.buffer.needs_host_step(held.index)
            || left.holds(Kind::Step, |(buffer, index)| {
                buffer.same(&held.buffer) && *index == held.index
            })
        {
            continue;
        }
        // A step waits for the pipelines its segments record, whose makings wake the thread as
        // they finish; one for them alone waits for no work on the GPU.
        let waits = held.buffer.step_waits(held.index);
        if waits.pipelines {
            continue;
        }
        if !waits.gpu {
            left.push(Kind::Step, (held.buffer.hold(), held.index));
            continue;
        }
        let gate = held.gate.get_or_insert_with(|| values.clone());
        gated.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        let device = held.buffer.device();
        let shared = device.shared();
        let reached = device.is_lost()
            || shared
                .queues
                .iter()
                .zip(gate.iter())
                .all(|(queue, &value)| queue.completed_value() >= value);
        if reached {
            left.push(Kind::Step, (held.buffer.hold(), held.index));
        } else {
            waiting = true;
        }
    }
    waiting
}

/// Whether the head of commit sequence number `sequence` waits behind a batch committed before it
/// that keeps its place ([`Order::is_behind`]): one whose binary semaphores' signals are submitted
/// and whose event values are reached, as is every batch before it on its line, and whose footprint
/// and the head's run in commit order ([`tracker::ordered`]). `memo` holds what this pass found.
fn is_behind(
    order: &Order<Held>,
    binary: &BinarySignals<vk::Semaphore>,
    memo: &mut Behind<Option<Footprint<ResourceKey>>>,
    sequence: u64,
) -> bool {
    order.is_behind(
        sequence,
        memo,
        (
            &mut |held: &Held| binary.ready(&held.binary_needs) && events_reachable(held),
            &mut |held: &Held| held.buffer.footprint(held.index),
            &|earlier: &Option<Footprint<ResourceKey>>, later: &Option<Footprint<ResourceKey>>| {
                tracker::ordered(earlier.as_ref(), later.as_ref())
            },
        ),
    )
}

/// Whether every event value `held` waits for is reached by a submitted signal or the host:
/// [`event_waits`] would answer, for a batch the order does not take yet.
fn events_reachable(held: &Held) -> bool {
    held.waits
        .iter()
        .all(|(event, value)| event.wait(*value).is_some())
}

/// The Vulkan waits of `held` for its events' signal indices, when every value it waits for is
/// reached by a submitted signal or the host; `None` while one is not.
fn event_waits(held: &Held) -> Option<Vec<(vk::Semaphore, u64)>> {
    let mut waits = Vec::new();
    for (event, value) in &held.waits {
        match event.wait(*value)? {
            WaitFor::Reached => {}
            WaitFor::Index(index) => waits.push((event.semaphore(), index)),
        }
    }
    Some(waits)
}
