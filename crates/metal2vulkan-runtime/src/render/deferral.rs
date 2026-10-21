//! Where a render encoder's calls are lowered (`DevicePolicy::render_lowering`): inside each
//! call (`RenderLoweringPolicy::AtCall`, the default), or after it returned
//! (`RenderLoweringPolicy::Worker`), where the program's encoder ([`RenderCommandEncoder`]) makes
//! each call's checks of its arguments and of the encoder's state (`render/call_checks.rs`), logs
//! it, and returns, and its lowering ([`RenderLowering`]) runs the logged calls later, in order,
//! as each would have run at its call.
//!
//! - **The log** is the one a rebind encodes an encoder again from (`render/replay.rs`): each
//!   call's bytes copied at the call and every object it names held by the log from the call, so
//!   neither a later write of the program's bytes nor a release moves what the call meant. A call
//!   its checks refuse returns the refusal and leaves nothing in the log.
//! - **Batches.** The calls are logged in batches of [`BATCH_CALLS`]: each full one is handed to
//!   the device's lowering thread (`render/lowering_worker.rs`), which lowers the batches it is
//!   handed in order while the program goes on. A program that outruns it waits, with
//!   [`HANDED_BATCHES`] batches of its encoder handed and not taken, until the thread takes one
//!   (`WorkCounts::render_log_waits`): an encoder's log never holds more than
//!   `(HANDED_BATCHES + 2) * BATCH_CALLS` calls not yet lowered, the batch the thread lowers and
//!   the one being filled counted.
//! - **Drains.** Where the program needs what the lowering leaves (the encoder's end, the device's
//!   work counts, a query of its pass: `tile_width`, `tile_height`) or makes a call the checks do
//!   not cover, its own thread lowers every call the lowering thread has not taken, after the
//!   batch that thread is lowering, if any (`WorkCounts::render_log_batches_by_program`, against
//!   `render_log_batches_by_thread`). So does a call past the cap where the device runs no
//!   lowering thread, or once the process exits.
//! - **At the call.** A call the checks do not cover is lowered inside the call, the calls logged
//!   before it first (`Checked::Now`), and from some calls on every call of the encoder is; so is
//!   every call of an encoder whose pass the device draws in color groups
//!   (`render/color_groups.rs`), whose checks follow the groups.
//! - **Detachment.** A logged call is lowered as one made while no buffer of the device was
//!   detached (`EncoderCore::judges_unattached`), as it was: none is logged while one is
//!   (`render/call_checks.rs`). A detach after the call refuses nothing of it, and its command
//!   buffer meets the detach as one encoded before it (`detach.rs`).
//! - **Refusals.** A call its lowering refuses fails its command buffer
//!   (`CommandBufferError::CallRefused`, naming the call): the encoder and every encoder after it
//!   run nothing ([`RenderLowering::refuse_call`]), and the encoder lowers none of its calls after
//!   that one. The command buffer keeps the call's error (`CommandBuffer::refusal`).
//! - **Notices.** A notice announced while a logged call is lowered names it
//!   (`DegradeNotice::call`, [`lowering::current`]), on whichever thread lowers it. A kind the
//!   device refuses while lowering (`DeviceOptions::lowering_refuses`) is not announced there: it
//!   refuses the call, as a refusal of its lowering does, the notice its error (`Error::Degraded`).
//! - **Emptied on the program's thread.** The lowering thread hands each batch back lowered, its
//!   calls still held; the program's thread empties it as it takes it to fill again, or as the
//!   encoder drains, ends or goes. By then what a lowered call named is held by the encoder's
//!   lowering or its command buffer (`ArgumentTable`'s retains).
//! - **Slots.** Each encoder's lowering, checks and batches lie in a slot the device keeps
//!   ([`DeferredEncoders`]), which the encoder's end or drop gives back for the device's next
//!   encoder, its batches' room kept.

use super::call_checks::{CallChecks, Checked};
use super::lowering_worker::LoweringWorker;
use super::replay::{render, DrawIndexed};
use super::{RenderCall, RenderCommandEncoder, RenderLowering};
use crate::device::{lock, Device};
use crate::error::{internal, EncoderCall, Error};
use crate::exit_guard;
use crate::replay::CallLog;
use metal2vulkan::device::RenderLoweringPolicy;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

/// The calls of a batch (the module docs): few enough that the lowering thread starts on a pass's
/// draws soon after the program makes them, many enough that the hand-off (a lock, and at most a
/// wake of the thread) is paid once for 32 calls. A draw's lowering took 1 to 2 µs on an M3 Max
/// (release, `host_ab many_draws`), so a batch is 30 to 70 µs of the thread's work.
pub(crate) const BATCH_CALLS: usize = 32;

/// The batches of one encoder handed to the lowering thread and not yet taken at which the
/// program's next full batch waits (the module docs): 512 calls, a few hundred µs of the thread's
/// work, enough to cover its wake and a pipeline making's wait for a draw. Each logged call holds
/// its own bytes (a `setBytes` call up to 4 KiB) and its objects, so an encoder holds at most about
/// 2 MiB of logged bytes beyond what its lowering holds, and runs at most 18 batches ahead of
/// what its command buffer has recorded.
pub(crate) const HANDED_BATCHES: usize = 16;

/// A run of logged calls, with the arguments of their indexed draws
/// (`RenderCall::DrawIndexedPrimitives`).
struct Batch {
    calls: CallLog<RenderCall>,
    indexed: Vec<DrawIndexed>,
    /// The index of its first call among its encoder's calls.
    first: u32,
}

impl Batch {
    /// An empty batch whose first call is call `first` of its encoder; it allocates nothing
    /// until its first call.
    fn new(first: u32) -> Batch {
        Batch {
            calls: CallLog::new(),
            indexed: Vec::new(),
            first,
        }
    }

    /// The index of the call after its last.
    fn next(&self) -> u32 {
        // An encoder's calls are counted in 32 bits (`EncoderCall::index`).
        self.first + self.calls.calls.len() as u32
    }

    /// Let go of its calls and what they hold, its room kept: its first call is call `first` of
    /// its encoder from now on.
    fn empty(&mut self, first: u32) {
        self.calls.clear();
        self.indexed.clear();
        self.first = first;
    }

    /// Lower its calls with `lowerer`, in order (the module docs): each as its identity, the first
    /// one refused failing the command buffer, and none after it lowered; counted as lowered on
    /// the lowering thread (`by_thread`) or the program's.
    fn lower(&self, lowerer: &mut Lowerer, by_thread: bool) {
        let Lowerer { lowering, refused } = lowerer;
        let Some(lowering) = lowering else {
            return;
        };
        if self.calls.calls.is_empty() || *refused {
            return;
        }
        let counts = &lowering.core.device().shared().counts;
        counts.render_log_batch_lowered(by_thread);
        lowering.core.judges_unattached(true);
        for (at, call) in (self.first..).zip(&self.calls.calls) {
            let identity = EncoderCall::new(call.selector(), at);
            let log = (&self.calls, &self.indexed[..]);
            let ran = lowering::during(identity, || render(lowering, log, call));
            let error = match (ran, lowering::take_refusal()) {
                (Err(error), _) => Some(error),
                (Ok(()), Some(notice)) => Some(Error::Degraded(Box::new(notice))),
                (Ok(()), None) => None,
            };
            if let Some(error) = error {
                // Metal's own checks of a call are its checks at the call's
                // (`render/call_checks.rs`): none is found as it is lowered.
                debug_assert!(
                    !matches!(error, Error::Invalid(_)),
                    "{identity:?}, logged, refused as it was lowered: {error}"
                );
                lowering.refuse_call(identity, error);
                *refused = true;
                break;
            }
        }
        lowering.core.judges_unattached(false);
    }
}

/// A render encoder's lowering, checks and batches (the module docs).
pub(crate) struct LoweringSlot {
    /// What the program's encoder works on, and the batches handed to the lowering thread.
    state: Mutex<SlotState>,
    /// Its lowering, which one thread at a time lowers a batch with: the lowering thread, or the
    /// program's at a drain, which holds `state` meanwhile, so the lowering thread starts no batch.
    lowerer: Mutex<Lowerer>,
    /// Signalled as the lowering thread takes a batch, hands one back, or leaves the slot.
    handed_back: Condvar,
}

/// A slot's lowering, and whether it refused a call.
struct Lowerer {
    /// Its encoder's lowering: `None` once the encoder ended, and in a slot not taken.
    lowering: Option<RenderLowering>,
    /// A call was refused as it was lowered: no later call is lowered.
    refused: bool,
}

struct SlotState {
    /// The program's encoder holds the slot.
    taken: bool,
    /// The checks at the call of the encoders that logged their calls: its encoder's while it
    /// logs them (`checking`), their lists' room kept for the next.
    checks: Option<CallChecks>,
    checking: bool,
    /// The batch the program logs into.
    filling: Batch,
    /// The batches handed to the lowering thread and not taken by it, in order, before `filling`.
    handed: VecDeque<Batch>,
    /// The lowering thread lowers a batch of the slot now.
    busy: bool,
    /// The program's thread drains the slot: the lowering thread takes no batch of it.
    draining: bool,
    /// The slot is in the lowering thread's queue.
    queued: bool,
    /// The lowering thread panicked lowering a batch of the slot (`render/lowering_worker.rs`).
    panicked: bool,
    /// Batches lowered, their calls still held until the program's thread empties them as it
    /// takes them to fill (the module docs).
    lowered: Vec<Batch>,
}

impl SlotState {
    /// Its encoder lowers its calls inside them from now on: its checks let go of what they hold.
    fn stop_checking(&mut self) {
        self.checking = false;
        if let Some(checks) = &mut self.checks {
            checks.release();
        }
    }

    /// Hand the batch being filled to the lowering thread, and fill a new one after it: a lowered
    /// one emptied, else a new one. Whether the slot is to be handed to the thread: it is in no
    /// queue.
    fn hand_off(&mut self) -> bool {
        let next = self.filling.next();
        let fresh = match self.lowered.pop() {
            Some(mut lowered) => {
                lowered.empty(next);
                lowered
            }
            None => Batch::new(next),
        };
        let full = std::mem::replace(&mut self.filling, fresh);
        self.handed.push_back(full);
        let unqueued = !self.queued;
        self.queued = true;
        unqueued
    }
}

impl LoweringSlot {
    /// The lowering thread's turn at the slot: each batch handed lowered, in order, inside a
    /// section of the exit guard, as the program's calls (`render/lowering_worker.rs`), and handed
    /// back, until none is handed or the program's thread drains the slot.
    pub(super) fn lower_handed(&self) {
        loop {
            let batch = {
                let Some(_section) = exit_guard::enter() else {
                    // The process exits: the program's thread lowers what is handed.
                    self.left_by_thread();
                    return;
                };
                let batch = {
                    let mut state = lock(&self.state);
                    let batch = match state.draining {
                        true => None,
                        false => state.handed.pop_front(),
                    };
                    let Some(batch) = batch else {
                        state.queued = false;
                        return;
                    };
                    state.busy = true;
                    batch
                };
                // A program that waits for room has it (`wait_for_room`).
                self.handed_back.notify_all();
                exit_guard::lowering_program_calls(|| batch.lower(&mut lock(&self.lowerer), true));
                batch
            };
            // Handed back once its section ended: an exit that follows finds the hook registered
            // after the section's first calls (`exit_guard`).
            let mut state = lock(&self.state);
            state.busy = false;
            state.lowered.push(batch);
            drop(state);
            self.handed_back.notify_all();
        }
    }

    /// The lowering thread leaves the slot with what is handed, unlowered: the process exits, or
    /// the thread ended.
    pub(super) fn left_by_thread(&self) {
        lock(&self.state).queued = false;
        self.handed_back.notify_all();
    }

    /// The lowering thread panicked lowering a batch of the slot, which is lost: every later
    /// drain panics on the program's thread.
    pub(super) fn lost_to_panic(&self) {
        let mut state = lock(&self.state);
        state.busy = false;
        state.queued = false;
        state.panicked = true;
        drop(state);
        self.handed_back.notify_all();
    }

    /// [`drain_locked`](Self::drain_locked) the slot.
    fn drain<T>(&self, past: u32, then: impl FnOnce(&mut Lowerer, &Batch) -> T) -> T {
        self.drain_locked(lock(&self.state), past, then)
    }

    /// Lower, on this thread, every call logged that the lowering thread has not taken, after the
    /// batch it lowers now, if any (the module docs), then run `then` of the lowering and the
    /// batch being filled, whose arenas hold `past` calls made of them and not logged (a call
    /// lowered at its call); every batch emptied then: its result.
    fn drain_locked<T>(
        &self,
        mut guard: MutexGuard<'_, SlotState>,
        past: u32,
        then: impl FnOnce(&mut Lowerer, &Batch) -> T,
    ) -> T {
        guard.draining = true;
        while guard.busy {
            guard =
                (self.handed_back.wait(guard)).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        assert!(
            !guard.panicked,
            "the device's lowering thread panicked lowering this render encoder's calls"
        );
        let state = &mut *guard;
        let mut lowerer = lock(&self.lowerer);
        for batch in state.handed.iter().chain([&state.filling]) {
            batch.lower(&mut lowerer, false);
        }
        let then = then(&mut lowerer, &state.filling);
        drop(lowerer);
        let next = state.filling.next() + past;
        for mut batch in state.handed.drain(..) {
            batch.empty(0);
            state.lowered.push(batch);
        }
        for batch in &mut state.lowered {
            batch.empty(0);
        }
        state.filling.empty(next);
        state.draining = false;
        then
    }

    /// The slot given back by its encoder, which ended or goes (the module docs): its lowering let
    /// go of (abandoning an encoder dropped unended, its calls not lowered), its checks and
    /// batches emptied, their room kept.
    fn give_back(&self) {
        let mut guard = lock(&self.state);
        guard.draining = true;
        while guard.busy {
            guard =
                (self.handed_back.wait(guard)).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        // Before the slot is free, so no later encoder's lowering goes here.
        let gone = {
            let mut lowerer = lock(&self.lowerer);
            lowerer.refused = false;
            lowerer.lowering.take()
        };
        let state = &mut *guard;
        state.stop_checking();
        for mut batch in state.handed.drain(..) {
            batch.empty(0);
            state.lowered.push(batch);
        }
        for batch in &mut state.lowered {
            batch.empty(0);
        }
        state.filling.empty(0);
        state.draining = false;
        state.panicked = false;
        state.taken = false;
        drop(guard);
        // Its drop abandons the command buffer of an encoder dropped unended, the slot's locks
        // released.
        drop(gone);
    }
}

/// A device's slots for its render encoders' lowerings (the module docs): every slot it made,
/// each taken by an open encoder or kept for the next; and its lowering thread, started with the
/// first batch an encoder hands off.
#[derive(Default)]
pub(crate) struct DeferredEncoders {
    slots: Mutex<Vec<Arc<LoweringSlot>>>,
    worker: OnceLock<Option<LoweringWorker>>,
}

impl DeferredEncoders {
    /// A slot for `lowering`'s encoder, which logs its calls, with its checks at the call: one
    /// kept, its checks' room reused, else a new one.
    fn take(&self, lowering: RenderLowering) -> Arc<LoweringSlot> {
        let mut slots = lock(&self.slots);
        let kept = slots.iter().find(|slot| !lock(&slot.state).taken).cloned();
        let slot = match kept {
            Some(slot) => slot,
            None => {
                let slot = Arc::new(LoweringSlot {
                    state: Mutex::new(SlotState {
                        taken: false,
                        checks: None,
                        checking: false,
                        filling: Batch::new(0),
                        handed: VecDeque::new(),
                        busy: false,
                        draining: false,
                        queued: false,
                        panicked: false,
                        lowered: Vec::new(),
                    }),
                    lowerer: Mutex::new(Lowerer {
                        lowering: None,
                        refused: false,
                    }),
                    handed_back: Condvar::new(),
                });
                slots.push(Arc::clone(&slot));
                slot
            }
        };
        let mut state = lock(&slot.state);
        state.taken = true;
        drop(slots);
        state.checking = true;
        match &mut state.checks {
            Some(checks) => checks.reset(&lowering),
            None => state.checks = Some(CallChecks::new(&lowering)),
        }
        lock(&slot.lowerer).lowering = Some(lowering);
        drop(state);
        slot
    }

    /// The device's lowering thread, started at the first call; `None` where the host starts no
    /// thread.
    fn worker(&self) -> Option<&LoweringWorker> {
        self.worker.get_or_init(LoweringWorker::start).as_ref()
    }

    /// Lower the calls every open encoder of the device logged, encoder by encoder: before a read
    /// of what their lowering leaves (`Device::work_counts`). Nothing where this thread lowers a
    /// logged call now (a message sink that reads the counts), which would wait for itself.
    pub(crate) fn lower_all(&self) {
        if lowering::current().is_some() {
            return;
        }
        let slots: Vec<Arc<LoweringSlot>> = lock(&self.slots).iter().map(Arc::clone).collect();
        for slot in slots {
            slot.drain(0, |_, _| ());
        }
    }

    /// Stop the lowering thread, as the device is torn down: every open encoder holds the device,
    /// so none is left to lower.
    pub(crate) fn stop(&self) {
        if let Some(Some(worker)) = self.worker.get() {
            worker.stop();
        }
    }
}

/// A device test's view of the device's slots (`tests::deferred_calls`, compiled under the
/// validation crate's GPU safety cfgs alone).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl DeferredEncoders {
    /// The slots open encoders hold now.
    pub(crate) fn taken(&self) -> Vec<Arc<LoweringSlot>> {
        let slots = lock(&self.slots);
        let taken = slots.iter().filter(|slot| lock(&slot.state).taken);
        taken.map(Arc::clone).collect()
    }
}

/// A device test's hold of a slot (`tests::deferred_calls`).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl LoweringSlot {
    /// Run `f` while holding the slot's lowering: the lowering thread lowers none of its batches
    /// until `f` returns.
    pub(crate) fn while_held<T>(&self, f: impl FnOnce() -> T) -> T {
        let _held = lock(&self.lowerer);
        f()
    }

    /// The batches of the slot handed to the lowering thread and not taken.
    pub(crate) fn handed(&self) -> usize {
        lock(&self.state).handed.len()
    }
}

/// A program's encoder's slot, with the calls it logged and their checks (the module docs),
/// given back to the device as the encoder goes.
pub(crate) struct Logged {
    slot: Arc<LoweringSlot>,
    /// The encoder's device, which its checks at the call work on.
    device: Device,
    /// It lowers every call inside it from now on (`Checked::Now { sticky: true }`).
    at_call: bool,
}

impl Drop for Logged {
    /// Give the slot back to the device: a lowering still there (an encoder dropped unended)
    /// goes, abandoning its command buffer, its calls not lowered.
    fn drop(&mut self) {
        self.slot.give_back();
    }
}

impl RenderCommandEncoder {
    /// The program's encoder over `lowering`, which lowers its calls as the device's policy says
    /// (the module docs).
    pub(super) fn new(lowering: RenderLowering) -> RenderCommandEncoder {
        let device = lowering.core.device();
        let logs = device.compile_device().policy().render_lowering == RenderLoweringPolicy::Worker
            && lowering.grouped.is_none();
        let (own, logged) = match logs {
            false => (Some(lowering), None),
            true => {
                let device = device.clone();
                let slot = device.shared().deferred.take(lowering);
                let at_call = false;
                let logged = Logged {
                    slot,
                    device,
                    at_call,
                };
                (None, Some(logged))
            }
        };
        RenderCommandEncoder {
            own,
            logged,
            next: 0,
            unshared: PhantomData,
        }
    }

    /// Whether the encoder lowers each call inside it (the module docs).
    pub(crate) fn lowers_at_call(&self) -> bool {
        self.logged.as_ref().is_none_or(|logged| logged.at_call)
    }

    /// Run `f`, a call, on the lowering now: an encoder that logged calls before it lowered them
    /// as it began lowering its calls at the call.
    pub(crate) fn at_call<T>(
        &mut self,
        f: impl FnOnce(&mut RenderLowering) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.next += 1;
        match (&mut self.own, &self.logged) {
            (Some(lowering), _) => f(lowering),
            (None, Some(logged)) => {
                let mut lowerer = lock(&logged.slot.lowerer);
                let lowering = (lowerer.lowering.as_mut())
                    .ok_or_else(|| internal("an ended encoder's call"))?;
                f(lowering)
            }
            (None, None) => Err(internal("an encoder without its lowering")),
        }
    }

    /// Run `f`, a call that returns nothing, on the lowering now, as [`at_call`](Self::at_call)
    /// does.
    pub(crate) fn at_call_unit(&mut self, f: impl FnOnce(&mut RenderLowering)) {
        self.next += 1;
        match (&mut self.own, &self.logged) {
            (Some(lowering), _) => f(lowering),
            (None, Some(logged)) => {
                if let Some(lowering) = lock(&logged.slot.lowerer).lowering.as_mut() {
                    f(lowering);
                }
            }
            (None, None) => {}
        }
    }

    /// Check and log the call `call` makes of the log (the module docs): a call its checks refuse
    /// returns the refusal and leaves the log as it was; one they do not cover is lowered now.
    pub(crate) fn defer(
        &mut self,
        call: impl FnOnce(&mut CallLog<RenderCall>) -> RenderCall,
    ) -> Result<(), Error> {
        self.defer_with(|calls, _| call(calls))
    }

    /// [`defer`](Self::defer) a call that returns nothing, whose checks refuse nothing.
    pub(crate) fn defer_unit(&mut self, call: impl FnOnce(&mut CallLog<RenderCall>) -> RenderCall) {
        let deferred = self.defer(call);
        debug_assert!(deferred.is_ok(), "a call that returns nothing refused");
    }

    /// [`defer`](Self::defer) a `draw_indexed_primitives` call, whose arguments `draw` makes of
    /// the log: they go into the list of them.
    pub(crate) fn defer_indexed(
        &mut self,
        draw: impl FnOnce(&mut CallLog<RenderCall>) -> DrawIndexed,
    ) -> Result<(), Error> {
        self.defer_with(|calls, indexed| {
            // Fewer than the log's calls, each of 24 bytes or more.
            let at = indexed.len() as u32;
            indexed.push(draw(calls));
            RenderCall::DrawIndexedPrimitives(at)
        })
    }

    /// [`defer`](Self::defer) the call `call` makes of the batch being filled and its list of
    /// indexed draws' arguments: a refused call leaves both as they were.
    fn defer_with(
        &mut self,
        call: impl FnOnce(&mut CallLog<RenderCall>, &mut Vec<DrawIndexed>) -> RenderCall,
    ) -> Result<(), Error> {
        let Some(logged) = &mut self.logged else {
            return Err(internal(
                "a call logged by an encoder that lowers it at the call",
            ));
        };
        let slot = &logged.slot;
        let mut state = lock(&slot.state);
        let SlotState {
            filling,
            checks,
            checking,
            ..
        } = &mut *state;
        let checks = match (*checking, checks) {
            (true, Some(checks)) => checks,
            _ => return Err(internal("a logged call unchecked")),
        };
        let Batch { calls, indexed, .. } = filling;
        let (mark, draws) = (calls.mark(), indexed.len());
        let call = call(calls, indexed);
        let checked = match checks.check(&logged.device, &call, (calls, indexed)) {
            Ok(checked) => checked,
            Err(error) => {
                calls.truncate(mark);
                indexed.truncate(draws);
                return Err(error);
            }
        };
        self.next += 1;
        match checked {
            Checked::Logged => {
                let extra = call.extra();
                calls.push(call, extra);
                if calls.calls.len() < BATCH_CALLS {
                    return Ok(());
                }
                let unqueued = state.hand_off();
                let full = state.handed.len() >= HANDED_BATCHES;
                drop(state);
                let worker = logged.device.shared().deferred.worker();
                if unqueued && !worker.is_some_and(|worker| worker.hand(slot)) {
                    lock(&slot.state).queued = false;
                }
                if full {
                    wait_for_room(slot, &logged.device, worker);
                }
                Ok(())
            }
            Checked::Now { sticky } => {
                if sticky {
                    logged.at_call = true;
                    state.stop_checking();
                }
                // The call is made of the batch being filled, after the calls logged in it.
                slot.drain_locked(state, 1, |lowerer, filling| match lowerer {
                    Lowerer { refused: true, .. } => Ok(()),
                    Lowerer {
                        lowering: Some(lowering),
                        ..
                    } => render(lowering, (&filling.calls, &filling.indexed), &call),
                    Lowerer { lowering: None, .. } => Err(internal("an ended encoder's call")),
                })
            }
        }
    }

    /// `f` of the lowering, the calls logged lowered first: an answer only the lowering knows.
    pub(super) fn lowered<T>(
        &self,
        f: impl FnOnce(&RenderLowering) -> Result<T, Error>,
    ) -> Result<T, Error> {
        match (&self.own, &self.logged) {
            (Some(lowering), _) => f(lowering),
            (None, Some(logged)) => logged.slot.drain(0, |lowerer, _| {
                let lowering = (lowerer.lowering.as_ref())
                    .ok_or_else(|| internal("an ended encoder's query"))?;
                f(lowering)
            }),
            (None, None) => Err(internal("an encoder without its lowering")),
        }
    }

    /// End the encoder (`end_encoding`): its calls logged lowered, then its lowering ended, inside
    /// the call, which returns a refusal of the end itself. Notices of the end of an encoder that
    /// logged its calls name it as `endEncoding`, after the encoder's calls.
    pub(super) fn end(self) -> Result<(), Error> {
        let logged = match (self.own, self.logged) {
            (Some(lowering), _) => return lowering.end_encoding(),
            (None, Some(logged)) => logged,
            (None, None) => return Err(internal("an encoder without its lowering")),
        };
        let slot = &logged.slot;
        let lowering = slot.drain(0, |lowerer, _| lowerer.lowering.take());
        lock(&slot.state).stop_checking();
        let lowering = lowering.ok_or_else(|| internal("an encoder ended twice"))?;
        match logged.at_call {
            true => lowering.end_encoding(),
            false => lowering::during(EncoderCall::new("endEncoding", self.next), || {
                lowering.end_encoding()
            }),
        }
    }
}

/// Wait, after a hand-off that left [`HANDED_BATCHES`] batches of `slot`'s encoder of `device`
/// handed and not taken, until the lowering thread takes one (the module docs); where `worker`,
/// the device's lowering thread, is none, lowers no more or left the slot, or the process exits,
/// lower them on this thread instead.
fn wait_for_room(slot: &LoweringSlot, device: &Device, worker: Option<&LoweringWorker>) {
    let mut state = lock(&slot.state);
    device.shared().counts.render_log_waited();
    loop {
        if state.handed.len() < HANDED_BATCHES {
            return;
        }
        let lowers = worker.is_some_and(LoweringWorker::runs) && state.queued;
        if !lowers || exit_guard::closed() || state.panicked {
            break;
        }
        state = (slot.handed_back.wait(state)).unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    slot.drain_locked(state, 0, |_, _| ());
}

/// The call being lowered on this thread, which the notices announced while it runs name, and a
/// notice of a kind the device refuses while lowering, which refuses it (the module docs).
pub(crate) mod lowering {
    use crate::degradation::DegradeNotice;
    use crate::error::EncoderCall;
    use std::cell::{Cell, RefCell};

    thread_local! {
        static LOWERING: Cell<Option<EncoderCall>> = const { Cell::new(None) };
        static REFUSED: RefCell<Option<DegradeNotice>> = const { RefCell::new(None) };
    }

    /// Run `f` as the lowering of `call`.
    pub(crate) fn during<T>(call: EncoderCall, f: impl FnOnce() -> T) -> T {
        let outer = LOWERING.with(|lowering| lowering.replace(Some(call)));
        let ran = f();
        LOWERING.with(|lowering| lowering.set(outer));
        ran
    }

    /// The call being lowered on this thread, if one is.
    pub(crate) fn current() -> Option<EncoderCall> {
        LOWERING.with(Cell::get)
    }

    /// Refuse the call being lowered for `notice`, a kind the device refuses while lowering: the
    /// first such notice of the call is kept.
    pub(crate) fn refuse(notice: DegradeNotice) {
        REFUSED.with(|refused| {
            refused.borrow_mut().get_or_insert(notice);
        });
    }

    /// The notice that refused the call being lowered, if one did, let go of here.
    pub(crate) fn take_refusal() -> Option<DegradeNotice> {
        REFUSED.with(|refused| refused.borrow_mut().take())
    }
}
