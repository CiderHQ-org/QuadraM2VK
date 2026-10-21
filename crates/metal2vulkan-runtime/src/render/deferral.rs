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
//! - **At the call.** A call the checks do not cover is lowered inside the call, the calls logged
//!   before it first (`Checked::Now`), and from some calls on every call of the encoder is; so is
//!   every call of an encoder whose pass the device draws in color groups
//!   (`render/color_groups.rs`), whose checks follow the groups.
//! - **When.** The calls logged are lowered as the encoder ends, and before anything reads what
//!   their lowering leaves: the device's work counts (`Device::work_counts`) and a query of the
//!   encoder's pass (`tile_width`, `tile_height`).
//! - **Detachment.** A logged call is lowered as one made while no buffer of the device was
//!   detached (`EncoderCore::judges_unattached`), as it was: none is logged while one is
//!   (`render/call_checks.rs`). A detach after the call refuses nothing of it, and its command
//!   buffer meets the detach as one encoded before it (`detach.rs`).
//! - **Refusals.** A call its lowering refuses fails its command buffer
//!   (`CommandBufferError::CallRefused`, naming the call): the encoder and every encoder after it
//!   run nothing ([`RenderLowering::refuse_call`]), and the encoder lowers none of its calls after
//!   that one. The command buffer keeps the call's error (`CommandBuffer::refusal`).
//! - **Notices.** A notice announced while a logged call is lowered names it
//!   (`DegradeNotice::call`, [`lowering::current`]). A kind the device refuses while lowering
//!   (`DeviceOptions::lowering_refuses`) is not announced there: it refuses the call, as a refusal
//!   of its lowering does, the notice its error (`Error::Degraded`).
//! - **Slots.** Each encoder's lowering, checks and calls not yet lowered lie in a slot the device
//!   keeps ([`DeferredEncoders`]), which a read lowers and the encoder's end or drop gives back
//!   for the device's next encoder, its log's room kept.

use super::call_checks::{CallChecks, Checked};
use super::replay::{render, DrawIndexed};
use super::{RenderCall, RenderCommandEncoder, RenderLowering};
use crate::device::lock;
use crate::error::{internal, EncoderCall, Error};
use crate::replay::CallLog;
use metal2vulkan::device::RenderLoweringPolicy;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// A render encoder's lowering, checks and calls not yet lowered (the module docs).
pub(crate) struct LoweringSlot(Mutex<SlotState>);

struct SlotState {
    /// The program's encoder holds the slot.
    taken: bool,
    /// Its encoder's lowering: `None` once the encoder ended, and in a slot not taken.
    lowering: Option<RenderLowering>,
    /// The checks at the call of the encoders that logged their calls: its encoder's while it
    /// logs them (`checking`), their lists' room kept for the next.
    checks: Option<CallChecks>,
    checking: bool,
    /// The calls logged since the last lowering, with the arguments of their indexed draws
    /// (`RenderCall::DrawIndexedPrimitives`).
    calls: CallLog<RenderCall>,
    indexed: Vec<DrawIndexed>,
    /// The index of the first of them.
    first: u32,
    /// A call was refused as it was lowered: no later call is lowered.
    refused: bool,
}

impl SlotState {
    /// Lower the calls logged, in order (the module docs): each as its identity, the first one
    /// refused failing the command buffer, and none after it lowered. Their log is kept.
    fn lower_logged(&mut self) {
        let SlotState {
            lowering,
            calls,
            indexed,
            first,
            refused,
            ..
        } = self;
        let Some(lowering) = lowering else {
            return;
        };
        if calls.calls.is_empty() || *refused {
            return;
        }
        lowering.core.judges_unattached(true);
        for (at, call) in (*first..).zip(&calls.calls) {
            let identity = EncoderCall::new(call.selector(), at);
            let ran = lowering::during(identity, || render(lowering, (calls, indexed), call));
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

    /// Let go of the log's calls, `called` more calls made: the next call logged is the one after
    /// them.
    fn clear(&mut self, called: u32) {
        self.first += self.calls.calls.len() as u32 + called;
        self.calls.clear();
        self.indexed.clear();
    }

    /// Lower the calls logged, then let go of them.
    fn lower(&mut self) {
        self.lower_logged();
        self.clear(0);
    }

    /// Lower the calls logged, then `call` inside its call, made of the log's arenas but not
    /// logged: its result. A call after one refused as it was lowered runs nothing.
    fn lower_with(&mut self, call: &RenderCall) -> Result<(), Error> {
        self.lower_logged();
        let ran = match (&mut self.lowering, self.refused) {
            (Some(lowering), false) => render(lowering, (&self.calls, &self.indexed), call),
            (Some(_), true) => Ok(()),
            (None, _) => Err(internal("an ended encoder's call")),
        };
        self.clear(1);
        ran
    }

    /// Its encoder lowers its calls inside them from now on: its checks let go of what they hold.
    fn stop_checking(&mut self) {
        self.checking = false;
        if let Some(checks) = &mut self.checks {
            checks.release();
        }
    }

    /// The slot given back: its lowering let go of (abandoning an encoder dropped unended), its
    /// checks and calls let go of, their room kept.
    fn give_back(&mut self) {
        self.lowering = None;
        self.stop_checking();
        self.calls.clear();
        self.indexed.clear();
        self.first = 0;
        self.refused = false;
        self.taken = false;
    }
}

/// A device's slots for its render encoders' lowerings (the module docs): every slot it made,
/// each taken by an open encoder or kept for the next.
#[derive(Default)]
pub(crate) struct DeferredEncoders(Mutex<Vec<Arc<LoweringSlot>>>);

impl DeferredEncoders {
    /// A slot for `lowering`'s encoder, which logs its calls, with its checks at the call: one
    /// kept, its checks' room reused, else a new one.
    fn take(&self, lowering: RenderLowering) -> Arc<LoweringSlot> {
        let checking = true;
        let mut slots = lock(&self.0);
        for slot in slots.iter() {
            let mut state = lock(&slot.0);
            if !state.taken {
                state.taken = true;
                state.checking = checking;
                if checking {
                    match &mut state.checks {
                        Some(checks) => checks.reset(&lowering),
                        None => state.checks = Some(CallChecks::new(&lowering)),
                    }
                }
                state.lowering = Some(lowering);
                drop(state);
                return Arc::clone(slot);
            }
        }
        let slot = Arc::new(LoweringSlot(Mutex::new(SlotState {
            taken: true,
            checks: checking.then(|| CallChecks::new(&lowering)),
            checking,
            lowering: Some(lowering),
            calls: CallLog::new(),
            indexed: Vec::new(),
            first: 0,
            refused: false,
        })));
        slots.push(Arc::clone(&slot));
        slot
    }

    /// Lower the calls every open encoder of the device logged, encoder by encoder: before a read
    /// of what their lowering leaves (`Device::work_counts`).
    pub(crate) fn lower_all(&self) {
        let slots: Vec<Arc<LoweringSlot>> = lock(&self.0).iter().map(Arc::clone).collect();
        for slot in slots {
            lock(&slot.0).lower();
        }
    }
}

/// A program's encoder's slot, with the calls it logged and their checks (the module docs),
/// given back to the device as the encoder goes.
pub(crate) struct Logged {
    slot: Arc<LoweringSlot>,
    /// It lowers every call inside it from now on (`Checked::Now { sticky: true }`).
    at_call: bool,
}

impl Drop for Logged {
    /// Give the slot back to the device: a lowering still there (an encoder dropped unended)
    /// goes, abandoning its command buffer, its calls not lowered.
    fn drop(&mut self) {
        lock(&self.slot.0).give_back();
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
                (None, Some(Logged { slot, at_call }))
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

    /// Run `f`, a call, on the lowering now.
    pub(crate) fn at_call<T>(
        &mut self,
        f: impl FnOnce(&mut RenderLowering) -> Result<T, Error>,
    ) -> Result<T, Error> {
        self.next += 1;
        match (&mut self.own, &self.logged) {
            (Some(lowering), _) => f(lowering),
            (None, Some(logged)) => {
                let mut state = lock(&logged.slot.0);
                let lowering =
                    (state.lowering.as_mut()).ok_or_else(|| internal("an ended encoder's call"))?;
                f(lowering)
            }
            (None, None) => Err(internal("an encoder without its lowering")),
        }
    }

    /// Run `f`, a call that returns nothing, on the lowering now.
    pub(crate) fn at_call_unit(&mut self, f: impl FnOnce(&mut RenderLowering)) {
        self.next += 1;
        match (&mut self.own, &self.logged) {
            (Some(lowering), _) => f(lowering),
            (None, Some(logged)) => {
                if let Some(lowering) = lock(&logged.slot.0).lowering.as_mut() {
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

    /// [`defer`](Self::defer) the call `call` makes of the log and its list of indexed draws'
    /// arguments: a refused call leaves both as they were.
    fn defer_with(
        &mut self,
        call: impl FnOnce(&mut CallLog<RenderCall>, &mut Vec<DrawIndexed>) -> RenderCall,
    ) -> Result<(), Error> {
        let Some(logged) = &mut self.logged else {
            return Err(internal(
                "a call logged by an encoder that lowers it at the call",
            ));
        };
        let mut state = lock(&logged.slot.0);
        let SlotState {
            lowering,
            calls,
            indexed,
            checks,
            checking,
            ..
        } = &mut *state;
        let (checks, lowering) = match (*checking, checks, lowering) {
            (true, Some(checks), Some(lowering)) => (checks, lowering),
            _ => return Err(internal("a logged call unchecked")),
        };
        let (mark, draws) = (calls.mark(), indexed.len());
        let call = call(calls, indexed);
        let checked = match checks.check(lowering.core.device(), &call, (calls, indexed)) {
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
                Ok(())
            }
            Checked::Now { sticky } => {
                if sticky {
                    logged.at_call = true;
                    state.stop_checking();
                }
                state.lower_with(&call)
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
            (None, Some(logged)) => {
                let mut state = lock(&logged.slot.0);
                state.lower();
                let lowering = (state.lowering.as_ref())
                    .ok_or_else(|| internal("an ended encoder's query"))?;
                f(lowering)
            }
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
        let lowering = {
            let mut state = lock(&logged.slot.0);
            state.lower();
            state.stop_checking();
            state.lowering.take()
        };
        let lowering = lowering.ok_or_else(|| internal("an encoder ended twice"))?;
        match logged.at_call {
            true => lowering.end_encoding(),
            false => lowering::during(EncoderCall::new("endEncoding", self.next), || {
                lowering.end_encoding()
            }),
        }
    }
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
