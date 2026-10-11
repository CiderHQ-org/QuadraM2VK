//! Reads of an argument the pipeline's function constants removed.
//!
//! Specialization makes each observed read of a texture argument the constants removed (a read,
//! sample, gather, query, or atomic), or each sample through a removed sampler argument,
//! [`FunctionConstant::AbsentRead`] (`air::canon::interface`). Metal gives such a read no value:
//! its compiler dropped a store of one, made a fragment color computed from one all-zero, and,
//! beside a removed texture table, dropped a kernel's whole body, so no one translation is Metal's.
//! Where the read runs, the function sets `StatusBits::ABSENT_ARGUMENT_READ` in the command
//! buffer's `ExecutionStatus` word (a relaxed device-scope `OR`, as the resident-set miss counter
//! writes it) and takes zero for the read's result; a consumer that reads the bit reports the
//! command buffer's execution as failed, so that zero reaches no result reported as good. The write
//! is on the read's path only: a run that never reaches the read is Metal's exactly, and a module
//! without one is unchanged.
//!
//! A load or atomic of threadgroup memory the constants removed is the same read
//! ([`FunctionConstant::AbsentMemory`]), and so is a texture operation on a removed texture or
//! sampler that lowering reaches through memory, an aggregate, a table, or a merge
//! (`texture::AbsentArgument`), which shares [`require`] and [`mark`].
//!
//! The stage states [`RuntimeFact::AbsentArgumentRead`] and, for the buffer's address,
//! [`RuntimeFact::ExecutionStatus`] with [`PushField::ExecutionStatusAddress`]. The write goes
//! through that device address, so the device must address buffers (`bufferDeviceAddress`,
//! `shaderInt64`), and a vertex or fragment function needs its stage's stores and atomics; a
//! device without them refuses the read as `FunctionConstantArgument`, naming what it needs.

use crate::air::intrinsics::{AirIntrinsic, FunctionConstant};
use crate::air::ir::{Callee, Inst, InstId, InstKind, Type, Value, ValueId};
use crate::bind::status::{StatusBits, StatusWord};
use crate::device::profile::BufferAccess;
use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::k32;
use crate::memory::roots::k64;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::PushField;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::DemoteTy;
use crate::spirv::ir::{AtomicOp, BinOp, Constant, Scope, Semantics, Type as SType};

/// An absent read ([`FunctionConstant::AbsentRead`]): the status write, then zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbsentRead {
    /// The lowered representation of the read's result; `None` for a read of no result.
    pub result: Option<DemoteTy>,
}

impl AbsentRead {
    /// The census name.
    pub fn name(&self) -> &'static str {
        "access.absent_read"
    }

    /// An upper estimate of the SPIR-V words one read emits: the push field's load, the word's
    /// address and pointer, the bit, and the atomic `OR`; the zero is a constant.
    pub fn words(&self) -> u32 {
        32
    }
}

/// Select the read of `inst` when it is a [`FunctionConstant::AbsentRead`] or
/// [`FunctionConstant::AbsentMemory`] call, `None` otherwise: the stage writes the
/// `ExecutionStatus` buffer through its device address.
pub fn select(cx: &mut SelectCx<'_>, at: InstId, inst: &Inst) -> Result<Option<AbsentRead>, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Ok(None);
    };
    let Callee::Air {
        op:
            AirIntrinsic::FunctionConstant(
                FunctionConstant::AbsentRead(_) | FunctionConstant::AbsentMemory(_),
            ),
        ..
    } = call.callee
    else {
        return Ok(None);
    };
    require(cx)?;
    let result = match cx.entry.module.types().get(inst.ty) {
        Some(Type::Void) => None,
        _ => Some(cx.lowered_repr(Value::Inst(at), inst.ty)?),
    };
    Ok(Some(AbsentRead { result }))
}

/// What an absent read needs, stated for the stage: the `ExecutionStatus` buffer's device
/// address, and the stage's stores and atomics; a device without them refuses the read as
/// `FunctionConstantArgument`, naming what it needs. The texture and sampler families' arms on
/// the absent handle need the same.
pub fn require(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    match status_short(cx) {
        Some(StatusShort::Addresses) => {
            return Err(Unsupported::needs_any_of(
                Feature::FunctionConstantArgument,
                vec![cx.profile.addressed_needs()],
            )
            .into());
        }
        Some(StatusShort::Stores(f)) => {
            return Err(Unsupported::needs(
                Feature::FunctionConstantArgument,
                Capability::Feature(f),
            )
            .into());
        }
        None => state_status(cx),
    }
    cx.facts
        .insert(RuntimeFact::AbsentArgumentRead { stage: cx.stage });
    Ok(())
}

/// What the stage lacks to write the command buffer's `ExecutionStatus` words: the buffer's
/// device address, or its stage's stores and atomics. Each refusal names its own feature at its
/// site ([`require`], `sync::barrier::require_seq_cst`), so the refusal surface reads each one's
/// condition (`scripts/refusal-surface`).
pub enum StatusShort {
    /// The device does not address buffers (`DeviceProfile::addressed_needs`).
    Addresses,
    /// The stage's stores and atomics, which the device has not enabled.
    Stores(VkFeature),
}

/// What the stage lacks to write the command buffer's `ExecutionStatus` words, or `None` where
/// it can ([`state_status`] then states the buffer's address).
pub fn status_short(cx: &SelectCx<'_>) -> Option<StatusShort> {
    if cx.profile.fields().buffer_access != BufferAccess::Addressed {
        return Some(StatusShort::Addresses);
    }
    let enabled = &cx.profile.fields().enabled.features;
    super::dynamic::stage_stores(cx.stage)
        .find(|f| !enabled.contains(f))
        .map(StatusShort::Stores)
}

/// State the `ExecutionStatus` buffer's address for the stage, which [`status_short`] found it
/// can write.
pub fn state_status(cx: &mut SelectCx<'_>) {
    cx.push.insert(PushField::ExecutionStatusAddress);
    cx.facts.insert(RuntimeFact::ExecutionStatus);
}

/// Emit absent read `s`, whose result is `vid`: set the bit, and define the result as zero.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &AbsentRead,
) -> BResult<()> {
    mark(cx, fb)?;
    let Some(d) = &s.result else {
        return Ok(());
    };
    let Some(ty) = crate::lower::arith::spirv_type(fb.module(), d)? else {
        return Err(cx.fail("an absent read without a lowered result type"));
    };
    let zero = fb.module().constant(Constant::Null { ty })?;
    let zero = fb.module().const_value(zero)?;
    cx.define(vid, zero);
    Ok(())
}

/// Set `StatusBits::ABSENT_ARGUMENT_READ` in the command buffer's `ExecutionStatus` word, where
/// the open block runs (a relaxed device-scope `OR`); [`require`] stated what it needs.
pub fn mark(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    mark_bits(cx, fb, StatusBits::ABSENT_ARGUMENT_READ)
}

/// Whether the stage can write the command buffer's `ExecutionStatus` word ([`require`]'s needs:
/// the buffer's device address and the stage's stores and atomics), stating the buffer's address
/// for it where it can.
pub fn status_writable(cx: &mut SelectCx<'_>) -> bool {
    let writable = status_short(cx).is_none();
    if writable {
        state_status(cx);
    }
    writable
}

/// Set `bits` in the command buffer's `ExecutionStatus` word, where the open block runs (a
/// relaxed device-scope `OR`); the stage stated the buffer's address ([`require`],
/// [`status_writable`]).
pub fn mark_bits(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    bits: StatusBits,
) -> BResult<()> {
    let word = status_word(cx, fb, StatusWord::Status)?;
    let bit = k32(fb, bits.0)?;
    fb.atomic(AtomicOp::Or, word, Scope::Device, Semantics::RELAXED, bit)?;
    Ok(())
}

/// The place of the command buffer's `ExecutionStatus` word `w` (a `u32` by device address);
/// the stage stated the buffer's address ([`state_status`]).
pub fn status_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    w: StatusWord,
) -> BResult<crate::spirv::ids::Value> {
    let status = cx.push_field(fb, PushField::ExecutionStatusAddress)?;
    let address = match w.byte_offset() {
        0 => status,
        at => {
            let off = k64(fb, u64::from(at))?;
            fb.binary(BinOp::IAdd, status, off)?
        }
    };
    let u32_ty = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    fb.device_place(address, u32_ty)
}
