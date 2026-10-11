//! The indirect-clamp prepass.
//!
//! Metal's `dispatchThreadgroupsWithIndirectBuffer:` takes GPU-written group counts up to
//! `2^32 - 1` per axis; Vulkan's `vkCmdDispatchIndirect` requires them within
//! `maxComputeWorkGroupCount`. When any axis of that limit is below `2^32 - 1`, the plan asks the
//! consumer to record [`IndirectClamp`] first: one invocation of the crate-owned module
//! [`indirect_clamp_module`] copies Metal's three words from the source into a consumer-provided
//! 12-byte scratch, clamps each axis to the device limit, and sets `OVERFLOW` in the command
//! buffer's `ExecutionStatus` when it reduced a count, which the consumer reports as a
//! command-buffer error. The real dispatch then reads the scratch. Excess is never silent, and
//! the consumer never reads the arguments on the CPU.
//!
//! **The module** (fixed per device: the limits are its constants). One `GLCompute` entry `main`
//! with local size `(1, 1, 1)`, no loops, three storage buffers in set 0 and one push constant:
//!
//! | Binding | Buffer | Range |
//! |---|---|---|
//! | [`IndirectClampModule::SOURCE_BINDING`] | the Metal indirect buffer | [`IndirectClamp::source_range`] |
//! | [`IndirectClampModule::SCRATCH_BINDING`] | the consumer's scratch | [`IndirectClamp::SCRATCH_BYTES`] |
//! | [`IndirectClampModule::STATUS_BINDING`] | the `ExecutionStatus` buffer | `ExecutionStatus::BYTES` |
//!
//! The push constant ([`IndirectClamp::push_constants`], `COMPUTE`, offset 0) is the byte distance
//! from the bound source range's start to the arguments (a multiple of 4), because a storage
//! descriptor's offset must be a multiple of `minStorageBufferOffsetAlignment` while Metal's
//! argument offset is only 4-aligned. Reads of the source are contained (an index past the bound
//! range reads 0), so a wrongly bound range can never make the prepass fault.
//!
//! **Synchronization** (consumer): the source write → `SHADER_READ` in `COMPUTE_SHADER` before
//! the prepass; the scratch `SHADER_WRITE` → `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` between
//! the prepass and the dispatch.

use std::sync::Arc;

use crate::bind::instantiate::{BufferRange, IndirectRef};
use crate::bind::status::{StatusBits, StatusWord};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{
    PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, Stage, Symbol, SyntheticBinding,
};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, LayoutRule,
    MemAccess, Scope, Semantics, StorageClass, Type, TypedModule,
};

/// A clamp prepass the consumer records before an indirect dispatch: one invocation of
/// [`indirect_clamp_module`] reads Metal's arguments at `source`, writes the clamped
/// `VkDispatchIndirectCommand` into a consumer-provided scratch of
/// [`IndirectClamp::SCRATCH_BYTES`], and sets `OVERFLOW` in the command buffer's `ExecutionStatus`
/// when it reduced a count.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndirectClamp {
    pub source: IndirectRef,
    /// `maxComputeWorkGroupCount` (the module's constants).
    pub limits: [u32; 3],
    /// The range of the source buffer to bind at [`IndirectClampModule::SOURCE_BINDING`]: from
    /// `source.offset` aligned down to `minStorageBufferOffsetAlignment` (and to 4) through the
    /// 12 argument bytes.
    pub source_range: BufferRange,
    /// The push constant bytes (`COMPUTE`, offset 0): the arguments' byte offset inside
    /// `source_range`.
    pub push_constants: [u8; 4],
}

impl IndirectClamp {
    /// Bytes of the scratch the clamped arguments are written to (`VkDispatchIndirectCommand`).
    pub const SCRATCH_BYTES: u32 = 12;

    /// The prepass for Metal arguments at `source`, clamped to `limits`, with the source bound at
    /// an offset aligned to `alignment` (`minStorageBufferOffsetAlignment`). `None` when
    /// `source.offset` is not a multiple of 4 (Metal and Vulkan both require it) or the range
    /// overflows.
    pub fn new(source: IndirectRef, limits: [u32; 3], alignment: u64) -> Option<IndirectClamp> {
        if !source.offset.is_multiple_of(4) {
            return None;
        }
        let a = alignment.max(4);
        let start = source.offset - source.offset % a;
        let residual = u32::try_from(source.offset - start).ok()?;
        let end = source.offset.checked_add(u64::from(Self::SCRATCH_BYTES))?;
        Some(IndirectClamp {
            source,
            limits,
            source_range: BufferRange {
                offset: start,
                range: end - start,
            },
            push_constants: residual.to_le_bytes(),
        })
    }
}

/// The crate-owned clamp module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndirectClampModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
    /// The group-count limits the module clamps to.
    pub limits: [u32; 3],
}

impl IndirectClampModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The Metal indirect buffer, a storage buffer.
    pub const SOURCE_BINDING: u32 = 0;
    /// The consumer's scratch (`VkDispatchIndirectCommand`), a storage buffer.
    pub const SCRATCH_BINDING: u32 = 1;
    /// The command buffer's `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 2;
    /// Bytes of the `COMPUTE` push-constant range at offset 0.
    pub const PUSH_BYTES: u32 = 4;
}

/// The binding symbol of the source buffer in the typed module ([`IndirectClampModule::SOURCE_BINDING`]).
pub const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the scratch buffer in the typed module ([`IndirectClampModule::SCRATCH_BINDING`]).
pub const SCRATCH: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the status buffer in the typed module ([`IndirectClampModule::STATUS_BINDING`]).
pub const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
};
/// The push field that carries the source residual: the same quantity as a descriptor slot's
/// residual bind offset, for the source slot.
const RESIDUAL: PushField = PushField::SlotBindOffset(SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Buffer,
    index: 0,
});

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The clamp module for `limits` as a typed module, before serialization: the module
/// [`indirect_clamp_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols [`SOURCE`], [`SCRATCH`], and [`STATUS`]; its push field is the source residual at
/// offset 0.
pub fn typed_module(limits: [u32; 3]) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let bool_ = b.ty(Type::Bool)?;
    let source = words_buffer(&mut b, SOURCE, u32_)?;
    let scratch = words_buffer(&mut b, SCRATCH, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let residual = b.push_member(RESIDUAL, u32_, 0)?;
    let push = b.push_global()?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let two = u32_value(&mut f, 2)?;
    let push_value = f.module().global_value(push)?;
    let member = u32_value(&mut f, residual.index)?;
    let residual_ptr = f.access_chain(push_value, &[member])?;
    let residual_bytes = f.load(residual_ptr, MemAccess::default())?;
    let first = f.binary(BinOp::ShiftRightLogical, residual_bytes, two)?;
    let source_value = f.module().global_value(source)?;
    let len = f.array_length(source_value, 0)?;
    let scratch_value = f.module().global_value(scratch)?;
    let no = f.module().constant(Constant::Scalar {
        ty: bool_,
        bits: ConstBits(0),
    })?;
    let mut overflow = f.module().const_value(no)?;
    for (axis, limit) in limits.into_iter().enumerate() {
        let i = u32_value(&mut f, axis as u32)?;
        let index = f.binary(BinOp::IAdd, first, i)?;
        // Contained: an index past the bound range reads word 0 and yields 0.
        let inb = f.binary(BinOp::ULessThan, index, len)?;
        let safe = f.select(inb, index, zero)?;
        let p = f.access_chain(source_value, &[zero, safe])?;
        let loaded = f.load(p, MemAccess::default())?;
        let v = f.select(inb, loaded, zero)?;
        let limit = u32_value(&mut f, limit)?;
        let over = f.binary(BinOp::UGreaterThan, v, limit)?;
        let clamped = f.select(over, limit, v)?;
        let out = f.access_chain(scratch_value, &[zero, i])?;
        f.store(out, clamped, MemAccess::default())?;
        overflow = f.binary(BinOp::LogicalOr, overflow, over)?;
    }
    let status_value = f.module().global_value(status)?;
    let word = u32_value(&mut f, StatusWord::Status.index())?;
    let bit = u32_value(&mut f, StatusBits::OVERFLOW.0)?;
    let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let p = fb.access_chain(status_value, &[zero, word])?;
        fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), overflow, then, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IndirectClampModule::SET),
        Symbol::Binding(s) if *s == SOURCE => Some(IndirectClampModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == SCRATCH => Some(IndirectClampModule::SCRATCH_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(IndirectClampModule::STATUS_BINDING),
        _ => None,
    }
}

/// The indirect-clamp module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`IndirectClampModule`].
pub fn indirect_clamp_module(device: &Device) -> Result<IndirectClampModule, Error> {
    let profile = device.profile();
    let limits = profile.fields().limits.max_compute_work_group_count;
    let m = typed_module(limits)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the indirect clamp needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(IndirectClampModule {
        spirv: words.into(),
        requirements: r.requirements,
        limits,
    })
}

#[cfg(test)]
mod tests;
