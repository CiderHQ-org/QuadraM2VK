//! One indirect Metal dispatch on a flat variant: the GPU path of
//! `dispatchThreadsWithIndirectBuffer:` and `dispatchThreadgroupsWithIndirectBuffer:` for a kernel
//! that does not observe its threadgroup partition.
//!
//! A flat variant (`LoweringInputs::flat_grid`, [`ReplayVariant::flat`](super::ReplayVariant::flat))
//! runs any grid in threadgroups of its own size, reading Metal's `threads_per_grid` from a region
//! record (`ReplayRegionAddress`). Where the grid and the threadgroup size are GPU data, one
//! invocation of the crate-owned module [`flat_dispatch_module`] reads Metal's arguments where
//! they lie and writes the record and the `VkDispatchIndirectCommand` the variant is then
//! dispatched with, as the device replay's preprocess does for one command of a flat variant
//! (the same derivation, `flat_dispatch`):
//!
//! - **Metal's answers**. A zero axis of the grid or of the group counts runs nothing, and so does
//!   a threadgroup with a zero axis or more threads than the state's
//!   `maxTotalThreadsPerThreadgroup`: the command is zero and nothing is reported. Any other
//!   grid's `threads_per_grid` (a thread dispatch's grid, or a threadgroup dispatch's group counts
//!   times its threadgroup size) is the record's grid, and its `⌈threads / N⌉` threadgroups of
//!   the variant's `N` threads spread over `x`, then `y`, then `z` within the device's group-count
//!   limits.
//! - **Folded groups** ([`FlatArguments::FoldedThreadgroups`]). The threadgroups form of a kernel
//!   that observes its partition runs on a folded variant (`VariantKey::folded_groups`) of the
//!   call's threadgroup size `L`: the same derivation with `N` the volume of `L`, so its `⌈threads
//!   / N⌉` threadgroups are Metal's group counts' product, spread likewise, and the record's
//!   nominal size is `L` packed, from which the variant counts Metal's threadgroups.
//! - **What it cannot run** is a grid whose threadgroups no group count within the limits holds,
//!   a threadgroup dispatch whose grid is above `2^32 - 1` threads on an axis, or a grid of more
//!   than `2^64 - 1` threads: the command is zero and the module sets
//!   [`StatusBits::REPLAY_FAULT`](crate::bind::StatusBits) with [`ReplayFaults::GROUP_COUNT`] in
//!   the command buffer's `ExecutionStatus`, which the consumer reports as the device replay's.
//!
//! **The scratch** is one consumer buffer of [`FlatIndirect::SCRATCH_BYTES`] with a device address
//! and `STORAGE_BUFFER | INDIRECT_BUFFER` usage: the record at [`FlatIndirect::RECORD_OFFSET`]
//! (the `DispatchRegion` value; its base is 0, and so is its nominal size but on a folded variant,
//! as a flat variant reads its grid alone), the command at [`FlatIndirect::COMMAND_OFFSET`] (both
//! written by the module), and the module's inputs at [`FlatIndirect::INPUTS_OFFSET`]
//! ([`FlatIndirect::inputs`], written by the consumer): Metal's arguments' device address, their
//! form, the call's threadgroup size for the threadgroups form, and the variant's threadgroup size
//! and maximum.
//!
//! **The module** (fixed per device: the group-count limits are its constants): one `GLCompute`
//! entry `main` of local size `(1, 1, 1)`, dispatched `(1, 1, 1)`, no descriptor, no loop, and
//! [`FlatDispatchModule::PUSH_BYTES`] of `COMPUTE` push constants
//! ([`FlatDispatchModule::push_constants`]): the scratch's device address and the
//! `ExecutionStatus` buffer's. It reads 12 bytes of the threadgroups form's arguments and 24 of
//! the threads form's, nothing past them.
//!
//! **Recording** (consumer): the inputs' write and Metal's arguments' writes (every earlier
//! command's, as Metal reads them when the dispatch runs) before the module (`COMPUTE_SHADER`,
//! `SHADER_READ`); then its `SHADER_WRITE` before the variant's dispatch reads the command
//! (`DRAW_INDIRECT`, `INDIRECT_COMMAND_READ`) and the record (`COMPUTE_SHADER`, `SHADER_READ`);
//! then the variant's dispatch, its `ReplayRegionAddress` the scratch's address plus
//! [`FlatIndirect::RECORD_OFFSET`], by `vkCmdDispatchIndirect` from the scratch at
//! [`FlatIndirect::COMMAND_OFFSET`]. A scratch reused for another dispatch is rewritten only after
//! that dispatch's reads.
//!
//! **Where it runs.** Profile A: the module and the variant reach memory by device address
//! (`bufferDeviceAddress`, `shaderInt64`). Elsewhere [`flat_dispatch_module`] names what is
//! missing (`GpuEncodedIndirectCommands`).

use std::sync::Arc;

use super::{at, flat_dispatch, k32, load32, load64, no_binding, report, store32, Types};
use crate::bind::status::ReplayFaults;
use crate::budget::{Budget, Meter};
use crate::device::profile::BufferAccess;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::lower::dispatch::NOMINAL_AXIS_BITS;
use crate::pipeline::layout::{PipelineLayout, PushConstantRange};
use crate::pipeline::sym::PushField;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ExecMode, ExecModel, MemAccess, Type, TypedModule, UnOp};
use crate::vk;

/// The input words, from [`FlatIndirect::INPUTS_OFFSET`]: Metal's arguments' device address
/// (`u64`), the form ([`FORM_THREADS`], [`FORM_THREADGROUPS`] or [`FORM_FOLDED`]), the variant's
/// threads per threadgroup and its maximum, and the call's threadgroup size (`u32` x 3).
const I_ARGUMENTS: u64 = 0;
const I_FORM: u64 = 8;
const I_SIZE: u64 = 12;
const I_MAX: u64 = 16;
const I_LOCAL: u64 = 20;

/// The form word: the threads form on a flat variant.
const FORM_THREADS: u32 = 0;
/// The form word: the threadgroups form on a flat variant.
const FORM_THREADGROUPS: u32 = 1;
/// The form word: the threadgroups form on a folded variant.
const FORM_FOLDED: u32 = 2;

/// Which Metal arguments a flat dispatch reads ([`FlatIndirect`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlatArguments {
    /// `MTLDispatchThreadsIndirectArguments` (24 bytes): the grid's threads, then the threadgroup
    /// size.
    Threads,
    /// `MTLDispatchThreadgroupsIndirectArguments` (12 bytes): the group counts, of threadgroups of
    /// the call's `threads_per_threadgroup`.
    Threadgroups { threads_per_threadgroup: [u32; 3] },
    /// `MTLDispatchThreadgroupsIndirectArguments` (12 bytes) run on a folded variant
    /// (`VariantKey::folded_groups`) of the call's `threads_per_threadgroup`, which is also the
    /// dispatch's threadgroup size: the group counts' product spreads over `x`, then `y`, then
    /// `z` within the device's limits, and the record's nominal size is the call's threadgroup
    /// size packed, from which the variant counts Metal's threadgroups.
    FoldedThreadgroups { threads_per_threadgroup: [u32; 3] },
}

impl FlatArguments {
    /// The arguments' bytes.
    pub fn bytes(self) -> u64 {
        match self {
            FlatArguments::Threads => 24,
            FlatArguments::Threadgroups { .. } | FlatArguments::FoldedThreadgroups { .. } => 12,
        }
    }
}

/// One indirect Metal dispatch run on a flat variant (the module docs): where its arguments lie,
/// their form, and the variant's threadgroup size and its state's
/// `maxTotalThreadsPerThreadgroup`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlatIndirect {
    /// The device address of Metal's arguments (a multiple of 4).
    pub arguments: u64,
    /// The arguments' form.
    pub form: FlatArguments,
    /// The flat variant's threadgroup size.
    pub threads_per_threadgroup: [u32; 3],
    /// The variant's state's `maxTotalThreadsPerThreadgroup` (1 to 1,024).
    pub max_total_threads: u32,
}

impl FlatIndirect {
    /// Bytes of the scratch.
    pub const SCRATCH_BYTES: u64 = 48 + 32;
    /// The region record the module writes and the variant reads (32 bytes, 16-aligned).
    pub const RECORD_OFFSET: u64 = 0;
    /// The `VkDispatchIndirectCommand` the module writes (12 bytes).
    pub const COMMAND_OFFSET: u64 = 32;
    /// The module's inputs the consumer writes ([`FlatIndirect::inputs`], 32 bytes, 8-aligned).
    pub const INPUTS_OFFSET: u64 = 48;

    /// The dispatch of Metal's arguments of `form` at device address `arguments`, on a flat
    /// variant of `threads_per_threadgroup` for a state of `max_total_threads`. `None` where the
    /// address is not a multiple of 4 (Metal's validation refuses such an offset), the variant's
    /// threadgroup has a zero axis or more than `2^32 - 1` threads, or the maximum is not 1 to
    /// 1,024: no flat variant is registered so ([`ReplayVariant::flat`](super::ReplayVariant::flat)).
    pub fn new(
        arguments: u64,
        form: FlatArguments,
        threads_per_threadgroup: [u32; 3],
        max_total_threads: u32,
    ) -> Option<FlatIndirect> {
        let size = threads_per_threadgroup
            .iter()
            .try_fold(1u32, |n, &axis| n.checked_mul(axis))?;
        let valid = arguments.is_multiple_of(4)
            && size != 0
            && (1..=super::MAX_THREADGROUP_THREADS).contains(&max_total_threads);
        valid.then_some(FlatIndirect {
            arguments,
            form,
            threads_per_threadgroup,
            max_total_threads,
        })
    }

    /// The bytes the consumer writes at [`FlatIndirect::INPUTS_OFFSET`] before the module runs.
    pub fn inputs(&self) -> [u8; 32] {
        // `new` held the product within a `u32`.
        let size = self
            .threads_per_threadgroup
            .iter()
            .fold(1u32, |n, &axis| n.saturating_mul(axis));
        let (form, local) = match self.form {
            FlatArguments::Threads => (FORM_THREADS, [0; 3]),
            FlatArguments::Threadgroups {
                threads_per_threadgroup,
            } => (FORM_THREADGROUPS, threads_per_threadgroup),
            FlatArguments::FoldedThreadgroups {
                threads_per_threadgroup,
            } => (FORM_FOLDED, threads_per_threadgroup),
        };
        let mut bytes = [0u8; 32];
        let mut put = |offset: u64, word: &[u8]| {
            let offset = offset as usize;
            bytes[offset..offset + word.len()].copy_from_slice(word);
        };
        put(I_ARGUMENTS, &self.arguments.to_le_bytes());
        put(I_FORM, &form.to_le_bytes());
        put(I_SIZE, &size.to_le_bytes());
        put(I_MAX, &self.max_total_threads.to_le_bytes());
        for (i, axis) in local.iter().enumerate() {
            put(I_LOCAL + 4 * i as u64, &axis.to_le_bytes());
        }
        bytes
    }
}

/// The crate-owned flat-dispatch module of one device (the module docs), validated like any
/// output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlatDispatchModule {
    /// SPIR-V words, entry point `main`, no descriptor.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (device addresses and `shaderInt64`).
    pub requirements: Requirements,
    /// The group-count limits the flat dispatch spreads its threadgroups within.
    pub limits: [u32; 3],
    /// The pipeline layout the consumer creates the module's pipeline with: no descriptor set,
    /// and one `COMPUTE` push-constant range of [`FlatDispatchModule::PUSH_BYTES`] at offset 0.
    pub layout: PipelineLayout,
}

impl FlatDispatchModule {
    /// Bytes of the `COMPUTE` push-constant range at offset 0.
    pub const PUSH_BYTES: u32 = 16;
    /// The scratch's device address (`u64`).
    pub const SCRATCH_PUSH: u32 = 0;
    /// The `ExecutionStatus` buffer's device address (`u64`).
    pub const STATUS_PUSH: u32 = 8;

    /// The push bytes for the scratch at device address `scratch` and the `ExecutionStatus`
    /// buffer at `status`.
    pub fn push_constants(scratch: u64, status: u64) -> [u8; 16] {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&scratch.to_le_bytes());
        bytes[8..].copy_from_slice(&status.to_le_bytes());
        bytes
    }
}

/// The flat-dispatch module of `device` (the module docs): built with the typed builder, checked
/// by the owned verifier against the device's profile (requirements included), and serialized.
/// A device without device addresses is `Unsupported{GpuEncodedIndirectCommands}` naming what it
/// lacks.
pub fn flat_dispatch_module(device: &Device) -> Result<FlatDispatchModule, Error> {
    let profile = device.profile();
    if profile.fields().buffer_access != BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![profile.addressed_needs()],
        )
        .into());
    }
    let limits = profile.fields().limits.max_compute_work_group_count;
    let m = typed_module(limits)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the flat-dispatch module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &no_binding)?;
    Ok(FlatDispatchModule {
        spirv: words.into(),
        requirements: r.requirements,
        limits,
        layout: PipelineLayout {
            sets: Vec::new(),
            push_constants: vec![PushConstantRange {
                stages: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: FlatDispatchModule::PUSH_BYTES,
            }],
            push_descriptor_eligible: false,
        },
    })
}

/// The flat-dispatch module for `limits` as a typed module, before serialization (what
/// [`flat_dispatch_module`] serializes, and what the CPU interpreter runs). Its push fields are
/// `ReplayRegionAddress` (the scratch, whose record is at its start) and
/// `ExecutionStatusAddress`, at [`FlatDispatchModule`]'s offsets.
pub(super) fn typed_module(limits: [u32; 3]) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let t = Types { u32_, u64_ };
    let m_scratch = b.push_member(
        PushField::ReplayRegionAddress,
        u64_,
        FlatDispatchModule::SCRATCH_PUSH,
    )?;
    let m_status = b.push_member(
        PushField::ExecutionStatusAddress,
        u64_,
        FlatDispatchModule::STATUS_PUSH,
    )?;
    let push = b.push_global()?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let fb = &mut f;

    let push_value = fb.module().global_value(push)?;
    let field = |fb: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let i = k32(fb, index)?;
        let p = fb.access_chain(push_value, &[i])?;
        fb.load(p, MemAccess::default())
    };
    let scratch = field(fb, m_scratch.index)?;
    let status = field(fb, m_status.index)?;

    // The inputs the consumer wrote.
    let inputs = at(fb, scratch, FlatIndirect::INPUTS_OFFSET)?;
    let input = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let a = at(fb, inputs, offset)?;
        load32(fb, t, a)
    };
    let arguments_at = at(fb, inputs, I_ARGUMENTS)?;
    let arguments = load64(fb, t, arguments_at)?;
    let form = input(fb, I_FORM)?;
    let size = input(fb, I_SIZE)?;
    let max_total = input(fb, I_MAX)?;
    let zero32 = k32(fb, 0)?;
    let threads_form = k32(fb, FORM_THREADS)?;
    let threadgroups = fb.binary(BinOp::INotEqual, form, threads_form)?;
    let folded_form = k32(fb, FORM_FOLDED)?;
    let folded = fb.binary(BinOp::IEqual, form, folded_form)?;

    // Metal's arguments: the grid (or group counts), then the threads form's threadgroup size;
    // the threadgroups form's is the call's, so nothing past its 12 bytes is read.
    let mut grid = [zero32; 3];
    for (i, g) in grid.iter_mut().enumerate() {
        let a = at(fb, arguments, 4 * i as u64)?;
        *g = load32(fb, t, a)?;
    }
    let from_call = at(fb, inputs, I_LOCAL)?;
    let from_arguments = at(fb, arguments, 12)?;
    let local_at = fb.select(threadgroups, from_call, from_arguments)?;
    let mut local = [zero32; 3];
    for (i, l) in local.iter_mut().enumerate() {
        let a = at(fb, local_at, 4 * i as u64)?;
        *l = load32(fb, t, a)?;
    }
    let flat = flat_dispatch(
        fb,
        t,
        (threadgroups, grid, local),
        (size, max_total),
        limits,
    )?;

    // The record: the `DispatchRegion` value, of which a flat variant reads the grid alone, and a
    // folded variant the grid and the call's threadgroup size, packed, which a threadgroup that
    // runs holds within 10 bits an axis (at most 1,024 threads).
    let record = at(fb, scratch, FlatIndirect::RECORD_OFFSET)?;
    let [x, y, z] = flat.threads;
    let one32 = k32(fb, 1)?;
    let mut packed = zero32;
    for (i, l) in local.into_iter().enumerate() {
        let less = fb.binary(BinOp::ISub, l, one32)?;
        let shift = k32(fb, NOMINAL_AXIS_BITS * i as u32)?;
        let axis = fb.binary(BinOp::ShiftLeftLogical, less, shift)?;
        packed = fb.binary(BinOp::BitwiseOr, packed, axis)?;
    }
    let nominal = fb.select(folded, packed, zero32)?;
    let words = [zero32, zero32, zero32, nominal, x, y, z, zero32];
    for (i, w) in words.into_iter().enumerate() {
        let a = at(fb, record, 4 * i as u64)?;
        store32(fb, t, a, w)?;
    }

    // The command: the flat dispatch's group count where it runs and fits, zero otherwise.
    let fits = fb.unary(UnOp::LogicalNot, flat.too_large)?;
    let go = fb.binary(BinOp::LogicalAnd, flat.runs, fits)?;
    let command = at(fb, scratch, FlatIndirect::COMMAND_OFFSET)?;
    for (i, g) in flat.groups.into_iter().enumerate() {
        let n = fb.select(go, g, zero32)?;
        let a = at(fb, command, 4 * i as u64)?;
        store32(fb, t, a, n)?;
    }

    // A grid Metal runs that no group count holds is the replay's `GROUP_COUNT` fault.
    let refused = fb.binary(BinOp::LogicalAnd, flat.runs, flat.too_large)?;
    let reported: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let faults = k32(fb, ReplayFaults::GROUP_COUNT.0)?;
        report(fb, t, status, faults)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), refused, reported, None)?;
    fb.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}
