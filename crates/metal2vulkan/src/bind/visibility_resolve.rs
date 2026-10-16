//! The boolean resolve of a pass's visibility results (step 6's third part).
//!
//! Metal's visibility results are occlusion queries. In `Counting` mode Metal writes the samples
//! that passed, which a precise query's result is. In `Boolean` mode Metal writes exactly 1 when
//! any sample passed and 0 otherwise (measured on an M4 with macOS 27.0.1); a Vulkan query's result
//! is a count (any nonzero value without the precise flag), so a copy alone cannot give it. The
//! consumer copies every query's 64-bit result to its Metal offset in the visibility buffer
//! (`vkCmdCopyQueryPoolResults`), then records [`VisibilityResolve`] for the pass's `Boolean`
//! offsets: one dispatch of the crate-owned module [`visibility_resolve_module`] rewrites each of
//! their words in place as Metal's 1 or 0. `Counting` words are left as copied.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`VisibilityResolveModule::LOCAL_SIZE`], 1, 1), no loops, no push constants, and two storage
//! buffers in set 0:
//!
//! | Binding | Buffer | Range |
//! |---|---|---|
//! | [`VisibilityResolveModule::TARGETS_BINDING`] | the consumer's upload of [`VisibilityResolve::targets`] | exactly its bytes |
//! | [`VisibilityResolveModule::DEST_BINDING`] | the visibility buffer | [`VisibilityResolve::dest_range`] |
//!
//! Invocation `i` below the targets' count rewrites the 64-bit word `targets[i]` of the bound
//! visibility range: 1 when either half is nonzero, else 0, with a high half of 0. The targets'
//! count is the bound table's length, so a range bound whole is the whole contract; a word past
//! the bound visibility range is skipped, so a wrongly bound range can never make the resolve
//! fault. Distinct targets touch distinct words, and the rewrite is idempotent.
//!
//! **Synchronization** (consumer): the result copies' `TRANSFER_WRITE` → `SHADER_STORAGE_READ`
//! and `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the resolve, and the upload's host write
//! before the submission; the resolve's writes are then ordered before the buffer's later uses as
//! any compute write is.

use std::sync::Arc;

use crate::bind::instantiate::BufferRange;
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

/// One dispatch of [`visibility_resolve_module`]: it rewrites, in place, the `Boolean` results
/// of a pass whose words lie in `dest_range` of the visibility buffer.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VisibilityResolve {
    /// The range of the visibility buffer to bind at [`VisibilityResolveModule::DEST_BINDING`]:
    /// from the first word's offset aligned down to `minStorageBufferOffsetAlignment` through
    /// the last word's end.
    pub dest_range: BufferRange,
    /// The table to upload (little-endian `u32`s) and bind, exactly, at
    /// [`VisibilityResolveModule::TARGETS_BINDING`]: the index of each result's 64-bit word
    /// inside `dest_range`, ascending.
    pub targets: Vec<u32>,
    /// Workgroups to dispatch along x, each of [`VisibilityResolveModule::LOCAL_SIZE`]
    /// invocations.
    pub group_count: u32,
}

impl VisibilityResolve {
    /// Bytes of one result in the visibility buffer.
    pub const RESULT_BYTES: u64 = 8;

    /// The dispatches that resolve the `Boolean` results at Metal's `offsets` (any order;
    /// repeats resolve once), each binding the visibility buffer at an offset aligned to
    /// `alignment` (`minStorageBufferOffsetAlignment`) over at most `max_range` bytes
    /// (`maxStorageBufferRange`) and dispatching at most `max_groups` workgroups
    /// (`maxComputeWorkGroupCount[0]`). Results too far apart for one binding, or too many for
    /// one dispatch, take more dispatches. No dispatch when `offsets` is empty. `None` when an
    /// offset is not a multiple of 8 (Metal requires it of a visibility offset), its word
    /// overflows, or one word does not fit `max_range` past its aligned offset, or `max_groups`
    /// is 0.
    pub fn plan(
        offsets: &[u64],
        alignment: u64,
        max_range: u32,
        max_groups: u32,
    ) -> Option<Vec<VisibilityResolve>> {
        let mut sorted = offsets.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        let max_targets =
            u64::from(max_groups).checked_mul(u64::from(VisibilityResolveModule::LOCAL_SIZE))?;
        let items = sorted.into_iter().map(|offset| (offset, ()));
        windows(items, alignment, max_range, max_targets)?
            .into_iter()
            .map(|(dest_range, items)| {
                let targets: Vec<u32> = items.into_iter().map(|(index, ())| index).collect();
                let groups =
                    (targets.len() as u64).div_ceil(u64::from(VisibilityResolveModule::LOCAL_SIZE));
                Some(VisibilityResolve {
                    dest_range,
                    targets,
                    group_count: u32::try_from(groups).ok()?,
                })
            })
            .collect()
    }

    /// The targets as the bytes to upload.
    pub fn target_bytes(&self) -> Vec<u8> {
        self.targets.iter().flat_map(|t| t.to_le_bytes()).collect()
    }
}

/// The windows of the visibility buffer that bind `items`, each at Metal's offset of its result
/// (ascending, each once): each window binds from an offset aligned down to `alignment` (at
/// least 8) over at most `max_range` bytes, and takes at most `max_items` items, each with the
/// index of its 64-bit word inside the window. A dispatch per window: [`VisibilityResolve`]'s
/// and the continuation combine's (`visibility_combine.rs`). `None` when an offset is not a
/// multiple of 8, its word overflows, or one word does not fit `max_range` past its aligned
/// offset, or `max_items` is 0 (with an item).
pub(super) fn windows<T>(
    items: impl IntoIterator<Item = (u64, T)>,
    alignment: u64,
    max_range: u32,
    max_items: u64,
) -> Option<Vec<(BufferRange, Vec<(u32, T)>)>> {
    let a = alignment.max(VisibilityResolve::RESULT_BYTES);
    let mut out: Vec<(BufferRange, Vec<(u32, T)>)> = Vec::new();
    for (offset, item) in items {
        if !offset.is_multiple_of(VisibilityResolve::RESULT_BYTES) {
            return None;
        }
        let end = offset.checked_add(VisibilityResolve::RESULT_BYTES)?;
        let joins = out.last().is_some_and(|(range, taken)| {
            end - range.offset <= u64::from(max_range) && (taken.len() as u64) < max_items
        });
        if !joins {
            let start = offset - offset % a;
            if end - start > u64::from(max_range) || max_items == 0 {
                return None;
            }
            let range = BufferRange {
                offset: start,
                range: 0,
            };
            out.push((range, Vec::new()));
        }
        let (range, taken) = out.last_mut()?;
        // Inside `max_range` (a `u32`) of the range's start: the index fits.
        let index =
            u32::try_from((offset - range.offset) / VisibilityResolve::RESULT_BYTES).ok()?;
        taken.push((index, item));
        range.range = end - range.offset;
    }
    Some(out)
}

/// The crate-owned visibility resolve module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibilityResolveModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl VisibilityResolveModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The table of targets, a storage buffer.
    pub const TARGETS_BINDING: u32 = 0;
    /// The visibility buffer, a storage buffer.
    pub const DEST_BINDING: u32 = 1;
    /// Invocations of one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 64;
}

/// The binding symbol of the targets in the typed module
/// ([`VisibilityResolveModule::TARGETS_BINDING`]).
pub const TARGETS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the visibility buffer in the typed module
/// ([`VisibilityResolveModule::DEST_BINDING`]).
pub const DEST: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
pub(super) fn words_buffer(
    b: &mut ModuleBuilder,
    sym: ResourceSlot,
    u32_: TypeId,
) -> BResult<GlobalId> {
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

pub(super) fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The resolve module as a typed module, before serialization: the module
/// [`visibility_resolve_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols [`TARGETS`] and [`DEST`].
pub fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let targets = words_buffer(&mut b, TARGETS, u32_)?;
    let dest = words_buffer(&mut b, DEST, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let i = f.composite_extract(invocation, &[0])?;
    let targets = f.module().global_value(targets)?;
    let count = f.array_length(targets, 0)?;
    let dest = f.module().global_value(dest)?;
    let dest_len = f.array_length(dest, 0)?;
    let listed = f.binary(BinOp::ULessThan, i, count)?;
    // Invocation `i` rewrites the word `targets[i]` names, when its two halves lie in the range.
    let resolve: ArmFn<'_, ()> = Box::new(move |f, _| {
        let p = f.access_chain(targets, &[zero, i])?;
        let target = f.load(p, MemAccess::default())?;
        let low_index = f.binary(BinOp::ShiftLeftLogical, target, one)?;
        let high_index = f.binary(BinOp::IAdd, low_index, one)?;
        let inside = f.binary(BinOp::ULessThan, high_index, dest_len)?;
        let rewrite: ArmFn<'_, ()> = Box::new(move |f, _| {
            let low_ptr = f.access_chain(dest, &[zero, low_index])?;
            let high_ptr = f.access_chain(dest, &[zero, high_index])?;
            let low = f.load(low_ptr, MemAccess::default())?;
            let high = f.load(high_ptr, MemAccess::default())?;
            let any = f.binary(BinOp::BitwiseOr, low, high)?;
            let passed = f.binary(BinOp::INotEqual, any, zero)?;
            let word = f.select(passed, one, zero)?;
            f.store(low_ptr, word, MemAccess::default())?;
            f.store(high_ptr, zero, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), inside, rewrite, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), listed, resolve, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: VisibilityResolveModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(VisibilityResolveModule::SET),
        Symbol::Binding(s) if *s == TARGETS => Some(VisibilityResolveModule::TARGETS_BINDING),
        Symbol::Binding(s) if *s == DEST => Some(VisibilityResolveModule::DEST_BINDING),
        _ => None,
    }
}

/// The visibility resolve module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`VisibilityResolveModule`].
pub fn visibility_resolve_module(device: &Device) -> Result<VisibilityResolveModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the visibility resolve needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(VisibilityResolveModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
