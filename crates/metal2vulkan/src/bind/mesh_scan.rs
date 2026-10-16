//! The mesh scan of an emulated mesh draw whose mesh threadgroups only the GPU knows: a draw of a
//! pipeline with an object function, whose object threadgroups each set the mesh grid they launch,
//! and an indirect draw, whose mesh grid is Metal's indirect arguments.
//!
//! One workgroup of [`MeshScanModule::LOCAL_SIZE`] invocations reads the draw's work buffer
//! (`SyntheticBinding::MeshWork`, [`MeshDispatch::work`](crate::bind::MeshDispatch::work)): the
//! grid each of its `O` object threadgroups set (slot 0's, of an indirect draw, from Metal's
//! arguments, held to Metal's limit of 1048576 threadgroups as an object function's grid is), and
//! writes, after the slots:
//!
//! - each object threadgroup's first mesh threadgroup, in object threadgroup order, and `T`, the
//!   draw's mesh threadgroups (the offsets the mesh function's wrapper bisects);
//! - in the header, `T`, where the draw's records begin in the command buffer's records pool
//!   (`RECORDS_BASE`), the mesh dispatch's `VkDispatchIndirectCommand` (`T` folded into rows of
//!   the header's `ROW` workgroups) and the draw's `VkDrawIndirectCommand` (`T · NP · k`
//!   vertices).
//!
//! The draw's records are reserved from the records pool's cursor
//! ([`StatusWord::MeshRecordsCursor`], 0 before the command buffer's first scan): `T` mesh
//! threadgroups' words. A draw whose records the rest of the pool cannot hold, or whose vertices
//! pass a `u32`, runs and draws nothing; the scan sets [`StatusBits::MESH_RECORDS_OVERFLOW`] and,
//! where it is the command buffer's first records fault of either kind (mesh or tessellation), the
//! consumer's number for the draw, its object threadgroups and the mesh threadgroups it needed
//! (saturated) in the fault words, which
//! [`ExecutionStatus::decode`](crate::bind::ExecutionStatus::decode) reports. Nothing is drawn
//! short silently.
//!
//! **Bindings** (set 0, storage buffers): [`MeshScanModule::WORK_BINDING`] (the draw's work
//! buffer, whole), `STATUS_BINDING` (the command buffer's `ExecutionStatus` buffer) and
//! `ARGUMENTS_BINDING` (for an indirect draw, the Metal indirect buffer's range from the
//! arguments' offset aligned down to `minStorageBufferOffsetAlignment`; for any other draw any 4
//! bytes, which nothing reads). Every read of the arguments is contained.
//!
//! **Synchronization** (consumer): the object dispatch's writes, the arguments' writes and the
//! upload of the work buffer → `SHADER_READ` in `COMPUTE_SHADER` before the scan; scans one
//! after another (each reads and writes the pool's cursor); the scan's writes →
//! `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` and `SHADER_READ` in `COMPUTE_SHADER` before the
//! mesh dispatch, and in `VERTEX_SHADER` before the draw.

use std::sync::Arc;

use crate::bind::index_bounds::finished;
use crate::bind::replay::checked_mul;
use crate::bind::status::{StatusBits, StatusWord};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::lower::mesh::work::{self, METAL_MESH_GRID_GROUPS};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, GlslOp, IoDir,
    LayoutRule, MemAccess, MemoryOrder, Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};

/// The crate-owned mesh scan module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshScanModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl MeshScanModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The draw's work buffer, a storage buffer.
    pub const WORK_BINDING: u32 = 0;
    /// The command buffer's `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 1;
    /// The Metal indirect buffer, a storage buffer.
    pub const ARGUMENTS_BINDING: u32 = 2;
    /// Invocations of the one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 64;
}

/// The work buffer's binding symbol in the typed module.
pub const WORK: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The arguments' binding symbol ([`MeshScanModule::ARGUMENTS_BINDING`]).
pub const ARGUMENTS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The status buffer's binding symbol ([`MeshScanModule::STATUS_BINDING`]).
pub const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
pub(crate) fn words_buffer(
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

pub(crate) fn k(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The pointer to word `index` of `buffer`'s runtime array.
pub(crate) fn word_ptr(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    f.access_chain(buffer, &[zero, index])
}

pub(crate) fn load(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let p = word_ptr(f, buffer, index)?;
    f.load(p, MemAccess::default())
}

pub(crate) fn load_at(f: &mut FunctionBuilder<'_>, buffer: Value, index: u32) -> BResult<Value> {
    let i = k(f, index)?;
    load(f, buffer, i)
}

pub(crate) fn store(
    f: &mut FunctionBuilder<'_>,
    buffer: Value,
    index: Value,
    v: Value,
) -> BResult<()> {
    let p = word_ptr(f, buffer, index)?;
    f.store(p, v, MemAccess::default())
}

pub(crate) fn store_at(
    f: &mut FunctionBuilder<'_>,
    buffer: Value,
    index: u32,
    v: Value,
) -> BResult<()> {
    let i = k(f, index)?;
    store(f, buffer, i, v)
}

/// Word `index` of `buffer` where it lies inside the buffer's bound range, else 0 (contained).
fn contained(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    let len = f.array_length(buffer, 0)?;
    let inside = f.binary(BinOp::ULessThan, index, len)?;
    let safe = f.select(inside, index, zero)?;
    let loaded = load(f, buffer, safe)?;
    f.select(inside, loaded, zero)
}

/// `a + b`, or `u32::MAX` where it wraps.
pub(crate) fn saturating_add(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let sum = f.binary(BinOp::IAdd, a, b)?;
    let wrapped = f.binary(BinOp::ULessThan, sum, a)?;
    let max = k(f, u32::MAX)?;
    f.select(wrapped, max, sum)
}

/// `a / b` rounded up (`b` not 0), without wrapping.
pub(crate) fn div_ceil(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let q = f.binary(BinOp::UDiv, a, b)?;
    let r = f.binary(BinOp::UMod, a, b)?;
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let partial = f.binary(BinOp::INotEqual, r, zero)?;
    let extra = f.select(partial, one, zero)?;
    f.binary(BinOp::IAdd, q, extra)
}

/// The threadgroups of grid `g`: `x · y · z` where the grid lies within Metal's limit of
/// [`METAL_MESH_GRID_GROUPS`] threadgroups (as an emulated object function holds the grid it
/// sets), else 0. The product compared wraps only where an earlier test fails.
fn bounded_groups(f: &mut FunctionBuilder<'_>, [x, y, z]: [Value; 3]) -> BResult<Value> {
    let one = k(f, 1)?;
    let total = k(f, METAL_MESH_GRID_GROUPS)?;
    let ny = f.glsl(GlslOp::UMax, &[y, one])?;
    let room = f.binary(BinOp::UDiv, total, ny)?;
    let mut all = f.binary(BinOp::ULessThanEqual, x, room)?;
    let xy = f.binary(BinOp::IMul, x, y)?;
    let nz = f.glsl(GlslOp::UMax, &[z, one])?;
    let room = f.binary(BinOp::UDiv, total, nz)?;
    let c = f.binary(BinOp::ULessThanEqual, xy, room)?;
    all = f.binary(BinOp::LogicalAnd, all, c)?;
    let xyz = f.binary(BinOp::IMul, xy, z)?;
    let zero = k(f, 0)?;
    f.select(all, xyz, zero)
}

/// Run `each(f, i)` for `i` from `lo` below `hi`.
pub(crate) fn for_range<'a>(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    (lo, hi): (Value, Value),
    each: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<()> + 'a,
) -> BResult<()> {
    let var = f.local_var(u32_, None)?;
    let counter = f.var_ptr(var)?;
    f.store(counter, lo, MemAccess::default())?;
    let body: LoopFn<'a, ()> = Box::new(move |f, _, lc| {
        let i = f.load(counter, MemAccess::default())?;
        let done = f.binary(BinOp::UGreaterThanEqual, i, hi)?;
        let exit: ArmFn<'_, ()> = Box::new(move |f, _| {
            lc.break_(f)?;
            Ok(ArmEnd::Exits)
        });
        f.if_(&mut (), done, exit, None)?;
        let one = k(f, 1)?;
        let next = f.binary(BinOp::IAdd, i, one)?;
        f.store(counter, next, MemAccess::default())?;
        each(f, i)
    });
    f.loop_(&mut (), LoopSpec::default(), body)
}

/// Run `then(f)` where `c` holds.
pub(crate) fn when<'a>(
    f: &mut FunctionBuilder<'_>,
    c: Value,
    then: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
) -> BResult<()> {
    let arm: ArmFn<'a, ()> = Box::new(move |f, _| {
        then(f)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), c, arm, None)?;
    Ok(())
}

/// The workgroup barrier over the partial sums.
pub(crate) fn barrier(f: &mut FunctionBuilder<'_>) -> BResult<()> {
    let sem = Semantics {
        order: MemoryOrder::AcquireRelease,
        workgroup_memory: true,
        ..Semantics::RELAXED
    };
    f.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)
}

/// The scan as a typed module, before serialization: the module [`mesh_scan_module`] serializes,
/// which the CPU interpreter runs. Its bindings are the symbols [`WORK`], [`ARGUMENTS`] and
/// [`STATUS`].
pub fn scan_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let lanes = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(MeshScanModule::LOCAL_SIZE)),
    })?;
    let partial_ty = b.ty(Type::Array {
        elem: u32_,
        len: lanes,
        stride: None,
    })?;
    let partial = b.plain_global(StorageClass::Workgroup, partial_ty, None)?;
    let total_var = b.plain_global(StorageClass::Workgroup, u32_, None)?;
    let work = words_buffer(&mut b, WORK, u32_)?;
    let arguments = words_buffer(&mut b, ARGUMENTS, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let invocation = b.builtin(
        crate::spv::BuiltIn::LocalInvocationIndex,
        IoDir::Input,
        u32_,
    )?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let work = f.module().global_value(work)?;
    let arguments = f.module().global_value(arguments)?;
    let status = f.module().global_value(status)?;
    let partial = f.module().global_value(partial)?;
    let total_var = f.module().global_value(total_var)?;
    let invocation = f.module().global_value(invocation)?;
    let lin = f.load(invocation, MemAccess::default())?;
    let zero = k(&mut f, 0)?;
    let one = k(&mut f, 1)?;
    let first = f.binary(BinOp::IEqual, lin, zero)?;

    let objects = load_at(&mut f, work, work::OBJECT_GROUPS)?;
    let slot_words = load_at(&mut f, work, work::SLOT_WORDS)?;
    let group_words = load_at(&mut f, work, work::GROUP_WORDS)?;
    let group_vertices = load_at(&mut f, work, work::GROUP_VERTICES)?;
    let pool_words = load_at(&mut f, work, work::POOL_WORDS)?;
    let draw_id = load_at(&mut f, work, work::DRAW_ID)?;
    let row = load_at(&mut f, work, work::ROW)?;
    let argument_word = load_at(&mut f, work, work::ARGUMENTS)?;
    let header = k(&mut f, work::HEADER)?;

    // An indirect draw's grid (slot 0, `O = 1`): Metal's arguments, read by every invocation.
    let indirect = f.binary(BinOp::INotEqual, argument_word, zero)?;
    let at = f.binary(BinOp::ISub, argument_word, one)?;
    let mut grid = Vec::with_capacity(3);
    for i in 0..3 {
        let i = k(&mut f, i)?;
        let w = f.binary(BinOp::IAdd, at, i)?;
        let w = f.select(indirect, w, zero)?;
        let v = contained(&mut f, arguments, w)?;
        grid.push(f.select(indirect, v, zero)?);
    }
    let argued = bounded_groups(&mut f, [grid[0], grid[1], grid[2]])?;
    let held = f.binary(BinOp::INotEqual, argued, zero)?;
    let write_grid = f.binary(BinOp::LogicalAnd, indirect, first)?;
    when(&mut f, write_grid, move |f| {
        let zero = k(f, 0)?;
        for (i, v) in grid.into_iter().enumerate() {
            let v = f.select(held, v, zero)?;
            let at = k(f, work::HEADER + i as u32)?;
            store(f, work, at, v)?;
        }
        Ok(())
    })?;
    // The mesh threadgroups object threadgroup `o` launches.
    let groups = move |f: &mut FunctionBuilder<'_>, o: Value| -> BResult<Value> {
        let into = f.binary(BinOp::IMul, o, slot_words)?;
        let slot = f.binary(BinOp::IAdd, header, into)?;
        let mut g = [slot; 3];
        for (i, v) in g.iter_mut().enumerate() {
            let off = k(f, i as u32)?;
            let w = f.binary(BinOp::IAdd, slot, off)?;
            *v = load(f, work, w)?;
        }
        let set = bounded_groups(f, g)?;
        f.select(indirect, argued, set)
    };

    // Each invocation's run of object threadgroups: `[lin · c, lin · c + c)` within `O`.
    let lanes = k(&mut f, MeshScanModule::LOCAL_SIZE)?;
    let chunk = div_ceil(&mut f, objects, lanes)?;
    let start = f.binary(BinOp::IMul, lin, chunk)?;
    let lo = f.glsl(GlslOp::UMin, &[start, objects])?;
    let end = f.binary(BinOp::IAdd, lo, chunk)?;
    let hi = f.glsl(GlslOp::UMin, &[end, objects])?;
    let sum_var = f.local_var(u32_, None)?;
    let sum = f.var_ptr(sum_var)?;
    f.store(sum, zero, MemAccess::default())?;
    for_range(&mut f, u32_, (lo, hi), move |f, o| {
        let n = groups(f, o)?;
        let s = f.load(sum, MemAccess::default())?;
        let s = saturating_add(f, s, n)?;
        f.store(sum, s, MemAccess::default())
    })?;
    let mine = f.load(sum, MemAccess::default())?;
    let own = f.access_chain(partial, &[lin])?;
    f.store(own, mine, MemAccess::default())?;
    barrier(&mut f)?;
    // The first invocation turns the sums into each run's first mesh threadgroup.
    when(&mut f, first, move |f| {
        let zero = k(f, 0)?;
        let lanes = k(f, MeshScanModule::LOCAL_SIZE)?;
        f.store(sum, zero, MemAccess::default())?;
        for_range(f, u32_, (zero, lanes), move |f, t| {
            let p = f.access_chain(partial, &[t])?;
            let part = f.load(p, MemAccess::default())?;
            let running = f.load(sum, MemAccess::default())?;
            f.store(p, running, MemAccess::default())?;
            let next = saturating_add(f, running, part)?;
            f.store(sum, next, MemAccess::default())
        })?;
        let total = f.load(sum, MemAccess::default())?;
        f.store(total_var, total, MemAccess::default())
    })?;
    barrier(&mut f)?;
    // The offsets of the run's object threadgroups.
    let slots = f.binary(BinOp::IMul, objects, slot_words)?;
    let offsets = f.binary(BinOp::IAdd, header, slots)?;
    let own = f.access_chain(partial, &[lin])?;
    let base = f.load(own, MemAccess::default())?;
    f.store(sum, base, MemAccess::default())?;
    for_range(&mut f, u32_, (lo, hi), move |f, o| {
        let running = f.load(sum, MemAccess::default())?;
        let at = f.binary(BinOp::IAdd, offsets, o)?;
        store(f, work, at, running)?;
        let n = groups(f, o)?;
        let next = saturating_add(f, running, n)?;
        f.store(sum, next, MemAccess::default())
    })?;
    // The first invocation reserves the records and writes the dispatch and the draw.
    when(&mut f, first, move |f| {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let total = f.load(total_var, MemAccess::default())?;
        let at = f.binary(BinOp::IAdd, offsets, objects)?;
        store(f, work, at, total)?;
        let cursor_at = k(f, StatusWord::MeshRecordsCursor.index())?;
        let cursor_ptr = word_ptr(f, status, cursor_at)?;
        let cursor = f.load(cursor_ptr, MemAccess::default())?;
        let used = f.glsl(GlslOp::UMin, &[cursor, pool_words])?;
        let room = f.binary(BinOp::ISub, pool_words, used)?;
        let gw = f.glsl(GlslOp::UMax, &[group_words, one])?;
        let fit = f.binary(BinOp::UDiv, room, gw)?;
        let fits = f.binary(BinOp::ULessThanEqual, total, fit)?;
        // The draw's vertices fit a `u32` where the product's high word is 0
        // (`replay::checked_mul`, whose docs say why).
        let gv = f.glsl(GlslOp::UMax, &[group_vertices, one])?;
        let (_, past) = checked_mul(f, total, gv, zero)?;
        let drawable = f.unary(UnOp::LogicalNot, past)?;
        let ok = f.binary(BinOp::LogicalAnd, fits, drawable)?;
        let t = f.select(ok, total, zero)?;
        let words = f.binary(BinOp::IMul, t, group_words)?;
        let next = f.binary(BinOp::IAdd, used, words)?;
        let cursor_after = f.select(ok, next, cursor)?;
        f.store(cursor_ptr, cursor_after, MemAccess::default())?;
        let records_base = f.select(ok, used, zero)?;
        let row = f.glsl(GlslOp::UMax, &[row, one])?;
        let x = f.glsl(GlslOp::UMin, &[t, row])?;
        let y = div_ceil(f, t, row)?;
        let vertices = f.binary(BinOp::IMul, t, group_vertices)?;
        for (at, v) in [
            (work::MESH_GROUPS, t),
            (work::RECORDS_BASE, records_base),
            (work::DISPATCH, x),
            (work::DISPATCH + 1, y),
            (work::DISPATCH + 2, one),
            (work::DRAW, vertices),
            (work::DRAW + 1, one),
            (work::DRAW + 2, zero),
            (work::DRAW + 3, zero),
        ] {
            store_at(f, work, at, v)?;
        }
        // A draw the pool cannot hold, or whose vertices pass a `u32`, draws nothing and says
        // so; the command buffer's first such draw is named.
        let over = f.unary(UnOp::LogicalNot, ok)?;
        when(f, over, move |f| {
            let zero = k(f, 0)?;
            let word = k(f, StatusWord::Status.index())?;
            let bit = k(f, StatusBits::MESH_RECORDS_OVERFLOW.0)?;
            let p = word_ptr(f, status, word)?;
            let old = f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
            let either = k(f, StatusBits::RECORDS_FAULTS.0)?;
            let had = f.binary(BinOp::BitwiseAnd, old, either)?;
            let new = f.binary(BinOp::IEqual, had, zero)?;
            when(f, new, move |f| {
                for (w, v) in [
                    (StatusWord::MeshFaultDraw, draw_id),
                    (StatusWord::MeshFaultObjects, objects),
                    (StatusWord::MeshFaultGroups, total),
                ] {
                    store_at(f, status, w.index(), v)?;
                }
                Ok(())
            })
        })
    })?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: MeshScanModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the scan's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(MeshScanModule::SET),
        Symbol::Binding(s) if *s == WORK => Some(MeshScanModule::WORK_BINDING),
        Symbol::Binding(s) if *s == ARGUMENTS => Some(MeshScanModule::ARGUMENTS_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(MeshScanModule::STATUS_BINDING),
        _ => None,
    }
}

/// The mesh scan module of `device`: built with the typed builder, verified, serialized, and
/// numbered with the fixed bindings of [`MeshScanModule`].
pub fn mesh_scan_module(device: &Device) -> Result<MeshScanModule, Error> {
    let (spirv, requirements) =
        finished(device, scan_typed_module()?, binding_of, "the mesh scan")?;
    Ok(MeshScanModule {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests;
