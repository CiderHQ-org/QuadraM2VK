//! The tessellation scan of an emulated tessellation draw: a draw of patches of a pipeline whose
//! post-tessellation vertex function runs as a plain vertex function on a device without
//! `tessellationShader` (`GraphicsState::emulated_tessellation`).
//!
//! One workgroup of [`TessellationScanModule::LOCAL_SIZE`] invocations reads the draw's work
//! buffer (`SyntheticBinding::TessellationWork`, `lower::tessellation::work`) and the encoder's
//! factor buffer, and writes the draw's records (`SyntheticBinding::TessellationRecords`), one per
//! instance and patch (`r = instance · patchCount + patch`), and its chunks' draws:
//!
//! 1. The draw's `patchCount`, `instanceCount`, `patchStart` and `baseInstance` are the header's
//!    first four words: a direct draw's the binder's, an indirect draw's Metal's
//!    `MTLDrawPatchIndirectArguments`, which the consumer copies there before the scan.
//! 2. `R = patchCount · instanceCount` records. Records in the command buffer's records pool
//!    (`work::POOL_WORDS` nonzero) are reserved from its cursor
//!    ([`StatusWord::MeshRecordsCursor`], which the mesh scan shares); a draw whose records the
//!    rest of the pool cannot hold, or whose `R` passes a `u32`, writes none and draws nothing,
//!    and the scan sets [`StatusBits::TESSELLATION_RECORDS_OVERFLOW`]: the command buffer's first
//!    records fault of either kind (mesh or tessellation) names its draw in the fault words, a
//!    tessellation draw with 0 in the object threadgroups word (a mesh draw's is at least 1) and
//!    `R` (saturated) in the mesh threadgroups word.
//! 3. In parallel over the records, each patch's factors, read at the native control stage's
//!    addressing (`link::tessellation::control`: the position `patchStart + patch`, the instance
//!    `instance + baseInstance`, from any even byte past the residual bind offset), are processed
//!    (the tessellator's `process`), and their triangles counted (`triangle_count`).
//! 4. Each record's first triangle is the exclusive prefix of the counts over `r`, held modulo
//!    `2^32` (the prefix itself is carried in two words); each chunk of
//!    `work::CHUNK_TRIANGLES` triangles names the record that holds its first triangle, and its
//!    `VkDrawIndirectCommand` draws its triangles' vertices (`vertexCount` 0 past the draw's).
//! 5. One invocation folds the tessellator's `stream_step` over the records in order: the batch
//!    fill, 0 at the draw's first record, carried from each record to the next, and each record's
//!    stream state the tile word its step gives at the fill before it.
//!
//! Invocations share only workgroup memory (the partial sums, `R` and the records' base), behind
//! workgroup barriers; each record and each chunk's first record is written by one invocation, and
//! the fold recomputes the factors it needs from the factor buffer, so no invocation reads a
//! storage word another wrote.
//!
//! **Bindings** (set 0, storage buffers): [`TessellationScanModule::WORK_BINDING`] (the draw's
//! work buffer, whole), `RECORDS_BINDING` (the draw's records: a region of its own, or the
//! command buffer's records pool), `STATUS_BINDING` (the command buffer's `ExecutionStatus`
//! buffer) and `FACTORS_BINDING` (the encoder's factor buffer from its bind offset aligned down,
//! or any 16 bytes, which cull every patch, without one): four, Vulkan's least
//! `maxPerStageDescriptorStorageBuffers`, so every device binds them. Every read of the factors is
//! contained.
//!
//! **Synchronization** (consumer): the factors' writes and the upload of the work buffer →
//! `SHADER_READ` in `COMPUTE_SHADER` before the scan; for an indirect draw the arguments' writes →
//! the copy of their 16 bytes into the work buffer's first four words, and the copy →
//! `SHADER_READ` in `COMPUTE_SHADER` before the scan; scans one after another (each reads and
//! writes the pool's cursor); the scan's writes → `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` and
//! `SHADER_READ` in `VERTEX_SHADER` before the draw.

use std::sync::Arc;

use crate::bind::index_bounds::finished;
use crate::bind::mesh_scan::{
    barrier, div_ceil, for_range, k, load, load_at, store, store_at, when, word_ptr, words_buffer,
};
use crate::bind::replay::checked_mul;
use crate::bind::status::{StatusBits, StatusWord};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::lower::tessellation::factors::{factor_bytes, factor_halves};
use crate::lower::tessellation::tessellator::{self, Facts};
use crate::lower::tessellation::work;
use crate::metal::{
    TessellationFactorStepFunction as Step, TessellationPartitionMode as Partition, Winding,
};
use crate::pipeline::interface::PatchDomain;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, GlslOp, IoDir, MemAccess,
    Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};

/// The pipeline state one tessellation scan bakes in: the tessellator's facts and the factors'
/// step function (`GraphicsState::emulated_tessellation`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationScanKernel {
    pub domain: PatchDomain,
    pub partition_mode: Partition,
    pub output_winding_order: Winding,
    pub max_tessellation_factor: u32,
    pub factor_step_function: Step,
    pub factor_scale_enabled: bool,
}

impl TessellationScanKernel {
    fn facts(&self) -> Facts {
        Facts {
            domain: self.domain,
            partition: self.partition_mode,
            winding: self.output_winding_order,
            max_factor: self.max_tessellation_factor,
            scale_enabled: self.factor_scale_enabled,
        }
    }

    /// The most triangles one patch of this pipeline draws, at every factor its maximum: the
    /// bound the binder and the consumer size a draw's chunks by (`work::chunks`).
    pub fn max_patch_triangles(&self) -> u32 {
        tessellator::max_triangle_count(&self.facts())
    }
}

/// The crate-owned tessellation scan module of one device and one [`TessellationScanKernel`],
/// validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TessellationScanModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl TessellationScanModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The draw's work buffer, a storage buffer.
    pub const WORK_BINDING: u32 = 0;
    /// The draw's records, a storage buffer.
    pub const RECORDS_BINDING: u32 = 1;
    /// The command buffer's `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 2;
    /// The encoder's factor buffer, a storage buffer.
    pub const FACTORS_BINDING: u32 = 3;
    /// Invocations of the one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 64;
}

const fn kernel_buffer(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// The work buffer's binding symbol in the typed module.
pub const WORK: ResourceSlot = kernel_buffer(0);
/// The records' binding symbol.
pub const RECORDS: ResourceSlot = kernel_buffer(1);
/// The factor buffer's binding symbol.
pub const FACTORS: ResourceSlot = kernel_buffer(2);
/// The status buffer's binding symbol.
pub const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
};

/// `(lo, hi) + x` with the carry into `hi`.
fn add64(
    f: &mut FunctionBuilder<'_>,
    (lo, hi): (Value, Value),
    x: Value,
) -> BResult<(Value, Value)> {
    let sum = f.binary(BinOp::IAdd, lo, x)?;
    let wrapped = f.binary(BinOp::ULessThan, sum, lo)?;
    let one = k(f, 1)?;
    let zero = k(f, 0)?;
    let carry = f.select(wrapped, one, zero)?;
    let hi = f.binary(BinOp::IAdd, hi, carry)?;
    Ok((sum, hi))
}

/// `(lo, hi) + (a, b)`.
fn add64_pair(
    f: &mut FunctionBuilder<'_>,
    x: (Value, Value),
    (a, b): (Value, Value),
) -> BResult<(Value, Value)> {
    let (lo, hi) = add64(f, x, a)?;
    let hi = f.binary(BinOp::IAdd, hi, b)?;
    Ok((lo, hi))
}

/// The chunk of the triangle `(lo, hi)`: `(hi << 2) | (lo >> 30)`.
fn chunk_of(f: &mut FunctionBuilder<'_>, (lo, hi): (Value, Value)) -> BResult<Value> {
    let two = k(f, 32 - work::CHUNK_SHIFT)?;
    let high = f.binary(BinOp::ShiftLeftLogical, hi, two)?;
    let shift = k(f, work::CHUNK_SHIFT)?;
    let low = f.binary(BinOp::ShiftRightLogical, lo, shift)?;
    f.binary(BinOp::BitwiseOr, high, low)
}

/// What every invocation reads of the draw.
#[derive(Clone, Copy)]
struct Draw {
    patch_count: Value,
    patch_start: Value,
    base_instance: Value,
    factor_offset: Value,
    factor_stride: Value,
    scale: Value,
}

/// The processed factors of record `r` of `d`: its patch's factors read at the native control
/// stage's addressing, processed.
fn processed(
    f: &mut FunctionBuilder<'_>,
    kernel: &TessellationScanKernel,
    factors: GlobalId,
    d: Draw,
    r: Value,
) -> BResult<[Value; 6]> {
    let instance = f.binary(BinOp::UDiv, r, d.patch_count)?;
    let patch = f.binary(BinOp::UMod, r, d.patch_count)?;
    let position = f.binary(BinOp::IAdd, d.patch_start, patch)?;
    let instance = f.binary(BinOp::IAdd, instance, d.base_instance)?;
    let mut address = d.factor_offset;
    if matches!(
        kernel.factor_step_function,
        Step::PerPatch | Step::PerPatchAndPerInstance
    ) {
        let size = k(f, factor_bytes(kernel.domain))?;
        let at = f.binary(BinOp::IMul, position, size)?;
        address = f.binary(BinOp::IAdd, address, at)?;
    }
    if matches!(
        kernel.factor_step_function,
        Step::PerInstance | Step::PerPatchAndPerInstance
    ) {
        let at = f.binary(BinOp::IMul, instance, d.factor_stride)?;
        address = f.binary(BinOp::IAdd, address, at)?;
    }
    let facts = kernel.facts();
    let (edges, insides) = facts.counts();
    let halves = factor_halves(f, factors, address, edges + insides)?;
    tessellator::emit_process(f, &facts, &halves, d.scale)
}

/// The scan of `kernel` as a typed module, before serialization: the module
/// [`tessellation_scan_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols [`WORK`], [`RECORDS`], [`STATUS`] and [`FACTORS`].
pub fn scan_typed_module(kernel: &TessellationScanKernel) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let lanes = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(TessellationScanModule::LOCAL_SIZE)),
    })?;
    let partial_ty = b.ty(Type::Array {
        elem: u32_,
        len: lanes,
        stride: None,
    })?;
    let partial_lo = b.plain_global(StorageClass::Workgroup, partial_ty, None)?;
    let partial_hi = b.plain_global(StorageClass::Workgroup, partial_ty, None)?;
    let four = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(4),
    })?;
    let shared_ty = b.ty(Type::Array {
        elem: u32_,
        len: four,
        stride: None,
    })?;
    // `R`, the records' base, and the total triangles' two words.
    let shared = b.plain_global(StorageClass::Workgroup, shared_ty, None)?;
    let work_buf = words_buffer(&mut b, WORK, u32_)?;
    let records = words_buffer(&mut b, RECORDS, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let factors = words_buffer(&mut b, FACTORS, u32_)?;
    let invocation = b.builtin(
        crate::spv::BuiltIn::LocalInvocationIndex,
        IoDir::Input,
        u32_,
    )?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let work_buf = f.module().global_value(work_buf)?;
    let records = f.module().global_value(records)?;
    let status = f.module().global_value(status)?;
    let partial_lo = f.module().global_value(partial_lo)?;
    let partial_hi = f.module().global_value(partial_hi)?;
    let shared = f.module().global_value(shared)?;
    let invocation = f.module().global_value(invocation)?;
    let lin = f.load(invocation, MemAccess::default())?;
    let zero = k(&mut f, 0)?;
    let first = f.binary(BinOp::IEqual, lin, zero)?;
    let shared_at = |f: &mut FunctionBuilder<'_>, i: u32| -> BResult<Value> {
        let i = k(f, i)?;
        f.access_chain(shared, &[i])
    };

    // 1. The draw's arguments, read by every invocation.
    let patch_count = load_at(&mut f, work_buf, work::PATCH_COUNT)?;
    let instance_count = load_at(&mut f, work_buf, work::INSTANCE_COUNT)?;
    let patch_start = load_at(&mut f, work_buf, work::PATCH_START)?;
    let base_instance = load_at(&mut f, work_buf, work::BASE_INSTANCE)?;
    let draw_id = load_at(&mut f, work_buf, work::DRAW_ID)?;
    let pool_words = load_at(&mut f, work_buf, work::POOL_WORDS)?;
    let chunks = load_at(&mut f, work_buf, work::CHUNKS)?;
    let factor_offset = load_at(&mut f, work_buf, work::FACTOR_OFFSET)?;
    let factor_stride = load_at(&mut f, work_buf, work::FACTOR_STRIDE)?;
    let scale_bits = load_at(&mut f, work_buf, work::SCALE)?;
    let scale = f.convert(ConvOp::Bitcast, scale_bits, f32_)?;
    let draw = Draw {
        patch_count,
        patch_start,
        base_instance,
        factor_offset,
        factor_stride,
        scale,
    };

    // 2. The records: `R` within a `u32` and, in the pool, its room; else none, and the fault.
    when(&mut f, first, move |f| {
        let zero = k(f, 0)?;
        let max = k(f, u32::MAX)?;
        // The records' count, `u32::MAX` past a `u32`, tested by the product's high word
        // (`replay::checked_mul`, whose docs say why).
        let (product, over) = checked_mul(f, patch_count, instance_count, zero)?;
        let fits_u32 = f.unary(UnOp::LogicalNot, over)?;
        let r = f.select(fits_u32, product, max)?;
        let pooled = f.binary(BinOp::INotEqual, pool_words, zero)?;
        let cursor_at = k(f, StatusWord::MeshRecordsCursor.index())?;
        let cursor_ptr = word_ptr(f, status, cursor_at)?;
        let cursor = f.load(cursor_ptr, MemAccess::default())?;
        let used = f.glsl(GlslOp::UMin, &[cursor, pool_words])?;
        let left = f.binary(BinOp::ISub, pool_words, used)?;
        let record_words = k(f, work::RECORD_WORDS)?;
        let fit = f.binary(BinOp::UDiv, left, record_words)?;
        let in_pool = f.binary(BinOp::ULessThanEqual, r, fit)?;
        let not_pooled = f.unary(UnOp::LogicalNot, pooled)?;
        let room_ok = f.binary(BinOp::LogicalOr, not_pooled, in_pool)?;
        let ok = f.binary(BinOp::LogicalAnd, fits_u32, room_ok)?;
        let kept = f.select(ok, r, zero)?;
        let reserve = f.binary(BinOp::LogicalAnd, ok, pooled)?;
        let words = f.binary(BinOp::IMul, kept, record_words)?;
        let next = f.binary(BinOp::IAdd, used, words)?;
        let cursor_after = f.select(reserve, next, cursor)?;
        f.store(cursor_ptr, cursor_after, MemAccess::default())?;
        let base = f.select(reserve, used, zero)?;
        store_at(f, work_buf, work::RECORDS, kept)?;
        store_at(f, work_buf, work::RECORDS_BASE, base)?;
        let p0 = shared_at(f, 0)?;
        f.store(p0, kept, MemAccess::default())?;
        let p1 = shared_at(f, 1)?;
        f.store(p1, base, MemAccess::default())?;
        let over = f.unary(UnOp::LogicalNot, ok)?;
        when(f, over, move |f| {
            let zero = k(f, 0)?;
            let word = k(f, StatusWord::Status.index())?;
            let bit = k(f, StatusBits::TESSELLATION_RECORDS_OVERFLOW.0)?;
            let p = word_ptr(f, status, word)?;
            let old = f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
            let either = k(f, StatusBits::RECORDS_FAULTS.0)?;
            let had = f.binary(BinOp::BitwiseAnd, old, either)?;
            let new = f.binary(BinOp::IEqual, had, zero)?;
            when(f, new, move |f| {
                let zero = k(f, 0)?;
                for (w, v) in [
                    (StatusWord::MeshFaultDraw, draw_id),
                    (StatusWord::MeshFaultObjects, zero),
                    (StatusWord::MeshFaultGroups, r),
                ] {
                    store_at(f, status, w.index(), v)?;
                }
                Ok(())
            })
        })
    })?;
    barrier(&mut f)?;
    let p0 = shared_at(&mut f, 0)?;
    let total_records = f.load(p0, MemAccess::default())?;
    let p1 = shared_at(&mut f, 1)?;
    let records_base = f.load(p1, MemAccess::default())?;
    let record_word = move |f: &mut FunctionBuilder<'_>, r: Value, word: u32| -> BResult<Value> {
        let rw = k(f, work::RECORD_WORDS)?;
        let at = f.binary(BinOp::IMul, r, rw)?;
        let at = f.binary(BinOp::IAdd, records_base, at)?;
        let w = k(f, word)?;
        f.binary(BinOp::IAdd, at, w)
    };

    // 3. Each invocation's run of records, `[lin · c, lin · c + c)` within `R`: processed and
    // counted, each count held in its record's first-triangle word until 4.
    let lanes = k(&mut f, TessellationScanModule::LOCAL_SIZE)?;
    let chunk = div_ceil(&mut f, total_records, lanes)?;
    let start = f.binary(BinOp::IMul, lin, chunk)?;
    let lo = f.glsl(GlslOp::UMin, &[start, total_records])?;
    let end = f.binary(BinOp::IAdd, lo, chunk)?;
    let hi = f.glsl(GlslOp::UMin, &[end, total_records])?;
    let sum_lo_var = f.local_var(u32_, None)?;
    let sum_lo = f.var_ptr(sum_lo_var)?;
    let sum_hi_var = f.local_var(u32_, None)?;
    let sum_hi = f.var_ptr(sum_hi_var)?;
    f.store(sum_lo, zero, MemAccess::default())?;
    f.store(sum_hi, zero, MemAccess::default())?;
    let factors_ = factors;
    let kernel_ = *kernel;
    for_range(&mut f, u32_, (lo, hi), move |f, r| {
        let p = processed(f, &kernel_, factors_, draw, r)?;
        let n = tessellator::emit_triangle_count(f, &kernel_.facts(), &p)?;
        for (i, v) in p.into_iter().enumerate() {
            let at = record_word(f, r, i as u32)?;
            store(f, records, at, v)?;
        }
        let at = record_word(f, r, work::FIRST_TRIANGLE)?;
        store(f, records, at, n)?;
        let s = (
            f.load(sum_lo, MemAccess::default())?,
            f.load(sum_hi, MemAccess::default())?,
        );
        let (l, h) = add64(f, s, n)?;
        f.store(sum_lo, l, MemAccess::default())?;
        f.store(sum_hi, h, MemAccess::default())
    })?;
    for (partial, sum) in [(partial_lo, sum_lo), (partial_hi, sum_hi)] {
        let mine = f.load(sum, MemAccess::default())?;
        let own = f.access_chain(partial, &[lin])?;
        f.store(own, mine, MemAccess::default())?;
    }
    barrier(&mut f)?;
    // The first invocation turns the sums into each run's first triangle, and the total.
    when(&mut f, first, move |f| {
        let zero = k(f, 0)?;
        let lanes = k(f, TessellationScanModule::LOCAL_SIZE)?;
        f.store(sum_lo, zero, MemAccess::default())?;
        f.store(sum_hi, zero, MemAccess::default())?;
        for_range(f, u32_, (zero, lanes), move |f, t| {
            let pl = f.access_chain(partial_lo, &[t])?;
            let ph = f.access_chain(partial_hi, &[t])?;
            let part = (
                f.load(pl, MemAccess::default())?,
                f.load(ph, MemAccess::default())?,
            );
            let running = (
                f.load(sum_lo, MemAccess::default())?,
                f.load(sum_hi, MemAccess::default())?,
            );
            f.store(pl, running.0, MemAccess::default())?;
            f.store(ph, running.1, MemAccess::default())?;
            let (l, h) = add64_pair(f, running, part)?;
            f.store(sum_lo, l, MemAccess::default())?;
            f.store(sum_hi, h, MemAccess::default())
        })?;
        let total_lo = f.load(sum_lo, MemAccess::default())?;
        let total_hi = f.load(sum_hi, MemAccess::default())?;
        let p2 = shared_at(f, 2)?;
        f.store(p2, total_lo, MemAccess::default())?;
        let p3 = shared_at(f, 3)?;
        f.store(p3, total_hi, MemAccess::default())
    })?;
    barrier(&mut f)?;

    // 4. Each record's first triangle, and each chunk's first record.
    let slot_word = |f: &mut FunctionBuilder<'_>, chunk: Value, word: u32| -> BResult<Value> {
        let sw = k(f, work::SLOT_WORDS)?;
        let at = f.binary(BinOp::IMul, chunk, sw)?;
        let base = k(f, work::HEADER + word)?;
        f.binary(BinOp::IAdd, base, at)
    };
    for (partial, sum) in [(partial_lo, sum_lo), (partial_hi, sum_hi)] {
        let own = f.access_chain(partial, &[lin])?;
        let base = f.load(own, MemAccess::default())?;
        f.store(sum, base, MemAccess::default())?;
    }
    for_range(&mut f, u32_, (lo, hi), move |f, r| {
        let zero = k(f, 0)?;
        let at = record_word(f, r, work::FIRST_TRIANGLE)?;
        let n = load(f, records, at)?;
        let running = (
            f.load(sum_lo, MemAccess::default())?,
            f.load(sum_hi, MemAccess::default())?,
        );
        store(f, records, at, running.0)?;
        let counts = f.binary(BinOp::INotEqual, n, zero)?;
        when(f, counts, move |f| {
            let zero = k(f, 0)?;
            let one = k(f, 1)?;
            let k0 = chunk_of(f, running)?;
            let mask = k(f, (work::CHUNK_TRIANGLES - 1) as u32)?;
            let within = f.binary(BinOp::BitwiseAnd, running.0, mask)?;
            let starts = f.binary(BinOp::IEqual, within, zero)?;
            let below = f.binary(BinOp::ULessThan, k0, chunks)?;
            let starts = f.binary(BinOp::LogicalAnd, starts, below)?;
            when(f, starts, move |f| {
                let at = slot_word(f, k0, work::SLOT_FIRST_RECORD)?;
                store(f, work_buf, at, r)
            })?;
            let last = f.binary(BinOp::ISub, n, one)?;
            let end = add64(f, running, last)?;
            let k1 = chunk_of(f, end)?;
            let crosses = f.binary(BinOp::INotEqual, k1, k0)?;
            let below = f.binary(BinOp::ULessThan, k1, chunks)?;
            let crosses = f.binary(BinOp::LogicalAnd, crosses, below)?;
            when(f, crosses, move |f| {
                let at = slot_word(f, k1, work::SLOT_FIRST_RECORD)?;
                store(f, work_buf, at, r)
            })
        })?;
        let (l, h) = add64(f, running, n)?;
        f.store(sum_lo, l, MemAccess::default())?;
        f.store(sum_hi, h, MemAccess::default())
    })?;

    // Each chunk's draw, and 5. the stream state, by the first invocation.
    let factors_ = factors;
    let kernel_ = *kernel;
    when(&mut f, first, move |f| {
        let zero = k(f, 0)?;
        let p2 = shared_at(f, 2)?;
        let total_lo = f.load(p2, MemAccess::default())?;
        let p3 = shared_at(f, 3)?;
        let total_hi = f.load(p3, MemAccess::default())?;
        // Chunk `c`'s triangles: `min(C, T - c · C)`, 0 at or past `T`.
        for_range(f, u32_, (zero, chunks), move |f, c| {
            let zero = k(f, 0)?;
            let one = k(f, 1)?;
            let shift = k(f, work::CHUNK_SHIFT)?;
            let start_lo = f.binary(BinOp::ShiftLeftLogical, c, shift)?;
            let back = k(f, 32 - work::CHUNK_SHIFT)?;
            let start_hi = f.binary(BinOp::ShiftRightLogical, c, back)?;
            // `T - start` in two words, and whether it is positive.
            let borrow = f.binary(BinOp::ULessThan, total_lo, start_lo)?;
            let rem_lo = f.binary(BinOp::ISub, total_lo, start_lo)?;
            let b = f.select(borrow, one, zero)?;
            let rem_hi = f.binary(BinOp::ISub, total_hi, start_hi)?;
            let rem_hi = f.binary(BinOp::ISub, rem_hi, b)?;
            let hi_above = f.binary(BinOp::UGreaterThan, total_hi, start_hi)?;
            let hi_equal = f.binary(BinOp::IEqual, total_hi, start_hi)?;
            let lo_above = f.binary(BinOp::UGreaterThan, total_lo, start_lo)?;
            let equal_above = f.binary(BinOp::LogicalAnd, hi_equal, lo_above)?;
            let positive = f.binary(BinOp::LogicalOr, hi_above, equal_above)?;
            let whole = f.binary(BinOp::INotEqual, rem_hi, zero)?;
            let c_tri = k(f, work::CHUNK_TRIANGLES as u32)?;
            let past = f.binary(BinOp::UGreaterThanEqual, rem_lo, c_tri)?;
            let full = f.binary(BinOp::LogicalOr, whole, past)?;
            let n = f.select(full, c_tri, rem_lo)?;
            let n = f.select(positive, n, zero)?;
            let three = k(f, 3)?;
            let vertices = f.binary(BinOp::IMul, n, three)?;
            let at = slot_word(f, c, work::SLOT_DRAW)?;
            for (i, v) in [vertices, one, zero, zero].into_iter().enumerate() {
                let i = k(f, i as u32)?;
                let w = f.binary(BinOp::IAdd, at, i)?;
                store(f, work_buf, w, v)?;
            }
            Ok(())
        })?;
        let used_var = f.local_var(u32_, None)?;
        let used = f.var_ptr(used_var)?;
        f.store(used, zero, MemAccess::default())?;
        for_range(f, u32_, (zero, total_records), move |f, r| {
            let before = f.load(used, MemAccess::default())?;
            let p = processed(f, &kernel_, factors_, draw, r)?;
            let (after, tiles) = tessellator::emit_stream_step(f, &kernel_.facts(), before, &p)?;
            let at = record_word(f, r, work::STREAM_STATE)?;
            store(f, records, at, tiles)?;
            f.store(used, after, MemAccess::default())
        })
    })?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: TessellationScanModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the scan's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(TessellationScanModule::SET),
        Symbol::Binding(s) if *s == WORK => Some(TessellationScanModule::WORK_BINDING),
        Symbol::Binding(s) if *s == RECORDS => Some(TessellationScanModule::RECORDS_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(TessellationScanModule::STATUS_BINDING),
        Symbol::Binding(s) if *s == FACTORS => Some(TessellationScanModule::FACTORS_BINDING),
        _ => None,
    }
}

/// The tessellation scan module of `device` for `kernel`: built with the typed builder,
/// verified, serialized, and numbered with the fixed bindings of [`TessellationScanModule`].
pub fn tessellation_scan_module(
    device: &Device,
    kernel: &TessellationScanKernel,
) -> Result<TessellationScanModule, Error> {
    let (spirv, requirements) = finished(
        device,
        scan_typed_module(kernel)?,
        binding_of,
        "the tessellation scan",
    )?;
    Ok(TessellationScanModule {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests;
