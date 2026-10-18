//! The index rewrite pass of an indexed draw whose primitives Vulkan's input assembly does not
//! form as Metal does: one dispatch before the render pass instance reads the draw's Metal indices
//! and writes the primitives Metal draws as a list of 32-bit indices, which the draw then draws as
//! a list over its arguments, with primitive restart off.
//!
//! The one program today is an indexed strip after Apple's `setPrimitiveRestartEnabled:index:`,
//! which restarts at the call's index `k'` alone, where Vulkan restarts at the
//! index type's all-ones index alone: the strip's pieces between restarts, each triangle in
//! Vulkan's strip order with its parity counted from its piece's start (a line strip's segments
//! as pairs), in order, so every primitive keeps its winding, its provoking vertex and its place
//! in `primitive_id`'s count, which Metal counts on across a restart (probe 5's `prim_*`). The
//! all-ones index, where it is not `k'`, is drawn as a vertex. Programs are a closed enum
//! ([`RewriteProgram`]): another way of forming primitives from indices is another arm of the same
//! pass, not another pass.
//!
//! - **Three dispatches** ([`RewritePhase`]) of workgroups of [`IndexRewriteModule::LOCAL_SIZE`]
//!   invocations. The draw's positions are split into one run per invocation of
//!   [`IndexRewritePass::groups`] workgroups. A position completes a primitive where neither it
//!   nor the [`RewriteProgram::vertices`] less one positions before it restarts, which a run
//!   decides from its own positions and the two before it, so the runs are independent but for
//!   two sums over the runs before each: its primitives (its first primitive's place in the
//!   output) and the start of the piece it begins in (the position after the last restart before
//!   it, from which a triangle's parity counts). [`RewritePhase::Count`] sums each group's runs;
//!   [`RewritePhase::Place`], one workgroup, places each group, reserves the output and writes
//!   the draw; [`RewritePhase::Write`] places each run within its group and writes its
//!   primitives. Each run walks a few dozen positions of a million-position draw, so the walks
//!   spread over the GPU: one workgroup walking every position (the first pass) took 56 ms for a
//!   million positions on an M4.
//! - **Positions** are Metal's: from the arguments' first index, each position inside the
//!   buffer's whole indices from the offset at its index, each past them at index 0, as the fetch
//!   reads it, restart included. Past the buffer every position reads 0, so the walk stops
//!   [`RewriteProgram::vertices`] positions past the last whole index inside; the rest of the
//!   draw's positions either all restart (`k'` = 0: no primitive) or each completes one primitive
//!   of index 0, written without a walk.
//! - **The output** is either the draw's own region ([`RewriteOutput::Owned`], a direct draw:
//!   its capacity [`RewriteProgram::capacity`] of its count, from index 0), or a reservation in
//!   the command buffer's index pool ([`RewriteOutput::Pooled`], an indirect draw, whose count
//!   only the GPU knows), shared with the index bounds pass (`index_bounds.rs`): the pool's
//!   counter in its first word, bytes reserved from its index region, which starts
//!   [`IndexRewritePass::POOL_HEADER_BYTES`] in. A draw whose primitives its output cannot hold
//!   draws nothing and sets [`StatusBits::INDEX_POOL_OVERFLOW`], which the consumer reports as the
//!   command buffer's error. Nothing is drawn short silently. Of a pooled output, the count
//!   dispatch reads nothing, the place dispatch reads and writes only the counter, and the write
//!   dispatch writes only index bytes, so a consumer may bind the counter for the first two and
//!   another buffer of the pool's bytes for the write, as the index bounds pass's plan and fill
//!   may.
//!
//! **The rewrite words** (one `u32` array of [`IndexRewritePass::BYTES`], a storage buffer, and
//! the indirect source of the draw):
//!
//! | Words | Written by | Contents |
//! |---|---|---|
//! | 0 to 7 | the consumer | [`IndexRewritePass::parameters`]: the arguments' first word in the bound arguments range plus 1 (0 for a direct draw), the index offset's byte in the bound indices range, the whole indices from the offset (saturated to `u32`), 1 for 32-bit indices, `k'`, the output's capacity in indices (owned) or the pool's index bytes (pooled), 1 where pooled, the groups |
//! | 8 to 12 | the consumer | a direct draw's Metal arguments (`MTLDrawIndexedPrimitivesIndirectArguments`: count, instances, first index 0, base vertex, base instance); 0 for an indirect draw, whose arguments are read from the Metal buffer |
//! | 16 to 20 | the place dispatch | the draw's `VkDrawIndexedIndirectCommand` ([`IndexRewritePass::DRAW_OFFSET`]) |
//! | 21 | the place dispatch | the primitives written |
//! | 22 to 25 | the place dispatch | the output's first word, 1 where the draw is written, the primitives before those past the walk, and those past it |
//! | 32 on | the count and place dispatches | two words per group: its primitives, then those before it; the start of its last piece, then of the piece it begins in (0 for none) |
//!
//! **Bindings** (set 0, storage buffers): [`IndexRewriteModule::WORDS_BINDING`],
//! `ARGUMENTS_BINDING` ([`IndexRewritePass::arguments_range`], or any 4 bytes for a direct draw),
//! `INDICES_BINDING` ([`IndexRewritePass::indices_range`], or any 4 bytes where it is `None`),
//! `OUTPUT_BINDING` (the draw's region, or the whole pool from its counter) and `STATUS_BINDING`
//! (the `ExecutionStatus` buffer). Every read of the arguments and indices is contained (an index
//! past the bound range reads 0), and every write lies inside the output.
//!
//! **Synchronization** (consumer): the arguments' and indices' writes and the upload of the
//! words → `SHADER_READ` in `COMPUTE_SHADER` before the count dispatch; each dispatch's
//! `SHADER_WRITE` → `SHADER_READ` and `SHADER_WRITE` in `COMPUTE_SHADER` before the next;
//! passes and index bounds plans one after another (each reads and writes the pool's counter);
//! the write dispatch's writes (and the place dispatch's) → `INDEX_READ` in `INDEX_INPUT` and
//! `INDIRECT_COMMAND_READ` in `DRAW_INDIRECT` before the draw.

use std::sync::Arc;

use crate::bind::index_bounds::{finished, word_range, IndexBoundsPass};
use crate::bind::instantiate::{BufferRange, IndirectRef};
use crate::bind::replay::checked_mul;
use crate::bind::status::{StatusBits, StatusWord};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ExecMode, ExecModel, GlslOp, IoDir,
    LayoutRule, MemAccess, MemoryOrder, Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};

/// How the pass forms a draw's primitives from its indices: one module each.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RewriteProgram {
    /// A line strip restarting at `k'` alone: each segment of a piece as a pair.
    LineStrip,
    /// A triangle strip restarting at `k'` alone: each triangle of a piece in Vulkan's strip
    /// order, its parity counted from the piece's start.
    TriangleStrip,
}

impl RewriteProgram {
    /// The indices of one primitive of the output list.
    pub fn vertices(self) -> u32 {
        match self {
            RewriteProgram::LineStrip => 2,
            RewriteProgram::TriangleStrip => 3,
        }
    }

    /// The most indices the program writes for a draw of `count` positions: a primitive at each
    /// position from the program's [`vertices`](Self::vertices)th on.
    pub fn capacity(self, count: u64) -> u64 {
        let v = u64::from(self.vertices());
        count.saturating_sub(v - 1) * v
    }
}

/// One of a pass's three dispatches, in the order the consumer records them (the module docs
/// say what each does): one module each, of each program.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RewritePhase {
    /// Over [`IndexRewritePass::groups`] workgroups: each sums its runs' primitives and finds
    /// the start of its last piece.
    Count,
    /// One workgroup: places each group, reserves the output, and writes the draw.
    Place,
    /// Over the groups again: each places its runs and writes their primitives.
    Write,
}

impl RewritePhase {
    /// The phases, in the order the consumer dispatches them.
    pub const ALL: [RewritePhase; 3] = [
        RewritePhase::Count,
        RewritePhase::Place,
        RewritePhase::Write,
    ];

    /// The workgroups of the phase's dispatch for `pass`.
    pub fn workgroups(self, pass: &IndexRewritePass) -> u32 {
        match self {
            RewritePhase::Place => 1,
            RewritePhase::Count | RewritePhase::Write => pass.groups,
        }
    }
}

/// Where a draw's Metal arguments are.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteSource {
    /// A direct draw: its index count, instances, base vertex and base instance.
    Direct {
        /// Metal's index count.
        count: u32,
        /// Metal's instance count.
        instances: u32,
        /// Metal's base vertex.
        base_vertex: i32,
        /// Metal's base instance.
        base_instance: u32,
    },
    /// An indirect draw: Metal's `MTLDrawIndexedPrimitivesIndirectArguments` at `arguments`.
    Indirect {
        /// The arguments' place in the Metal indirect buffer.
        arguments: IndirectRef,
    },
}

/// Where the pass writes a draw's indices.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RewriteOutput {
    /// A region of the draw's own, of `capacity` indices, written from index 0.
    Owned {
        /// The region's indices.
        capacity: u32,
    },
    /// A reservation in the command buffer's index pool of `pool_bytes` index bytes (a multiple
    /// of 4) behind its header.
    Pooled {
        /// The pool's index bytes.
        pool_bytes: u32,
    },
}

/// One draw's index rewrite pass: what the consumer binds and uploads (the module docs say how).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexRewritePass {
    /// The range of the Metal indirect buffer to bind at
    /// [`IndexRewriteModule::ARGUMENTS_BINDING`]: from the arguments' offset aligned down to
    /// `minStorageBufferOffsetAlignment` (and to 4) through their 20 bytes; `None` for a direct
    /// draw.
    pub arguments_range: Option<BufferRange>,
    /// The range of the Metal index buffer to bind at [`IndexRewriteModule::INDICES_BINDING`]:
    /// from the index offset aligned down likewise through the whole indices inside the buffer,
    /// to a whole word (`index_bounds::word_range`); `None` when none lies inside it from the
    /// offset.
    pub indices_range: Option<BufferRange>,
    /// The words' first [`IndexRewritePass::PARAMETER_WORDS`] words, which the consumer uploads
    /// before the pass.
    pub parameters: [u32; IndexRewritePass::PARAMETER_WORDS],
    /// The workgroups of the count and write dispatches ([`RewritePhase::workgroups`]): one per
    /// [`IndexRewritePass::GROUP_POSITIONS`] positions the walk may take, from 1 to
    /// [`IndexRewritePass::MAX_GROUPS`].
    pub groups: u32,
}

impl IndexRewritePass {
    /// Bytes of the rewrite words: through the groups' words of [`MAX_GROUPS`](Self::MAX_GROUPS).
    pub const BYTES: u64 = 4 * (Output::Groups as u64 + 2 * Self::MAX_GROUPS as u64);
    /// The most workgroups of the count and write dispatches.
    pub const MAX_GROUPS: u32 = 256;
    /// The positions of a walk per workgroup of the count and write dispatches, below which a
    /// draw takes fewer: 32 per invocation.
    pub const GROUP_POSITIONS: u64 = 32 * IndexRewriteModule::LOCAL_SIZE as u64;
    /// The most positions a walk takes past the buffer's whole indices: the most vertices of a
    /// program's primitive ([`RewriteProgram::vertices`]).
    const WALK_PAST: u64 = 3;
    /// The consumer's words at the rewrite words' start: the parameters and a direct draw's
    /// arguments.
    pub const PARAMETER_WORDS: usize = 16;
    /// The draw's `VkDrawIndexedIndirectCommand` in the rewrite words.
    pub const DRAW_OFFSET: u64 = 64;
    /// The index pool's header, as the index bounds pass's: its counter, then reserved bytes.
    pub const POOL_HEADER_BYTES: u64 = IndexBoundsPass::POOL_HEADER_BYTES;
    /// Bytes of Metal's `MTLDrawIndexedPrimitivesIndirectArguments`.
    const ARGUMENT_BYTES: u64 = 20;

    /// The pass of a draw from `source` restarting at `restart` (`k'`), drawing indices of
    /// `index_size` bytes (2 or 4) from `index_offset` of an index buffer whose bytes end at
    /// `index_end` (its length, in the same terms as the offset; its `VkBuffer` holds the bytes
    /// to `index_end` rounded up to 4 there, as the padding does), into `output`, with ranges bound
    /// at offsets aligned to `alignment` (`minStorageBufferOffsetAlignment`) over at most
    /// `max_range` bytes (`maxStorageBufferRange`). `None` when the arguments' offset is not a
    /// multiple of 4, the index size is neither 2 nor 4, the index offset is not a multiple of
    /// it, a pool's bytes are not a multiple of 4, or a range overflows or exceeds `max_range`.
    pub fn new(
        source: RewriteSource,
        restart: u32,
        (index_size, index_offset, index_end): (u64, u64, u64),
        output: RewriteOutput,
        alignment: u64,
        max_range: u32,
    ) -> Option<IndexRewritePass> {
        if !matches!(index_size, 2 | 4) || !index_offset.is_multiple_of(index_size) {
            return None;
        }
        let aligned = |offset: u64, end: u64| word_range(offset, end, alignment, max_range);
        let mut parameters = [0; Self::PARAMETER_WORDS];
        let arguments_range = match source {
            RewriteSource::Direct {
                count,
                instances,
                base_vertex,
                base_instance,
            } => {
                parameters[Parameter::Arguments as usize] = count;
                parameters[Parameter::Arguments as usize + 1] = instances;
                parameters[Parameter::Arguments as usize + 3] = base_vertex as u32;
                parameters[Parameter::Arguments as usize + 4] = base_instance;
                None
            }
            RewriteSource::Indirect { arguments } => {
                if !arguments.offset.is_multiple_of(4) {
                    return None;
                }
                let end = arguments.offset.checked_add(Self::ARGUMENT_BYTES)?;
                let (range, residual) = aligned(arguments.offset, end)?;
                parameters[Parameter::ArgumentsWord as usize] =
                    u32::try_from(residual / 4 + 1).ok()?;
                Some(range)
            }
        };
        let inside = index_end.saturating_sub(index_offset) / index_size;
        // The positions the walk may take: an indirect draw's count is the GPU's.
        let walk = inside.saturating_add(Self::WALK_PAST);
        let walk = match source {
            RewriteSource::Direct { count, .. } => walk.min(u64::from(count)),
            RewriteSource::Indirect { .. } => walk,
        };
        let groups = walk
            .div_ceil(Self::GROUP_POSITIONS)
            .clamp(1, u64::from(Self::MAX_GROUPS)) as u32;
        let (indices_range, indices_residual) = if inside == 0 {
            (None, 0)
        } else {
            let end = index_offset.checked_add(inside.checked_mul(index_size)?)?;
            let (range, residual) = aligned(index_offset, end)?;
            (Some(range), residual)
        };
        let (capacity, pooled) = match output {
            RewriteOutput::Owned { capacity } => (capacity, 0),
            RewriteOutput::Pooled { pool_bytes } => {
                if !pool_bytes.is_multiple_of(4) {
                    return None;
                }
                (pool_bytes, 1)
            }
        };
        parameters[Parameter::IndicesByte as usize] = u32::try_from(indices_residual).ok()?;
        parameters[Parameter::Inside as usize] = u32::try_from(inside).unwrap_or(u32::MAX);
        parameters[Parameter::Wide as usize] = u32::from(index_size == 4);
        parameters[Parameter::Restart as usize] = restart;
        parameters[Parameter::Capacity as usize] = capacity;
        parameters[Parameter::Pooled as usize] = pooled;
        parameters[Parameter::Groups as usize] = groups;
        Some(IndexRewritePass {
            arguments_range,
            indices_range,
            parameters,
            groups,
        })
    }

    /// The parameters as the bytes to upload at the rewrite words' start.
    pub fn parameter_bytes(&self) -> Vec<u8> {
        self.parameters
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }
}

/// The consumer's words, by index into the rewrite words.
#[derive(Clone, Copy)]
enum Parameter {
    ArgumentsWord = 0,
    IndicesByte = 1,
    Inside = 2,
    Wide = 3,
    Restart = 4,
    Capacity = 5,
    Pooled = 6,
    Groups = 7,
    Arguments = 8,
}

/// The pass's words in the rewrite words, by index.
#[derive(Clone, Copy)]
enum Output {
    Draw = 16,
    Primitives = 21,
    /// The output's first word.
    Base = 22,
    /// 1 where the draw is written.
    Written = 23,
    /// The primitives before those past the walk.
    TailAt = 24,
    /// The primitives past the walk.
    Tail = 25,
    /// The groups' words, two per group.
    Groups = 32,
}

/// The crate-owned module of one program on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexRewriteModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl IndexRewriteModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The draw's rewrite words, a storage buffer.
    pub const WORDS_BINDING: u32 = 0;
    /// The Metal indirect buffer, a storage buffer.
    pub const ARGUMENTS_BINDING: u32 = 1;
    /// The Metal index buffer, a storage buffer.
    pub const INDICES_BINDING: u32 = 2;
    /// The draw's output: its region, or the command buffer's index pool, a storage buffer.
    pub const OUTPUT_BINDING: u32 = 3;
    /// The command buffer's `ExecutionStatus` buffer, a storage buffer.
    pub const STATUS_BINDING: u32 = 4;
    /// Invocations of each workgroup, along x: Vulkan's least `maxComputeWorkGroupSize[0]`
    /// and `maxComputeWorkGroupInvocations`, so every device takes it.
    pub const LOCAL_SIZE: u32 = 128;
}

/// The words' binding symbol in the typed module.
pub const WORDS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The arguments' binding symbol ([`IndexRewriteModule::ARGUMENTS_BINDING`]).
pub const ARGUMENTS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The indices' binding symbol ([`IndexRewriteModule::INDICES_BINDING`]).
pub const INDICES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};
/// The output's binding symbol ([`IndexRewriteModule::OUTPUT_BINDING`]).
pub const OUTPUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 3,
};
/// The status buffer's binding symbol ([`IndexRewriteModule::STATUS_BINDING`]).
pub const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
};

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

fn k(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// The pointer to word `index` of `buffer`'s runtime array.
fn word_ptr(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let zero = k(f, 0)?;
    f.access_chain(buffer, &[zero, index])
}

fn load(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value) -> BResult<Value> {
    let p = word_ptr(f, buffer, index)?;
    f.load(p, MemAccess::default())
}

fn load_at(f: &mut FunctionBuilder<'_>, buffer: Value, index: u32) -> BResult<Value> {
    let i = k(f, index)?;
    load(f, buffer, i)
}

fn store(f: &mut FunctionBuilder<'_>, buffer: Value, index: Value, v: Value) -> BResult<()> {
    let p = word_ptr(f, buffer, index)?;
    f.store(p, v, MemAccess::default())
}

fn store_at(f: &mut FunctionBuilder<'_>, buffer: Value, index: u32, v: Value) -> BResult<()> {
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

/// Store `v` at word `index` of `buffer` where it lies inside the buffer's bound range.
fn contained_store(
    f: &mut FunctionBuilder<'_>,
    buffer: Value,
    index: Value,
    v: Value,
) -> BResult<()> {
    let len = f.array_length(buffer, 0)?;
    let inside = f.binary(BinOp::ULessThan, index, len)?;
    when(f, inside, move |f| store(f, buffer, index, v))
}

/// `a + b`, or `u32::MAX` where it wraps.
fn saturating_add(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let sum = f.binary(BinOp::IAdd, a, b)?;
    let wrapped = f.binary(BinOp::ULessThan, sum, a)?;
    let max = k(f, u32::MAX)?;
    f.select(wrapped, max, sum)
}

/// `a - b`, or 0 where `b` is larger.
fn saturating_sub(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let diff = f.binary(BinOp::ISub, a, b)?;
    let over = f.binary(BinOp::UGreaterThan, b, a)?;
    let zero = k(f, 0)?;
    f.select(over, zero, diff)
}

/// `a / b` rounded up (`b` not 0), without wrapping.
fn div_ceil(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let q = f.binary(BinOp::UDiv, a, b)?;
    let r = f.binary(BinOp::UMod, a, b)?;
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let partial = f.binary(BinOp::INotEqual, r, zero)?;
    let extra = f.select(partial, one, zero)?;
    f.binary(BinOp::IAdd, q, extra)
}

/// Run `each(f, i)` for `i` from `lo` below `hi`, `step` apart (`step` not 0).
fn for_range<'a>(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    (lo, hi, step): (Value, Value, Value),
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
        // The next `i`, or past `hi` where it would wrap.
        let next = saturating_add(f, i, step)?;
        f.store(counter, next, MemAccess::default())?;
        each(f, i)
    });
    f.loop_(&mut (), LoopSpec::default(), body)
}

/// Run `then(f)` where `c` holds.
fn when<'a>(
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

/// The workgroup barrier over the runs' words.
fn barrier(f: &mut FunctionBuilder<'_>) -> BResult<()> {
    let sem = Semantics {
        order: MemoryOrder::AcquireRelease,
        workgroup_memory: true,
        ..Semantics::RELAXED
    };
    f.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)
}

/// What reads the draw's Metal index at a position.
#[derive(Clone, Copy)]
struct Fetch {
    indices: Value,
    first_byte: Value,
    inside: Value,
    start: Value,
    wide: Value,
}

impl Fetch {
    /// The index at the draw's position `q` (Metal position `start + q`): where it lies inside
    /// the whole indices (and did not wrap past `start`), its index, else 0; 4 bytes each where
    /// `wide`, else 2.
    fn at(self, f: &mut FunctionBuilder<'_>, q: Value) -> BResult<Value> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let two = k(f, 2)?;
        let three = k(f, 3)?;
        let p = f.binary(BinOp::IAdd, self.start, q)?;
        let unwrapped = f.binary(BinOp::UGreaterThanEqual, p, self.start)?;
        let below = f.binary(BinOp::ULessThan, p, self.inside)?;
        let valid = f.binary(BinOp::LogicalAnd, unwrapped, below)?;
        let shift = f.select(self.wide, two, one)?;
        let offset = f.binary(BinOp::ShiftLeftLogical, p, shift)?;
        let byte = f.binary(BinOp::IAdd, self.first_byte, offset)?;
        let word = f.binary(BinOp::ShiftRightLogical, byte, two)?;
        let safe = f.select(valid, word, zero)?;
        let loaded = contained(f, self.indices, safe)?;
        let half = f.binary(BinOp::BitwiseAnd, byte, two)?;
        let bits = f.binary(BinOp::ShiftLeftLogical, half, three)?;
        let shifted = f.binary(BinOp::ShiftRightLogical, loaded, bits)?;
        let mask = k(f, 0xFFFF)?;
        let narrow = f.binary(BinOp::BitwiseAnd, shifted, mask)?;
        let value = f.select(self.wide, loaded, narrow)?;
        f.select(valid, value, zero)
    }
}

/// The pass's dispatch of `phase` for `program` as a typed module, before serialization: the
/// module [`index_rewrite_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols [`WORDS`], [`ARGUMENTS`], [`INDICES`], [`OUTPUT`] and [`STATUS`].
pub fn rewrite_typed_module(program: RewriteProgram, phase: RewritePhase) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let most = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(IndexRewritePass::MAX_GROUPS)),
    })?;
    let list_ty = b.ty(Type::Array {
        elem: u32_,
        len: most,
        stride: None,
    })?;
    // Per run (count and write) or per group (place): its primitives, then those before it; the
    // start of its last piece, then of the piece it begins in.
    let counts = b.plain_global(StorageClass::Workgroup, list_ty, None)?;
    let starts = b.plain_global(StorageClass::Workgroup, list_ty, None)?;
    let words = words_buffer(&mut b, WORDS, u32_)?;
    let arguments = words_buffer(&mut b, ARGUMENTS, u32_)?;
    let indices = words_buffer(&mut b, INDICES, u32_)?;
    let output = words_buffer(&mut b, OUTPUT, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let invocation = b.builtin(
        crate::spv::BuiltIn::LocalInvocationIndex,
        IoDir::Input,
        u32_,
    )?;
    let workgroup = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let g = Globals {
        words: f.module().global_value(words)?,
        arguments: f.module().global_value(arguments)?,
        indices: f.module().global_value(indices)?,
        output: f.module().global_value(output)?,
        status: f.module().global_value(status)?,
        counts: f.module().global_value(counts)?,
        starts: f.module().global_value(starts)?,
        u32_,
    };
    let invocation = f.module().global_value(invocation)?;
    let lin = f.load(invocation, MemAccess::default())?;
    let workgroup = f.module().global_value(workgroup)?;
    let workgroup = f.load(workgroup, MemAccess::default())?;
    let group = f.composite_extract(workgroup, &[0])?;
    let at = Invocation { lin, group };
    let d = DrawWalk::read(&mut f, g, program.vertices())?;
    match phase {
        RewritePhase::Count => count_phase(&mut f, g, d, at)?,
        RewritePhase::Place => place_phase(&mut f, g, d, lin)?,
        RewritePhase::Write => write_phase(&mut f, g, d, at, program)?,
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: IndexRewriteModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The module's buffers and workgroup lists, and its `u32` type.
#[derive(Clone, Copy)]
struct Globals {
    words: Value,
    arguments: Value,
    indices: Value,
    output: Value,
    status: Value,
    /// Per run or group: its primitives, then those before it.
    counts: Value,
    /// Per run or group: the start of its last piece, then of the piece it begins in.
    starts: Value,
    u32_: TypeId,
}

impl Globals {
    /// A `u32` function variable holding `init`, by its pointer.
    fn local(self, f: &mut FunctionBuilder<'_>, init: Value) -> BResult<Value> {
        let var = f.local_var(self.u32_, None)?;
        let ptr = f.var_ptr(var)?;
        f.store(ptr, init, MemAccess::default())?;
        Ok(ptr)
    }

    /// Store `count` and `start` as entry `i` of the workgroup lists.
    fn list(
        self,
        f: &mut FunctionBuilder<'_>,
        i: Value,
        (count, start): (Value, Value),
    ) -> BResult<()> {
        let p = f.access_chain(self.counts, &[i])?;
        f.store(p, count, MemAccess::default())?;
        let p = f.access_chain(self.starts, &[i])?;
        f.store(p, start, MemAccess::default())
    }

    /// The first of group `group`'s two words.
    fn group_word(self, f: &mut FunctionBuilder<'_>, group: Value) -> BResult<Value> {
        let two = k(f, 2)?;
        let first = k(f, Output::Groups as u32)?;
        let w = f.binary(BinOp::IMul, group, two)?;
        f.binary(BinOp::IAdd, first, w)
    }
}

/// An invocation: its index in its workgroup, and its workgroup's along x.
#[derive(Clone, Copy)]
struct Invocation {
    lin: Value,
    group: Value,
}

/// An invocation's run of positions, `[lo, hi)`; its place among the runs, and the runs.
#[derive(Clone, Copy)]
struct Run {
    lo: Value,
    hi: Value,
    t: Value,
    runs: Value,
}

/// What every dispatch of a pass reads of its draw: how the walk fetches, `k'`, how far it walks,
/// Metal's other arguments and the output's terms.
#[derive(Clone, Copy)]
struct DrawWalk {
    fetch: Fetch,
    restart: Value,
    /// The positions the runs walk.
    walked: Value,
    /// The draw's positions past the walk.
    tail: Value,
    instances: Value,
    base: Value,
    base_instance: Value,
    capacity: Value,
    /// Whether the output is the index pool.
    pooled: Value,
    /// The groups, from 1 to `MAX_GROUPS`.
    groups: Value,
    /// The program's vertices per primitive.
    v: u32,
}

impl DrawWalk {
    fn read(f: &mut FunctionBuilder<'_>, g: Globals, v: u32) -> BResult<DrawWalk> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let argument_word = load_at(f, g.words, Parameter::ArgumentsWord as u32)?;
        let first_byte = load_at(f, g.words, Parameter::IndicesByte as u32)?;
        let inside = load_at(f, g.words, Parameter::Inside as u32)?;
        let wide = load_at(f, g.words, Parameter::Wide as u32)?;
        let restart = load_at(f, g.words, Parameter::Restart as u32)?;
        let capacity = load_at(f, g.words, Parameter::Capacity as u32)?;
        let pooled = load_at(f, g.words, Parameter::Pooled as u32)?;
        let groups = load_at(f, g.words, Parameter::Groups as u32)?;
        let most = k(f, IndexRewritePass::MAX_GROUPS)?;
        let groups = f.glsl(GlslOp::UMin, &[groups, most])?;
        let groups = f.glsl(GlslOp::UMax, &[groups, one])?;
        let wide = f.binary(BinOp::INotEqual, wide, zero)?;
        let pooled = f.binary(BinOp::INotEqual, pooled, zero)?;
        // Metal's arguments: count, instances, first index, base vertex, base instance; an
        // indirect draw's from its buffer, a direct draw's from the words.
        let indirect = f.binary(BinOp::INotEqual, argument_word, zero)?;
        let at = f.binary(BinOp::ISub, argument_word, one)?;
        let mut argument = Vec::with_capacity(5);
        for i in 0..5 {
            let i = k(f, i)?;
            let w = f.binary(BinOp::IAdd, at, i)?;
            let w = f.select(indirect, w, zero)?;
            let read = contained(f, g.arguments, w)?;
            let given = k(f, Parameter::Arguments as u32)?;
            let given = f.binary(BinOp::IAdd, given, i)?;
            let given = load(f, g.words, given)?;
            argument.push(f.select(indirect, read, given)?);
        }
        let (count, instances, start, base, base_instance) = (
            argument[0],
            argument[1],
            argument[2],
            argument[3],
            argument[4],
        );
        let fetch = Fetch {
            indices: g.indices,
            first_byte,
            inside,
            start,
            wide,
        };
        // The walk: the positions inside the buffer from the first index, and the program's
        // vertices past them (which read 0); past it every position reads 0.
        let vk_ = k(f, v)?;
        let held = saturating_sub(f, inside, start)?;
        let reach = saturating_add(f, held, vk_)?;
        let walked = f.glsl(GlslOp::UMin, &[count, reach])?;
        let tail = f.binary(BinOp::ISub, count, walked)?;
        Ok(DrawWalk {
            fetch,
            restart,
            walked,
            tail,
            instances,
            base,
            base_instance,
            capacity,
            pooled,
            groups,
            v,
        })
    }

    /// `at`'s run: one of `groups · LOCAL_SIZE` runs of the walk's positions, in order, each of
    /// as many as the walk over the runs, rounded up; empty where its workgroup is past the
    /// groups.
    fn run(self, f: &mut FunctionBuilder<'_>, at: Invocation) -> BResult<Run> {
        let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
        let runs = f.binary(BinOp::IMul, self.groups, lanes)?;
        let before = f.binary(BinOp::IMul, at.group, lanes)?;
        let t = f.binary(BinOp::IAdd, before, at.lin)?;
        let chunk = div_ceil(f, self.walked, runs)?;
        // Below 2^32 for a run among the runs: `t · chunk` is at most the walk less one where
        // the chunk is at least the runs, and below 2^31 where it is fewer.
        let from = f.binary(BinOp::IMul, t, chunk)?;
        let among = f.binary(BinOp::ULessThan, at.group, self.groups)?;
        let from = f.select(among, from, self.walked)?;
        let lo = f.glsl(GlslOp::UMin, &[from, self.walked])?;
        let end = saturating_add(f, lo, chunk)?;
        let hi = f.glsl(GlslOp::UMin, &[end, self.walked])?;
        Ok(Run { lo, hi, t, runs })
    }

    /// Before position `lo`: the positions of its piece there, at most `v - 1` (those of `lo - 1`,
    /// `lo - 2` that exist, back to the first that restarts), and the indices at `lo - 1` and
    /// `lo - 2` (0 where there is none).
    fn behind(self, f: &mut FunctionBuilder<'_>, lo: Value) -> BResult<(Value, Value, Value)> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let two = k(f, 2)?;
        let before = |f: &mut FunctionBuilder<'_>, n: Value| -> BResult<(Value, Value)> {
            let has = f.binary(BinOp::UGreaterThanEqual, lo, n)?;
            let q = f.binary(BinOp::ISub, lo, n)?;
            let fetched = self.fetch.at(f, q)?;
            let value = f.select(has, fetched, zero)?;
            let other = f.binary(BinOp::INotEqual, value, self.restart)?;
            Ok((value, f.binary(BinOp::LogicalAnd, has, other)?))
        };
        let (old, live1) = before(f, one)?;
        let (older, live2) = before(f, two)?;
        let streak = f.select(live1, one, zero)?;
        let streak = match self.v {
            2 => streak,
            _ => {
                let live2 = f.binary(BinOp::LogicalAnd, live1, live2)?;
                f.select(live2, two, streak)?
            }
        };
        Ok((streak, old, older))
    }

    /// A position reads `value`: whether it restarts, and whether it completes a primitive, its
    /// piece's positions through it (at most `v`) at `streak`.
    fn step(
        self,
        f: &mut FunctionBuilder<'_>,
        streak: Value,
        value: Value,
    ) -> BResult<(Value, Value)> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let vk_ = k(f, self.v)?;
        let restarts = f.binary(BinOp::IEqual, value, self.restart)?;
        let s = f.load(streak, MemAccess::default())?;
        let grown = f.binary(BinOp::IAdd, s, one)?;
        let grown = f.glsl(GlslOp::UMin, &[grown, vk_])?;
        let s = f.select(restarts, zero, grown)?;
        f.store(streak, s, MemAccess::default())?;
        let completes = f.binary(BinOp::UGreaterThanEqual, s, vk_)?;
        Ok((restarts, completes))
    }

    /// `run`'s primitives and the start of its last piece (0 where nothing in it restarts), its
    /// piece's positions before it `streak`.
    fn count(
        self,
        f: &mut FunctionBuilder<'_>,
        g: Globals,
        run: Run,
        streak: Value,
    ) -> BResult<(Value, Value)> {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let streak = g.local(f, streak)?;
        let count = g.local(f, zero)?;
        let start = g.local(f, zero)?;
        for_range(f, g.u32_, (run.lo, run.hi, one), move |f, q| {
            let value = self.fetch.at(f, q)?;
            let (restarts, completes) = self.step(f, streak, value)?;
            let zero = k(f, 0)?;
            let one = k(f, 1)?;
            let add = f.select(completes, one, zero)?;
            let c = f.load(count, MemAccess::default())?;
            let c = f.binary(BinOp::IAdd, c, add)?;
            f.store(count, c, MemAccess::default())?;
            let next = f.binary(BinOp::IAdd, q, one)?;
            let s = f.load(start, MemAccess::default())?;
            let s = f.select(restarts, next, s)?;
            f.store(start, s, MemAccess::default())
        })?;
        let count = f.load(count, MemAccess::default())?;
        let start = f.load(start, MemAccess::default())?;
        Ok((count, start))
    }
}

/// The first invocation's scan of the workgroup lists' first `n` entries, in order, from `place`
/// and `carry`: each count becomes the primitives before its entry (`place` and the counts
/// before it, saturated), each start the start of the piece its entry begins in (the greatest of
/// `carry` and the starts before it). The two totals' pointers.
fn scan(
    f: &mut FunctionBuilder<'_>,
    g: Globals,
    n: Value,
    (place, carry): (Value, Value),
) -> BResult<(Value, Value)> {
    let zero = k(f, 0)?;
    let one = k(f, 1)?;
    let place = g.local(f, place)?;
    let carry = g.local(f, carry)?;
    for_range(f, g.u32_, (zero, n, one), move |f, i| {
        let p = f.access_chain(g.counts, &[i])?;
        let count = f.load(p, MemAccess::default())?;
        let before = f.load(place, MemAccess::default())?;
        f.store(p, before, MemAccess::default())?;
        let next = saturating_add(f, before, count)?;
        f.store(place, next, MemAccess::default())?;
        let p = f.access_chain(g.starts, &[i])?;
        let start = f.load(p, MemAccess::default())?;
        let begun = f.load(carry, MemAccess::default())?;
        f.store(p, begun, MemAccess::default())?;
        let next = f.glsl(GlslOp::UMax, &[begun, start])?;
        f.store(carry, next, MemAccess::default())
    })?;
    Ok((place, carry))
}

/// [`RewritePhase::Count`]: each run's primitives and last piece's start, summed over the
/// workgroup into its group's words.
fn count_phase(
    f: &mut FunctionBuilder<'_>,
    g: Globals,
    d: DrawWalk,
    at: Invocation,
) -> BResult<()> {
    let run = d.run(f, at)?;
    let (streak, _, _) = d.behind(f, run.lo)?;
    let counted = d.count(f, g, run, streak)?;
    g.list(f, at.lin, counted)?;
    barrier(f)?;
    let zero = k(f, 0)?;
    let first = f.binary(BinOp::IEqual, at.lin, zero)?;
    let among = f.binary(BinOp::ULessThan, at.group, d.groups)?;
    let first = f.binary(BinOp::LogicalAnd, first, among)?;
    when(f, first, move |f| {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
        let (total, last) = scan(f, g, lanes, (zero, zero))?;
        let total = f.load(total, MemAccess::default())?;
        let last = f.load(last, MemAccess::default())?;
        let w = g.group_word(f, at.group)?;
        store(f, g.words, w, total)?;
        let w = f.binary(BinOp::IAdd, w, one)?;
        store(f, g.words, w, last)
    })
}

/// [`RewritePhase::Place`]: each group's words become the primitives before it and the start of
/// the piece it begins in; the output is reserved, the draw written, and what the write
/// dispatch needs stated.
fn place_phase(f: &mut FunctionBuilder<'_>, g: Globals, d: DrawWalk, lin: Value) -> BResult<()> {
    let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
    // The groups' words into the workgroup lists, the invocations taking turns.
    for_range(f, g.u32_, (lin, d.groups, lanes), move |f, i| {
        let one = k(f, 1)?;
        let w = g.group_word(f, i)?;
        let count = load(f, g.words, w)?;
        let w = f.binary(BinOp::IAdd, w, one)?;
        let start = load(f, g.words, w)?;
        g.list(f, i, (count, start))
    })?;
    barrier(f)?;
    let zero = k(f, 0)?;
    let first = f.binary(BinOp::IEqual, lin, zero)?;
    when(f, first, move |f| {
        let zero = k(f, 0)?;
        let one = k(f, 1)?;
        let vk_ = k(f, d.v)?;
        let (total, _) = scan(f, g, d.groups, (zero, zero))?;
        for_range(f, g.u32_, (zero, d.groups, one), move |f, i| {
            let one = k(f, 1)?;
            let w = g.group_word(f, i)?;
            let p = f.access_chain(g.counts, &[i])?;
            let before = f.load(p, MemAccess::default())?;
            store(f, g.words, w, before)?;
            let w = f.binary(BinOp::IAdd, w, one)?;
            let p = f.access_chain(g.starts, &[i])?;
            let begun = f.load(p, MemAccess::default())?;
            store(f, g.words, w, begun)
        })?;
        // Past the walk, every position reads 0: none restarts where `k'` is not 0, so each
        // completes a primitive of index 0 (the walk ends `v` positions of 0 past the buffer).
        let walked_total = f.load(total, MemAccess::default())?;
        let some = f.binary(BinOp::INotEqual, d.restart, zero)?;
        let tail_drawn = f.select(some, d.tail, zero)?;
        let primitives = saturating_add(f, walked_total, tail_drawn)?;
        // The output's indices, where they fit a `u32`: where the product's high word is 0
        // (`replay::checked_mul`, whose docs say why).
        let (written, past) = checked_mul(f, primitives, vk_, zero)?;
        let countable = f.unary(UnOp::LogicalNot, past)?;
        // Placed in the pool's room, or in the draw's own region.
        let counter_ptr = word_ptr(f, g.output, zero)?;
        let counter = f.load(counter_ptr, MemAccess::default())?;
        let used = f.glsl(GlslOp::UMin, &[counter, d.capacity])?;
        let room = f.binary(BinOp::ISub, d.capacity, used)?;
        let four = k(f, 4)?;
        let room = f.binary(BinOp::UDiv, room, four)?;
        let room = f.select(d.pooled, room, d.capacity)?;
        let fits = f.binary(BinOp::ULessThanEqual, written, room)?;
        let ok = f.binary(BinOp::LogicalAnd, countable, fits)?;
        let reserve = f.binary(BinOp::LogicalAnd, ok, d.pooled)?;
        when(f, reserve, move |f| {
            let four = k(f, 4)?;
            let bytes = f.binary(BinOp::IMul, written, four)?;
            let next = f.binary(BinOp::IAdd, used, bytes)?;
            f.store(counter_ptr, next, MemAccess::default())
        })?;
        let first_index = f.binary(BinOp::UDiv, used, four)?;
        let first_index = f.select(d.pooled, first_index, zero)?;
        let header = k(f, (IndexRewritePass::POOL_HEADER_BYTES / 4) as u32)?;
        let pool_base = f.binary(BinOp::IAdd, header, first_index)?;
        let base_word = f.select(d.pooled, pool_base, zero)?;
        let ok_word = f.select(ok, one, zero)?;
        for (word, value) in [
            (Output::Base, base_word),
            (Output::Written, ok_word),
            (Output::TailAt, walked_total),
            (Output::Tail, tail_drawn),
        ] {
            store_at(f, g.words, word as u32, value)?;
        }
        let drawn = f.select(ok, written, zero)?;
        let primitives_drawn = f.select(ok, primitives, zero)?;
        for (i, value) in [drawn, d.instances, first_index, d.base, d.base_instance]
            .into_iter()
            .enumerate()
        {
            store_at(f, g.words, Output::Draw as u32 + i as u32, value)?;
        }
        store_at(f, g.words, Output::Primitives as u32, primitives_drawn)?;
        // A draw whose primitives the output cannot hold draws nothing and says so.
        let over = f.unary(UnOp::LogicalNot, ok)?;
        when(f, over, move |f| {
            let word = k(f, StatusWord::Status.index())?;
            let bit = k(f, StatusBits::INDEX_POOL_OVERFLOW.0)?;
            let p = word_ptr(f, g.status, word)?;
            f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
            Ok(())
        })
    })
}

/// [`RewritePhase::Write`]: each run, placed after the runs before it from its group's words,
/// writes its primitives where the draw is written; the runs then write the primitives past the
/// walk.
fn write_phase(
    f: &mut FunctionBuilder<'_>,
    g: Globals,
    d: DrawWalk,
    at: Invocation,
    program: RewriteProgram,
) -> BResult<()> {
    let run = d.run(f, at)?;
    let (streak, old, older) = d.behind(f, run.lo)?;
    let counted = d.count(f, g, run, streak)?;
    g.list(f, at.lin, counted)?;
    barrier(f)?;
    let zero = k(f, 0)?;
    let first = f.binary(BinOp::IEqual, at.lin, zero)?;
    let among = f.binary(BinOp::ULessThan, at.group, d.groups)?;
    let first = f.binary(BinOp::LogicalAnd, first, among)?;
    when(f, first, move |f| {
        let one = k(f, 1)?;
        let lanes = k(f, IndexRewriteModule::LOCAL_SIZE)?;
        let w = g.group_word(f, at.group)?;
        let place = load(f, g.words, w)?;
        let w = f.binary(BinOp::IAdd, w, one)?;
        let carry = load(f, g.words, w)?;
        scan(f, g, lanes, (place, carry))?;
        Ok(())
    })?;
    barrier(f)?;
    let written = load_at(f, g.words, Output::Written as u32)?;
    let ok = f.binary(BinOp::INotEqual, written, zero)?;
    let base_word = load_at(f, g.words, Output::Base as u32)?;
    let v = d.v;
    when(f, ok, move |f| {
        let one = k(f, 1)?;
        let vk_ = k(f, v)?;
        let p = f.access_chain(g.counts, &[at.lin])?;
        let place = f.load(p, MemAccess::default())?;
        let p = f.access_chain(g.starts, &[at.lin])?;
        let begun = f.load(p, MemAccess::default())?;
        let into = f.binary(BinOp::IMul, place, vk_)?;
        let at0 = f.binary(BinOp::IAdd, base_word, into)?;
        let next_word = g.local(f, at0)?;
        let streak = g.local(f, streak)?;
        let piece = g.local(f, begun)?;
        let older = g.local(f, older)?;
        let old = g.local(f, old)?;
        for_range(f, g.u32_, (run.lo, run.hi, one), move |f, q| {
            let value = d.fetch.at(f, q)?;
            let (restarts, completes) = d.step(f, streak, value)?;
            let one = k(f, 1)?;
            let next = f.binary(BinOp::IAdd, q, one)?;
            let p = f.load(piece, MemAccess::default())?;
            let p = f.select(restarts, next, p)?;
            f.store(piece, p, MemAccess::default())?;
            let a = f.load(older, MemAccess::default())?;
            let b = f.load(old, MemAccess::default())?;
            when(f, completes, move |f| {
                let w = f.load(next_word, MemAccess::default())?;
                let primitive = match program {
                    RewriteProgram::LineStrip => vec![b, value],
                    RewriteProgram::TriangleStrip => {
                        // Vulkan's strip order: the piece's `i`th triangle (`q - p - 2`) takes
                        // `(a, b, c)` where `i` is even, `(a, c, b)` where it is odd.
                        let zero = k(f, 0)?;
                        let one = k(f, 1)?;
                        let i = f.binary(BinOp::ISub, q, p)?;
                        let parity = f.binary(BinOp::BitwiseAnd, i, one)?;
                        let odd = f.binary(BinOp::INotEqual, parity, zero)?;
                        let second = f.select(odd, value, b)?;
                        let third = f.select(odd, b, value)?;
                        vec![a, second, third]
                    }
                };
                for (i, index) in primitive.into_iter().enumerate() {
                    let i = k(f, i as u32)?;
                    let slot = f.binary(BinOp::IAdd, w, i)?;
                    contained_store(f, g.output, slot, index)?;
                }
                let vk_ = k(f, v)?;
                let next = f.binary(BinOp::IAdd, w, vk_)?;
                f.store(next_word, next, MemAccess::default())
            })?;
            f.store(older, b, MemAccess::default())?;
            f.store(old, value, MemAccess::default())
        })?;
        // The primitives of index 0 past the walk, the runs taking turns.
        let tail_at = load_at(f, g.words, Output::TailAt as u32)?;
        let tail = load_at(f, g.words, Output::Tail as u32)?;
        for_range(f, g.u32_, (run.t, tail, run.runs), move |f, j| {
            let vk_ = k(f, v)?;
            let zero = k(f, 0)?;
            let primitive = f.binary(BinOp::IAdd, tail_at, j)?;
            let into = f.binary(BinOp::IMul, primitive, vk_)?;
            let w = f.binary(BinOp::IAdd, base_word, into)?;
            for i in 0..v {
                let i = k(f, i)?;
                let slot = f.binary(BinOp::IAdd, w, i)?;
                contained_store(f, g.output, slot, zero)?;
            }
            Ok(())
        })
    })
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(IndexRewriteModule::SET),
        Symbol::Binding(s) if *s == WORDS => Some(IndexRewriteModule::WORDS_BINDING),
        Symbol::Binding(s) if *s == ARGUMENTS => Some(IndexRewriteModule::ARGUMENTS_BINDING),
        Symbol::Binding(s) if *s == INDICES => Some(IndexRewriteModule::INDICES_BINDING),
        Symbol::Binding(s) if *s == OUTPUT => Some(IndexRewriteModule::OUTPUT_BINDING),
        Symbol::Binding(s) if *s == STATUS => Some(IndexRewriteModule::STATUS_BINDING),
        _ => None,
    }
}

/// The index rewrite module of `program`'s dispatch of `phase` for `device`: built with the typed
/// builder, verified, serialized, and numbered with the fixed bindings of [`IndexRewriteModule`].
pub fn index_rewrite_module(
    device: &Device,
    program: RewriteProgram,
    phase: RewritePhase,
) -> Result<IndexRewriteModule, Error> {
    let (spirv, requirements) = finished(
        device,
        rewrite_typed_module(program, phase)?,
        binding_of,
        "the index rewrite",
    )?;
    Ok(IndexRewriteModule {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests;
