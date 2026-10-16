//! A post-tessellation vertex function on a device without `tessellationShader`: a plain `Vertex`
//! entry whose wrapper ([`wrap_entry`]) runs the tessellator's decode and placement for the
//! vertex it draws, and hands the function its inputs through `Private` variables in place of the
//! evaluation stage's `TessCoord` and patch inputs (the module docs).

use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::stage_io;
use crate::lower::tessellation::factors::word;
use crate::lower::tessellation::tessellator::{self, Facts};
use crate::lower::tessellation::{element_array, work, Declared, EvalPlan};
use crate::metal::TessellationControlPointIndexType as IndexType;
use crate::pipeline::descriptor::TessellationDescriptor;
use crate::pipeline::interface::PatchDomain;
use crate::pipeline::link::tessellation::control::{CONTROL_POINT_INDEX_SLOT, PATCH_INDEX_SLOT};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, GlslOp, IoDir, LayoutRule, MemAccess, StorageClass, Type,
};

/// What an emulated post-tessellation vertex function bakes in: the render pipeline descriptor's
/// tessellation state it was compiled for (`LoweringInputs::tessellation`, Metal's defaults
/// without one), as the tessellator's facts too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Emulation {
    pub desc: TessellationDescriptor,
    pub facts: Facts,
    /// The stage-IO object to which the function writes its patch's position, the flat varying a
    /// fragment function reads as its `[[primitive_id]]` (`IoRole::PrimitiveId`), where its
    /// pipeline's fragment function reads it (`LoweringInputs::primitive_id_varying`).
    pub position_out: Option<usize>,
}

/// The binding symbol of an emulated tessellation draw's work buffer
/// (`SyntheticBinding::TessellationWork`), declared with the vertex stage's symbols.
pub const WORK_SLOT: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Synthetic(SyntheticBinding::TessellationWork),
    index: 0,
};

/// The binding symbol of an emulated tessellation draw's records
/// (`SyntheticBinding::TessellationRecords`), declared with the vertex stage's symbols.
pub const RECORDS_SLOT: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Synthetic(SyntheticBinding::TessellationRecords),
    index: 0,
};

/// Plan the emulation of a `domain` patch's evaluation entry on `cx`'s profile: the push field of
/// the chunk the draw draws, and under `LoweringInputs::primitive_id_varying` the primitive ID's
/// flat output ([`finish`] adds the index buffers' fields).
pub(super) fn plan(cx: &mut SelectCx<'_>, domain: PatchDomain) -> Result<Emulation, Error> {
    let desc = cx.inputs.tessellation.unwrap_or_default();
    cx.push.insert(PushField::TessellationChunk);
    let position_out = match cx.inputs.primitive_id_varying {
        true => Some(stage_io::primitive_id_varying(&mut cx.families.stage_io)?),
        false => None,
    };
    Ok(Emulation {
        desc,
        facts: Facts {
            domain,
            partition: desc.partition_mode,
            winding: desc.output_winding_order,
            max_factor: desc.max_tessellation_factor,
            scale_enabled: desc.factor_scale_enabled,
        },
        position_out,
    })
}

/// The push fields of the index buffers an emulated entry `em` reads, once its reads are planned
/// (whether it reads its `patch` index and its control points' `elements`): the patch index
/// buffer's residual and whether the draw binds one where it reads either, and the control-point
/// index buffer's residual where it reads control points of a pipeline with a control-point index
/// type. The storage buffers the entry binds beside its own ([`declare`]: the work buffer, the
/// records, and those index buffers) count toward the stage's per-stage limits, which the plan
/// checks once every family has counted (`select::finish`): a function they take past a limit is
/// refused, typed, as one of its own buffers would be.
pub(super) fn finish(cx: &mut SelectCx<'_>, (patch, elements): (bool, bool), em: &Emulation) {
    let mut buffers = 2;
    if patch || elements {
        cx.push.insert(PushField::PatchIndexOffset);
        cx.push.insert(PushField::PatchIndexBound);
        buffers += 1;
    }
    if elements && em.desc.control_point_index_type != IndexType::None {
        cx.push.insert(PushField::ControlPointIndexOffset);
        buffers += 1;
    }
    let d = &mut cx.descriptors;
    d.storage_buffers = d.storage_buffers.saturating_add(buffers);
}

/// A `Block { uint[] }` storage buffer bound at `sym`.
fn words_buffer(mb: &mut ModuleBuilder, sym: ResourceSlot) -> BResult<GlobalId> {
    let u32_ = uint(mb)?;
    let words = mb.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[words])?;
    let block = mb.ty(Type::Struct {
        members: [words].into(),
        layout: Some(layout),
    })?;
    mb.resource(sym, StorageClass::StorageBuffer, block)
}

fn uint(mb: &mut ModuleBuilder) -> BResult<TypeId> {
    mb.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

/// Declare the emulated entry's objects into `d`: its inputs as `Private` variables the wrapper
/// writes, `VertexIndex`, the draw's work buffer and records, and the patch index and
/// control-point index buffers where the function reads what they index.
pub(super) fn declare(
    mb: &mut ModuleBuilder,
    e: &EvalPlan,
    em: &Emulation,
    d: &mut Declared,
) -> BResult<()> {
    let u32_ = uint(mb)?;
    let private = |mb: &mut ModuleBuilder, ty| mb.plain_global(StorageClass::Private, ty, None);
    if e.coord {
        let f32_ = mb.ty(Type::Float { width: 32 })?;
        let v3 = mb.ty(Type::Vector { elem: f32_, len: 3 })?;
        d.coord = Some(private(mb, v3)?);
    }
    if e.patch {
        d.patch = Some(private(mb, u32_)?);
    }
    if e.instance {
        d.instance = Some(private(mb, u32_)?);
    }
    if e.base_instance {
        d.base_instance = Some(private(mb, u32_)?);
    }
    if e.elements {
        let ty = element_array(mb, e.control_points)?;
        d.elements = Some(private(mb, ty)?);
    }
    d.vertex_index = Some(mb.builtin(crate::spv::BuiltIn::VertexIndex, IoDir::Input, u32_)?);
    d.work = Some(words_buffer(mb, WORK_SLOT)?);
    d.records = Some(words_buffer(mb, RECORDS_SLOT)?);
    if e.patch || e.elements {
        d.patch_indices = Some(words_buffer(mb, PATCH_INDEX_SLOT)?);
    }
    if e.elements && em.desc.control_point_index_type != IndexType::None {
        d.control_point_indices = Some(words_buffer(mb, CONTROL_POINT_INDEX_SLOT)?);
    }
    Ok(())
}

fn k(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// Word `index` of the `uint[]` buffer `buffer`.
fn load(fb: &mut FunctionBuilder<'_>, buffer: GlobalId, index: Value) -> BResult<Value> {
    let base = fb.module().global_value(buffer)?;
    let zero = k(fb, 0)?;
    let ptr = fb.access_chain(base, &[zero, index])?;
    fb.load(ptr, MemAccess::default())
}

fn load_at(fb: &mut FunctionBuilder<'_>, buffer: GlobalId, index: u32) -> BResult<Value> {
    let i = k(fb, index)?;
    load(fb, buffer, i)
}

fn store_global(fb: &mut FunctionBuilder<'_>, g: Option<GlobalId>, v: Value) -> BResult<()> {
    if let Some(g) = g {
        let p = fb.module().global_value(g)?;
        fb.store(p, v, MemAccess::default())?;
    }
    Ok(())
}

/// The last `r` of `[lo, hi]` for which `holds(r)` (a predicate false from some point on past
/// `lo`, which it is taken to hold at), by bisection.
fn bisect<'a>(
    fb: &mut FunctionBuilder<'_>,
    (lo, hi): (Value, Value),
    holds: impl Fn(&mut FunctionBuilder<'_>, Value) -> BResult<Value> + 'a,
) -> BResult<Value> {
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let lo_var = fb.local_var(u32_, None)?;
    let lo_ptr = fb.var_ptr(lo_var)?;
    let hi_var = fb.local_var(u32_, None)?;
    let hi_ptr = fb.var_ptr(hi_var)?;
    fb.store(lo_ptr, lo, MemAccess::default())?;
    fb.store(hi_ptr, hi, MemAccess::default())?;
    let body: LoopFn<'a, ()> = Box::new(move |fb, _, lc| {
        let lo = fb.load(lo_ptr, MemAccess::default())?;
        let hi = fb.load(hi_ptr, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, lo, hi)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(&mut (), done, exit, None)?;
        // `lo + ceil((hi - lo) / 2)`: above `lo`, at most `hi`.
        let one = k(fb, 1)?;
        let span = fb.binary(BinOp::ISub, hi, lo)?;
        let span = fb.binary(BinOp::IAdd, span, one)?;
        let half = fb.binary(BinOp::ShiftRightLogical, span, one)?;
        let mid = fb.binary(BinOp::IAdd, lo, half)?;
        let yes = holds(fb, mid)?;
        let below = fb.binary(BinOp::ISub, mid, one)?;
        let lo = fb.select(yes, mid, lo)?;
        let hi = fb.select(yes, hi, below)?;
        fb.store(lo_ptr, lo, MemAccess::default())?;
        fb.store(hi_ptr, hi, MemAccess::default())
    });
    fb.loop_(&mut (), LoopSpec::default(), body)?;
    fb.load(lo_ptr, MemAccess::default())
}

/// Metal's patch index of the patch at `position`: the patch index buffer's word at byte
/// `residual + 4 · position` where the draw binds one (`bound` nonzero), else the position, as the
/// native control stage reads it (`control::patch_index`).
fn patch_index(
    fb: &mut FunctionBuilder<'_>,
    buffer: GlobalId,
    (residual, bound): (Value, Value),
    position: Value,
) -> BResult<Value> {
    let four = k(fb, 4)?;
    let bytes = fb.binary(BinOp::IMul, position, four)?;
    let address = fb.binary(BinOp::IAdd, residual, bytes)?;
    let two = k(fb, 2)?;
    let at = fb.binary(BinOp::ShiftRightLogical, address, two)?;
    let indexed = word(fb, buffer, at)?;
    let zero = k(fb, 0)?;
    let bound = fb.binary(BinOp::INotEqual, bound, zero)?;
    fb.select(bound, indexed, position)
}

/// The element of control point `k` of the patch whose index is `patch` (`n` control points):
/// `patch · n + k`, or with a control-point index type the index there in `buffer` from byte
/// `residual`, as the native control stage reads it (`control::elements`).
fn element(
    fb: &mut FunctionBuilder<'_>,
    (n, index_type): (u32, IndexType),
    indices: Option<(GlobalId, Value)>,
    patch: Value,
    k_: u32,
) -> BResult<Value> {
    let n = k(fb, n)?;
    let first = fb.binary(BinOp::IMul, patch, n)?;
    let kk = k(fb, k_)?;
    let e = fb.binary(BinOp::IAdd, first, kk)?;
    let Some((buffer, residual)) = indices else {
        return Ok(e);
    };
    let size = match index_type {
        IndexType::UInt16 => 2,
        _ => 4,
    };
    let size_ = k(fb, size)?;
    let bytes = fb.binary(BinOp::IMul, e, size_)?;
    let address = fb.binary(BinOp::IAdd, residual, bytes)?;
    let two = k(fb, 2)?;
    let at = fb.binary(BinOp::ShiftRightLogical, address, two)?;
    let w = word(fb, buffer, at)?;
    if size != 2 {
        return Ok(w);
    }
    let half = fb.binary(BinOp::BitwiseAnd, address, two)?;
    let three = k(fb, 3)?;
    let shift = fb.binary(BinOp::ShiftLeftLogical, half, three)?;
    let w = fb.binary(BinOp::ShiftRightLogical, w, shift)?;
    let mask = k(fb, 0xffff)?;
    fb.binary(BinOp::BitwiseAnd, w, mask)
}

/// The entry of an emulated post-tessellation vertex function: the vertex `VertexIndex` names
/// (triangle `t = VertexIndex / 3` of the draw's chunk, its corner `VertexIndex % 3`), its record
/// by bisection over the chunk's records, its point by the tessellator's decode and placement,
/// and its patch, instance and control points' elements, written to the variables the function
/// reads; then the function. Any other function's entry is `body`.
pub fn wrap_entry(cx: &mut EmitCx<'_>, mb: &mut ModuleBuilder, body: FuncId) -> BResult<FuncId> {
    let Some(e) = cx.plan.families.tessellation.eval.as_ref() else {
        return Ok(body);
    };
    let Some(em) = e.emulated else {
        return Ok(body);
    };
    let shape = (e.control_points, e.coord);
    let d = cx.decl.families.tessellation.clone();
    let position_out = match em.position_out {
        Some(i) => match cx.decl.families.stage_io.globals.get(i) {
            Some(&g) => Some(g),
            None => {
                return Err(
                    cx.fail("an emulated tessellation entry without its primitive ID output")
                )
            }
        },
        None => None,
    };
    let void = mb.ty(Type::Void)?;
    let mut fb = mb.function(void)?;
    prologue(cx, &mut fb, shape, (em, d, position_out), body)?;
    fb.finish()
}

/// The wrapper's body ([`wrap_entry`]).
fn prologue(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (control_points, read_coord): (u32, bool),
    (em, d, position_out): (Emulation, Declared, Option<GlobalId>),
    body: FuncId,
) -> BResult<()> {
    let (Some(vertex_index), Some(work_buf), Some(records)) = (d.vertex_index, d.work, d.records)
    else {
        return Err(cx.fail("an emulated tessellation entry without its draw's buffers"));
    };
    let vi = {
        let p = fb.module().global_value(vertex_index)?;
        fb.load(p, MemAccess::default())?
    };
    let chunk = cx.push_field(fb, PushField::TessellationChunk)?;
    let three = k(fb, 3)?;
    let t = fb.binary(BinOp::UDiv, vi, three)?;
    let corner = fb.binary(BinOp::UMod, vi, three)?;
    // The chunk's records: from its first, through the next chunk's first where that chunk draws,
    // else the draw's last.
    let one = k(fb, 1)?;
    let zero = k(fb, 0)?;
    let total = load_at(fb, work_buf, work::RECORDS)?;
    let chunks = load_at(fb, work_buf, work::CHUNKS)?;
    let slot_words = k(fb, work::SLOT_WORDS)?;
    let slot = fb.binary(BinOp::IMul, chunk, slot_words)?;
    let header = k(fb, work::HEADER)?;
    let slot = fb.binary(BinOp::IAdd, header, slot)?;
    let at = k(fb, work::SLOT_FIRST_RECORD)?;
    let at = fb.binary(BinOp::IAdd, slot, at)?;
    let first = load(fb, work_buf, at)?;
    let next = fb.binary(BinOp::IAdd, chunk, one)?;
    let has_next = fb.binary(BinOp::ULessThan, next, chunks)?;
    let next_slot = fb.binary(BinOp::IAdd, slot, slot_words)?;
    let next_slot = fb.select(has_next, next_slot, zero)?;
    let draw_at = k(fb, work::SLOT_DRAW)?;
    let draw_at = fb.binary(BinOp::IAdd, next_slot, draw_at)?;
    let next_vertices = load(fb, work_buf, draw_at)?;
    let next_draws = fb.binary(BinOp::INotEqual, next_vertices, zero)?;
    let next_draws = fb.binary(BinOp::LogicalAnd, has_next, next_draws)?;
    let first_at = k(fb, work::SLOT_FIRST_RECORD)?;
    let first_at = fb.binary(BinOp::IAdd, next_slot, first_at)?;
    let next_first = load(fb, work_buf, first_at)?;
    let last = fb.binary(BinOp::ISub, total, one)?;
    let last = fb.select(next_draws, next_first, last)?;
    // Each record's key in the chunk: its first triangle less the chunk's, modulo `2^32`.
    let records_base = load_at(fb, work_buf, work::RECORDS_BASE)?;
    let shift = k(fb, work::CHUNK_SHIFT)?;
    let chunk_first = fb.binary(BinOp::ShiftLeftLogical, chunk, shift)?;
    let record_word = move |fb: &mut FunctionBuilder<'_>, r: Value, w: u32| -> BResult<Value> {
        let rw = k(fb, work::RECORD_WORDS)?;
        let at = fb.binary(BinOp::IMul, r, rw)?;
        let at = fb.binary(BinOp::IAdd, records_base, at)?;
        let w = k(fb, w)?;
        let at = fb.binary(BinOp::IAdd, at, w)?;
        load(fb, records, at)
    };
    let key = move |fb: &mut FunctionBuilder<'_>, r: Value| -> BResult<Value> {
        let ft = record_word(fb, r, work::FIRST_TRIANGLE)?;
        fb.binary(BinOp::ISub, ft, chunk_first)
    };
    let r = bisect(fb, (first, last), move |fb, r| {
        let key = key(fb, r)?;
        fb.binary(BinOp::ULessThanEqual, key, t)
    })?;
    let l = key(fb, r)?;
    let l = fb.binary(BinOp::ISub, t, l)?;
    let mut p = [zero; 6];
    for (i, x) in p.iter_mut().enumerate() {
        *x = record_word(fb, r, i as u32)?;
    }
    let tiles = record_word(fb, r, work::STREAM_STATE)?;
    let ids = tessellator::emit_decode(fb, &em.facts, l, &p, tiles)?;
    let is0 = fb.binary(BinOp::IEqual, corner, zero)?;
    let is1 = fb.binary(BinOp::IEqual, corner, one)?;
    let id = fb.select(is1, ids[1], ids[2])?;
    let id = fb.select(is0, ids[0], id)?;
    if read_coord {
        let lanes = tessellator::emit_place(fb, &em.facts, id, &p)?;
        let f32_ = fb.module().ty(Type::Float { width: 32 })?;
        let v3 = fb.module().ty(Type::Vector { elem: f32_, len: 3 })?;
        let coord = fb.composite_construct(v3, &lanes)?;
        store_global(fb, d.coord, coord)?;
    }
    // The patch's position in the draw and its instance: record `r` is patch `r % patchCount` of
    // instance `r / patchCount`.
    let patch_count = load_at(fb, work_buf, work::PATCH_COUNT)?;
    let patch_count = fb.glsl(GlslOp::UMax, &[patch_count, one])?;
    let in_draw = fb.binary(BinOp::UMod, r, patch_count)?;
    let instance = fb.binary(BinOp::UDiv, r, patch_count)?;
    let start = load_at(fb, work_buf, work::PATCH_START)?;
    let position = fb.binary(BinOp::IAdd, start, in_draw)?;
    if let Some(out) = position_out {
        let out = fb.module().global_value(out)?;
        fb.store(out, position, MemAccess::default())?;
    }
    let base_instance = load_at(fb, work_buf, work::BASE_INSTANCE)?;
    store_global(fb, d.base_instance, base_instance)?;
    let instance = fb.binary(BinOp::IAdd, instance, base_instance)?;
    store_global(fb, d.instance, instance)?;
    if let Some(buffer) = d.patch_indices {
        let residual = cx.push_field(fb, PushField::PatchIndexOffset)?;
        let bound = cx.push_field(fb, PushField::PatchIndexBound)?;
        let patch = patch_index(fb, buffer, (residual, bound), position)?;
        store_global(fb, d.patch, patch)?;
        if let Some(elements) = d.elements {
            let indices = match d.control_point_indices {
                Some(buffer) => Some((
                    buffer,
                    cx.push_field(fb, PushField::ControlPointIndexOffset)?,
                )),
                None => None,
            };
            let base = fb.module().global_value(elements)?;
            for q in 0..control_points {
                let x = element(
                    fb,
                    (control_points, em.desc.control_point_index_type),
                    indices,
                    patch,
                    q,
                )?;
                let quad = k(fb, q / 4)?;
                let lane = k(fb, q % 4)?;
                let ptr = fb.access_chain(base, &[quad, lane])?;
                fb.store(ptr, x, MemAccess::default())?;
            }
        }
    }
    fb.call(body, &[])?;
    fb.ret(None)
}
