//! A fragment function's `half` color outputs whose last operation Metal computes at `float`
//! precision.
//!
//! Metal's compiler chooses per output (an M4 under macOS 27.0.1 and the M3 Max under macOS 26.7,
//! plain Metal, every lane of each case alike): under fast math, a `half` output whose value is a
//! product, a sum or difference, or a fused multiply-add of `half`s, and which has no other use,
//! reaches a 32-bit float attachment that does not blend unrounded, the product exact and the
//! others rounded to `float` once; a product reaches such an attachment blended with factors one
//! and one under `Add` unrounded too, added to the destination in `float`. Everywhere else in those
//! cases the value is the `half` the function computes.
//!
//! [`plan`] marks those operations for the channels `LoweringInputs::half_output_lanes` lists
//! ([`ArithPlan::unrounded`](super::ArithPlan::unrounded)): selection computes each in `float`
//! on the widened operands, which are `half`s, and rounds it to nothing narrower, so the output
//! stores the `float` Metal stores. A mark needs every fast-math flag on the operation (the
//! probes' fast math; a partial set was not measured), one return, and a route from the
//! operation to the returned output through `insertelement`, `shufflevector` and `insertvalue`
//! alone, each used once, that stores each of its lanes in one listed channel at most. A
//! regrouped tree's addition (`contract::chain`) and a chain of fused multiply-adds are not
//! marked: their rounding at `float` was not measured.
//!
//! An output to a listed attachment records its note ([`half_output_note`]) unless every channel
//! the attachment stores takes a marked lane or a constant, which Metal and the function store
//! alike.
//!
//! The note follows the module's math mode: a module compiled with fast math records
//! `PrecisionNote::HalfOutputFastMathBlend`, where Metal's compiler, folding the blend into the
//! function, reads the output's `float` or its `half` and fuses or regroups the blend per pipeline,
//! so no fixed form is Metal's; any other module (without fast math, or naming no mode) records
//! `PrecisionNote::HalfOutputPrecision`.
//!
//! A module without fast math (`air.compile.fast_math_disable`) records no note itself: Metal
//! stores the `half` it returns, so only a blend can store another value. It lists the attachment
//! (`StageIo::half_blend_colors`, [`note_half_output`]), and the pipeline's stage records
//! `PrecisionNote::HalfOutputPrecision` where that attachment blends
//! (`color_writes::blends_output`). A module with fast math lists an attachment it reaches as a
//! color attachment too (`StageIo::half_fast_colors`), and the pipeline's stage records
//! `PrecisionNote::HalfOutputFastMathBlend` where that attachment stores some channel
//! (`color_writes::stores_output`): one whose write mask stores none holds nothing the output could
//! differ by. A module naming no math mode keeps the note at any attachment: what Metal's compiler
//! does there was not measured.

use std::collections::{BTreeMap, BTreeSet};

use super::contract;
use crate::air::intrinsics::{AirIntrinsic, LlvmFloatOp, LlvmIntrinsic, Math, MathFn};
use crate::air::ir::{BinOp as AirBin, Callee, Const, FastMath, InstId, InstKind, Type, Value};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::select::SelectCx;
use crate::lower::stage_io::Returned;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::lower_inputs::HalfOutputLane;
use crate::pipeline::sym::Stage;

/// Every fast-math flag: Metal's fast math, which the probes measured.
fn fast() -> FastMath {
    FastMath::ALL
        .iter()
        .fold(FastMath::NONE, |s, (f, _)| s.union(*f))
}

/// What one lane of a returned output is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// A constant (or `poison`) lane.
    Constant,
    /// Lane `lane` of instruction `at`, reached through single-use moves alone.
    Lane { at: InstId, lane: u32 },
    /// Anything else.
    Other,
}

/// The use count of every instruction's result in the body.
fn use_counts(cx: &SelectCx<'_>, m: &mut Meter) -> Result<Vec<u32>, Error> {
    let mut uses = vec![0u32; cx.body.insts.len()];
    for (_, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        inst.kind.for_each_value(|v| {
            if let Value::Inst(d) = v {
                if let Some(u) = uses.get_mut(d.raw() as usize) {
                    *u = u.saturating_add(1);
                }
            }
        });
    }
    Ok(uses)
}

/// The constant integer `v`, if it is one.
fn const_int(cx: &SelectCx<'_>, v: Value) -> Option<u64> {
    let Value::Const(c) = v else { return None };
    match cx.entry.module.consts().get(c)?.value {
        Const::Int(b) => Some(b),
        _ => None,
    }
}

/// The lane count of the vector type of instruction or value `v` (1 for a scalar).
fn lanes(cx: &SelectCx<'_>, v: Value) -> Option<u32> {
    let ty = cx.value_ty(v)?;
    match cx.entry.module.types().get(ty)? {
        Type::Vector { len, .. } => Some(*len),
        _ => Some(1),
    }
}

/// Whether instruction `at` yields a `half` or a vector of `half`.
fn is_half(cx: &SelectCx<'_>, at: InstId) -> bool {
    let Some(ty) = cx.value_ty(Value::Inst(at)) else {
        return false;
    };
    let types = cx.entry.module.types();
    let scalar = match types.get(ty) {
        Some(Type::Vector { elem, .. }) => types.get(*elem),
        other => other,
    };
    matches!(scalar, Some(Type::Half))
}

/// Lane `lane` of `v`, followed back through moves each used once (`v` itself is the route's
/// last move, whose use the caller counted).
fn source(
    cx: &SelectCx<'_>,
    uses: &[u32],
    mut v: Value,
    mut lane: u32,
    m: &mut Meter,
) -> Result<Source, Error> {
    loop {
        m.tick(1)?;
        let at = match v {
            Value::Const(_) => return Ok(Source::Constant),
            Value::Arg(_) => return Ok(Source::Other),
            Value::Inst(at) => at,
        };
        let Some(inst) = cx.body.insts.get(at) else {
            return Ok(Source::Other);
        };
        let once = |w: Value| match w {
            Value::Inst(d) => uses.get(d.raw() as usize) == Some(&1),
            _ => true,
        };
        match &inst.kind {
            InstKind::InsertElement { vec, elt, idx } => match const_int(cx, *idx) {
                Some(k) if k == u64::from(lane) => {
                    if !once(*elt) {
                        return Ok(Source::Other);
                    }
                    v = *elt;
                    lane = 0;
                }
                Some(_) => {
                    if !once(*vec) {
                        return Ok(Source::Other);
                    }
                    v = *vec;
                }
                None => return Ok(Source::Other),
            },
            InstKind::ShuffleVector { a, b, mask } => {
                let Some(pick) = mask.get(lane as usize).copied() else {
                    return Ok(Source::Other);
                };
                let Some(pick) = pick else {
                    return Ok(Source::Constant);
                };
                let Some(n) = lanes(cx, *a) else {
                    return Ok(Source::Other);
                };
                let (w, k) = if pick < n { (*a, pick) } else { (*b, pick - n) };
                if !once(w) {
                    return Ok(Source::Other);
                }
                v = w;
                lane = k;
            }
            _ => return Ok(Source::Lane { at, lane }),
        }
    }
}

/// The returned value's field `field` (the value itself where the entry returns one output),
/// followed back through `insertvalue`s each used once.
fn field(
    cx: &SelectCx<'_>,
    uses: &[u32],
    returned: Value,
    field: u32,
    m: &mut Meter,
) -> Result<Option<Value>, Error> {
    let once = |w: Value| match w {
        Value::Inst(d) => uses.get(d.raw() as usize) == Some(&1),
        _ => true,
    };
    if cx.families.stage_io.returned == Returned::Value {
        return Ok(once(returned).then_some(returned));
    }
    let mut v = returned;
    loop {
        m.tick(1)?;
        let Value::Inst(at) = v else { return Ok(None) };
        let Some(inst) = cx.body.insts.get(at) else {
            return Ok(None);
        };
        let InstKind::InsertValue { agg, val, indices } = &inst.kind else {
            return Ok(None);
        };
        if indices.as_ref() == [field] {
            return Ok(once(*val).then_some(*val));
        }
        if !once(*agg) {
            return Ok(None);
        }
        v = *agg;
    }
}

/// Whether instruction `at` is a last operation Metal computes at `float` precision into a
/// channel taking `lane`: every fast-math flag, on `half`s, used once (by the route), and a
/// product for either channel, or a sum, difference, or fused multiply-add for a channel that
/// does not blend.
fn keeps_float(cx: &SelectCx<'_>, uses: &[u32], at: InstId, lane: HalfOutputLane) -> bool {
    let plan: &contract::Plan = &cx.families.arith.contract;
    if uses.get(at.raw() as usize) != Some(&1)
        || plan.folded.contains(&at)
        || plan.members.contains(&at)
        || !is_half(cx, at)
    {
        return false;
    }
    let Some(InstKind::BinOp { op, fmf, .. }) = cx.body.insts.get(at).map(|i| &i.kind) else {
        return false;
    };
    if *fmf != fast() {
        return false;
    }
    let fused = plan.fused.get(&at);
    if fused.is_some_and(|f| !f.inner.is_empty()) {
        return false;
    }
    matches!(
        (lane, op, fused),
        (_, AirBin::FMul, None) | (HalfOutputLane::Float32, AirBin::FAdd | AirBin::FSub, _)
    )
}

/// The note a `half` color output records where Metal may hand it to the attachment otherwise
/// than the function stores it (the module docs): `PrecisionNote::HalfOutputFastMathBlend` in a
/// module compiled with fast math (`air.compile.fast_math_enable`), whose blend Metal's compiler
/// fuses and regroups per pipeline; `PrecisionNote::HalfOutputPrecision` in any other, a module
/// that names no math mode too (no module is taken into the fast-math class unless it states
/// fast math).
pub(crate) fn half_output_note(cx: &SelectCx<'_>) -> PrecisionNote {
    match cx.entry.compile.fast_math {
        Some(true) => PrecisionNote::HalfOutputFastMathBlend,
        Some(false) | None => PrecisionNote::HalfOutputPrecision,
    }
}

/// Record that a `half` color output reaches attachment `color` where Metal may store it otherwise
/// than the function: in a module compiled without fast math (`air.compile.fast_math_disable`)
/// the attachment itself (`StageIo::half_blend_colors`), whose note the pipeline decides by
/// whether the attachment blends, since Metal stores the `half` the function returns wherever it
/// does not (the sweep: the `half4` output and a `float4` of its `half` store alike
/// under every blend and format); in a module compiled with fast math the attachment too
/// (`StageIo::half_fast_colors`), whose note the pipeline decides by whether the attachment stores
/// some channel, since one that stores none (its write mask none, or absent from the pipeline)
/// holds nothing the output could differ by, but [`half_output_note`] itself for an attachment
/// the function writes in the storage form, which the pipeline's color attachments do not show;
/// in a module naming no math mode [`half_output_note`] itself. Nothing where the function
/// computes Metal's 16-bit float blend there itself (`DevicePolicy::half_blend`, or a fetched
/// attachment's blend), which stores Metal's value.
pub(crate) fn note_half_output(cx: &mut SelectCx<'_>, color: u32) {
    // Metal's 16-bit float blend, computed by the function: the attachment stores Metal's value.
    if crate::lower::fragment::blends_half_in_storage(cx, color) {
        return;
    }
    let math = cx.entry.compile.fast_math;
    let in_storage = crate::lower::fragment::written_in_storage(cx, color);
    if math.is_none() || (math == Some(true) && in_storage) {
        let note = half_output_note(cx);
        cx.precision.insert(note);
        return;
    }
    let io = &mut cx.families.stage_io.interface;
    let colors = match math {
        Some(false) => &mut io.half_blend_colors,
        _ => &mut io.half_fast_colors,
    };
    if let Err(at) = colors.binary_search(&color) {
        colors.insert(at, color);
    }
}

/// Mark the `half` color outputs' last operations Metal computes at `float` precision
/// (`ArithPlan::unrounded`), and record [`half_output_note`] for an output to a listed attachment
/// one of whose stored channels takes neither such a lane nor a constant (the module docs).
pub fn plan(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    if cx.stage != Stage::Fragment || cx.inputs.half_output_lanes.is_empty() {
        return Ok(());
    }
    // The outputs to listed attachments that hold `half`s: each one's field and channels.
    let outputs: Vec<(u32, u32, [Option<HalfOutputLane>; 4])> = cx
        .families
        .stage_io
        .outputs
        .iter()
        .filter(|o| o.link.form.float_fmt() == Some(super::legal::FloatFmt::Half))
        .filter_map(|o| {
            let color = o.color?;
            let lanes = *cx.inputs.half_output_lanes.get(&color)?;
            Some((o.field, color, lanes))
        })
        .collect();
    if outputs.is_empty() {
        return Ok(());
    }
    let mut returns = Vec::new();
    for (_, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        if let InstKind::Ret(v) = &inst.kind {
            returns.push(*v);
        }
    }
    let returned = match returns.as_slice() {
        [Some(v)] => Some(*v),
        _ => None,
    };
    let uses = use_counts(cx, m)?;
    // Each instruction's lanes, with the channel each reaches; a lane reaching two is refused.
    let mut reached: BTreeMap<InstId, BTreeMap<u32, HalfOutputLane>> = BTreeMap::new();
    let mut refused: BTreeSet<InstId> = BTreeSet::new();
    // Each output's stored channels: the instruction lane each takes, or none (a constant).
    let mut stored: Vec<Vec<Option<(InstId, u32)>>> = Vec::new();
    let mut settled = Vec::new();
    for (f, _, lanes) in &outputs {
        let value = match returned {
            Some(r) => field(cx, &uses, r, *f, m)?,
            None => None,
        };
        let mut channels = Vec::new();
        let mut ok = value.is_some();
        for (k, lane) in lanes.iter().enumerate() {
            let Some(lane) = *lane else { continue };
            let Some(value) = value else { break };
            match source(cx, &uses, value, k as u32, m)? {
                Source::Constant => channels.push(None),
                Source::Lane { at, lane: l } => {
                    let entry = reached.entry(at).or_default();
                    if entry.insert(l, lane).is_some() {
                        refused.insert(at);
                    }
                    channels.push(Some((at, l)));
                }
                Source::Other => ok = false,
            }
        }
        stored.push(channels);
        settled.push(ok);
    }
    let mut unrounded = BTreeSet::new();
    for (at, lanes) in &reached {
        if refused.contains(at) {
            continue;
        }
        let every = lanes.values().all(|&l| keeps_float(cx, &uses, *at, l));
        if every {
            unrounded.insert(*at);
        }
    }
    for ((channels, ok), (_, color, _)) in stored.iter().zip(settled).zip(&outputs) {
        let all = ok
            && channels
                .iter()
                .all(|c| c.is_none_or(|(at, _)| unrounded.contains(&at)));
        if !all {
            note_half_output(cx, *color);
        }
    }
    cx.families.arith.unrounded.extend(unrounded);
    Ok(())
}

/// Record `PrecisionNote::ExtendedRangeBlendSource` where Metal's blend of the function's color
/// output into an extended-range attachment may round the output otherwise than
/// `extended_range::blend`'s model, which holds for an output Metal cannot see into under fast
/// math: a `float4` output loaded from memory (all 63 blends of 21 tuples into 550, 552 and 554,
/// byte for byte). Metal's compiler folds the blend into the function, so its rounding of the
/// source follows how the function computed it and the module's math mode (an Apple M4 under macOS
/// 27.0.1):
///
/// - a module that names no math mode: whether Metal's compiler fuses the blend is unknown.
///   A module without fast math (`air.compile.fast_math_disable`, safe and relaxed math alike)
///   records none: Metal rounds every product to `half` before its sum, for any output, which the
///   function computes (`extended_range::blend::BlendMath::Rounded`: every key of both sweeps under
///   safe and relaxed math: the source computed, a product with a loaded one: 21 computed `float4`
///   tuples: the lane table's every made blend into the six forms, the `_sRGB` ones too);
/// - under fast math, an output whose lanes are not each a load, of either type (a product: six of
///   the 21 tuples differ by a code on 35 to 432 of 4, 096 texels; the lerp, common-factor and kept
///   forms; 26 computed `float4` shapes, each of 21 tuples, of which Metal's compiler rounds a
///   third of the shapes otherwise again; and the consumer's `half4` of an integer hash, which
///   Metal on its M3 Max blends one code apart from the model under five of 22 tuples into 550, 552
///   and 554).
///
/// A computed `half4` under fast math matched the model in the shapes first measured (a product
/// used once, "over" and one-and-one), but not in every shape, so it is announced as a `float4` is:
/// only a load is proven. Every blended extended-range output of the function is checked: a lane is
/// a load where its value, followed back through single-use moves from the returned field, is a
/// `load`.
///
/// A `float4` of whole-vector arithmetic over loads and constants ([`whole_vector_arithmetic`]) is
/// rounded as a load, so it records none, but under a blend one of whose lane classes takes a key
/// of `extended_range::computed`'s tables (Metal rounds such a source otherwise there) or into an
/// `_sRGB` form (not swept).
///
/// Each output records the note of its element type: `PrecisionNote::ExtendedRangeFloatBlendSource`
/// an output of `float` lanes, `PrecisionNote::ExtendedRangeBlendSource` any other (`half`), so a
/// consumer tells a `float4` output's possible difference from a `half4` one's.
pub fn note_extended_range_sources(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    use crate::air::abi::{AbiInt, OutputRole};
    if cx.stage != Stage::Fragment {
        return Ok(());
    }
    let (_, tys) = crate::lower::stage_io::output_types(cx)?;
    let blended: Vec<(u32, u32, crate::air::ir::TypeId)> = cx
        .entry
        .ep
        .outputs
        .iter()
        .zip(tys)
        .filter_map(|(o, ty)| match o.role {
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(color),
                index: 0,
                ..
            } if cx
                .inputs
                .extended_range_colors
                .get(&color)
                .is_some_and(|e| e.blended) =>
            {
                Some((color, o.field, ty))
            }
            _ => None,
        })
        .collect();
    if blended.is_empty() {
        return Ok(());
    }
    let types = cx.entry.module.types();
    let fields: Vec<(u32, u32, u32, PrecisionNote)> = blended
        .into_iter()
        .map(|(color, field, ty)| {
            let (elem, len) = match types.get(ty) {
                Some(Type::Vector { elem, len }) => (types.get(*elem), *len),
                other => (other, 1),
            };
            let note = match elem {
                Some(Type::Float) => PrecisionNote::ExtendedRangeFloatBlendSource,
                _ => PrecisionNote::ExtendedRangeBlendSource,
            };
            (color, field, len, note)
        })
        .collect();
    match cx.entry.compile.fast_math {
        Some(true) => {}
        Some(false) => return Ok(()),
        None => {
            for (_, _, _, note) in fields {
                cx.precision.insert(note);
            }
            return Ok(());
        }
    }
    let mut returns = Vec::new();
    for (_, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        if let InstKind::Ret(v) = &inst.kind {
            returns.push(*v);
        }
    }
    let returned = match returns.as_slice() {
        [Some(v)] => Some(*v),
        _ => None,
    };
    let uses = use_counts(cx, m)?;
    for (color, f, n, note) in fields {
        let value = match returned {
            Some(r) => field(cx, &uses, r, f, m)?,
            None => None,
        };
        let loaded = match value {
            None => false,
            Some(value) => {
                let mut lanes_loaded = true;
                for lane in 0..n {
                    let is_load = match source(cx, &uses, value, lane, m)? {
                        Source::Lane { at, .. } => matches!(
                            cx.body.insts.get(at).map(|i| &i.kind),
                            Some(InstKind::Load { .. })
                        ),
                        Source::Constant | Source::Other => false,
                    };
                    lanes_loaded &= is_load;
                }
                // A `float4` of whole-vector arithmetic over loads and constants blends as a loaded
                // one but under the computed class's keys and into an `_sRGB` form; a write of none
                // of the attachment's channels blends nothing.
                let computed = !lanes_loaded
                    && whole_vector_arithmetic(cx, &uses, value, m)?
                    && crate::lower::fragment::shader_blend(cx, color).is_some_and(
                        |(eq, form, nothing)| match eq {
                            Some(eq) => {
                                crate::lower::arith::extended_range::computed::holds_as_loaded(
                                    &eq, form,
                                )
                            }
                            None => nothing,
                        },
                    );
                lanes_loaded || computed
            }
        };
        if !loaded {
            cx.precision.insert(note);
        }
    }
    Ok(())
}

/// Whether `v` is a `float4` (four `float` lanes) of whole-vector arithmetic over vector loads
/// and constants: a product, sum, difference or quotient carrying every fast-math flag, or a fused
/// multiply-add, of four `float` lanes, each used once, over such operations, `float4` loads (of
/// any address space) and `float4` constants, reading a load at least (the class Metal's compiler
/// rounds as a load, but under the keys of `extended_range::computed`: twelve such shapes byte for
/// byte alike on 21 blends, and a product of two loads under every blend of the sweeps and the
/// lane table). A lane moved alone (`insertelement`, `shufflevector`), a texture read, any other
/// call or operation, and a `half4` are outside it.
fn whole_vector_arithmetic(
    cx: &SelectCx<'_>,
    uses: &[u32],
    v: Value,
    m: &mut Meter,
) -> Result<bool, Error> {
    let types = cx.entry.module.types();
    let float4 = |w: Value| {
        cx.value_ty(w)
            .and_then(|t| types.get(t))
            .is_some_and(|t| match t {
                Type::Vector { elem, len: 4 } => matches!(types.get(*elem), Some(Type::Float)),
                _ => false,
            })
    };
    let fma = |callee: &Callee| {
        matches!(
            callee,
            Callee::Llvm {
                op: LlvmIntrinsic::Float {
                    op: LlvmFloatOp::Fma | LlvmFloatOp::FMulAdd,
                    ..
                },
                ..
            } | Callee::Air {
                op: AirIntrinsic::Math(Math { f: MathFn::Fma, .. }),
                ..
            }
        )
    };
    let mut loads = 0u32;
    let mut stack = vec![v];
    while let Some(w) = stack.pop() {
        m.tick(1)?;
        if !float4(w) {
            return Ok(false);
        }
        let at = match w {
            Value::Const(_) => continue,
            Value::Arg(_) => return Ok(false),
            Value::Inst(at) => at,
        };
        let Some(inst) = cx.body.insts.get(at) else {
            return Ok(false);
        };
        // An operation inside the tree is used by it alone (the root by the returned field).
        let once = w == v || uses.get(at.raw() as usize) == Some(&1);
        match &inst.kind {
            InstKind::Load {
                volatile: false,
                atomic: None,
                ..
            } => loads += 1,
            InstKind::BinOp {
                op: AirBin::FMul | AirBin::FAdd | AirBin::FSub | AirBin::FDiv,
                fmf,
                a,
                b,
                ..
            } if *fmf == fast() && once => stack.extend([*a, *b]),
            InstKind::Call(call) if fma(&call.callee) && once => {
                stack.extend(call.args.iter().copied())
            }
            _ => return Ok(false),
        }
    }
    Ok(loads > 0)
}
