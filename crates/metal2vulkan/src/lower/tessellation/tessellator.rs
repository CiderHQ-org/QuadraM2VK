//! Metal's tessellator as SPIR-V function bodies, for a device without `tessellationShader`
//! (`lower::tessellation::emulated`, `bind::tessellation_scan`).
//!
//! The emulation draws a patch's triangles as plain vertices. These emitters say which triangles,
//! in which order, with which first corner, at which domain coordinates, bit for bit as Metal's
//! tessellator draws them on the M3 Max (macOS 27.0.1, the probe below):
//!
//! - [`emit_process`]: a patch's factors (`half` bits) and the encoder's scale to its processed
//!   factors, 16.16 fixed point, 0 for a culled patch;
//! - [`emit_triangle_count`]: the patch's triangles;
//! - [`emit_stream_step`]: Metal's batch of points across a draw, from one patch to the next, and
//!   the patch's tile word;
//! - [`emit_decode`]: triangle `l`'s points, its first corner first;
//! - [`emit_place`]: a point's domain coordinates.
//!
//! **Points and triangles.** Metal's points and triangles are those of the D3D11 reference
//! tessellator (Microsoft's `CHWTessellator`, MIT; the copy in Mesa's `d3d11/tessellator.cpp`):
//! its factor processing (`QuadProcessTessFactors`, `TriProcessTessFactors`), after Metal's own
//! two steps, the clamp to `maxTessellationFactor` (one less under `fractional_odd`) and `pow2`'s
//! rounding up to a power of two; its 16.16 point placement (`ComputeTessFactorContext`,
//! `PlacePointIn1D`); and its stitching (`StitchTransition`, `StitchRegular`).
//!
//! **Metal's order.** A patch draws, in order:
//! 1. the reference's outer transition ring (the outer edges stitched to the first inside
//!    ring), edge by edge in the reference's order, each triangle's first corner by its stitch
//!    step (an outside step, an inside step, the middle quad's two triangles, a triangle pointing
//!    in or out); a patch whose every factor is 1 draws the reference's 2 (quad) or 1 (triangle)
//!    triangles;
//! 2. the center: a quad's middle cell (both inside counts odd) or a triangle's center triangle
//!    (odd inside parity);
//! 3. the interior as regions of cells, each cell's two triangles in turn: a quad's 4 quadrants
//!    (a pinwheel around the center), a triangle's 3 sectors (each ring's corner quad and the
//!    halves of its edges' quads). A region's cells go row-major, or row-major over tiles and
//!    row-major in each tile when the batch tiles it.
//!
//! **The batch.** Metal draws a draw's patches (instance-major) through one batch of 512 points,
//! empty at the draw's start. A patch whose points fit the batch's free room is drawn whole.
//! Otherwise its transition ring goes in as one unit when it fits, or edge by edge, starting a
//! fresh batch at an edge that does not fit, unless the patch's rest (the remaining edges and the
//! interior) fits a fresh batch whole, in which case the rest is drawn whole there. Each region
//! then takes tiles of side `T`, the largest odd `T` whose `(T + 1)^2` points fit the free room
//! (a fresh batch, `T = 21`, when `T < 5`, unless no region was drawn yet and the interior fits a
//! fresh batch whole), each tile charging its points, starting a fresh batch when one does not
//! fit. A batch that is exactly full is closed. The fill is the scan's carried state, 0..=512.
//!
//! **No float operation sees a NaN or an infinity** (a Vulkan implementation may assume none
//! without `SignedZeroInfNanPreserve`): non-finite and non-positive factors are classified on
//! their bits; every placement is integer arithmetic; the only float operations are the factor
//! processing on finite values and the exact conversion of 16.16 locations.
//!
//! The probe (2026-10-16, M3 Max) compared every point bit for bit, the triangle set, each
//! triangle's first corner and the order within and across patches over 1550 random multi-patch
//! draws (6136 patch runs: quad and triangle, every partition, factors 0 to 64), with no
//! difference; a draw whose only patch is culled draws nothing.

use crate::metal::{TessellationPartitionMode as Partition, Winding};
use crate::pipeline::interface::PatchDomain;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, ControlRule};
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, MemAccess, Type};

/// The pipeline state the tessellator bakes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Facts {
    pub domain: PatchDomain,
    pub partition: Partition,
    pub winding: Winding,
    /// `maxTessellationFactor`.
    pub max_factor: u32,
    /// `tessellationFactorScaleEnabled`.
    pub scale_enabled: bool,
}

impl Facts {
    /// The edge and inside factors of a patch.
    pub fn counts(&self) -> (usize, usize) {
        match self.domain {
            PatchDomain::Triangle => (3, 1),
            PatchDomain::Quad => (4, 2),
        }
    }

    fn quad(&self) -> bool {
        self.domain == PatchDomain::Quad
    }

    /// Whether the partition's parity follows each factor (`integer`, `pow2`).
    fn integer(&self) -> bool {
        !matches!(
            self.partition,
            Partition::FractionalOdd | Partition::FractionalEven
        )
    }
}

/// The most segments a factor of a pipeline of `facts` makes.
fn max_segments(facts: &Facts) -> u32 {
    let m = facts.max_factor.clamp(1, 64);
    match facts.partition {
        Partition::Pow2 => m.next_power_of_two().min(64),
        Partition::FractionalOdd => (m.saturating_sub(1).max(1) | 1).min(63),
        Partition::FractionalEven => ((m + 1) & !1).clamp(2, 64),
        _ => m,
    }
}

/// The most triangles one patch of a pipeline of `facts` draws: `2n^2` for a quad, `3n^2 / 2`
/// for a triangle, `n` the most segments a factor makes.
pub fn max_triangle_count(facts: &Facts) -> u32 {
    let n = max_segments(facts);
    match facts.domain {
        PatchDomain::Triangle => 3 * n * n / 2,
        PatchDomain::Quad => 2 * n * n,
    }
}

/// 1.0 in 16.16.
const ONE: u32 = 0x1_0000;
/// The points one batch holds.
const BATCH: u32 = 512;
/// A point id's kind (bits 30 and 31): an interior point, `(a << 8) | b` for a quad (points `a`
/// and `b` of the `u` and `v` inside factors), `(edge << 16) | (ring << 8) | q` for a triangle.
const ID_INNER: u32 = 0;
/// An outer edge's point: `(edge << 16) | q`, point `q` of the edge's factor.
const ID_OUTER: u32 = 1 << 30;
/// A fixed point: bit 0 `u = 1`, bit 1 `v = 1`, bit 2 a triangle's center `(1/3, 1/3)`.
const ID_FIXED: u32 = 2 << 30;

// ---- Small emitters ----

fn uint(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })
}

fn float(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(Type::Float { width: 32 })
}

fn k(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

fn kf(fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<Value> {
    let ty = float(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v.to_bits())),
    })?;
    fb.module().const_value(c)
}

fn kb(fb: &mut FunctionBuilder<'_>, v: bool) -> BResult<Value> {
    let ty = fb.module().ty(Type::Bool)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v)),
    })?;
    fb.module().const_value(c)
}

fn add(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::IAdd, a, b)
}

fn addk(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::IAdd, a, b)
}

fn sub(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::ISub, a, b)
}

fn subk(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::ISub, a, b)
}

fn ksub(fb: &mut FunctionBuilder<'_>, a: u32, b: Value) -> BResult<Value> {
    let a = k(fb, a)?;
    fb.binary(BinOp::ISub, a, b)
}

fn mul(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::IMul, a, b)
}

fn mulk(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::IMul, a, b)
}

fn udiv(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::UDiv, a, b)
}

fn umod(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::UMod, a, b)
}

fn shr(fb: &mut FunctionBuilder<'_>, a: Value, n: u32) -> BResult<Value> {
    let n = k(fb, n)?;
    fb.binary(BinOp::ShiftRightLogical, a, n)
}

fn shl(fb: &mut FunctionBuilder<'_>, a: Value, n: u32) -> BResult<Value> {
    let n = k(fb, n)?;
    fb.binary(BinOp::ShiftLeftLogical, a, n)
}

fn andk(fb: &mut FunctionBuilder<'_>, a: Value, m: u32) -> BResult<Value> {
    let m = k(fb, m)?;
    fb.binary(BinOp::BitwiseAnd, a, m)
}

fn or(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::BitwiseOr, a, b)
}

fn ork(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::BitwiseOr, a, b)
}

fn eq(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::IEqual, a, b)
}

fn eqk(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::IEqual, a, b)
}

fn nek(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::INotEqual, a, b)
}

fn lt(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::ULessThan, a, b)
}

fn ltk(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::ULessThan, a, b)
}

fn le(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::ULessThanEqual, a, b)
}

fn lek(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::ULessThanEqual, a, b)
}

fn gt(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::UGreaterThan, a, b)
}

fn gek(fb: &mut FunctionBuilder<'_>, a: Value, b: u32) -> BResult<Value> {
    let b = k(fb, b)?;
    fb.binary(BinOp::UGreaterThanEqual, a, b)
}

fn land(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::LogicalAnd, a, b)
}

fn lor(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.binary(BinOp::LogicalOr, a, b)
}

fn lnot(fb: &mut FunctionBuilder<'_>, a: Value) -> BResult<Value> {
    fb.unary(crate::spirv::ir::UnOp::LogicalNot, a)
}

fn sel(fb: &mut FunctionBuilder<'_>, c: Value, a: Value, b: Value) -> BResult<Value> {
    fb.select(c, a, b)
}

/// `c ? a : b` for constants.
fn selk(fb: &mut FunctionBuilder<'_>, c: Value, a: u32, b: u32) -> BResult<Value> {
    let a = k(fb, a)?;
    let b = k(fb, b)?;
    fb.select(c, a, b)
}

fn umin(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.glsl(GlslOp::UMin, &[a, b])
}

fn umax(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    fb.glsl(GlslOp::UMax, &[a, b])
}

/// `1` or `0` as a `uint`.
fn bit(fb: &mut FunctionBuilder<'_>, c: Value) -> BResult<Value> {
    selk(fb, c, 1, 0)
}

/// Value `index` of `first` followed by `rest`, for a small `index`: a chain of selects.
fn pick(
    fb: &mut FunctionBuilder<'_>,
    index: Value,
    first: Value,
    rest: impl IntoIterator<Item = Value>,
) -> BResult<Value> {
    let mut out = first;
    for (i, v) in rest.into_iter().enumerate() {
        let at = eqk(fb, index, i as u32 + 1)?;
        out = sel(fb, at, v, out)?;
    }
    Ok(out)
}

/// `then` where `c` holds; nothing else.
fn when<'a>(
    fb: &mut FunctionBuilder<'_>,
    c: Value,
    then: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
) -> BResult<()> {
    let arm: ArmFn<'a, ()> = Box::new(move |fb, _| {
        then(fb)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), c, arm, None)?;
    Ok(())
}

/// `c ? then : otherwise`, each arm computing `N` values; only the taken arm runs.
fn branch<'a, const N: usize>(
    fb: &mut FunctionBuilder<'_>,
    c: Value,
    then: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<[Value; N]> + 'a,
    otherwise: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<[Value; N]> + 'a,
) -> BResult<[Value; N]> {
    let t: ArmFn<'a, ()> = Box::new(move |fb, _| Ok(ArmEnd::FallThrough(then(fb)?.to_vec())));
    let f: ArmFn<'a, ()> = Box::new(move |fb, _| Ok(ArmEnd::FallThrough(otherwise(fb)?.to_vec())));
    let merged = fb.if_(&mut (), c, t, Some(f))?;
    merged
        .0
        .try_into()
        .map_err(|_| BuildError::Control(ControlRule::ArmValues))
}

/// A `uint` `Function` variable holding `init`.
fn var(fb: &mut FunctionBuilder<'_>, init: Value) -> BResult<Value> {
    let u32_ = uint(fb)?;
    let v = fb.local_var(u32_, None)?;
    let p = fb.var_ptr(v)?;
    fb.store(p, init, MemAccess::default())?;
    Ok(p)
}

fn get(fb: &mut FunctionBuilder<'_>, p: Value) -> BResult<Value> {
    fb.load(p, MemAccess::default())
}

fn set(fb: &mut FunctionBuilder<'_>, p: Value, v: Value) -> BResult<()> {
    fb.store(p, v, MemAccess::default())
}

/// `each(i)` for `i` from 0 below `n`.
fn for_count<'a>(
    fb: &mut FunctionBuilder<'_>,
    n: Value,
    each: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<()> + 'a,
) -> BResult<()> {
    let zero = k(fb, 0)?;
    let counter = var(fb, zero)?;
    let body: LoopFn<'a, ()> = Box::new(move |fb, _, lc| {
        let i = get(fb, counter)?;
        let done = fb.binary(BinOp::UGreaterThanEqual, i, n)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(&mut (), done, exit, None)?;
        let next = addk(fb, i, 1)?;
        set(fb, counter, next)?;
        each(fb, i)
    });
    fb.loop_(&mut (), LoopSpec::default(), body)
}

// ---- One factor ----

/// What the stitching and counting read of one processed factor `f`.
#[derive(Clone, Copy, Debug)]
struct Factor {
    odd: Value,
    /// The context's half points (`numHalfTessFactorPoints`).
    num_half: Value,
    /// `NumPointsForTessFactor` (an inside factor's at least 4 odd, 3 even).
    points: Value,
}

/// The parity of processed factor `f`: the partition's, or under `integer` and `pow2` the
/// level's, an inside level of 1 even.
fn parity(fb: &mut FunctionBuilder<'_>, facts: &Facts, f: Value, inside: bool) -> BResult<Value> {
    match facts.partition {
        Partition::FractionalOdd => kb(fb, true),
        Partition::FractionalEven => kb(fb, false),
        _ => {
            let level = shr(fb, f, 16)?;
            let low = andk(fb, level, 1)?;
            let odd = eqk(fb, low, 1)?;
            if inside {
                let not_one = nek(fb, f, ONE)?;
                land(fb, odd, not_one)
            } else {
                Ok(odd)
            }
        }
    }
}

/// `(f + 1) / 2`, a half more when odd or exactly a half (`ComputeTessFactorContext`).
fn half_factor(fb: &mut FunctionBuilder<'_>, f: Value, odd: Value) -> BResult<Value> {
    let f1 = addk(fb, f, 1)?;
    let half = shr(fb, f1, 1)?;
    let is_half = eqk(fb, half, 0x8000)?;
    let bump = lor(fb, odd, is_half)?;
    let up = addk(fb, half, 0x8000)?;
    sel(fb, bump, up, half)
}

fn factor(fb: &mut FunctionBuilder<'_>, facts: &Facts, f: Value, inside: bool) -> BResult<Factor> {
    let odd = parity(fb, facts, f, inside)?;
    let half = half_factor(fb, f, odd)?;
    let ceil = addk(fb, half, 0xffff)?;
    let num_half = shr(fb, ceil, 16)?;
    let twice = shl(fb, num_half, 1)?;
    let even = selk(fb, odd, 0, 1)?;
    let points = add(fb, twice, even)?;
    let points = if inside {
        let least = selk(fb, odd, 4, 3)?;
        umax(fb, points, least)?
    } else {
        points
    };
    Ok(Factor {
        odd,
        num_half,
        points,
    })
}

/// A 1D placement context (`TESS_FACTOR_CONTEXT`).
#[derive(Clone, Copy, Debug)]
struct Ctx {
    odd: Value,
    frac: Value,
    num_half: Value,
    split: Value,
    inv_floor: Value,
    inv_ceil: Value,
}

/// `v` without its most significant set bit (0 stays 0).
fn remove_msb(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
    let msb = fb.glsl(GlslOp::FindUMsb, &[v])?;
    let msb = andk(fb, msb, 31)?;
    let one = k(fb, 1)?;
    let top = fb.binary(BinOp::ShiftLeftLogical, one, msb)?;
    let nonzero = nek(fb, v, 0)?;
    let zero = k(fb, 0)?;
    let top = sel(fb, nonzero, top, zero)?;
    fb.binary(BinOp::BitwiseXor, v, top)
}

/// `s_fixedReciprocal[n]`: `65536 / n` rounded to nearest, `(2^17 + n) / 2n` (no tie arises for
/// `n <= 64`), with `n` at least 1.
fn reciprocal(fb: &mut FunctionBuilder<'_>, n: Value) -> BResult<Value> {
    let one = k(fb, 1)?;
    let n = umax(fb, n, one)?;
    let top = addk(fb, n, 1 << 17)?;
    let twice = shl(fb, n, 1)?;
    udiv(fb, top, twice)
}

/// `ComputeTessFactorContext` of processed factor `f` of parity `odd`.
fn context(fb: &mut FunctionBuilder<'_>, f: Value, odd: Value) -> BResult<Ctx> {
    let half = half_factor(fb, f, odd)?;
    let floor = andk(fb, half, 0xffff_0000)?;
    let ceil = addk(fb, half, 0xffff)?;
    let ceil = andk(fb, ceil, 0xffff_0000)?;
    let frac = sub(fb, half, floor)?;
    let num_half = shr(fb, ceil, 16)?;
    let fl = shr(fb, floor, 16)?;
    let below = subk(fb, fl, 1)?;
    let odd_split = remove_msb(fb, below)?;
    let odd_split = shl(fb, odd_split, 1)?;
    let odd_split = addk(fb, odd_split, 1)?;
    let fl_one = eqk(fb, fl, 1)?;
    let zero = k(fb, 0)?;
    let odd_split = sel(fb, fl_one, zero, odd_split)?;
    let even_split = remove_msb(fb, fl)?;
    let even_split = shl(fb, even_split, 1)?;
    let even_split = addk(fb, even_split, 1)?;
    let split = sel(fb, odd, odd_split, even_split)?;
    let whole = eq(fb, ceil, floor)?;
    let past = addk(fb, num_half, 1)?;
    let split = sel(fb, whole, past, split)?;
    let odd_bit = bit(fb, odd)?;
    let fs = shl(fb, fl, 1)?;
    let fs = sub(fb, fs, odd_bit)?;
    let cs = shl(fb, num_half, 1)?;
    let cs = sub(fb, cs, odd_bit)?;
    Ok(Ctx {
        odd,
        frac,
        num_half,
        split,
        inv_floor: reciprocal(fb, fs)?,
        inv_ceil: reciprocal(fb, cs)?,
    })
}

/// `PlacePointIn1D`: point `point`'s location, 16.16.
fn place_1d(fb: &mut FunctionBuilder<'_>, c: &Ctx, point: Value) -> BResult<Value> {
    let flip = fb.binary(BinOp::UGreaterThanEqual, point, c.num_half)?;
    let twice = shl(fb, c.num_half, 1)?;
    let mirrored = sub(fb, twice, point)?;
    let odd_bit = bit(fb, c.odd)?;
    let mirrored = sub(fb, mirrored, odd_bit)?;
    let p = sel(fb, flip, mirrored, point)?;
    let middle = eq(fb, p, c.num_half)?;
    let past = gt(fb, p, c.split)?;
    let less = subk(fb, p, 1)?;
    let on_floor = sel(fb, past, less, p)?;
    let lf = mul(fb, on_floor, c.inv_floor)?;
    let lc = mul(fb, p, c.inv_ceil)?;
    let rest = ksub(fb, ONE, c.frac)?;
    let lf = mul(fb, lf, rest)?;
    let lc = mul(fb, lc, c.frac)?;
    let loc = add(fb, lf, lc)?;
    let loc = addk(fb, loc, 0x8000)?;
    let loc = shr(fb, loc, 16)?;
    let flipped = ksub(fb, ONE, loc)?;
    let loc = sel(fb, flip, flipped, loc)?;
    let half = k(fb, 0x8000)?;
    sel(fb, middle, half, loc)
}

// ---- Factor processing ----

/// `half` bits `h` (below `0x7c00`, so finite and not negative) as a `float`, exactly:
/// `significand · 2^(max(e, 1) - 25)`, by `Ldexp` and a multiply by `2^-25`, both exact.
fn half_value(fb: &mut FunctionBuilder<'_>, h: Value) -> BResult<Value> {
    let f32_ = float(fb)?;
    let exp = shr(fb, h, 10)?;
    let mant = andk(fb, h, 0x3ff)?;
    let subnormal = eqk(fb, exp, 0)?;
    let normal = ork(fb, mant, 0x400)?;
    let significand = sel(fb, subnormal, mant, normal)?;
    let one = k(fb, 1)?;
    let exp = sel(fb, subnormal, one, exp)?;
    let m = fb.convert(ConvOp::ConvertUToF, significand, f32_)?;
    let scaled = fb.glsl(GlslOp::Ldexp, &[m, exp])?;
    let ulp = kf(fb, f32::from_bits(0x3300_0000))?; // 2^-25
    fb.binary(BinOp::FMul, scaled, ulp)
}

/// Finite `x` clamped to `[lo, hi]` by ordered comparisons and selects.
fn clamp_f(fb: &mut FunctionBuilder<'_>, x: Value, lo: Value, hi: Value) -> BResult<Value> {
    let above = fb.binary(BinOp::FOrdGreaterThanEqual, x, lo)?;
    let x = sel(fb, above, x, lo)?;
    let below = fb.binary(BinOp::FOrdLessThanEqual, x, hi)?;
    sel(fb, below, x, hi)
}

/// Finite `x` in `[1, 64]` rounded up to a power of two.
fn pow2(fb: &mut FunctionBuilder<'_>, x: Value) -> BResult<Value> {
    let u32_ = uint(fb)?;
    let f32_ = float(fb)?;
    let up = fb.glsl(GlslOp::Ceil, &[x])?;
    let n = fb.convert(ConvOp::ConvertFToU, up, u32_)?;
    let below = subk(fb, n, 1)?;
    let msb = fb.glsl(GlslOp::FindUMsb, &[below])?;
    let shift = addk(fb, msb, 1)?;
    let one = k(fb, 1)?;
    let p = fb.binary(BinOp::ShiftLeftLogical, one, shift)?;
    fb.convert(ConvOp::ConvertUToF, p, f32_)
}

/// `floatToIDotF<15, 16, false>` of `v` in `[1, 64]`: `v · 2^16` rounded to nearest even, on
/// the bits.
fn to_fixed(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Value> {
    let u32_ = uint(fb)?;
    let bits = fb.convert(ConvOp::Bitcast, v, u32_)?;
    let exp = shr(fb, bits, 23)?;
    let mant = andk(fb, bits, 0x7f_ffff)?;
    let m = ork(fb, mant, 0x80_0000)?;
    // The bits dropped: 7 - (exp - 127), 1 to 7.
    let extra = ksub(fb, 134, exp)?;
    let one = k(fb, 1)?;
    let lsb = fb.binary(BinOp::ShiftLeftLogical, one, extra)?;
    let mask = subk(fb, lsb, 1)?;
    let half = shr(fb, lsb, 1)?;
    let kept = fb.binary(BinOp::BitwiseAnd, m, lsb)?;
    let kept = nek(fb, kept, 0)?;
    let rest = fb.binary(BinOp::BitwiseAnd, m, mask)?;
    let over = gt(fb, rest, half)?;
    let round = lor(fb, kept, over)?;
    let up = add(fb, m, half)?;
    let m = sel(fb, round, up, m)?;
    fb.binary(BinOp::ShiftRightLogical, m, extra)
}

/// The processed factors of a patch whose factors' `half` bits are `halves` (each a `uint`),
/// under `scale` (a `float`, read only where the pipeline enables the scale: the encoder's scale
/// rounded to `half`, an infinite one as `2^32`): each 16.16, all 0 for a culled patch.
///
/// Metal clamps each factor to `maxTessellationFactor` (one less under `fractional_odd`) and
/// `pow2` rounds a positive one up to a power of two; then the reference's processing: a patch
/// with an edge factor not above 0 (or NaN) is culled; edges clamp to the partition's range and
/// round up under `integer` and `pow2`; under `fractional_odd` an edge (or a quad's inside)
/// factor above 1 makes the inside factors' lower bound `1 + 2^-16`; insides clamp (a NaN to the
/// lower bound) and round alike; each converts to 16.16, rounding to nearest even.
pub fn emit_process(
    fb: &mut FunctionBuilder<'_>,
    facts: &Facts,
    halves: &[Value],
    scale: Value,
) -> BResult<[Value; 6]> {
    let (ne, ni) = facts.counts();
    let upper = match facts.partition {
        Partition::FractionalOdd => facts.max_factor.saturating_sub(1),
        _ => facts.max_factor,
    } as f32;
    let (lo, hi) = match facts.partition {
        Partition::FractionalEven => (2.0f32, 64.0f32),
        Partition::FractionalOdd => (1.0, 63.0),
        _ => (1.0, 64.0),
    };
    // A maximum of 0 (`fractional_odd` at 1) takes every edge to 0: every patch is culled.
    let mut culled = kb(fb, upper <= 0.0)?;
    let mut raw = Vec::new();
    let mut positive = Vec::new();
    for &h in halves.iter().take(ne + ni) {
        let h = andk(fb, h, 0xffff)?;
        // Positive and finite: 0x0001..=0x7bff; positive infinity 0x7c00.
        let from = subk(fb, h, 1)?;
        let finite = lek(fb, from, 0x7bff - 1)?;
        let infinite = eqk(fb, h, 0x7c00)?;
        let pos = lor(fb, finite, infinite)?;
        let zero_bits = k(fb, 0)?;
        let h = sel(fb, finite, h, zero_bits)?;
        let v = half_value(fb, h)?;
        let v = if facts.scale_enabled {
            fb.binary(BinOp::FMul, v, scale)?
        } else {
            v
        };
        // Metal's clamp to the pipeline's maximum: infinity becomes it too.
        let up = kf(fb, upper)?;
        let below = fb.binary(BinOp::FOrdLessThanEqual, v, up)?;
        let v = sel(fb, below, v, up)?;
        let v = sel(fb, infinite, up, v)?;
        let v = if facts.partition == Partition::Pow2 {
            let one = kf(fb, 1.0)?;
            let top = kf(fb, 64.0)?;
            let c = clamp_f(fb, v, one, top)?;
            let p = pow2(fb, c)?;
            sel(fb, pos, p, v)?
        } else {
            v
        };
        raw.push(v);
        positive.push(pos);
    }
    for &pos in positive.iter().take(ne) {
        let not = lnot(fb, pos)?;
        culled = lor(fb, culled, not)?;
    }
    let lo_v = kf(fb, lo)?;
    let hi_v = kf(fb, hi)?;
    let mut levels = Vec::new();
    for &v in raw.iter().take(ne) {
        let v = clamp_f(fb, v, lo_v, hi_v)?;
        let v = if facts.integer() {
            fb.glsl(GlslOp::Ceil, &[v])?
        } else {
            v
        };
        levels.push(v);
    }
    let lo_in = if facts.partition == Partition::FractionalOdd {
        let above = kf(fb, 1.0 + f32::from_bits(0x3700_0000))?; // 1 + 2^-17
        let mut frame = kb(fb, false)?;
        let checked = if facts.quad() { ne + ni } else { ne };
        for &v in levels.iter().chain(raw.iter().skip(ne)).take(checked) {
            // A non-positive inside factor is 0 here, which is not above.
            let over = fb.binary(BinOp::FOrdGreaterThan, v, above)?;
            frame = lor(fb, frame, over)?;
        }
        let framed = kf(fb, 1.0 + f32::from_bits(0x3780_0000))?; // 1 + 2^-16
        sel(fb, frame, framed, lo_v)?
    } else {
        lo_v
    };
    for (&v, &pos) in raw.iter().zip(&positive).skip(ne).take(ni) {
        let c = clamp_f(fb, v, lo_in, hi_v)?;
        let v = sel(fb, pos, c, lo_in)?;
        let v = if facts.integer() {
            fb.glsl(GlslOp::Ceil, &[v])?
        } else {
            v
        };
        levels.push(v);
    }
    let zero = k(fb, 0)?;
    let mut out = [zero; 6];
    for (slot, &v) in out.iter_mut().zip(&levels) {
        let fixed = to_fixed(fb, v)?;
        *slot = sel(fb, culled, zero, fixed)?;
    }
    Ok(out)
}

// ---- A patch's structure ----

/// What the counting, the batch and the decode read of a patch.
#[derive(Clone, Copy, Debug)]
struct Info {
    culled: Value,
    minimum: Value,
    /// The outer edges (a triangle's fourth unused).
    outer: [Factor; 4],
    /// The inside factors (`u`, `v`; a triangle's one twice).
    inside: [Factor; 2],
}

fn info(fb: &mut FunctionBuilder<'_>, facts: &Facts, p: &[Value; 6]) -> BResult<Info> {
    let (ne, ni) = facts.counts();
    let mut culled = kb(fb, false)?;
    for &f in p.iter().take(ne) {
        let zero = eqk(fb, f, 0)?;
        culled = lor(fb, culled, zero)?;
    }
    let mut minimum = kb(fb, facts.partition != Partition::FractionalEven)?;
    for &f in p.iter().take(ne + ni) {
        let one = eqk(fb, f, ONE)?;
        minimum = land(fb, minimum, one)?;
    }
    let [e0, e1, e2, e3, i0, i1] = *p;
    let o0 = factor(fb, facts, e0, false)?;
    let o1 = factor(fb, facts, e1, false)?;
    let o2 = factor(fb, facts, e2, false)?;
    let (outer, inside) = if facts.quad() {
        let o3 = factor(fb, facts, e3, false)?;
        let u = factor(fb, facts, i0, true)?;
        let v = factor(fb, facts, i1, true)?;
        ([o0, o1, o2, o3], [u, v])
    } else {
        // A triangle's inside factor follows its three edges.
        let inside = factor(fb, facts, e3, true)?;
        ([o0, o1, o2, o2], [inside, inside])
    };
    Ok(Info {
        culled,
        minimum,
        outer,
        inside,
    })
}

impl Info {
    /// The patch's outer edges in order, each with the inside factor stitched to it (a quad's
    /// `v` for edges 0 and 2, `u` for 1 and 3; a triangle's one, its fourth unused).
    fn edges(&self, facts: &Facts) -> [(Factor, Factor); 4] {
        let [o0, o1, o2, o3] = self.outer;
        let [u, v] = self.inside;
        if facts.quad() {
            [(o0, v), (o1, u), (o2, v), (o3, u)]
        } else {
            [(o0, u), (o1, u), (o2, u), (o3, u)]
        }
    }
}

/// The triangles of a transition ring's edge of factor `outer` stitched to `inside`.
fn ring_edge_triangles(
    fb: &mut FunctionBuilder<'_>,
    outer: Factor,
    inside: Factor,
) -> BResult<Value> {
    let n = add(fb, inside.points, outer.points)?;
    subk(fb, n, 4)
}

/// A quad's interior cells along `u` and `v`.
fn grid(fb: &mut FunctionBuilder<'_>, i: &Info) -> BResult<(Value, Value)> {
    Ok((
        subk(fb, i.inside[0].points, 3)?,
        subk(fb, i.inside[1].points, 3)?,
    ))
}

/// A triangle's regular rings: the reference's rings past the transition ring.
fn tri_rings(fb: &mut FunctionBuilder<'_>, i: &Info) -> BResult<Value> {
    let n = addk(fb, i.inside[0].points, 1)?;
    let rings = shr(fb, n, 1)?;
    subk(fb, rings, 2)
}

/// The interior's regions (rows, columns), in Metal's order.
fn regions(fb: &mut FunctionBuilder<'_>, facts: &Facts, i: &Info) -> BResult<Vec<(Value, Value)>> {
    if facts.quad() {
        let (nu, nv) = grid(fb, i)?;
        let hu = shr(fb, nu, 1)?;
        let hv = shr(fb, nv, 1)?;
        let cu = sub(fb, nu, hu)?;
        let cv = sub(fb, nv, hv)?;
        Ok(vec![
            (hu, sub(fb, nv, hv)?),
            (hv, cu),
            (sub(fb, nu, cu)?, cv),
            (sub(fb, nv, cv)?, sub(fb, nu, hu)?),
        ])
    } else {
        let rings = tri_rings(fb, i)?;
        let odd = bit(fb, i.inside[0].odd)?;
        let c = add(fb, rings, odd)?;
        Ok(vec![(rings, c); 3])
    }
}

/// The triangles of the transition ring.
fn ring_triangles(fb: &mut FunctionBuilder<'_>, facts: &Facts, i: &Info) -> BResult<Value> {
    let (ne, _) = facts.counts();
    let mut total = k(fb, 0)?;
    for &(outer, inside) in i.edges(facts).iter().take(ne) {
        let n = ring_edge_triangles(fb, outer, inside)?;
        total = add(fb, total, n)?;
    }
    Ok(total)
}

/// The triangles a patch of processed factors `p` draws (0 for a culled patch).
pub fn emit_triangle_count(
    fb: &mut FunctionBuilder<'_>,
    facts: &Facts,
    p: &[Value; 6],
) -> BResult<Value> {
    let i = info(fb, facts, p)?;
    let ring = ring_triangles(fb, facts, &i)?;
    let interior = if facts.quad() {
        let (nu, nv) = grid(fb, &i)?;
        let cells = mul(fb, nu, nv)?;
        shl(fb, cells, 1)?
    } else {
        let rings = tri_rings(fb, &i)?;
        let odd = bit(fb, i.inside[0].odd)?;
        let c = add(fb, rings, odd)?;
        let cells = mul(fb, rings, c)?;
        let six = mulk(fb, cells, 6)?;
        add(fb, six, odd)?
    };
    let n = add(fb, ring, interior)?;
    let least = k(fb, if facts.quad() { 2 } else { 1 })?;
    let n = sel(fb, i.minimum, least, n)?;
    let zero = k(fb, 0)?;
    sel(fb, i.culled, zero, n)
}

// ---- The batch ----

/// The tile side code `c` (side `2c - 1`) of the largest odd side whose `(side + 1)^2` points fit
/// `free`: the count of `j` in `1..=11` with `(2j)^2 <= free`.
fn side_code(fb: &mut FunctionBuilder<'_>, free: Value) -> BResult<Value> {
    let mut c = k(fb, 0)?;
    for j in 1..=11u32 {
        let fits = gek(fb, free, 4 * j * j)?;
        let b = bit(fb, fits)?;
        c = add(fb, c, b)?;
    }
    Ok(c)
}

/// The tile shape (rows, columns) of an `r` by `c` region at side `t`.
fn tile_shape(
    fb: &mut FunctionBuilder<'_>,
    r: Value,
    c: Value,
    t: Value,
) -> BResult<(Value, Value)> {
    let t1 = addk(fb, t, 1)?;
    let b = mul(fb, t1, t1)?;
    let r1 = addk(fb, r, 1)?;
    let wide = udiv(fb, b, r1)?;
    let wide = subk(fb, wide, 1)?;
    let wide = umin(fb, c, wide)?;
    let c1 = addk(fb, c, 1)?;
    let tall = udiv(fb, b, c1)?;
    let tall = subk(fb, tall, 1)?;
    let tall = umin(fb, r, tall)?;
    let short = lt(fb, r, t)?;
    let narrow = lt(fb, c, t)?;
    let rows = sel(fb, narrow, tall, t)?;
    let rows = sel(fb, short, r, rows)?;
    let cols = sel(fb, narrow, c, t)?;
    let cols = sel(fb, short, wide, cols)?;
    Ok((rows, cols))
}

/// The fill after `n` tiles of `cost` points each, from fill `used`: a tile that does not fit
/// starts a fresh batch.
fn charge(fb: &mut FunctionBuilder<'_>, used: Value, cost: Value, n: Value) -> BResult<Value> {
    let room = ksub(fb, BATCH, used)?;
    let fits = udiv(fb, room, cost)?;
    let all = le(fb, n, fits)?;
    let nf = mul(fb, n, cost)?;
    let within = add(fb, used, nf)?;
    let rest = sub(fb, n, fits)?;
    let batch = k(fb, BATCH)?;
    let per = udiv(fb, batch, cost)?;
    let last = subk(fb, rest, 1)?;
    let last = umod(fb, last, per)?;
    let last = addk(fb, last, 1)?;
    let fresh = mul(fb, last, cost)?;
    let after = sel(fb, all, within, fresh)?;
    let none = eqk(fb, n, 0)?;
    sel(fb, none, used, after)
}

/// The batch fill after a patch of processed factors `p` drawn at batch fill `used`, and the
/// patch's tile word, which its decode reads: per region (5 bits each, region 0 lowest), 0 for a
/// region drawn whole, else `(T - 1) / 2` for a region drawn in tiles of side `T`. A culled
/// patch leaves the fill as it is.
pub fn emit_stream_step(
    fb: &mut FunctionBuilder<'_>,
    facts: &Facts,
    used: Value,
    p: &[Value; 6],
) -> BResult<(Value, Value)> {
    let (ne, _) = facts.counts();
    let i = info(fb, facts, p)?;
    // The patch's points, its interior's, its transition ring's and each ring edge's.
    let mut outer = k(fb, 0)?;
    for f in i.outer.iter().take(ne) {
        outer = add(fb, outer, f.points)?;
    }
    let outer = subk(fb, outer, ne as u32)?;
    let (interior, cells, ring_inside) = if facts.quad() {
        let a = subk(fb, i.inside[0].points, 2)?;
        let b = subk(fb, i.inside[1].points, 2)?;
        let interior = mul(fb, a, b)?;
        let (nu, nv) = grid(fb, &i)?;
        let nu0 = eqk(fb, nu, 0)?;
        let nv0 = eqk(fb, nv, 0)?;
        let flat = lor(fb, nu0, nv0)?;
        let cells = lnot(fb, flat)?;
        let sum = add(fb, nu, nv)?;
        let line = addk(fb, sum, 1)?;
        let frame = shl(fb, sum, 1)?;
        (interior, cells, sel(fb, flat, line, frame)?)
    } else {
        let n = i.inside[0].points;
        let r = shr(fb, n, 1)?;
        let r = subk(fb, r, 1)?;
        let r1 = addk(fb, r, 1)?;
        let rr = mul(fb, r, r1)?;
        let even = mulk(fb, rr, 3)?;
        let even = addk(fb, even, 1)?;
        let sq = mul(fb, r, r)?;
        let odd = mulk(fb, sq, 3)?;
        let interior = sel(fb, i.inside[0].odd, odd, even)?;
        let rings = tri_rings(fb, &i)?;
        let some = nek(fb, rings, 0)?;
        let cells = lor(fb, i.inside[0].odd, some)?;
        let e = subk(fb, n, 2)?;
        let single = eqk(fb, e, 1)?;
        let e1 = subk(fb, e, 1)?;
        let around = mulk(fb, e1, 3)?;
        let one = k(fb, 1)?;
        (interior, cells, sel(fb, single, one, around)?)
    };
    let points = add(fb, outer, interior)?;
    let zero = k(fb, 0)?;
    let interior = sel(fb, cells, interior, zero)?;
    let ring = add(fb, outer, ring_inside)?;
    let mut sides = Vec::new();
    for &(outer, inside) in i.edges(facts).iter().take(ne) {
        let inside = subk(fb, inside.points, 2)?;
        sides.push(add(fb, outer.points, inside)?);
    }
    let full = gek(fb, used, BATCH)?;
    let used0 = sel(fb, full, zero, used)?;
    let with_all = add(fb, used0, points)?;
    let whole = lek(fb, with_all, BATCH)?;
    // The ring as one unit, or edge by edge.
    let with_ring = add(fb, used0, ring)?;
    let ring_fits = lek(fb, with_ring, BATCH)?;
    let mut u = used0;
    let mut done = kb(fb, false)?;
    let mut rest_used = zero;
    // Each edge's rest: its side, the later edges' and the interior.
    let mut rests = Vec::new();
    let mut rest = interior;
    for &side in sides.iter().rev() {
        rest = add(fb, rest, side)?;
        rests.push(rest);
    }
    for (&side, &rest) in sides.iter().zip(rests.iter().rev()) {
        let not_done = lnot(fb, done)?;
        let with = add(fb, u, side)?;
        let over = gek(fb, with, BATCH + 1)?;
        let over = land(fb, over, not_done)?;
        let rest_fits = lek(fb, rest, BATCH)?;
        let take = land(fb, over, rest_fits)?;
        rest_used = sel(fb, take, rest, rest_used)?;
        done = lor(fb, done, take)?;
        let fresh = sel(fb, over, zero, u)?;
        let next = add(fb, fresh, side)?;
        u = sel(fb, done, u, next)?;
    }
    let u = sel(fb, ring_fits, with_ring, u)?;
    let edgewise = lnot(fb, ring_fits)?;
    let done = land(fb, done, edgewise)?;
    // The regions, each whole or in tiles.
    let one = k(fb, 1)?;
    let state = var(fb, u)?;
    let word = var(fb, zero)?;
    let first = var(fb, one)?;
    let finished = var(fb, zero)?;
    let skip = lor(fb, whole, done)?;
    let skip = lor(fb, skip, i.minimum)?;
    let skip = lor(fb, skip, i.culled)?;
    let tiled = lnot(fb, skip)?;
    let regions = regions(fb, facts, &i)?;
    when(fb, tiled, move |fb| {
        for (index, (r, c)) in regions.into_iter().enumerate() {
            let fin = get(fb, finished)?;
            let open = eqk(fb, fin, 0)?;
            let rows = nek(fb, r, 0)?;
            let cols = nek(fb, c, 0)?;
            let cells = land(fb, rows, cols)?;
            let live = land(fb, open, cells)?;
            when(fb, live, move |fb| {
                let u = get(fb, state)?;
                let free = ksub(fb, BATCH, u)?;
                let code = side_code(fb, free)?;
                let small = ltk(fb, code, 3)?;
                let f = get(fb, first)?;
                let is_first = eqk(fb, f, 1)?;
                let interior_fits = lek(fb, interior, BATCH)?;
                let alone = land(fb, small, is_first)?;
                let alone = land(fb, alone, interior_fits)?;
                let zero = k(fb, 0)?;
                set(fb, first, zero)?;
                let one = k(fb, 1)?;
                let fin = sel(fb, alone, one, zero)?;
                set(fb, finished, fin)?;
                let interior_u = sel(fb, alone, interior, u)?;
                set(fb, state, interior_u)?;
                let tile = lnot(fb, alone)?;
                when(fb, tile, move |fb| {
                    // Too small a side: a fresh batch, side 21.
                    let zero = k(fb, 0)?;
                    let u = sel(fb, small, zero, u)?;
                    let fresh = k(fb, 11)?;
                    let code = sel(fb, small, fresh, code)?;
                    let w = get(fb, word)?;
                    let field = subk(fb, code, 1)?;
                    let field = shl(fb, field, 5 * index as u32)?;
                    let w = or(fb, w, field)?;
                    set(fb, word, w)?;
                    let t = shl(fb, code, 1)?;
                    let t = subk(fb, t, 1)?;
                    let (th, tw) = tile_shape(fb, r, c, t)?;
                    set(fb, state, u)?;
                    // Bands of `th` rows: full tiles of `tw` columns, then a narrower one.
                    let th1 = subk(fb, th, 1)?;
                    let bands = add(fb, r, th1)?;
                    let bands = udiv(fb, bands, th)?;
                    let fulls = udiv(fb, c, tw)?;
                    let covered = mul(fb, fulls, tw)?;
                    let narrow = sub(fb, c, covered)?;
                    for_count(fb, bands, move |fb, band| {
                        let top = mul(fb, band, th)?;
                        let left = sub(fb, r, top)?;
                        let h = umin(fb, th, left)?;
                        let h1 = addk(fb, h, 1)?;
                        let tw1 = addk(fb, tw, 1)?;
                        let cost = mul(fb, h1, tw1)?;
                        let u = get(fb, state)?;
                        let u = charge(fb, u, cost, fulls)?;
                        let n1 = addk(fb, narrow, 1)?;
                        let cost = mul(fb, h1, n1)?;
                        let some = nek(fb, narrow, 0)?;
                        let n = bit(fb, some)?;
                        let u = charge(fb, u, cost, n)?;
                        set(fb, state, u)
                    })
                })
            })?;
        }
        Ok(())
    })?;
    let tiled_used = get(fb, state)?;
    let tiled_word = get(fb, word)?;
    // A patch of every factor 1: its points, in a fresh batch when they do not fit.
    let with_outer = add(fb, used0, outer)?;
    let fits_min = lek(fb, with_outer, BATCH)?;
    let min_used = sel(fb, fits_min, with_outer, outer)?;
    let after = sel(fb, done, rest_used, tiled_used)?;
    let after = sel(fb, whole, with_all, after)?;
    let after = sel(fb, i.minimum, min_used, after)?;
    let after = sel(fb, i.culled, used, after)?;
    let plain = lor(fb, done, whole)?;
    let plain = lor(fb, plain, i.minimum)?;
    let plain = lor(fb, plain, i.culled)?;
    let word = sel(fb, plain, zero, tiled_word)?;
    Ok((after, word))
}

// ---- Decode ----

/// `finalPointPositionTable[j]` for `j` in `0..=32`: 0, 32, then `15 + j/2` for even `j`, and for
/// odd `j` with `m = (j - 1) / 2 = 2^c · o` (`o` odd), `2^(3 - c) + (o - 1) / 2`.
fn final_position(fb: &mut FunctionBuilder<'_>, j: Value) -> BResult<Value> {
    let half = shr(fb, j, 1)?;
    let even_v = addk(fb, half, 15)?;
    let m = subk(fb, j, 1)?;
    let m = shr(fb, m, 1)?;
    let c = fb.glsl(GlslOp::FindILsb, &[m])?;
    let c = andk(fb, c, 31)?;
    let eight = k(fb, 8)?;
    let level = fb.binary(BinOp::ShiftRightLogical, eight, c)?;
    let c1 = addk(fb, c, 1)?;
    let c1 = andk(fb, c1, 31)?;
    let o = fb.binary(BinOp::ShiftRightLogical, m, c1)?;
    let odd_v = add(fb, level, o)?;
    let low = andk(fb, j, 1)?;
    let is_odd = eqk(fb, low, 1)?;
    let v = sel(fb, is_odd, odd_v, even_v)?;
    let one = eqk(fb, j, 1)?;
    let top = k(fb, 32)?;
    let v = sel(fb, one, top, v)?;
    let zero = eqk(fb, j, 0)?;
    let z = k(fb, 0)?;
    sel(fb, zero, z, v)
}

/// Triangle `t` of a `StitchTransition` half (the first half's walk; the second half mirrors it)
/// between inside and outside rows of `in_half` and `out_half` half points: its kind (1 an
/// outside step, 2 an inside step) and the outside and inside steps before it.
fn walk(
    fb: &mut FunctionBuilder<'_>,
    t: Value,
    in_half: Value,
    out_half: Value,
) -> BResult<[Value; 3]> {
    let zero = k(fb, 0)?;
    let count = var(fb, zero)?;
    let o = var(fb, zero)?;
    let i = var(fb, zero)?;
    let kind = var(fb, zero)?;
    let at_o = var(fb, zero)?;
    let at_i = var(fb, zero)?;
    let steps = k(fb, 33)?;
    for_count(fb, steps, move |fb, j| {
        let f = final_position(fb, j)?;
        let n = get(fb, count)?;
        let ov = get(fb, o)?;
        let iv = get(fb, i)?;
        // An inside step (never at j = 0), then an outside step.
        let inner = lt(fb, f, in_half)?;
        let past0 = nek(fb, j, 0)?;
        let inner = land(fb, inner, past0)?;
        let hit = eq(fb, n, t)?;
        let hit = land(fb, hit, inner)?;
        let kv = get(fb, kind)?;
        let two = k(fb, 2)?;
        let kv = sel(fb, hit, two, kv)?;
        let ao = get(fb, at_o)?;
        let ao = sel(fb, hit, ov, ao)?;
        let ai = get(fb, at_i)?;
        let ai = sel(fb, hit, iv, ai)?;
        let step = bit(fb, inner)?;
        let n = add(fb, n, step)?;
        let iv = add(fb, iv, step)?;
        let outer = lt(fb, f, out_half)?;
        let hit = eq(fb, n, t)?;
        let hit = land(fb, hit, outer)?;
        let one = k(fb, 1)?;
        let kv = sel(fb, hit, one, kv)?;
        let ao = sel(fb, hit, ov, ao)?;
        let ai = sel(fb, hit, iv, ai)?;
        let step = bit(fb, outer)?;
        let n = add(fb, n, step)?;
        let ov = add(fb, ov, step)?;
        set(fb, count, n)?;
        set(fb, o, ov)?;
        set(fb, i, iv)?;
        set(fb, kind, kv)?;
        set(fb, at_o, ao)?;
        set(fb, at_i, ai)
    })?;
    Ok([get(fb, kind)?, get(fb, at_o)?, get(fb, at_i)?])
}

/// The id of outer edge `edge`'s walk point `w` (`edge` a `uint`).
fn outer_id(
    fb: &mut FunctionBuilder<'_>,
    facts: &Facts,
    edge: Value,
    last: Value,
    w: Value,
) -> BResult<Value> {
    let forward = if facts.quad() {
        let one = eqk(fb, edge, 1)?;
        let two = eqk(fb, edge, 2)?;
        lor(fb, one, two)?
    } else {
        eqk(fb, edge, 1)?
    };
    let back = sub(fb, last, w)?;
    let q = sel(fb, forward, w, back)?;
    let e = shl(fb, edge, 16)?;
    let id = or(fb, e, q)?;
    ork(fb, id, ID_OUTER)
}

/// The id of a quad's grid point `(a, b)`.
fn grid_id(fb: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let a = shl(fb, a, 8)?;
    let id = or(fb, a, b)?;
    ork(fb, id, ID_INNER)
}

/// The id of a triangle's inside ring `ring`'s edge `edge` walk point `w` (the reference's
/// numbering: an edge's last point is the next edge's first; a ring of one point per edge is
/// the center).
fn ring_id(
    fb: &mut FunctionBuilder<'_>,
    n: Value,
    edge: Value,
    ring: Value,
    w: Value,
) -> BResult<Value> {
    let twice = shl(fb, ring, 1)?;
    let points = sub(fb, n, twice)?;
    let center = eqk(fb, points, 1)?;
    let last = subk(fb, points, 1)?;
    let wraps = eq(fb, w, last)?;
    let next = addk(fb, edge, 1)?;
    let three = eqk(fb, next, 3)?;
    let zero = k(fb, 0)?;
    let next = sel(fb, three, zero, next)?;
    let edge = sel(fb, wraps, next, edge)?;
    let w = sel(fb, wraps, zero, w)?;
    let end = subk(fb, n, 1)?;
    let end = sub(fb, end, ring)?;
    let back = sub(fb, end, w)?;
    let ahead = add(fb, ring, w)?;
    let forward = eqk(fb, edge, 1)?;
    let q = sel(fb, forward, ahead, back)?;
    let e = shl(fb, edge, 16)?;
    let r = shl(fb, ring, 8)?;
    let id = or(fb, e, r)?;
    let id = or(fb, id, q)?;
    let id = ork(fb, id, ID_INNER)?;
    let center_id = k(fb, ID_FIXED | 4)?;
    sel(fb, center, center_id, id)
}

/// The id of the transition ring's inside walk point `w` on edge `edge`.
fn inner_id(
    fb: &mut FunctionBuilder<'_>,
    facts: &Facts,
    i: &Info,
    edge: Value,
    w: Value,
) -> BResult<Value> {
    if facts.quad() {
        let end_u = subk(fb, i.inside[0].points, 2)?;
        let end_v = subk(fb, i.inside[1].points, 2)?;
        let one = k(fb, 1)?;
        let down = sub(fb, end_v, w)?;
        let right = addk(fb, w, 1)?;
        let left = sub(fb, end_u, w)?;
        let a = pick(fb, edge, one, [right, end_u, left])?;
        let b = pick(fb, edge, down, [one, right, end_v])?;
        grid_id(fb, a, b)
    } else {
        let one = k(fb, 1)?;
        ring_id(fb, i.inside[0].points, edge, one, w)
    }
}

/// Triangle `l` of the transition ring.
fn decode_ring(
    fb: &mut FunctionBuilder<'_>,
    facts: &Facts,
    i: &Info,
    l: Value,
) -> BResult<[Value; 3]> {
    let (ne, _) = facts.counts();
    let edges = i.edges(facts);
    // The edge and the triangle within it.
    let mut edge = k(fb, 0)?;
    let mut base = k(fb, 0)?;
    let mut start = k(fb, 0)?;
    for (e, &(outer, inside)) in edges.iter().take(ne).enumerate() {
        let n = ring_edge_triangles(fb, outer, inside)?;
        let next = add(fb, start, n)?;
        if e + 1 < ne {
            let past = fb.binary(BinOp::UGreaterThanEqual, l, next)?;
            let ev = k(fb, e as u32 + 1)?;
            edge = sel(fb, past, ev, edge)?;
            base = sel(fb, past, next, base)?;
        }
        start = next;
    }
    let t = sub(fb, l, base)?;
    let [first, rest @ ..] = edges;
    let field = |fb: &mut FunctionBuilder<'_>, get: fn(&(Factor, Factor)) -> Value| {
        pick(fb, edge, get(&first), rest.iter().take(ne - 1).map(get))
    };
    let o_points = field(fb, |(o, _)| o.points)?;
    let o_half = field(fb, |(o, _)| o.num_half)?;
    let o_odd = field(fb, |(o, _)| o.odd)?;
    let i_points = field(fb, |(_, i)| i.points)?;
    let i_half = field(fb, |(_, i)| i.num_half)?;
    let i_odd = field(fb, |(_, i)| i.odd)?;
    let ob = bit(fb, o_odd)?;
    let out_half = sub(fb, o_half, ob)?;
    let ib = bit(fb, i_odd)?;
    let in_half = sub(fb, i_half, ib)?;
    let one = k(fb, 1)?;
    let in_steps = umax(fb, in_half, one)?;
    let in_steps = subk(fb, in_steps, 1)?;
    let first_half = add(fb, out_half, in_steps)?;
    // The middle: the quad (2) when both are odd, one triangle when the parities differ.
    let same = fb.binary(BinOp::LogicalEqual, i_odd, o_odd)?;
    let quad_mid = land(fb, same, i_odd)?;
    let differ = lnot(fb, same)?;
    let two = k(fb, 2)?;
    let zero = k(fb, 0)?;
    let mid = sel(fb, quad_mid, two, zero)?;
    let mid = sel(fb, differ, one, mid)?;
    let in_first = lt(fb, t, first_half)?;
    let past_mid = add(fb, first_half, mid)?;
    let in_mid = lt(fb, t, past_mid)?;
    let not_first = lnot(fb, in_first)?;
    let in_mid = land(fb, in_mid, not_first)?;
    let second = sub(fb, t, past_mid)?;
    let last = subk(fb, first_half, 1)?;
    let mirrored = sub(fb, last, second)?;
    let walked = sel(fb, in_first, t, mirrored)?;
    let [wk, wo, wi] = walk(fb, walked, in_half, out_half)?;
    // The second half's positions, from its ends.
    let end_o = subk(fb, o_points, 1)?;
    let end_i = subk(fb, i_points, 3)?;
    let is_out = eqk(fb, wk, 1)?;
    let back_o = sub(fb, end_o, wo)?;
    let o_less = bit(fb, is_out)?;
    let back_o = sub(fb, back_o, o_less)?;
    let back_i = sub(fb, end_i, wi)?;
    let is_in = eqk(fb, wk, 2)?;
    let i_less = bit(fb, is_in)?;
    let back_i = sub(fb, back_i, i_less)?;
    let o = sel(fb, in_first, wo, back_o)?;
    let iw = sel(fb, in_first, wi, back_i)?;
    // The middle's kinds: 3 then 4, 5 (inside even), 6 (inside odd).
    let at = sub(fb, t, first_half)?;
    let second_tri = eqk(fb, at, 1)?;
    let quad_kind = selk(fb, second_tri, 4, 3)?;
    let single = selk(fb, i_odd, 6, 5)?;
    let mid_kind = sel(fb, quad_mid, quad_kind, single)?;
    let kind = sel(fb, in_mid, mid_kind, wk)?;
    let o = sel(fb, in_mid, out_half, o)?;
    let iw = sel(fb, in_mid, in_steps, iw)?;
    // Corners by kind: 1 and 5 (o, i, o+1); 4 (o, i+1, o+1); 2, 3 and 6 (i, i+1, o).
    let last_o = subk(fb, o_points, 1)?;
    let o1 = addk(fb, o, 1)?;
    let i1 = addk(fb, iw, 1)?;
    let po = outer_id(fb, facts, edge, last_o, o)?;
    let po1 = outer_id(fb, facts, edge, last_o, o1)?;
    let pi = inner_id(fb, facts, i, edge, iw)?;
    let pi1 = inner_id(fb, facts, i, edge, i1)?;
    let k1 = eqk(fb, kind, 1)?;
    let k5 = eqk(fb, kind, 5)?;
    let k4 = eqk(fb, kind, 4)?;
    let outside_first = lor(fb, k1, k5)?;
    let outside_first = lor(fb, outside_first, k4)?;
    let c0 = sel(fb, outside_first, po, pi)?;
    let second_in = sel(fb, k4, pi1, pi)?;
    let c1 = sel(fb, outside_first, second_in, pi1)?;
    let c2 = sel(fb, outside_first, po1, po)?;
    Ok([c0, c1, c2])
}

/// Cell `x` of an `r` by `c` region drawn in `th` by `tw` tiles: tiles row-major, each tile's
/// cells row-major.
fn tiled_cell(
    fb: &mut FunctionBuilder<'_>,
    (r, c): (Value, Value),
    (th, tw): (Value, Value),
    x: Value,
) -> BResult<(Value, Value)> {
    // Divisors at least 1: an empty region's path is computed and discarded.
    let one = k(fb, 1)?;
    let band_cells = mul(fb, th, c)?;
    let band_cells = umax(fb, band_cells, one)?;
    let band = udiv(fb, x, band_cells)?;
    let ti = mul(fb, band, th)?;
    let before = mul(fb, band, band_cells)?;
    let rest = sub(fb, x, before)?;
    let left = sub(fb, r, ti)?;
    let h = umin(fb, th, left)?;
    let tile_cells = mul(fb, h, tw)?;
    let tile_cells = umax(fb, tile_cells, one)?;
    let col = udiv(fb, rest, tile_cells)?;
    let tj = mul(fb, col, tw)?;
    let before = mul(fb, col, tile_cells)?;
    let rest = sub(fb, rest, before)?;
    let right = sub(fb, c, tj)?;
    let w = umin(fb, tw, right)?;
    let w = umax(fb, w, one)?;
    let di = udiv(fb, rest, w)?;
    let dj = umod(fb, rest, w)?;
    Ok((add(fb, ti, di)?, add(fb, tj, dj)?))
}

/// Region `index`'s tile shape from the tile word: whole, or tiles of the coded side.
fn region_shape(
    fb: &mut FunctionBuilder<'_>,
    (r, c): (Value, Value),
    index: Value,
    tiles: Value,
) -> BResult<(Value, Value)> {
    let shift = mulk(fb, index, 5)?;
    let code = fb.binary(BinOp::ShiftRightLogical, tiles, shift)?;
    let code = andk(fb, code, 0x1f)?;
    let t = shl(fb, code, 1)?;
    let t = addk(fb, t, 1)?;
    let (th, tw) = tile_shape(fb, r, c, t)?;
    let whole = eqk(fb, code, 0)?;
    Ok((sel(fb, whole, r, th)?, sel(fb, whole, c, tw)?))
}

/// The region holding interior triangle `l` (past the center): its index, size and the
/// triangle's index within it.
fn find_region(
    fb: &mut FunctionBuilder<'_>,
    regions: &[(Value, Value)],
    l: Value,
) -> BResult<(Value, (Value, Value), Value)> {
    let zero = k(fb, 0)?;
    let mut index = zero;
    let mut base = zero;
    let (mut r, mut c) = (zero, zero);
    let mut start = zero;
    for (n, &(rr, cc)) in regions.iter().enumerate() {
        if n == 0 {
            (r, c) = (rr, cc);
        } else {
            // Past the earlier regions: in this one.
            let past = fb.binary(BinOp::UGreaterThanEqual, l, start)?;
            let iv = k(fb, n as u32)?;
            index = sel(fb, past, iv, index)?;
            base = sel(fb, past, start, base)?;
            r = sel(fb, past, rr, r)?;
            c = sel(fb, past, cc, c)?;
        }
        let cells = mul(fb, rr, cc)?;
        let tris = shl(fb, cells, 1)?;
        start = add(fb, start, tris)?;
    }
    Ok((index, (r, c), sub(fb, l, base)?))
}

/// A quad's interior triangle `l` (past the transition ring).
fn decode_quad_interior(
    fb: &mut FunctionBuilder<'_>,
    i: &Info,
    l: Value,
    tiles: Value,
) -> BResult<[Value; 3]> {
    let (nu, nv) = grid(fb, i)?;
    let hu = shr(fb, nu, 1)?;
    let hv = shr(fb, nv, 1)?;
    let cu = sub(fb, nu, hu)?;
    let cv = sub(fb, nv, hv)?;
    let both = fb.binary(BinOp::BitwiseAnd, nu, nv)?;
    let both = andk(fb, both, 1)?;
    let center = eqk(fb, both, 1)?;
    let two = selk(fb, center, 2, 0)?;
    let at_center = lt(fb, l, two)?;
    let past = sub(fb, l, two)?;
    let regions = vec![
        (hu, sub(fb, nv, hv)?),
        (hv, cu),
        (sub(fb, nu, cu)?, cv),
        (sub(fb, nv, cv)?, sub(fb, nu, hu)?),
    ];
    let (index, rc, lr) = find_region(fb, &regions, past)?;
    let shape = region_shape(fb, rc, index, tiles)?;
    let x = shr(fb, lr, 1)?;
    let (ci, cj) = tiled_cell(fb, rc, shape, x)?;
    // The quadrant's cell: Q1 (hu-1-i, hv+j), Q2 (cu-1-j, hv-1-i), Q3 (cu+i, cv-1-j),
    // Q4 (hu+j, cv+i).
    let hu1 = subk(fb, hu, 1)?;
    let q1u = sub(fb, hu1, ci)?;
    let q1v = add(fb, hv, cj)?;
    let cu1 = subk(fb, cu, 1)?;
    let q2u = sub(fb, cu1, cj)?;
    let hv1 = subk(fb, hv, 1)?;
    let q2v = sub(fb, hv1, ci)?;
    let q3u = add(fb, cu, ci)?;
    let cv1 = subk(fb, cv, 1)?;
    let q3v = sub(fb, cv1, cj)?;
    let q4u = add(fb, hu, cj)?;
    let q4v = add(fb, cv, ci)?;
    let u0 = pick(fb, index, q1u, [q2u, q3u, q4u])?;
    let v0 = pick(fb, index, q1v, [q2v, q3v, q4v])?;
    let zero = k(fb, 0)?;
    let q = sel(fb, at_center, zero, index)?;
    let u0 = sel(fb, at_center, hu, u0)?;
    let v0 = sel(fb, at_center, hv, v0)?;
    let half = andk(fb, l, 1)?;
    let half = eqk(fb, half, 1)?;
    // Corner c is the cell's corner `cyc[q + s_c]`, `cyc` = (0,0), (1,0), (1,1), (0,1) and
    // `s` = (0, 3, 1) for the first triangle, (1, 3, 2) for the second.
    let mut out = [zero; 3];
    for (slot, (s0, s1)) in out.iter_mut().zip([(0u32, 1u32), (3, 3), (1, 2)]) {
        let s = selk(fb, half, s1, s0)?;
        let x = add(fb, q, s)?;
        let x = andk(fb, x, 3)?;
        let x1 = addk(fb, x, 1)?;
        let du = shr(fb, x1, 1)?;
        let du = andk(fb, du, 1)?;
        let dv = shr(fb, x, 1)?;
        let a = add(fb, u0, du)?;
        let a = addk(fb, a, 1)?;
        let b = add(fb, v0, dv)?;
        let b = addk(fb, b, 1)?;
        *slot = grid_id(fb, a, b)?;
    }
    Ok(out)
}

/// A triangle's interior triangle `l` (past the transition ring).
fn decode_tri_interior(
    fb: &mut FunctionBuilder<'_>,
    i: &Info,
    l: Value,
    tiles: Value,
) -> BResult<[Value; 3]> {
    let n = i.inside[0].points;
    let rings = tri_rings(fb, i)?;
    let odd = bit(fb, i.inside[0].odd)?;
    let zero = k(fb, 0)?;
    let one = k(fb, 1)?;
    let two = k(fb, 2)?;
    // The center triangle (odd), first: the innermost ring's edge starts.
    let at_center = lt(fb, l, odd)?;
    let innermost = shr(fb, n, 1)?;
    let innermost = subk(fb, innermost, 1)?;
    let alone = eqk(fb, rings, 0)?;
    let past = sub(fb, l, odd)?;
    let c = add(fb, rings, odd)?;
    let cells = mul(fb, rings, c)?;
    let tris = shl(fb, cells, 1)?;
    let tris = umax(fb, tris, one)?;
    let s = udiv(fb, past, tris)?;
    let s = umin(fb, s, two)?;
    let before = mul(fb, s, tris)?;
    let lr = sub(fb, past, before)?;
    let shape = region_shape(fb, (rings, c), s, tiles)?;
    let x = shr(fb, lr, 1)?;
    let (ci, cj) = tiled_cell(fb, (rings, c), shape, x)?;
    // The cell's half: the parity of the triangle within its sector.
    let lr_half = andk(fb, lr, 1)?;
    let lr_half = eqk(fb, lr_half, 1)?;
    let shifted = fb.binary(BinOp::UGreaterThanEqual, cj, odd)?;
    let cj_less = sub(fb, cj, odd)?;
    let cj_less = sel(fb, shifted, cj_less, zero)?;
    let kk = umax(fb, ci, cj_less)?;
    let numring = addk(fb, n, 1)?;
    let numring = shr(fb, numring, 1)?;
    let r = subk(fb, numring, 1)?;
    let r = sub(fb, r, kk)?;
    let r_out = subk(fb, r, 1)?;
    let prev = addk(fb, s, 2)?;
    let three = k(fb, 3)?;
    let prev = umod(fb, prev, three)?;
    let row_k = eq(fb, ci, kk)?;
    let kodd = add(fb, kk, odd)?;
    let col_k = eq(fb, cj, kodd)?;
    let corner = land(fb, row_k, col_k)?;
    let off_corner = lnot(fb, col_k)?;
    let start = land(fb, row_k, off_corner)?;
    // Start: edge s, quad k + odd - 1 - j; end: edge s - 1, quad k + odd + i.
    let m_start = subk(fb, kodd, 1)?;
    let m_start = sub(fb, m_start, cj)?;
    let m_end = add(fb, kodd, ci)?;
    let m = sel(fb, start, m_start, m_end)?;
    let e = sel(fb, start, s, prev)?;
    let m1 = addk(fb, m, 1)?;
    let m2 = addk(fb, m, 2)?;
    // The corner's last outside step: the outside ring's edge points less 2.
    let twice = shl(fb, r_out, 1)?;
    let o_points = sub(fb, n, twice)?;
    let o_before_last = subk(fb, o_points, 2)?;
    // (edge, ring, walk) per corner.
    let choose = |fb: &mut FunctionBuilder<'_>,
                  corner_v: [(Value, Value, Value); 2],
                  start_v: [(Value, Value, Value); 2],
                  end_v: [(Value, Value, Value); 2]|
     -> BResult<(Value, Value, Value)> {
        let pick3 = |fb: &mut FunctionBuilder<'_>,
                     a: (Value, Value, Value),
                     b: (Value, Value, Value)|
         -> BResult<(Value, Value, Value)> {
            Ok((
                sel(fb, lr_half, b.0, a.0)?,
                sel(fb, lr_half, b.1, a.1)?,
                sel(fb, lr_half, b.2, a.2)?,
            ))
        };
        let cv = pick3(fb, corner_v[0], corner_v[1])?;
        let sv = pick3(fb, start_v[0], start_v[1])?;
        let ev = pick3(fb, end_v[0], end_v[1])?;
        Ok((
            {
                let x = sel(fb, start, sv.0, ev.0)?;
                sel(fb, corner, cv.0, x)?
            },
            {
                let x = sel(fb, start, sv.1, ev.1)?;
                sel(fb, corner, cv.1, x)?
            },
            {
                let x = sel(fb, start, sv.2, ev.2)?;
                sel(fb, corner, cv.2, x)?
            },
        ))
    };
    // Corner: (o(s,1), o(s,0), i(s,0)), (i(s,0), o(s,0), o(s-1, last-1)).
    // Start: (o(m+2), o(m+1), i(m+1)), (i(m+1), o(m+1), i(m)).
    // End: (i(m+1), o(m+2), i(m)), (i(m), o(m+2), o(m+1)).
    let corners = [
        choose(
            fb,
            [(s, r_out, one), (s, r, zero)],
            [(e, r_out, m2), (e, r, m1)],
            [(e, r, m1), (e, r, m)],
        )?,
        choose(
            fb,
            [(s, r_out, zero), (s, r_out, zero)],
            [(e, r_out, m1), (e, r_out, m1)],
            [(e, r_out, m2), (e, r_out, m2)],
        )?,
        choose(
            fb,
            [(s, r, zero), (prev, r_out, o_before_last)],
            [(e, r, m1), (e, r, m)],
            [(e, r, m), (e, r_out, m1)],
        )?,
    ];
    let mut out = [zero; 3];
    for (slot, &(e, ring, w)) in out.iter_mut().zip(&corners) {
        *slot = ring_id(fb, n, e, ring, w)?;
    }
    // The center: (c1, c0, c2) alone, else (c2, c1, c0).
    let c0 = ring_id(fb, n, zero, innermost, zero)?;
    let c1 = ring_id(fb, n, one, innermost, zero)?;
    let c2 = ring_id(fb, n, two, innermost, zero)?;
    let a0 = sel(fb, alone, c1, c2)?;
    let a1 = sel(fb, alone, c0, c1)?;
    let a2 = sel(fb, alone, c2, c0)?;
    Ok([
        sel(fb, at_center, a0, out[0])?,
        sel(fb, at_center, a1, out[1])?,
        sel(fb, at_center, a2, out[2])?,
    ])
}

/// The points of triangle `l` of a patch of processed factors `p` and tile word `tiles`, its
/// first corner first.
pub fn emit_decode(
    fb: &mut FunctionBuilder<'_>,
    facts: &Facts,
    l: Value,
    p: &[Value; 6],
    tiles: Value,
) -> BResult<[Value; 3]> {
    let i = info(fb, facts, p)?;
    let facts_ = *facts;
    let ring = ring_triangles(fb, facts, &i)?;
    let in_ring = lt(fb, l, ring)?;
    let corners = branch(
        fb,
        i.minimum,
        move |fb| {
            // The reference's 2 (quad) or 1 (triangle) triangles.
            let c = |fb: &mut FunctionBuilder<'_>, u: u32, v: u32| k(fb, ID_FIXED | u | (v << 1));
            if facts_.quad() {
                let first = eqk(fb, l, 0)?;
                let c00 = c(fb, 0, 0)?;
                let c01 = c(fb, 0, 1)?;
                let c10 = c(fb, 1, 0)?;
                let c11 = c(fb, 1, 1)?;
                Ok([sel(fb, first, c00, c01)?, sel(fb, first, c01, c11)?, c10])
            } else {
                Ok([c(fb, 0, 0)?, c(fb, 0, 1)?, c(fb, 1, 0)?])
            }
        },
        move |fb| {
            branch(
                fb,
                in_ring,
                move |fb| decode_ring(fb, &facts_, &i, l),
                move |fb| {
                    let past = sub(fb, l, ring)?;
                    if facts_.quad() {
                        decode_quad_interior(fb, &i, past, tiles)
                    } else {
                        decode_tri_interior(fb, &i, past, tiles)
                    }
                },
            )
        },
    )?;
    Ok(match facts.winding {
        Winding::CounterClockwise => [corners[0], corners[2], corners[1]],
        _ => corners,
    })
}

// ---- Place ----

/// The domain coordinates of point `id` of a patch of processed factors `p`: `(u, v, 0)` for a
/// quad, `(u, v, w)` for a triangle, `w = (1 - u) - v`; each a 16.16 location converted exactly.
pub fn emit_place(
    fb: &mut FunctionBuilder<'_>,
    facts: &Facts,
    id: Value,
    p: &[Value; 6],
) -> BResult<[Value; 3]> {
    let (ne, _) = facts.counts();
    let kind = shr(fb, id, 30)?;
    let is_outer = eqk(fb, kind, 1)?;
    let is_fixed = eqk(fb, kind, 2)?;
    let edge = shr(fb, id, 16)?;
    let edge = andk(fb, edge, 3)?;
    let q16 = andk(fb, id, 0xffff)?;
    let hi8 = shr(fb, id, 8)?;
    let hi8 = andk(fb, hi8, 0xff)?;
    let lo8 = andk(fb, id, 0xff)?;
    // The first location: an outer point along its edge's factor; a quad's grid point along `u`;
    // a triangle's ring point along the inside factor.
    let [e0, e1, e2, e3, i0, i1] = *p;
    // A triangle's inside factor follows its three edges.
    let inside_f = if facts.quad() { i0 } else { e3 };
    let edge_f = pick(fb, edge, e0, [e1, e2, e3].into_iter().take(ne - 1))?;
    let f1 = sel(fb, is_outer, edge_f, inside_f)?;
    let edge_odd = {
        let mut out = parity(fb, facts, e0, false)?;
        for (n, &f) in p.iter().take(ne).enumerate().skip(1) {
            let o = parity(fb, facts, f, false)?;
            let at = eqk(fb, edge, n as u32)?;
            out = sel(fb, at, o, out)?;
        }
        out
    };
    let inside_odd = parity(fb, facts, inside_f, true)?;
    let odd1 = sel(fb, is_outer, edge_odd, inside_odd)?;
    let ctx1 = context(fb, f1, odd1)?;
    let index1 = if facts.quad() {
        sel(fb, is_outer, q16, hi8)?
    } else {
        sel(fb, is_outer, q16, lo8)?
    };
    let loc1 = place_1d(fb, &ctx1, index1)?;
    let one = k(fb, ONE)?;
    let zero = k(fb, 0)?;
    let (u, v) = if facts.quad() {
        let fv = i1;
        let odd_v = parity(fb, facts, fv, true)?;
        let ctx_v = context(fb, fv, odd_v)?;
        let loc_v = place_1d(fb, &ctx_v, lo8)?;
        // Outer: edge 0 (0, l), 1 (l, 0), 2 (1, l), 3 (l, 1).
        let ou = pick(fb, edge, zero, [loc1, one, loc1])?;
        let ov = pick(fb, edge, loc1, [zero, loc1, one])?;
        let u = sel(fb, is_outer, ou, loc1)?;
        let v = sel(fb, is_outer, ov, loc_v)?;
        (u, v)
    } else {
        // A ring point: the perpendicular `round(place(ring) · 2/3)`, along less half of it.
        let ring_loc = place_1d(fb, &ctx1, hi8)?;
        let perp = mulk(fb, ring_loc, 0xaaaa)?;
        let perp = addk(fb, perp, 0x8000)?;
        let perp = shr(fb, perp, 16)?;
        let half = addk(fb, perp, 1)?;
        let half = shr(fb, half, 1)?;
        let along = sub(fb, loc1, half)?;
        let rest = ksub(fb, ONE, along)?;
        let rest = sub(fb, rest, perp)?;
        let iu = pick(fb, edge, perp, [along, along])?;
        let iv = pick(fb, edge, along, [perp, rest])?;
        // Outer: edge 0 (0, l), 1 (l, 0), 2 (l, 1 - l).
        let back = ksub(fb, ONE, loc1)?;
        let ou = pick(fb, edge, zero, [loc1, loc1])?;
        let ov = pick(fb, edge, loc1, [zero, back])?;
        (sel(fb, is_outer, ou, iu)?, sel(fb, is_outer, ov, iv)?)
    };
    // Fixed points: a corner's bits, or a triangle's center.
    let bu = andk(fb, id, 1)?;
    let bu = nek(fb, bu, 0)?;
    let bv = andk(fb, id, 2)?;
    let bv = nek(fb, bv, 0)?;
    let center = andk(fb, id, 4)?;
    let center = nek(fb, center, 0)?;
    let third = k(fb, 0x5555)?;
    let fu = sel(fb, bu, one, zero)?;
    let fu = sel(fb, center, third, fu)?;
    let fv = sel(fb, bv, one, zero)?;
    let fv = sel(fb, center, third, fv)?;
    let u = sel(fb, is_fixed, fu, u)?;
    let v = sel(fb, is_fixed, fv, v)?;
    let f32_ = float(fb)?;
    let scale = kf(fb, 1.0 / 65536.0)?;
    let uf = fb.convert(ConvOp::ConvertUToF, u, f32_)?;
    let uf = fb.binary(BinOp::FMul, uf, scale)?;
    let vf = fb.convert(ConvOp::ConvertUToF, v, f32_)?;
    let vf = fb.binary(BinOp::FMul, vf, scale)?;
    let w = if facts.quad() {
        kf(fb, 0.0)?
    } else {
        let one_f = kf(fb, 1.0)?;
        let w = fb.binary(BinOp::FSub, one_f, uf)?;
        fb.binary(BinOp::FSub, w, vf)?
    };
    Ok([uf, vf, w])
}
