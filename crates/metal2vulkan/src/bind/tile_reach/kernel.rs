//! The tile reach kernel: [`typed_module`], which [`tile_reach_module`](super::tile_reach_module)
//! serializes and the CPU interpreter runs. It bins a pass's primitives into the tile mask by the
//! rule of the module docs, step for step as the reference function the tests hold computes it
//! (`tests/reference.rs`, whose names each step below gives).
//!
//! **Exactness.** The kernel has no float instruction: it holds every `f32` as its bits (`u32`)
//! and computes the window transform's and the clipper's arithmetic in integers, as IEEE defines
//! it ([`quotient`], [`product`], [`sum`]: each rounded once to nearest even, subnormal operands
//! and results exact, Rust's `/`, `*`, `+` and `-`), and the comparisons from the bits
//! ([`negative`], [`nonnegative`], [`nonpositive`], [`exceeds`]). So a device that flushes
//! subnormals to zero, contracts a multiply and an add, or compiles with fast math computes what
//! the reference does (lavapipe flushes: a vertex at `z = -1e-40` crosses the near plane, which
//! it did not see). The snap to 1/256 of a pixel reads the window position's bits ([`snap`]), and
//! every test after it is in 64-bit integers, which the coordinate limit (2^29 units, the render
//! area at most 16384 pixels a side) keeps from overflowing.
//!
//! **No integer division.** Compilers lower a 64-bit division into a long chain of 64-bit steps,
//! and a kernel holding many of them grows past the registers of a device, spilling (RADV's
//! compiler then got some of its lines' reaches wrong in wave64). So the kernel divides nothing
//! but by constants: the significands' quotient is found a bit at a time in 32-bit words
//! ([`significand_quotient`]), the tiles a box meets by shifts (a tile's sides are powers of two),
//! a line's box over its length by products ([`small_quotients`]), and the small-primitive cull
//! tests each sample position its box holds ([`holds_sample`]).

use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, GlslOp,
    IoDir, LayoutRule, MemAccess, Scope, Semantics, StorageClass, Type, TypedModule, UnOp,
};

use super::{layout, TileReachModule as M, AREA_LIMIT};

/// The binding symbols of the primitives, the pass and the tile mask in the typed module.
pub(super) const PRIMITIVES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
pub(super) const PASS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
pub(super) const TILES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// Units of a pixel a vertex snaps to.
const UNIT: i64 = 256;
/// The largest snapped coordinate: past it a coordinate is clamped and flagged unmeasured.
const LIMIT: u32 = 1 << 29;
/// The most vertices the clipper holds: 9, the most a triangle's polygon reaches. Sutherland-Hodgman
/// keeps each vertex on the plane's inner side and adds one for each edge whose ends lie on
/// opposite sides. Around a closed polygon of `n` vertices, `k` of them inside, the side changes
/// at most `2 min(k, n - k)` times, so the result holds at most `k + 2 min(k, n - k)` vertices:
/// at most 4 from 3, 6 from 4 and 9 from 6. The planes are at most three (near, far, eye), so a
/// triangle's polygon holds at most 9 and a segment's 2. The count looks only at which side each
/// vertex lies on, so it holds whatever the rounding of the vertices the clipper makes, even where
/// that rounding leaves a polygon not convex (exact arithmetic gives at most 6). The test
/// `the_clipper_holds_its_most_vertices` clips a hexagon to 9.
pub(super) const POLYGON: u32 = 9;
/// The largest point size Metal draws, in pixels: a larger size draws as this one.
const POINT_SIZE_LIMIT: f32 = 511.0;

/// The words of the per-primitive context `bin` reads (a `Private` array of `i64`): the region,
/// the tile's width and height in units, the tiles of a row and of a column, the first row of
/// tiles of the primitive's slice, and the tile's width and height in units as powers of two.
mod ctx {
    pub(super) const RX0: u32 = 0;
    pub(super) const RX1: u32 = 1;
    pub(super) const RY0: u32 = 2;
    pub(super) const RY1: u32 = 3;
    pub(super) const TW: u32 = 4;
    pub(super) const TH: u32 = 5;
    pub(super) const TILES_X: u32 = 6;
    pub(super) const TILES_Y: u32 = 7;
    pub(super) const BASE: u32 = 8;
    pub(super) const SHIFT_X: u32 = 9;
    pub(super) const SHIFT_Y: u32 = 10;
    pub(super) const WORDS: u32 = 11;
}

/// The kernel's types.
#[derive(Clone, Copy)]
struct T {
    bool_: TypeId,
    u32_: TypeId,
    u64_: TypeId,
    i64_: TypeId,
    /// A clip-space vertex: its four `f32`s' bits (`uvec4`).
    vec4: TypeId,
}

/// What the main function's parts take: the types, the routines, the buffers, the flags and the
/// clipper's arrays (pointers), and the primitive's state.
#[derive(Clone, Copy)]
struct K {
    t: T,
    r: Routines,
    near_far: Value,
    unmeasured: Value,
    poly: Value,
    scratch: Value,
    n: Value,
    pts: Value,
    /// The viewport's origin and size (`f32` bits).
    vp: [Value; 4],
    /// Whether depth clamps rather than clips (`bool`).
    clamp: Value,
    /// The raw cull mode and winding, and the pass's samples (`u32`).
    cull: Value,
    winding: Value,
    samples: Value,
}

fn k32(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

fn k64(f: &mut FunctionBuilder<'_>, t: T, v: i64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: t.i64_,
        bits: ConstBits(v as u64),
    })?;
    f.module().const_value(c)
}

fn ku64(f: &mut FunctionBuilder<'_>, t: T, v: u64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: t.u64_,
        bits: ConstBits(v),
    })?;
    f.module().const_value(c)
}

/// The `bool` `false`.
fn falsy(f: &mut FunctionBuilder<'_>, t: T) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: t.bool_,
        bits: ConstBits(0),
    })?;
    f.module().const_value(c)
}

fn op(f: &mut FunctionBuilder<'_>, o: BinOp, a: Value, b: Value) -> BResult<Value> {
    f.binary(o, a, b)
}

/// `a o v` for a `u32` constant `v`.
fn opk(f: &mut FunctionBuilder<'_>, o: BinOp, a: Value, v: u32) -> BResult<Value> {
    let c = k32(f, v)?;
    f.binary(o, a, c)
}

/// `a o v` for an `i64` constant `v`.
fn op64(f: &mut FunctionBuilder<'_>, t: T, o: BinOp, a: Value, v: i64) -> BResult<Value> {
    let c = k64(f, t, v)?;
    f.binary(o, a, c)
}

fn sel(f: &mut FunctionBuilder<'_>, c: Value, a: Value, b: Value) -> BResult<Value> {
    f.select(c, a, b)
}

fn and(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    f.binary(BinOp::LogicalAnd, a, b)
}

fn or(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    f.binary(BinOp::LogicalOr, a, b)
}

fn not(f: &mut FunctionBuilder<'_>, a: Value) -> BResult<Value> {
    f.unary(UnOp::LogicalNot, a)
}

/// The smaller of two `i64`s.
fn smin(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let less = f.binary(BinOp::SLessThan, a, b)?;
    f.select(less, a, b)
}

/// The larger of two `i64`s.
fn smax(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let more = f.binary(BinOp::SGreaterThan, a, b)?;
    f.select(more, a, b)
}

/// The magnitude of an `i64`.
fn sabs(f: &mut FunctionBuilder<'_>, t: T, a: Value) -> BResult<Value> {
    let zero = k64(f, t, 0)?;
    let neg = f.binary(BinOp::SLessThan, a, zero)?;
    let minus = f.unary(UnOp::SNegate, a)?;
    f.select(neg, minus, a)
}

/// A `u32` as an `i64` of its value.
fn widen(f: &mut FunctionBuilder<'_>, t: T, a: Value) -> BResult<Value> {
    let u = f.convert(ConvOp::UConvert, a, t.u64_)?;
    f.convert(ConvOp::Bitcast, u, t.i64_)
}

/// The low 32 bits of an `i64` or a `u64`, as a `u32`.
fn low(f: &mut FunctionBuilder<'_>, t: T, a: Value) -> BResult<Value> {
    let u = match f.value_ty(a)? == t.i64_ {
        true => f.convert(ConvOp::Bitcast, a, t.u64_)?,
        false => a,
    };
    f.convert(ConvOp::UConvert, u, t.u32_)
}

/// `n / 256` rounded down (`i64`).
fn floor_unit(f: &mut FunctionBuilder<'_>, n: Value) -> BResult<Value> {
    opk(f, BinOp::ShiftRightArithmetic, n, 8)
}

/// `n / 256` rounded up (`i64`).
fn ceil_unit(f: &mut FunctionBuilder<'_>, n: Value) -> BResult<Value> {
    let minus = f.unary(UnOp::SNegate, n)?;
    let q = floor_unit(f, minus)?;
    f.unary(UnOp::SNegate, q)
}

fn load(f: &mut FunctionBuilder<'_>, p: Value) -> BResult<Value> {
    f.load(p, MemAccess::default())
}

fn store(f: &mut FunctionBuilder<'_>, p: Value, v: Value) -> BResult<()> {
    f.store(p, v, MemAccess::default())
}

/// Element `i` (a `u32`) of the array a pointer points at.
fn at(f: &mut FunctionBuilder<'_>, array: Value, i: Value) -> BResult<Value> {
    f.access_chain(array, &[i])
}

/// Element `i` (a constant) of the array a pointer points at.
fn atk(f: &mut FunctionBuilder<'_>, array: Value, i: u32) -> BResult<Value> {
    let i = k32(f, i)?;
    f.access_chain(array, &[i])
}

/// Word `i` (a `u32`) of a buffer of words.
fn word(f: &mut FunctionBuilder<'_>, buffer: Value, i: Value) -> BResult<Value> {
    let zero = k32(f, 0)?;
    let p = f.access_chain(buffer, &[zero, i])?;
    load(f, p)
}

/// Set the `u32` flag a pointer points at where `c` holds.
fn flag_if(f: &mut FunctionBuilder<'_>, flag: Value, c: Value) -> BResult<()> {
    let old = load(f, flag)?;
    let one = k32(f, 1)?;
    let new = sel(f, c, one, old)?;
    store(f, flag, new)
}

/// Run `body` where `c` holds.
fn when<'a>(
    f: &mut FunctionBuilder<'_>,
    c: Value,
    body: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
) -> BResult<()> {
    let arm: ArmFn<'_, ()> = Box::new(move |f, _| {
        body(f)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), c, arm, None)?;
    Ok(())
}

/// Run `then` where `c` holds and `other` where it does not.
fn either<'a>(
    f: &mut FunctionBuilder<'_>,
    c: Value,
    then: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
    other: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
) -> BResult<()> {
    let yes: ArmFn<'_, ()> = Box::new(move |f, _| {
        then(f)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    let no: ArmFn<'_, ()> = Box::new(move |f, _| {
        other(f)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), c, yes, Some(no))?;
    Ok(())
}

/// Run `body(i)` for each `u32` `i` from `first` while `i < end` (`end` read once, before).
fn count<'a>(
    f: &mut FunctionBuilder<'_>,
    t: T,
    first: u32,
    end: Value,
    body: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<()> + 'a,
) -> BResult<()> {
    let var = f.local_var(t.u32_, None)?;
    let p = f.var_ptr(var)?;
    let first = k32(f, first)?;
    store(f, p, first)?;
    let walk: LoopFn<'_, ()> = Box::new(move |f, _, lc| {
        let i = load(f, p)?;
        let done = op(f, BinOp::UGreaterThanEqual, i, end)?;
        let stop: ArmFn<'_, ()> = Box::new(|f, _| {
            lc.break_(f)?;
            Ok(ArmEnd::Exits)
        });
        f.if_(&mut (), done, stop, None)?;
        let next = opk(f, BinOp::IAdd, i, 1)?;
        store(f, p, next)?;
        body(f, i)
    });
    f.loop_(&mut (), LoopSpec::default(), walk)
}

/// Run `body(i)` for each `i64` `i` from `lo` to `hi` inclusive.
fn span<'a>(
    f: &mut FunctionBuilder<'_>,
    t: T,
    lo: Value,
    hi: Value,
    body: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<()> + 'a,
) -> BResult<()> {
    let var = f.local_var(t.i64_, None)?;
    let p = f.var_ptr(var)?;
    store(f, p, lo)?;
    let walk: LoopFn<'_, ()> = Box::new(move |f, _, lc| {
        let i = load(f, p)?;
        let done = op(f, BinOp::SGreaterThan, i, hi)?;
        let stop: ArmFn<'_, ()> = Box::new(|f, _| {
            lc.break_(f)?;
            Ok(ArmEnd::Exits)
        });
        f.if_(&mut (), done, stop, None)?;
        let next = op64(f, t, BinOp::IAdd, i, 1)?;
        store(f, p, next)?;
        body(f, i)
    });
    f.loop_(&mut (), LoopSpec::default(), walk)
}

// ---- f32 arithmetic in integers ----

/// The exponent field of `f32` bits.
fn exponent(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let e = opk(f, BinOp::ShiftRightLogical, b, 23)?;
    opk(f, BinOp::BitwiseAnd, e, 0xff)
}

/// Whether `f32` bits are not finite.
fn not_finite(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let e = exponent(f, b)?;
    opk(f, BinOp::IEqual, e, 0xff)
}

/// Whether `f32` bits are subnormal (zero exponent, a fraction).
fn subnormal(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let e = exponent(f, b)?;
    let tiny = opk(f, BinOp::IEqual, e, 0)?;
    let fr = opk(f, BinOp::BitwiseAnd, b, 0x7f_ffff)?;
    let some = opk(f, BinOp::INotEqual, fr, 0)?;
    and(f, tiny, some)
}

/// Whether `f32` bits are a NaN.
fn nan(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let m = opk(f, BinOp::BitwiseAnd, b, 0x7fff_ffff)?;
    opk(f, BinOp::UGreaterThan, m, 0x7f80_0000)
}

/// Whether the `f32` of bits `b` is below zero (`x < 0.0`): its sign set, not a zero or a NaN.
fn negative(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let m = opk(f, BinOp::BitwiseAnd, b, 0x7fff_ffff)?;
    let some = opk(f, BinOp::INotEqual, m, 0)?;
    let number = opk(f, BinOp::ULessThanEqual, m, 0x7f80_0000)?;
    let signed = opk(f, BinOp::UGreaterThanEqual, b, 0x8000_0000)?;
    let n = and(f, some, number)?;
    and(f, n, signed)
}

/// Whether the `f32` of bits `b` is at least zero (`x >= 0.0`): neither below zero nor a NaN.
fn nonnegative(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let below = negative(f, b)?;
    let nan = nan(f, b)?;
    let out = or(f, below, nan)?;
    not(f, out)
}

/// Whether the `f32` of bits `b` is at most zero (`x <= 0.0`): below zero or a zero.
fn nonpositive(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let below = negative(f, b)?;
    let m = opk(f, BinOp::BitwiseAnd, b, 0x7fff_ffff)?;
    let zero = opk(f, BinOp::IEqual, m, 0)?;
    or(f, below, zero)
}

/// Whether the `f32` of bits `b` exceeds the positive finite `limit` (`x > limit`): positive
/// floats order as their bits.
fn exceeds(f: &mut FunctionBuilder<'_>, b: Value, limit: f32) -> BResult<Value> {
    let nan = nan(f, b)?;
    let number = not(f, nan)?;
    let above = opk(f, BinOp::ULessThan, b, 0x8000_0000)?;
    let past = opk(f, BinOp::UGreaterThan, b, limit.to_bits())?;
    let a = and(f, number, above)?;
    and(f, a, past)
}

/// The classes and parts of `f32` bits: the sign bit (`u32`, 0 or `0x8000_0000`), whether a NaN,
/// an infinity or a zero, and its significand and exponent ([`normalized`]).
#[derive(Clone, Copy)]
struct Unpacked {
    sign: Value,
    nan: Value,
    inf: Value,
    zero: Value,
    m: Value,
    e: Value,
}

fn unpack(f: &mut FunctionBuilder<'_>, t: T, b: Value) -> BResult<Unpacked> {
    let sign = opk(f, BinOp::BitwiseAnd, b, 0x8000_0000)?;
    let e = exponent(f, b)?;
    let fr = opk(f, BinOp::BitwiseAnd, b, 0x7f_ffff)?;
    let top = opk(f, BinOp::IEqual, e, 0xff)?;
    let some = opk(f, BinOp::INotEqual, fr, 0)?;
    let whole = not(f, some)?;
    let nan = and(f, top, some)?;
    let inf = and(f, top, whole)?;
    let m = opk(f, BinOp::BitwiseAnd, b, 0x7fff_ffff)?;
    let zero = opk(f, BinOp::IEqual, m, 0)?;
    let (m, e) = normalized(f, t, e, fr, zero)?;
    Ok(Unpacked {
        sign,
        nan,
        inf,
        zero,
        m,
        e,
    })
}

/// An `f32` significand and exponent of finite, non-zero bits `e` and `fr` (exponent field and
/// fraction): `(m, e')` with `m` in `[2^23, 2^24)` and the value `m * 2^(e' - 150)` (`e'` an
/// `i64`). A zero gives `2^23` and an exponent no caller reads.
fn normalized(
    f: &mut FunctionBuilder<'_>,
    t: T,
    e: Value,
    fr: Value,
    zero: Value,
) -> BResult<(Value, Value)> {
    let normal = opk(f, BinOp::INotEqual, e, 0)?;
    let implicit = opk(f, BinOp::BitwiseOr, fr, 0x80_0000)?;
    let m = sel(f, normal, implicit, fr)?;
    let one_bit = k32(f, 0x80_0000)?;
    let m = sel(f, zero, one_bit, m)?;
    let msb = f.glsl(GlslOp::FindUMsb, &[m])?;
    let k23 = k32(f, 23)?;
    let lz = op(f, BinOp::ISub, k23, msb)?;
    let m = op(f, BinOp::ShiftLeftLogical, m, lz)?;
    let one = k32(f, 1)?;
    let e = sel(f, normal, e, one)?;
    let e = widen(f, t, e)?;
    let lz = widen(f, t, lz)?;
    let e = op(f, BinOp::ISub, e, lz)?;
    Ok((m, e))
}

/// The position of the leading one of a non-zero `u64`, an `i64` (`FindUMsb` takes 32 bits).
fn msb64(f: &mut FunctionBuilder<'_>, t: T, s: Value) -> BResult<Value> {
    let hi = opk(f, BinOp::ShiftRightLogical, s, 32)?;
    let hi = low(f, t, hi)?;
    let lo = low(f, t, s)?;
    let some_hi = opk(f, BinOp::INotEqual, hi, 0)?;
    let mh = f.glsl(GlslOp::FindUMsb, &[hi])?;
    let mh = opk(f, BinOp::IAdd, mh, 32)?;
    let ml = f.glsl(GlslOp::FindUMsb, &[lo])?;
    let msb = sel(f, some_hi, mh, ml)?;
    widen(f, t, msb)
}

/// The bits of the `f32` of sign `sign` (a sign bit) and magnitude `s * 2^k` for the `u64` `s`
/// whose leading one is at `p` and whose value's biased exponent is `biased` (both `i64`; `k`
/// follows from them), rounded once to nearest even, with `sticky` (a `bool`) where the exact
/// magnitude lies past `s`'s last bit: subnormal where `biased` is not positive, infinite past the
/// largest finite value. Every caller keeps at least six bits below the rounded significand.
fn pack(
    f: &mut FunctionBuilder<'_>,
    t: T,
    sign: Value,
    s: Value,
    sticky: Value,
    p: Value,
    biased: Value,
) -> BResult<Value> {
    let zero_e = k64(f, t, 0)?;
    let one_e = k64(f, t, 1)?;
    let sub = op(f, BinOp::SLessThanEqual, biased, zero_e)?;
    let below = op(f, BinOp::ISub, one_e, biased)?;
    let extra = sel(f, sub, below, zero_e)?;
    let shift = op64(f, t, BinOp::ISub, p, 23)?;
    let shift = op(f, BinOp::IAdd, shift, extra)?;
    let k63 = k64(f, t, 63)?;
    let shift = smin(f, shift, k63)?;
    let shift = smax(f, shift, zero_e)?;
    let sh = low(f, t, shift)?;
    let mant = op(f, BinOp::ShiftRightLogical, s, sh)?;
    let one64 = ku64(f, t, 1)?;
    let zero64 = ku64(f, t, 0)?;
    let unit = op(f, BinOp::ShiftLeftLogical, one64, sh)?;
    let mask = op(f, BinOp::ISub, unit, one64)?;
    let rem = op(f, BinOp::BitwiseAnd, s, mask)?;
    let half = opk(f, BinOp::ShiftRightLogical, unit, 1)?;
    let above = op(f, BinOp::UGreaterThan, rem, half)?;
    let tie = op(f, BinOp::IEqual, rem, half)?;
    let odd = op(f, BinOp::BitwiseAnd, mant, one64)?;
    let odd = op(f, BinOp::INotEqual, odd, zero64)?;
    let tie_up = or(f, sticky, odd)?;
    let tie_up = and(f, tie, tie_up)?;
    let up = or(f, above, tie_up)?;
    let carry = sel(f, up, one64, zero64)?;
    let mant = op(f, BinOp::IAdd, mant, carry)?;
    let mant = low(f, t, mant)?;
    // A normal result's field below its significand's implicit bit, which carries into it.
    let field = op(f, BinOp::ISub, biased, one_e)?;
    let field = sel(f, sub, zero_e, field)?;
    let k300 = k64(f, t, 300)?;
    let field = smin(f, field, k300)?;
    let field = low(f, t, field)?;
    let field = opk(f, BinOp::ShiftLeftLogical, field, 23)?;
    let enc = op(f, BinOp::IAdd, field, mant)?;
    let over = opk(f, BinOp::UGreaterThanEqual, enc, 0x7f80_0000)?;
    let inf_bits = k32(f, 0x7f80_0000)?;
    let enc = sel(f, over, inf_bits, enc)?;
    op(f, BinOp::BitwiseOr, sign, enc)
}

/// `floor(mx * 2^30 / my)` of significands `mx` and `my` in `[2^23, 2^24)` (`u32`s), in
/// `(2^29, 2^31)`, and whether a remainder is left (a `bool`): restoring division, one quotient
/// bit a step from the place of `2^0` in `mx / my` (below 2) to that of `2^-30`, in 32-bit words
/// (the remainder stays below `2 my`), dividing nothing (the module docs).
fn significand_quotient(
    f: &mut FunctionBuilder<'_>,
    t: T,
    mx: Value,
    my: Value,
) -> BResult<(Value, Value)> {
    let rest = f.local_var(t.u32_, None)?;
    let rest = f.var_ptr(rest)?;
    let quot = f.local_var(t.u32_, None)?;
    let quot = f.var_ptr(quot)?;
    store(f, rest, mx)?;
    let zero = k32(f, 0)?;
    store(f, quot, zero)?;
    let steps = k32(f, 31)?;
    count(f, t, 0, steps, move |f, _| {
        let r = load(f, rest)?;
        let fits = op(f, BinOp::UGreaterThanEqual, r, my)?;
        let less = op(f, BinOp::ISub, r, my)?;
        let r = sel(f, fits, less, r)?;
        let r = opk(f, BinOp::ShiftLeftLogical, r, 1)?;
        store(f, rest, r)?;
        let (one, zero) = (k32(f, 1)?, k32(f, 0)?);
        let bit = sel(f, fits, one, zero)?;
        let q = load(f, quot)?;
        let q = opk(f, BinOp::ShiftLeftLogical, q, 1)?;
        let q = op(f, BinOp::BitwiseOr, q, bit)?;
        store(f, quot, q)
    })?;
    let q = load(f, quot)?;
    let r = load(f, rest)?;
    let left = opk(f, BinOp::INotEqual, r, 0)?;
    Ok((q, left))
}

/// The bits of the IEEE `f32` quotient `x / y` of bits `x` and `y`, rounded once to nearest
/// even, subnormal operands and results exact (Rust's `x / y`; a NaN is `0x7fc0_0000`): from the
/// operands' significands' quotient to 31 bits and whether a remainder is left
/// ([`significand_quotient`]).
fn quotient(f: &mut FunctionBuilder<'_>, t: T, x: Value, y: Value) -> BResult<Value> {
    let ux = unpack(f, t, x)?;
    let uy = unpack(f, t, y)?;
    let sign = op(f, BinOp::BitwiseXor, ux.sign, uy.sign)?;
    let nan = or(f, ux.nan, uy.nan)?;
    let infs = and(f, ux.inf, uy.inf)?;
    let zeros = and(f, ux.zero, uy.zero)?;
    let nan = or(f, nan, infs)?;
    let nan = or(f, nan, zeros)?;
    let inf = or(f, ux.inf, uy.zero)?;
    let zero = or(f, ux.zero, uy.inf)?;
    // The significands' quotient, scaled by 2^30: its leading one at 29 or 30.
    let (q, sticky) = significand_quotient(f, t, ux.m, uy.m)?;
    let q = f.convert(ConvOp::UConvert, q, t.u64_)?;
    let p = msb64(f, t, q)?;
    // The value's leading one is 2^(p - 30 + ex - ey).
    let d = op(f, BinOp::ISub, ux.e, uy.e)?;
    let biased = op(f, BinOp::IAdd, d, p)?;
    let biased = op64(f, t, BinOp::IAdd, biased, 97)?;
    let r = pack(f, t, sign, q, sticky, p, biased)?;
    special(
        f,
        sign,
        [(nan, None), (inf, Some(0x7f80_0000)), (zero, Some(0))],
        r,
    )
}

/// The bits of the IEEE `f32` product `x * y` of bits `x` and `y`, rounded once to nearest even,
/// subnormal operands and results exact (Rust's `x * y`): from the 48-bit product of the
/// significands.
fn product(f: &mut FunctionBuilder<'_>, t: T, x: Value, y: Value) -> BResult<Value> {
    let ux = unpack(f, t, x)?;
    let uy = unpack(f, t, y)?;
    let sign = op(f, BinOp::BitwiseXor, ux.sign, uy.sign)?;
    let nan = or(f, ux.nan, uy.nan)?;
    let a = and(f, ux.inf, uy.zero)?;
    let b = and(f, ux.zero, uy.inf)?;
    let nan = or(f, nan, a)?;
    let nan = or(f, nan, b)?;
    let inf = or(f, ux.inf, uy.inf)?;
    let zero = or(f, ux.zero, uy.zero)?;
    let mx = f.convert(ConvOp::UConvert, ux.m, t.u64_)?;
    let my = f.convert(ConvOp::UConvert, uy.m, t.u64_)?;
    let s = op(f, BinOp::IMul, mx, my)?;
    let p = msb64(f, t, s)?;
    // The value's leading one is 2^(p + ex + ey - 300).
    let e = op(f, BinOp::IAdd, ux.e, uy.e)?;
    let biased = op(f, BinOp::IAdd, e, p)?;
    let biased = op64(f, t, BinOp::ISub, biased, 173)?;
    let exact = falsy(f, t)?;
    let r = pack(f, t, sign, s, exact, p, biased)?;
    special(
        f,
        sign,
        [(nan, None), (inf, Some(0x7f80_0000)), (zero, Some(0))],
        r,
    )
}

/// The bits of the IEEE `f32` sum `x + y` of bits `x` and `y`, rounded once to nearest even,
/// subnormal operands and results exact (Rust's `x + y`): the smaller magnitude aligned to the
/// larger's with 30 bits below its significand, the bits shifted past them jammed into the last
/// (so a sum or a difference rounds as the exact one), and an exact cancellation `+0`.
fn sum(f: &mut FunctionBuilder<'_>, t: T, x: Value, y: Value) -> BResult<Value> {
    let ux = unpack(f, t, x)?;
    let uy = unpack(f, t, y)?;
    let apart = op(f, BinOp::INotEqual, ux.sign, uy.sign)?;
    let nan = or(f, ux.nan, uy.nan)?;
    let infs = and(f, ux.inf, uy.inf)?;
    let opposed = and(f, infs, apart)?;
    let nan = or(f, nan, opposed)?;
    // The larger magnitude first.
    let higher = op(f, BinOp::SGreaterThan, ux.e, uy.e)?;
    let level = op(f, BinOp::IEqual, ux.e, uy.e)?;
    let more = op(f, BinOp::UGreaterThanEqual, ux.m, uy.m)?;
    let level_more = and(f, level, more)?;
    let x_big = or(f, higher, level_more)?;
    let (eb, es) = (sel(f, x_big, ux.e, uy.e)?, sel(f, x_big, uy.e, ux.e)?);
    let (mb, ms) = (sel(f, x_big, ux.m, uy.m)?, sel(f, x_big, uy.m, ux.m)?);
    let sign = sel(f, x_big, ux.sign, uy.sign)?;
    let mb = f.convert(ConvOp::UConvert, mb, t.u64_)?;
    let mb = opk(f, BinOp::ShiftLeftLogical, mb, 30)?;
    let ms = f.convert(ConvOp::UConvert, ms, t.u64_)?;
    let ms = opk(f, BinOp::ShiftLeftLogical, ms, 30)?;
    let d = op(f, BinOp::ISub, eb, es)?;
    let k63 = k64(f, t, 63)?;
    let d = smin(f, d, k63)?;
    let d = low(f, t, d)?;
    let shifted = op(f, BinOp::ShiftRightLogical, ms, d)?;
    let one64 = ku64(f, t, 1)?;
    let zero64 = ku64(f, t, 0)?;
    let unit = op(f, BinOp::ShiftLeftLogical, one64, d)?;
    let mask = op(f, BinOp::ISub, unit, one64)?;
    let lost = op(f, BinOp::BitwiseAnd, ms, mask)?;
    let sticky = op(f, BinOp::INotEqual, lost, zero64)?;
    let jam = sel(f, sticky, one64, zero64)?;
    let small = op(f, BinOp::BitwiseOr, shifted, jam)?;
    let plus = op(f, BinOp::IAdd, mb, small)?;
    let minus = op(f, BinOp::ISub, mb, small)?;
    let s = sel(f, apart, minus, plus)?;
    let cancelled = op(f, BinOp::IEqual, s, zero64)?;
    let p = msb64(f, t, s)?;
    // The value's leading one is 2^(p + eb - 180).
    let biased = op(f, BinOp::IAdd, eb, p)?;
    let biased = op64(f, t, BinOp::ISub, biased, 53)?;
    let exact = falsy(f, t)?;
    let r = pack(f, t, sign, s, exact, p, biased)?;
    let zero = k32(f, 0)?;
    let r = sel(f, cancelled, zero, r)?;
    // A zero operand gives the other (two zeros, -0 only where both are); an infinity itself.
    let both = op(f, BinOp::BitwiseAnd, ux.sign, uy.sign)?;
    let zeros = and(f, ux.zero, uy.zero)?;
    let r = sel(f, uy.zero, x, r)?;
    let r = sel(f, ux.zero, y, r)?;
    let r = sel(f, zeros, both, r)?;
    let r = sel(f, uy.inf, y, r)?;
    let r = sel(f, ux.inf, x, r)?;
    let nan_bits = k32(f, 0x7fc0_0000)?;
    sel(f, nan, nan_bits, r)
}

/// `r`, but the first of `cases` that holds: a NaN (`None`), or `sign` with the given magnitude
/// bits.
fn special(
    f: &mut FunctionBuilder<'_>,
    sign: Value,
    cases: [(Value, Option<u32>); 3],
    r: Value,
) -> BResult<Value> {
    let mut r = r;
    for (holds, magnitude) in cases.into_iter().rev() {
        let v = match magnitude {
            Some(m) => opk(f, BinOp::BitwiseOr, sign, m)?,
            None => k32(f, 0x7fc0_0000)?,
        };
        r = sel(f, holds, v, r)?;
    }
    Ok(r)
}

/// Define a routine `(u32, u32) -> u32` of `f32` bits computing `body`.
fn define_binary(
    b: &mut ModuleBuilder,
    t: T,
    body: fn(&mut FunctionBuilder<'_>, T, Value, Value) -> BResult<Value>,
) -> BResult<FuncId> {
    let mut f = b.routine(t.u32_)?;
    let x = f.param(t.u32_)?;
    let y = f.param(t.u32_)?;
    let r = body(&mut f, t, x, y)?;
    f.ret(Some(r))?;
    f.finish()
}

/// The kernel's routines: the `f32` arithmetic on bits and `bin`.
#[derive(Clone, Copy)]
struct Routines {
    divide: FuncId,
    multiply: FuncId,
    add: FuncId,
    bin: FuncId,
}

/// Call routine `r` on `x` and `y`.
fn call2(f: &mut FunctionBuilder<'_>, r: FuncId, x: Value, y: Value) -> BResult<Value> {
    f.call(r, &[x, y])?
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))
}

/// The bits of `x / y` ([`quotient`]).
fn divide(f: &mut FunctionBuilder<'_>, k: &K, x: Value, y: Value) -> BResult<Value> {
    call2(f, k.r.divide, x, y)
}

/// The bits of `x * y` ([`product`]).
fn mul(f: &mut FunctionBuilder<'_>, k: &K, x: Value, y: Value) -> BResult<Value> {
    call2(f, k.r.multiply, x, y)
}

/// The bits of `x + y` ([`sum`]).
fn add(f: &mut FunctionBuilder<'_>, k: &K, x: Value, y: Value) -> BResult<Value> {
    call2(f, k.r.add, x, y)
}

/// The bits of `-x`.
fn neg(f: &mut FunctionBuilder<'_>, x: Value) -> BResult<Value> {
    opk(f, BinOp::BitwiseXor, x, 0x8000_0000)
}

/// The bits of `x - y`: `x + (-y)`, as IEEE defines it.
fn sub(f: &mut FunctionBuilder<'_>, k: &K, x: Value, y: Value) -> BResult<Value> {
    let minus = neg(f, y)?;
    add(f, k, x, minus)
}

/// The `f32` coordinate of bits `c` (pixels) snapped to 1/256 of a pixel, ties to even (an `i64`), and
/// whether it lies within the coordinates measured (a `bool`): past the limit it is clamped to
/// it, and a value not finite is 0 (the reference's `snap_clamped`). From its bits: `c * 256` is
/// `m * 2^(e - 142)`.
fn snap(f: &mut FunctionBuilder<'_>, t: T, c: Value) -> BResult<(Value, Value)> {
    let b = c;
    let e = exponent(f, b)?;
    let finite = opk(f, BinOp::INotEqual, e, 0xff)?;
    let fr = opk(f, BinOp::BitwiseAnd, b, 0x7f_ffff)?;
    let normal = opk(f, BinOp::INotEqual, e, 0)?;
    let implicit = opk(f, BinOp::BitwiseOr, fr, 0x80_0000)?;
    let m = sel(f, normal, implicit, fr)?;
    let one = k32(f, 1)?;
    let e = sel(f, normal, e, one)?;
    // From 148 on, |c * 256| >= 2^29: clamped. From 142, a whole number: m shifted left.
    let big = opk(f, BinOp::UGreaterThanEqual, e, 148)?;
    let whole = opk(f, BinOp::UGreaterThanEqual, e, 142)?;
    let k142 = k32(f, 142)?;
    let k147 = k32(f, 147)?;
    let left_e = sel(f, big, k147, e)?;
    let left_e = sel(f, whole, left_e, k142)?;
    let lsh = op(f, BinOp::ISub, left_e, k142)?;
    let left = op(f, BinOp::ShiftLeftLogical, m, lsh)?;
    // Below 142: m shifted right by 142 - e, rounded to nearest even; past 24 places, zero.
    let right_e = sel(f, whole, k142, e)?;
    let r = op(f, BinOp::ISub, k142, right_e)?;
    let gone = opk(f, BinOp::UGreaterThanEqual, r, 25)?;
    let k24 = k32(f, 24)?;
    let r = sel(f, gone, k24, r)?;
    let q = op(f, BinOp::ShiftRightLogical, m, r)?;
    let unit = op(f, BinOp::ShiftLeftLogical, one, r)?;
    let mask = op(f, BinOp::ISub, unit, one)?;
    let rem = op(f, BinOp::BitwiseAnd, m, mask)?;
    let half = op(f, BinOp::ShiftRightLogical, unit, one)?;
    let above = op(f, BinOp::UGreaterThan, rem, half)?;
    let tie = op(f, BinOp::IEqual, rem, half)?;
    let odd = op(f, BinOp::BitwiseAnd, q, one)?;
    let odd = opk(f, BinOp::INotEqual, odd, 0)?;
    let tie = and(f, tie, odd)?;
    let up = or(f, above, tie)?;
    let zero = k32(f, 0)?;
    let carry = sel(f, up, one, zero)?;
    let right = op(f, BinOp::IAdd, q, carry)?;
    let right = sel(f, gone, zero, right)?;
    let mag = sel(f, whole, left, right)?;
    let limit = k32(f, LIMIT)?;
    let mag = sel(f, big, limit, mag)?;
    let mag = sel(f, finite, mag, zero)?;
    let mag = widen(f, t, mag)?;
    let neg = opk(f, BinOp::ShiftRightLogical, b, 31)?;
    let neg = opk(f, BinOp::INotEqual, neg, 0)?;
    let minus = f.unary(UnOp::SNegate, mag)?;
    let value = sel(f, neg, minus, mag)?;
    let small = not(f, big)?;
    let measured = and(f, finite, small)?;
    Ok((value, measured))
}

/// The snapped window position of the clip-space vertex `v` (four `f32`s' bits) under the viewport,
/// flagging unmeasured a coordinate past the limit or not finite (the reference's `snapped`).
fn window(f: &mut FunctionBuilder<'_>, k: &K, v: [Value; 4]) -> BResult<[Value; 2]> {
    let t = k.t;
    let [ox, oy, w, h] = k.vp;
    let half = k32(f, 0.5f32.to_bits())?;
    let px = mul(f, k, w, half)?;
    let py = mul(f, k, h, half)?;
    let xn = divide(f, k, v[0], v[3])?;
    let yn = divide(f, k, v[1], v[3])?;
    let a = mul(f, k, px, xn)?;
    let b = add(f, k, ox, px)?;
    let x = add(f, k, a, b)?;
    let npy = neg(f, py)?;
    let a = mul(f, k, npy, yn)?;
    let b = add(f, k, oy, py)?;
    let y = add(f, k, a, b)?;
    let (sx, mx) = snap(f, t, x)?;
    let (sy, my) = snap(f, t, y)?;
    let measured = and(f, mx, my)?;
    let off = not(f, measured)?;
    flag_if(f, k.unmeasured, off)?;
    Ok([sx, sy])
}

/// The four components of a vertex (`uvec4`).
fn parts(f: &mut FunctionBuilder<'_>, v: Value) -> BResult<[Value; 4]> {
    Ok([
        f.composite_extract(v, &[0])?,
        f.composite_extract(v, &[1])?,
        f.composite_extract(v, &[2])?,
        f.composite_extract(v, &[3])?,
    ])
}

/// The signed distance (`f32` bits) of the clip-space vertex `v` from plane `p` (a `u32`: 0 near, `z`; 1 far,
/// `w - z`; 2 the eye, `w`), inside where not negative (the reference's `depth_planes` and its
/// `w` plane).
fn distance(f: &mut FunctionBuilder<'_>, k: &K, v: [Value; 4], p: Value) -> BResult<Value> {
    let far = sub(f, k, v[3], v[2])?;
    let is_near = opk(f, BinOp::IEqual, p, 0)?;
    let is_far = opk(f, BinOp::IEqual, p, 1)?;
    let d = sel(f, is_far, far, v[3])?;
    sel(f, is_near, v[2], d)
}

/// Clip the `inputs` (two or three `vec4`s) into the polygon (`k.poly`, `k.n`): flagged and empty
/// where a component is not finite, flagged where one is subnormal; then by the near and far
/// planes under depth clipping (empty where wholly past one, flagged near-far where crossing it)
/// and by `w >= 0` where a vertex has `w <= 0` (flagged unmeasured), Sutherland-Hodgman in `f32`
/// (the reference's `depth_clipped` and `clip`).
fn clip(f: &mut FunctionBuilder<'_>, k: &K, inputs: &[Value]) -> BResult<()> {
    let t = k.t;
    let mut nonfinite = falsy(f, t)?;
    let mut tiny = nonfinite;
    for &v in inputs {
        for b in parts(f, v)? {
            let nf = not_finite(f, b)?;
            nonfinite = or(f, nonfinite, nf)?;
            let sn = subnormal(f, b)?;
            tiny = or(f, tiny, sn)?;
        }
    }
    flag_if(f, k.unmeasured, tiny)?;
    flag_if(f, k.unmeasured, nonfinite)?;
    for (i, &v) in inputs.iter().enumerate() {
        let p = atk(f, k.poly, i as u32)?;
        store(f, p, v)?;
    }
    let len = k32(f, inputs.len() as u32)?;
    let zero = k32(f, 0)?;
    let n = sel(f, nonfinite, zero, len)?;
    store(f, k.n, n)?;
    let planes = k32(f, 3)?;
    let k = *k;
    count(f, t, 0, planes, move |f, p| {
        let n = load(f, k.n)?;
        let eye = opk(f, BinOp::IEqual, p, 2)?;
        let clips = not(f, k.clamp)?;
        let active = or(f, eye, clips)?;
        let some = opk(f, BinOp::INotEqual, n, 0)?;
        let go = and(f, active, some)?;
        when(f, go, move |f| plane(f, &k, p, n))
    })
}

/// Clip the polygon of `n` vertices by plane `p` where the reference does (the module's
/// [`clip`]).
fn plane(f: &mut FunctionBuilder<'_>, k: &K, p: Value, n: Value) -> BResult<()> {
    let t = k.t;
    let any_neg = f.local_var(t.u32_, None)?;
    let any_neg = f.var_ptr(any_neg)?;
    let all_neg = f.local_var(t.u32_, None)?;
    let all_neg = f.var_ptr(all_neg)?;
    let any_eye = f.local_var(t.u32_, None)?;
    let any_eye = f.var_ptr(any_eye)?;
    let (zero, one) = (k32(f, 0)?, k32(f, 1)?);
    store(f, any_neg, zero)?;
    store(f, all_neg, one)?;
    store(f, any_eye, zero)?;
    let kk = *k;
    count(f, t, 0, n, move |f, j| {
        let v = at(f, kk.poly, j)?;
        let v = load(f, v)?;
        let v = parts(f, v)?;
        let d = distance(f, &kk, v, p)?;
        let neg = negative(f, d)?;
        let eye = nonpositive(f, v[3])?;
        flag_if(f, any_neg, neg)?;
        flag_if(f, any_eye, eye)?;
        let pos = not(f, neg)?;
        let old = load(f, all_neg)?;
        let zero = k32(f, 0)?;
        let new = sel(f, pos, zero, old)?;
        store(f, all_neg, new)
    })?;
    let any_neg = load(f, any_neg)?;
    let any_neg = opk(f, BinOp::INotEqual, any_neg, 0)?;
    let all_neg = load(f, all_neg)?;
    let all_neg = opk(f, BinOp::INotEqual, all_neg, 0)?;
    let any_eye = load(f, any_eye)?;
    let any_eye = opk(f, BinOp::INotEqual, any_eye, 0)?;
    let eye = opk(f, BinOp::IEqual, p, 2)?;
    let depth = not(f, eye)?;
    let empty = and(f, depth, all_neg)?;
    let trigger = sel(f, eye, any_eye, any_neg)?;
    let k = *k;
    either(
        f,
        empty,
        move |f| {
            let zero = k32(f, 0)?;
            store(f, k.n, zero)
        },
        move |f| {
            when(f, trigger, move |f| {
                let depth = not(f, eye)?;
                flag_if(f, k.near_far, depth)?;
                flag_if(f, k.unmeasured, eye)?;
                cut(f, &k, p, n)
            })
        },
    )
}

/// Sutherland-Hodgman: the polygon of `n` vertices cut by plane `p` into the scratch array, then
/// copied back (the reference's `clip`; a polygon of two vertices is a segment).
fn cut(f: &mut FunctionBuilder<'_>, k: &K, p: Value, n: Value) -> BResult<()> {
    let t = k.t;
    let out = f.local_var(t.u32_, None)?;
    let out = f.var_ptr(out)?;
    let zero = k32(f, 0)?;
    store(f, out, zero)?;
    let segment = opk(f, BinOp::IEqual, n, 2)?;
    let one = k32(f, 1)?;
    let edges = sel(f, segment, one, n)?;
    let k = *k;
    count(f, t, 0, edges, move |f, i| {
        let next = opk(f, BinOp::IAdd, i, 1)?;
        let wraps = op(f, BinOp::IEqual, next, n)?;
        let zero = k32(f, 0)?;
        let next = sel(f, wraps, zero, next)?;
        let a = at(f, k.poly, i)?;
        let a = load(f, a)?;
        let b = at(f, k.poly, next)?;
        let b = load(f, b)?;
        let av = parts(f, a)?;
        let bv = parts(f, b)?;
        let da = distance(f, &k, av, p)?;
        let db = distance(f, &k, bv, p)?;
        let ina = nonnegative(f, da)?;
        let inb = nonnegative(f, db)?;
        when(f, ina, move |f| push(f, &k, out, a))?;
        let crosses = op(f, BinOp::LogicalNotEqual, ina, inb)?;
        when(f, crosses, move |f| {
            let gap = sub(f, &k, da, db)?;
            let s = divide(f, &k, da, gap)?;
            let mut made = [av[0]; 4];
            for c in 0..4 {
                let step = sub(f, &k, bv[c], av[c])?;
                let step = mul(f, &k, s, step)?;
                made[c] = add(f, &k, av[c], step)?;
            }
            let v = f.composite_construct(k.t.vec4, &made)?;
            push(f, &k, out, v)
        })?;
        let last = and(f, segment, inb)?;
        when(f, last, move |f| push(f, &k, out, b))
    })?;
    let made = load(f, out)?;
    count(f, t, 0, made, move |f, j| {
        let from = at(f, k.scratch, j)?;
        let v = load(f, from)?;
        let to = at(f, k.poly, j)?;
        store(f, to, v)
    })?;
    store(f, k.n, made)
}

/// Append `v` to the scratch array (never past its length, which no clip reaches: [`POLYGON`]).
fn push(f: &mut FunctionBuilder<'_>, k: &K, out: Value, v: Value) -> BResult<()> {
    let i = load(f, out)?;
    let room = opk(f, BinOp::ULessThan, i, POLYGON)?;
    let k = *k;
    when(f, room, move |f| {
        let p = at(f, k.scratch, i)?;
        store(f, p, v)?;
        let next = opk(f, BinOp::IAdd, i, 1)?;
        store(f, out, next)
    })
}

/// A job for `bin`: of each axis, the closed box `[lo, hi]` over its denominator `den`, and
/// `lo / den` and `hi / den` rounded down (`i64`s: `[lo, hi, den, lo / den, hi / den]`); and up to
/// three half-planes `gx * x + gy * y + c >= 0` (in units), unused ones zero.
struct Job {
    x: [Value; 5],
    y: [Value; 5],
    planes: [[Value; 3]; 3],
}

/// An axis of a job of denominator 1: `[lo, hi, 1, lo, hi]`.
fn whole(f: &mut FunctionBuilder<'_>, t: T, lo: Value, hi: Value) -> BResult<[Value; 5]> {
    let one = k64(f, t, 1)?;
    Ok([lo, hi, one, lo, hi])
}

/// Bin a job (the function [`define_bin`] defines).
fn call_bin(f: &mut FunctionBuilder<'_>, k: &K, job: Job) -> BResult<()> {
    let mut args = Vec::with_capacity(BIN_PARAMS);
    args.extend(job.x);
    args.extend(job.y);
    for p in job.planes {
        args.extend(p);
    }
    f.call(k.r.bin, &args)?;
    Ok(())
}

/// Three unused half-planes.
fn no_planes(f: &mut FunctionBuilder<'_>, t: T) -> BResult<[[Value; 3]; 3]> {
    let zero = k64(f, t, 0)?;
    Ok([[zero; 3]; 3])
}

/// The parameters of `bin` ([`define_bin`]).
const BIN_PARAMS: usize = 19;

/// The function `bin(x, y, planes)` of a [`Job`]'s 19 `i64`s, in its fields' order: each tile of
/// the primitive's slice the job meets, within the region (the context), sets its bit of the mask.
/// A tile meets it where its rectangle clipped by the region (half open) is not empty, the job's
/// box meets it, and no half-plane separates it, the rectangle's far edges approached from inside
/// (the reference's `bin` and `meets`). The tiles' sides are powers of two, so the tiles the box
/// can meet follow from its bounds over their denominators by shifts.
fn define_bin(b: &mut ModuleBuilder, t: T, tiles: GlobalId, context: GlobalId) -> BResult<FuncId> {
    let void = b.ty(Type::Void)?;
    let mut f = b.routine(void)?;
    let mut p = Vec::with_capacity(BIN_PARAMS);
    for _ in 0..BIN_PARAMS {
        p.push(f.param(t.i64_)?);
    }
    bin_body(&mut f, t, tiles, context, &p)?;
    f.ret(None)?;
    f.finish()
}

/// The body of `bin` over its parameters `p` ([`define_bin`]).
fn bin_body(
    f: &mut FunctionBuilder<'_>,
    t: T,
    tiles: GlobalId,
    context: GlobalId,
    p: &[Value],
) -> BResult<()> {
    let [lo_x, hi_x, den_x, qlo_x, qhi_x] = [p[0], p[1], p[2], p[3], p[4]];
    let [lo_y, hi_y, den_y, qlo_y, qhi_y] = [p[5], p[6], p[7], p[8], p[9]];
    let planes = [
        [p[10], p[11], p[12]],
        [p[13], p[14], p[15]],
        [p[16], p[17], p[18]],
    ];
    let context = f.module().global_value(context)?;
    let tiles = f.module().global_value(tiles)?;
    let mut c = [lo_x; ctx::WORDS as usize];
    for (i, w) in c.iter_mut().enumerate() {
        let at = atk(f, context, i as u32)?;
        *w = load(f, at)?;
    }
    let [rx0, rx1, ry0, ry1, tw, th, tiles_x, tiles_y, base, shift_x, shift_y] = c;
    // The tiles the box and the region can meet: `floor(lo / den / size)` is
    // `floor(floor(lo / den) / size)`, a shift of the job's quotient by the size's power of two.
    let range = |f: &mut FunctionBuilder<'_>,
                 q: [Value; 2],
                 r0: Value,
                 r1: Value,
                 shift: Value,
                 count: Value|
     -> BResult<(Value, Value)> {
        let shift = low(f, t, shift)?;
        let first = op(f, BinOp::ShiftRightArithmetic, q[0], shift)?;
        let zero = k64(f, t, 0)?;
        let first = smax(f, first, zero)?;
        let r_first = op(f, BinOp::ShiftRightArithmetic, r0, shift)?;
        let first = smax(f, first, r_first)?;
        let last = op(f, BinOp::ShiftRightArithmetic, q[1], shift)?;
        let end = op64(f, t, BinOp::ISub, count, 1)?;
        let last = smin(f, last, end)?;
        let r_end = op64(f, t, BinOp::ISub, r1, 1)?;
        let r_last = op(f, BinOp::ShiftRightArithmetic, r_end, shift)?;
        let last = smin(f, last, r_last)?;
        Ok((first, last))
    };
    let (tx0, tx1) = range(f, [qlo_x, qhi_x], rx0, rx1, shift_x, tiles_x)?;
    let (ty0, ty1) = range(f, [qlo_y, qhi_y], ry0, ry1, shift_y, tiles_y)?;
    let words = f.array_length(tiles, 0)?;
    let words = widen(f, t, words)?;
    span(f, t, ty0, ty1, move |f, ty| {
        span(f, t, tx0, tx1, move |f, tx| {
            let edge =
                |f: &mut FunctionBuilder<'_>, i: Value, size: Value, r0: Value, r1: Value| {
                    let start = op(f, BinOp::IMul, i, size)?;
                    let end = op(f, BinOp::IAdd, start, size)?;
                    Ok::<_, BuildError>((smax(f, r0, start)?, smin(f, r1, end)?))
                };
            let (x0, x1) = edge(f, tx, tw, rx0, rx1)?;
            let (y0, y1) = edge(f, ty, th, ry0, ry1)?;
            let wide = op(f, BinOp::SLessThan, x0, x1)?;
            let tall = op(f, BinOp::SLessThan, y0, y1)?;
            let mut ok = and(f, wide, tall)?;
            for (lo, hi, den, a0, a1) in [(lo_x, hi_x, den_x, x0, x1), (lo_y, hi_y, den_y, y0, y1)]
            {
                let far = op(f, BinOp::IMul, a1, den)?;
                let near = op(f, BinOp::IMul, a0, den)?;
                let before = op(f, BinOp::SLessThan, lo, far)?;
                let after = op(f, BinOp::SGreaterThanEqual, hi, near)?;
                ok = and(f, ok, before)?;
                ok = and(f, ok, after)?;
            }
            let zero = k64(f, t, 0)?;
            for [gx, gy, c] in planes {
                let gx_pos = op(f, BinOp::SGreaterThan, gx, zero)?;
                let gy_pos = op(f, BinOp::SGreaterThan, gy, zero)?;
                let x = sel(f, gx_pos, x1, x0)?;
                let y = sel(f, gy_pos, y1, y0)?;
                let a = op(f, BinOp::IMul, gx, x)?;
                let b = op(f, BinOp::IMul, gy, y)?;
                let v = op(f, BinOp::IAdd, a, b)?;
                let v = op(f, BinOp::IAdd, v, c)?;
                let below = op(f, BinOp::SLessThan, v, zero)?;
                let on = op(f, BinOp::IEqual, v, zero)?;
                let toward = or(f, gx_pos, gy_pos)?;
                let on = and(f, on, toward)?;
                let out = or(f, below, on)?;
                let kept = not(f, out)?;
                ok = and(f, ok, kept)?;
            }
            let row = op(f, BinOp::IAdd, base, ty)?;
            let tile = op(f, BinOp::IMul, row, tiles_x)?;
            let tile = op(f, BinOp::IAdd, tile, tx)?;
            let w = opk(f, BinOp::ShiftRightArithmetic, tile, 5)?;
            let held = op(f, BinOp::SLessThan, w, words)?;
            let ok = and(f, ok, held)?;
            when(f, ok, move |f| {
                let w = low(f, t, w)?;
                let tile = low(f, t, tile)?;
                let bit = opk(f, BinOp::BitwiseAnd, tile, 31)?;
                let one = k32(f, 1)?;
                let bit = op(f, BinOp::ShiftLeftLogical, one, bit)?;
                let zero = k32(f, 0)?;
                let p = f.access_chain(tiles, &[zero, w])?;
                f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
                Ok(())
            })
        })
    })
}

/// Where a point of the `f32` `size` pixels (not negative, at most 511) reaches about its snapped
/// centre, in units (`i64`s): how far back its square starts and its extent from there, half its
/// size back with the start rounded half up, and its size rounded to the nearest 1/16 of a pixel,
/// a half up (the reference's `point_extent`). In integers from its bits: `size * 256 * 2^24` is `m << (e -
/// 118)`, exact from `e = 118` (a size from half a unit) to 135 (511); below, both are zero.
fn point_extent(f: &mut FunctionBuilder<'_>, t: T, size: Value) -> BResult<(Value, Value)> {
    let b = size;
    let e = exponent(f, b)?;
    let fr = opk(f, BinOp::BitwiseAnd, b, 0x7f_ffff)?;
    let normal = opk(f, BinOp::INotEqual, e, 0)?;
    let implicit = opk(f, BinOp::BitwiseOr, fr, 0x80_0000)?;
    let m = sel(f, normal, implicit, fr)?;
    let tiny = opk(f, BinOp::ULessThan, e, 118)?;
    let k118 = k32(f, 118)?;
    let e = sel(f, tiny, k118, e)?;
    let sh = op(f, BinOp::ISub, e, k118)?;
    let m = f.convert(ConvOp::UConvert, m, t.u64_)?;
    let n = op(f, BinOp::ShiftLeftLogical, m, sh)?;
    let n = f.convert(ConvOp::Bitcast, n, t.i64_)?;
    // ceil((n - 2^24) / 2^25): half the size less half a unit, rounded up.
    let less = op64(f, t, BinOp::ISub, n, 1 << 24)?;
    let minus = f.unary(UnOp::SNegate, less)?;
    let down = opk(f, BinOp::ShiftRightArithmetic, minus, 25)?;
    let back = f.unary(UnOp::SNegate, down)?;
    // n / 2^28 (the size in sixteenths of a pixel) rounded to nearest, a half up.
    let up = op64(f, t, BinOp::IAdd, n, 1 << 27)?;
    let sixteenths = opk(f, BinOp::ShiftRightArithmetic, up, 28)?;
    let extent = op64(f, t, BinOp::IMul, sixteenths, 16)?;
    let zero = k64(f, t, 0)?;
    let back = sel(f, tiny, zero, back)?;
    let extent = sel(f, tiny, zero, extent)?;
    Ok((back, extent))
}

/// A point's reach (the reference's `point`): none and flagged where its position is not finite
/// or at or behind the eye, or past the depth range under depth clipping; flagged where a
/// component is subnormal, or its size negative, not a number or subnormal (drawn as 0); its size
/// clamped to 511; the square of [`point_extent`] about its snapped centre.
fn point(f: &mut FunctionBuilder<'_>, k: &K, v: Value, size: Value) -> BResult<()> {
    let t = k.t;
    let c = parts(f, v)?;
    let mut bad = falsy(f, t)?;
    let mut tiny = bad;
    for b in c {
        let nf = not_finite(f, b)?;
        bad = or(f, bad, nf)?;
        let sn = subnormal(f, b)?;
        tiny = or(f, tiny, sn)?;
    }
    let behind = nonpositive(f, c[3])?;
    let bad = or(f, bad, behind)?;
    flag_if(f, k.unmeasured, bad)?;
    let good = not(f, bad)?;
    let k = *k;
    when(f, good, move |f| {
        flag_if(f, k.unmeasured, tiny)?;
        let far = sub(f, &k, c[3], c[2])?;
        let near_out = negative(f, c[2])?;
        let far_out = negative(f, far)?;
        let out = or(f, near_out, far_out)?;
        let clips = not(f, k.clamp)?;
        let past = and(f, clips, out)?;
        flag_if(f, k.unmeasured, past)?;
        let drawn = not(f, past)?;
        when(f, drawn, move |f| {
            let ok = nonnegative(f, size)?;
            let not_ok = not(f, ok)?;
            flag_if(f, k.unmeasured, not_ok)?;
            let zero = k32(f, 0)?;
            let s = sel(f, ok, size, zero)?;
            let sn = subnormal(f, s)?;
            flag_if(f, k.unmeasured, sn)?;
            let limit = k32(f, POINT_SIZE_LIMIT.to_bits())?;
            let big = exceeds(f, s, POINT_SIZE_LIMIT)?;
            let s = sel(f, big, limit, s)?;
            let [cx, cy] = window(f, &k, c)?;
            let (back, extent) = point_extent(f, t, s)?;
            let x0 = op(f, BinOp::ISub, cx, back)?;
            let x1 = op(f, BinOp::IAdd, x0, extent)?;
            let y0 = op(f, BinOp::ISub, cy, back)?;
            let y1 = op(f, BinOp::IAdd, y0, extent)?;
            let job = Job {
                x: whole(f, t, x0, x1)?,
                y: whole(f, t, y0, y1)?,
                planes: no_planes(f, t)?,
            };
            call_bin(f, &k, job)
        })
    })
}

/// A line's reach (the reference's `line`): its segment clipped, its ends snapped; a line of no
/// length flagged and binned as the square of a pixel about its end; another as the
/// parallelogram half a pixel across its major axis, extended half a pixel past each end along
/// it (a 45-degree line taken as y-major): the box of its major axis in units, of its minor axis
/// over the major's length `d`, and the two half-planes `|dx (y - py) - dy (x - px)| <= 128 d`.
fn line(f: &mut FunctionBuilder<'_>, k: &K, ends: [Value; 2]) -> BResult<()> {
    clip(f, k, &ends)?;
    let n = load(f, k.n)?;
    let two = opk(f, BinOp::UGreaterThanEqual, n, 2)?;
    let k = *k;
    when(f, two, move |f| {
        let t = k.t;
        let a = atk(f, k.poly, 0)?;
        let a = load(f, a)?;
        let b = atk(f, k.poly, 1)?;
        let b = load(f, b)?;
        let a = parts(f, a)?;
        let b = parts(f, b)?;
        let p = window(f, &k, a)?;
        let q = window(f, &k, b)?;
        let dx = op(f, BinOp::ISub, q[0], p[0])?;
        let dy = op(f, BinOp::ISub, q[1], p[1])?;
        let zx = op64(f, t, BinOp::IEqual, dx, 0)?;
        let zy = op64(f, t, BinOp::IEqual, dy, 0)?;
        let zero_length = and(f, zx, zy)?;
        either(
            f,
            zero_length,
            move |f| {
                let one = k32(f, 1)?;
                store(f, k.unmeasured, one)?;
                let mut square = [[p[0]; 5]; 2];
                for (axis, c) in [p[0], p[1]].into_iter().enumerate() {
                    let lo = op64(f, t, BinOp::ISub, c, 128)?;
                    let hi = op64(f, t, BinOp::IAdd, c, 128)?;
                    square[axis] = whole(f, t, lo, hi)?;
                }
                let job = Job {
                    x: square[0],
                    y: square[1],
                    planes: no_planes(f, t)?,
                };
                call_bin(f, &k, job)
            },
            move |f| parallelogram(f, &k, p, q, [dx, dy]),
        )
    })
}

/// The job of a line from `p` to `q` of length `delta`, not zero ([`line`]).
fn parallelogram(
    f: &mut FunctionBuilder<'_>,
    k: &K,
    p: [Value; 2],
    q: [Value; 2],
    delta: [Value; 2],
) -> BResult<()> {
    let t = k.t;
    let [dx, dy] = delta;
    let adx = sabs(f, t, dx)?;
    let ady = sabs(f, t, dy)?;
    let x_major = op(f, BinOp::SGreaterThan, adx, ady)?;
    let d = sel(f, x_major, adx, ady)?;
    let half_d = op64(f, t, BinOp::IMul, d, 128)?;
    // The box of axis `a` as the major axis (in units, over 1) and as the minor one (over `d`),
    // with the minor one's bounds over `d` rounded down: `floor(start / d)` is
    // `p + floor(-along / d)` and `floor(end / d)` is `q + floor(along / d)`.
    let mut major = [[p[0]; 2]; 2];
    let mut minor = [[p[0]; 4]; 2];
    for a in 0..2 {
        let lo = smin(f, p[a], q[a])?;
        let hi = smax(f, p[a], q[a])?;
        major[a] = [
            op64(f, t, BinOp::ISub, lo, 128)?,
            op64(f, t, BinOp::IAdd, hi, 128)?,
        ];
        let along = op64(f, t, BinOp::IMul, delta[a], 128)?;
        let start = op(f, BinOp::IMul, p[a], d)?;
        let start = op(f, BinOp::ISub, start, along)?;
        let end = op(f, BinOp::IMul, q[a], d)?;
        let end = op(f, BinOp::IAdd, end, along)?;
        let lo = smin(f, start, end)?;
        let hi = smax(f, start, end)?;
        let (ahead, back) = small_quotients(f, t, along, d)?;
        let start_q = op(f, BinOp::IAdd, p[a], back)?;
        let end_q = op(f, BinOp::IAdd, q[a], ahead)?;
        let lo_q = smin(f, start_q, end_q)?;
        let hi_q = smax(f, start_q, end_q)?;
        minor[a] = [
            op(f, BinOp::ISub, lo, half_d)?,
            op(f, BinOp::IAdd, hi, half_d)?,
            op64(f, t, BinOp::ISub, lo_q, 128)?,
            op64(f, t, BinOp::IAdd, hi_q, 128)?,
        ];
    }
    let one = k64(f, t, 1)?;
    let mut axes = [[p[0]; 5]; 2];
    for (a, axis) in axes.iter_mut().enumerate() {
        // Axis `a` is the major one where `x_major` is `a == 0`.
        let is_major = match a {
            0 => x_major,
            _ => not(f, x_major)?,
        };
        let [lo, hi] = major[a];
        let [mlo, mhi, mqlo, mqhi] = minor[a];
        *axis = [
            sel(f, is_major, lo, mlo)?,
            sel(f, is_major, hi, mhi)?,
            sel(f, is_major, one, d)?,
            sel(f, is_major, lo, mqlo)?,
            sel(f, is_major, hi, mqhi)?,
        ];
    }
    // dx (y - py) - dy (x - px) = dx y - dy x + c0, c0 = dy px - dx py.
    let a = op(f, BinOp::IMul, dy, p[0])?;
    let b = op(f, BinOp::IMul, dx, p[1])?;
    let c0 = op(f, BinOp::ISub, a, b)?;
    let ndy = f.unary(UnOp::SNegate, dy)?;
    let ndx = f.unary(UnOp::SNegate, dx)?;
    let c_a = op(f, BinOp::IAdd, c0, half_d)?;
    let c_b = op(f, BinOp::ISub, half_d, c0)?;
    let zero = k64(f, t, 0)?;
    call_bin(
        f,
        k,
        Job {
            x: axes[0],
            y: axes[1],
            planes: [[ndy, dx, c_a], [dy, ndx, c_b], [zero; 3]],
        },
    )
}

/// `(floor(along / d), floor(-along / d))` for `i64`s `along` and `d > 0` with `|along| <= 128 d`
/// (a line's minor axis): from `floor(|along| / d)`, at most 128, found a bit at a time from 128
/// down by products (no division), and whether it is exact.
fn small_quotients(
    f: &mut FunctionBuilder<'_>,
    t: T,
    along: Value,
    d: Value,
) -> BResult<(Value, Value)> {
    let u = sabs(f, t, along)?;
    let mut q = k64(f, t, 0)?;
    for bit in [128, 64, 32, 16, 8, 4, 2, 1] {
        let more = op64(f, t, BinOp::IAdd, q, bit)?;
        let product = op(f, BinOp::IMul, more, d)?;
        let fits = op(f, BinOp::SLessThanEqual, product, u)?;
        q = sel(f, fits, more, q)?;
    }
    let product = op(f, BinOp::IMul, q, d)?;
    let exact = op(f, BinOp::IEqual, product, u)?;
    let (zero, one) = (k64(f, t, 0)?, k64(f, t, 1)?);
    let up = sel(f, exact, zero, one)?;
    let ceil = op(f, BinOp::IAdd, q, up)?;
    let down = f.unary(UnOp::SNegate, ceil)?;
    let ahead_pos = op(f, BinOp::SGreaterThanEqual, along, zero)?;
    Ok((sel(f, ahead_pos, q, down)?, sel(f, ahead_pos, down, q)?))
}

/// A triangle's reach (the reference's `triangle`): its polygon clipped, its vertices snapped,
/// and each triangle of its fan from the first vertex not of zero area, not culled by face and
/// not by the small-primitive cull, binned as its box and its three edges' half-planes.
fn triangle(f: &mut FunctionBuilder<'_>, k: &K, vertices: [Value; 3]) -> BResult<()> {
    clip(f, k, &vertices)?;
    let n = load(f, k.n)?;
    let three = opk(f, BinOp::UGreaterThanEqual, n, 3)?;
    let k = *k;
    when(f, three, move |f| {
        let t = k.t;
        count(f, t, 0, n, move |f, j| {
            let v = at(f, k.poly, j)?;
            let v = load(f, v)?;
            let v = parts(f, v)?;
            let [x, y] = window(f, &k, v)?;
            let i = opk(f, BinOp::IMul, j, 2)?;
            let px = at(f, k.pts, i)?;
            store(f, px, x)?;
            let i = opk(f, BinOp::IAdd, i, 1)?;
            let py = at(f, k.pts, i)?;
            store(f, py, y)
        })?;
        let last = opk(f, BinOp::ISub, n, 1)?;
        count(f, t, 1, last, move |f, i| fan(f, &k, i))
    })
}

/// The snapped vertex `j` (a `u32`) of the polygon.
fn vertex(f: &mut FunctionBuilder<'_>, k: &K, j: Value) -> BResult<[Value; 2]> {
    let i = opk(f, BinOp::IMul, j, 2)?;
    let x = at(f, k.pts, i)?;
    let x = load(f, x)?;
    let i = opk(f, BinOp::IAdd, i, 1)?;
    let y = at(f, k.pts, i)?;
    let y = load(f, y)?;
    Ok([x, y])
}

/// The fan's triangle of the polygon's vertices 0, `i` and `i + 1` ([`triangle`]).
fn fan(f: &mut FunctionBuilder<'_>, k: &K, i: Value) -> BResult<()> {
    let t = k.t;
    let zero32 = k32(f, 0)?;
    let a = vertex(f, k, zero32)?;
    let b = vertex(f, k, i)?;
    let next = opk(f, BinOp::IAdd, i, 1)?;
    let c = vertex(f, k, next)?;
    let ux = op(f, BinOp::ISub, b[0], a[0])?;
    let uy = op(f, BinOp::ISub, b[1], a[1])?;
    let vx = op(f, BinOp::ISub, c[0], a[0])?;
    let vy = op(f, BinOp::ISub, c[1], a[1])?;
    let l = op(f, BinOp::IMul, ux, vy)?;
    let r = op(f, BinOp::IMul, uy, vx)?;
    let area = op(f, BinOp::ISub, l, r)?;
    let zero = k64(f, t, 0)?;
    let some = op(f, BinOp::INotEqual, area, zero)?;
    // Clockwise in window coordinates, y down, is a positive area.
    let pos = op(f, BinOp::SGreaterThan, area, zero)?;
    let cw = opk(f, BinOp::IEqual, k.winding, 0)?;
    let front = op(f, BinOp::LogicalEqual, pos, cw)?;
    let back = not(f, front)?;
    let cull_front = opk(f, BinOp::IEqual, k.cull, 1)?;
    let cull_back = opk(f, BinOp::IEqual, k.cull, 2)?;
    let culled_front = and(f, cull_front, front)?;
    let culled_back = and(f, cull_back, back)?;
    let culled = or(f, culled_front, culled_back)?;
    let kept = not(f, culled)?;
    let go = and(f, some, kept)?;
    let k = *k;
    when(f, go, move |f| {
        let (one, minus) = (k64(f, t, 1)?, k64(f, t, -1)?);
        let sign = sel(f, pos, one, minus)?;
        let tri = [a, b, c];
        let culled = sample_culled(f, &k, tri, sign)?;
        let drawn = not(f, culled)?;
        when(f, drawn, move |f| {
            let mut x = [a[0]; 5];
            let mut y = [a[0]; 5];
            for (axis, out) in [(0, &mut x), (1, &mut y)] {
                let lo = smin(f, a[axis], b[axis])?;
                let lo = smin(f, lo, c[axis])?;
                let hi = smax(f, a[axis], b[axis])?;
                let hi = smax(f, hi, c[axis])?;
                *out = whole(f, t, lo, hi)?;
            }
            let mut planes = no_planes(f, t)?;
            for (e, plane) in planes.iter_mut().enumerate() {
                let (p, q) = (tri[e], tri[(e + 1) % 3]);
                // sign * cross(q - p, (x, y) - p) >= 0.
                let rise = op(f, BinOp::ISub, p[1], q[1])?;
                let gx = op(f, BinOp::IMul, sign, rise)?;
                let run = op(f, BinOp::ISub, q[0], p[0])?;
                let gy = op(f, BinOp::IMul, sign, run)?;
                let a = op(f, BinOp::IMul, gx, p[0])?;
                let b = op(f, BinOp::IMul, gy, p[1])?;
                let s = op(f, BinOp::IAdd, a, b)?;
                let c = f.unary(UnOp::SNegate, s)?;
                *plane = [gx, gy, c];
            }
            call_bin(f, &k, Job { x, y, planes })
        })
    })
}

/// The default sample positions of a pixel in units (Metal's, Vulkan's standard ones) at 1, 2 and
/// 4 samples (the reference's `sample_offsets`).
const OFFSETS: [&[[i64; 2]]; 3] = [
    &[[128, 128]],
    &[[192, 192], [64, 64]],
    &[[96, 32], [224, 96], [32, 160], [160, 224]],
];

/// Whether the small-primitive cull drops the triangle `tri` of orientation `sign` (the
/// reference's `sample_culled`): its closed box holds at most 12 sample positions (24 at 4
/// samples) and its closed area none of them, position by position (`covers_sample`). An
/// offset whose columns the box holds none of holds no position, so it is not walked.
fn sample_culled(
    f: &mut FunctionBuilder<'_>,
    k: &K,
    tri: [[Value; 2]; 3],
    sign: Value,
) -> BResult<Value> {
    let t = k.t;
    let four = opk(f, BinOp::IEqual, k.samples, 4)?;
    let two = opk(f, BinOp::IEqual, k.samples, 2)?;
    let many = or(f, four, two)?;
    let untrue = falsy(f, t)?;
    let truthy = not(f, untrue)?;
    let mut lo = [tri[0][0]; 2];
    let mut hi = [tri[0][0]; 2];
    for axis in 0..2 {
        let l = smin(f, tri[0][axis], tri[1][axis])?;
        lo[axis] = smin(f, l, tri[2][axis])?;
        let h = smax(f, tri[0][axis], tri[1][axis])?;
        hi[axis] = smax(f, h, tri[2][axis])?;
    }
    let zero = k64(f, t, 0)?;
    // The sample positions' columns or rows within `[lo, hi]` at offset `o`.
    let span_of = |f: &mut FunctionBuilder<'_>, lo: Value, hi: Value, o: Value| {
        let a = op(f, BinOp::ISub, hi, o)?;
        let a = floor_unit(f, a)?;
        let b = op(f, BinOp::ISub, lo, o)?;
        let b = ceil_unit(f, b)?;
        let s = op(f, BinOp::ISub, a, b)?;
        let s = op64(f, t, BinOp::IAdd, s, 1)?;
        smax(f, s, zero)
    };
    let mut offsets = Vec::with_capacity(4);
    let mut boxed = zero;
    for s in 0..4 {
        let mut o = [zero; 2];
        for (axis, slot) in o.iter_mut().enumerate() {
            let at = |n: usize| OFFSETS[n].get(s).map_or(0, |p| p[axis]);
            let v4 = k64(f, t, at(2))?;
            let v2 = k64(f, t, at(1))?;
            let v1 = k64(f, t, at(0))?;
            let v = sel(f, two, v2, v1)?;
            *slot = sel(f, four, v4, v)?;
        }
        let active = match s {
            0 => truthy,
            1 => many,
            _ => four,
        };
        let cols = span_of(f, lo[0], hi[0], o[0])?;
        let rows = span_of(f, lo[1], hi[1], o[1])?;
        let held = op(f, BinOp::IMul, cols, rows)?;
        let held = sel(f, active, held, zero)?;
        boxed = op(f, BinOp::IAdd, boxed, held)?;
        let some = op(f, BinOp::SGreaterThan, cols, zero)?;
        let walk = and(f, active, some)?;
        offsets.push((o, walk));
    }
    let (k24, k12) = (k64(f, t, 24)?, k64(f, t, 12)?);
    let tested = sel(f, four, k24, k12)?;
    let few = op(f, BinOp::SLessThanEqual, boxed, tested)?;
    let covers = f.local_var(t.u32_, None)?;
    let covers = f.var_ptr(covers)?;
    let zero32 = k32(f, 0)?;
    store(f, covers, zero32)?;
    when(f, few, move |f| {
        for (o, walk) in offsets {
            when(f, walk, move |f| {
                let mut ends = [[o[0]; 2]; 2];
                for (axis, end) in ends.iter_mut().enumerate() {
                    let first = op(f, BinOp::ISub, lo[axis], o[axis])?;
                    let last = op(f, BinOp::ISub, hi[axis], o[axis])?;
                    *end = [ceil_unit(f, first)?, floor_unit(f, last)?];
                }
                span(f, t, ends[1][0], ends[1][1], move |f, row| {
                    span(f, t, ends[0][0], ends[0][1], move |f, column| {
                        let hit = holds_sample(f, t, tri, sign, o, [column, row])?;
                        flag_if(f, covers, hit)
                    })
                })
            })?;
        }
        Ok(())
    })?;
    let covers = load(f, covers)?;
    let none = opk(f, BinOp::IEqual, covers, 0)?;
    and(f, few, none)
}

/// Whether the sample position of column and row `at` at offset `o` (`256 at + o` units) lies
/// within the closed triangle `tri` of orientation `sign`: on the inner side of each edge or on it,
/// `sign * cross(b - a, p - a) >= 0` (the reference's `covers_sample`, position by position: the
/// box holds at most 24 of them).
fn holds_sample(
    f: &mut FunctionBuilder<'_>,
    t: T,
    tri: [[Value; 2]; 3],
    sign: Value,
    o: [Value; 2],
    at: [Value; 2],
) -> BResult<Value> {
    let mut point = [o[0]; 2];
    for (axis, c) in point.iter_mut().enumerate() {
        let u = op64(f, t, BinOp::IMul, at[axis], UNIT)?;
        *c = op(f, BinOp::IAdd, u, o[axis])?;
    }
    let zero = k64(f, t, 0)?;
    let mut sides = [zero; 3];
    for (e, inner) in sides.iter_mut().enumerate() {
        let (a, b) = (tri[e], tri[(e + 1) % 3]);
        let run = op(f, BinOp::ISub, b[0], a[0])?;
        let rise = op(f, BinOp::ISub, b[1], a[1])?;
        let px = op(f, BinOp::ISub, point[0], a[0])?;
        let py = op(f, BinOp::ISub, point[1], a[1])?;
        let l = op(f, BinOp::IMul, run, py)?;
        let r = op(f, BinOp::IMul, rise, px)?;
        let cross = op(f, BinOp::ISub, l, r)?;
        let cross = op(f, BinOp::IMul, sign, cross)?;
        *inner = op(f, BinOp::SGreaterThanEqual, cross, zero)?;
    }
    let two = and(f, sides[0], sides[1])?;
    and(f, two, sides[2])
}

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

/// A `Function` array of `len` elements of `elem`, as its pointer.
fn array_var(f: &mut FunctionBuilder<'_>, elem: TypeId, len: u32) -> BResult<Value> {
    let len = f.u32_const(len)?;
    let ty = f.module().ty(Type::Array {
        elem,
        len,
        stride: None,
    })?;
    let v = f.local_var(ty, None)?;
    f.var_ptr(v)
}

/// The module as a typed module, before serialization: the module
/// [`tile_reach_module`](super::tile_reach_module) serializes, which the CPU interpreter runs.
/// Its bindings are the symbols `PRIMITIVES`, `PASS` and `TILES`.
pub(super) fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let (t, uvec3) = types(&mut b)?;
    let prims = words_buffer(&mut b, PRIMITIVES, t.u32_)?;
    let pass = words_buffer(&mut b, PASS, t.u32_)?;
    let tiles = words_buffer(&mut b, TILES, t.u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let (r, context) = routines(&mut b, t, tiles)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let g = [prims, pass, context, invocation];
    main_body(&mut f, t, r, g)?;
    f.ret(None)?;
    let id = f.finish()?;
    entry(&mut b, id)?;
    b.finalize()
}

/// The kernel's routines, over the mask `tiles`, and the `Private` context `bin` reads.
fn routines(b: &mut ModuleBuilder, t: T, tiles: GlobalId) -> BResult<(Routines, GlobalId)> {
    let words = b.constant(Constant::Scalar {
        ty: t.u32_,
        bits: ConstBits(u64::from(ctx::WORDS)),
    })?;
    let ctx_ty = b.ty(Type::Array {
        elem: t.i64_,
        len: words,
        stride: None,
    })?;
    let context = b.plain_global(StorageClass::Private, ctx_ty, None)?;
    let r = Routines {
        divide: define_binary(b, t, quotient)?,
        multiply: define_binary(b, t, product)?,
        add: define_binary(b, t, sum)?,
        bin: define_bin(b, t, tiles, context)?,
    };
    Ok((r, context))
}

/// The kernel's types, and the `uvec3` of its invocation's id.
fn types(b: &mut ModuleBuilder) -> BResult<(T, TypeId)> {
    let bool_ = b.ty(Type::Bool)?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let i64_ = b.ty(Type::Int {
        width: 64,
        signed: true,
    })?;
    let vec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let t = T {
        bool_,
        u32_,
        u64_,
        i64_,
        vec4,
    };
    Ok((t, uvec3))
}

/// Make `id` the `GLCompute` entry point of local size ([`TileReachModule::LOCAL_SIZE`](M), 1,
/// 1).
fn entry(b: &mut ModuleBuilder, id: FuncId) -> BResult<()> {
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: M::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    Ok(())
}

/// The words of one record of [`arithmetic_module`]: in, the bits of an `f32` `x`, `y` and a
/// point size; out, the bits of the kernel's quotient `x / y`, product `x * y` and sum `x + y`,
/// `x` snapped (an `i64`, its low word first) and whether it lay within the coordinates measured
/// (1 or 0), and the size's point extent (how far back and the extent, each a word).
#[cfg(test)]
pub(super) const ARITHMETIC_WORDS: u32 = 11;

/// A probe of the kernel's arithmetic for the tests: invocation `i` reads record `i` of the
/// buffer `PRIMITIVES` (records of [`ARITHMETIC_WORDS`], as many as it holds) and writes the
/// results of [`quotient`], [`product`], [`sum`], [`snap`] and [`point_extent`] into it, as the
/// kernel computes them.
#[cfg(test)]
pub(super) fn arithmetic_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let (t, uvec3) = types(&mut b)?;
    let records = words_buffer(&mut b, PRIMITIVES, t.u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let divide = define_binary(&mut b, t, quotient)?;
    let multiply = define_binary(&mut b, t, product)?;
    let plus = define_binary(&mut b, t, sum)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let records = f.module().global_value(records)?;
    let invocation = f.module().global_value(invocation)?;
    let gid = load(&mut f, invocation)?;
    let i = f.composite_extract(gid, &[0])?;
    let len = f.array_length(records, 0)?;
    let held = opk(&mut f, BinOp::UDiv, len, ARITHMETIC_WORDS)?;
    let inside = op(&mut f, BinOp::ULessThan, i, held)?;
    when(&mut f, inside, move |f| {
        let base = opk(f, BinOp::IMul, i, ARITHMETIC_WORDS)?;
        let field = |f: &mut FunctionBuilder<'_>, at: u32| {
            let w = opk(f, BinOp::IAdd, base, at)?;
            let zero = k32(f, 0)?;
            f.access_chain(records, &[zero, w])
        };
        let mut input = [base; 3];
        for (at, v) in input.iter_mut().enumerate() {
            let p = field(f, at as u32)?;
            *v = load(f, p)?;
        }
        let [x, y, size] = input;
        let q = call2(f, divide, x, y)?;
        let m = call2(f, multiply, x, y)?;
        let a = call2(f, plus, x, y)?;
        let (s, measured) = snap(f, t, x)?;
        let s_lo = low(f, t, s)?;
        let s_hi = opk(f, BinOp::ShiftRightArithmetic, s, 32)?;
        let s_hi = low(f, t, s_hi)?;
        let (one, zero) = (k32(f, 1)?, k32(f, 0)?);
        let measured = sel(f, measured, one, zero)?;
        let (back, extent) = point_extent(f, t, size)?;
        let back = low(f, t, back)?;
        let extent = low(f, t, extent)?;
        for (at, v) in [
            (3, q),
            (4, m),
            (5, a),
            (6, s_lo),
            (7, s_hi),
            (8, measured),
            (9, back),
            (10, extent),
        ] {
            let p = field(f, at)?;
            store(f, p, v)?;
        }
        Ok(())
    })?;
    f.ret(None)?;
    let id = f.finish()?;
    entry(&mut b, id)?;
    b.finalize()
}

/// The words of one record of [`clip_module`]: the plane (0 near, 1 far, 2 the eye), the
/// polygon's vertices' count, and [`POLYGON`] vertices (four `f32`s' bits each); out, the count
/// and the vertices of the polygon clipped.
#[cfg(test)]
pub(super) const CLIP_WORDS: u32 = 2 + 4 * POLYGON;

/// A probe of the clipper for the tests: invocation `i` reads record `i` of the buffer
/// `PRIMITIVES` (records of [`CLIP_WORDS`], as many as it holds), clips its polygon by its plane
/// as the kernel does where the plane cuts it ([`cut`]), and writes the result into it.
#[cfg(test)]
pub(super) fn clip_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let (t, uvec3) = types(&mut b)?;
    let records = words_buffer(&mut b, PRIMITIVES, t.u32_)?;
    let tiles = words_buffer(&mut b, TILES, t.u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let (r, _) = routines(&mut b, t, tiles)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let records = f.module().global_value(records)?;
    let invocation = f.module().global_value(invocation)?;
    let gid = load(&mut f, invocation)?;
    let i = f.composite_extract(gid, &[0])?;
    let len = f.array_length(records, 0)?;
    let held = opk(&mut f, BinOp::UDiv, len, CLIP_WORDS)?;
    let inside = op(&mut f, BinOp::ULessThan, i, held)?;
    let mut flags = [i; 2];
    for flag in &mut flags {
        let v = f.local_var(t.u32_, None)?;
        *flag = f.var_ptr(v)?;
    }
    let poly = array_var(&mut f, t.vec4, POLYGON)?;
    let scratch = array_var(&mut f, t.vec4, POLYGON)?;
    let pts = array_var(&mut f, t.i64_, 2 * POLYGON)?;
    let n = f.local_var(t.u32_, None)?;
    let n = f.var_ptr(n)?;
    when(&mut f, inside, move |f| {
        let base = opk(f, BinOp::IMul, i, CLIP_WORDS)?;
        let field = |f: &mut FunctionBuilder<'_>, at: u32| {
            let w = opk(f, BinOp::IAdd, base, at)?;
            let zero = k32(f, 0)?;
            f.access_chain(records, &[zero, w])
        };
        let p = field(f, 0)?;
        let plane = load(f, p)?;
        let p = field(f, 1)?;
        let count = load(f, p)?;
        for j in 0..POLYGON {
            let mut c = [i; 4];
            for (at, w) in c.iter_mut().enumerate() {
                let p = field(f, 2 + 4 * j + at as u32)?;
                *w = load(f, p)?;
            }
            let v = f.composite_construct(t.vec4, &c)?;
            let to = atk(f, poly, j)?;
            store(f, to, v)?;
        }
        store(f, n, count)?;
        let zero = k32(f, 0)?;
        let k = K {
            t,
            r,
            near_far: flags[0],
            unmeasured: flags[1],
            poly,
            scratch,
            n,
            pts,
            vp: [zero; 4],
            clamp: falsy(f, t)?,
            cull: zero,
            winding: zero,
            samples: zero,
        };
        cut(f, &k, plane, count)?;
        let made = load(f, n)?;
        let p = field(f, 1)?;
        store(f, p, made)?;
        for j in 0..POLYGON {
            let from = atk(f, poly, j)?;
            let v = load(f, from)?;
            for c in 0..4 {
                let w = f.composite_extract(v, &[c])?;
                let p = field(f, 2 + 4 * j + c)?;
                store(f, p, w)?;
            }
        }
        Ok(())
    })?;
    f.ret(None)?;
    let id = f.finish()?;
    entry(&mut b, id)?;
    b.finalize()
}

/// The entry's body: invocation `i` below the header's count bins primitive `i`, and counts it
/// in the header's near-far and unmeasured words where it is flagged so.
fn main_body(
    f: &mut FunctionBuilder<'_>,
    t: T,
    r: Routines,
    globals: [GlobalId; 4],
) -> BResult<()> {
    let [prims, pass, context, invocation] = globals;
    let prims = f.module().global_value(prims)?;
    let pass = f.module().global_value(pass)?;
    let context = f.module().global_value(context)?;
    let invocation = f.module().global_value(invocation)?;
    let gid = load(f, invocation)?;
    let i = f.composite_extract(gid, &[0])?;
    let at = k32(f, layout::COUNT)?;
    let total = word(f, pass, at)?;
    let inside = op(f, BinOp::ULessThan, i, total)?;
    let near_far = f.local_var(t.u32_, None)?;
    let near_far = f.var_ptr(near_far)?;
    let unmeasured = f.local_var(t.u32_, None)?;
    let unmeasured = f.var_ptr(unmeasured)?;
    let zero = k32(f, 0)?;
    store(f, near_far, zero)?;
    store(f, unmeasured, zero)?;
    let poly = array_var(f, t.vec4, POLYGON)?;
    let scratch = array_var(f, t.vec4, POLYGON)?;
    let pts = array_var(f, t.i64_, 2 * POLYGON)?;
    let n = f.local_var(t.u32_, None)?;
    let n = f.var_ptr(n)?;
    let parts = Parts {
        t,
        r,
        prims,
        pass,
        context,
        near_far,
        unmeasured,
        poly,
        scratch,
        n,
        pts,
    };
    when(f, inside, move |f| {
        primitive(f, parts, i)?;
        for (flag, w) in [
            (near_far, M::NEAR_FAR_WORD),
            (unmeasured, M::UNMEASURED_WORD),
        ] {
            let set = load(f, flag)?;
            let set = opk(f, BinOp::INotEqual, set, 0)?;
            when(f, set, move |f| {
                let zero = k32(f, 0)?;
                let w = k32(f, w)?;
                let p = f.access_chain(pass, &[zero, w])?;
                let one = k32(f, 1)?;
                f.atomic(AtomicOp::IAdd, p, Scope::Device, Semantics::RELAXED, one)?;
                Ok(())
            })?;
        }
        Ok(())
    })
}

/// What [`primitive`] starts from: the main function's buffers and variables.
#[derive(Clone, Copy)]
struct Parts {
    t: T,
    r: Routines,
    prims: Value,
    pass: Value,
    context: Value,
    near_far: Value,
    unmeasured: Value,
    poly: Value,
    scratch: Value,
    n: Value,
    pts: Value,
}

/// Bin primitive `i`: read its record and its state (a state past the pass's is flagged
/// unmeasured and reaches nothing), set the context, and bin it by its kind (another kind is
/// flagged unmeasured and reaches nothing).
fn primitive(f: &mut FunctionBuilder<'_>, p: Parts, i: Value) -> BResult<()> {
    let t = p.t;
    let base = opk(f, BinOp::IMul, i, M::PRIMITIVE_WORDS)?;
    let field = |f: &mut FunctionBuilder<'_>, at: u32| {
        let w = opk(f, BinOp::IAdd, base, at)?;
        word(f, p.prims, w)
    };
    let kind = field(f, layout::KIND)?;
    let state = field(f, layout::STATE)?;
    let layer = field(f, layout::LAYER)?;
    let size = field(f, layout::SIZE)?;
    let mut vertices = Vec::with_capacity(3);
    for v in 0..3 {
        let mut c = Vec::with_capacity(4);
        for i in 0..4 {
            let w = field(f, layout::VERTICES + 4 * v + i)?;
            c.push(w);
        }
        vertices.push(f.composite_construct(t.vec4, &c)?);
    }
    let len = f.array_length(p.pass, 0)?;
    let header = opk(f, BinOp::UGreaterThanEqual, len, M::HEADER_WORDS)?;
    let after = opk(f, BinOp::ISub, len, M::HEADER_WORDS)?;
    let states = opk(f, BinOp::UDiv, after, M::STATE_WORDS)?;
    let zero = k32(f, 0)?;
    let states = sel(f, header, states, zero)?;
    let valid = op(f, BinOp::ULessThan, state, states)?;
    let invalid = not(f, valid)?;
    flag_if(f, p.unmeasured, invalid)?;
    let vertices = [vertices[0], vertices[1], vertices[2]];
    when(f, valid, move |f| {
        let k = setup(f, p, state, layer)?;
        let is = |f: &mut FunctionBuilder<'_>, v: u32| opk(f, BinOp::IEqual, kind, v);
        let point_kind = is(f, M::POINT)?;
        when(f, point_kind, move |f| point(f, &k, vertices[0], size))?;
        let line_kind = is(f, M::LINE)?;
        when(f, line_kind, move |f| {
            line(f, &k, [vertices[0], vertices[1]])
        })?;
        let triangle_kind = is(f, M::TRIANGLE)?;
        when(f, triangle_kind, move |f| triangle(f, &k, vertices))?;
        let other = opk(f, BinOp::UGreaterThan, kind, M::TRIANGLE)?;
        flag_if(f, k.unmeasured, other)
    })
}

/// Read the primitive's state and set the context: the region (the viewport's, flagged
/// unmeasured where its edges lie past the coordinates measured, the scissor's and the render
/// area's), the tiles and the slice (a layer past the pass's renders its first).
fn setup(f: &mut FunctionBuilder<'_>, p: Parts, state: Value, layer: Value) -> BResult<K> {
    let t = p.t;
    let header = |f: &mut FunctionBuilder<'_>, at: u32| {
        let at = k32(f, at)?;
        word(f, p.pass, at)
    };
    let aw = header(f, layout::AREA_W)?;
    let ah = header(f, layout::AREA_H)?;
    let samples = header(f, layout::SAMPLES)?;
    let layers = header(f, layout::LAYERS)?;
    let base = opk(f, BinOp::IMul, state, M::STATE_WORDS)?;
    let base = opk(f, BinOp::IAdd, base, M::HEADER_WORDS)?;
    let field = |f: &mut FunctionBuilder<'_>, at: u32| {
        let w = opk(f, BinOp::IAdd, base, at)?;
        word(f, p.pass, w)
    };
    let mut vp = [state; 4];
    for (i, v) in vp.iter_mut().enumerate() {
        *v = field(f, layout::VIEWPORT + i as u32)?;
    }
    let mut scissor = [state; 4];
    for (i, s) in scissor.iter_mut().enumerate() {
        let w = field(f, layout::SCISSOR + i as u32)?;
        *s = widen(f, t, w)?;
    }
    let cull = field(f, layout::CULL)?;
    let winding = field(f, layout::WINDING)?;
    let clip_mode = field(f, layout::DEPTH_CLIP)?;
    let clamp = opk(f, BinOp::IEqual, clip_mode, 1)?;
    // The viewport's extent along each axis, snapped.
    let mut measured = Vec::with_capacity(4);
    let mut view = [[state; 2]; 2];
    for axis in 0..2 {
        let (o, e) = (vp[axis], vp[axis + 2]);
        let (a, ma) = snap(f, t, o)?;
        let end = call2(f, p.r.add, o, e)?;
        let (b, mb) = snap(f, t, end)?;
        measured.push(ma);
        measured.push(mb);
        view[axis] = [smin(f, a, b)?, smax(f, a, b)?];
    }
    let mut all = measured[0];
    for &m in &measured[1..] {
        all = and(f, all, m)?;
    }
    let off = not(f, all)?;
    flag_if(f, p.unmeasured, off)?;
    let mut region = [state; 4];
    for (axis, extent) in [aw, ah].into_iter().enumerate() {
        let limit = k32(f, AREA_LIMIT)?;
        let big = op(f, BinOp::UGreaterThan, extent, limit)?;
        let extent = sel(f, big, limit, extent)?;
        let extent = widen(f, t, extent)?;
        let extent = op64(f, t, BinOp::IMul, extent, UNIT)?;
        let start = op64(f, t, BinOp::IMul, scissor[axis], UNIT)?;
        let end = op(f, BinOp::IAdd, scissor[axis], scissor[axis + 2])?;
        let end = op64(f, t, BinOp::IMul, end, UNIT)?;
        let zero = k64(f, t, 0)?;
        let lo = smax(f, view[axis][0], start)?;
        let lo = smax(f, lo, zero)?;
        let hi = smin(f, view[axis][1], end)?;
        let hi = smin(f, hi, extent)?;
        region[2 * axis] = lo;
        region[2 * axis + 1] = hi;
    }
    let four = opk(f, BinOp::IEqual, samples, 4)?;
    let (k16, k32_) = (k32(f, 16)?, k32(f, 32)?);
    let th_px = sel(f, four, k16, k32_)?;
    // The tile's sides in pixels and in units as powers of two.
    let (k4, k5) = (k32(f, 4)?, k32(f, 5)?);
    let th_log = sel(f, four, k4, k5)?;
    let mut tiles = [state; 2];
    let mut shifts = [state; 2];
    for (axis, (extent, size, log)) in [(aw, k32_, k5), (ah, th_px, th_log)]
        .into_iter()
        .enumerate()
    {
        let limit = k32(f, AREA_LIMIT)?;
        let big = op(f, BinOp::UGreaterThan, extent, limit)?;
        let extent = sel(f, big, limit, extent)?;
        let up = op(f, BinOp::IAdd, extent, size)?;
        let up = opk(f, BinOp::ISub, up, 1)?;
        let n = op(f, BinOp::ShiftRightLogical, up, log)?;
        tiles[axis] = widen(f, t, n)?;
        let units = opk(f, BinOp::IAdd, log, UNIT.trailing_zeros())?;
        shifts[axis] = widen(f, t, units)?;
    }
    let tw = k64(f, t, 32 * UNIT)?;
    let th = widen(f, t, th_px)?;
    let th = op64(f, t, BinOp::IMul, th, UNIT)?;
    let inside = op(f, BinOp::ULessThan, layer, layers)?;
    let zero = k32(f, 0)?;
    let slice = sel(f, inside, layer, zero)?;
    let slice = widen(f, t, slice)?;
    let first = op(f, BinOp::IMul, slice, tiles[1])?;
    for (at, v) in [
        (ctx::RX0, region[0]),
        (ctx::RX1, region[1]),
        (ctx::RY0, region[2]),
        (ctx::RY1, region[3]),
        (ctx::TW, tw),
        (ctx::TH, th),
        (ctx::TILES_X, tiles[0]),
        (ctx::TILES_Y, tiles[1]),
        (ctx::BASE, first),
        (ctx::SHIFT_X, shifts[0]),
        (ctx::SHIFT_Y, shifts[1]),
    ] {
        let w = atk(f, p.context, at)?;
        store(f, w, v)?;
    }
    Ok(K {
        t,
        r: p.r,
        near_far: p.near_far,
        unmeasured: p.unmeasured,
        poly: p.poly,
        scratch: p.scratch,
        n: p.n,
        pts: p.pts,
        vp,
        clamp,
        cull,
        winding,
        samples,
    })
}
