//! The arc tangent family: `atan`, `asin`, and `acos` in both of Metal's modes, and the generic
//! part of precise `atan2`.
//!
//! **Precise `atan`** is Metal's own form ([`PRECISE_ATAN_P`]): with `a = |x|`, `t = a` below
//! `RN(tan(π/8))`, `(a - 1) rcp(a + 1)` below `RN(tan(3π/8))`, and `-rcp(a)` from there, Metal's
//! reciprocal ([`E::recip`]); then `K + fma(t, RN(RN(t t) P), t)` with `K` zero, `RN(π/4)` and
//! `RN(π/2)`.
//!
//! **Precise `asin` and `acos`** are Metal's own forms, which follow Apple's runtime library and
//! match Metal's measured words: with `a = |x|`, `asin` is the operand below [`ASIN_TINY`],
//! polynomials of `a` below 0.1 ([`ASIN_NEAR`]) and 0.36 ([`ASIN_LOW`]), `a` times polynomials of
//! `a^2` below 0.62 ([`ASIN_MID`]) and 0.815 ([`ASIN_HIGH`]), and from there `RN(π/2) + w P(a)`
//! ([`ASIN_FAR`], fused) of Metal's root `w = z rsqrt(z)` of `z = 1 - a^2` (fused below 0.99,
//! `RN(1 - a) RN(1 + a)` from it), with `x`'s sign; `acos` is `RN(π/2 - a) - a^3 / 6` (fused) to
//! [`ACOS_NEAR_END`], a polynomial of `a` to 1/2 ([`ACOS_MID`]), and `P(u) w` of `u = 1 - a` and
//! Metal's root `w = u rsqrt(u)` above ([`ACOS_FAR`]), taken from `RN(π)` for a negative `x`. Each
//! polynomial is Horner's rule in fused steps from the highest coefficient, and the inverse root is
//! the hardware's model ([`E::fast_rsqrt`]).
//!
//! **Fast `asin` and `acos`** are Metal's own forms ([`FAST_ASIN_P`]): one polynomial below 0.7 and
//! the half-angle form with Metal's root of `(1 - |x|) / 2` from there.
//!
//! **Precise `atan2`** (finite nonzero operands; `math::atan2_special` answers the rest) is
//! Metal's precise `atan` of the correctly rounded quotient `y / x` ([`E::div`]), and that plus
//! `RN(π)` toward a negative `x` (minus it where `y` is negative too).
//!
//! Metal's values at the special operands (cases 087758cb and cc234732): `atan` and `asin` are
//! odd and `acos(±1)` is `+0` and `π`, as the reductions give; an infinite `atan` operand gives
//! `±π/2`; a denormal operand is read as [`denormal::operand`] says, and a zero then answers
//! `+0` for precise `atan` and `fast::asin` (whatever its sign) and the zero itself otherwise;
//! below `2^-12` in magnitude the precise `atan(x)` rounds to `x`, which is selected, so a device
//! that flushes the operand's cube answers it too. `|x| > 1` gives `asin` and `acos` Metal's NaN.

use crate::lower::arith::denormal;
use crate::lower::arith::libm::{Fun, Mode, E, F32};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, UnOp};

/// Metal's precise `asin(a)` is `a` below this (`0x3a9285fe`, about `1.118e-3`).
pub(crate) const ASIN_TINY: u32 = 0x3A92_85FE;

/// Metal's precise `asin(a)` below 0.1: a polynomial of degree 5 in `a` (`f32` bits, lowest
/// first).
pub(crate) const ASIN_NEAR: [u32; 6] = [
    0xAD75_05AF,
    0x3F80_0000,
    0xB52F_1CE4,
    0x3E2A_B123,
    0xB9D6_2D9E,
    0x3D9F_092F,
];

/// Metal's precise `asin(a)` from 0.1 to 0.36: a polynomial of degree 6 in `a`.
pub(crate) const ASIN_LOW: [u32; 7] = [
    0x35E8_9736,
    0x3F7F_FBDB,
    0x3A78_7C88,
    0x3E22_F112,
    0x3D0C_250A,
    0xBC47_34E4,
    0x3DE2_9FE7,
];

/// Metal's precise `asin(a)` from 0.36 to 0.62: `a Q(a^2)`, `Q` of degree 5.
pub(crate) const ASIN_MID: [u32; 6] = [
    0x3F7F_FF08,
    0x3E2B_0F18,
    0x3D91_80AC,
    0x3D85_A0DD,
    0xBCD1_3BD9,
    0x3DBC_499F,
];

/// Metal's precise `asin(a)` from 0.62 to [`ASIN_FAR_START`]: `a Q(a^2)`, `Q` of degree 6.
pub(crate) const ASIN_HIGH: [u32; 7] = [
    0x3F81_6B2F,
    0x3CC1_EB4C,
    0x3F58_74B5,
    0xC00B_91CD,
    0x406B_7509,
    0xC04E_468F,
    0x3FA2_2DD9,
];

/// Where Metal's precise `asin` takes its root form (`0x3f50a3d7`, about 0.815).
pub(crate) const ASIN_FAR_START: u32 = 0x3F50_A3D7;

/// Metal's precise `asin(a)` from [`ASIN_FAR_START`]: `RN(π/2) + w P(a)`, `P` of degree 4.
pub(crate) const ASIN_FAR: [u32; 5] = [
    0xBFC6_F7E7,
    0x3F65_16AB,
    0xBEFB_9783,
    0x3E3A_F607,
    0xBD01_89D6,
];

/// Metal's precise `acos(a)` is `RN(π/2 - a) - a^3 / 6` to this (`0x3d99a8ca`, about 0.075).
pub(crate) const ACOS_NEAR_END: u32 = 0x3D99_A8CA;

/// Metal's precise `acos(a)` from [`ACOS_NEAR_END`] to 1/2: a polynomial of degree 8 in `a`.
pub(crate) const ACOS_MID: [u32; 9] = [
    0x3FC9_0FD4,
    0xBF7F_FDCF,
    0xBA23_8B67,
    0xBE24_30FB,
    0xBD1D_272C,
    0x3D8D_89DA,
    0xBEA9_EE80,
    0x3ECA_14C3,
    0xBE8C_EF6E,
];

/// Metal's precise `acos(a)` above 1/2: `P(u) w` of `u = 1 - a`, `P` of degree 6.
pub(crate) const ACOS_FAR: [u32; 7] = [
    0x3FB5_04F3,
    0x3DF1_5BE6,
    0x3CD9_3DAE,
    0x3C00_E404,
    0x3B38_D9E4,
    0x3A23_5EB2,
    0x3A5B_BFCA,
];

/// `x` as `fun` reads it ([`denormal::read`]).
pub(crate) fn read(e: &mut E<'_, '_>, fun: Fun, x: Value) -> BResult<Value> {
    match fun.float_fn() {
        Some(op) => denormal::read(e.fb, op, F32, x),
        None => Ok(x),
    }
}

/// `Q(s)` of Metal's fast `atan(t) = t Q(t^2)` on `[0, 1]`, its `f32` coefficients by their bits,
/// read off Metal's words (clean-room fit): Horner's rule in `s = t t` (rounded),
/// each step one fused multiply-add, then `t Q` rounded.
pub(crate) const FAST_ATAN_Q: [u32; 8] = [
    0x3F7F_FFFF,
    0xBEAA_A90B,
    0x3E4C_81C3,
    0xBE0F_A948,
    0x3DCB_A0B2,
    0xBD75_5EA6,
    0x3CC9_2C07,
    0xBB9C_A55F,
];

/// `RN(π/2)`, which Metal's fast `atan` subtracts from above 1.
const HALF_PI_BITS: u32 = 0x3FC9_0FDB;

/// Metal's fast `atan(x)`: `t Q(t^2)` of [`FAST_ATAN_Q`] for `t = |x|` to 1, and `RN(π/2) - t
/// Q(t^2)` of Metal's reciprocal `t = rcp(|x|)` above (`E::recip`; an infinity and a magnitude past
/// `2^126`, whose reciprocal Metal flushes, give `RN(π/2)`), with `x`'s sign. A magnitude at or
/// below `2^-126` gives a zero: Metal flushes the product `t Q`, which is below `2^-126` there
/// before rounding. Metal's word on every `float` of `[2^-12, 1]` and every 256th of the others,
/// and above 1 wherever the reciprocal is Metal's.
fn fast_atan(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let x = read(e, Fun::Atan(Mode::Fast), x)?;
    let a = e.abs(x)?;
    let one = e.c(1.0)?;
    let above = e.gt(a, one)?;
    let r = e.recip(a)?;
    let t = e.sel(above, r, a)?;
    let s = e.mul(t, t)?;
    let (last, rest) = FAST_ATAN_Q
        .split_last()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
    let mut q = e.c(f32::from_bits(*last))?;
    for c in rest.iter().rev() {
        let c = e.c(f32::from_bits(*c))?;
        q = e.fused(q, s, c)?;
    }
    let core = e.mul(t, q)?;
    let h = e.c(f32::from_bits(HALF_PI_BITS))?;
    let far = e.sub(h, core)?;
    let r = e.sel(above, far, core)?;
    let least = e.c(f32::MIN_POSITIVE)?;
    let flushed = e.le(a, least)?;
    let zero = e.c(0.0)?;
    let r = e.sel(flushed, zero, r)?;
    e.copysign(r, x)
}

/// `P(s)` of Metal's precise `atan(t) = t + t s P(s)`, `s = t t` (rounded), for `|t|` up to
/// `tan(π/8)`, its `f32` coefficients by their bits: the coefficients of the public Cephes `atanf`
/// (Moshier), which a clean-room fit of Metal's words converged on, rounded to
/// `float`. Horner's rule with one fused multiply-add per step.
pub(crate) const PRECISE_ATAN_P: [u32; 4] = [0xBEAA_AA2A, 0x3E4C_925F, 0xBE0E_1B85, 0x3DA4_F0D1];

/// `RN(tan(π/8))` and `RN(tan(3π/8))`, from which Metal's precise `atan` reduces its operand.
const TAN_PI_8_BITS: u32 = 0x3ED4_13CD;
const TAN_3PI_8_BITS: u32 = 0x401A_827A;

/// `RN(π/4)`, which Metal's precise `atan` adds between [`TAN_PI_8_BITS`] and
/// [`TAN_3PI_8_BITS`].
const QUARTER_PI_BITS: u32 = 0x3F49_0FDB;

/// Metal's precise `atan(x)`: with `a = |x|`, `t = a` below `RN(tan(π/8))`, `(a - 1) rcp(a + 1)`
/// below `RN(tan(3π/8))`, and `-rcp(a)` from there, of Metal's reciprocal ([`E::recip`]; an
/// infinity and a magnitude past `2^126`, whose reciprocal Metal flushes, give `RN(π/2)`); then
/// `K + fma(t, RN(RN(t t) P(t t)), t)` of [`PRECISE_ATAN_P`] with `K` zero, `RN(π/4)` and `RN(π/2)`
/// there, and `x`'s sign. Below `2^-12` it is `x` (selected, so the emulated fused steps never take
/// products that small), and a zero gives `+0`, a flushed denormal too. Metal's word on every
/// `float` of `[0.25, 4)` and every 256th positive one elsewhere (the M3) wherever
/// the reciprocal is Metal's, which it is at all but 0.08% of them.
fn precise_atan(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let x = read(e, Fun::Atan(Mode::Precise), x)?;
    metal_atan(e, x)
}

/// [`precise_atan`] of an operand already read as Metal reads it: also `atan2`'s.
fn metal_atan(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let a = e.abs(x)?;
    let low = e.c(f32::from_bits(TAN_PI_8_BITS))?;
    let high = e.c(f32::from_bits(TAN_3PI_8_BITS))?;
    let near = e.lt(a, low)?;
    let middle = e.lt(a, high)?;
    let one = e.c(1.0)?;
    let up = e.add(a, one)?;
    let down = e.sub(a, one)?;
    let divisor = e.sel(middle, up, a)?;
    let r = e.recip(divisor)?;
    let quotient = e.mul(down, r)?;
    let zero = e.c(0.0)?;
    let far = e.sub(zero, r)?;
    let t = e.sel(middle, quotient, far)?;
    let t = e.sel(near, a, t)?;
    let s = e.mul(t, t)?;
    let (last, rest) = PRECISE_ATAN_P
        .split_last()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
    let mut p = e.c(f32::from_bits(*last))?;
    for c in rest.iter().rev() {
        let c = e.c(f32::from_bits(*c))?;
        p = e.fused(p, s, c)?;
    }
    let ps = e.mul(p, s)?;
    let core = e.fused(t, ps, t)?;
    let quarter = e.c(f32::from_bits(QUARTER_PI_BITS))?;
    let half = e.c(f32::from_bits(HALF_PI_BITS))?;
    let k = e.sel(middle, quarter, half)?;
    let k = e.sel(near, zero, k)?;
    let r = e.add(k, core)?;
    let r = e.copysign(r, x)?;
    let tiny = e.c(2f32.powi(-12))?;
    let is_tiny = e.lt(a, tiny)?;
    let r = e.sel(is_tiny, x, r)?;
    // Precise `atan(-0)` is `+0` (a flushed denormal too).
    let is_zero = e.fb.binary(BinOp::FOrdEqual, a, zero)?;
    e.sel(is_zero, zero, r)
}

/// Metal's `atan(x)` in mode `m`.
pub(crate) fn atan(e: &mut E<'_, '_>, m: Mode, x: Value) -> BResult<Value> {
    match m {
        Mode::Fast => fast_atan(e, x),
        Mode::Precise => precise_atan(e, x),
    }
}

/// `P(s)` of Metal's fast `asin(x) = x P(x^2)` below 0.7, its `f32` coefficients by their bits,
/// read off Metal's words (clean-room fit): Horner's rule in `s = x x` (rounded),
/// each step one fused multiply-add. Its fast `acos` takes the same `P`, and both take it at `z =
/// (1 - |x|) / 2` from 0.7 on.
pub(crate) const FAST_ASIN_P: [u32; 7] = [
    0x3F80_0001,
    0x3E2A_A575,
    0x3D9A_CEE0,
    0x3D1C_4BA7,
    0x3D81_F491,
    0xBD73_BDCF,
    0x3DD4_CDC2,
];

/// `RN(π)`, which Metal's fast `acos` of a negative operand from -0.7 down starts from.
const PI_BITS: u32 = 0x4049_0FDB;

/// The pieces of Metal's fast `asin` and `acos` of magnitude `a`: whether `a` is below 0.7,
/// [`FAST_ASIN_P`] at `a^2` there and at `z = (1 - a) / 2` from 0.7 on, `z`, `z`'s inverse
/// root, and whether `z` is zero (`a = 1`, whose root is then `0`).
struct FastArcs {
    near: Value,
    p: Value,
    z: Value,
    rs: Value,
    zero_z: Value,
}

fn fast_arcs(e: &mut E<'_, '_>, a: Value) -> BResult<FastArcs> {
    let bound = e.c(0.7)?;
    let near = e.lt(a, bound)?;
    let s = e.mul(a, a)?;
    let (one, half) = (e.c(1.0)?, e.c(0.5)?);
    // `1 - a` is exact from `a = 1/2` on (Sterbenz), and so is its half.
    let rest = e.sub(one, a)?;
    let z = e.mul(rest, half)?;
    let v = e.sel(near, s, z)?;
    let (last, rest) = FAST_ASIN_P
        .split_last()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
    let mut p = e.c(f32::from_bits(*last))?;
    for c in rest.iter().rev() {
        let c = e.c(f32::from_bits(*c))?;
        p = e.fused(p, v, c)?;
    }
    // `z`'s inverse root where it is read (`z` is zero or at least `2^-25` from 0.7 on); 1/4
    // elsewhere, so no lane takes a denormal's.
    let quarter = e.c(0.25)?;
    let zr = e.sel(near, quarter, z)?;
    let rs = e.fast_rsqrt(zr)?;
    let zero = e.c(0.0)?;
    let zero_z = e.fb.binary(BinOp::FOrdEqual, z, zero)?;
    Ok(FastArcs {
        near,
        p,
        z,
        rs,
        zero_z,
    })
}

/// Metal's fast `asin(x)`: `|x| P(x^2)` of [`FAST_ASIN_P`] below 0.7, and from there `RN(π/2) - 2 w
/// P(z)` in one fused multiply-add, `z = (1 - |x|) / 2` and `w = z rsqrt(z)` (Metal's fast square
/// root), with `x`'s sign; `+0` at a zero (`fast::asin(-0)` is `+0`, a flushed denormal too) and
/// Metal's NaN, with `x`'s sign, past 1 in magnitude. Metal's `rsqrt` is the hardware's, which is
/// correctly rounded on 95.6% of significands; the routine takes its model ([`E::fast_rsqrt`]), and
/// so is Metal's word on every lane below 0.7 that its polynomial is and on 99.88 % of `[0.7, 1)`
/// (97.39 % with the correctly rounded root).
fn fast_asin(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let x = read(e, Fun::Asin(Mode::Fast), x)?;
    let a = e.abs(x)?;
    let k = fast_arcs(e, a)?;
    let near = e.mul(a, k.p)?;
    let w = e.mul(k.z, k.rs)?;
    let zero = e.c(0.0)?;
    let w = e.sel(k.zero_z, zero, w)?;
    let minus_two = e.c(-2.0)?;
    let m2w = e.mul(minus_two, w)?;
    let h = e.c(f32::from_bits(HALF_PI_BITS))?;
    let far = e.fused(m2w, k.p, h)?;
    let r = e.sel(k.near, near, far)?;
    let r = e.copysign(r, x)?;
    let is_zero = e.fb.binary(BinOp::FOrdEqual, a, zero)?;
    let r = e.sel(is_zero, zero, r)?;
    let one = e.c(1.0)?;
    let domain = e.gt(a, one)?;
    let nan = e.metal_nan(Some(x))?;
    e.sel(domain, nan, r)
}

/// Metal's fast `acos(x)`: `RN(π/2) - x P(x^2)` of [`FAST_ASIN_P`] below 0.7 in magnitude, in one
/// fused multiply-add; from there, with `z = (1 - |x|) / 2`, Metal's root `w` of `z`, `2 RN(w
/// P(z))` for `x > 0` and `RN(π) - 2 w P(z)` (fused) for `x < 0`; Metal's NaN past 1. The root is
/// `w1 = z r` refined once, `w1 + w1 RN(1/2 - w1 RN(r / 2))` (two fused steps), of the inverse root
/// `r` (Metal's hardware `rsqrt`'s model, as in [`fast_asin`]): Metal's word on every lane below
/// 0.7 that the polynomial is and on 99.94 % of `[0.7, 1)` and 99.99 % of `(-1, -0.7]` (99.05 % and
/// 99.82 % with the correctly rounded root).
fn fast_acos(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let x = read(e, Fun::Acos(Mode::Fast), x)?;
    let a = e.abs(x)?;
    let k = fast_arcs(e, a)?;
    let h = e.c(f32::from_bits(HALF_PI_BITS))?;
    let minus_one = e.c(-1.0)?;
    let nx = e.mul(minus_one, x)?;
    let near = e.fused(nx, k.p, h)?;
    let half = e.c(0.5)?;
    let w1 = e.mul(k.z, k.rs)?;
    let hr = e.mul(half, k.rs)?;
    let nw1 = e.mul(minus_one, w1)?;
    let t = e.fused(nw1, hr, half)?;
    let w = e.fused(w1, t, w1)?;
    let zero = e.c(0.0)?;
    let w = e.sel(k.zero_z, zero, w)?;
    let wp = e.mul(w, k.p)?;
    let two = e.c(2.0)?;
    let positive = e.mul(two, wp)?;
    let minus_two = e.c(-2.0)?;
    let m2w = e.mul(minus_two, w)?;
    let pi = e.c(f32::from_bits(PI_BITS))?;
    let negative = e.fused(m2w, k.p, pi)?;
    let above = e.gt(x, zero)?;
    let far = e.sel(above, positive, negative)?;
    let r = e.sel(k.near, near, far)?;
    let one = e.c(1.0)?;
    let domain = e.gt(a, one)?;
    let nan = e.metal_nan(None)?;
    e.sel(domain, nan, r)
}

/// `z rsqrt(z)` of `z >= 0`, Metal's root through its hardware's inverse root ([`E::fast_rsqrt`]),
/// and `0` at `z = 0` (whose inverse root is infinite).
fn metal_root(e: &mut E<'_, '_>, z: Value) -> BResult<Value> {
    let r = e.fast_rsqrt(z)?;
    let w = e.mul(z, r)?;
    let zero = e.c(0.0)?;
    let at_zero = e.fb.binary(BinOp::FOrdEqual, z, zero)?;
    e.sel(at_zero, zero, w)
}

/// Whether the magnitude `a` lies below the `float` of bits `bound` (by the bits).
fn below(e: &mut E<'_, '_>, a: Value, bound: u32) -> BResult<Value> {
    let ab = e.bits(a)?;
    let k = e.u(bound)?;
    e.iop(BinOp::ULessThan, ab, k)
}

/// Metal's `asin(x)` in mode `m`; the precise one Metal's form (the module's list), each region
/// computed and the operand's selected.
pub(crate) fn asin(e: &mut E<'_, '_>, m: Mode, x: Value) -> BResult<Value> {
    if m == Mode::Fast {
        return fast_asin(e, x);
    }
    let x = read(e, Fun::Asin(m), x)?;
    let a = e.abs(x)?;
    // From 0.815: `RN(π/2) + w P(a)`, `z = 1 - a^2` fused below 0.99.
    let one = e.c(1.0)?;
    let minus_a = e.fb.unary(UnOp::FNegate, a)?;
    let fused = e.fused(minus_a, a, one)?;
    let down = e.sub(one, a)?;
    let up = e.add(a, one)?;
    let product = e.mul(down, up)?;
    let near_one = e.c(0.99)?;
    let below_near_one = e.lt(a, near_one)?;
    let z = e.sel(below_near_one, fused, product)?;
    let w = metal_root(e, z)?;
    let p = e.fused_poly(&ASIN_FAR, a)?;
    let h = e.c(f32::from_bits(HALF_PI_BITS))?;
    let r = e.fused(w, p, h)?;
    // Below 0.815 and 0.62: `a Q(a^2)`.
    let s = e.mul(a, a)?;
    let q = e.fused_poly(&ASIN_HIGH, s)?;
    let high = e.mul(q, a)?;
    let in_high = below(e, a, ASIN_FAR_START)?;
    let r = e.sel(in_high, high, r)?;
    let q = e.fused_poly(&ASIN_MID, s)?;
    let mid = e.mul(q, a)?;
    let bound = e.c(0.62)?;
    let in_mid = e.lt(a, bound)?;
    let r = e.sel(in_mid, mid, r)?;
    // Below 0.36 and 0.1: polynomials of `a`.
    let low = e.fused_poly(&ASIN_LOW, a)?;
    let bound = e.c(0.36)?;
    let in_low = e.lt(a, bound)?;
    let r = e.sel(in_low, low, r)?;
    let near = e.fused_poly(&ASIN_NEAR, a)?;
    let bound = e.c(0.1)?;
    let in_near = e.lt(a, bound)?;
    let r = e.sel(in_near, near, r)?;
    let r = e.copysign(r, x)?;
    // Below `ASIN_TINY`: the operand itself (a kept denormal included).
    let tiny = below(e, a, ASIN_TINY)?;
    let r = e.sel(tiny, x, r)?;
    // `|x| > 1`: Metal's NaN.
    let domain = e.gt(a, one)?;
    let nan = e.metal_nan(None)?;
    e.sel(domain, nan, r)
}

/// Metal's `acos(x)` in mode `m`; the precise one Metal's form (the module's list), each region
/// computed and the operand's selected, then `RN(π - r)` for a negative `x`.
pub(crate) fn acos(e: &mut E<'_, '_>, m: Mode, x: Value) -> BResult<Value> {
    if m == Mode::Fast {
        return fast_acos(e, x);
    }
    let x = read(e, Fun::Acos(m), x)?;
    let a = e.abs(x)?;
    // Above 1/2: `P(u) w` of `u = 1 - a` (exact) and `w = u rsqrt(u)`; `+0` at `a = 1`.
    let one = e.c(1.0)?;
    let u = e.sub(one, a)?;
    let w = metal_root(e, u)?;
    let p = e.fused_poly(&ACOS_FAR, u)?;
    let r = e.mul(p, w)?;
    // To 1/2: a polynomial of `a`.
    let mid = e.fused_poly(&ACOS_MID, a)?;
    let half = e.c(0.5)?;
    let in_mid = e.le(a, half)?;
    let r = e.sel(in_mid, mid, r)?;
    // To `ACOS_NEAR_END`: `RN(π/2 - a) - a^3 / 6`, fused.
    let s = e.mul(a, a)?;
    let cube = e.mul(a, s)?;
    let h = e.c(f32::from_bits(HALF_PI_BITS))?;
    let d = e.sub(h, a)?;
    let sixth = e.c(f32::from_bits(0xBE2A_AAAB))?;
    let near = e.fused(cube, sixth, d)?;
    let end = e.c(f32::from_bits(ACOS_NEAR_END))?;
    let in_near = e.le(a, end)?;
    let r = e.sel(in_near, near, r)?;
    // A negative `x`: `RN(π - r)`.
    let pi = e.c(f32::from_bits(PI_BITS))?;
    let turned = e.sub(pi, r)?;
    let zero = e.c(0.0)?;
    let negative = e.lt(x, zero)?;
    let r = e.sel(negative, turned, r)?;
    // `|x| > 1`: Metal's NaN.
    let domain = e.gt(a, one)?;
    let nan = e.metal_nan(None)?;
    e.sel(domain, nan, r)
}

/// The generic part of Metal's precise `atan2(y, x)` for finite nonzero normal operands (any
/// other lane's answer is `math::atan2_special`'s, a zero or denormal answer included): Metal's
/// precise `atan` ([`metal_atan`]) of the correctly rounded quotient `q = y / x` ([`E::div`]), and
/// `RN(atan(q) + RN(π))` toward a negative `x`, `RN(atan(q) - RN(π))` where `y` is negative too.
/// Metal's word on every finite normal pair of our probe (16.7 million, the M4) whose answer
/// is normal, wherever the reciprocal in its `atan` is Metal's, which it is at all but 0.012% of
/// them.
pub(crate) fn atan2(e: &mut E<'_, '_>, y: Value, x: Value) -> BResult<Value> {
    let q = e.div(y, x)?;
    let t = metal_atan(e, q)?;
    turned(e, t, y, x)
}

/// `t`, the arc tangent of `y / x`, turned to `atan2(y, x)`'s quadrant: `RN(t + RN(π))` toward a
/// negative `x`, `RN(t - RN(π))` where `y` is negative too, and `t` itself toward a positive
/// one.
fn turned(e: &mut E<'_, '_>, t: Value, y: Value, x: Value) -> BResult<Value> {
    let pi = e.c(f32::from_bits(PI_BITS))?;
    let up = e.add(t, pi)?;
    let down = e.sub(t, pi)?;
    let zero = e.c(0.0)?;
    let below = e.lt(y, zero)?;
    let turned = e.sel(below, down, up)?;
    let behind = e.lt(x, zero)?;
    e.sel(behind, turned, t)
}

/// `P(s)` of Metal's `half` `atan(a) = a P(a a)` to 1 and `RN(π/2) - t P(t t)` of `t = 1 / a`
/// above, its `f32` coefficients by their bits: read off Metal's words at every `half` (the M3; the
/// M4's, are the same) by a clean-room fit. Horner's rule in `s` (rounded), each step a product and
/// a sum, both rounded.
pub(crate) const HALF_ATAN_P: [u32; 4] = [0x3F7F_F382, 0xBEA7_01E6, 0x3E20_517B, 0xBD38_2B03];

/// Metal's `half` `atan(x)` before its rounding to `half`, of a `half` `x` held as `float`: with
/// `a = |x|`, `a P(a a)` of [`HALF_ATAN_P`] to 1, and above it `RN(π/2) - t P(t t)` of the
/// correctly rounded `t = 1 / a` ([`E::div`]; `0` for an infinite `a`), with `x`'s sign, so a zero
/// gives itself. Rounded to `half`, Metal's word at every `half`, whatever the steps' fusion and
/// whether `t` is the quotient or Metal's reciprocal: the form is no closer to any word's rounding
/// edge than that.
pub(crate) fn half_atan(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let a = e.abs(x)?;
    let one = e.c(1.0)?;
    let above = e.gt(a, one)?;
    // The quotient's divisor is `a` above 1 and 1 elsewhere, so it is finite on every lane
    // that reads it; an infinite `a` takes `t = 0`.
    let divisor = e.sel(above, a, one)?;
    let t = e.div(one, divisor)?;
    let max = e.c(f32::MAX)?;
    let infinite = e.gt(a, max)?;
    let zero = e.c(0.0)?;
    let t = e.sel(infinite, zero, t)?;
    let t = e.sel(above, t, a)?;
    let s = e.mul(t, t)?;
    let p = half_atan_p(e, s)?;
    let core = e.mul(t, p)?;
    let h = e.c(f32::from_bits(HALF_PI_BITS))?;
    let far = e.sub(h, core)?;
    let r = e.sel(above, far, core)?;
    e.copysign(r, x)
}

/// [`HALF_ATAN_P`] at `s`.
fn half_atan_p(e: &mut E<'_, '_>, s: Value) -> BResult<Value> {
    let c = HALF_ATAN_P.map(f32::from_bits);
    e.horner(&c, s)
}

/// Metal's `half` `atan2(y, x)` before its rounding to `half`, of finite nonzero `half` operands
/// held as `float` (any other lane's answer is `math::atan2_special`'s): [`half_atan`] of the
/// correctly rounded quotient `q = y / x` ([`E::div`], normal for every such pair), turned to the
/// quadrant as the precise `atan2` is (`RN(π)` added toward a negative `x`, subtracted where `y` is
/// negative too). Rounded to `half`, Metal's word at every such pair of our probe (65,536
/// pairs; the M4's words are the same).
pub(crate) fn half_atan2(e: &mut E<'_, '_>, y: Value, x: Value) -> BResult<Value> {
    let q = e.div(y, x)?;
    let t = half_atan(e, q)?;
    turned(e, t, y, x)
}
