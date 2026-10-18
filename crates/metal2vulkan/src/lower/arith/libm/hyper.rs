//! The hyperbolic functions and their inverses: Metal's precise `sinh`, `cosh`, `tanh`, `asinh`,
//! `acosh`, and `atanh` of a `float`, which MSL Table 8.1 bounds at 4, 4, 5, 4, 4, and 5 ulp. Each
//! is Metal's own form, which matches Metal's measured words and follows Apple's runtime library:
//! polynomials near zero and in the middle ranges, then compositions of Metal's precise `exp`
//! ([`exp::exp_base`]), `log2` ([`log::metal_log2`], with its table), and reciprocal
//! ([`E::recip`]).
//!
//! - `sinh` and `cosh` ([`hyper`]): polynomials below 1.5, `(E ∓ 1/E) / 2` of `E = e^a` to
//!   `12 ln 2`, and `e^a / 2` past it;
//! - `tanh`: a polynomial of `s = a^2` below 0.35, one of `a` ([`TANH_MIDDLE`]) to 1.25, `(t - 1)
//!   (1 / (t + 1))` of `t = e^(2a)` below `12 ln 2`, and 1 from it;
//! - `asinh` ([`arc`]): the operand below [`ASINH_TINY`], `a Q(a^2)` ([`ASINH_NEAR`]) below 0.35,
//!   a polynomial of `a` ([`ASINH_MIDDLE`]) below 1.5, `RN(RN(log2(w)) RN(ln 2))` of `w = RN(a +
//!   RN(z rsqrt(z)))` with `z = RN(a^2 + 1)` fused below 500, and `RN(RN(RN(log2(a)) RN(ln 2)) +
//!   RN(ln 2))` from 500;
//! - `acosh`: `L + RN(err rcp(w))` of the same `w = RN(q + x)` with `q = RN(z rsqrt(z))` and `z =
//!   RN(RN(x - 1) RN(x + 1))`, `err = q - (w - x)` the sum's rounding error, and `L` the
//!   polynomial `t P(t)` of `t = w - 1` ([`ACOSH_LOG1P`]) for `w` in `[1.0154, 1.2811)`, else
//!   `RN(RN(log2(w)) RN(ln 2))`; from [`ACOSH_LARGE`] on, the `log2(x)` form above;
//! - `atanh`: the operand to [`ATANH_TINY`], a polynomial of `a` ([`ATANH_NEAR`]) to 1/4, and
//!   `RN(RN(log2(q)) RN(ln 2)) / 2` of `q = RN(RN(1 + a) rcp(RN(1 - a)))` above.
//!
//! The polynomials are Horner's rule from the highest coefficient, each step one fused
//! multiply-add, their `f32` coefficients by their bits, lowest first. Metal's word at every
//! operand where its hardware's primitives are the models': its `log2` and reciprocal are fitted
//! models (a unit off at a few lanes), and so is the hardware's inverse root that its `asinh` and
//! `acosh` take ([`E::fast_rsqrt`], Metal's word at 99.82 % of `[1, 4)`): with it `asinh` is
//! Metal's word at 99.86 % of `[1.5, 4)` and `acosh` at 99.87 % of `[1, 16)` (98.67 % and 98.81 %
//! with the correctly rounded root).
//!
//! Metal's precise functions return a denormal operand itself (case 7b283404), as `sinh` and
//! `tanh` do, and `cosh` of one is `1`. The odd functions take the operand's sign at the end.
//! Metal's values at the special operands (case 7b283404, every lane): `sinh(±inf) = ±inf`,
//! `cosh(±inf) = +inf`, `tanh(±inf) = ±1`, `asinh(±inf) = ±inf`, `acosh(+inf) = +inf`, `atanh(±1)
//! = ±inf`; `acosh` below 1 (a zero and a denormal among them) and `atanh` beyond 1 answer
//! Metal's NaN `0x7fc00000`, and a NaN operand a NaN.
//!
//! Metal's fast `asinh` and `acosh` ([`fast_arc`]) and its `half` `asinh` ([`half_asinh`]) are
//! their own shorter forms: `log2` of a sum the hardware's inverse root's product is fused into.

use crate::lower::arith::libm::exp;
use crate::lower::arith::libm::log;
use crate::lower::arith::libm::E;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, UnOp};

/// Metal's polynomials near zero: `1 + s q(s)` of `s = x x` by Horner's rule,
/// each step one fused multiply-add, `q`'s `f32` coefficients by their bits. `sinh(x) = x (1 + s
/// q)` and `cosh(x) = 1 + s q` below 1.5 in magnitude, `tanh(x) = x (1 + s q)` below 0.35.
pub(crate) const SINH_Q: [u32; 4] = [0x3E2A_AAA8, 0x3C08_894C, 0x394F_C8DE, 0x3642_2719];
pub(crate) const COSH_Q: [u32; 4] = [0x3EFF_FFF3, 0x3D2A_AC9E, 0x3AB5_B161, 0x37DC_86D5];
pub(crate) const TANH_Q: [u32; 5] = [
    0xBEAA_AAAB,
    0x3E08_8889,
    0xBD5D_0DD3,
    0x3CB3_280E,
    0xBC11_3C58,
];

/// Metal's `tanh(a)` from 0.35 to 1.25: a polynomial of degree 8 in `a`.
pub(crate) const TANH_MIDDLE: [u32; 9] = [
    0xB95D_6161,
    0x3F80_668A,
    0xBCA5_F7E6,
    0xBE84_2FE5,
    0xBE33_C007,
    0x3ECC_0D09,
    0xBE80_CAEC,
    0x3D98_DFD6,
    0xBC13_2FA1,
];

/// Metal's `asinh(a)` from [`ASINH_TINY`] to 0.35: `a Q(a^2)`, `Q` of degree 3.
pub(crate) const ASINH_NEAR: [u32; 4] = [0x3F7F_FFFF, 0xBE2A_A75C, 0x3D98_9340, 0xBD1B_89C2];

/// Metal's `asinh(a)` from 0.35 to 1.5: a polynomial of degree 8 in `a`.
pub(crate) const ASINH_MIDDLE: [u32; 9] = [
    0x39A6_3B75,
    0x3F7F_0E87,
    0x3C8F_A3F1,
    0xBE56_5E46,
    0x3D4C_6073,
    0x3D93_5FC0,
    0xBD83_E03D,
    0x3CB3_FAF1,
    0xBB3C_F9B9,
];

/// Metal's `log(w)` next to 1 in its `acosh`: `t P(t)` of `t = w - 1`, `P` of degree 5.
pub(crate) const ACOSH_LOG1P: [u32; 6] = [
    0x3F7F_FFFE,
    0xBEFF_FEA0,
    0x3EAA_837C,
    0xBE7C_1408,
    0x3E33_542C,
    0xBDA9_125E,
];

/// Metal's `atanh(a)` from [`ATANH_TINY`] to 1/4: a polynomial of degree 7 in `a`.
pub(crate) const ATANH_NEAR: [u32; 8] = [
    0xAEE6_1A7E,
    0x3F80_0000,
    0xB658_5560,
    0x3EAA_B74B,
    0xBAB8_927B,
    0x3E58_0007,
    0xBD36_5927,
    0x3E63_29B2,
];

/// To these magnitudes (bits) Metal's `sinh` and `tanh` are the operand and its `cosh` is 1.
pub(crate) const SINH_TINY: u32 = 0x3A85_2018;
pub(crate) const TANH_TINY: u32 = 0x3A38_9BA2;
pub(crate) const COSH_TINY: u32 = 0x3A35_04F2;

/// Below 1.5 `sinh` and `cosh` are their polynomials; below 0.35 `tanh` is [`TANH_Q`]'s.
pub(crate) const POLY_END: u32 = 0x3FC0_0000;
pub(crate) const TANH_POLY_END: u32 = 0x3EB3_3333;

/// Past 1.25 `tanh` is `(t - 1) / (t + 1)` of `t = e^(2a)`; to it, [`TANH_MIDDLE`].
pub(crate) const TANH_QUOTIENT: u32 = 0x3FA0_0000;

/// `12 ln 2`: past it `sinh` and `cosh` are `e^a / 2`, and from it `tanh` is 1.
pub(crate) const HALF_EXP: u32 = 0x4105_1592;

/// Below this magnitude (bits) Metal's `asinh` is its operand.
pub(crate) const ASINH_TINY: u32 = 0x3A8F_950B;

/// Below this magnitude (bits, the `float` below 0.35) `asinh` is [`ASINH_NEAR`]'s.
pub(crate) const ASINH_NEAR_END: u32 = 0x3EB3_3332;

/// From this magnitude (bits, 1.5) `asinh` is its root form; below, [`ASINH_MIDDLE`]'s.
pub(crate) const ASINH_ROOT: u32 = 0x3FC0_0000;

/// From this magnitude `asinh` is `log(a) + log(2)` of its `log2`.
pub(crate) const ASINH_LARGE: f32 = 500.0;

/// From this operand (bits, 1750.2775) `acosh` is `log(x) + log(2)` of its `log2`.
pub(crate) const ACOSH_LARGE: u32 = 0x44DA_C8E0;

/// `acosh`'s `w` from this (bits) below [`ACOSH_LOG1P_END`] takes [`ACOSH_LOG1P`].
pub(crate) const ACOSH_LOG1P_START: u32 = 0x3F81_F7F1;
pub(crate) const ACOSH_LOG1P_END: u32 = 0x3FA3_FC0B;

/// To this magnitude (bits) Metal's `atanh` is its operand.
pub(crate) const ATANH_TINY: u32 = 0x3A40_0000;

/// Above this magnitude Metal's `atanh` is its form of its `log2` and reciprocal.
pub(crate) const ATANH_FORM: f32 = 0.25;

/// Which hyperbolic function a routine answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hyper {
    Sinh,
    Cosh,
    Tanh,
}

/// Which inverse hyperbolic function a routine answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arc {
    Asinh,
    Acosh,
    Atanh,
}

/// Which of Metal's fast inverse hyperbolic functions [`fast_arc`] answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FastArc {
    Asinh,
    Acosh,
}

/// `1 + s q(s)` of [`SINH_Q`], [`COSH_Q`], or [`TANH_Q`].
fn near_zero(e: &mut E<'_, '_>, q: &[u32], s: Value) -> BResult<Value> {
    let acc = e.fused_poly(q, s)?;
    let one = e.c(1.0)?;
    e.fused(acc, s, one)
}

/// Whether the magnitude's bits `ab` are below `bits` (`ab` of a NaN is above every bound).
fn below(e: &mut E<'_, '_>, ab: Value, bits: u32) -> BResult<Value> {
    let b = e.u(bits)?;
    e.iop(BinOp::ULessThan, ab, b)
}

/// Metal's precise `which` of `x` (`tanh`'s middle range):
/// - `sinh` and `cosh`: the operand (`sinh`) or 1 (`cosh`) to [`SINH_TINY`] / [`COSH_TINY`],
///   their polynomials below 1.5, `(E - 1/E) / 2` and `(E + 1/E) / 2` of `E = e^a` to `12 ln 2`,
///   and past it `e^a / 2` ([`exp::exp_half`], so `2 h` is [`exp::exp_base`]'s `E`);
/// - `tanh`: the operand to [`TANH_TINY`], its polynomial below 0.35, [`TANH_MIDDLE`] to 1.25,
///   `(t - 1) (1 / (t + 1))` of `t = e^(2a)` below `12 ln 2`, and 1 from it.
///
/// Each `1 / E` and `1 / (t + 1)` is Metal's reciprocal ([`E::recip`]). With Metal's own reciprocal
/// these forms are Metal's word at every operand of `[1.5, 8)` (`sinh`, `cosh`) and `(1.25, 12 ln
/// 2)` (`tanh`); with the model's they miss 520, 512, and 83,177 of those 20,971,520, 20,971,520,
/// and 23,401,873 operands (an open near-equivalent), where the correctly rounded reciprocal missed
/// 18,099, 17,989, and 2,390,824.
pub(crate) fn hyper(e: &mut E<'_, '_>, which: Hyper, x: Value) -> BResult<Value> {
    let a = e.abs(x)?;
    let ab = e.bits(a)?;
    let s = e.mul(a, a)?;
    let one = e.c(1.0)?;
    let r = match which {
        Hyper::Sinh | Hyper::Cosh => {
            let sinh = which == Hyper::Sinh;
            let h = exp::exp_half(e, a)?;
            let big = e.add(h, h)?;
            let inv = e.recip(big)?;
            let mid = if sinh {
                e.sub(big, inv)?
            } else {
                e.add(big, inv)?
            };
            let half = e.c(0.5)?;
            let mid = e.mul(half, mid)?;
            let far = below(e, ab, HALF_EXP + 1)?;
            let r = e.sel(far, mid, h)?;
            let p = near_zero(e, if sinh { &SINH_Q } else { &COSH_Q }, s)?;
            let (near, tiny, tiny_bits) = if sinh {
                (e.mul(a, p)?, a, SINH_TINY)
            } else {
                (p, one, COSH_TINY)
            };
            let poly = below(e, ab, POLY_END)?;
            let r = e.sel(poly, near, r)?;
            let small = below(e, ab, tiny_bits + 1)?;
            e.sel(small, tiny, r)?
        }
        Hyper::Tanh => {
            let top = e.c(f32::from_bits(HALF_EXP))?;
            let over = e.gt(a, top)?;
            let ac = e.sel(over, top, a)?;
            let a2 = e.add(ac, ac)?;
            let t = exp::exp_base(e, a2, false)?;
            let tm = e.sub(t, one)?;
            let tp = e.add(t, one)?;
            let inv = e.recip(tp)?;
            let quot = e.mul(tm, inv)?;
            let ones = below(e, ab, HALF_EXP)?;
            let r = e.sel(ones, quot, one)?;
            let middle = e.fused_poly(&TANH_MIDDLE, a)?;
            let quotient = below(e, ab, TANH_QUOTIENT + 1)?;
            let r = e.sel(quotient, middle, r)?;
            let p = near_zero(e, &TANH_Q, s)?;
            let near = e.mul(a, p)?;
            let poly = below(e, ab, TANH_POLY_END)?;
            let r = e.sel(poly, near, r)?;
            let small = below(e, ab, TANH_TINY + 1)?;
            e.sel(small, a, r)?
        }
    };
    let r = if which == Hyper::Cosh {
        r
    } else {
        e.copysign(r, x)?
    };
    let nan = e.is_nan(x)?;
    e.sel(nan, x, r)
}

/// `RN(RN(log2(v)) RN(ln 2))`, and past the switch `large` `RN(that + RN(ln 2))` of `far`
/// instead: one `log2`, of the selected argument.
fn ln_or_twice(e: &mut E<'_, '_>, large: Value, v: Value, far: Value) -> BResult<(Value, Value)> {
    let arg = e.sel(large, far, v)?;
    let l2 = log::metal_log2(e, arg)?;
    let ln2 = e.c(f32::from_bits(log::LN_2_BITS))?;
    let m = e.mul(l2, ln2)?;
    let twice = e.add(m, ln2)?;
    Ok((m, twice))
}

/// Metal's precise `which` of `x`: see the module's list.
pub(crate) fn arc(e: &mut E<'_, '_>, which: Arc, x: Value) -> BResult<Value> {
    let one = e.c(1.0)?;
    let zero = e.c(0.0)?;
    let a = e.abs(x)?;
    let ab = e.bits(a)?;
    match which {
        Arc::Asinh => {
            let z = e.fused(a, a, one)?;
            let rz = e.fast_rsqrt(z)?;
            let q = e.mul(z, rz)?;
            let w = e.add(a, q)?;
            let top = e.c(ASINH_LARGE)?;
            let large = e.ge(a, top)?;
            let (m, twice) = ln_or_twice(e, large, w, a)?;
            let r = e.sel(large, twice, m)?;
            let middle = e.fused_poly(&ASINH_MIDDLE, a)?;
            let in_middle = below(e, ab, ASINH_ROOT)?;
            let r = e.sel(in_middle, middle, r)?;
            let s = e.mul(a, a)?;
            let p = e.fused_poly(&ASINH_NEAR, s)?;
            let near = e.mul(p, a)?;
            let is_near = below(e, ab, ASINH_NEAR_END)?;
            let r = e.sel(is_near, near, r)?;
            let tiny = below(e, ab, ASINH_TINY)?;
            let r = e.sel(tiny, a, r)?;
            let r = e.copysign(r, x)?;
            let nan = e.is_nan(x)?;
            e.sel(nan, x, r)
        }
        Arc::Acosh => {
            // `q = z rsqrt(z)` is `sqrt(z)`; at `x = 1` (`z = 0`) it is 0.
            let down = e.sub(x, one)?;
            let up = e.add(x, one)?;
            let z = e.mul(down, up)?;
            let rz = e.fast_rsqrt(z)?;
            let q = e.mul(z, rz)?;
            let at_one = e.fb.binary(BinOp::FOrdEqual, z, zero)?;
            let q = e.sel(at_one, zero, q)?;
            let w = e.add(q, x)?;
            let back = e.sub(w, x)?;
            let err = e.sub(q, back)?;
            let top = e.c(f32::from_bits(ACOSH_LARGE))?;
            let large = e.ge(x, top)?;
            let (m, twice) = ln_or_twice(e, large, w, x)?;
            let t = e.sub(w, one)?;
            let p = e.fused_poly(&ACOSH_LOG1P, t)?;
            let near = e.mul(t, p)?;
            let wb = e.bits(w)?;
            let start = e.u(ACOSH_LOG1P_START)?;
            let offset = e.iop(BinOp::ISub, wb, start)?;
            let span = e.u(ACOSH_LOG1P_END - ACOSH_LOG1P_START)?;
            let is_near = e.iop(BinOp::ULessThan, offset, span)?;
            let l = e.sel(is_near, near, m)?;
            let rw = e.recip(w)?;
            let tail = e.mul(err, rw)?;
            let root = e.add(l, tail)?;
            let r = e.sel(large, twice, root)?;
            let below_one = e.lt(x, one)?;
            let nan = e.metal_nan(None)?;
            let r = e.sel(below_one, nan, r)?;
            let x_nan = e.is_nan(x)?;
            e.sel(x_nan, x, r)
        }
        Arc::Atanh => {
            let half = e.c(0.5)?;
            let up = e.add(one, a)?;
            let down = e.sub(one, a)?;
            let rd = e.recip(down)?;
            let q = e.mul(up, rd)?;
            let l2 = log::metal_log2(e, q)?;
            let ln2 = e.c(f32::from_bits(log::LN_2_BITS))?;
            let m = e.mul(l2, ln2)?;
            let m = e.mul(m, half)?;
            let near = e.fused_poly(&ATANH_NEAR, a)?;
            let quarter = e.c(ATANH_FORM)?;
            let formed = e.gt(a, quarter)?;
            let r = e.sel(formed, m, near)?;
            let inf = e.c(f32::INFINITY)?;
            let at_one = e.fb.binary(BinOp::FOrdEqual, a, one)?;
            let r = e.sel(at_one, inf, r)?;
            let tiny = below(e, ab, ATANH_TINY + 1)?;
            let r = e.sel(tiny, a, r)?;
            let r = e.copysign(r, x)?;
            let beyond = e.gt(a, one)?;
            let nan = e.metal_nan(None)?;
            let r = e.sel(beyond, nan, r)?;
            let x_nan = e.is_nan(x)?;
            e.sel(x_nan, x, r)
        }
    }
}

/// Metal's fast `asinh` or `acosh` (`which`) of `x`: MSL Table 8.2's `log(x + sqrt(s))` of `s =
/// RN(RN(x x) ± 1)` in Metal's own form, `RN(RN(log2(v)) RN(ln 2))` of `v = fma(s, rsqrt(s), x)`:
/// its hardware's inverse root ([`E::fast_rsqrt`]) times `s` is the root, and the product is fused
/// into the sum, so where `x` is negative and large the sum keeps the root's error
/// (`fast::asinh(-6.84e13)` is 14.39, and NaN where the error is below zero). Where `s` is 0 or
/// `+inf` the root is `s` itself: `v = RN(x + s)`, so `asinh(-inf)` is NaN (`-inf + inf`) and
/// `asinh(-FLT_MAX)` and `acosh(-FLT_MAX)` are `+inf`. Below zero (`acosh` inside `(-1, 1)`) the
/// answer is Metal's NaN. On case f6524207's 1,024 lanes this is Metal's word on the M3 and the M4
/// wherever its `log2`'s model ([`log::metal_log2`]) is (`v` outside `[0.996, 1.031)`) and its
/// inverse root's is: on every lane but 42 and 10 of `v` next to 1 and a single lane of the root
/// (host replay).
pub(crate) fn fast_arc(e: &mut E<'_, '_>, which: FastArc, x: Value) -> BResult<Value> {
    let one = e.c(1.0)?;
    let zero = e.c(0.0)?;
    let inf = e.c(f32::INFINITY)?;
    let sq = e.mul(x, x)?;
    let s = match which {
        FastArc::Asinh => e.add(sq, one)?,
        FastArc::Acosh => e.sub(sq, one)?,
    };
    let rs = e.fast_rsqrt(s)?;
    // The product `s rsqrt(s)` as `(s 2^-64) (rsqrt(s) 2^64)`, the same product exactly, whose
    // factors the emulated fused step splits without overflow (`s` reaches `2^128`).
    let (down, up) = (
        e.c(f32::from_bits(0x1F80_0000))?,
        e.c(f32::from_bits(0x5F80_0000))?,
    );
    let s_down = e.mul(s, down)?;
    let rs_up = e.mul(rs, up)?;
    let v = e.fused(s_down, rs_up, x)?;
    let at_zero = e.fb.binary(BinOp::FOrdEqual, s, zero)?;
    let at_inf = e.fb.binary(BinOp::FOrdEqual, s, inf)?;
    let ends = e.iop(BinOp::LogicalOr, at_zero, at_inf)?;
    let plain = e.add(x, s)?;
    let v = e.sel(ends, plain, v)?;
    let l2 = log::metal_log2(e, v)?;
    let ln2 = e.c(f32::from_bits(log::LN_2_BITS))?;
    let r = e.mul(l2, ln2)?;
    // Metal's inverse root of a negative operand is its magnitude's ([`E::fast_rsqrt`]).
    let negative = e.lt(s, zero)?;
    let nan = e.metal_nan(None)?;
    e.sel(negative, nan, r)
}

/// The `float` bits below which Metal's `half` `asinh(x)` is `x` (`2^-11`).
const HALF_ASINH_TINY: u32 = 0x3A00_0000;

/// Metal's `half` `asinh` before its rounding to `half`, of a `half` `x` held as `float`: its own
/// form, which follows Apple's runtime library: `x` itself below `2^-11` in magnitude and at an
/// infinity, else `RN(RN(log2(v)) RN(ln 2))` with `x`'s sign of `v = fma(z, rsqrt(z), |x|)`, `z =
/// fma(x, x, 1)` and its hardware's inverse root ([`E::fast_rsqrt`]), the fast `asinh`'s sum of
/// `|x|` ([`fast_arc`]).
pub(crate) fn half_asinh(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let one = e.c(1.0)?;
    let z = e.fused(x, x, one)?;
    let rz = e.fast_rsqrt(z)?;
    let a = e.abs(x)?;
    let v = e.fused(z, rz, a)?;
    let l2 = log::metal_log2(e, v)?;
    let ln2 = e.c(f32::from_bits(log::LN_2_BITS))?;
    let l = e.mul(l2, ln2)?;
    let minus_l = e.fb.unary(UnOp::FNegate, l)?;
    let zero = e.c(0.0)?;
    let negative = e.lt(x, zero)?;
    let r = e.sel(negative, minus_l, l)?;
    let ab = e.bits(a)?;
    let tiny = below(e, ab, HALF_ASINH_TINY)?;
    let inf = e.u(0x7F80_0000)?;
    let at_inf = e.iop(BinOp::IEqual, ab, inf)?;
    let itself = e.iop(BinOp::LogicalOr, tiny, at_inf)?;
    e.sel(itself, x, r)
}
