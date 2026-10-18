//! The logarithms: Metal's `log`, `log2`, and `log10` of a `float`, which MSL Table 8.1 bounds at 4
//! ulp, Metal's precise `powr` (16 ulp), and the core the inverse hyperbolic functions share.
//!
//! **The core** ([`ln_parts`]): a positive normal `x` is `2^e (1 + f)` with `1 + f` in
//! `[sqrt(1/2), sqrt(2)]` ([`reduce`]; `f = m - 1` is exact), and `log(1 + f) = 2 atanh(s)` with
//! `s = f / (2 + f)`, `|s| <= 0.1716`. `s` is the device's quotient and the low part from its
//! exact remainder ([`E::div_pair`]); `2s` is exact, `(2/3) s^3` is formed from exact products
//! (Dekker's, with `s`'s low part to the first order), and the rest `s^5 R(s^2)` in plain `float`,
//! `R` of degree 3 fitted to a relative error of `2e-8`. `e ln 2` is exact in its high part (`ln 2`
//! to 16 bits) and the terms are summed by exact two-term sums from the largest, so the pair is
//! accurate to about `2^-40` relative: enough for the exact value [`metal_log2`] rounds.
//!
//! **Metal's `log2`** ([`metal_log2`]): what Metal's hardware computes, read off every `float` of
//! the M4 (the M3 answers every one alike). It is the exponent plus `L(m)` of the significand,
//! rounded toward zero, and `L(m) - log2(m)` is a smooth error per 32nd of `m` (about `2^-26` at
//! most) under a noise of about `2^-31` the model does not reproduce: [`TABLE`] fits the smooth
//! part, so the routine is `RZ(log2(x) + eps(m))`, and next to 1 (`[1 - 2^-8, 1 + 2^-5)`), where
//! the hardware keeps a relative accuracy the model lacks, `RZ(log2(x))`. Metal's word at
//! 2,130,133,076 of the 2,130,706,432 positive normal operands (99.973 %); a word one unit off at
//! the others (an open near-equivalent), and within 3 ulp of the exact value everywhere, as
//! Metal's. Metal's fast `log2` is the same function, so `air.fast_log2`
//! and the fast functions' formulas take the same routine (before SEMANTICS 1262 they took the core
//! rounded toward zero without the table, Metal's word on 99.76 % of those operands).
//!
//! **Metal's precise `log` and `log10`** ([`log`]): its `log2` times `RN(ln 2)` or
//! `RN(log10(2))` (every `float` of the sweep's binades `2^-40` to `2^40`), and next to 1 its own
//! polynomials ([`LOG_NEAR`], [`LOG10_NEAR`]), Metal's word at every `float` of their ranges.
//!
//! **Metal's precise `powr`** ([`powr`]), its own form, which follows Apple's runtime library and
//! matches Metal's measured words: `log2(x)` as a pair of `float`s through a table of nine centres
//! of the significand ([`POWR_TABLE`]) and a polynomial of degree 5 in fused steps ([`POWR_P`]),
//! its product with `y` as a pair, and Metal's hardware `exp2` of the high part (`exp::exp2`, its
//! model) grown by the low part in one fused step (`exp::grown`). With Metal's measured `exp2` it
//! is Metal's word at every measured operand; through the model, wherever the model is Metal's
//! `exp2` (99.905 % of its fractions).
//!
//! **Special operands** (case 7b283404, every lane, both math modes): a zero or a denormal (read
//! as a zero) gives `-inf`, a negative number Metal's NaN (`0x7fc00000`), `+inf` `+inf`, and a
//! NaN a NaN. `powr`'s magnitude of a zero or denormal base is `+0` toward a positive exponent and
//! `+inf` toward a negative one, of an infinite base the reverse; its NaN cases and its zero
//! exponent are `math::powr_nan`'s and `math::pow_special`'s around it.

use crate::lower::arith::libm::exp;
use crate::lower::arith::libm::pair::constant;
use crate::lower::arith::libm::E;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, MemAccess};
use std::f64::consts::{LN_2, LOG2_E};

/// `R(z)` of `log((1 + s) / (1 - s)) = 2s + (2/3) s^3 + s^5 R(s^2)` on `[0, 0.0295]`.
const R: [f32; 4] = [0.4, 0.285_715_3, 0.222_046_78, 0.191_227_05];

/// The bits of the largest `float` below `sqrt(2)`'s rounding: a significand above it is halved.
const SQRT2_BITS: u32 = 0x3FB5_04F3;

/// Metal's `log2` table ([`metal_log2`]): per 32nd of the significand `m` of `x`, the `f32`
/// coefficients (by their bits, lowest first) of the quintic in `t` (the offset in the 32nd, in
/// `[0, 1)`) that models the hardware's error `L(m) - log2(m)`, where Metal's `log2(x)` is `L(m)`
/// plus the exponent rounded toward zero. Fitted by least squares to the bounds Metal's words set
/// on `L(m)` at every significand (the binades `[1/2, 1)`, `[1, 2)`, `[2, 4)`;
/// the words of `[1, 1 + 1/32)` and `[1 - 1/64, 1)` left out, where the hardware keeps the relative
/// accuracy no such model gives). The first segment's constant is 0, so the powers of 2 are exact.
pub(crate) const TABLE: [[u32; 6]; 32] = [
    [
        0x0000_0000,
        0xB3E1_5503,
        0x350D_60F3,
        0xB55F_52F0,
        0x34F6_A16A,
        0xB33B_CEFE,
    ],
    [
        0x3200_A205,
        0xB403_237D,
        0x34F5_6DB3,
        0xB52B_3158,
        0x3493_6140,
        0x3259_3010,
    ],
    [
        0x31EF_171D,
        0xB3E6_F706,
        0x34C2_0549,
        0xB4E8_A17E,
        0x33EF_7B7F,
        0x3382_A9BC,
    ],
    [
        0xAF8C_3B2F,
        0xB399_4960,
        0x3497_04EF,
        0xB4B8_39BA,
        0x3368_BD46,
        0x339C_C268,
    ],
    [
        0xB1D0_EEE3,
        0xB328_8A32,
        0x3476_6816,
        0xB4B4_59C9,
        0x33FE_9C95,
        0x330A_D5FC,
    ],
    [
        0x3278_7D57,
        0xB3F8_71CA,
        0x34B1_0C55,
        0xB4CA_ED27,
        0x33FD_87E4,
        0x32D8_6AFC,
    ],
    [
        0x321C_AC81,
        0xB3CA_8A2B,
        0x34AB_D3B9,
        0xB4CD_22B1,
        0x3411_93F7,
        0x31E8_3355,
    ],
    [
        0x3235_26B5,
        0xB3CF_59CD,
        0x349B_9225,
        0xB4B9_37DF,
        0x3404_1A18,
        0x3259_FCED,
    ],
    [
        0x3226_177F,
        0xB3F2_3D17,
        0x349B_8A38,
        0xB486_5353,
        0x31D1_C99F,
        0x3358_3A58,
    ],
    [
        0x3208_AAA4,
        0xB391_BAB0,
        0x346E_AA13,
        0xB487_A4DD,
        0x338D_D500,
        0x32B6_7DDF,
    ],
    [
        0x31A4_DA68,
        0xB3B7_48F3,
        0x3485_0421,
        0xB48E_1F2E,
        0x3397_D850,
        0x329D_4FA1,
    ],
    [
        0xB092_988F,
        0xB2DC_B39D,
        0x340A_3A65,
        0xB43F_2DB8,
        0x3398_36BF,
        0x3236_3EAF,
    ],
    [
        0xB21E_AB87,
        0x3193_9904,
        0x338C_A47A,
        0xB40F_AA7F,
        0x336C_6EFB,
        0x3286_AE63,
    ],
    [
        0xB12C_6079,
        0xB239_CB47,
        0x338C_391B,
        0xB3C1_E86A,
        0x328C_A645,
        0x32DE_FB2D,
    ],
    [
        0xB160_347D,
        0xB288_D371,
        0x33EB_A160,
        0xB424_44BD,
        0x336B_1948,
        0x3210_ECC6,
    ],
    [
        0x320F_FEDF,
        0xB3AD_882C,
        0x3451_91B6,
        0xB42B_4D5E,
        0x3183_7957,
        0x32D7_0DBE,
    ],
    [
        0x31F9_8607,
        0xB380_183C,
        0x3429_3C97,
        0xB41C_9839,
        0x32CC_416E,
        0x3296_3EAF,
    ],
    [
        0x320F_8CE4,
        0xB348_3D19,
        0x340D_438A,
        0xB438_B378,
        0x339B_4CEE,
        0xB12B_21E9,
    ],
    [
        0x3268_3BD7,
        0xB38F_0693,
        0x340A_8AE4,
        0xB3E9_2BA0,
        0x320F_2AA2,
        0x3296_CA96,
    ],
    [
        0x320F_AD9D,
        0xB39C_87A8,
        0x3439_619A,
        0xB427_B4DF,
        0x3317_6254,
        0x31E8_2E9C,
    ],
    [
        0x30C7_D330,
        0xB208_9D73,
        0x3343_37EE,
        0xB3BA_E63E,
        0x3357_8954,
        0xB063_F2D5,
    ],
    [
        0x30BE_1221,
        0xB327_50E3,
        0x33D9_6E04,
        0xB3D3_3908,
        0x321C_F270,
        0x325F_24B9,
    ],
    [
        0x312C_4836,
        0xB28F_336A,
        0x335C_CDB0,
        0xB3A3_99F0,
        0x3329_5877,
        0x30BC_7099,
    ],
    [
        0x3223_44FE,
        0xB392_EA61,
        0x3432_EF2E,
        0xB41F_31D3,
        0x3314_E9F5,
        0x30D3_0DEE,
    ],
    [
        0x3237_EDC2,
        0xB386_7642,
        0x3404_7D32,
        0xB3D6_EBC3,
        0x3291_7F87,
        0x31F1_04E6,
    ],
    [
        0x3145_916D,
        0xB2E8_D3E6,
        0x33A3_FCC8,
        0xB393_2998,
        0x31BE_CD6E,
        0x322E_FDB5,
    ],
    [
        0xB00E_B02A,
        0xB2BA_5507,
        0x3389_FD0C,
        0xB3A6_D06B,
        0x32ED_F541,
        0x303A_561D,
    ],
    [
        0x2F9E_3183,
        0xB2D9_4D80,
        0x3385_6DD9,
        0xB360_917B,
        0xB053_D071,
        0x324D_9D1D,
    ],
    [
        0xB186_950C,
        0xB28A_4992,
        0x338E_BB9D,
        0xB376_2A50,
        0x2E9F_DE63,
        0x3231_6D8E,
    ],
    [
        0x31F1_16B8,
        0xB307_A725,
        0x3399_2D14,
        0xB35A_6910,
        0xB2CF_20D7,
        0x32AC_B86E,
    ],
    [
        0x31F9_B76F,
        0xB3BC_C5AB,
        0x34A9_01D6,
        0xB53A_96D3,
        0x3558_BAB8,
        0xB4BC_8B62,
    ],
    [
        0xB186_9558,
        0x32D5_CEF8,
        0xB397_760F,
        0x341E_A898,
        0xB446_D126,
        0x33BF_EEEA,
    ],
];

/// The coefficients per segment of [`TABLE`].
const COEFFICIENTS: u32 = 6;

/// An upper estimate of the SPIR-V words of [`TABLE`] in a module: its 192 constants, the
/// composite, and the array, pointer, and variable.
pub(crate) const TABLE_WORDS: u32 = 192 * 4 + 195 + 24;

/// Declare [`TABLE`] in the module: a `Private` `u32[192]` of the coefficients' bits, segment
/// `s`'s six from word `6 s`.
pub(crate) fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    super::declare_words(mb, TABLE.as_flattened())
}

/// From `PLAIN_LOW` to below `PLAIN_END` (`[1 - 2^-8, 1 + 2^-5)`, bits) Metal's `log2` is taken
/// as the exact value rounded toward zero: [`TABLE`]'s model is not there.
pub(crate) const PLAIN_LOW: u32 = 0x3F7F_0000;
pub(crate) const PLAIN_END: u32 = 0x3F84_0000;

/// `RN(ln 2)` and `RN(log10(2))`, by their bits: Metal's precise `log` and `log10` away from 1
/// are its `log2` times them.
pub(crate) const LN_2_BITS: u32 = 0x3F31_7218;
pub(crate) const LOG10_2_BITS: u32 = 0x3E9A_209B;

/// Metal's precise `log` or `log10` next to 1: for `x` strictly between the
/// bits `low` and `end`, `RN(f fma(p(f), f, lead))` of `f = x - 1` (exact there), `p` by Horner's
/// rule with fused steps, `f32` coefficients by their bits, lowest first. Each is Metal's word at
/// every `float` of its range.
pub(crate) struct Near {
    pub low: u32,
    pub end: u32,
    pub lead: u32,
    pub p: &'static [u32],
}

/// `log`'s: `(61/64, 17/16)`, `lead` 1.
pub(crate) const LOG_NEAR: Near = Near {
    low: 0x3F74_0000,
    end: 0x3F88_0000,
    lead: 0x3F80_0000,
    p: &[0xBEFF_FFF4, 0x3EAA_AC4E, 0xBE80_449E, 0x3E47_04F8],
};

/// `log10`'s: `(495/512, 543/512)`, `lead` `log10(e)` rounded down.
pub(crate) const LOG10_NEAR: Near = Near {
    low: 0x3F77_8000,
    end: 0x3F87_C000,
    lead: 0x3EDE_5BD8,
    p: &[0xBE5E_5CE8, 0x3E14_5662, 0xBDD5_5B94],
};

/// Which logarithm a routine answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Base {
    E,
    Ten,
}

/// `c` to 16 significant bits and the `float` nearest the rest, so that `e c_hi` is exact for an
/// integer `e` of at most 8 bits.
fn short(c: f64) -> (f32, f32) {
    let hi = (c * 65536.0).floor() / 65536.0;
    (hi as f32, (c - hi) as f32)
}

/// `x = 2^e (1 + f)` for a positive normal `x`: `e` (a `float`) and `f` in
/// `[sqrt(1/2) - 1, sqrt(2) - 1]`, both exact.
pub(crate) fn reduce(e: &mut E<'_, '_>, x: Value) -> BResult<(Value, Value)> {
    let (ef, m) = significand(e, x)?;
    let onef = e.c(1.0)?;
    let f = e.sub(m, onef)?;
    Ok((ef, f))
}

/// `x = 2^e m` for a positive normal `x`: `e` (a `float`) and `m` in `(sqrt(1/2), sqrt(2)]`
/// (bits up to [`SQRT2_BITS`]), both exact.
fn significand(e: &mut E<'_, '_>, x: Value) -> BResult<(Value, Value)> {
    let b = e.bits(x)?;
    let k23 = e.u(23)?;
    let biased = e.iop(BinOp::ShiftRightLogical, b, k23)?;
    let (fraction, one_bits) = (e.u(0x007F_FFFF)?, e.u(0x3F80_0000)?);
    let m = e.iop(BinOp::BitwiseAnd, b, fraction)?;
    let m = e.iop(BinOp::BitwiseOr, m, one_bits)?;
    let bound = e.u(SQRT2_BITS)?;
    let big = e.iop(BinOp::UGreaterThan, m, bound)?;
    let lead = e.u(0x0080_0000)?;
    let halved = e.iop(BinOp::ISub, m, lead)?;
    let m = e.sel(big, halved, m)?;
    let one = e.u(1)?;
    let up = e.iop(BinOp::IAdd, biased, one)?;
    let biased = e.sel(big, up, biased)?;
    let bf = e.ufloat(biased)?;
    let bias = e.c(127.0)?;
    let ef = e.sub(bf, bias)?;
    let m = e.float(m)?;
    Ok((ef, m))
}

/// The core: `e ln 2 + log(1 + fh + fl)` as a pair, for an integral `e` of at most 8 bits, `fh`
/// in `[sqrt(1/2) - 1, sqrt(2) - 1]` (zero or at least `2^-25` in magnitude), and `fl` a low
/// part below an ulp of `fh`.
pub(crate) fn ln_parts(
    e: &mut E<'_, '_>,
    ef: Value,
    fh: Value,
    fl: Value,
) -> BResult<(Value, Value)> {
    let two = e.c(2.0)?;
    let (dh, dl) = e.fast_two_sum(two, fh)?;
    let dl = e.add(dl, fl)?;
    let (sh, sl) = e.div_pair((fh, fl), (dh, dl))?;
    // `s^2` and `s^3` exactly, then `(2/3) s^3 + 2 s^2 sl`.
    let (q, q_err) = e.two_prod(sh, sh)?;
    let (c, c_err) = e.two_prod(q, sh)?;
    let qe = e.mul(q_err, sh)?;
    let c_err = e.add(c_err, qe)?;
    let (tt_hi, tt_lo) = constant(2.0 / 3.0);
    let tt = e.c(tt_hi)?;
    let (t, t_err) = e.two_prod(c, tt)?;
    let ttl = e.c(tt_lo)?;
    let t1 = e.mul(c, ttl)?;
    let t2 = e.mul(c_err, tt)?;
    let t12 = e.add(t1, t2)?;
    let t_err = e.add(t_err, t12)?;
    let qs = e.mul(q, sl)?;
    let qs = e.add(qs, qs)?;
    let t_err = e.add(t_err, qs)?;
    // `s^5 R(s^2)`.
    let r = e.horner(&R, q)?;
    let s5 = e.mul(c, q)?;
    let s5 = e.mul(s5, r)?;
    // `e ln 2`.
    let (l2_hi, l2_lo) = short(LN_2);
    let l2h = e.c(l2_hi)?;
    let l2l = e.c(l2_lo)?;
    let eh = e.mul(ef, l2h)?;
    let el = e.mul(ef, l2l)?;
    // `eh + 2 sh + t + (2 sl + t_err + s5 + el)`, from the largest (`|eh| >= 0.69 > |2 sh|` or
    // `eh = 0`).
    let s2 = e.add(sh, sh)?;
    let (a, a_err) = e.fast_two_sum(eh, s2)?;
    let (b, b_err) = e.two_sum(a, t)?;
    let ab = e.add(a_err, b_err)?;
    let sl2 = e.add(sl, sl)?;
    let rest = e.add(sl2, t_err)?;
    let tail = e.add(s5, el)?;
    let rest = e.add(rest, tail)?;
    let rest = e.add(ab, rest)?;
    e.fast_two_sum(b, rest)
}

/// `x` where it is a positive normal `float`, `1` elsewhere: the core's operand on every lane,
/// the special operands' answers selected after it.
pub(crate) fn positive_normal(e: &mut E<'_, '_>, x: Value) -> BResult<(Value, Value)> {
    let b = e.bits(x)?;
    let lead = e.u(0x0080_0000)?;
    let offset = e.iop(BinOp::ISub, b, lead)?;
    let span = e.u(0x7F00_0000)?;
    let normal = e.iop(BinOp::ULessThan, offset, span)?;
    let one = e.c(1.0)?;
    let xs = e.sel(normal, x, one)?;
    Ok((xs, normal))
}

/// `(h + l) c` as a pair for a split constant `c`, `h + l` a pair.
fn times(e: &mut E<'_, '_>, (h, l): (Value, Value), c: f64) -> BResult<(Value, Value)> {
    let (c_hi, c_lo) = constant(c);
    let ch = e.c(c_hi)?;
    let (p, p_err) = e.two_prod(h, ch)?;
    let cl = e.c(c_lo)?;
    let a = e.mul(h, cl)?;
    let b = e.mul(l, ch)?;
    let ab = e.add(a, b)?;
    let rest = e.add(p_err, ab)?;
    e.fast_two_sum(p, rest)
}

/// Metal's `log2` of `x`, in both modes (its fast `log2` is its precise one): `RZ(log2(x) +
/// eps(m))`, rounded toward zero, `eps` the hardware's error at the significand `m` of `x`
/// ([`TABLE`]), and `0` from [`PLAIN_LOW`] to below [`PLAIN_END`]. `log2(x)` is `ef + log2(1 + f)`
/// of [`reduce`]'s parts, the logarithm a pair (exact at the powers of 2, accurate to about
/// `2^-41`), so the sum with `ef` is exact and the one rounding is the last: where the exact
/// remainder's sign is not the sum's, the sum steps one unit toward zero. Metal's special operands
/// are [`specials`]'.
pub(crate) fn metal_log2(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let (xs, _) = positive_normal(e, x)?;
    let (ef, f) = reduce(e, xs)?;
    let zero = e.c(0.0)?;
    let ln = ln_parts(e, zero, f, zero)?;
    let (h, l) = times(e, ln, LOG2_E)?;
    let (s, r) = e.two_sum(ef, h)?;
    let eps = fitted_error(e, xs)?;
    let tail = e.add(l, eps)?;
    let tail = e.add(r, tail)?;
    let (s, r) = e.two_sum(s, tail)?;
    // Toward zero: one unit down in magnitude where a nonzero remainder has the other sign.
    let (sb, rb) = (e.bits(s)?, e.bits(r)?);
    let (sign, magnitude, zero_u, one) = (e.u(0x8000_0000)?, e.u(0x7FFF_FFFF)?, e.u(0)?, e.u(1)?);
    let differ = e.iop(BinOp::BitwiseXor, sb, rb)?;
    let differ = e.iop(BinOp::BitwiseAnd, differ, sign)?;
    let opposite = e.iop(BinOp::INotEqual, differ, zero_u)?;
    let rm = e.iop(BinOp::BitwiseAnd, rb, magnitude)?;
    let nonzero = e.iop(BinOp::INotEqual, rm, zero_u)?;
    let step = e.iop(BinOp::LogicalAnd, opposite, nonzero)?;
    let down = e.iop(BinOp::ISub, sb, one)?;
    let v = e.sel(step, down, sb)?;
    let v = e.float(v)?;
    specials(e, x, v)
}

/// [`TABLE`]'s `eps(m)` at the positive normal `xs`: segment `k >> 18` of its fraction `k`, at
/// `t = (k mod 2^18) 2^-18`, and `0` from [`PLAIN_LOW`] to below [`PLAIN_END`].
fn fitted_error(e: &mut E<'_, '_>, xs: Value) -> BResult<Value> {
    let table = e.log2_table()?;
    let b = e.bits(xs)?;
    let (fraction, low, k18) = (e.u(0x007F_FFFF)?, e.u(0x3_FFFF)?, e.u(18)?);
    let k = e.iop(BinOp::BitwiseAnd, b, fraction)?;
    let segment = e.iop(BinOp::ShiftRightLogical, k, k18)?;
    let t = e.iop(BinOp::BitwiseAnd, k, low)?;
    let t = e.ufloat(t)?;
    let scale = e.c(1.0 / 262_144.0)?;
    let t = e.mul(t, scale)?;
    let width = e.u(COEFFICIENTS)?;
    let first = e.iop(BinOp::IMul, segment, width)?;
    let base = e.fb.module().global_value(table)?;
    let coefficient = |e: &mut E<'_, '_>, j: u32| -> BResult<Value> {
        let j = e.u(j)?;
        let i = e.iop(BinOp::IAdd, first, j)?;
        let p = e.fb.access_chain(base, &[i])?;
        let w = e.fb.load(p, MemAccess::default())?;
        e.float(w)
    };
    // Horner's rule in plain `float`: `eps` is below `2^-25`, so its own rounding errors (about
    // `2^-48`) move the result only where the sum lies that close to a `float`.
    let mut eps = coefficient(e, COEFFICIENTS - 1)?;
    for j in (0..COEFFICIENTS - 1).rev() {
        let c = coefficient(e, j)?;
        let p = e.mul(eps, t)?;
        eps = e.add(p, c)?;
    }
    let plain_low = e.u(PLAIN_LOW)?;
    let offset = e.iop(BinOp::ISub, b, plain_low)?;
    let span = e.u(PLAIN_END - PLAIN_LOW)?;
    let plain = e.iop(BinOp::ULessThan, offset, span)?;
    let zero = e.c(0.0)?;
    e.sel(plain, zero, eps)
}

/// `RN(f q)` of `f = x - 1` and `q = fma(p(f), f, lead)`, `p` by Horner's rule with fused steps
/// ([`Near`]); exact where `x` lies in the near form's range.
fn near_one(e: &mut E<'_, '_>, x: Value, near: &Near) -> BResult<Value> {
    let one = e.c(1.0)?;
    let f = e.sub(x, one)?;
    let (last, rest) = near
        .p
        .split_last()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
    let mut p = e.c(f32::from_bits(*last))?;
    for c in rest.iter().rev() {
        let c = e.c(f32::from_bits(*c))?;
        p = e.fused(p, f, c)?;
    }
    let lead = e.c(f32::from_bits(near.lead))?;
    let q = e.fused(p, f, lead)?;
    e.mul(f, q)
}

/// Metal's precise `log` or `log10` of `x`: the precise [`metal_log2`]'s product with `RN(ln 2)` or
/// `RN(log10(2))`, and inside their [`Near`] ranges next to 1 their own polynomials.
pub(crate) fn log(e: &mut E<'_, '_>, base: Base, x: Value) -> BResult<Value> {
    let l2 = metal_log2(e, x)?;
    let (c, near) = match base {
        Base::E => (LN_2_BITS, &LOG_NEAR),
        Base::Ten => (LOG10_2_BITS, &LOG10_NEAR),
    };
    let c = e.c(f32::from_bits(c))?;
    let far = e.mul(l2, c)?;
    let b = e.bits(x)?;
    let first = e.u(near.low + 1)?;
    let offset = e.iop(BinOp::ISub, b, first)?;
    let span = e.u(near.end - near.low - 1)?;
    let inside = e.iop(BinOp::ULessThan, offset, span)?;
    let r = near_one(e, x, near)?;
    let r = e.sel(inside, r, far)?;
    specials(e, x, r)
}

/// `r` with Metal's answers at `x`'s special operands: `-inf` for a zero or denormal, its NaN for
/// a negative number, `+inf` for `+inf`, and a NaN `x` itself.
fn specials(e: &mut E<'_, '_>, x: Value, r: Value) -> BResult<Value> {
    let b = e.bits(x)?;
    let (sign, field) = (e.u(0x8000_0000)?, e.u(0x7F80_0000)?);
    let zero = e.u(0)?;
    let s = e.iop(BinOp::BitwiseAnd, b, sign)?;
    let negative = e.iop(BinOp::INotEqual, s, zero)?;
    let ex = e.iop(BinOp::BitwiseAnd, b, field)?;
    let zero_or_denormal = e.iop(BinOp::IEqual, ex, zero)?;
    let inf = e.iop(BinOp::IEqual, b, field)?;
    let nan = e.metal_nan(None)?;
    let r = e.sel(negative, nan, r)?;
    let minus_inf = e.c(f32::NEG_INFINITY)?;
    let r = e.sel(zero_or_denormal, minus_inf, r)?;
    let plus_inf = e.c(f32::INFINITY)?;
    let r = e.sel(inf, plus_inf, r)?;
    let x_nan = e.is_nan(x)?;
    e.sel(x_nan, x, r)
}

/// [`powr`]'s reduction table, one row per centre `c` of the reduction (a significand `m` in
/// `(sqrt(1/2), sqrt(2)]` takes the row of its bits `20..24` less 3, whose centre is the left end
/// of its sixteenth (below 1) or eighth (from 1) of a binade, or 1 next to 1): `1 / c` and
/// `log2(c)` as pairs of `float`s by their bits, `[hi(1 / c), lo(1 / c), hi(log2 c), lo(log2 c)]`, each `hi` the
/// `float` nearest the value and `lo` the `float` nearest the rest (`libm::split` of the `f64`
/// value; `powr_table_is_the_centres_split`). With them the form is Metal's word at every
/// measured operand.
pub(crate) const POWR_TABLE: [[u32; 4]; 9] = [
    [0x3FBA_2E8C, 0xB33A_2E8C, 0xBF0A_62B0, 0xB2FE_68B0],
    [0x3FAA_AAAB, 0xB32A_AAAB, 0xBED4_7FCC, 0x3267_EF5A],
    [0x3F9D_89D9, 0xB344_EC4F, 0xBE99_5FF7, 0xB15C_3B9A],
    [0x3F92_4925, 0xB35B_6DB7, 0xBE45_44C0, 0xB1AB_FBD3],
    [0x3F80_0000, 0, 0, 0],
    [0x3F80_0000, 0, 0, 0],
    [0x3F63_8E39, 0xB1E3_8E39, 0x3E2E_00D2, 0xB140_852F],
    [0x3F4C_CCCD, 0xB24C_CCCD, 0x3EA4_D3C2, 0x323C_D1B9],
    [0x3F3A_2E8C, 0xB2BA_2E8C, 0x3EEB_3A9F, 0x2F4B_A83C],
];

/// An upper estimate of the SPIR-V words of [`POWR_TABLE`] in a module: its 36 constants (the
/// zeros shared), the composite, and the array, pointer, and variable.
pub(crate) const POWR_TABLE_WORDS: u32 = 36 * 4 + 39 + 24;

/// Declare [`POWR_TABLE`] in the module: a `Private` `u32[36]`, row `k`'s four from word `4 k`.
pub(crate) fn declare_powr(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    super::declare_words(mb, POWR_TABLE.as_flattened())
}

/// `P(u)` of [`powr`]'s logarithm, `log2(1 + u) = u (log2(e) + u P(u))` for `|u|` up to `1/8`:
/// `f32` coefficients by their bits, lowest first, evaluated by Horner's rule in fused steps
/// (Metal's polynomial; its word at every measured operand).
pub(crate) const POWR_P: [u32; 6] = [
    0xBF38_AA3B,
    0x3EF6_3846,
    0xBEB8_AB4B,
    0x3E93_D4FF,
    0xBE75_68A7,
    0x3E30_0BB9,
];

/// The bound `y` is clamped to before the product: past `2^31` every `y log2(x)` of `x` other
/// than 1 is beyond the exponential's range (`|log2(x)| >= 2^-24`), so the answer is the same,
/// and the products' exact errors stay in range.
const POWR_Y_BOUND: f32 = 2_147_483_648.0;

/// `log2(x)` of a positive normal `x` as Metal's `powr` forms it, a pair `(h, l)`: `x = 2^n m`
/// ([`significand`]), `m = c (1 + u)` of the row's centre `c`, `u = m (1 / c) - 1` as a pair (the
/// product's error exact, the low part of `1 / c` in a fused step); `L = log2(1 + u)` as the pair
/// `(u_h + u_l) (q_h + q_l)` of `q = log2(e) + u_h P(u_h)`; then `n + L + log2(c)` by two-term
/// sums, the larger term first.
fn powr_log2(e: &mut E<'_, '_>, x: Value) -> BResult<(Value, Value)> {
    let (ef, m) = significand(e, x)?;
    // The row: bits 20 to 23 of `m`, less 3 (`m` from `0x3F35_04F4` up to `SQRT2_BITS`).
    let mb = e.bits(m)?;
    let (k20, nibble, three, four) = (e.u(20)?, e.u(15)?, e.u(3)?, e.u(4)?);
    let row = e.iop(BinOp::ShiftRightLogical, mb, k20)?;
    let row = e.iop(BinOp::BitwiseAnd, row, nibble)?;
    let row = e.iop(BinOp::ISub, row, three)?;
    let first = e.iop(BinOp::IMul, row, four)?;
    let table = e.powr_table()?;
    let base = e.fb.module().global_value(table)?;
    let entry = |e: &mut E<'_, '_>, j: u32| -> BResult<Value> {
        let j = e.u(j)?;
        let i = e.iop(BinOp::IAdd, first, j)?;
        let p = e.fb.access_chain(base, &[i])?;
        let w = e.fb.load(p, MemAccess::default())?;
        e.float(w)
    };
    let (inv_hi, inv_lo) = (entry(e, 0)?, entry(e, 1)?);
    let (log_hi, log_lo) = (entry(e, 2)?, entry(e, 3)?);
    // `m / c` as a pair, `|p| >= |p_l|`, then `u = m / c - 1` (`m / c` is in `[15/16, 9/8)`, so
    // the difference with 1 is exact).
    let p = e.mul(m, inv_hi)?;
    let p_err = e.exact_error(m, inv_hi, p)?;
    let p_l = e.fused(m, inv_lo, p_err)?;
    let (s, s_l) = e.fast_two_sum(p, p_l)?;
    let minus_one = e.c(-1.0)?;
    let u = e.add(s, minus_one)?;
    let (uh, ul) = e.fast_two_sum(u, s_l)?;
    // `q = log2(e) + t` of `t = u_h P(u_h)`, `|t|` below a tenth, so `log2(e)` leads.
    let poly = e.fused_poly(&POWR_P, uh)?;
    let t = e.mul(poly, uh)?;
    let (l2e_hi, l2e_lo) = constant(LOG2_E);
    let l2e = e.c(l2e_hi)?;
    let q0 = e.add(t, l2e)?;
    let d = e.sub(l2e, q0)?;
    let d = e.add(t, d)?;
    let l2e_lo = e.c(l2e_lo)?;
    let q0_l = e.add(d, l2e_lo)?;
    let (qh, ql) = e.fast_two_sum(q0, q0_l)?;
    // `L = (u_h + u_l) (q_h + q_l)`: the high product's exact error, then the cross terms in
    // fused steps.
    let lp = e.mul(uh, qh)?;
    let l_err = e.exact_error(qh, uh, lp)?;
    let l_err = e.fused(qh, ul, l_err)?;
    let l_err = e.fused(ql, uh, l_err)?;
    let (lh, ll) = e.fast_two_sum(lp, l_err)?;
    // `n + L`: `|n| >= 1 > |L|`, or `n = 0`.
    let (a, a_l) = e.fast_two_sum(ef, lh)?;
    let r = e.add(ll, a_l)?;
    let (g, gl) = e.fast_two_sum(a, r)?;
    // `+ log2(c)`: the larger of `log2(c)` and `g` first.
    let s2 = e.add(log_hi, g)?;
    let (h_abs, g_abs) = (e.abs(log_hi)?, e.abs(g)?);
    let g_larger = e.lt(h_abs, g_abs)?;
    let d1 = e.sub(log_hi, s2)?;
    let d1 = e.add(g, d1)?;
    let d1 = e.add(gl, d1)?;
    let by_log = e.add(log_lo, d1)?;
    let d2 = e.sub(g, s2)?;
    let d2 = e.add(log_hi, d2)?;
    let d2 = e.add(log_lo, d2)?;
    let by_g = e.add(gl, d2)?;
    let rest = e.sel(g_larger, by_g, by_log)?;
    e.fast_two_sum(s2, rest)
}

/// The magnitude of Metal's precise `powr(x, y)` (and of `pow`'s, of `|x|`), Metal's own form: `z =
/// y log2(x)` as a pair of [`powr_log2`]'s pair (the high product's error exact, the low part's
/// product in a fused step), one unit of the last place down where `z_h` is at least 128 and `z_l`
/// negative, and then Metal's `exp2` of `z_h` grown by `z_l ln 2` in one fused step (`exp::grown`).
/// With `|x|`'s zero (a denormal read as one) and infinity selected; a NaN operand gives a NaN.
pub(crate) fn powr(e: &mut E<'_, '_>, x: Value, y: Value) -> BResult<Value> {
    let ax = e.abs(x)?;
    let (xs, _) = positive_normal(e, ax)?;
    let (gh, gl) = powr_log2(e, xs)?;
    let (yc, _) = e.clamp(y, POWR_Y_BOUND)?;
    let p = e.mul(gh, yc)?;
    let p_err = e.exact_error(gh, yc, p)?;
    let p_err = e.fused(gl, yc, p_err)?;
    let (zh, zl) = e.fast_two_sum(p, p_err)?;
    // From 128 with a negative low part, the high part one unit lower, so that its `exp2` is
    // finite where the pair's value is below 128.
    let (top, zero) = (e.c(128.0)?, e.c(0.0)?);
    let at_top = e.ge(zh, top)?;
    let below = e.lt(zl, zero)?;
    let lower = e.iop(BinOp::LogicalAnd, at_top, below)?;
    let zb = e.bits(zh)?;
    let one = e.u(1)?;
    let zb = e.iop(BinOp::ISub, zb, one)?;
    let z1 = e.float(zb)?;
    let step = e.sub(zh, z1)?;
    let zl1 = e.add(zl, step)?;
    let zh = e.sel(lower, z1, zh)?;
    let zl = e.sel(lower, zl1, zl)?;
    let big = exp::exp2(e, zh)?;
    let ln2 = e.c(f32::from_bits(LN_2_BITS))?;
    let growth = e.mul(zl, ln2)?;
    let r = exp::grown(e, big, growth)?;
    // `y = 1`: `|x|` itself, as Metal answers before its form (case 7b283404, where the form is a
    // unit above at three of the nine such lanes).
    let one_f = e.c(1.0)?;
    let y_one = e.fb.binary(BinOp::FOrdEqual, y, one_f)?;
    let r = e.sel(y_one, ax, r)?;
    // `|x|` zero or denormal, or infinite: `+0` or `+inf` by `y`'s sign.
    let b = e.bits(ax)?;
    let field = e.u(0x7F80_0000)?;
    let zero_u = e.u(0)?;
    let ex = e.iop(BinOp::BitwiseAnd, b, field)?;
    let x_zero = e.iop(BinOp::IEqual, ex, zero_u)?;
    let x_inf = e.iop(BinOp::IEqual, b, field)?;
    let inf = e.c(f32::INFINITY)?;
    let y_pos = e.gt(y, zero)?;
    let at_zero = e.sel(y_pos, zero, inf)?;
    let at_inf = e.sel(y_pos, inf, zero)?;
    let r = e.sel(x_zero, at_zero, r)?;
    let r = e.sel(x_inf, at_inf, r)?;
    let xy = e.add(x, y)?;
    let x_nan = e.is_nan(x)?;
    let y_nan = e.is_nan(y)?;
    let any_nan = e.iop(BinOp::LogicalOr, x_nan, y_nan)?;
    e.sel(any_nan, xy, r)
}
