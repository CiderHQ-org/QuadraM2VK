//! The exponentials: Metal's `exp2` of a `float`, its precise `exp` and `exp10` (MSL Table 8.1
//! bounds each at 4 ulp), and the growth step they and `powr` share.
//!
//! **Metal's `exp2`** ([`exp2`]): what Metal's hardware computes, measured on every `float` operand
//! of the Apple M4 and on 2^24 of the M3, which answers alike. An operand below `2^-23` in
//! magnitude reads as 0, and a NaN gives `0x7fc00000`. Otherwise `k = floor(x 2^23)` exactly,
//! `n = k >> 23` and `f = k mod 2^23`, and the answer is `2^n (1 + T(f) 2^-23)`: `+0` where its
//! exponent field would be 0 or below (Metal flushes a denormal result: `exp2(-127)` is `+0`),
//! `+inf` from 255. `T` is the 23-bit fraction of a cubic per 32nd of `f`, which [`TABLE`] gives as
//! Horner's rule in fixed point with each product truncated: it answers Metal's `T` at 8,380,655 of
//! the 2^23 fractions (99.905 %), and its word is one unit of the last place off at the other
//! 7,953, which no fixed-point form tried reproduces (an open near-equivalent).
//!
//! **Metal's precise `exp` and `exp10`** ([`exp_base`]): Metal's own composition of its `exp2`,
//! measured on every lane of case 7b283404 on the M3 and the M4, and on every operand of
//! magnitude `[2^-40, 2^9)` whose answer is normal on the M4: `E = exp2(p)`
//! of the rounded product `p = x c_hi` (`c_hi + c_lo` the constant `log2(e)` or `log2(10)`),
//! corrected by the product's error `d = fma(x, c_lo, fma(x, c_hi, -p))` as the fused
//! `fma(E, RN(d ln 2), E)`, and `E` itself where it is `+inf`, `+0`, or a NaN. Each fused step is
//! exact on every device ([`E::fused`], [`fused_growth`]).
//!
//! **Metal's growth step** ([`grown`]): `fma(E, t, E)` of a power `E` of its `exp2` and a small
//! correction `t`, which its `exp`, `exp10`, and `powr` take, exact on every device.

use crate::lower::arith::libm::pair::constant;
use crate::lower::arith::libm::E;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, MemAccess};
use std::f64::consts::{LN_2, LOG2_10, LOG2_E};

/// Metal's `exp2` table ([`exp2`]): for each 32nd `s` of the fraction `f`, the fixed-point
/// coefficients `[c0, c1, c2, c3]` of the cubic that gives `T(f)` of the offset `t = f mod 2^18`,
/// by Horner's rule with each product truncated: `u2 = c2 + (c3 t >> 18)`,
/// `u1 = c1 + (u2 t >> 18)`, `Y = c0 + (u1 t >> 18)`, and `T = Y >> 9`. Each `Y` is below `2^32`
/// (the largest is `0xfffffef0`, at the last fraction; every coefficient is positive, so `Y` grows
/// with `t`), `c3 t` below `2^32`, and `u2 t` and `u1 t` below `2^39` and `2^46`. Fitted to Metal's
/// `T` on every fraction: equal at 8,380,655 of 2^23.
pub(crate) const TABLE: [[u32; 4]; 32] = [
    [0x0000_01B8, 0x058B_90BC, 0x000F_5F93, 0x0000_1CF2],
    [0x059B_0EF9, 0x05AA_A5FA, 0x000F_B693, 0x0000_1CF3],
    [0x0B55_8879, 0x05CA_6A3A, 0x0010_0E12, 0x0000_1DF4],
    [0x1130_1EB9, 0x05EA_E03B, 0x0010_678F, 0x0000_1EF6],
    [0x172B_8599, 0x060C_0BB9, 0x0010_C495, 0x0000_1EF2],
    [0x1D48_74D9, 0x062D_F1B8, 0x0011_2218, 0x0000_1FF0],
    [0x2387_A898, 0x0650_9600, 0x0011_8189, 0x0000_20F8],
    [0x29E9_E119, 0x0673_FBB9, 0x0011_E494, 0x0000_20F3],
    [0x306F_E259, 0x0698_283F, 0x0012_4882, 0x0000_2200],
    [0x371A_7539, 0x06BD_1FB8, 0x0012_AE97, 0x0000_22F1],
    [0x3DEA_667A, 0x06E2_E632, 0x0013_1720, 0x0000_23ED],
    [0x44E0_87D9, 0x0709_7FBC, 0x0013_8389, 0x0000_23FC],
    [0x4BFD_AF19, 0x0730_F279, 0x0013_F091, 0x0000_24F7],
    [0x5342_B718, 0x0759_4282, 0x0014_6000, 0x0000_2600],
    [0x5AB0_7F98, 0x0782_7441, 0x0014_D204, 0x0000_26FC],
    [0x6247_EC99, 0x07AC_8D76, 0x0015_461D, 0x0000_27ED],
    [0x6A09_E819, 0x07D7_91B6, 0x0015_BD9F, 0x0000_28EA],
    [0x71F7_6039, 0x0803_87B9, 0x0016_3713, 0x0000_29F4],
    [0x7A11_48F9, 0x0830_7339, 0x0016_B514, 0x0000_29F3],
    [0x8258_9B58, 0x085E_5B7E, 0x0017_340E, 0x0000_2AF5],
    [0x8ACE_55D9, 0x088D_44F8, 0x0017_B615, 0x0000_2BF3],
    [0x9373_7CD8, 0x08BD_34FF, 0x0018_3B07, 0x0000_2CFC],
    [0x9C49_19F9, 0x08EE_31F9, 0x0018_C312, 0x0000_2DF5],
    [0xA550_3CD9, 0x0920_41F8, 0x0019_4D95, 0x0000_2EF3],
    [0xAE89_FB59, 0x0953_69FA, 0x0019_DB91, 0x0000_2FF5],
    [0xB7F7_70F8, 0x0987_B0FD, 0x001A_6C8E, 0x0000_30F6],
    [0xC199_BF79, 0x09BD_1D76, 0x001B_009E, 0x0000_31EB],
    [0xCB72_0F79, 0x09F3_B576, 0x001B_969C, 0x0000_33EE],
    [0xD581_8F78, 0x0A2B_7EFD, 0x001C_318D, 0x0000_34F7],
    [0xDFC9_74F9, 0x0A64_80F5, 0x001C_D01F, 0x0000_35EC],
    [0xEA4A_FBD9, 0x0A9E_C2F4, 0x001D_71A0, 0x0000_36EC],
    [0xF507_6779, 0x0ADA_4B32, 0x001E_1727, 0x0000_37E6],
];

/// An upper estimate of the SPIR-V words of [`TABLE`] in a module: its 128 constants, the
/// composite, and the array, pointer, and variable.
pub(crate) const TABLE_WORDS: u32 = 128 * 4 + 131 + 24;

/// Declare [`TABLE`] in the module: a `Private` `u32[128]` initialized to the coefficients, `s`'s
/// four from word `4 s`.
pub(crate) fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    super::declare_words(mb, TABLE.as_flattened())
}

/// The magnitude past which Metal's `exp2` is `+0` or `+inf`: [`exp2`] clamps its operand to it.
pub(crate) const LIMIT: f32 = 200.0;

/// `(a b) >> 18` of `u32` values `a` and `b` whose product is below `2^50`.
fn product_shifted(e: &mut E<'_, '_>, a: Value, b: Value) -> BResult<Value> {
    let (lo, hi) = e.umul(a, b)?;
    let (k14, k18) = (e.u(14)?, e.u(18)?);
    let hi = e.iop(BinOp::ShiftLeftLogical, hi, k14)?;
    let lo = e.iop(BinOp::ShiftRightLogical, lo, k18)?;
    e.iop(BinOp::BitwiseOr, hi, lo)
}

/// Metal's `exp2(x)` (the module docs): the reduction exact on `float`s (`x 2^23` and its floor,
/// `n = floor(k 2^-23)`, and `f = k - n 2^23`, each exact; `x` clamped to `[-200, 200]` first,
/// where `n` is past the flush and the infinity both ways), and `T` of [`TABLE`] on `u32`s.
pub(crate) fn exp2(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let table = e.exp2_table()?;
    let nan = e.is_nan(x)?;
    let zero = e.c(0.0)?;
    let ax = e.abs(x)?;
    let tiny = e.c(f32::EPSILON)?;
    let tiny = e.lt(ax, tiny)?;
    let x = e.sel(tiny, zero, x)?;
    let x = e.sel(nan, zero, x)?;
    let (x, _) = e.clamp(x, LIMIT)?;
    let (up, down) = (e.c(8_388_608.0)?, e.c(1.0 / 8_388_608.0)?);
    let k = e.mul(x, up)?;
    let k = e.floor(k)?;
    let n = e.mul(k, down)?;
    let n = e.floor(n)?;
    let whole = e.mul(n, up)?;
    let f = e.sub(k, whole)?;
    let f = e.utrunc(f)?;
    // `n + 256`, in `[56, 456]`.
    let offset = e.c(256.0)?;
    let n = e.add(n, offset)?;
    let n = e.utrunc(n)?;
    let (k2, k9, k18, k23) = (e.u(2)?, e.u(9)?, e.u(18)?, e.u(23)?);
    let s = e.iop(BinOp::ShiftRightLogical, f, k18)?;
    let low = e.u(0x3_FFFF)?;
    let t = e.iop(BinOp::BitwiseAnd, f, low)?;
    let first = e.iop(BinOp::ShiftLeftLogical, s, k2)?;
    let base = e.fb.module().global_value(table)?;
    let coefficient = |e: &mut E<'_, '_>, j: u32| -> BResult<Value> {
        let j = e.u(j)?;
        let i = e.iop(BinOp::IAdd, first, j)?;
        let p = e.fb.access_chain(base, &[i])?;
        e.fb.load(p, MemAccess::default())
    };
    let c3 = coefficient(e, 3)?;
    let p3 = e.iop(BinOp::IMul, c3, t)?;
    let p3 = e.iop(BinOp::ShiftRightLogical, p3, k18)?;
    let c2 = coefficient(e, 2)?;
    let u2 = e.iop(BinOp::IAdd, c2, p3)?;
    let p2 = product_shifted(e, u2, t)?;
    let c1 = coefficient(e, 1)?;
    let u1 = e.iop(BinOp::IAdd, c1, p2)?;
    let p1 = product_shifted(e, u1, t)?;
    let c0 = coefficient(e, 0)?;
    let y = e.iop(BinOp::IAdd, c0, p1)?;
    let fraction = e.iop(BinOp::ShiftRightLogical, y, k9)?;
    // The exponent field `n + 127`: `+0` where it is 0 or below, `+inf` from 255.
    let (bias, top) = (e.u(129)?, e.u(384)?);
    let field = e.iop(BinOp::ISub, n, bias)?;
    let field = e.iop(BinOp::ShiftLeftLogical, field, k23)?;
    let r = e.iop(BinOp::BitwiseOr, field, fraction)?;
    let under = e.iop(BinOp::ULessThanEqual, n, bias)?;
    let over = e.iop(BinOp::UGreaterThanEqual, n, top)?;
    let (inf, flushed, quiet) = (e.u(0x7F80_0000)?, e.u(0)?, e.u(super::METAL_NAN)?);
    let r = e.sel(over, inf, r)?;
    let r = e.sel(under, flushed, r)?;
    let r = e.sel(nan, quiet, r)?;
    e.float(r)
}

/// `fma(m, t, m)`, `m (1 + t)` rounded once, for a positive normal `m` and `|t|` below `2^-8`, on
/// every device and without a fused operation: on `m`'s significand `M` in `[1, 2)`, the exact
/// product `M t = y + y_err` ([`E::two_prod`]) rounded to odd ([`E::odd`]), and `RN(M + o)`,
/// which is `RN(M + M t)`: `o` lies on a grid at least 2^-30 fine, so `M + o` is on the same side
/// of each midpoint of the result's grid as `M + M t`, and on none where that is not on one
/// (Boldo and Melquiond's rounding to odd). Then `m`'s exponent back on the bits: `+inf` past the
/// largest finite `float`, `+0` below `2^-126`. A product below `2^-100`, whose error
/// [`E::exact_error`] may not give exactly, leaves `M` unchanged either way. Unlike
/// [`E::fused`], the scaling keeps it exact where `m t` would underflow.
fn fused_growth(e: &mut E<'_, '_>, m: Value, t: Value) -> BResult<Value> {
    let mb = e.bits(m)?;
    let (fraction, one) = (e.u(0x007F_FFFF)?, e.u(0x3F80_0000)?);
    let mm = e.iop(BinOp::BitwiseAnd, mb, fraction)?;
    let mm = e.iop(BinOp::BitwiseOr, mm, one)?;
    let mm = e.float(mm)?;
    let (y, y_err) = e.two_prod(mm, t)?;
    let o = e.odd(y, y_err)?;
    let r = e.add(mm, o)?;
    let rb = e.bits(r)?;
    let k23 = e.u(23)?;
    let mf = e.iop(BinOp::ShiftRightLogical, mb, k23)?;
    let rf = e.iop(BinOp::ShiftRightLogical, rb, k23)?;
    // The result's exponent field plus 127.
    let sum = e.iop(BinOp::IAdd, mf, rf)?;
    let scale = e.iop(BinOp::ShiftLeftLogical, mf, k23)?;
    let bias = e.u(127 << 23)?;
    let out = e.iop(BinOp::IAdd, rb, scale)?;
    let out = e.iop(BinOp::ISub, out, bias)?;
    let (top, bottom) = (e.u(382)?, e.u(127)?);
    let over = e.iop(BinOp::UGreaterThanEqual, sum, top)?;
    let under = e.iop(BinOp::ULessThanEqual, sum, bottom)?;
    let (inf, zero) = (e.u(0x7F80_0000)?, e.u(0)?);
    let out = e.sel(over, inf, out)?;
    let out = e.sel(under, zero, out)?;
    e.float(out)
}

/// Metal's precise `exp(x)` (`ten` false) or `exp10(x)` (`ten` true), as Metal composes it of its
/// `exp2` (the module docs): `fma(E, RN(d ln 2), E)` of `d = fma(x, c_lo, fma(x, c_hi, -p))`,
/// both fused steps exact where they decide the answer (`x c_hi` above `2^-100`; below, `E` is 1
/// and the answer 1).
pub(crate) fn exp_base(e: &mut E<'_, '_>, x: Value, ten: bool) -> BResult<Value> {
    composed(e, x, ten, false)
}

/// `e^x / 2` as Metal's `sinh` and `cosh` form it for `x` from 1.5 on: [`exp_base`]'s composition
/// of `E = exp2(p - 1)`, so it is finite up to 89.4. It is exactly half of [`exp_base`]'s answer
/// wherever that is finite (the composition scales exactly), and up to 89.4 it is an open
/// near-equivalent (Metal's word on 62 % of those operands, the rest one or two units of the last
/// place off).
pub(crate) fn exp_half(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    composed(e, x, false, true)
}

/// [`exp_base`] and [`exp_half`]: `p - 1` is exact for every `p` from 1 on, where `exp_half`'s
/// answer is used.
fn composed(e: &mut E<'_, '_>, x: Value, ten: bool, halved: bool) -> BResult<Value> {
    let (hi, lo) = constant(if ten { LOG2_10 } else { LOG2_E });
    let hi = e.c(hi)?;
    let p = e.mul(x, hi)?;
    let err = e.exact_error(x, hi, p)?;
    let lo = e.c(lo)?;
    let d = e.fused(x, lo, err)?;
    let p = if halved {
        let one = e.c(1.0)?;
        e.sub(p, one)?
    } else {
        p
    };
    let big = exp2(e, p)?;
    let ln2 = e.c(LN_2 as f32)?;
    let t = e.mul(d, ln2)?;
    grown(e, big, t)
}

/// `fma(E, t, E)` of a value `E` of Metal's [`exp2`] and `|t|` below `2^-8` (or any `t` where `E`
/// is `+inf` or `+0`), as Metal grows its `exp2` in its `exp`, `exp10`, and `powr`
/// ([`fused_growth`]), and `E` itself where it is `+inf`, `+0`, or a NaN.
pub(crate) fn grown(e: &mut E<'_, '_>, big: Value, t: Value) -> BResult<Value> {
    let r = fused_growth(e, big, t)?;
    let b = e.bits(big)?;
    let (field, zero) = (e.u(0x7F80_0000)?, e.u(0)?);
    let top = e.iop(BinOp::BitwiseAnd, b, field)?;
    let top = e.iop(BinOp::IEqual, top, field)?;
    let flushed = e.iop(BinOp::IEqual, b, zero)?;
    let ends = e.iop(BinOp::LogicalOr, top, flushed)?;
    e.sel(ends, big, r)
}
