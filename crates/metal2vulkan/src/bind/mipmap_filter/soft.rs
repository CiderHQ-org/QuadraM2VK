//! Binary floating point in integers for the kernel (`kernel.rs`): the Apple M3 Max's tail
//! arithmetic and the values a level keeps, each rounding exact (the module docs' "The M3
//! Max's tail"). A finite value is `(-1)^neg m 2^e` with `m` a `u64`; the precision of a
//! rounding is a [`Prec`] of values, since a field's record holds it.

use super::kernel::{bin, bink, c32, c64, select, K};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, UnOp};

/// A value: `(-1)^neg m 2^e` (`m` a `u64`, `e` an `i32` in a `u32`), unless `inf` or `nan`.
#[derive(Clone, Copy)]
pub(super) struct Soft {
    pub(super) neg: Value,
    pub(super) m: Value,
    pub(super) e: Value,
    pub(super) inf: Value,
    pub(super) nan: Value,
}

/// A rounding's significand bits and least normal exponent (`i32` in a `u32`).
#[derive(Clone, Copy)]
pub(super) struct Prec {
    pub(super) bits: Value,
    pub(super) least: Value,
}

/// The constant `false`.
pub(super) fn falsy(f: &mut FunctionBuilder<'_>) -> BResult<Value> {
    let zero = c32(f, 0)?;
    bin(f, BinOp::INotEqual, zero, zero)
}

/// The index of the highest set bit of `u64` `v`, -1 for 0 (an `i32` in a `u32`).
pub(super) fn msb64(f: &mut FunctionBuilder<'_>, k: &K, v: Value) -> BResult<Value> {
    let thirty_two = c32(f, 32)?;
    let hi = bin(f, BinOp::ShiftRightLogical, v, thirty_two)?;
    let hi = f.convert(ConvOp::UConvert, hi, k.u32_)?;
    let lo = f.convert(ConvOp::UConvert, v, k.u32_)?;
    let mh = f.glsl(GlslOp::FindUMsb, &[hi])?;
    let mh = bin(f, BinOp::IAdd, mh, thirty_two)?;
    let ml = f.glsl(GlslOp::FindUMsb, &[lo])?;
    let zero = c32(f, 0)?;
    let high = bin(f, BinOp::INotEqual, hi, zero)?;
    select(f, high, mh, ml)
}

/// A finite value of `m` and `e`, its sign `neg`.
fn finite(f: &mut FunctionBuilder<'_>, neg: Value, m: Value, e: Value) -> BResult<Soft> {
    let no = falsy(f)?;
    Ok(Soft {
        neg,
        m,
        e,
        inf: no,
        nan: no,
    })
}

/// `(-1)^neg (mag + s) 2^exp` rounded to nearest even at `pr` (subnormals on the least normal's
/// quantum), where `s` is a positive amount below one unit when `sticky` (some bits a sum lost
/// below `mag`'s last). `mag` is below `2^63`. No overflow is possible where it is used.
pub(super) fn round(
    f: &mut FunctionBuilder<'_>,
    k: &K,
    neg: Value,
    mag: Value,
    exp: Value,
    sticky: Value,
    pr: Prec,
) -> BResult<Soft> {
    let zero = c32(f, 0)?;
    let one = c32(f, 1)?;
    let b = msb64(f, k, mag)?;
    let x = bin(f, BinOp::IAdd, b, exp)?;
    let floor = f.glsl(GlslOp::SMax, &[x, pr.least])?;
    let below = bin(f, BinOp::ISub, pr.bits, one)?;
    let ulp = bin(f, BinOp::ISub, floor, below)?;
    let drop = bin(f, BinOp::ISub, ulp, exp)?;
    let dropping = bin(f, BinOp::SGreaterThan, drop, zero)?;
    let d = select(f, dropping, drop, one)?;
    let sixty_four = c32(f, 64)?;
    let big = bin(f, BinOp::UGreaterThanEqual, d, sixty_four)?;
    let cap = c32(f, 63)?;
    let d63 = bink(f, BinOp::BitwiseAnd, d, 63)?;
    let d63 = select(f, big, cap, d63)?;
    let zero64 = c64(f, k, 0)?;
    let one64 = c64(f, k, 1)?;
    let shifted = bin(f, BinOp::ShiftRightLogical, mag, d63)?;
    let kept = select(f, big, zero64, shifted)?;
    let bit = bin(f, BinOp::ShiftLeftLogical, one64, d63)?;
    let low = bin(f, BinOp::ISub, bit, one64)?;
    let all = c64(f, k, u64::MAX)?;
    let low = select(f, big, all, low)?;
    let rem = bin(f, BinOp::BitwiseAnd, mag, low)?;
    let d1 = bin(f, BinOp::ISub, d63, one)?;
    let half = bin(f, BinOp::ShiftLeftLogical, one64, d1)?;
    let top = c64(f, k, 1 << 63)?;
    let half = select(f, big, top, half)?;
    let over = bin(f, BinOp::UGreaterThan, rem, half)?;
    let at = bin(f, BinOp::IEqual, rem, half)?;
    let at_sticky = bin(f, BinOp::LogicalAnd, at, sticky)?;
    let above = bin(f, BinOp::LogicalOr, over, at_sticky)?;
    let exact = f.unary(UnOp::LogicalNot, sticky)?;
    let tie = bin(f, BinOp::LogicalAnd, at, exact)?;
    let lsb = bin(f, BinOp::BitwiseAnd, kept, one64)?;
    let odd = bin(f, BinOp::INotEqual, lsb, zero64)?;
    let tie_odd = bin(f, BinOp::LogicalAnd, tie, odd)?;
    let up = bin(f, BinOp::LogicalOr, above, tie_odd)?;
    let step = select(f, up, one64, zero64)?;
    let rounded = bin(f, BinOp::IAdd, kept, step)?;
    let minus = bin(f, BinOp::ISub, zero, drop)?;
    let l = f.glsl(GlslOp::UMin, &[minus, cap])?;
    let widened = bin(f, BinOp::ShiftLeftLogical, mag, l)?;
    let r = select(f, dropping, rounded, widened)?;
    let carried = bin(f, BinOp::ShiftLeftLogical, one64, pr.bits)?;
    let carry = bin(f, BinOp::IEqual, r, carried)?;
    let halved = bin(f, BinOp::ShiftRightLogical, r, one)?;
    let r = select(f, carry, halved, r)?;
    let ulp_up = bin(f, BinOp::IAdd, ulp, one)?;
    let ulp = select(f, carry, ulp_up, ulp)?;
    finite(f, neg, r, ulp)
}

/// A value of a float field's bits: sign `neg`, exponent field `ef`, mantissa field `mf`; the
/// format `(mb, emax, bias)`: `mb` mantissa bits, the exponent field of infinities and NaNs
/// `emax`, and the exponent bias plus `mb`.
pub(super) fn unpack(
    f: &mut FunctionBuilder<'_>,
    k: &K,
    neg: Value,
    ef: Value,
    mf: Value,
    (mb, emax, bias): (Value, Value, Value),
) -> BResult<Soft> {
    let zero = c32(f, 0)?;
    let one = c32(f, 1)?;
    let special = bin(f, BinOp::IEqual, ef, emax)?;
    let quiet = bin(f, BinOp::INotEqual, mf, zero)?;
    let nan = bin(f, BinOp::LogicalAnd, special, quiet)?;
    let loud = f.unary(UnOp::LogicalNot, quiet)?;
    let inf = bin(f, BinOp::LogicalAnd, special, loud)?;
    let normal = bin(f, BinOp::INotEqual, ef, zero)?;
    let hidden = bin(f, BinOp::ShiftLeftLogical, one, mb)?;
    let with_hidden = bin(f, BinOp::BitwiseOr, mf, hidden)?;
    let m = select(f, normal, with_hidden, mf)?;
    let m = select(f, special, zero, m)?;
    let m = f.convert(ConvOp::UConvert, m, k.u64_)?;
    let en = bin(f, BinOp::ISub, ef, bias)?;
    let es = bin(f, BinOp::ISub, one, bias)?;
    let e = select(f, normal, en, es)?;
    Ok(Soft {
        neg,
        m,
        e,
        inf,
        nan,
    })
}

/// The value of `f32` bits.
pub(super) fn unpack_f32(f: &mut FunctionBuilder<'_>, k: &K, bits: Value) -> BResult<Soft> {
    let s = bink(f, BinOp::ShiftRightLogical, bits, 31)?;
    let zero = c32(f, 0)?;
    let neg = bin(f, BinOp::INotEqual, s, zero)?;
    let ef = bink(f, BinOp::ShiftRightLogical, bits, 23)?;
    let ef = bink(f, BinOp::BitwiseAnd, ef, 0xff)?;
    let mf = bink(f, BinOp::BitwiseAnd, bits, 0x7f_ffff)?;
    let format = (c32(f, 23)?, c32(f, 255)?, c32(f, 150)?);
    unpack(f, k, neg, ef, mf, format)
}

/// The value of half-float bits.
pub(super) fn unpack_f16(f: &mut FunctionBuilder<'_>, k: &K, bits: Value) -> BResult<Soft> {
    let s = bink(f, BinOp::ShiftRightLogical, bits, 15)?;
    let s = bink(f, BinOp::BitwiseAnd, s, 1)?;
    let zero = c32(f, 0)?;
    let neg = bin(f, BinOp::INotEqual, s, zero)?;
    let ef = bink(f, BinOp::ShiftRightLogical, bits, 10)?;
    let ef = bink(f, BinOp::BitwiseAnd, ef, 0x1f)?;
    let mf = bink(f, BinOp::BitwiseAnd, bits, 0x3ff)?;
    let format = (c32(f, 10)?, c32(f, 31)?, c32(f, 25)?);
    unpack(f, k, neg, ef, mf, format)
}

/// The `f32` bits of a value of at most 24 significant bits on `f32`'s quantum (zeros +0, NaNs
/// quiet).
pub(super) fn pack_f32(f: &mut FunctionBuilder<'_>, k: &K, v: Soft) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let zero64 = c64(f, k, 0)?;
    let sign = c32(f, 0x8000_0000)?;
    let sign = select(f, v.neg, sign, zero)?;
    let b = msb64(f, k, v.m)?;
    let x = bin(f, BinOp::IAdd, b, v.e)?;
    let least = c32(f, -126i32 as u32)?;
    let normal = bin(f, BinOp::SGreaterThanEqual, x, least)?;
    let twenty_three = c32(f, 23)?;
    let cap = c32(f, 63)?;
    let shift = bin(f, BinOp::ISub, twenty_three, b)?;
    let shift = f.glsl(GlslOp::UMin, &[shift, cap])?;
    let mn = bin(f, BinOp::ShiftLeftLogical, v.m, shift)?;
    let mn = f.convert(ConvOp::UConvert, mn, k.u32_)?;
    let mn = bink(f, BinOp::BitwiseAnd, mn, 0x7f_ffff)?;
    let biased = bink(f, BinOp::IAdd, x, 127)?;
    let biased = bink(f, BinOp::ShiftLeftLogical, biased, 23)?;
    let nbits = bin(f, BinOp::BitwiseOr, sign, biased)?;
    let nbits = bin(f, BinOp::BitwiseOr, nbits, mn)?;
    let sub = bink(f, BinOp::IAdd, v.e, 149)?;
    let sub = f.glsl(GlslOp::UMin, &[sub, cap])?;
    let sm = bin(f, BinOp::ShiftLeftLogical, v.m, sub)?;
    let sm = f.convert(ConvOp::UConvert, sm, k.u32_)?;
    let sbits = bin(f, BinOp::BitwiseOr, sign, sm)?;
    let bits = select(f, normal, nbits, sbits)?;
    let is_zero = bin(f, BinOp::IEqual, v.m, zero64)?;
    let bits = select(f, is_zero, zero, bits)?;
    let inf = bink(f, BinOp::BitwiseOr, sign, 0x7f80_0000)?;
    let bits = select(f, v.inf, inf, bits)?;
    let nan = c32(f, 0x7fc0_0000)?;
    select(f, v.nan, nan, bits)
}

/// The half-float bits of a value that is a half float (zeros +0).
pub(super) fn pack_f16(f: &mut FunctionBuilder<'_>, k: &K, v: Soft) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let zero64 = c64(f, k, 0)?;
    let sign = c32(f, 0x8000)?;
    let sign = select(f, v.neg, sign, zero)?;
    let b = msb64(f, k, v.m)?;
    let x = bin(f, BinOp::IAdd, b, v.e)?;
    let least = c32(f, -14i32 as u32)?;
    let normal = bin(f, BinOp::SGreaterThanEqual, x, least)?;
    let ten = c32(f, 10)?;
    let cap = c32(f, 63)?;
    let shift = bin(f, BinOp::ISub, ten, b)?;
    let shift = f.glsl(GlslOp::UMin, &[shift, cap])?;
    let mn = bin(f, BinOp::ShiftLeftLogical, v.m, shift)?;
    let mn = f.convert(ConvOp::UConvert, mn, k.u32_)?;
    let mn = bink(f, BinOp::BitwiseAnd, mn, 0x3ff)?;
    let biased = bink(f, BinOp::IAdd, x, 15)?;
    let biased = bink(f, BinOp::ShiftLeftLogical, biased, 10)?;
    let nbits = bin(f, BinOp::BitwiseOr, sign, biased)?;
    let nbits = bin(f, BinOp::BitwiseOr, nbits, mn)?;
    let sub = bink(f, BinOp::IAdd, v.e, 24)?;
    let sub = f.glsl(GlslOp::UMin, &[sub, cap])?;
    let sm = bin(f, BinOp::ShiftLeftLogical, v.m, sub)?;
    let sm = f.convert(ConvOp::UConvert, sm, k.u32_)?;
    let sbits = bin(f, BinOp::BitwiseOr, sign, sm)?;
    let bits = select(f, normal, nbits, sbits)?;
    let is_zero = bin(f, BinOp::IEqual, v.m, zero64)?;
    let bits = select(f, is_zero, zero, bits)?;
    let inf = bink(f, BinOp::BitwiseOr, sign, 0x7c00)?;
    let bits = select(f, v.inf, inf, bits)?;
    let nan = c32(f, 0x7e00)?;
    select(f, v.nan, nan, bits)
}

/// `n / d` (`n` an `i32` of magnitude below `2^18`, `d` a `u32` from 1 to `2^17`) rounded to
/// nearest even at `pr`: the quotient's 27 bits or more and whether a remainder is left.
pub(super) fn ratio(
    f: &mut FunctionBuilder<'_>,
    k: &K,
    n: Value,
    d: Value,
    pr: Prec,
) -> BResult<Soft> {
    let zero = c32(f, 0)?;
    let neg = bin(f, BinOp::SLessThan, n, zero)?;
    let a = f.glsl(GlslOp::SAbs, &[n])?;
    let a = f.convert(ConvOp::UConvert, a, k.u64_)?;
    let shift = c32(f, 44)?;
    let num = bin(f, BinOp::ShiftLeftLogical, a, shift)?;
    let d = f.convert(ConvOp::UConvert, d, k.u64_)?;
    let q = bin(f, BinOp::UDiv, num, d)?;
    let back = bin(f, BinOp::IMul, q, d)?;
    let rem = bin(f, BinOp::ISub, num, back)?;
    let zero64 = c64(f, k, 0)?;
    let sticky = bin(f, BinOp::INotEqual, rem, zero64)?;
    let exp = c32(f, -44i32 as u32)?;
    round(f, k, neg, q, exp, sticky, pr)
}

/// The exact sum of two finite values of at most 36 significant bits each: `(neg, mag, exp,
/// sticky)`, the sum `(-1)^neg (mag + s) 2^exp` with `s` below one unit (`sticky` where it is
/// not 0), `mag` below `2^63` and, where `sticky`, at least `2^60`.
fn add(
    f: &mut FunctionBuilder<'_>,
    k: &K,
    x: (Value, Value, Value),
    y: (Value, Value, Value),
) -> BResult<(Value, Value, Value, Value)> {
    let zero64 = c64(f, k, 0)?;
    let one64 = c64(f, k, 1)?;
    let sixty_one = c32(f, 61)?;
    let cap = c32(f, 63)?;
    // Each magnitude's top bit moved to bit 61 (a zero stays 0).
    let normalize = |f: &mut FunctionBuilder<'_>, (n, m, e): (Value, Value, Value)| {
        let b = msb64(f, k, m)?;
        let l = bin(f, BinOp::ISub, sixty_one, b)?;
        let l = f.glsl(GlslOp::UMin, &[l, cap])?;
        let mm = bin(f, BinOp::ShiftLeftLogical, m, l)?;
        let ee = bin(f, BinOp::ISub, e, l)?;
        let z = bin(f, BinOp::IEqual, m, zero64)?;
        BResult::Ok((n, mm, ee, z))
    };
    let (nx, mx, ex, zx) = normalize(f, x)?;
    let (ny, my, ey, zy) = normalize(f, y)?;
    let x_first = bin(f, BinOp::SGreaterThanEqual, ex, ey)?;
    let nonzero_x = f.unary(UnOp::LogicalNot, zx)?;
    let x_first = bin(f, BinOp::LogicalAnd, nonzero_x, x_first)?;
    let x_first = bin(f, BinOp::LogicalOr, zy, x_first)?;
    let a = select(f, x_first, mx, my)?;
    let ea = select(f, x_first, ex, ey)?;
    let na = select(f, x_first, nx, ny)?;
    let b = select(f, x_first, my, mx)?;
    let eb = select(f, x_first, ey, ex)?;
    let nb = select(f, x_first, ny, nx)?;
    // b aligned to a's exponent, the bits it loses kept as `sticky`.
    let d = bin(f, BinOp::ISub, ea, eb)?;
    let sixty_four = c32(f, 64)?;
    let far = bin(f, BinOp::UGreaterThanEqual, d, sixty_four)?;
    let d63 = f.glsl(GlslOp::UMin, &[d, cap])?;
    let shifted = bin(f, BinOp::ShiftRightLogical, b, d63)?;
    let b2 = select(f, far, zero64, shifted)?;
    let bit = bin(f, BinOp::ShiftLeftLogical, one64, d63)?;
    let low = bin(f, BinOp::ISub, bit, one64)?;
    let all = c64(f, k, u64::MAX)?;
    let low = select(f, far, all, low)?;
    let lost = bin(f, BinOp::BitwiseAnd, b, low)?;
    let sticky = bin(f, BinOp::INotEqual, lost, zero64)?;
    let same = bin(f, BinOp::LogicalEqual, na, nb)?;
    let sum = bin(f, BinOp::IAdd, a, b2)?;
    let a_ge = bin(f, BinOp::UGreaterThanEqual, a, b2)?;
    let borrow = select(f, sticky, one64, zero64)?;
    let ab = bin(f, BinOp::ISub, a, b2)?;
    let ab = bin(f, BinOp::ISub, ab, borrow)?;
    let ba = bin(f, BinOp::ISub, b2, a)?;
    let diff = select(f, a_ge, ab, ba)?;
    let nd = select(f, a_ge, na, nb)?;
    let mag = select(f, same, sum, diff)?;
    let neg = select(f, same, na, nd)?;
    Ok((neg, mag, ea, sticky))
}

/// `a` where `pick`, else `b`.
pub(super) fn choose(f: &mut FunctionBuilder<'_>, pick: Value, a: Soft, b: Soft) -> BResult<Soft> {
    Ok(Soft {
        neg: select(f, pick, a.neg, b.neg)?,
        m: select(f, pick, a.m, b.m)?,
        e: select(f, pick, a.e, b.e)?,
        inf: select(f, pick, a.inf, b.inf)?,
        nan: select(f, pick, a.nan, b.nan)?,
    })
}

/// The M3 Max's mix of `a` and `b` at `t = w / 2048` (`w` a `u32` from 0 to 2047):
/// `a (1 - t) + b t` with `b t` rounded first and the sum rounded, both at `pr`; IEEE's
/// infinities and NaNs (a NaN or an infinity times a weight of 0 is a NaN).
pub(super) fn mix(
    f: &mut FunctionBuilder<'_>,
    k: &K,
    a: Soft,
    b: Soft,
    w: Value,
    pr: Prec,
) -> BResult<Soft> {
    let zero = c32(f, 0)?;
    let all = c32(f, 2048)?;
    let s = bin(f, BinOp::ISub, all, w)?;
    let no = falsy(f)?;
    // b t, rounded.
    let w64 = f.convert(ConvOp::UConvert, w, k.u64_)?;
    let bm = bin(f, BinOp::IMul, b.m, w64)?;
    let be = bink(f, BinOp::ISub, b.e, 11)?;
    let bt = round(f, k, b.neg, bm, be, no, pr)?;
    // a (1 - t), exact.
    let s64 = f.convert(ConvOp::UConvert, s, k.u64_)?;
    let am = bin(f, BinOp::IMul, a.m, s64)?;
    let ae = bink(f, BinOp::ISub, a.e, 11)?;
    let (neg, mag, exp, sticky) = add(f, k, (a.neg, am, ae), (bt.neg, bt.m, bt.e))?;
    let r = round(f, k, neg, mag, exp, sticky, pr)?;
    // Infinities and NaNs.
    let a_dead = bin(f, BinOp::IEqual, s, zero)?;
    let b_dead = bin(f, BinOp::IEqual, w, zero)?;
    let a_zero_inf = bin(f, BinOp::LogicalAnd, a.inf, a_dead)?;
    let b_zero_inf = bin(f, BinOp::LogicalAnd, b.inf, b_dead)?;
    let both = bin(f, BinOp::LogicalAnd, a.inf, b.inf)?;
    let opposed = bin(f, BinOp::LogicalNotEqual, a.neg, b.neg)?;
    let clash = bin(f, BinOp::LogicalAnd, both, opposed)?;
    let nan = bin(f, BinOp::LogicalOr, a.nan, b.nan)?;
    let nan = bin(f, BinOp::LogicalOr, nan, a_zero_inf)?;
    let nan = bin(f, BinOp::LogicalOr, nan, b_zero_inf)?;
    let nan = bin(f, BinOp::LogicalOr, nan, clash)?;
    let any_inf = bin(f, BinOp::LogicalOr, a.inf, b.inf)?;
    let finite_nan = f.unary(UnOp::LogicalNot, nan)?;
    let inf = bin(f, BinOp::LogicalAnd, any_inf, finite_nan)?;
    let inf_neg = select(f, a.inf, a.neg, b.neg)?;
    let neg = select(f, inf, inf_neg, r.neg)?;
    Ok(Soft {
        neg,
        m: r.m,
        e: r.e,
        inf,
        nan,
    })
}

/// `v den` rounded to the nearest integer, ties to even, as an `i32` (magnitude at most
/// `2^30`): `v` finite, of at most 24 significant bits, `den` below `2^17`.
pub(super) fn times_rne(f: &mut FunctionBuilder<'_>, k: &K, v: Soft, den: Value) -> BResult<Value> {
    let zero = c32(f, 0)?;
    let zero64 = c64(f, k, 0)?;
    let one64 = c64(f, k, 1)?;
    let d = f.convert(ConvOp::UConvert, den, k.u64_)?;
    let mag = bin(f, BinOp::IMul, v.m, d)?;
    let (one, twenty_two, cap) = (c32(f, 1)?, c32(f, 22)?, c32(f, 63)?);
    let nonneg = bin(f, BinOp::SGreaterThanEqual, v.e, zero)?;
    let huge = bin(f, BinOp::SGreaterThan, v.e, twenty_two)?;
    let el = f.glsl(GlslOp::UMin, &[v.e, twenty_two])?;
    let left = bin(f, BinOp::ShiftLeftLogical, mag, el)?;
    let s = bin(f, BinOp::ISub, zero, v.e)?;
    let s = f.glsl(GlslOp::UMin, &[s, cap])?;
    let s = f.glsl(GlslOp::UMax, &[s, one])?;
    let kept = bin(f, BinOp::ShiftRightLogical, mag, s)?;
    let bit = bin(f, BinOp::ShiftLeftLogical, one64, s)?;
    let low = bin(f, BinOp::ISub, bit, one64)?;
    let rem = bin(f, BinOp::BitwiseAnd, mag, low)?;
    let s1 = bink(f, BinOp::ISub, s, 1)?;
    let half = bin(f, BinOp::ShiftLeftLogical, one64, s1)?;
    let over = bin(f, BinOp::UGreaterThan, rem, half)?;
    let at = bin(f, BinOp::IEqual, rem, half)?;
    let lsb = bin(f, BinOp::BitwiseAnd, kept, one64)?;
    let odd = bin(f, BinOp::INotEqual, lsb, zero64)?;
    let tie_odd = bin(f, BinOp::LogicalAnd, at, odd)?;
    let up = bin(f, BinOp::LogicalOr, over, tie_odd)?;
    let step = select(f, up, one64, zero64)?;
    let right = bin(f, BinOp::IAdd, kept, step)?;
    let r = select(f, nonneg, left, right)?;
    let most = c64(f, k, 1 << 30)?;
    let r = select(f, huge, most, r)?;
    let r = f.glsl(GlslOp::UMin, &[r, most])?;
    let r = f.convert(ConvOp::UConvert, r, k.u32_)?;
    let minus = bin(f, BinOp::ISub, zero, r)?;
    select(f, v.neg, minus, r)
}

/// `(m, e)` of a value read as at least 0 (a negative value, a zero or a NaN as `m = 0`), for
/// `RGB9E5Float`'s word.
pub(super) fn nonnegative(f: &mut FunctionBuilder<'_>, k: &K, v: Soft) -> BResult<(Value, Value)> {
    let zero64 = c64(f, k, 0)?;
    let gone = bin(f, BinOp::LogicalOr, v.neg, v.nan)?;
    let m = select(f, gone, zero64, v.m)?;
    Ok((m, v.e))
}
