//! The square roots: Metal's precise `sqrt` and `rsqrt` of a `float`, which MSL Table 8.1 states as
//! correctly rounded, as the `half` table (8.3) does for its own.
//!
//! **Normal positive operands.** `x = y 2^(2k)` with `y` in `[1, 4)` (the significand with the
//! exponent's parity), so `sqrt(x) = sqrt(y) 2^k` and `rsqrt(x) = rsqrt(y) 2^-k`, each a normal
//! `float` whose rounding is `y`'s root's, scaled exactly. The device's `Sqrt` of `y`, or its
//! reciprocal by the device's division, is refined by one Newton step on its exact residual (`y - s^2`, or `1 - y r^2` from
//! Dekker's products), which leaves it within a little over half an ulp whatever the device's
//! error; the correctly rounded root is then that value or one of its neighbours, decided by
//! exact integer comparisons with the midpoints between them. With `Y = y 2^23` (an integer below
//! `2^25`) and the candidate `S = s 2^23`, `sqrt(y)` is above the midpoint `(2S + 1) 2^-24` when
//! `Y 2^25 > (2S + 1)^2`; with `R = r 2^24`, `rsqrt(y)` is above `(2R + 1) 2^-25` when `2^73 >
//! (2R + 1)^2 Y`. Both sides are integers and never equal, so there is no tie; the products are
//! two and three words of `OpUMulExtended`.
//!
//! **Every other operand** (a zero, a denormal, a negative value, an infinity, or a NaN) takes the
//! device's `Sqrt`, and `rsqrt` its reciprocal by the device's division: `sqrt(±0) = ±0`,
//! `rsqrt(±0) = ±inf`, `sqrt(+inf) = +inf`, `rsqrt(+inf) = +0`, a NaN for a negative operand or a
//! NaN, and a denormal read as the device reads it. How Metal reads a denormal operand of `sqrt` and `rsqrt` is the denormal model's
//! ([`denormal::operand`]), which hands the routine its operand.

use crate::lower::arith::libm::atan::read;
use crate::lower::arith::libm::{Fun, E};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, Contraction, GlslOp};

#[cfg(doc)]
use crate::lower::arith::denormal;

/// Which root a routine answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Root {
    Sqrt,
    Rsqrt,
}

/// Whether the 64-bit value `(alo, ahi)` is above `(blo, bhi)` (words low first).
fn above(
    e: &mut E<'_, '_>,
    (alo, ahi): (Value, Value),
    (blo, bhi): (Value, Value),
) -> BResult<Value> {
    let hi_above = e.iop(BinOp::UGreaterThan, ahi, bhi)?;
    let hi_equal = e.iop(BinOp::IEqual, ahi, bhi)?;
    let lo_above = e.iop(BinOp::UGreaterThan, alo, blo)?;
    let tie_above = e.iop(BinOp::LogicalAnd, hi_equal, lo_above)?;
    e.iop(BinOp::LogicalOr, hi_above, tie_above)
}

/// The correctly rounded `S` from a candidate within one unit of it: `S + 1` where `up(2S + 1)`
/// holds, `S - 1` where `down(2S - 1)` does, `S` otherwise.
fn nearest(
    e: &mut E<'_, '_>,
    s: Value,
    up: &dyn Fn(&mut E<'_, '_>, Value) -> BResult<Value>,
    down: &dyn Fn(&mut E<'_, '_>, Value) -> BResult<Value>,
) -> BResult<Value> {
    let one = e.u(1)?;
    let twice = e.iop(BinOp::ShiftLeftLogical, s, one)?;
    let odd_up = e.iop(BinOp::IAdd, twice, one)?;
    let odd_down = e.iop(BinOp::ISub, twice, one)?;
    let is_up = up(e, odd_up)?;
    let is_down = down(e, odd_down)?;
    let plus = e.iop(BinOp::IAdd, s, one)?;
    let minus = e.iop(BinOp::ISub, s, one)?;
    let r = e.sel(is_down, minus, s)?;
    e.sel(is_up, plus, r)
}

/// Metal's precise `which` of `x`.
pub(crate) fn root(e: &mut E<'_, '_>, which: Root, x: Value) -> BResult<Value> {
    let fun = match which {
        Root::Sqrt => Fun::Sqrt,
        Root::Rsqrt => Fun::Rsqrt,
    };
    let x = read(e, fun, x)?;
    let bits = e.bits(x)?;
    // A positive normal operand: bits from 0x0080_0000 below 0x7F80_0000.
    let lead = e.u(0x0080_0000)?;
    let offset = e.iop(BinOp::ISub, bits, lead)?;
    let span = e.u(0x7F00_0000)?;
    let normal = e.iop(BinOp::ULessThan, offset, span)?;
    // The biased exponent `B`, the parity `p` of `B - 127`, and `y`'s bits: the fraction under the
    // biased exponent `127 + p`.
    let (k23, one) = (e.u(23)?, e.u(1)?);
    let biased = e.iop(BinOp::ShiftRightLogical, bits, k23)?;
    let b1 = e.iop(BinOp::IAdd, biased, one)?;
    let p = e.iop(BinOp::BitwiseAnd, b1, one)?;
    let (fraction, k127) = (e.u(0x007F_FFFF)?, e.u(127)?);
    let f = e.iop(BinOp::BitwiseAnd, bits, fraction)?;
    let yb = e.iop(BinOp::IAdd, k127, p)?;
    let yb = e.iop(BinOp::ShiftLeftLogical, yb, k23)?;
    let yb = e.iop(BinOp::BitwiseOr, f, yb)?;
    let y = e.float(yb)?;
    // `Y = y 2^23`, the significand shifted by the parity.
    let m = e.iop(BinOp::BitwiseOr, f, lead)?;
    let big_y = e.iop(BinOp::ShiftLeftLogical, m, p)?;
    // `k + 127 = (B + 127 - p) / 2`.
    let half_k = e.iop(BinOp::IAdd, biased, k127)?;
    let half_k = e.iop(BinOp::ISub, half_k, p)?;
    let k_biased = e.iop(BinOp::ShiftRightLogical, half_k, one)?;
    let r = match which {
        Root::Sqrt => {
            // `s1 = s + (y - s^2) / (2s)`, the residual exact.
            let s = e.device_sqrt(y)?;
            let d = e.remainder(y, s, s)?;
            let two_s = e.add(s, s)?;
            let c = e.device_div(d, two_s)?;
            let s1 = e.add(s, c)?;
            let scale = e.c(8_388_608.0)?;
            let s1 = e.mul(s1, scale)?;
            let s1 = e.utrunc(s1)?;
            // `Y 2^25` as two words, compared with `(2S ± 1)^2`.
            let (k25, k7) = (e.u(25)?, e.u(7)?);
            let ylo = e.iop(BinOp::ShiftLeftLogical, big_y, k25)?;
            let yhi = e.iop(BinOp::ShiftRightLogical, big_y, k7)?;
            let s = nearest(
                e,
                s1,
                &|e, odd| {
                    let sq = e.umul(odd, odd)?;
                    above(e, (ylo, yhi), sq)
                },
                &|e, odd| {
                    let sq = e.umul(odd, odd)?;
                    above(e, sq, (ylo, yhi))
                },
            )?;
            // `S 2^(k - 23)`: an exact product, `S` at most `2^24`.
            let sf = e.ufloat(s)?;
            let shift = e.iop(BinOp::ISub, k_biased, k23)?;
            let scale = e.iop(BinOp::ShiftLeftLogical, shift, k23)?;
            let scale = e.float(scale)?;
            e.mul(sf, scale)?
        }
        Root::Rsqrt => {
            // `r1 = r + r h / 2` with `h = 1 - y r^2`, from the exact products `r^2 = th + tl`
            // and `y th = uh + ul`; `r` is the device's `1 / sqrt(y)`.
            let s = e.device_sqrt(y)?;
            let one_f = e.c(1.0)?;
            let r = e.device_div(one_f, s)?;
            let th = e.mul(r, r)?;
            let tl = e.product_error(r, r, th)?;
            let uh = e.mul(y, th)?;
            let ul = e.product_error(y, th, uh)?;
            let v = e.mul(y, tl)?;
            let half = e.c(0.5)?;
            let h = e.sub(one_f, uh)?;
            let h = e.sub(h, ul)?;
            let h = e.sub(h, v)?;
            let hh = e.mul(h, half)?;
            let c = e.mul(r, hh)?;
            let r1 = e.add(r, c)?;
            let scale = e.c(16_777_216.0)?;
            let r1 = e.mul(r1, scale)?;
            let r1 = e.utrunc(r1)?;
            // The third word of `(2R ± 1)^2 Y` against that of `2^73`, 512.
            let k512 = e.u(512)?;
            let top = |e: &mut E<'_, '_>, odd: Value| -> BResult<Value> {
                let (plo, phi) = e.umul(odd, odd)?;
                let (_, a1) = e.umul(plo, big_y)?;
                let (b0, b1) = e.umul(phi, big_y)?;
                let (_, carry) = e.uadd(a1, b0)?;
                e.iop(BinOp::IAdd, b1, carry)
            };
            let r = nearest(
                e,
                r1,
                &|e, odd| {
                    let w = top(e, odd)?;
                    e.iop(BinOp::ULessThan, w, k512)
                },
                &|e, odd| {
                    let w = top(e, odd)?;
                    e.iop(BinOp::UGreaterThanEqual, w, k512)
                },
            )?;
            // `R 2^(-k - 24)`, the exponent's bias `254 - (k + 127) - 24`.
            let rf = e.ufloat(r)?;
            let k230 = e.u(230)?;
            let shift = e.iop(BinOp::ISub, k230, k_biased)?;
            let scale = e.iop(BinOp::ShiftLeftLogical, shift, k23)?;
            let scale = e.float(scale)?;
            e.mul(rf, scale)?
        }
    };
    // The device's `Sqrt` of another operand, and its reciprocal by the device's division.
    let device = e.fb.glsl(GlslOp::Sqrt, &[x])?;
    let device = match which {
        Root::Sqrt => device,
        Root::Rsqrt => {
            let one_f = e.c(1.0)?;
            e.fb.binary_with(BinOp::FDiv, one_f, device, Contraction::Forbidden)?
        }
    };
    let r = e.sel(normal, r, device)?;
    // A negative normal or infinite operand: Metal's NaN (a negative denormal is read as the
    // device reads it).
    let least = e.c(-f32::MIN_POSITIVE)?;
    let domain = e.le(x, least)?;
    let nan = e.metal_nan(None)?;
    e.sel(domain, nan, r)
}
