//! Metal's precise `tan` of a `float`: its own form, which follows Apple's runtime library and
//! matches Metal's measured words (the M3 Max: every `float` of `[0.25, 4)`, every
//! 16th from `2^-11`, every 256th from 4, and every 4096th from `2^21`).
//!
//! **Reduction** to quarter turns, `r = x 2/π - k`, by Metal's own reduction
//! ([`quarter::turns`], with the rounding error of its large operands' difference added).
//!
//! **Quotient.** With `s = r^2`, `N = r P(s)` ([`TAN_P`]) and `D = Q(s)` ([`TAN_Q`]), each by
//! Horner's rule in fused steps from the highest coefficient. `tan(x)` is `N / D` for an even
//! `k` and `-D / N` for an odd one, correctly rounded: Metal's reciprocal with two fused
//! corrections gives the correctly rounded quotient at every measured operand, so the module's
//! precise division routine ([`E::div`]) answers it on every device.
//!
//! **Special operands**: below `2^-11` in magnitude (a zero and a denormal included) `x` itself;
//! an infinity or a NaN, Metal's NaN.

use crate::lower::arith::libm::atan::read;
use crate::lower::arith::libm::quarter::{self, Error};
use crate::lower::arith::libm::{Fun, E};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, UnOp};

/// `P(s)` of `N = r P(s)` (`f32` bits, lowest first).
pub(crate) const TAN_P: [u32; 5] = [
    0x3FC9_0FDB,
    0xBF25_5DE7,
    0x3DA3_35DD,
    0xBB99_67FB,
    0x3927_F34F,
];

/// `Q(s)`, the denominator `D` (`f32` bits, lowest first).
pub(crate) const TAN_Q: [u32; 6] = [
    0x3F80_0000,
    0xBF9D_E9E6,
    0x3E81_E0F7,
    0xBCAA_E925,
    0x3A70_B770,
    0xB7C9_C073,
];

/// The `float` bits below which `tan(x)` is `x` (`2^-11`).
const TINY: u32 = 0x3A00_0000;

/// Metal's precise `tan(x)` (the module's doc).
pub(crate) fn tan(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let x = read(e, Fun::Tan, x)?;
    let (r, k) = quarter::turns(e, x, Error::Compensated)?;
    // `N / D` or `-D / N`.
    let s = e.mul(r, r)?;
    let pn = e.fused_poly(&TAN_P, s)?;
    let n = e.mul(r, pn)?;
    let d = e.fused_poly(&TAN_Q, s)?;
    let (one_u, zero_u) = (e.u(1)?, e.u(0)?);
    let odd = e.iop(BinOp::BitwiseAnd, k, one_u)?;
    let odd = e.iop(BinOp::INotEqual, odd, zero_u)?;
    let minus_d = e.fb.unary(UnOp::FNegate, d)?;
    let num = e.sel(odd, minus_d, n)?;
    let den = e.sel(odd, n, d)?;
    let q = e.div(num, den)?;
    // Below 2^-11: `x`; an infinity or a NaN: Metal's NaN.
    let ax = e.abs(x)?;
    let bits = e.bits(ax)?;
    let tiny = e.u(TINY)?;
    let is_tiny = e.iop(BinOp::ULessThan, bits, tiny)?;
    let q = e.sel(is_tiny, x, q)?;
    let inf = e.u(0x7F80_0000)?;
    let finite = e.iop(BinOp::ULessThan, bits, inf)?;
    let nan = e.metal_nan(None)?;
    e.sel(finite, q, nan)
}
