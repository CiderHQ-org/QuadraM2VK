//! Metal's `sinpi`, `cospi` and `tanpi` of a `float`: Metal's own forms, which follow Apple's
//! runtime library, of half turns and its hardware's `sinc` (`sincos`), the same in both math
//! modes. With the hardware's measured `sinc` and reciprocal they are Metal's word at every
//! measured operand; through the models of both, at 99.5 % to 99.99 %.
//!
//! **Half turns.** `2x = 2n + t` with `n` the integer nearest `x` (ties to even) and `t` in
//! `[-1, 1]`, so `sin(π x) = ±sin(π t / 2)`: `s = RN(2x + b)` of `b = ±2^24` with `x`'s sign,
//! whose last bit is `n`'s, and `t = 2x + (b - s)`. Every step but `s` is exact (`2x` is a scaling,
//! `b - s` a difference of two floats within a factor of two, and `t` a multiple of `2x`'s unit
//! below 1), so the steps Metal fuses are plain sums here. From `2^23` in magnitude `2x + b` is
//! no longer within a unit of `s`, and `cospi` and `tanpi` take `b = 0` (`t = 0`, `s = 2x`).
//!
//! **The functions**, of `S` the hardware's `sinc` (`sin(π t / 2) = t S(t)`):
//! - `sinpi(x) = t S(t)`, `t` negated for an odd `n`; `+0` from `2^23`.
//! - `cospi(x) = u S(u)` of `u = RN(1 - |t|)`, negated for an odd `n`; 1 from `2^24`.
//! - `tanpi(x) = RN(a rcp(c'))` of `a = RN(t S(t))`, `c = S(RN(1 - |t|))` and
//!   `c' = RN(c - c |t|)` in one fused step, by Metal's reciprocal ([`super::Fun::Recip`]); `+0`
//!   from `2^24`. At a half-integer `c'` is `+0` and the reciprocal `+inf`.
//!
//! Metal reads a denormal `x` as the zero of its sign (`sinpi` and `tanpi` `+0`, `cospi` 1), and
//! answers its NaN, `0x7fc00000`, for an infinity or a NaN.

use crate::lower::arith::libm::atan::read;
use crate::lower::arith::libm::sincos::sinc;
use crate::lower::arith::libm::{Fun, E};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, UnOp};

/// Which of Metal's `π`-scaled functions [`pi`] computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pi {
    Sin,
    Cos,
    Tan,
}

/// The `float` bits of `2^23`, from which `2x + 2^24` is not within a unit of its rounding.
const HALF_TURNS_EXACT_BELOW: u32 = 0x4B00_0000;

/// The `float` bits of `2^24`, from which `cospi` is 1 and `tanpi` `+0`.
const WHOLE: u32 = 0x4B80_0000;

/// Metal's `which` of `x` (the module's doc), the routine of `fun`, whose operand table it reads
/// `x` by.
pub(crate) fn pi(e: &mut E<'_, '_>, fun: Fun, which: Pi, x: Value) -> BResult<Value> {
    let x = read(e, fun, x)?;
    let ax = e.abs(x)?;
    let bits = e.bits(ax)?;
    // From `2^24` (and for an infinity or a NaN) every function's answer is selected below, so
    // the steps take `+0` instead, where `2x` would overflow or `t` be NaN.
    let whole = e.u(WHOLE)?;
    let past = e.iop(BinOp::UGreaterThanEqual, bits, whole)?;
    let zero = e.c(0.0)?;
    let x = e.sel(past, zero, x)?;
    let big = e.u(HALF_TURNS_EXACT_BELOW)?;
    let large = e.iop(BinOp::UGreaterThanEqual, bits, big)?;
    let unit = e.c(16_777_216.0)?;
    let b = e.copysign(unit, x)?;
    let b = match which {
        // From `2^23` `sinpi` is `+0` whatever the steps give.
        Pi::Sin => b,
        Pi::Cos | Pi::Tan => e.sel(large, zero, b)?,
    };
    let two = e.c(2.0)?;
    let x2 = e.mul(x, two)?;
    let s = e.add(x2, b)?;
    let d = e.sub(b, s)?;
    let t = e.add(x2, d)?;
    let (one_u, zero_u) = (e.u(1)?, e.u(0)?);
    let sb = e.bits(s)?;
    let odd = e.iop(BinOp::BitwiseAnd, sb, one_u)?;
    let odd = e.iop(BinOp::INotEqual, odd, zero_u)?;
    let at = e.abs(t)?;
    let one = e.c(1.0)?;
    let rest = e.sub(one, at)?;
    let negated = |e: &mut E<'_, '_>, v: Value| -> BResult<Value> {
        let minus = e.fb.unary(UnOp::FNegate, v)?;
        e.sel(odd, minus, v)
    };
    let v = match which {
        Pi::Sin => {
            let t = negated(e, t)?;
            let st = sinc(e, at)?;
            let v = e.mul(t, st)?;
            e.sel(large, zero, v)?
        }
        Pi::Cos => {
            let u = negated(e, rest)?;
            let su = sinc(e, rest)?;
            let v = e.mul(u, su)?;
            e.sel(past, one, v)?
        }
        Pi::Tan => {
            let st = sinc(e, at)?;
            let a = e.mul(t, st)?;
            let c = sinc(e, rest)?;
            let minus_at = e.fb.unary(UnOp::FNegate, at)?;
            let den = e.fused(c, minus_at, c)?;
            let q = e.recip(den)?;
            let v = e.mul(a, q)?;
            e.sel(past, zero, v)?
        }
    };
    let inf = e.u(0x7F80_0000)?;
    let finite = e.iop(BinOp::ULessThan, bits, inf)?;
    let nan = e.metal_nan(None)?;
    e.sel(finite, v, nan)
}
