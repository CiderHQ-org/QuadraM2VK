//! The trigonometric functions' other routines: Metal's fast `sin` and `cos`, and Metal's `half`
//! `tan`. Its precise `sin` and `cos` are Metal's own form (`libm::sincos`), its precise `tan` too
//! (`libm::tan`), and its `sinpi`, `cospi` and `tanpi` (`libm::pi`).
//!
//! **Metal's fast `sin` and `cos`** below their flush point (`math::fast_trig_flush` answers
//! past it) are its precise functions: they agree on 864 of the 865 such lanes of cases 087758cb
//! and cc234732 (the one apart, `fast::sin(-2^-12)`, is an ulp below), and on every `float` of
//! `[0.25, 1)`, within 2.2 ulp of the exact value inside and outside `[-π, π]`. Inside `[-π, π]`,
//! where Apple bounds them by `2^-13` absolute (MSL Table 8.2) and every device measured meets it,
//! the device's `Sin` or `Cos` answers; past `π`, where the table states no bound and the devices
//! are up to 1.8 off (ANV; RADV 0.21, NVIDIA 0.56), the precise routine does, per lane under a
//! branch, so a wave whose lanes all lie inside `[-π, π]` runs the device's instruction alone. A
//! denormal operand joins the branch too: Metal keeps it (`fast::sin(x) = x`, `fast::cos(x) = 1`,
//! case cc234732) where RADV, ANV, and NVIDIA flush it, and the routine's selects answer it
//! ([`denormal::operand`]).

use crate::lower::arith::libm::atan::read;
use crate::lower::arith::libm::{quarter, sincos, Fun, E};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, GlslOp, UnOp};

#[cfg(doc)]
use crate::lower::arith::denormal;

/// Which function a routine answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Trig {
    Sin,
    Cos,
}

/// Metal's fast `which` (`Sin` or `Cos`) of `x` below its flush point: the device's instruction
/// where `|x| < float(π)` (or `x` is NaN, whose answer the flush selects), the precise routine
/// from `float(π)` up, which is past `π`, and for a denormal `x`, which Metal keeps
/// ([`denormal::operand`]) where a device that flushes it answers its zero, under a branch on the
/// lane's `|x|`. So the device answers only inside `[-π, π]`, where Apple bounds its answer
/// (MSL Table 8.2) and the user accepts it; at `±float(π)` it was outside that bound on RADV, ANV
/// and the RTX 5080, where the routine is Metal's word. The denormal is told by its
/// bits, so a device that flushes in comparisons tells it too.
pub(crate) fn fast(e: &mut E<'_, '_>, which: Trig, x: Value) -> BResult<Value> {
    let (op, fun) = match which {
        Trig::Sin => (GlslOp::Sin, Fun::FastSin),
        Trig::Cos => (GlslOp::Cos, Fun::FastCos),
    };
    let x = read(e, fun, x)?;
    let ax = e.abs(x)?;
    let pi = e.c(std::f32::consts::PI)?;
    let past = e.ge(ax, pi)?;
    let bits = e.bits(ax)?;
    let one = e.u(1)?;
    let below = e.iop(BinOp::ISub, bits, one)?;
    let largest = e.u(0x007F_FFFF)?;
    let denormal = e.iop(BinOp::ULessThan, below, largest)?;
    let past = e.iop(BinOp::LogicalOr, past, denormal)?;
    let (rough, fma, helpers) = (e.rough, e.fma, e.helpers);
    let routine: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let mut e = E {
            fb,
            rough,
            fma,
            helpers,
        };
        let v = sincos::sin_cos(&mut e, which, x)?;
        Ok(ArmEnd::FallThrough(vec![v]))
    });
    let inside: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let device = fb.glsl(op, &[x])?;
        Ok(ArmEnd::FallThrough(vec![device]))
    });
    let merged = e.fb.if_(&mut (), past, routine, Some(inside))?;
    merged
        .0
        .first()
        .copied()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))
}

/// `P(s)` of Metal's `half` `tan(x) = r P(r r)` on the reduced argument `r` in quarter turns,
/// its `f32` coefficients by their bits, lowest first (the values Metal's form takes). Horner's
/// rule in `s` (rounded), each step one fused multiply-add.
pub(crate) const HALF_TAN_P: [u32; 4] = [0x3FC9_0D3B, 0x3FA6_963B, 0x3F8C_CDD0, 0x400F_C280];

/// Metal's `half` `tan(x)` before its rounding to `half`, of a `half` `x` held as `float`: its own
/// form, which follows Apple's runtime library. The quarter turns `k` are the low bits of `s =
/// fma(x, c0, m)`, `m = 1.5 2^23`, and `r = fma(x, c1, fma(x, c0, m - s))` of `2/π`'s first two
/// parts ([`quarter::QUARTER`]), `t = r P(r r)` of [`HALF_TAN_P`], and `t` for an even `k`, Metal's
/// reciprocal of `-t` ([`E::recip`]) for an odd one. A zero gives `+0`, and an infinity or a NaN
/// Metal's NaN. Rounded to `half`, Metal's word at every `half`.
pub(crate) fn half_tan(e: &mut E<'_, '_>, x: Value) -> BResult<Value> {
    let ax = e.abs(x)?;
    let bits = e.bits(ax)?;
    let inf = e.u(0x7F80_0000)?;
    let finite = e.iop(BinOp::ULessThan, bits, inf)?;
    let zero = e.c(0.0)?;
    let xr = e.sel(finite, x, zero)?;
    let [c0, c1, _] = quarter::QUARTER.map(f32::from_bits);
    let (c0, c1, m) = (e.c(c0)?, e.c(c1)?, e.c(12_582_912.0)?);
    let s = e.fused(xr, c0, m)?;
    let k = e.bits(s)?;
    let minus_k = e.sub(m, s)?;
    let r = e.fused(xr, c0, minus_k)?;
    let r = e.fused(xr, c1, r)?;
    let r2 = e.mul(r, r)?;
    let p = e.fused_poly(&HALF_TAN_P, r2)?;
    let t = e.mul(r, p)?;
    let minus_t = e.fb.unary(UnOp::FNegate, t)?;
    let cot = e.recip(minus_t)?;
    let (one_u, zero_u) = (e.u(1)?, e.u(0)?);
    let odd = e.iop(BinOp::BitwiseAnd, k, one_u)?;
    let odd = e.iop(BinOp::INotEqual, odd, zero_u)?;
    let v = e.sel(odd, cot, t)?;
    let nan = e.metal_nan(None)?;
    e.sel(finite, v, nan)
}
