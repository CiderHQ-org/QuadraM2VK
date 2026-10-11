//! Unevaluated sums of two `float`s (`hi + lo`, `|lo|` at most about half an ulp of `hi`), the
//! arithmetic the second stage's routines carry their intermediate values in: each operation is
//! exact or rounds its low part only, so a value keeps about 44 significant bits through the
//! reductions, and only the routine's last addition rounds to the result. The device's division and
//! square root are corrected by their exact remainders, as stage 1's routines do, so their error
//! does not reach the low part. A product's exact error is [`E::exact_error`]'s: one `OpFmaKHR`
//! where the profile has it, Dekker's product elsewhere, the same normal `float` for every product
//! whose error an answer depends on (each lies above `2^-100`), so a routine answers the same with
//! and without it.

use crate::lower::arith::libm::E;
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::Value;
use crate::spirv::ir::BinOp;

/// A `float` constant as `hi + lo`, `hi` the `float` nearest `v` and `lo` the `float` nearest the
/// rest.
pub(crate) fn constant(v: f64) -> (f32, f32) {
    super::split(v)
}

impl E<'_, '_> {
    /// `a + b` as the exact two-term sum `(s, e)`, for any `a` and `b` whose sum does not overflow
    /// (Knuth's TwoSum).
    pub fn two_sum(&mut self, a: Value, b: Value) -> BResult<(Value, Value)> {
        let s = self.add(a, b)?;
        let bb = self.sub(s, a)?;
        let aa = self.sub(s, bb)?;
        let da = self.sub(a, aa)?;
        let db = self.sub(b, bb)?;
        let err = self.add(da, db)?;
        Ok((s, err))
    }

    /// `a * b` as the exact two-term product `(p, e)` ([`E::exact_error`]'s range).
    pub fn two_prod(&mut self, a: Value, b: Value) -> BResult<(Value, Value)> {
        let p = self.mul(a, b)?;
        let err = self.exact_error(a, b, p)?;
        Ok((p, err))
    }

    /// `n - q d` exactly, for a quotient or root `q` of `n` by `d` correct to a few ulps
    /// ([`E::exact_error`]'s range): `n - p` is exact by Sterbenz's lemma.
    fn residual(&mut self, n: Value, q: Value, d: Value) -> BResult<Value> {
        let p = self.mul(q, d)?;
        let err = self.exact_error(q, d, p)?;
        let r = self.sub(n, p)?;
        self.sub(r, err)
    }

    /// `(nh + nl) / (dh + dl)` as a pair: the device's quotient `q` of the high parts and the
    /// correction `(nh - q dh + nl - q dl) / dh`, its first remainder exact.
    pub fn div_pair(
        &mut self,
        (nh, nl): (Value, Value),
        (dh, dl): (Value, Value),
    ) -> BResult<(Value, Value)> {
        let q = self.div(nh, dh)?;
        let r = self.residual(nh, q, dh)?;
        let qd = self.mul(q, dl)?;
        let rest = self.sub(nl, qd)?;
        let r = self.add(r, rest)?;
        let lo = self.div(r, dh)?;
        Ok((q, lo))
    }

    /// `s + err` rounded to odd, for a sum `s` and its exact residual `err` (`s = RN(s + err)`, as
    /// [`E::two_sum`] and [`E::two_prod`] give them): `s` where `err` is zero, else whichever of `s`
    /// and its neighbour toward zero has its last bit set, by its bits (Boldo and Melquiond's
    /// rounding to odd). A zero `s` stays: its residual is zero but where an operand underflowed.
    pub fn odd(&mut self, s: Value, err: Value) -> BResult<Value> {
        let (sb, eb) = (self.bits(s)?, self.bits(err)?);
        let (magnitude, zero) = (self.u(0x7FFF_FFFF)?, self.u(0)?);
        let ea = self.iop(BinOp::BitwiseAnd, eb, magnitude)?;
        let inexact = self.iop(BinOp::INotEqual, ea, zero)?;
        let sa = self.iop(BinOp::BitwiseAnd, sb, magnitude)?;
        let nonzero = self.iop(BinOp::INotEqual, sa, zero)?;
        let adjust = self.iop(BinOp::LogicalAnd, inexact, nonzero)?;
        // Toward zero by one unit where the residual's sign is not `s`'s.
        let k31 = self.u(31)?;
        let other = self.iop(BinOp::BitwiseXor, sb, eb)?;
        let other = self.iop(BinOp::ShiftRightLogical, other, k31)?;
        let down = self.iop(BinOp::ISub, sb, other)?;
        let last = self.u(1)?;
        let o = self.iop(BinOp::BitwiseOr, down, last)?;
        let o = self.sel(adjust, o, sb)?;
        self.float(o)
    }

    /// `a b + c` rounded once, as a fused multiply-add rounds it: one `OpFmaKHR` where the profile
    /// has it ([`E::exact_error`]'s route), else a call of the module's [`super::Fun::Fused`]
    /// routine ([`E::fused_inline`]), so a routine's many fused steps share one emulation (an
    /// error where the module has none). Exact where `a b` lies above `2^-100` and nothing
    /// overflows.
    pub fn fused(&mut self, a: Value, b: Value, c: Value) -> BResult<Value> {
        if self.fma {
            return self.fb.fma(a, b, c);
        }
        let f = self
            .helpers
            .fused
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
        self.fb
            .call(f, &[a, b, c])?
            .ok_or(BuildError::Type(TypeRule::FunctionSignature))
    }

    /// `a b + c` rounded once by Boldo and Melquiond's emulation, [`super::Fun::Fused`]'s body: `a
    /// b = uh + ul` exactly, `c + uh = th + tl` exactly, and `RN(th + RO(tl + ul))`. Exact where
    /// `a b` lies above `2^-100` ([`E::exact_error`]'s range) and nothing overflows.
    pub fn fused_inline(&mut self, a: Value, b: Value, c: Value) -> BResult<Value> {
        let (uh, ul) = self.two_prod(a, b)?;
        let (th, tl) = self.two_sum(c, uh)?;
        let (v, v_err) = self.two_sum(tl, ul)?;
        let v = self.odd(v, v_err)?;
        self.add(th, v)
    }

    /// Whether `x` is a NaN, by its bits (no float mode folds an integer comparison).
    pub fn is_nan(&mut self, x: Value) -> BResult<Value> {
        let b = self.bits(x)?;
        let (m, inf) = (self.u(0x7FFF_FFFF)?, self.u(0x7F80_0000)?);
        let a = self.iop(BinOp::BitwiseAnd, b, m)?;
        self.iop(BinOp::UGreaterThan, a, inf)
    }

    /// `v` clamped to `[-bound, bound]` (a NaN stays a NaN), and whether it was clamped.
    pub fn clamp(&mut self, v: Value, bound: f32) -> BResult<(Value, Value)> {
        let (hi, lo) = (self.c(bound)?, self.c(-bound)?);
        let over = self.gt(v, hi)?;
        let under = self.lt(v, lo)?;
        let r = self.sel(over, hi, v)?;
        let r = self.sel(under, lo, r)?;
        let clamped = self.iop(BinOp::LogicalOr, over, under)?;
        Ok((r, clamped))
    }
}
