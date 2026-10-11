//! The remainder: Metal's precise `fmod(x, y)` of `float` operands, `x - n y` with `n` the quotient
//! truncated toward zero, which MSL Table 8.1 states exact (0 ulp), as the `half` table (8.3) does
//! for its own.
//!
//! **Generic operands.** With `|x| = Mx 2^(Ex - 150)` and `|y| = My 2^(Ey - 150)` (`M` the 24-bit
//! significand, a denormal's without its leading bit and at `E = 1`), `|x|` is at least `|y|`
//! where computed (a denormal `x` only: `y` is normal there), and the remainder is `(Mx 2^(Ex - Ey) mod My) 2^(Ey - 150)`: the integer
//! `Mx 2^(Ex - Ey)` is reduced modulo `My` eight bits at a time (`(r << 8) mod My` stays below
//! `2^32`), first the gap's low three bits with `Mx` itself, then a loop of one step per remaining
//! eight, at most 31. The remainder, below `My`, is an exact `float` (a normal one by its
//! exponent field, a denormal one by its bits), with `x`'s sign: so a zero remainder is the zero of
//! `x`'s sign, and the device's denormal mode does not touch it.
//!
//! **Special operands** (as C99 and Metal answer, case 087758cb, and the half reading for `half`):
//! `|x|` below `|y|` gives `x` itself, which covers a zero `x` and an infinite `y` (`fmod(x,
//! ±inf) = x`); an infinite or NaN `x`, a NaN `y`, or a zero `y` gives a NaN of the device's
//! arithmetic.
//!
//! **Denormal operands**, as Metal reads them (case 087758cb): a denormal dividend as itself
//! (`fmod(2^-149, -1) = 2^-149`, `fmod(2^-127, 8) = 2^-127`) and a denormal divisor as a zero
//! (`fmod(2, 2^-149)` and `fmod(-6588397.5, 2^-149)` are NaN), decided on the divisor's exponent
//! field, so the answer is Metal's whatever the device's denormal mode. The reading is per
//! operand, so it lives here rather than in the per-operation table ([`denormal::operand`], which
//! has no `fmod` row).

use crate::lower::arith::libm::E;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, MemAccess, Type, UnOp};

#[cfg(doc)]
use crate::lower::arith::denormal;

/// The significand `M` and the biased exponent `E` (at least 1) of the magnitude bits `a` of a
/// finite `float`.
fn parts(e: &mut E<'_, '_>, a: Value) -> BResult<(Value, Value)> {
    let (k23, one, zero) = (e.u(23)?, e.u(1)?, e.u(0)?);
    let biased = e.iop(BinOp::ShiftRightLogical, a, k23)?;
    let (fraction, lead) = (e.u(0x007F_FFFF)?, e.u(0x0080_0000)?);
    let f = e.iop(BinOp::BitwiseAnd, a, fraction)?;
    let denormal = e.iop(BinOp::IEqual, biased, zero)?;
    let implicit = e.sel(denormal, zero, lead)?;
    let m = e.iop(BinOp::BitwiseOr, f, implicit)?;
    let biased = e.sel(denormal, one, biased)?;
    Ok((m, biased))
}

/// Metal's precise `fmod(x, y)`.
pub(crate) fn fmod(e: &mut E<'_, '_>, x: Value, y: Value) -> BResult<Value> {
    let (bx, by) = (e.bits(x)?, e.bits(y)?);
    let (magnitude, sign_bit) = (e.u(0x7FFF_FFFF)?, e.u(0x8000_0000)?);
    let ax = e.iop(BinOp::BitwiseAnd, bx, magnitude)?;
    let ay = e.iop(BinOp::BitwiseAnd, by, magnitude)?;
    let sign = e.iop(BinOp::BitwiseAnd, bx, sign_bit)?;
    // The NaN lanes, `x` itself, and the lanes the remainder is computed for.
    let (inf, zero) = (e.u(0x7F80_0000)?, e.u(0)?);
    let x_nonfinite = e.iop(BinOp::UGreaterThanEqual, ax, inf)?;
    let y_nan = e.iop(BinOp::UGreaterThan, ay, inf)?;
    // A zero or denormal `y`, by its exponent field: Metal reads a denormal divisor as a zero.
    let y_exponent = e.iop(BinOp::BitwiseAnd, ay, inf)?;
    let y_zero = e.iop(BinOp::IEqual, y_exponent, zero)?;
    let nan_lane = e.iop(BinOp::LogicalOr, x_nonfinite, y_nan)?;
    let nan_lane = e.iop(BinOp::LogicalOr, nan_lane, y_zero)?;
    let keep_x = e.iop(BinOp::ULessThan, ax, ay)?;
    let special = e.iop(BinOp::LogicalOr, nan_lane, keep_x)?;
    let (mx, ex) = parts(e, ax)?;
    let (my, ey) = parts(e, ay)?;
    // A zero or denormal `y`'s lane divides by one (it answers a NaN).
    let one = e.u(1)?;
    let my = e.sel(y_zero, one, my)?;
    // The gap `Ex - Ey`, zero where nothing is computed: its low three bits shift `Mx`, and the
    // rest counts the loop's eight-bit steps.
    let gap = e.iop(BinOp::ISub, ex, ey)?;
    let gap = e.sel(special, zero, gap)?;
    let (seven, three) = (e.u(7)?, e.u(3)?);
    let head = e.iop(BinOp::BitwiseAnd, gap, seven)?;
    let steps = e.iop(BinOp::ShiftRightLogical, gap, three)?;
    let first = e.iop(BinOp::ShiftLeftLogical, mx, head)?;
    let first = e.iop(BinOp::UMod, first, my)?;
    let r = steps_mod(e, first, my, steps)?;
    // `r 2^(Ey - 150)`: `float(r)` with `Ey - 150` added to its exponent field where the result is
    // normal (`127 + msb(r) + Ey - 150 >= 1`), `r << (Ey - 1)` where it is denormal.
    let rf = e.ufloat(r)?;
    let rb = e.bits(rf)?;
    let k23 = e.u(23)?;
    let rexp = e.iop(BinOp::ShiftRightLogical, rb, k23)?;
    let sum = e.iop(BinOp::IAdd, rexp, ey)?;
    let k151 = e.u(151)?;
    let normal = e.iop(BinOp::UGreaterThanEqual, sum, k151)?;
    let nonzero = e.iop(BinOp::INotEqual, r, zero)?;
    let normal = e.iop(BinOp::LogicalAnd, normal, nonzero)?;
    let k150 = e.u(150)?;
    let delta = e.iop(BinOp::ISub, ey, k150)?;
    let delta = e.iop(BinOp::ShiftLeftLogical, delta, k23)?;
    let as_normal = e.iop(BinOp::IAdd, rb, delta)?;
    let k31 = e.u(31)?;
    let shift = e.iop(BinOp::ISub, ey, one)?;
    let shift = e.iop(BinOp::BitwiseAnd, shift, k31)?;
    let as_denormal = e.iop(BinOp::ShiftLeftLogical, r, shift)?;
    let rbits = e.sel(normal, as_normal, as_denormal)?;
    let rbits = e.iop(BinOp::BitwiseOr, rbits, sign)?;
    let rem = e.float(rbits)?;
    let rem = e.sel(keep_x, x, rem)?;
    // A NaN of the device's arithmetic: `((x - x) + (y - y)) inf`, which is `0 inf` for a finite
    // `x` and a zero `y`.
    let dx = e.sub(x, x)?;
    let dy = e.sub(y, y)?;
    let d = e.add(dx, dy)?;
    let inf_f = e.c(f32::INFINITY)?;
    let nan = e.mul(d, inf_f)?;
    // An infinite `x` or a zero or denormal `y`, neither a NaN: Metal's NaN.
    let x_nan = e.iop(BinOp::UGreaterThan, ax, inf)?;
    let any_nan = e.iop(BinOp::LogicalOr, x_nan, y_nan)?;
    let domain = e.iop(BinOp::LogicalOr, x_nonfinite, y_zero)?;
    let no_nan = e.fb.unary(UnOp::LogicalNot, any_nan)?;
    let domain = e.iop(BinOp::LogicalAnd, domain, no_nan)?;
    let metal = e.metal_nan(None)?;
    let nan = e.sel(domain, metal, nan)?;
    e.sel(nan_lane, nan, rem)
}

/// `r 2^(8 steps) mod m` for `r` below `m`: a loop of `steps` iterations of `r = (r << 8) mod m`.
fn steps_mod(e: &mut E<'_, '_>, r: Value, m: Value, steps: Value) -> BResult<Value> {
    let ut = e.fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let rv = e.fb.local_var(ut, None)?;
    let rp = e.fb.var_ptr(rv)?;
    let iv = e.fb.local_var(ut, None)?;
    let ip = e.fb.var_ptr(iv)?;
    let (zero, one, eight) = (e.u(0)?, e.u(1)?, e.u(8)?);
    e.fb.store(rp, r, MemAccess::default())?;
    e.fb.store(ip, zero, MemAccess::default())?;
    let body: LoopFn<'_, ()> = Box::new(move |fb, cx, lc| {
        let i = fb.load(ip, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, i, steps)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let next = fb.binary(BinOp::IAdd, i, one)?;
        fb.store(ip, next, MemAccess::default())?;
        let v = fb.load(rp, MemAccess::default())?;
        let v = fb.binary(BinOp::ShiftLeftLogical, v, eight)?;
        let v = fb.binary(BinOp::UMod, v, m)?;
        fb.store(rp, v, MemAccess::default())
    });
    e.fb.loop_(&mut (), LoopSpec::default(), body)?;
    e.fb.load(rp, MemAccess::default())
}
