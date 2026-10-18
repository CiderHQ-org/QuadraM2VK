//! The emission core of the math intrinsics: integer functions (`air` integer math,
//! `llvm.{s,u}{min,max}`, `llvm.fsh{l,r}`, `llvm.bswap`, `llvm.ctpop`, saturating arithmetic,
//! `llvm.abs`, `llvm.ct{l,t}z`), float functions (`air` float math, `llvm.{fma,fmuladd,fabs,
//! minnum,maxnum}`), and `air.convert`.
//!
//! Exact sequences are used wherever `GLSL.std.450` differs from Metal: `clz`/`ctz` from
//! `FindUMsb`/`FindILsb` with the zero case, `rotate` and the funnel shifts with an in-range
//! second shift (`(x >> 1) >> (N - 1 - s)`), `round` (half away from zero) from `Trunc`, `mix`
//! as Metal computes it (on `float` and `half` `fma(t, y, fma(-t, x, x))` by the profile's fused
//! multiply-add route, `metal_mix`; elsewhere `x * (1 - t) + y * t`), and saturating
//! float-to-integer `air.convert` (NaN to 0, out-of-range to the bound, measured on Apple GPUs)
//! with the clamp applied before the conversion so no conversion is out of range. `sign`, `fract`,
//! `saturate`, `median3`, and on `float` and `half` `fmin`, `fmax`, `min3`, `max3`, and `clamp`
//! answer as Metal does at their special operands (zeros, NaNs, infinities, `f32` denormals), which
//! `FSign`, `Fract`, `NClamp`, and `NMin`/`NMax` leave to the device. Where `GLSL.std.450` is a
//! different function from Metal's, Metal's is emitted: `sinpi`, `cospi`, and `tanpi` are Metal's
//! own forms (`libm`) and elsewhere reduce their argument exactly instead of scaling it by
//! `float(π)`, `fast_tanh` on `float` is the MSL fast-math formula, and the sRGB conversions go
//! through Metal's twelve-bit tables ([`srgb`]). Functions whose Vulkan precision is looser than
//! correct rounding record a `PrecisionNote` at selection.

use crate::air::intrinsics::PackFormat;
use crate::lower::arith::denormal;
use crate::lower::arith::divide::{self, Reciprocal, Rough};
use crate::lower::arith::exact_division::{div_exact_lanes, per_lane};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::libm::{self, Lib};
use crate::lower::arith::nan::{self, NanTest};
use crate::lower::arith::narrow;
use crate::lower::arith::ops::{fsplat, mask, ones, sext, splat, ty, CastKind};
use crate::lower::arith::srgb;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, TypeRule};
use crate::spirv::ids::{FuncId, GlobalId, Value};
use crate::spirv::ir::{BinOp, Contraction, ConvOp, GlslOp, Type, UnOp};

/// Whether an integer function reads its operands as signed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sign {
    Signed,
    Unsigned,
}

/// An integer function on held integers of one form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IntFn {
    /// Signed absolute value (the unsigned one is the operand itself).
    Abs,
    AbsDiff(Sign),
    Min(Sign),
    Max(Sign),
    Clamp(Sign),
    Min3(Sign),
    Max3(Sign),
    MulHi(Sign),
    MadHi(Sign),
    /// `a * b + c` saturated from the exact (double-width) result.
    MadSat(Sign),
    AddSat(Sign),
    SubSat(Sign),
    HAdd(Sign),
    RHAdd(Sign),
    /// `OpBitFieldExtract` on 32-bit containers, shifts on 64-bit ones (Vulkan's bit-field
    /// instructions are 32-bit only).
    ExtractBits(Sign),
    /// As [`IntFn::ExtractBits`].
    InsertBits,
    Popcount,
    ReverseBits,
    Bswap,
    Rotate,
    Clz,
    Ctz,
    Fshl,
    Fshr,
}

impl IntFn {
    /// The operand count.
    pub fn arity(self) -> usize {
        match self {
            IntFn::Abs | IntFn::Popcount | IntFn::ReverseBits | IntFn::Bswap => 1,
            IntFn::Clz | IntFn::Ctz => 1,
            IntFn::AbsDiff(_)
            | IntFn::Min(_)
            | IntFn::Max(_)
            | IntFn::MulHi(_)
            | IntFn::AddSat(_)
            | IntFn::SubSat(_)
            | IntFn::HAdd(_)
            | IntFn::RHAdd(_)
            | IntFn::Rotate => 2,
            IntFn::Clamp(_)
            | IntFn::Min3(_)
            | IntFn::Max3(_)
            | IntFn::MadHi(_)
            | IntFn::MadSat(_)
            | IntFn::ExtractBits(_)
            | IntFn::Fshl
            | IntFn::Fshr => 3,
            IntFn::InsertBits => 4,
        }
    }

    /// An upper estimate of the SPIR-V words per use on `bits`-wide integers.
    pub fn words(self, bits: u8) -> u32 {
        match self {
            IntFn::Bswap => 16 + 32 * u32::from(bits / 8),
            IntFn::Clz | IntFn::Ctz | IntFn::Popcount | IntFn::ReverseBits => 128,
            _ => 96,
        }
    }
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

fn glsl(fb: &mut FunctionBuilder<'_>, op: GlslOp, args: &[Value]) -> BResult<Value> {
    fb.glsl(op, args)
}

fn arg(args: &[Value], i: usize) -> BResult<Value> {
    args.get(i)
        .copied()
        .ok_or(crate::spirv::error::BuildError::Operand {
            op: crate::spv::Op::ExtInst,
            rule: crate::spirv::error::OperandRule::Arity,
        })
}

/// The unsigned 32-bit form with `f`'s lanes.
fn u32_form(f: Form) -> Form {
    f.with(Held::Int {
        bits: 32,
        container: 32,
    })
}

/// `ldexp(x, k)` (`x` of form `f`, `k` a held `i32` of its lanes) as Metal answers it for every
/// exponent. GLSL's `Ldexp` leaves an exponent past `+128` and a product past the format's range
/// undefined, and may flush any result of an exponent below `-126`, so a `float`
/// `ldexp(2^100, -150)` (`2^-50`) could read zero. A `float` (and a `bfloat`, held in its `f32`)
/// is computed on the container's bits ([`ldexp_bits`]; a `bfloat`'s denormals, which Metal's
/// `bfloat` environment leaves unmeasured, read and flush as a `float`'s); a `half`, held exactly
/// in an `f32`, is
/// scaled by `2^k` with `k` clamped to `[-64, 64]`, which keeps every product an exact normal
/// `f32` and saturates it past `half`'s range, and the caller rounds it to `half` once. A
/// `double` keeps the device's `Ldexp`: Metal has no `double` to answer it.
fn ldexp(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, k: Value) -> BResult<Value> {
    match f.float_fmt() {
        Some(FloatFmt::F32 | FloatFmt::BFloat) => ldexp_bits(fb, f, x, k),
        Some(FloatFmt::Half) => {
            let i = u32_form(f);
            let (lo, hi) = (splat(fb, i, (-64i32) as u32 as u64)?, splat(fb, i, 64)?);
            let k = glsl(fb, GlslOp::SClamp, &[k, lo, hi])?;
            let bias = splat(fb, i, 127)?;
            let e = bin(fb, BinOp::IAdd, k, bias)?;
            let at = splat(fb, i, 23)?;
            let e = bin(fb, BinOp::ShiftLeftLogical, e, at)?;
            let ft = ty(fb, f)?;
            let scale = fb.convert(ConvOp::Bitcast, e, ft)?;
            bin(fb, BinOp::FMul, x, scale)
        }
        _ => glsl(fb, GlslOp::Ldexp, &[x, k]),
    }
}

/// [`ldexp`] of an `f32` container on its bits, as Metal's `float` answers (the
/// denormal probes): a zero or a denormal `x` (reads as zero) is the zero of its sign; an infinity
/// or a NaN is `x`, bit for bit; otherwise the exponent field `e + k` (with `k` clamped to
/// `[-512, 512]`, which saturates every answer) is an infinity of `x`'s sign from 255 up, the
/// zero of its sign at 0 and below (a result below `2^-126` is flushed, tininess before
/// rounding: the scaled significand is exact), and else replaces `x`'s exponent field.
fn ldexp_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, k: Value) -> BResult<Value> {
    let i = u32_form(f);
    let it = ty(fb, i)?;
    let bits = fb.convert(ConvOp::Bitcast, x, it)?;
    let sign = splat(fb, i, 0x8000_0000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, bits, sign)?;
    let at = splat(fb, i, 23)?;
    let field = splat(fb, i, 0xFF)?;
    let e = bin(fb, BinOp::ShiftRightLogical, bits, at)?;
    let e = bin(fb, BinOp::BitwiseAnd, e, field)?;
    let (lo, hi) = (splat(fb, i, (-512i32) as u32 as u64)?, splat(fb, i, 512)?);
    let k = glsl(fb, GlslOp::SClamp, &[k, lo, hi])?;
    let ne = bin(fb, BinOp::IAdd, e, k)?;
    let keep = splat(fb, i, 0x807F_FFFF)?;
    let kept = bin(fb, BinOp::BitwiseAnd, bits, keep)?;
    let moved = bin(fb, BinOp::ShiftLeftLogical, ne, at)?;
    let scaled = bin(fb, BinOp::BitwiseOr, kept, moved)?;
    let inf = splat(fb, i, 0x7F80_0000)?;
    let inf = bin(fb, BinOp::BitwiseOr, sign, inf)?;
    let zero = splat(fb, i, 0)?;
    let under = bin(fb, BinOp::SLessThanEqual, ne, zero)?;
    let r = fb.select(under, sign, scaled)?;
    let over = bin(fb, BinOp::SGreaterThanEqual, ne, field)?;
    let r = fb.select(over, inf, r)?;
    let special = bin(fb, BinOp::IEqual, e, field)?;
    let r = fb.select(special, bits, r)?;
    let small = bin(fb, BinOp::IEqual, e, zero)?;
    let r = fb.select(small, sign, r)?;
    let ft = ty(fb, f)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// The low and high 32-bit halves of a 64-bit container.
fn halves(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<(Value, Value)> {
    let t32 = ty(fb, u32_form(f))?;
    let lo = fb.convert(ConvOp::UConvert, x, t32)?;
    let k = splat(fb, f, 32)?;
    let hi = bin(fb, BinOp::ShiftRightLogical, x, k)?;
    let hi = fb.convert(ConvOp::UConvert, hi, t32)?;
    Ok((lo, hi))
}

/// A 32-bit lane result widened to `f`'s container.
fn widen(fb: &mut FunctionBuilder<'_>, f: Form, r: Value) -> BResult<Value> {
    if f.container() == 32 {
        return Ok(r);
    }
    let t = ty(fb, f)?;
    fb.convert(ConvOp::UConvert, r, t)
}

/// `(x << s) | ((y >> 1) >> (N - 1 - s))` for `s < N` (the funnel/rotate core without an
/// out-of-range shift).
fn funnel_left(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    s: Value,
) -> BResult<Value> {
    let n1 = splat(fb, f, u64::from(f.bits()) - 1)?;
    let one = splat(fb, f, 1)?;
    let hi = bin(fb, BinOp::ShiftLeftLogical, x, s)?;
    let y1 = bin(fb, BinOp::ShiftRightLogical, y, one)?;
    let rest = bin(fb, BinOp::ISub, n1, s)?;
    let lo = bin(fb, BinOp::ShiftRightLogical, y1, rest)?;
    let r = bin(fb, BinOp::BitwiseOr, hi, lo)?;
    mask(fb, f, r)
}

/// `s mod N`.
fn amount(fb: &mut FunctionBuilder<'_>, f: Form, s: Value) -> BResult<Value> {
    let n = splat(fb, f, u64::from(f.bits()))?;
    bin(fb, BinOp::UMod, s, n)
}

/// Emit integer function `op` on `args` (held integers of form `f`).
pub fn int_fn(fb: &mut FunctionBuilder<'_>, op: IntFn, f: Form, args: &[Value]) -> BResult<Value> {
    let a = arg(args, 0)?;
    let signed = |s: Sign| s == Sign::Signed;
    // The operand as its signed or unsigned value in the container.
    let ext = |fb: &mut FunctionBuilder<'_>, s: Sign, x: Value| -> BResult<Value> {
        if signed(s) {
            sext(fb, f, x)
        } else {
            Ok(x)
        }
    };
    let minmax = |fb: &mut FunctionBuilder<'_>, s: Sign, max: bool, x, y| -> BResult<Value> {
        let op = match (signed(s), max) {
            (true, true) => GlslOp::SMax,
            (true, false) => GlslOp::SMin,
            (false, true) => GlslOp::UMax,
            (false, false) => GlslOp::UMin,
        };
        glsl(fb, op, &[x, y])
    };
    let n = u64::from(f.bits());
    match op {
        IntFn::Abs => {
            let x = sext(fb, f, a)?;
            let r = glsl(fb, GlslOp::SAbs, &[x])?;
            mask(fb, f, r)
        }
        IntFn::AbsDiff(s) => {
            let (x, y) = (ext(fb, s, a)?, ext(fb, s, arg(args, 1)?)?);
            let gt = if signed(s) {
                BinOp::SGreaterThan
            } else {
                BinOp::UGreaterThan
            };
            let c = bin(fb, gt, x, y)?;
            let d1 = bin(fb, BinOp::ISub, x, y)?;
            let d2 = bin(fb, BinOp::ISub, y, x)?;
            let r = fb.select(c, d1, d2)?;
            mask(fb, f, r)
        }
        IntFn::Min(s) | IntFn::Max(s) | IntFn::Min3(s) | IntFn::Max3(s) => {
            let max = matches!(op, IntFn::Max(_) | IntFn::Max3(_));
            let (x, y) = (ext(fb, s, a)?, ext(fb, s, arg(args, 1)?)?);
            let mut r = minmax(fb, s, max, x, y)?;
            if matches!(op, IntFn::Min3(_) | IntFn::Max3(_)) {
                let z = ext(fb, s, arg(args, 2)?)?;
                r = minmax(fb, s, max, r, z)?;
            }
            mask(fb, f, r)
        }
        IntFn::Clamp(s) => {
            let x = ext(fb, s, a)?;
            let (lo, hi) = (ext(fb, s, arg(args, 1)?)?, ext(fb, s, arg(args, 2)?)?);
            let r = minmax(fb, s, true, x, lo)?;
            let r = minmax(fb, s, false, r, hi)?;
            mask(fb, f, r)
        }
        IntFn::MulHi(s) | IntFn::MadHi(s) => {
            let (x, y) = (ext(fb, s, a)?, ext(fb, s, arg(args, 1)?)?);
            let wide = if signed(s) {
                BinOp::SMulExtended
            } else {
                BinOp::UMulExtended
            };
            let p = bin(fb, wide, x, y)?;
            let hi = fb.composite_extract(p, &[1])?;
            let mut r = if f.legalized() {
                // The product's bits [N, 2N): (lo >> N) | (hi << (C - N)).
                let lo = fb.composite_extract(p, &[0])?;
                let sh = splat(fb, f, n)?;
                let lo = bin(fb, BinOp::ShiftRightLogical, lo, sh)?;
                let up = splat(fb, f, u64::from(f.container()) - n)?;
                let hi = bin(fb, BinOp::ShiftLeftLogical, hi, up)?;
                bin(fb, BinOp::BitwiseOr, lo, hi)?
            } else {
                hi
            };
            if matches!(op, IntFn::MadHi(_)) {
                r = bin(fb, BinOp::IAdd, r, arg(args, 2)?)?;
            }
            mask(fb, f, r)
        }
        IntFn::AddSat(Sign::Unsigned) => {
            let b = arg(args, 1)?;
            let s = bin(fb, BinOp::IAdd, a, b)?;
            if f.legalized() {
                // No wrap in the container: clamp to the width's maximum.
                let m = splat(fb, f, ones(f.bits()))?;
                glsl(fb, GlslOp::UMin, &[s, m])
            } else {
                let wrapped = bin(fb, BinOp::ULessThan, s, a)?;
                let m = splat(fb, f, ones(f.bits()))?;
                fb.select(wrapped, m, s)
            }
        }
        IntFn::SubSat(Sign::Unsigned) => {
            let b = arg(args, 1)?;
            let under = bin(fb, BinOp::UGreaterThan, b, a)?;
            let d = bin(fb, BinOp::ISub, a, b)?;
            let z = splat(fb, f, 0)?;
            fb.select(under, z, d)
        }
        IntFn::AddSat(Sign::Signed) | IntFn::SubSat(Sign::Signed) if f.legalized() => {
            // Each operand keyed: its zero-extended bits with bit N - 1 flipped, its value plus
            // 2^(N-1). N + 1 <= C bits: the keys' sum, or their difference plus 2^N, is the exact
            // result plus 2^N and never below zero, so clamping it unsigned to
            // [2^(N-1), 3 * 2^(N-1) - 1] and taking 2^(N-1) away leaves the saturated result's
            // key, which flipped back is the result, zero-extended. The operands sign-extended,
            // summed, clamped signed to the width's range and masked come out sign-extended on
            // the RTX 3090 (NVIDIA 615.71.09), which drops that mask.
            let half = splat(fb, f, 1 << (n - 1))?;
            let x = bin(fb, BinOp::BitwiseXor, a, half)?;
            let y = bin(fb, BinOp::BitwiseXor, arg(args, 1)?, half)?;
            let t = if matches!(op, IntFn::AddSat(_)) {
                bin(fb, BinOp::IAdd, x, y)?
            } else {
                let up = splat(fb, f, 1 << n)?;
                let x = bin(fb, BinOp::IAdd, x, up)?;
                bin(fb, BinOp::ISub, x, y)?
            };
            let hi = splat(fb, f, 3 * (1 << (n - 1)) - 1)?;
            let t = glsl(fb, GlslOp::UMax, &[t, half])?;
            let t = glsl(fb, GlslOp::UMin, &[t, hi])?;
            let k = bin(fb, BinOp::ISub, t, half)?;
            bin(fb, BinOp::BitwiseXor, k, half)
        }
        IntFn::AddSat(Sign::Signed) | IntFn::SubSat(Sign::Signed) => {
            let add = matches!(op, IntFn::AddSat(_));
            let o = if add { BinOp::IAdd } else { BinOp::ISub };
            let y = arg(args, 1)?;
            let s = bin(fb, o, a, y)?;
            // Overflow iff the result's sign differs from what the operands allow.
            let xs = bin(fb, BinOp::BitwiseXor, a, s)?;
            let t = if add {
                bin(fb, BinOp::BitwiseXor, y, s)?
            } else {
                bin(fb, BinOp::BitwiseXor, a, y)?
            };
            let both = bin(fb, BinOp::BitwiseAnd, xs, t)?;
            let z = splat(fb, f, 0)?;
            let over = bin(fb, BinOp::SLessThan, both, z)?;
            // x < 0 ? MIN : MAX == (x >>a (C-1)) ^ MAX.
            let c1 = splat(fb, f, n - 1)?;
            let sgn = bin(fb, BinOp::ShiftRightArithmetic, a, c1)?;
            let max = splat(fb, f, ones(f.bits() - 1))?;
            let sat = bin(fb, BinOp::BitwiseXor, sgn, max)?;
            fb.select(over, sat, s)
        }
        IntFn::HAdd(s) | IntFn::RHAdd(s) => {
            let (x, y) = (ext(fb, s, a)?, ext(fb, s, arg(args, 1)?)?);
            let one = splat(fb, f, 1)?;
            let shr = if signed(s) {
                BinOp::ShiftRightArithmetic
            } else {
                BinOp::ShiftRightLogical
            };
            let hx = bin(fb, shr, x, one)?;
            let hy = bin(fb, shr, y, one)?;
            let both = if matches!(op, IntFn::HAdd(_)) {
                bin(fb, BinOp::BitwiseAnd, x, y)?
            } else {
                bin(fb, BinOp::BitwiseOr, x, y)?
            };
            let carry = bin(fb, BinOp::BitwiseAnd, both, one)?;
            let r = bin(fb, BinOp::IAdd, hx, hy)?;
            let r = bin(fb, BinOp::IAdd, r, carry)?;
            mask(fb, f, r)
        }
        IntFn::ExtractBits(s) if f.container() == 64 => {
            extract_bits64(fb, f, signed(s), a, arg(args, 1)?, arg(args, 2)?)
        }
        IntFn::ExtractBits(s) => {
            let r = fb.bit_field_extract(signed(s), a, arg(args, 1)?, arg(args, 2)?)?;
            mask(fb, f, r)
        }
        IntFn::InsertBits if f.container() == 64 => {
            insert_bits64(fb, f, a, arg(args, 1)?, arg(args, 2)?, arg(args, 3)?)
        }
        IntFn::InsertBits => fb.bit_field_insert(a, arg(args, 1)?, arg(args, 2)?, arg(args, 3)?),
        IntFn::MadSat(s) => mad_sat(fb, f, signed(s), a, arg(args, 1)?, arg(args, 2)?),
        IntFn::Popcount => {
            if f.container() == 32 {
                return fb.unary(UnOp::BitCount, a);
            }
            let (lo, hi) = halves(fb, f, a)?;
            let cl = fb.unary(UnOp::BitCount, lo)?;
            let ch = fb.unary(UnOp::BitCount, hi)?;
            let r = bin(fb, BinOp::IAdd, cl, ch)?;
            widen(fb, f, r)
        }
        IntFn::ReverseBits => {
            let full = if f.container() == 32 {
                fb.unary(UnOp::BitReverse, a)?
            } else {
                let (lo, hi) = halves(fb, f, a)?;
                let rl = fb.unary(UnOp::BitReverse, lo)?;
                let rh = fb.unary(UnOp::BitReverse, hi)?;
                let rl = widen(fb, f, rl)?;
                let rh = widen(fb, f, rh)?;
                let k = splat(fb, f, 32)?;
                let rl = bin(fb, BinOp::ShiftLeftLogical, rl, k)?;
                bin(fb, BinOp::BitwiseOr, rl, rh)?
            };
            if !f.legalized() {
                return Ok(full);
            }
            let sh = splat(fb, f, u64::from(f.container()) - n)?;
            bin(fb, BinOp::ShiftRightLogical, full, sh)
        }
        IntFn::Bswap => {
            let bytes = n / 8;
            let ff = splat(fb, f, 0xFF)?;
            let mut acc: Option<Value> = None;
            for i in 0..bytes {
                let byte = if i == 0 {
                    a
                } else {
                    let k = splat(fb, f, 8 * i)?;
                    bin(fb, BinOp::ShiftRightLogical, a, k)?
                };
                let byte = bin(fb, BinOp::BitwiseAnd, byte, ff)?;
                let to = 8 * (bytes - 1 - i);
                let placed = if to == 0 {
                    byte
                } else {
                    let k = splat(fb, f, to)?;
                    bin(fb, BinOp::ShiftLeftLogical, byte, k)?
                };
                acc = Some(match acc {
                    None => placed,
                    Some(x) => bin(fb, BinOp::BitwiseOr, x, placed)?,
                });
            }
            match acc {
                Some(x) => Ok(x),
                None => Ok(a),
            }
        }
        IntFn::Rotate => {
            let s = amount(fb, f, arg(args, 1)?)?;
            funnel_left(fb, f, a, a, s)
        }
        IntFn::Fshl => {
            let s = amount(fb, f, arg(args, 2)?)?;
            funnel_left(fb, f, a, arg(args, 1)?, s)
        }
        IntFn::Fshr => {
            // (b >> s) | ((a << 1) << (N - 1 - s)).
            let b = arg(args, 1)?;
            let s = amount(fb, f, arg(args, 2)?)?;
            let n1 = splat(fb, f, n - 1)?;
            let one = splat(fb, f, 1)?;
            let lo = bin(fb, BinOp::ShiftRightLogical, b, s)?;
            let a1 = bin(fb, BinOp::ShiftLeftLogical, a, one)?;
            let rest = bin(fb, BinOp::ISub, n1, s)?;
            let hi = bin(fb, BinOp::ShiftLeftLogical, a1, rest)?;
            let r = bin(fb, BinOp::BitwiseOr, hi, lo)?;
            mask(fb, f, r)
        }
        IntFn::Clz => {
            let u = u32_form(f);
            // 31 - FindUMsb(x): 32 for zero (FindUMsb gives -1).
            let clz32 = |fb: &mut FunctionBuilder<'_>, x: Value| -> BResult<Value> {
                let m = glsl(fb, GlslOp::FindUMsb, &[x])?;
                let k = splat(fb, u, 31)?;
                bin(fb, BinOp::ISub, k, m)
            };
            let r = if f.container() == 32 {
                clz32(fb, a)?
            } else {
                let (lo, hi) = halves(fb, f, a)?;
                let cl = clz32(fb, lo)?;
                let ch = clz32(fb, hi)?;
                let z = splat(fb, u, 0)?;
                let hz = bin(fb, BinOp::IEqual, hi, z)?;
                let k = splat(fb, u, 32)?;
                let low = bin(fb, BinOp::IAdd, cl, k)?;
                let r = fb.select(hz, low, ch)?;
                widen(fb, f, r)?
            };
            if !f.legalized() {
                return Ok(r);
            }
            let pad = splat(fb, f, u64::from(f.container()) - n)?;
            bin(fb, BinOp::ISub, r, pad)
        }
        IntFn::Ctz => {
            let u = u32_form(f);
            let z = splat(fb, u, 0)?;
            let r = if f.container() == 32 {
                fb.glsl(GlslOp::FindILsb, &[a])?
            } else {
                let (lo, hi) = halves(fb, f, a)?;
                let ll = glsl(fb, GlslOp::FindILsb, &[lo])?;
                let lh = glsl(fb, GlslOp::FindILsb, &[hi])?;
                let k = splat(fb, u, 32)?;
                let lh = bin(fb, BinOp::IAdd, lh, k)?;
                let lnz = bin(fb, BinOp::INotEqual, lo, z)?;
                let r = fb.select(lnz, ll, lh)?;
                widen(fb, f, r)?
            };
            let fz = splat(fb, f, 0)?;
            let zero = bin(fb, BinOp::IEqual, a, fz)?;
            let w = splat(fb, f, n)?;
            fb.select(zero, w, r)
        }
    }
}

/// Scalar `u32` operand `v` of a bit-field function as a 64-bit amount with `f`'s lanes.
fn amount64(fb: &mut FunctionBuilder<'_>, f: Form, v: Value) -> BResult<Value> {
    let s = f.scalar_form();
    let t = ty(fb, s)?;
    let v = fb.convert(ConvOp::UConvert, v, t)?;
    if f.lanes == 1 {
        return Ok(v);
    }
    let vt = ty(fb, f)?;
    fb.composite_construct(vt, &vec![v; usize::from(f.lanes)])
}

/// `extract_bits` on a 64-bit container: `(x << (64 - o - n)) >> (64 - n)` (arithmetic when
/// `signed`), 0 when `n` is 0 (Metal leaves `o + n` beyond the width undefined).
fn extract_bits64(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    signed: bool,
    x: Value,
    offset: Value,
    count: Value,
) -> BResult<Value> {
    let (o, n) = (amount64(fb, f, offset)?, amount64(fb, f, count)?);
    let k64 = splat(fb, f, 64)?;
    let z = splat(fb, f, 0)?;
    // A zero count would shift by 64: shift by 0 and select 0 instead.
    let none = bin(fb, BinOp::IEqual, n, z)?;
    let up = bin(fb, BinOp::ISub, k64, o)?;
    let up = bin(fb, BinOp::ISub, up, n)?;
    let down = bin(fb, BinOp::ISub, k64, n)?;
    let down = fb.select(none, z, down)?;
    let up = fb.select(none, z, up)?;
    let v = bin(fb, BinOp::ShiftLeftLogical, x, up)?;
    let shr = if signed {
        BinOp::ShiftRightArithmetic
    } else {
        BinOp::ShiftRightLogical
    };
    let v = bin(fb, shr, v, down)?;
    let v = fb.select(none, z, v)?;
    mask(fb, f, v)
}

/// `insert_bits` on a 64-bit container: the `n` bits at `o` from `y`, the rest from `x`.
fn insert_bits64(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    offset: Value,
    count: Value,
) -> BResult<Value> {
    let (o, n) = (amount64(fb, f, offset)?, amount64(fb, f, count)?);
    let k64 = splat(fb, f, 64)?;
    let z = splat(fb, f, 0)?;
    let all = splat(fb, f, u64::MAX)?;
    // The field mask: all ones shifted down by 64 - n (n = 0 gives no field), then up by o.
    let none = bin(fb, BinOp::IEqual, n, z)?;
    let down = bin(fb, BinOp::ISub, k64, n)?;
    let down = fb.select(none, z, down)?;
    let m = bin(fb, BinOp::ShiftRightLogical, all, down)?;
    let m = fb.select(none, z, m)?;
    let m = bin(fb, BinOp::ShiftLeftLogical, m, o)?;
    let keep = fb.unary(UnOp::Not, m)?;
    let keep = bin(fb, BinOp::BitwiseAnd, x, keep)?;
    let ins = bin(fb, BinOp::ShiftLeftLogical, y, o)?;
    let ins = bin(fb, BinOp::BitwiseAnd, ins, m)?;
    let r = bin(fb, BinOp::BitwiseOr, keep, ins)?;
    mask(fb, f, r)
}

/// `mad_sat`: `a * b + c` computed exactly as a double-width `(hi, lo)` pair of containers and
/// saturated to the AIR width's range.
fn mad_sat(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    signed: bool,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Value> {
    let (a, b, c) = if signed {
        (sext(fb, f, a)?, sext(fb, f, b)?, sext(fb, f, c)?)
    } else {
        (a, b, c)
    };
    let wide = if signed {
        BinOp::SMulExtended
    } else {
        BinOp::UMulExtended
    };
    let p = bin(fb, wide, a, b)?;
    let lo = fb.composite_extract(p, &[0])?;
    let hi = fb.composite_extract(p, &[1])?;
    let s = bin(fb, BinOp::IAddCarry, lo, c)?;
    let lo = fb.composite_extract(s, &[0])?;
    let carry = fb.composite_extract(s, &[1])?;
    let hi = bin(fb, BinOp::IAdd, hi, carry)?;
    let z = splat(fb, f, 0)?;
    let n = u64::from(f.bits());
    let cw = u64::from(f.container());
    if !signed {
        let max = splat(fb, f, ones(f.bits()))?;
        let over_hi = bin(fb, BinOp::INotEqual, hi, z)?;
        let over_lo = bin(fb, BinOp::UGreaterThan, lo, max)?;
        let over = bin(fb, BinOp::LogicalOr, over_hi, over_lo)?;
        return fb.select(over, max, lo);
    }
    // A negative addend adds -1 to the high word (its sign extension).
    let cneg = bin(fb, BinOp::SLessThan, c, z)?;
    let m1 = splat(fb, f, u64::MAX)?;
    let ext = fb.select(cneg, m1, z)?;
    let hi = bin(fb, BinOp::IAdd, hi, ext)?;
    // The pair is one container value when hi is lo's sign extension.
    let k = splat(fb, f, cw - 1)?;
    let lo_sign = bin(fb, BinOp::ShiftRightArithmetic, lo, k)?;
    let fits = bin(fb, BinOp::IEqual, hi, lo_sign)?;
    let min = splat(fb, f, (!0u64) << (n - 1))?;
    let max = splat(fb, f, ones(f.bits() - 1))?;
    let r = glsl(fb, GlslOp::SMax, &[lo, min])?;
    let r = glsl(fb, GlslOp::SMin, &[r, max])?;
    let neg = bin(fb, BinOp::SLessThan, hi, z)?;
    let sat = fb.select(neg, min, max)?;
    let r = fb.select(fits, r, sat)?;
    mask(fb, f, r)
}

/// A float function on floats of one form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatFn {
    Abs,
    Floor,
    Ceil,
    Trunc,
    /// Round half away from zero.
    Round,
    /// Round half to even.
    Rint,
    /// `x - floor(x)`, below one (`metal_fract`).
    Fract,
    /// `air.fast_fract` on `float`: [`FloatFn::Fract`] without the bound below one, and the
    /// canonical NaN at an infinity where `infinite_nan` (a call with every fast-math flag), else
    /// `+0` as `fract`.
    FastFract {
        infinite_nan: bool,
    },
    /// `air.fract` on `half` with every fast-math flag (Metal has no `fast_` `half` function):
    /// [`FloatFn::Fract`], but the largest `half` below one (`0x3BFF`) for a NaN and for an
    /// infinity, where `fmin(x - floor(x), 0x3BFF)` loses its NaN to the bound.
    FlaggedFract,
    /// Metal's `sign` (`metal_sign`), with and without fast math.
    Sign,
    /// Metal's `saturate` (`metal_saturate`).
    Saturate,
    /// `fmin` (`air.fmin`, `air.fast_fmin`, `air.min`; `llvm.minnum` where [`FloatFn::MinNum`]
    /// refines to it): Metal's on `float` and `half` (`metal_min_max`), the device's `NMin` on the
    /// other formats.
    Min,
    /// `fmax`, as [`FloatFn::Min`] with the order reversed.
    Max,
    /// `clamp(x, lo, hi)`: `fmin(fmax(x, lo), hi)` with Metal's `fmin` and `fmax` on `float` and
    /// `half`, the device's `NClamp` on the other formats.
    Clamp,
    /// `min3(a, b, c)`: `fmin(fmin(a, b), c)`, Metal's on `float` and `half`.
    Min3,
    /// `max3(a, b, c)`: `fmax(fmax(a, b), c)`, Metal's on `float` and `half`.
    Max3,
    /// `llvm.minnum`: LLVM's function, the device's `NMin`, on the formats other than `float`.
    /// Metal's compiler emits it for `a < b ? a : b` under fast math, with every flag; on `float`
    /// it refines to [`FloatFn::Min`] with any flags, which Metal answers alike
    /// (`nan_minmax_fast`).
    MinNum,
    /// `llvm.maxnum`, as [`FloatFn::MinNum`] for `a > b ? a : b` and [`FloatFn::Max`].
    MaxNum,
    /// Metal's `median3` (`metal_median3`).
    Median3,
    /// `air.fast_fmedian3` on `float`: [`FloatFn::Median3`], except that two NaN first operands
    /// give the second.
    FastMedian3,
    Mix,
    Ldexp,
    /// `llvm.fmuladd` on `bfloat` and `double`: fused or not, as the implementation chooses.
    /// `float` and `half` select the fused multiply-add Metal computes ([`super::fma`]).
    MulAdd,
    /// `llvm.copysign`: the first operand's magnitude with the second's sign bit, moved as bits
    /// of the `f32` container, so exact for NaNs and zeros on every device. Not on `double`.
    CopySign,
    // Device precision (`PrecisionNote::DeviceArithmetic`).
    /// Metal's precise `sqrt`: correctly rounded (MSL Table 8.1). On `float` and `half`, the
    /// device's instruction where it is Metal's own in a module compiled with precise functions,
    /// the translation's routine otherwise ([`libm`]).
    Sqrt,
    /// `fma` on `double` (`GLSL.std.450 Fma`) and `bfloat` (`narrow::bf16_fma`). `float` and `half`
    /// select the correctly rounded [`super::fma`].
    Fma,
    /// Metal's precise `fmod`: exact (Table 8.1), `fmod(x, ±inf) = x`; as [`FloatFn::Sqrt`].
    Fmod,
    Dot,
    // Device precision (`PrecisionNote::DeviceTranscendentals`).
    /// Metal's precise `rsqrt`: correctly rounded (Table 8.1); as [`FloatFn::Sqrt`].
    Rsqrt,
    /// `air.fast_rsqrt` on `float`: Metal's fast `rsqrt`, within 2 ulp (Table 8.2); the device's
    /// `InverseSqrt`.
    FastRsqrt,
    /// `air.fast_sqrt` on `float`: `x · rsqrt(x)` as Metal composes it, with Metal's zeros and
    /// `+inf` (`fast_sqrt_of`).
    FastSqrt,
    /// Metal's precise `exp`: within 4 ulp (MSL Table 8.1). On `float` and `half`, where the
    /// device's functions are Metal's, the device's instruction in a module compiled with precise
    /// functions and Metal's own composition of the device's `exp2` in a fast-math one (as
    /// [`FloatFn::Exp10`]); the translation's routine elsewhere ([`libm`]).
    Exp,
    /// `air.fast_exp` on `float`: `exp2(x · log2(e))` as Metal composes it, the product rounded,
    /// of Metal's `exp2` ([`FloatFn::Exp2`]).
    FastExp,
    /// Metal's `exp2`, the same function in both modes: the device's instruction where it is
    /// Metal's own, the translation's routine otherwise.
    Exp2,
    /// Metal's precise `exp10`: Metal's own composition of the device's `exp2` where that is
    /// Metal's, the translation's routine on `float` and `half` otherwise, as no instruction is it.
    Exp10,
    /// `air.fast_exp10` on `float`: `exp2(x · log2(10))` (MSL Table 8.2), the product rounded.
    FastExp10,
    /// Metal's precise `log`; as [`FloatFn::Exp`].
    Log,
    /// `air.fast_log` on `float`: `log2(x) · ln(2)` as Metal composes it, the product rounded,
    /// of Metal's `log2` ([`FloatFn::Log2`]): every lane of case f6524207.
    FastLog,
    /// Metal's `log2`, the same function in both modes; as [`FloatFn::Exp2`].
    Log2,
    /// `air.fast_log2` on `float`: Metal's `log2` ([`FloatFn::Log2`]), the device's instruction
    /// where it is Metal's own, elsewhere the translation's routine of it (`libm::Fun::Log2`).
    FastLog2,
    /// Metal's precise `log10`; as [`FloatFn::Exp10`].
    Log10,
    /// `air.fast_log10` on `float`: `log2(x) · log10(2)` (MSL Table 8.2), the product rounded:
    /// every lane of case f6524207.
    FastLog10,
    /// Metal's precise `sin`: within 4 ulp (MSL Table 8.1). On `float` and `half`, the device's
    /// instruction where it is Metal's own in a module compiled with precise functions, the
    /// translation's routine otherwise ([`libm`]).
    Sin,
    /// Metal's precise `cos`: within 4 ulp; as [`FloatFn::Sin`].
    Cos,
    /// `air.fast_sin` on `float`: [`FloatFn::Sin`], but `+0` past `float(π/2) · 2^22` and for an
    /// infinity or NaN, as Metal answers (`fast_trig_flush`). The device's `Sin` where its
    /// functions are Metal's, in either mode; elsewhere the device's `Sin` inside `[-π, π]`
    /// (Apple's `2^-13` bound) and the translation's precise routine past it ([`libm`]).
    FastSin,
    /// `air.fast_cos` on `float`: [`FloatFn::Cos`], but `+0` past `float(π/2) · 2^22` and for an
    /// infinity or NaN, as Metal answers (`fast_trig_flush`); as [`FloatFn::FastSin`].
    FastCos,
    /// Metal's precise `tan`: within 6 ulp; as [`FloatFn::Sin`].
    Tan,
    /// `air.fast_tan` on `float`: Metal's fast `tan`, the formula `sin(x) · (1 / cos(x))` of MSL
    /// Table 8.2: the device's `Tan` where its functions are Metal's in a fast-math module, and
    /// elsewhere the formula of [`FloatFn::FastSin`] and [`FloatFn::FastCos`] divided by Metal's
    /// fast division (`fast_tan`), a call of its [`FastDivision`] routine.
    FastTan {
        /// The fast division's reciprocal on the profile.
        reciprocal: Reciprocal,
    },
    /// Metal's precise `asin`: within 4 ulp (MSL Table 8.1). On `float`, the device's
    /// instruction where it is Metal's own in a module compiled with precise functions, the
    /// translation's routine otherwise ([`libm`]).
    Asin,
    /// `air.fast_asin` on `float`: within 5 ulp (Table 8.2); as [`FloatFn::Asin`], in Metal's
    /// fast mode.
    FastAsin,
    /// Metal's precise `acos`: within 4 ulp; as [`FloatFn::Asin`].
    Acos,
    /// `air.fast_acos` on `float`: within 5 ulp; as [`FloatFn::FastAsin`].
    FastAcos,
    /// Metal's precise `atan`: within 5 ulp; as [`FloatFn::Asin`].
    Atan,
    /// `air.fast_atan` on `float`: within 5 ulp; as [`FloatFn::FastAsin`].
    FastAtan,
    /// Metal's precise `sinh`; as [`FloatFn::Exp`].
    Sinh,
    /// `air.fast_sinh` on `float`: `(exp(x) - exp(-x)) · 0.5` of Metal's fast `exp` (MSL Table
    /// 8.2), or the device's `Sinh` where it is Metal's in a fast-math module.
    FastSinh,
    /// Metal's precise `cosh`; as [`FloatFn::Exp`].
    Cosh,
    /// `air.fast_cosh` on `float`: `(exp(x) + exp(-x)) · 0.5`; as [`FloatFn::FastSinh`].
    FastCosh,
    /// Metal's precise `tanh` (within 5 ulp): the device's instruction where its functions are
    /// Metal's (SPIRV-Cross emits `precise::tanh` in either mode), the routine otherwise.
    Tanh,
    /// `air.fast_tanh` on `float`: the MSL fast-math accuracy table defines it as the formula
    /// `(t - 1) / (t + 1)` with `t = exp(2x)`, not as an accuracy bound, so it is a different
    /// function from `tanh` (it is `NaN` once `t` overflows). Emitted as that formula with
    /// `t = Exp2(x * 2 log2(e))` (`tanh_of_exp`). Narrow forms keep [`FloatFn::Tanh`]: the
    /// fast-math table is the single-precision one.
    FastTanh {
        /// The fast division's reciprocal on the profile.
        reciprocal: Reciprocal,
    },
    /// Metal's precise `asinh`; as [`FloatFn::Exp`].
    Asinh,
    /// `air.fast_asinh` on `float` with every flag: Metal's fast formula
    /// `log(x + sqrt(x * x + 1))` (MSL Table 8.2) in Metal's own form (`libm::Fun::FastAsinh`),
    /// which is the canonical NaN at `-inf` (`-inf + inf`), or the device's `Asinh` where it is
    /// Metal's in a fast-math module.
    FastAsinh,
    /// Metal's precise `acosh`; as [`FloatFn::Exp`].
    Acosh,
    /// `air.fast_acosh` on `float`: `log(x + sqrt(x * x - 1))`; as [`FloatFn::FastAsinh`].
    FastAcosh,
    /// Metal's precise `atanh`; as [`FloatFn::Exp`].
    Atanh,
    /// `air.fast_atanh` on `float`: `0.5 · log((1 + x) / (1 - x))` (MSL Table 8.2) with Metal's
    /// fast division, or the device's `Atanh` where it is Metal's in a fast-math module.
    FastAtanh {
        /// The fast division's reciprocal on the profile.
        reciprocal: Reciprocal,
    },
    /// Metal's `sinpi`: within 4 ulp (MSL Table 8.1). On `float` and `half`, Metal's own form,
    /// the translation's routine ([`libm`]); on a `bfloat` the exact reduction around the device's
    /// `Sin` and `Cos` (`pi_scaled`).
    SinPi,
    /// Metal's `cospi`; as [`FloatFn::SinPi`].
    CosPi,
    /// Metal's `tanpi`: within 6 ulp; as [`FloatFn::SinPi`].
    TanPi,
    /// `air.fast_sinpi` on `float`: Metal's fast `sinpi`, which is its precise one
    /// ([`FloatFn::SinPi`]): the routine, but where the device's functions are Metal's, the exact
    /// reduction around its `Sin` and `Cos` (`pi_scaled`), within Apple's `2^-13` bound on
    /// `[-1, 1]` (MSL Table 8.2).
    FastSinPi,
    /// `air.fast_cospi` on `float`; as [`FloatFn::FastSinPi`].
    FastCosPi,
    /// `air.fast_tanpi` on `float`; as [`FloatFn::FastSinPi`] (Table 8.2: `tan(x * pi)`).
    FastTanPi,
    Powr,
    /// `air.fast_powr` on `float`: `exp2(y · log2(x))` as Metal composes it, the product rounded.
    FastPowr,
    /// Metal `pow`: C99 Annex F semantics (negative bases with integral exponents, the zero,
    /// one, and infinity cases) around `GLSL.std.450 Pow` of the magnitude, as Metal answers
    /// its special operands (`pow_special`).
    Pow,
    /// `air.fast_pow` on `float`: [`FloatFn::Pow`] around the magnitude `exp2(y · log2|x|)`, as
    /// Metal composes it, except that a NaN base to the exponent one gives the canonical NaN
    /// rather than itself.
    FastPow,
    /// Metal `atan2(y, x)`: C99 Annex F's special values, as Metal answers them
    /// (`atan2_special`), around the generic value: on `float`, the device's `Atan2` where its
    /// functions are Metal's (always `precise::atan2` there), the translation's routine
    /// otherwise ([`libm`]).
    Atan2,
    /// `air.fast_atan2` on `float`: `atan(y / x)` turned by `±π` when `x` is below zero, as Metal
    /// answers it (`fast_atan2`): `0 / 0` and `inf / inf` are NaN, where `atan2` answers C99's
    /// values, and the turn follows `y`'s comparison with zero, so `fast::atan2(-0, -1)` is `+π`.
    FastAtan2 {
        /// The quotient is `y` times `x`'s reciprocal flushed to zero beyond `2^126`, as Metal
        /// answers a call that does not carry every fast-math flag; a call that does divides
        /// over the whole range.
        reciprocal_flushed: bool,
    },
}

/// The precision class of a float function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatPrecision {
    Exact,
    DeviceArithmetic,
    DeviceTranscendental,
}

impl FloatFn {
    /// The operand count.
    pub fn arity(self) -> usize {
        use FloatFn as F;
        match self {
            F::Min
            | F::Max
            | F::MinNum
            | F::MaxNum
            | F::Ldexp
            | F::Fmod
            | F::Dot
            | F::Powr
            | F::FastPowr
            | F::Pow
            | F::FastPow
            | F::Atan2
            | F::FastAtan2 { .. }
            | F::CopySign => 2,
            F::Clamp
            | F::Min3
            | F::Max3
            | F::Median3
            | F::FastMedian3
            | F::Mix
            | F::MulAdd
            | F::Fma => 3,
            _ => 1,
        }
    }

    pub fn precision(self) -> FloatPrecision {
        use FloatFn as F;
        match self {
            F::Sqrt | F::Fma | F::Fmod | F::Dot => FloatPrecision::DeviceArithmetic,
            F::Rsqrt
            | F::FastRsqrt
            | F::FastSqrt
            | F::Exp
            | F::FastExp
            | F::Exp2
            | F::Exp10
            | F::FastExp10
            | F::Log
            | F::FastLog
            | F::Log2
            | F::FastLog2
            | F::Log10
            | F::FastLog10
            | F::Sin
            | F::Cos
            | F::FastSin
            | F::FastCos
            | F::Tan
            | F::FastTan { .. }
            | F::Asin
            | F::FastAsin
            | F::Acos
            | F::FastAcos
            | F::Atan
            | F::FastAtan
            | F::Sinh
            | F::FastSinh
            | F::Cosh
            | F::FastCosh
            | F::Tanh
            | F::FastTanh { .. }
            | F::Asinh
            | F::FastAsinh
            | F::Acosh
            | F::FastAcosh
            | F::Atanh
            | F::FastAtanh { .. }
            | F::SinPi
            | F::CosPi
            | F::TanPi
            | F::FastSinPi
            | F::FastCosPi
            | F::FastTanPi
            | F::Powr
            | F::FastPowr
            | F::Pow
            | F::FastPow
            | F::Atan2
            | F::FastAtan2 { .. } => FloatPrecision::DeviceTranscendental,
            _ => FloatPrecision::Exact,
        }
    }

    /// Whether the lowering answers Metal at `f32` denormal operands and results itself, whatever
    /// the device's denormal mode, so it records no `PrecisionNote::DeviceDenormals`: the functions
    /// that read the operand's exponent field (`sign`, `saturate`, `fract`, `median3`, `fmin`,
    /// `fmax`, `min3`, `max3`, and `clamp`, which compare a denormal as zero and return the
    /// operand, and `atan2`, which also flushes a denormal generic result), `copysign` and `fabs`
    /// (on `float`, [`sign_on_bits`]), which move bits, `ldexp` (on `float`, [`ldexp_bits`]), which
    /// reads a denormal operand as the zero of its sign and flushes a denormal result on the bits,
    /// and `fast::sqrt`, which selects the zero of a denormal operand's sign and has no denormal
    /// result. The arc tangent family (`asin`, `acos`, and `atan`, both modes), precise `sin`,
    /// `cos`, and `tan`, and `sinpi`, `cospi`, and `tanpi` do too on `float`: each is either the
    /// device's instruction where that is Metal's own function, or the translation's routine, which
    /// reads a denormal operand as Metal does (`denormal::operand`) and has no denormal result for
    /// a normal operand (where llvmpipe's `atan(2^-126)` is `0x007fff52`); and
    /// `fast::sin`, `fast::cos`, and `fast::tan` keep a denormal operand as Metal does, where RADV,
    /// ANV, and NVIDIA flush it (case cc234732). `fast::atan2` does not: its quotient is the
    /// device's division, whose operands `denormal::coverage` reads as Metal does. The second stage's
    /// functions do: their routines read a denormal operand as Metal does (`exp` and the hyperbolic
    /// functions as itself below `2^-12`, the logarithms and `pow`'s base as a zero) and flush a
    /// denormal result, and the fast formulas' steps have no denormal intermediate on which their
    /// result depends (`fast::sinh` of a denormal is `(1 - 1) · 0.5 = +0`, as Metal answers, case
    /// f6524207). `round` answers `+0` for every zero it reaches, whatever zero the device's
    /// `Trunc` gives a denormal, which Metal's `round` of a denormal is (the denormal probes, both
    /// modes); precise `fmod` on `float` is the remainder's routine, which reads a denormal divisor
    /// as a zero on the bits and computes the remainder in integers (`libm::fmod`), or on MoltenVK
    /// Metal's own.
    pub fn metal_denormals(self) -> bool {
        use FloatFn as F;
        matches!(
            self,
            F::Abs
                | F::Ldexp
                | F::Round
                | F::Fmod
                | F::Sign
                | F::FastSqrt
                | F::Saturate
                | F::Fract
                | F::FastFract { .. }
                | F::FlaggedFract
                | F::Median3
                | F::FastMedian3
                | F::Min
                | F::Max
                | F::Clamp
                | F::Min3
                | F::Max3
                | F::CopySign
                | F::Atan2
                | F::Sin
                | F::Cos
                | F::Tan
                | F::FastSin
                | F::FastCos
                | F::FastTan { .. }
                | F::SinPi
                | F::CosPi
                | F::TanPi
                | F::Asin
                | F::FastAsin
                | F::Acos
                | F::FastAcos
                | F::Atan
                | F::FastAtan
                | F::Exp
                | F::FastExp
                | F::Exp2
                | F::Exp10
                | F::FastExp10
                | F::Log
                | F::FastLog
                | F::Log2
                | F::FastLog2
                | F::Log10
                | F::FastLog10
                | F::Sinh
                | F::FastSinh
                | F::Cosh
                | F::FastCosh
                | F::Tanh
                | F::FastTanh { .. }
                | F::Asinh
                | F::FastAsinh
                | F::Acosh
                | F::FastAcosh
                | F::Atanh
                | F::FastAtanh { .. }
                | F::Pow
                | F::FastPow
                | F::Powr
                | F::FastPowr
        )
    }

    /// Whether the device's order (`RoundingOrderRoute::Native`) answers the function by the
    /// device's instruction where Metal's answer is computed around one without a routine:
    /// `fast::sqrt`, the device's `Sqrt`, where Metal's is `x · rsqrt(x)` with its own answers at
    /// a zero, a denormal and `+inf` ([`fast_sqrt_of`]).
    pub fn native_instruction(self) -> bool {
        matches!(self, FloatFn::FastSqrt)
    }

    /// Whether the lowering on lanes of format `fmt` answers Metal's NaN bit pattern itself, so it
    /// records no `PrecisionNote::DeviceNanEncoding` and needs no select: the functions that select
    /// their NaN (or answer a number for one: `sign`, `saturate`, `fast::sin`, `fast::cos`),
    /// `fast::tanh`, whose result is Metal's fast quotient, `median3`, and on `float` and `half`
    /// `fmin`, `fmax`, `min3`, `max3`, and `clamp` (`metal_min_max`), which return an operand,
    /// `copysign`, which moves bits, and `fabs` on `float`, which clears the sign bit on the bits
    /// ([`sign_on_bits`]; a `half` `fabs` answers its canonical NaN by a select).
    pub fn metal_nans(self, fmt: FloatFmt) -> bool {
        use FloatFn as F;
        if self == F::Abs {
            return sign_on_bits(Some(fmt));
        }
        if matches!(self, F::Min | F::Max | F::Clamp | F::Min3 | F::Max3) {
            return metal_picks(Some(fmt));
        }
        matches!(
            self,
            F::Sign
                | F::Saturate
                | F::Fract
                | F::FastFract { .. }
                | F::FlaggedFract
                | F::Median3
                | F::FastMedian3
                | F::CopySign
                | F::Atan2
                | F::FastAtan2 { .. }
                | F::FastTanh { .. }
                | F::Pow
                | F::FastPow
                | F::FastSin
                | F::FastCos
        )
    }

    /// Whether the function divides by a device division besides its [`FloatFn::precision`]
    /// class (`PrecisionNote::DeviceArithmetic`): `fast_tanh`'s quotient, whose divisor leaves
    /// the range where Metal's fast division and Vulkan's `OpFDiv` bound their error (`2^-126` to
    /// `2^126`) as `t` approaches overflow, `fast_atan2`'s `y · rcp(x)` (Metal's reciprocal's
    /// model, or the device's division where it is Metal's), `fast::tan`'s quotient of its sine by
    /// its cosine, `fast::atanh`'s `(1 + x) / (1 - x)`, and `fast::tanpi`'s quotient where the
    /// device's functions are Metal's.
    pub fn divides(self) -> bool {
        matches!(
            self,
            FloatFn::FastTanh { .. }
                | FloatFn::FastAtan2 { .. }
                | FloatFn::FastTan { .. }
                | FloatFn::FastAtanh { .. }
                | FloatFn::FastTanPi
        )
    }

    /// Whether the function is lowered on 64-bit floats: `GLSL.std.450` offers the
    /// transcendental families on 16/32-bit floats only, and `copysign` moves the bits of an
    /// `f32` container.
    pub fn allows_f64(self) -> bool {
        self.precision() != FloatPrecision::DeviceTranscendental && self != FloatFn::CopySign
    }

    /// Whether the result on values of a narrow format is again a value of the format without
    /// rounding (it is an operand, a constant of the format, or an integral value of one).
    /// `fract` is not: `x - floor(x)` of a negative non-integral `x` can need more bits than
    /// the format has (half `-0.300048828125` gives `0.699951171875`), so it rounds once, before
    /// its bound (`metal_fract`).
    pub fn closed_on_narrow(self) -> bool {
        use FloatFn as F;
        matches!(
            self,
            F::Abs
                | F::Floor
                | F::Ceil
                | F::Trunc
                | F::Round
                | F::Rint
                | F::Sign
                | F::Saturate
                | F::Min
                | F::Max
                | F::MinNum
                | F::MaxNum
                | F::Clamp
                | F::Min3
                | F::Max3
                | F::Median3
                | F::FastMedian3
                | F::CopySign
        )
    }

    /// An upper estimate of the SPIR-V words per use on form `f`, including the roundings of a
    /// narrow form.
    pub fn words(self, f: Form) -> u32 {
        let half = f.float_fmt() == Some(FloatFmt::Half);
        let base = match metal_half(self, f) {
            // `half_tanh`: the quotient's operands and product, and the far operands' test and
            // signed one (its routines' calls are counted below).
            FloatFn::Tanh if half => 160,
            // Metal's other `half` functions by its fast compositions (`metal_half`), and a
            // scalar's constants of the rounding.
            FloatFn::FastExp
            | FloatFn::FastExp10
            | FloatFn::FastLog
            | FloatFn::FastLog2
            | FloatFn::FastLog10
                if half =>
            {
                128
            }
            FloatFn::FastSinh | FloatFn::FastCosh if half => 128 + libm::use_words(f.lanes, 1),
            FloatFn::Sign | FloatFn::Saturate | FloatFn::Fract | FloatFn::FastFract { .. } => 96,
            // `fract`, and the operand's exponent test and the bound's select.
            FloatFn::FlaggedFract => 128,
            // The steps and the `+0` of a zero result.
            FloatFn::Round => 128,
            // The device's function and the operands' tests where Metal answers NaN, the precise
            // sign's on the bits, each operand classified once: 226 words with the routine's calls
            // on `float4`.
            FloatFn::Powr => 208,
            // And the infinite product's tests and select, on the same classification
            // (`fast_powr_special`).
            FloatFn::FastPowr => 208 + INFINITE_PRODUCT_WORDS,
            // Metal's formulas of its fast `exp` (two calls of `exp2`'s routine, the second
            // counted here).
            FloatFn::FastSinh | FloatFn::FastCosh => 96 + libm::use_words(f.lanes, 1),
            // Two selects per step for the operand and its compared value, and each operand's
            // flush test.
            FloatFn::Median3 | FloatFn::FastMedian3 => 192,
            // Metal's pick: the operands' NaN tests (the bits, 6 instructions each, where the
            // entry does not preserve NaNs), the comparison, and two selects per step, and each
            // operand's flush test.
            FloatFn::Min | FloatFn::Max => 160,
            FloatFn::Clamp | FloatFn::Min3 | FloatFn::Max3 => 288,
            // The device's remainder and, on `half`, the zero's tests and selects.
            FloatFn::Fmod if f.float_fmt() == Some(FloatFmt::Half) => 192,
            // `|x|`, and C99's special cases around the magnitude (`pow_special`): 390 words
            // with the routine's calls on `float4`.
            FloatFn::Pow => 344,
            // And `pow(x, 1)`'s flush of an `f32` denormal `x` (`pow_special`).
            FloatFn::FastPow => 344 + INFINITE_PRODUCT_WORDS + denormal::flush_words(f),
            // The magnitude's test against the last evaluated quadrant (the routine's calls are
            // counted below).
            FloatFn::FastSin | FloatFn::FastCos => 48,
            // Both flushes, the denormal operand's select, and the call of the fast division's
            // routine ([`FastDivision`], whose words are the module's).
            FloatFn::FastTan { .. } => 160 + FAST_DIVISION_CALL_WORDS,
            // The product and Metal's zeros and infinity.
            FloatFn::FastSqrt => 96,
            // The special operands' tests and the selected constant's sign.
            FloatFn::Atan2 => 320,
            // The quotient, its special values and sign, and the turn.
            FloatFn::FastAtan2 { .. } => 400,
            // Where no routine answers them (a `bfloat`), the period reduction, the exact axis
            // arms, both quadrant selections, `tanpi`'s quarter turns, and the finiteness guard around
            // the device's `Sin` and `Cos` (`pi_scaled`); the routines' calls are counted below.
            FloatFn::SinPi | FloatFn::CosPi | FloatFn::TanPi if libm::uses(self, f).is_empty() => {
                384
            }
            // Through the routine: its call, counted below, and Metal's NaN's select.
            FloatFn::SinPi | FloatFn::CosPi | FloatFn::TanPi => 112,
            // The routine's call, or where the device's functions are Metal's the reduction
            // around its `Sin` and `Cos`, which the routine's words do not cover.
            FloatFn::FastSinPi | FloatFn::FastCosPi | FloatFn::FastTanPi => 384,
            // The tie correction of a bfloat fused multiply-add (`narrow::bf16_fma`).
            FloatFn::Fma if f.float_fmt() == Some(FloatFmt::BFloat) => 208,
            // `t`, the quotient's operands, their `NoContraction` decorations, and the call of
            // the fast division's routine ([`FastDivision`], whose words are the module's).
            FloatFn::FastTanh { .. } => 80 + FAST_DIVISION_CALL_WORDS,
            // `1 ± x`, the call of the fast division's routine, and the halving.
            FloatFn::FastAtanh { .. } => 96 + FAST_DIVISION_CALL_WORDS,
            // Metal's composition where its `exp2` is the device's (`metal_exp`): five products
            // and sums, two fused ones, the device's `exp2`, and the ends' tests: 136 words on
            // `float4`.
            FloatFn::Exp | FloatFn::Exp10 => 128,
            // Metal's composition where its `log2` is the device's (`metal_log`): the device's
            // `log2` and the product, the polynomial's fused steps, and the range test.
            FloatFn::Log | FloatFn::Log10 => 200,
            // The exponent's clamp and the bits' fields, tests and selects (`ldexp_bits`).
            FloatFn::Ldexp => 160,
            _ => 48,
        };
        // A routine's calls, one per lane; the routine itself is the module's. A `half` function's
        // zero select (`half_zero`), its operand below a magnitude (`half_small`), `half` `cospi`'s
        // zero sign (`half_cospi_zero`), and `half` `asin`'s NaN sign (`half_signed_nan`).
        let base = base
            + libm::uses(self, f)
                .iter()
                .map(|r| libm::use_words(f.lanes, r.arity()))
                .sum::<u32>()
            + half_zero(self, f).map_or(0, |_| 24)
            + half_small(self, f).map_or(0, |_| 40)
            + if half && self == FloatFn::CosPi {
                96
            } else {
                0
            }
            + if half_signed_nan(self, f) { 40 } else { 0 };
        let roundings = match self {
            _ if self.closed_on_narrow() => 0,
            FloatFn::Mix => 3,
            FloatFn::MulAdd => 2,
            _ => 1,
        };
        base + roundings * narrow::round_words(f)
    }
}

/// Emit float function `op` on `args` (floats of form `f`; `Ldexp`'s exponent is a held `i32`
/// and `Dot` gives the scalar), calling `lib`'s routines where the device's instruction does not
/// answer the function. Metal's operand-picking functions (`metal_min_max`, `metal_median3`) test
/// their operands for NaNs by `test`, the test the entry's float modes leave exact
/// ([`NanTest::for_modes`]). A `half` `fmod` answers Metal's zero remainder (`half_fmod_zero`).
pub fn float_fn(
    fb: &mut FunctionBuilder<'_>,
    op: FloatFn,
    f: Form,
    contract: Contraction,
    test: NanTest,
    args: &[Value],
    lib: &Lib<'_>,
) -> BResult<Value> {
    use FloatFn as F;
    let r = float_fn_core(fb, op, f, contract, test, args, lib)?;
    let r = match op {
        _ if op.closed_on_narrow() => r,
        // The device's order: a `half` `mix`, the device's `FMix` in `float`, rounds once.
        F::Mix if f.float_fmt() == Some(FloatFmt::Half) && lib.orders_natively() => {
            narrow::round(fb, f, r)?
        }
        // Rounded per step below.
        F::Mix | F::MulAdd | F::Fract | F::FastFract { .. } | F::FlaggedFract => r,
        F::Dot => narrow::round(fb, f.scalar_form(), r)?,
        // The square root of a narrow value is a zero or at least `2^-12`.
        F::Sqrt => narrow::round_exact_below_normal(fb, f, r)?,
        F::Fmod => {
            let r = narrow::round(fb, f, r)?;
            half_fmod_zero(fb, f, arg(args, 0)?, r)?
        }
        _ => narrow::round(fb, f, r)?,
    };
    let r = match half_zero(op, f) {
        Some(zero) => half_zero_answer(fb, f, zero, arg(args, 0)?, r)?,
        None => r,
    };
    let r = match half_small(op, f) {
        Some(limit) => half_small_answer(fb, f, limit, arg(args, 0)?, r)?,
        None => r,
    };
    let r = if f.float_fmt() == Some(FloatFmt::Half) && op == F::CosPi {
        half_cospi_zero(fb, f, arg(args, 0)?, r)?
    } else {
        r
    };
    if half_signed_nan(op, f) {
        half_signed_nan_answer(fb, f, arg(args, 0)?, r)
    } else {
        Ok(r)
    }
}

/// The `float` function whose answer, rounded to `half`, is Metal's `half` function `op` on form
/// `f`, where that is not `op`'s own (`op` otherwise, and on another format): Metal's `half`
/// `exp`, `exp10`, `log`, `log2`, `log10`, `cosh`, `pow`, and `powr` are its fast `float`
/// functions of the widened operands, rounded, and so is its `sinh` from `2^-12`
/// (`half_small`). Each is Metal's word at every `half` operand of our probe (the M3; the
/// M4's words, are the same), `pow` and `powr` at each of 65,536 operand pairs,
/// and the precise function is not at 2 operands of `cosh`, 3 pairs of `pow` and `powr`, and 4
/// operands of `sinh`, whose fast one is again Metal's with the translation's `exp2` and `log2`
/// in place of Metal's. Its `tanh` is `half_tanh`, and its `atan`, `atan2`, `tan`, `sin`, `cos`,
/// and `asinh` are its own `half` forms, which no `float` function rounds to: `libm::uses` gives
/// them their routines.
pub(crate) fn metal_half(op: FloatFn, f: Form) -> FloatFn {
    if f.float_fmt() != Some(FloatFmt::Half) {
        return op;
    }
    match op {
        FloatFn::Exp => FloatFn::FastExp,
        FloatFn::Exp10 => FloatFn::FastExp10,
        FloatFn::Log => FloatFn::FastLog,
        FloatFn::Log2 => FloatFn::FastLog2,
        FloatFn::Log10 => FloatFn::FastLog10,
        FloatFn::Sinh => FloatFn::FastSinh,
        FloatFn::Cosh => FloatFn::FastCosh,
        FloatFn::Pow => FloatFn::FastPow,
        FloatFn::Powr => FloatFn::FastPowr,
        _ => op,
    }
}

/// Metal's `half` `tanh(x)` on `half` lanes `x` (form `f`, held in `f32`) before its rounding:
/// `(t - 1) · rcp(t + 1)` of `t = exp2(x · RN(2 log2 e))`, MSL Table 8.2's `fast::tanh` of
/// Metal's `exp2` and reciprocal (the routines `libm::Fun::Exp2` and `libm::Fun::Recip`, or the
/// device's instructions where they are Metal's), and `±1` from `|x| = 16`, where the answer
/// rounds to `±1` (the quotient is a NaN past `x ≈ 44`, where Metal's `half` `tanh` is `1`).
/// Below `2^-13` the operand itself (`half_small`). Every `half` of our probes is
/// this, with the translation's `exp2` and reciprocal in place of Metal's (and with a correctly
/// rounded reciprocal or quotient in place of the reciprocal); Metal's precise `float` `tanh`,
/// rounded, is not at 7 of them, nor its fast one, rounded, below `2^-13` and past `44`.
fn half_tanh(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, lib: &Lib<'_>) -> BResult<Value> {
    let exact = Contraction::Forbidden;
    let k = wide_splat(fb, f, 2.0 * std::f64::consts::LOG2_E)?;
    let p = fb.binary_with(BinOp::FMul, x, k, exact)?;
    let t = routine_or(
        fb,
        lib,
        FloatFn::Tanh,
        f,
        libm::Fun::Exp2,
        GlslOp::Exp2,
        &[p],
    )?;
    let one = wide_splat(fb, f, 1.0)?;
    let num = fb.binary_with(BinOp::FSub, t, one, exact)?;
    let den = fb.binary_with(BinOp::FAdd, t, one, exact)?;
    let recip = libm::Fun::Recip;
    let r = if libm::uses(FloatFn::Tanh, f).contains(&recip) && !lib.device(recip) {
        lib.call(fb, recip, f, &[den])?
    } else {
        fb.binary_with(BinOp::FDiv, one, den, exact)?
    };
    let q = fb.binary_with(BinOp::FMul, num, r, exact)?;
    let ax = glsl(fb, GlslOp::FAbs, &[x])?;
    let limit = wide_splat(fb, f, 16.0)?;
    let far = bin(fb, BinOp::FOrdGreaterThanEqual, ax, limit)?;
    let unit = copysign(fb, f, one, x)?;
    fb.select(far, unit, q)
}

/// The magnitude below which Metal's `half` function `op` (form `f`) answers its operand
/// itself, where the `float` function it rounds does not: `sinh` below `2^-12` and `tanh` below
/// `2^-13`, where `fast::sinh(x)` and `fast::tanh(x)`, of a difference of nearly equal
/// exponentials, lose the operand's low bits and the exact value rounds to the operand (every
/// such `half` of our probes). `None` for another function or format.
fn half_small(op: FloatFn, f: Form) -> Option<f64> {
    if f.float_fmt() != Some(FloatFmt::Half) {
        return None;
    }
    match op {
        FloatFn::Sinh => Some(2f64.powi(-12)),
        FloatFn::Tanh => Some(2f64.powi(-13)),
        _ => None,
    }
}

/// `r`, the answer of a `half` function of `x` (form `f`), with `x` itself where `|x|` is below
/// `limit` (`half_small`).
fn half_small_answer(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    limit: f64,
    x: Value,
    r: Value,
) -> BResult<Value> {
    let ax = glsl(fb, GlslOp::FAbs, &[x])?;
    let l = fsplat(fb, f, limit)?;
    let small = bin(fb, BinOp::FOrdLessThan, ax, l)?;
    fb.select(small, x, r)
}

/// `r`, Metal's `half` `cospi(x)` (form `f`), with its zero's sign: at `x = n + 1/2` Metal
/// answers `-0` where `n = floor(x)` is even and `+0` where it is odd, the sign of
/// `-sinpi(x - 1/2)` of its `half` `sinpi`, whose zero at an odd integer is `-0` (each of the
/// 4,096 such `half` operands of our probes), where its `float` `cospi` answers
/// `+0`. `n` and `n / 2` are exact in the container. The zero is chosen in the bits: in a
/// fast-math entry (no `SignedZeroInfNanPreserve`) a float select between `-0` and `+0` may be
/// folded, and the RTX 3090 answered `-0` at the odd `n` of case 32931370.
fn half_cospi_zero(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, r: Value) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let rb = fb.convert(ConvOp::Bitcast, r, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let rm = bin(fb, BinOp::BitwiseAnd, rb, magnitude)?;
    let uz = splat(fb, u, 0)?;
    let zero = bin(fb, BinOp::IEqual, rm, uz)?;
    let n = glsl(fb, GlslOp::Floor, &[x])?;
    let half = fsplat(fb, f, 0.5)?;
    let h = bin(fb, BinOp::FMul, n, half)?;
    let hf = glsl(fb, GlslOp::Floor, &[h])?;
    let even = bin(fb, BinOp::FOrdEqual, hf, h)?;
    let minus = splat(fb, u, 0x8000_0000)?;
    let signed = fb.select(even, minus, uz)?;
    let b = fb.select(zero, signed, rb)?;
    fb.convert(ConvOp::Bitcast, b, ft)
}

/// Whether float function `op` on form `f` is Metal's `half` `asin`, whose NaN at `|x| > 1` has
/// `x`'s sign in both math modes (cases 48500c31 and 32931370), where its `float` precise `asin`'s,
/// which the routine and a device whose functions are Metal's answer, is the quiet NaN without a
/// sign (case 087758cb).
fn half_signed_nan(op: FloatFn, f: Form) -> bool {
    f.float_fmt() == Some(FloatFmt::Half) && op == FloatFn::Asin
}

/// `r`, the answer of a `half` function of `x` (form `f`), with `x`'s sign where `|x| > 1`.
fn half_signed_nan_answer(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    r: Value,
) -> BResult<Value> {
    let ax = glsl(fb, GlslOp::FAbs, &[x])?;
    let one = fsplat(fb, f, 1.0)?;
    let domain = bin(fb, BinOp::FOrdGreaterThan, ax, one)?;
    let signed = copysign(fb, f, r, x)?;
    fb.select(domain, signed, r)
}

/// Metal's answer for a zero operand of a `half` function where it is not its `float`
/// function's, which the device's instruction and the routines compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HalfZero {
    /// `+0`, whatever the zero's sign.
    Positive,
    /// The zero itself.
    Operand,
}

/// The [`HalfZero`] of float function `op` on form `f`, where it is measured: Metal's `half`
/// `sin`, `tan`, and `asin` of either zero are `+0`, and its `atan` of `-0` is `-0`, in both
/// math modes (the half probe, cases dc0fee8f and 3ef7c1c5), where its `float` `sin(-0)` and
/// `tan(-0)` are `-0`, precise `asin(-0)` is `-0`, and precise `atan(-0)` is `+0` (cases 087758cb
/// and cc234732); and its `half` `sinpi(-0)` is `-0`, where its `float` one is `+0`. `None` for any
/// other function or format.
fn half_zero(op: FloatFn, f: Form) -> Option<HalfZero> {
    if f.float_fmt() != Some(FloatFmt::Half) {
        return None;
    }
    match op {
        FloatFn::Sin | FloatFn::Tan | FloatFn::Asin => Some(HalfZero::Positive),
        FloatFn::Atan | FloatFn::SinPi => Some(HalfZero::Operand),
        _ => None,
    }
}

/// `r`, the answer of a `half` function of `x` (form `f`), with a zero operand's answer
/// `zero` selected (an operand is a zero when its widened value is: a `half` has no `f32`
/// denormal).
fn half_zero_answer(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    zero: HalfZero,
    x: Value,
    r: Value,
) -> BResult<Value> {
    let z = fsplat(fb, f, 0.0)?;
    let at_zero = bin(fb, BinOp::FOrdEqual, x, z)?;
    let answer = match zero {
        HalfZero::Positive => z,
        HalfZero::Operand => x,
    };
    fb.select(at_zero, answer, r)
}

/// [`float_fn`] before the final rounding to a narrow format.
fn float_fn_core(
    fb: &mut FunctionBuilder<'_>,
    op: FloatFn,
    f: Form,
    contract: Contraction,
    test: NanTest,
    args: &[Value],
    lib: &Lib<'_>,
) -> BResult<Value> {
    use FloatFn as F;
    let a = arg(args, 0)?;
    // Metal's `half` functions, by the `float` function whose rounded answer each is.
    if f.float_fmt() == Some(FloatFmt::Half) && op == F::Tanh {
        return half_tanh(fb, f, a, lib);
    }
    let op = metal_half(op, f);
    // The device's instruction `o`, or `op`'s routine where the device does not answer it.
    let lib1 = |fb: &mut FunctionBuilder<'_>, o: GlslOp, x: Value| -> BResult<Value> {
        match libm::uses(op, f) {
            [r] if !lib.device(*r) => lib.call(fb, *r, f, &[x]),
            _ => glsl(fb, o, &[x]),
        }
    };
    let g1 = |fb: &mut FunctionBuilder<'_>, o: GlslOp| glsl(fb, o, &[a]);
    let times = |fb: &mut FunctionBuilder<'_>, k: f64| -> BResult<Value> {
        let c = wide_splat(fb, f, k)?;
        bin(fb, BinOp::FMul, a, c)
    };
    match op {
        F::Abs if sign_on_bits(f.float_fmt()) => sign_bits(fb, f, a, SignBit::Clear),
        F::Abs => g1(fb, GlslOp::FAbs),
        F::Floor => g1(fb, GlslOp::Floor),
        F::Ceil => g1(fb, GlslOp::Ceil),
        F::Trunc => g1(fb, GlslOp::Trunc),
        F::Rint => g1(fb, GlslOp::RoundEven),
        F::Round => {
            // t = trunc(x); |x - t| >= 0.5 ? t + (x < 0 ? -1 : 1) : t. `x - t` is exact.
            let t = g1(fb, GlslOp::Trunc)?;
            let d = bin(fb, BinOp::FSub, a, t)?;
            let d = glsl(fb, GlslOp::FAbs, &[d])?;
            let half = fsplat(fb, f, 0.5)?;
            let up = bin(fb, BinOp::FOrdGreaterThanEqual, d, half)?;
            let z = fsplat(fb, f, 0.0)?;
            let neg = bin(fb, BinOp::FOrdLessThan, a, z)?;
            let (m1, p1) = (fsplat(fb, f, -1.0)?, fsplat(fb, f, 1.0)?);
            let step = fb.select(neg, m1, p1)?;
            let away = bin(fb, BinOp::FAdd, t, step)?;
            let r = fb.select(up, away, t)?;
            positive_zero(fb, f, r)
        }
        F::Fract => metal_fract(fb, f, a, true, false),
        F::FastFract { infinite_nan } => metal_fract(fb, f, a, false, infinite_nan),
        F::FlaggedFract => {
            let r = metal_fract(fb, f, a, true, false)?;
            nonfinite_below_one(fb, f, a, r)
        }
        F::Sign => metal_sign(fb, f, a),
        // Metal's `clamp(x, +0, 1)` at every operand: the device's `NClamp` where the minimum and
        // maximum functions take it (measured to be Metal's, or native device maths,
        // `PrecisionNote::NativeMinMax`).
        F::Saturate if metal_picks(f.float_fmt()) && lib.device_min_max() => {
            let (z, one) = (fsplat(fb, f, 0.0)?, fsplat(fb, f, 1.0)?);
            glsl(fb, GlslOp::NClamp, &[a, z, one])
        }
        F::Saturate => metal_saturate(fb, f, a),
        // The device's `NMin`, `NMax` and `NClamp` where they are measured to be Metal's, the arms
        // below.
        F::Min | F::Max | F::Clamp | F::Min3 | F::Max3
            if metal_picks(f.float_fmt()) && !lib.device_min_max() =>
        {
            metal_min_max(fb, op, f, test, args)
        }
        F::Min | F::MinNum => glsl(fb, GlslOp::NMin, &[a, arg(args, 1)?]),
        F::Max | F::MaxNum => glsl(fb, GlslOp::NMax, &[a, arg(args, 1)?]),
        F::Clamp => glsl(fb, GlslOp::NClamp, &[a, arg(args, 1)?, arg(args, 2)?]),
        F::Min3 | F::Max3 => {
            let o = if op == F::Min3 {
                GlslOp::NMin
            } else {
                GlslOp::NMax
            };
            let r = glsl(fb, o, &[a, arg(args, 1)?])?;
            glsl(fb, o, &[r, arg(args, 2)?])
        }
        F::Median3 | F::FastMedian3 => {
            let (b, c) = (arg(args, 1)?, arg(args, 2)?);
            metal_median3(fb, f, test, [a, b, c], op == F::FastMedian3)
        }
        // `float` and `half` take Metal's fused form ([`metal_mix`]), a strategy of its own with
        // its fused multiply-adds' routines; the device's order takes the device's `FMix`
        // (`PrecisionNote::NativeMix`).
        F::Mix if matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) => {
            if !lib.orders_natively() {
                return Err(BuildError::Type(TypeRule::FunctionSignature));
            }
            glsl(fb, GlslOp::FMix, &[a, arg(args, 1)?, arg(args, 2)?])
        }
        F::Mix => {
            // The blend `x * (1 - t) + y * t` on `bfloat` and `double`, which Metal's fused form is
            // not measured on: its multiplies and add carry the call's contraction, and each step
            // rounds to a narrow format. `1 - t` and the final sum add two values of the format.
            let (y, t) = (arg(args, 1)?, arg(args, 2)?);
            let one = fsplat(fb, f, 1.0)?;
            let w = fb.binary_with(BinOp::FSub, one, t, contract)?;
            let w = narrow::round_exact_below_normal(fb, f, w)?;
            let p = fb.binary_with(BinOp::FMul, a, w, contract)?;
            let p = narrow::round(fb, f, p)?;
            let q = fb.binary_with(BinOp::FMul, y, t, contract)?;
            let q = narrow::round(fb, f, q)?;
            let r = fb.binary_with(BinOp::FAdd, p, q, contract)?;
            narrow::round_exact_below_normal(fb, f, r)
        }
        F::Ldexp => ldexp(fb, f, a, arg(args, 1)?),
        F::CopySign => copysign(fb, f, a, arg(args, 1)?),
        F::MulAdd => {
            // A narrow form takes the unfused choice, which is exact (each step rounds once; the
            // sum adds two values of the format).
            let p = bin(fb, BinOp::FMul, a, arg(args, 1)?)?;
            let p = narrow::round(fb, f, p)?;
            let r = bin(fb, BinOp::FAdd, p, arg(args, 2)?)?;
            narrow::round_exact_below_normal(fb, f, r)
        }
        F::Sqrt => lib1(fb, GlslOp::Sqrt, a),
        F::Fma if f.float_fmt() == Some(FloatFmt::BFloat) => {
            narrow::bf16_fma(fb, f, a, arg(args, 1)?, arg(args, 2)?)
        }
        F::Fma => glsl(fb, GlslOp::Fma, &[a, arg(args, 1)?, arg(args, 2)?]),
        F::Fmod => {
            let y = arg(args, 1)?;
            match libm::uses(op, f) {
                [r] if !lib.device(*r) => lib.call(fb, *r, f, &[a, y]),
                _ => bin(fb, BinOp::FRem, a, y),
            }
        }
        F::Dot => bin(fb, BinOp::Dot, a, arg(args, 1)?),
        F::Rsqrt => lib1(fb, GlslOp::InverseSqrt, a),
        F::FastRsqrt => g1(fb, GlslOp::InverseSqrt),
        // The device's order: its `Sqrt`.
        F::FastSqrt if lib.orders_natively() => g1(fb, GlslOp::Sqrt),
        F::FastSqrt => {
            let r = g1(fb, GlslOp::InverseSqrt)?;
            fast_sqrt_of(fb, f, a, r)
        }
        // A device whose functions are Metal's answers precise `exp` in a precise module; in a
        // fast-math one, Metal's own composition of its `exp2` does.
        F::Exp => match libm::uses(op, f) {
            [r] if !lib.device(*r) => lib.call(fb, *r, f, &[a]),
            [_] if !lib.metal_in(libm::Mode::Precise) => {
                metal_exp(fb, f, a, std::f64::consts::LOG2_E)
            }
            _ => g1(fb, GlslOp::Exp),
        },
        F::Exp2 => lib1(fb, GlslOp::Exp2, a),
        // Likewise precise `log`: in a fast-math module Metal's composition of its `log2`.
        F::Log => match libm::uses(op, f) {
            [r] if !lib.device(*r) => lib.call(fb, *r, f, &[a]),
            [_] if !lib.metal_in(libm::Mode::Precise) => {
                metal_log(fb, f, a, &libm::log::LOG_NEAR, libm::log::LN_2_BITS)
            }
            _ => g1(fb, GlslOp::Log),
        },
        F::Log2 | F::FastLog2 => lib1(fb, GlslOp::Log2, a),
        // No instruction is Metal's `exp10` or `log10`: their routines, or Metal's own
        // compositions of the device's `exp2` and `log2` where those are Metal's.
        F::Exp10 | F::Log10 => match libm::uses(op, f) {
            [r] if !lib.device(*r) => lib.call(fb, *r, f, &[a]),
            _ if op == F::Exp10 => metal_exp(fb, f, a, std::f64::consts::LOG2_10),
            _ => metal_log(fb, f, a, &libm::log::LOG10_NEAR, libm::log::LOG10_2_BITS),
        },
        // Metal's fast compositions of its `exp2` and `log2` (MSL Table 8.2).
        F::FastExp | F::FastExp10 => {
            let k = if op == F::FastExp {
                std::f64::consts::LOG2_E
            } else {
                std::f64::consts::LOG2_10
            };
            let x = times(fb, k)?;
            routine_or(fb, lib, op, f, libm::Fun::Exp2, GlslOp::Exp2, &[x])
        }
        F::FastLog | F::FastLog10 => {
            let l = routine_or(fb, lib, op, f, libm::Fun::Log2, GlslOp::Log2, &[a])?;
            let k = if op == F::FastLog {
                std::f64::consts::LN_2
            } else {
                std::f64::consts::LOG10_2
            };
            let c = wide_splat(fb, f, k)?;
            fb.binary_with(BinOp::FMul, l, c, Contraction::Forbidden)
        }
        F::Sin => lib1(fb, GlslOp::Sin, a),
        F::Cos => lib1(fb, GlslOp::Cos, a),
        F::FastSin => {
            let r = lib1(fb, GlslOp::Sin, a)?;
            fast_trig_flush(fb, f, a, r)
        }
        F::FastCos => {
            let r = lib1(fb, GlslOp::Cos, a)?;
            fast_trig_flush(fb, f, a, r)
        }
        F::Tan => lib1(fb, GlslOp::Tan, a),
        // A device whose functions are Metal's in a fast-math module: its `Tan` is `fast::tan`.
        F::FastTan { .. } if lib.metal_in(libm::Mode::Fast) => g1(fb, GlslOp::Tan),
        // Its quotient is a call of the fast division's routine, which the caller declares:
        // [`fast_tan`].
        F::FastTan { .. } => Err(BuildError::Type(TypeRule::FunctionSignature)),
        F::Asin | F::FastAsin => lib1(fb, GlslOp::Asin, a),
        F::Acos | F::FastAcos => lib1(fb, GlslOp::Acos, a),
        F::Atan | F::FastAtan => lib1(fb, GlslOp::Atan, a),
        F::Sinh => lib1(fb, GlslOp::Sinh, a),
        F::Cosh => lib1(fb, GlslOp::Cosh, a),
        F::Tanh => lib1(fb, GlslOp::Tanh, a),
        F::Asinh => lib1(fb, GlslOp::Asinh, a),
        F::Acosh => lib1(fb, GlslOp::Acosh, a),
        F::Atanh => lib1(fb, GlslOp::Atanh, a),
        // A device whose functions are Metal's in a fast-math module: its instruction is Metal's
        // fast function. Elsewhere Metal's formula (MSL Table 8.2).
        F::FastSinh if lib.metal_in(libm::Mode::Fast) => g1(fb, GlslOp::Sinh),
        F::FastCosh if lib.metal_in(libm::Mode::Fast) => g1(fb, GlslOp::Cosh),
        F::FastSinh | F::FastCosh => fast_sinh_cosh(fb, op, f, a, lib),
        // Metal's own forms of its fast `asinh` and `acosh` (`libm::hyper::fast_arc`), or the
        // device's instructions where they are Metal's fast functions.
        F::FastAsinh => lib1(fb, GlslOp::Asinh, a),
        F::FastAcosh => lib1(fb, GlslOp::Acosh, a),
        // Their quotients are calls of the fast division's routine, which the caller declares:
        // [`fast_tanh`], [`fast_atanh`].
        F::FastTanh { .. } | F::FastAtanh { .. } => {
            Err(BuildError::Type(TypeRule::FunctionSignature))
        }
        F::SinPi | F::CosPi | F::TanPi | F::FastSinPi | F::FastCosPi | F::FastTanPi => {
            let which = match op {
                F::SinPi | F::FastSinPi => PiScaled::Sin,
                F::CosPi | F::FastCosPi => PiScaled::Cos,
                _ => PiScaled::Tan,
            };
            // Metal's own form, the routine; where none answers (a `bfloat`, a fast call where the
            // device's functions are Metal's, or a unit test that leaves them to the device), the
            // device's `Sin` and `Cos` of `float(π) r`.
            match libm::uses(op, f) {
                [fun] if !lib.device(*fun) => lib.call(fb, *fun, f, &[a]),
                _ => pi_scaled(
                    fb,
                    f,
                    which,
                    a,
                    &|fb, residue| {
                        let pi = wide_splat(fb, f, std::f64::consts::PI)?;
                        let angle = bin(fb, BinOp::FMul, pi, residue)?;
                        Ok((
                            glsl(fb, GlslOp::Sin, &[angle])?,
                            glsl(fb, GlslOp::Cos, &[angle])?,
                        ))
                    },
                    &|fb, num, den| bin(fb, BinOp::FDiv, num, den),
                ),
            }
        }
        F::Powr => {
            let y = arg(args, 1)?;
            let r = routine_or(fb, lib, op, f, libm::Fun::Powr, GlslOp::Pow, &[a, y])?;
            powr_nan(fb, f, a, y, r)
        }
        F::FastPowr => {
            let y = arg(args, 1)?;
            let r = exp2_of_product(fb, lib, op, f, y, a)?;
            fast_powr_special(fb, f, a, y, r)
        }
        // C99 Annex F's special cases around the magnitude of `|x|`, which `GLSL.std.450 Pow`
        // defines only for a positive base.
        F::Pow | F::FastPow => {
            let y = arg(args, 1)?;
            let ax = glsl(fb, GlslOp::FAbs, &[a])?;
            let magnitude = if op == F::Pow {
                routine_or(fb, lib, op, f, libm::Fun::Powr, GlslOp::Pow, &[ax, y])?
            } else {
                let r = exp2_of_product(fb, lib, op, f, y, ax)?;
                infinite_product(fb, f, y, ax, r)?
            };
            pow_special(fb, f, a, y, magnitude, op == F::Pow)
        }
        F::Atan2 => {
            let x = arg(args, 1)?;
            let generic = match libm::uses(op, f) {
                [r] if !lib.device(*r) => lib.call(fb, *r, f, &[a, x])?,
                _ => glsl(fb, GlslOp::Atan2, &[a, x])?,
            };
            atan2_special(fb, f, a, x, generic)
        }
        F::FastAtan2 { reciprocal_flushed } => {
            // Metal's fast `atan` of its fast quotient `y · rcp(x)`: each the device's where it is
            // Metal's in the module's mode, the routines otherwise.
            let atan = |fb: &mut FunctionBuilder<'_>, q: Value| {
                routine_or(
                    fb,
                    lib,
                    op,
                    f,
                    libm::Fun::Atan(libm::Mode::Fast),
                    GlslOp::Atan,
                    &[q],
                )
            };
            let divide = |fb: &mut FunctionBuilder<'_>, n: Value, d: Value| -> BResult<Value> {
                let recip = libm::Fun::Recip;
                if libm::uses(op, f).contains(&recip) && !lib.device(recip) {
                    let r = lib.call(fb, recip, f, &[d])?;
                    fb.binary_with(BinOp::FMul, n, r, Contraction::Forbidden)
                } else {
                    bin(fb, BinOp::FDiv, n, d)
                }
            };
            fast_atan2(fb, f, a, arg(args, 1)?, reciprocal_flushed, &atan, &divide)
        }
    }
}

/// `llvm.copysign(a, b)` on floats of form `f` held in an `f32` container: `a`'s bits with the
/// sign bit replaced by `b`'s (LangRef), so a NaN keeps its payload and a zero takes `b`'s sign
/// whatever the device's float arithmetic does with them. A narrow value's container is the
/// widened value, whose sign bit is the format's.
fn copysign(fb: &mut FunctionBuilder<'_>, f: Form, a: Value, b: Value) -> BResult<Value> {
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let ab = fb.convert(ConvOp::Bitcast, a, ut)?;
    let bb = fb.convert(ConvOp::Bitcast, b, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let ab = bin(fb, BinOp::BitwiseAnd, ab, magnitude)?;
    let bb = bin(fb, BinOp::BitwiseAnd, bb, sign)?;
    let r = bin(fb, BinOp::BitwiseOr, ab, bb)?;
    let ft = ty(fb, f)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// Whether `fneg` and `fabs` on lanes of format `fmt` move the sign as an integer on the bits
/// ([`sign_bits`]): on `float`, where Metal flips or clears the sign bit alone, a NaN keeping its
/// payload and a denormal kept, with and without fast-math flags (`nan_signbits`,
/// `nan_signbits_fast`, every lane). The devices' `OpFNegate` and `FAbs` differ there: RADV's
/// `fneg` keeps a NaN's sign, NVIDIA's `fabs` answers `0x7FFFFFFF`, and RADV and ANV flush a
/// denormal. A `half` is held widened and its `fneg` and `fabs` answer the canonical `half` NaN
/// (class D); `bfloat` and `double` are not measured.
pub fn sign_on_bits(fmt: Option<FloatFmt>) -> bool {
    fmt == Some(FloatFmt::F32)
}

/// What [`sign_bits`] does to the sign bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignBit {
    /// `fneg`.
    Flip,
    /// `fabs`.
    Clear,
}

/// `x` (floats of form `f`, `f32` lanes) with its sign bit flipped or cleared as an integer on
/// the bits, whatever the device's float modes: Metal's `float` `fneg` and `fabs`
/// ([`sign_on_bits`]).
pub fn sign_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, op: SignBit) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let r = match op {
        SignBit::Flip => {
            let sign = splat(fb, u, 0x8000_0000)?;
            bin(fb, BinOp::BitwiseXor, bits, sign)?
        }
        SignBit::Clear => {
            let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
            bin(fb, BinOp::BitwiseAnd, bits, magnitude)?
        }
    };
    let ft = ty(fb, f)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// The constant `k` at the container's precision: an internal constant of a sequence that runs
/// in the container (`f32` for a narrow form) before its final rounding.
fn wide_splat(fb: &mut FunctionBuilder<'_>, f: Form, k: f64) -> BResult<Value> {
    let wide = match f.float_fmt() {
        Some(n) if n.narrow() => f.with(Held::Float(FloatFmt::F32)),
        _ => f,
    };
    fsplat(fb, wide, k)
}

// ---- Metal's functions at their special operands ----

/// Whether lanes of `x` (floats of form `f`) are zeros or denormals of an `f32`: Metal's `float`
/// functions read an operand whose exponent field is zero as a zero (its denormals are flushed,
/// A1 cases aed3faaa, 87d13612, and f9a12252). Decided on the exponent field, so the answer is
/// the same whatever the device's own denormal mode. `None` for another format: a widened `half`
/// is never an `f32` denormal, and whether Metal flushes `bfloat` denormals is not measured.
fn flushed(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Option<Value>> {
    if f.float_fmt() != Some(FloatFmt::F32) {
        return Ok(None);
    }
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let exp = splat(fb, u, 0x7F80_0000)?;
    let e = bin(fb, BinOp::BitwiseAnd, bits, exp)?;
    let zero = splat(fb, u, 0)?;
    Ok(Some(bin(fb, BinOp::IEqual, e, zero)?))
}

/// Whether lanes of `x` (floats of form `f`) are `f32` denormals: a nonzero magnitude whose
/// exponent field is zero, told by the bits whatever the device's own denormal mode. `None` for
/// another format, as [`flushed`].
fn denormal_lanes(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Option<Value>> {
    if f.float_fmt() != Some(FloatFmt::F32) {
        return Ok(None);
    }
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let a = bin(fb, BinOp::BitwiseAnd, bits, magnitude)?;
    let one = splat(fb, u, 1)?;
    let below = bin(fb, BinOp::ISub, a, one)?;
    let largest = splat(fb, u, 0x007F_FFFF)?;
    Ok(Some(bin(fb, BinOp::ULessThan, below, largest)?))
}

/// Whether lanes of `x` (floats of form `f`, held in a 32-bit container) have the bits `bits`:
/// an integer comparison, which no float mode lets a device fold.
fn has_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, bits: u64) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let k = splat(fb, u, bits)?;
    bin(fb, BinOp::IEqual, b, k)
}

/// `r` (`round`'s result, floats of form `f`) with `+0` for `-0` on `float` and `half`: Metal's
/// `round` never answers `-0` (`-0.3` and `-0` round to `+0`, precise and fast; `half` in both flag
/// states, `nan_half` and `nan_half_fast`), as `trunc(x)` plus a `+0` step does. Moved in the bits,
/// so no signed-zero mode folds it; `bfloat` keeps `r` (not measured).
fn positive_zero(fb: &mut FunctionBuilder<'_>, f: Form, r: Value) -> BResult<Value> {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return Ok(r);
    }
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let b = fb.convert(ConvOp::Bitcast, r, ut)?;
    let minus_zero = splat(fb, u, 0x8000_0000)?;
    let zero = splat(fb, u, 0)?;
    let is = bin(fb, BinOp::IEqual, b, minus_zero)?;
    let b = fb.select(is, zero, b)?;
    fb.convert(ConvOp::Bitcast, b, ft)
}

/// `r` (a `half` `fmod`'s remainder of `x`, floats of form `f`) with Metal's zero: a zero
/// remainder of a nonzero dividend is `+0`, and a zero dividend's is the dividend itself
/// (`fmod(-2, 0.5)` is `+0`, `fmod(-0, 1)` is `-0`), with or without fast-math flags
/// (`nan_fmod_zero`, `nan_fmod_zero_fast`). A `float` remainder keeps the dividend's sign, as C's
/// `fmod` does, and so does Metal's: it and every other format keep `r`. Tested on the bits, so no
/// signed-zero mode folds it.
fn half_fmod_zero(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, r: Value) -> BResult<Value> {
    if f.float_fmt() != Some(FloatFmt::Half) {
        return Ok(r);
    }
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let zero = splat(fb, u, 0)?;
    let rb = fb.convert(ConvOp::Bitcast, r, ut)?;
    let rm = bin(fb, BinOp::BitwiseAnd, rb, magnitude)?;
    let r_zero = bin(fb, BinOp::IEqual, rm, zero)?;
    let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
    let xm = bin(fb, BinOp::BitwiseAnd, xb, magnitude)?;
    let x_zero = bin(fb, BinOp::IEqual, xm, zero)?;
    let z = fb.select(x_zero, xb, zero)?;
    let b = fb.select(r_zero, z, rb)?;
    fb.convert(ConvOp::Bitcast, b, ft)
}

/// `r` (a `half` `fract` with every fast-math flag, floats of form `f`) with the largest `half`
/// below one where `x` is a NaN or an infinity, as Metal answers there under fast math
/// (`nan_half_fast`: `0x3BFF` on all 32 NaN and 7 infinite lanes): its `x - floor(x)` is a NaN, and
/// its bound `fmin` returns the other operand. Told by the exponent field, so no float mode folds
/// the test.
fn nonfinite_below_one(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    r: Value,
) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let exp = splat(fb, u, 0x7F80_0000)?;
    let e = bin(fb, BinOp::BitwiseAnd, b, exp)?;
    let nonfinite = bin(fb, BinOp::IEqual, e, exp)?;
    let bound = splat(fb, f, below_one_bits(f))?;
    fb.select(nonfinite, bound, r)
}

/// The classes of `powr`'s operands `x` and `y` (floats of form `f`, a `float` or a `half` in its
/// 32-bit container) that its special cases read, each told once from the bits and shared by
/// every rule that reads it: each operand's bits and magnitude's, whether it is a zero to Metal (an
/// `f32` denormal is one, [`metal_zero`]) and an infinity of either sign, and whether `x` is one.
/// Integer comparisons, which no float mode folds.
struct PowrOperands {
    x_bits: Value,
    x_magnitude: Value,
    x_zero: Value,
    x_inf: Value,
    x_one: Value,
    y_bits: Value,
    y_magnitude: Value,
    y_zero: Value,
    y_inf: Value,
}

impl PowrOperands {
    fn of(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, y: Value) -> BResult<Self> {
        let u = u32_form(f);
        let ut = ty(fb, u)?;
        let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
        let inf = splat(fb, u, 0x7F80_0000)?;
        let class = |fb: &mut FunctionBuilder<'_>, v: Value| -> BResult<[Value; 4]> {
            let b = fb.convert(ConvOp::Bitcast, v, ut)?;
            let m = bin(fb, BinOp::BitwiseAnd, b, magnitude)?;
            let zero = match f.float_fmt() {
                // The exponent field zero: the magnitude below the least normal's.
                Some(FloatFmt::F32) => {
                    let normal = splat(fb, u, 0x0080_0000)?;
                    bin(fb, BinOp::ULessThan, m, normal)?
                }
                _ => metal_zero(fb, f, v)?,
            };
            let is_inf = bin(fb, BinOp::IEqual, m, inf)?;
            Ok([b, m, zero, is_inf])
        };
        let [x_bits, x_magnitude, x_zero, x_inf] = class(fb, x)?;
        let [y_bits, y_magnitude, y_zero, y_inf] = class(fb, y)?;
        let one = splat(fb, u, u64::from(1f32.to_bits()))?;
        let x_one = bin(fb, BinOp::IEqual, x_bits, one)?;
        Ok(Self {
            x_bits,
            x_magnitude,
            x_zero,
            x_inf,
            x_one,
            y_bits,
            y_magnitude,
            y_zero,
            y_inf,
        })
    }

    /// Whether `log2(x)` is infinite: `x` a zero to Metal or an infinity.
    fn log_infinite(&self, fb: &mut FunctionBuilder<'_>) -> BResult<Value> {
        bin(fb, BinOp::LogicalOr, self.x_zero, self.x_inf)
    }

    /// Whether neither operand is a NaN: the larger magnitude's bits are an infinity's or less.
    fn numbers(&self, fb: &mut FunctionBuilder<'_>, f: Form) -> BResult<Value> {
        let larger = glsl(fb, GlslOp::UMax, &[self.x_magnitude, self.y_magnitude])?;
        let inf = splat(fb, u32_form(f), 0x7F80_0000)?;
        bin(fb, BinOp::ULessThanEqual, larger, inf)
    }

    /// Metal's NaN where `exp2(y · log2(x))` is one for operands that are not NaNs ([`powr_nan`]),
    /// `negative` where `x` is below zero as the mode reads it: `0 · inf` (`y` a zero, `log2(x)`
    /// infinite) and `inf · 0` (`y` infinite, `x` one).
    fn nan(
        &self,
        fb: &mut FunctionBuilder<'_>,
        negative: Value,
        log_infinite: Value,
    ) -> BResult<Value> {
        let zero_times_inf = bin(fb, BinOp::LogicalAnd, self.y_zero, log_infinite)?;
        let inf_times_zero = bin(fb, BinOp::LogicalAnd, self.y_inf, self.x_one)?;
        let nan = bin(fb, BinOp::LogicalOr, negative, zero_times_inf)?;
        bin(fb, BinOp::LogicalOr, nan, inf_times_zero)
    }

    /// IEEE's `exp2` of an infinite `y · log2(x)` (`x` not negative where it matters): `+0` where
    /// the product is `-inf`, `+inf` where it is `+inf`, by the signs of `log2(x)` (below zero
    /// where `|x|` is below one, a zero included) and of `y` (its sign bit).
    fn infinite_answer(&self, fb: &mut FunctionBuilder<'_>, f: Form) -> BResult<Value> {
        let one = splat(fb, u32_form(f), u64::from(1f32.to_bits()))?;
        let l_negative = bin(fb, BinOp::ULessThan, self.x_magnitude, one)?;
        let y_negative = bin(fb, BinOp::INotEqual, self.y_bits, self.y_magnitude)?;
        let negative = bin(fb, BinOp::LogicalNotEqual, l_negative, y_negative)?;
        let (zero, inf) = (wide_splat(fb, f, 0.0)?, wide_splat(fb, f, f64::INFINITY)?);
        fb.select(negative, zero, inf)
    }
}

/// `powr(x, y)`'s result `r` (floats of form `f`) with Metal's canonical NaN where Metal's
/// `exp2(y * log2(x))` is a NaN for finite or infinite operands: `x` below zero (`-inf` included),
/// `y` zero with `x` a zero or an infinity (`0 * inf`), and `y` infinite with `x` one (`inf * 0`).
/// Devices answer `0` or `1` there (`GLSL.std.450 Pow` leaves `x < 0` and `x = 0, y <= 0`
/// undefined). The operands' classes are read from the bits once ([`PowrOperands`]), and an `f32`
/// denormal operand is a zero ([`metal_zero`]) to the zero tests, as Metal reads it. Metal's
/// precise `powr` reads the sign from the bits: a negative denormal's is NaN (case 7b283404 and
/// f6524207's `powr(-2^-149, 0.5)`, the M3 and the M4); its `fast::powr` is [`fast_powr_special`].
/// `float` and `half` (whose NaN lanes are the same rule's, `nan_half` and `nan_half_fast`, every
/// lane); `bfloat` keeps `r` (not measured).
pub(super) fn powr_nan(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    r: Value,
) -> BResult<Value> {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return Ok(r);
    }
    let c = PowrOperands::of(fb, f, x, y)?;
    // The sign bit of a nonzero `x`, a denormal's and a NaN's included: the bits above `-0`'s.
    let minus_zero = splat(fb, u32_form(f), 0x8000_0000)?;
    let negative = bin(fb, BinOp::UGreaterThan, c.x_bits, minus_zero)?;
    let log_infinite = c.log_infinite(fb)?;
    let nan = c.nan(fb, negative, log_infinite)?;
    let k = fsplat(fb, f, f64::NAN)?;
    fb.select(nan, k, r)
}

/// Metal's `fast::powr(x, y)` around `r`, the composition's `exp2(y · log2(x))`
/// ([`exp2_of_product`]): [`powr_nan`]'s NaN where `x` is below zero as `fast::powr` reads it
/// (a negative `f32` denormal is a zero, so `fast::powr(-2^-149, 0.5)` is `+0`, case f6524207 on
/// the M3 and the M4), over [`infinite_product`]'s IEEE value, both on one classification of the
/// operands ([`PowrOperands`]). The infinite product is selected wherever `log2(x)` or `y` is
/// infinite and neither operand is a NaN: the lanes of those where the product is `0 · inf`, and
/// those of a negative `x`, are the NaN rule's, which overrides them. `float` and `half`; `bfloat`
/// keeps `r` (not measured).
pub(super) fn fast_powr_special(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    r: Value,
) -> BResult<Value> {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return Ok(r);
    }
    let c = PowrOperands::of(fb, f, x, y)?;
    // `x` below zero and not a zero to Metal, compared as `fast::powr` compares: an `f32` at or
    // below the negative least normal (so neither a denormal nor a zero), a `half` below zero.
    let negative = if f.float_fmt() == Some(FloatFmt::F32) {
        let least = fsplat(fb, f, -f64::from(f32::MIN_POSITIVE))?;
        bin(fb, BinOp::FOrdLessThanEqual, x, least)?
    } else {
        let z = fsplat(fb, f, 0.0)?;
        bin(fb, BinOp::FOrdLessThan, x, z)?
    };
    let log_infinite = c.log_infinite(fb)?;
    let nan = c.nan(fb, negative, log_infinite)?;
    let infinite = bin(fb, BinOp::LogicalOr, log_infinite, c.y_inf)?;
    let numbers = c.numbers(fb, f)?;
    let infinite = bin(fb, BinOp::LogicalAnd, infinite, numbers)?;
    let answer = c.infinite_answer(fb, f)?;
    let r = fb.select(infinite, answer, r)?;
    let k = fsplat(fb, f, f64::NAN)?;
    fb.select(nan, k, r)
}

/// `r`, with `+0` in the lanes where `x` is [`flushed`].
fn zero_where_flushed(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, r: Value) -> BResult<Value> {
    match flushed(fb, f, x)? {
        Some(t) => {
            let z = fsplat(fb, f, 0.0)?;
            fb.select(t, z, r)
        }
        None => Ok(r),
    }
}

/// Metal's `sign(x)` (A1 cases 87d13612 and f9a12252, precise and fast alike): `1` above zero,
/// `-1` below it, and `+0` for either zero, a NaN, and an `f32` denormal ([`flushed`]).
/// `GLSL.std.450 FSign` keeps the sign of a zero and leaves a NaN's result undefined.
fn metal_sign(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let z = fsplat(fb, f, 0.0)?;
    let (m1, p1) = (fsplat(fb, f, -1.0)?, fsplat(fb, f, 1.0)?);
    let lt = bin(fb, BinOp::FOrdLessThan, x, z)?;
    let r = fb.select(lt, m1, z)?;
    let gt = bin(fb, BinOp::FOrdGreaterThan, x, z)?;
    let r = fb.select(gt, p1, r)?;
    zero_where_flushed(fb, f, x, r)
}

/// Metal's `fract(x)` (A1 case aed3faaa): `x - floor(x)`, rounded once to a narrow format, then
/// bounded below one (`fmin` with the largest value below one) when `bounded`; `air.fast_fract`
/// on `float` is not bounded (`fast::fract(-1e-10)` is `1.0`). An infinity gives `+0`, or the
/// canonical NaN where `infinite_nan`, and an `f32` zero or denormal gives `+0` ([`flushed`]); a
/// NaN gives the canonical NaN `0x7fc00000` on `float` and itself, bit for bit, on `half` (`0x7c01`
/// stays `0x7c01`). A `bfloat` NaN takes `float`'s answer (not measured). `GLSL.std.450 Fract` of
/// an infinity is NaN.
///
/// Unbounded, the one is selected outright where `x - floor(x) = 1 + x` rounds to it
/// ([`rounds_to_one_bits`]), not left to the subtraction: Mesa's NIR folds `x - floor(x)` into
/// its `ffract`, which RADV's `v_fract` bounds below one (`0x3f7fffff`).
fn metal_fract(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    bounded: bool,
    infinite_nan: bool,
) -> BResult<Value> {
    // `floor(x)` of a narrow value is a value of the format, so the difference is exact below the
    // smallest normal half.
    let fl = glsl(fb, GlslOp::Floor, &[x])?;
    let d = bin(fb, BinOp::FSub, x, fl)?;
    let d = narrow::round_exact_below_normal(fb, f, d)?;
    let r = if bounded {
        let below_one = splat(fb, f, below_one_bits(f))?;
        glsl(fb, GlslOp::NMin, &[d, below_one])?
    } else {
        let (z, one) = (fsplat(fb, f, 0.0)?, fsplat(fb, f, 1.0)?);
        let lo = splat(fb, f, rounds_to_one_bits(f))?;
        let above = bin(fb, BinOp::FOrdGreaterThanEqual, x, lo)?;
        let below = bin(fb, BinOp::FOrdLessThan, x, z)?;
        let near = bin(fb, BinOp::LogicalAnd, above, below)?;
        fb.select(near, one, d)?
    };
    // An infinity's `fract` is `+0`; under every fast-math flag `fast::fract`'s is the canonical
    // NaN, its `x - floor(x)`.
    let at_inf = if infinite_nan {
        fsplat(fb, f, f64::NAN)?
    } else {
        fsplat(fb, f, 0.0)?
    };
    let inf = fb.unary(UnOp::IsInf, x)?;
    let r = fb.select(inf, at_inf, r)?;
    let r = zero_where_flushed(fb, f, x, r)?;
    let nan = fb.unary(UnOp::IsNan, x)?;
    let n = if f.float_fmt() == Some(FloatFmt::Half) {
        x
    } else {
        fsplat(fb, f, f64::NAN)?
    };
    fb.select(nan, n, r)
}

/// Metal's `saturate(x)` (A1 case aed3faaa, precise and fast alike): `x` clamped to `[0, 1]`,
/// with `+0` for either zero, a NaN, and an `f32` denormal ([`flushed`]); a `half` denormal is
/// kept. `NClamp` leaves the sign of a zero result to the device.
fn metal_saturate(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let (z, one) = (fsplat(fb, f, 0.0)?, fsplat(fb, f, 1.0)?);
    let lt = bin(fb, BinOp::FOrdLessThan, x, one)?;
    let r = fb.select(lt, x, one)?;
    let gt = bin(fb, BinOp::FOrdGreaterThan, x, z)?;
    let r = fb.select(gt, r, z)?;
    zero_where_flushed(fb, f, x, r)
}

/// An operand of [`metal_median3`] with the value it compares as.
#[derive(Clone, Copy)]
struct Compared {
    value: Value,
    as_: Value,
}

/// Metal's `min(p, q)` (`less`) or `max(p, q)` of [`metal_median3`]: `p` when `q` is NaN (by
/// `test`), else `p` when it compares `less` (or greater) than `q`, else `q`.
fn pick(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    test: NanTest,
    order: BinOp,
    p: Compared,
    q: Compared,
) -> BResult<Compared> {
    let q_nan = nan::is_nan(fb, f, test, q.value)?;
    let before = bin(fb, order, p.as_, q.as_)?;
    let keep = bin(fb, BinOp::LogicalOr, q_nan, before)?;
    Ok(Compared {
        value: fb.select(keep, p.value, q.value)?,
        as_: fb.select(keep, p.as_, q.as_)?,
    })
}

/// Metal's `median3(a, b, c)` (A1 case aed3faaa, every lane of precise `float`, fast `float`,
/// and `half`): `min(max(c, min(a, b)), max(a, b))` with Metal's `min` and `max` ([`pick`]), so
/// the result is always one of the operands, bit for bit (a zero keeps its sign, a NaN its
/// payload). `float` compares the operands [`flushed`] (an `f32` denormal compares as a zero)
/// but returns the operand itself. `air.fast_fmedian3` (`fast`) returns `b` when `a` and `b`
/// are both NaN. `NMin`/`NMax` order a NaN differently and may give either zero.
fn metal_median3(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    test: NanTest,
    [a, b, c]: [Value; 3],
    fast: bool,
) -> BResult<Value> {
    let compared = |fb: &mut FunctionBuilder<'_>, x: Value| -> BResult<Compared> {
        Ok(Compared {
            value: x,
            as_: zero_where_flushed(fb, f, x, x)?,
        })
    };
    let (ca, cb, cc) = (compared(fb, a)?, compared(fb, b)?, compared(fb, c)?);
    let lo = pick(fb, f, test, BinOp::FOrdLessThan, ca, cb)?;
    let hi = pick(fb, f, test, BinOp::FOrdGreaterThan, ca, cb)?;
    let m = pick(fb, f, test, BinOp::FOrdGreaterThan, cc, lo)?;
    let r = pick(fb, f, test, BinOp::FOrdLessThan, m, hi)?.value;
    if !fast {
        return Ok(r);
    }
    let a_nan = nan::is_nan(fb, f, test, a)?;
    let b_nan = nan::is_nan(fb, f, test, b)?;
    let both = bin(fb, BinOp::LogicalAnd, a_nan, b_nan)?;
    fb.select(both, b, r)
}

/// Whether Metal's `fmin` and `fmax` family is measured on lanes of format `fmt`, so
/// `metal_min_max` emits it: `float` (`nan_minmax` without fast math, `nan_minmax_fast` with
/// every flag, and both at zeros and denormals) and `half` (`nan_half`, `nan_half_fast`). `bfloat`
/// and `double` keep the device's `NMin`, `NMax`, and `NClamp`.
pub fn metal_picks(fmt: Option<FloatFmt>) -> bool {
    matches!(fmt, Some(FloatFmt::F32 | FloatFmt::Half))
}

/// Metal's `fmin(p, q)` (`order` less) or `fmax(p, q)` (greater): `q` where `p` is NaN, else
/// [`pick`]'s `p` where `q` is NaN or `p` compares before `q`, else `q`. So two NaNs answer `q`'s
/// bits and a tie (`±0`, or an `f32` denormal beside a zero, which compares as one) answers `q`.
fn metal_fmin(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    test: NanTest,
    order: BinOp,
    p: Compared,
    q: Compared,
) -> BResult<Compared> {
    let p_nan = nan::is_nan(fb, f, test, p.value)?;
    let r = pick(fb, f, test, order, p, q)?;
    Ok(Compared {
        value: fb.select(p_nan, q.value, r.value)?,
        as_: fb.select(p_nan, q.as_, r.as_)?,
    })
}

/// Metal's `fmin`, `fmax`, `min3`, `max3`, and `clamp` (`op`) of `args` on `float` or `half`
/// lanes of form `f` ([`metal_picks`]), the NaN tests by `test`. Measured on every lane of
/// `nan_minmax` and `nan_minmax_fast` (`float`, without fast math and with every flag, the same
/// answers), their zeros-and-denormals cases, and `nan_half` and `nan_half_fast`: the result is an
/// operand, bit for bit ([`metal_fmin`]: a NaN loses, two NaNs and a tie answer the second);
/// `min3(a, b, c)` is `fmin(fmin(a, b), c)`, `max3` likewise, and `clamp(x, lo, hi)` is
/// `fmin(fmax(x, lo), hi)`. `float` compares the operands [`flushed`] (an `f32` denormal compares
/// as a zero) but returns the operand itself. `NMin` and `NMax` leave two NaNs, a tie of zeros,
/// and a denormal's comparison to the device.
pub(super) fn metal_min_max(
    fb: &mut FunctionBuilder<'_>,
    op: FloatFn,
    f: Form,
    test: NanTest,
    args: &[Value],
) -> BResult<Value> {
    use FloatFn as F;
    let compared = |fb: &mut FunctionBuilder<'_>, i: usize| -> BResult<Compared> {
        let x = arg(args, i)?;
        Ok(Compared {
            value: x,
            as_: zero_where_flushed(fb, f, x, x)?,
        })
    };
    let (less, greater) = (BinOp::FOrdLessThan, BinOp::FOrdGreaterThan);
    let (a, b) = (compared(fb, 0)?, compared(fb, 1)?);
    let r = match op {
        F::Min => metal_fmin(fb, f, test, less, a, b)?,
        F::Max => metal_fmin(fb, f, test, greater, a, b)?,
        F::Min3 | F::Max3 => {
            let order = if op == F::Min3 { less } else { greater };
            let ab = metal_fmin(fb, f, test, order, a, b)?;
            let c = compared(fb, 2)?;
            metal_fmin(fb, f, test, order, ab, c)?
        }
        F::Clamp => {
            let low = metal_fmin(fb, f, test, greater, a, b)?;
            let c = compared(fb, 2)?;
            metal_fmin(fb, f, test, less, low, c)?
        }
        // Not one of Metal's minimum and maximum functions.
        _ => return Err(BuildError::Type(TypeRule::FunctionSignature)),
    };
    Ok(r.value)
}

/// The largest magnitude Metal's `fast::sin` and `fast::cos` evaluate: `float(π/2) · 2^22`
/// (`0x4ac90fdb`), past which the quadrant index no longer fits in 22 bits.
pub(super) const FAST_TRIG_LIMIT: f32 = 6_588_397.5;

/// Metal's `fast::sin(x)` or `fast::cos(x)` from `r`, the device's `GLSL.std.450 Sin` or `Cos` of
/// `x`: `+0` where `|x|` is past [`FAST_TRIG_LIMIT`] or `x` is NaN (an unordered compare), `r`
/// elsewhere. Measured on Apple M3 Max: `fast::cos(6588397.5)` tracks `precise::cos` and the next
/// float answers `0` (the old engine's bisection), every lane of probe case 08bc6611 past the limit
/// (158, infinities and NaNs among them) answers `+0` for both, whatever the sign, and A1 cases
/// 179b6aa6, 3c92eb70, and 1f239ba7 read the flush. Below the limit the two are the precise
/// functions within a few ulp (the device's, `PrecisionNote::DeviceTranscendentals`).
pub(super) fn fast_trig_flush(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    r: Value,
) -> BResult<Value> {
    let ax = glsl(fb, GlslOp::FAbs, &[x])?;
    let limit = wide_splat(fb, f, f64::from(FAST_TRIG_LIMIT))?;
    let past = bin(fb, BinOp::FUnordGreaterThan, ax, limit)?;
    let zero = wide_splat(fb, f, 0.0)?;
    fb.select(past, zero, r)
}

/// Metal's `fast::tanh(x)` (`op`, [`FloatFn::FastTanh`]) on `float` lanes `x` (form `f`):
/// [`tanh_of_exp`] of `t = exp2(x · 2 log2 e)`, Metal's `exp2` ([`routine_or`]), its quotient by
/// `divide` (a call of its [`FastDivision`] routine).
pub(super) fn fast_tanh(
    fb: &mut FunctionBuilder<'_>,
    op: FloatFn,
    f: Form,
    x: Value,
    lib: &Lib<'_>,
    divide: DivideOf<'_>,
) -> BResult<Value> {
    // One operation to Metal: no step of it is fused or regrouped (`tanh_of_exp`).
    let k = wide_splat(fb, f, 2.0 * std::f64::consts::LOG2_E)?;
    let x = fb.binary_with(BinOp::FMul, x, k, Contraction::Forbidden)?;
    let t = routine_or(fb, lib, op, f, libm::Fun::Exp2, GlslOp::Exp2, &[x])?;
    tanh_of_exp(fb, f, divide, t)
}

/// Metal's `fast::atanh(x)` (`op`, [`FloatFn::FastAtanh`]) on `float` lanes `x` (form `f`): the
/// device's `Atanh` where `lib` says its functions are Metal's in a fast-math module; elsewhere
/// MSL Table 8.2's `0.5 · log((1 + x) · rcp(1 - x))`, the quotient by `divide` (a call of its
/// [`FastDivision`] routine, Metal's fast division) and the logarithm Metal's fast `log`
/// ([`fast_log_of`]), each step its own. The formula answers Metal's specials itself: a zero and a
/// denormal give `+0`, `±1` gives `±inf`, an infinity gives NaN (`inf · 0`), and `±FLT_MAX` gives
/// `-inf`, the reciprocal of `∓FLT_MAX` flushed (case f6524207, every special lane). On that case
/// the formula of correctly rounded `log2` and reciprocal matches Metal on 871 of 1,024 lanes; the
/// rest is Metal's `log2`.
pub(super) fn fast_atanh(
    fb: &mut FunctionBuilder<'_>,
    op: FloatFn,
    f: Form,
    x: Value,
    lib: &Lib<'_>,
    divide: DivideOf<'_>,
) -> BResult<Value> {
    if lib.metal_in(libm::Mode::Fast) {
        return glsl(fb, GlslOp::Atanh, &[x]);
    }
    let exact = Contraction::Forbidden;
    let one = wide_splat(fb, f, 1.0)?;
    let num = fb.binary_with(BinOp::FAdd, one, x, exact)?;
    let den = fb.binary_with(BinOp::FSub, one, x, exact)?;
    let q = divide(fb, num, den)?;
    let l = fast_log_of(fb, lib, op, f, q)?;
    let half = wide_splat(fb, f, 0.5)?;
    fb.binary_with(BinOp::FMul, half, l, exact)
}

/// Metal's `fast::tan(x)` on `float` lanes `x` (form `f`, [`FloatFn::FastTan`]): the device's `Tan`
/// where `lib` says its functions are Metal's in a fast-math module; elsewhere Apple's formula
/// `sin(x) · (1 / cos(x))` (MSL Table 8.2) of Metal's fast sine and cosine, each the device's
/// instruction or its routine with Metal's flush past the last evaluated quadrant
/// ([`fast_trig_flush`]), their quotient by `divide` (a call of its [`FastDivision`] routine). A
/// denormal `x` is its own `fast::tan`, as Metal keeps it, where the quotient's product reads a
/// denormal sine as a zero (`denormal::operand`).
pub(super) fn fast_tan(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    lib: &Lib<'_>,
    divide: DivideOf<'_>,
) -> BResult<Value> {
    if lib.metal_in(libm::Mode::Fast) {
        return glsl(fb, GlslOp::Tan, &[x]);
    }
    let fast = |fb: &mut FunctionBuilder<'_>, r: libm::Fun, o: GlslOp| -> BResult<Value> {
        let v = if lib.device(r) {
            glsl(fb, o, &[x])?
        } else {
            lib.call(fb, r, f, &[x])?
        };
        fast_trig_flush(fb, f, x, v)
    };
    let sin = fast(fb, libm::Fun::FastSin, GlslOp::Sin)?;
    let cos = fast(fb, libm::Fun::FastCos, GlslOp::Cos)?;
    let r = divide(fb, sin, cos)?;
    match denormal_lanes(fb, f, x)? {
        Some(d) => fb.select(d, x, r),
        None => Ok(r),
    }
}

/// `(t - 1) / (t + 1)`: `tanh(x)` from `t = exp(2x)`, the MSL fast-math definition of
/// `fast::tanh`, divided as Metal's fast division does (`divide`, [`fast_divide`]): past `x ≈
/// 43.7`, where `t + 1` passes `2^126`, the flushed reciprocal gives `0` (Metal's answer at `x =
/// 44`, A1 case a919193f), and `t = +inf` gives `inf · 0 = NaN`, as Metal does past `x ≈ 44.4`.
///
/// `air.fast_tanh` is one operation to Metal, whatever the call's fast-math flags, so every step
/// here (and `x · 2 log2 e` before it) is `NoContraction`: the device rounds each on its own and
/// never fuses or regroups one with the caller's arithmetic. Mesa's `lower_flrp` rule (`1 + c ·
/// (b - 1)` to `(1 - c) + b · c`) took a caller's `1 + tanh(x)` with the quotient's steps and
/// answered `t` for it, where Metal's `1 + tanh(-24.65)` is `0`: RC3 cases e72f550f and 2d5f3949's
/// `gelu(-8)` was `0.5 · x · t` on RADV and x86 lavapipe, not Metal's `-0`.
pub(super) fn tanh_of_exp(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    divide: DivideOf<'_>,
    t: Value,
) -> BResult<Value> {
    let exact = Contraction::Forbidden;
    let one = wide_splat(fb, f, 1.0)?;
    let num = fb.binary_with(BinOp::FSub, t, one, exact)?;
    let den = fb.binary_with(BinOp::FAdd, t, one, exact)?;
    divide(fb, num, den)
}

/// Metal's fast reciprocal of `y` (floats of form `f`): `1 / y` by `reciprocal`, the device's
/// division or `RN(1 / y)` (Metal's own is neither on every lane, so both are
/// `PrecisionNote::DeviceArithmetic`), with the two ranges where Metal's answer is a flush,
/// measured on Apple M3 Max (probe case 08bc6611's `1 / y` lanes): past `2^126` the reciprocal is
/// an `f32` denormal, which Metal flushes to the zero of `y`'s sign (an infinity included), and an
/// `f32` denormal `y` is a zero to Metal, whose reciprocal is the infinity of `y`'s sign. Vulkan
/// bounds `OpFDiv` only for a divisor in `[2^-126, 2^126]`, so both ranges are selected rather than
/// left to the device.
pub(super) fn fast_reciprocal(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    y: Value,
    contract: Contraction,
    reciprocal: Reciprocal,
) -> BResult<Value> {
    let device = move |fb: &mut FunctionBuilder<'_>, a: Value, b: Value| {
        fb.binary_with(BinOp::FDiv, a, b, contract)
    };
    fast_reciprocal_with(fb, f, y, reciprocal, &device)
}

/// [`fast_reciprocal`] with `rough` the device's division (a test moves it within Vulkan's
/// bound).
pub(super) fn fast_reciprocal_with(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    y: Value,
    reciprocal: Reciprocal,
    rough: Rough<'_>,
) -> BResult<Value> {
    let one = wide_splat(fb, f, 1.0)?;
    let r = match reciprocal {
        Reciprocal::Device | Reciprocal::Native => rough(fb, one, y)?,
        Reciprocal::Rounded => rounded_reciprocal(fb, f, one, y, rough)?,
    };
    let ay = glsl(fb, GlslOp::FAbs, &[y])?;
    let limit = wide_splat(fb, f, 2f64.powi(126))?;
    let past = bin(fb, BinOp::FOrdGreaterThan, ay, limit)?;
    let zero = wide_splat(fb, f, 0.0)?;
    let signed_zero = copysign(fb, f, zero, y)?;
    let r = fb.select(past, signed_zero, r)?;
    match flushed(fb, f, y)? {
        Some(z) => {
            let inf = wide_splat(fb, f, f64::INFINITY)?;
            let signed_inf = copysign(fb, f, inf, y)?;
            fb.select(z, signed_inf, r)
        }
        None => Ok(r),
    }
}

/// `RN(1 / y)` of `f32` lanes `y` (form `f`) whose magnitude is in `[2^-126, 2^126]`, from
/// `rough`'s estimate ([`divide::rounded_reciprocal_bits`]); the other lanes are
/// [`fast_reciprocal`]'s flushes, or a NaN's. A `y` past `2^125`, whose estimate may fall below
/// `2^-126`, is halved first and its reciprocal halved after, each by its exponent field: both
/// moves are exact there, and the result is at least `2^-126`. `one` is `1.0`.
fn rounded_reciprocal(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    one: Value,
    y: Value,
    rough: Rough<'_>,
) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let yb = fb.convert(ConvOp::Bitcast, y, ut)?;
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let ay = bin(fb, BinOp::BitwiseAnd, yb, magnitude)?;
    // `|y|` in `(2^125, 2^126]`: one less in the exponent field.
    let low = splat(fb, u, 0x7E00_0001)?;
    let span = splat(fb, u, 0x80_0000)?;
    let off = bin(fb, BinOp::ISub, ay, low)?;
    let high = bin(fb, BinOp::ULessThan, off, span)?;
    let zero = splat(fb, u, 0)?;
    let shift = fb.select(high, span, zero)?;
    let sb = bin(fb, BinOp::ISub, yb, shift)?;
    let ys = fb.convert(ConvOp::Bitcast, sb, ft)?;
    let ny = fb.unary(UnOp::FNegate, ys)?;
    let r = rough(fb, one, ys)?;
    let rb = divide::rounded_reciprocal_bits(fb, f, one, ny, yb, r)?;
    let rb = bin(fb, BinOp::ISub, rb, shift)?;
    fb.convert(ConvOp::Bitcast, rb, ft)
}

/// An upper estimate of the SPIR-V words of `fast_divide` on form `f` by `reciprocal`: the
/// reciprocal, its two flushes' tests and selects, the product, and Metal's answers at denormals
/// and NaNs (`metal_fast_product`), with their constants, types, and the `GLSL.std.450` import.
pub fn fast_divide_words(f: Form, reciprocal: Reciprocal) -> u32 {
    match reciprocal {
        // The device's `OpFDiv` alone.
        Reciprocal::Native => 16,
        _ => 320 + 48 * u32::from(f.lanes) + reciprocal.words(f),
    }
}

/// Metal's fast division of two values of one form: a call of its [`FastDivision`] routine, or
/// (a unit test) [`fast_divide`] itself.
pub(super) type DivideOf<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, Value, Value) -> BResult<Value>;

/// The module-scope routine of Metal's fast division (`fast_divide`) on `float` lanes of
/// `lanes` lanes, by `reciprocal`, its device division `NoContraction` where `exact`. Every `arcp`
/// division, `fast::tanh`, `fast::fmod`, and `fast::tan` on those lanes calls it, so a module holds
/// it once, as it holds the precise division's routines; expanded at every use, Metal's answers at
/// denormals and NaNs made each use about 330 to 570 words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FastDivision {
    pub lanes: u8,
    pub reciprocal: Reciprocal,
    pub exact: bool,
}

impl FastDivision {
    /// The routine of a fast division of form `f` (`float` lanes) by `reciprocal`, whose device
    /// division contracts as `contract` allows.
    pub fn new(f: Form, reciprocal: Reciprocal, contract: Contraction) -> FastDivision {
        FastDivision {
            lanes: f.lanes,
            reciprocal,
            exact: contract == Contraction::Forbidden,
        }
    }

    /// The routine's operand form.
    fn form(self) -> Form {
        Form {
            held: Held::Float(FloatFmt::F32),
            lanes: self.lanes,
        }
    }

    /// The contraction of the routine's device division.
    fn contract(self) -> Contraction {
        if self.exact {
            Contraction::Forbidden
        } else {
            Contraction::Allowed
        }
    }

    /// An upper estimate of the routine's SPIR-V words: `fast_divide`'s, and the function's
    /// type, parameters, and frame.
    pub fn words(self) -> u32 {
        fast_divide_words(self.form(), self.reciprocal) + 32
    }
}

/// An upper estimate of the SPIR-V words of one call of a [`FastDivision`] routine.
pub const FAST_DIVISION_CALL_WORDS: u32 = 24;

/// Define routine `r`: a function of two `float` values of `r.lanes` lanes returning Metal's
/// fast quotient (`fast_divide`).
pub fn define_fast_division(mb: &mut ModuleBuilder, r: FastDivision) -> BResult<FuncId> {
    let e = mb.ty(Type::Float { width: 32 })?;
    let t = if r.lanes == 1 {
        e
    } else {
        mb.ty(Type::Vector {
            elem: e,
            len: r.lanes,
        })?
    };
    let mut fb = mb.routine(t)?;
    let (x, y) = (fb.param(t)?, fb.param(t)?);
    let q = fast_divide(&mut fb, r.form(), x, y, r.contract(), r.reciprocal)?;
    fb.ret(Some(q))?;
    fb.finish()
}

/// Metal's fast quotient of `x` and `y` through `routine`, a [`define_fast_division`].
pub fn call_fast_division(
    fb: &mut FunctionBuilder<'_>,
    routine: FuncId,
    x: Value,
    y: Value,
) -> BResult<Value> {
    fb.call(routine, &[x, y])?
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))
}

/// Metal's fast quotient `x / y` as the device's own `OpFDiv(x, y)`, contracting as routine `r`'s
/// device division does: the form of an `arcp` division whose quotient is the device's
/// (`ArithPlan::device_quotient`), where no routine is called.
pub fn device_fast_division(
    fb: &mut FunctionBuilder<'_>,
    r: FastDivision,
    x: Value,
    y: Value,
) -> BResult<Value> {
    fb.binary_with(BinOp::FDiv, x, y, r.contract())
}

/// Metal's fast division `x / y`: `x` times its fast reciprocal of `y` ([`fast_reciprocal`] by
/// `reciprocal`), two roundings, as Metal computes it (every lane of probe case 08bc6611's fast `x
/// / y` equals `x` times its fast `1 / y`), with Metal's answers at `f32` denormals and NaNs
/// ([`metal_fast_product`]).
pub(super) fn fast_divide(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    contract: Contraction,
    reciprocal: Reciprocal,
) -> BResult<Value> {
    // Native device maths: the device's quotient itself.
    if reciprocal == Reciprocal::Native {
        return fb.binary_with(BinOp::FDiv, x, y, contract);
    }
    let r = fast_reciprocal(fb, f, y, contract, reciprocal)?;
    if f.float_fmt() != Some(FloatFmt::F32) {
        return fb.binary_with(BinOp::FMul, x, r, contract);
    }
    metal_fast_product(fb, f, x, r)
}

/// `x · r`, the product that ends Metal's fast division, as Metal answers it on `f32` lanes
/// (form `f`) whatever the device's denormal mode and NaN encoding: a denormal `x` is a zero of its
/// sign ([`flushed`]); a product whose 24-bit rounding, with an unbounded exponent, is below
/// `2^-126` is a zero of its sign (Metal's precise quotient flushes by that rounding, a probe case
/// c10cba13; the product scaled by `2^24` is a normal there, so its rounding decides, and the
/// device's own rounding of a denormal result, or its flush, does not); and a NaN is `0x7FC00000`.
/// Metal answered `0x7fc00000` for every NaN fast quotient measured (a probe cases 914cc4d2 and
/// 711cc190, 155 and 62 lanes, probe case 08bc6611, A1 case a919193f), where RADV answered
/// `0xffc00000` for `0 / 0` and a NaN divisor, and a denormal numerator over a denormal divisor
/// `NaN` (`0 · inf`, A1 case f9a12252), where lavapipe kept the numerator and answered `+inf`. No
/// step is fused or regrouped (`NoContraction`), and the selects keep the device from fusing the
/// product into its user. The NaN is told by the bits, which no float mode folds.
fn metal_fast_product(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, r: Value) -> BResult<Value> {
    let exact = Contraction::Forbidden;
    let zero = wide_splat(fb, f, 0.0)?;
    let x = match flushed(fb, f, x)? {
        Some(z) => {
            let signed_zero = copysign(fb, f, zero, x)?;
            fb.select(z, signed_zero, x)?
        }
        None => x,
    };
    let q = fb.binary_with(BinOp::FMul, x, r, exact)?;
    // The same product `2^24` larger: exact up to its rounding, which is the 24-bit one.
    let up = wide_splat(fb, f, 2f64.powi(24))?;
    let xs = fb.binary_with(BinOp::FMul, x, up, exact)?;
    let s = fb.binary_with(BinOp::FMul, xs, r, exact)?;
    let abs_s = glsl(fb, GlslOp::FAbs, &[s])?;
    let tiny_limit = wide_splat(fb, f, 2f64.powi(-102))?;
    let tiny = bin(fb, BinOp::FOrdLessThan, abs_s, tiny_limit)?;
    // A 24-bit rounding of at least `2^-126` (`s` is a number: not `inf · 0`), which the device
    // answered below it (a flush before rounding): `2^-126`.
    let rounds_normal = bin(fb, BinOp::FOrdGreaterThanEqual, abs_s, tiny_limit)?;
    let abs_q = glsl(fb, GlslOp::FAbs, &[q])?;
    let least = wide_splat(fb, f, 2f64.powi(-126))?;
    let below = bin(fb, BinOp::FOrdLessThan, abs_q, least)?;
    let raised = bin(fb, BinOp::LogicalAnd, rounds_normal, below)?;
    let signed_least = copysign(fb, f, least, s)?;
    let q = fb.select(raised, signed_least, q)?;
    let signed_zero = copysign(fb, f, zero, s)?;
    let q = fb.select(tiny, signed_zero, q)?;
    // Told by the bits: the routine runs in entries of every float mode, and one without
    // `SignedZeroInfNanPreserve` lets a device fold `OpIsNan`.
    let nan = nan::is_nan(fb, f, NanTest::Bits, q)?;
    let canonical = fsplat(fb, f, f64::NAN)?;
    fb.select(nan, canonical, q)
}

/// A fused multiply-add of three floats (`super::fma::emit` with its routine; a unit test
/// substitutes a reference).
pub(super) type FmaOf<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value>;

/// Metal's `fast::fmod(x, y)` on `float`: `fma(-trunc(q), y, x)` of its fast quotient
/// `q = x / y` (`divide`, a call of its [`FastDivision`] routine), the fused multiply-add
/// rounding once (`fma`; Metal's answers for `f32` denormals and NaNs are its). Every lane of
/// probe case 08bc6611 (1,024, specials among them) is this expression on Metal's own quotient; the
/// quotient's reciprocal is a near-equivalent of Metal's (`PrecisionNote::DeviceArithmetic`), so
/// `fmod(1e10, 3)` is Metal's `-512` exactly where that reciprocal of `3` is Metal's. Metal reads a
/// denormal `x` or `y` as the zero of its sign and answers no denormal, with and without fast math
/// (every pair of 28 patterns): the caller flushes the operands and the result where
/// the fused multiply-add is the device's and the profile asks Metal's functions; under native
/// device maths they are the device's (`PrecisionNote::DeviceDenormals`).
pub(super) fn fast_fmod(
    fb: &mut FunctionBuilder<'_>,
    x: Value,
    y: Value,
    divide: DivideOf<'_>,
    fma: FmaOf<'_>,
) -> BResult<Value> {
    let q = divide(fb, x, y)?;
    let t = glsl(fb, GlslOp::Trunc, &[q])?;
    let n = fb.unary(UnOp::FNegate, t)?;
    fma(fb, &[n, y, x])
}

/// Metal's LLVM `frem` on `float` without every fast-math flag: `x - y * trunc(q)` of its precise
/// quotient `q = x / y` (`divide`, a call of the shared routine, which reads a denormal operand as
/// a zero and flushes a denormal quotient), the product and the difference each rounded on its own
/// (`NoContraction`), the product read as Metal reads an `f32` denormal where `flush` (the caller
/// flushes the operands and the result). Not the exact remainder: on the M3 Max
/// `frem(1, 0.1)` is `+0` where `fmod` gives `0x3dcccccb`, and this expression is Metal's answer on
/// all 49,753 measured pairs, every class of operand.
pub(super) fn metal_frem(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    (x, y): (Value, Value),
    divide: DivideOf<'_>,
    flush: bool,
) -> BResult<Value> {
    let q = divide(fb, x, y)?;
    let t = glsl(fb, GlslOp::Trunc, &[q])?;
    let p = fb.binary_no_contraction(BinOp::FMul, y, t)?;
    let p = if flush { denormal::flush(fb, f, p)? } else { p };
    fb.binary_no_contraction(BinOp::FSub, x, p)
}

/// Metal's precise `mix(x, y, t)` on `float` and `half`: `fma(t, y, fma(-t, x, x))`, each fused
/// multiply-add rounding once (`fma`, the profile's route), with and without fast-math flags. Every
/// lane of `nan_functions`, `nan_functions_fast`, `nan_half`, and `nan_half_fast` whose answer is a
/// number is this expression, where the blend `x * (1 - t) + y * t` misses lanes (`mix(inf, 1, 0)`
/// is NaN: `fma(-0, inf, inf)`; `mix(-0, -0, 0)` is `+0`; `half` `mix(-0.3, 0.7, 0.2)` is
/// `0xAE68`). `t` is negated on its bits, which no float mode changes. Each fused multiply-add
/// reads an `f32` denormal operand as the zero of its sign and flushes a denormal result (an Apple
/// M3 Max under macOS 27.0.1: every finite triple of 28 patterns, denormals
/// among them, in both modes); where `flush`, the inner one's result is read so
/// ([`denormal::flush`]), for a device whose `OpFmaKHR` may keep one; the operands and the result
/// are the caller's to flush.
pub(super) fn metal_mix(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    [x, y, t]: [Value; 3],
    fma: FmaOf<'_>,
    flush: bool,
) -> BResult<Value> {
    let minus_t = sign_bits(fb, f, t, SignBit::Flip)?;
    let rest = fma(fb, &[minus_t, x, x])?;
    let rest = if flush {
        denormal::flush(fb, f, rest)?
    } else {
        rest
    };
    fma(fb, &[t, y, rest])
}

/// Metal's `dot(x, y)` on `float` or `half` vectors of form `f`: the products summed lane by lane
/// in one fused chain, `fma(x[n-1], y[n-1], ... fma(x[1], y[1], x[0] * y[0]))`, the first product
/// rounded to `f`'s format on its own and each later lane added by one fused multiply-add (`fma`,
/// the profile's route), with and without fast-math flags. On 4,096 lanes of two, three and four
/// lanes each, in both modes, every `float` and `half` answer of the M3 Max and the M4 is this
/// chain's (a `half` keeps its denormals). Where `flush`, each of the chain's `float` values before
/// the last is read as Metal reads a denormal ([`denormal::flush`]), for a device whose multiply
/// and `OpFmaKHR` may keep one; the operands and the result are the caller's to flush. Where
/// `tiny` (`float` lanes), the first product is Metal's where the device's rounding of a product
/// below `2^-126` reaches `±2^-126`, by that check ([`denormal::tiny_product`]).
pub(super) fn metal_dot(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    [x, y]: [Value; 2],
    fma: FmaOf<'_>,
    (flush, tiny): (bool, Option<denormal::TinyCheck>),
) -> BResult<Value> {
    let lane = |fb: &mut FunctionBuilder<'_>, i: u32| -> BResult<(Value, Value)> {
        if f.lanes == 1 {
            Ok((x, y))
        } else {
            Ok((
                fb.composite_extract(x, &[i])?,
                fb.composite_extract(y, &[i])?,
            ))
        }
    };
    metal_chain(
        fb,
        f.scalar_form(),
        u32::from(f.lanes),
        &lane,
        fma,
        (flush, tiny),
    )
}

/// Lane `i`'s two operands of a dot product.
pub(super) type LaneOf<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, u32) -> BResult<(Value, Value)>;

/// [`metal_dot`]'s fused chain over `lanes` lanes of scalar form `s`, lane `i`'s operands by
/// `lane`, each taken as the chain reaches it; its first product Metal's at `2^-126` where `tiny`.
pub(super) fn metal_chain(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    lanes: u32,
    lane: LaneOf<'_>,
    fma: FmaOf<'_>,
    (flush, tiny): (bool, Option<denormal::TinyCheck>),
) -> BResult<Value> {
    let step = |fb: &mut FunctionBuilder<'_>, v: Value| -> BResult<Value> {
        if flush {
            denormal::flush(fb, s, v)
        } else {
            Ok(v)
        }
    };
    let (x0, y0) = lane(fb, 0)?;
    // A `half` product is exact in its `float` container, and rounded once.
    let p = fb.binary_no_contraction(BinOp::FMul, x0, y0)?;
    let p = match tiny {
        Some(check) => denormal::tiny_product(fb, s, [x0, y0], p, check)?,
        None => p,
    };
    let mut sum = narrow::round(fb, s, p)?;
    for i in 1..lanes {
        sum = step(fb, sum)?;
        let (xi, yi) = lane(fb, i)?;
        sum = fma(fb, &[xi, yi, sum])?;
    }
    Ok(sum)
}

/// An upper estimate of the SPIR-V words of [`metal_dot`] on form `f` whose fused multiply-adds
/// take `fma_words` each: two extractions per lane, the product, its check where `tiny` and its
/// rounding, and per later lane the fused multiply-add and, where `flush`, the flush before it.
pub(super) fn metal_dot_words(
    f: Form,
    fma_words: u32,
    (flush, tiny): (bool, Option<denormal::TinyCheck>),
) -> u32 {
    let s = f.scalar_form();
    let lanes = u32::from(f.lanes);
    let step = fma_words + if flush { denormal::flush_words(s) } else { 0 };
    let check = tiny.map_or(0, |check| denormal::tiny_product_words(s, check));
    (10 * lanes + 5 + narrow::round_words(s) + check).saturating_add(lanes.saturating_sub(1) * step)
}

/// Which of Metal's pi-scaled functions [`pi_scaled`] computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PiScaled {
    Sin,
    Cos,
    Tan,
}

/// `sin(π r)` and `cos(π r)` of the exact residue `r` in `[-1/4, 1/4]` (`GLSL.std.450`
/// `Sin`/`Cos` of `float(π) r`; a unit test substitutes a stand-in).
pub(super) type SinCosOf<'a> =
    &'a dyn Fn(&mut FunctionBuilder<'_>, Value) -> BResult<(Value, Value)>;

/// The quotient of two values, for [`pi_scaled`]'s `tanpi`.
pub(super) type QuotientOf<'a> =
    &'a dyn Fn(&mut FunctionBuilder<'_>, Value, Value) -> BResult<Value>;

/// Metal `sinpi`/`cospi`/`tanpi` of `x` (form `f`, computed in its `f32` container) by an exact
/// period reduction (the old engine's device-measured lowering), not `sin(float(π) * x)`:
/// `float(π)` is `8.74e-8` above π, so the pre-multiply misses every zero crossing (`sinpi(1)`
/// came back `-8.74e-8`, `cospi(0.5)` `-4.37e-8`, and `tan(float(π) * 0.5)` has no pole at all),
/// and it hands `Sin`/`Cos` a large argument (`cospi(1e7)` came back 0). Both functions have
/// period 2, a power of two, so every step but the last multiply is exact:
///
/// ```text
/// x2 = x - 2 roundEven(x / 2)     in [-1, 1]
/// n  = roundEven(2 x2)             in {-2, -1, 0, 1, 2}
/// r  = x2 - n / 2                  in [-1/4, 1/4]
/// sinpi(x) = [ sin, cos, -sin, -cos][n mod 4](π r)
/// cospi(x) = [ cos, -sin, -cos, sin][n mod 4](π r)
/// tanpi(x) = sinpi(x) / cospi(x), and ±1 where |r| = 1/4
/// ```
///
/// This is the form where no routine answers (a `bfloat`; on `float` and `half` the routines are
/// Metal's own forms). The sine and cosine of `π r` are `sin_cos`'s of the residue, the device's
/// `Sin` and `Cos` of `float(π) r` (a unit test substitutes a stand-in). `n` is selected by three
/// float compares (never converted, so a large `x` cannot overflow an integer). At a multiple of
/// one half `r` is exactly `+0`, and the sine and cosine there are selected as `+0` and `1` rather
/// than taken from the device (Vulkan bounds `Sin` and `Cos` only to `2^-11` absolute, so a
/// device's `Sin(+0)` need not be `+0`), so the result is an exact `0` or `±1`, and `tanpi` divides
/// an exact `±1` by the `+0` below into the correctly signed infinity. The quotient is
/// `quotient`'s, the device's `OpFDiv`. For a normal `x` no operand of it is a denormal: a nonzero
/// `sinpi` is at least `π 2^-126` in magnitude (`r` is `x` below 1, and a multiple of `2^-23`
/// above); a denormal `x` reaches the first multiply and difference, which read it as the device
/// does: Metal's reading where the profile's functions are Metal's (the deviation, or flushed
/// before), the device's under native device maths (`PrecisionNote::DeviceDenormals`). At a
/// quarter turn (`r = ±1/4`, exact like every step before the multiply) the two arms are the sine
/// and cosine of `π/4`, equal in magnitude, so `tanpi` is exactly `±1` with the quotient's sign,
/// whatever the device's `Sin` and `Cos` round them to (on lavapipe they differ by one ulp, and the
/// quotient by one). With the zeros and the poles these are the only rational values `tanpi` takes
/// at a rational argument (Niven's theorem), and `sinpi` and `cospi` take a rational value at a
/// dyadic `r` in `[-1/4, 1/4]` only at `r = 0`, so all three are exact on any device wherever their
/// value is a float. Metal's signed zeros are measured, not derived: `sinpi` is `-0` at odd
/// integers (the `-sin` arm negates `+0`), and `cospi` is `+0` at every half-integer, including
/// those whose arm is `-sin`, so that arm selects `+0` when `r` is 0 (a negation, or `0 - sin`,
/// which Metal's compiler folds into one, would give `-0`). A non-finite `x` (tested on its bits,
/// which no fast-math mode folds) gives the quiet NaN `0x7fc00000`, as Metal does for `±inf` and
/// NaN.
pub(super) fn pi_scaled(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    which: PiScaled,
    x: Value,
    sin_cos: SinCosOf<'_>,
    quotient: QuotientOf<'_>,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: f64| wide_splat(fb, f, k);
    let (half, two, zero) = (c(fb, 0.5)?, c(fb, 2.0)?, c(fb, 0.0)?);
    let (one, minus_one) = (c(fb, 1.0)?, c(fb, -1.0)?);
    // x2 = x - 2 roundEven(x / 2): scalings by powers of two, and a difference below 1.
    let scaled = bin(fb, BinOp::FMul, x, half)?;
    let whole = glsl(fb, GlslOp::RoundEven, &[scaled])?;
    let doubled = bin(fb, BinOp::FMul, whole, two)?;
    let reduced = bin(fb, BinOp::FSub, x, doubled)?;
    let twice = bin(fb, BinOp::FMul, reduced, two)?;
    let quadrant = glsl(fb, GlslOp::RoundEven, &[twice])?;
    let offset = bin(fb, BinOp::FMul, quadrant, half)?;
    let residue = bin(fb, BinOp::FSub, reduced, offset)?;
    let (sine, cosine) = sin_cos(fb, residue)?;
    // On an axis (`r = +0`) the arms are exact whatever the device's `Sin` and `Cos` of `+0` are.
    let axis = bin(fb, BinOp::FOrdEqual, residue, zero)?;
    let sine = fb.select(axis, zero, sine)?;
    let cosine = fb.select(axis, one, cosine)?;
    let at_zero = bin(fb, BinOp::FOrdEqual, quadrant, zero)?;
    let at_one = bin(fb, BinOp::FOrdEqual, quadrant, one)?;
    let at_minus_one = bin(fb, BinOp::FOrdEqual, quadrant, minus_one)?;
    // The arms for n = 0, 1, -1, and ±2 (which select the same value).
    let pick = |fb: &mut FunctionBuilder<'_>, arms: [Value; 4]| -> BResult<Value> {
        let inner = fb.select(at_minus_one, arms[2], arms[3])?;
        let middle = fb.select(at_one, arms[1], inner)?;
        fb.select(at_zero, arms[0], middle)
    };
    let sin_arms = |fb: &mut FunctionBuilder<'_>| -> BResult<[Value; 4]> {
        let ns = fb.unary(UnOp::FNegate, sine)?;
        let nc = fb.unary(UnOp::FNegate, cosine)?;
        Ok([sine, cosine, nc, ns])
    };
    let cos_arms = |fb: &mut FunctionBuilder<'_>| -> BResult<[Value; 4]> {
        let ns = fb.unary(UnOp::FNegate, sine)?;
        let ns = fb.select(axis, zero, ns)?;
        let nc = fb.unary(UnOp::FNegate, cosine)?;
        Ok([cosine, ns, sine, nc])
    };
    let r = match which {
        PiScaled::Sin => {
            let arms = sin_arms(fb)?;
            pick(fb, arms)?
        }
        PiScaled::Cos => {
            let arms = cos_arms(fb)?;
            pick(fb, arms)?
        }
        PiScaled::Tan => {
            let arms = sin_arms(fb)?;
            let num = pick(fb, arms)?;
            let arms = cos_arms(fb)?;
            let den = pick(fb, arms)?;
            let quotient = quotient(fb, num, den)?;
            // A quarter turn: both arms are nonzero, so the quotient's sign bit is the answer's.
            let quarter = c(fb, 0.25)?;
            let magnitude = glsl(fb, GlslOp::FAbs, &[residue])?;
            let at_quarter = bin(fb, BinOp::FOrdEqual, magnitude, quarter)?;
            let unit = copysign(fb, f, one, quotient)?;
            fb.select(at_quarter, unit, quotient)?
        }
    };
    // Finite: the exponent field is not all ones.
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let exp = splat(fb, u, 0x7F80_0000)?;
    let e = bin(fb, BinOp::BitwiseAnd, bits, exp)?;
    let finite = bin(fb, BinOp::INotEqual, e, exp)?;
    let nan = c(fb, f64::NAN)?;
    fb.select(finite, r, nan)
}

/// The words of `infinite_product` on a scalar: about 23 instructions and their constants.
const INFINITE_PRODUCT_WORDS: u32 = 128;

/// `exp2(y · log2(x))` with the product rounded, of Metal's `exp2` and `log2` ([`routine_or`]):
/// Metal's `fast::powr(x, y)`, and its `fast::pow`'s magnitude of `|x|` (`op`, on form `f`).
/// Every lane of probe case 08bc6611 (1,024, specials among them) gives Metal's `fast::powr` as
/// this expression of Metal's own `exp2` and `log2`, and its `fast::pow` at a base that is not
/// negative. Where the product is infinite the callers select IEEE's answer by the operands' bits
/// ([`infinite_product`], [`fast_powr_special`]), so a device's arithmetic on the infinite
/// intermediates cannot move it.
fn exp2_of_product(
    fb: &mut FunctionBuilder<'_>,
    lib: &Lib<'_>,
    op: FloatFn,
    f: Form,
    y: Value,
    x: Value,
) -> BResult<Value> {
    let l = routine_or(fb, lib, op, f, libm::Fun::Log2, GlslOp::Log2, &[x])?;
    let p = bin(fb, BinOp::FMul, y, l)?;
    routine_or(fb, lib, op, f, libm::Fun::Exp2, GlslOp::Exp2, &[p])
}

/// `r`, the value of `exp2(y · log2(x))` (floats of form `f`, `x` not negative where it
/// matters), with the value IEEE arithmetic gives it where the product is infinite, told by the
/// operands' bits ([`PowrOperands`]): `log2(x)` is `-inf` at a base Metal reads as zero (an `f32`
/// denormal included, as the fast `log2` reads it) and `+inf` at `+inf`, and the product of
/// either with a nonzero `y`, or of an infinite `y` with a `log2(x)` that is not zero, is `+inf`
/// (answer `+inf`) or `-inf` (answer `+0`) by their signs. Where the product is a NaN (a zero
/// times an infinity) or an operand is one, `r` is kept for the callers' NaN rules. A fast-math
/// entry declares no `SignedZeroInfNanPreserve`, so a device may compute those infinite
/// intermediates otherwise: the RTX 5080 answered `1` at every one (case f6524207's `fast::pow`
/// at infinite bases and `fast::powr` at zero and denormal ones, case 32931370's `half` `powr`),
/// where Metal and the Mesa venues answer IEEE's value. `fast::powr` reads the
/// same classes for its NaN rule as well ([`fast_powr_special`]).
pub(super) fn infinite_product(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    y: Value,
    x: Value,
    r: Value,
) -> BResult<Value> {
    if !matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half)) {
        return Ok(r);
    }
    let c = PowrOperands::of(fb, f, x, y)?;
    // `log2(x)` infinite; then a nonzero `y` makes the product infinite.
    let log_infinite = c.log_infinite(fb)?;
    let y_nonzero = fb.unary(UnOp::LogicalNot, c.y_zero)?;
    let by_base = bin(fb, BinOp::LogicalAnd, log_infinite, y_nonzero)?;
    // An infinite `y` times a `log2(x)` that is not zero.
    let x_not_one = fb.unary(UnOp::LogicalNot, c.x_one)?;
    let by_exponent = bin(fb, BinOp::LogicalAnd, c.y_inf, x_not_one)?;
    let infinite = bin(fb, BinOp::LogicalOr, by_base, by_exponent)?;
    let numbers = c.numbers(fb, f)?;
    let infinite = bin(fb, BinOp::LogicalAnd, infinite, numbers)?;
    let answer = c.infinite_answer(fb, f)?;
    fb.select(infinite, answer, r)
}

/// `fun` of `args` (floats of form `f`) where float function `op` calls it: `fun`'s routine
/// where `op` takes it ([`libm::uses`]) and the device's instruction does not answer it, the
/// device's instruction `o` otherwise.
fn routine_or(
    fb: &mut FunctionBuilder<'_>,
    lib: &Lib<'_>,
    op: FloatFn,
    f: Form,
    fun: libm::Fun,
    o: GlslOp,
    args: &[Value],
) -> BResult<Value> {
    if libm::uses(op, f).contains(&fun) && !lib.device(fun) {
        lib.call(fb, fun, f, args)
    } else {
        glsl(fb, o, args)
    }
}

/// Metal's precise `exp10(x)` (`c` is `log2(10)`) or `exp(x)` (`log2(e)`) from the device's
/// `exp2` (floats of form `f` in a 32-bit container, the constants the container's), as Metal
/// composes it: `E = exp2(p)` of the rounded product `p = x c_hi`, `c_hi + c_lo = c`, corrected by
/// the product's error `d = fma(x, c_lo, fma(x, c_hi, -p))` as `fma(E, d ln 2, E)`, and `E` itself
/// where it is `+inf` or `+0` (where `x` is infinite `d` is a NaN). Every lane of case 7b283404
/// on the M3 and the M4 is this of Metal's `exp2`, for both, and so is every normal answer of
/// magnitude `[2^-40, 2^9)` on the M4. Each step is its own: no device fuses or regroups it.
fn metal_exp(fb: &mut FunctionBuilder<'_>, f: Form, x: Value, c: f64) -> BResult<Value> {
    let exact = Contraction::Forbidden;
    let (hi, lo) = libm::split(c);
    let c_hi = wide_splat(fb, f, f64::from(hi))?;
    let c_lo = wide_splat(fb, f, f64::from(lo))?;
    let p = fb.binary_with(BinOp::FMul, x, c_hi, exact)?;
    let minus_p = fb.unary(UnOp::FNegate, p)?;
    let e = glsl(fb, GlslOp::Fma, &[x, c_hi, minus_p])?;
    let d = glsl(fb, GlslOp::Fma, &[x, c_lo, e])?;
    let big = glsl(fb, GlslOp::Exp2, &[p])?;
    let ln2 = wide_splat(fb, f, f64::from(std::f64::consts::LN_2 as f32))?;
    let t = fb.binary_with(BinOp::FMul, d, ln2, exact)?;
    let r = glsl(fb, GlslOp::Fma, &[big, t, big])?;
    let inf = has_bits(fb, f, big, 0x7F80_0000)?;
    let zero = has_bits(fb, f, big, 0)?;
    let ends = bin(fb, BinOp::LogicalOr, inf, zero)?;
    fb.select(ends, big, r)
}

/// Metal's precise `log(x)` or `log10(x)` from the device's `log2` (floats of form `f` in a
/// 32-bit container, the constants the container's), as Metal composes it: its `log2` times `RN(ln
/// 2)` or `RN(log10 2)` (the constant `c`, by its bits), and strictly inside `near`'s range next to
/// 1 its polynomial `RN(t fma(p(t), t, lead))` of `t = x - 1`, by fused steps. Each step is its
/// own: no device fuses or regroups it.
fn metal_log(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    near: &libm::log::Near,
    c: u32,
) -> BResult<Value> {
    let exact = Contraction::Forbidden;
    let l = glsl(fb, GlslOp::Log2, &[x])?;
    let k = wide_splat(fb, f, f64::from(f32::from_bits(c)))?;
    let far = fb.binary_with(BinOp::FMul, l, k, exact)?;
    let one = wide_splat(fb, f, 1.0)?;
    let t = fb.binary_with(BinOp::FSub, x, one, exact)?;
    let (last, rest) = near
        .p
        .split_last()
        .ok_or(BuildError::Type(TypeRule::FunctionSignature))?;
    let mut p = wide_splat(fb, f, f64::from(f32::from_bits(*last)))?;
    for c in rest.iter().rev() {
        let c = wide_splat(fb, f, f64::from(f32::from_bits(*c)))?;
        p = glsl(fb, GlslOp::Fma, &[p, t, c])?;
    }
    let lead = wide_splat(fb, f, f64::from(f32::from_bits(near.lead)))?;
    let q = glsl(fb, GlslOp::Fma, &[p, t, lead])?;
    let close = fb.binary_with(BinOp::FMul, t, q, exact)?;
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let first = splat(fb, u, u64::from(near.low + 1))?;
    let offset = bin(fb, BinOp::ISub, b, first)?;
    let span = splat(fb, u, u64::from(near.end - near.low - 1))?;
    let inside = bin(fb, BinOp::ULessThan, offset, span)?;
    fb.select(inside, close, far)
}

/// Metal's `fast::sinh(x)` or `fast::cosh(x)` (`op`) on `float` lanes `x` (form `f`): MSL Table
/// 8.2's `(e(x) - e(-x)) · 0.5` and `(e(x) + e(-x)) · 0.5` of its fast `exp`,
/// `e(v) = exp2(v · log2 e)` ([`routine_or`]; `-x · log2 e` is the product negated). One
/// operation to Metal: no step is fused with another or with the caller's arithmetic. On case
/// f6524207 the formula of correctly rounded `exp2` matches Metal on 768 and 822 of 1,024 lanes,
/// every special one among them (a zero and a denormal give `+0` and `1`, an infinity `±inf` and
/// `+inf`); the rest is Metal's `exp2`, which is not correctly rounded.
fn fast_sinh_cosh(
    fb: &mut FunctionBuilder<'_>,
    op: FloatFn,
    f: Form,
    x: Value,
    lib: &Lib<'_>,
) -> BResult<Value> {
    let exact = Contraction::Forbidden;
    let k = wide_splat(fb, f, std::f64::consts::LOG2_E)?;
    let p = fb.binary_with(BinOp::FMul, x, k, exact)?;
    let up = routine_or(fb, lib, op, f, libm::Fun::Exp2, GlslOp::Exp2, &[p])?;
    let n = fb.unary(UnOp::FNegate, p)?;
    let down = routine_or(fb, lib, op, f, libm::Fun::Exp2, GlslOp::Exp2, &[n])?;
    let combine = if op == FloatFn::FastSinh {
        BinOp::FSub
    } else {
        BinOp::FAdd
    };
    let s = fb.binary_with(combine, up, down, exact)?;
    let half = wide_splat(fb, f, 0.5)?;
    fb.binary_with(BinOp::FMul, s, half, exact)
}

/// Metal's fast `log(v)` of `float` lanes `v` (form `f`) where float function `op` calls it:
/// `log2(v) · ln 2` ([`routine_or`]), each step its own.
fn fast_log_of(
    fb: &mut FunctionBuilder<'_>,
    lib: &Lib<'_>,
    op: FloatFn,
    f: Form,
    v: Value,
) -> BResult<Value> {
    let l = routine_or(fb, lib, op, f, libm::Fun::Log2, GlslOp::Log2, &[v])?;
    let c = wide_splat(fb, f, std::f64::consts::LN_2)?;
    fb.binary_with(BinOp::FMul, l, c, Contraction::Forbidden)
}

/// Metal's `fast::sqrt(x)` on `float` from `r`, the device's `GLSL.std.450 InverseSqrt(x)`
/// (probe case 08bc6611, every lane): `x · r`, whose product is NaN at a zero and an infinity, so
/// those answer as Metal does: a zero or an `f32` denormal gives the zero of its sign
/// ([`flushed`]), and `+inf` gives `+inf`. A negative `x` or a NaN gives NaN.
pub(super) fn fast_sqrt_of(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    r: Value,
) -> BResult<Value> {
    let p = bin(fb, BinOp::FMul, x, r)?;
    let inf = wide_splat(fb, f, f64::INFINITY)?;
    let at_inf = bin(fb, BinOp::FOrdEqual, x, inf)?;
    let p = fb.select(at_inf, inf, p)?;
    let zero = metal_zero(fb, f, x)?;
    let z = wide_splat(fb, f, 0.0)?;
    let signed = copysign(fb, f, z, x)?;
    fb.select(zero, signed, p)
}

/// The C99 Annex F (F.9.4.4) special cases of `pow(x, y)` around `magnitude`, the value
/// `GLSL.std.450 Pow(|x|, y)` gives, as Metal answers them (A1 case 116e052a, every lane of
/// precise `float`, fast `float`, and `half`). Later selects win:
///
/// - `pow(±0, y)` is `+inf` for `y < 0` and `+0` for `y > 0` (signed for odd integral `y`);
///   an `f32` denormal base is a zero ([`flushed`]);
/// - `pow(-1, y)` is `1` for integral or infinite `y` (odd integers negate below);
/// - a negative base to an odd integer negates; a negative finite nonzero base (not an `f32`
///   denormal, which is `-0`) to a non-integral finite exponent is NaN;
/// - a NaN operand gives the canonical NaN (`0x7fc00000`, a `half` `0x7e00`), whatever its
///   payload or sign: this overrides the zero-base and `-1` results, because C99 makes
///   `pow(±0, NaN)` and `pow(-1, NaN)` NaN, and it does not trust `Pow` to propagate a NaN;
/// - `pow(x, 1)` is `x` for every other `x`, and precise `pow` on `float` (`precise`, not
///   `half`) keeps a NaN `x` itself, bit for bit (`pow(-NaN, 1)` is `0xffc00000`, where
///   `fast::pow` and `half` answer the canonical NaN); `fast::pow` reads an `f32` denormal `x`
///   as the zero of its sign there (`fast::pow(-2^-149, 1)` is `-0`), where precise `pow`
///   answers the denormal itself (the denormal probes' `pow` word, cases 638c7906 and
///   c9b6bacb, on the M3 Max and the M4);
/// - `pow(+1, y)` is `1` for every `y`, even a NaN;
/// - `pow(x, ±0)` is `1` for every `x`, even a NaN, and so is `pow(x, y)` for an `f32`
///   denormal `y` ([`flushed`]).
pub(super) fn pow_special(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
    magnitude: Value,
    precise: bool,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: f64| wide_splat(fb, f, k);
    let (zero, one, half) = (c(fb, 0.0)?, c(fb, 1.0)?, c(fb, 0.5)?);
    let (inf, ninf, nan) = (
        c(fb, f64::INFINITY)?,
        c(fb, f64::NEG_INFINITY)?,
        c(fb, f64::NAN)?,
    );
    // Whether an operand is a zero (an `f32` denormal is one to Metal).
    let is_zero = |fb: &mut FunctionBuilder<'_>, v: Value| -> BResult<Value> {
        match flushed(fb, f, v)? {
            Some(t) => Ok(t),
            None => bin(fb, BinOp::FOrdEqual, v, zero),
        }
    };
    // y integral; y odd (an odd integer halves to a fraction of exactly one half).
    let ty_ = glsl(fb, GlslOp::Trunc, &[y])?;
    let yint = bin(fb, BinOp::FOrdEqual, ty_, y)?;
    let hy = bin(fb, BinOp::FMul, y, half)?;
    let fr = glsl(fb, GlslOp::Fract, &[hy])?;
    let yodd = bin(fb, BinOp::FOrdEqual, fr, half)?;
    // Zero base.
    let xz = is_zero(fb, x)?;
    let yneg = bin(fb, BinOp::FOrdLessThan, y, zero)?;
    let zb = fb.select(yneg, inf, zero)?;
    let base = fb.select(xz, zb, magnitude)?;
    // A base of magnitude one.
    let ax = glsl(fb, GlslOp::FAbs, &[x])?;
    let x1 = bin(fb, BinOp::FOrdEqual, ax, one)?;
    let base = fb.select(x1, one, base)?;
    // The sign bit of x (so -0 and -inf count).
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
    let k31 = splat(fb, u, 31)?;
    let sb = bin(fb, BinOp::ShiftRightLogical, xb, k31)?;
    let uz = splat(fb, u, 0)?;
    let sneg = bin(fb, BinOp::INotEqual, sb, uz)?;
    let flip = bin(fb, BinOp::LogicalAnd, sneg, yodd)?;
    let nb = fb.unary(UnOp::FNegate, base)?;
    let r = fb.select(flip, nb, base)?;
    // A negative finite nonzero base to a non-integral exponent (a negative `f32` denormal is
    // `-0` to Metal, so `pow` of it answers as the zero base does).
    let xneg = bin(fb, BinOp::FOrdLessThan, x, zero)?;
    let xfin = bin(fb, BinOp::FOrdGreaterThan, x, ninf)?;
    let fractional = fb.unary(UnOp::LogicalNot, yint)?;
    let nonzero = fb.unary(UnOp::LogicalNot, xz)?;
    let bad = bin(fb, BinOp::LogicalAnd, xneg, xfin)?;
    let bad = bin(fb, BinOp::LogicalAnd, bad, fractional)?;
    let bad = bin(fb, BinOp::LogicalAnd, bad, nonzero)?;
    let r = fb.select(bad, nan, r)?;
    // NaN operands give the canonical NaN (after the zero-base and -1 cases, which they
    // override).
    let ynan = fb.unary(UnOp::IsNan, y)?;
    let xnan = fb.unary(UnOp::IsNan, x)?;
    let any_nan = bin(fb, BinOp::LogicalOr, xnan, ynan)?;
    let r = fb.select(any_nan, nan, r)?;
    // pow(x, 1) = x, an `f32` denormal `x` flushed under `fast::pow`.
    let y1 = bin(fb, BinOp::FOrdEqual, y, one)?;
    let keep = if precise && f.float_fmt() != Some(FloatFmt::Half) {
        y1
    } else {
        let number = fb.unary(UnOp::LogicalNot, xnan)?;
        bin(fb, BinOp::LogicalAnd, y1, number)?
    };
    let kept = if !precise && f.float_fmt() == Some(FloatFmt::F32) {
        denormal::flush(fb, f, x)?
    } else {
        x
    };
    let r = fb.select(keep, kept, r)?;
    let xp1 = bin(fb, BinOp::FOrdEqual, x, one)?;
    let r = fb.select(xp1, one, r)?;
    let y0 = is_zero(fb, y)?;
    fb.select(y0, one, r)
}

/// Whether lanes of `v` (floats of form `f`) are zeros to Metal: either zero, and an `f32`
/// denormal ([`flushed`]).
fn metal_zero(fb: &mut FunctionBuilder<'_>, f: Form, v: Value) -> BResult<Value> {
    match flushed(fb, f, v)? {
        Some(t) => Ok(t),
        None => {
            let zero = wide_splat(fb, f, 0.0)?;
            bin(fb, BinOp::FOrdEqual, v, zero)
        }
    }
}

/// Whether the sign bit of `v` (floats of form `f`, held in an `f32` container) is set, so `-0`,
/// a negative denormal, and `-inf` count.
fn sign_set(fb: &mut FunctionBuilder<'_>, f: Form, v: Value) -> BResult<Value> {
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let k31 = splat(fb, u, 31)?;
    let s = bin(fb, BinOp::ShiftRightLogical, bits, k31)?;
    let uz = splat(fb, u, 0)?;
    bin(fb, BinOp::INotEqual, s, uz)
}

/// The C99 Annex F (F.9.1.4) special cases of `atan2(y, x)` around `generic`, the value
/// `GLSL.std.450 Atan2(y, x)` gives, as Metal answers them (probe case 6802ab13, every lane of
/// sixteen operands squared on precise `float` and `half`, and A1 case 58e56ca3). Vulkan leaves
/// `Atan2` undefined where both operands are zero and bounds it only to 4096 ulp elsewhere, so
/// a zero's sign and the axes' angles are the device's (llvmpipe answers `atan2(-0, 1) = +0` and
/// `atan2(inf, inf)` 55 ulp below `π/4`). Each special answer is a constant with `y`'s sign:
///
/// - a zero `y` gives `±0` toward a positive `x` (`+0` included) and `±π` toward a negative one
///   (`-0` and `-inf` included);
/// - an infinite `y` gives `±π/4` toward `+inf`, `±3π/4` toward `-inf`, and `±π/2` otherwise;
/// - a finite nonzero `y` gives `±π/2` over a zero `x`, `±0` toward `+inf`, and `±π` toward
///   `-inf`;
/// - a NaN operand gives the canonical NaN (`0x7fc00000`, a `half` `0x7e00`).
///
/// The constants are `π`, `π/2`, `π/4`, and `3π/4` correctly rounded to the format (Metal's
/// answers: `0x40490fdb`, `0x3fc90fdb`, `0x3f490fdb`, `0x4016cbe4`; `half` `0x4248`, `0x3e48`,
/// `0x3a48`, `0x40b6`, which the final rounding gives from the `f32` constants). On `float` an
/// `f32` denormal operand is a zero and a generic answer of an `f32` denormal (or zero) is `+0`,
/// whatever its sign (`atan2(-1, FLT_MAX)` is `+0`); a `half` keeps its denormals.
pub(super) fn atan2_special(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    y: Value,
    x: Value,
    generic: Value,
) -> BResult<Value> {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let c = |fb: &mut FunctionBuilder<'_>, k: f64| wide_splat(fb, f, k);
    let (zero, half_pi, pi) = (c(fb, 0.0)?, c(fb, FRAC_PI_2)?, c(fb, PI)?);
    let (quarter, three_quarters) = (c(fb, FRAC_PI_4)?, c(fb, 3.0 * FRAC_PI_4)?);
    let nan = c(fb, f64::NAN)?;
    let r = zero_where_flushed(fb, f, generic, generic)?;
    let (yz, xz) = (metal_zero(fb, f, y)?, metal_zero(fb, f, x)?);
    let (yi, xi) = (fb.unary(UnOp::IsInf, y)?, fb.unary(UnOp::IsInf, x)?);
    let xneg = sign_set(fb, f, x)?;
    // The answer on the x axis: 0 toward +x, π toward -x.
    let axis = fb.select(xneg, pi, zero)?;
    // A zero x (y finite nonzero): π/2; otherwise (an infinite x) the axis.
    let k = fb.select(xz, half_pi, axis)?;
    let diagonal = fb.select(xneg, three_quarters, quarter)?;
    let at_inf = fb.select(xi, diagonal, half_pi)?;
    let k = fb.select(yi, at_inf, k)?;
    let k = fb.select(yz, axis, k)?;
    let y_special = bin(fb, BinOp::LogicalOr, yz, yi)?;
    let x_special = bin(fb, BinOp::LogicalOr, xz, xi)?;
    let special = bin(fb, BinOp::LogicalOr, y_special, x_special)?;
    let signed = copysign(fb, f, k, y)?;
    let r = fb.select(special, signed, r)?;
    let ynan = fb.unary(UnOp::IsNan, y)?;
    let xnan = fb.unary(UnOp::IsNan, x)?;
    let any_nan = bin(fb, BinOp::LogicalOr, ynan, xnan)?;
    fb.select(any_nan, nan, r)
}

/// The arc tangent of a quotient (`GLSL.std.450 Atan`; a unit test substitutes a stand-in
/// because the interpreter does not model it).
pub(super) type AtanOf<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, Value) -> BResult<Value>;

/// Metal `fast::atan2(y, x)` on `float` (probe case c11eb8cb, every lane of sixteen operands
/// squared, in both of its compiles): `atan(q)` of the quotient `q = y / x`, turned by `π` when
/// `x` is below zero (by `-π` when `y` is, too), with Metal's special values:
///
/// - an `f32` denormal operand is a zero, and a quotient that is an `f32` denormal is a zero;
/// - the quotient's sign is the exclusive or of the operands' sign bits, so a zero quotient
///   gives that signed zero and an infinite one (a zero `x`, an infinite `y`, or an overflow)
///   `±π/2`;
/// - `0 / 0`, `inf / inf`, and a NaN operand give the canonical NaN (`0x7fc00000`);
/// - the turn reads `x < 0` and `y >= 0` as comparisons, after the flush: `fast::atan2(-0, -1)`
///   is `+π` (where `atan2` answers `-π`), and `fast::atan2(1, -0)` is `-π/2`.
///
/// Beyond `2^126` the quotient follows the call's fast-math flags. A call that carries every
/// flag (case c11eb8cb, compiled with fast math) divides over the whole range
/// (`fast::atan2(FLT_MAX, FLT_MAX)` is `atan(1)`): the division is scaled by `1/4` when `|x|` is
/// at least `2^126`, an exact scaling that keeps the divisor in the range where Metal's fast
/// division and Vulkan's `OpFDiv` bound their error. Any other call (`reciprocal_flushed`; case
/// 6802ab13, compiled without fast math, measured with no flag) multiplies `y` by `x`'s
/// reciprocal flushed to zero beyond `2^126` (the reciprocal is an `f32` denormal there), so the
/// quotient is the signed zero, or NaN for an infinite `y` (`fast::atan2(FLT_MAX, FLT_MAX)` is
/// `+0`, `fast::atan2(inf, FLT_MAX)` is NaN). A call with only some of the flags is not
/// measured; it takes the second answer, which differs only where the divisor leaves the range
/// MSL bounds the fast division in.
///
/// Every finite normal pair of our probe (16.7 million, the M4) is `atan(q)` of Metal's fast
/// `atan` and its fast quotient `q = y · rcp(x)`, and `RN(atan(q) ± RN(π))` toward a negative
/// `x`: `divide` gives `q` (Metal's reciprocal's routine, or the device's quotient where it is
/// Metal's), `atan` the arc tangent (Metal's fast `atan`'s routine, or the device's where it is
/// Metal's). Every special value above is exact.
pub(super) fn fast_atan2(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    y: Value,
    x: Value,
    reciprocal_flushed: bool,
    atan: AtanOf<'_>,
    divide: DivideOf<'_>,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: f64| wide_splat(fb, f, k);
    let zero = c(fb, 0.0)?;
    let (pi, minus_pi) = (c(fb, std::f64::consts::PI)?, c(fb, -std::f64::consts::PI)?);
    let half_pi = c(fb, std::f64::consts::FRAC_PI_2)?;
    let nan = c(fb, f64::NAN)?;
    let (yz, xz) = (metal_zero(fb, f, y)?, metal_zero(fb, f, x)?);
    let (yi, xi) = (fb.unary(UnOp::IsInf, y)?, fb.unary(UnOp::IsInf, x)?);
    let ax = glsl(fb, GlslOp::FAbs, &[x])?;
    let limit = c(fb, 2f64.powi(126))?;
    // The quotient, and where the reciprocal is flushed (`None` when it is not).
    let (q, beyond) = if reciprocal_flushed {
        let q = divide(fb, y, x)?;
        (q, Some(bin(fb, BinOp::FOrdGreaterThan, ax, limit)?))
    } else {
        // Both operands scaled by 1/4 when the divisor is at least 2^126.
        let (one, quarter) = (c(fb, 1.0)?, c(fb, 0.25)?);
        let big = bin(fb, BinOp::FOrdGreaterThanEqual, ax, limit)?;
        let s = fb.select(big, quarter, one)?;
        let ys = bin(fb, BinOp::FMul, y, s)?;
        let xs = bin(fb, BinOp::FMul, x, s)?;
        (divide(fb, ys, xs)?, None)
    };
    // The quotient's sign: the exclusive or of the operands' sign bits.
    let u = narrow::u32_form(f);
    let ut = ty(fb, u)?;
    let yb = fb.convert(ConvOp::Bitcast, y, ut)?;
    let xb = fb.convert(ConvOp::Bitcast, x, ut)?;
    let sb = bin(fb, BinOp::BitwiseXor, yb, xb)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let sb = bin(fb, BinOp::BitwiseAnd, sb, sign)?;
    let ft = ty(fb, f)?;
    let signed_zero = fb.convert(ConvOp::Bitcast, sb, ft)?;
    let signed_half_pi = copysign(fb, f, half_pi, signed_zero)?;
    // A zero quotient: a zero y, an infinite x, or a quotient Metal flushes.
    let q_flushed = metal_zero(fb, f, q)?;
    let qz = bin(fb, BinOp::LogicalOr, yz, xi)?;
    let qz = bin(fb, BinOp::LogicalOr, qz, q_flushed)?;
    let qz = match beyond {
        Some(b) => bin(fb, BinOp::LogicalOr, qz, b)?,
        None => qz,
    };
    // An infinite quotient: a zero x, an infinite y, or a quotient that overflows.
    let q_inf = fb.unary(UnOp::IsInf, q)?;
    let qi = bin(fb, BinOp::LogicalOr, xz, yi)?;
    let qi = bin(fb, BinOp::LogicalOr, qi, q_inf)?;
    let a = atan(fb, q)?;
    let a = fb.select(qz, signed_zero, a)?;
    let a = fb.select(qi, signed_half_pi, a)?;
    // The turn, read after the flush: x below zero, by π when y is at least zero.
    let below = bin(fb, BinOp::FOrdLessThan, x, zero)?;
    let x_nonzero = fb.unary(UnOp::LogicalNot, xz)?;
    let xneg = bin(fb, BinOp::LogicalAnd, below, x_nonzero)?;
    let at_least = bin(fb, BinOp::FOrdGreaterThanEqual, y, zero)?;
    let ynonneg = bin(fb, BinOp::LogicalOr, at_least, yz)?;
    let turn = fb.select(ynonneg, pi, minus_pi)?;
    let turned = bin(fb, BinOp::FAdd, a, turn)?;
    let r = fb.select(xneg, turned, a)?;
    // 0 / 0, inf / inf (or an infinite y times a flushed reciprocal), and NaN operands.
    let zeros = bin(fb, BinOp::LogicalAnd, yz, xz)?;
    let infinite_x = match beyond {
        Some(b) => bin(fb, BinOp::LogicalOr, xi, b)?,
        None => xi,
    };
    let infs = bin(fb, BinOp::LogicalAnd, yi, infinite_x)?;
    let ynan = fb.unary(UnOp::IsNan, y)?;
    let xnan = fb.unary(UnOp::IsNan, x)?;
    let undefined = bin(fb, BinOp::LogicalOr, zeros, infs)?;
    let undefined = bin(fb, BinOp::LogicalOr, undefined, ynan)?;
    let undefined = bin(fb, BinOp::LogicalOr, undefined, xnan)?;
    fb.select(undefined, nan, r)
}

/// The container bits of the largest value below 1.0 of `f`'s format.
fn below_one_bits(f: Form) -> u64 {
    let significand = match f.float_fmt() {
        Some(FloatFmt::F64) => 53,
        Some(FloatFmt::F32) => 24,
        Some(FloatFmt::Half) => 11,
        _ => 8,
    };
    f.const_bits(1.0 - 2f64.powi(-significand))
}

/// The bits of `-2^-(p + 1)` in `f`'s format of `p` significand bits: the least negative `x`
/// whose `1 + x` rounds to one rather than to the largest value below it (the tie rounds to one's
/// even significand), so every `x` in `[-2^-(p + 1), 0)` has `x - floor(x)` rounded to one.
fn rounds_to_one_bits(f: Form) -> u64 {
    let significand = match f.float_fmt() {
        Some(FloatFmt::F64) => 53,
        Some(FloatFmt::F32) => 24,
        Some(FloatFmt::Half) => 11,
        _ => 8,
    };
    f.const_bits(-(2f64.powi(-(significand + 1))))
}

/// The (mantissa bits, exponent bias, all-ones exponent) of a float width.
pub(super) fn float_layout(bits: u8) -> (u32, i64, i64) {
    match bits {
        64 => (52, 1023, 2047),
        32 => (23, 127, 255),
        _ => (10, 15, 31),
    }
}

/// The bits of `2^k` in a `bits`-wide float (infinity when it overflows).
pub(super) fn pow2_bits(bits: u8, k: u32) -> u64 {
    let (m, bias, top) = float_layout(bits);
    let e = i64::from(k) + bias;
    if e >= top {
        (top as u64) << m
    } else {
        (e as u64) << m
    }
}

/// The value of the container float with bit pattern `b`: an `f64` when `bits` is 64, else an
/// `f32` (every narrow float is held in an `f32`).
pub(super) fn float_value(bits: u8, b: u64) -> f64 {
    if bits == 64 {
        f64::from_bits(b)
    } else {
        f64::from(f32::from_bits(b as u32))
    }
}

/// The bits of the largest finite float below `2^k` (`k >= 1`).
pub(super) fn below_pow2_bits(bits: u8, k: u32) -> u64 {
    let (m, bias, top) = float_layout(bits);
    let e = (i64::from(k) - 1 + bias).min(top - 1);
    ((e as u64) << m) | ((1u64 << m) - 1)
}

/// How `air.convert` converts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConvertKind {
    /// Float to float (`FConvert`, or the value itself at one width).
    Float,
    /// Integer (`signed` source) to float.
    ToFloat { signed: bool },
    /// Float to integer, saturating: NaN to 0, out-of-range to the destination's bound.
    ToInt { signed: bool },
    /// Integer (`signed` source) to integer.
    Int { signed: bool },
    /// An 8-bit float (carried in an `i8`) to a float: exact.
    FromFp8(narrow::Fp8Layout),
    /// An 8-bit float (carried in an `i8`) to an integer: its value as an `f32` (exact, as
    /// [`ConvertKind::FromFp8`]) converted as [`ConvertKind::ToInt`] (saturating, NaN to 0).
    Fp8ToInt {
        fp8: narrow::Fp8Layout,
        signed: bool,
    },
    /// A `float` or `half` to `bool`: true unless the value is a zero or an `f32` denormal, so a
    /// NaN is true. Metal's reading (A1 cases 099ba79b and 9a824047, with and without fast math) is
    /// C++'s `x != 0` with `float` denormals flushed; the exponent field of the `f32` container
    /// decides it, whatever the device's denormal mode (a `half` is held widened, so its denormals
    /// are normal `f32` values and true, as Metal answers).
    ToBool,
    /// An integer to `bool`: C++'s `x != 0`. MSL's `bool`-vector conversion of an integer vector is
    /// what the front end emits as this `air.convert`; a scalar conversion is the front end's own
    /// `icmp ne x, 0`. Signedness does not matter.
    IntToBool,
}

/// Emit `air.convert` of `x` from `from` to `to`, a float source's NaN told by `test` (the
/// entry's [`NanTest::for_width`] of the source's container).
pub fn convert(
    fb: &mut FunctionBuilder<'_>,
    k: ConvertKind,
    from: Form,
    to: Form,
    x: Value,
    test: NanTest,
) -> BResult<Value> {
    use crate::lower::arith::ops::cast;
    match k {
        ConvertKind::FromFp8(l) => {
            let u = narrow::u32_form(from);
            let v = narrow::fp8_to_f32(fb, l, u, to, x)?;
            // Every 8-bit float value is a value of every wider format: no rounding.
            if to.float_fmt() == Some(FloatFmt::F64) {
                let t = ty(fb, to)?;
                fb.convert(ConvOp::FConvert, v, t)
            } else {
                Ok(v)
            }
        }
        ConvertKind::Fp8ToInt { fp8, signed } => {
            let f = from.with(Held::Float(FloatFmt::F32));
            let v = narrow::fp8_to_f32(fb, fp8, narrow::u32_form(from), f, x)?;
            convert(fb, ConvertKind::ToInt { signed }, f, to, v, test)
        }
        ConvertKind::Float => cast(fb, CastKind::FpResize, from, to, x),
        // A held integer is zero-extended in its container, so the container is zero exactly
        // when the value is.
        ConvertKind::IntToBool => {
            let zero = splat(fb, from, 0)?;
            bin(fb, BinOp::INotEqual, x, zero)
        }
        ConvertKind::ToBool => {
            let u = narrow::u32_form(from);
            let ut = ty(fb, u)?;
            let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
            let exp = splat(fb, u, 0x7F80_0000)?;
            let e = bin(fb, BinOp::BitwiseAnd, bits, exp)?;
            let zero = splat(fb, u, 0)?;
            bin(fb, BinOp::INotEqual, e, zero)
        }
        ConvertKind::ToFloat { signed } => cast(
            fb,
            if signed {
                CastKind::SiToFp
            } else {
                CastKind::UiToFp
            },
            from,
            to,
            x,
        ),
        ConvertKind::Int { signed } => {
            if from.is_bool() {
                return cast(fb, CastKind::ZExt, from, to, x);
            }
            let widening = to.bits() >= from.bits();
            let c = match (widening, signed) {
                (true, true) => CastKind::SExt,
                (true, false) => CastKind::ZExt,
                (false, _) => CastKind::Trunc,
            };
            if widening && to.bits() == from.bits() {
                return Ok(x);
            }
            cast(fb, c, from, to, x)
        }
        ConvertKind::ToInt { signed } => {
            let n = u32::from(to.bits());
            // A narrow float is exactly its container's value: clamp in the container.
            let fw = from.container();
            let k = if signed { n - 1 } else { n };
            // Clamp into [lo, largest float below 2^k] first, so the conversion is in range. A
            // signed `lo` is `-2^k`, a float of both containers (`k <= 63`): below it the clamp
            // gives the minimum.
            let lo = if signed {
                splat(fb, from, pow2_bits(fw, k) | (1u64 << (fw - 1)))?
            } else {
                fsplat(fb, from, 0.0)?
            };
            let hi_bits = below_pow2_bits(fw, k);
            let hi = splat(fb, from, hi_bits)?;
            let c = glsl(fb, GlslOp::NClamp, &[x, lo, hi])?;
            let t = ty(fb, to)?;
            let op = if signed {
                ConvOp::ConvertFToS
            } else {
                ConvOp::ConvertFToU
            };
            let mut r = fb.convert(op, c, t)?;
            // At or above `2^k` the result is the maximum. The clamp already gives it when the
            // largest float below `2^k` truncates to `2^k - 1` (the float holds every integer up
            // to it); otherwise it is selected.
            if float_value(fw, hi_bits).trunc() < (2f64.powi(k as i32) - 1.0) {
                let limit = splat(fb, from, pow2_bits(fw, k))?;
                let over = bin(fb, BinOp::FOrdGreaterThanEqual, x, limit)?;
                let max = splat(fb, to, ones(if signed { to.bits() - 1 } else { to.bits() }))?;
                r = fb.select(over, max, r)?;
            }
            // `NClamp` takes NaN to the lower bound; Metal converts it to 0, the unsigned lower
            // bound. Without `SignedZeroInfNanPreserve` for the container a device may assume no
            // NaN, in `NClamp` and `OpIsNan` alike, so the bits tell it.
            if signed || test == NanTest::Bits {
                let nan = nan::is_nan(fb, from, test, x)?;
                let z = splat(fb, to, 0)?;
                r = fb.select(nan, z, r)?;
            }
            mask(fb, to, r)
        }
    }
}

/// The packed-channel layout of a normalized `air.pack`/`air.unpack` format: each channel's
/// width, low channel first, and whether channels are signed-normalized; `None` for the packed
/// float formats (`rg11b10f`, `rgb9e5`, [`float_unpack_lanes`]). `rgb565` holds red in bits
/// 0..5, green in 5..11, and blue in 11..16 (device-measured by the old engine); the `snorm`
/// `rgb10a2` fields are two's complement at `rgb10a2`'s positions (measured).
pub fn pack_layout(format: PackFormat) -> Option<(&'static [u32], bool)> {
    Some(match format {
        PackFormat::Unorm4x8 | PackFormat::Unorm4x8Srgb => (&[8, 8, 8, 8], false),
        PackFormat::Snorm4x8 => (&[8, 8, 8, 8], true),
        PackFormat::Unorm2x16 => (&[16, 16], false),
        PackFormat::Snorm2x16 => (&[16, 16], true),
        PackFormat::UnormRgb10a2 => (&[10, 10, 10, 2], false),
        PackFormat::SnormRgb10a2 => (&[10, 10, 10, 2], true),
        PackFormat::UnormRgb565 => (&[5, 6, 5], false),
        PackFormat::UnormRg11b10f | PackFormat::UnormRgb9e5 => return None,
    })
}

/// How a normalized `air.pack` format rounds a channel `x` (clamped) times its scale `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackRounding {
    /// The exact product, to nearest even ([`round_exact_product_lanes`]; device-measured for
    /// `rgb10a2`).
    ExactProduct,
    /// The `f32` product `fl(x * m)`, to nearest even: Metal's `pack_float_to_unorm565` answers
    /// this on all 32 words of `the-rgb565-pack-at-the-exact-tie`, where the exact product
    /// disagrees on 8 (e.g. green `0.03968254 * 63`, exactly `2.50000009`, is `fl` = `2.5`, and
    /// Metal answers 2).
    FloatProduct,
}

/// The rounding of normalized pack format `format`: the float product for `rgb565` and the
/// `snorm` `rgb10a2`, whose Metal observations contradict the exact product; the exact product for
/// the others (measured for the `unorm` `rgb10a2`, unmeasured for the 8- and 16-bit formats).
pub fn pack_rounding(format: PackFormat) -> PackRounding {
    match format {
        PackFormat::UnormRgb565 | PackFormat::SnormRgb10a2 => PackRounding::FloatProduct,
        _ => PackRounding::ExactProduct,
    }
}

/// The normalized value a NaN channel of `air.pack.<format>` packs as: `-1` for the `snorm`
/// `rgb10a2`, which Metal measures so for every `float` and `half` NaN (it is what `clamp(x, -1,
/// 1)` gives when its `max` ignores a NaN), and 0 for the others (the MSL normalized conversions).
pub fn pack_nan(format: PackFormat) -> f64 {
    if format == PackFormat::SnormRgb10a2 {
        -1.0
    } else {
        0.0
    }
}

/// How a normalized `air.unpack` format converts a channel code `c` to a float.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnpackRounding {
    /// `c / d` correctly rounded ([`div_exact`]; snorm: of the magnitude clamped to `d`).
    ///
    /// [`div_exact`]: crate::lower::arith::exact_division::div_exact
    Quotient,
    /// `max(c * fl(1/d), -1)`, the product with the `f32` reciprocal rounded to nearest even
    /// ([`mul_exact`]): Metal's `snorm` `rgb10a2` on every code of every field, where the quotient
    /// differs on 267,279 of 1,049,600 measured words by one ULP.
    ReciprocalProduct,
}

/// The conversion of normalized unpack format `format`'s channels ([`UnpackRounding`]).
pub fn unpack_rounding(format: PackFormat) -> UnpackRounding {
    match format {
        PackFormat::SnormRgb10a2 => UnpackRounding::ReciprocalProduct,
        _ => UnpackRounding::Quotient,
    }
}

/// The float lanes of a packed float format this family unpacks (`rg11b10f`, `rgb9e5`: 3), or
/// `None` for a normalized one. Packing to them is not lowered (Metal's rounding, overflow, and NaN
/// rules for these encodings are not established).
pub fn float_unpack_lanes(format: PackFormat) -> Option<u8> {
    match format {
        PackFormat::UnormRg11b10f | PackFormat::UnormRgb9e5 => Some(3),
        _ => None,
    }
}

/// The AIR width of the packed integer of `format` (`rgb565` packs into a `ushort`).
pub fn packed_bits(format: PackFormat) -> u8 {
    if format == PackFormat::UnormRgb565 {
        16
    } else {
        32
    }
}

/// The error of an `air.pack`/`air.unpack` whose format the emission cannot take (a packed float
/// format to pack to, or the sRGB format without the module's table): selection rules both out.
fn unpackable() -> crate::spirv::error::BuildError {
    crate::spirv::error::BuildError::Operand {
        op: crate::spv::Op::ExtInst,
        rule: crate::spirv::error::OperandRule::Arity,
    }
}

/// `air.pack.<format>`: float lanes `x` (form `f`) to the packed integer (a `u32`; a `ushort`
/// for `rgb565`, zero-extended). Each channel is converted as Metal converts a float to a
/// normalized integer: clamped to `[0, 1]` (`[-1, 1]` for snorm; NaN to [`pack_nan`]), and the
/// product with `2^N - 1` (`2^(N-1) - 1`) rounded to the nearest even integer: the **exact**
/// product ([`round_exact_product_lanes`]), except for `rgb565` and the `snorm` `rgb10a2`, which
/// round the `f32` product ([`pack_rounding`]). The sRGB format encodes the color channels through
/// Metal's twelve-bit table ([`srgb::search`] of the exact product with 4095), which `srgb` (the
/// module's table) holds.
///
/// Every channel runs one lane of one vector sequence, its scale a per-lane constant: the
/// instructions are lane-wise, so each lane computes exactly what it computed alone. Only the sRGB
/// table reads are per channel. A unorm channel's result is at most its scale (`v <= 1`), so only a
/// snorm channel (two's complement) is masked to its field before it is placed.
pub fn pack(
    fb: &mut FunctionBuilder<'_>,
    format: PackFormat,
    f: Form,
    x: Value,
    srgb: Option<GlobalId>,
    test: NanTest,
) -> BResult<Value> {
    let Some((widths, snorm)) = pack_layout(format) else {
        return Err(unpackable());
    };
    if usize::from(f.lanes) != widths.len() {
        return Err(unpackable());
    }
    let s = f.with(Held::Float(FloatFmt::F32));
    let u = narrow::u32_form(s);
    let color = if format == PackFormat::Unorm4x8Srgb {
        3
    } else {
        0
    };
    // NaN converts to the format's value (0, even for the snorm formats where `NClamp` alone
    // would give -1; -1 for the `snorm` `rgb10a2`), chosen explicitly so the answer does not
    // depend on the device's NaN handling; `test` is the bits where the entry lets a device
    // fold `OpIsNan`.
    let lo = fsplat(fb, s, if snorm { -1.0 } else { 0.0 })?;
    let one = fsplat(fb, s, 1.0)?;
    let nan = nan::is_nan(fb, s, test, x)?;
    let z = fsplat(fb, s, pack_nan(format))?;
    let v = fb.select(nan, z, x)?;
    let v = glsl(fb, GlslOp::NClamp, &[v, lo, one])?;
    let scales: Vec<u32> = widths
        .iter()
        .enumerate()
        .map(|(i, w)| match (i < color, snorm) {
            (true, _) => 4095,
            (false, true) => (1u32 << (w - 1)) - 1,
            (false, false) => (1u32 << w) - 1,
        })
        .collect();
    let r = match pack_rounding(format) {
        PackRounding::ExactProduct => round_exact_product_lanes(fb, s, v, &scales, snorm)?,
        PackRounding::FloatProduct => {
            // The `f32` product, rounded to nearest even (two's complement for snorm).
            let m: Vec<u64> = scales.iter().map(|m| s.const_bits(f64::from(*m))).collect();
            let m = per_lane(fb, s, &m)?;
            let p = fb.binary_with(BinOp::FMul, v, m, Contraction::Forbidden)?;
            let p = glsl(fb, GlslOp::RoundEven, &[p])?;
            let ut = ty(fb, u)?;
            let op = if snorm {
                ConvOp::ConvertFToS
            } else {
                ConvOp::ConvertFToU
            };
            fb.convert(op, p, ut)?
        }
    };
    // The sRGB color channels' indices search the table one lane at a time.
    let r = if color > 0 {
        let table = srgb.ok_or_else(unpackable)?;
        let mut fields = Vec::with_capacity(widths.len());
        for i in 0..widths.len() {
            let field = fb.composite_extract(r, &[i as u32])?;
            fields.push(if i < color {
                srgb::search(fb, table, field)?
            } else {
                field
            });
        }
        let ut = ty(fb, u)?;
        fb.composite_construct(ut, &fields)?
    } else {
        r
    };
    let r = if snorm {
        let m: Vec<u64> = widths.iter().map(|w| ones(*w as u8)).collect();
        let m = per_lane(fb, u, &m)?;
        bin(fb, BinOp::BitwiseAnd, r, m)?
    } else {
        r
    };
    let mut at = Vec::with_capacity(widths.len());
    let mut next = 0u64;
    for w in widths {
        at.push(next);
        next += u64::from(*w);
    }
    let sh = per_lane(fb, u, &at)?;
    let r = bin(fb, BinOp::ShiftLeftLogical, r, sh)?;
    let mut acc = fb.composite_extract(r, &[0])?;
    for i in 1..widths.len() {
        let field = fb.composite_extract(r, &[i as u32])?;
        acc = bin(fb, BinOp::BitwiseOr, acc, field)?;
    }
    Ok(acc)
}

/// [`round_exact_product_lanes`] with one scale `m` for every lane.
#[cfg(test)]
pub fn round_exact_product(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    m: u32,
    snorm: bool,
) -> BResult<Value> {
    round_exact_product_lanes(fb, s, v, &[m], snorm)
}

/// `v` (`f32` lanes of form `s`, in `[0, 1]`, or `[-1, 1]` when `snorm`) times `m[i]` for lane
/// `i` (one scale for every lane when `m` has one; each below `2^16`), rounded to the nearest
/// integer, ties to even, **from the exact product**, as `u32` lanes (two's complement for a
/// negative `snorm` result). The sequence is lane-wise.
///
/// Rounding `fl(v * m)` differs where the multiply itself rounds onto a half-integer; Metal
/// rounds the exact product (measured for `rgb10a2` by the old engine's device case
/// `round-the-exact-product-into-ten-ten-ten-two`). The product is formed exactly in integers,
/// independent of the device's rounding mode and of whether it fuses: with `v = mant * 2^(e -
/// 150)` (`e` the biased exponent, `mant` the significand with its leading one), `v * m =
/// (mant * m) / 2^s` for `s = 150 - e >= 23`. `mant * m < 2^40` is the `UMulExtended` pair;
/// shifted right by 9 with a sticky bit it fits 31 bits, and `s >= 41` (which includes every
/// denormal and zero) gives a product below one half, so 0.
pub fn round_exact_product_lanes(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    m: &[u32],
    snorm: bool,
) -> BResult<Value> {
    let u = narrow::u32_form(s);
    let c = |fb: &mut FunctionBuilder<'_>, k: u64| splat(fb, u, k);
    let a = if snorm {
        glsl(fb, GlslOp::FAbs, &[v])?
    } else {
        v
    };
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, a, ut)?;
    let k23 = c(fb, 23)?;
    let e = bin(fb, BinOp::ShiftRightLogical, bits, k23)?;
    let man = c(fb, 0x7F_FFFF)?;
    let mant = bin(fb, BinOp::BitwiseAnd, bits, man)?;
    let lead = c(fb, 0x80_0000)?;
    let mant = bin(fb, BinOp::BitwiseOr, mant, lead)?;
    let m: Vec<u64> = m.iter().map(|m| u64::from(*m)).collect();
    let mk = per_lane(fb, u, &m)?;
    let p = bin(fb, BinOp::UMulExtended, mant, mk)?;
    let lo = fb.composite_extract(p, &[0])?;
    let hi = fb.composite_extract(p, &[1])?;
    // s = 150 - e; zero when s >= 41; t = min(s, 40) - 9 in [14, 31].
    let k150 = c(fb, 150)?;
    let sh = bin(fb, BinOp::ISub, k150, e)?;
    let k41 = c(fb, 41)?;
    let zero = bin(fb, BinOp::UGreaterThanEqual, sh, k41)?;
    let k40 = c(fb, 40)?;
    let sh = glsl(fb, GlslOp::UMin, &[sh, k40])?;
    let k9 = c(fb, 9)?;
    let t = bin(fb, BinOp::ISub, sh, k9)?;
    // x = (mant * m) >> 9 with the dropped bits as a sticky lowest bit.
    let low9 = c(fb, 0x1FF)?;
    let dropped = bin(fb, BinOp::BitwiseAnd, lo, low9)?;
    let z = c(fb, 0)?;
    let one = c(fb, 1)?;
    let lost = bin(fb, BinOp::INotEqual, dropped, z)?;
    let sticky = fb.select(lost, one, z)?;
    let x = bin(fb, BinOp::ShiftRightLogical, lo, k9)?;
    let up = bin(fb, BinOp::ShiftLeftLogical, hi, k23)?;
    let x = bin(fb, BinOp::BitwiseOr, x, up)?;
    let x = bin(fb, BinOp::BitwiseOr, x, sticky)?;
    // Round x / 2^t to nearest even.
    let q = bin(fb, BinOp::ShiftRightLogical, x, t)?;
    let unit = bin(fb, BinOp::ShiftLeftLogical, one, t)?;
    let below = bin(fb, BinOp::ISub, unit, one)?;
    let rem = bin(fb, BinOp::BitwiseAnd, x, below)?;
    let t1 = bin(fb, BinOp::ISub, t, one)?;
    let half = bin(fb, BinOp::ShiftLeftLogical, one, t1)?;
    let gt = bin(fb, BinOp::UGreaterThan, rem, half)?;
    let eq = bin(fb, BinOp::IEqual, rem, half)?;
    let lsb = bin(fb, BinOp::BitwiseAnd, q, one)?;
    let odd = bin(fb, BinOp::INotEqual, lsb, z)?;
    let tie_up = bin(fb, BinOp::LogicalAnd, eq, odd)?;
    let round_up = bin(fb, BinOp::LogicalOr, gt, tie_up)?;
    let inc = fb.select(round_up, one, z)?;
    let r = bin(fb, BinOp::IAdd, q, inc)?;
    let r = fb.select(zero, z, r)?;
    if !snorm {
        return Ok(r);
    }
    let fz = fsplat(fb, s, 0.0)?;
    let neg = bin(fb, BinOp::FOrdLessThan, v, fz)?;
    let nr = bin(fb, BinOp::ISub, z, r)?;
    fb.select(neg, nr, r)
}

/// `air.unpack.<format>`: the packed integer `w` (a `u32`; a `ushort` for `rgb565`) to float
/// lanes of form `f`. Each channel of a normalized format is converted as Metal converts a
/// normalized integer to a float: `c / (2^N - 1)` (snorm: `max(c / (2^(N-1) - 1), -1)`), the
/// quotient correctly rounded ([`div_exact`]), except the `snorm` `rgb10a2`, which multiplies by
/// the `f32` reciprocal ([`unpack_rounding`]); the sRGB format decodes the color channels through
/// Metal's twelve-bit table (`k / 4095` of the step [`srgb::step`]), which `srgb` (the module's
/// table) holds. A narrow form rounds each lane (so rounds twice, as Metal's `half` answers do).
/// The packed float formats decode exactly (`unpack_float`).
///
/// Every channel runs one lane of one vector sequence, its field and divisor per-lane constants:
/// the instructions are lane-wise, so each lane computes exactly what it computed alone. Only the
/// sRGB table reads are per channel.
///
/// [`div_exact`]: crate::lower::arith::exact_division::div_exact
pub fn unpack(
    fb: &mut FunctionBuilder<'_>,
    format: PackFormat,
    f: Form,
    w: Value,
    srgb: Option<GlobalId>,
) -> BResult<Value> {
    if float_unpack_lanes(format).is_some() {
        return unpack_float(fb, format, f, w);
    }
    let Some((widths, snorm)) = pack_layout(format) else {
        return Err(unpackable());
    };
    if usize::from(f.lanes) != widths.len() {
        return Err(unpackable());
    }
    let s = f.with(Held::Float(FloatFmt::F32));
    let u = narrow::u32_form(s);
    let ut = ty(fb, u)?;
    // Lane `i` holds field `i`: the word shifted down to the field and masked to its width.
    let mut at = Vec::with_capacity(widths.len());
    let mut next = 0u64;
    for n in widths {
        at.push(next);
        next += u64::from(*n);
    }
    let spread = fb.composite_construct(ut, &vec![w; widths.len()])?;
    let sh = per_lane(fb, u, &at)?;
    let v = bin(fb, BinOp::ShiftRightLogical, spread, sh)?;
    let m: Vec<u64> = widths.iter().map(|n| ones(*n as u8)).collect();
    let m = per_lane(fb, u, &m)?;
    let v = bin(fb, BinOp::BitwiseAnd, v, m)?;
    let q = if format == PackFormat::Unorm4x8Srgb {
        // The color channels' steps over 4095, the alpha channel's byte over 255.
        let table = srgb.ok_or_else(unpackable)?;
        let mut k = Vec::with_capacity(widths.len());
        for i in 0..widths.len() {
            let b = fb.composite_extract(v, &[i as u32])?;
            k.push(if i < 3 { srgb::step(fb, table, b)? } else { b });
        }
        let k = fb.composite_construct(ut, &k)?;
        let d: Vec<u32> = (0..widths.len())
            .map(|i| if i < 3 { 4095 } else { 255 })
            .collect();
        div_exact_lanes(fb, u, k, &d)?
    } else if snorm {
        let d: Vec<u32> = widths.iter().map(|n| (1u32 << (n - 1)) - 1).collect();
        let up: Vec<u64> = widths.iter().map(|n| u64::from(32 - n)).collect();
        let up = per_lane(fb, u, &up)?;
        let v = bin(fb, BinOp::ShiftLeftLogical, v, up)?;
        let v = bin(fb, BinOp::ShiftRightArithmetic, v, up)?;
        let z = splat(fb, u, 0)?;
        let neg = bin(fb, BinOp::SLessThan, v, z)?;
        let minus = bin(fb, BinOp::ISub, z, v)?;
        let mag = fb.select(neg, minus, v)?;
        match unpack_rounding(format) {
            UnpackRounding::Quotient => {
                // The magnitude, clamped to the scale (the most negative code is -1 exactly).
                let dk: Vec<u64> = d.iter().map(|d| u64::from(*d)).collect();
                let dk = per_lane(fb, u, &dk)?;
                let mag = glsl(fb, GlslOp::UMin, &[mag, dk])?;
                let q = div_exact_lanes(fb, u, mag, &d)?;
                let nq = fb.unary(UnOp::FNegate, q)?;
                fb.select(neg, nq, q)?
            }
            UnpackRounding::ReciprocalProduct => {
                // `max(c * fl(1/d), -1)`: the magnitude's product with its lane's reciprocal,
                // the code's sign, and the bound, which only the most negative code reaches
                // (`-512 * fl(1/511)` is below -1). Comparing with -1 is exact on every device.
                let r: Vec<u32> = d.iter().map(|d| (1.0f32 / *d as f32).to_bits()).collect();
                let p = mul_exact(fb, u, mag, &r)?;
                let pb = fb.convert(ConvOp::Bitcast, p, ut)?;
                let sign = splat(fb, u, 0x8000_0000)?;
                let nb = bin(fb, BinOp::BitwiseOr, pb, sign)?;
                let b = fb.select(neg, nb, pb)?;
                let st = ty(fb, s)?;
                let q = fb.convert(ConvOp::Bitcast, b, st)?;
                let m1 = fsplat(fb, s, -1.0)?;
                let below = bin(fb, BinOp::FOrdLessThan, q, m1)?;
                fb.select(below, m1, q)?
            }
        }
    } else {
        let d: Vec<u32> = widths.iter().map(|n| (1u32 << n) - 1).collect();
        div_exact_lanes(fb, u, v, &d)?
    };
    narrow::round(fb, f, q)
}

/// `air.unpack` of a packed float format (the field layouts are the old engine's, device-measured
/// against Metal's `rg11b10f<float3>` and `rgb9e5<float3>` reads):
///
/// - `rg11b10f`: red in bits 0..11 and green in 11..22 are unsigned floats with a 5-bit exponent
///   (bias 15) above a 6-bit mantissa; blue in 22..32 has a 5-bit mantissa. Each is the half
///   whose exponent and (left-aligned) mantissa bits they are, infinities and NaNs included.
/// - `rgb9e5`: three 9-bit mantissas (bits 0..9, 9..18, 18..27) share the exponent in bits
///   27..32: each channel is `mantissa * 2^(exponent - 24)`, without an implicit leading one.
///
/// Every channel value is a `half` value (so also a `float` one), so nothing rounds: the
/// `rg11b10f` fields convert with the exact half-bits decoding, and the
/// `rgb9e5` scale `2^(exponent - 24)` is a normal `f32` built from its bits, whose product with
/// the mantissa is exact.
fn unpack_float(
    fb: &mut FunctionBuilder<'_>,
    format: PackFormat,
    f: Form,
    w: Value,
) -> BResult<Value> {
    let s = f.scalar_form().with(Held::Float(FloatFmt::F32));
    let u = narrow::u32_form(s);
    let c = |fb: &mut FunctionBuilder<'_>, k: u64| splat(fb, u, k);
    let mut lanes = Vec::with_capacity(3);
    if format == PackFormat::UnormRgb9e5 {
        let k27 = c(fb, 27)?;
        let e = bin(fb, BinOp::ShiftRightLogical, w, k27)?;
        let k103 = c(fb, 103)?;
        let e = bin(fb, BinOp::IAdd, e, k103)?;
        let k23 = c(fb, 23)?;
        let e = bin(fb, BinOp::ShiftLeftLogical, e, k23)?;
        let st = ty(fb, s)?;
        let scale = fb.convert(ConvOp::Bitcast, e, st)?;
        for at in [0u64, 9, 18] {
            let v = if at == 0 {
                w
            } else {
                let k = c(fb, at)?;
                bin(fb, BinOp::ShiftRightLogical, w, k)?
            };
            let m9 = c(fb, 0x1FF)?;
            let v = bin(fb, BinOp::BitwiseAnd, v, m9)?;
            let v = fb.convert(ConvOp::ConvertUToF, v, st)?;
            lanes.push(bin(fb, BinOp::FMul, v, scale)?);
        }
    } else {
        let half = Form {
            held: Held::Float(FloatFmt::Half),
            lanes: 1,
        };
        for (at, man) in [(0u64, 6u64), (11, 6), (22, 5)] {
            let v = if at == 0 {
                w
            } else {
                let k = c(fb, at)?;
                bin(fb, BinOp::ShiftRightLogical, w, k)?
            };
            let km = c(fb, man)?;
            let e = bin(fb, BinOp::ShiftRightLogical, v, km)?;
            let m5 = c(fb, 0x1F)?;
            let e = bin(fb, BinOp::BitwiseAnd, e, m5)?;
            let k10 = c(fb, 10)?;
            let e = bin(fb, BinOp::ShiftLeftLogical, e, k10)?;
            let mm = c(fb, (1 << man) - 1)?;
            let m = bin(fb, BinOp::BitwiseAnd, v, mm)?;
            let up = c(fb, 10 - man)?;
            let m = bin(fb, BinOp::ShiftLeftLogical, m, up)?;
            let h = bin(fb, BinOp::BitwiseOr, e, m)?;
            lanes.push(narrow::from_bits(fb, half, h)?);
        }
    }
    let t = ty(fb, f)?;
    fb.composite_construct(t, &lanes)
}

/// `a * r` rounded to nearest even as an `f32`, for an integer `0 <= a < 2^10` (`u32` lanes of
/// form `u`) and a positive normal `f32` constant `r` per lane (its bits `r_bits[i]` for lane
/// `i`, at most `2^117`; a lane-wise sequence), with integer operations only, so it is exact on
/// every device whatever its rounding mode and under every fast-math mode (Metal's `snorm`
/// `rgb10a2` unpack multiplies by the reciprocal).
///
/// With `r = m * 2^er` (`m` its 24-bit significand, leading one included), the product `p = a *
/// m` is below `2^34`: `UMulExtended` gives it as two words. Its leading one is at bit `b = 23 +
/// s` with `s` in `[0, 10]` (`m >= 2^23`), so the significand is `p >> s`, rounded to nearest
/// even on the `s` dropped bits (`s <= 10` keeps every shift below 32), and the biased exponent
/// is `s + er + 150`. As in [`div_exact`], the bits are `((s + er + 149) << 23) + q` plus the
/// rounding, which carries into the exponent when `q` is all ones. `a = 0` is `+0`.
///
/// [`div_exact`]: crate::lower::arith::exact_division::div_exact
pub fn mul_exact(
    fb: &mut FunctionBuilder<'_>,
    u: Form,
    a: Value,
    r_bits: &[u32],
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, v: u64| splat(fb, u, v);
    let m: Vec<u64> = r_bits
        .iter()
        .map(|r| u64::from((r & 0x7F_FFFF) | 0x80_0000))
        .collect();
    // er + 149 = (biased exponent - 150) + 149.
    let bias: Vec<u64> = r_bits
        .iter()
        .map(|r| u64::from(((r >> 23) & 0xFF).saturating_sub(1)))
        .collect();
    let (z, one) = (c(fb, 0)?, c(fb, 1)?);
    let zero = bin(fb, BinOp::IEqual, a, z)?;
    let mk = per_lane(fb, u, &m)?;
    let p = bin(fb, BinOp::UMulExtended, a, mk)?;
    let lo = fb.composite_extract(p, &[0])?;
    let hi = fb.composite_extract(p, &[1])?;
    // The leading bit: 32 + msb(hi) when hi is nonzero, else msb(lo) (>= 23 unless a = 0).
    let hi_set = bin(fb, BinOp::INotEqual, hi, z)?;
    let hm = glsl(fb, GlslOp::FindUMsb, &[hi])?;
    let k32 = c(fb, 32)?;
    let hm = bin(fb, BinOp::IAdd, hm, k32)?;
    let lm = glsl(fb, GlslOp::FindUMsb, &[lo])?;
    let msb = fb.select(hi_set, hm, lm)?;
    // s = msb - 23, in [0, 10]; t = max(s, 1) keeps the shifts below 32 when s = 0 (then hi = 0
    // and nothing is dropped).
    let k23 = c(fb, 23)?;
    let s = bin(fb, BinOp::ISub, msb, k23)?;
    let k10 = c(fb, 10)?;
    let s = glsl(fb, GlslOp::UMin, &[s, k10])?;
    let t = glsl(fb, GlslOp::UMax, &[s, one])?;
    let q = bin(fb, BinOp::ShiftRightLogical, lo, s)?;
    let up_sh = bin(fb, BinOp::ISub, k32, t)?;
    let top = bin(fb, BinOp::ShiftLeftLogical, hi, up_sh)?;
    let q = bin(fb, BinOp::BitwiseOr, q, top)?;
    let unit = bin(fb, BinOp::ShiftLeftLogical, one, s)?;
    let below = bin(fb, BinOp::ISub, unit, one)?;
    let rem = bin(fb, BinOp::BitwiseAnd, lo, below)?;
    let t1 = bin(fb, BinOp::ISub, t, one)?;
    let half = bin(fb, BinOp::ShiftLeftLogical, one, t1)?;
    let some = bin(fb, BinOp::INotEqual, s, z)?;
    let gt = bin(fb, BinOp::UGreaterThan, rem, half)?;
    let eq = bin(fb, BinOp::IEqual, rem, half)?;
    let lsb = bin(fb, BinOp::BitwiseAnd, q, one)?;
    let odd = bin(fb, BinOp::INotEqual, lsb, z)?;
    let tie_up = bin(fb, BinOp::LogicalAnd, eq, odd)?;
    let up = bin(fb, BinOp::LogicalOr, gt, tie_up)?;
    let up = bin(fb, BinOp::LogicalAnd, some, up)?;
    let inc = fb.select(up, one, z)?;
    let kb = per_lane(fb, u, &bias)?;
    let be = bin(fb, BinOp::IAdd, s, kb)?;
    let be = bin(fb, BinOp::ShiftLeftLogical, be, k23)?;
    let bits = bin(fb, BinOp::IAdd, be, q)?;
    let bits = bin(fb, BinOp::IAdd, bits, inc)?;
    let bits = fb.select(zero, z, bits)?;
    let ft = ty(fb, u.with(Held::Float(FloatFmt::F32)))?;
    fb.convert(ConvOp::Bitcast, bits, ft)
}
