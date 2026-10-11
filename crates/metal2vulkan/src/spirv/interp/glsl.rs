//! The exact subset of `GLSL.std.450`: rounding, sign, min/max/clamp, `Fma`, `Sqrt`, `Length`,
//! `Ldexp`, `Modf`/`Frexp`, bit finding, `Cross`, and the half packs. Every float result is rounded once
//! per IEEE operation; the transcendental and approximate instructions are [`Unmodelled`].

use crate::spirv::ids::Value;
use crate::spirv::ir::GlslOp;

use super::float;
use super::sched::Prog;
use super::value::{mask, sext, Val};
use super::{malformed, IResult, TrapKind, Unmodelled};

/// Evaluate `op` on `vals` (the values of `args`) with result type `res`.
pub(super) fn eval(p: &Prog<'_>, op: GlslOp, args: &[Value], vals: &[Val]) -> IResult<Val> {
    let first = *args
        .first()
        .ok_or_else(|| malformed("GLSL instruction without operands"))?;
    let n = p.t.num_of(first)?;
    let w = n.width;
    let comps = |i: usize| -> IResult<Vec<u64>> {
        vals.get(i)
            .ok_or_else(|| malformed("GLSL operand count"))?
            .comps()
    };
    // Component-wise over `k` operands of the first operand's shape.
    let lanes = |k: usize, f: &dyn Fn(&[u64]) -> IResult<u64>| -> IResult<Val> {
        let cs: Vec<Vec<u64>> = (0..k).map(comps).collect::<IResult<_>>()?;
        let len = cs.first().map_or(0, Vec::len);
        let mut out = Vec::with_capacity(len);
        for j in 0..len {
            let row: Vec<u64> = cs
                .iter()
                .map(|c| c.get(j).copied().ok_or_else(|| malformed("GLSL shape")))
                .collect::<IResult<_>>()?;
            out.push(f(&row)?);
        }
        Ok(Val::from_comps(out))
    };
    let f = |x: u64| float::to_f64(x, w);
    let r = |x: f64| float::from_f64(x, w);
    let at = |row: &[u64], i: usize| -> IResult<u64> {
        row.get(i).copied().ok_or_else(|| malformed("GLSL arity"))
    };
    let fmin = |a: u64, b: u64| float::bin(w, a, b, f64::min);
    let fmax = |a: u64, b: u64| float::bin(w, a, b, f64::max);
    // `NMin`, `NMax` and `NClamp` pick an operand as Metal's `fmin`, `fmax` and `clamp` do, one of
    // the answers Vulkan allows: a NaN operand loses, and two NaNs and a tie answer the second
    // operand's bits. A denormal compares by its value, as every instruction here reads it (Metal's
    // compares a 32-bit one as a zero: the device's reading, which the driver facts measure).
    let pick = |p: u64, q: u64, less: bool| -> IResult<u64> {
        let (a, b) = (f(p)?, f(q)?);
        if a.is_nan() {
            return Ok(q);
        }
        if b.is_nan() {
            return Ok(p);
        }
        let before = if less { a < b } else { a > b };
        Ok(if before { p } else { q })
    };
    match op {
        GlslOp::Round => lanes(1, &|x| r(f(at(x, 0)?)?.round())),
        GlslOp::RoundEven => lanes(1, &|x| r(f(at(x, 0)?)?.round_ties_even())),
        GlslOp::Trunc => lanes(1, &|x| r(f(at(x, 0)?)?.trunc())),
        GlslOp::Floor => lanes(1, &|x| r(f(at(x, 0)?)?.floor())),
        GlslOp::Ceil => lanes(1, &|x| r(f(at(x, 0)?)?.ceil())),
        GlslOp::FAbs => lanes(1, &|x| Ok(at(x, 0)? & !(1u64 << (w - 1)))),
        GlslOp::Fract => lanes(1, &|x| {
            let a = at(x, 0)?;
            float::sub(w, a, r(f(a)?.floor())?)
        }),
        GlslOp::Sqrt => lanes(1, &|x| r(f(at(x, 0)?)?.sqrt())),
        // The root of the components' squares summed in order, each step rounded once.
        GlslOp::Length => {
            let mut sum = r(0.0)?;
            for x in comps(0)? {
                sum = float::add(w, sum, float::mul(w, x, x)?)?;
            }
            Ok(Val::S(r(f(sum)?.sqrt())?))
        }
        GlslOp::FSign => lanes(1, &|x| {
            let v = f(at(x, 0)?)?;
            if v.is_nan() {
                return Err(TrapKind::Undefined("FSign of NaN").err());
            }
            r(if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                v
            })
        }),
        GlslOp::SAbs => lanes(1, &|x| Ok(sext(at(x, 0)?, w).unsigned_abs() & mask(w))),
        GlslOp::SSign => lanes(1, &|x| Ok(sext(at(x, 0)?, w).signum() as u64 & mask(w))),
        GlslOp::FMin => lanes(2, &|x| fmin(at(x, 0)?, at(x, 1)?)),
        GlslOp::FMax => lanes(2, &|x| fmax(at(x, 0)?, at(x, 1)?)),
        GlslOp::NMin => lanes(2, &|x| pick(at(x, 0)?, at(x, 1)?, true)),
        GlslOp::NMax => lanes(2, &|x| pick(at(x, 0)?, at(x, 1)?, false)),
        GlslOp::UMin => lanes(2, &|x| Ok(at(x, 0)?.min(at(x, 1)?))),
        GlslOp::UMax => lanes(2, &|x| Ok(at(x, 0)?.max(at(x, 1)?))),
        GlslOp::SMin => lanes(2, &|x| {
            let (a, b) = (at(x, 0)?, at(x, 1)?);
            Ok(if sext(a, w) <= sext(b, w) { a } else { b })
        }),
        GlslOp::SMax => lanes(2, &|x| {
            let (a, b) = (at(x, 0)?, at(x, 1)?);
            Ok(if sext(a, w) >= sext(b, w) { a } else { b })
        }),
        GlslOp::FClamp => lanes(3, &|x| fmin(fmax(at(x, 0)?, at(x, 1)?)?, at(x, 2)?)),
        GlslOp::NClamp => lanes(3, &|x| {
            let low = pick(at(x, 0)?, at(x, 1)?, false)?;
            pick(low, at(x, 2)?, true)
        }),
        GlslOp::UClamp => lanes(3, &|x| Ok(at(x, 0)?.max(at(x, 1)?).min(at(x, 2)?))),
        GlslOp::SClamp => lanes(3, &|x| {
            let (v, lo, hi) = (sext(at(x, 0)?, w), sext(at(x, 1)?, w), sext(at(x, 2)?, w));
            Ok(v.max(lo).min(hi) as u64 & mask(w))
        }),
        GlslOp::Step => lanes(2, &|x| {
            r(if f(at(x, 1)?)? < f(at(x, 0)?)? {
                0.0
            } else {
                1.0
            })
        }),
        GlslOp::FMix => lanes(3, &|x| {
            let (a, b, t) = (at(x, 0)?, at(x, 1)?, at(x, 2)?);
            let one = r(1.0)?;
            let s = float::sub(w, one, t)?;
            float::add(w, float::mul(w, a, s)?, float::mul(w, b, t)?)
        }),
        GlslOp::Fma => lanes(3, &|x| float::fma(w, at(x, 0)?, at(x, 1)?, at(x, 2)?)),
        GlslOp::Ldexp => lanes(2, &|x| {
            let e = sext(at(x, 1)?, 32).clamp(-4000, 4000) as i32;
            let mut v = f(at(x, 0)?)?;
            let mut left = e;
            // Scale in steps that stay exact in f64 until the final rounding.
            while left != 0 {
                let step = left.clamp(-512, 512);
                v *= 2f64.powi(step);
                left -= step;
            }
            r(v)
        }),
        GlslOp::ModfStruct => {
            let (mut frac, mut whole) = (Vec::new(), Vec::new());
            for x in comps(0)? {
                let v = f(x)?;
                let i = v.trunc();
                whole.push(r(i)?);
                frac.push(r(if v.is_infinite() {
                    0.0f64.copysign(v)
                } else {
                    v - i
                })?);
            }
            Ok(Val::C(
                vec![Val::from_comps(frac), Val::from_comps(whole)].into(),
            ))
        }
        GlslOp::FrexpStruct => {
            let (mut mant, mut exps) = (Vec::new(), Vec::new());
            for x in comps(0)? {
                let v = f(x)?;
                if !v.is_finite() {
                    return Err(TrapKind::Undefined("Frexp of an infinity or NaN").err());
                }
                let (m, e) = frexp(v);
                mant.push(r(m)?);
                exps.push(e as u32 as u64);
            }
            Ok(Val::C(
                vec![Val::from_comps(mant), Val::from_comps(exps)].into(),
            ))
        }
        GlslOp::FindILsb => lanes(1, &|x| {
            let v = at(x, 0)? as u32;
            Ok(if v == 0 {
                u64::from(u32::MAX)
            } else {
                u64::from(v.trailing_zeros())
            })
        }),
        GlslOp::FindUMsb => lanes(1, &|x| {
            let v = at(x, 0)? as u32;
            Ok(if v == 0 {
                u64::from(u32::MAX)
            } else {
                u64::from(31 - v.leading_zeros())
            })
        }),
        GlslOp::FindSMsb => lanes(1, &|x| {
            let v = at(x, 0)? as u32;
            // For a negative value, the most significant 0 bit.
            let v = if v & 0x8000_0000 != 0 { !v } else { v };
            Ok(if v == 0 {
                u64::from(u32::MAX)
            } else {
                u64::from(31 - v.leading_zeros())
            })
        }),
        GlslOp::Cross => {
            let (a, b) = (comps(0)?, comps(1)?);
            let g = |v: &[u64], i: usize| -> IResult<u64> {
                v.get(i).copied().ok_or_else(|| malformed("cross operand"))
            };
            let term = |i: usize, j: usize| -> IResult<u64> {
                float::sub(
                    w,
                    float::mul(w, g(&a, i)?, g(&b, j)?)?,
                    float::mul(w, g(&a, j)?, g(&b, i)?)?,
                )
            };
            Ok(Val::from_comps(vec![term(1, 2)?, term(2, 0)?, term(0, 1)?]))
        }
        GlslOp::PackHalf2x16 => {
            let c = comps(0)?;
            let mut out = 0u64;
            for (i, x) in c.iter().enumerate() {
                let h = float::f64_to_f16(float::to_f64(*x, 32)?);
                out |= u64::from(h) << (16 * i);
            }
            Ok(Val::S(out & mask(32)))
        }
        GlslOp::UnpackHalf2x16 => {
            let x = comps(0)?
                .first()
                .copied()
                .ok_or_else(|| malformed("unpack operand"))?;
            let lo = float::from_f64(float::f16_to_f64(x as u16), 32)?;
            let hi = float::from_f64(float::f16_to_f64((x >> 16) as u16), 32)?;
            Ok(Val::from_comps(vec![lo, hi]))
        }
        other => Err(Unmodelled::Glsl(other).err()),
    }
}

/// `v = m * 2^e` with `0.5 <= |m| < 1` (and `(0, 0)` for zero), exactly.
fn frexp(v: f64) -> (f64, i32) {
    if v == 0.0 {
        return (v, 0);
    }
    let mut e = 0i32;
    let mut m = v;
    while m.abs() >= 1.0 {
        m /= 2.0;
        e += 1;
    }
    while m.abs() < 0.5 {
        m *= 2.0;
        e -= 1;
    }
    (m, e)
}
