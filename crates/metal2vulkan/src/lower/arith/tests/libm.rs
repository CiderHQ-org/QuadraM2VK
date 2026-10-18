//! The math-library routines on the interpreter: each within its bound of the exact value over
//! operands across the whole range, and Metal's answers at the special operands, bit for bit. A
//! routine divides and takes square roots only through the module's correctly rounded routines, so
//! its answers are every device's; those of `sqrt` and `rsqrt` correct the device's own, which
//! their tests also move to the edge of what they correct.
//!
//! The reference is the host's `f64` function of the exact operand, rounded nowhere: the error is
//! measured in ulps of the `float` grid at the exact value (the result's exponent, `2^-149` below
//! the normal range).

use super::*;
use crate::lower::arith::libm::{self, Fun, Mode};
use crate::spirv::serialize::serialize;
use std::cell::Cell;

/// `float` scalars.
pub(super) fn f32s() -> Form {
    float(32)
}

/// The ulp of the `float` grid at `v`.
fn ulp32(v: f64) -> f64 {
    let a = v.abs();
    if a < 2f64.powi(-126) {
        return 2f64.powi(-149);
    }
    2f64.powi(a.log2().floor() as i32 - 23)
}

/// Operands: every 2^k from `2^-126` to `2^127` and their neighbours, both signs, pseudo-random
/// bit patterns of every exponent, and the interval edges each routine reduces by.
pub(super) fn operands(seed: u64, random: usize) -> Vec<u64> {
    let mut v = Vec::new();
    for e in 1u64..255 {
        for m in [0u64, 1, 0x7F_FFFF, 0x40_0000] {
            for s in [0u64, 1] {
                v.push((s << 31) | (e << 23) | m);
            }
        }
    }
    for x in [
        0.4375f32,
        0.6875,
        1.1875,
        2.4375,
        0.5,
        1.0,
        0.25,
        0.75,
        0.975,
        0.999_999_9,
        1.5,
        3.0,
        67_108_864.0,
        67_108_860.0,
    ] {
        for d in [-2i64, -1, 0, 1, 2] {
            let b = (i64::from(x.to_bits()) + d) as u64;
            v.push(b);
            v.push(b | 0x8000_0000);
        }
    }
    let mut rng = Rng(seed);
    for _ in 0..random {
        let r = rng.next() & 0xFFFF_FFFF;
        if r & 0x7F80_0000 != 0x7F80_0000 {
            v.push(r);
        }
        // Concentrated where the functions have most of their range.
        let near = (rng.next() & 0x00FF_FFFF) | 0x3E00_0000;
        v.push(near | (rng.next() & 0x8000_0000));
    }
    v
}

/// The routine of `fun` built rough (its division and square root moved) or exact, on a profile
/// whose `float` fused multiply-add is `OpFmaKHR` where `fma`.
pub(super) fn routine_module(fun: Fun, rough: bool, fma: bool) -> TypedModule {
    let f = f32s();
    let ins = vec![f; fun.arity()];
    kernel_defs(
        &ins,
        f,
        &|b| {
            Ok(vec![if rough {
                libm::define_rough(b, fun, fma)?
            } else {
                libm::define_alone(b, fun, fma)?
            }])
        },
        &|fb, x, funcs| {
            fb.call(funcs[0], x)?
                .ok_or(crate::spirv::error::BuildError::Type(
                    crate::spirv::error::TypeRule::FunctionSignature,
                ))
        },
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("{fun:?}: build: {e:?}"))
}

/// A module of the helpers `fun`'s routine calls ([`Fun::needs`]) alone, each called from the
/// entry so that none leaves as dead: taken from [`routine_module`]'s words, it leaves the
/// routine's own (each helper is charged to its own census name, [`Fun::Recip`]'s routine with its
/// table; Metal's `exp2`, `log2`, and reciprocal tables are charged to the routines that read
/// them, [`Fun::words`]).
pub(super) fn helpers_module(fun: Fun, fma: bool) -> TypedModule {
    let f = f32s();
    let ins = vec![f; fun.arity()];
    let needs = libm::Needs {
        exp2: false,
        log2: false,
        rcp: false,
        rsqrt: false,
        quarter: false,
        sinc: false,
        ..fun.needs()
    };
    // The helpers `define_helpers` defines (the ones `fun` calls and the ones they call), for the
    // entry to call each by its own signature.
    let defined = Cell::new(libm::Helpers::default());
    kernel_defs(
        &ins,
        f,
        &|b| {
            let h = libm::define_helpers(b, needs, false, fma)?;
            defined.set(h);
            Ok(h.fused
                .into_iter()
                .chain(h.division)
                .chain(h.recip)
                .chain(h.inverse_root)
                .chain(h.fast_inverse_root)
                .collect())
        },
        &|fb, x, _| {
            let call = |fb: &mut FunctionBuilder<'_>, id: FuncId, args: &[Value]| {
                fb.call(id, args)?
                    .ok_or(crate::spirv::error::BuildError::Type(
                        crate::spirv::error::TypeRule::FunctionSignature,
                    ))
            };
            let h = defined.get();
            let mut v = x[0];
            if let Some(id) = h.fused {
                v = call(fb, id, &[v, x[0], v])?;
            }
            if let Some(id) = h.division {
                v = call(fb, id, &[v, x[0]])?;
            }
            if let Some(id) = h.recip {
                v = call(fb, id, &[v])?;
            }
            if let Some(id) = h.inverse_root {
                v = call(fb, id, &[v])?;
            }
            if let Some(id) = h.fast_inverse_root {
                v = call(fb, id, &[v])?;
            }
            Ok(v)
        },
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("{fun:?}: helpers: {e:?}"))
}

/// Run the routine of `fun` over `cases` (one operand list each), against `exact(operands)` (an
/// `f64`, or `None` where the lane is not measured), accepting `limit` ulps; returns the worst.
pub(super) fn worst_error(
    fun: Fun,
    rough: bool,
    cases: &[Vec<u64>],
    exact: &dyn Fn(&[f32]) -> Option<f64>,
    limit: f64,
) -> f64 {
    let worst = Cell::new(0f64);
    let f = f32s();
    let ins = vec![f; fun.arity()];
    let cases: Vec<Case> = cases
        .iter()
        .map(|c| c.iter().map(|b| vec![*b]).collect())
        .collect();
    check_module(
        &format!("{fun:?} rough={rough}"),
        routine_module(fun, rough, false),
        &ins,
        f,
        &cases,
        &|c| {
            let xs: Vec<f32> = c.iter().map(|l| f32v(l[0])).collect();
            exact(&xs).map(|r| vec![r.to_bits()])
        },
        &|got, want| {
            let (g, w) = (f64::from(f32v(got)), f64v(want));
            if w.is_nan() {
                return g.is_nan();
            }
            let e = (g - w).abs() / ulp32(w);
            if e > worst.get() {
                worst.set(e);
            }
            e <= limit
        },
    );
    worst.get()
}

/// The one-operand cases of `xs`.
pub(super) fn unary(xs: &[u64]) -> Vec<Vec<u64>> {
    xs.iter().map(|x| vec![*x]).collect()
}

/// Metal's precise `asin` or `acos` of a normal or zero `x` as the model computes it
/// (`libm::atan`'s list): Metal's polynomials by fused Horner steps and its root `z rsqrt(z)` of
/// Metal's inverse root's model ([`rsqrt_model`], 0 at `z = 0`); Metal's NaN past 1.
pub(super) fn precise_arc_model(asin: bool, x: f32) -> f32 {
    use crate::lower::arith::libm::atan::{
        ACOS_FAR, ACOS_MID, ACOS_NEAR_END, ASIN_FAR, ASIN_FAR_START, ASIN_HIGH, ASIN_LOW, ASIN_MID,
        ASIN_NEAR, ASIN_TINY,
    };
    let poly = |c: &[u32], v: f32| {
        c.iter()
            .rev()
            .skip(1)
            .fold(f32::from_bits(c[c.len() - 1]), |acc, k| {
                acc.mul_add(v, f32::from_bits(*k))
            })
    };
    let root = |z: f32| if z == 0.0 { 0.0 } else { z * rsqrt_model(z) };
    let (half_pi, pi) = (f32::from_bits(0x3FC9_0FDB), f32::from_bits(0x4049_0FDB));
    let a = x.abs();
    if x.is_nan() {
        return x;
    }
    if asin && a.to_bits() < ASIN_TINY {
        return x;
    }
    if a > 1.0 {
        return f32::from_bits(0x7FC0_0000);
    }
    if asin {
        let r = if a < 0.1 {
            poly(&ASIN_NEAR, a)
        } else if a < 0.36 {
            poly(&ASIN_LOW, a)
        } else if a < 0.62 {
            poly(&ASIN_MID, a * a) * a
        } else if a.to_bits() < ASIN_FAR_START {
            poly(&ASIN_HIGH, a * a) * a
        } else {
            let z = if a < 0.99 {
                (-a).mul_add(a, 1.0)
            } else {
                (1.0 - a) * (a + 1.0)
            };
            root(z).mul_add(poly(&ASIN_FAR, a), half_pi)
        };
        r.copysign(x)
    } else {
        let r = if a.to_bits() <= ACOS_NEAR_END {
            (a * (a * a)).mul_add(f32::from_bits(0xBE2A_AAAB), half_pi - a)
        } else if a <= 0.5 {
            poly(&ACOS_MID, a)
        } else {
            let u = 1.0 - a;
            poly(&ACOS_FAR, u) * root(u)
        };
        if x < 0.0 {
            pi - r
        } else {
            r
        }
    }
}

/// The precise `asin` and `acos` routines answer [`precise_arc_model`] bit for bit, with and
/// without `OpFmaKHR`, over the normal and zero operands in `[-1, 1]` of [`operands`], every
/// region's edges, and every 65,537th `float` of `[0, 1]` of both signs, and lie within Apple's
/// bound (4 ulp, MSL Table 8.1) of the exact value there. Part `part` of `of` checks every
/// `of`-th operand.
fn precise_asin_and_acos_are_metals(part: usize, of: usize) {
    use crate::lower::arith::libm::atan::{ACOS_NEAR_END, ASIN_FAR_START, ASIN_TINY};
    let mut xs: Vec<u64> = operands(0x07A4_E297, 4000)
        .into_iter()
        .filter(|b| {
            let v = f32v(*b);
            v.abs() <= 1.0 && (v == 0.0 || v.is_normal())
        })
        .collect();
    for edge in [
        ASIN_TINY,
        0.1f32.to_bits(),
        0.36f32.to_bits(),
        0.62f32.to_bits(),
        ASIN_FAR_START,
        0.99f32.to_bits(),
        ACOS_NEAR_END,
        0.5f32.to_bits(),
        1.0f32.to_bits(),
    ] {
        for d in 0..=64u32 {
            let b = (edge + d).wrapping_sub(32);
            if f32::from_bits(b) <= 1.0 {
                xs.push(u64::from(b));
            }
        }
    }
    xs.extend((0..=0x3F80_0000u32).step_by(65_537).map(u64::from));
    let signed: Vec<u64> = xs
        .iter()
        .flat_map(|b| [*b & 0x7FFF_FFFF, *b | 0x8000_0000])
        .collect();
    let signed = part_of(signed, part, of);
    let f = f32s();
    for asin in [true, false] {
        let fun = if asin {
            Fun::Asin(Mode::Precise)
        } else {
            Fun::Acos(Mode::Precise)
        };
        for fma in [false, true] {
            check_module(
                &format!("{fun:?} fma={fma}"),
                routine_module(fun, false, fma),
                &[f],
                f,
                &signed.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
                &|c| {
                    Some(vec![u64::from(
                        precise_arc_model(asin, f32v(c[0][0])).to_bits(),
                    )])
                },
                &|g, w| g == w,
            );
        }
        let worst = signed
            .iter()
            .map(|b| {
                let x = f32v(*b);
                let e = if asin {
                    f64::from(x).asin()
                } else {
                    f64::from(x).acos()
                };
                let w = f64::from(precise_arc_model(asin, x));
                let r = e.abs() as f32;
                let ulp = f64::from(f32::from_bits(r.to_bits() + 1)) - f64::from(r);
                if e == 0.0 {
                    0.0
                } else {
                    (w - e).abs() / ulp
                }
            })
            .fold(0f64, f64::max);
        assert!(worst <= 4.0, "{fun:?}: {worst:.3} ulp");
    }
}

mod precise_asin_and_acos_routines_are_metals {
    parts!(super::precise_asin_and_acos_are_metals, 4; part_1_of_4 = 0, part_2_of_4 = 1,
        part_3_of_4 = 2, part_4_of_4 = 3);
}

/// Metal's reduction of a finite `x` to quarter turns as the model computes it
/// (`libm::quarter::turns`): `(r, k)`, `k` modulo `2^32`, with the rounding error of the large
/// operands' difference where `compensated` (`tan`).
pub(super) fn metal_quarter_turns(x: f32, compensated: bool) -> (f32, u32) {
    use crate::lower::arith::libm::quarter::{QUARTER, QUARTER_TABLE};
    let a = x.to_bits() & 0x7FFF_FFFF;
    let rounded = |c: f32, v: f32, like: f32| {
        let b = 8_388_608f32.copysign(like);
        c.mul_add(v, b) - b
    };
    // An integer-valued `float` below `2^32` in magnitude, modulo `2^32`.
    let int = |k: f32| (k as i64) as u32;
    if a < 0x4A00_0000 {
        let [c0, c1, c2] = QUARTER.map(f32::from_bits);
        let k = rounded(c0, x, x);
        let r = c2.mul_add(x, c1.mul_add(x, c0.mul_add(x, -k)));
        return (r, int(k));
    }
    let [h, m, l] = QUARTER_TABLE[((a - 0x4A00_0000) >> 24) as usize].map(f32::from_bits);
    let xs = x * f32::from_bits(0x0080_0000);
    let p = xs * h;
    let k0 = if p.abs() >= 8_388_608.0 {
        p
    } else {
        rounded(h, xs, p)
    };
    let r0 = h.mul_add(xs, -k0);
    let r1 = m.mul_add(xs, r0);
    let k1 = (r1 + 0.5f32.copysign(r1)).trunc();
    let d = r0 - k1;
    let r = l.mul_add(xs, m.mul_add(xs, d));
    let r = if compensated {
        let e1 = d - r0;
        r + ((-k1 - e1) + (r0 - (d - e1)))
    } else {
        r
    };
    (r, int(k0).wrapping_add(int(k1)))
}

/// Metal's precise `tan(x)` as the model computes it: the routine's form (`libm::tan`) in `f32`
/// arithmetic, each of Metal's fused steps one `mul_add`, and the quotient correctly rounded.
pub(super) fn metal_tan_model(x: f32) -> f32 {
    use crate::lower::arith::libm::tan::{TAN_P, TAN_Q};
    let a = x.to_bits() & 0x7FFF_FFFF;
    if a >= 0x7F80_0000 {
        return f32::from_bits(0x7FC0_0000);
    }
    if a < 0x3A00_0000 {
        return x;
    }
    let poly = |c: &[u32], v: f32| {
        let (last, rest) = c.split_last().expect("coefficients");
        rest.iter().rev().fold(f32::from_bits(*last), |acc, k| {
            acc.mul_add(v, f32::from_bits(*k))
        })
    };
    let (r, k) = metal_quarter_turns(x, true);
    let s = r * r;
    let n = r * poly(&TAN_P, s);
    let d = poly(&TAN_Q, s);
    if k & 1 == 0 {
        n / d
    } else {
        -d / n
    }
}

/// The model of Metal's hardware `sinc` of `|t| <= 1` (`libm::sincos::SINC`), in 64-bit integers.
pub(super) fn metal_sinc_model(at: f32) -> f32 {
    use crate::lower::arith::libm::sincos::SINC;
    let m = (at * 8_388_608.0) as u32;
    if m == 0 {
        return std::f32::consts::FRAC_PI_2;
    }
    if m >= 1 << 23 {
        return 1.0;
    }
    let row = SINC[(m >> 18) as usize];
    let d = i64::from(m & 0x3_FFFF);
    let c = |i: usize| i64::from(row[i] as i32);
    let t2 = c(3) + ((c(4) * d) >> 18);
    let t1 = c(2) + ((t2 * d) >> 18);
    let w = c(1) + ((t1 * d) >> 18);
    f32::from_bits(row[0].wrapping_add(((w + 2048) >> 12) as u32))
}

/// Metal's precise `sin(x)` (or `cos(x)` where `cos`) as the model computes it: the routine's form
/// (`libm::sincos`) in `f32` arithmetic, the reduction's fused steps each one `mul_add`, and the
/// `sinc` [`metal_sinc_model`].
pub(super) fn metal_sin_cos_model(x: f32, cos: bool) -> f32 {
    let a = x.to_bits() & 0x7FFF_FFFF;
    if a >= 0x7F80_0000 {
        return f32::from_bits(0x7FC0_0000);
    }
    if a <= 0x3980_0000 {
        return if cos { 1.0 } else { x };
    }
    let (r, k) = metal_quarter_turns(x, false);
    let odd = k & 1 != 0;
    let t = match (cos, odd) {
        (false, false) => r,
        (true, true) => -r,
        _ => 1.0 - r.abs(),
    };
    let t = if k & 2 != 0 { -t } else { t };
    t * metal_sinc_model(t.abs())
}

/// The precise `tan` routine answers [`metal_tan_model`] bit for bit, with and without `OpFmaKHR`,
/// over [`trig_operands`] (every binade, and the floats nearest multiples of `π/2` to `2^40 π/2`),
/// every reduction's and region's edges, and every 65,537th finite `float`, both signs; and the
/// model lies within 5 ulp of the exact value there (Apple's bound: 6; Metal's worst over every
/// 251st `float` from `2^-11` and the floats nearest the first `2^22` multiples of `π/2`, by the
/// model, is 4.77). Part `part` of `of` checks every `of`-th operand.
fn precise_tan_is_metals(part: usize, of: usize) {
    let mut xs = trig_operands();
    for edge in [0x3A00_0000u32, 0x4A00_0000, 0x4B00_0000, 0x7F00_0000] {
        for d in 0..=64u32 {
            xs.push(u64::from((edge + d).wrapping_sub(32)));
        }
    }
    xs.extend((0..0x7F80_0000u32).step_by(65_537).map(u64::from));
    let signed: Vec<u64> = xs
        .iter()
        .flat_map(|b| [*b & 0x7FFF_FFFF, *b | 0x8000_0000])
        .filter(|b| f32v(*b).is_finite())
        .collect();
    let signed = part_of(signed, part, of);
    let f = f32s();
    for fma in [false, true] {
        check_module(
            &format!("Tan fma={fma}"),
            routine_module(Fun::Tan, false, fma),
            &[f],
            f,
            &signed.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
            &|c| Some(vec![u64::from(metal_tan_model(f32v(c[0][0])).to_bits())]),
            &|g, w| g == w,
        );
    }
    let worst = signed
        .iter()
        .map(|b| {
            let x = f32v(*b);
            let e = f64::from(x).tan();
            let r = e.abs() as f32;
            let ulp = f64::from(f32::from_bits(r.to_bits() + 1)) - f64::from(r);
            (f64::from(metal_tan_model(x)) - e).abs() / ulp
        })
        .fold(0f64, f64::max);
    assert!(worst <= 5.0, "Tan: {worst:.3} ulp");
}

mod precise_tan_routine_is_metals {
    parts!(super::precise_tan_is_metals, 4; part_1_of_4 = 0, part_2_of_4 = 1,
        part_3_of_4 = 2, part_4_of_4 = 3);
}

/// [`metal_tan_model`] is Metal's precise `tan` at words measured on the M3 Max
/// in each of its ranges (below 1/4, `[1/4, 1)`, `[1, 4)`, `[4, 2^21)` and from `2^21`), where
/// Metal's word is not the correctly rounded value, and the routine answers them bit for bit, with
/// and without `OpFmaKHR`.
#[test]
fn precise_tan_answers_metals_measured_words() {
    let words: [(u32, u32); 40] = [
        (0x3A00_1A10, 0x3A00_1A10),
        (0x3A00_3BC0, 0x3A00_3BC0),
        (0x3A00_4240, 0x3A00_4240),
        (0x3A00_52C0, 0x3A00_52C0),
        (0x3A00_7C70, 0x3A00_7C70),
        (0x3A00_9700, 0x3A00_9700),
        (0x3A00_B390, 0x3A00_B390),
        (0x3A00_CEC0, 0x3A00_CEC0),
        (0x3E80_0420, 0x3E82_C093),
        (0x3E80_08EF, 0x3E82_C5B2),
        (0x3E80_09E4, 0x3E82_C6B7),
        (0x3E80_0C6D, 0x3E82_C969),
        (0x3E80_139A, 0x3E82_D10E),
        (0x3E80_17FA, 0x3E82_D5B7),
        (0x3E80_1C05, 0x3E82_DA07),
        (0x3E80_200E, 0x3E82_DE52),
        (0x3F80_02AB, 0x3FC7_6246),
        (0x3F80_05D6, 0x3FC7_6D21),
        (0x3F80_0687, 0x3FC7_6F80),
        (0x3F80_0812, 0x3FC7_74CC),
        (0x3F80_0C21, 0x3FC7_82B6),
        (0x3F80_0EB8, 0x3FC7_8B98),
        (0x3F80_1172, 0x3FC7_94F3),
        (0x3F80_140A, 0x3FC7_9DD7),
        (0x4081_FA00, 0x3FA8_28A7),
        (0x4085_2000, 0x3FCF_BB23),
        (0x4085_D400, 0x3FDA_54AF),
        (0x4087_6100, 0x3FF4_CA15),
        (0x408B_FA00, 0x4036_0324),
        (0x408E_9700, 0x4074_0F56),
        (0x4092_5E00, 0x40E5_B628),
        (0x4095_4A00, 0x41A9_B476),
        (0x4A29_E000, 0x3F89_4A74),
        (0x4A5F_E000, 0x4159_4F3A),
        (0x4A6B_5000, 0x3F3D_7462),
        (0x4A85_A000, 0x3FB8_AE5D),
        (0x4AC6_C000, 0xBFB3_8744),
        (0x4AF1_C000, 0xBDDA_1E74),
        (0x4B1E_9000, 0xBF8D_35D3),
        (0x4B4B_4000, 0xBF3A_A6CC),
    ];
    let f = f32s();
    for (x, want) in words {
        assert_eq!(
            metal_tan_model(f32::from_bits(x)).to_bits(),
            want,
            "model at {x:#010x}"
        );
    }
    for fma in [false, true] {
        check_module(
            &format!("Tan fma={fma}"),
            routine_module(Fun::Tan, false, fma),
            &[f],
            f,
            &words
                .iter()
                .map(|(x, _)| vec![vec![u64::from(*x)]])
                .collect::<Vec<Case>>(),
            &|c| {
                let x = c[0][0] as u32;
                words
                    .iter()
                    .find(|(b, _)| *b == x)
                    .map(|(_, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
    }
}

/// [`precise_arc_model`] is Metal's precise `asin` and `acos` at words measured on the M4 in every
/// region, where Metal's word is not the correctly rounded value, and the
/// routines answer them bit for bit.
#[test]
fn precise_asin_and_acos_answer_metals_measured_words() {
    let asin: [(u32, u32); 24] = [
        (0x39E8_9770, 0x39E8_9770),
        (0x3A9E_4650, 0x3A9E_4652),
        (0x3C3A_A7B0, 0x3C3A_A8B8),
        (0x3DCC_CCC0, 0x3DCD_2486),
        (0x3DCC_CD50, 0x3DCD_2517),
        (0x3E7F_FFC0, 0x3E81_5F2E),
        (0x3E80_0002, 0x3E81_5F50),
        (0x3EB8_51E0, 0x3EBC_8D8F),
        (0x3EB8_51FA, 0x3EBC_8DAB),
        (0x3ED7_DE29, 0x3EDE_D627),
        (0x3EF3_2927, 0x3EFD_6083),
        (0x3F1E_B851, 0x3F2B_32B7),
        (0x3F1E_B852, 0x3F2B_32BA),
        (0x3F31_6BFF, 0x3F44_0609),
        (0x3F41_7C9F, 0x3F5B_5CF2),
        (0x3F50_A3D5, 0x3F73_E60B),
        (0x3F50_A3D7, 0x3F73_E610),
        (0x3F5C_1ABE, 0x3F84_75CB),
        (0x3F6C_7255, 0x3F96_B5A7),
        (0x3F7D_70A3, 0x3FB6_F1E1),
        (0x3F7D_70A8, 0x3FB6_F1F3),
        (0x3F7E_4A48, 0x3FBA_4282),
        (0x3F7F_2591, 0x3FBE_9BBD),
        (0x3F7F_FFFF, 0x3FC9_048B),
    ];
    let acos: [(u32, u32); 20] = [
        (0x3980_0230, 0x3FC9_07DB),
        (0x3B6A_12A0, 0x3FC8_9AD2),
        (0x3D12_1140, 0x3FC4_7F12),
        (0x3E7F_FF80, 0x3FA8_B817),
        (0x3E80_0000, 0x3FA8_B806),
        (0x3EAB_11F4, 0x3F9D_74AF),
        (0x3ED5_92EB, 0x3F91_FC08),
        (0x3EFF_FFFD, 0x3F86_0A92),
        (0x3F00_0001, 0x3F86_0A92),
        (0x3F28_BA28, 0x3F59_E76C),
        (0x3F53_1DA8, 0x3F19_E6A3),
        (0x3F7F_FFFE, 0x39FF_FFFF),
        (0xBE80_0000, 0x3FE9_67B0),
        (0xBEA4_A4AE, 0x3FF2_F7A4),
        (0xBEC9_50F0, 0x3FFC_C98F),
        (0xBEFF_FFFE, 0x4006_0A92),
        (0xBF00_0004, 0x4006_0A94),
        (0xBF28_E452, 0x4012_A406),
        (0xBF53_6D36, 0x4022_B96A),
        (0xBF7F_FFFF, 0x4049_0A33),
    ];
    let f = f32s();
    for (is_asin, words) in [(true, &asin[..]), (false, &acos[..])] {
        for (x, w) in words {
            assert_eq!(
                precise_arc_model(is_asin, f32::from_bits(*x)).to_bits(),
                *w,
                "{x:#010x}: the model against Metal's word"
            );
        }
        let fun = if is_asin {
            Fun::Asin(Mode::Precise)
        } else {
            Fun::Acos(Mode::Precise)
        };
        for fma in [false, true] {
            check_module(
                &format!("{fun:?} fma={fma} at Metal's measured words"),
                routine_module(fun, false, fma),
                &[f],
                f,
                &words
                    .iter()
                    .map(|(x, _)| vec![vec![u64::from(*x)]])
                    .collect::<Vec<Case>>(),
                &|c| {
                    words
                        .iter()
                        .find(|(x, _)| u64::from(*x) == c[0][0])
                        .map(|(_, w)| vec![u64::from(*w)])
                },
                &|g, w| g == w,
            );
        }
    }
}

/// Metal's fast `atan` (the M3; the M4's words are the same) at lanes where its
/// word is not the correctly rounded one: below `2^-12` (where it is `x` less an ulp), on `[2^-12,
/// 1]`, on `(1, 4)` through Metal's reciprocal, and up to `2^126`; each word bit for bit, both
/// signs.
#[test]
fn fast_atan_answers_metals_words() {
    let words: [(u64, u64); 44] = [
        (0x07EE2100, 0x07EE20FF),
        (0x0B769C00, 0x0B769BFF),
        (0x12360D00, 0x12360CFF),
        (0x19984C00, 0x19984BFF),
        (0x2A2D3200, 0x2A2D31FF),
        (0x30B09300, 0x30B092FF),
        (0x39F45200, 0x39F451FE),
        (0x3A1BB100, 0x3A1BB0FE),
        (0x3A7D9600, 0x3A7D95FA),
        (0x3B694700, 0x3B6946BE),
        (0x3C4EDA00, 0x3C4ED72F),
        (0x3D627900, 0x3D623E06),
        (0x3E8B0DCA, 0x3E87C75B),
        (0x3EADBE05, 0x3EA77FD2),
        (0x3EAF7020, 0x3EA904B5),
        (0x3EC46393, 0x3EBB870D),
        (0x3ECE5D03, 0x3EC42A60),
        (0x3ED0FC2C, 0x3EC66AB7),
        (0x3EEBD321, 0x3EDCFE97),
        (0x3EED7E44, 0x3EDE5E89),
        (0x3EEF6840, 0x3EDFF12F),
        (0x3F2509B8, 0x3F12979C),
        (0x3F499ABD, 0x3F2AC5F6),
        (0x3F766465, 0x3F442AB1),
        (0x3F8C35F1, 0x3F54B56F),
        (0x3F8D30B0, 0x3F559887),
        (0x3F995055, 0x3F600937),
        (0x3FB092A0, 0x3F718C27),
        (0x3FC0F063, 0x3F7C2BCE),
        (0x3FCDE59B, 0x3F81DE40),
        (0x4019B6D5, 0x3F968FFC),
        (0x4023C994, 0x3F99618A),
        (0x4033533E, 0x3F9D2EE5),
        (0x403A05CE, 0x3F9EA5E2),
        (0x403EDAE4, 0x3F9FA5CC),
        (0x40531E61, 0x3FA362E0),
        (0x4186A600, 0x3FC17744),
        (0x4311E800, 0x3FC82F47),
        (0x44C38B00, 0x3FC8FAE9),
        (0x45B78D00, 0x3FC90A47),
        (0x45C5CD00, 0x3FC90AAE),
        (0x49783F00, 0x3FC90FD3),
        (0x498D9600, 0x3FC90FD4),
        (0x4A15D200, 0x3FC90FD8),
    ];
    let cases: Vec<Case> = words
        .iter()
        .flat_map(|(x, _)| [vec![vec![*x]], vec![vec![*x | 0x8000_0000]]])
        .collect();
    for fma in [false, true] {
        let f = f32s();
        check_module(
            &format!("fast atan fma={fma}"),
            routine_module(Fun::Atan(Mode::Fast), false, fma),
            &[f],
            f,
            &cases,
            &|c| {
                let x = c[0][0];
                words
                    .iter()
                    .find(|(w, _)| *w == x & 0x7FFF_FFFF)
                    .map(|(_, r)| vec![*r | (x & 0x8000_0000)])
            },
            &|g, w| g == w,
        );
    }
}

/// Metal's precise `atan` (the M3) at lanes where its word is not the correctly
/// rounded one: below `RN(tan(π/8))` (from `2^-12`), at it (where the next reduction starts), up to
/// `RN(tan(3π/8))` and above it through Metal's reciprocal, and up to `2^126`; each word bit for
/// bit, both signs. These lanes are among those where the reciprocal model is Metal's.
#[test]
fn precise_atan_answers_metals_words() {
    let words: [(u64, u64); 48] = [
        (0x39916500, 0x39916500),
        (0x3B832900, 0x3B8328D2),
        (0x3BEA3D00, 0x3BEA3BFB),
        (0x3D125400, 0x3D124414),
        (0x3D87D700, 0x3D87A423),
        (0x3DC2AD00, 0x3DC217B4),
        (0x3E87E074, 0x3E84D0B4),
        (0x3E8BAC29, 0x3E885ACC),
        (0x3E90A7ED, 0x3E8CFB3D),
        (0x3EA771A5, 0x3EA1D493),
        (0x3EB040CD, 0x3EA9BF5F),
        (0x3EB5733E, 0x3EAE611E),
        (0x3EB7AB9A, 0x3EB0595E),
        (0x3EC2EF62, 0x3EBA424A),
        (0x3EC65C06, 0x3EBD3E3C),
        (0x3ECBB20A, 0x3EC1DDCA),
        (0x3ECC3332, 0x3EC24D42),
        (0x3ECD713C, 0x3EC35F6C),
        (0x3ED413CD, 0x3EC90FDA),
        (0x3EDD796F, 0x3ED107CB),
        (0x3EDE548B, 0x3ED1C041),
        (0x3EE479D4, 0x3ED6E61D),
        (0x3EEA4427, 0x3EDBB510),
        (0x3EF344C7, 0x3EE31A02),
        (0x3F3B7B7B, 0x3F21D207),
        (0x3F61E992, 0x3F391980),
        (0x3F75FB18, 0x3F43F3FB),
        (0x3F791339, 0x3F458D5F),
        (0x3F87D7E8, 0x3F50AB80),
        (0x3F889C03, 0x3F516368),
        (0x3FB84D51, 0x3F76B92C),
        (0x4023481D, 0x3F993F26),
        (0x40236D99, 0x3F994920),
        (0x4045AC53, 0x3FA0FBAB),
        (0x40507C74, 0x3FA2F020),
        (0x40538662, 0x3FA3745A),
        (0x4053C455, 0x3FA37EBB),
        (0x405F2247, 0x3FA54EF2),
        (0x4063ECE0, 0x3FA605CB),
        (0x406B61E9, 0x3FA7149C),
        (0x4071498B, 0x3FA7DFFA),
        (0x4076441B, 0x3FA8845C),
        (0x407F35F1, 0x3FA99C8E),
        (0x43005500, 0x3FC81086),
        (0x43F34700, 0x3FC8CC82),
        (0x45A40200, 0x3FC9099D),
        (0x498DC600, 0x3FC90FD4),
        (0x4C158B00, 0x3FC90FDB),
    ];
    let cases: Vec<Case> = words
        .iter()
        .flat_map(|(x, _)| [vec![vec![*x]], vec![vec![*x | 0x8000_0000]]])
        .collect();
    for fma in [false, true] {
        let f = f32s();
        check_module(
            &format!("precise atan fma={fma}"),
            routine_module(Fun::Atan(Mode::Precise), false, fma),
            &[f],
            f,
            &cases,
            &|c| {
                let x = c[0][0];
                words
                    .iter()
                    .find(|(w, _)| *w == x & 0x7FFF_FFFF)
                    .map(|(_, r)| vec![*r | (x & 0x8000_0000)])
            },
            &|g, w| g == w,
        );
    }
}

/// Metal's fast `atan`, `asin`, and `acos` and its precise `atan` are within Apple's bound for them
/// (5 ulp, MSL Tables 8.1 and 8.2) over their ranges; Metal's own words are within 3 ulp (both
/// `atan`s, on every `float` of `[0.25, 4)`) and 4 ulp (`asin` and `acos`, on every `float` of
/// `[0.25, 1)`, both signs for `acos`) of the exact value.
#[test]
fn metal_arc_forms_stay_within_apples_bound() {
    let xs: Vec<u64> = operands(0x07A4_E297, 4000)
        .into_iter()
        .filter(|b| {
            f32v(*b).is_finite() && (f32v(*b) == 0.0 || f32v(*b).abs() >= f32::MIN_POSITIVE)
        })
        .collect();
    let in_domain: Vec<u64> = xs
        .iter()
        .copied()
        .filter(|b| f32v(*b).abs() <= 1.0)
        .collect();
    for (fun, set, exact) in [
        (
            Fun::Atan(Mode::Fast),
            &xs,
            (|x: &[f32]| (x[0].abs() > f32::MIN_POSITIVE).then(|| f64::from(x[0]).atan()))
                as fn(&[f32]) -> Option<f64>,
        ),
        (Fun::Asin(Mode::Fast), &in_domain, |x: &[f32]| {
            (x[0] != 0.0).then(|| f64::from(x[0]).asin())
        }),
        (Fun::Acos(Mode::Fast), &in_domain, |x: &[f32]| {
            Some(f64::from(x[0]).acos())
        }),
        (Fun::Atan(Mode::Precise), &xs, |x: &[f32]| {
            Some(f64::from(x[0]).atan())
        }),
    ] {
        let w = worst_error(fun, false, &unary(set), &|x| exact(x), 5.0);
        eprintln!("{fun:?}: {w:.3} ulp");
    }
}

/// Metal's fast `asin` and `acos` (the M4; the M3's are the same at every `half` operand) at lanes
/// where its word is not the correctly rounded one: below 0.25 (stepping 16 `float`s from `2^-12`),
/// on `[0.25, 0.7)`, and from 0.7 to 1, for `acos` both signs. Each word bit for bit (`asin` both
/// signs). The last 8 of `asin` and 12 of `acos` are from 0.7 on where the form with the correctly
/// rounded inverse root is not Metal's word either, and the hardware's inverse root's model is.
#[test]
fn fast_asin_and_acos_answer_metals_words() {
    let asin: [(u64, u64); 32] = [
        (0x3CF62CB0, 0x3CF6362F),
        (0x3999ECE0, 0x3999ECE1),
        (0x399EE730, 0x399EE731),
        (0x3AD716F0, 0x3AD716F8),
        (0x3B4C04D0, 0x3B4C04E8),
        (0x39D2A8D0, 0x39D2A8D2),
        (0x3CFA3980, 0x3CFA4379),
        (0x3A9EADF0, 0x3A9EADF4),
        (0x3E99830C, 0x3E9BE90A),
        (0x3EA5706C, 0x3EA8765D),
        (0x3F10A18C, 0x3F19B338),
        (0x3ED863CF, 0x3EDF6994),
        (0x3EBE29E1, 0x3EC2D4ED),
        (0x3ED80DF2, 0x3EDF0ADB),
        (0x3E81842B, 0x3E82F05E),
        (0x3EE6624D, 0x3EEEF746),
        (0x3F73B78D, 0x3FA14019),
        (0x3F38AD57, 0x3F4E49DE),
        (0x3F478662, 0x3F64C96B),
        (0x3F6689CD, 0x3F8F7D45),
        (0x3F7AAB17, 0x3FAEE47E),
        (0x3F745A1C, 0x3FA24D1F),
        (0x3F48A433, 0x3F669399),
        (0x3F430BCE, 0x3F5DC1E4),
        (0x3F33_3352, 0x3F46_809E),
        (0x3F38_689B, 0x3F4D_E6B0),
        (0x3F3F_2320, 0x3F57_CDA0),
        (0x3F45_2DAF, 0x3F61_1330),
        (0x3F4C_A8BC, 0x3F6D_2725),
        (0x3F54_2347, 0x3F7A_0A05),
        (0x3F63_DE37, 0x3F8C_7DAD),
        (0x3F7F_ED13, 0x3FC5_FC55),
    ];
    let acos: [(u64, u64); 44] = [
        (0x3E5C38E0, 0x3FAD5143),
        (0x3B091CE0, 0x3FC8CB4D),
        (0x3E5736D0, 0x3FADF545),
        (0x3C547030, 0x3FC766F8),
        (0x3AB4C110, 0x3FC8E2AB),
        (0x39A395D0, 0x3FC905A2),
        (0x3F1FE3CF, 0x3F656E66),
        (0x3F1AF5B2, 0x3F6BAF03),
        (0x3EDAECD9, 0x3F908232),
        (0x3F224F40, 0x3F625156),
        (0x3EE8C12D, 0x3F8CA7ED),
        (0x3F270BD8, 0x3F5C2171),
        (0x3F438211, 0x3F33A6EE),
        (0x3F6D44BD, 0x3EC513A9),
        (0x3F3473B6, 0x3F49DCEE),
        (0x3F4BF531, 0x3F26228A),
        (0x3F765AEE, 0x3E8CFD24),
        (0x3F60CDDB, 0x3EFF6723),
        (0x3F531A80, 0x3F19EC3A),
        (0x3F568C06, 0x3F13B9BF),
        (0xBEAFE43F, 0x3FF5F300),
        (0xBECC61AA, 0x3FFD9F39),
        (0xBEA71227, 0x3FF39BDD),
        (0xBECB9768, 0x3FFD6818),
        (0xBEBF2072, 0x3FFA0781),
        (0xBEA0682E, 0x3FF1D9BA),
        (0xBF45B1F7, 0x401D00AE),
        (0xBF7F80FA, 0x4045139B),
        (0xBF641A15, 0x402AE7AC),
        (0xBF5BF761, 0x4026B187),
        (0xBF65828E, 0x402BB09D),
        (0xBF679D26, 0x402CE683),
        (0x3F33_3640, 0x3F4B_9AFD),
        (0x3F4E_A39D, 0x3F21_A56C),
        (0x3F58_5EB5, 0x3F10_598A),
        (0x3F63_D477, 0x3EF2_737B),
        (0x3F73_7C51, 0x3EA0_C062),
        (0x3F7F_FE3A, 0x3BF1_109C),
        (0xBF33_368F, 0x4016_2938),
        (0xBF4C_3C82, 0x401F_A4C1),
        (0xBF54_7157, 0x4023_2D5A),
        (0xBF5B_8C97, 0x4026_7D7A),
        (0xBF66_1E0A, 0x402C_08D7),
        (0xBF7F_8B3F, 0x4045_3DBB),
    ];
    for fma in [false, true] {
        let f = f32s();
        let cases: Vec<Case> = asin
            .iter()
            .flat_map(|(x, _)| [vec![vec![*x]], vec![vec![*x | 0x8000_0000]]])
            .collect();
        check_module(
            &format!("fast asin fma={fma}"),
            routine_module(Fun::Asin(Mode::Fast), false, fma),
            &[f],
            f,
            &cases,
            &|c| {
                let x = c[0][0];
                asin.iter()
                    .find(|(w, _)| *w == x & 0x7FFF_FFFF)
                    .map(|(_, r)| vec![*r | (x & 0x8000_0000)])
            },
            &|g, w| g == w,
        );
        let cases: Vec<Case> = acos.iter().map(|(x, _)| vec![vec![*x]]).collect();
        check_module(
            &format!("fast acos fma={fma}"),
            routine_module(Fun::Acos(Mode::Fast), false, fma),
            &[f],
            f,
            &cases,
            &|c| {
                acos.iter()
                    .find(|(w, _)| *w == c[0][0])
                    .map(|(_, r)| vec![*r])
            },
            &|g, w| g == w,
        );
    }
}

/// The generic part of precise `atan2`, Metal's own form, is within Apple's bound (6 ulp, MSL Table
/// 8.1) for finite nonzero normal operands, quotients from `2^-200` to `2^200` included; Metal's
/// own words are within 3.3 ulp of the exact value (every finite normal pair of our probe
/// whose answer is normal). Its quotients are the module's correctly rounded division routine's,
/// whatever the device's.
#[test]
fn atan2_routine_stays_within_its_bound() {
    let xs = operands(0x051A_BE11, 600);
    let normal: Vec<u64> = xs
        .iter()
        .copied()
        .filter(|b| {
            let v = f32v(*b);
            v.is_normal()
        })
        .collect();
    let mut rng = Rng(0x2BAD_F00D);
    let mut pairs = Vec::new();
    for _ in 0..6000 {
        let y = normal[(rng.next() as usize) % normal.len()];
        let x = normal[(rng.next() as usize) % normal.len()];
        pairs.push(vec![y, x]);
    }
    // Quotients near one and near the reduction's edges.
    for k in [0.4375f32, 0.6875, 1.0, 1.1875, 2.4375, 0.999_999_9] {
        for s in [1.0f32, -1.0] {
            pairs.push(vec![
                u64::from((k * s).to_bits()),
                u64::from(1.0f32.to_bits()),
            ]);
            pairs.push(vec![
                u64::from(1.0f32.to_bits()),
                u64::from((k * s).to_bits()),
            ]);
        }
    }
    let w = worst_error(
        Fun::Atan2,
        false,
        &pairs,
        &|x| {
            let r = f64::from(x[0]).atan2(f64::from(x[1]));
            // A result `atan2_special` flushes (below 2^-126) is its, not the routine's.
            (r.abs() >= 2f64.powi(-126)).then_some(r)
        },
        6.0,
    );
    eprintln!("Atan2: {w:.3}");
}

/// Metal's precise `atan2(y, x)` (the M4) at pairs where its word is not the
/// correctly rounded one, five in each quadrant's quotients below `2^-12`, below `RN(tan(π/8))`,
/// below `RN(tan(3π/8))`, and above, toward a positive and a negative `x`; each word bit for bit.
/// These pairs are among those where the reciprocal model is Metal's.
#[test]
fn precise_atan2_answers_metals_words() {
    let words: [(u64, u64, u64); 40] = [
        (0xC7CB9EAA, 0x3F8001D8, 0xBFC90F8B),
        (0xC29D223D, 0x3FF4603E, 0xBFC5F3BF),
        (0xC1334FA1, 0x40946D3E, 0xBF96D535),
        (0x3F59E032, 0xC1842DFF, 0x4045C4A8),
        (0x41E2023E, 0x3D7F73A3, 0x3FC8C785),
        (0xC176DECC, 0xC257C9F3, 0xC0373BBB),
        (0x3C023B14, 0x41844686, 0x39FC0AF2),
        (0xB7327783, 0xB7A34141, 0xC0290C3C),
        (0x474B21BF, 0x3F581529, 0x3FC90F53),
        (0x3F764DFF, 0x4287BD5D, 0x3C683EC1),
        (0xBBFCA34D, 0xBF033F5B, 0xC048197D),
        (0x3FC7FDF7, 0xC2AEB69E, 0x4047EAD9),
        (0x3E7BA212, 0x3FD7E2F6, 0x3E1426DE),
        (0x0868E679, 0x82516CA0, 0x3FC9170D),
        (0x40D4D09C, 0xC1854DCC, 0x4030C221),
        (0x3F6B30D1, 0xBEA6EA53, 0x3FF4B56A),
        (0xC09627FA, 0xC0E202E3, 0xC023880C),
        (0xC1123F6F, 0xC18DFA94, 0xC02A9F97),
        (0xBF8CE01F, 0x4092549B, 0xBE71DB4D),
        (0xC10E031C, 0x417EEAFD, 0xBF021DFD),
        (0x400AADC7, 0x3FB582E0, 0x3F7DC6B1),
        (0xBDB60C6F, 0x408F06BF, 0xBCA2E6AD),
        (0xC18A40F2, 0x412B56FD, 0xBF820DDF),
        (0x3FD72868, 0x3F800000, 0x3F845E41),
        (0x3DB7A5FD, 0xB9A8757B, 0x3FC98545),
        (0x40A8B4DB, 0x40FC10E4, 0x3F16FE71),
        (0xC188098E, 0xB8EC1249, 0xBFC91013),
        (0xBEF9A9A0, 0xB820258A, 0xBFC9126C),
        (0xC147F77F, 0xC0ADD2A8, 0xBFFD8AB0),
        (0x1BCA6E57, 0xA57C6253, 0x40490FD5),
        (0x416E7672, 0xC162697D, 0x40152347),
        (0x1D34B8B5, 0xA75EAF2B, 0x40490FD8),
        (0xBBABB6BE, 0x42222524, 0xB9078DC2),
        (0xCF3CE185, 0xDA1A7835, 0xC0490FDA),
        (0xCA02C1C0, 0xD5C95DBB, 0xC0490FDB),
        (0x40B93FD1, 0x47417508, 0x38F5237C),
        (0xC1ABF335, 0xCE0DB5EC, 0xC0490FDB),
        (0x3BD48266, 0x41E76EAD, 0x396B117B),
        (0x3BB0A5ED, 0x41EBC17C, 0x393FD11B),
        (0x3BA6C4AF, 0x41DEBF99, 0x393FA9CA),
    ];
    let cases: Vec<Case> = words
        .iter()
        .map(|(y, x, _)| vec![vec![*y], vec![*x]])
        .collect();
    for fma in [false, true] {
        let f = f32s();
        check_module(
            &format!("precise atan2 fma={fma}"),
            routine_module(Fun::Atan2, false, fma),
            &[f, f],
            f,
            &cases,
            &|c| {
                words
                    .iter()
                    .find(|(y, x, _)| *y == c[0][0] && *x == c[1][0])
                    .map(|(_, _, r)| vec![*r])
            },
            &|g, w| g == w,
        );
    }
}

/// Metal's answers at the special operands (cases 087758cb and cc234732), bit for bit: zeros and
/// denormals as each function and mode reads them, infinities, `±1`, and the smallest normal.
#[test]
fn arc_tangent_family_routines_answer_metals_special_values() {
    const HALF_PI: u64 = 0x3FC9_0FDB;
    let specials: [(Fun, u64, u64); 28] = [
        (Fun::Atan(Mode::Precise), 0x0000_0000, 0x0000_0000),
        (Fun::Atan(Mode::Precise), 0x8000_0000, 0x0000_0000),
        (Fun::Atan(Mode::Precise), 0x8000_0001, 0x0000_0000),
        (Fun::Atan(Mode::Precise), 0x807F_FFFF, 0x0000_0000),
        (Fun::Atan(Mode::Precise), 0x8080_0000, 0x8080_0000),
        (Fun::Atan(Mode::Precise), 0x7F80_0000, HALF_PI),
        (Fun::Atan(Mode::Precise), 0xFF80_0000, HALF_PI | 0x8000_0000),
        (Fun::Atan(Mode::Precise), 0x7F7F_FFFF, HALF_PI),
        (Fun::Atan(Mode::Precise), 0x3F80_0000, 0x3F49_0FDB),
        (Fun::Atan(Mode::Fast), 0x8000_0000, 0x8000_0000),
        (Fun::Atan(Mode::Fast), 0x8000_0001, 0x8000_0000),
        (Fun::Atan(Mode::Fast), 0x0000_0001, 0x0000_0000),
        (Fun::Atan(Mode::Fast), 0xFF80_0000, HALF_PI | 0x8000_0000),
        (Fun::Asin(Mode::Precise), 0x8000_0000, 0x8000_0000),
        (Fun::Asin(Mode::Precise), 0x0000_0001, 0x0000_0001),
        (Fun::Asin(Mode::Precise), 0x807F_FFFF, 0x807F_FFFF),
        (Fun::Asin(Mode::Precise), 0x3F80_0000, HALF_PI),
        (Fun::Asin(Mode::Precise), 0xBF80_0000, HALF_PI | 0x8000_0000),
        (Fun::Asin(Mode::Fast), 0x8000_0000, 0x0000_0000),
        (Fun::Asin(Mode::Fast), 0x8000_0001, 0x0000_0000),
        (Fun::Asin(Mode::Fast), 0x007F_FFFF, 0x0000_0000),
        (Fun::Acos(Mode::Precise), 0x8000_0001, HALF_PI),
        (Fun::Acos(Mode::Precise), 0x3F80_0000, 0x0000_0000),
        (Fun::Acos(Mode::Precise), 0xBF80_0000, 0x4049_0FDB),
        (Fun::Acos(Mode::Fast), 0x0000_0000, HALF_PI),
        (Fun::Acos(Mode::Fast), 0xBF80_0000, 0x4049_0FDB),
        (Fun::Acos(Mode::Fast), 0x3F80_0000, 0x0000_0000),
        (Fun::Asin(Mode::Fast), 0x3F80_0000, HALF_PI),
    ];
    for rough in [false, true] {
        for (fun, x, want) in specials {
            let f = f32s();
            check_module(
                &format!("{fun:?}({x:#010x}) rough={rough}"),
                routine_module(fun, rough, false),
                &[f],
                f,
                &[vec![vec![x]]],
                &|_| Some(vec![want]),
                &|g, w| g == w,
            );
        }
        // A domain error is a NaN.
        for fun in [
            Fun::Asin(Mode::Precise),
            Fun::Asin(Mode::Fast),
            Fun::Acos(Mode::Precise),
            Fun::Acos(Mode::Fast),
        ] {
            for x in [
                0x3F80_0001u64,
                0xBF80_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC0_0000,
            ] {
                let f = f32s();
                check_module(
                    &format!("{fun:?}({x:#010x}) rough={rough}"),
                    routine_module(fun, rough, false),
                    &[f],
                    f,
                    &[vec![vec![x]]],
                    &|_| Some(vec![0x7FC0_0000]),
                    &|g, _| f32v(g).is_nan(),
                );
            }
        }
    }
}

/// Each routine's words, with its constants and types, stay within [`Fun::words`], with and without
/// `OpFmaKHR`, and the estimate is at most twice the larger.
#[test]
fn libm_routines_fit_their_word_estimates() {
    let module_words = |m: TypedModule| serialize(m).expect("serialize").words.len();
    let mut off = Vec::new();
    for fun in [
        Fun::Atan(Mode::Precise),
        Fun::Atan(Mode::Fast),
        Fun::Asin(Mode::Precise),
        Fun::Asin(Mode::Fast),
        Fun::Acos(Mode::Precise),
        Fun::Acos(Mode::Fast),
        Fun::Atan2,
        Fun::Sin,
        Fun::Cos,
        Fun::Tan,
        Fun::FastSin,
        Fun::FastCos,
        Fun::SinPi(Mode::Precise),
        Fun::CosPi(Mode::Precise),
        Fun::TanPi(Mode::Precise),
        Fun::SinPi(Mode::Fast),
        Fun::CosPi(Mode::Fast),
        Fun::TanPi(Mode::Fast),
        Fun::Sqrt,
        Fun::Rsqrt,
        Fun::Fmod,
        Fun::Fused,
        Fun::HalfAtan,
        Fun::HalfAtan2,
        Fun::HalfTan,
        Fun::HalfSin,
        Fun::HalfCos,
        Fun::HalfAsinh,
    ] {
        // The estimate covers the larger emission (without `OpFmaKHR` a fused step is
        // emulated) and is not loose by more than half of it.
        let bound = fun.words() as usize;
        let mut largest = 0;
        for fma in [false, true] {
            let bare = module_words(helpers_module(fun, fma));
            let used = module_words(routine_module(fun, false, fma));
            let emitted = used - bare;
            eprintln!("{fun:?} fma={fma}: {emitted} words against {bound}");
            if emitted > bound {
                off.push(format!("{fun:?} fma={fma}: {emitted} words over {bound}"));
            }
            largest = largest.max(emitted);
        }
        if 2 * largest < bound {
            off.push(format!("{fun:?}: {bound} words for at most {largest}"));
        }
    }
    assert!(off.is_empty(), "{off:#?}");
}

/// A `half` call of the arc tangent family takes a routine before its rounding (the `float` routine
/// of `asin` and `acos`, precise: Metal's `half` functions are the same in both modes; `atan`'s own
/// `half` form, `half_forms_follow_their_model`), so it is within a `half` ulp of the exact value
/// over every pattern, and it answers Metal's `half` zeros (the half probe, cases dc0fee8f and
/// 3ef7c1c5): `asin(±0)` is `+0` and `atan(±0)` the zero itself, where the `float` routines answer
/// `asin(-0) = -0` and `atan(-0) = +0`. Part `part` of `of` checks every `of`-th pattern.
fn half_arc_functions(part: usize, of: usize) {
    use std::collections::BTreeMap;
    let h = float(16);
    let functions: [(FloatFn, Fun, fn(f64) -> f64); 3] = [
        (FloatFn::Atan, Fun::HalfAtan, f64::atan),
        (FloatFn::Asin, Fun::Asin(Mode::Precise), f64::asin),
        (FloatFn::Acos, Fun::Acos(Mode::Precise), f64::acos),
    ];
    let cases: Vec<Case> = (0..=0xFFFFu64)
        .filter(|b| b & 0x7C00 != 0x7C00 || b & 0x03FF == 0)
        .map(|b| vec![vec![b]])
        .collect();
    let cases = part_of(cases, part, of);
    for (op, fun, exact) in functions {
        let m = kernel_defs(
            &[h],
            h,
            &|b| Ok(vec![libm::define_alone(b, fun, false)?]),
            &|fb, x, funcs| {
                let map: BTreeMap<Fun, FuncId> = funcs.iter().map(|id| (fun, *id)).collect();
                math::float_fn(
                    fb,
                    op,
                    h,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::new(None, &map),
                )
            },
            true,
            false,
        )
        .unwrap_or_else(|e| panic!("{op:?}: build: {e:?}"));
        check_module(
            &format!("half {op:?}"),
            m,
            &[h],
            h,
            &cases,
            &|c| {
                let b = c[0][0];
                let x = f16_to_f64(b as u16);
                let r = match (op, x == 0.0) {
                    (FloatFn::Asin, true) => 0.0,
                    (FloatFn::Atan, true) => x,
                    _ => exact(x),
                };
                Some(vec![u64::from(f64_to_f16(r))])
            },
            &|g, w| {
                let nan = |b: u64| b & 0x7C00 == 0x7C00 && b & 0x03FF != 0;
                if nan(w) || nan(g) {
                    return nan(w) && nan(g);
                }
                // A zero answer is exact; another is within one `half` ulp of the same sign.
                if w & 0x7FFF == 0 || g & 0x7FFF == 0 {
                    return g == w;
                }
                (g ^ w) & 0x8000 == 0 && (g as i64 - w as i64).abs() <= 1
            },
        );
    }
}

mod half_arc_functions_take_the_routines_and_answer_metals_zeros {
    parts!(super::half_arc_functions, 6; part_1_of_6 = 0, part_2_of_6 = 1,
        part_3_of_6 = 2, part_4_of_6 = 3, part_5_of_6 = 4, part_6_of_6 = 5);
}

/// Metal's `half` `atan` of a `half` held as `float`, before its rounding: the test's model of
/// `Fun::HalfAtan`'s routine (`libm::atan::half_atan`).
fn half_atan_form(x: f32) -> f32 {
    let p = libm::atan::HALF_ATAN_P.map(f32::from_bits);
    let a = x.abs();
    let above = a > 1.0;
    let t = match (above, a.is_infinite()) {
        (true, true) => 0.0,
        (true, false) => 1.0 / a,
        (false, _) => a,
    };
    let s = t * t;
    let q = ((p[3] * s + p[2]) * s + p[1]) * s + p[0];
    let core = t * q;
    let r = if above {
        std::f32::consts::FRAC_PI_2 - core
    } else {
        core
    };
    r.copysign(x)
}

/// Metal's `half` `atan2(y, x)` of finite nonzero `half` operands held as `float`, before its
/// rounding: the model of `Fun::HalfAtan2`'s routine.
fn half_atan2_form(y: f32, x: f32) -> f32 {
    let t = half_atan_form(y / x);
    match (x < 0.0, y < 0.0) {
        (true, true) => t - std::f32::consts::PI,
        (true, false) => t + std::f32::consts::PI,
        _ => t,
    }
}

/// Metal's `half` `tan` of a `half` held as `float`, before its rounding: the model of
/// `Fun::HalfTan`'s routine (`libm::trig::half_tan`), each fused step one `mul_add` and Metal's
/// reciprocal its model.
fn half_tan_form(x: f32) -> f32 {
    let (r, k) = half_quarter_turns(x, 12_582_912.0);
    let p = libm::trig::HALF_TAN_P.map(f32::from_bits);
    let s = r * r;
    let t = r * p[3].mul_add(s, p[2]).mul_add(s, p[1]).mul_add(s, p[0]);
    if k & 1 != 0 {
        libm::rcp::model(-t)
    } else {
        t
    }
}

/// The reduction of Metal's `half` `sin`, `cos`, and `tan`: `(r, k)` of `s = fma(x, c0, magic)`,
/// `k` its bits and `r = fma(x, c1, fma(x, c0, magic - s))` of `2/π`'s first two parts.
fn half_quarter_turns(x: f32, magic: f32) -> (f32, u32) {
    let [c0, c1, _] = libm::quarter::QUARTER.map(f32::from_bits);
    let s = x.mul_add(c0, magic);
    let r = x.mul_add(c0, magic - s);
    (x.mul_add(c1, r), s.to_bits())
}

/// Metal's `half` `sin` (`cos` where `cos`) of a `half` held as `float`, before its rounding: the
/// model of `Fun::HalfSin`'s and `Fun::HalfCos`'s routine (`libm::sincos::half_sin_cos`), the
/// `sinc` [`metal_sinc_model`].
fn half_sin_cos_form(x: f32, cos: bool) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    if !cos && x.abs() < 1.0 / 64.0 {
        return x;
    }
    let (r, k) = half_quarter_turns(x, if cos { 12_582_913.0 } else { 12_582_912.0 });
    let t = if k & 1 != 0 { 1.0 - r.abs() } else { r };
    let t = if k & 2 != 0 { -t } else { t };
    t * metal_sinc_model(t.abs())
}

/// A module of `half` function `op` (operands of form `h`) lowered as a call takes it, with the
/// routine `fun` (and its helpers) defined.
pub(super) fn half_module(op: FloatFn, fun: Fun) -> TypedModule {
    use std::collections::BTreeMap;
    let h = float(16);
    kernel_defs(
        &vec![h; fun.arity()],
        h,
        &|b| Ok(vec![libm::define_alone(b, fun, false)?]),
        &|fb, x, funcs| {
            let map: BTreeMap<Fun, FuncId> = funcs.iter().map(|id| (fun, *id)).collect();
            math::float_fn(
                fb,
                op,
                h,
                Contraction::Forbidden,
                NanTest::IsNan,
                x,
                &Lib::new(None, &map),
            )
        },
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("{op:?}: build: {e:?}"))
}

/// A `half` `atan`, `tan`, `sin`, `cos`, or `atan2` call answers its model of Metal's own `half`
/// form rounded to `half` ([`half_atan_form`], [`half_tan_form`], [`half_sin_cos_form`],
/// [`half_atan2_form`]) at every `half` (`atan2`: at the 65,536 pairs of our probe, `y` every
/// pattern and `x` its hash, and at those of finite nonzero operands), a zero `sin` or `tan`
/// operand giving `+0` and a NaN or an infinite operand a NaN. Rounded so, each model is Metal's
/// word at every `half` of our probes (`atan`, `atan2`, `tan`, `sin`, `cos`; the M3
/// and the M4); Metal's words where they are not the correctly rounded ones are
/// `half_forms_answer_metals_words`'. Part `part` of `of` checks every `of`-th
/// run.
fn half_forms(part: usize, of: usize) {
    let h = float(16);
    let nan = |b: u64| b & 0x7C00 == 0x7C00 && b & 0x03FF != 0;
    let half_of = |x: f32| u64::from(f64_to_f16(f64::from(x)));
    let all: Vec<Case> = (0..=0xFFFFu64).map(|b| vec![vec![b]]).collect();
    let finite = |b: u64| b & 0x7C00 != 0x7C00 && b & 0x7FFF != 0;
    let pairs: Vec<Case> = (0..=0xFFFFu32)
        .map(|i| (u64::from(i), u64::from(i.wrapping_mul(2_654_435_761) >> 16)))
        .filter(|(y, x)| finite(*y) && finite(*x))
        .map(|(y, x)| vec![vec![y], vec![x]])
        .collect();
    let mut runs = Vec::new();
    for (op, fun) in [
        (FloatFn::Atan, Fun::HalfAtan),
        (FloatFn::Tan, Fun::HalfTan),
        (FloatFn::Sin, Fun::HalfSin),
        (FloatFn::Cos, Fun::HalfCos),
        (FloatFn::Atan2, Fun::HalfAtan2),
    ] {
        let cases = if op == FloatFn::Atan2 { &pairs } else { &all };
        for run in cases.chunks(4096) {
            runs.push((op, fun, run));
        }
    }
    for (op, fun, run) in part_of(runs, part, of) {
        check_module(
            &format!("half {op:?}"),
            half_module(op, fun),
            &vec![h; fun.arity()],
            h,
            run,
            &|c| {
                let x = f16_to_f64(c[0][0] as u16) as f32;
                let r = match op {
                    FloatFn::Atan2 => half_atan2_form(x, f16_to_f64(c[1][0] as u16) as f32),
                    FloatFn::Tan | FloatFn::Sin if x == 0.0 => 0.0,
                    FloatFn::Tan if !x.is_finite() => f32::NAN,
                    FloatFn::Tan => half_tan_form(x),
                    FloatFn::Sin => half_sin_cos_form(x, false),
                    FloatFn::Cos => half_sin_cos_form(x, true),
                    _ => half_atan_form(x),
                };
                Some(vec![half_of(r)])
            },
            &|g, w| g == w || (nan(g) && nan(w)),
        );
    }
}

mod half_forms_follow_their_model {
    parts!(super::half_forms, 12;
        part_1_of_12 = 0, part_2_of_12 = 1, part_3_of_12 = 2, part_4_of_12 = 3,
        part_5_of_12 = 4, part_6_of_12 = 5, part_7_of_12 = 6, part_8_of_12 = 7,
        part_9_of_12 = 8, part_10_of_12 = 9, part_11_of_12 = 10, part_12_of_12 = 11);
}

/// Metal's `half` `atan`, `tan`, `sin`, `cos`, and `atan2` words (the M3; the M4's are the same)
/// where they are not the correctly rounded ones, bit for bit: `atan` and `tan` at 24 operands
/// each, spread over the magnitudes where they differ (3,728 and 2,280 `half` values), `sin` and
/// `cos` at every nonzero one (6 and 2), and `atan2` at six pairs per quadrant (of the 3,210 probe
/// pairs where it differs).
#[test]
fn half_forms_answer_metals_words() {
    let h = float(16);
    let atan: [(u64, u64); 24] = [
        (0x237F, 0x237E),
        (0x23D0, 0x23CF),
        (0x25BF, 0x25BE),
        (0x2610, 0x260F),
        (0x2661, 0x2660),
        (0x26B2, 0x26B1),
        (0x2703, 0x2702),
        (0x280F, 0x280E),
        (0x2860, 0x285F),
        (0x2A49, 0x2A47),
        (0x2A9A, 0x2A98),
        (0x2BCC, 0x2BC9),
        (0x2CEF, 0x2CEC),
        (0x2EC3, 0x2EBC),
        (0x30AF, 0x30A6),
        (0x345E, 0x3444),
        (0x35BE, 0x3584),
        (0x36E0, 0x367F),
        (0x38FA, 0x3873),
        (0x3A5C, 0x395F),
        (0x3C05, 0x3A4E),
        (0x3D86, 0x3B8E),
        (0x40BD, 0x3CAF),
        (0xDC01, 0xBE45),
    ];
    let tan: [(u64, u64); 24] = [
        (0x2745, 0x2745),
        (0xA776, 0xA776),
        (0x27A8, 0x27A8),
        (0xA8B8, 0xA8B8),
        (0x2BE1, 0x2BE3),
        (0xAF68, 0xAF70),
        (0x34CF, 0x34F6),
        (0xB9AD, 0xBADF),
        (0x3E2D, 0x50A8),
        (0xC2C1, 0xB3AD),
        (0x4660, 0x2DE4),
        (0xCA2C, 0x333F),
        (0xCDFB, 0x4150),
        (0x525F, 0x3AC9),
        (0xD635, 0x4170),
        (0x5B2E, 0x3707),
        (0xDF5E, 0xB447),
        (0x63E8, 0x36E7),
        (0xE872, 0xC9F3),
        (0x6C71, 0xBE9B),
        (0xF0B3, 0xC340),
        (0x749D, 0xBA78),
        (0xF8C0, 0xB39C),
        (0xFBE5, 0x3C57),
    ];
    let sin: [(u64, u64); 6] = [
        (0x32B3, 0x32A6),
        (0x51F5, 0xB810),
        (0x5CB0, 0xBC00),
        (0xB2B3, 0xB2A6),
        (0xD1F5, 0x3810),
        (0xDCB0, 0x3C00),
    ];
    let cos: [(u64, u64); 2] = [(0x4578, 0x397E), (0xC578, 0x397E)];
    let atan2: [(u64, u64, u64); 24] = [
        (0x001A, 0x11A2, 0x189D),
        (0x1253, 0x328A, 0x1BBC),
        (0x255B, 0x424E, 0x1ECB),
        (0x370C, 0x55BE, 0x1CE8),
        (0x4B12, 0x608D, 0x2636),
        (0x7BF4, 0x7458, 0x3D37),
        (0x07B2, 0x86E6, 0x409B),
        (0x2957, 0xA757, 0x4058),
        (0x3EC1, 0xB74B, 0x3F57),
        (0x4F99, 0xADB5, 0x3E4B),
        (0x5F75, 0xE583, 0x41A1),
        (0x7935, 0xFA00, 0x40DA),
        (0x800F, 0x0218, 0xA729),
        (0x9370, 0x1325, 0xBA71),
        (0xA558, 0x2480, 0xBAF7),
        (0xB89C, 0x4945, 0xAAFD),
        (0xCD22, 0x6FD0, 0x9941),
        (0xFBE1, 0x7312, 0xBD66),
        (0x81A6, 0x8C4B, 0xC218),
        (0xA128, 0x9C33, 0xBFD5),
        (0xB6B4, 0xAF85, 0xBF61),
        (0xC95B, 0xCC44, 0xC12A),
        (0xE502, 0xDC4A, 0xBF21),
        (0xFBAE, 0xEE05, 0xBEAC),
    ];
    for (op, fun, words) in [
        (FloatFn::Atan, Fun::HalfAtan, &atan[..]),
        (FloatFn::Tan, Fun::HalfTan, &tan[..]),
        (FloatFn::Sin, Fun::HalfSin, &sin[..]),
        (FloatFn::Cos, Fun::HalfCos, &cos[..]),
    ] {
        let cases: Vec<Case> = words.iter().map(|(x, _)| vec![vec![*x]]).collect();
        check_module(
            &format!("half {op:?} words"),
            half_module(op, fun),
            &[h],
            h,
            &cases,
            &|c| {
                words
                    .iter()
                    .find(|(x, _)| *x == c[0][0])
                    .map(|(_, r)| vec![*r])
            },
            &|g, w| g == w,
        );
    }
    let cases: Vec<Case> = atan2
        .iter()
        .map(|(y, x, _)| vec![vec![*y], vec![*x]])
        .collect();
    check_module(
        "half Atan2 words",
        half_module(FloatFn::Atan2, Fun::HalfAtan2),
        &[h, h],
        h,
        &cases,
        &|c| {
            atan2
                .iter()
                .find(|(y, x, _)| *y == c[0][0] && *x == c[1][0])
                .map(|(_, _, r)| vec![*r])
        },
        &|g, w| g == w,
    );
}

/// At a domain error of its function every routine answers Metal's NaN bit for bit, a constant, so
/// no device's NaN encoding reaches it (Metal's rows of cases 087758cb and cc234732, every such
/// lane): `0x7fc00000` for an infinite operand of `sin`, `cos`, and `tan`, `|x| > 1` of precise
/// `asin` and of `acos`, a negative normal or infinite operand of `sqrt` and `rsqrt`, and an
/// infinite dividend or a zero or denormal divisor of `fmod`; and `fast::asin`'s carries the
/// operand's sign. Metal's `half` `asin` of `|x| > 1` carries the operand's sign in both modes
/// (cases 48500c31 and 32931370), its `half` `acos`'s does not.
#[test]
fn routines_answer_metals_nan_at_their_domain_errors() {
    use std::collections::BTreeMap;
    const NAN: u64 = 0x7FC0_0000;
    const NEG_NAN: u64 = 0xFFC0_0000;
    let f = f32s();
    let unary: Vec<(Fun, u64, u64)> = [
        (Fun::Sin, 0x7F80_0000u64, NAN),
        (Fun::Sin, 0xFF80_0000, NAN),
        (Fun::Cos, 0x7F80_0000, NAN),
        (Fun::Cos, 0xFF80_0000, NAN),
        (Fun::Tan, 0x7F80_0000, NAN),
        (Fun::Tan, 0xFF80_0000, NAN),
        (Fun::Asin(Mode::Precise), 0x3F80_0001, NAN),
        (Fun::Asin(Mode::Precise), 0xBF80_0001, NAN),
        (Fun::Asin(Mode::Precise), 0xC020_0000, NAN),
        (Fun::Asin(Mode::Precise), 0xFF80_0000, NAN),
        (Fun::Asin(Mode::Fast), 0x4020_0000, NAN),
        (Fun::Asin(Mode::Fast), 0x7F80_0000, NAN),
        (Fun::Asin(Mode::Fast), 0xC020_0000, NEG_NAN),
        (Fun::Asin(Mode::Fast), 0xFF80_0000, NEG_NAN),
        (Fun::Acos(Mode::Precise), 0xC020_0000, NAN),
        (Fun::Acos(Mode::Precise), 0x7F80_0000, NAN),
        (Fun::Acos(Mode::Fast), 0xC020_0000, NAN),
        (Fun::Acos(Mode::Fast), 0xFF80_0000, NAN),
        (Fun::Sqrt, 0xBF80_0000, NAN),
        (Fun::Sqrt, 0x8080_0000, NAN),
        (Fun::Sqrt, 0xFF80_0000, NAN),
        (Fun::Rsqrt, 0xBF80_0000, NAN),
        (Fun::Rsqrt, 0x8080_0000, NAN),
        (Fun::Rsqrt, 0xFF80_0000, NAN),
    ]
    .to_vec();
    for rough in [false, true] {
        for (fun, x, want) in &unary {
            check_module(
                &format!("{fun:?}({x:#010x}) rough={rough}"),
                routine_module(*fun, rough, false),
                &[f],
                f,
                &[vec![vec![*x]]],
                &|_| Some(vec![*want]),
                &|g, w| g == w,
            );
        }
        for (x, y) in [
            (0x7F80_0000u64, 0x3F80_0000u64),
            (0xFF80_0000, 0x4000_0000),
            (0x3F80_0000, 0x0000_0000),
            (0xC000_0000, 0x8000_0000),
            (0x4000_0000, 0x0000_0001),
            (0x0000_0001, 0x0000_0000),
        ] {
            check_module(
                &format!("Fmod({x:#010x}, {y:#010x}) rough={rough}"),
                routine_module(Fun::Fmod, rough, false),
                &[f, f],
                f,
                &[vec![vec![x], vec![y]]],
                &|_| Some(vec![NAN]),
                &|g, w| g == w,
            );
        }
    }
    // `half` `asin` and `acos` through the precise routines.
    let h = float(16);
    for (op, fun, cases) in [
        (
            FloatFn::Asin,
            Fun::Asin(Mode::Precise),
            [
                (0x4100u64, 0x7E00u64),
                (0xC100, 0xFE00),
                (0x7C00, 0x7E00),
                (0xFC00, 0xFE00),
            ],
        ),
        (
            FloatFn::Acos,
            Fun::Acos(Mode::Precise),
            [
                (0x4100, 0x7E00),
                (0xC100, 0x7E00),
                (0x7C00, 0x7E00),
                (0xFC00, 0x7E00),
            ],
        ),
    ] {
        let m = kernel_defs(
            &[h],
            h,
            &|b| Ok(vec![libm::define_alone(b, fun, false)?]),
            &|fb, x, funcs| {
                let map: BTreeMap<Fun, FuncId> = funcs.iter().map(|id| (fun, *id)).collect();
                math::float_fn(
                    fb,
                    op,
                    h,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::new(None, &map),
                )
            },
            true,
            false,
        )
        .unwrap_or_else(|e| panic!("{op:?}: build: {e:?}"));
        let inputs: Vec<Case> = cases.iter().map(|(x, _)| vec![vec![*x]]).collect();
        check_module(
            &format!("half {op:?}"),
            m,
            &[h],
            h,
            &inputs,
            &|c| {
                let x = c[0][0];
                cases.iter().find(|(b, _)| *b == x).map(|(_, w)| vec![*w])
            },
            &|g, w| g == w,
        );
    }
}

/// Operands for the trigonometric routines: [`operands`], and the floats nearest the first 4,096
/// and some larger multiples of `π/2` with their neighbours, where the reduction cancels most.
fn trig_operands() -> Vec<u64> {
    let mut v = operands(0x0B5E_55ED, 3000);
    let mut rng = Rng(0x7A11_C0DE);
    let ks = (1u64..4096).chain((0..2000).map(|_| rng.next() % (1 << 40)));
    for k in ks {
        let x = (k as f64 * std::f64::consts::FRAC_PI_2) as f32;
        if !x.is_finite() {
            continue;
        }
        for d in [-1i64, 0, 1] {
            let b = (i64::from(x.to_bits()) + d) as u64;
            v.push(b);
            v.push(b | 0x8000_0000);
        }
    }
    v.retain(|b| f32v(*b).is_finite());
    v
}

/// The `OpFmaKHR` instructions of `m`.
fn fma_count(m: TypedModule) -> usize {
    let words = serialize(m).expect("serialize").words;
    let mut at = 5;
    let mut n = 0;
    while at < words.len() {
        let w = words[at];
        if w & 0xFFFF == crate::spv::Op::FmaKHR as u32 {
            n += 1;
        }
        at += (w >> 16).max(1) as usize;
    }
    n
}

/// On a profile whose `float` fused multiply-add is `OpFmaKHR`, the trigonometric routines take it
/// for a product's exact error, Metal's precise `atan` and `atan2` for their own fused steps, its
/// `tanpi` for its denominator, and its `half` `tan` for its reduction and polynomial, and no other
/// first-stage routine does (`sinpi`, `cospi`, and the `half` `atan` and `atan2` among them); on
/// any other profile no routine does.
#[test]
fn trig_routines_take_fma_where_the_profile_has_it() {
    for fun in [
        Fun::Sin,
        Fun::Cos,
        Fun::Tan,
        Fun::FastSin,
        Fun::FastCos,
        Fun::SinPi(Mode::Precise),
        Fun::CosPi(Mode::Precise),
        Fun::TanPi(Mode::Precise),
        Fun::SinPi(Mode::Fast),
        Fun::CosPi(Mode::Fast),
        Fun::TanPi(Mode::Fast),
        Fun::Atan(Mode::Precise),
        Fun::Atan2,
        Fun::Sqrt,
        Fun::Rsqrt,
        Fun::Fmod,
        Fun::HalfAtan,
        Fun::HalfAtan2,
        Fun::HalfTan,
        Fun::HalfSin,
        Fun::HalfCos,
        Fun::HalfAsinh,
    ] {
        let fuses = matches!(
            fun,
            Fun::Sin
                | Fun::Cos
                | Fun::Tan
                | Fun::FastSin
                | Fun::FastCos
                | Fun::TanPi(_)
                | Fun::Atan(Mode::Precise)
                | Fun::Atan2
                | Fun::HalfTan
                | Fun::HalfSin
                | Fun::HalfCos
                | Fun::HalfAsinh
        );
        // The routine's own: a helper takes `OpFmaKHR` on its own account (the precise
        // division's fused path).
        let own =
            |fma| fma_count(routine_module(fun, false, fma)) - fma_count(helpers_module(fun, fma));
        assert_eq!(own(false), 0, "{fun:?}");
        let fused = own(true);
        assert_eq!(fused > 0, fuses, "{fun:?}: {fused} OpFmaKHR");
    }
}

/// The precise `sin` and `cos` routines answer [`metal_sin_cos_model`] bit for bit, with and
/// without `OpFmaKHR`, over [`trig_operands`] (every binade, and the floats nearest multiples of
/// `π/2` to `2^40 π/2`), the edges of the reduction, the tiny operands' bound and the `sinc`'s
/// segments, and every 65,537th finite `float`, both signs; and the model lies within Apple's
/// bound of the exact value there (4 ulp). Part `part` of `of` checks every `of`-th operand.
fn precise_sin_cos_is_metals(part: usize, of: usize) {
    let mut xs = trig_operands();
    for edge in [0x3980_0000u32, 0x4A00_0000, 0x4B00_0000, 0x7F00_0000] {
        for d in 0..=64u32 {
            xs.push(u64::from((edge + d).wrapping_sub(32)));
        }
    }
    // Operands whose folded value starts a segment of the `sinc` (`t = j / 32`, `x = t π/2`).
    for j in 1..32u32 {
        let x = (f64::from(j) / 32.0 * std::f64::consts::FRAC_PI_2) as f32;
        for d in 0..=8u32 {
            xs.push(u64::from((x.to_bits() + d).wrapping_sub(4)));
        }
    }
    xs.extend((0..0x7F80_0000u32).step_by(65_537).map(u64::from));
    let signed: Vec<u64> = xs
        .iter()
        .flat_map(|b| [*b & 0x7FFF_FFFF, *b | 0x8000_0000])
        .filter(|b| f32v(*b).is_finite())
        .collect();
    let signed = part_of(signed, part, of);
    let f = f32s();
    for (fun, cos) in [(Fun::Sin, false), (Fun::Cos, true)] {
        for fma in [false, true] {
            check_module(
                &format!("{fun:?} fma={fma}"),
                routine_module(fun, false, fma),
                &[f],
                f,
                &signed.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
                &|c| {
                    Some(vec![u64::from(
                        metal_sin_cos_model(f32v(c[0][0]), cos).to_bits(),
                    )])
                },
                &|g, w| g == w,
            );
        }
        let exact = if cos { f64::cos } else { f64::sin };
        let worst = signed
            .iter()
            .map(|b| {
                let x = f32v(*b);
                let e = exact(f64::from(x));
                let r = e.abs() as f32;
                let ulp = f64::from(f32::from_bits(r.to_bits() + 1)) - f64::from(r);
                (f64::from(metal_sin_cos_model(x, cos)) - e).abs() / ulp
            })
            .fold(0f64, f64::max);
        eprintln!("{fun:?}: {worst:.3} ulp");
        assert!(worst <= 4.0, "{fun:?}: {worst:.3} ulp");
    }
}

mod precise_sin_and_cos_routines_are_metals {
    parts!(super::precise_sin_cos_is_metals, 8; part_1_of_8 = 0, part_2_of_8 = 1,
        part_3_of_8 = 2, part_4_of_8 = 3, part_5_of_8 = 4, part_6_of_8 = 5, part_7_of_8 = 6,
        part_8_of_8 = 7);
}

/// [`metal_sin_cos_model`] is Metal's precise `sin` and `cos` at words measured on the M3 Max, 8 of
/// each in `[2^-12, 1/4)` and in `[1/4, 1)`, where Metal's word is not the
/// correctly rounded value, and the routines answer them bit for bit, with and without `OpFmaKHR`.
#[test]
fn precise_sin_and_cos_answer_metals_measured_words() {
    let words: [(u32, u32); 32] = [
        (0x3E80_3039, 0x3E7D_B4E7),
        (0x3EA0_3039, 0x3E9D_9678),
        (0x3EC0_3039, 0x3EBB_B4F2),
        (0x3EE0_303A, 0x3ED9_17C5),
        (0x3F00_3039, 0x3EF5_CBE2),
        (0x3F20_3039, 0x3F15_F009),
        (0x3F40_303C, 0x3F2E_A328),
        (0x3F60_3039, 0x3F44_9CA0),
        (0x3983_0390, 0x3983_038F),
        (0x3A43_03D0, 0x3A43_03CE),
        (0x3B04_92B0, 0x3B04_92A9),
        (0x3BC3_0FF0, 0x3BC3_0FA5),
        (0x3C83_0390, 0x3C83_0221),
        (0x3D43_0390, 0x3D42_F0B5),
        (0x3E03_0390, 0x3E02_A821),
        (0x3EC3_03A0, 0x3EBE_553E),
        (0x3E80_3039, 0x3F78_04AC),
        (0x3EA0_303C, 0x3F73_9289),
        (0x3EC0_303B, 0x3F6E_2CE8),
        (0x3EE0_303A, 0x3F67_D930),
        (0x3F00_303A, 0x3F60_921E),
        (0x3F20_3039, 0x3F4F_7F0C),
        (0x3F40_3039, 0x3F3B_2F13),
        (0x3F60_3039, 0x3F23_F358),
        (0x3983_0390, 0x3F80_0000),
        (0x3A43_0390, 0x3F7F_FFFA),
        (0x3B03_0390, 0x3F7F_FFDF),
        (0x3BC3_0390, 0x3F7F_FED8),
        (0x3C83_0390, 0x3F7F_F79F),
        (0x3D43_03A0, 0x3F7F_B5BB),
        (0x3E03_03B0, 0x3F7D_E854),
        (0x3EC3_03C0, 0x3F6D_A75C),
    ];
    let f = f32s();
    for (i, (x, want)) in words.into_iter().enumerate() {
        let (fun, cos) = if i < 16 {
            (Fun::Sin, false)
        } else {
            (Fun::Cos, true)
        };
        let got = metal_sin_cos_model(f32::from_bits(x), cos).to_bits();
        assert_eq!(got, want, "{fun:?} model at {x:#010x}");
        for fma in [false, true] {
            check_module(
                &format!("{fun:?}({x:#010x}) fma={fma}"),
                routine_module(fun, false, fma),
                &[f],
                f,
                &[vec![vec![u64::from(x)]]],
                &|_| Some(vec![u64::from(want)]),
                &|g, w| g == w,
            );
        }
    }
}

/// Metal's precise `sin`, `cos`, and `tan` at their special operands (cases 087758cb and cc234732):
/// a zero or a denormal is its own `sin` and `tan` and has `cos` 1, signs kept, and an infinity or
/// NaN answers a NaN.
#[test]
fn trig_routines_answer_metals_special_values() {
    let f = f32s();
    for rough in [false, true] {
        for x in [
            0x0000_0000u64,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x007F_FFFF,
            0x807F_FFFF,
            0x0080_0000,
            0x8080_0000,
            0x3900_0000,
            0xB880_0000,
        ] {
            for (fun, want) in [(Fun::Sin, x), (Fun::Cos, 0x3F80_0000), (Fun::Tan, x)] {
                check_module(
                    &format!("{fun:?}({x:#010x}) rough={rough}"),
                    routine_module(fun, rough, false),
                    &[f],
                    f,
                    &[vec![vec![x]]],
                    &|_| Some(vec![want]),
                    &|g, w| g == w,
                );
            }
        }
        for x in [0x7F80_0000u64, 0xFF80_0000, 0x7FC0_0000] {
            for fun in [Fun::Sin, Fun::Cos, Fun::Tan] {
                check_module(
                    &format!("{fun:?}({x:#010x}) rough={rough}"),
                    routine_module(fun, rough, false),
                    &[f],
                    f,
                    &[vec![vec![x]]],
                    &|_| Some(vec![0x7FC0_0000]),
                    &|g, _| f32v(g).is_nan(),
                );
            }
        }
    }
}

/// Metal's `sinpi`, `cospi`, or `tanpi` (`op`) of `x` as the model computes it: the routine's form
/// (`libm::pi`) in `f32` arithmetic, its fused step one `mul_add`, the `sinc` [`metal_sinc_model`],
/// and Metal's reciprocal through its model (`+inf` of the `+0` at a half-integer).
pub(super) fn metal_pi_model(op: FloatFn, x: f32) -> f32 {
    use crate::lower::arith::libm::rcp;
    let a = x.to_bits() & 0x7FFF_FFFF;
    if a >= 0x7F80_0000 {
        return f32::from_bits(0x7FC0_0000);
    }
    // Metal reads a denormal as the zero of its sign.
    let x = if a < 0x0080_0000 { 0f32.copysign(x) } else { x };
    let (large, whole) = (a >= 0x4B00_0000, a >= 0x4B80_0000);
    let b = if large && op != FloatFn::SinPi {
        0.0
    } else {
        16_777_216f32.copysign(x)
    };
    let x2 = x * 2.0;
    let s = x2 + b;
    let t = x2 + (b - s);
    let odd = s.to_bits() & 1 != 0;
    let at = t.abs();
    let rest = 1.0 - at;
    let signed = |v: f32| if odd { -v } else { v };
    match op {
        FloatFn::SinPi if large => 0.0,
        FloatFn::SinPi => signed(t) * metal_sinc_model(at),
        FloatFn::CosPi if whole => 1.0,
        FloatFn::CosPi => signed(rest) * metal_sinc_model(rest),
        _ if whole => 0.0,
        _ => {
            let a = t * metal_sinc_model(at);
            let c = metal_sinc_model(rest);
            let den = c.mul_add(-at, c);
            let q = if den == 0.0 {
                f32::INFINITY
            } else {
                rcp::model(den)
            };
            a * q
        }
    }
}

/// `sin(π x)`, `cos(π x)`, or `tan(π x)` (`op`) of the exact `x` in `f64`, by the exact quarter
/// turn `q` and residue `u` in `[-1/4, 1/4]` of `x = q/2 + u` (mod 2), so that the zeros and
/// poles are exact; `None` at a pole.
fn exact_pi(op: FloatFn, x: f32) -> Option<f64> {
    let x = f64::from(x);
    let r = x - 2.0 * (x / 2.0).round_ties_even();
    let q = (2.0 * r).round_ties_even();
    let u = r - q / 2.0;
    let (su, cu) = (
        (std::f64::consts::PI * u).sin(),
        (std::f64::consts::PI * u).cos(),
    );
    let (s, c) = match q as i64 {
        0 => (su, cu),
        1 => (cu, -su),
        -1 => (-cu, su),
        _ => (-su, -cu),
    };
    match op {
        FloatFn::SinPi => Some(s),
        FloatFn::CosPi => Some(c),
        _ if c == 0.0 => None,
        _ => Some(s / c),
    }
}

/// The operands of the `π`-scaled routines' checks: [`operands`], every 131,071st finite `float`,
/// the floats about `2^23` and `2^24` (where the half turns change form) and about the first 256
/// quarter turns, the tiny ones, Metal's measured operands, [`DENORMALS`], the infinities, and a
/// NaN, both signs.
fn pi_operands() -> Vec<u64> {
    let mut xs = operands(0x5150_0C05, 3000);
    xs.extend((0..0x7F80_0000u32).step_by(131_071).map(u64::from));
    for edge in [0x4B00_0000u32, 0x4B80_0000, 0x0080_0000, 0x3380_0000] {
        xs.extend((0..=64u32).map(|d| u64::from(edge + d - 32)));
    }
    for k in 1..=256u32 {
        let b = (k as f32 / 4.0).to_bits();
        xs.extend((0..=4u32).map(|d| u64::from(b + d - 2)));
    }
    xs.extend(
        PI_WORDS
            .iter()
            .flat_map(|(_, words)| words.iter().map(|w| u64::from(w.0))),
    );
    xs.extend(DENORMALS);
    xs.extend([0x7F80_0000, 0x7FC0_0000]);
    xs.iter()
        .flat_map(|b| [*b & 0x7FFF_FFFF, *b | 0x8000_0000])
        .collect()
}

/// The `sinpi`, `cospi`, and `tanpi` routines answer [`metal_pi_model`] bit for bit, with and
/// without `OpFmaKHR`, over [`pi_operands`]. Part `part` of `of` checks every `of`-th operand.
fn precise_pi_is_metals(part: usize, of: usize) {
    let f = f32s();
    let xs = part_of(pi_operands(), part, of);
    let cases: Vec<Case> = xs.iter().map(|x| vec![vec![*x]]).collect();
    for (fun, op) in [
        (Fun::SinPi(Mode::Precise), FloatFn::SinPi),
        (Fun::CosPi(Mode::Precise), FloatFn::CosPi),
        (Fun::TanPi(Mode::Precise), FloatFn::TanPi),
    ] {
        for fma in [false, true] {
            check_module(
                &format!("{fun:?} fma={fma}"),
                routine_module(fun, false, fma),
                &[f],
                f,
                &cases,
                &|c| Some(vec![u64::from(metal_pi_model(op, f32v(c[0][0])).to_bits())]),
                &|g, w| g == w,
            );
        }
    }
}

mod pi_routines_are_metals {
    parts!(super::precise_pi_is_metals, 6; part_1_of_6 = 0, part_2_of_6 = 1,
        part_3_of_6 = 2, part_4_of_6 = 3, part_5_of_6 = 4, part_6_of_6 = 5);
}

/// [`metal_pi_model`] lies within 3 ulp of the exact value for `sinpi` and `cospi`, and within 5
/// for `tanpi`, over [`pi_operands`] and every 4,099th finite normal `float`, both signs (the worst
/// over every `float` from `2^-11` by the model, our probe's analysis: 2.26 and 4.25); and it
/// answers a denormal operand as its zero.
#[test]
fn pi_model_stays_near_the_exact_value() {
    let mut xs = pi_operands();
    xs.extend(
        (0x0080_0000..0x7F80_0000u32)
            .step_by(4_099)
            .flat_map(|b| [b, b | 0x8000_0000])
            .map(u64::from),
    );
    for (op, limit) in [
        (FloatFn::SinPi, 3.0),
        (FloatFn::CosPi, 3.0),
        (FloatFn::TanPi, 5.0),
    ] {
        let mut worst = (0f64, 0u64);
        for b in &xs {
            let x = f32v(*b);
            let got = metal_pi_model(op, x);
            if !x.is_finite() {
                assert_eq!(got.to_bits(), 0x7FC0_0000, "{op:?}({b:#010x})");
                continue;
            }
            if x != 0.0 && x.is_subnormal() {
                let zero = metal_pi_model(op, 0.0);
                assert_eq!(got.to_bits(), zero.to_bits(), "{op:?}({b:#010x})");
                continue;
            }
            let Some(w) = exact_pi(op, x) else {
                assert!(got.is_infinite(), "{op:?}({b:#010x}) = {got:e} at a pole");
                continue;
            };
            if w.abs() > f64::from(f32::MAX) {
                continue;
            }
            let e = (f64::from(got) - w).abs() / ulp32(w);
            if e > worst.0 {
                worst = (e, *b);
            }
        }
        eprintln!("{op:?}: worst {:.4} ulp at {:#010x}", worst.0, worst.1);
        assert!(
            worst.0 <= limit,
            "{op:?}: {:.4} ulp at {:#010x}",
            worst.0,
            worst.1
        );
    }
}

/// Metal's `sinpi`, `cospi`, and `tanpi` (precise and fast answer alike) at operands measured on
/// the M3 Max (the M4's, are the same), two in each of five ranges where Metal's word is not the
/// correctly rounded value: about `1/4`, about 1, the tiny operands, from 4, and the small negative
/// ones.
const PI_WORDS: [(FloatFn, [(u32, u32); 10]); 3] = [
    (
        FloatFn::SinPi,
        [
            (0x3E80_3003, 0x3F35_3A40),
            (0x3E80_4004, 0x3F35_4BFF),
            (0x3F80_5005, 0xBBFB_62E4),
            (0x3F80_A00A, 0xBC7B_60FF),
            (0x3380_0000, 0x3449_0FDA),
            (0x3380_B010, 0x344A_2469),
            (0x4080_B100, 0x3D8A_E8A3),
            (0x4084_2600, 0x3ECA_CE62),
            (0xBA00_6840, 0xBAC9_B396),
            (0xBA01_38C0, 0xBACA_FB19),
        ],
    ),
    (
        FloatFn::CosPi,
        [
            (0x3E80_2002, 0x3F34_E163),
            (0x3E80_4004, 0x3F34_BDCA),
            (0x3F80_2002, 0xBF7F_FFB0),
            (0x3F80_7007, 0xBF7F_FC39),
            (0x38A3_A790, 0x3F80_0000),
            (0x38A4_57A0, 0x3F80_0000),
            (0x4080_B100, 0x3F7F_6913),
            (0x4081_6200, 0x3F7D_A502),
            (0xBA02_D9C0, 0x3F7F_FFEC),
            (0xBA03_4200, 0x3F7F_FFEC),
        ],
    ),
    (
        FloatFn::TanPi,
        [
            (0x3E80_3003, 0x3F80_4B82),
            (0x3E80_5005, 0x3F80_7DEF),
            (0x3F80_2002, 0x3B49_1C96),
            (0x3F80_5005, 0x3BFB_64C9),
            (0x3380_0000, 0x3449_0FDA),
            (0x3380_B010, 0x344A_2469),
            (0x4082_1300, 0x3E53_7445),
            (0x4084_2600, 0x3EDC_DF5A),
            (0xBA01_A100, 0xBACB_9EEC),
            (0xBA02_0940, 0xBACC_42AC),
        ],
    ),
];

/// [`metal_pi_model`] is Metal's word at [`PI_WORDS`] and at the exact answers of
/// [`METAL_SINPI_COSPI`] and [`METAL_TANPI`] (the zeros, poles, units, quarter turns, large and
/// non-finite operands), and the routines of a precise and a fast call (Metal's fast functions
/// are its precise ones) answer them bit for bit, with and without `OpFmaKHR`.
#[test]
fn pi_routines_answer_metals_measured_words() {
    let f = f32s();
    for (op, of) in [
        (FloatFn::SinPi, Fun::SinPi as fn(Mode) -> Fun),
        (FloatFn::CosPi, Fun::CosPi),
        (FloatFn::TanPi, Fun::TanPi),
    ] {
        let mut words: Vec<(u32, u32)> = PI_WORDS
            .iter()
            .filter(|(o, _)| *o == op)
            .flat_map(|(_, w)| w.iter().copied())
            .collect();
        words.extend(match op {
            FloatFn::SinPi => METAL_SINPI_COSPI.iter().map(|r| (r.0, r.1)).collect(),
            FloatFn::CosPi => METAL_SINPI_COSPI.iter().map(|r| (r.0, r.2)).collect(),
            _ => METAL_TANPI.to_vec(),
        });
        for (x, want) in &words {
            let got = metal_pi_model(op, f32::from_bits(*x)).to_bits();
            assert_eq!(got, *want, "{op:?} model at {x:#010x}: {got:#010x}");
        }
        let cases: Vec<Case> = words
            .iter()
            .map(|(x, _)| vec![vec![u64::from(*x)]])
            .collect();
        for (fun, fma) in [Mode::Precise, Mode::Fast]
            .into_iter()
            .flat_map(|m| [(of(m), false), (of(m), true)])
        {
            check_module(
                &format!("{fun:?} fma={fma} at Metal's measured operands"),
                routine_module(fun, false, fma),
                &[f],
                f,
                &cases,
                &|c| {
                    let x = c[0][0] as u32;
                    words
                        .iter()
                        .find(|(m, _)| *m == x)
                        .map(|(_, w)| vec![u64::from(*w)])
                },
                &|g, w| g == w,
            );
        }
    }
}

/// `a 2^k` against one, for `a` below `2^127`.
fn against_one(a: u128, k: i32) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a, k) {
        (0, _) => Ordering::Less,
        (_, 0..) => {
            if a == 1 && k == 0 {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        }
        _ if -k >= 127 => Ordering::Less,
        _ => a.cmp(&(1u128 << -k)),
    }
}

/// A positive finite `float` as `(M, e)`, `M 2^e`, `M` its integer significand.
fn significand(x: f32) -> (u128, i32) {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32;
    let f = u128::from(b & 0x007F_FFFF);
    if e == 0 {
        (f, -149)
    } else {
        (f | (1 << 23), e - 150)
    }
}

/// The correctly rounded `rsqrt` of a positive finite `float` (a positive normal answer): the
/// `f64` estimate or a neighbour, whichever lies between the midpoints `m` around the exact
/// value `t`, decided exactly (`m < t` exactly when `m^2 x < 1`).
pub(super) fn rsqrt_exact(x: f32) -> f32 {
    use std::cmp::Ordering;
    let (mx, ex) = significand(x);
    let estimate = (1.0 / f64::from(x).sqrt()) as f32;
    let candidates = [
        estimate,
        f32::from_bits(estimate.to_bits() + 1),
        f32::from_bits(estimate.to_bits() - 1),
    ];
    for r in candidates {
        let (mr, er) = significand(r);
        let up = (2 * mr + 1) * (2 * mr + 1) * mx;
        let below_upper = against_one(up, 2 * er - 2 + ex) == Ordering::Greater;
        let (low, k) = if mr == 1 << 23 {
            ((4 * mr - 1) * (4 * mr - 1) * mx, 2 * er - 4 + ex)
        } else {
            ((2 * mr - 1) * (2 * mr - 1) * mx, 2 * er - 2 + ex)
        };
        let above_lower = against_one(low, k) == Ordering::Less;
        if below_upper && above_lower {
            return r;
        }
    }
    panic!("rsqrt({x:e}): no candidate is correctly rounded");
}

/// Metal's fast `rsqrt` of `y` as the model computes it and the routine forms it (`libm::rsqrt`):
/// `y = m 4^k` with `m` in `[1, 4)`, `RN(hi + RN(lo + eps(m) 2^-24))` of `hi = RN(1 / sqrt(m))`,
/// `lo = RN(RN(hi d) / 2)` of `d = 1 - m hi^2` from exact products, and `eps` by Horner's rule in
/// plain `float`, scaled by `2^-k`; `+inf` of a zero or denormal, `+0` of `+inf`, and a NaN of a
/// NaN. For operands at or above `+0`, as its callers pass.
pub(super) fn rsqrt_model(y: f32) -> f32 {
    let b = y.to_bits();
    let n = (b >> 23) & 0xFF;
    if y.is_nan() {
        return y;
    }
    if n == 0 {
        return f32::INFINITY;
    }
    if b == 0x7F80_0000 {
        return 0.0;
    }
    let k = b & 0x007F_FFFF;
    let binade = (n & 1) ^ 1;
    let m = f32::from_bits((0x3F80_0000 + (binade << 23)) | k);
    let hi = rsqrt_exact(m);
    let p = hi * hi;
    let pe = (f64::from(hi) * f64::from(hi) - f64::from(p)) as f32;
    let q = m * p;
    let qe = (f64::from(m) * f64::from(p) - f64::from(q)) as f32;
    let w = m * pe;
    let d = (1.0 - q) - (qe + w);
    let lo = (hi * d) * 0.5;
    let row = libm::rsqrt::TABLE[((k >> 18) + 32 * binade) as usize].map(f32::from_bits);
    let t = (k & 0x3_FFFF) as f32 / 262_144.0;
    let eps = row.iter().rev().fold(0.0f32, |acc, c| acc * t + c);
    let r = hi + (lo + eps / 16_777_216.0);
    let shift = ((127 + binade as i32 - n as i32) / 2) << 23;
    f32::from_bits(r.to_bits().wrapping_add(shift as u32))
}

/// Operands for the roots: the positive normal [`operands`] and the magnitudes of the negative
/// ones, the edges of the reduction's `[1, 4)`, and perfect squares.
fn root_operands() -> Vec<u64> {
    let mut v: Vec<u64> = operands(0x0C0F_FEE5, 4000)
        .into_iter()
        .map(|b| b & 0x7FFF_FFFF)
        .filter(|b| f32v(*b).is_normal())
        .collect();
    for x in [1.0f32, 2.0, 4.0, 0.5, 0.25] {
        for d in [-2i64, -1, 0, 1, 2] {
            v.push((i64::from(x.to_bits()) + d) as u64);
        }
    }
    for k in 1u32..4096 {
        let s = (k * k) as f32;
        v.push(u64::from(s.to_bits()));
        v.push(u64::from((s * 2f32.powi(-100)).to_bits()));
    }
    v.extend([0x0080_0000u64, 0x7F7F_FFFF, 0x0080_0001, 0x7F7F_FFFE]);
    v
}

/// Precise `sqrt` and `rsqrt` are correctly rounded (MSL Table 8.1) at every positive normal
/// operand checked, with the device's division and square root correctly rounded or moved 2 and 3
/// ulp.
#[test]
fn root_routines_are_correctly_rounded() {
    let xs: Vec<Case> = root_operands().iter().map(|b| vec![vec![*b]]).collect();
    let f = f32s();
    for rough in [false, true] {
        for (fun, exact) in [
            (Fun::Sqrt, f32::sqrt as fn(f32) -> f32),
            (Fun::Rsqrt, rsqrt_exact),
        ] {
            check_module(
                &format!("{fun:?} rough={rough}"),
                routine_module(fun, rough, false),
                &[f],
                f,
                &xs,
                &|c| Some(vec![u64::from(exact(f32v(c[0][0])).to_bits())]),
                &|g, w| g == w,
            );
        }
    }
}

/// Every other operand of the roots takes the device's `Sqrt` (and `rsqrt` its reciprocal): Metal's
/// answers at the zeros, infinities, and negative operands (case 087758cb), and a denormal as the
/// device reads it (the interpreter keeps it; how Metal reads one is the denormal model's).
#[test]
fn root_routines_answer_the_special_operands() {
    let f = f32s();
    let nan = 0x7FC0_0000u64;
    let denormal = |x: u32, rsqrt: bool| {
        let s = f32::from_bits(x).sqrt();
        u64::from(if rsqrt { 1.0 / s } else { s }.to_bits())
    };
    let cases: [(Fun, u64, u64); 18] = [
        (Fun::Sqrt, 0x0000_0000, 0x0000_0000),
        (Fun::Sqrt, 0x8000_0000, 0x8000_0000),
        (Fun::Sqrt, 0x7F80_0000, 0x7F80_0000),
        (Fun::Sqrt, 0xFF80_0000, nan),
        (Fun::Sqrt, 0xBF80_0000, nan),
        (Fun::Sqrt, 0x8080_0000, nan),
        (Fun::Sqrt, 0x7FC0_0000, nan),
        (Fun::Sqrt, 0x0000_0001, denormal(1, false)),
        (Fun::Sqrt, 0x007F_FFFF, denormal(0x007F_FFFF, false)),
        (Fun::Rsqrt, 0x0000_0000, 0x7F80_0000),
        (Fun::Rsqrt, 0x8000_0000, 0xFF80_0000),
        (Fun::Rsqrt, 0x7F80_0000, 0x0000_0000),
        (Fun::Rsqrt, 0xFF80_0000, nan),
        (Fun::Rsqrt, 0xBF80_0000, nan),
        (Fun::Rsqrt, 0x8080_0000, nan),
        (Fun::Rsqrt, 0x7FC0_0000, nan),
        (Fun::Rsqrt, 0x0000_0001, denormal(1, true)),
        (Fun::Rsqrt, 0x007F_FFFF, denormal(0x007F_FFFF, true)),
    ];
    for rough in [false, true] {
        for (fun, x, want) in cases {
            check_module(
                &format!("{fun:?}({x:#010x}) rough={rough}"),
                routine_module(fun, rough, false),
                &[f],
                f,
                &[vec![vec![x]]],
                &|_| Some(vec![want]),
                &|g, w| g == w || (f32v(g).is_nan() && f32v(w).is_nan()),
            );
        }
    }
}

/// Precise `fmod` is exact (MSL Table 8.1: 0 ulp), bit for bit against C's `fmod` (Rust's `%`) of
/// the operands as Metal reads them (a denormal divisor as a zero): operands of every exponent,
/// denormal dividends included, exponent gaps up to 253, equal and near-equal magnitudes, `fmod(x,
/// ±inf) = x`, and a NaN for an infinite or NaN `x`, a NaN `y`, or a zero or denormal `y`. A zero
/// remainder is the zero of `x`'s sign.
#[test]
fn fmod_routine_is_exact() {
    let mut pool: Vec<u64> = operands(0x0F0D_D1CE, 400)
        .into_iter()
        .filter(|b| f32v(*b).is_finite())
        .collect();
    let denormals = [
        0x0000_0001u64,
        0x0000_0003,
        0x0040_0000,
        0x007F_FFFF,
        0x0000_0100,
    ];
    for d in denormals {
        pool.push(d);
        pool.push(d | 0x8000_0000);
    }
    let mut rng = Rng(0x3E3E_F00D);
    let mut pairs: Vec<Vec<u64>> = Vec::new();
    for _ in 0..12000 {
        let x = pool[(rng.next() as usize) % pool.len()];
        let y = pool[(rng.next() as usize) % pool.len()];
        pairs.push(vec![x, y]);
    }
    // Near-equal magnitudes, small multiples, and the widest gaps.
    for _ in 0..2000 {
        let y = pool[(rng.next() as usize) % pool.len()];
        let k = (rng.next() % 9) as f32 - 4.0;
        let x = f32v(y) * k + f32v(y) * 2f32.powi(-((rng.next() % 30) as i32));
        if x.is_finite() {
            pairs.push(vec![u64::from(x.to_bits()), y]);
        }
        pairs.push(vec![y, y ^ 0x8000_0000]);
    }
    for (x, y) in [
        (0x7F7F_FFFFu64, 0x0000_0001u64),
        (0xFF7F_FFFF, 0x0000_0001),
        (0x7F7F_FFFF, 0x0080_0001),
        (0x7F7F_FFFF, 0x4040_0000),
        (0x4000_0000, 0x0000_0001),
        (0xCAC9_0FDB, 0x0000_0001),
        (0x4AC9_0FDC, 0x0080_0000),
        (0x3FC0_0000, 0xBF40_0000),
        (0x8000_0000, 0x4000_0000),
        (0xFF7F_FFFF, 0xBF80_0000),
    ] {
        pairs.push(vec![x, y]);
    }
    // The special operands.
    for x in [
        0x3F80_0000u64,
        0xBFC9_0FDB,
        0x0080_0000,
        0x8000_0001,
        0x0000_0000,
    ] {
        for y in [
            0x7F80_0000u64,
            0xFF80_0000,
            0x0000_0000,
            0x8000_0000,
            0x7FC0_0000,
        ] {
            pairs.push(vec![x, y]);
        }
    }
    for x in [0x7F80_0000u64, 0xFF80_0000, 0x7FC0_0000] {
        for y in [0x3F80_0000u64, 0x7F80_0000, 0x0000_0001] {
            pairs.push(vec![x, y]);
        }
    }
    let f = f32s();
    check_module(
        "Fmod",
        routine_module(Fun::Fmod, false, false),
        &[f, f],
        f,
        &pairs
            .iter()
            .map(|c| c.iter().map(|b| vec![*b]).collect())
            .collect::<Vec<Case>>(),
        &|c| {
            let y = c[1][0];
            let y = if y & 0x7F80_0000 == 0 {
                y & 0x8000_0000
            } else {
                y
            };
            Some(vec![u64::from((f32v(c[0][0]) % f32v(y)).to_bits())])
        },
        &|g, w| g == w || (f32v(g).is_nan() && f32v(w).is_nan()),
    );
}

/// Metal's precise `fmod` reads a denormal dividend as itself and a denormal divisor as a zero
/// (case 087758cb, every such lane): `fmod(2, 2^-149)` and `fmod(-6588397.5, 2^-149)` are NaN,
/// `fmod(±2^-149, y)` is the dividend, and so is `fmod(2^-127, 8)`; on every device, since the
/// routine decides on the operands' bits.
#[test]
fn fmod_routine_reads_metals_denormal_operands() {
    let f = f32s();
    let nan = 0x7FC0_0000u64;
    for (x, y, want) in [
        (0x4000_0000u64, 0x0000_0001u64, nan),
        (0xCAC9_0FDB, 0x0000_0001, nan),
        (0x3F80_0000, 0x807F_FFFF, nan),
        (0x0000_0003, 0x0000_0002, nan),
        (0x0000_0000, 0x0040_0000, nan),
        (0x0000_0001, 0xBF80_0000, 0x0000_0001),
        (0x8000_0001, 0x3F00_0000, 0x8000_0001),
        (0x0040_0000, 0x4100_0000, 0x0040_0000),
        (0x007F_FFFF, 0x0080_0000, 0x007F_FFFF),
    ] {
        check_module(
            &format!("Fmod({x:#010x}, {y:#010x})"),
            routine_module(Fun::Fmod, false, false),
            &[f, f],
            f,
            &[vec![vec![x], vec![y]]],
            &|_| Some(vec![want]),
            &|g, w| g == w || (f32v(g).is_nan() && f32v(w).is_nan()),
        );
    }
}

/// A `half` call of `sqrt`, `rsqrt`, and `fmod` takes the `float` routine before its rounding
/// (Metal's `half` functions are the same in both modes), so it is Metal's `half` function (MSL
/// Table 8.3: correctly rounded, correctly rounded, and exact) over every pattern of the roots and
/// a grid of `fmod`'s pairs: the `float` roots' rounding to `half` is exact (24 bits hold twice
/// `half`'s 11 and two more for `sqrt`; checked exhaustively for `rsqrt`), and the remainder of two
/// `half` values is one. The half reading: `fmod(x, ±inf) = x` (where lavapipe, RADV, and NVIDIA
/// answer a NaN) and `fmod(1.5, -0.75) = 0` (RADV: 0.75). A zero remainder is Metal's `half` one:
/// `+0` for a nonzero dividend, the dividend itself for a zero one. Part `part` of `of` of the runs
/// of 4,096 operands.
fn half_roots_and_fmod(part: usize, of: usize) {
    use std::collections::BTreeMap;
    let h = float(16);
    let nan = |b: u64| b & 0x7C00 == 0x7C00 && b & 0x03FF != 0;
    let module = |op: FloatFn, fun: Fun| {
        kernel_defs(
            &vec![h; fun.arity()],
            h,
            &|b| Ok(vec![libm::define_alone(b, fun, false)?]),
            &|fb, x, funcs| {
                let map: BTreeMap<Fun, FuncId> = funcs.iter().map(|id| (fun, *id)).collect();
                math::float_fn(
                    fb,
                    op,
                    h,
                    Contraction::Forbidden,
                    NanTest::IsNan,
                    x,
                    &Lib::new(None, &map),
                )
            },
            true,
            false,
        )
        .unwrap_or_else(|e| panic!("{op:?}: build: {e:?}"))
    };
    let all: Vec<Case> = (0..=0xFFFFu64)
        .filter(|b| !nan(*b))
        .map(|b| vec![vec![b]])
        .collect();
    let half_of = |x: f32| u64::from(f64_to_f16(f64::from(x)));
    let mut pairs: Vec<Case> = Vec::new();
    for x in (0..=0xFFFFu64).step_by(251) {
        for y in (3..=0xFFFFu64).step_by(241) {
            pairs.push(vec![vec![x], vec![y]]);
        }
    }
    for (x, y) in [
        (0x3E00u64, 0xBA00u64),
        (0x3C00, 0x7C00),
        (0xBC00, 0xFC00),
        (0x7BFF, 0x7C00),
        (0x0001, 0xFC00),
    ] {
        pairs.push(vec![vec![x], vec![y]]);
    }
    let mut runs = Vec::new();
    for (op, fun) in [
        (FloatFn::Sqrt, Fun::Sqrt),
        (FloatFn::Rsqrt, Fun::Rsqrt),
        (FloatFn::Fmod, Fun::Fmod),
    ] {
        let cases = if op == FloatFn::Fmod { &pairs } else { &all };
        for run in cases.chunks(4096) {
            runs.push((op, fun, run));
        }
    }
    for (op, fun, run) in part_of(runs, part, of) {
        check_module(
            &format!("half {op:?}"),
            module(op, fun),
            &vec![h; fun.arity()],
            h,
            run,
            &|c| {
                let x = f16_to_f64(c[0][0] as u16);
                let r = match op {
                    FloatFn::Fmod => {
                        let y = f16_to_f64(c[1][0] as u16);
                        let r = (x as f32) % (y as f32);
                        if r == 0.0 && x != 0.0 {
                            0.0
                        } else {
                            r
                        }
                    }
                    FloatFn::Sqrt => (x as f32).sqrt(),
                    _ if x > 0.0 && x.is_finite() => rsqrt_exact(x as f32),
                    _ => 1.0 / (x as f32).sqrt(),
                };
                Some(vec![half_of(r)])
            },
            &|g, w| g == w || (nan(g) && nan(w)),
        );
    }
}

mod half_roots_and_fmod_take_the_routines {
    parts!(super::half_roots_and_fmod, 8;
        part_1_of_8 = 0, part_2_of_8 = 1, part_3_of_8 = 2, part_4_of_8 = 3,
        part_5_of_8 = 4, part_6_of_8 = 5, part_7_of_8 = 6, part_8_of_8 = 7);
}

/// The [`trig_operands`] from `float(π)` up in magnitude, past `π`, where Metal's fast `sin` and
/// `cos` take the precise routine (inside, the device's instruction answers, which the
/// interpreter does not model).
fn past_pi_operands() -> Vec<u64> {
    let mut v = trig_operands();
    assert!(v.contains(&u64::from(std::f32::consts::PI.to_bits())));
    v.retain(|b| f32v(*b).abs() >= std::f32::consts::PI);
    v
}

/// Metal's fast `sin` and `cos` past `π` (where Apple states no bound and RADV, ANV, and NVIDIA are
/// up to 1.8 off) are its precise functions, which the routines compute under the branch:
/// [`metal_sin_cos_model`] bit for bit. Part `part` of `of` checks every `of`-th operand.
fn fast_trig_past_pi(part: usize, of: usize) {
    let xs = part_of(past_pi_operands(), part, of);
    let f = f32s();
    for (fun, cos) in [(Fun::FastSin, false), (Fun::FastCos, true)] {
        check_module(
            &format!("{fun:?} past π"),
            routine_module(fun, false, false),
            &[f],
            f,
            &xs.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
            &|c| {
                Some(vec![u64::from(
                    metal_sin_cos_model(f32v(c[0][0]), cos).to_bits(),
                )])
            },
            &|g, w| g == w,
        );
    }
}

mod fast_trig_routines_take_the_precise_routine_past_pi {
    parts!(super::fast_trig_past_pi, 6; part_1_of_6 = 0, part_2_of_6 = 1,
        part_3_of_6 = 2, part_4_of_6 = 3, part_5_of_6 = 4, part_6_of_6 = 5);
}

/// `fast::tan` off a device whose functions are Metal's in a fast-math module is Apple's formula
/// `sin(x) · (1 / cos(x))` (MSL Table 8.2) of Metal's fast functions divided by its fast division's
/// routine, by either reciprocal: past `π` the formula of Metal's functions
/// ([`metal_sin_cos_model`]) bit for bit, the reciprocal and the product each rounded once, and
/// past Metal's flush point, where both flush to `+0`, the canonical NaN `0 · (1 / 0)` that Metal
/// answers at 148 of the 160 such lanes of case cc234732. Part `part` of `of` checks every `of`-th
/// operand past `π`.
fn fast_tan_formula(part: usize, of: usize) {
    use std::collections::BTreeMap;
    let f = f32s();
    let module = |reciprocal: Reciprocal| {
        let division = math::FastDivision::new(f, reciprocal, Contraction::Forbidden);
        kernel_defs(
            &[f],
            f,
            &|b| {
                Ok(vec![
                    libm::define_alone(b, Fun::FastSin, false)?,
                    libm::define_alone(b, Fun::FastCos, false)?,
                    math::define_fast_division(b, division)?,
                ])
            },
            &|fb, x, funcs| {
                let map: BTreeMap<Fun, FuncId> = [Fun::FastSin, Fun::FastCos]
                    .into_iter()
                    .zip(funcs.iter().copied())
                    .collect();
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    math::call_fast_division(fb, funcs[2], p, q)
                };
                math::fast_tan(fb, f, x[0], &Lib::new(None, &map), &divide)
            },
            true,
            false,
        )
        .unwrap_or_else(|e| panic!("FastTan {reciprocal:?}: build: {e:?}"))
    };
    let limit = f64::from(math::FAST_TRIG_LIMIT);
    let (below, past): (Vec<u64>, Vec<u64>) = part_of(past_pi_operands(), part, of)
        .into_iter()
        .partition(|b| f64::from(f32v(*b)).abs() <= limit);
    for reciprocal in [Reciprocal::Device, Reciprocal::Rounded] {
        check_module(
            "FastTan past π",
            module(reciprocal),
            &[f],
            f,
            &below.iter().map(|b| vec![vec![*b]]).collect::<Vec<Case>>(),
            &|c| {
                let x = f32v(c[0][0]);
                let (s, c) = (metal_sin_cos_model(x, false), metal_sin_cos_model(x, true));
                Some(vec![u64::from((s * (1.0 / c)).to_bits())])
            },
            &|g, w| g == w,
        );
        check_module(
            "FastTan of a denormal",
            module(reciprocal),
            &[f],
            f,
            &DENORMALS
                .iter()
                .map(|b| vec![vec![*b]])
                .collect::<Vec<Case>>(),
            &|c| Some(vec![c[0][0]]),
            &|g, w| g == w,
        );
        check_module(
            "FastTan past the flush point",
            module(reciprocal),
            &[f],
            f,
            &past.iter().map(|b| vec![vec![*b]]).collect::<Vec<Case>>(),
            &|_| Some(vec![0x7FC0_0000]),
            &|g, w| g == w,
        );
    }
}

mod fast_tan_is_apples_formula_of_the_fast_functions {
    parts!(super::fast_tan_formula, 8; part_1_of_8 = 0, part_2_of_8 = 1, part_3_of_8 = 2,
        part_4_of_8 = 3, part_5_of_8 = 4, part_6_of_8 = 5, part_7_of_8 = 6, part_8_of_8 = 7);
}

/// `f32` denormals of both signs: the smallest, the largest, and between.
pub(super) const DENORMALS: [u64; 8] = [
    0x0000_0001,
    0x8000_0001,
    0x007F_FFFF,
    0x807F_FFFF,
    0x0040_0000,
    0x8040_0000,
    0x0000_1234,
    0x8012_3456,
];

/// Metal keeps a denormal operand of `fast::sin` and `fast::cos` (case cc234732, every such lane:
/// `fast::sin(x) = x`, `fast::cos(x) = 1`), where RADV, ANV, and NVIDIA flush it: the routines send
/// a denormal lane, told by its bits, to the precise routine, whose selects answer it (the device's
/// instruction, which the interpreter does not model, is not reached).
#[test]
fn fast_trig_routines_keep_a_denormal() {
    let f = f32s();
    for rough in [false, true] {
        for (fun, cos) in [(Fun::FastSin, false), (Fun::FastCos, true)] {
            check_module(
                &format!("{fun:?} of a denormal rough={rough}"),
                routine_module(fun, rough, false),
                &[f],
                f,
                &DENORMALS
                    .iter()
                    .map(|b| vec![vec![*b]])
                    .collect::<Vec<Case>>(),
                &|c| Some(vec![if cos { 0x3F80_0000 } else { c[0][0] }]),
                &|g, w| g == w,
            );
        }
    }
}

/// The device's divisions and square roots in `m`: `OpFDiv`, and `GLSL.std.450`'s `Sqrt` (31)
/// and `InverseSqrt` (32), the only extended instruction set the routines import.
fn device_divisions_and_roots(m: TypedModule) -> usize {
    let words = serialize(m).expect("serialize").words;
    let mut at = 5;
    let mut n = 0;
    while at < words.len() {
        let w = words[at];
        let op = w & 0xFFFF;
        let root =
            op == crate::spv::Op::ExtInst as u32 && matches!(words.get(at + 4), Some(31 | 32));
        n += usize::from(op == crate::spv::Op::FDiv as u32 || root);
        at += (w >> 16).max(1) as usize;
    }
    n
}

/// No routine's answer depends on the device's division or square root, which Vulkan bounds at 2.5
/// and 3 ulp: no routine divides or takes a square root but through the module's correctly rounded
/// routines ([`Fun::needs`]), with and without `OpFmaKHR`. Only those of `sqrt` and `rsqrt` take
/// the device's, which they correct exactly (`root_routines_are_correctly_rounded`, with the
/// device's moved), as the precise division routine does its quotient
/// (`precise_division_answers_metal_on_both_paths`). So a routine's answers on the interpreter,
/// whose division and square root are correctly rounded, are its answers on every device.
#[test]
fn routines_divide_and_take_roots_only_through_their_helpers() {
    let mut wrong = Vec::new();
    for fun in [
        Fun::Atan(Mode::Precise),
        Fun::Atan(Mode::Fast),
        Fun::Asin(Mode::Precise),
        Fun::Asin(Mode::Fast),
        Fun::Acos(Mode::Precise),
        Fun::Acos(Mode::Fast),
        Fun::Atan2,
        Fun::Sin,
        Fun::Cos,
        Fun::Tan,
        Fun::FastSin,
        Fun::FastCos,
        Fun::SinPi(Mode::Precise),
        Fun::CosPi(Mode::Precise),
        Fun::TanPi(Mode::Precise),
        Fun::SinPi(Mode::Fast),
        Fun::CosPi(Mode::Fast),
        Fun::TanPi(Mode::Fast),
        Fun::Fmod,
        Fun::Exp2,
        Fun::Exp,
        Fun::Exp10,
        Fun::Log2,
        Fun::Log,
        Fun::Log10,
        Fun::Powr,
        Fun::Sinh,
        Fun::Cosh,
        Fun::Tanh,
        Fun::Asinh,
        Fun::Acosh,
        Fun::Atanh,
        Fun::FastAsinh,
        Fun::FastAcosh,
        Fun::Recip,
        Fun::FastRsqrt,
        Fun::Fused,
        Fun::HalfAtan,
        Fun::HalfAtan2,
        Fun::HalfTan,
    ] {
        for fma in [false, true] {
            let own = device_divisions_and_roots(routine_module(fun, false, fma))
                - device_divisions_and_roots(helpers_module(fun, fma));
            if own != 0 {
                wrong.push(format!("{fun:?} fma={fma}: {own}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Each routine calls exactly the helpers [`Fun::needs`] lists: it is defined with them, and not
/// without any of them.
#[test]
fn each_routine_needs_what_it_calls() {
    let mut wrong = Vec::new();
    for fun in [
        Fun::Atan(Mode::Precise),
        Fun::Atan(Mode::Fast),
        Fun::Asin(Mode::Precise),
        Fun::Asin(Mode::Fast),
        Fun::Acos(Mode::Precise),
        Fun::Acos(Mode::Fast),
        Fun::Atan2,
        Fun::Sin,
        Fun::Cos,
        Fun::Tan,
        Fun::FastSin,
        Fun::FastCos,
        Fun::SinPi(Mode::Precise),
        Fun::CosPi(Mode::Precise),
        Fun::TanPi(Mode::Precise),
        Fun::SinPi(Mode::Fast),
        Fun::CosPi(Mode::Fast),
        Fun::TanPi(Mode::Fast),
        Fun::Sqrt,
        Fun::Rsqrt,
        Fun::Fmod,
        Fun::Exp2,
        Fun::Exp,
        Fun::Exp10,
        Fun::Log2,
        Fun::Log,
        Fun::Log10,
        Fun::Powr,
        Fun::Sinh,
        Fun::Cosh,
        Fun::Tanh,
        Fun::Asinh,
        Fun::Acosh,
        Fun::Atanh,
        Fun::FastAsinh,
        Fun::FastAcosh,
        Fun::Recip,
        Fun::FastRsqrt,
        Fun::Fused,
        Fun::HalfAtan,
        Fun::HalfAtan2,
        Fun::HalfTan,
        Fun::HalfSin,
        Fun::HalfCos,
        Fun::HalfAsinh,
    ] {
        let needs = fun.needs();
        let mut b = ModuleBuilder::new();
        let helpers = libm::define_helpers(&mut b, needs, false, false).expect("helpers");
        if let Err(e) = libm::define(&mut b, fun, false, helpers) {
            wrong.push(format!("{fun:?} with its helpers: {e:?}"));
        }
        for (missing, without) in [
            (
                needs.division,
                libm::Helpers {
                    division: None,
                    ..helpers
                },
            ),
            (
                needs.exp2,
                libm::Helpers {
                    exp2: None,
                    ..helpers
                },
            ),
            (
                needs.log2,
                libm::Helpers {
                    log2: None,
                    ..helpers
                },
            ),
            (
                needs.rcp,
                libm::Helpers {
                    rcp: None,
                    ..helpers
                },
            ),
            (
                needs.recip,
                libm::Helpers {
                    recip: None,
                    ..helpers
                },
            ),
            (
                needs.inverse_root,
                libm::Helpers {
                    inverse_root: None,
                    ..helpers
                },
            ),
            (
                needs.rsqrt,
                libm::Helpers {
                    rsqrt: None,
                    ..helpers
                },
            ),
            (
                needs.fast_inverse_root,
                libm::Helpers {
                    fast_inverse_root: None,
                    ..helpers
                },
            ),
            (
                needs.fused,
                libm::Helpers {
                    fused: None,
                    ..helpers
                },
            ),
            (
                needs.quarter,
                libm::Helpers {
                    quarter: None,
                    ..helpers
                },
            ),
            (
                needs.sinc,
                libm::Helpers {
                    sinc: None,
                    ..helpers
                },
            ),
            (
                needs.powr,
                libm::Helpers {
                    powr: None,
                    ..helpers
                },
            ),
        ] {
            if missing && libm::define(&mut b, fun, false, without).is_ok() {
                wrong.push(format!("{fun:?} is defined without a helper it lists"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
