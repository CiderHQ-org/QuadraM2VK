//! The second stage's routines on the interpreter: the exponentials, the logarithms, `powr`, and
//! the hyperbolic functions and their inverses, each Metal's measured form where it is identified
//! (bit for bit against its model) and otherwise within an ulp of the exact value over its range
//! with the device's division and square root exact and moved to the edge of what the routine
//! corrects; Metal's answers at the special operands, bit for bit; and the fast functions' formulas
//! at theirs.

use super::libm::{
    f32s, helpers_module, operands, routine_module, rsqrt_model, unary, worst_error, DENORMALS,
};
use super::*;
use crate::lower::arith::libm::rcp::model as rcp_model;
use crate::lower::arith::libm::{self, Fun};
use crate::spirv::serialize::serialize;
use std::collections::BTreeMap;

/// A float uniformly in `[lo, hi]`.
fn uniform(rng: &mut Rng, lo: f64, hi: f64) -> u64 {
    let t = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
    u64::from(((lo + (hi - lo) * t) as f32).to_bits())
}

/// `n` floats of magnitude log-uniform in `[2^lo, 2^hi]`, both signs.
fn magnitudes(rng: &mut Rng, lo: f64, hi: f64, n: usize) -> Vec<u64> {
    (0..n)
        .map(|_| {
            let b = uniform(rng, lo, hi);
            let x = f64::from(f32v(b)).exp2() as f32;
            u64::from(x.to_bits()) | (rng.next() & 0x8000_0000)
        })
        .collect()
}

/// `x` and its neighbours `d` ulps away, `d` in `-k..=k`.
fn around(x: f32, k: i64) -> Vec<u64> {
    (-k..=k)
        .map(|d| (i64::from(x.to_bits()) + d) as u64)
        .collect()
}

/// Operands of the exponentials and the hyperbolic functions: [`operands`], uniform ones across
/// the range where the results are finite and normal, small magnitudes, and the neighbours of the
/// reduction's edges (the integers and half-integers).
fn exp_operands(seed: u64) -> Vec<u64> {
    let mut rng = Rng(seed);
    let mut v = operands(seed, 1500);
    for (lo, hi, n) in [
        (-150.0, 130.0, 2500),
        (-40.0, 40.0, 1500),
        (-1.0, 1.0, 1000),
    ] {
        v.extend((0..n).map(|_| uniform(&mut rng, lo, hi)));
    }
    v.extend(magnitudes(&mut rng, -30.0, 3.0, 1500));
    for k in -30..=30 {
        for x in [k as f32, k as f32 + 0.5, k as f32 * std::f32::consts::LN_2] {
            v.extend(around(x, 2));
        }
    }
    v.retain(|b| f32v(*b).is_finite() && !is_denormal(*b));
    v
}

/// Positive normal operands of the logarithms: [`operands`], the floats near 1 (where the
/// result cancels most), random ones, and the neighbours of `√2` and `√2 / 2`, where the reduction
/// changes exponent.
fn log_operands(seed: u64) -> Vec<u64> {
    let mut rng = Rng(seed);
    let mut v = operands(seed, 2000);
    v.extend(around(1.0, 1500));
    v.extend((0..1500).map(|_| uniform(&mut rng, 0.5, 2.0)));
    v.extend(magnitudes(&mut rng, -126.0, 127.9, 2000));
    v.extend(around(f32::from_bits(0x3FB5_04F3), 4));
    v.extend(around(f32::from_bits(0x3F35_04F3), 4));
    v.retain(|b| {
        let x = f32v(*b);
        x.is_normal() && x > 0.0
    });
    v
}

/// Whether float bits `b` are an `f32` denormal.
fn is_denormal(b: u64) -> bool {
    b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0
}

/// `r` where it is a finite normal `float` value away from the edges of the range, where the
/// result's rounding to infinity or its flush below `2^-126` is checked by the special operands.
fn in_range(r: f64) -> Option<f64> {
    let a = r.abs();
    let normal = a == 0.0 || a >= 1.001 * f64::from(f32::MIN_POSITIVE);
    (r.is_finite() && a <= 0.999 * f64::from(f32::MAX) && normal).then_some(r)
}

/// Metal's `log2` of positive normal float bits `x` as the model computes it: `RZ(log2(x) + eps)`,
/// `eps` the table's ([`libm::log::TABLE`]) quintic at the fraction's 32nd by Horner's rule in
/// `float`, 0 in `[PLAIN_LOW, PLAIN_END)`. `None` where the sum lies within `2^-38` of its
/// magnitude of a `float` (the routine's exact value is a pair accurate to about `2^-41`, the
/// model's a double).
fn log2_model(x: u32) -> Option<u32> {
    use libm::log::{PLAIN_END, PLAIN_LOW, TABLE};
    let v = f32::from_bits(x);
    if !(v.is_normal() && v > 0.0) {
        return None;
    }
    let k = x & 0x7F_FFFF;
    let eps = if (PLAIN_LOW..PLAIN_END).contains(&x) {
        0.0
    } else {
        let c = TABLE[(k >> 18) as usize].map(f32::from_bits);
        let t = (k & 0x3_FFFF) as f32 / 262_144.0;
        let mut e = c[5];
        for j in (0..5).rev() {
            e = e * t + c[j];
        }
        f64::from(e)
    };
    let w = f64::from(v).log2() + eps;
    let r = w as f32;
    let r = if f64::from(r).abs() > w.abs() {
        f32::from_bits(r.to_bits() - 1)
    } else {
        r
    };
    let away = f64::from(f32::from_bits(r.to_bits() + 1));
    let tol = w.abs() * 2f64.powi(-38);
    let near = (w - f64::from(r)).abs() < tol || (away - w).abs() < tol;
    (!near).then_some(r.to_bits())
}

/// Metal's precise `log` or `log10` of positive normal float bits `x` as the model computes it:
/// inside the [`libm::log::Near`] range `RN(f fma(p(f), f, lead))`, elsewhere [`log2_model`]'s
/// fitted word times `RN(ln 2)` or `RN(log10(2))`; `log2`'s routines' [`log2_model`].
fn log_model(fun: Fun, x: u32) -> Option<u32> {
    use libm::log::{LOG10_NEAR, LOG_NEAR};
    let (near, c) = match fun {
        Fun::Log => (&LOG_NEAR, std::f32::consts::LN_2),
        Fun::Log10 => (&LOG10_NEAR, std::f32::consts::LOG10_2),
        _ => return log2_model(x),
    };
    if near.low < x && x < near.end {
        let f = f32::from_bits(x) - 1.0;
        let (last, rest) = near.p.split_last()?;
        let mut p = f32::from_bits(*last);
        for c in rest.iter().rev() {
            p = p.mul_add(f, f32::from_bits(*c));
        }
        let q = p.mul_add(f, f32::from_bits(near.lead));
        return Some((f * q).to_bits());
    }
    log2_model(x).map(|l| (f32::from_bits(l) * c).to_bits())
}

/// The `log2`, `log`, and `log10` routines answer [`log_model`] bit for bit over the logarithms'
/// operands, the edges of the plain and near ranges and of the table's 32nds, and uniform operands
/// next to 1 and in `[1/2, 4)`, and stay within 3 ulp of the exact value: part `part` of `of` of
/// the runs of 2,048 operands.
fn logarithms_are_metals(part: usize, of: usize) {
    use libm::log::{LOG10_NEAR, LOG_NEAR, PLAIN_END, PLAIN_LOW};
    let mut rng = Rng(0x1065_1066);
    let mut xs = log_operands(0x1065_1065);
    for edge in [
        PLAIN_LOW,
        PLAIN_END,
        LOG_NEAR.low,
        LOG_NEAR.end,
        LOG10_NEAR.low,
        LOG10_NEAR.end,
    ] {
        xs.extend(around(f32::from_bits(edge), 3));
    }
    for s in 0..32u32 {
        for e in [0x3F00_0000u32, 0x3F80_0000, 0x4000_0000, 0x7F00_0000] {
            xs.extend(around(f32::from_bits(e | (s << 18)), 2));
        }
    }
    for (lo, hi) in [(0.95, 1.07), (0.5, 4.0)] {
        xs.extend((0..2000).map(|_| uniform(&mut rng, lo, hi)));
    }
    xs.retain(|b| {
        let x = f32v(*b);
        x.is_normal() && x > 0.0
    });
    let mut runs = Vec::new();
    for fun in [Fun::Log2, Fun::Log, Fun::Log10] {
        let modelled: Vec<u64> = xs
            .iter()
            .copied()
            .filter(|b| log_model(fun, *b as u32).is_some())
            .collect();
        for run in modelled.chunks(2048) {
            runs.push((fun, run.to_vec()));
        }
    }
    let f = f32s();
    for (fun, run) in part_of(runs, part, of) {
        check_module(
            &format!("{fun:?}"),
            routine_module(fun, false, false),
            &[f],
            f,
            &run.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
            &|c| log_model(fun, c[0][0] as u32).map(|w| vec![u64::from(w)]),
            &|g, w| g == w,
        );
        let exact: fn(f64) -> f64 = match fun {
            Fun::Log2 => f64::log2,
            Fun::Log => f64::ln,
            _ => f64::log10,
        };
        let w = worst_error(
            fun,
            false,
            &unary(&run),
            &|x| in_range(exact(f64::from(x[0]))),
            3.0,
        );
        eprintln!("{fun:?}, {} operands: {w:.3}", run.len());
    }
}

mod logarithm_routines_are_metals {
    parts!(super::logarithms_are_metals, 6;
        part_1_of_6 = 0, part_2_of_6 = 1, part_3_of_6 = 2, part_4_of_6 = 3, part_5_of_6 = 4,
        part_6_of_6 = 5);
}

/// The routines answer Metal's precise `log2`, `log`, and `log10` at measured words of the M4: the
/// far forms, the near polynomials and their edges, and the plain range's
/// first operand.
#[test]
fn logarithm_routines_answer_metals_measured_words() {
    let words: [(Fun, u64, u64); 43] = [
        (Fun::Log2, 0x3F7F_0000, 0xBBB9_06CE),
        (Fun::Log2, 0x4000_0001, 0x3F80_0001),
        (Fun::Log2, 0x3FB5_04F3, 0x3EFF_FFFF),
        (Fun::Log2, 0x3F35_04F3, 0xBF00_0000),
        (Fun::Log2, 0x0080_0001, 0xC2FB_FFFF),
        (Fun::Log2, 0x7F7F_FFFF, 0x42FF_FFFF),
        (Fun::Log2, 0x3F00_0001, 0xBF7F_FFFD),
        (Fun::Log2, 0x4B00_0001, 0x41B8_0000),
        (Fun::Log2, 0x3FC9_0FDB, 0x3F26_C874),
        (Fun::Log2, 0x4049_0FDB, 0x3FD3_643A),
        (Fun::Log2, 0x3E4C_CCCD, 0xC014_9A78),
        (Fun::Log2, 0x42C8_0000, 0x40D4_9A78),
        (Fun::Log2, 0x3F9D_70A4, 0x3E98_E9BE),
        (Fun::Log2, 0x3F60_0000, 0xBE45_44BF),
        (Fun::Log, 0x3F74_0001, 0xBD44_A540),
        (Fun::Log, 0x3F87_FFFF, 0x3D78_5168),
        (Fun::Log, 0x3F80_0001, 0x33FF_FFFF),
        (Fun::Log, 0x3F7F_FFFF, 0xB380_0000),
        (Fun::Log, 0x3F74_0000, 0xBD44_A54F),
        (Fun::Log, 0x3F7A_0000, 0xBCC2_492A),
        (Fun::Log, 0x3F85_0000, 0x3D1C_F43E),
        (Fun::Log, 0x4000_0001, 0x3F31_7219),
        (Fun::Log, 0x3FB5_04F3, 0x3EB1_7217),
        (Fun::Log, 0x4049_0FDB, 0x3F92_8683),
        (Fun::Log, 0x3E4C_CCCD, 0xBFCE_020F),
        (Fun::Log, 0x42C8_0000, 0x4093_5D8E),
        (Fun::Log, 0x3F9D_70A4, 0x3E53_FB86),
        (Fun::Log, 0x3F60_0000, 0xBE08_BC73),
        (Fun::Log10, 0x3F77_8001, 0xBC70_445A),
        (Fun::Log10, 0x3F87_BFFF, 0x3CD1_23E8),
        (Fun::Log10, 0x3F80_0001, 0x335E_5BD7),
        (Fun::Log10, 0x3F7F_FFFF, 0xB2DE_5BD8),
        (Fun::Log10, 0x3F77_8000, 0xBC70_4479),
        (Fun::Log10, 0x3F87_C000, 0x3CD1_2402),
        (Fun::Log10, 0x3F7A_0000, 0xBC28_C128),
        (Fun::Log10, 0x3F85_0000, 0x3C88_541E),
        (Fun::Log10, 0x4000_0001, 0x3E9A_209C),
        (Fun::Log10, 0x3FB5_04F3, 0x3E1A_209A),
        (Fun::Log10, 0x4049_0FDB, 0x3EFE_8A6F),
        (Fun::Log10, 0x3E4C_CCCD, 0xBF32_EFB3),
        (Fun::Log10, 0x42C8_0000, 0x4000_0000),
        (Fun::Log10, 0x3F9D_70A4, 0x3DB8_202C),
        (Fun::Log10, 0x3F60_0000, 0xBD6D_88F6),
    ];
    let f = f32s();
    for fun in [Fun::Log2, Fun::Log, Fun::Log10] {
        let cases: Vec<(u64, u64)> = words
            .iter()
            .filter(|(g, _, _)| *g == fun)
            .map(|(_, x, w)| (*x, *w))
            .collect();
        check_module(
            &format!("{fun:?} at Metal's words"),
            routine_module(fun, false, false),
            &[f],
            f,
            &cases
                .iter()
                .map(|(x, _)| vec![vec![*x]])
                .collect::<Vec<Case>>(),
            &|c| {
                let x = c[0][0];
                cases.iter().find(|(b, _)| *b == x).map(|(_, w)| vec![*w])
            },
            &|g, w| g == w,
        );
    }
}

/// Metal's `exp2` of float bits `x` as the model computes it ([`libm::exp::TABLE`]): an operand
/// below `2^-23` in magnitude reads as 0, a NaN gives `0x7fc00000`, and otherwise `2^n (1 + T(f)
/// 2^-23)` of `k = floor(x 2^23)`, `n = k >> 23`, `f = k mod 2^23`, `+0` where the exponent field
/// is 0 or below and `+inf` from 255.
fn exp2_model(x: u32) -> u32 {
    let v = f32::from_bits(x);
    if v.is_nan() {
        return 0x7FC0_0000;
    }
    let v = if v.abs() < f32::EPSILON {
        0.0
    } else {
        v.clamp(-200.0, 200.0)
    };
    let k = (f64::from(v) * 8_388_608.0).floor() as i64;
    let (n, f) = (k >> 23, (k & 0x7F_FFFF) as u64);
    let [c0, c1, c2, c3] = libm::exp::TABLE[(f >> 18) as usize].map(u64::from);
    let t = f & 0x3_FFFF;
    let u2 = c2 + ((c3 * t) >> 18);
    let u1 = c1 + ((u2 * t) >> 18);
    let y = c0 + ((u1 * t) >> 18);
    assert!(y < 1 << 32, "{x:#010x}: Y {y:#x}");
    match n + 127 {
        e if e <= 0 => 0,
        e if e >= 255 => 0x7F80_0000,
        e => ((e as u32) << 23) | (y >> 9) as u32,
    }
}

/// Metal's precise `exp` (`c` is `log2(e)`) or `exp10` (`log2(10)`) of float bits `x`, as it
/// composes its `exp2` of the model's ([`exp2_model`]): `fma(E, RN(d ln 2), E)` of `E = exp2(p)`,
/// `p = RN(x c_hi)`, and `d = fma(x, c_lo, fma(x, c_hi, -p))`, `E` itself where it is `+inf`, `+0`,
/// or a NaN; a denormal result is `+0`.
fn exp_model(x: u32, c: f64) -> u32 {
    exp_composed(x, c, false)
}

/// [`exp_model`], and with `halved` its composition of `E = exp2(p - 1)`: `e^x / 2` as Metal's
/// `sinh` and `cosh` form it from 1.5 on.
fn exp_composed(x: u32, c: f64, halved: bool) -> u32 {
    let x = f32::from_bits(x);
    let (hi, lo) = libm::split(c);
    let p = x * hi;
    let d = x.mul_add(lo, x.mul_add(hi, -p));
    let p = if halved { p - 1.0 } else { p };
    let big = f32::from_bits(exp2_model(p.to_bits()));
    if big.is_infinite() || big == 0.0 || big.is_nan() {
        return big.to_bits();
    }
    let r = big.mul_add(d * (std::f64::consts::LN_2 as f32), big);
    if r.is_subnormal() {
        0
    } else {
        r.to_bits()
    }
}

/// The `exp2` routine answers its model ([`exp2_model`]) and the `exp` and `exp10` routines Metal's
/// compositions of it ([`exp_model`]) bit for bit, over the exponentials' operands and fractions of
/// `[1, 2)` of both signs, and the models are within Apple's 4 ulp of the exact value: part `part`
/// of `of` of the runs of 2,048 operands.
fn exponentials_are_metals(part: usize, of: usize) {
    let mut xs = exp_operands(0x0E59_0E59);
    let mut rng = Rng(0x0E5A_0E5A);
    xs.extend((0..4096).map(|_| 0x3F80_0000 | (rng.next() & 0x807F_FFFF)));
    let mut runs = Vec::new();
    for (fun, exact) in [
        (Fun::Exp2, f64::exp2 as fn(f64) -> f64),
        (Fun::Exp, f64::exp),
        (Fun::Exp10, |x: f64| 10f64.powf(x)),
    ] {
        for run in xs.chunks(2048) {
            runs.push((fun, exact, run.to_vec()));
        }
    }
    let f = f32s();
    for (fun, exact, run) in part_of(runs, part, of) {
        let model = |x: u64| -> u64 {
            let x = x as u32;
            u64::from(match fun {
                Fun::Exp2 => exp2_model(x),
                Fun::Exp => exp_model(x, std::f64::consts::LOG2_E),
                _ => exp_model(x, std::f64::consts::LOG2_10),
            })
        };
        let mut worst = 0f64;
        for x in &run {
            let r = f64::from(f32v(model(*x)));
            if let Some(v) = in_range(exact(f64::from(f32v(*x)))) {
                let ulp = 2f64.powi((v.abs().log2().floor() as i32).max(-126) - 23);
                worst = worst.max((r - v).abs() / ulp);
            }
        }
        assert!(worst <= 4.0, "{fun:?}: the model is {worst:.3} ulp off");
        eprintln!("{fun:?}: the model within {worst:.3} ulp");
        check_module(
            &format!("{fun:?}"),
            routine_module(fun, false, false),
            &[f],
            f,
            &run.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
            &|c| Some(vec![model(c[0][0])]),
            &|g, w| g == w,
        );
    }
}

mod exponential_routines_are_metals {
    parts!(super::exponentials_are_metals, 6;
        part_1_of_6 = 0, part_2_of_6 = 1, part_3_of_6 = 2, part_4_of_6 = 3,
        part_5_of_6 = 4, part_6_of_6 = 5);
}

/// The `exp2` routine answers Metal's words where the model does, measured on the Apple M4:
/// `exp2(0.5)` is `0x3fb504f4`, an ulp above `√2`'s rounding, as Metal answers,
/// and the other binades' and signs' reductions and flushes alike.
#[test]
fn exp2_routine_answers_metals_measured_words() {
    let measured: [(u32, u32); 20] = [
        (0x3F80_0000, 0x4000_0000),
        (0x3F00_0000, 0x3FB5_04F4),
        (0x3DCC_CCCD, 0x3F89_2FDF),
        (0x3E99_999A, 0x3F9D_9624),
        (0x3F33_3333, 0x3FCF_EFC6),
        (0xBE99_999A, 0x3F4F_EFC6),
        (0xBFD9_999A, 0x3E9D_9624),
        (0x4049_0FD0, 0x410D_330C),
        (0xC0B0_0000, 0x3CB5_04F4),
        (0x4124_0000, 0x4498_37F1),
        (0x42C9_8000, 0x71D7_44FD),
        (0xC2C8_999A, 0x0D4F_EFA9),
        (0x42FF_CCCD, 0x7F6E_DB51),
        (0xC2FB_CCCD, 0x0089_2FD6),
        (0x3580_0000, 0x3F80_0006),
        (0xB480_0000, 0x3F7F_FFFE),
        (0xB400_0000, 0x3F7F_FFFF),
        (0xB3FF_FFFF, 0x3F80_0000),
        (0x3F7F_BE77, 0x3FFF_D297),
        (0xBF40_0000, 0x3F18_37F1),
    ];
    let f = f32s();
    check_module(
        "Exp2 at Metal's measured words",
        routine_module(Fun::Exp2, false, false),
        &[f],
        f,
        &measured
            .iter()
            .map(|(x, _)| vec![vec![u64::from(*x)]])
            .collect::<Vec<Case>>(),
        &|c| {
            measured
                .iter()
                .find(|(x, _)| u64::from(*x) == c[0][0])
                .map(|(_, w)| vec![u64::from(*w)])
        },
        &|g, w| g == w,
    );
    for (x, w) in measured {
        assert_eq!(exp2_model(x), w, "the model at {x:#010x}");
    }
}

/// Metal's precise `powr` magnitude of a positive normal `x` and a finite `y` as the model
/// computes it (`libm::log::powr`): `log2(x)` as a pair through [`libm::log::POWR_TABLE`] and
/// [`libm::log::POWR_P`], its product with `y` as a pair, one unit of the last place down from 128
/// with a negative low part, and [`exp2_model`] of the high part grown as `fma(E, RN(z_l RN(ln 2)),
/// E)`, `E` itself where it is `+inf` or `+0`; a denormal result is `+0`.
fn powr_model(x: f32, y: f32) -> u32 {
    use libm::log::{POWR_P, POWR_TABLE};
    if y == 1.0 {
        return x.to_bits();
    }
    let b = x.to_bits();
    let (mut mb, mut n) = (
        (b & 0x7F_FFFF) | 0x3F80_0000,
        ((b >> 23) & 0xFF) as i32 - 127,
    );
    if mb > 0x3FB5_04F3 {
        mb -= 0x0080_0000;
        n += 1;
    }
    let m = f32::from_bits(mb);
    let row = POWR_TABLE[((mb >> 20) & 15) as usize - 3].map(f32::from_bits);
    let p = m * row[0];
    let p_l = m.mul_add(row[1], m.mul_add(row[0], -p));
    let s = p + p_l;
    let s_l = p_l - (s - p);
    let u = s - 1.0;
    let uh = u + s_l;
    let ul = s_l - (uh - u);
    let t = horner_model(&POWR_P, uh) * uh;
    let (l2e, l2e_lo) = libm::split(std::f64::consts::LOG2_E);
    let q0 = t + l2e;
    let q0_l = (t + (l2e - q0)) + l2e_lo;
    let qh = q0 + q0_l;
    let ql = q0_l - (qh - q0);
    let lp = uh * qh;
    let l_err = ql.mul_add(uh, qh.mul_add(ul, qh.mul_add(uh, -lp)));
    let lh = lp + l_err;
    let ll = l_err - (lh - lp);
    let ef = n as f32;
    let a = ef + lh;
    let r = ll + (lh - (a - ef));
    let g = a + r;
    let gl = r - (g - a);
    let (log_hi, log_lo) = (row[2], row[3]);
    let s2 = log_hi + g;
    let rest = if log_hi.abs() < g.abs() {
        gl + (log_lo + (log_hi + (g - s2)))
    } else {
        log_lo + (gl + (g + (log_hi - s2)))
    };
    let gh = s2 + rest;
    let gl = rest - (gh - s2);
    let y = y.clamp(-2_147_483_648.0, 2_147_483_648.0);
    let zp = gh * y;
    let z_err = gl.mul_add(y, gh.mul_add(y, -zp));
    let mut zh = zp + z_err;
    let mut zl = z_err - (zh - zp);
    if zh >= 128.0 && zl < 0.0 {
        let lower = f32::from_bits(zh.to_bits() - 1);
        zl += zh - lower;
        zh = lower;
    }
    let big = f32::from_bits(exp2_model(zh.to_bits()));
    if big.is_infinite() || big == 0.0 {
        return big.to_bits();
    }
    let r = big.mul_add(zl * f32::from_bits(libm::log::LN_2_BITS), big);
    if r.is_subnormal() {
        0
    } else {
        r.to_bits()
    }
}

/// Positive normal bases and exponents for `powr`: `y log2(x)` across the exponential's range,
/// small integral exponents, and bases near 1 to large powers.
fn powr_pairs(seed: u64) -> Vec<Vec<u64>> {
    let mut rng = Rng(seed);
    let bases = log_operands(seed ^ 0x0B45_0B45);
    let mut pairs = Vec::new();
    for _ in 0..6000 {
        let x = bases[(rng.next() as usize) % bases.len()];
        let l = f64::from(f32v(x)).log2();
        if l == 0.0 {
            continue;
        }
        let t = f64::from(f32v(uniform(&mut rng, -140.0, 130.0)));
        pairs.push(vec![x, u64::from(((t / l) as f32).to_bits())]);
    }
    for _ in 0..2000 {
        let x = bases[(rng.next() as usize) % bases.len()];
        let y = (rng.next() % 41) as f32 - 20.0;
        pairs.push(vec![x, u64::from(y.to_bits())]);
    }
    for k in 1..400i64 {
        for x in [
            1.0f32 + k as f32 * f32::EPSILON,
            1.0 - k as f32 * f32::EPSILON / 2.0,
        ] {
            let l = f64::from(x).log2();
            for t in [-100.0, -10.0, 1.0, 50.0, 120.0, 127.99] {
                pairs.push(vec![
                    u64::from(x.to_bits()),
                    u64::from(((t / l) as f32).to_bits()),
                ]);
            }
        }
    }
    pairs
}

/// The `powr` routine answers its model (Metal's form, [`powr_model`]) bit for bit, with and
/// without `OpFmaKHR`, and the model is within Apple's 16 ulp of `x^y` wherever the result is
/// finite and normal: part `part` of `of` of the runs of 2,048 pairs.
fn powr_is_metals_form(part: usize, of: usize) {
    let pairs = powr_pairs(0x0907_0907);
    let mut worst = 0f64;
    for c in &pairs {
        let (x, y) = (f32v(c[0]), f32v(c[1]));
        let r = f64::from(f32::from_bits(powr_model(x, y)));
        if let Some(v) = in_range(f64::from(x).powf(f64::from(y))) {
            let ulp = 2f64.powi((v.abs().log2().floor() as i32).max(-126) - 23);
            worst = worst.max((r - v).abs() / ulp);
        }
    }
    assert!(worst <= 16.0, "the model is {worst:.3} ulp off");
    eprintln!("powr: the model within {worst:.3} ulp");
    let f = f32s();
    let runs: Vec<(bool, Vec<Vec<u64>>)> = [false, true]
        .into_iter()
        .flat_map(|fma| pairs.chunks(2048).map(move |r| (fma, r.to_vec())))
        .collect();
    for (fma, run) in part_of(runs, part, of) {
        check_module(
            &format!("Powr fma={fma}"),
            routine_module(Fun::Powr, false, fma),
            &[f, f],
            f,
            &run.iter()
                .map(|c| vec![vec![c[0]], vec![c[1]]])
                .collect::<Vec<Case>>(),
            &|c| Some(vec![u64::from(powr_model(f32v(c[0][0]), f32v(c[1][0])))]),
            &|g, w| g == w,
        );
    }
}

mod powr_routine_is_metals_form {
    parts!(super::powr_is_metals_form, 4;
        part_1_of_4 = 0, part_2_of_4 = 1, part_3_of_4 = 2, part_4_of_4 = 3);
}

/// Metal's precise `powr` words on the M3 Max (case 7b283404's `powr` lanes, the M4's the same)
/// where Metal's word is not the correctly rounded `x^y`, each one unit of the last place off it:
/// the model and the routine answer them.
const POWR_WORDS: [(u32, u32, u32); 24] = [
    (0x3966_6C74, 0xC09E_DDCB, 0x5DA0_C338),
    (0x3D67_2CF1, 0xC044_0FB6, 0x45D0_9B37),
    (0x3E00_0000, 0xC07B_F941, 0x4560_9B9F),
    (0x3E00_C07E, 0x3F5D_6038, 0x3E2A_6ECB),
    (0x3E10_3271, 0xC1EA_DB2D, 0x6902_34C8),
    (0x3E6B_C386, 0xC076_78BC, 0x438E_FD21),
    (0x3E7D_8B6D, 0x414E_7502, 0x3281_330A),
    (0x3E8E_F404, 0x401F_D291, 0x3D29_51B6),
    (0x3E9A_CA4E, 0xBF2D_D5DE, 0x4010_32E3),
    (0x3F10_D34F, 0xC0C0_0000, 0x41F4_0A3B),
    (0x3F5C_4D74, 0x3FC9_A051, 0x3F4A_1239),
    (0x3F80_0004, 0xC2AD_D0D0, 0x3F7F_FD48),
    (0x3F80_000D, 0x42A4_38BE, 0x3F80_042B),
    (0x3F80_0400, 0x4604_6C53, 0x4034_134F),
    (0x3F81_1AFD, 0x3FC0_0000, 0x3F81_A965),
    (0x3FAA_7AE9, 0x4080_0000, 0x4049_6381),
    (0x406C_AFCD, 0xC110_0000, 0x3701_A690),
    (0x40C9_0FDA, 0x40F9_3B26, 0x49C9_01AA),
    (0x4120_0000, 0xC214_0000, 0x0208_1CEB),
    (0x41A3_5CE2, 0x4071_9F39, 0x47AC_73BB),
    (0x4242_C75D, 0x3C98_4C8C, 0x3F89_96AB),
    (0x4252_1A85, 0xC175_EB28, 0x138F_2C5B),
    (0x427C_BBA6, 0xBD85_02D6, 0x3F43_9146),
    (0x4DAE_4DB4, 0x3FFE_BE9D, 0x5BC3_9F39),
];

/// Metal's precise `powr(x, 1)` is `x` itself (every one of case 7b283404's nine lanes with `y =
/// 1`, the M3's and the M4's): these three are where the form alone is a unit above.
const POWR_AT_ONE: [(u32, u32, u32); 3] = [
    (0x3FC9_C1AA, 0x3F80_0000, 0x3FC9_C1AA),
    (0x3FDF_B484, 0x3F80_0000, 0x3FDF_B484),
    (0x3FFD_BEE5, 0x3F80_0000, 0x3FFD_BEE5),
];

/// The `powr` routine and its model answer Metal's measured words ([`POWR_WORDS`],
/// [`POWR_AT_ONE`]).
#[test]
fn powr_answers_metals_measured_words() {
    let words: Vec<(u32, u32, u32)> = POWR_WORDS.iter().chain(&POWR_AT_ONE).copied().collect();
    for &(x, y, w) in &words {
        let got = powr_model(f32::from_bits(x), f32::from_bits(y));
        assert_eq!(got, w, "the model at ({x:#010x}, {y:#010x}): {got:#010x}");
    }
    let f = f32s();
    for fma in [false, true] {
        check_module(
            &format!("Powr fma={fma} at Metal's measured words"),
            routine_module(Fun::Powr, false, fma),
            &[f, f],
            f,
            &words
                .iter()
                .map(|(x, y, _)| vec![vec![u64::from(*x)], vec![u64::from(*y)]])
                .collect::<Vec<Case>>(),
            &|c| {
                words
                    .iter()
                    .find(|(x, y, _)| [u64::from(*x), u64::from(*y)] == [c[0][0], c[1][0]])
                    .map(|(_, _, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
    }
}

/// The centres `c` of `powr`'s reduction, one per row of [`libm::log::POWR_TABLE`]: the left end
/// of the row's sixteenth (below 1) or eighth (from 1) of a binade, or 1 next to 1.
const POWR_CENTRES: [f64; 9] = [0.6875, 0.75, 0.8125, 0.875, 1.0, 1.0, 1.125, 1.25, 1.375];

/// [`libm::log::POWR_TABLE`] is `1 / c` and `log2(c)` of each centre of [`POWR_CENTRES`] split into
/// a `float` pair, and each centre is the left end of its row's range of significands (or 1 next to
/// 1).
#[test]
fn powr_table_is_the_centres_split() {
    use libm::log::POWR_TABLE;
    for (k, (c, row)) in POWR_CENTRES.iter().zip(POWR_TABLE).enumerate() {
        let (ih, il) = libm::split(1.0 / c);
        let (lh, ll) = libm::split(c.log2());
        let want = [ih, il, lh, ll].map(f32::to_bits);
        assert_eq!(row, want, "row {k}: centre {c}");
        // Row `k` holds the significands whose bits 20 to 23 (with the exponent's last bit) are
        // `k + 3`.
        let left = f32::from_bits(0x3F00_0000 | ((k as u32 + 3) << 20));
        assert!(
            *c == f64::from(left) || *c == 1.0,
            "row {k}: centre {c}, left end {left}"
        );
    }
}

/// Every table of the math library is declared a whole number of 16-byte rows
/// (`libm::declare_words`), the quarter-turn table's 165 words followed by three zeros, so each
/// table starts on a 16-byte boundary of the tables declared before it. Behind the unpadded 165,
/// NVIDIA's driver read every word of the `powr` table one index low (case 7b283404 on the RTX
/// 5080).
#[test]
fn math_tables_are_whole_16_byte_rows() {
    let f = f32s();
    let needs = libm::Needs {
        exp2: true,
        log2: true,
        rcp: true,
        rsqrt: true,
        quarter: true,
        sinc: true,
        powr: true,
        ..libm::Needs::default()
    };
    let module = kernel_defs(
        &[f],
        f,
        &|b| {
            libm::define_helpers(b, needs, false, true)?;
            Ok(Vec::new())
        },
        &|_, x, _| Ok(x[0]),
        false,
        false,
    )
    .expect("build");
    let words = serialize(module).expect("words").words;
    const OP_TYPE_ARRAY: u32 = 28;
    const OP_CONSTANT: u32 = 43;
    let mut constants = BTreeMap::new();
    let mut lengths = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (n, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        match op {
            OP_CONSTANT if n == 4 => {
                constants.insert(words[i + 2], words[i + 3]);
            }
            OP_TYPE_ARRAY => lengths.push(constants[&words[i + 3]]),
            _ => {}
        }
        i += n.max(1);
    }
    lengths.sort_unstable();
    assert_eq!(lengths, [36, 128, 160, 168, 192, 256, 320]);
}

/// `P(v)` of the coefficients `c` (bits, lowest first) by Horner's rule from the highest in fused
/// steps, as the routines evaluate Metal's polynomials.
fn horner_model(c: &[u32], v: f32) -> f32 {
    let mut acc = f32::from_bits(c[c.len() - 1]);
    for k in c[..c.len() - 1].iter().rev() {
        acc = acc.mul_add(v, f32::from_bits(*k));
    }
    acc
}

/// Metal's precise `asinh`, `acosh`, or `atanh` of `xb` as the model computes it, of
/// [`log2_model`]'s fitted word, [`rcp_model`], and Metal's fast inverse root's model
/// [`rsqrt_model`] (libm::hyper's list); `None` where [`log2_model`] is.
fn arc_model(fun: Fun, xb: u32) -> Option<u32> {
    use libm::hyper::*;
    let x = f32::from_bits(xb);
    if x.is_nan() {
        return Some(xb);
    }
    let ln2 = f32::from_bits(libm::log::LN_2_BITS);
    let l2 = |v: f32| log2_model(v.to_bits()).map(f32::from_bits);
    let a = x.abs();
    let ab = a.to_bits();
    let r = match fun {
        Fun::Asinh => {
            let r = if ab < ASINH_TINY {
                a
            } else if ab < ASINH_NEAR_END {
                horner_model(&ASINH_NEAR, a * a) * a
            } else if ab < ASINH_ROOT {
                horner_model(&ASINH_MIDDLE, a)
            } else if a.is_infinite() {
                a
            } else if a >= ASINH_LARGE {
                l2(a)? * ln2 + ln2
            } else {
                let z = a.mul_add(a, 1.0);
                l2(a + z * rsqrt_model(z))? * ln2
            };
            r.copysign(x)
        }
        Fun::Acosh => {
            if x < 1.0 {
                return Some(0x7FC0_0000);
            }
            if x.is_infinite() {
                x
            } else if xb >= ACOSH_LARGE {
                l2(x)? * ln2 + ln2
            } else {
                let z = (x - 1.0) * (x + 1.0);
                let q = if z == 0.0 { 0.0 } else { z * rsqrt_model(z) };
                let w = q + x;
                let err = q - (w - x);
                let off = w.to_bits().wrapping_sub(ACOSH_LOG1P_START);
                let l = if off < ACOSH_LOG1P_END - ACOSH_LOG1P_START {
                    let t = w - 1.0;
                    t * horner_model(&ACOSH_LOG1P, t)
                } else {
                    l2(w)? * ln2
                };
                l + err * rcp_model(w)
            }
        }
        _ => {
            if a > 1.0 {
                return Some(0x7FC0_0000);
            }
            let r = if ab <= ATANH_TINY {
                a
            } else if a == 1.0 {
                f32::INFINITY
            } else if a <= ATANH_FORM {
                horner_model(&ATANH_NEAR, a)
            } else {
                let q = (1.0 + a) * rcp_model(1.0 - a);
                l2(q)? * ln2 * 0.5
            };
            r.copysign(x)
        }
    };
    Some(r.to_bits())
}

/// The ulp error of `w` against the exact `e` (in units of the `float` nearest `e`'s magnitude).
fn ulps(w: u32, e: f64) -> f64 {
    let ulp = f64::from(f32::from_bits((e.abs() as f32).to_bits() + 1)) - f64::from(e.abs() as f32);
    (f64::from(f32::from_bits(w)) - e).abs() / ulp
}

/// The inverse hyperbolic routines answer [`arc_model`] bit for bit over the exponentials'
/// operands, each region's edges, uniform operands in each region, and every 262,147th positive
/// `float` (both signs for the odd functions), and lie within Apple's bounds of the exact value
/// there (4, 4, and 5 ulp): part `part` of `of` of the runs of 2,048 operands.
fn inverse_hyperbolics_are_metals(part: usize, of: usize) {
    use libm::hyper::*;
    let mut rng = Rng(0x4790_4790);
    let mut xs = exp_operands(0x51A4_51A4);
    xs.extend(magnitudes(&mut rng, -30.0, 127.9, 2000));
    xs.extend((0..0x7F80_0000u32).step_by(262_147).map(u64::from));
    for edge in [
        ASINH_TINY,
        ASINH_NEAR_END,
        ASINH_ROOT,
        ASINH_LARGE.to_bits(),
        ACOSH_LARGE,
        ATANH_TINY,
        ATANH_FORM.to_bits(),
        0x3F80_0000,
        0x3FC5_83AB,
    ] {
        xs.extend(around(f32::from_bits(edge), 32));
    }
    for (lo, hi) in [
        (1.0, 1.6),
        (0.0, 0.35),
        (0.35, 1.5),
        (1.5, 500.0),
        (0.25, 1.0),
    ] {
        xs.extend((0..1500).map(|_| uniform(&mut rng, lo, hi)));
    }
    xs.retain(|b| !f32v(*b).is_nan());
    let signed: Vec<u64> = xs
        .iter()
        .flat_map(|b| [*b & 0x7FFF_FFFF, *b | 0x8000_0000])
        .collect();
    let mut runs = Vec::new();
    for (fun, set, exact, bound) in [
        (Fun::Asinh, &signed, f64::asinh as fn(f64) -> f64, 4.0),
        (Fun::Acosh, &xs, f64::acosh, 4.0),
        (Fun::Atanh, &signed, f64::atanh, 5.0),
    ] {
        let cases: Vec<(u32, u32)> = set
            .iter()
            .filter_map(|b| arc_model(fun, *b as u32).map(|w| (*b as u32, w)))
            .collect();
        for run in cases.chunks(2048) {
            runs.push((fun, exact, bound, run.to_vec()));
        }
    }
    let f = f32s();
    for (fun, exact, bound, run) in part_of(runs, part, of) {
        check_module(
            &format!("{fun:?}"),
            routine_module(fun, false, false),
            &[f],
            f,
            &run.iter()
                .map(|(x, _)| vec![vec![u64::from(*x)]])
                .collect::<Vec<Case>>(),
            &|c| {
                run.iter()
                    .find(|(x, _)| u64::from(*x) == c[0][0])
                    .map(|(_, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
        let worst = run
            .iter()
            .filter_map(|(x, w)| {
                let e = exact(f64::from(f32::from_bits(*x)));
                (e.is_finite() && e.abs() >= f64::from(f32::MIN_POSITIVE)).then(|| ulps(*w, e))
            })
            .fold(0f64, f64::max);
        assert!(worst <= bound, "{fun:?}: {worst:.3} ulp");
    }
}

mod inverse_hyperbolic_routines_are_metals {
    parts!(super::inverse_hyperbolics_are_metals, 6;
        part_1_of_6 = 0, part_2_of_6 = 1, part_3_of_6 = 2, part_4_of_6 = 3,
        part_5_of_6 = 4, part_6_of_6 = 5);
}

/// [`arc_model`] is Metal's word at operands measured on the M3 in every region, most where Metal's
/// word is not the correctly rounded one (the last 16 where the forms with the correctly rounded
/// inverse root are not Metal's word either), and the routines answer them bit for bit.
#[test]
fn inverse_hyperbolics_answer_metals_measured_words() {
    let measured: [(Fun, u32, u32); 135] = [
        (Fun::Asinh, 0x3FC0_0000, 0x3F98_EE00),
        (Fun::Asinh, 0x3FC0_0004, 0x3F98_EE02),
        (Fun::Asinh, 0x3FD8_0001, 0x3FA5_B10A),
        (Fun::Asinh, 0x3FF0_0002, 0x3FB1_7218),
        (Fun::Asinh, 0x4008_0000, 0x3FBF_C455),
        (Fun::Asinh, 0x4020_0000, 0x3FD2_D877),
        (Fun::Asinh, 0x4038_0001, 0x3FE3_9AB6),
        (Fun::Asinh, 0x4050_0000, 0x3FF2_849D),
        (Fun::Asinh, 0x4068_0002, 0x3FFF_EFCB),
        (Fun::Asinh, 0x4084_D400, 0x4008_5CE9),
        (Fun::Asinh, 0x4284_D300, 0x409C_735E),
        (Fun::Asinh, 0x43F9_FFFF, 0x40DD_0C56),
        (Fun::Asinh, 0x43FA_0000, 0x40DD_0C55),
        (Fun::Asinh, 0x4484_D200, 0x40F5_2BB5),
        (Fun::Asinh, 0x4C84_D900, 0x4196_0466),
        (Fun::Asinh, 0x5484_D200, 0x41EE_BD05),
        (Fun::Asinh, 0x5C84_D200, 0x4223_BB08),
        (Fun::Asinh, 0x6484_D200, 0x4250_178E),
        (Fun::Asinh, 0x6C84_D200, 0x427C_7414),
        (Fun::Asinh, 0x7484_D300, 0x4294_6850),
        (Fun::Asinh, 0x7C84_D300, 0x42AA_9693),
        (Fun::Asinh, 0x7E84_D300, 0x42B0_2224),
        (Fun::Asinh, 0x7F7F_FF00, 0x42B2_D4FA),
        (Fun::Acosh, 0x3FC5_83AB, 0x3F80_0000),
        (Fun::Acosh, 0x3FC5_83AD, 0x3F80_0001),
        (Fun::Acosh, 0x3FFC_D337, 0x3FA6_B8E2),
        (Fun::Acosh, 0x4034_22C3, 0x3FD8_EF85),
        (Fun::Acosh, 0x406B_724A, 0x3FFD_05DD),
        (Fun::Acosh, 0x40A2_C1D7, 0x4013_D4CD),
        (Fun::Acosh, 0x40DA_1160, 0x4026_D5A6),
        (Fun::Acosh, 0x4111_60EA, 0x4039_661C),
        (Fun::Acosh, 0x4148_B076, 0x404E_2055),
        (Fun::Acosh, 0x4184_D300, 0x4060_1DE2),
        (Fun::Acosh, 0x4384_D200, 0x40C8_CF28),
        (Fun::Acosh, 0x44DA_C8D8, 0x4102_9220),
        (Fun::Acosh, 0x44DA_C8E1, 0x4102_9222),
        (Fun::Acosh, 0x4584_D300, 0x4110_C43C),
        (Fun::Acosh, 0x4D84_D500, 0x41A1_1B4A),
        (Fun::Acosh, 0x5584_D200, 0x41F9_D426),
        (Fun::Acosh, 0x5D84_D200, 0x4229_4699),
        (Fun::Acosh, 0x6584_D300, 0x4255_A326),
        (Fun::Acosh, 0x6D84_D300, 0x4280_FFD6),
        (Fun::Acosh, 0x7584_D200, 0x4297_2E15),
        (Fun::Acosh, 0x7984_D300, 0x42A2_453A),
        (Fun::Acosh, 0x7D84_D200, 0x42AD_5C58),
        (Fun::Acosh, 0x7F7F_FF00, 0x42B2_D4FA),
        (Fun::Asinh, 0x39E8_9800, 0x39E8_9800),
        (Fun::Asinh, 0x3E80_0000, 0x3E7D_67D8),
        (Fun::Asinh, 0x3EA8_0011, 0x3EA5_1F79),
        (Fun::Asinh, 0x3ECF_FFFF, 0x3ECA_AA7C),
        (Fun::Asinh, 0x3EF7_FFFF, 0x3EEF_33EA),
        (Fun::Asinh, 0x3F1F_FFFF, 0x3F17_13A6),
        (Fun::Asinh, 0x3F47_FFFF, 0x3F37_CC39),
        (Fun::Asinh, 0x3F70_0000, 0x3F56_23AC),
        (Fun::Asinh, 0x3F98_0000, 0x3F81_0462),
        (Fun::Acosh, 0x3F80_0008, 0x3AB5_04F3),
        (Fun::Acosh, 0x3F89_9999, 0x3EC5_13E7),
        (Fun::Acosh, 0x3F93_3333, 0x3F0A_855A),
        (Fun::Acosh, 0x3F9C_CCCD, 0x3F28_A9B7),
        (Fun::Acosh, 0x3FA6_6666, 0x3F41_A594),
        (Fun::Acosh, 0x3FB0_0000, 0x3F57_4D0A),
        (Fun::Acosh, 0x3FB9_9999, 0x3F6A_9205),
        (Fun::Acosh, 0x3FC3_3334, 0x3F7C_055E),
        (Fun::Acosh, 0x4180_0100, 0x405D_BF17),
        (Fun::Acosh, 0x493F_FA00, 0x4164_4AE6),
        (Fun::Acosh, 0x50FF_F300, 0x41C7_9FF2),
        (Fun::Acosh, 0x58BF_EB00, 0x420F_05AC),
        (Fun::Acosh, 0x607F_E400, 0x4239_C300),
        (Fun::Acosh, 0x683F_DD00, 0x4264_F8A5),
        (Fun::Acosh, 0x6FFF_DD00, 0x4287_DB13),
        (Fun::Acosh, 0x77BF_CF00, 0x429D_75CF),
        (Fun::Atanh, 0x3E80_0002, 0x3E82_C579),
        (Fun::Atanh, 0x3E85_57FC, 0x3E88_7CCA),
        (Fun::Atanh, 0x3E8A_AFD1, 0x3E8E_3C7D),
        (Fun::Atanh, 0x3E90_0679, 0x3E94_03EE),
        (Fun::Atanh, 0x3E95_5DF7, 0x3E99_D5C2),
        (Fun::Atanh, 0x3E9A_B87F, 0x3E9F_B4DB),
        (Fun::Atanh, 0x3EA0_0C10, 0x3EA5_96B9),
        (Fun::Atanh, 0x3EA5_589A, 0x3EAB_7BA5),
        (Fun::Atanh, 0x3EAA_B0FD, 0x3EB1_7933),
        (Fun::Atanh, 0x3EB0_0941, 0x3EB7_82AE),
        (Fun::Atanh, 0x3EB5_6275, 0x3EBD_99DC),
        (Fun::Atanh, 0x3EBA_B6D6, 0x3EC3_B8BC),
        (Fun::Atanh, 0x3EC0_0122, 0x3EC9_D9A8),
        (Fun::Atanh, 0x3EC5_454E, 0x3ED0_0199),
        (Fun::Atanh, 0x3ECA_8C3D, 0x3ED6_3B8E),
        (Fun::Atanh, 0x3ECF_D8E5, 0x3EDC_8BD9),
        (Fun::Atanh, 0x3ED5_21ED, 0x3EE2_E81A),
        (Fun::Atanh, 0x3EDA_66C4, 0x3EE9_5040),
        (Fun::Atanh, 0x3EDF_A03E, 0x3EEF_BC0F),
        (Fun::Atanh, 0x3EE4_DFBA, 0x3EF6_4194),
        (Fun::Atanh, 0x3EEA_1651, 0x3EFC_CF15),
        (Fun::Atanh, 0x3EF0_8232, 0x3F02_7E61),
        (Fun::Atanh, 0x3EF7_FAF1, 0x3F07_5492),
        (Fun::Atanh, 0x3EFF_700B, 0x3F0C_3F6E),
        (Fun::Atanh, 0x3F08_C4B1, 0x3F18_986B),
        (Fun::Atanh, 0x3F12_3AAF, 0x3F26_3971),
        (Fun::Atanh, 0x3F1B_03E1, 0x3F33_AB1D),
        (Fun::Atanh, 0x3F21_96C7, 0x3F3E_50C3),
        (Fun::Atanh, 0x3F28_2932, 0x3F49_8CB3),
        (Fun::Atanh, 0x3F2E_C051, 0x3F55_8184),
        (Fun::Atanh, 0x3F35_23A9, 0x3F61_DF28),
        (Fun::Atanh, 0x3F3B_61C8, 0x3F6E_D506),
        (Fun::Atanh, 0x3F41_9F3D, 0x3F7C_D330),
        (Fun::Atanh, 0x3F4C_5103, 0x3F8B_F41D),
        (Fun::Atanh, 0x3F58_A4E3, 0x3F9F_15A6),
        (Fun::Atanh, 0x3F63_5748, 0x3FB4_D0F0),
        (Fun::Atanh, 0x3F69_A218, 0x3FC5_824C),
        (Fun::Atanh, 0x3F6F_DF48, 0x3FDB_3FDD),
        (Fun::Atanh, 0x3F76_38FC, 0x3FFC_1812),
        (Fun::Atanh, 0x3F7F_FFFE, 0x4105_1591),
        (Fun::Atanh, 0x39B8_9C00, 0x39B8_9C00),
        (Fun::Atanh, 0x3A20_0000, 0x3A20_0000),
        (Fun::Atanh, 0x3AC0_0000, 0x3AC0_0008),
        (Fun::Atanh, 0x3B60_7000, 0x3B60_7039),
        (Fun::Atanh, 0x3C00_3500, 0x3C00_35AB),
        (Fun::Atanh, 0x3CA0_0900, 0x3CA0_0E36),
        (Fun::Atanh, 0x3D40_0100, 0x3D40_250C),
        (Fun::Atanh, 0x3DE0_0000, 0x3DE0_E651),
        (Fun::Asinh, 0x3FC0_002A, 0x3F98_EE17),
        (Fun::Asinh, 0x3FD2_D856, 0x3FA3_0A46),
        (Fun::Asinh, 0x3FE4_4F82, 0x3FAB_D693),
        (Fun::Asinh, 0x400D_8489, 0x3FC4_63A8),
        (Fun::Asinh, 0x4029_D277, 0x3FD9_F44D),
        (Fun::Asinh, 0x4044_E4A1, 0x3FEB_D221),
        (Fun::Asinh, 0x4062_DB4B, 0x3FFD_2C2E),
        (Fun::Asinh, 0x407F_FFC9, 0x4006_0FB8),
        (Fun::Acosh, 0x3F80_0002, 0x3A35_04F3),
        (Fun::Acosh, 0x3F9D_1641, 0x3F29_78C7),
        (Fun::Acosh, 0x3FC4_1B34, 0x3F7D_96BF),
        (Fun::Acosh, 0x3FFC_E8BB, 0x3FA6_C583),
        (Fun::Acosh, 0x4044_F0B8, 0x3FE5_134E),
        (Fun::Acosh, 0x409A_42F7, 0x4010_53F9),
        (Fun::Acosh, 0x40FB_21B7, 0x402F_F49C),
        (Fun::Acosh, 0x417F_FF39, 0x405D_BE66),
    ];
    for (fun, x, w) in measured {
        assert_eq!(arc_model(fun, x), Some(w), "{fun:?}'s model at {x:#010x}");
    }
    let f = f32s();
    for fun in [Fun::Asinh, Fun::Acosh, Fun::Atanh] {
        let mine: Vec<(u32, u32)> = measured
            .iter()
            .filter(|(g, _, _)| *g == fun)
            .map(|(_, x, w)| (*x, *w))
            .collect();
        check_module(
            &format!("{fun:?} at Metal's measured words"),
            routine_module(fun, false, false),
            &[f],
            f,
            &mine
                .iter()
                .map(|(x, _)| vec![vec![u64::from(*x)]])
                .collect::<Vec<Case>>(),
            &|c| {
                mine.iter()
                    .find(|(x, _)| u64::from(*x) == c[0][0])
                    .map(|(_, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
    }
}

/// Metal's fast `asinh` or `acosh` of `xb` as the model computes it (`libm::hyper::fast_arc`):
/// `RN(log2(v) RN(ln 2))` of `v = fma(s, rsqrt(s), x)` and `s = RN(RN(x x) ± 1)` with Metal's
/// inverse root's model [`rsqrt_model`] and [`log2_model`], `v = RN(x + s)` where `s` is 0 or
/// `+inf`, and Metal's NaN where `s` is below zero; an `f32` denormal operand is a zero. `None`
/// where [`log2_model`] is, and for a NaN `v`.
fn fast_arc_model(fun: Fun, xb: u32) -> Option<u32> {
    let x = f32::from_bits(xb);
    let x = if x.is_subnormal() {
        0f32.copysign(x)
    } else {
        x
    };
    let s = if fun == Fun::FastAsinh {
        x * x + 1.0
    } else {
        x * x - 1.0
    };
    if s < 0.0 {
        return Some(0x7FC0_0000);
    }
    let v = if s == 0.0 || s.is_infinite() {
        x + s
    } else {
        s.mul_add(rsqrt_model(s), x)
    };
    let l2 = if v.is_nan() {
        return None;
    } else if v.to_bits() & 0x7F80_0000 == 0 {
        f32::NEG_INFINITY
    } else if v < 0.0 {
        return Some(0x7FC0_0000);
    } else if v.is_infinite() {
        v
    } else {
        f32::from_bits(log2_model(v.to_bits())?)
    };
    Some((l2 * f32::from_bits(libm::log::LN_2_BITS)).to_bits())
}

/// The fast `asinh` and `acosh` routines answer [`fast_arc_model`] bit for bit, with and without
/// `OpFmaKHR`, over the exponentials' operands, magnitudes of either sign from `2^-30` to `2^127`
/// (where the fused sum keeps the root's error for a negative `x`), and the neighbours of 1 and -1:
/// part `part` of `of` of the runs of 2,048 operands.
fn fast_arcs_are_metals_form(part: usize, of: usize) {
    let mut rng = Rng(0x4820_4820);
    let mut xs = exp_operands(0x4821_4821);
    xs.extend(magnitudes(&mut rng, -30.0, 127.9, 1500));
    xs.extend(around(1.0, 32));
    xs.extend(around(-1.0, 32));
    xs.retain(|b| f32v(*b).is_finite());
    let mut runs = Vec::new();
    for fun in [Fun::FastAsinh, Fun::FastAcosh] {
        let cases: Vec<(u32, u32)> = xs
            .iter()
            .filter_map(|b| fast_arc_model(fun, *b as u32).map(|w| (*b as u32, w)))
            .collect();
        for fma in [false, true] {
            for run in cases.chunks(2048) {
                runs.push((fun, fma, run.to_vec()));
            }
        }
    }
    let f = f32s();
    for (fun, fma, run) in part_of(runs, part, of) {
        check_module(
            &format!("{fun:?} fma={fma}"),
            routine_module(fun, false, fma),
            &[f],
            f,
            &run.iter()
                .map(|(x, _)| vec![vec![u64::from(*x)]])
                .collect::<Vec<Case>>(),
            &|c| {
                run.iter()
                    .find(|(x, _)| u64::from(*x) == c[0][0])
                    .map(|(_, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
    }
}

mod fast_inverse_hyperbolic_routines_are_metals_form {
    parts!(super::fast_arcs_are_metals_form, 8;
        part_1_of_8 = 0, part_2_of_8 = 1, part_3_of_8 = 2, part_4_of_8 = 3,
        part_5_of_8 = 4, part_6_of_8 = 5, part_7_of_8 = 6, part_8_of_8 = 7);
}

/// Metal's fast `asinh` and `acosh` words on the M3 Max (case f6524207, the M4's the same), most
/// where they are not the correctly rounded function: below `-1e7` the fused sum leaves the root's
/// error (`asinh` and `acosh` alike there), and from `-1e4` on a negative `acosh` is a number. The
/// model and the routines answer them.
#[test]
fn fast_arcs_answer_metals_measured_words() {
    let measured: [(Fun, u32, u32); 32] = [
        (Fun::FastAsinh, 0xFF7F_FFFF, 0x7F80_0000),
        (Fun::FastAsinh, 0xD96D_F87F, 0x7FC0_0000),
        (Fun::FastAsinh, 0xD678_EF4C, 0x4166_3E90),
        (Fun::FastAsinh, 0xD438_A2DF, 0x40D8_9E8D),
        (Fun::FastAsinh, 0xD200_1B7B, 0x4101_F0DF),
        (Fun::FastAsinh, 0xCE52_18BC, 0x4050_D6AA),
        (Fun::FastAsinh, 0xCC35_418C, 0x3F81_6E95),
        (Fun::FastAsinh, 0xCB28_390F, 0xBF1B_C5E5),
        (Fun::FastAsinh, 0xCAC9_0FDC, 0xC002_7053),
        (Fun::FastAsinh, 0xC589_ADD5, 0xC130_0AB4),
        (Fun::FastAsinh, 0xC5A8_A26C, 0xC112_840A),
        (Fun::FastAsinh, 0xC4C1_8A29, 0xC100_AEB2),
        (Fun::FastAsinh, 0xC291_D119, 0xC09F_7158),
        (Fun::FastAsinh, 0xC11B_5C07, 0xC03E_033D),
        (Fun::FastAsinh, 0xBF29_5707, 0xBF1E_EE94),
        (Fun::FastAsinh, 0xBDD7_1180, 0xBDD6_ACCF),
        (Fun::FastAsinh, 0x3E02_D2E7, 0x3E02_7873),
        (Fun::FastAsinh, 0x3F66_8CEF, 0x3F4F_2E8A),
        (Fun::FastAsinh, 0x3FFE_A25D, 0x3FB8_2C59),
        (Fun::FastAsinh, 0x41B7_F547, 0x4075_0C78),
        (Fun::FastAsinh, 0x42B0_0000, 0x40A5_74DE),
        (Fun::FastAsinh, 0x4AC9_0FDC, 0x4183_26D8),
        (Fun::FastAsinh, 0x6892_3E06, 0x7F80_0000),
        (Fun::FastAcosh, 0xFF7F_FFFF, 0x7F80_0000),
        (Fun::FastAcosh, 0xD678_EF4C, 0x4166_3E90),
        (Fun::FastAcosh, 0xCB28_390F, 0xBF1B_C5E5),
        (Fun::FastAcosh, 0xC589_ADD5, 0xC130_0AB4),
        (Fun::FastAcosh, 0xC0D7_A67F, 0x7FC0_0000),
        (Fun::FastAcosh, 0x3F7F_FFFB, 0x7FC0_0000),
        (Fun::FastAcosh, 0x4128_2940, 0x4042_C3F3),
        (Fun::FastAcosh, 0x497F_FFFF, 0x4168_E5BE),
        (Fun::FastAcosh, 0x3F80_0000, 0x0000_0000),
    ];
    for (fun, x, w) in measured {
        assert_eq!(
            fast_arc_model(fun, x),
            Some(w),
            "{fun:?}'s model at {x:#010x}"
        );
    }
    let f = f32s();
    for fun in [Fun::FastAsinh, Fun::FastAcosh] {
        let mine: Vec<(u32, u32)> = measured
            .iter()
            .filter(|(g, _, _)| *g == fun)
            .map(|(_, x, w)| (*x, *w))
            .collect();
        for fma in [false, true] {
            check_module(
                &format!("{fun:?} fma={fma} at Metal's measured words"),
                routine_module(fun, false, fma),
                &[f],
                f,
                &mine
                    .iter()
                    .map(|(x, _)| vec![vec![u64::from(*x)]])
                    .collect::<Vec<Case>>(),
                &|c| {
                    mine.iter()
                        .find(|(x, _)| u64::from(*x) == c[0][0])
                        .map(|(_, w)| vec![u64::from(*w)])
                },
                &|g, w| g == w,
            );
        }
    }
}

/// Metal's `half` `asinh` of `half` bits `b`, rounded to `half`, as the model computes it
/// (`libm::hyper::half_asinh`): `x` below `2^-11` in magnitude and at an infinity, else `RN(log2(v)
/// RN(ln 2))` with `x`'s sign of `v = fma(z, rsqrt(z), |x|)` and `z = fma(x, x, 1)`, with
/// [`rsqrt_model`] and [`log2_model`]; `None` for a NaN and where [`log2_model`] is.
fn half_asinh_model(b: u16) -> Option<u64> {
    let x = f16_to_f64(b) as f32;
    if x.is_nan() {
        return None;
    }
    let r = if x.abs() < 1.0 / 2048.0 || x.is_infinite() {
        x
    } else {
        let z = x.mul_add(x, 1.0);
        let v = z.mul_add(rsqrt_model(z), x.abs());
        let l = f32::from_bits(log2_model(v.to_bits())?) * f32::from_bits(libm::log::LN_2_BITS);
        if x < 0.0 {
            -l
        } else {
            l
        }
    };
    Some(u64::from(f64_to_f16(f64::from(r))))
}

/// A `half` `asinh` call answers [`half_asinh_model`] at every `half` it gives a value for (the
/// model is Metal's word at every `half` of our probe, the M3; the M4's are the same): part
/// `part` of `of` of the runs of 2,048 operands.
fn half_asinh_is_metals_form(part: usize, of: usize) {
    let h = float(16);
    let cases: Vec<(u64, u64)> = (0..=0xFFFFu16)
        .filter_map(|b| half_asinh_model(b).map(|w| (u64::from(b), w)))
        .collect();
    let runs: Vec<&[(u64, u64)]> = cases.chunks(2048).collect();
    for run in part_of(runs, part, of) {
        check_module(
            "half Asinh",
            super::libm::half_module(FloatFn::Asinh, Fun::HalfAsinh),
            &[h],
            h,
            &run.iter()
                .map(|(b, _)| vec![vec![*b]])
                .collect::<Vec<Case>>(),
            &|c| {
                run.iter()
                    .find(|(b, _)| *b == c[0][0])
                    .map(|(_, w)| vec![*w])
            },
            &|g, w| g == w,
        );
    }
}

mod half_asinh_routine_is_metals_form {
    parts!(super::half_asinh_is_metals_form, 8;
        part_1_of_8 = 0, part_2_of_8 = 1, part_3_of_8 = 2, part_4_of_8 = 3,
        part_5_of_8 = 4, part_6_of_8 = 5, part_7_of_8 = 6, part_8_of_8 = 7);
}

/// Metal's `half` `asinh` words where they are not the correctly rounded ones (all four of our
/// probe, the M3; the M4's are the same): the model and the routine answer them.
#[test]
fn half_asinh_answers_metals_words() {
    let h = float(16);
    let measured: [(u64, u64); 4] = [
        (0x29C5, 0x29C4),
        (0x32BD, 0x32B0),
        (0xA9C5, 0xA9C4),
        (0xB2BD, 0xB2B0),
    ];
    for (b, w) in measured {
        assert_eq!(half_asinh_model(b as u16), Some(w), "the model at {b:#06x}");
    }
    check_module(
        "half Asinh at Metal's words",
        super::libm::half_module(FloatFn::Asinh, Fun::HalfAsinh),
        &[h],
        h,
        &measured
            .iter()
            .map(|(b, _)| vec![vec![*b]])
            .collect::<Vec<Case>>(),
        &|c| {
            measured
                .iter()
                .find(|(b, _)| *b == c[0][0])
                .map(|(_, w)| vec![*w])
        },
        &|g, w| g == w,
    );
}

/// [`rcp_model`] answers Metal's reciprocal at words measured on the Apple M4 (a
/// library compiled with `mathMode` fast): where Metal's word is `RN(1 / y)`, one unit above it,
/// and one below, in `[1, 2)`, the same significands in the lowest and highest binades whose
/// reciprocal is normal, and negated.
#[test]
fn metal_reciprocal_model_answers_measured_words() {
    let measured: [(u32, u32); 23] = [
        (0x3F80_0000, 0x3F80_0000),
        (0x3F80_05D1, 0x3F7F_F45E),
        (0x3F86_446F, 0x3F74_0CEA),
        (0x3F8E_EDE9, 0x3F65_4299),
        (0x3FBB_D1E9, 0x3F2E_7707),
        (0x3FC0_0000, 0x3F2A_AAAB),
        (0x3FC9_66D0, 0x3F22_B326),
        (0x3FD0_E1BE, 0x3F1C_DF98),
        (0x3FD6_6159, 0x3F18_D993),
        (0x3FDB_C21E, 0x3F15_1BFA),
        (0x3FDB_D94E, 0x3F15_0C40),
        (0x3FE4_BF14, 0x3F0F_4017),
        (0x3FE8_959C, 0x3F0C_E2F1),
        (0x3FF4_6C2C, 0x3F06_101C),
        (0x3FF8_E791, 0x3F03_A61C),
        (0x3FFF_FFFF, 0x3F00_0001),
        (0x00C9_66D0, 0x7E22_B326),
        (0x0080_05D1, 0x7E7F_F45E),
        (0x7E49_66D0, 0x00A2_B326),
        (0x7E5B_C21E, 0x0095_1BFA),
        (0xBFC9_66D0, 0xBF22_B326),
        (0xBF80_05D1, 0xBF7F_F45E),
        (0xBFDB_C21E, 0xBF15_1BFA),
    ];
    for (y, w) in measured {
        assert_eq!(
            rcp_model(f32::from_bits(y)).to_bits(),
            w,
            "{y:#010x}: the model against Metal's word"
        );
    }
}

/// [`Fun::Recip`]'s routine answers [`rcp_model`] bit for bit, with and without `OpFmaKHR`, over
/// magnitudes from `2^-126` to `2^128` of both signs and around the ends of the binades it scales
/// between, and Metal's flushes: the infinity of a zero's or a denormal's sign, and the zero of the
/// sign past `2^126`, an infinity included; a NaN gives a NaN.
#[test]
fn metal_reciprocal_routine_answers_the_model() {
    const ANY_NAN: u64 = 0x7FC0_0000;
    let mut rng = Rng(0x7EC1_7EC1);
    let mut xs = magnitudes(&mut rng, -126.0, 128.0, 3000);
    for e in [-126, -125, -1, 0, 1, 125, 126, 127] {
        let p = 2f32.powi(e);
        xs.extend(around(p, 2));
        xs.extend(around(-p, 2));
    }
    xs.extend(DENORMALS);
    xs.extend([
        0,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0xFF80_0001,
    ]);
    let expect = |b: u64| -> u64 {
        let b = b as u32;
        let y = f32::from_bits(b);
        let sign = b & 0x8000_0000;
        u64::from(if y.is_nan() {
            return ANY_NAN;
        } else if b & 0x7F80_0000 == 0 {
            sign | 0x7F80_0000
        } else if b & 0x7FFF_FFFF > 0x7E80_0000 {
            sign
        } else {
            rcp_model(y).to_bits()
        })
    };
    let f = f32s();
    for fma in [false, true] {
        check_module(
            &format!("Recip fma={fma}"),
            routine_module(Fun::Recip, false, fma),
            &[f],
            f,
            &xs.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
            &|c| Some(vec![expect(c[0][0])]),
            &|g, w| g == w || (w == ANY_NAN && f32v(g).is_nan()),
        );
    }
}

/// [`rsqrt_model`] is Metal's fast `rsqrt` at words measured on the M3 Max and the M4 (the same
/// words): in `[1, 4)`, in the binades below and above (every 256th `float`), all where Metal's
/// word is not the correctly rounded root, and its specials (`+inf` of a zero and a denormal, `+0`
/// of `+inf`); and [`Fun::FastRsqrt`]'s routine answers them bit for bit.
#[test]
fn metal_fast_rsqrt_answers_metals_measured_words() {
    let measured: [(u32, u32); 43] = [
        (0x3F80_0927, 0x3F7F_F6DA),
        (0x3F8C_3CE2, 0x3F74_933E),
        (0x3F94_EDE1, 0x3F6D_54D0),
        (0x3FA6_C81A, 0x3F60_4513),
        (0x3FAF_0148, 0x3F5A_EFF2),
        (0x3FB8_4D43, 0x3F55_5807),
        (0x3FC3_D76E, 0x3F4E_F682),
        (0x3FCB_095A, 0x3F4B_4346),
        (0x3FD2_09C2, 0x3F47_D8A1),
        (0x3FDE_75ED, 0x3F42_2FAB),
        (0x3FEB_2FCC, 0x3F3C_DC01),
        (0x3FF7_BBDE, 0x3F38_03BD),
        (0x4001_45CE, 0x3F34_2047),
        (0x4009_F8EC, 0x3F2E_5AD5),
        (0x4014_732B, 0x3F28_16DC),
        (0x4020_4D77, 0x3F21_C177),
        (0x402D_AC35, 0x3F1B_679C),
        (0x403C_70A5, 0x3F15_30F7),
        (0x404B_8318, 0x3F0F_8F7A),
        (0x4057_1727, 0x3F0B_A4A1),
        (0x4060_8753, 0x3F08_AD35),
        (0x406A_671A, 0x3F05_C453),
        (0x4073_B7E4, 0x3F03_2F7E),
        (0x407F_EE87, 0x3F00_045F),
        (0x07EE_C200, 0x5B3B_7111),
        (0x112E_0100, 0x569B_41BA),
        (0x1A78_9300, 0x5201_E5D9),
        (0x23BD_B700, 0x4D52_473A),
        (0x2CF7_8500, 0x48B8_1821),
        (0x3637_C200, 0x4417_147D),
        (0x3F7F_E600, 0x3F80_0681),
        (0x4080_1000, 0x3EFF_F002),
        (0x497F_E600, 0x3A80_0681),
        (0x527F_E600, 0x3600_0681),
        (0x5B7F_D300, 0x3180_0B42),
        (0x647F_D300, 0x2D00_0B42),
        (0x6D7F_BB00, 0x2880_1144),
        (0x767F_BB00, 0x2400_1144),
        (0x7F7F_BB00, 0x1F80_1144),
        (0x0000_0000, 0x7F80_0000),
        (0x0000_0100, 0x7F80_0000),
        (0x006C_E400, 0x7F80_0000),
        (0x7F80_0000, 0x0000_0000),
    ];
    for (y, w) in measured {
        assert_eq!(
            rsqrt_model(f32::from_bits(y)).to_bits(),
            w,
            "{y:#010x}: the model against Metal's word"
        );
    }
    let f = f32s();
    for fma in [false, true] {
        check_module(
            &format!("FastRsqrt fma={fma} at Metal's measured words"),
            routine_module(Fun::FastRsqrt, false, fma),
            &[f],
            f,
            &measured
                .iter()
                .map(|(y, _)| vec![vec![u64::from(*y)]])
                .collect::<Vec<Case>>(),
            &|c| {
                measured
                    .iter()
                    .find(|(y, _)| u64::from(*y) == c[0][0])
                    .map(|(_, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
    }
}

/// [`Fun::FastRsqrt`]'s routine answers [`rsqrt_model`] bit for bit, with and without `OpFmaKHR`,
/// over positive magnitudes from `2^-126` to `2^128`, around the powers of 2 it reduces by, every
/// 65,537th `float` of `[1, 4)`, and its specials: `+inf` of a zero and a denormal, `+0` of `+inf`,
/// and a NaN of a NaN.
#[test]
fn metal_fast_rsqrt_routine_answers_the_model() {
    const ANY_NAN: u64 = 0x7FC0_0000;
    let mut rng = Rng(0x2591_2591);
    let mut xs = magnitudes(&mut rng, -126.0, 128.0, 3000);
    for e in [-126, -125, -2, -1, 0, 1, 2, 3, 125, 126, 127] {
        xs.extend(around(2f32.powi(e), 2));
    }
    xs.extend((0x3F80_0000u32..0x4080_0000).step_by(65_537).map(u64::from));
    xs.retain(|b| f32v(*b) > 0.0);
    xs.extend(DENORMALS.iter().map(|b| b & 0x7FFF_FFFF));
    xs.extend([0, 0x7F7F_FFFF, 0x7F80_0000, 0x7FC0_0000]);
    let f = f32s();
    for fma in [false, true] {
        check_module(
            &format!("FastRsqrt fma={fma}"),
            routine_module(Fun::FastRsqrt, false, fma),
            &[f],
            f,
            &xs.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
            &|c| {
                let y = f32v(c[0][0]);
                Some(vec![if y.is_nan() {
                    ANY_NAN
                } else {
                    u64::from(rsqrt_model(y).to_bits())
                }])
            },
            &|g, w| g == w || (w == ANY_NAN && f32v(g).is_nan()),
        );
    }
}

/// `1 + s q(s)` of Metal's coefficients `q` (`libm::hyper`), each step one fused multiply-add.
fn near_zero_model(q: &[u32], s: f32) -> f32 {
    horner_model(q, s).mul_add(s, 1.0)
}

/// Metal's precise `sinh`, `cosh`, or `tanh` of `x` with its reciprocals the
/// model's ([`rcp_model`]): the polynomials near zero, `(E -+ 1/E) / 2` of its precise
/// `exp` ([`exp_model`]) and `e^a / 2` ([`exp_composed`]) past `12 ln 2`, `tanh`'s `(t - 1) (1 /
/// (t + 1))` of `t = e^(2a)`, and its polynomial from 0.35 to 1.25.
fn hyper_model(fun: Fun, xb: u32) -> u32 {
    use libm::hyper::*;
    let x = f32::from_bits(xb);
    if x.is_nan() {
        return xb;
    }
    let a = x.abs();
    let ab = a.to_bits();
    let s = a * a;
    let r = match fun {
        Fun::Sinh | Fun::Cosh => {
            let sinh = fun == Fun::Sinh;
            if ab <= if sinh { SINH_TINY } else { COSH_TINY } {
                if sinh {
                    a
                } else {
                    1.0
                }
            } else if ab < POLY_END {
                let p = near_zero_model(if sinh { &SINH_Q } else { &COSH_Q }, s);
                if sinh {
                    a * p
                } else {
                    p
                }
            } else {
                let h = f32::from_bits(exp_composed(ab, std::f64::consts::LOG2_E, true));
                if ab > HALF_EXP {
                    h
                } else {
                    let big = h + h;
                    let inv = rcp_model(big);
                    0.5 * if sinh { big - inv } else { big + inv }
                }
            }
        }
        _ => {
            if ab <= TANH_TINY {
                a
            } else if ab < TANH_POLY_END {
                a * near_zero_model(&TANH_Q, s)
            } else if ab <= TANH_QUOTIENT {
                horner_model(&TANH_MIDDLE, a)
            } else if ab >= HALF_EXP {
                1.0
            } else {
                let t = f32::from_bits(exp_model((a + a).to_bits(), std::f64::consts::LOG2_E));
                (t - 1.0) * rcp_model(t + 1.0)
            }
        }
    };
    if fun == Fun::Cosh { r } else { r.copysign(x) }.to_bits()
}

/// The `sinh`, `cosh`, and `tanh` routines answer [`hyper_model`] bit for bit over the
/// exponentials' operands, every region's edges, and uniform operands in each region: part `part`
/// of `of` of the runs of 2,048 operands.
fn hyperbolics_are_metals(part: usize, of: usize) {
    use libm::hyper::*;
    let mut rng = Rng(0x4791_4791);
    let mut xs = exp_operands(0x51A5_51A5);
    for edge in [
        SINH_TINY,
        COSH_TINY,
        TANH_TINY,
        POLY_END,
        TANH_POLY_END,
        TANH_QUOTIENT,
        HALF_EXP,
        0x42B1_7218,
        0x42B2_D4FC,
    ] {
        xs.extend(around(f32::from_bits(edge), 3));
        xs.extend(around(-f32::from_bits(edge), 3));
    }
    for (lo, hi) in [
        (-0.35, 0.35),
        (-1.5, 1.5),
        (0.35, 1.25),
        (1.25, 8.4),
        (-90.0, 90.0),
    ] {
        xs.extend((0..1500).map(|_| uniform(&mut rng, lo, hi)));
    }
    xs.extend(magnitudes(&mut rng, -12.0, -1.0, 1500));
    xs.retain(|b| !f32v(*b).is_nan());
    let mut runs = Vec::new();
    for fun in [Fun::Sinh, Fun::Cosh, Fun::Tanh] {
        for run in xs.chunks(2048) {
            runs.push((fun, run.to_vec()));
        }
    }
    let f = f32s();
    for (fun, run) in part_of(runs, part, of) {
        check_module(
            &format!("{fun:?}"),
            routine_module(fun, false, false),
            &[f],
            f,
            &run.iter().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
            &|c| Some(vec![u64::from(hyper_model(fun, c[0][0] as u32))]),
            &|g, w| g == w,
        );
    }
}

mod hyperbolic_routines_are_metals {
    parts!(super::hyperbolics_are_metals, 6;
        part_1_of_6 = 0, part_2_of_6 = 1, part_3_of_6 = 2, part_4_of_6 = 3,
        part_5_of_6 = 4, part_6_of_6 = 5);
}

/// The `sinh`, `cosh`, and `tanh` routines answer Metal's words, measured on the Apple M4, at each
/// region's edges and inside it, both signs: the operand itself to the tiny bound and the
/// polynomials' first word past it, the polynomials, the compositions of the precise `exp`, among
/// them words where the correctly rounded reciprocal is one unit off and Metal's is not, and
/// `e^a / 2` and `tanh`'s 1 from `12 ln 2`; and `tanh`'s polynomial from 0.35 to 1.25 at words
/// measured on the M3 where Metal's word is not the correctly rounded one.
#[test]
fn hyperbolic_routines_answer_metals_measured_words() {
    let measured: [(Fun, u32, u32); 54] = [
        (Fun::Sinh, 0x3A85_2018, 0x3A85_2018),
        (Fun::Sinh, 0x3A85_2019, 0x3A85_201A),
        (Fun::Sinh, 0x3D00_0000, 0x3D00_0555),
        (Fun::Sinh, 0x3F00_0000, 0x3F05_6680),
        (Fun::Sinh, 0x3FBF_FFFF, 0x4008_461C),
        (Fun::Sinh, 0x3FC0_0000, 0x4008_461E),
        (Fun::Sinh, 0x4000_0000, 0x4068_1E7B),
        (Fun::Sinh, 0x4080_0000, 0x41DA_51C1),
        (Fun::Sinh, 0x4105_1592, 0x44FF_FFFF),
        (Fun::Sinh, 0x4105_1593, 0x4500_0009),
        (Fun::Sinh, 0x4180_0000, 0x4A87_975F),
        (Fun::Sinh, 0x42B0_0000, 0x7E78_82B7),
        (Fun::Sinh, 0xBF00_0000, 0xBF05_6680),
        (Fun::Sinh, 0xC000_0000, 0xC068_1E7B),
        (Fun::Sinh, 0x3FD6_533D, 0x4024_BE84),
        (Fun::Sinh, 0x400C_E2F5, 0x408E_D33F),
        (Fun::Sinh, 0x4085_0099, 0x41FF_4864),
        (Fun::Cosh, 0x3A35_04F2, 0x3F80_0000),
        (Fun::Cosh, 0x3A35_04F3, 0x3F80_0002),
        (Fun::Cosh, 0x3E80_0000, 0x3F84_0558),
        (Fun::Cosh, 0x3F80_0000, 0x3FC5_83AA),
        (Fun::Cosh, 0x3FBF_FFFF, 0x4016_8DE0),
        (Fun::Cosh, 0x3FC0_0000, 0x4016_8DE2),
        (Fun::Cosh, 0x4040_0000, 0x4121_1525),
        (Fun::Cosh, 0x4105_1592, 0x4500_0000),
        (Fun::Cosh, 0x4105_1593, 0x4500_0009),
        (Fun::Cosh, 0x4280_0000, 0x6D21_2CC2),
        (Fun::Cosh, 0xC040_0000, 0x4121_1525),
        (Fun::Cosh, 0x3FD6_533D, 0x4030_BD2C),
        (Fun::Cosh, 0x400F_0487, 0x4097_343D),
        (Fun::Cosh, 0x4085_0099, 0x41FF_687A),
        (Fun::Tanh, 0x3A38_9BA2, 0x3A38_9BA2),
        (Fun::Tanh, 0x3A38_9BA3, 0x3A38_9BA1),
        (Fun::Tanh, 0x3E00_0000, 0x3DFE_ACC9),
        (Fun::Tanh, 0x3EB3_3333, 0x3EAC_396A),
        (Fun::Tanh, 0x3FA0_0001, 0x3F59_291E),
        (Fun::Tanh, 0x3FC0_0000, 0x3F67_B7CC),
        (Fun::Tanh, 0x4040_0000, 0x3F7E_BBE8),
        (Fun::Tanh, 0x4080_0000, 0x3F7F_D40C),
        (Fun::Tanh, 0x4105_1591, 0x3F7F_FFFE),
        (Fun::Tanh, 0x4105_1592, 0x3F80_0000),
        (Fun::Tanh, 0xC000_0000, 0xBF76_CA84),
        (Fun::Tanh, 0x3FC1_2FA0, 0x3F68_249E),
        (Fun::Tanh, 0x402A_AB55, 0x3F7D_8A49),
        (Fun::Tanh, 0x40AA_DFAA, 0x3F7F_FCFD),
        (Fun::Tanh, 0x4101_CA7D, 0x3F7F_FFFE),
        (Fun::Tanh, 0x3EB3_3336, 0x3EAC_396C),
        (Fun::Tanh, 0x3ED0_CCCC, 0x3EC5_F23A),
        (Fun::Tanh, 0x3EEE_6666, 0x3EDE_8B71),
        (Fun::Tanh, 0x3F0C_0005, 0x3EFF_10C0),
        (Fun::Tanh, 0x3F29_999A, 0x3F14_7C80),
        (Fun::Tanh, 0x3F47_3333, 0x3F26_D12B),
        (Fun::Tanh, 0x3F64_CCCC, 0x3F36_96F5),
        (Fun::Tanh, 0x3F82_6666, 0x3F44_F492),
    ];
    let f = f32s();
    for fun in [Fun::Sinh, Fun::Cosh, Fun::Tanh] {
        let mine: Vec<(u32, u32)> = measured
            .iter()
            .filter(|(g, _, _)| *g == fun)
            .map(|(_, x, w)| (*x, *w))
            .collect();
        for (x, w) in &mine {
            assert_eq!(hyper_model(fun, *x), *w, "{fun:?}'s model at {x:#010x}");
        }
        check_module(
            &format!("{fun:?} at Metal's measured words"),
            routine_module(fun, false, false),
            &[f],
            f,
            &mine
                .iter()
                .map(|(x, _)| vec![vec![u64::from(*x)]])
                .collect::<Vec<Case>>(),
            &|c| {
                mine.iter()
                    .find(|(x, _)| u64::from(*x) == c[0][0])
                    .map(|(_, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
    }
}

/// Every second-stage routine answers bit for bit the same with `OpFmaKHR` for a product's exact
/// error as with Dekker's product, over the accuracy tests' operands: each product whose error an
/// answer depends on lies above `2^-100`, where both are the same exact `float`. Part `part` of
/// `of` of the runs of 1,024 operands.
fn same_with_fma(part: usize, of: usize) {
    let xs = exp_operands(0x0F3A_0F3A);
    let ls = log_operands(0x0F3B_0F3B);
    let mut rng = Rng(0x0F3C_0F3C);
    let pairs: Vec<Vec<u64>> = (0..4000)
        .map(|_| {
            let x = ls[(rng.next() as usize) % ls.len()];
            let l = f64::from(f32v(x)).log2();
            let t = f64::from(f32v(uniform(&mut rng, -140.0, 130.0)));
            let y = if l == 0.0 { t } else { t / l };
            vec![x, u64::from((y as f32).to_bits())]
        })
        .collect();
    let mut runs = Vec::new();
    for fun in [
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
    ] {
        let operands = match fun {
            Fun::Powr => pairs.clone(),
            Fun::Log2 | Fun::Log | Fun::Log10 => unary(&ls),
            _ => unary(&xs),
        };
        let cases: Vec<Case> = operands
            .iter()
            .map(|c| c.iter().map(|b| vec![*b]).collect())
            .collect();
        for run in cases.chunks(1024) {
            runs.push((fun, run.to_vec()));
        }
    }
    let f = f32s();
    let mut differ = Vec::new();
    for (fun, run) in part_of(runs, part, of) {
        let ins = vec![f; fun.arity()];
        let plain = run_module("plain", routine_module(fun, false, false), &ins, f, &run);
        let fused = run_module("fused", routine_module(fun, false, true), &ins, f, &run);
        for ((c, p), q) in run.iter().zip(&plain).zip(&fused) {
            if p != q {
                differ.push(format!("{fun:?} {c:x?}: {p:x?} without, {q:x?} with"));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

mod second_stage_routines_answer_the_same_with_fma {
    parts!(super::same_with_fma, 16;
        part_1_of_16 = 0, part_2_of_16 = 1, part_3_of_16 = 2, part_4_of_16 = 3,
        part_5_of_16 = 4, part_6_of_16 = 5, part_7_of_16 = 6, part_8_of_16 = 7,
        part_9_of_16 = 8, part_10_of_16 = 9, part_11_of_16 = 10, part_12_of_16 = 11,
        part_13_of_16 = 12, part_14_of_16 = 13, part_15_of_16 = 14, part_16_of_16 = 15);
}

/// Signed zeros, denormals, infinities, the largest finite floats, and the smallest normals.
const ZERO: u64 = 0x0000_0000;
const NEG_ZERO: u64 = 0x8000_0000;
const INF: u64 = 0x7F80_0000;
const NEG_INF: u64 = 0xFF80_0000;
const MAX: u64 = 0x7F7F_FFFF;
const NEG_MAX: u64 = 0xFF7F_FFFF;
const MIN: u64 = 0x0080_0000;
const NEG_MIN: u64 = 0x8080_0000;
const ONE: u64 = 0x3F80_0000;
const NEG_ONE: u64 = 0xBF80_0000;
const NAN: u64 = 0x7FC0_0000;

/// The operands whose answer is the operand itself to Metal's odd functions below `2^-12`.
const TINY: [u64; 6] = [ZERO, NEG_ZERO, MIN, NEG_MIN, DENORMALS[0], DENORMALS[3]];

/// Metal's precise answers at the special operands (case 7b283404, every such lane), bit for bit:
/// zeros, denormals, and the smallest normals; infinities; the largest finite floats where the
/// answer is infinite or `±1`; `±1` where Metal's answer is exact; and its NaN `0x7fc00000` at a
/// domain error. A NaN operand gives a NaN. (Metal's `asinh(±FLT_MAX)` and `acosh(FLT_MAX)`,
/// `±0x42b2d4fb`, are 1.15 ulp below the exact value, and are its forms' words.) Part `part` of
/// `of` of the routines, each over all of its special operands in one run.
fn metals_special_values(part: usize, of: usize) {
    let mut specials: Vec<(Fun, u64, u64)> = Vec::new();
    for fun in [Fun::Exp2, Fun::Exp, Fun::Exp10] {
        for x in [ZERO, NEG_ZERO, MIN, NEG_MIN].into_iter().chain(DENORMALS) {
            specials.push((fun, x, ONE));
        }
        specials.extend([
            (fun, INF, INF),
            (fun, NEG_INF, ZERO),
            (fun, MAX, INF),
            (fun, NEG_MAX, ZERO),
        ]);
    }
    specials.extend([
        (Fun::Exp2, ONE, 0x4000_0000),
        (Fun::Exp2, NEG_ONE, 0x3F00_0000),
        (Fun::Exp2, u64::from((-126f32).to_bits()), MIN),
        (Fun::Exp2, u64::from((-127f32).to_bits()), ZERO),
        (Fun::Exp2, u64::from(128f32.to_bits()), INF),
        (Fun::Exp10, NEG_ONE, 0x3DCC_CCCD),
        // A denormal result is flushed (case 7b283404: `exp(-87.34)` is `+0`).
        (Fun::Exp, u64::from((-87.34f32).to_bits()), ZERO),
    ]);
    for fun in [Fun::Log2, Fun::Log, Fun::Log10] {
        for x in [ZERO, NEG_ZERO].into_iter().chain(DENORMALS) {
            specials.push((fun, x, NEG_INF));
        }
        specials.extend([
            (fun, INF, INF),
            (fun, ONE, ZERO),
            (fun, NEG_ONE, NAN),
            (fun, NEG_MIN, NAN),
            (fun, NEG_MAX, NAN),
            (fun, NEG_INF, NAN),
        ]);
    }
    specials.push((Fun::Log2, MIN, 0xC2FC_0000));
    for x in TINY.into_iter().chain(DENORMALS) {
        specials.push((Fun::Sinh, x, x));
        specials.push((Fun::Tanh, x, x));
        specials.push((Fun::Asinh, x, x));
        specials.push((Fun::Atanh, x, x));
        specials.push((Fun::Cosh, x, ONE));
        specials.push((Fun::Acosh, x, NAN));
    }
    specials.extend([
        (Fun::Sinh, INF, INF),
        (Fun::Sinh, NEG_INF, NEG_INF),
        (Fun::Sinh, MAX, INF),
        (Fun::Sinh, NEG_MAX, NEG_INF),
        (Fun::Cosh, INF, INF),
        (Fun::Cosh, NEG_INF, INF),
        (Fun::Cosh, NEG_MAX, INF),
        (Fun::Tanh, INF, ONE),
        (Fun::Tanh, NEG_INF, NEG_ONE),
        (Fun::Tanh, MAX, ONE),
        (Fun::Tanh, NEG_MAX, NEG_ONE),
        (Fun::Asinh, INF, INF),
        (Fun::Asinh, NEG_INF, NEG_INF),
        (Fun::Asinh, MAX, 0x42B2_D4FB),
        (Fun::Asinh, NEG_MAX, 0xC2B2_D4FB),
        (Fun::Acosh, INF, INF),
        (Fun::Acosh, MAX, 0x42B2_D4FB),
        (Fun::Acosh, ONE, ZERO),
        (Fun::Acosh, NEG_ONE, NAN),
        (Fun::Acosh, NEG_INF, NAN),
        (Fun::Acosh, NEG_MAX, NAN),
        (Fun::Atanh, ONE, INF),
        (Fun::Atanh, NEG_ONE, NEG_INF),
        // To `0x3a400000` Metal's `atanh` is its operand.
        (Fun::Atanh, 0x39C0_0000, 0x39C0_0000),
        (Fun::Atanh, 0xB9FF_FF00, 0xB9FF_FF00),
        (Fun::Atanh, INF, NAN),
        (Fun::Atanh, NEG_INF, NAN),
        (Fun::Atanh, MAX, NAN),
        (Fun::Atanh, NEG_MAX, NAN),
    ]);
    // `powr`'s own answers at a zero, denormal, and infinite base (`math::powr_nan` answers the
    // NaNs around them), and exact powers.
    let two = u64::from(2f32.to_bits());
    let minus_two = u64::from((-2f32).to_bits());
    let powr = [
        (ZERO, two, ZERO),
        (ZERO, minus_two, INF),
        (DENORMALS[0], two, ZERO),
        (DENORMALS[0], minus_two, INF),
        (INF, two, INF),
        (INF, minus_two, ZERO),
        (two, u64::from(3f32.to_bits()), u64::from(8f32.to_bits())),
        (u64::from(4f32.to_bits()), u64::from(0.5f32.to_bits()), two),
        (two, NEG_ONE, u64::from(0.5f32.to_bits())),
        (two, u64::from(128f32.to_bits()), INF),
        (two, u64::from((-127f32).to_bits()), ZERO),
        (MAX, two, INF),
    ];
    let checks = vec![
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
    ];
    let f = f32s();
    for fun in part_of(checks, part, of) {
        if fun == Fun::Powr {
            check_module(
                "Powr at its special operands",
                routine_module(Fun::Powr, false, false),
                &[f, f],
                f,
                &powr
                    .iter()
                    .map(|(x, y, _)| vec![vec![*x], vec![*y]])
                    .collect::<Vec<Case>>(),
                &|c| {
                    powr.iter()
                        .find(|(x, y, _)| [*x, *y] == [c[0][0], c[1][0]])
                        .map(|(_, _, w)| vec![*w])
                },
                &|g, w| g == w,
            );
            continue;
        }
        let mut want: BTreeMap<u64, u64> = BTreeMap::new();
        for (_, x, w) in specials.iter().filter(|(g, _, _)| *g == fun) {
            let other = want.insert(*x, *w);
            assert!(
                other.is_none_or(|o| o == *w),
                "{fun:?}({x:#010x}) has two answers"
            );
        }
        check_module(
            &format!("{fun:?} at its special operands"),
            routine_module(fun, false, false),
            &[f],
            f,
            &want.keys().map(|x| vec![vec![*x]]).collect::<Vec<Case>>(),
            &|c| want.get(&c[0][0]).map(|w| vec![*w]),
            &|g, w| g == w,
        );
        // A NaN operand gives a NaN.
        check_module(
            &format!("{fun:?}(NaN)"),
            routine_module(fun, false, false),
            &[f],
            f,
            &[vec![vec![NAN]], vec![vec![0xFFC0_1234]]],
            &|_| Some(vec![NAN]),
            &|g, _| f32v(g).is_nan(),
        );
    }
}

mod exponential_and_hyperbolic_routines_answer_metals_special_values {
    parts!(super::metals_special_values, 4;
        part_1_of_4 = 0, part_2_of_4 = 1, part_3_of_4 = 2, part_4_of_4 = 3);
}

/// The module of float function `op` on form `f` whose routines `funs` are defined, for
/// [`math::float_fn`].
fn function_module(op: FloatFn, f: Form, funs: &[Fun]) -> TypedModule {
    kernel_defs(
        &vec![f; op.arity()],
        f,
        &|b| {
            funs.iter()
                .map(|fun| libm::define_alone(b, *fun, false))
                .collect()
        },
        &|fb, x, ids| {
            let map: BTreeMap<Fun, FuncId> =
                funs.iter().copied().zip(ids.iter().copied()).collect();
            math::float_fn(
                fb,
                op,
                f,
                Contraction::Forbidden,
                NanTest::IsNan,
                x,
                &Lib::new(None, &map),
            )
        },
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("{op:?} {f:?}: build: {e:?}"))
}

/// Metal's fast functions off a device whose instructions are Metal's are MSL Table 8.2's formulas
/// of the `exp2` and `log2` routines, which answer Metal's fast functions at the special operands
/// bit for bit (case f6524207, every such lane): a zero, a denormal, and the smallest normal give
/// `exp`'s 1 and the others' `+0`, whatever the sign; `fast::asinh(-inf)` is NaN (`-inf + inf`),
/// and `fast::asinh(-FLT_MAX)` and `fast::acosh(-FLT_MAX)` are `+inf`.
#[test]
fn fast_formulas_answer_metals_special_values() {
    let f = f32s();
    let small = [ZERO, NEG_ZERO, MIN, NEG_MIN, DENORMALS[0], DENORMALS[3]];
    let mut specials: Vec<(FloatFn, u64, u64)> = Vec::new();
    for x in small {
        for op in [FloatFn::FastExp, FloatFn::FastExp10, FloatFn::FastCosh] {
            specials.push((op, x, ONE));
        }
        for op in [FloatFn::FastSinh, FloatFn::FastAsinh] {
            specials.push((op, x, ZERO));
        }
        specials.push((FloatFn::FastAcosh, x, NAN));
    }
    for x in [ZERO, NEG_ZERO].into_iter().chain(DENORMALS) {
        specials.push((FloatFn::FastLog, x, NEG_INF));
        specials.push((FloatFn::FastLog10, x, NEG_INF));
    }
    specials.extend([
        (FloatFn::FastExp, INF, INF),
        (FloatFn::FastExp, NEG_INF, ZERO),
        (FloatFn::FastExp, MAX, INF),
        (FloatFn::FastExp, NEG_MAX, ZERO),
        (FloatFn::FastExp10, INF, INF),
        (FloatFn::FastExp10, NEG_INF, ZERO),
        (FloatFn::FastLog, INF, INF),
        (FloatFn::FastLog, ONE, ZERO),
        (FloatFn::FastLog, NEG_ONE, NAN),
        (FloatFn::FastLog, NEG_INF, NAN),
        (FloatFn::FastLog, MIN, 0xC2AE_AC50),
        (FloatFn::FastLog10, INF, INF),
        (FloatFn::FastLog10, ONE, ZERO),
        (FloatFn::FastLog10, NEG_MAX, NAN),
        (FloatFn::FastSinh, INF, INF),
        (FloatFn::FastSinh, NEG_INF, NEG_INF),
        (FloatFn::FastSinh, MAX, INF),
        (FloatFn::FastSinh, NEG_MAX, NEG_INF),
        (FloatFn::FastCosh, INF, INF),
        (FloatFn::FastCosh, NEG_INF, INF),
        (FloatFn::FastCosh, MAX, INF),
        (FloatFn::FastAsinh, INF, INF),
        (FloatFn::FastAsinh, NEG_INF, NAN),
        (FloatFn::FastAsinh, MAX, INF),
        (FloatFn::FastAsinh, NEG_MAX, INF),
        (FloatFn::FastAcosh, INF, INF),
        (FloatFn::FastAcosh, NEG_INF, NAN),
        (FloatFn::FastAcosh, MAX, INF),
        (FloatFn::FastAcosh, NEG_MAX, INF),
        (FloatFn::FastAcosh, ONE, ZERO),
        (FloatFn::FastAcosh, NEG_ONE, NAN),
    ]);
    let ops: Vec<FloatFn> = {
        let mut ops: Vec<FloatFn> = specials.iter().map(|(op, _, _)| *op).collect();
        ops.dedup();
        ops.sort_by_key(|op| format!("{op:?}"));
        ops.dedup();
        ops
    };
    for op in ops {
        let funs = libm::uses(op, f);
        let cases: Vec<(u64, u64)> = specials
            .iter()
            .filter(|(o, _, _)| *o == op)
            .map(|(_, x, w)| (*x, *w))
            .collect();
        check_module(
            &format!("{op:?} at its special operands"),
            function_module(op, f, funs),
            &[f],
            f,
            &cases
                .iter()
                .map(|(x, _)| vec![vec![*x]])
                .collect::<Vec<Case>>(),
            &|c| {
                let x = c[0][0];
                cases.iter().find(|(b, _)| *b == x).map(|(_, w)| vec![*w])
            },
            &|g, w| g == w || (w == NAN && f32v(g).is_nan()),
        );
    }
}

/// `fast::atanh` off a device whose instructions are Metal's is `0.5 · log((1 + x) · rcp(1 - x))`
/// of the `log2` routine and Metal's fast division, by either reciprocal, which answers Metal's
/// fast `atanh` at the special operands bit for bit (case f6524207, every such lane): `+0` for a
/// zero, a denormal, and the smallest normal, `±inf` at `±1`, NaN for an infinity, and `-inf` for
/// `±FLT_MAX`, whose divisor's reciprocal Metal flushes.
#[test]
fn fast_atanh_is_the_formula_with_the_fast_division() {
    let f = f32s();
    let mut cases: Vec<(u64, u64)> = [ZERO, NEG_ZERO, MIN, NEG_MIN, DENORMALS[0], DENORMALS[3]]
        .into_iter()
        .map(|x| (x, ZERO))
        .collect();
    cases.extend([
        (ONE, INF),
        (NEG_ONE, NEG_INF),
        (INF, NAN),
        (NEG_INF, NAN),
        (MAX, NEG_INF),
        (NEG_MAX, NEG_INF),
    ]);
    for reciprocal in [Reciprocal::Device, Reciprocal::Rounded] {
        let op = FloatFn::FastAtanh { reciprocal };
        let division = math::FastDivision::new(f, reciprocal, Contraction::Forbidden);
        let m = kernel_defs(
            &[f],
            f,
            &|b| {
                Ok(vec![
                    libm::define_alone(b, Fun::Log2, false)?,
                    math::define_fast_division(b, division)?,
                ])
            },
            &|fb, x, ids| {
                let map: BTreeMap<Fun, FuncId> = [(Fun::Log2, ids[0])].into_iter().collect();
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    math::call_fast_division(fb, ids[1], p, q)
                };
                math::fast_atanh(fb, op, f, x[0], &Lib::new(None, &map), &divide)
            },
            true,
            false,
        )
        .unwrap_or_else(|e| panic!("{op:?}: build: {e:?}"));
        check_module(
            &format!("{op:?} at its special operands"),
            m,
            &[f],
            f,
            &cases
                .iter()
                .map(|(x, _)| vec![vec![*x]])
                .collect::<Vec<Case>>(),
            &|c| {
                let x = c[0][0];
                cases.iter().find(|(b, _)| *b == x).map(|(_, w)| vec![*w])
            },
            &|g, w| g == w || (w == NAN && f32v(g).is_nan()),
        );
    }
}

/// A `half` call of a second-stage function takes the `float` routine before its rounding, so it is
/// within a `half` ulp of the exact value over every fifth finite pattern (in runs within the
/// interpreter's step budget) whose result is a finite `half`: part `part` of `of` of the runs.
fn half_takes_the_routines(part: usize, of: usize) {
    let h = float(16);
    let cases: Vec<Case> = (0..=0xFFFFu64)
        .step_by(5)
        .filter(|b| b & 0x7C00 != 0x7C00)
        .map(|b| vec![vec![b]])
        .collect();
    let functions: [(FloatFn, fn(f64) -> f64); 12] = [
        (FloatFn::Exp2, f64::exp2),
        (FloatFn::Exp, f64::exp),
        (FloatFn::Exp10, |x| 10f64.powf(x)),
        (FloatFn::Log2, f64::log2),
        (FloatFn::Log, f64::ln),
        (FloatFn::Log10, f64::log10),
        (FloatFn::Sinh, f64::sinh),
        (FloatFn::Cosh, f64::cosh),
        (FloatFn::Tanh, f64::tanh),
        (FloatFn::Asinh, f64::asinh),
        (FloatFn::Acosh, f64::acosh),
        (FloatFn::Atanh, f64::atanh),
    ];
    let reference = |exact: fn(f64) -> f64, c: &Case| {
        let r = exact(f16_to_f64(c[0][0] as u16));
        // Finite `half` results; the infinities and NaNs are the `float` routine's.
        (r.is_finite() && r.abs() <= 65504.0).then(|| vec![u64::from(f64_to_f16(r))])
    };
    let mut runs = Vec::new();
    for (op, exact) in functions {
        let defined: Vec<Case> = cases
            .iter()
            .filter(|c| reference(exact, c).is_some())
            .cloned()
            .collect();
        for run in defined.chunks(2048) {
            runs.push((op, exact, run.to_vec()));
        }
    }
    for (op, exact, run) in part_of(runs, part, of) {
        check_module(
            &format!("half {op:?}"),
            function_module(op, h, libm::uses(op, h)),
            &[h],
            h,
            &run,
            &|c| reference(exact, c),
            &|g, w| {
                let nan = |b: u64| b & 0x7C00 == 0x7C00 && b & 0x03FF != 0;
                if nan(g) {
                    return false;
                }
                // A zero answer is exact up to its sign; another is within one `half` ulp of the
                // same sign.
                if w & 0x7FFF == 0 || g & 0x7FFF == 0 {
                    return g & 0x7FFF == w & 0x7FFF;
                }
                (g ^ w) & 0x8000 == 0 && (g as i64 - w as i64).abs() <= 1
            },
        );
    }
}

mod half_exponentials_and_hyperbolics_take_the_routines {
    parts!(super::half_takes_the_routines, 4;
        part_1_of_4 = 0, part_2_of_4 = 1, part_3_of_4 = 2, part_4_of_4 = 3);
}

/// Metal's `half` functions that are not its precise `float` functions rounded
/// (`math::metal_half`, `half_tanh`, `half_small`, `half_cospi_zero`, and `sinpi(-0)`): Metal's
/// words at measured operands of our probe (the M4's, are the same), those where the precise
/// routines rounded were not Metal's among them, the edges of the operand ranges each answers by,
/// and others; and `tanpi`, its `float` form rounded. The binary functions' second operand is the
/// probe's, `(i * 2654435761) >> 16` of the first's bits `i` (32-bit arithmetic).
#[test]
fn half_functions_answer_metals_measured_words() {
    let h = float(16);
    let unary: [(FloatFn, &[(u16, u16)]); 11] = [
        (
            FloatFn::Tanh,
            &[
                (0x2744, 0x2743),
                (0x2745, 0x2745),
                (0x2748, 0x2748),
                (0x2894, 0x2893),
                (0x2895, 0x2895),
                (0x2D86, 0x2D83),
                (0xA746, 0xA746),
                (0x0A68, 0x0A68),
                (0x0C01, 0x0C01),
                (0x4C00, 0x3C00),
                (0x5A00, 0x3C00),
                (0xFC00, 0xBC00),
                (0x3C00, 0x3A18),
                (0xC400, 0xBBFF),
            ],
        ),
        (
            FloatFn::Sinh,
            &[
                (0x321B, 0x3224),
                (0x4758, 0x620A),
                (0xB21B, 0xB224),
                (0xC758, 0xE20A),
                (0x0A68, 0x0A68),
                (0x0A71, 0x0A71),
                (0x0C00, 0x0C00),
                (0x3C00, 0x3CB3),
            ],
        ),
        (
            FloatFn::Cosh,
            &[
                (0x3DEF, 0x40A3),
                (0xBDEF, 0x40A3),
                (0x3C00, 0x3E2C),
                (0x4900, 0x7161),
            ],
        ),
        (
            FloatFn::CosPi,
            &[
                (0x3800, 0x8000),
                (0x3E00, 0x0000),
                (0x4100, 0x8000),
                (0x4300, 0x0000),
                (0xB800, 0x0000),
                (0xBE00, 0x8000),
                (0xC100, 0x0000),
                (0x6200, 0x3C00),
                (0x3400, 0x39A8),
            ],
        ),
        (
            FloatFn::SinPi,
            &[
                (0x8000, 0x8000),
                (0x0000, 0x0000),
                (0x3C00, 0x8000),
                (0x4000, 0x0000),
                (0x4200, 0x8000),
                (0xBC00, 0x8000),
                (0xC000, 0x0000),
                (0x3800, 0x3C00),
            ],
        ),
        // Metal's `float` `tanpi` of the widened operand, rounded.
        (
            FloatFn::TanPi,
            &[
                (0x0000, 0x0000),
                (0x8000, 0x0000),
                (0x0001, 0x0003),
                (0x3400, 0x3C00),
                (0x3800, 0x7C00),
                (0xB800, 0xFC00),
                (0x3C00, 0x0000),
                (0x3E00, 0xFC00),
                (0x3555, 0x3EED),
                (0x3BFF, 0x9648),
                (0x37FF, 0x6518),
                (0x7C00, 0x7E00),
            ],
        ),
        (
            FloatFn::Exp,
            &[
                (0x3C00, 0x4170),
                (0x4A00, 0x7C00),
                (0xC880, 0x080B),
                (0x2E66, 0x3C6C),
                (0x0441, 0x3C00),
            ],
        ),
        (
            FloatFn::Exp10,
            &[
                (0x3C00, 0x4900),
                (0x4000, 0x5640),
                (0xBC00, 0x2E66),
                (0x3555, 0x404F),
            ],
        ),
        (
            FloatFn::Log,
            &[
                (0x3C01, 0x13FF),
                (0x4A00, 0x40F8),
                (0x0400, 0xC8DA),
                (0x3BFF, 0x9000),
            ],
        ),
        (
            FloatFn::Log2,
            &[
                (0x3C01, 0x15C5),
                (0x4A00, 0x432C),
                (0x0001, 0xCE00),
                (0x3BFF, 0x91C6),
            ],
        ),
        (
            FloatFn::Log10,
            &[
                (0x3C01, 0x0EF2),
                (0x4A00, 0x3C51),
                (0x0400, 0xC437),
                (0x7BFF, 0x44D1),
            ],
        ),
    ];
    for (op, words) in unary {
        let cases: Vec<Case> = words
            .iter()
            .map(|(x, _)| vec![vec![u64::from(*x)]])
            .collect();
        check_module(
            &format!("half {op:?}"),
            function_module(op, h, libm::uses(op, h)),
            &[h],
            h,
            &cases,
            &|c| {
                words
                    .iter()
                    .find(|(x, _)| u64::from(*x) == c[0][0])
                    .map(|(_, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
    }
    let pairs: [(u16, u16, u16); 7] = [
        (0x07D1, 0xAF9E, 0x41DC),
        (0x41D1, 0x4130, 0x4BF8),
        (0x4F23, 0xC023, 0x13FF),
        (0x3C00, 0x0085, 0x3C00),
        (0x4000, 0xDE6C, 0x0000),
        (0x4100, 0x15E5, 0x3C01),
        (0x3555, 0x049D, 0x3C00),
    ];
    let cases: Vec<Case> = pairs
        .iter()
        .map(|(x, y, _)| vec![vec![u64::from(*x)], vec![u64::from(*y)]])
        .collect();
    for op in [FloatFn::Pow, FloatFn::Powr] {
        check_module(
            &format!("half {op:?}"),
            function_module(op, h, libm::uses(op, h)),
            &[h, h],
            h,
            &cases,
            &|c| {
                pairs
                    .iter()
                    .find(|(x, y, _)| (u64::from(*x), u64::from(*y)) == (c[0][0], c[1][0]))
                    .map(|(_, _, w)| vec![u64::from(*w)])
            },
            &|g, w| g == w,
        );
    }
}

/// Metal's fast `pow` and `powr` (and its `half` ones, the same compositions) where `y · log2(x)`
/// is infinite answer IEEE's value whatever the composition gave there (`math::infinite_product`
/// over a composition that answered `1`, as the RTX 5080 did without `SignedZeroInfNanPreserve`):
/// Metal's word at case f6524207's special pairs and case 32931370's `half`
/// ones, and the composition's own value where the product is finite, a NaN, or an operand is one.
#[test]
fn an_infinite_product_answers_ieee_whatever_the_composition_gave() {
    let (f, h) = (f32s(), float(16));
    let one = u64::from(1.0f32.to_bits());
    let one_h = 0x3C00;
    // (x, y, answer); `None`: the composition's value kept.
    let words: [(u64, u64, Option<u64>); 15] = [
        (0x0000_0000, 0x3F80_0000, Some(0)),
        (0x8000_0000, 0x4000_0000, Some(0)),
        (0x0000_0001, 0xBF80_0000, Some(0x7F80_0000)),
        (0x8000_0001, 0x3F00_0000, Some(0)),
        (0x0040_0000, 0x4100_0000, Some(0)),
        (0x7F80_0000, 0xC000_0000, Some(0)),
        (0x7F80_0000, 0x4100_0000, Some(0x7F80_0000)),
        (0x3F00_0000, 0x7F80_0000, Some(0)),
        (0x4000_0000, 0x7F80_0000, Some(0x7F80_0000)),
        (0x3F00_0000, 0xFF80_0000, Some(0x7F80_0000)),
        (0x4000_0000, 0xFF80_0000, Some(0)),
        (0x3F80_0000, 0x7F80_0000, None),
        (0x0000_0000, 0x0000_0000, None),
        (0x7FC0_0000, 0x4000_0000, None),
        (0x4000_0000, 0x4040_0000, None),
    ];
    let half_words: [(u64, u64, Option<u64>); 8] = [
        (0x0000, 0x3C00, Some(0)),
        (0x8000, 0x4000, Some(0)),
        (0x7C00, 0xC000, Some(0)),
        (0x7C00, 0x4800, Some(0x7C00)),
        (0x3800, 0x7C00, Some(0)),
        (0x4000, 0xFC00, Some(0)),
        (0x0001, 0x3C00, None),
        (0x3C00, 0x7C00, None),
    ];
    for (form, words, kept) in [(f, &words[..], one), (h, &half_words[..], one_h)] {
        let module = kernel_defs(
            &[form, form],
            form,
            &|_| Ok(Vec::new()),
            &|fb, x, _| {
                let r = crate::lower::arith::ops::fsplat(fb, form, 1.0)?;
                math::infinite_product(fb, form, x[1], x[0], r)
            },
            false,
            false,
        )
        .expect("build");
        let cases: Vec<Case> = words
            .iter()
            .map(|(x, y, _)| vec![vec![*x], vec![*y]])
            .collect();
        check_module(
            &format!("infinite product {form:?}"),
            module,
            &[form, form],
            form,
            &cases,
            &|c| {
                words
                    .iter()
                    .find(|(x, y, _)| (*x, *y) == (c[0][0], c[1][0]))
                    .map(|(_, _, w)| vec![w.unwrap_or(kept)])
            },
            &|g, w| g == w,
        );
    }
}

/// Each second-stage routine's words, with its constants and types, stay within [`Fun::words`] with
/// and without `OpFmaKHR` (the estimate at most twice the larger), and each function's call site
/// within [`FloatFn::words`] on `float` and `float4` (and on `half` and `half4` the ones Metal
/// answers by its own `half` functions, `math::metal_half`), on a device whose functions are its
/// own and on one whose functions are Metal's, precise or fast (`exp10`'s, `log10`'s, and the fast
/// module's `exp`'s compositions): the module's growth over one that defines the same routines and
/// returns its operand.
#[test]
fn second_stage_fits_its_word_estimates() {
    let module_words = |m: TypedModule| serialize(m).expect("serialize").words.len();
    let f = f32s();
    let mut off = Vec::new();
    for fun in [
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
    ] {
        let bound = fun.words() as usize;
        let mut largest = 0;
        for fma in [false, true] {
            let bare = module_words(helpers_module(fun, fma));
            let emitted = module_words(routine_module(fun, false, fma)) - bare;
            eprintln!("{fun:?} fma={fma}: {emitted} words against {bound}");
            if emitted > bound {
                off.push(format!("{fun:?} fma={fma}: {emitted} words over {bound}"));
            }
            largest = largest.max(emitted);
        }
        // The estimate covers the larger emission (without `OpFmaKHR`, each fused step is
        // emulated) and is not loose by more than half.
        if 2 * largest < bound {
            off.push(format!("{fun:?}: {bound} words for at most {largest}"));
        }
    }
    let h = float(16);
    let half_ops = [
        FloatFn::Exp,
        FloatFn::Exp10,
        FloatFn::Log2,
        FloatFn::Log,
        FloatFn::Log10,
        FloatFn::Powr,
        FloatFn::Pow,
        FloatFn::Sinh,
        FloatFn::Cosh,
        FloatFn::Tanh,
    ];
    for form in [f, vec(f, 4), h, vec(h, 4)] {
        for op in [
            FloatFn::Exp2,
            FloatFn::Exp,
            FloatFn::Exp10,
            FloatFn::FastExp,
            FloatFn::FastExp10,
            FloatFn::Log2,
            FloatFn::FastLog2,
            FloatFn::Log,
            FloatFn::Log10,
            FloatFn::FastLog,
            FloatFn::FastLog10,
            FloatFn::Powr,
            FloatFn::FastPowr,
            FloatFn::Pow,
            FloatFn::FastPow,
            FloatFn::Sinh,
            FloatFn::FastSinh,
            FloatFn::Cosh,
            FloatFn::FastCosh,
            FloatFn::Tanh,
            FloatFn::Asinh,
            FloatFn::FastAsinh,
            FloatFn::Acosh,
            FloatFn::FastAcosh,
            FloatFn::Atanh,
        ] {
            if form.float_fmt() == Some(FloatFmt::Half) && !half_ops.contains(&op) {
                continue;
            }
            let funs = libm::uses(op, form);
            let defined = |body: BodyCalling<'_>| {
                kernel_defs(
                    &vec![form; op.arity()],
                    form,
                    &|b| {
                        funs.iter()
                            .map(|fun| libm::define_alone(b, *fun, false))
                            .collect()
                    },
                    body,
                    false,
                    false,
                )
                .unwrap_or_else(|e| panic!("{op:?} {form:?}: build: {e:?}"))
            };
            let bare = module_words(defined(&|_, x, _| Ok(x[0])));
            for metal in [None, Some(libm::Mode::Precise), Some(libm::Mode::Fast)] {
                let used = module_words(defined(&|fb, x, ids| {
                    let map: BTreeMap<Fun, FuncId> =
                        funs.iter().copied().zip(ids.iter().copied()).collect();
                    math::float_fn(
                        fb,
                        op,
                        form,
                        Contraction::Forbidden,
                        NanTest::IsNan,
                        x,
                        &Lib::new(metal, &map),
                    )
                }));
                let (emitted, bound) = (used - bare, op.words(form) as usize);
                eprintln!(
                    "{op:?} x{} {metal:?}: {emitted} words against {bound}",
                    form.lanes
                );
                if emitted > bound {
                    off.push(format!(
                        "{op:?} x{} {metal:?}: {emitted} words over {bound}",
                        form.lanes
                    ));
                }
            }
        }
    }
    assert!(off.is_empty(), "{}", off.join("\n"));
}
