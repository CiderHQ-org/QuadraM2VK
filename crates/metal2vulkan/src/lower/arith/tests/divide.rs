//! Metal's precise division: both routine paths against Metal's quotient, on the interpreter, with
//! the device's estimate exact and perturbed to the edge of Vulkan's 2.5 ulp; and the reciprocal
//! step the fused path's proof rests on, over every significand.
//!
//! The reference is Metal's, as the probe read it: an `f32` denormal operand is a zero of its sign,
//! the quotient is rounded once to nearest even to 24 bits with an unbounded exponent (the host's
//! `f64` quotient of two `f32` values is never within an `f64` ulp of an `f32` rounding midpoint
//! without being one, so rounding it to 24 bits rounds once), a result below the smallest normal
//! after that rounding is a zero of its sign (probe case c10cba13), and a NaN is `0x7FC00000`.

use super::*;
use crate::lower::arith::divide::{self, Path, Rough, Routine};
use crate::lower::arith::legal::{f16_to_f64, f64_to_f16};
use crate::lower::arith::narrow;

/// `float` lanes of `lanes` lanes.
fn floats(lanes: u8) -> Form {
    vec(float(32), lanes)
}

/// An `f32` denormal as a zero of its sign.
fn ftz(x: u64) -> u64 {
    if x & 0x7F80_0000 == 0 {
        x & 0x8000_0000
    } else {
        x
    }
}

/// Metal's precise quotient of `f32` bits `x` and `y`.
fn metal_div32(x: u64, y: u64) -> u64 {
    let (x, y) = (ftz(x), ftz(y));
    let q = f64::from(f32v(x)) / f64::from(f32v(y));
    if q.is_nan() {
        return 0x7FC0_0000;
    }
    let smallest = 2f64.powi(-126);
    if q != 0.0 && q.abs() < smallest {
        // Rounded to 24 bits in `f32`'s normal range (the scaling is exact both ways), then
        // flushed unless the rounding reached the smallest normal.
        let scale = 2f64.powi(200);
        let rounded = f64::from((q * scale) as f32) / scale;
        if rounded.abs() < smallest {
            return u64::from(q.is_sign_negative()) << 31;
        }
        return u64::from((rounded as f32).to_bits());
    }
    u64::from((q as f32).to_bits())
}

/// `f32` bits of sign `s`, biased exponent `e`, and significand field `m`.
pub(super) fn bits(s: u64, e: u64, m: u64) -> u64 {
    (s << 31) | (e << 23) | m
}

/// Operand pairs: specials, every class against every other, the flush boundary (the quotient
/// `2^-126 - 2^-150`, which has 24 bits and is flushed although the denormal grid would round it
/// up to the smallest normal, and its neighbors), overflow, all-ones significands, divisors past
/// `2^126`, the edges of the range the fused path once kept to (`[2^-31, 2^33)`), divisors near
/// `2^40` (the probe's timing), quotients near rounding midpoints at moderate exponents and
/// at every exponent (where the fused path's exponent arithmetic decides the overflow and the
/// flush), and random bits.
fn pairs32() -> Vec<[u64; 2]> {
    let mut rng = Rng(0x2790_D1F1_0000_0001);
    let specials: Vec<u64> = [
        0,
        bits(1, 0, 0),
        bits(0, 0, 1),
        bits(1, 0, 0x7F_FFFF),
        bits(0, 1, 0),
        bits(0, 127, 0),
        bits(1, 127, 0x40_0000),
        bits(0, 127, 0x7F_FFFF),
        bits(0, 254, 0x7F_FFFF),
        bits(0, 253, 0),
        bits(0, 254, 0),
        bits(0, 95, 0x7F_FFFF),
        bits(0, 96, 0),
        bits(0, 159, 0x7F_FFFF),
        bits(0, 160, 0),
        bits(0, 255, 0),
        bits(1, 255, 0),
        bits(0, 255, 1),
        0x7FC0_0000,
    ]
    .to_vec();
    let mut out = Vec::new();
    for &x in &specials {
        for &y in &specials {
            out.push([x, y]);
        }
    }
    // (2 - 2^-23) * 2^a over 2^(a + 127): exactly 2^-126 - 2^-150, either sign.
    for a in [0u64, 1, 60, 126] {
        for s in 0..4u64 {
            out.push([bits(s & 1, 127 - a, 0x7F_FFFF), bits(s >> 1, 254 - a, 0)]);
            // Just below and above: a numerator one ulp off.
            out.push([bits(s & 1, 127 - a, 0x7F_FFFE), bits(s >> 1, 254 - a, 0)]);
            out.push([bits(s & 1, 128 - a, 0), bits(s >> 1, 254 - a, 0)]);
        }
    }
    // Overflow and the largest finite quotient.
    out.push([bits(0, 254, 0x7F_FFFF), bits(0, 126, 0x7F_FFFF)]);
    out.push([bits(0, 254, 0x7F_FFFF), bits(0, 127, 0)]);
    out.push([bits(1, 254, 0x40_0000), bits(0, 126, 0)]);
    // Quotients near a rounding midpoint: X * 2^t / Y within 2^-10 of a half.
    let mut near = 0;
    while near < 96 {
        let (mx, my) = (rng.next() & 0x7F_FFFF, rng.next() & 0x7F_FFFF);
        let (big_x, big_y) = (mx | 0x80_0000, my | 0x80_0000);
        let t = if big_x >= big_y { 23 } else { 24 };
        let r = (big_x << (t + 1)) % (2 * big_y);
        if r.abs_diff(big_y) * 1024 >= 2 * big_y {
            continue;
        }
        near += 1;
        let (ex, ey) = (64 + rng.next() % 128, 64 + rng.next() % 128);
        out.push([bits(rng.next() & 1, ex, mx), bits(rng.next() & 1, ey, my)]);
    }
    // All-ones divisors and random operands, at moderate exponents.
    for _ in 0..96 {
        let ex = 96 + rng.next() % 64;
        let ey = 96 + rng.next() % 64;
        out.push([
            bits(rng.next() & 1, ex, rng.next() & 0x7F_FFFF),
            bits(rng.next() & 1, ey, 0x7F_FFFF),
        ]);
        out.push([
            bits(rng.next() & 1, ex, rng.next() & 0x7F_FFFF),
            bits(rng.next() & 1, ey, rng.next() & 0x7F_FFFF),
        ]);
    }
    // Random bit patterns over the whole range.
    for _ in 0..256 {
        out.push([rng.next() & 0xFFFF_FFFF, rng.next() & 0xFFFF_FFFF]);
    }
    // Divisors near 2^40 (all-ones and random significands) over random numerators.
    for _ in 0..32 {
        let y = bits(rng.next() & 1, 166 + rng.next() % 2, rng.next() & 0x7F_FFFF);
        out.push([rng.next() & 0xFFFF_FFFF, y]);
        out.push([rng.next() & 0xFFFF_FFFF, bits(0, 166, 0x7F_FFFF)]);
    }
    // Quotients near a rounding midpoint at every exponent: past the largest finite, below the
    // smallest normal, and across both.
    let mut near = 0;
    while near < 192 {
        let (mx, my) = (rng.next() & 0x7F_FFFF, rng.next() & 0x7F_FFFF);
        let (big_x, big_y) = (mx | 0x80_0000, my | 0x80_0000);
        let t = if big_x >= big_y { 23 } else { 24 };
        let r = (big_x << (t + 1)) % (2 * big_y);
        if r.abs_diff(big_y) * 1024 >= 2 * big_y {
            continue;
        }
        near += 1;
        let (ex, ey) = (1 + rng.next() % 254, 1 + rng.next() % 254);
        out.push([bits(rng.next() & 1, ex, mx), bits(rng.next() & 1, ey, my)]);
    }
    out
}

/// The cases of `pairs` in lanes of `lanes`.
fn lane_cases(pairs: &[[u64; 2]], lanes: u8) -> Vec<Case> {
    let n = usize::from(lanes);
    (0..pairs.len())
        .step_by(n)
        .map(|i| {
            (0..2)
                .map(|k| (0..n).map(|l| pairs[(i + l) % pairs.len()][k]).collect())
                .collect()
        })
        .collect()
}

/// The device's quotient of `float` lanes (form `f`) moved by 0, +1, -2, or -1 ulp (by the
/// divisor's low bits): within 2.5 ulp of the exact one, as Vulkan allows, also where the move
/// crosses a power of two.
pub(super) fn perturbed(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    y: Value,
) -> BResult<Value> {
    let u = super::super::narrow::u32_form(f);
    let ut = ops::ty(fb, u)?;
    let ft = ops::ty(fb, f)?;
    let q = fb.binary(BinOp::FDiv, x, y)?;
    let qb = fb.convert(ConvOp::Bitcast, q, ut)?;
    let yb = fb.convert(ConvOp::Bitcast, y, ut)?;
    let three = ops::splat(fb, u, 3)?;
    let sel = fb.binary(BinOp::BitwiseAnd, yb, three)?;
    let mut out = qb;
    for (k, delta) in [(1u64, 1u64), (2, 0xFFFF_FFFE), (3, 0xFFFF_FFFF)] {
        let kv = ops::splat(fb, u, k)?;
        let dv = ops::splat(fb, u, delta)?;
        let hit = fb.binary(BinOp::IEqual, sel, kv)?;
        let moved = fb.binary(BinOp::IAdd, qb, dv)?;
        out = fb.select(hit, moved, out)?;
    }
    fb.convert(ConvOp::Bitcast, out, ft)
}

/// Define a precise division routine of `lanes` lanes by `path` with estimate `rough`.
fn define_with(b: &mut ModuleBuilder, lanes: u8, path: Path, rough: Rough<'_>) -> BResult<FuncId> {
    divide::define_with(b, Routine { lanes, path }, rough)
}

fn check_divide(what: &str, lanes: u8, path: Path, rough: Rough<'_>, pairs: &[[u64; 2]]) {
    let f = floats(lanes);
    let m = kernel_defs(
        &[f, f],
        f,
        &|b| Ok(vec![define_with(b, lanes, path, rough)?]),
        &|fb, x, funcs| divide::emit(fb, f, x[0], x[1], funcs[0]),
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
    check_module(
        what,
        m,
        &[f, f],
        f,
        &lane_cases(pairs, lanes),
        &|c| {
            Some(
                (0..c[0].len())
                    .map(|l| metal_div32(c[0][l], c[1][l]))
                    .collect(),
            )
        },
        &|g, w| g == w,
    );
}

/// Both paths answer Metal's precise quotient bit for bit, on scalars and vectors, with the
/// device's estimate exact and moved to the edge of 2.5 ulp; the vectors mix lanes of every
/// exponent with zeros, denormals, infinities, and NaNs.
#[test]
fn precise_division_answers_metal_on_both_paths() {
    let pairs = pairs32();
    let exact: Rough<'_> = &|fb, x, y| fb.binary(BinOp::FDiv, x, y);
    for path in [Path::Integer, Path::Fused] {
        for lanes in [1u8, 3] {
            let moved = move |fb: &mut FunctionBuilder<'_>, x: Value, y: Value| {
                perturbed(fb, floats(lanes), x, y)
            };
            for (name, rough) in [("exact", exact), ("moved", &moved as Rough<'_>)] {
                check_divide(
                    &format!("{path:?} x{lanes} {name}"),
                    lanes,
                    path,
                    rough,
                    &pairs,
                );
            }
        }
    }
}

/// The reciprocal step of [`Path::Fused`]: from every estimate within 2.5 ulp of `1 / y`, two
/// fused Newton steps give `RN(1 / y)` for every significand of `y` but the all-ones one, where
/// they give the even neighbor below and one ulp up is `RN(1 / y)` (the operation the routine
/// applies there). The path runs it on the divisor's significand in `[1, 2)`, which this covers.
#[test]
fn markstein_reciprocal_is_correctly_rounded() {
    for m in 0u32..(1 << 23) {
        let big_y = u64::from(m | 0x80_0000);
        let y = f32::from_bits(0x3F80_0000 | m);
        // RN(1 / y) = 2^23 / Y: in units of 2^-24, 2^47 / Y, rounded to nearest even.
        let rn = if m == 0 {
            0x3F80_0000
        } else {
            let mut q = (1u64 << 47) / big_y;
            let r = (1u64 << 47) - q * big_y;
            if 2 * r > big_y || (2 * r == big_y && q & 1 == 1) {
                q += 1;
            }
            if q == 1 << 24 {
                0x3F80_0000
            } else {
                0x3F00_0000 | (q - (1 << 23)) as u32
            }
        };
        for d in -5i32..=5 {
            let rb = rn.wrapping_add_signed(d);
            // Within 2.5 ulp of 1 / y (2^-24 in (1/2, 1); at y = 1, the ulp above one, 2^-23):
            // |r * Y - 2^23| <= 2.5 ulp * Y, every term exact in f64 (r and Y have 24 bits).
            let ulp = if m == 0 {
                2f64.powi(-23)
            } else {
                2f64.powi(-24)
            };
            let yf = big_y as f64;
            if (f64::from(f32::from_bits(rb)) * yf - 2f64.powi(23)).abs() > 2.5 * ulp * yf {
                continue;
            }
            let mut r = f32::from_bits(rb);
            for _ in 0..2 {
                let t = (-y).mul_add(r, 1.0);
                r = t.mul_add(r, r);
            }
            let got = r.to_bits() | u32::from(m == 0x7F_FFFF);
            assert_eq!(got, rn, "y = {y:e} (significand {m:#x}), estimate {rb:#x}");
        }
    }
}

/// [`Path::Halves`]'s quotient of two `half` values rounds to their exact quotient's `half`, at
/// every scale: for every pair of 11-bit significands `X / 2^10` and `Y / 2^10` (in `[1, 2)`) and
/// every estimate of `1 / y` within 2.5 ulp of it, `q = x · r` and one fused step `q + r · (x -
/// y · q)` give a quotient on the same point of the 12-bit grid as `X / Y` or strictly between
/// the same two neighboring points. Every `half` value and every `half` rounding midpoint (the
/// denormal ones and the overflow threshold 65520 included) is such a point at its scale, so the
/// quotient and the exact one round to the same `half`. The routine's steps stay normal for every
/// `half` operand (the divisor in `[2^-24, 65504]`, the quotient in `[2^-40, 2^40]`), where
/// scaling an operand by a power of two scales every step and the set of estimates exactly, and
/// negating one negates them, so these pairs cover every pair of finite nonzero `half` values.
#[test]
fn one_fused_correction_rounds_every_half_quotient() {
    for my in 0u32..1024 {
        let big_y = 1024 + my;
        let y = big_y as f32 / 1024.0;
        // RN(1 / y), and every float within 2.5 ulp of 1 / y around it (2^-24 in (1/2, 1); at
        // y = 1, the ulp above one, 2^-23): `f64` holds each product `r · y` exactly (r has 24
        // bits and y 11).
        let rn = 1.0 / y;
        let yf = f64::from(y);
        let ulp = if my == 0 {
            2f64.powi(-23)
        } else {
            2f64.powi(-24)
        };
        let estimates: Vec<f32> = (-5i32..=5)
            .map(|d| f32::from_bits(rn.to_bits().wrapping_add_signed(d)))
            .filter(|r| (f64::from(*r) * yf - 1.0).abs() <= 2.5 * ulp * yf)
            .collect();
        assert!(estimates.len() >= 4, "y = {y}: {estimates:?}");
        for mx in 0u32..1024 {
            let big_x = 1024 + mx;
            let x = big_x as f32 / 1024.0;
            // The 12-bit grid of X / Y's binade: spacing 2^-11 in [1, 2), 2^-12 in (1/2, 1).
            let t = if big_x >= big_y { 11 } else { 12 };
            let scaled = u64::from(big_x) << t;
            let (lo, rem) = (scaled / u64::from(big_y), scaled % u64::from(big_y));
            let point = |k: u64| k as f64 / f64::from(1u32 << t);
            for &r in &estimates {
                let q = x * r;
                let e = (-y).mul_add(q, x);
                let q = f64::from(e.mul_add(r, q));
                if rem == 0 {
                    assert_eq!(q, point(lo), "{big_x} / {big_y}, estimate {r:e}");
                } else {
                    assert!(
                        point(lo) < q && q < point(lo + 1),
                        "{big_x} / {big_y}, estimate {r:e}: {q:e} is not in ({:e}, {:e})",
                        point(lo),
                        point(lo + 1)
                    );
                }
            }
        }
    }
}

/// Metal's `half` quotient of half bits `x` and `y`: the exact quotient rounded once to `half`
/// (the host's `f64` quotient of two `half` values rounds to `half` once), `half` denormals kept.
fn metal_div16(x: u64, y: u64) -> u64 {
    u64::from(f64_to_f16(f16_to_f64(x as u16) / f16_to_f64(y as u16)))
}

/// `half` operand pairs: every pair of the edges (zeros, denormals, the smallest normal, one and
/// its neighbors, the largest finite, infinities, a NaN, three), and random bits.
fn pairs16() -> Vec<[u64; 2]> {
    let edges: [u64; 16] = [
        0, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x3C01, 0x3BFF, 0x7BFF, 0xFBFF,
        0x7C00, 0xFC00, 0x7E00, 0x4200,
    ];
    let mut out = Vec::new();
    for a in edges {
        for b in edges {
            out.push([a, b]);
        }
    }
    let mut rng = Rng(0x2790_D1F1_0000_0016);
    for _ in 0..1500 {
        let r = rng.next();
        out.push([r & 0xFFFF, (r >> 16) & 0xFFFF]);
    }
    out
}

/// A `half` quotient through every path is Metal's, the exact quotient rounded once to `half`,
/// with the device's `float` estimate exact and moved to the edge of 2.5 ulp (where the estimate
/// rounded to `half` alone would cross a rounding midpoint), on scalars and vectors.
#[test]
fn precise_half_division_rounds_once() {
    let pairs = pairs16();
    let exact: Rough<'_> = &|fb, x, y| fb.binary(BinOp::FDiv, x, y);
    for path in [Path::Integer, Path::Fused, Path::Halves] {
        for lanes in [1u8, 2] {
            let moved = move |fb: &mut FunctionBuilder<'_>, x: Value, y: Value| {
                perturbed(fb, floats(lanes), x, y)
            };
            for (name, rough) in [("exact", exact), ("moved", &moved as Rough<'_>)] {
                let what = format!("half {path:?} x{lanes} {name}");
                let f = vec(float(16), lanes);
                let m = kernel_defs(
                    &[f, f],
                    f,
                    &|b| Ok(vec![define_with(b, lanes, path, rough)?]),
                    &|fb, x, funcs| divide::emit(fb, f, x[0], x[1], funcs[0]),
                    true,
                    false,
                )
                .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
                check_module(
                    &what,
                    m,
                    &[f, f],
                    f,
                    &lane_cases(&pairs, lanes),
                    &|c| {
                        Some(
                            (0..c[0].len())
                                .map(|l| metal_div16(c[0][l], c[1][l]))
                                .collect(),
                        )
                    },
                    &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
                );
            }
        }
    }
}

/// The routines' words stay within their estimates.
#[test]
fn precise_division_routines_fit_their_word_estimates() {
    for path in [Path::Integer, Path::Fused, Path::Halves] {
        for lanes in [1u8, 4] {
            let r = Routine { lanes, path };
            let mut b = ModuleBuilder::new();
            divide::define(&mut b, r).expect("define");
            let void = b.ty(Type::Void).expect("void");
            let id = {
                let mut fb = b.function(void).expect("entry");
                fb.ret(None).expect("ret");
                fb.finish().expect("finish")
            };
            b.entry(
                ExecModel::GLCompute,
                id,
                vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
            )
            .expect("entry");
            let m = b.finalize().expect("finalize");
            let words = crate::spirv::serialize::serialize(m)
                .expect("serialize")
                .words
                .len();
            assert!(
                words <= r.words() as usize && words * 3 >= r.words() as usize * 2,
                "{path:?} x{lanes}: {words} words against the estimate {} (within its last third)",
                r.words()
            );
        }
    }
}

/// The path follows the profile: fused where the `float` fused multiply-add is `OpFmaKHR`,
/// integer elsewhere; a `half` division on the fused path takes the halves' path where the
/// profile's `half` rounding is not the device's own. Every path is some committed profile's.
#[test]
fn the_precise_division_path_follows_the_profile() {
    let mut seen = std::collections::BTreeSet::new();
    for (stem, p) in committed_profiles().expect("profiles") {
        let want = if p.fields().leaf.fma32 {
            Path::Fused
        } else {
            Path::Integer
        };
        assert_eq!(
            Routine::on(&p, floats(2)),
            Routine {
                lanes: 2,
                path: want
            },
            "{stem}"
        );
        let native = narrow::native_rounding(&p, narrow::quantize_form(&p)).is_some();
        let want_half = if want == Path::Fused && !native {
            Path::Halves
        } else {
            want
        };
        assert_eq!(
            Routine::on(&p, vec(float(16), 2)),
            Routine {
                lanes: 2,
                path: want_half
            },
            "{stem}: half"
        );
        seen.insert(want);
        seen.insert(want_half);
    }
    assert_eq!(
        seen.len(),
        3,
        "every path is some committed profile's: {seen:?}"
    );
}

/// Metal's `frem` without every fast-math flag: `x - y * trunc(x / y)` on operands read as zeros
/// where denormal, of Metal's precise quotient, the product and the difference rounded to `f32` and
/// flushed where denormal, a NaN the canonical one.
fn metal_frem32(x: u64, y: u64) -> u64 {
    let (x, y) = (ftz(x), ftz(y));
    let t = f32v(metal_div32(x, y)).trunc();
    let p = ftz(u64::from((f32v(y) * t).to_bits()));
    let r = f32v(x) - f32v(p);
    if r.is_nan() {
        return 0x7FC0_0000;
    }
    ftz(u64::from(r.to_bits()))
}

/// Metal's `frem` on the M3 Max without fast math (our own probe): the
/// operands' bits and the answer's, sampled from the 49,753 measured pairs, which
/// [`metal_frem32`] answers all of bit for bit. Not the exact remainder: `frem(1, 0.1)` is `+0`
/// where `fmod` gives `0x3DCCCCCB`.
const FREM: [(u32, u32, u32); 24] = [
    (0x3F80_0000, 0x3DCC_CCCD, 0x0000_0000),
    (0xBF80_0000, 0x3DCC_CCCD, 0x0000_0000),
    (0x8000_0001, 0x3F80_0000, 0x0000_0000),
    (0x526F_4968, 0xBE9B_F0CA, 0x4680_0000),
    (0x5865_577A, 0xD43B_FA15, 0x537A_C000),
    (0x4072_A1A8, 0xBF1A_5671, 0x3E31_FFE0),
    (0x1DE5_2695, 0x9AD9_E1A5, 0x1A07_2E00),
    (0x5174_BA86, 0xCB28_C38D, 0x4AEF_8000),
    (0x79C5_B1DB, 0xF20B_EC8C, 0x720B_8000),
    (0x61ED_4285, 0x5230_FE72, 0xD600_0000),
    (0xF1A1_BD5E, 0xE9A9_D63A, 0xE930_0000),
    (0xB31B_12F4, 0x3059_3991, 0xB015_9E80),
    (0xA33A_821C, 0x9D89_EB99, 0x9D07_2000),
    (0xBE99_999A, 0xF149_F2CA, 0xBE99_999A),
    (0x3F80_0000, 0xBE99_999A, 0x3DCC_CCC8),
    (0x4669_071D, 0x4602_9548, 0x45CC_E3AA),
    (0x8A6A_493E, 0x008F_24EF, 0x8000_0000),
    (0x997D_1870, 0x118E_32E4, 0x9164_0000),
    (0x7F80_0000, 0xFF7F_FFFF, 0x7FC0_0000),
    (0xBF80_0000, 0x7F80_0000, 0x7FC0_0000),
    (0x0040_0000, 0xFFC0_0001, 0x7FC0_0000),
    (0x0000_0000, 0x007F_FFFF, 0x7FC0_0000),
    (0x811E_B274, 0x8053_3F6A, 0x7FC0_0000),
    (0x0029_DF79, 0x07EC_0A0E, 0x0000_0000),
];

/// `math::metal_frem` with its operands, product and result flushed and its NaN canonicalized (as
/// selection wraps it) answers [`metal_frem32`] on both division paths, scalar and vector, at
/// [`pairs32`] and at every measured pair, which the reference answers as Metal did.
#[test]
fn frem_answers_metal_on_both_paths() {
    for &(x, y, want) in &FREM {
        assert_eq!(
            metal_frem32(u64::from(x), u64::from(y)),
            u64::from(want),
            "the reference at ({x:#010x}, {y:#010x})"
        );
    }
    let mut pairs = pairs32();
    pairs.extend(FREM.iter().map(|&(x, y, _)| [u64::from(x), u64::from(y)]));
    let exact: Rough<'_> = &|fb, x, y| fb.binary(BinOp::FDiv, x, y);
    for path in [Path::Integer, Path::Fused] {
        for lanes in [1u8, 3] {
            let f = floats(lanes);
            let what = format!("frem {path:?} x{lanes}");
            let m = kernel_defs(
                &[f, f],
                f,
                &|b| Ok(vec![define_with(b, lanes, path, exact)?]),
                &|fb, x, funcs| {
                    let flush =
                        |fb: &mut FunctionBuilder<'_>, v| super::super::denormal::flush(fb, f, v);
                    let (p, q) = (flush(fb, x[0])?, flush(fb, x[1])?);
                    let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                        divide::emit(fb, f, p, q, funcs[0])
                    };
                    let r = math::metal_frem(fb, f, (p, q), &divide, true)?;
                    let r = flush(fb, r)?;
                    let nan = super::super::nan::is_nan(fb, f, NanTest::Bits, r)?;
                    let canonical = ops::fsplat(fb, f, f64::NAN)?;
                    fb.select(nan, canonical, r)
                },
                true,
                false,
            )
            .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
            check_module(
                &what,
                m,
                &[f, f],
                f,
                &lane_cases(&pairs, lanes),
                &|c| {
                    Some(
                        (0..c[0].len())
                            .map(|l| metal_frem32(c[0][l], c[1][l]))
                            .collect(),
                    )
                },
                &|g, w| g == w,
            );
        }
    }
}
