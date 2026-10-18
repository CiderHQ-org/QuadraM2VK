//! Correctly rounded fused multiply-add: every route against one rounding of the exact `a * b + c`,
//! on the interpreter.
//!
//! The `float` reference is the host's `f32::mul_add` (one rounding, denormals kept); the `half`
//! reference rounds the exact sum, formed as an integer multiple of `2^-48`, once to nearest
//! even. Beside random and special operands, the inputs hold the triples where a shortcut
//! rounds twice: the exact sum within a `double` ulp of a `float` midpoint (which a `double` sum
//! rounded to `float` gets wrong without round-to-odd), with the addend dominant and with the
//! product dominant, and the `half` triples whose `float` sum lands on a `half` midpoint.

use super::*;
use crate::lower::arith::fma::{self, Fma, Route, Routine};
use std::collections::{BTreeMap, BTreeSet};

fn floats(lanes: u8) -> Form {
    vec(float(32), lanes)
}

/// `f32` bits of sign `s`, unbiased exponent `e` (a normal), and significand field `m`.
fn mk32(s: u64, e: i32, m: u64) -> u64 {
    (s << 31) | (((e + 127) as u64) << 23) | m
}

/// `half` bits of sign `s`, unbiased exponent `e` (a normal), and significand field `m`.
fn mk16(s: u64, e: i32, m: u64) -> u64 {
    (s << 15) | (((e + 15) as u64) << 10) | m
}

fn fma32(a: u64, b: u64, c: u64) -> u64 {
    u64::from(f32v(a).mul_add(f32v(b), f32v(c)).to_bits())
}

/// An `f32` denormal as a zero of its sign.
fn ftz(x: u64) -> u64 {
    if x & 0x7F80_0000 == 0 {
        x & 0x8000_0000
    } else {
        x
    }
}

/// Whether the exact `a * b + c` of finite `f32`s is nonzero and below `2^-126` in magnitude: the
/// product is exact in `f64`, and the sum's rounding error is exact (TwoSum), so the sum rounded
/// to `f64` and its error place the exact value against `2^-126`, a `f64`.
fn exact_below_normal(a: u64, b: u64, c: u64) -> bool {
    let (a, b, c) = (f32v(a), f32v(b), f32v(c));
    if !(a.is_finite() && b.is_finite() && c.is_finite()) {
        return false;
    }
    let p = f64::from(a) * f64::from(b);
    let c = f64::from(c);
    let s = p + c;
    let v = s - p;
    let e = (p - (s - v)) + (c - v);
    let normal = f64::from(f32::MIN_POSITIVE);
    (s != 0.0 && s.abs() < normal) || (s.abs() == normal && e != 0.0 && (e < 0.0) == (s > 0.0))
}

/// Metal's `fma` (probe cases e98b103f and f7850ba7): denormal operands read as zeros, the one
/// rounding, a result whose exact value is below `2^-126` a zero of its sign before rounding, and
/// every NaN `0x7FC00000`.
fn metal_fma32(a: u64, b: u64, c: u64) -> u64 {
    let (a, b, c) = (ftz(a), ftz(b), ftz(c));
    let r = fma32(a, b, c);
    let r = if exact_below_normal(a, b, c) {
        r & 0x8000_0000
    } else {
        r
    };
    if f32v(r).is_nan() {
        0x7FC0_0000
    } else {
        r
    }
}

/// Whether a `double` sum rounded to `float` differs from the fused result (the hazard the
/// `double` route's round-to-odd removes).
fn double_rounds_wrong(a: u64, b: u64, c: u64) -> bool {
    let s = f64::from(f32v(a)) * f64::from(f32v(b)) + f64::from(f32v(c));
    u64::from((s as f32).to_bits()) != fma32(a, b, c)
}

/// The `float` triples: specials and every pairing of edge values, random bit patterns, nearby
/// normals, cancellations, denormal results, products past the largest float that the addend
/// brings back, and the `double` double-rounding hazards.
fn float_triples() -> Vec<[u64; 3]> {
    let edges: [u64; 19] = [
        0,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0040_0000,
        0x007F_FFFF,
        0x0080_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F80_0001,
        0x3F7F_FFFF,
        0x4040_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0x5F00_0000,
        0x1F00_0000,
    ];
    let mut out = Vec::new();
    for a in edges {
        for b in edges {
            for c in edges {
                out.push([a, b, c]);
            }
        }
    }
    let mut rng = Rng(0xF3A1_0001);
    for _ in 0..1500 {
        let r = rng.next();
        out.push([r & 0xFFFF_FFFF, r >> 32, rng.next() & 0xFFFF_FFFF]);
    }
    let normal = |rng: &mut Rng, lo: i32, span: u64| {
        let r = rng.next();
        mk32(r & 1, lo + ((r >> 1) % span) as i32, (r >> 16) & 0x7F_FFFF)
    };
    for _ in 0..1500 {
        out.push([
            normal(&mut rng, -20, 40),
            normal(&mut rng, -20, 40),
            normal(&mut rng, -30, 60),
        ]);
    }
    // Cancellations: `c` within a few ulps of `-RN(a * b)`, or some binades below it.
    for i in 0..1000u64 {
        let (a, b) = (normal(&mut rng, -60, 120), normal(&mut rng, -60, 120));
        let p = u64::from((f32v(a) * f32v(b)).to_bits());
        if (p & 0x7FFF_FFFF) < 0x0080_0000 || (p & 0x7F80_0000) == 0x7F80_0000 {
            continue;
        }
        let k = (rng.next() % 7) as i64 - 3;
        let mag = ((p & 0x7FFF_FFFF) as i64 + k) as u64;
        let c = if i % 2 == 0 {
            mag | ((p & 0x8000_0000) ^ 0x8000_0000)
        } else {
            let e = ((p >> 23) & 0xFF) as i32 - 127 - 1 - (rng.next() % 30) as i32;
            normal(&mut rng, e, 1)
        };
        out.push([a, b, c]);
    }
    // Denormal products and results.
    for _ in 0..1000 {
        let a = normal(&mut rng, -80, 20);
        let b = normal(&mut rng, -80, 20);
        let r = rng.next();
        let c = if r.is_multiple_of(3) {
            (r >> 2) & 0x807F_FFFF
        } else {
            normal(&mut rng, -130, 8)
        };
        out.push([a, b, c]);
    }
    // Products past the largest float that the addend brings back.
    for _ in 0..200 {
        let ea = 64 + (rng.next() % 64) as i32;
        let a = normal(&mut rng, ea, 1);
        let b = normal(&mut rng, 128 - ea, 1);
        let c = 0xFF7F_FFFF ^ (a & 0x8000_0000) ^ (b & 0x8000_0000);
        out.push([a, b, c]);
    }
    // The addend dominant: P = 2^(ec-24) (1 - 2^-46 u^2) just below half an ulp of an odd `c`,
    // so the `double` sum is the `float` midpoint, which ties away from `c`.
    let mut n = 0;
    while n < 200 {
        let u = 1 + rng.next() % 360;
        let ec = (rng.next() % 190) as i32 - 90;
        let c = mk32(rng.next() & 1, ec, (rng.next() & 0x7F_FFFF) | 1);
        let ka = (rng.next() % 81) as i32 - 40;
        let kb = ec - 24 - ka;
        if !(-126..=127).contains(&ka) || !(-125..=127).contains(&kb) {
            continue;
        }
        let a = mk32(rng.next() & 1, ka, u);
        let b = mk32(0, kb - 1, (1 << 23) - 2 * u);
        if double_rounds_wrong(a, b, c) {
            out.push([a, b, c]);
            n += 1;
        }
    }
    // The product dominant: its bits past the `float` midpoint bit are few, and `c` cancels
    // them but for a residue far below an ulp of the `double` sum.
    let mut n = 0;
    let mut tries = 0;
    while n < 200 && tries < 4_000_000 {
        tries += 1;
        let ma = (rng.next() & 0x7F_FFFF) | 0x80_0000;
        let mb = (rng.next() & 0x7F_FFFF) | 0x80_0000;
        let p = ma * mb;
        let top = 64 - p.leading_zeros() as i32;
        let mid = top - 25;
        if (p >> mid) & 1 == 0 {
            continue;
        }
        let low = p & ((1 << mid) - 1);
        let lowlen = 64 - low.leading_zeros() as i32;
        if low == 0 || lowlen > 17 {
            continue;
        }
        let shift = lowlen + 7 + (rng.next() % (18 - lowlen as u64).max(1)) as i32;
        if shift > 24 {
            continue;
        }
        let tiny = lowlen - shift;
        let toward = if rng.next() & 1 == 0 { 1i64 } else { -1 };
        let cint = -((low << -tiny) as i64) + toward;
        let cmag = cint.unsigned_abs();
        if 64 - cmag.leading_zeros() > 24 {
            continue;
        }
        let ea = (rng.next() % 101) as i32 - 50;
        let eb = (rng.next() % 101) as i32 - 50;
        let cexp = tiny + ea + eb - 46;
        let cv = cint as f64 * 2f64.powi(cexp);
        let c = u64::from((cv as f32).to_bits());
        if f64::from(f32v(c)) != cv {
            continue;
        }
        let a = mk32(0, ea, ma & 0x7F_FFFF);
        let b = mk32(0, eb, mb & 0x7F_FFFF);
        if double_rounds_wrong(a, b, c) {
            out.push([a, b, c]);
            n += 1;
        }
    }
    assert!(n >= 100, "only {n} product-dominant hazards");
    // The sticky bit decides: the product is exactly a `float` tie (`(1 + 2^-12)^2 = 1 + 2^-11 +
    // 2^-24` above an even last bit, `(1 + 2^-23) * 1.5` above an odd one) and an addend of
    // the sign that breaks the tie lies below the integer route's 64 bits; scaled by `2^k` and
    // negated.
    for (a, b, c_sign) in [
        (0x3F80_0800u64, 0x3F80_0800u64, 0u64),
        (0x3F80_0001, 0x3FC0_0000, 0x8000_0000),
    ] {
        for k in [-60i64, -20, 0, 20, 60] {
            for t in [-100i64, -70, -50, -42, -40, -39] {
                let ae = ((a >> 23) & 0xFF) as i64 + k;
                let ce = 127 + t + k;
                if !(1..=254).contains(&ae) || !(1..=254).contains(&ce) {
                    continue;
                }
                let a_k = (a & 0x807F_FFFF) | ((ae as u64) << 23);
                let c_k = c_sign | ((ce as u64) << 23);
                for neg in [0u64, 0x8000_0000] {
                    out.push([a_k ^ neg, b, c_k ^ neg]);
                }
            }
        }
    }
    out
}

/// The cases of `triples` on `lanes` lanes: lane `l` of case `i` is triple `i + l`.
fn lane_cases(triples: &[[u64; 3]], lanes: u8) -> Vec<Case> {
    let n = usize::from(lanes);
    (0..triples.len())
        .step_by(n)
        .map(|i| {
            (0..3)
                .map(|k| {
                    (0..n)
                        .map(|l| triples[(i + l) % triples.len()][k])
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Bits equal, or both NaN.
fn same(f: Form) -> impl Fn(u64, u64) -> bool {
    move |g, w| g == w || (is_nan(f, g) && is_nan(f, w))
}

/// Run `fma` over `cases` (through a defined routine when its route has one; on vectors, per
/// lane through its route's routine on one lane when `per_lane`; checked, on operands read as
/// Metal reads them, as a module flushes them) and compare with `reference` per lane: bit for bit
/// when `exact`, else any NaN for a NaN.
fn check_fma(
    what: &str,
    fma: Fma,
    per_lane: bool,
    cases: &[Case],
    reference: &dyn Fn(u64, u64, u64) -> u64,
    exact: bool,
) {
    let f = fma.form;
    let routine = fma.routine();
    // The module's routines: the use's and, per lane, its route's on one lane.
    let mut fmas: BTreeSet<Routine> = routine.into_iter().collect();
    if per_lane {
        fmas.extend(routine.map(|r| Routine {
            route: r.route,
            lanes: 1,
        }));
    }
    // The routine of `OpFmaKHR`'s small results is never defined per lane.
    assert_eq!(
        routine.is_some_and(|r| fma::per_lane(r, &fmas)),
        per_lane && routine.is_some_and(|r| r.route != Route::Khr) && f.lanes > 1,
        "{what}"
    );
    let m = kernel_defs(
        &[f, f, f],
        f,
        &|b| {
            let mut defined = BTreeMap::new();
            for r in &fmas {
                let id = fma::define_among(b, *r, &fmas, &defined)?;
                defined.insert(*r, id);
            }
            Ok(routine
                .and_then(|r| defined.get(&r).copied())
                .into_iter()
                .collect())
        },
        &|fb, x, funcs| {
            let x = match fma.checked() {
                true => x
                    .iter()
                    .map(|v| crate::lower::arith::denormal::flush(fb, f, *v))
                    .collect::<BResult<Vec<_>>>()?,
                false => x.to_vec(),
            };
            fma::emit(fb, fma, &x, funcs.first().copied())
        },
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
    check_module(
        what,
        m,
        &[f, f, f],
        f,
        cases,
        &|c| {
            Some(
                (0..c[0].len())
                    .map(|l| reference(c[0][l], c[1][l], c[2][l]))
                    .collect(),
            )
        },
        &|g, w| if exact { g == w } else { same(f)(g, w) },
    );
}

/// The three `float` routes round the exact `a * b + c` once, on scalars and vectors (a vector
/// routine's sequence on its lanes, and per lane), specials, signed zeros, denormals, overflow, and
/// the double-rounding hazards included. The routes the translation computes answer Metal's `fma`
/// bit for bit (its denormal flush and NaN); `OpFmaKHR` is the device's (here the interpreter's
/// IEEE `fma`), and checked by either routine, on operands read as Metal reads them, Metal's
/// but for the NaN's encoding.
#[test]
fn float_fused_multiply_add_rounds_once_on_every_route() {
    let triples = float_triples();
    for (route, tiny, reference, exact) in [
        (Route::Khr, None, fma32 as fn(u64, u64, u64) -> u64, false),
        (Route::Khr, Some(Route::Wide), metal_fma32, false),
        (Route::Khr, Some(Route::Soft), metal_fma32, false),
        (Route::Wide, None, metal_fma32, true),
        (Route::Soft, None, metal_fma32, true),
    ] {
        for (lanes, per_lane) in [(1u8, false), (3, false), (3, true)] {
            let fma = Fma {
                form: floats(lanes),
                route,
                tiny,
            };
            check_fma(
                &format!("fma {route:?} checked by {tiny:?} x{lanes}, per lane {per_lane}"),
                fma,
                per_lane,
                &lane_cases(&triples, lanes),
                &reference,
                exact,
            );
        }
    }
}

/// `OpFmaKHR` checked (`fma::checked`) is Metal's `fma` on every device Vulkan allows, on operands
/// read as Metal reads them: whether the device keeps a denormal result, flushes it to a zero of
/// its sign or to `+0`, or flushes an exact value below `2^-126` before rounding, a result above
/// `2^-126` in magnitude is Metal's, and the routine answers the rest. The triples reach the
/// results the device rounds up to `2^-126` (`2^-126 * (1 - 2^-24)` among the edges), and the
/// test counts them.
#[test]
fn a_checked_fused_multiply_add_is_metals_in_every_denormal_mode() {
    let devices: [(&str, fn(u64, u64, u64) -> u64); 4] = [
        ("kept", fma32),
        ("flushed", |a, b, c| ftz(fma32(a, b, c))),
        ("flushed to +0", |a, b, c| {
            let r = fma32(a, b, c);
            if r & 0x7F80_0000 == 0 {
                0
            } else {
                r
            }
        }),
        ("flushed before rounding", |a, b, c| {
            let r = fma32(a, b, c);
            if exact_below_normal(a, b, c) {
                r & 0x8000_0000
            } else {
                r
            }
        }),
    ];
    let triples: Vec<[u64; 3]> = float_triples().into_iter().map(|t| t.map(ftz)).collect();
    let rounded_up = triples
        .iter()
        .filter(|[a, b, c]| {
            exact_below_normal(*a, *b, *c) && fma32(*a, *b, *c) & 0x7FFF_FFFF == 0x0080_0000
        })
        .count();
    assert!(rounded_up >= 1, "{rounded_up} results rounded up to 2^-126");
    for (what, device) in devices {
        for [a, b, c] in &triples {
            let r = device(*a, *b, *c);
            let small = r & 0x7FFF_FFFF <= 0x0080_0000;
            let want = metal_fma32(*a, *b, *c);
            // By the whole routine (the integer route), and by the routine of small results.
            for (routine, answer) in [
                ("Metal's fma", metal_fma32(*a, *b, *c)),
                ("small", small_results(*a, *b, *c, r)),
            ] {
                let checked = if small { answer } else { r };
                assert!(
                    same(float(32))(checked, want),
                    "{what}, {routine}: fma({a:#010x}, {b:#010x}, {c:#010x}): {checked:#010x}, \
                     Metal {want:#010x}"
                );
            }
        }
    }
}

/// `fma::small` on one lane, in Rust's IEEE `double`: Metal's `fma` of `a`, `b`, `c` from `r`,
/// their `OpFmaKHR` on some device, where `r` is at most `2^-126` in magnitude, else `r`.
fn small_results(a: u64, b: u64, c: u64, r: u64) -> u64 {
    let read = |x: u64| match x & 0x7F80_0000 {
        0 => 0.0,
        _ => f64::from(f32v(x)),
    };
    let (x, y, z) = (read(a), read(b), read(c));
    let p = x * y;
    let s = p + z;
    let sp = s - z;
    let sz = s - sp;
    let e = (p - sp) + (z - sz);
    let normal = f64::from(f32::MIN_POSITIVE);
    let size = s.abs();
    let tiny = size < normal || (size == normal && e * s < 0.0);
    match r & 0x7FFF_FFFF <= 0x0080_0000 && tiny {
        // An exact zero sum: IEEE's sign, on the operands' bits.
        true if s == 0.0 => (a ^ b) & c & 0x8000_0000,
        true if s < 0.0 => 0x8000_0000,
        true => 0,
        false => r,
    }
}

/// The blends' exact `float` multiply-add (`fma::float_fma_exact`, of a `float` output's blend into
/// an extended-range or 16-bit float attachment) rounds the exact `a * b + c` once on operands below
/// 2 in magnitude whose parts do not underflow, on one lane and four, and its zero is IEEE's: `-0`
/// for a negative zero product and a `-0` addend (`-0 · 1 + -0`, the zero alpha Metal stores for a
/// `float4` output's blend over a `-0` texel), `+0` for opposite zeros and for a product the addend
/// cancels. The emulation's own sum loses the first (`th + v` adds a `+0` to a `-0`). An infinite
/// operand's infinite result is that infinity, where the emulation's own steps give a NaN (a
/// `float4` output's infinity over a texel, which Metal stores as the `half` infinity).
#[test]
fn the_blends_float_multiply_add_rounds_once_and_signs_its_zeros() {
    fn operand(rng: &mut Rng) -> u64 {
        let r = rng.next();
        mk32(r & 1, ((r >> 1) % 15) as i32 - 14, (r >> 8) & 0x7F_FFFF)
    }
    let edges: [u64; 8] = [
        0,
        0x8000_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3F00_0000,
        0xBF00_0000,
        0x3FFF_FFFF,
        0xBFFF_FFFF,
    ];
    let mut triples = Vec::new();
    for a in edges {
        for b in edges {
            for c in edges {
                triples.push([a, b, c]);
            }
        }
    }
    // Infinite operands whose results are infinite.
    triples.extend([
        [0x7F80_0000, 0x3F80_0000, 0x3F00_0000],
        [0xFF80_0000, 0x3F00_0000, 0xBF80_0000],
        [0x3F80_0000, 0x3F00_0000, 0x7F80_0000],
        [0xBFFF_FFFF, 0x3F80_0000, 0xFF80_0000],
        [0x7F80_0000, 0x3F80_0000, 0x7F80_0000],
    ]);
    let mut rng = Rng(0xB1E0_F32A);
    for _ in 0..2000 {
        triples.push([operand(&mut rng), operand(&mut rng), operand(&mut rng)]);
    }
    // Products the addend cancels: factors of 12 significand bits, whose product is a `float`.
    for _ in 0..300 {
        let (a, b) = (operand(&mut rng) & !0xFFF, operand(&mut rng) & !0xFFF);
        let p = f32v(a) * f32v(b);
        triples.push([a, b, u64::from((-p).to_bits())]);
    }
    assert!(triples.contains(&[0x8000_0000, 0x3F80_0000, 0x8000_0000]));
    for lanes in [1u8, 4] {
        let f = floats(lanes);
        check(
            &format!("the blends' float fma x{lanes}"),
            &[f, f, f],
            f,
            &|fb, x| fma::float_fma_exact(fb, f, x[0], x[1], x[2]),
            &lane_cases(&triples, lanes),
            &|c| {
                Some(
                    (0..c[0].len())
                        .map(|l| fma32(c[0][l], c[1][l], c[2][l]))
                        .collect(),
                )
            },
            false,
        );
    }
}

/// The `double` and integer routes never take the unfused shortcut where it differs: the
/// hazard families hold triples a `double` sum rounded to `float` gets wrong, and the
/// cancellations triples where the unfused result differs from the fused one.
#[test]
fn the_float_triples_reach_the_hazards() {
    let t = float_triples();
    let hazards = t
        .iter()
        .filter(|[a, b, c]| double_rounds_wrong(*a, *b, *c))
        .count();
    let unfused = t
        .iter()
        .filter(|[a, b, c]| {
            let u = u64::from((f32v(*a) * f32v(*b) + f32v(*c)).to_bits());
            let f = fma32(*a, *b, *c);
            u != f && !(is_nan(float(32), u) && is_nan(float(32), f))
        })
        .count();
    assert!(hazards >= 300, "{hazards} double-rounding hazards");
    assert!(unfused >= 1000, "{unfused} triples where unfused differs");
}

/// The exact `a * b + c` of `half` bits rounded once to `half` (nearest even): the operands
/// scaled by `2^24` are integers, so the sum is an integer multiple of `2^-48`.
fn fma16(a: u64, b: u64, c: u64) -> u64 {
    let (x, y, z) = (
        f16_to_f64(a as u16),
        f16_to_f64(b as u16),
        f16_to_f64(c as u16),
    );
    if !x.is_finite() || !y.is_finite() || !z.is_finite() {
        return u64::from(f64_to_f16(x * y + z));
    }
    let scale = 2f64.powi(24);
    let (xi, yi, zi) = (
        (x * scale) as i128,
        (y * scale) as i128,
        (z * scale) as i128,
    );
    let n = xi * yi + (zi << 24);
    if n == 0 {
        // IEEE: the sum of two zeros of one sign keeps it; any other exact zero is `+0`.
        let p_neg = x.is_sign_negative() != y.is_sign_negative();
        let both = xi * yi == 0 && zi == 0 && p_neg && z.is_sign_negative();
        return if both { 0x8000 } else { 0 };
    }
    let m = n.unsigned_abs();
    let msb = 127 - m.leading_zeros() as i32;
    // Keep 11 bits, or the `2^-24` bit of a denormal (the sum's bit 24).
    let drop = (msb - 10).max(24) as u32;
    let q = m >> drop;
    let rem = m & ((1u128 << drop) - 1);
    let half = 1u128 << (drop - 1);
    let q = if rem > half || (rem == half && q & 1 == 1) {
        q + 1
    } else {
        q
    };
    let bits = if msb > 34 {
        ((((msb - 34) as u128) << 10) + q).min(0x7C00)
    } else {
        q
    };
    (bits as u64) | if n < 0 { 0x8000 } else { 0 }
}

/// The `half` triples: specials and edge pairings, random patterns, the triples whose `float`
/// sum lands on a `half` midpoint (`c` odd, `P = 2^(ec-11) (1 - 2^-20 u^2)` just inside half an
/// ulp of `c`), and products near `-c`.
fn half_triples() -> Vec<[u64; 3]> {
    let edges: [u64; 16] = [
        0, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x3C01, 0x3BFF, 0x7BFF, 0xFBFF,
        0x7C00, 0xFC00, 0x7E00, 0x4200,
    ];
    let mut out = Vec::new();
    for a in edges {
        for b in edges {
            for c in edges {
                out.push([a, b, c]);
            }
        }
    }
    let mut rng = Rng(0xF3A1_0016);
    for _ in 0..1500 {
        let r = rng.next();
        out.push([r & 0xFFFF, (r >> 16) & 0xFFFF, (r >> 32) & 0xFFFF]);
    }
    let mut n = 0;
    while n < 300 {
        let u = 1 + rng.next() % 11;
        let ec = (rng.next() % 28) as i32 - 12;
        let c = mk16(rng.next() & 1, ec, (rng.next() & 0x3FF) | 1);
        let ka = (rng.next() % 28) as i32 - 13;
        let kb = ec - 11 - ka;
        if !(-13..=15).contains(&ka) || !(-13..=15).contains(&kb) {
            continue;
        }
        let a = mk16(rng.next() & 1, ka, u);
        let b = mk16(0, kb - 1, 1024 - 2 * u);
        let s = f16_to_f64(a as u16) * f16_to_f64(b as u16) + f16_to_f64(c as u16);
        let twice = u64::from(f64_to_f16(f64::from(s as f32)));
        if twice != fma16(a, b, c) {
            out.push([a, b, c]);
            n += 1;
        }
    }
    for _ in 0..600 {
        let r = rng.next();
        let a = mk16(r & 1, ((r >> 1) % 15) as i32 - 7, (r >> 8) & 0x3FF);
        let b = mk16(
            (r >> 20) & 1,
            ((r >> 21) % 15) as i32 - 7,
            (r >> 28) & 0x3FF,
        );
        let p = f64_to_f16(f16_to_f64(a as u16) * f16_to_f64(b as u16));
        let k = (rng.next() % 7) as i64 - 3;
        let c = ((i64::from(p & 0x7FFF) + k).clamp(0, 0x7BFF) as u64)
            | (u64::from(p & 0x8000) ^ 0x8000);
        out.push([a, b, c]);
    }
    out
}

/// A `half` fused multiply-add rounds once on every profile: the exact `float` sum moved to odd,
/// then the one rounding to `half` (no `F16FmaDoubleRounding`).
#[test]
fn half_fused_multiply_add_rounds_once() {
    let triples = half_triples();
    for (lanes, per_lane) in [(1u8, false), (2, false), (2, true)] {
        let fma = Fma {
            form: vec(float(16), lanes),
            route: Route::HalfOdd,
            tiny: None,
        };
        check_fma(
            &format!("fma half x{lanes}, per lane {per_lane}"),
            fma,
            per_lane,
            &lane_cases(&triples, lanes),
            &fma16,
            false,
        );
    }
}

/// The `half` reference itself: exact sums of a few hand-checked triples, signed zeros, the
/// overflow to infinity, and a denormal result.
#[test]
fn the_half_reference_rounds_the_exact_sum() {
    // 2 * 3 - 6 = +0; -0 * 1 + -0 = -0; 1 * 1 + 2^-24 rounds to 1.
    assert_eq!(fma16(0x4000, 0x4200, 0xC600), 0);
    assert_eq!(fma16(0x8000, 0x3C00, 0x8000), 0x8000);
    assert_eq!(fma16(0x3C00, 0x3C00, 0x0001), 0x3C00);
    // 1 * 1 + 2^-11 ties to 1 (even); 1 * 1 + 2^-11 + 2^-21 rounds up to 1 + 2^-10.
    assert_eq!(fma16(0x3C00, 0x3C00, 0x1000), 0x3C00);
    assert_eq!(fma16(0x3C00, 0x3C00, 0x1001), 0x3C01);
    // 65504 * 2 - 65504 = 65504; 256 * 256 - 0 overflows.
    assert_eq!(fma16(0x7BFF, 0x4000, 0xFBFF), 0x7BFF);
    assert_eq!(fma16(0x5C00, 0x5C00, 0), 0x7C00);
    // 2^-24 * 0.5 + 0 = 2^-25 ties to +0; 3 * 2^-24 * 0.5 ties to 2 * 2^-24.
    assert_eq!(fma16(0x0001, 0x3800, 0), 0);
    assert_eq!(fma16(0x0003, 0x3800, 0), 0x0002);
}

/// The route follows the profile: `OpFmaKHR` where `shaderFmaFloat32` is enabled, else `double`
/// where 64-bit floats and integers are, else the integer routine; `half` has its own, the device's
/// 16-bit `Fma` where the driver is measured to compute it as Metal's `half` `fma` and
/// `shaderFloat16` is enabled (the MoltenVK captures alone), else `OpFmaKHR` on 16 bits where
/// `shaderFmaFloat16` is enabled and the profile rounds under the 16-bit float controls, under
/// either policy, and `bfloat` and `double` keep their own lowering.
///
/// Under native device maths (the default policy) a profile whose functions are not Metal's takes
/// the device's `Fma` ([`Route::Native`]) where it would take a routine. Under
/// `FloatMathsPolicy::Exact`, `OpFmaKHR` on a profile whose functions are not Metal's own (not
/// MoltenVK, whose `fma` is Metal's) is checked by the `double` routine where it is offered, else
/// the integer one ([`fma::tiny_route`]), whether or not the device reads its denormals as Metal
/// does (`DriverDeviation::MetalFloatDenormals`): it rounds before it flushes.
#[test]
fn the_fused_multiply_add_route_follows_the_profile() {
    use crate::device::profile::FloatFunctions;
    use crate::error::DriverDeviation;
    use crate::pipeline::testing::exact_maths;
    use crate::spirv::testing::{committed_profiles_with, is_moltenvk_capture};
    // The `half` route a profile offers whatever its policy: Metal's measured `Fma`, or `OpFmaKHR`
    // on 16 bits under the float controls, or none.
    let exact_half = |p: &crate::device::DeviceProfile| {
        let l = &p.fields().leaf;
        if p.takes_driver_deviation(DriverDeviation::MetalHalfFma) && l.float16 {
            Some(Route::F16)
        } else if l.fma16
            && crate::lower::arith::narrow::quantize_form(p)
                == crate::spirv::builder::QuantizeForm::Controlled
        {
            Some(Route::F16Khr)
        } else {
            None
        }
    };
    let mut native = 0;
    let mut khr16 = 0;
    for (stem, p) in committed_profiles().expect("profiles") {
        let l = &p.fields().leaf;
        if p.fields().float_functions != FloatFunctions::Native {
            continue;
        }
        native += 1;
        let want = if l.fma32 { Route::Khr } else { Route::Native };
        let f = Fma::on(&p, float(32)).expect("float");
        assert_eq!((f.route, f.routine()), (want, None), "{stem}");
        let h = Fma::on(&p, float(16)).expect("half");
        let half = exact_half(&p).unwrap_or(Route::Native);
        assert_eq!((h.route, h.routine()), (half, None), "{stem}");
        khr16 += usize::from(half == Route::F16Khr);
    }
    assert!(native >= 10, "{native} profiles take native device maths");
    assert!(khr16 >= 2, "{khr16} profiles take OpFmaKHR on 16 bits");
    let mut seen = std::collections::BTreeSet::new();
    let mut f16 = Vec::new();
    let mut checked = 0;
    let mut flushing = 0;
    let mut metals = 0;
    for (stem, p) in committed_profiles_with(exact_maths()).expect("profiles") {
        let l = &p.fields().leaf;
        let want = if l.fma32 {
            Route::Khr
        } else if l.float64 && l.int64 {
            Route::Wide
        } else {
            Route::Soft
        };
        let f = Fma::on(&p, float(32)).expect("float");
        assert_eq!(f.route, want, "{stem}");
        // Metal's own functions (MoltenVK) take its `fma`, unchecked.
        let device = p.fields().float_functions == FloatFunctions::Device;
        let tiny = (want == Route::Khr && device).then_some(if l.float64 && l.int64 {
            Route::Wide
        } else {
            Route::Soft
        });
        assert_eq!(f.tiny, tiny, "{stem}");
        // `OpFmaKHR` checked through `double` calls the routine of its small results alone.
        assert_eq!(
            f.routine(),
            match want {
                Route::Khr => tiny.map(|t| if t == Route::Wide { Route::Khr } else { t }),
                _ => Some(want),
            }
            .map(|route| Routine { route, lanes: 1 }),
            "{stem}"
        );
        checked += usize::from(tiny.is_some());
        metals += usize::from(want == Route::Khr && !device);
        flushing += usize::from(
            tiny.is_some() && p.takes_driver_deviation(DriverDeviation::MetalFloatDenormals),
        );
        let half = exact_half(&p).unwrap_or(Route::HalfOdd);
        let h = Fma::on(&p, float(16)).expect("half");
        assert_eq!(h.route, half, "{stem}");
        assert_eq!(
            h.routine(),
            (half == Route::HalfOdd).then_some(Routine {
                route: half,
                lanes: 1
            }),
            "{stem}"
        );
        if half == Route::F16 {
            f16.push(stem.clone());
        }
        assert_eq!(half == Route::F16, is_moltenvk_capture(&stem), "{stem}");
        assert_eq!(Fma::on(&p, bfloat()), None, "{stem}");
        assert_eq!(Fma::on(&p, float(64)), None, "{stem}");
        seen.insert(want);
    }
    assert_eq!(
        seen,
        fma::FLOAT_ROUTES.into_iter().collect(),
        "every float route is some committed profile's"
    );
    assert!(f16.len() >= 2, "{f16:?}");
    assert!(checked >= 1, "{checked} profiles check OpFmaKHR");
    assert!(
        flushing >= 1,
        "{flushing} profiles check OpFmaKHR on a device that flushes its denormals"
    );
    assert!(
        metals >= 1,
        "{metals} profiles take Metal's own OpFmaKHR unchecked"
    );
}

/// The words of a module of the functions `defs` defines and an empty entry point.
fn defined_words(defs: &dyn Fn(&mut ModuleBuilder) -> BResult<()>) -> usize {
    let mut b = ModuleBuilder::new();
    defs(&mut b).expect("define");
    let void = b.ty(Type::Void).expect("void");
    let fb = b.function(void).expect("entry");
    let id = {
        let mut fb = fb;
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
    crate::spirv::serialize::serialize(m)
        .expect("serialize")
        .words
        .len()
}

/// The routines' words stay within their estimates: each routine's function, defined alone, and
/// each use; and a routine defined per lane, beside its route's routine on one lane, its type
/// included. `half`'s in the integer rounding and in the device's quantize. A use of `OpFmaKHR`
/// checked ([`Fma::checked`]) stays within its own estimate, on one lane and four, by the integer
/// routine and by the routine of its small results.
#[test]
fn fused_multiply_add_routines_fit_their_word_estimates() {
    use crate::spirv::builder::QuantizeForm;
    let lanes_and_routes = [1u8, 4]
        .into_iter()
        .flat_map(|lanes| [(lanes, Route::Soft), (lanes, Route::Wide)]);
    for (lanes, tiny) in lanes_and_routes {
        let fma = Fma {
            form: floats(lanes),
            route: Route::Khr,
            tiny: Some(tiny),
        };
        let r = fma.routine().expect("a checked use's routine");
        // The routine and a function of three floats that answers their `fma`, or its first.
        let module = |uses: bool| {
            defined_words(&|b| {
                let id = fma::define(b, r)?;
                let e = b.ty(Type::Float { width: 32 })?;
                let t = match lanes {
                    1 => e,
                    len => b.ty(Type::Vector { elem: e, len })?,
                };
                let mut fb = b.routine(t)?;
                let xyz = [fb.param(t)?, fb.param(t)?, fb.param(t)?];
                let v = match uses {
                    true => fma::emit(&mut fb, fma, &xyz, Some(id))?,
                    false => xyz[0],
                };
                fb.ret(Some(v))?;
                fb.finish().map(drop)
            })
        };
        let words = module(true) - module(false);
        assert!(
            words <= fma.words() as usize,
            "checked by {tiny:?} x{lanes}: {words} words against the estimate {}",
            fma.words()
        );
    }
    let within = |what: &str, words: usize, estimate: u32| {
        let estimate = estimate as usize;
        assert!(
            words <= estimate && words * 3 >= estimate * 2,
            "{what}: {words} words against the estimate {estimate} (within its last third)"
        );
    };
    for q in [QuantizeForm::Integer, QuantizeForm::Device] {
        let in_q = |defs: &dyn Fn(&mut ModuleBuilder) -> BResult<()>| {
            defined_words(&|b| {
                b.set_quantize_form(q);
                defs(b)
            })
        };
        for route in [Route::Wide, Route::Soft, Route::HalfOdd] {
            let one = Routine { route, lanes: 1 };
            let alone = in_q(&|b| fma::define(b, one).map(drop));
            for lanes in [1u8, 4] {
                let r = Routine { route, lanes };
                let words = in_q(&|b| fma::define(b, r).map(drop));
                within(&format!("{route:?} x{lanes} ({q:?})"), words, r.words(q));
            }
            for lanes in [2u8, 4] {
                let r = Routine { route, lanes };
                let both = in_q(&|b| {
                    let id = fma::define(b, one)?;
                    fma::define_per_lane(b, r, id).map(drop)
                });
                within(
                    &format!("{route:?} x{lanes} per lane ({q:?})"),
                    both - alone,
                    r.per_lane_words(),
                );
            }
        }
        // The routine of `OpFmaKHR`'s small results, never defined per lane.
        for lanes in [1u8, 4] {
            let r = Routine {
                route: Route::Khr,
                lanes,
            };
            let words = in_q(&|b| fma::define(b, r).map(drop));
            within(&format!("Khr x{lanes} ({q:?})"), words, r.words(q));
        }
    }
}

/// A route a module calls on one lane and on more is defined per lane on more; one called only on
/// `k` (two or more) vector lane counts gets its routine on one lane where the `k - 1` sequences
/// that saves outweigh the per-lane routines twice (the integer and `double` routes on two lane
/// counts and on three, and `half`'s on three, not on two, where its rounding is the device's
/// quantize; in the integer rounding, on two too); a route called on one lane count keeps its one
/// routine, on any lanes. Each routine is charged its per-lane words where it is defined per lane.
#[test]
fn a_route_on_several_lane_counts_holds_its_sequence_once() {
    use crate::spirv::builder::QuantizeForm;
    let q = QuantizeForm::Device;
    let r = |route, lanes| Routine { route, lanes };
    let mut fmas = BTreeSet::from([
        r(Route::Soft, 2),
        r(Route::Soft, 4),
        r(Route::HalfOdd, 2),
        r(Route::HalfOdd, 4),
        r(Route::Wide, 1),
        r(Route::Wide, 3),
    ]);
    fma::complete(&mut fmas, q);
    assert_eq!(
        fmas,
        BTreeSet::from([
            r(Route::Soft, 1),
            r(Route::Soft, 2),
            r(Route::Soft, 4),
            r(Route::HalfOdd, 2),
            r(Route::HalfOdd, 4),
            r(Route::Wide, 1),
            r(Route::Wide, 3),
        ])
    );
    let per_lane: Vec<Routine> = fmas
        .iter()
        .copied()
        .filter(|x| fma::per_lane(*x, &fmas))
        .collect();
    assert_eq!(
        per_lane,
        [r(Route::Wide, 3), r(Route::Soft, 2), r(Route::Soft, 4)]
    );
    // A double route on two vector lane counts gets its one-lane routine too, and so does one on
    // three, which saves two sequences, as does `half`'s on three.
    for lanes in [&[2u8, 4][..], &[2, 3, 4]] {
        let mut double: BTreeSet<Routine> = lanes.iter().map(|&n| r(Route::Wide, n)).collect();
        fma::complete(&mut double, q);
        assert!(double.contains(&r(Route::Wide, 1)), "{double:?}");
    }
    let mut half = BTreeSet::from([
        r(Route::HalfOdd, 2),
        r(Route::HalfOdd, 3),
        r(Route::HalfOdd, 4),
    ]);
    fma::complete(&mut half, q);
    assert!(half.contains(&r(Route::HalfOdd, 1)), "{half:?}");
    for x in &fmas {
        let want = if per_lane.contains(x) {
            x.per_lane_words()
        } else {
            x.words(q)
        };
        assert_eq!(x.words_among(&fmas, q), want, "{x:?}");
    }
    // Completing again adds nothing.
    let again = fmas.clone();
    fma::complete(&mut fmas, q);
    assert_eq!(fmas, again);
    // In the integer rounding a `half` sequence is the larger by its rounding: on two and four
    // lanes too, its routines call the one-lane routine.
    let mut half = BTreeSet::from([r(Route::HalfOdd, 2), r(Route::HalfOdd, 4)]);
    fma::complete(&mut half, QuantizeForm::Integer);
    assert!(half.contains(&r(Route::HalfOdd, 1)), "{half:?}");
}

/// Run Metal's `mix` ([`math::metal_mix`]) through `fma`'s route (its routine defined when it
/// has one) over `cases` and compare with `reference` per lane, as [`check_fma`] does.
fn check_mix(
    what: &str,
    fma: Fma,
    cases: &[Case],
    reference: &dyn Fn(u64, u64, u64) -> u64,
    exact: bool,
) {
    let f = fma.form;
    let routine = fma.routine();
    let m = kernel_defs(
        &[f, f, f],
        f,
        &|b| {
            Ok(routine
                .map(|r| fma::define(b, r))
                .transpose()?
                .into_iter()
                .collect())
        },
        &|fb, x, funcs| {
            let id = funcs.first().copied();
            let fused = |fb: &mut FunctionBuilder<'_>, args: &[Value]| fma::emit(fb, fma, args, id);
            math::metal_mix(fb, f, [x[0], x[1], x[2]], &fused, false)
        },
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
    check_module(
        what,
        m,
        &[f, f, f],
        f,
        cases,
        &|c| {
            Some(
                (0..c[0].len())
                    .map(|l| reference(c[0][l], c[1][l], c[2][l]))
                    .collect(),
            )
        },
        &|g, w| if exact { g == w } else { same(f)(g, w) },
    );
}

/// Metal's recorded `float` `mix(x, y, t)` words, `(x, y, t, answer)`: `nan_functions`'
/// `mix(inf, 1, 0)` is NaN (`fma(-0, inf, inf)`) and `mix(-0, -0, 0)` is `+0`; `nu::mix`
/// (978cc308) at `t = 1` answers `y` where `x + (y - x) * t` loses it; `mix-an-infinite-endpoint`
/// (412028fc) answers NaN for an infinite or NaN endpoint weighted out, `+0` for `mix(1, -0, 1)`,
/// and the ordinary blends.
fn metal_float_mixes() -> Vec<[u64; 4]> {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    [
        (inf, 1.0, 0.0, nan),
        (-0.0, -0.0, 0.0, 0.0),
        (1e10, 1.0, 1.0, 1.0),
        (0.0, 7.0, 1.0, 7.0),
        (-1e10, 3.0, 1.0, 3.0),
        (2.0, 5.0, 1.0, 5.0),
        (inf, 5.0, 1.0, nan),
        (5.0, inf, 0.0, nan),
        (nan, 5.0, 1.0, nan),
        (5.0, nan, 0.0, nan),
        (inf, inf, 1.0, nan),
        (1.0, -0.0, 1.0, 0.0),
        (3.0, 7.0, 1.0, 7.0),
        (3.0, 7.0, 0.0, 3.0),
        (2.0, 4.0, 0.5, 3.0),
    ]
    .iter()
    .map(|&(x, y, t, r): &(f32, f32, f32, f32)| {
        let r = if r.is_nan() { 0x7FC0_0000 } else { r.to_bits() };
        [x, y, t]
            .map(|v| u64::from(v.to_bits()))
            .into_iter()
            .chain([u64::from(r)])
            .collect::<Vec<_>>()
            .try_into()
            .expect("four")
    })
    .collect()
}

/// Metal's `mix(x, y, t)` on `float` and `half` is `fma(t, y, fma(-t, x, x))`, each fused
/// multiply-add rounding once on the profile's route, on scalars and vectors. Against the
/// composition of each route's reference over the fused multiply-add triples (as `(x, y, t)`):
/// bit for bit on the routines, which answer Metal's `fma` ([`metal_fma32`]); `OpFmaKHR` and the
/// `half` route by value, a NaN for a NaN. And Metal's recorded words: [`metal_float_mixes`] on
/// every `float` route (bit for bit on the routines), and `nan_half`'s `mix(-0.3, 0.7, 0.2)`
/// (`0xB4CD`, `0x399A`, `0x3266`), `0xAE68` where the blend gives `0xAE66`, and `mix(inf, 0, 0)`
/// and `mix(inf, 1, 0.5)`, NaN.
#[test]
fn mix_is_metals_two_fused_multiply_adds() {
    let flip32 = |t: u64| t ^ 0x8000_0000;
    let triples = float_triples();
    let pinned = metal_float_mixes();
    for (route, exact) in [
        (Route::Khr, false),
        (Route::Wide, true),
        (Route::Soft, true),
    ] {
        let fused = move |a: u64, b: u64, c: u64| match route {
            Route::Khr => fma32(a, b, c),
            _ => metal_fma32(a, b, c),
        };
        let reference = move |x: u64, y: u64, t: u64| fused(t, y, fused(flip32(t), x, x));
        for lanes in [1u8, 3] {
            let fma = Fma {
                form: floats(lanes),
                route,
                tiny: None,
            };
            check_mix(
                &format!("mix {route:?} x{lanes}"),
                fma,
                &lane_cases(&triples, lanes),
                &reference,
                exact,
            );
            let rows: Vec<[u64; 3]> = pinned.iter().map(|r| [r[0], r[1], r[2]]).collect();
            let want = |x: u64, y: u64, t: u64| {
                pinned
                    .iter()
                    .find(|r| [r[0], r[1], r[2]] == [x, y, t])
                    .map_or(0, |r| r[3])
            };
            check_mix(
                &format!("mix {route:?} x{lanes}, Metal's words"),
                fma,
                &lane_cases(&rows, lanes),
                &want,
                exact,
            );
        }
    }
    let half_triples = half_triples();
    let reference = |x: u64, y: u64, t: u64| fma16(t, y, fma16(t ^ 0x8000, x, x));
    let pinned: [[u64; 4]; 3] = [
        [0xB4CD, 0x399A, 0x3266, 0xAE68],
        [0x7C00, 0, 0, 0x7E00],
        [0x7C00, 0x3C00, 0x3800, 0x7E00],
    ];
    for lanes in [1u8, 2] {
        let fma = Fma {
            form: vec(float(16), lanes),
            route: Route::HalfOdd,
            tiny: None,
        };
        check_mix(
            &format!("mix half x{lanes}"),
            fma,
            &lane_cases(&half_triples, lanes),
            &reference,
            false,
        );
        let rows: Vec<[u64; 3]> = pinned.iter().map(|r| [r[0], r[1], r[2]]).collect();
        let want = |x: u64, y: u64, t: u64| {
            pinned
                .iter()
                .find(|r| [r[0], r[1], r[2]] == [x, y, t])
                .map_or(0, |r| r[3])
        };
        check_mix(
            &format!("mix half x{lanes}, Metal's words"),
            fma,
            &lane_cases(&rows, lanes),
            &want,
            false,
        );
    }
    // The blend misses Metal's `half` word, which the fused form answers.
    let (x, y, t) = (f16_to_f64(0xB4CD), f16_to_f64(0x399A), f16_to_f64(0x3266));
    let q = |v: f64| f16_to_f64(f64_to_f16(v));
    assert_eq!(f64_to_f16(q(q(x * q(1.0 - t)) + q(y * t))), 0xAE66);
}

/// Metal's recorded `float` `dot` words (a probe of 4,096 lanes; the M3 Max and the M4 answer every
/// word alike, with and without fast math): `x`, `y`, and the `dot` of their first two, three and
/// four lanes, at lanes where the fused chain alone gives all three (every other order of the
/// chain, the pairwise tree, the unfused sum and the exact sum miss one).
const METAL_FLOAT_DOTS: [([u32; 4], [u32; 4], [u32; 3]); 6] = [
    (
        [0xBF1DBDC5, 0xC128181E, 0x3F354B70, 0x3E3EE94D],
        [0x40F04319, 0xBEE176DD, 0x3EDB66A8, 0xBFFC6F88],
        [0xB599D686, 0x3E9B6015, 0xBD8382FE],
    ),
    (
        [0x3EF1628A, 0xBEB13510, 0xBFCD5931, 0x3F5BA54E],
        [0xC1647CE8, 0xC19B9E64, 0xBF1335AB, 0xBE75DE97],
        [0x34226900, 0x3F6C2A8A, 0x3F376D7E],
    ),
    (
        [0xBF176CF7, 0xBEE9DC9D, 0xBF412D4B, 0x3FEF20EF],
        [0x41233C41, 0x41121529, 0xC13A10CB, 0x3F967F46],
        [0xC1234799, 0xBFB70024, 0x3F445073],
    ),
    (
        [0x40E6B4B3, 0x41690D30, 0x3F8F6CD2, 0x3F2017B4],
        [0xC10270AC, 0x40812091, 0x3ECD7C24, 0xBED2C469],
        [0xB78A61F4, 0x3EE63D39, 0x3E44DDE7],
    ),
    (
        [0x414EAD1A, 0xBEFDED98, 0xC005B664, 0xC04BDDB4],
        [0x41380482, 0x4395C65E, 0x40918EB8, 0xC0049177],
        [0x37C39918, 0xC1180D9B, 0xC039EDCB],
    ),
    (
        [0x3F2121E3, 0xC1504426, 0x40D96C9D, 0x3EC6C559],
        [0xBFD689B8, 0xC0608ADE, 0xC0D7159F, 0x402DE9ED],
        [0x42327452, 0xBF87085A, 0x37A4C7B4],
    ),
];

/// As [`METAL_FLOAT_DOTS`], on `half`: lanes where the chain, rounding each step to `half`,
/// gives all three words, and the `float` sum rounded once to `half` misses one.
const METAL_HALF_DOTS: [([u16; 4], [u16; 4], [u16; 3]); 6] = [
    (
        [0xC727, 0xC4B5, 0x44D4, 0xC26F],
        [0xC15C, 0xB60C, 0xC3F0, 0x39C3],
        [0x4D3D, 0x3F2A, 0xB835],
    ),
    (
        [0x47FB, 0xBA56, 0xC4DE, 0x322F],
        [0xC570, 0xD2DA, 0xBF0B, 0xBB49],
        [0x1CF0, 0x484A, 0x4833],
    ),
    (
        [0x3F28, 0x4464, 0xBE4B, 0xC32A],
        [0xB34F, 0x2DF5, 0xB0E5, 0x323D],
        [0x8130, 0x33B3, 0xB753],
    ),
    (
        [0x4B64, 0x42EF, 0x3E65, 0x481C],
        [0xB8A4, 0xC01C, 0x4476, 0x3C2C],
        [0xCBDA, 0xC849, 0x1680],
    ),
    (
        [0x3B64, 0x3CF3, 0xCACA, 0xC15F],
        [0x4B0C, 0x3847, 0x3CDB, 0x302C],
        [0x4AD8, 0xC197, 0xC24A],
    ),
    (
        [0xB3DD, 0xB0C3, 0xBB51, 0x3CD8],
        [0xC76B, 0x4A20, 0x31C7, 0xBA80],
        [0x0E80, 0xB145, 0xBC98],
    ),
];

/// Run Metal's `dot` ([`math::metal_dot`]) of two vectors of `lanes` lanes through `fma`'s route
/// on one lane (its routine defined when it has one), flushing the chain's values where `flush`,
/// over `cases` and compare with `reference(x, y)`: bit for bit when `exact`, else any NaN for a
/// NaN.
fn check_dot(
    what: &str,
    fma: Fma,
    lanes: u8,
    flush: bool,
    cases: &[Case],
    reference: &dyn Fn(&[u64], &[u64]) -> u64,
    exact: bool,
) {
    let s = fma.form;
    let v = vec(s, lanes);
    let routine = fma.routine();
    let m = kernel_defs(
        &[v, v],
        s,
        &|b| {
            Ok(routine
                .map(|r| fma::define(b, r))
                .transpose()?
                .into_iter()
                .collect())
        },
        &|fb, x, funcs| {
            let id = funcs.first().copied();
            let fused = |fb: &mut FunctionBuilder<'_>, args: &[Value]| fma::emit(fb, fma, args, id);
            math::metal_dot(fb, v, [x[0], x[1]], &fused, (flush, None))
        },
        true,
        false,
    )
    .unwrap_or_else(|e| panic!("{what}: build: {e:?}"));
    check_module(
        what,
        m,
        &[v, v],
        s,
        cases,
        &|c| Some(vec![reference(&c[0], &c[1])]),
        &|g, w| if exact { g == w } else { same(s)(g, w) },
    );
}

/// The cases of two vectors of `lanes` lanes: `x`'s lanes from the first column of `lanes`
/// consecutive `triples`, `y`'s from the second.
fn dot_cases(triples: &[[u64; 3]], lanes: u8) -> Vec<Case> {
    let n = usize::from(lanes);
    (0..triples.len())
        .step_by(n)
        .map(|i| {
            (0..2)
                .map(|k| {
                    (0..n)
                        .map(|l| triples[(i + l) % triples.len()][k])
                        .collect()
                })
                .collect()
        })
        .collect()
}

/// Metal's `dot` on `float` and `half` is the products' fused chain, `fma(x[n-1], y[n-1], ...
/// fma(x[1], y[1], x[0] * y[0]))`, on two, three and four lanes. On every `float` route against the
/// chain of each route's fused multiply-add over the fused multiply-add triples (bit for bit on the
/// routines, which answer Metal's `fma`; with the chain's values flushed on `OpFmaKHR` where the
/// device may keep a denormal), and against Metal's recorded words ([`METAL_FLOAT_DOTS`], bit for
/// bit on every route); on `half`'s routine the same against the `half` chain and
/// [`METAL_HALF_DOTS`].
#[test]
fn dot_is_metals_fused_chain() {
    let triples = float_triples();
    for (route, flush, exact) in [
        (Route::Khr, false, false),
        (Route::Khr, true, false),
        (Route::Wide, false, true),
        (Route::Soft, false, true),
    ] {
        let step = move |a: u64, b: u64, c: u64| match (route, flush) {
            (Route::Khr, false) => fma32(a, b, c),
            (Route::Khr, true) => fma32(a, b, ftz(c)),
            _ => metal_fma32(a, b, c),
        };
        let reference = move |x: &[u64], y: &[u64]| {
            let mut sum = u64::from((f32v(x[0]) * f32v(y[0])).to_bits());
            for l in 1..x.len() {
                sum = step(x[l], y[l], sum);
            }
            sum
        };
        let fma = Fma {
            form: floats(1),
            route,
            tiny: None,
        };
        for lanes in [2u8, 3, 4] {
            let what = format!("dot {route:?} x{lanes}, flush {flush}");
            let cases = dot_cases(&triples, lanes);
            check_dot(&what, fma, lanes, flush, &cases, &reference, exact);
            let n = usize::from(lanes);
            let pinned: Vec<Case> = METAL_FLOAT_DOTS
                .iter()
                .map(|(x, y, _)| {
                    [x, y]
                        .map(|v| v[..n].iter().map(|b| u64::from(*b)).collect())
                        .to_vec()
                })
                .collect();
            let want = |x: &[u64], _: &[u64]| {
                METAL_FLOAT_DOTS
                    .iter()
                    .find(|(mx, ..)| mx[..n].iter().map(|b| u64::from(*b)).eq(x.iter().copied()))
                    .map_or(0, |(.., w)| u64::from(w[n - 2]))
            };
            check_dot(
                &format!("{what}, Metal's words"),
                fma,
                lanes,
                flush,
                &pinned,
                &want,
                true,
            );
        }
    }
    let half_triples = half_triples();
    let reference = |x: &[u64], y: &[u64]| {
        let mut sum = fma16(x[0], y[0], 0x8000);
        for l in 1..x.len() {
            sum = fma16(x[l], y[l], sum);
        }
        sum
    };
    let fma = Fma {
        form: float(16),
        route: Route::HalfOdd,
        tiny: None,
    };
    for lanes in [2u8, 3, 4] {
        let what = format!("dot half x{lanes}");
        check_dot(
            &what,
            fma,
            lanes,
            false,
            &dot_cases(&half_triples, lanes),
            &reference,
            false,
        );
        let n = usize::from(lanes);
        let pinned: Vec<Case> = METAL_HALF_DOTS
            .iter()
            .map(|(x, y, _)| {
                [x, y]
                    .map(|v| v[..n].iter().map(|b| u64::from(*b)).collect())
                    .to_vec()
            })
            .collect();
        let want = |x: &[u64], _: &[u64]| {
            METAL_HALF_DOTS
                .iter()
                .find(|(mx, ..)| mx[..n].iter().map(|b| u64::from(*b)).eq(x.iter().copied()))
                .map_or(0, |(.., w)| u64::from(w[n - 2]))
        };
        check_dot(
            &format!("{what}, Metal's words"),
            fma,
            lanes,
            false,
            &pinned,
            &want,
            true,
        );
    }
}
