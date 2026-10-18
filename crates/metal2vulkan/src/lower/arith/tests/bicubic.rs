//! Interpreter differential tests of the bicubic filter's routines
//! (`lower::sampler::bicubic::routines`), which compute with the exact texel conversions of
//! `texel_round`, and of the linear filter's. References are plain Rust: Metal's rules in exact
//! integer arithmetic for the exact routines, Metal's accumulators exactly through `f64` for the
//! accumulated sums (the linear filter's, and the bicubic filter's of 32-bit float and half
//! formats), and the routines' own `f32` order (IEEE, as the interpreter evaluates it) for the
//! other formats' bicubic sums, whose differences from Metal the precision note records. Pinned
//! lanes are Metal's own answers.

use super::texel_round::half_of;
use super::*;
use crate::lower::arith::texel_round::HalfRounding;
use crate::lower::sampler::bicubic::routines::{self, Routine, RowIndex};
use crate::metal::bicubic::{weights, BicubicTexels as T};

fn u() -> Form {
    int(32)
}

fn v4() -> Form {
    vec(float(32), 4)
}

/// `n / d` (`0 <= n <= d`) rounded to nearest even with `p` significand bits and least normal
/// exponent `emin`: exact integer arithmetic.
fn round_ratio(n: u64, d: u64, p: i32, emin: i32) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let (n, d) = (u128::from(n), u128::from(d));
    let mut e = 0i32;
    while n << ((-e) as u32) < d {
        e -= 1;
    }
    let e = e.max(emin);
    let num = n << (p - 1 - e) as u32;
    let (q, r) = (num / d, num % d);
    let up = 2 * r > d || (2 * r == d && q & 1 == 1);
    (q + u128::from(up)) as f64 * 2f64.powi(e - p + 1)
}

/// An exact format: its flags, code range, and grid `D`.
#[derive(Clone, Copy, Debug)]
struct Exact {
    flags: u32,
    lo: i64,
    hi: i64,
    /// The code scale `2^b - 1` (`2^(b-1) - 1` signed).
    scale: i64,
    d: i64,
}

const EXACT_FORMATS: [Exact; 4] = [
    Exact {
        flags: T::UNORM | T::EXACT,
        lo: 0,
        hi: 255,
        scale: 255,
        d: 4080,
    },
    Exact {
        flags: T::SNORM | T::EXACT,
        lo: -128,
        hi: 127,
        scale: 127,
        d: 2032,
    },
    Exact {
        flags: T::UNORM | T::EXACT | T::WIDE16,
        lo: 0,
        hi: 65535,
        scale: 65535,
        d: 65535,
    },
    Exact {
        flags: T::SNORM | T::EXACT | T::WIDE16,
        lo: -32768,
        hi: 32767,
        scale: 32767,
        d: 32767,
    },
];

impl Exact {
    /// The value Vulkan reads for code `c` (`f32` bits): `c / scale` correctly rounded, `-1` for
    /// the least signed code.
    fn value(self, c: i64) -> u64 {
        let c = c.max(-self.scale);
        let m = round_ratio(c.unsigned_abs(), self.scale as u64, 24, -126);
        u64::from((m.copysign(c as f64) as f32).to_bits())
    }

    /// The code the sum reads for code `c`, on the grid: `-2^(b-1)` reads as `-(2^(b-1) - 1)`.
    fn step(self, c: i64) -> i64 {
        c.max(-self.scale) * (self.d / self.scale)
    }

    /// Metal's result for grid step `n` (clamped): `f32` bits of `n / D`, or of the `half`.
    fn result(self, n: i64, half: bool) -> u64 {
        let lo = if self.flags & T::SNORM != 0 {
            -self.d
        } else {
            0
        };
        let n = n.clamp(lo, self.d);
        let (p, emin) = if half { (11, -14) } else { (24, -126) };
        let m = round_ratio(n.unsigned_abs(), self.d as u64, p, emin);
        u64::from((m.copysign(n as f64) as f32).to_bits())
    }
}

/// Codes: random, the range's ends, and equal codes (a sum that is that code exactly).
fn codes(f: Exact, rng: &mut Rng, n: usize) -> Vec<i64> {
    let span = (f.hi - f.lo + 1) as u64;
    let pick = rng.next() % 8;
    (0..n)
        .map(|i| match pick {
            0 => f.hi,
            1 => f.lo,
            2 => [f.lo, f.hi][i % 2],
            3 => f.lo + (rng.next() % 3) as i64,
            _ => f.lo + (rng.next() % span) as i64,
        })
        .collect()
}

fn fraction(rng: &mut Rng) -> u64 {
    match rng.next() % 6 {
        0 => 0,
        1 => 128,
        2 => 255,
        _ => rng.next() % 256,
    }
}

/// The weights routine is [`weights`] at every fraction.
#[test]
fn the_weights_routine_is_metals_table() {
    let cases: Vec<Case> = (0..256u64).map(|k| vec![vec![k]]).collect();
    check(
        "bicubic weights",
        &[u()],
        vec(int(32), 4),
        &|fb, x| {
            let p = routines::define_plane(fb.module())?;
            Ok(fb.call(p.weights, &[x[0]])?.expect("weights"))
        },
        &cases,
        &|c| {
            let w = weights(arg(c, 0) as u32);
            Some(w.iter().map(|w| u64::from(*w as u32)).collect())
        },
        false,
    );
}

/// The exact 2D routine is Metal's rule on every exact format: the codes' exact sum under the
/// table's weights, rounded half up onto the grid, clamped, and divided exactly, to `f32` or to
/// `half`.
#[test]
fn the_exact_plane_rounds_the_exact_sum_half_up() {
    let mut ins = vec![v4(); 16];
    ins.extend([u(), u(), u()]);
    let mut rng = Rng(0x00b1_c0b1_c5ee_d001);
    for f in EXACT_FORMATS {
        let mut cases = Vec::new();
        for _ in 0..600 {
            let (kx, ky) = (fraction(&mut rng), fraction(&mut rng));
            let taps: Vec<Vec<i64>> = (0..16).map(|_| codes(f, &mut rng, 4)).collect();
            let mut c: Case = taps
                .iter()
                .map(|t| t.iter().map(|c| f.value(*c)).collect())
                .collect();
            c.extend([vec![kx], vec![ky], vec![u64::from(f.flags)]]);
            cases.push((c, taps));
        }
        for half in [false, true] {
            let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                let p = routines::define_plane(fb.module())?;
                let wx = fb.call(p.weights, &[x[16]])?.expect("weights");
                let wy = fb.call(p.weights, &[x[17]])?.expect("weights");
                let h = ops::splat(fb, boolean(), u64::from(half))?;
                let mut args = x[..16].to_vec();
                args.extend([wx, wy, x[18], h]);
                Ok(fb.call(p.exact, &args)?.expect("exact"))
            };
            let reference = |c: &Case| {
                let (_, taps) = cases.iter().find(|(k, _)| k == c)?;
                let (wx, wy) = (weights(arg(c, 16) as u32), weights(arg(c, 17) as u32));
                let lanes = (0..4).map(|l| {
                    let mut s: i64 = 0;
                    for j in 0..4 {
                        for i in 0..4 {
                            let w = i64::from(wx[i]) * i64::from(wy[j]);
                            s += f.step(taps[4 * j + i][l]) * w;
                        }
                    }
                    f.result((s + (1 << 31)) >> 32, half)
                });
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("exact plane {f:?} half={half}"),
                &ins,
                v4(),
                &body,
                &only,
                &reference,
                false,
            );
        }
    }
}

/// The exact 1D routine: `⌊(c₀ (256 - k) + c₁ k + 128) / 256⌋` on the grid, every fraction.
#[test]
fn the_exact_line_rounds_the_exact_sum_half_up() {
    let ins = [v4(), v4(), u(), u()];
    let mut rng = Rng(0x00b1_c0b1_c5ee_d002);
    for f in EXACT_FORMATS {
        let mut cases = Vec::new();
        for k in 0..256u64 {
            for _ in 0..3 {
                let taps: Vec<Vec<i64>> = (0..2).map(|_| codes(f, &mut rng, 4)).collect();
                let mut c: Case = taps
                    .iter()
                    .map(|t| t.iter().map(|c| f.value(*c)).collect())
                    .collect();
                c.extend([vec![k], vec![u64::from(f.flags)]]);
                cases.push((c, taps));
            }
        }
        for half in [false, true] {
            let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                let l = routines::define_line(fb.module())?;
                let h = ops::splat(fb, boolean(), u64::from(half))?;
                Ok(fb
                    .call(l.exact, &[x[0], x[1], x[2], x[3], h])?
                    .expect("exact"))
            };
            let reference = |c: &Case| {
                let (_, taps) = cases.iter().find(|(k, _)| k == c)?;
                let k = arg(c, 2) as i64;
                let lanes = (0..4).map(|l| {
                    let s = f.step(taps[0][l]) * (256 - k) + f.step(taps[1][l]) * k;
                    f.result((s + 128) >> 8, half)
                });
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("exact line {f:?} half={half}"),
                &ins,
                v4(),
                &body,
                &only,
                &reference,
                false,
            );
        }
    }
}

/// The grid step of a 2D exact sum of format lane `f`'s codes `codes` (taps `(i, j)`, `(i + 1,
/// j)`, `(i, j + 1)`, `(i + 1, j + 1)`) at fractions `k` ([`quad_weights`]), its terms reduced by
/// `terms` as a 3D sum's are ([`grid_step`]): Metal's truncates a 16-bit format's onto `2^-8` of
/// its codes and every other grid's onto `2^-4`, where only an sRGB color lane's step loses bits
/// (an 8-bit lane's `16 · code` term is a multiple of 16).
fn quad_step(f: Exact, codes: &[i64; 4], [kx, ky]: [u32; 2], terms: Terms) -> i64 {
    grid_step(f, codes, &quad_weights(kx, ky), terms)
}

/// The routines' case of a 2D footprint: the four taps' values by lane (`codes[lane][tap]`), the
/// fractions, and the format's flags.
fn quad_case(lanes: [Exact; 4], codes: &[[i64; 4]; 4], k: [u32; 2]) -> Case {
    let mut c: Case = (0..4)
        .map(|i| (0..4).map(|l| lanes[l].value(codes[l][i])).collect())
        .collect();
    c.extend(k.map(|k| vec![u64::from(k)]));
    c.push(vec![u64::from(lanes[0].flags)]);
    c
}

/// A body calling the exact linear 2D routine on a [`quad_case`]'s inputs, the `half` result
/// constant.
fn quad_body(half: bool) -> impl Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value> {
    move |fb, x| {
        let q = routines::define_quad(fb.module())?;
        let h = ops::splat(fb, boolean(), u64::from(half))?;
        let mut args = x[..7].to_vec();
        args.push(h);
        Ok(fb.call(q.exact, &args)?.expect("exact"))
    }
}

/// The inputs of a [`quad_case`].
fn quad_ins() -> Vec<Form> {
    let mut ins = vec![v4(); 4];
    ins.extend([u(), u(), u()]);
    ins
}

/// The exact linear 2D routine is Metal's sum ([`quad_step`]) on random codes of every exact format
/// and an 8-bit sRGB format's raw steps, at fractions on and off the edges, with either result; for
/// the 8-bit formats, whose terms lose nothing to Metal's truncation, it is the same step as the
/// exact 2D bicubic routine's under the weights `[0, (256 - k) · 256, k · 256, 0]`.
#[test]
fn the_exact_quad_is_metals_2d_sum() {
    let ins = quad_ins();
    let mut rng = Rng(0x00b1_c0b1_c5ee_d005);
    let mut formats: Vec<[Exact; 4]> = EXACT_FORMATS.iter().map(|f| [*f; 4]).collect();
    formats.push(srgb_lanes());
    for lanes in formats {
        let mut cases = Vec::new();
        for n in 0..1200u32 {
            let k = match n {
                0..256 => [n, fraction(&mut rng) as u32],
                256..512 => [fraction(&mut rng) as u32, n - 256],
                _ => [fraction(&mut rng) as u32, fraction(&mut rng) as u32],
            };
            let cs: [[i64; 4]; 4] = std::array::from_fn(|l| {
                let c = codes(lanes[l], &mut rng, 4);
                std::array::from_fn(|i| c[i])
            });
            cases.push((quad_case(lanes, &cs, k), cs));
        }
        for half in [false, true] {
            let quad = quad_body(half);
            // The bicubic routine on the footprint whose inner taps are the quad's.
            let plane = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                let p = routines::define_plane(fb.module())?;
                let h = ops::splat(fb, boolean(), u64::from(half))?;
                let z = ops::splat(fb, v4(), 0)?;
                let weights = |fb: &mut FunctionBuilder<'_>, k: Value| -> BResult<Value> {
                    let c256 = ops::splat(fb, u(), 256)?;
                    let eight = ops::splat(fb, u(), 8)?;
                    let zero = ops::splat(fb, u(), 0)?;
                    let rest = fb.binary(BinOp::ISub, c256, k)?;
                    let w0 = fb.binary(BinOp::ShiftLeftLogical, rest, eight)?;
                    let w1 = fb.binary(BinOp::ShiftLeftLogical, k, eight)?;
                    let ty = ops::ty(fb, vec(int(32), 4))?;
                    fb.composite_construct(ty, &[zero, w0, w1, zero])
                };
                let wx = weights(fb, x[4])?;
                let wy = weights(fb, x[5])?;
                let mut args = vec![z, z, z, z, z, x[0], x[1], z, z, x[2], x[3], z, z, z, z, z];
                args.extend([wx, wy, x[6], h]);
                Ok(fb.call(p.exact, &args)?.expect("exact"))
            };
            let reference = |c: &Case| {
                let (_, cs) = cases.iter().find(|(k, _)| k == c)?;
                let k = [arg(c, 4) as u32, arg(c, 5) as u32];
                let lanes = (0..4)
                    .map(|l| lanes[l].result(quad_step(lanes[l], &cs[l], k, Terms::Floor), half));
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            let mut bodies = vec![("quad", &quad as Body<'_>)];
            if lanes[0].flags & (T::WIDE16 | T::SRGB) == 0 {
                bodies.push(("plane", &plane as Body<'_>));
            }
            for (what, body) in bodies {
                check(
                    &format!("exact {what} {:?} half={half}", lanes[0]),
                    &ins,
                    v4(),
                    body,
                    &only,
                    &reference,
                    false,
                );
            }
        }
    }
}

/// A lane of the 2D probes (Apple M4, macOS 27.0.1; the M3 Max's results byte-identical) that pins
/// one rule of Metal's 2D exact sum: the format's lanes, the fractions as Metal reads them, the
/// taps' codes by lane (an sRGB color lane's table steps), Metal's `float` result's bits by lane,
/// and a reading the lane rules out.
struct QuadPin {
    what: &'static str,
    lanes: [Exact; 4],
    k: [u32; 2],
    codes: [[i64; 4]; 4],
    want: [u32; 4],
    ruled: Ruled<[u32; 2]>,
}

fn quad_pins() -> Vec<QuadPin> {
    vec![
        QuadPin {
            what: "a 16-bit format's terms truncated",
            lanes: [EXACT_FORMATS[2]; 4],
            k: [110, 24],
            codes: [
                [55762, 22205, 14530, 30349],
                [12683, 52504, 11120, 38166],
                [62640, 32318, 36064, 28365],
                [47199, 3782, 64660, 28308],
            ],
            want: [0x3f1a_2a9a, 0x3ee3_98e4, 0x3f3b_9fbc, 0x3eee_02ee],
            ruled: Ruled::Terms(Terms::Exact),
        },
        QuadPin {
            what: "a signed 16-bit format's negative terms truncated downward",
            lanes: [EXACT_FORMATS[3]; 4],
            k: [135, 87],
            codes: [
                [-4143, 29752, 2902, -6881],
                [-25761, -24584, 25085, -31919],
                [29263, -24141, 29816, 24618],
                [3275, 18827, 3196, 10039],
            ],
            want: [0x3e81_a903, 0xbf0e_e11e, 0x3e9b_2136, 0x3e9a_8535],
            ruled: Ruled::Terms(Terms::TowardZero),
        },
        QuadPin {
            what: "an axis whose two taps are one texel weighs (256, 0)",
            lanes: [EXACT_FORMATS[2]; 4],
            k: [0, 128],
            codes: [
                [45370, 45370, 48395, 48395],
                [62103, 62103, 1003, 1003],
                [22797, 22797, 60938, 60938],
                [26059, 26059, 37169, 37169],
            ],
            want: [0x3f37_23b7, 0x3ef6_82f7, 0x3f23_8ca4, 0x3ef6_fcf7],
            ruled: Ruled::Inputs([107, 128]),
        },
        QuadPin {
            what: "an sRGB color lane's steps truncated",
            lanes: srgb_lanes(),
            k: [130, 75],
            codes: [
                [363, 1916, 811, 2035],
                [72, 1480, 1646, 3568],
                [91, 1342, 1689, 1381],
                [120, 221, 155, 93],
            ],
            want: [0x3e9a_49a5, 0x3ea5_aa5b, 0x3e70_8f09, 0x3f1d_edee],
            ruled: Ruled::Terms(Terms::Exact),
        },
    ]
}

/// The exact linear 2D routine answers Metal's own lanes, each pinning one rule ([`quad_pins`]),
/// which the reference answers too and the ruled-out reading does not.
#[test]
fn the_exact_quad_answers_metals_pinned_lanes() {
    let pins = quad_pins();
    for p in &pins {
        let result = |k: [u32; 2], terms: Terms| -> [u64; 4] {
            std::array::from_fn(|l| {
                let f = p.lanes[l];
                f.result(quad_step(f, &p.codes[l], k, terms), false)
            })
        };
        let want = p.want.map(u64::from);
        assert_eq!(result(p.k, Terms::Floor), want, "reference: {}", p.what);
        let other = match p.ruled {
            Ruled::Inputs(k) => result(k, Terms::Floor),
            Ruled::Terms(terms) => result(p.k, terms),
        };
        assert_ne!(other, want, "{:?} answers too: {}", p.ruled, p.what);
    }
    let set: Vec<(Case, &QuadPin)> = pins
        .iter()
        .map(|p| (quad_case(p.lanes, &p.codes, p.k), p))
        .collect();
    let only: Vec<Case> = set.iter().map(|(c, _)| c.clone()).collect();
    let want = |c: &Case| {
        let (_, p) = set.iter().find(|(k, _)| k == c)?;
        Some(p.want.map(u64::from).to_vec())
    };
    check(
        "pinned quad lanes",
        &quad_ins(),
        v4(),
        &quad_body(false),
        &only,
        &want,
        false,
    );
}

/// The corner routine of a linear cube sample is Metal's made-up texel for every exact format and
/// an 8-bit sRGB format's raw steps: per lane `⌊(6 X + 5 Y + 5 Z + 8) / 16⌋` of the three texels'
/// codes (a signed format's least code read as the next one, an sRGB color lane's steps), as its
/// value on the grid, on random codes and the ranges' ends. A float format's corner is its `x`
/// face's texel, whatever the others.
#[test]
fn the_cube_corner_is_metals_made_up_texel() {
    let ins = vec![v4(), v4(), v4(), u()];
    let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
        let c = routines::define_cube(fb.module())?;
        Ok(fb.call(c.corner, x)?.expect("corner"))
    };
    let mut rng = Rng(0x00c0_be5e_ed02_2001);
    let mut formats: Vec<[Exact; 4]> = EXACT_FORMATS.iter().map(|f| [*f; 4]).collect();
    formats.push(srgb_lanes());
    for lanes in formats {
        let mut cases = Vec::new();
        for _ in 0..600 {
            let cs: [[i64; 3]; 4] = std::array::from_fn(|l| {
                let c = codes(lanes[l], &mut rng, 3);
                [c[0], c[1], c[2]]
            });
            let mut case: Case = (0..3)
                .map(|t| (0..4).map(|l| lanes[l].value(cs[l][t])).collect())
                .collect();
            case.push(vec![u64::from(lanes[0].flags)]);
            cases.push((case, cs));
        }
        let reference = |c: &Case| {
            let (_, cs) = cases.iter().find(|(k, _)| k == c)?;
            let lanes = (0..4).map(|l| {
                let f = lanes[l];
                let [x, y, z] = cs[l].map(|c| c.max(-f.scale));
                let made = (6 * x + 5 * y + 5 * z + 8).div_euclid(16);
                f.result(made * (f.d / f.scale), false)
            });
            Some(lanes.collect())
        };
        let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
        check(
            &format!("corner {:?}", lanes[0]),
            &ins,
            v4(),
            &body,
            &only,
            &reference,
            false,
        );
    }
    // A float format's corner is the `x` face's texel, bit for bit (a NaN, an infinity, and a
    // `-0` included).
    let mut floats = Vec::new();
    for flags in [T::LINEAR, T::HALF] {
        for n in 0..64u64 {
            let mut texel = |k: u64| -> Vec<u64> {
                (0..4)
                    .map(|l| match (n + k + l) % 9 {
                        0 => u64::from(f32::NAN.to_bits()),
                        1 => u64::from(f32::NEG_INFINITY.to_bits()),
                        2 => u64::from((-0.0f32).to_bits()),
                        _ => rng.next() & 0xffff_ffff,
                    })
                    .collect()
            };
            let mut case: Case = vec![texel(0), texel(3), texel(5)];
            case.push(vec![u64::from(flags)]);
            floats.push(case);
        }
    }
    check(
        "corner float",
        &ins,
        v4(),
        &body,
        &floats,
        &|c: &Case| Some(c[0].clone()),
        false,
    );
}

/// A float texel: finite in `[-4, 4]`, or, one tap in `special` lanes, a NaN or an infinity.
fn float_texel(rng: &mut Rng, half: bool) -> f32 {
    let x = (rng.next() % 8001) as f32 / 1000.0 - 4.0;
    if half {
        f16_to_f64(f64_to_f16(f64::from(x))) as f32
    } else {
        x
    }
}

const SPECIALS: [f32; 3] = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];

/// Metal's answer for the specials among `taps` (each `(value, counted)`): NaN for a NaN or both
/// infinities, else the infinity present, else `None`.
fn special_result(taps: impl Iterator<Item = (f32, bool)>) -> Option<f32> {
    let (mut nan, mut pos, mut neg) = (false, false, false);
    for (x, counted) in taps {
        if !counted {
            continue;
        }
        nan |= x.is_nan();
        pos |= x == f32::INFINITY;
        neg |= x == f32::NEG_INFINITY;
    }
    if nan || (pos && neg) {
        Some(f32::NAN)
    } else if pos {
        Some(f32::INFINITY)
    } else if neg {
        Some(f32::NEG_INFINITY)
    } else {
        None
    }
}

/// The `half` result of the emulated sum `r` for format flags `flags`.
fn as_half(flags: u32, r: f32) -> f32 {
    let mode = if flags & T::LINEAR != 0 {
        HalfRounding::TowardZeroSaturating
    } else {
        HalfRounding::NearestEven
    };
    if flags & T::HALF != 0 {
        r
    } else {
        f32::from_bits(half_of(r, mode))
    }
}

/// A 2D sample's result through [`routines::Plane::sums`] as a sample composes it: the footprint
/// `x[..16]` (row-major) read row by row, the fractions `x[16]` and `x[17]`, the format's flags
/// `x[18]`, and the format's result selected on the flags (at run time here, on the
/// specialization at a sample). No axis's four taps read one texel.
fn plane_sample(fb: &mut FunctionBuilder<'_>, x: &[Value], half: bool) -> BResult<Value> {
    let apart = ops::splat(fb, boolean(), 0)?;
    plane_sample_whole(fb, x, half, (apart, apart))
}

/// [`plane_sample`] where `whole` says, per axis (`bool`), whether its four taps read one texel.
fn plane_sample_whole(
    fb: &mut FunctionBuilder<'_>,
    x: &[Value],
    half: bool,
    whole: (Value, Value),
) -> BResult<Value> {
    let p = routines::define_plane(fb.module())?;
    let h = ops::splat(fb, boolean(), u64::from(half))?;
    let mut fetch = |fb: &mut FunctionBuilder<'_>, j: RowIndex| -> BResult<[Value; 4]> {
        match j {
            RowIndex::At(j) => {
                let at = 4 * j as usize;
                Ok([x[at], x[at + 1], x[at + 2], x[at + 3]])
            }
            // The row the counter names, selected among the four.
            RowIndex::Counter(k) => {
                let mut row = [x[12], x[13], x[14], x[15]];
                for j in (0..3).rev() {
                    let c = ops::splat(fb, u(), j as u64)?;
                    let is = fb.binary(BinOp::IEqual, k, c)?;
                    for (i, r) in row.iter_mut().enumerate() {
                        *r = fb.select(is, x[4 * j + i], *r)?;
                    }
                }
                Ok(row)
            }
        }
    };
    let sums = p.sums(
        fb,
        &mut fetch,
        (x[16], x[17]),
        (x[16], x[17]),
        whole,
        (x[18], h),
        false,
    )?;
    let float = sums.float.expect("an emulated sum");
    let mut bit = |b: u32| -> BResult<Value> {
        let m = ops::splat(fb, u(), u64::from(b))?;
        let z = ops::splat(fb, u(), 0)?;
        let and = fb.binary(BinOp::BitwiseAnd, x[18], m)?;
        fb.binary(BinOp::INotEqual, and, z)
    };
    let (is_half, is_linear, is_exact) = (bit(T::HALF)?, bit(T::LINEAR)?, bit(T::EXACT)?);
    let f = float.select(fb, is_half, is_linear)?;
    fb.select(is_exact, sums.exact, f)
}

/// The emulated 2D routine: a 32-bit float format's bilinear sum through Metal's linear
/// accumulator ([`metal_linear`]), a half format's Catmull-Rom sum through its two-pass
/// accumulator ([`metal_catmull_rom`]), any other format's Catmull-Rom sum in `f32` under
/// Metal's weights; the clamps, Metal's non-finite rule, and the `half` result's conversion.
#[test]
fn the_float_plane_takes_each_formats_sum_and_metals_specials() {
    let mut ins = vec![v4(); 16];
    ins.extend([u(), u(), u()]);
    let mut rng = Rng(0x00b1_c0b1_c5ee_d003);
    let formats = [
        T::LINEAR,
        T::HALF,
        T::UNORM,
        T::SNORM,
        0, // a slot bound to nothing: the sum unchanged
    ];
    for flags in formats {
        let half_format = flags & T::HALF != 0;
        let mut cases = Vec::new();
        for n in 0..400 {
            let (kx, ky) = (fraction(&mut rng), fraction(&mut rng));
            let mut taps: Vec<[f32; 4]> = (0..16)
                .map(|_| std::array::from_fn(|_| float_texel(&mut rng, half_format)))
                .collect();
            if flags & (T::UNORM | T::SNORM) != 0 {
                for t in &mut taps {
                    for x in t.iter_mut() {
                        *x = if flags & T::UNORM != 0 {
                            x.abs() / 4.0
                        } else {
                            *x / 4.0
                        };
                    }
                }
            } else if n % 2 == 1 {
                // One or two specials per lane, anywhere in the footprint.
                for l in [0usize, 1, 2, 3] {
                    for _ in 0..1 + rng.next() % 2 {
                        let at = (rng.next() % 16) as usize;
                        taps[at][l] = SPECIALS[(rng.next() % 3) as usize];
                    }
                }
            }
            let mut c: Case = taps
                .iter()
                .map(|t| t.iter().map(|x| u64::from(x.to_bits())).collect())
                .collect();
            c.extend([vec![kx], vec![ky], vec![u64::from(flags)]]);
            cases.push((c, taps));
        }
        for half in [false, true] {
            let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| plane_sample(fb, x, half);
            let reference = |c: &Case| {
                let (_, taps) = cases.iter().find(|(k, _)| k == c)?;
                let (kx, ky) = (arg(c, 16) as u32, arg(c, 17) as u32);
                let fw = |k: u32| weights(k).map(|w| w as f32 / 65536.0);
                let (wx, wy) = (fw(kx), fw(ky));
                let lanes = (0..4).map(|l| {
                    let at = |j: usize, i: usize| taps[4 * j + i][l];
                    if flags & T::LINEAR != 0 {
                        let w = quad_weights(kx, ky);
                        let t = [
                            (at(1, 1), w[0]),
                            (at(1, 2), w[1]),
                            (at(2, 1), w[2]),
                            (at(2, 2), w[3]),
                        ];
                        return u64::from(metal_linear(&t, flags, half).to_bits());
                    }
                    if half_format {
                        let t: [f32; 16] = std::array::from_fn(|n| at(n / 4, n % 4));
                        let specials = t.iter().map(|x| (*x, true));
                        let r = special_result(specials)
                            .unwrap_or_else(|| metal_catmull_rom(&t, kx, ky));
                        return u64::from(r.to_bits());
                    }
                    let r = {
                        let mut s = 0f32;
                        for (j, wyj) in wy.iter().enumerate() {
                            let mut row = 0f32;
                            for (i, wxi) in wx.iter().enumerate() {
                                let t = at(j, i) * wxi;
                                row = if i == 0 { t } else { row + t };
                            }
                            let t = row * wyj;
                            s = if j == 0 { t } else { s + t };
                        }
                        s
                    };
                    let r = if flags & T::UNORM != 0 {
                        r.clamp(0.0, 1.0)
                    } else if flags & T::SNORM != 0 {
                        r.clamp(-1.0, 1.0)
                    } else {
                        r
                    };
                    let r = if half { as_half(flags, r) } else { r };
                    u64::from(r.to_bits())
                });
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("float plane flags={flags:#x} half={half}"),
                &ins,
                v4(),
                &body,
                &only,
                &reference,
                false,
            );
        }
    }
}

/// A finite tap that strains an exact sum: any `f32` bit pattern (or `half` one), a denormal,
/// or, for the second tap, one cancelling the first, one ulp from it, or one a chosen number of
/// binades below it.
fn straining_texel(rng: &mut Rng, half: bool, first: Option<f32>) -> f32 {
    let any = |rng: &mut Rng| loop {
        let x = if half {
            f16_to_f64((rng.next() & 0xFFFF) as u16) as f32
        } else {
            f32::from_bits(rng.next() as u32)
        };
        if x.is_finite() {
            return x;
        }
    };
    let pick = rng.next() % 8;
    match (first, pick) {
        (Some(a), 0) => -a,
        (Some(a), 1) => {
            let up = if half {
                f16_to_f64(f64_to_f16(f64::from(a)) ^ 1) as f32
            } else {
                f32::from_bits(a.to_bits() ^ 1)
            };
            if rng.next() & 1 == 0 {
                -up
            } else {
                up
            }
        }
        (Some(a), 2) => {
            let d = (rng.next() % 72) as i32;
            let x = a * 2f32.powi(-d) * if rng.next() & 1 == 0 { -1.0 } else { 1.0 };
            let x = if half {
                f16_to_f64(f64_to_f16(f64::from(x))) as f32
            } else {
                x
            };
            if x.is_finite() {
                x
            } else {
                any(rng)
            }
        }
        (_, 3) if !half => f32::from_bits(rng.next() as u32 & 0x807F_FFFF),
        (_, 3) => f16_to_f64((rng.next() & 0x83FF) as u16) as f32,
        _ => any(rng),
    }
}

/// The 1D routine's 32-bit float and half sums are Metal's linear accumulator, against
/// [`metal_linear`], on taps that strain it: every exponent, denormals, cancellation, neighbours
/// one ulp apart, exponent gaps up to 71 binades, and the ties of `k = 128`. Each lane's result
/// is checked as a `float` and as a `half` result.
#[test]
fn the_float_line_is_metals_accumulator() {
    let ins = [v4(), v4(), u(), u()];
    let mut rng = Rng(0x00b1_c0b1_c5ee_d005);
    for flags in [T::LINEAR, T::HALF] {
        let half_format = flags & T::HALF != 0;
        let mut cases = Vec::new();
        for n in 0..2000 {
            let k = match n % 4 {
                0 => 128,
                1 => [1, 64, 192, 255][(rng.next() % 4) as usize],
                _ => rng.next() % 256,
            };
            let first: [f32; 4] =
                std::array::from_fn(|_| straining_texel(&mut rng, half_format, None));
            let second: [f32; 4] =
                std::array::from_fn(|l| straining_texel(&mut rng, half_format, Some(first[l])));
            let taps = [first, second];
            let mut c: Case = taps
                .iter()
                .map(|t| t.iter().map(|x| u64::from(x.to_bits())).collect())
                .collect();
            c.extend([vec![k], vec![u64::from(flags)]]);
            cases.push((c, taps));
        }
        for half in [false, true] {
            let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                let l = routines::define_line(fb.module())?;
                let h = ops::splat(fb, boolean(), u64::from(half))?;
                Ok(fb
                    .call(l.float, &[x[0], x[1], x[2], x[3], h])?
                    .expect("float"))
            };
            let reference = |c: &Case| {
                let (_, taps) = cases.iter().find(|(key, _)| key == c)?;
                let w = span_weights(arg(c, 2) as u32);
                let lanes = (0..4).map(|l| {
                    let t = [(taps[0][l], w[0]), (taps[1][l], w[1])];
                    u64::from(metal_linear(&t, flags, half).to_bits())
                });
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("accumulated float line flags={flags:#x} half={half}"),
                &ins,
                v4(),
                &body,
                &only,
                &reference,
                false,
            );
        }
    }
}

/// The emulated 1D routine: a 32-bit float or half format's sum through Metal's linear
/// accumulator ([`metal_sum`]), every other format's bilinear in `f32`, a zero-weight second
/// tap skipped, the half format's rounding, and the non-finite rule on the taps of nonzero
/// weight.
#[test]
fn the_float_line_skips_a_zero_weight_tap() {
    let ins = [v4(), v4(), u(), u()];
    let mut rng = Rng(0x00b1_c0b1_c5ee_d004);
    for flags in [T::LINEAR, T::HALF, T::UNORM] {
        let half_format = flags & T::HALF != 0;
        let mut cases = Vec::new();
        for n in 0..600 {
            let k = fraction(&mut rng);
            let mut taps: [[f32; 4]; 2] = std::array::from_fn(|_| {
                std::array::from_fn(|_| float_texel(&mut rng, half_format))
            });
            if flags & T::UNORM != 0 {
                for x in taps.iter_mut().flatten() {
                    *x = x.abs() / 4.0;
                }
            } else if n % 2 == 1 {
                for l in [0usize, 1, 2, 3] {
                    let at = (rng.next() % 2) as usize;
                    taps[at][l] = SPECIALS[(rng.next() % 3) as usize];
                }
            }
            let mut c: Case = taps
                .iter()
                .map(|t| t.iter().map(|x| u64::from(x.to_bits())).collect())
                .collect();
            c.extend([vec![k], vec![u64::from(flags)]]);
            cases.push((c, taps));
        }
        for half in [false, true] {
            let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                let l = routines::define_line(fb.module())?;
                let h = ops::splat(fb, boolean(), u64::from(half))?;
                Ok(fb
                    .call(l.float, &[x[0], x[1], x[2], x[3], h])?
                    .expect("float"))
            };
            let reference = |c: &Case| {
                let (_, taps) = cases.iter().find(|(k, _)| k == c)?;
                let k = arg(c, 2) as u32;
                let lanes = (0..4).map(|l| {
                    let (a, b) = (taps[0][l], taps[1][l]);
                    let float = flags & (T::LINEAR | T::HALF) != 0;
                    let r = if float {
                        let w = span_weights(k);
                        metal_sum(&[(a, w[0]), (b, w[1])], half_format)
                    } else if k == 0 {
                        a
                    } else {
                        a * ((256 - k) as f32 / 256.0) + b * (k as f32 / 256.0)
                    };
                    let r =
                        special_result([(a, float), (b, float && k != 0)].into_iter()).unwrap_or(r);
                    let r = if half { as_half(flags, r) } else { r };
                    u64::from(r.to_bits())
                });
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("float line flags={flags:#x} half={half}"),
                &ins,
                v4(),
                &body,
                &only,
                &reference,
                false,
            );
        }
    }
}

/// `x` (exact in `f64`) rounded to `f32` with ties away from zero; a zero or denormal result
/// `+0`.
fn f32_away(x: f64) -> f32 {
    let r = x as f32;
    let d = x - f64::from(r);
    let r = if d == 0.0 {
        r
    } else {
        // The other neighbour of `x`, on `d`'s side of `r`.
        let toward = |r: f32, up: bool| {
            let b = r.to_bits();
            let away = (r >= 0.0) == up;
            f32::from_bits(if r == 0.0 {
                1 | if up { 0 } else { 0x8000_0000 }
            } else if away {
                b + 1
            } else {
                b - 1
            })
        };
        let other = toward(r, d > 0.0);
        if (f64::from(other) - x).abs() == d.abs() && other.abs() > r.abs() {
            other
        } else {
            r
        }
    };
    if r.abs() < f32::MIN_POSITIVE {
        0.0
    } else {
        r
    }
}

/// Metal's linear accumulator over the finite taps `taps` (each with its integer weight in units of
/// `2^-16`) of a 32-bit float format, or a half one when `half_format`, exactly through `f64`,
/// independently of the routine's integer alignment: each present term `v · w` (a nonzero tap of
/// nonzero weight; exact, 24 and 17 bits) truncated toward zero onto the grid `2^(E - b)`, `E` the
/// largest present `e(v) + e(w)`, `b` 38 or 17; the truncated terms summed (exact: at most eight,
/// integers below `2^(b + 5)` grid units); the sum rounded with ties away from zero, to `f32` (a
/// denormal `+0`) or `half` (exact in `f32` first: at most 21 bits, exponent at least `-57`); a
/// zero `+0`.
fn metal_sum(taps: &[(f32, u32)], half_format: bool) -> f32 {
    let bits = if half_format { 17 } else { 38 };
    let exp = |x: f64| ((x.abs().to_bits() >> 52) as i32) - 1023;
    let present: Vec<(f64, f64)> = taps
        .iter()
        .filter(|(v, w)| *v != 0.0 && *w != 0)
        .map(|(v, w)| (f64::from(*v), f64::from(*w) / 65536.0))
        .collect();
    let Some(top) = present.iter().map(|(v, w)| exp(*v) + exp(*w)).max() else {
        return 0.0;
    };
    let unit = 2f64.powi(top - bits);
    let s: f64 = present
        .iter()
        .map(|(v, w)| ((v * w) / unit).trunc() * unit)
        .sum();
    if s == 0.0 {
        return 0.0;
    }
    if half_format {
        let h = f32::from_bits(half_of(s as f32, HalfRounding::NearestAway));
        if h == 0.0 {
            0.0
        } else {
            h
        }
    } else {
        f32_away(s)
    }
}

/// Metal's linear result for format flags `flags` (`LINEAR` or `HALF`) of the taps `taps` (each
/// with its integer weight): Metal's non-finite rule on the taps of nonzero weight, else
/// [`metal_sum`], then the `half` result's conversion when `half`.
fn metal_linear(taps: &[(f32, u32)], flags: u32, half: bool) -> f32 {
    let live: Vec<bool> = taps.iter().map(|(_, w)| *w != 0).collect();
    metal_linear_live(taps, &live, flags, half)
}

/// [`metal_linear`] with Metal's non-finite rule on the taps `live` names (a 3D footprint's taps
/// of nonzero product weight, whose rounded weight may be zero).
fn metal_linear_live(taps: &[(f32, u32)], live: &[bool], flags: u32, half: bool) -> f32 {
    let half_format = flags & T::HALF != 0;
    let special = special_result(taps.iter().zip(live).map(|((v, _), l)| (*v, *l)));
    let r = special.unwrap_or_else(|| metal_sum(taps, half_format));
    if half {
        as_half(flags, r)
    } else {
        r
    }
}

/// `x` (exact in `f64`) rounded to `half` with ties away from zero, as an `f32`: a zero `+0`,
/// a magnitude past `half`'s largest finite value an infinity.
fn half_away(x: f64) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let e = (((x.abs().to_bits() >> 52) as i32) - 1023).max(-14);
    let quantum = 2f64.powi(e - 10);
    let n = x.abs() / quantum;
    let r = (n.floor() + f64::from(u8::from(n - n.floor() >= 0.5))) * quantum;
    let r = if r > 65504.0 { f64::INFINITY } else { r };
    if r == 0.0 {
        0.0
    } else {
        (r as f32).copysign(x as f32)
    }
}

/// Metal's Catmull-Rom sum of a half format's finite footprint `taps` (row `j`, column `i` at
/// `4j + i`) at fractions `(kx, ky)`, exactly through `f64`, independently of the routine's integer
/// alignment:
///
/// - row `j`'s present terms `v · wx_i` (exact: 24 and 17 bits) truncated toward zero onto
///   `2^(E_j - 17)`, `E_j` the row's largest `e(v) + e(wx_i)` (a unit weight's `e` read as `-1`),
///   and summed (exact: below `2^22` grid units);
/// - the rows' terms `R_j · wy_j` (exact: 22 and 17 bits) of the rows with a present term and a
///   nonzero `wy_j` truncated toward zero onto `2^(E - 20)`, `E` the largest `E_j + e(wy_j)`,
///   summed (exact: below `2^29` grid units), and rounded to `half` with ties away from zero
///   ([`half_away`]).
fn metal_catmull_rom(taps: &[f32; 16], kx: u32, ky: u32) -> f32 {
    let exp = |x: f64| ((x.abs().to_bits() >> 52) as i32) - 1023;
    let unit_capped = |w: i32| 31 - w.unsigned_abs().min(0xFFFF).leading_zeros() as i32 - 16;
    let (wx, wy) = (weights(kx), weights(ky));
    let truncated = |x: f64, unit: f64| (x / unit).trunc() * unit;
    let mut rows = Vec::new();
    for (j, w) in wy.iter().enumerate() {
        let present: Vec<(f64, i32)> = (0..4)
            .filter(|&i| taps[4 * j + i] != 0.0 && wx[i] != 0)
            .map(|i| (f64::from(taps[4 * j + i]), wx[i]))
            .collect();
        let Some(e) = present.iter().map(|(v, w)| exp(*v) + unit_capped(*w)).max() else {
            continue;
        };
        if *w == 0 {
            continue;
        }
        let unit = 2f64.powi(e - 17);
        let r: f64 = present
            .iter()
            .map(|(v, w)| truncated(v * f64::from(*w) / 65536.0, unit))
            .sum();
        rows.push((r, e, f64::from(*w) / 65536.0));
    }
    let Some(top) = rows.iter().map(|(_, e, w)| e + exp(*w)).max() else {
        return 0.0;
    };
    let unit = 2f64.powi(top - 20);
    half_away(rows.iter().map(|(r, _, w)| truncated(r * w, unit)).sum())
}

/// The weights of a 1D sample at fraction `k`, and of a 2D one at `(kx, ky)` (taps `(i, j)`,
/// `(i + 1, j)`, `(i, j + 1)`, `(i + 1, j + 1)`), in units of `2^-16`.
fn span_weights(k: u32) -> [u32; 2] {
    [(256 - k) * 256, k * 256]
}

fn quad_weights(kx: u32, ky: u32) -> [u32; 4] {
    [
        (256 - kx) * (256 - ky),
        kx * (256 - ky),
        (256 - kx) * ky,
        kx * ky,
    ]
}

/// Random footprints for the accumulated routines: taps that strain the accumulator
/// ([`straining_texel`]: every exponent, denormals, cancellation, neighbours one ulp apart, wide
/// exponent gaps), with specials in one case of four and zeros of both signs and tiny values
/// in another, at fractions on and off the edges.
fn accumulated_cases(
    rng: &mut Rng,
    half_format: bool,
    taps: usize,
    n: usize,
) -> Vec<Vec<[f32; 4]>> {
    (0..n)
        .map(|c| {
            let first: [f32; 4] = std::array::from_fn(|_| straining_texel(rng, half_format, None));
            let mut t: Vec<[f32; 4]> = vec![first];
            for _ in 1..taps {
                t.push(std::array::from_fn(|l| {
                    straining_texel(rng, half_format, Some(first[l]))
                }));
            }
            match c % 4 {
                1 => {
                    for l in [0usize, 1, 2, 3] {
                        for _ in 0..1 + rng.next() % 2 {
                            let at = (rng.next() as usize) % taps;
                            t[at][l] = SPECIALS[(rng.next() % 3) as usize];
                        }
                    }
                }
                2 => {
                    let tiny = if half_format {
                        [0.0, -0.0, 5.960_464_5e-8, -5.960_464_5e-8]
                    } else {
                        [0.0, -0.0, 1e-38, -1e-38]
                    };
                    for x in t.iter_mut().flatten() {
                        *x = tiny[(rng.next() % 4) as usize];
                    }
                }
                _ => {}
            }
            t
        })
        .collect()
}

/// The accumulated linear 2D routine is Metal's accumulator ([`metal_linear`]) on random straining
/// footprints, with Metal's non-finite rule on the taps of nonzero weight and the `half` result's
/// conversion.
#[test]
fn the_float_quad_is_metals_accumulator() {
    let mut ins = vec![v4(); 4];
    ins.extend([u(), u(), u()]);
    let mut rng = Rng(0x00b1_c0b1_c5ee_d006);
    for flags in [T::LINEAR, T::HALF] {
        let half_format = flags & T::HALF != 0;
        let mut cases = Vec::new();
        for taps in accumulated_cases(&mut rng, half_format, 4, 1600) {
            let (kx, ky) = (fraction(&mut rng), fraction(&mut rng));
            let mut c: Case = taps
                .iter()
                .map(|t| t.iter().map(|x| u64::from(x.to_bits())).collect())
                .collect();
            c.extend([vec![kx], vec![ky], vec![u64::from(flags)]]);
            cases.push((c, taps));
        }
        for half in [false, true] {
            let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                let q = routines::define_quad(fb.module())?;
                let h = ops::splat(fb, boolean(), u64::from(half))?;
                let mut args = x[..7].to_vec();
                args.push(h);
                Ok(fb.call(q.float, &args)?.expect("float"))
            };
            let reference = |c: &Case| {
                let (_, taps) = cases.iter().find(|(k, _)| k == c)?;
                let w = quad_weights(arg(c, 4) as u32, arg(c, 5) as u32);
                let lanes = (0..4).map(|l| {
                    let t: Vec<(f32, u32)> = (0..4).map(|n| (taps[n][l], w[n])).collect();
                    u64::from(metal_linear(&t, flags, half).to_bits())
                });
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("float quad flags={flags:#x} half={half}"),
                &ins,
                v4(),
                &body,
                &only,
                &reference,
                false,
            );
        }
    }
}

/// The accumulated linear 1D routine is Metal's accumulator ([`metal_linear`]) on random straining
/// taps, with Metal's non-finite rule on the taps of nonzero weight and the `half` result's
/// conversion.
#[test]
fn the_float_span_is_metals_accumulator() {
    let ins = [v4(), v4(), u(), u()];
    let mut rng = Rng(0x00b1_c0b1_c5ee_d007);
    for flags in [T::LINEAR, T::HALF] {
        let half_format = flags & T::HALF != 0;
        let mut cases = Vec::new();
        for (n, taps) in accumulated_cases(&mut rng, half_format, 2, 2000)
            .into_iter()
            .enumerate()
        {
            let k = match n % 4 {
                0 => [64, 128, 192][(rng.next() % 3) as usize],
                _ => fraction(&mut rng),
            };
            let mut c: Case = taps
                .iter()
                .map(|t| t.iter().map(|x| u64::from(x.to_bits())).collect())
                .collect();
            c.extend([vec![k], vec![u64::from(flags)]]);
            cases.push((c, taps));
        }
        for half in [false, true] {
            let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
                let s = routines::define_span(fb.module())?;
                let h = ops::splat(fb, boolean(), u64::from(half))?;
                Ok(fb
                    .call(s.float, &[x[0], x[1], x[2], x[3], h])?
                    .expect("float"))
            };
            let reference = |c: &Case| {
                let (_, taps) = cases.iter().find(|(k, _)| k == c)?;
                let w = span_weights(arg(c, 2) as u32);
                let lanes = (0..4).map(|l| {
                    let t = [(taps[0][l], w[0]), (taps[1][l], w[1])];
                    u64::from(metal_linear(&t, flags, half).to_bits())
                });
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("float span flags={flags:#x} half={half}"),
                &ins,
                v4(),
                &body,
                &only,
                &reference,
                false,
            );
        }
    }
}

/// Metal's weights of a linear 3D footprint's eight taps (the `s` axis's fastest, then the `t`
/// axis's, then the `r` axis's), in units of `2^-16`: `wx · wy · wz / 256` rounded to nearest, a
/// tie up on the `t` axis's first tap in Metal's frame (its second when `flip`) and down on the
/// other.
fn block_weights(kx: u32, ky: u32, kz: u32, flip: bool) -> [u32; 8] {
    let (wx, wy, wz) = ([256 - kx, kx], [256 - ky, ky], [256 - kz, kz]);
    std::array::from_fn(|i| {
        let p = wx[i & 1] * wy[(i >> 1) & 1] * wz[i >> 2];
        let (q, r) = (p >> 8, p & 255);
        let first = ((i >> 1) & 1 == 1) == flip;
        q + u32::from(r > 128 || (r == 128 && first))
    })
}

/// How a 2D or 3D exact sum reduces its terms `step · w`: Metal's, each truncated onto `2^-s` of
/// the grid toward negative infinity; or a reading Metal's lanes rule out, the exact sum or each
/// term truncated toward zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Terms {
    Floor,
    Exact,
    TowardZero,
}

/// The grid step of a 2D or 3D exact sum of format lane `f`'s codes `codes` under weights `w`,
/// its terms reduced by `terms`: `T = Σ ⌊step(c) · w / 2^s⌋` (`s` 8 for a 16-bit format, else 4),
/// then `⌊(T + 2^(15 - s)) / 2^(16 - s)⌋`; the exact sum's `⌊(S + 2^15) / 2^16⌋`.
fn grid_step(f: Exact, codes: &[i64], w: &[u32], terms: Terms) -> i64 {
    let s = if f.flags & T::WIDE16 != 0 { 8 } else { 4 };
    let products = codes.iter().zip(w).map(|(c, w)| f.step(*c) * i64::from(*w));
    let reduced = |t: i64| (t + (1 << (15 - s))) >> (16 - s);
    match terms {
        Terms::Exact => (products.sum::<i64>() + (1 << 15)) >> 16,
        Terms::Floor => reduced(products.map(|p| p >> s).sum()),
        Terms::TowardZero => reduced(products.map(|p| p / (1 << s)).sum()),
    }
}

/// An 8-bit sRGB format's color lane as a raw linear sample reads it: the table's step `k` as `k /
/// 4095`, on the grid 4095.
const SRGB_COLOR: Exact = Exact {
    flags: T::UNORM | T::EXACT | T::SRGB,
    lo: 0,
    hi: 4095,
    scale: 4095,
    d: 4095,
};

/// An 8-bit sRGB format's lanes as a raw linear sample reads them: three color lanes
/// ([`SRGB_COLOR`]) and the alpha lane's code on `16 · 255`.
fn srgb_lanes() -> [Exact; 4] {
    let alpha = Exact {
        flags: SRGB_COLOR.flags,
        ..EXACT_FORMATS[0]
    };
    [SRGB_COLOR, SRGB_COLOR, SRGB_COLOR, alpha]
}

/// The routines' case of a 3D footprint: the eight taps' values by lane (`codes[lane][tap]`),
/// the fractions, and the format's flags.
fn block_case(lanes: [Exact; 4], codes: &[[i64; 8]; 4], k: [u32; 3]) -> Case {
    let mut c: Case = (0..8)
        .map(|i| (0..4).map(|l| lanes[l].value(codes[l][i])).collect())
        .collect();
    c.extend(k.map(|k| vec![u64::from(k)]));
    c.push(vec![u64::from(lanes[0].flags)]);
    c
}

/// A body calling a 3D routine (`exact` or not) on a [`block_case`]'s inputs, the `t` axis's
/// frame `flip` and the `half` result constants.
fn block_body(
    exact: bool,
    flip: bool,
    half: bool,
) -> impl Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value> {
    move |fb, x| {
        let b = routines::define_block(fb.module())?;
        let f = ops::splat(fb, boolean(), u64::from(flip))?;
        let h = ops::splat(fb, boolean(), u64::from(half))?;
        let mut args = x[..11].to_vec();
        args.extend([f, x[11], h]);
        let routine = if exact { b.exact } else { b.float };
        Ok(fb.call(routine, &args)?.expect("block"))
    }
}

/// The inputs of a [`block_case`].
fn block_ins() -> Vec<Form> {
    let mut ins = vec![v4(); 8];
    ins.extend([u(), u(), u(), u()]);
    ins
}

/// The exact linear 3D routine is Metal's sum of truncated terms ([`grid_step`]) under Metal's
/// 16-bit weights ([`block_weights`]), on random codes of every exact format and an 8-bit sRGB
/// format's raw steps, at fractions on and off the edges (`128` with odd partners makes ties), with
/// either frame of the `t` axis and either result.
#[test]
fn the_exact_block_truncates_each_term() {
    let ins = block_ins();
    let mut rng = Rng(0x00b1_c0b1_c5ee_d00a);
    let mut formats: Vec<[Exact; 4]> = EXACT_FORMATS.iter().map(|f| [*f; 4]).collect();
    formats.push(srgb_lanes());
    for lanes in formats {
        let mut cases = Vec::new();
        for _ in 0..600 {
            let k = [0; 3].map(|_| fraction(&mut rng) as u32);
            let cs: [[i64; 8]; 4] = std::array::from_fn(|l| {
                let c = codes(lanes[l], &mut rng, 8);
                std::array::from_fn(|i| c[i])
            });
            cases.push((block_case(lanes, &cs, k), cs));
        }
        for (half, flip) in [(false, false), (false, true), (true, false), (true, true)] {
            let reference = |c: &Case| {
                let (_, cs) = cases.iter().find(|(k, _)| k == c)?;
                let w = block_weights(arg(c, 8) as u32, arg(c, 9) as u32, arg(c, 10) as u32, flip);
                let lanes = (0..4)
                    .map(|l| lanes[l].result(grid_step(lanes[l], &cs[l], &w, Terms::Floor), half));
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("exact block {:?} flip={flip} half={half}", lanes[0]),
                &ins,
                v4(),
                &block_body(true, flip, half),
                &only,
                &reference,
                false,
            );
        }
    }
}

/// The accumulated linear 3D routine is Metal's accumulator ([`metal_linear`]) under Metal's 16-bit
/// weights ([`block_weights`]) on random straining footprints, with Metal's non-finite rule on the
/// taps of nonzero product weight (every tap but the second of an axis whose fraction is 0) and the
/// `half` result's conversion.
#[test]
fn the_float_block_is_metals_accumulator() {
    let ins = block_ins();
    let mut rng = Rng(0x00b1_c0b1_c5ee_d00b);
    for flags in [T::LINEAR, T::HALF] {
        let half_format = flags & T::HALF != 0;
        let mut cases = Vec::new();
        for taps in accumulated_cases(&mut rng, half_format, 8, 1200) {
            let mut c: Case = taps
                .iter()
                .map(|t| t.iter().map(|x| u64::from(x.to_bits())).collect())
                .collect();
            c.extend([0; 3].map(|_| vec![fraction(&mut rng)]));
            c.push(vec![u64::from(flags)]);
            cases.push((c, taps));
        }
        for (half, flip) in [(false, false), (false, true), (true, false), (true, true)] {
            let reference = |c: &Case| {
                let (_, taps) = cases.iter().find(|(k, _)| k == c)?;
                let k = [arg(c, 8) as u32, arg(c, 9) as u32, arg(c, 10) as u32];
                let w = block_weights(k[0], k[1], k[2], flip);
                let live: Vec<bool> = (0..8)
                    .map(|n| (0..3).all(|axis| n >> axis & 1 == 0 || k[axis] != 0))
                    .collect();
                let lanes = (0..4).map(|l| {
                    let t: Vec<(f32, u32)> = (0..8).map(|n| (taps[n][l], w[n])).collect();
                    u64::from(metal_linear_live(&t, &live, flags, half).to_bits())
                });
                Some(lanes.collect())
            };
            let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
            check(
                &format!("float block flags={flags:#x} flip={flip} half={half}"),
                &ins,
                v4(),
                &block_body(false, flip, half),
                &only,
                &reference,
                false,
            );
        }
    }
}

/// A reading a pinned lane rules out: other inputs (a 2D lane's fractions, a 3D lane's and the
/// frame of its `t` axis), or other terms.
#[derive(Clone, Copy, Debug)]
enum Ruled<K> {
    Inputs(K),
    Terms(Terms),
}

/// A lane of the 3D probe (Apple M4, macOS 27.0.1; the M3's results byte-identical) that pins one
/// rule of Metal's 3D exact sum: the format's lanes, the fractions and the `t` axis's frame as
/// Metal reads them, the taps' codes by lane, Metal's `float` result's bits by lane, and a reading
/// the lane rules out.
struct BlockPin {
    what: &'static str,
    lanes: [Exact; 4],
    k: [u32; 3],
    flip: bool,
    codes: [[i64; 8]; 4],
    want: [u32; 4],
    ruled: Ruled<([u32; 3], bool)>,
}

fn block_pins() -> Vec<BlockPin> {
    let unorm8 = [EXACT_FORMATS[0]; 4];
    vec![
        BlockPin {
            what: "a tie rounds up on the t axis's first tap",
            lanes: unorm8,
            k: [32, 188, 17],
            flip: false,
            codes: [
                [87, 129, 91, 118, 196, 150, 216, 82],
                [106, 253, 208, 121, 217, 218, 86, 125],
                [240, 232, 162, 44, 214, 239, 158, 182],
                [198, 93, 173, 3, 145, 168, 47, 19],
            ],
            want: [0x3eca_0a0a, 0x3f2e_eeef, 0x3f2c_9c9d, 0x3f1b_3b3b],
            ruled: Ruled::Inputs(([32, 188, 17], true)),
        },
        BlockPin {
            what: "a mirrored t axis's reflected half: the second tap is first",
            lanes: unorm8,
            k: [188, 152, 188],
            flip: true,
            codes: [
                [141, 240, 54, 84, 211, 47, 48, 236],
                [10, 232, 54, 90, 1, 56, 174, 71],
                [125, 231, 162, 133, 89, 148, 94, 182],
                [121, 168, 32, 147, 233, 144, 15, 82],
            ],
            want: [0x3f0f_bfc0, 0x3ead_edee, 0x3f19_5959, 0x3ee3_2323],
            ruled: Ruled::Inputs(([188, 152, 188], false)),
        },
        BlockPin {
            what: "a 16-bit format's terms truncated",
            lanes: [EXACT_FORMATS[2]; 4],
            k: [0, 11, 35],
            flip: false,
            codes: [
                [47325, 47325, 9669, 9669, 26735, 26735, 28906, 28906],
                [23044, 23044, 39306, 39306, 25040, 25040, 3152, 3152],
                [13038, 13038, 11721, 11721, 16745, 16745, 58031, 58031],
                [5573, 5573, 57244, 57244, 40777, 40777, 7301, 7301],
            ],
            want: [0x3f28_76a8, 0x3eb9_e0ba, 0x3e56_a8d7, 0x3e3d_28bd],
            ruled: Ruled::Terms(Terms::Exact),
        },
        BlockPin {
            what: "an sRGB format's table steps truncated",
            lanes: srgb_lanes(),
            k: [193, 135, 139],
            flip: false,
            codes: [
                [1286, 578, 869, 346, 1125, 151, 3844, 2611],
                [3272, 3435, 210, 210, 567, 614, 3468, 2991],
                [825, 614, 448, 899, 53, 1646, 1342, 3879],
                [74, 50, 66, 244, 66, 164, 204, 102],
            ],
            want: [0x3e97_8979, 0x3ee4_ae4b, 0x3ec6_6c67, 0x3f05_2525],
            ruled: Ruled::Terms(Terms::Exact),
        },
        BlockPin {
            what: "a signed 16-bit format's negative terms truncated downward",
            lanes: [EXACT_FORMATS[3]; 4],
            k: [255, 48, 90],
            flip: false,
            codes: [
                [16108, 19923, 31736, -16334, 0, 0, 0, 0],
                [-10059, 24725, -29267, -11208, 0, 0, 0, 0],
                [31063, 27920, 730, 4137, 0, 0, 0, 0],
                [4366, -14624, -21393, -30638, 0, 0, 0, 0],
            ],
            want: [0x3e85_350a, 0x3eb5_016a, 0x3eed_c9dc, 0xbeb1_e964],
            ruled: Ruled::Terms(Terms::TowardZero),
        },
        BlockPin {
            what: "an axis whose two taps are one texel weighs (256, 0)",
            lanes: unorm8,
            k: [0, 232, 0],
            flip: false,
            codes: [
                [121, 121, 73, 73, 121, 121, 73, 73],
                [246, 246, 95, 95, 246, 246, 95, 95],
                [79, 79, 248, 248, 79, 79, 248, 248],
                [233, 233, 39, 39, 233, 233, 39, 39],
            ],
            want: [0x3e9b_9b9c, 0x3edb_3b3b, 0x3f69_1919, 0x3e65_a5a6],
            ruled: Ruled::Inputs(([50, 232, 40], false)),
        },
    ]
}

/// The exact linear 3D routine answers Metal's own lanes, each pinning one rule
/// ([`block_pins`]), which the reference answers too and the ruled-out reading does not.
#[test]
fn the_exact_block_answers_metals_pinned_lanes() {
    let pins = block_pins();
    for p in &pins {
        let result = |k: [u32; 3], flip: bool, terms: Terms| -> [u64; 4] {
            let w = block_weights(k[0], k[1], k[2], flip);
            std::array::from_fn(|l| {
                let f = p.lanes[l];
                f.result(grid_step(f, &p.codes[l], &w, terms), false)
            })
        };
        let want = p.want.map(u64::from);
        assert_eq!(
            result(p.k, p.flip, Terms::Floor),
            want,
            "reference: {}",
            p.what
        );
        let other = match p.ruled {
            Ruled::Inputs((k, flip)) => result(k, flip, Terms::Floor),
            Ruled::Terms(terms) => result(p.k, p.flip, terms),
        };
        assert_ne!(other, want, "{:?} answers too: {}", p.ruled, p.what);
    }
    let ins = block_ins();
    for flip in [false, true] {
        let set: Vec<(Case, &BlockPin)> = pins
            .iter()
            .filter(|p| p.flip == flip)
            .map(|p| (block_case(p.lanes, &p.codes, p.k), p))
            .collect();
        let only: Vec<Case> = set.iter().map(|(c, _)| c.clone()).collect();
        let want = |c: &Case| {
            let (_, p) = set.iter().find(|(k, _)| k == c)?;
            Some(p.want.map(u64::from).to_vec())
        };
        check(
            &format!("pinned block lanes flip={flip}"),
            &ins,
            v4(),
            &block_body(true, flip, false),
            &only,
            &want,
            false,
        );
    }
}

/// The 2D routine's half format sum is Metal's two-pass accumulator, against [`metal_catmull_rom`],
/// on footprints that strain it ([`accumulated_cases`]: every exponent, denormals, cancellation
/// against the first tap, neighbours one ulp apart, wide exponent gaps, zeros, and specials
/// anywhere in the footprint), at fractions on and off the edges, `x` and `y` fractions of 0
/// included.
#[test]
fn the_float_plane_half_sum_is_metals_two_pass_accumulator() {
    let mut ins = vec![v4(); 16];
    ins.extend([u(), u(), u()]);
    let mut rng = Rng(0x00b1_c0b1_c5ee_d008);
    let flags = T::HALF;
    let mut cases = Vec::new();
    for taps in accumulated_cases(&mut rng, true, 16, 1200) {
        let (kx, ky) = (fraction(&mut rng), fraction(&mut rng));
        let mut c: Case = taps
            .iter()
            .map(|t| t.iter().map(|x| u64::from(x.to_bits())).collect())
            .collect();
        c.extend([vec![kx], vec![ky], vec![u64::from(flags)]]);
        cases.push((c, taps));
    }
    for half in [false, true] {
        let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| plane_sample(fb, x, half);
        let reference = |c: &Case| {
            let (_, taps) = cases.iter().find(|(k, _)| k == c)?;
            let (kx, ky) = (arg(c, 16) as u32, arg(c, 17) as u32);
            let lanes = (0..4).map(|l| {
                let t: [f32; 16] = std::array::from_fn(|n| taps[n][l]);
                let r = special_result(t.iter().map(|x| (*x, true)))
                    .unwrap_or_else(|| metal_catmull_rom(&t, kx, ky));
                u64::from(r.to_bits())
            });
            Some(lanes.collect())
        };
        let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
        check(
            &format!("two-pass float plane half={half}"),
            &ins,
            v4(),
            &body,
            &only,
            &reference,
            false,
        );
    }
}

/// The 2D routine answers Metal's own results (apple-m3-max-macos26.7) on footprints of a
/// half-float format that pin the two-pass accumulator's rules: a row's cancellation leaving a
/// small term (P2d, 420f03ef), its result in `half`'s subnormal range, a tie, a vertical
/// cancellation at an `x` fraction of 0 whose rows' grid the unit weight's exponent sets
/// (3dc8ffeb), and a result past `half`'s range (1d5028be). [`metal_catmull_rom`] agrees with
/// each, and the exact sum rounded once with none but the last.
#[test]
fn the_two_pass_accumulator_answers_metals_pinned_lanes() {
    let bits = |b: [u32; 16]| b.map(f32::from_bits);
    let sparse = |at: &[(usize, u32)]| {
        let mut t = [0u32; 16];
        for (n, b) in at {
            t[*n] = *b;
        }
        bits(t)
    };
    let big = 0xc77f_e000;
    let pos = 0x477f_e000;
    let lanes: Vec<(&str, u32, u32, [f32; 16], f32)> = vec![
        (
            "row cancellation beside a small term",
            9,
            128,
            sparse(&[
                (3, 0x41e9_e000),
                (8, 0x46c8_c000),
                (9, 0x4108_2000),
                (10, 0x46a3_a000),
            ]),
            f32::from_bits(0x4098_a000),
        ),
        (
            "row cancellation to a subnormal result",
            9,
            128,
            sparse(&[
                (2, 0x3c63_a000),
                (3, 0xbc27_2000),
                (10, 0xc224_0000),
                (11, 0xc4a4_4000),
            ]),
            f32::from_bits(0xb794_0000),
        ),
        (
            "row cancellation beside a tiny term",
            9,
            128,
            sparse(&[
                (1, 0xbb35_6000),
                (4, 0x4088_e000),
                (6, 0x405f_2000),
                (7, 0x3c51_6000),
            ]),
            f32::from_bits(0x3930_6000),
        ),
        (
            "a tie and a tiny term",
            254,
            128,
            sparse(&[(2, 0xc280_0000), (8, 0x3988_a000)]),
            4.0,
        ),
        (
            "a vertical cancellation at x fraction 0",
            0,
            128,
            sparse(&[(1, 0x4512_8000), (5, 0x3e31_8000), (13, 0xc512_8000)]),
            f32::from_bits(0x3dc7_a000),
        ),
        (
            "a vertical cancellation at x fraction 0 to a subnormal",
            0,
            128,
            sparse(&[(1, 0xc0c0_4000), (5, 0xb4a0_0000), (13, 0x40c0_4000)]),
            f32::from_bits(0xb400_0000),
        ),
        (
            "past half's range",
            128,
            128,
            bits([
                big, big, big, big, big, pos, pos, big, big, pos, pos, big, big, big, big, big,
            ]),
            f32::INFINITY,
        ),
    ];
    for (what, kx, ky, taps, want) in &lanes {
        let r = metal_catmull_rom(taps, *kx, *ky);
        assert_eq!(r.to_bits(), want.to_bits(), "reference: {what}");
        let (wx, wy) = (weights(*kx), weights(*ky));
        let exact: f64 = (0..16)
            .map(|n| f64::from(taps[n]) * f64::from(wx[n % 4]) * f64::from(wy[n / 4]))
            .sum::<f64>()
            / 2f64.powi(32);
        let rounded = half_away(exact);
        assert_eq!(
            rounded.to_bits() == want.to_bits(),
            want.is_infinite(),
            "the exact sum (rounded once) answers {rounded} at {what}"
        );
    }
    let splat4 = |x: f32| vec![u64::from(x.to_bits()); 4];
    let cases: Vec<(Case, f32)> = lanes
        .iter()
        .map(|(_, kx, ky, taps, want)| {
            let mut c: Case = taps.iter().map(|x| splat4(*x)).collect();
            c.extend([
                vec![u64::from(*kx)],
                vec![u64::from(*ky)],
                vec![u64::from(T::HALF)],
            ]);
            (c, *want)
        })
        .collect();
    let mut ins = vec![v4(); 16];
    ins.extend([u(), u(), u()]);
    let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| plane_sample(fb, x, false);
    let only: Vec<Case> = cases.iter().map(|(c, _)| c.clone()).collect();
    check(
        "pinned two-pass lanes",
        &ins,
        v4(),
        &body,
        &only,
        &|c| {
            let (_, want) = cases.iter().find(|(k, _)| k == c)?;
            Some(vec![u64::from(want.to_bits()); 4])
        },
        false,
    );
}

/// A half format's axis whose four taps read one texel (each index clamped onto the edge texel
/// past it) is weighed as that texel alone at the unit weight, Metal's weights at fraction 0: two
/// lanes Metal answered (apple-m3-max-macos26.7, a probe on `r16Float` under `clamp_to_edge`),
/// where four copies under the fraction's weights miss by one `half` ulp, and footprints that
/// strain the accumulator ([`accumulated_cases`]) with whole axes repeating one texel, against
/// [`metal_catmull_rom`] at fraction 0 on those axes.
#[test]
fn a_half_axis_on_one_texel_is_weighed_alone() {
    // Each row's four taps one texel (the `x` axis whole), or each column's (the `y` axis).
    let rows = |r: [u32; 4]| -> [f32; 16] { std::array::from_fn(|n| f32::from_bits(r[n / 4])) };
    let columns = |c: [u32; 4]| -> [f32; 16] { std::array::from_fn(|n| f32::from_bits(c[n % 4])) };
    let at = |whole: bool, k: u64| if whole { 0 } else { k as u32 };
    let pinned = [
        (
            "the x axis past the far edge",
            (212, 17),
            (true, false),
            rows([0xbe93_4000, 0xbfa2_8000, 0x3f6d_2000, 0x3fc9_a000]),
            f32::from_bits(0xbf9b_2000),
        ),
        (
            "the y axis past the far edge",
            (94, 36),
            (false, true),
            columns([0x4005_0000, 0x3f85_2000, 0xbfd9_6000, 0xbee5_2000]),
            f32::from_bits(0xbc2e_0000),
        ),
    ];
    // Each case's fractions and axes' words, and Metal's answer where it is pinned.
    let mut cases: Vec<([u64; 4], Option<f32>)> = Vec::new();
    let mut footprints: Vec<Vec<[f32; 4]>> = Vec::new();
    for (what, (kx, ky), (wx, wy), taps, want) in &pinned {
        let r = metal_catmull_rom(taps, at(*wx, *kx), at(*wy, *ky));
        assert_eq!(r.to_bits(), want.to_bits(), "reference: {what}");
        let copies = metal_catmull_rom(taps, *kx as u32, *ky as u32);
        assert_ne!(copies.to_bits(), want.to_bits(), "four copies: {what}");
        footprints.push(taps.iter().map(|x| [*x; 4]).collect());
        cases.push(([*kx, *ky, u64::from(*wx), u64::from(*wy)], Some(*want)));
    }
    let mut rng = Rng(0x00b1_c0b1_c5ee_d565);
    for (n, mut taps) in accumulated_cases(&mut rng, true, 16, 600)
        .into_iter()
        .enumerate()
    {
        let (wx, wy) = [(true, false), (false, true), (true, true)][n % 3];
        for j in 0..4 {
            for i in 0..4 {
                let (fi, fj) = (if wx { 1 } else { i }, if wy { 1 } else { j });
                taps[4 * j + i] = taps[4 * fj + fi];
            }
        }
        let (kx, ky) = (fraction(&mut rng), fraction(&mut rng));
        footprints.push(taps);
        cases.push(([kx, ky, u64::from(wx), u64::from(wy)], None));
    }
    // Each case's inputs: the footprint, the fractions, the format's flags, and the axes' words.
    let inputs: Vec<(Case, Option<f32>, &Vec<[f32; 4]>)> = cases
        .iter()
        .zip(&footprints)
        .map(|((k, want), taps)| {
            let mut c: Case = taps
                .iter()
                .map(|t| t.iter().map(|x| u64::from(x.to_bits())).collect())
                .collect();
            c.extend([
                vec![k[0]],
                vec![k[1]],
                vec![u64::from(T::HALF)],
                vec![k[2]],
                vec![k[3]],
            ]);
            (c, *want, taps)
        })
        .collect();
    let mut ins = vec![v4(); 16];
    ins.extend([u(), u(), u(), u(), u()]);
    let reference = |c: &Case| {
        let (_, want, taps) = inputs.iter().find(|(k, _, _)| k == c)?;
        if let Some(w) = want {
            return Some(vec![u64::from(w.to_bits()); 4]);
        }
        let (kx, ky) = (arg(c, 16), arg(c, 17));
        let (wx, wy) = (arg(c, 19) != 0, arg(c, 20) != 0);
        let lanes = (0..4).map(|l| {
            let t: [f32; 16] = std::array::from_fn(|n| taps[n][l]);
            let r = special_result(t.iter().map(|x| (*x, true)))
                .unwrap_or_else(|| metal_catmull_rom(&t, at(wx, kx), at(wy, ky)));
            u64::from(r.to_bits())
        });
        Some(lanes.collect())
    };
    let only: Vec<Case> = inputs.iter().map(|(c, _, _)| c.clone()).collect();
    for half in [false, true] {
        let body = |fb: &mut FunctionBuilder<'_>, x: &[Value]| {
            let zero = ops::splat(fb, u(), 0)?;
            let wx = fb.binary(BinOp::INotEqual, x[19], zero)?;
            let wy = fb.binary(BinOp::INotEqual, x[20], zero)?;
            plane_sample_whole(fb, x, half, (wx, wy))
        };
        check(
            &format!("one-texel axes half={half}"),
            &ins,
            v4(),
            &body,
            &only,
            &reference,
            false,
        );
    }
}

/// `2^e` for a normal `f32` exponent `e`.
fn p2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// A footprint whose result pins one rule of Metal's accumulator: the format's flags, the fractions
/// (`ky` `None` for the 1D routine, which reads the first two taps), the taps, and Metal's result
/// (`float`).
struct Pinned {
    what: &'static str,
    flags: u32,
    kx: u32,
    ky: Option<u32>,
    taps: [f32; 4],
    want: f32,
}

fn pinned() -> Vec<Pinned> {
    let (f, h) = (T::LINEAR, T::HALF);
    let lane = |what, flags, (kx, ky), taps, want| Pinned {
        what,
        flags,
        kx,
        ky,
        taps,
        want,
    };
    let fine = p2(-10) + p2(-33);
    vec![
        // The grid: 38 bits below the largest term's `e(v) + e(w)` (here `0 + (-2)`), so the
        // small term `fine / 512 = 2^-19 + 2^-42` loses `2^-42`, which `f32` would keep.
        lane(
            "grid truncation",
            f,
            (128, Some(1)),
            [1.0, -1.0, fine, 0.0],
            p2(-19),
        ),
        lane(
            "grid truncation toward zero",
            f,
            (128, Some(1)),
            [1.0, -1.0, -fine, 0.0],
            -p2(-19),
        ),
        // 17 bits for a half format: `(1/2 + 2^-11) / 512` loses `2^-20`, a `half` step.
        lane(
            "half grid truncation",
            h,
            (128, Some(1)),
            [1.0, -1.0, 0.5 + p2(-11), 0.0],
            p2(-10),
        ),
        // 1D: a large term on a tie and a tiny opposite one below the grid, which the exact sum
        // keeps (it rounds the sum down) and the grid drops (the tie rounds away).
        lane(
            "1D grid truncation at a tie",
            f,
            (64, None),
            [1.0 + p2(-23), -p2(-60), 0.0, 0.0],
            0.75 + p2(-23),
        ),
        lane(
            "1D half grid truncation at a tie",
            h,
            (64, None),
            [1.0 + p2(-10), -p2(-20), 0.0, 0.0],
            0.75 + p2(-10),
        ),
        // A tap of zero weight sets no grid (it would make the grid 4) and its special is
        // skipped.
        lane(
            "zero-weight taps",
            f,
            (0, Some(0)),
            [fine, p2(40), p2(40), p2(40)],
            fine,
        ),
        lane(
            "half zero-weight taps",
            h,
            (0, Some(0)),
            [0.5 + p2(-11), p2(15), p2(15), p2(15)],
            0.5 + p2(-11),
        ),
        lane(
            "zero-weight specials",
            f,
            (0, Some(0)),
            [1.5, f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
            1.5,
        ),
        lane(
            "1D zero-weight special",
            f,
            (0, None),
            [2.5, f32::NAN, 0.0, 0.0],
            2.5,
        ),
        // Ties round away from zero: `1 + 2^-24` to `1 + 2^-23` (nearest even would give 1).
        lane(
            "tie away",
            f,
            (128, Some(0)),
            [1.0, 1.0 + p2(-23), 0.0, 0.0],
            1.0 + p2(-23),
        ),
        lane(
            "negative tie away",
            f,
            (128, Some(0)),
            [-1.0, -1.0 - p2(-23), 0.0, 0.0],
            -1.0 - p2(-23),
        ),
        lane(
            "1D tie away",
            f,
            (128, None),
            [1.0, 1.0 + p2(-23), 0.0, 0.0],
            1.0 + p2(-23),
        ),
        lane(
            "half tie away",
            h,
            (128, Some(0)),
            [1.0, 1.0 + p2(-10), 0.0, 0.0],
            1.0 + p2(-10),
        ),
        lane(
            "half denormal tie away",
            h,
            (128, Some(0)),
            [p2(-24), 0.0, 0.0, 0.0],
            p2(-24),
        ),
        // A denormal `f32` sum is `+0`, and a `half` sum that rounds to zero, of either sign.
        lane(
            "denormal +0",
            f,
            (128, Some(0)),
            [-p2(-126), 0.0, 0.0, 0.0],
            0.0,
        ),
        lane(
            "1D denormal +0",
            f,
            (128, None),
            [-p2(-126), 0.0, 0.0, 0.0],
            0.0,
        ),
        lane(
            "half zero +0",
            h,
            (192, Some(0)),
            [-p2(-24), 0.0, 0.0, 0.0],
            0.0,
        ),
        lane(
            "1D half zero +0",
            h,
            (192, None),
            [-p2(-24), 0.0, 0.0, 0.0],
            0.0,
        ),
    ]
}

/// Each rule of Metal's accumulator on a footprint that pins it ([`pinned`]): grid truncation
/// (toward zero, 38 or 17 bits), taps of zero weight, ties away from zero, and a denormal or zero
/// sum `+0`. The reference ([`metal_linear`]) agrees with every pinned result.
#[test]
fn the_accumulator_answers_metals_pinned_lanes() {
    let lanes = pinned();
    for p in &lanes {
        let w: Vec<u32> = match p.ky {
            Some(ky) => quad_weights(p.kx, ky).to_vec(),
            None => span_weights(p.kx).to_vec(),
        };
        let t: Vec<(f32, u32)> = p.taps.iter().copied().zip(w).collect();
        let r = metal_linear(&t, p.flags, false);
        assert_eq!(r.to_bits(), p.want.to_bits(), "reference: {}", p.what);
    }
    let splat4 = |x: f32| vec![u64::from(x.to_bits()); 4];
    let quads: Vec<(Case, &Pinned)> = lanes
        .iter()
        .filter_map(|p| {
            let ky = p.ky?;
            let mut c: Case = p.taps.iter().map(|x| splat4(*x)).collect();
            c.extend([
                vec![u64::from(p.kx)],
                vec![u64::from(ky)],
                vec![u64::from(p.flags)],
            ]);
            Some((c, p))
        })
        .collect();
    let spans: Vec<(Case, &Pinned)> = lanes
        .iter()
        .filter(|p| p.ky.is_none())
        .map(|p| {
            let mut c: Case = p.taps[..2].iter().map(|x| splat4(*x)).collect();
            c.extend([vec![u64::from(p.kx)], vec![u64::from(p.flags)]]);
            (c, p)
        })
        .collect();
    let want = |set: &[(Case, &Pinned)], c: &Case| {
        let (_, p) = set.iter().find(|(k, _)| k == c)?;
        Some(vec![u64::from(p.want.to_bits()); 4])
    };
    let mut ins = vec![v4(); 4];
    ins.extend([u(), u(), u()]);
    let quad = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
        let q = routines::define_quad(fb.module())?;
        let h = ops::splat(fb, boolean(), 0)?;
        let mut args = x[..7].to_vec();
        args.push(h);
        Ok(fb.call(q.float, &args)?.expect("float"))
    };
    let only: Vec<Case> = quads.iter().map(|(c, _)| c.clone()).collect();
    check(
        "pinned quad lanes",
        &ins,
        v4(),
        &quad,
        &only,
        &|c| want(&quads, c),
        false,
    );
    let span = |fb: &mut FunctionBuilder<'_>, x: &[Value]| -> BResult<Value> {
        let s = routines::define_span(fb.module())?;
        let h = ops::splat(fb, boolean(), 0)?;
        Ok(fb
            .call(s.float, &[x[0], x[1], x[2], x[3], h])?
            .expect("float"))
    };
    let only: Vec<Case> = spans.iter().map(|(c, _)| c.clone()).collect();
    let ins = [v4(), v4(), u(), u()];
    check(
        "pinned span lanes",
        &ins,
        v4(),
        &span,
        &only,
        &|c| want(&spans, c),
        false,
    );
}

/// [`Routine::words`] bounds each routine set's emission.
#[test]
fn bicubic_routines_words_bound_their_emission() {
    use crate::spirv::serialize::serialize;
    let size = |body: Body<'_>| -> usize {
        let m = kernel(&[u()], u(), body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let without = size(&|_, x| Ok(x[0]));
    let plane = size(&|fb, x| {
        routines::define_plane(fb.module())?;
        Ok(x[0])
    }) - without;
    let line = size(&|fb, x| {
        routines::define_line(fb.module())?;
        Ok(x[0])
    }) - without;
    let quad = size(&|fb, x| {
        routines::define_quad(fb.module())?;
        Ok(x[0])
    }) - without;
    let span = size(&|fb, x| {
        routines::define_span(fb.module())?;
        Ok(x[0])
    }) - without;
    let block = size(&|fb, x| {
        routines::define_block(fb.module())?;
        Ok(x[0])
    }) - without;
    let cube = size(&|fb, x| {
        routines::define_cube(fb.module())?;
        Ok(x[0])
    }) - without;
    eprintln!(
        "bicubic routines: plane {plane} words, line {line} words, quad {quad} words, span {span} \
         words, block {block} words, cube {cube} words"
    );
    assert!(
        plane <= Routine::Plane.words() as usize,
        "the 2D routines are {plane} words, bound {}",
        Routine::Plane.words()
    );
    assert!(
        line <= Routine::Line.words() as usize,
        "the 1D routines are {line} words, bound {}",
        Routine::Line.words()
    );
    assert!(
        quad <= Routine::Quad.words() as usize,
        "the linear 2D routines are {quad} words, bound {}",
        Routine::Quad.words()
    );
    assert!(
        span <= Routine::Span.words() as usize,
        "the linear 1D routines are {span} words, bound {}",
        Routine::Span.words()
    );
    assert!(
        block <= Routine::Block.words() as usize,
        "the linear 3D routines are {block} words, bound {}",
        Routine::Block.words()
    );
    assert!(
        cube <= Routine::Cube.words() as usize,
        "the linear cube routines are {cube} words, bound {}",
        Routine::Cube.words()
    );
}

/// The scalar values a value of type `ty` holds (a vector's lanes, a struct's members' sum; a
/// pointer none: what it points to is counted where it is loaded).
fn scalars(m: &TypedModule, ty: TypeId) -> usize {
    match m.types().get(ty).expect("a type") {
        Type::Vector { len, .. } => usize::from(*len),
        Type::Struct { members, .. } => members.iter().map(|t| scalars(m, *t)).sum(),
        Type::Pointer { .. } | Type::Void => 0,
        _ => 1,
    }
}

/// What of `f` a device keeps: the instructions without a result but a store to a `Function`
/// variable, the stores to a variable a kept load reads, and every instruction and phi whose
/// result one kept reads, or the blocks' terminators read (the routines have no side effect, so a
/// call whose result nothing reads is dead), as `(block, index)`, phi results, and variables.
struct Kept {
    insts: std::collections::BTreeSet<(usize, usize)>,
    phis: std::collections::BTreeSet<Value>,
    vars: std::collections::BTreeSet<crate::spirv::ids::VarId>,
}

fn kept(m: &TypedModule, f: FuncId) -> Kept {
    use crate::spirv::ir::{Inst, ValueOrigin};
    use std::collections::BTreeSet;
    let func = &m.functions()[f.index()];
    let var_of = |v: Value| match m.value(v).expect("a value").origin {
        ValueOrigin::Var(id) => Some(id),
        _ => None,
    };
    let mut needed = BTreeSet::new();
    for b in &func.blocks {
        m.blocks()[b.index()].term.for_each_value(|v| {
            needed.insert(v);
        });
    }
    let mut k = Kept {
        insts: BTreeSet::new(),
        phis: BTreeSet::new(),
        vars: BTreeSet::new(),
    };
    loop {
        let before = (k.insts.len(), k.phis.len(), k.vars.len());
        for (bi, b) in func.blocks.iter().enumerate() {
            let blk = &m.blocks()[b.index()];
            for (ii, n) in blk.insts.iter().enumerate() {
                let keep = match &n.inst {
                    Inst::Store { ptr, .. } => var_of(*ptr).is_none_or(|v| k.vars.contains(&v)),
                    _ => n.result.is_none_or(|r| needed.contains(&r)),
                };
                if keep && k.insts.insert((bi, ii)) {
                    n.inst.for_each_value(|v| {
                        needed.insert(v);
                    });
                    if let Inst::Load { ptr, .. } = &n.inst {
                        k.vars.extend(var_of(*ptr));
                    }
                }
            }
            for p in &blk.phis {
                if needed.contains(&p.result) && k.phis.insert(p.result) {
                    needed.extend(p.incoming.iter().map(|(v, _)| *v));
                }
            }
        }
        if (k.insts.len(), k.phis.len(), k.vars.len()) == before {
            return k;
        }
    }
}

/// The most scalar values `f` holds live at once once a device inlines every call in it and
/// drops what nothing reads ([`kept`]): a backward liveness over its blocks (a phi's operands live
/// out of their predecessor), each point weighed by the scalars its live values hold; at a call,
/// the values live across it beside the callee's own peak. A constant, a module variable, and a
/// `Function` variable's pointer hold no register; a kept `Function` variable's contents count
/// throughout the function. The emitted form's live state, before a device's scheduler and register
/// allocator, which work from it: a regression in what a sample keeps live shows here first.
fn peak_live(m: &TypedModule, f: FuncId) -> usize {
    use crate::spirv::ids::BlockId;
    use crate::spirv::ir::{Inst, Terminator, ValueOrigin};
    use std::collections::BTreeSet;
    let Kept { insts, phis, vars } = kept(m, f);
    let counted = |v: Value| {
        matches!(
            m.value(v).expect("a value").origin,
            ValueOrigin::Param { .. } | ValueOrigin::Inst(_) | ValueOrigin::Phi(_)
        )
    };
    let weigh = |set: &BTreeSet<Value>| -> usize {
        set.iter()
            .map(|v| scalars(m, m.value(*v).expect("a value").ty))
            .sum()
    };
    let func = &m.functions()[f.index()];
    let block = |b: BlockId| &m.blocks()[b.index()];
    let succs = |b: BlockId| -> Vec<BlockId> {
        match &block(b).term {
            Terminator::Branch(t) => vec![*t],
            Terminator::CondBranch { t, f, .. } => vec![*t, *f],
            Terminator::Switch { default, cases, .. } => std::iter::once(*default)
                .chain(cases.iter().map(|c| c.1))
                .collect(),
            _ => Vec::new(),
        }
    };
    let live_phis = |b: BlockId| block(b).phis.iter().filter(|p| phis.contains(&p.result));
    // What block `bi` reads before defining it (its phis' operands excluded), and defines.
    let local = |bi: usize, b: BlockId| -> (BTreeSet<Value>, BTreeSet<Value>) {
        let blk = block(b);
        let (mut uses, mut defs) = (BTreeSet::new(), BTreeSet::new());
        defs.extend(live_phis(b).map(|p| p.result));
        let mut read = |v: Value, defs: &BTreeSet<Value>| {
            if counted(v) && !defs.contains(&v) {
                uses.insert(v);
            }
        };
        for (ii, n) in blk.insts.iter().enumerate() {
            if insts.contains(&(bi, ii)) {
                n.inst.for_each_value(|v| read(v, &defs));
                defs.extend(n.result);
            }
        }
        blk.term.for_each_value(|v| read(v, &defs));
        (uses, defs)
    };
    // The values `b` passes to its successors' kept phis.
    let phi_uses = |b: BlockId| -> BTreeSet<Value> {
        succs(b)
            .into_iter()
            .flat_map(live_phis)
            .flat_map(|p| p.incoming.iter())
            .filter(|(v, from)| *from == b && counted(*v))
            .map(|(v, _)| *v)
            .collect()
    };
    let sets: Vec<_> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(bi, b)| local(bi, *b))
        .collect();
    let at = |b: BlockId| {
        func.blocks
            .iter()
            .position(|x| *x == b)
            .expect("a block of f")
    };
    let mut live_in = vec![BTreeSet::new(); func.blocks.len()];
    let mut live_out = vec![BTreeSet::new(); func.blocks.len()];
    loop {
        let mut changed = false;
        for (i, b) in func.blocks.iter().enumerate().rev() {
            let mut out = phi_uses(*b);
            for s in succs(*b) {
                let defined: BTreeSet<Value> = live_phis(s).map(|p| p.result).collect();
                out.extend(live_in[at(s)].difference(&defined).copied());
            }
            let (uses, defs) = &sets[i];
            let mut inn: BTreeSet<Value> = out.difference(defs).copied().collect();
            inn.extend(uses.iter().copied());
            // A phi's result is live into its block from the edge.
            inn.extend(live_phis(*b).map(|p| p.result).filter(|v| out.contains(v)));
            if inn != live_in[i] || out != live_out[i] {
                live_in[i] = inn;
                live_out[i] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let vars: usize = vars
        .iter()
        .map(|v| scalars(m, m.vars()[v.index()].pointee))
        .sum();
    let mut peak = 0;
    for (bi, b) in func.blocks.iter().enumerate() {
        let blk = block(*b);
        let mut live = live_out[bi].clone();
        blk.term.for_each_value(|v| {
            if counted(v) {
                live.insert(v);
            }
        });
        peak = peak.max(weigh(&live));
        for (ii, n) in blk.insts.iter().enumerate().rev() {
            if !insts.contains(&(bi, ii)) {
                continue;
            }
            if let Some(r) = n.result {
                live.remove(&r);
            }
            if let Inst::Call { func: callee, args } = &n.inst {
                let held: BTreeSet<Value> =
                    live.iter().copied().filter(|v| !args.contains(v)).collect();
                peak = peak.max(weigh(&held) + peak_live(m, *callee));
            }
            n.inst.for_each_value(|v| {
                if counted(v) {
                    live.insert(v);
                }
            });
            peak = peak.max(weigh(&live));
        }
    }
    peak + vars
}

/// The lane operations `f` comes to once a device inlines every call in it and drops what
/// nothing reads ([`kept`]): each kept instruction weighed by the scalars its result holds (a
/// `uvec4` operation four, `OpUMulExtended`'s two words eight), a call counting its callee's, and
/// a composite's construction or extraction and a bitcast none (moves a device folds into its
/// operands). A device scalarizes the vector operations, so this is what it executes per
/// invocation, before it shares the work two inlined calls repeat on one argument.
fn lane_ops(m: &TypedModule, f: FuncId) -> usize {
    use crate::spirv::ir::Inst;
    let Kept { insts, phis, .. } = kept(m, f);
    let func = &m.functions()[f.index()];
    let mut n = phis.len();
    for (bi, ii) in &insts {
        let node = &m.blocks()[func.blocks[*bi].index()].insts[*ii];
        n += match &node.inst {
            Inst::Call { func: callee, .. } => lane_ops(m, *callee),
            Inst::CompositeConstruct { .. }
            | Inst::CompositeExtract { .. }
            | Inst::Convert {
                op: ConvOp::Bitcast,
                ..
            } => 0,
            _ => node
                .result
                .map_or(0, |r| scalars(m, m.value(r).expect("a value").ty)),
        };
    }
    n
}

/// A 2D sample of a format of `flags`'s class as a device compiles it: [`routines::Plane::sums`]
/// composed as a sample composes it, the class's result returned (the specialization folds the
/// selection, and what the class does not read is dead), each tap born where the sample reads
/// it (a value of the function's one input, as a fetch of the coordinates), the fractions too.
fn class_sample(flags: u32) -> (TypedModule, FuncId) {
    use std::cell::Cell;
    let id = Cell::new(None);
    let defs = |mb: &mut ModuleBuilder| -> BResult<Vec<FuncId>> {
        let plane = routines::define_plane(mb)?;
        let f32_ = mb.ty(Type::Float { width: 32 })?;
        let vf = mb.ty(Type::Vector { elem: f32_, len: 4 })?;
        let u32_ = mb.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let mut fb = mb.function(vf)?;
        let x = fb.param(u32_)?;
        let fb_ = &mut fb;
        // A tap of row `j` (a `u32` value), column `i`: `x + 4j + i`, read as a float.
        let tap = |fb: &mut FunctionBuilder<'_>, j: Value, i: u64| -> BResult<Value> {
            let four = ops::splat(fb, u(), 4)?;
            let at = fb.binary(BinOp::IMul, j, four)?;
            let i = ops::splat(fb, u(), i)?;
            let at = fb.binary(BinOp::IAdd, at, i)?;
            let s = fb.binary(BinOp::IAdd, x, at)?;
            let s = fb.convert(ConvOp::Bitcast, s, f32_)?;
            fb.composite_construct(vf, &[s; 4])
        };
        let mut fetch = |fb: &mut FunctionBuilder<'_>, j: RowIndex| -> BResult<[Value; 4]> {
            let j = match j {
                RowIndex::At(j) => ops::splat(fb, u(), u64::from(j))?,
                RowIndex::Counter(k) => k,
            };
            Ok([
                tap(fb, j, 0)?,
                tap(fb, j, 1)?,
                tap(fb, j, 2)?,
                tap(fb, j, 3)?,
            ])
        };
        let byte = ops::splat(fb_, u(), 255)?;
        let eight = ops::splat(fb_, u(), 8)?;
        let kx = fb_.binary(BinOp::BitwiseAnd, x, byte)?;
        let ky = fb_.binary(BinOp::ShiftRightLogical, x, eight)?;
        let ky = fb_.binary(BinOp::BitwiseAnd, ky, byte)?;
        let bits = ops::splat(fb_, u(), u64::from(flags))?;
        let half = ops::splat(fb_, boolean(), 0)?;
        let apart = ops::splat(fb_, boolean(), 0)?;
        let sums = plane.sums(
            fb_,
            &mut fetch,
            (kx, ky),
            (kx, ky),
            (apart, apart),
            (bits, half),
            false,
        )?;
        let float = sums.float.expect("an emulated sum");
        let r = if flags & T::EXACT != 0 {
            sums.exact
        } else if flags & T::HALF != 0 {
            float.half
        } else if flags & T::LINEAR != 0 {
            float.linear
        } else {
            float.cubic
        };
        fb.ret(Some(r))?;
        let f = fb.finish()?;
        id.set(Some(f));
        Ok(vec![f])
    };
    let m = kernel_defs(
        &[u()],
        v4(),
        &defs,
        &|fb, x, f| Ok(fb.call(f[0], &[x[0]])?.expect("the sample")),
        false,
        false,
    )
    .expect("build");
    (m, id.get().expect("the sample"))
}

/// What a device compiles for a half format's 2D sample holds no more live state than a 32-bit
/// float format's, and comes to no more instructions (a pipeline above 5 s to create is a bug).
/// Holding the footprint's sixteen taps beside its accumulators, a half format's sample spilled
/// registers on an Intel Iris Xe (ANV) and took 16 to 32 s to create; passing each row as the
/// sample read it ended the spills, but its four rows' arithmetic, inlined twice, came to 35,167
/// instructions against the 32-bit float sample's 22,002. The half format's routines now loop over
/// the rows once ([`routines::Plane::sums`]), the body one row's arithmetic, which a device that
/// keeps the loop compiles once; each row's sum and reference is one word a channel, so a device
/// that unrolls the loop reads each tap once and carries one word a row and channel across it.
#[test]
fn a_half_plane_sample_holds_no_more_than_a_float_one() {
    let measure = |flags: u32| {
        let (m, f) = class_sample(flags);
        (peak_live(&m, f), lane_ops(&m, f))
    };
    let (half, linear) = (measure(T::HALF), measure(T::LINEAR));
    let (exact, cubic) = (measure(T::UNORM | T::EXACT), measure(T::UNORM));
    eprintln!(
        "2D samples, (peak live scalars, lane operations): half {half:?}, 32-bit float \
         {linear:?}, exact {exact:?}, other {cubic:?}"
    );
    assert!(
        half.0 <= linear.0,
        "a half format's 2D sample holds {} scalars live, a 32-bit float format's {}",
        half.0,
        linear.0
    );
    assert!(
        half.1 <= linear.1,
        "a half format's 2D sample comes to {} lane operations, a 32-bit float format's {}",
        half.1,
        linear.1
    );
}
