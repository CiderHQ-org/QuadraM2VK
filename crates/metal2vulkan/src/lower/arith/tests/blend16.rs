//! A fetched 16-bit float attachment's blend constant differences under fast math, on the
//! interpreter (`extended_range::blend::blend_half_float`): `BlendColor` with `OneMinusBlendColor`
//! and `BlendAlpha` with `OneMinusBlendAlpha`, either as the source factor, under `Subtract` and
//! `ReverseSubtract`, for the color channels and alpha alike, against Metal's measured form. The
//! minuend's product is fused into the difference, the subtrahend's rounded first, in the
//! arithmetic of the output's type: in `half` (a `half4` output) the blend color reads `h(c)` and
//! its one minus `h(1 - h(c))`, each sum and product rounded to `half`; in `float` (a `float4`
//! output) they read `c` and `f(1 - c)`, the result the `float` the caller converts.
//!
//! Plain Metal on the M3 Max under macOS 27.0.1, our own MSL, every channel of 16,384 random pixels
//! of each such state of the 677 and 1,350 state sweeps, `half4` and `float4` outputs, under the
//! blend color (0.3, 0.6, 0.9, 0.45), where blue's `h(1 - h(c))` and `h(1 - c)` differ: this form
//! holds Metal's bits in every channel, signed zeros included (through narrower outputs too). The
//! second blend color puts every color channel where the two differ. Before the form was built, the
//! color channels' blend color difference took the rounded form (both products rounded, one minus
//! of the color's `f32`), which differs from Metal's in some channels, and neither difference
//! counted as fitted.
//!
//! Without fast math a zero factor keeps its product (`a_safe_blend_keeps_a_zero_factors_product`):
//! each product rounded, `x · 0` a NaN where `x` is an infinity or a NaN and else a zero of the
//! signed operand's sign, their IEEE 754 sum rounded (`-0` only where both terms are `-0`), every
//! NaN the quiet `0x7FC00000`, and `SourceAlphaSaturated`'s minimum Metal's: a NaN operand gives
//! the other, two zeros `+0` (plain Metal on the M3 Max under macOS 27.0.1, each
//! operation with a zero factor beside each factor, color and alpha, into `RGBA16Float` and
//! `RG16Float` from `half4` and `float4` outputs, sources and texels holding signed zeros,
//! infinities and NaNs: every lane). Before the form was built, a zero factor dropped its term,
//! which kept neither the NaN of an infinite or NaN operand nor the zero sum's sign.
//!
//! Under fast math a zero factor drops its term, and a lone term's zero takes Metal's sign
//! (`a_fast_blend_s_lone_term_takes_metal_s_zero`): `R16Float` and `RG16Float` have no alpha, so
//! one minus the destination alpha is a zero factor; and into `RG16Float` or `RGBA16Float` a
//! destination's product subtracted alone by one minus the source's color or alpha negates the
//! factor, `h(d · h(z - 1))`, as does any one minus in alpha of a `float4` output (our probes
//! `rb16b`, `rb16c` and `rb16e`, the M3 Max under macOS 27.0.1, every lane). Before, the destination
//! alpha of those formats was the texel's one, whose `1 - 1` kept the product `d · 0` in the sum,
//! and every lone term negated its operand: the zero's sign was off Metal's.

use super::*;
use crate::lower::arith::extended_range::blend::{blend_half_float, Blend, BlendArith, BlendMath};
use crate::metal::{BlendFactor as F, BlendOperation as O, ColorWriteMask, PixelFormat};
use crate::pipeline::lower_inputs::BlendEquation;

/// `x` rounded to `half`, nearest even.
fn h(x: f64) -> f64 {
    f16_to_f64(f64_to_f16(x))
}

/// The `f32` bits of a random `half` of magnitude in `[2^-8, 2^4)`, or a signed zero one time in
/// 16; a random `float` of that range where `float`.
fn operand(rng: &mut Rng, float: bool) -> u64 {
    let w = rng.next();
    let sign = if w & 1 == 1 { -1.0f64 } else { 1.0 };
    if (w >> 1) & 15 == 0 {
        return u64::from((0.0 * sign as f32).to_bits());
    }
    let exponent = ((w >> 5) % 12) as i32 - 8;
    let v = match float {
        true => 1.0 + ((w >> 16) & 0x7F_FFFF) as f64 / 8_388_608.0,
        false => 1.0 + ((w >> 16) & 0x3FF) as f64 / 1024.0,
    };
    u64::from(((sign * v * 2f64.powi(exponent)) as f32).to_bits())
}

/// Metal's difference in lane `l` (alpha the last) of source `s` over destination
/// `d`, factors `fs` and `fd` under `op`, blend color `bc`, in `arith`: the `f32` bits of the
/// `half` (`BlendArith::Half`) or the `float`.
fn metal(
    arith: BlendArith,
    (fs, fd, op): (F, F, O),
    bc: [f32; 4],
    l: usize,
    (s, d): (f32, f32),
) -> u64 {
    let factor = |f: F| {
        let alpha = l == 3 || matches!(f, F::BlendAlpha | F::OneMinusBlendAlpha);
        let c = bc[if alpha { 3 } else { l }];
        let one_minus = matches!(f, F::OneMinusBlendColor | F::OneMinusBlendAlpha);
        match (arith, one_minus) {
            (BlendArith::Half, false) => h(f64::from(c)),
            (BlendArith::Half, true) => h(1.0 - h(f64::from(c))),
            (BlendArith::Float, false) => f64::from(c),
            (BlendArith::Float, true) => f64::from(1.0f32 - c),
        }
    };
    let (ks, kd) = (factor(fs), factor(fd));
    // The minuend's product fused, the subtrahend's rounded first.
    let ((x, kx), (y, ky)) = match op {
        O::Subtract => ((s, ks), (d, kd)),
        _ => ((d, kd), (s, ks)),
    };
    match arith {
        // Products and sums of these `half`s are exact in `f64`.
        BlendArith::Half => {
            let r = f64::from(x).mul_add(kx, -h(f64::from(y) * ky));
            u64::from((h(r) as f32).to_bits())
        }
        BlendArith::Float => {
            let (kx, ky) = (kx as f32, ky as f32);
            u64::from(x.mul_add(kx, -(y * ky)).to_bits())
        }
    }
}

#[test]
fn a_fetched_blend_constant_difference_is_metals() {
    let f4 = vec(float(32), 4);
    let u4 = vec(int(32), 4);
    let colors: [[f32; 4]; 2] = [[0.3, 0.6, 0.9, 0.45], [0.7, 0.55, 0.321, 0.8]];
    let pairs = [
        (F::BlendColor, F::OneMinusBlendColor),
        (F::OneMinusBlendColor, F::BlendColor),
        (F::BlendAlpha, F::OneMinusBlendAlpha),
        (F::OneMinusBlendAlpha, F::BlendAlpha),
    ];
    for arith in [BlendArith::Half, BlendArith::Float] {
        let float = arith == BlendArith::Float;
        let mut rng = Rng(0xD1FF_B1E0 ^ u64::from(float));
        let mut cases: Vec<Case> = Vec::new();
        for bc in colors {
            let color: Vec<u64> = bc.iter().map(|c| u64::from(c.to_bits())).collect();
            for _ in 0..300 {
                let s = (0..4).map(|_| operand(&mut rng, float)).collect();
                let d = (0..4).map(|_| operand(&mut rng, false)).collect();
                cases.push(vec![s, d, color.clone()]);
            }
        }
        for (fs, fd) in pairs {
            for op in [O::Subtract, O::ReverseSubtract] {
                let eq = BlendEquation {
                    source_rgb: fs,
                    destination_rgb: fd,
                    rgb_operation: op,
                    source_alpha: fs,
                    destination_alpha: fd,
                    alpha_operation: op,
                };
                let what = format!("{arith:?} {fs:?} {fd:?} {op:?}");
                let b = Blend::half_float(
                    &eq,
                    PixelFormat::RGBA16Float,
                    ColorWriteMask::ALL,
                    BlendMath::Fast,
                )
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert!(b.fitted(), "{what}: {b:?}");
                for q in HALF_FORMS {
                    check(
                        &format!("the fetched blend's {what}, half by {q:?}"),
                        &[f4, f4, u4],
                        f4,
                        &|fb, x| {
                            fb.module().set_quantize_form(q);
                            blend_half_float(fb, &b, arith, (x[0], x[1]), Some(x[2]))
                        },
                        &cases,
                        &|c| {
                            let bc: [f32; 4] = std::array::from_fn(|l| f32v(c[2][l]));
                            Some(
                                (0..4)
                                    .map(|l| {
                                        let sd = (f32v(c[0][l]), f32v(c[1][l]));
                                        metal(arith, (fs, fd, op), bc, l, sd)
                                    })
                                    .collect(),
                            )
                        },
                        false,
                    );
                }
            }
        }
    }
}

/// The `f32` bits of a special `half` (a signed zero, an infinity, a NaN of either sign, one, the
/// smallest subnormal and normal, the largest finite) one time in three, else a random `half` (or
/// `float` where `float`) of [`operand`].
fn special(rng: &mut Rng, float: bool) -> u64 {
    const SPECIALS: [u32; 12] = [
        0x0000_0000,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0xFFC0_0000,
        0x3F80_0000,
        0xBF80_0000,
        0x3380_0000,
        0x3880_0000,
        0x477F_E000,
        0xC77F_E000,
    ];
    let w = rng.next();
    match w % 3 {
        0 => u64::from(SPECIALS[((w >> 2) % 12) as usize]),
        _ => operand(rng, float),
    }
}

/// `x` rounded in `arith`: to `half`, nearest even, or to `float`.
fn rounded(arith: BlendArith, x: f64) -> f64 {
    match arith {
        BlendArith::Half => h(x),
        BlendArith::Float => f64::from(x as f32),
    }
}

/// Metal's `min` in `SourceAlphaSaturated`: a NaN operand gives the other, two
/// zeros `+0`.
fn metal_min(x: f64, y: f64) -> f64 {
    match (x.is_nan(), y.is_nan()) {
        (true, _) => y,
        (_, true) => x,
        _ if x == 0.0 && y == 0.0 => 0.0,
        _ => x.min(y),
    }
}

/// Metal's blend without fast math in lane `l` (alpha the last) of source `s` over
/// destination `d`, factors `fs` and `fd` under `op`, blend color `bc`, in `arith`: each product
/// rounded, a zero factor's kept, the IEEE 754 sum rounded; the `f32` bits of the `half` or the
/// `float`, every NaN `0x7FC00000`.
fn metal_safe(
    arith: BlendArith,
    (fs, fd, op): (F, F, O),
    bc: [f32; 4],
    l: usize,
    (s, d): ([f32; 4], [f32; 4]),
) -> u64 {
    let alpha = l == 3;
    let r = |x: f64| rounded(arith, x);
    let (sl, dl) = (f64::from(s[l]), f64::from(d[l]));
    let (sa, da) = (f64::from(s[3]), f64::from(d[3]));
    let factor = |f: F| match f {
        F::Zero => 0.0,
        F::One => 1.0,
        F::SourceColor => sl,
        F::OneMinusSourceAlpha => r(1.0 - sa),
        F::DestinationColor => dl,
        F::OneMinusDestinationAlpha => r(1.0 - da),
        F::SourceAlphaSaturated if alpha => 1.0,
        F::SourceAlphaSaturated => metal_min(sa, r(1.0 - da)),
        F::BlendColor => r(f64::from(bc[l])),
        _ => panic!("no factor {f:?} here"),
    };
    // Every product of two `half`s or two `float`s is exact in `f64`.
    let (ps, pd) = (r(sl * factor(fs)), r(dl * factor(fd)));
    let v = r(match op {
        O::Add => ps + pd,
        O::Subtract => ps - pd,
        O::ReverseSubtract => pd - ps,
        _ => panic!("no operation {op:?} here"),
    });
    match v.is_nan() {
        true => 0x7FC0_0000,
        false => u64::from((v as f32).to_bits()),
    }
}

#[test]
fn a_safe_blend_keeps_a_zero_factors_product() {
    let f4 = vec(float(32), 4);
    let u4 = vec(int(32), 4);
    let bc: [f32; 4] = [0.3, 0.6, 0.9, 0.45];
    let others = [
        F::One,
        F::SourceColor,
        F::OneMinusSourceAlpha,
        F::DestinationColor,
        F::OneMinusDestinationAlpha,
        F::SourceAlphaSaturated,
        F::BlendColor,
    ];
    let pairs: Vec<(F, F)> = (others.iter())
        .flat_map(|&o| [(F::Zero, o), (o, F::Zero)])
        .chain([(F::Zero, F::Zero)])
        .collect();
    for arith in [BlendArith::Half, BlendArith::Float] {
        let float = arith == BlendArith::Float;
        let mut rng = Rng(0x2E80_FAC7 ^ u64::from(float));
        let color: Vec<u64> = bc.iter().map(|c| u64::from(c.to_bits())).collect();
        let mut cases: Vec<Case> = Vec::new();
        // The saturated factor's zeros: a source alpha of `-0` beside a destination alpha of one.
        for z in [0x8000_0000u64, 0] {
            let s = vec![0xBF80_0000, 0x3F80_0000, 0x8000_0000, z];
            let d = vec![0x3F80_0000, 0x8000_0000, 0xBF80_0000, 0x3F80_0000];
            cases.push(vec![s, d, color.clone()]);
        }
        for _ in 0..400 {
            let s = (0..4).map(|_| special(&mut rng, float)).collect();
            let d = (0..4).map(|_| special(&mut rng, false)).collect();
            cases.push(vec![s, d, color.clone()]);
        }
        for &(fs, fd) in &pairs {
            for op in [O::Add, O::Subtract, O::ReverseSubtract] {
                let eq = BlendEquation {
                    source_rgb: fs,
                    destination_rgb: fd,
                    rgb_operation: op,
                    source_alpha: fs,
                    destination_alpha: fd,
                    alpha_operation: op,
                };
                let what = format!("{arith:?} {fs:?} {fd:?} {op:?}");
                let b = Blend::half_float(
                    &eq,
                    PixelFormat::RGBA16Float,
                    ColorWriteMask::ALL,
                    BlendMath::Rounded,
                )
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                for q in HALF_FORMS {
                    check_accept(
                        &format!("the safe blend's {what}, half by {q:?}"),
                        &[f4, f4, u4],
                        f4,
                        &|fb, x| {
                            fb.module().set_quantize_form(q);
                            blend_half_float(fb, &b, arith, (x[0], x[1]), Some(x[2]))
                        },
                        &cases,
                        &|c| {
                            let s: [f32; 4] = std::array::from_fn(|l| f32v(c[0][l]));
                            let d: [f32; 4] = std::array::from_fn(|l| f32v(c[1][l]));
                            Some(
                                (0..4)
                                    .map(|l| metal_safe(arith, (fs, fd, op), bc, l, (s, d)))
                                    .collect(),
                            )
                        },
                        &|got, want| {
                            let nan = |w: u64| f32v(w).is_nan();
                            got == want || (q == QuantizeForm::Controlled && nan(got) && nan(want))
                        },
                    );
                }
            }
        }
    }
}

/// The values of [`a_fast_blend_s_lone_term_takes_metal_s_zero`]'s grid: the signed zeros, one,
/// -1, 2.5, -0.75, the infinities and a NaN, each exact in `half` and every product and `1 - x` of
/// them too.
const GRID: [f32; 9] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.5,
    -0.75,
    f32::INFINITY,
    -f32::INFINITY,
    f32::NAN,
];

/// Metal's blend under fast math (our probes `rb16b`, `rb16c`, `rb16e`) in lane `l` (alpha the
/// last) of an attachment of `channels` channels, of source `s` over destination `d`, factors `fs`
/// and `fd` under `op`, blend color one, one of whose terms is dropped by a zero factor (`None`
/// where neither is), in `arith`: the remaining term, its operand negated or, a destination's
/// product subtracted by one minus a source value into two or more channels (any one minus in
/// alpha in `float`), its factor; both dropped `+0`.
fn metal_fast_term(
    arith: BlendArith,
    (fs, fd, op): (F, F, O),
    channels: u32,
    l: usize,
    (s, d): ([f32; 4], [f32; 4]),
) -> Option<f64> {
    let alpha = l == 3;
    let (sl, dl) = (f64::from(s[l]), f64::from(d[l]));
    let sa = f64::from(s[3]);
    let da = if channels == 4 { f64::from(d[3]) } else { 1.0 };
    // Each factor: `None` where it is zero by the state, else its `1 - z` (`Some(z)`) or value;
    // the blend color one.
    let factor = |f: F| -> Option<(Option<f64>, f64)> {
        let z = match f {
            F::Zero => return None,
            F::OneMinusDestinationAlpha if channels < 4 => return None,
            F::One | F::BlendColor => return Some((None, 1.0)),
            F::SourceColor => return Some((None, sl)),
            F::SourceAlpha => return Some((None, sa)),
            F::DestinationColor => return Some((None, dl)),
            F::DestinationAlpha => return Some((None, da)),
            F::SourceAlphaSaturated if alpha => return Some((None, 1.0)),
            F::SourceAlphaSaturated => return Some((None, metal_min(sa, 1.0 - da))),
            F::OneMinusBlendColor | F::OneMinusBlendAlpha => 1.0,
            F::OneMinusSourceAlpha => sa,
            F::OneMinusSourceColor if alpha => sa,
            F::OneMinusSourceColor => sl,
            F::OneMinusDestinationAlpha => da,
            F::OneMinusDestinationColor if alpha => da,
            F::OneMinusDestinationColor => dl,
            _ => panic!("no factor {f:?} here"),
        };
        Some((Some(z), 1.0 - z))
    };
    let source_value = |f: F| matches!(f, F::OneMinusSourceAlpha | F::OneMinusSourceColor);
    let (ks, kd) = (factor(fs), factor(fd));
    let (x, k, f, sign) = match (ks, kd) {
        (None, None) => return Some(0.0),
        (Some(_), Some(_)) => return None,
        (Some(k), None) => (sl, k, fs, if op == O::ReverseSubtract { -1.0 } else { 1.0 }),
        (None, Some(k)) => (dl, k, fd, if op == O::Subtract { -1.0 } else { 1.0 }),
    };
    let destination = kd.is_some();
    let into = destination
        && sign < 0.0
        && channels > 1
        && k.0.is_some()
        && (source_value(f) || (alpha && arith == BlendArith::Float));
    Some(match (into, k.0) {
        (true, Some(z)) => x * (z - 1.0),
        _ => sign * (x * k.1),
    })
}

#[test]
fn a_fast_blend_s_lone_term_takes_metal_s_zero() {
    let f4 = vec(float(32), 4);
    let u4 = vec(int(32), 4);
    // The probes' states with a zero factor.
    let states = [
        (F::One, F::OneMinusDestinationAlpha, O::Add),
        (F::One, F::OneMinusDestinationAlpha, O::Subtract),
        (F::One, F::OneMinusDestinationAlpha, O::ReverseSubtract),
        (F::SourceAlpha, F::OneMinusDestinationAlpha, O::Add),
        (F::SourceAlpha, F::OneMinusDestinationAlpha, O::Subtract),
        (F::OneMinusDestinationAlpha, F::One, O::Subtract),
        (
            F::OneMinusDestinationAlpha,
            F::SourceAlpha,
            O::ReverseSubtract,
        ),
        (
            F::OneMinusDestinationAlpha,
            F::OneMinusDestinationAlpha,
            O::Subtract,
        ),
        (F::Zero, F::OneMinusSourceAlpha, O::Add),
        (F::Zero, F::OneMinusSourceAlpha, O::Subtract),
        (F::Zero, F::OneMinusSourceColor, O::Subtract),
        (F::Zero, F::OneMinusDestinationAlpha, O::Subtract),
        (F::Zero, F::OneMinusDestinationColor, O::Subtract),
        (F::Zero, F::OneMinusBlendColor, O::Subtract),
        (F::Zero, F::OneMinusBlendAlpha, O::Subtract),
        (F::Zero, F::BlendColor, O::Subtract),
        (F::Zero, F::SourceColor, O::Subtract),
        (F::Zero, F::SourceAlpha, O::Subtract),
        (F::Zero, F::SourceAlphaSaturated, O::Subtract),
        (F::Zero, F::One, O::Subtract),
        (F::Zero, F::DestinationAlpha, O::Add),
        (F::Zero, F::DestinationAlpha, O::Subtract),
        (F::Zero, F::DestinationColor, O::Subtract),
        (F::OneMinusSourceAlpha, F::Zero, O::Subtract),
        (F::OneMinusSourceAlpha, F::Zero, O::ReverseSubtract),
        (F::OneMinusSourceColor, F::Zero, O::ReverseSubtract),
        (F::OneMinusDestinationAlpha, F::Zero, O::ReverseSubtract),
        (F::OneMinusDestinationColor, F::Zero, O::ReverseSubtract),
        (F::OneMinusBlendColor, F::Zero, O::ReverseSubtract),
        (F::SourceAlpha, F::Zero, O::Subtract),
        (F::SourceAlpha, F::Zero, O::ReverseSubtract),
        (F::SourceAlphaSaturated, F::Zero, O::ReverseSubtract),
        (F::DestinationColor, F::Zero, O::ReverseSubtract),
        (F::DestinationAlpha, F::Zero, O::ReverseSubtract),
        (F::One, F::Zero, O::ReverseSubtract),
        (F::Zero, F::Zero, O::Add),
        (F::Zero, F::Zero, O::Subtract),
    ];
    let formats = [
        (PixelFormat::R16Float, 1),
        (PixelFormat::RG16Float, 2),
        (PixelFormat::RGBA16Float, 4),
    ];
    let bits = |x: f32| u64::from(x.to_bits());
    for arith in [BlendArith::Half, BlendArith::Float] {
        for (pf, channels) in formats {
            // Each source of color `GRID[a]` and alpha `GRID[b]` over a texel of color `GRID[c]`
            // and alpha `GRID[e]`, or one, the texel's, where the format has no alpha.
            let n = GRID.len();
            let alphas = if channels == 4 { n } else { 1 };
            let mut cases: Vec<Case> = Vec::new();
            for i in 0..n * n * n * alphas {
                let (a, b, c, e) = (
                    i / (n * n * alphas),
                    (i / (n * alphas)) % n,
                    (i / alphas) % n,
                    i % alphas,
                );
                let da = if channels == 4 { GRID[e] } else { 1.0 };
                let s = [GRID[a], GRID[a], GRID[a], GRID[b]].map(bits).to_vec();
                let d = [GRID[c], GRID[c], GRID[c], da].map(bits).to_vec();
                cases.push(vec![s, d, vec![bits(1.0); 4]]);
            }
            // A zero factor of the state or the format's, which drops its term.
            let dropped = |f: F| f == F::Zero || (f == F::OneMinusDestinationAlpha && channels < 4);
            for (fs, fd, op) in states
                .into_iter()
                .filter(|&(fs, fd, _)| dropped(fs) || dropped(fd))
            {
                let eq = BlendEquation {
                    source_rgb: fs,
                    destination_rgb: fd,
                    rgb_operation: op,
                    source_alpha: fs,
                    destination_alpha: fd,
                    alpha_operation: op,
                };
                let what = format!("{arith:?} {pf:?} {fs:?} {fd:?} {op:?}");
                let b = Blend::half_float(&eq, pf, ColorWriteMask::ALL, BlendMath::Fast)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert!(b.fitted(), "{what}: {b:?}");
                // The channels the format stores: the store drops the others.
                let stored = channels as usize;
                for q in HALF_FORMS {
                    check_accept(
                        &format!("the fast blend's {what}, half by {q:?}"),
                        &[f4, f4, u4],
                        f4,
                        &|fb, x| {
                            fb.module().set_quantize_form(q);
                            blend_half_float(fb, &b, arith, (x[0], x[1]), Some(x[2]))
                        },
                        &cases,
                        &|c| {
                            let s: [f32; 4] = std::array::from_fn(|l| f32v(c[0][l]));
                            let d: [f32; 4] = std::array::from_fn(|l| f32v(c[1][l]));
                            let lanes = (0..4).map(|l| match l < stored {
                                true => metal_fast_term(arith, (fs, fd, op), channels, l, (s, d))
                                    .map(|v| bits(v as f32)),
                                // A lane the store drops: any answer.
                                false => Some(u64::MAX),
                            });
                            lanes.collect()
                        },
                        &|got, want| {
                            let nan = |w: u64| f32v(w).is_nan();
                            want == u64::MAX || got == want || (nan(got) && nan(want))
                        },
                    );
                }
            }
        }
    }
}
