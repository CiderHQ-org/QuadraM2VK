//! A fetched 8-bit unorm attachment's blend on the interpreter
//! (`extended_range::blend::blend_unorm8`, then the 8-bit unorm rule of `arith::color_write`):
//! premultiplied source-over (one over one minus the source alpha, added, for color and alpha),
//! of a `half4` or `float4` output, under fast math and without, against Metal's measured
//! arithmetic. Metal clamps the output to `[0, 1]` (a NaN to 0), reads the destination's code `c`
//! as `h(c / 255)` (`f(c / 255)` for a `float4` output), blends in the output's type (fast math:
//! `h(fma(d, h(1 - a), s))`; without: `h(s + h(d · h(1 - a)))`; in `float` likewise), and stores
//! the result's code by its unorm write (a NaN 0, clamped, the exact product with 255 rounded to
//! nearest). The function writes that code as `c` times the `f32` nearest `1 / 255`.
//!
//! EV FB8-sw1-m3 (plain Metal on the M3 Max under macOS 27.0.1, our own MSL of the guest's
//! shape, a 4096 x 961 target): every `half` alpha in `[0, 1]` against every destination code of
//! each channel, four source draws per pair (half aimed within four units in the last place of a
//! code's midpoint, 2% out of range, infinite or NaN), each of `half4` and `float4`, fast and safe:
//! this arithmetic gives Metal's code in every one of 63 million channels per variant, into
//! `BGRA8Unorm`, and `RGBA8Unorm`, `RG8Unorm` and `R8Unorm` alike. Here every alpha code's `half`
//! (or `float`) meets every destination code in some lane, beside sources aimed at the codes'
//! midpoints, out of range and NaN, and texels read a unit in the last place off `c / 255`.

use super::*;
use crate::lower::arith::extended_range::blend::{blend_unorm8, Blend, BlendArith, BlendMath};
use crate::metal::{BlendFactor as F, BlendOperation as O, ColorWriteMask, PixelFormat};
use crate::pipeline::lower_inputs::{BlendEquation, ChannelWrite};

/// `x` rounded to `half`, nearest even.
fn h(x: f64) -> f64 {
    f16_to_f64(f64_to_f16(x))
}

/// `x` clamped to `[0, 1]`, a NaN to 0.
fn clamp01(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

/// Metal's code of source lane `s` (alpha `a`, both as the output holds them) over destination
/// code `c`, in `arith` under `math` (the module docs).
fn metal_code(arith: BlendArith, math: BlendMath, (s, a): (f64, f64), c: u32) -> u32 {
    let (s, a) = (clamp01(s), clamp01(a));
    let v = match arith {
        BlendArith::Half => {
            let d = h(f64::from(c) / 255.0);
            let m = h(1.0 - a);
            // Products and sums of these `half`s are exact in `f64`.
            match math {
                BlendMath::Fast => h(d * m + s),
                BlendMath::Rounded => h(s + h(d * m)),
            }
        }
        BlendArith::Float => {
            let d = c as f32 / 255.0;
            let (s, a) = (s as f32, a as f32);
            let m = 1.0 - a;
            f64::from(match math {
                BlendMath::Fast => d.mul_add(m, s),
                BlendMath::Rounded => s + d * m,
            })
        }
    };
    (clamp01(v) * 255.0 + 0.5).floor() as u32
}

/// The `f32` bits of a source lane: a `half` (or a `float` where `float`) aimed within four
/// units in the last place of the midpoint between two codes of the blend of destination code
/// `c` under alpha `a`, a uniform one in `[0, 1]`, or, one time in 16, out of range, an infinity
/// or a NaN.
fn source(rng: &mut Rng, float: bool, (a, c): (f64, u32)) -> u64 {
    let w = rng.next();
    const SPECIALS: [f32; 8] = [
        -0.0,
        -1.0,
        2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.5,
        -0.25,
    ];
    if w.is_multiple_of(16) {
        return u64::from(SPECIALS[((w >> 4) % 8) as usize].to_bits());
    }
    let x = if (w >> 8) & 1 == 0 {
        ((w >> 16) & 0xFFFF) as f64 / 65535.0
    } else {
        let k = ((w >> 16) % 255) as f64;
        let ulp = if float {
            2f64.powi(-24)
        } else {
            2f64.powi(-11)
        };
        let step = ((w >> 32) % 9) as f64 - 4.0;
        (k + 0.5) / 255.0 - f64::from(c) / 255.0 * (1.0 - a) + step * ulp
    };
    let x = x.clamp(0.0, 1.0);
    match float {
        true => u64::from((x as f32).to_bits()),
        false => u64::from((h(x) as f32).to_bits()),
    }
}

/// The `f32` bits of a texel lane of code `c` as a storage image reads it: `c / 255`, a unit in
/// the last place off it one time in four (a decode the function must read as `c` all the same).
fn texel(rng: &mut Rng, c: u32) -> u64 {
    let v = c as f32 / 255.0;
    let v = match rng.next() % 8 {
        0 if c > 0 => f32::from_bits(v.to_bits() - 1),
        1 if c < 255 => f32::from_bits(v.to_bits() + 1),
        _ => v,
    };
    u64::from(v.to_bits())
}

#[test]
fn a_fetched_eight_bit_unorm_blend_stores_metals_code() {
    let f4 = vec(float(32), 4);
    let over = BlendEquation {
        source_rgb: F::One,
        destination_rgb: F::OneMinusSourceAlpha,
        rgb_operation: O::Add,
        source_alpha: F::One,
        destination_alpha: F::OneMinusSourceAlpha,
        alpha_operation: O::Add,
    };
    let unorm = Some(ChannelWrite::Unorm { bits: 8 });
    let one_over = f64::from(1.0f32 / 255.0);
    for arith in [BlendArith::Half, BlendArith::Float] {
        let float = arith == BlendArith::Float;
        let mut rng = Rng(0xB1E8_0C0D ^ u64::from(float));
        // Every alpha code's value against every destination code: alpha code `j`, and each
        // case's lanes the codes `4k` to `4k + 3`.
        let mut cases: Vec<Case> = Vec::new();
        for j in 0..256u32 {
            let a = match float {
                true => f64::from(j as f32 / 255.0),
                false => h(f64::from(j) / 255.0),
            };
            let a_bits = u64::from((a as f32).to_bits());
            for k in 0..64u32 {
                let codes: Vec<u32> = (0..4).map(|l| 4 * k + l).collect();
                let mut s: Vec<u64> = codes[..3]
                    .iter()
                    .map(|&c| source(&mut rng, float, (a, c)))
                    .collect();
                s.push(a_bits);
                let d = codes.iter().map(|&c| texel(&mut rng, c)).collect();
                cases.push(vec![s, d]);
            }
        }
        for math in [BlendMath::Fast, BlendMath::Rounded] {
            for pf in [PixelFormat::BGRA8Unorm, PixelFormat::R8Unorm] {
                let what = format!("{arith:?} {math:?} {pf:?}");
                let b = Blend::unorm8(&over, pf, ColorWriteMask::ALL, math)
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                check(
                    &format!("the fetched 8-bit unorm blend, {what}"),
                    &[f4, f4],
                    f4,
                    &|fb, x| {
                        let v = blend_unorm8(fb, &b, arith, (x[0], x[1]))?;
                        crate::lower::arith::color_write::round(fb, f4, v, &[unorm; 4], None)
                    },
                    &cases,
                    &|c| {
                        let a = f64::from(f32v(c[0][3]));
                        Some(
                            (0..4)
                                .map(|l| {
                                    let code = (f64::from(f32v(c[1][l])) * 255.0).round() as u32;
                                    let s = f64::from(f32v(c[0][l]));
                                    // A format without alpha stores no alpha: the blend
                                    // leaves the lane the texel's.
                                    let got = match (pf, l) {
                                        (PixelFormat::R8Unorm, 3) => code,
                                        _ => metal_code(arith, math, (s, a), code),
                                    };
                                    u64::from(((got as f32) * one_over as f32).to_bits())
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

/// Another blend of an 8-bit unorm attachment, a write mask that keeps a channel, and another
/// format are not measured: `Blend::unorm8` refuses each `NotImplemented(FramebufferFetch)`.
#[test]
fn another_eight_bit_unorm_blend_is_not_built() {
    let over = BlendEquation {
        source_rgb: F::One,
        destination_rgb: F::OneMinusSourceAlpha,
        rgb_operation: O::Add,
        source_alpha: F::One,
        destination_alpha: F::OneMinusSourceAlpha,
        alpha_operation: O::Add,
    };
    let refused = |r: Result<Blend, crate::error::Error>| match r {
        Err(crate::error::Error::Unsupported(u)) => {
            u.feature == crate::error::Feature::FramebufferFetch
                && u.needs == crate::error::Needs::NotImplemented
        }
        _ => false,
    };
    let math = BlendMath::Fast;
    let rgb = ColorWriteMask::RED
        .union(ColorWriteMask::GREEN)
        .union(ColorWriteMask::BLUE);
    let cases = [
        (
            BlendEquation {
                source_rgb: F::SourceAlpha,
                ..over
            },
            PixelFormat::BGRA8Unorm,
            ColorWriteMask::ALL,
        ),
        (
            BlendEquation {
                alpha_operation: O::Max,
                ..over
            },
            PixelFormat::RGBA8Unorm,
            ColorWriteMask::ALL,
        ),
        (over, PixelFormat::BGRA8Unorm, rgb),
        (over, PixelFormat::RGBA8UnormSRGB, ColorWriteMask::ALL),
    ];
    for (eq, pf, mask) in cases {
        assert!(
            refused(Blend::unorm8(&eq, pf, mask, math)),
            "{eq:?} {pf:?} {mask:?}"
        );
    }
}
