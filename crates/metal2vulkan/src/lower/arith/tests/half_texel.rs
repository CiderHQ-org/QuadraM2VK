//! Interpreter differential tests of the conversion of a texel to a `half` result by the bound
//! format's class (`lower::texture::half`). References are plain Rust: Metal's measured rules on
//! exact values (a normalized code's value `c / (2^b - 1)`, a grid step's `n / D`), and, for a
//! value off its format's grid (a filter other than Metal's), the routine's own `f32` step
//! `RoundEven(x · D)`. Each runs in every [`QuantizeForm`], whose rounding the routine shares where
//! it is Metal's.

use super::texel_round::half_of;
use super::*;
use crate::lower::arith::texel_round::HalfRounding;
use crate::lower::texture::half::{self, HalfFamilies, HalfForm, TexelLanes};
use crate::metal::texel_format::{texel_format, PackedTexels, TexelFormat as C};
use crate::metal::PixelFormat as P;
use crate::spirv::builder::QuantizeForm;

/// The module roundings the routine is built under.
const FORMS: [QuantizeForm; 4] = [
    QuantizeForm::Integer,
    QuantizeForm::Device,
    QuantizeForm::Converted,
    QuantizeForm::Controlled,
];

fn v4() -> Form {
    vec(float(32), 4)
}

/// The lanes of an access of form `form` in the earlier tests: a read's one texel's, a sample's a
/// value that may be filtered (those classes convert alike either way).
fn lanes_of(form: HalfForm) -> TexelLanes {
    match form {
        HalfForm::Read => TexelLanes::Texel,
        HalfForm::Sample => TexelLanes::Filtered,
    }
}

/// The routine at class `class` (an input), form `form`, and lanes `lanes`, in a module rounding to
/// `half` by `q`.
fn body_lanes(
    form: HalfForm,
    lanes: TexelLanes,
    q: QuantizeForm,
) -> impl Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value> {
    body_families(form, lanes, q, HalfFamilies::Every)
}

/// [`body_lanes`] of the routine telling apart `families`.
fn body_families(
    form: HalfForm,
    lanes: TexelLanes,
    q: QuantizeForm,
    families: HalfFamilies,
) -> impl Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value> {
    move |fb, x| {
        fb.module().set_quantize_form(q);
        let r = half::define(fb.module(), families)?;
        let lanes = lanes.value(fb)?;
        half::convert(fb, r, x[0], x[1], (form, lanes))
    }
}

/// Run the routine over `texels` at `class`, `form`, and `lanes`, against `reference` of each
/// lane's index and value, in every [`QuantizeForm`].
fn run_lanes(
    what: &str,
    class: u32,
    (form, lanes): (HalfForm, TexelLanes),
    texels: &[[u64; 4]],
    reference: &dyn Fn(usize, u64) -> u64,
) {
    let cases: Vec<Case> = texels
        .iter()
        .map(|t| vec![t.to_vec(), vec![u64::from(class)]])
        .collect();
    for q in FORMS {
        check(
            &format!("{what} {form:?} {lanes:?} {q:?}"),
            &[v4(), int(32)],
            v4(),
            &body_lanes(form, lanes, q),
            &cases,
            &|c| {
                Some(
                    c[0].iter()
                        .enumerate()
                        .map(|(i, x)| reference(i, *x))
                        .collect(),
                )
            },
            false,
        );
    }
}

/// Run the routine over `texels` (four lanes each, `f32` bits) at `class` and `form`, against
/// `reference` lane by lane, in every [`QuantizeForm`].
fn run(
    what: &str,
    class: u32,
    form: HalfForm,
    texels: &[[u64; 4]],
    reference: &dyn Fn(u64) -> u64,
) {
    run_lanes(what, class, (form, lanes_of(form)), texels, &|_, x| {
        reference(x)
    });
}

/// `x` in groups of four lanes, the last padded with zeros.
fn quads(x: &[u64]) -> Vec<[u64; 4]> {
    x.chunks(4)
        .map(|c| {
            let mut q = [0u64; 4];
            q[..c.len()].copy_from_slice(c);
            q
        })
        .collect()
}

/// `f32` bits of the `half` nearest the quotient `v` (`f64`'s correctly rounded quotient, whose
/// second rounding to `half` is innocuous, `texel_round`'s tests' note), ties to even.
fn half_bits(v: f64) -> u64 {
    u64::from((f16_to_f64(f64_to_f16(v)) as f32).to_bits())
}

/// A normalized format: its class, signedness, code scale `2^b - 1` (`2^(b-1) - 1` signed), and
/// grid `D`.
#[derive(Clone, Copy, Debug)]
struct Norm {
    class: u32,
    signed: bool,
    scale: i64,
    d: i64,
}

/// The treatment bits of `pf`.
fn bits(pf: P) -> u32 {
    texel_format(pf).0
}

fn norms() -> [Norm; 4] {
    [
        Norm {
            class: bits(P::RGBA8Unorm),
            signed: false,
            scale: 255,
            d: C::GRID8 as i64,
        },
        Norm {
            class: bits(P::RGBA8Snorm),
            signed: true,
            scale: 127,
            d: 16 * 127,
        },
        Norm {
            class: bits(P::RGBA16Unorm),
            signed: false,
            scale: 65535,
            d: C::GRID16 as i64,
        },
        Norm {
            class: bits(P::RGBA16Snorm),
            signed: true,
            scale: 32767,
            d: 32767,
        },
    ]
}

impl Norm {
    fn codes(self) -> Vec<i64> {
        let (lo, hi) = if self.signed {
            (-self.scale - 1, self.scale)
        } else {
            (0, self.scale)
        };
        let step = if hi > 255 { 7 } else { 1 };
        (lo..=hi)
            .step_by(step)
            .chain([lo, lo + 1, -1, 0, 1, hi - 1, hi])
            .filter(|c| (lo..=hi).contains(c))
            .collect()
    }

    /// Vulkan's value of code `c` (`f32` bits): `c / scale` correctly rounded, `-1` for the
    /// least signed code.
    fn value(self, c: i64) -> u64 {
        let c = c.max(-self.scale);
        u64::from(((c as f64 / self.scale as f64) as f32).to_bits())
    }

    /// `f32` bits of grid step `n`'s value `n / D`.
    fn on_grid(self, n: i64) -> u64 {
        u64::from(((n as f64 / self.d as f64) as f32).to_bits())
    }

    /// The routine's `half` result of grid step `n`: `+0` for zero, else `n / D` to nearest
    /// even.
    fn result(self, n: i64) -> u64 {
        if n == 0 {
            return 0;
        }
        half_bits(n as f64 / self.d as f64)
    }

    /// The routine's step of `f32` value `x`: the product rounded to `f32`, clamped, to nearest
    /// even.
    fn step(self, x: u64) -> i64 {
        let d = self.d as f32;
        let y = (f32::from_bits(x as u32) * d).clamp(-d, d);
        y.round_ties_even() as i64
    }
}

/// A read or a nearest sample of a normalized texel is its code's value `c / (2^b - 1)`
/// correctly rounded to `half` (`-1` for the least signed code); a linear sample's value on
/// the format's grid is its step `n / D` to nearest even.
#[test]
fn a_normalized_texel_converts_on_its_grid() {
    for f in norms() {
        let codes = f.codes();
        let texels: Vec<u64> = codes.iter().map(|c| f.value(*c)).collect();
        // Each value's code (the least signed code reads as the next one's `-1`).
        let code: std::collections::BTreeMap<u64, i64> = codes
            .iter()
            .map(|c| (f.value(*c), (*c).max(-f.scale)))
            .collect();
        let want = |x: u64| {
            let c = code.get(&x).copied().unwrap_or(0);
            if c == 0 {
                0
            } else {
                half_bits(c as f64 / f.scale as f64)
            }
        };
        for form in [HalfForm::Read, HalfForm::Sample] {
            run(
                &format!("codes {f:?}"),
                f.class,
                form,
                &quads(&texels),
                &want,
            );
        }
        let lo = if f.signed { -f.d } else { 0 };
        let mut rng = Rng(0x00b1_f0b1_f5ee_d000 ^ f.d as u64);
        // The steps whose `f32` quotient, or product by the reciprocal, rounds to another `half`
        // on the 16-bit grids (`texel_round::ones_ratio_to_half`), both signs.
        let twice = [65327, 65391, 65455, 65519, 32727, 32759];
        let steps: Vec<i64> = (0..2000)
            .map(|_| lo + (rng.next() % (f.d - lo + 1) as u64) as i64)
            .chain([lo, lo + 1, -1, 0, 1, f.d - 1, f.d])
            .chain(twice.iter().flat_map(|n| [*n, -*n]))
            .filter(|n| (lo..=f.d).contains(n))
            .collect();
        let sums: Vec<u64> = steps.iter().map(|n| f.on_grid(*n)).collect();
        run(
            &format!("grid {f:?}"),
            f.class,
            HalfForm::Sample,
            &quads(&sums),
            &|x| f.result(f.step(x)),
        );
        // A value off the grid (another device's filter) takes the nearest step.
        let off: Vec<u64> = (0..2000)
            .map(|_| {
                let v = (rng.next() % 1_000_001) as f64 / 1e6;
                let v = if f.signed && rng.next() & 1 == 0 {
                    -v
                } else {
                    v
                };
                u64::from((v as f32).to_bits())
            })
            .collect();
        run(
            &format!("off grid {f:?}"),
            f.class,
            HalfForm::Sample,
            &quads(&off),
            &|x| f.result(f.step(x)),
        );
    }
}

/// Every half below `2^-13` (the denormals and the least normal binade), random `f32` patterns of
/// every magnitude, and the rounding edges: the tie below the least denormal, the overflow, the
/// infinities, an `f32` denormal, and `-0`.
fn float_texels() -> Vec<u64> {
    let mut rng = Rng(0x00b1_f0b1_f5ee_d001);
    let mut out: Vec<u64> = (0u32..0x800)
        .map(|h| u64::from((f16_to_f64(h as u16) as f32).to_bits()))
        .collect();
    out.extend((0..4000).map(|_| rng.next() & 0xFFFF_FFFF));
    out.extend([
        0x3300_0000u64, // 2^-25: the tie below the smallest half denormal
        0x477F_E000,    // 65504
        0x477F_F000,    // 65520: the tie past the largest half
        0x4780_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x0000_0001,
        0x8000_0000,
    ]);
    out
}

/// A 32-bit float texel: a read or gather to nearest even, past `65504` to the infinity, half
/// denormals kept; a sample toward zero, saturating at `±65504`. A half-float texel: a read of
/// the half itself, and a linear sum to nearest with ties away from zero. With every tie of
/// `texel_round`'s probes (each half denormal's, every 13th normal half's, the top binade's), where
/// a sample's directed rounding leaves the module's to nearest even.
#[test]
fn a_float_texel_rounds_by_its_access() {
    let mut texels = float_texels();
    texels.extend(super::texel_round::half_probe_bits());
    let texels = quads(&texels);
    let rule =
        |mode: HalfRounding| move |x: u64| u64::from(half_of(f32::from_bits(x as u32), mode));
    run(
        "float32",
        bits(P::RGBA32Float),
        HalfForm::Read,
        &texels,
        &rule(HalfRounding::NearestEven),
    );
    run(
        "float32",
        bits(P::RGBA32Float),
        HalfForm::Sample,
        &texels,
        &rule(HalfRounding::TowardZeroSaturating),
    );
    run(
        "half",
        bits(P::RGBA16Float),
        HalfForm::Read,
        &texels,
        &rule(HalfRounding::NearestEven),
    );
    run(
        "half",
        bits(P::RGBA16Float),
        HalfForm::Sample,
        &texels,
        &rule(HalfRounding::NearestAway),
    );
}

/// A format whose conversion is not measured takes every other `half` texel's exact rounding
/// (`narrow::round_texel`): to nearest even, half denormals kept, a NaN or an infinity itself, in
/// either form.
#[test]
fn an_unmeasured_format_takes_the_exact_rounding() {
    let texels = quads(&float_texels());
    let exact = |x: u64| u64::from(half_of(f32::from_bits(x as u32), HalfRounding::NearestEven));
    // Unmeasured: no family bit, channels alone, or a normalized family without `EXACT` (the
    // bicubic treatment of an sRGB or packed format).
    for class in [
        C::OTHER.0,
        0xF << 4,
        crate::metal::bicubic::BicubicTexels::UNORM | 0xF << 4,
    ] {
        for form in [HalfForm::Read, HalfForm::Sample] {
            run(&format!("class {class}"), class, form, &texels, &exact);
        }
    }
}

/// A two-plane YCbCr format's one texel and a 2D gather's lanes: its step of `1/8192` (or of
/// `1/2048`, a `half` already) rounds to `half` with ties away from zero, as Metal's read and
/// nearest sample of `YCBCR8_420_2P_sRGB`, `YCBCR10_420_2P_sRGB` and three 12-bit and
/// extended-range formats did (every channel), and its gathers, and every step of each grid: the
/// extended-range steps signed within `[-2, 2]`. A value that may be filtered keeps nearest even (a
/// filtered value is the device's).
#[test]
fn a_ycbcr_texel_rounds_its_step_with_ties_away() {
    for (pf, d, lo, hi) in [
        (P::YCBCR84202P, 2048, 0, 2048),
        (P::YCBCR84202PSRGB, 8192, 0, 8192),
        (P::YCBCR124202P, 8192, 0, 8192),
        (P::YCBCR104202PXR, 8192, -16384, 16384),
    ] {
        assert!(
            PackedTexels::of(pf).is_some_and(PackedTexels::ycbcr),
            "{pf:?}"
        );
        let values: Vec<u64> = (lo..=hi)
            .map(|k: i32| u64::from((k as f32 / d as f32).to_bits()))
            .collect();
        let texels = quads(&values);
        let rounded = |mode: HalfRounding| {
            move |_, x: u64| u64::from(half_of(f32::from_bits(x as u32), mode))
        };
        for form in [HalfForm::Read, HalfForm::Sample] {
            let gathers = (0..4).map(TexelLanes::Gathered);
            for lanes in [TexelLanes::Texel].into_iter().chain(gathers) {
                run_lanes(
                    &format!("{pf:?} {lanes:?}"),
                    bits(pf),
                    (form, lanes),
                    &texels,
                    &rounded(HalfRounding::NearestAway),
                );
            }
            run_lanes(
                &format!("{pf:?}"),
                bits(pf),
                (form, TexelLanes::Filtered),
                &texels,
                &rounded(HalfRounding::NearestEven),
            );
        }
        // A step of `1/8192` halfway between two halves rounds away from zero: `4098 / 8192`
        // (between the halves `4096 / 8192` and `4100 / 8192`) to `4100 / 8192`.
        if d == 8192 {
            let away = half_of(4098.0 / 8192.0, HalfRounding::NearestAway);
            assert_eq!(f32::from_bits(away) * 8192.0, 4100.0);
        }
    }
}

/// The device values a packed channel of code scale `scale` reads code `c` as: its exact value,
/// lavapipe's product by the reciprocal, and a 4-, 5- or 6-bit code's bits replicated to 8 (ANV's
/// and NVIDIA's `r / 255`), each as `f32` bits.
fn packed_values(scale: u32, c: u32) -> Vec<u64> {
    let exact = (f64::from(c) / f64::from(scale)) as f32;
    let product = c as f32 * (1.0f32 / scale as f32);
    let mut out = vec![u64::from(exact.to_bits()), u64::from(product.to_bits())];
    let bits = 32 - scale.leading_zeros();
    if (4..=6).contains(&bits) {
        let r = (c << (8 - bits)) | (c >> (2 * bits - 8));
        out.push(u64::from((r as f32 / 255.0).to_bits()));
    }
    out
}

/// Quads of every packed class's codes as the devices read them (`packed_values`), each lane its
/// channel's code, and each lane's code, for a class of `scales`.
fn packed_quads(scales: [u32; 4]) -> (Vec<[u64; 4]>, Vec<[u32; 4]>) {
    let (mut quads, mut codes) = (Vec::new(), Vec::new());
    let most = scales.iter().copied().max().unwrap_or(1);
    for k in 0..=most {
        let code = scales.map(|s| k % (s + 1));
        let values: Vec<Vec<u64>> = (0..4).map(|i| packed_values(scales[i], code[i])).collect();
        let n = values.iter().map(Vec::len).max().unwrap_or(1);
        for v in 0..n {
            quads.push(std::array::from_fn(|i| {
                values[i][v.min(values[i].len() - 1)]
            }));
            codes.push(code);
        }
    }
    (quads, codes)
}

/// A packed format's one texel is each channel's code value `c / (2^b - 1)` correctly rounded to
/// `half`, from whatever value the device reads it as: exactly, by the reciprocal, or replicated
/// to 8 bits; a gather's lanes each its gathered channel's. A value that may be filtered keeps the
/// `OTHER` arm's exact rounding, under the field alone, under a swizzled texture's class (its
/// corner alone), and under a packed normalized format's own class (`EXACT` beside the field, whose
/// grids are per channel), where its one texel is still its code's value. A small float's one texel
/// is its value to nearest even, a YCbCr format's one texel and gather lanes to nearest with ties
/// away from zero.
#[test]
fn a_packed_texel_converts_from_its_codes() {
    let half_of_code = |c: u32, s: u32| match c {
        0 => 0,
        _ => half_bits(f64::from(c) / f64::from(s)),
    };
    let other = |x: u64| u64::from(half_of(f32::from_bits(x as u32), HalfRounding::NearestEven));
    for p in PackedTexels::ALL {
        let class = p.class().0;
        let Some(scales) = p.scales() else {
            let texels = quads(&float_texels());
            for (form, lanes) in [
                (HalfForm::Read, TexelLanes::Texel),
                (HalfForm::Sample, TexelLanes::Texel),
                (HalfForm::Sample, TexelLanes::Filtered),
                (HalfForm::Read, TexelLanes::Gathered(1)),
            ] {
                let away = p.ycbcr() && lanes != TexelLanes::Filtered;
                let mode = match away {
                    true => HalfRounding::NearestAway,
                    false => HalfRounding::NearestEven,
                };
                run_lanes(&format!("{p:?}"), class, (form, lanes), &texels, &|_, x| {
                    u64::from(half_of(f32::from_bits(x as u32), mode))
                });
            }
            continue;
        };
        let (texels, codes) = packed_quads(scales);
        let code_of: std::collections::BTreeMap<[u64; 4], [u32; 4]> =
            texels.iter().copied().zip(codes.iter().copied()).collect();
        for form in [HalfForm::Read, HalfForm::Sample] {
            run_lanes(
                &format!("{p:?}"),
                class,
                (form, TexelLanes::Texel),
                &texels,
                &|i, x| {
                    let quad = texels.iter().find(|q| q[i] == x).expect("a lane's quad");
                    half_of_code(code_of[quad][i], scales[i])
                },
            );
        }
        for ch in 0..4u32 {
            // Every lane a value of channel `ch`'s scale.
            let s = scales[ch as usize];
            let gathered: Vec<[u64; 4]> = (0..=s)
                .flat_map(|c| packed_values(s, c).into_iter().map(move |v| (c, v)))
                .collect::<Vec<_>>()
                .chunks(4)
                .map(|q| std::array::from_fn(|i| q[i.min(q.len() - 1)].1))
                .collect();
            let code: std::collections::BTreeMap<u64, u32> = (0..=s)
                .flat_map(|c| packed_values(s, c).into_iter().map(move |v| (v, c)))
                .collect();
            run_lanes(
                &format!("{p:?} channel {ch}"),
                class,
                (HalfForm::Read, TexelLanes::Gathered(ch)),
                &gathered,
                &|_, x| half_of_code(code[&x], s),
            );
        }
        run_lanes(
            &format!("{p:?}"),
            class,
            (HalfForm::Sample, TexelLanes::Filtered),
            &texels,
            &|_, x| other(x),
        );
        let full = texel_format(match p {
            PackedTexels::Rgb565 => P::B5G6R5Unorm,
            PackedTexels::Rgb5A1 => P::A1BGR5Unorm,
            PackedTexels::Rgba4 => P::ABGR4Unorm,
            PackedTexels::Argb8 => P::ARGB8Unorm,
            _ => P::RGB10A2Unorm,
        });
        // A swizzled texture's class keeps the corner alone (the binder clears the field and a
        // packed normalized family).
        run_lanes(
            &format!("{p:?} swizzled"),
            full.swizzled().0,
            (HalfForm::Read, TexelLanes::Texel),
            &texels,
            &|_, x| other(x),
        );
        // The format's own class: its one texel as the field's, a filtered value the `OTHER`
        // arm's.
        if p.filter_grids().is_none() {
            continue;
        }
        run_lanes(
            &format!("{p:?} class"),
            full.0,
            (HalfForm::Read, TexelLanes::Texel),
            &texels,
            &|i, x| {
                let quad = texels.iter().find(|q| q[i] == x).expect("a lane's quad");
                half_of_code(code_of[quad][i], scales[i])
            },
        );
        run_lanes(
            &format!("{p:?} class filtered"),
            full.0,
            (HalfForm::Sample, TexelLanes::Filtered),
            &texels,
            &|_, x| other(x),
        );
    }
}

/// [`half::ROUTINE_WORDS`] bounds the routine's emission, and [`half::CALL_WORDS`] a call's.
#[test]
fn the_half_texel_routine_words_bound_its_emission() {
    use crate::spirv::serialize::serialize;
    let ins = [v4(), int(32)];
    let size = |body: Body<'_>| -> usize {
        let m = kernel(&ins, v4(), body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    for q in FORMS {
        let without = size(&|fb, x| {
            fb.module().set_quantize_form(q);
            Ok(x[0])
        });
        let defined = size(&|fb, x| {
            fb.module().set_quantize_form(q);
            half::define(fb.module(), HalfFamilies::Every)?;
            Ok(x[0])
        }) - without;
        // The routine of a device whose texel is Metal's tells apart fewer families, so it is
        // shorter wherever it differs (the `Integer` form tells apart every one).
        let metal = size(&|fb, x| {
            fb.module().set_quantize_form(q);
            half::define(fb.module(), HalfFamilies::MetalSampler)?;
            Ok(x[0])
        }) - without;
        match q {
            QuantizeForm::Integer => assert_eq!(metal, defined, "{q:?}"),
            QuantizeForm::Device | QuantizeForm::Converted | QuantizeForm::Controlled => assert!(
                metal < defined,
                "{q:?}: the Metal sampler's routine is {metal} words, every family's {defined}"
            ),
        }
        eprintln!("half texel routine {q:?}: Metal sampler's {metal} words");
        let called =
            size(&body_lanes(HalfForm::Sample, TexelLanes::Filtered, q)) - without - defined;
        assert!(
            defined as u64 <= half::ROUTINE_WORDS,
            "{q:?}: the routine is {defined} words, bound {}",
            half::ROUTINE_WORDS
        );
        assert!(
            called <= half::CALL_WORDS as usize,
            "{q:?}: a call is {called} words, bound {}",
            half::CALL_WORDS
        );
        eprintln!("half texel routine {q:?}: {defined} words, a call {called}");
    }
}

/// On every grid the routine divides by multiplying (the 8-bit grids and the packed code scales),
/// the product of every step `m` by the reciprocal `RN(1 / D)` rounds to the `half` of `m / D` (to
/// nearest even) from both of its `f32` neighbours, so the product may round either way; on the
/// 16-bit grids it does not (`texel_round::ones_ratio_to_half` takes them). Exact in `f64`: the
/// product of a 24-bit reciprocal and a step below `2^16` has at most 40 bits.
#[test]
fn the_reciprocal_product_rounds_like_the_quotient() {
    let half = |v: f64| f16_to_f64(f64_to_f16(v));
    let neighbours = |p: f64| {
        let r = p as f32;
        let (lo, hi) = match f64::from(r).partial_cmp(&p) {
            Some(std::cmp::Ordering::Less) => (r, f32::from_bits(r.to_bits() + 1)),
            Some(std::cmp::Ordering::Greater) => (f32::from_bits(r.to_bits() - 1), r),
            _ => (r, r),
        };
        [f64::from(lo), f64::from(hi)]
    };
    let misses = |d: u32| {
        let c = f64::from(1.0f32 / d as f32);
        (0..=d)
            .filter(|m| {
                let want = half(f64::from(*m) / f64::from(d));
                neighbours(f64::from(*m) * c)
                    .iter()
                    .any(|v| half(*v) != want)
            })
            .count()
    };
    let mut grids = vec![C::GRID8, C::GRID8S];
    grids.extend(
        PackedTexels::ALL
            .iter()
            .filter_map(|p| p.scales())
            .flatten(),
    );
    for d in grids {
        assert_eq!(misses(d), 0, "the product by 1 / {d}");
    }
    assert!(misses(C::GRID16) > 0 && misses(C::GRID16S) > 0);
}

/// On a device that samples through Metal's own texture unit, an 8-bit sRGB format's `f32` texel is
/// Metal's `float` value, a step `n / D` rounded to nearest even on the grids `1/4095` (a color
/// channel's table step, or a linear sum rounded onto it), `1/4080` (alpha's linear sum), and
/// `1/255` (alpha's code), and the conversion's `OTHER` arm rounds that to `half` to nearest even.
/// For every step it is the step's own rounding to `half`, Metal's `half` value, so that class's
/// conversion is Metal's there (`lower::texture::unconverted_class`). Exact: integer arithmetic on
/// the step and on the `f32` value.
#[test]
fn metals_srgb_float_steps_round_to_their_half_steps() {
    for d in [4095u32, 4080, 255] {
        let misses = steps_rounding_twice(d);
        assert!(misses.is_empty(), "the steps n / {d} at n in {misses:?}");
    }
}

/// `p / q` (in `[0, 1]`) rounded to `half` to nearest even: the value `k · 2^x` as `(k, x)` with
/// `k` odd (or zero), so that equal values are equal pairs.
fn half_of_ratio(p: u128, q: u128) -> (u128, i32) {
    if p == 0 {
        return (0, 0);
    }
    // The exponent `e` of `p / q` (`2^e <= p / q < 2^(e + 1)`), at least half's least normal
    // one, -14.
    let mut e = 0i32;
    while e > -14 && (p << e.unsigned_abs()) < q {
        e -= 1;
    }
    // `p / q` in units of `2^(e - 10)`, rounded to nearest even.
    let scaled = p << (10 - e).unsigned_abs();
    let (mut k, r) = (scaled / q, scaled % q);
    if 2 * r > q || (2 * r == q && k % 2 == 1) {
        k += 1;
    }
    let mut x = e - 10;
    while k % 2 == 0 {
        k /= 2;
        x += 1;
    }
    (k, x)
}

/// The `f32` value `n / d` rounded to nearest even (IEEE division of two exact operands), as the
/// ratio of its significand to a power of two.
fn f32_step(n: u32, d: u32) -> (u128, u128) {
    let bits = (n as f32 / d as f32).to_bits();
    if bits == 0 {
        return (0, 1);
    }
    let significand = u128::from(bits & 0x7f_ffff | 0x80_0000);
    let exponent = ((bits >> 23) & 0xff) as i32 - 150;
    assert!(exponent < 0, "{n} / {d}");
    (significand, 1u128 << exponent.unsigned_abs())
}

/// The steps `n` of grid `1/d` (`0 <= n <= d`) whose `f32` value rounds to another `half` than
/// the step itself: exact, by integer arithmetic on the step and on the `f32` value.
fn steps_rounding_twice(d: u32) -> Vec<u32> {
    (0..=d)
        .filter(|&n| {
            let (p, q) = f32_step(n, d);
            half_of_ratio(u128::from(n), u128::from(d)) != half_of_ratio(p, q)
        })
        .collect()
}

/// Metal's `float` texel of every grid whose `half` the routine of a device that samples through
/// Metal's own texture unit leaves to the `OTHER` arm is a step `n / D` rounded to `f32` to nearest
/// even, and that value's rounding to `half` is the step's own, Metal's `half`, for every step: the
/// 8-bit normalized codes (`1/255`, `1/127`) and filter grids (`1/4080`, `1/2032`), every packed
/// code scale and filter grid, the sRGB table's `1/4095`, and the YCbCr grids `1/2048` and `1/8192`
/// (whose ties the routine still breaks away from zero on `8192`). The 16-bit grids are not: `65455
/// / 65535`, `65519 / 65535` and `32759 / 32767` round twice, so the routine keeps their exact
/// `half` (our probes differ at exactly those steps). A signed step's value is
/// the negation of its magnitude's, so the magnitudes cover both signs. Exact: integer arithmetic
/// on the step and on the `f32` value.
#[test]
fn metal_float_steps_round_to_their_half() {
    let mut grids = vec![255u32, 127, C::GRID8, C::GRID8S, 4095, 2048, 8192];
    for p in PackedTexels::ALL {
        grids.extend(p.scales().into_iter().flatten());
        grids.extend(p.filter_grids().into_iter().flatten());
    }
    grids.sort_unstable();
    grids.dedup();
    for d in grids {
        let misses = steps_rounding_twice(d);
        assert!(misses.is_empty(), "the steps n / {d} at n in {misses:?}");
    }
    assert_eq!(steps_rounding_twice(C::GRID16), vec![65455, 65519]);
    assert_eq!(steps_rounding_twice(C::GRID16S), vec![32759]);
}

/// On a device that samples through Metal's own texture unit, the routine that tells apart only the
/// families whose `half` is not Metal's `float` texel rounded to nearest even answers as the
/// routine of every family on Metal's texels, for every class: the normalized families' codes and
/// grid steps, every packed field's codes and filter steps, the half-float and small float
/// families' halves, the 8-bit sRGB table's steps, the YCbCr fields' steps, and any value of a
/// 32-bit float class and of `OTHER`; in every access form and lanes, in both roundings it takes
/// (`Device`, `Converted`). Off Metal's grids (another device's filter) the two differ: the routine
/// of every family snaps an 8-bit value onto its grid first.
#[test]
fn the_metal_sampler_routine_answers_as_every_family_on_metal_s_texels() {
    let mut rng = Rng(0x00b1_f0b1_f5ee_d643);
    let f32_bits = |v: f64| u64::from((v as f32).to_bits());
    // A class's texels in lane order (lane `i` its channel `i`, or for a gather of channel `c`
    // that channel of four texels), as Metal's `float` values, and the access forms that read
    // them.
    #[derive(Clone, Copy)]
    enum Reach {
        /// Any access: every channel on one grid, or values exact in `half`.
        Any,
        /// One texel's channels: a read, a sample of one texel, or a filter's sum of one.
        Texel,
        /// Channel `c` of four texels: a gather of it.
        Gather(u32),
        /// A linear filter's sums: a filtered sample.
        Filtered,
    }
    impl Reach {
        fn reaches(self, lanes: TexelLanes) -> bool {
            match self {
                Reach::Any => true,
                Reach::Texel => !matches!(lanes, TexelLanes::Gathered(_)),
                Reach::Gather(c) => lanes == TexelLanes::Gathered(c),
                Reach::Filtered => lanes == TexelLanes::Filtered,
            }
        }
    }
    let gathered = 2;
    let mut classes: Vec<(String, u32, Vec<u64>, Reach)> = Vec::new();
    for f in norms() {
        let every = f.d <= 4096;
        // The 8-bit families' every code and step; a sample of the 16-bit ones', whose arm the
        // two routines share.
        let codes = f.codes();
        let codes = codes.iter().step_by(if every { 1 } else { 13 });
        let mut texels: Vec<u64> = codes.map(|c| f.value(*c)).collect();
        let lo = if f.signed { -f.d } else { 0 };
        let steps: Vec<i64> = match every {
            true => (lo..=f.d).collect(),
            false => (0..1500)
                .map(|_| lo + (rng.next() % (f.d - lo + 1) as u64) as i64)
                .chain([65455, 65519, 32759, -32759, lo, f.d])
                .filter(|n| (lo..=f.d).contains(n))
                .collect(),
        };
        texels.extend(steps.iter().map(|n| f.on_grid(*n)));
        classes.push((format!("{f:?}"), f.class, texels, Reach::Any));
    }
    let packed = [
        P::B5G6R5Unorm,
        P::A1BGR5Unorm,
        P::ABGR4Unorm,
        P::BGR5A1Unorm,
        P::RGB10A2Unorm,
        P::BGR10A2Unorm,
        P::ARGB8Unorm,
    ];
    for pf in packed {
        let field = PackedTexels::of(pf).expect("a packed format");
        let scales = field.scales().expect("code scales");
        let mut code = |s: u32| f32_bits((rng.next() % u64::from(s + 1)) as f64 / f64::from(s));
        let texels = (0..600).flat_map(|_| scales.map(&mut code)).collect();
        classes.push((format!("{pf:?}"), bits(pf), texels, Reach::Texel));
        let s = scales[gathered as usize];
        let texels = (0..2400).map(|_| code(s)).collect();
        let name = format!("{pf:?} channel {gathered}");
        classes.push((name, bits(pf), texels, Reach::Gather(gathered)));
        // A linear filter's sums lie on the field's filter grids, which only a filtered sample
        // reaches: a read, a gather and a single texel are the field's codes.
        if let Some(grids) = field.filter_grids() {
            let mut sums = Vec::new();
            for _ in 0..600 {
                for g in grids {
                    let n = rng.next() % u64::from(g + 1);
                    sums.push(f32_bits(n as f64 / f64::from(g)));
                }
            }
            classes.push((format!("{pf:?} sums"), bits(pf), sums, Reach::Filtered));
        }
    }
    // Every finite `half` of either sign, as a half-float or small-float texel.
    let halves: Vec<u64> = (0u32..0x7C00)
        .step_by(7)
        .flat_map(|h| [h, h | 0x8000])
        .map(|h| u64::from((f16_to_f64(h as u16) as f32).to_bits()))
        .collect();
    for pf in [P::RGBA16Float, P::RGB9E5Float] {
        classes.push((format!("{pf:?}"), bits(pf), halves.clone(), Reach::Any));
    }
    // An 8-bit sRGB format's decoded steps (its color channels'; alpha's codes are an 8-bit
    // normalized format's), and its filter's sums on the 8-bit grid.
    let srgb = bits(P::RGBA8UnormSRGB);
    let steps: Vec<u64> = (0..=4095u32)
        .map(|k| f32_bits(f64::from(k) / 4095.0))
        .collect();
    let texels = steps
        .chunks(3)
        .zip((0..=255u32).cycle())
        .flat_map(|(rgb, a)| {
            let mut t = [0; 4];
            t[..rgb.len()].copy_from_slice(rgb);
            t[3] = f32_bits(f64::from(a) / 255.0);
            t
        })
        .collect();
    classes.push(("RGBA8Unorm_sRGB".into(), srgb, texels, Reach::Texel));
    let name = format!("RGBA8Unorm_sRGB channel {gathered}");
    classes.push((name, srgb, steps, Reach::Gather(gathered)));
    let sums = (0..=C::GRID8)
        .map(|n| f32_bits(f64::from(n) / f64::from(C::GRID8)))
        .collect();
    classes.push(("RGBA8Unorm_sRGB sums".into(), srgb, sums, Reach::Filtered));
    for p in PackedTexels::ALL.into_iter().filter(|p| p.ycbcr()) {
        let pf = P::ALL
            .iter()
            .copied()
            .find(|&pf| PackedTexels::of(pf) == Some(p))
            .expect("a format of the field");
        let d: i64 = match p {
            PackedTexels::YCbCr2048 => 2048,
            _ => 8192,
        };
        let lo = if p.filter_extended() { -2 * d } else { 0 };
        let hi = if p.filter_extended() { 2 * d } else { d };
        let texels = (lo..=hi)
            .step_by(((hi - lo) / 2000).max(1) as usize)
            .chain([lo, hi, 1, -1, 4097, 4099, -4097])
            .filter(|k| (lo..=hi).contains(k))
            .map(|k| f32_bits(k as f64 / d as f64))
            .collect();
        classes.push((format!("{pf:?}"), bits(pf), texels, Reach::Any));
    }
    let float = bits(P::RGBA32Float);
    classes.push(("RGBA32Float".into(), float, float_texels(), Reach::Any));
    classes.push(("OTHER".into(), C::OTHER.0, float_texels(), Reach::Any));
    let ins = [v4(), int(32)];
    let same = |g: &Vec<u64>, w: &Vec<u64>| {
        g.len() == w.len()
            && g.iter().zip(w).all(|(g, w)| {
                g == w || (f32::from_bits(*g as u32).is_nan() && f32::from_bits(*w as u32).is_nan())
            })
    };
    let forms = [
        (HalfForm::Read, TexelLanes::Texel),
        (HalfForm::Read, TexelLanes::Gathered(2)),
        (HalfForm::Sample, TexelLanes::Texel),
        (HalfForm::Sample, TexelLanes::Filtered),
    ];
    for q in [
        QuantizeForm::Device,
        QuantizeForm::Converted,
        QuantizeForm::Controlled,
    ] {
        for (form, lanes) in forms {
            let build = |families| {
                kernel_with(
                    &ins,
                    v4(),
                    &body_families(form, lanes, q, families),
                    true,
                    false,
                )
                .expect("build")
            };
            let (every, metal) = (
                build(HalfFamilies::Every),
                build(HalfFamilies::MetalSampler),
            );
            let reached = classes.iter().filter(|c| c.3.reaches(lanes));
            for (name, class, texels, _) in reached {
                let what = format!("{name} {form:?} {lanes:?} {q:?}");
                let cases: Vec<Case> = quads(texels)
                    .iter()
                    .map(|t| vec![t.to_vec(), vec![u64::from(*class)]])
                    .collect();
                let want = run_module(&what, every.clone(), &ins, v4(), &cases);
                let got = run_module(&what, metal.clone(), &ins, v4(), &cases);
                let mut differ = Vec::new();
                for (i, c) in cases.iter().enumerate() {
                    if !same(&got[i], &want[i]) {
                        differ.push(format!(
                            "{:x?}: got {:x?}, want {:x?}",
                            c[0], got[i], want[i]
                        ));
                    }
                }
                differ.truncate(8);
                assert!(differ.is_empty(), "{what}:\n{}", differ.join("\n"));
            }
        }
    }
    // Off the grid, the routine of every family snaps an 8-bit value onto it first: below the
    // tie `0.5 + 2^-12` between two halves, the nearest step `2041 / 4080` lies above it.
    let class = bits(P::RGBA8Unorm);
    let off: Vec<u64> = std::iter::once(f32_bits(0.500_24))
        .chain((0..1023).map(|_| f32_bits((rng.next() % 1_000_001) as f64 / 1e6)))
        .collect();
    let cases: Vec<Case> = quads(&off)
        .iter()
        .map(|t| vec![t.to_vec(), vec![u64::from(class)]])
        .collect();
    let q = QuantizeForm::Converted;
    let lanes = (HalfForm::Sample, TexelLanes::Filtered);
    let build = |families| {
        kernel_with(
            &ins,
            v4(),
            &body_families(lanes.0, lanes.1, q, families),
            true,
            false,
        )
        .expect("build")
    };
    let want = run_module("off grid", build(HalfFamilies::Every), &ins, v4(), &cases);
    let got = run_module(
        "off grid",
        build(HalfFamilies::MetalSampler),
        &ins,
        v4(),
        &cases,
    );
    assert!(
        !same(&got[0], &want[0]),
        "off grid: {:x?} {:x?}",
        got[0],
        want[0]
    );
}

/// The values of a 16-bit normalized class the two forms of its `half` are compared on: every step
/// `n / D` of its grid `D`, both signs, and values off it of every kind: random `f32` patterns of
/// every magnitude, random values within and past `±1`, values whose step is `-0` or `+0`, and the
/// specials.
fn grid16_values(f: Norm, rng: &mut Rng) -> Vec<u64> {
    let mut out: Vec<u64> = (-f.d..=f.d).map(|n| f.on_grid(n)).collect();
    out.extend((0..4000).map(|_| rng.next() & 0xFFFF_FFFF));
    out.extend((0..4000).map(|_| {
        let v = (rng.next() % 2_400_001) as f64 / 1e6 - 1.2;
        u64::from((v as f32).to_bits())
    }));
    let tiny = 0.5 / f.d as f32;
    out.extend(
        [
            0.0f32,
            -0.0,
            tiny,
            -tiny,
            tiny * 0.999,
            -tiny * 0.999,
            1e-30,
            -1e-30,
            1.0,
            -1.0,
            1.0 + tiny,
            -1.0 - tiny,
            2.0,
            -2.0,
            f32::MAX,
            f32::MIN,
        ]
        .map(|v| u64::from(v.to_bits())),
    );
    out.extend([
        0x0000_0001,
        0x8000_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0xFFC0_0000,
        0x7F80_0001,
        0xFFBF_FFFF,
    ]);
    out
}

/// The routine of a device that samples through Metal's own texture unit converts a 16-bit
/// normalized texel by one product (`grid16_by_product`, its factor `texel_round::ones_ratio_factor`),
/// where the routine of every family takes integer arithmetic alone
/// (`texel_round::ones_ratio_to_half`); the two answer alike, bit for bit, on every value of
/// [`grid16_values`] (every step of both grids, both signs, and values off the grids), in both
/// roundings the routine takes and in a read's and a filtered sample's forms. Under native half
/// rounding it keeps the integer form, which answers alike too.
fn grid16_forms_agree(part: usize, of: usize) {
    let mut rng = Rng(0x00b1_f0b1_f5ee_d016);
    let ins = [v4(), int(32)];
    let grids: Vec<Norm> = norms().into_iter().filter(|f| f.d > 4096).collect();
    assert_eq!(grids.len(), 2, "the 16-bit grids");
    let forms = [
        (HalfForm::Read, TexelLanes::Texel),
        (HalfForm::Sample, TexelLanes::Filtered),
    ];
    for f in grids {
        let cases: Vec<Case> = quads(&grid16_values(f, &mut rng))
            .iter()
            .map(|t| vec![t.to_vec(), vec![u64::from(f.class)]])
            .collect();
        let cases = part_of(cases, part, of);
        for q in [
            QuantizeForm::Device,
            QuantizeForm::Converted,
            QuantizeForm::Controlled,
        ] {
            for (form, lanes) in forms {
                let build = |families, native: bool| {
                    let body = body_families(form, lanes, q, families);
                    kernel_with(
                        &ins,
                        v4(),
                        &move |fb: &mut FunctionBuilder<'_>, x: &[Value]| {
                            fb.module().set_native_half(native);
                            body(fb, x)
                        },
                        true,
                        false,
                    )
                    .expect("build")
                };
                let what = format!("{f:?} {form:?} {lanes:?} {q:?}");
                let want = run_module(&what, build(HalfFamilies::Every, false), &ins, v4(), &cases);
                for native in [false, true] {
                    let got = run_module(
                        &what,
                        build(HalfFamilies::MetalSampler, native),
                        &ins,
                        v4(),
                        &cases,
                    );
                    let differ: Vec<String> = cases
                        .iter()
                        .zip(got.iter().zip(&want))
                        .filter(|(_, (g, w))| g != w)
                        .take(8)
                        .map(|(c, (g, w))| format!("{:x?}: got {g:x?}, want {w:x?}", c[0]))
                        .collect();
                    assert!(
                        differ.is_empty(),
                        "{what} native {native}:\n{}",
                        differ.join("\n")
                    );
                }
            }
        }
    }
}

parts!(grid16_forms_agree, 8;
    the_16_bit_grids_convert_by_one_product_as_by_integers_0 = 0,
    the_16_bit_grids_convert_by_one_product_as_by_integers_1 = 1,
    the_16_bit_grids_convert_by_one_product_as_by_integers_2 = 2,
    the_16_bit_grids_convert_by_one_product_as_by_integers_3 = 3,
    the_16_bit_grids_convert_by_one_product_as_by_integers_4 = 4,
    the_16_bit_grids_convert_by_one_product_as_by_integers_5 = 5,
    the_16_bit_grids_convert_by_one_product_as_by_integers_6 = 6,
    the_16_bit_grids_convert_by_one_product_as_by_integers_7 = 7,
);

/// The grid families' `half` comes from integer arithmetic, selected after the rounding of the
/// texel itself, so every rounding the routine leaves to the device (the `Converted` form's
/// barrier, a `NoContraction` `x + -0.0`, and the `Device` form's `OpQuantizeToF16`) reads the
/// routine's texel parameter directly: where a device specializes a grid class, the selection
/// drops it whole, and no `[[clang::optnone]]` call is left at the access on MoltenVK.
#[test]
fn the_device_rounding_reads_the_texel_alone() {
    use crate::spirv::serialize::serialize;
    use std::collections::BTreeSet;
    // SPIR-V opcodes and the `NoContraction` decoration.
    const OP_DECORATE: u32 = 71;
    const OP_FUNCTION: u32 = 54;
    const OP_FUNCTION_PARAMETER: u32 = 55;
    const OP_FUNCTION_END: u32 = 56;
    const OP_QUANTIZE_TO_F16: u32 = 116;
    const OP_FADD: u32 = 129;
    const NO_CONTRACTION: u32 = 42;
    for q in [QuantizeForm::Device, QuantizeForm::Converted] {
        let m = kernel(&[v4(), int(32)], v4(), &|fb, x| {
            fb.module().set_quantize_form(q);
            half::define(fb.module(), HalfFamilies::Every)?;
            Ok(x[0])
        })
        .expect("build");
        let words = serialize(m).expect("serialize").words;
        let mut insts = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let n = ((words[i] >> 16) as usize).max(1);
            insts.push(&words[i..(i + n).min(words.len())]);
            i += n;
        }
        let decorated: BTreeSet<u32> = insts
            .iter()
            .filter(|w| w[0] & 0xFFFF == OP_DECORATE && w.get(2) == Some(&NO_CONTRACTION))
            .map(|w| w[1])
            .collect();
        // The routine is the module's function whose first parameter is a `vec4` and that has
        // four parameters; its texel is that first parameter.
        let mut texel = None;
        let mut params = Vec::new();
        let mut roundings = Vec::new();
        let mut body = Vec::new();
        for w in &insts {
            match w[0] & 0xFFFF {
                OP_FUNCTION => {
                    params.clear();
                    body.clear();
                }
                OP_FUNCTION_PARAMETER => params.push(w[2]),
                OP_FUNCTION_END => {
                    if params.len() == 4 {
                        texel = Some(params[0]);
                        roundings = body.clone();
                    }
                }
                OP_FADD if decorated.contains(&w[2]) => body.push((w[2], w[3])),
                OP_QUANTIZE_TO_F16 => body.push((w[2], w[3])),
                _ => {}
            }
        }
        let texel = texel.expect("the routine");
        assert!(!roundings.is_empty(), "{q:?}: the routine rounds the texel");
        for (result, operand) in roundings {
            assert_eq!(
                operand, texel,
                "{q:?}: the device's rounding %{result} reads %{operand}, not the texel"
            );
        }
    }
}
