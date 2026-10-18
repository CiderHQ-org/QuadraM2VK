//! Interpreter differential tests of the `float` value of one texel by the bound format's class
//! (`lower::texture::float_texel`). References are plain Rust: Metal's measured answer, a code's
//! value `c / (2^b - 1)` correctly rounded to `f32`, from each value a device reads the code as
//! (exactly, by the reciprocal, or a short channel's bits replicated to 8).

use super::*;
use crate::lower::texture::float_texel;
use crate::lower::texture::half::TexelLanes;
use crate::metal::texel_format::{texel_format, PackedTexels, TexelFormat as C};
use crate::metal::PixelFormat as P;

fn v4() -> Form {
    vec(float(32), 4)
}

/// The routine at class `class` (an input) and lanes `lanes`.
fn body(lanes: TexelLanes) -> impl Fn(&mut FunctionBuilder<'_>, &[Value]) -> BResult<Value> {
    move |fb, x| {
        let r = float_texel::define(fb.module())?;
        let lanes = lanes.value(fb)?;
        float_texel::convert(fb, r, x[0], x[1], lanes)
    }
}

/// Run the routine over `texels` (four lanes each, `f32` bits) at `class` and `lanes`, against
/// `reference` of each lane's index and value, bit for bit (a NaN's payload included).
fn run(
    what: &str,
    class: u32,
    lanes: TexelLanes,
    texels: &[[u64; 4]],
    reference: &dyn Fn(usize, u64) -> u64,
) {
    let cases: Vec<Case> = texels
        .iter()
        .map(|t| vec![t.to_vec(), vec![u64::from(class)]])
        .collect();
    check_accept(
        &format!("{what} {lanes:?}"),
        &[v4(), int(32)],
        v4(),
        &body(lanes),
        &cases,
        &|c| {
            Some(
                c[0].iter()
                    .enumerate()
                    .map(|(i, x)| reference(i, *x))
                    .collect(),
            )
        },
        &|g, w| g == w,
    );
}

/// `x` in groups of four lanes, the last padded with its last value.
fn quads(x: &[u64]) -> Vec<[u64; 4]> {
    x.chunks(4)
        .map(|c| std::array::from_fn(|i| c[i.min(c.len() - 1)]))
        .collect()
}

/// `f32` bits of `c / s` correctly rounded (`f64`'s correctly rounded quotient, whose second
/// rounding cannot meet an `f32` tie: `s` is odd, so `c / s` is dyadic only where it is an
/// integer), `+0` for zero.
fn ratio_bits(c: i64, s: i64) -> u64 {
    u64::from(((c as f64 / s as f64) as f32).to_bits())
}

/// The values a device reads code `c` of code scale `s` (signed where `signed`) as: its exact
/// value, the product by the reciprocal (lavapipe's), and for a 4-, 5- or 6-bit code its bits
/// replicated to 8 (ANV's and NVIDIA's `r / 255`); the least signed code reads `-1`.
fn device_values(c: i64, s: i64, signed: bool) -> Vec<u64> {
    let c = if signed { c.max(-s) } else { c };
    let exact = ratio_bits(c, s);
    let product = u64::from((c as f32 * (1.0f32 / s as f32)).to_bits());
    let mut out = vec![exact, product];
    let bits = 64 - s.leading_zeros() as i64;
    if !signed && (4..=6).contains(&bits) {
        let r = (c << (8 - bits)) | (c >> (2 * bits - 8));
        out.push(u64::from((r as f32 / 255.0).to_bits()));
    }
    out
}

/// A normalized family's one texel, a read's or a gather's, is its code's value correctly
/// rounded to `f32` from the value the device reads (lavapipe's reciprocal one ulp off included),
/// `-1` for the least signed code; a value that may be filtered is the device's.
#[test]
fn a_normalized_texel_is_its_codes_value() {
    for (pf, s, signed) in [
        (P::RGBA8Unorm, 255i64, false),
        (P::RGBA8Snorm, 127, true),
        (P::RGBA16Unorm, 65535, false),
        (P::RGBA16Snorm, 32767, true),
    ] {
        let class = texel_format(pf).0;
        let (lo, hi) = if signed { (-s - 1, s) } else { (0, s) };
        let step = if hi > 255 { 13 } else { 1 };
        let codes: Vec<i64> = (lo..=hi)
            .step_by(step)
            .chain([lo, lo + 1, -1, 0, 1, hi - 1, hi])
            .filter(|c| (lo..=hi).contains(c))
            .collect();
        let pairs: Vec<(u64, i64)> = codes
            .iter()
            .flat_map(|&c| {
                device_values(c, s, signed)
                    .into_iter()
                    .map(move |v| (v, c.max(-s)))
            })
            .collect();
        let code: std::collections::BTreeMap<u64, i64> = pairs.iter().copied().collect();
        let texels = quads(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
        let want = |_: usize, x: u64| ratio_bits(code[&x], s);
        run(&format!("{pf:?}"), class, TexelLanes::Texel, &texels, &want);
        run(
            &format!("{pf:?}"),
            class,
            TexelLanes::Gathered(2),
            &texels,
            &want,
        );
        run(
            &format!("{pf:?}"),
            class,
            TexelLanes::Filtered,
            &texels,
            &|_, x| x,
        );
    }
}

/// A packed format's one texel is each channel's code value `c / (2^b - 1)` correctly rounded to
/// `f32`, from whatever value the device reads it as; a gather's lanes each the gathered
/// channel's; a value that may be filtered, and a swizzled texture's class (no packed field), the
/// device's.
#[test]
fn a_packed_texel_is_its_codes_value() {
    for p in PackedTexels::ALL {
        let Some(scales) = p.scales() else {
            continue;
        };
        let class = p.class().0;
        let most = scales.iter().copied().max().unwrap_or(1);
        let mut texels = Vec::new();
        let mut codes = Vec::new();
        for k in 0..=most {
            let code = scales.map(|s| k % (s + 1));
            let values: Vec<Vec<u64>> = (0..4)
                .map(|i| device_values(i64::from(code[i]), i64::from(scales[i]), false))
                .collect();
            let n = values.iter().map(Vec::len).max().unwrap_or(1);
            for v in 0..n {
                texels.push(std::array::from_fn::<u64, 4, _>(|i| {
                    values[i][v.min(values[i].len() - 1)]
                }));
                codes.push(code);
            }
        }
        // Each lane's value names its code on that lane's scale.
        let lane_code: Vec<std::collections::BTreeMap<u64, u32>> = (0..4)
            .map(|i| {
                texels
                    .iter()
                    .zip(&codes)
                    .map(|(t, c)| (t[i], c[i]))
                    .collect()
            })
            .collect();
        run(
            &format!("{p:?}"),
            class,
            TexelLanes::Texel,
            &texels,
            &|i, x| ratio_bits(i64::from(lane_code[i][&x]), i64::from(scales[i])),
        );
        for (ch, s) in (0u32..).zip(scales) {
            let pairs: Vec<(u64, u32)> = (0..=s)
                .flat_map(|c| {
                    device_values(i64::from(c), i64::from(s), false)
                        .into_iter()
                        .map(move |v| (v, c))
                })
                .collect();
            let code: std::collections::BTreeMap<u64, u32> = pairs.iter().copied().collect();
            run(
                &format!("{p:?} channel {ch}"),
                class,
                TexelLanes::Gathered(ch),
                &quads(&pairs.iter().map(|p| p.0).collect::<Vec<_>>()),
                &|_, x| ratio_bits(i64::from(code[&x]), i64::from(s)),
            );
        }
        run(
            &format!("{p:?}"),
            class,
            TexelLanes::Filtered,
            &texels,
            &|_, x| x,
        );
        run(
            &format!("{p:?} swizzled"),
            class & !C::PACKED_MASK,
            TexelLanes::Texel,
            &texels,
            &|_, x| x,
        );
        // A packed normalized format's own class (its `EXACT` family beside the field) reads
        // alike: its one texel the code's value, a filtered value the device's.
        let full = match p {
            PackedTexels::Rgb565 => P::B5G6R5Unorm,
            PackedTexels::Rgb5A1 => P::BGR5A1Unorm,
            PackedTexels::Rgba4 => P::ABGR4Unorm,
            PackedTexels::Rgb10A2 => P::BGR10A2Unorm,
            PackedTexels::SmallFloat
            | PackedTexels::Argb8
            | PackedTexels::YCbCr2048
            | PackedTexels::YCbCr8192
            | PackedTexels::YCbCrExtended8192 => continue,
        };
        let full = texel_format(full);
        assert!(full.packed_exact(), "{p:?}");
        run(
            &format!("{p:?} class"),
            full.0,
            TexelLanes::Texel,
            &texels,
            &|i, x| ratio_bits(i64::from(lane_code[i][&x]), i64::from(scales[i])),
        );
        run(
            &format!("{p:?} class"),
            full.0,
            TexelLanes::Filtered,
            &texels,
            &|_, x| x,
        );
    }
}

/// A small or half float's one texel is the device's value, a NaN Metal's `0x7fc00000`
/// (`RG11B10Float`'s rows; a half-float texel's, whatever the device's widening
/// kept of its sign and payload); a value that may be filtered keeps its payload. Any class of
/// neither a normalized family, a packed field nor the half-float family (32-bit floats, sRGB, a
/// corner alone, `OTHER`) is the device's value, bit for bit.
#[test]
fn a_small_or_half_float_reads_metals_nan_and_other_classes_the_devices_value() {
    let mut rng = Rng(0x00f1_0a7e_7e1e_0001);
    let mut values: Vec<u64> = (0..4000).map(|_| rng.next() & 0xFFFF_FFFF).collect();
    values.extend([
        0x7fa8_0000u64,
        0x7fd6_0000,
        0xffc0_0001,
        0x7f80_0001,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0001,
        0x8000_0000,
        0x3f80_0000,
    ]);
    let texels = quads(&values);
    let small = PackedTexels::SmallFloat.class().0;
    let nan = |x: u64| f32::from_bits(x as u32).is_nan();
    run("small float", small, TexelLanes::Texel, &texels, &|_, x| {
        if nan(x) {
            0x7fc0_0000
        } else {
            x
        }
    });
    run(
        "small float",
        small,
        TexelLanes::Gathered(0),
        &texels,
        &|_, x| if nan(x) { 0x7fc0_0000 } else { x },
    );
    run(
        "small float",
        small,
        TexelLanes::Filtered,
        &texels,
        &|_, x| x,
    );
    // The half floats, and a swizzled small float (its class keeps the family alone).
    let swizzled = texel_format(P::RG11B10Float).swizzled();
    assert_eq!(swizzled.0 & C::PACKED_MASK, 0);
    let halves = [P::R16Float, P::RG16Float, P::RGBA16Float, P::RGBX16Float]
        .map(|pf| (format!("{pf:?}"), texel_format(pf).0))
        .into_iter()
        .chain([("swizzled RG11B10Float".to_owned(), swizzled.0)]);
    for (what, class) in halves {
        for lanes in [
            TexelLanes::Texel,
            TexelLanes::Gathered(0),
            TexelLanes::Gathered(3),
        ] {
            run(&what, class, lanes, &texels, &|_, x| {
                if nan(x) {
                    0x7fc0_0000
                } else {
                    x
                }
            });
        }
        run(&what, class, TexelLanes::Filtered, &texels, &|_, x| x);
    }
    for pf in [
        P::RGBA32Float,
        P::R32Float,
        P::RGBA8UnormSRGB,
        P::BC7RGBAUnorm,
        P::RGBA8Uint,
    ] {
        let class = texel_format(pf).0;
        run(
            &format!("{pf:?}"),
            class,
            TexelLanes::Texel,
            &texels,
            &|_, x| x,
        );
    }
}

/// A YCbCr format's one texel is the device's value, bit for bit: its view holds Metal's read
/// `k / D` exactly, and its class's `EXACT` family sums on its field's grid, never on a normalized
/// family's (whose `1/4080` step would read `k = 1` of `1/2048` as `2/4080`): each step of the grid
/// `2048`, of `8192`, and of the signed `8192` within `[-2, 2]`.
#[test]
fn a_ycbcr_texel_is_the_devices_value() {
    for (pf, d, lo, hi) in [
        (P::YCBCR84202P, 2048, 0, 2048),
        (P::YCBCR124202P, 8192, 0, 8192),
        (P::YCBCR104202PXR, 8192, -16384, 16384),
    ] {
        let class = texel_format(pf);
        assert!(
            PackedTexels::of(pf).is_some_and(PackedTexels::ycbcr),
            "{pf:?}"
        );
        let values: Vec<u64> = (lo..=hi)
            .map(|k: i32| u64::from((k as f32 / d as f32).to_bits()))
            .collect();
        let texels = quads(&values);
        for lanes in [
            TexelLanes::Texel,
            TexelLanes::Gathered(1),
            TexelLanes::Filtered,
        ] {
            run(&format!("{pf:?}"), class.0, lanes, &texels, &|_, x| x);
        }
    }
}

/// [`float_texel::ROUTINE_WORDS`] bounds the routine's emission, and
/// [`float_texel::CALL_WORDS`] a call's.
#[test]
fn the_float_texel_routine_words_bound_its_emission() {
    use crate::spirv::serialize::serialize;
    let ins = [v4(), int(32)];
    let size = |body: Body<'_>| -> usize {
        let m = kernel(&ins, v4(), body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let without = size(&|_, x| Ok(x[0]));
    let defined = size(&|fb, x| {
        float_texel::define(fb.module())?;
        Ok(x[0])
    }) - without;
    let called = size(&body(TexelLanes::Texel)) - without - defined;
    assert!(
        defined as u64 <= float_texel::ROUTINE_WORDS,
        "the routine is {defined} words, bound {}",
        float_texel::ROUTINE_WORDS
    );
    assert!(
        called <= float_texel::CALL_WORDS as usize,
        "a call is {called} words, bound {}",
        float_texel::CALL_WORDS
    );
    eprintln!("float texel routine: {defined} words, a call {called}");
}
