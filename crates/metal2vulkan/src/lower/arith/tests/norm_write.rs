//! Interpreter differential tests of a written `float` texel rounded to the codes Metal stores in
//! a normalized storage texture by the bound format's class (`lower::texture::norm_write`).
//! References are plain Rust: Metal's measured rule, each channel's value clamped
//! to its range (a NaN 0) times its maximum code `m` exactly, rounded to nearest with a tie away
//! from zero, a 1-bit channel 1 only above one half; the value written `c / m` correctly rounded,
//! with the clamped value's sign.

use super::*;
use crate::lower::texture::norm_write;
use crate::metal::texel_format::{texel_format, PackedTexels};
use crate::metal::PixelFormat as P;

fn v4() -> Form {
    vec(float(32), 4)
}

/// The routine at the class an input holds.
fn body(fb: &mut FunctionBuilder<'_>, x: &[Value]) -> BResult<Value> {
    let r = norm_write::define(fb.module())?;
    norm_write::convert(fb, r, x[0], x[1])
}

/// Metal's code of `x` in a channel of maximum code `m` (signed where `signed`), as an `i64`.
fn metal_code(x: f32, m: u32, signed: bool) -> i64 {
    let lo = if signed { -1.0 } else { 0.0 };
    let v = if x.is_nan() { 0.0 } else { x.clamp(lo, 1.0) };
    if m == 1 {
        return i64::from(v > 0.5);
    }
    // The product of a 24-bit significand and a 16-bit integer is exact in `f64`; `round` takes a
    // tie away from zero.
    (f64::from(v) * f64::from(m)).round() as i64
}

/// The `f32` bits the routine writes for `x` in such a channel: Metal's code over `m`, correctly
/// rounded, with the clamped value's sign (`-0.0` for a negative value whose code is 0).
fn written(x: f32, m: u32, signed: bool) -> u64 {
    let lo = if signed { -1.0 } else { 0.0 };
    let v = if x.is_nan() { 0.0 } else { x.clamp(lo, 1.0) };
    let c = metal_code(x, m, signed);
    let q = c.unsigned_abs() as f32 / m as f32;
    u64::from(if v < 0.0 { -q } else { q }.to_bits())
}

/// Each lane's maximum code and sign of `pf`'s class: an unpacked family's in every lane, a
/// packed field's own scales.
fn scales(pf: P) -> ([u32; 4], bool) {
    let signed = matches!(
        pf,
        P::R8Snorm | P::RG8Snorm | P::RGBA8Snorm | P::R16Snorm | P::RG16Snorm | P::RGBA16Snorm
    );
    if let Some(s) = PackedTexels::of(pf).and_then(PackedTexels::scales) {
        return (s, signed);
    }
    let wide = matches!(
        pf,
        P::R16Unorm | P::RG16Unorm | P::RGBA16Unorm | P::R16Snorm | P::RG16Snorm | P::RGBA16Snorm
    );
    let m = match (wide, signed) {
        (false, false) => 255,
        (false, true) => 127,
        (true, false) => 65535,
        (true, true) => 32767,
    };
    ([m; 4], signed)
}

/// The values beside each code and each tie of a channel of maximum `m` (every `step`-th), two
/// `f32` steps either side, over its range.
fn around_codes(m: u32, signed: bool, step: usize) -> Vec<f32> {
    let lo = if signed { -i64::from(m) } else { 0 };
    let mut out = Vec::new();
    for k in (lo..=i64::from(m)).step_by(step).chain([i64::from(m)]) {
        for c in [k as f64, k as f64 + 0.5, k as f64 - 0.5] {
            let x = (c / f64::from(m)) as f32;
            for d in 0..=2 {
                out.push(f32::from_bits(x.to_bits().wrapping_add(d)));
                out.push(f32::from_bits(x.to_bits().wrapping_sub(d)));
            }
        }
    }
    out
}

fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        f32::from_bits(0x3f00_0001),
        f32::from_bits(0x3eff_ffff),
        1.5,
        -1.5,
        2.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MIN_POSITIVE,
        1e-10,
        -1e-10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xFFC0_0001),
        f32::from_bits(0x7F80_0001),
    ]
}

/// Run the routine over `values`, each in every lane, at `class`, against `reference` of each
/// lane's index and value, bit for bit.
fn run(what: &str, class: u32, values: &[f32], reference: &dyn Fn(usize, f32) -> u64) {
    let cases: Vec<Case> = values
        .iter()
        .map(|x| vec![vec![u64::from(x.to_bits()); 4], vec![u64::from(class)]])
        .collect();
    check_accept(
        what,
        &[v4(), int(32)],
        v4(),
        &body,
        &cases,
        &|c| {
            let x = f32::from_bits(c[0][0] as u32);
            Some((0..4).map(|i| reference(i, x)).collect())
        },
        &|g, w| g == w,
    );
}

/// Every normalized format's class rounds each channel to Metal's code on its own scale: the 8-
/// and 16-bit unorm and snorm families in every lane, the packed formats' 10:10:10:2, 5:6:5, 4:4:4:4
/// and 5:5:5:1 channels each on its own, a 1-bit alpha's one half to 0.
#[test]
fn a_normalized_write_rounds_each_channel_to_metals_code() {
    for pf in [
        P::R8Unorm,
        P::RGBA8Unorm,
        P::BGRA8Unorm,
        P::A8Unorm,
        P::RGBA8Snorm,
        P::RGBA16Unorm,
        P::RG16Snorm,
        P::RGBA16Snorm,
        P::RGB10A2Unorm,
        P::BGR10A2Unorm,
        P::B5G6R5Unorm,
        P::A1BGR5Unorm,
        P::ABGR4Unorm,
        P::BGR5A1Unorm,
    ] {
        let class = texel_format(pf).0;
        let (ms, signed) = scales(pf);
        let mut values = specials();
        let mut seen = Vec::new();
        for m in ms {
            if seen.contains(&m) {
                continue;
            }
            seen.push(m);
            let step = if m > 1023 { 97 } else { 1 };
            values.extend(around_codes(m, signed, step));
        }
        run(&format!("{pf:?}"), class, &values, &|i, x| {
            written(x, ms[i], signed)
        });
    }
}

/// Every other class writes the texel as computed, a NaN's payload included: the float,
/// half-float and small-float families, an 8-bit sRGB format's (its storage choice encodes it),
/// `ARGB8Unorm`'s field without a family (its storage choice rounds it), a YCbCr format's field
/// and `OTHER`.
#[test]
fn another_class_writes_the_texel_as_computed() {
    let mut values = specials();
    values.extend(around_codes(255, true, 7));
    for pf in [
        P::RGBA32Float,
        P::RGBA16Float,
        P::RG11B10Float,
        P::RGBA8UnormSRGB,
        P::ARGB8Unorm,
        P::Invalid,
    ] {
        let class = texel_format(pf).0;
        run(&format!("{pf:?}"), class, &values, &|_, x| {
            u64::from(x.to_bits())
        });
    }
    let ycbcr = PackedTexels::YCbCr2048.class().0
        | crate::metal::bicubic::BicubicTexels::UNORM
        | crate::metal::bicubic::BicubicTexels::EXACT;
    run("a YCbCr field", ycbcr, &values, &|_, x| {
        u64::from(x.to_bits())
    });
}

/// The value written, `c / m` within three `f32` ulps (a device's division is within 2.5 ulps),
/// times `m` lies within `2^-5` of `c`, so a device that rounds the product to nearest stores `c`
/// whatever it does at a tie: every code of every scale the routine takes.
#[test]
fn a_written_code_value_stores_its_code() {
    for m in [1u32, 3, 15, 31, 63, 127, 255, 1023, 32767, 65535] {
        for c in 0..=m {
            let q = c as f32 / m as f32;
            for d in -3i32..=3 {
                let v = f32::from_bits(q.to_bits().wrapping_add_signed(d));
                if c == 0 && d < 0 {
                    continue;
                }
                let product = f64::from(v) * f64::from(m);
                assert!(
                    (product - f64::from(c)).abs() < 1.0 / 32.0,
                    "m {m} code {c}: {v:e} times m is {product}"
                );
            }
        }
    }
}

#[test]
fn the_norm_write_routine_words_bound_its_emission() {
    use crate::spirv::serialize::serialize;
    let ins = [v4(), int(32)];
    let size = |body: Body<'_>| -> usize {
        let m = kernel(&ins, v4(), body).expect("build");
        serialize(m).expect("serialize").words.len()
    };
    let without = size(&|_, x| Ok(x[0]));
    let defined = size(&|fb, x| {
        norm_write::define(fb.module())?;
        Ok(x[0])
    }) - without;
    let called = size(&body) - without - defined;
    assert!(
        defined as u64 <= norm_write::ROUTINE_WORDS,
        "the routine is {defined} words, bound {}",
        norm_write::ROUTINE_WORDS
    );
    assert!(
        called <= norm_write::CALL_WORDS as usize,
        "a call is {called} words, bound {}",
        norm_write::CALL_WORDS
    );
    eprintln!("norm write routine: {defined} words, a call {called}");
}
