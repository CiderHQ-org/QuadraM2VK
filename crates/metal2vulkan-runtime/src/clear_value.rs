//! The texels Metal's `Clear` load action writes, and the Vulkan clear values that write the same
//! texels.
//!
//! Metal takes a clear color as four doubles, a clear depth as a double and a clear stencil as a
//! 32-bit value. Its conversion to each format was measured on an Apple M4 (macOS 27.0.1, by our
//! Metal probes), and is copied here channel by channel:
//!
//! - **Normalized** channels go through `float` (round to nearest even), then NaN is 0, the value
//!   is clamped to `[0, 1]` (`[-1, 1]` signed), and scaled to the nearest integer: 0.5/255 is 1
//!   and 0.5/65535 is 0, as the exact product of the `float` says. The channels of the packed
//!   16-bit formats (B5G6R5, A1BGR5, ABGR4, BGR5A1) go through `half` too, and halfway values go
//!   to even: an ABGR4 alpha of 0.1 is 1 (`half` 0.09998 × 15), and an A1BGR5 alpha of 0.5 is 0.
//!   sRGB channels are encoded with the sRGB curve before the scaling; their alpha is linear.
//! - **Integer** channels truncate the double toward zero and saturate; NaN is 0. No `float`
//!   step: 16777217 stays 16777217.
//! - **Half** channels go through `float` and then `half`, each rounding to nearest even
//!   (1 + 2^-11 + 2^-40 is 1.0); overflow is infinity, subnormals stay, and NaN of either sign
//!   is `0x7e00`. **Float** channels round to nearest even, keeping subnormals; NaN keeps its
//!   sign (`0x7fc00000`, `0xffc00000`).
//! - **RG11B10** channels truncate their mantissas; negative values are 0, a finite value past
//!   the largest is the largest, infinity stays, and NaN of either sign is `0x7e0` (`0x3f0` in
//!   the 10-bit channel). **RGB9E5** takes its exponent from the largest channel and truncates
//!   every mantissa; negative values and NaN are 0, and values past 65408 are 65408.
//! - **Depth** goes through `float`; `Depth16Unorm` truncates (0.5 is `0x7fff`). Metal refuses a
//!   clear depth outside `[0, 1]` (the caller's check). **Stencil** keeps the low 8 bits.
//!
//! Vulkan converts a clear value to the format as its own rules say, which leave rounding to the
//! implementation. So each clear value here is one the conversion maps exactly: the integer itself
//! for integer formats, `k / m` for a normalized integer `k` (whose exact product with `m` is
//! within a hair of `k`), the decoded sRGB value for an sRGB channel, and the exact `float` value
//! of a `half`, small-float or shared-exponent result. A `Depth16Unorm` clear is `(k + 1/4) / m`, a
//! quarter step above the stored integer `k` (or 1.0 for the largest), which rounding to nearest
//! and truncating both map to `k`: MoltenVK hands the value to Metal, which truncates, and `k / m`
//! lies a hair below `k` for some `k` (1/65535 stored 0 there). A NaN of a `half` or small-float
//! channel has no such value: Vulkan leaves its bits to the device, and RADV and lavapipe write
//! other NaNs than Metal's canonical quiet one (a Metal probe, the M3 Max and an M4 alike:
//! `0x7e00`, and `0x7e0` and `0x3f0` for `RG11B10Float`, the sign dropped). [`nan_texel`] gives
//! such a texel's bits, which `nan_clear.rs` writes before the pass. A `float` NaN keeps its sign
//! (`0x7fc00000`, `0xffc00000`), which every device measured writes.
//!
//! An **extended-range** attachment holds its raw texels in an unsigned integer format, so its
//! clear value is the raw texel Metal stores (`ExtendedRange::clear_texel`): each channel goes
//! through `float` and then `half`, each rounding to nearest even, and the `half` takes the code a
//! kernel's write of it stores (on an M4 under macOS 27.0.1: every texel of 8,192 random clear
//! colors of each of the six formats). That is one of Metal's two routes
//! (`metal::format::ClearRoute`), which a bfloat attachment's codes take too: where Metal keeps the
//! level compressed and the pass has no draw call (`nan_clear::keeps_nan_sign`, the pass's encoder
//! choosing at its end), a bfloat channel's `float` is rounded straight to bfloat, on the M4 under
//! 27.0.1 and the M3 Max under 26.7 alike, and a `BGR10_XR` or `BGRA10_XR` channel's code is
//! straight from its `float`, `BGR10_XR`'s bits 30 and 31 zero (the M4 under 27.0.1); the two-plane
//! formats store the same on either route.
//!
//! A **word** attachment (`metal::format::WordAttachment`: a packed 16-bit, one- or two-channel
//! 8-bit sRGB or `RGB9E5Float` format the device holds but does not render) renders through the
//! view of its word, so its clear value is the word Metal's clear stores ([`word`]).
//!
//! A **two-plane YCbCr** attachment whose store is matched (`metal::format::ycbcr_renders`) renders
//! into floats, which the pass's store encodes into the planes as Metal stores a drawn texel: its
//! clear value is the clear color through `float` and then `half`, each rounding to nearest even
//! (NaN kept), which the store clamps and encodes (every byte of the clears' planes is the encoding
//! of the clear color's `half`s; the 10-bit formats' 320 clears alike). The store takes a drawn
//! 10-bit texel's `half` toward zero (`bind::ycbcr_store`), which keeps a `half`, so a cleared
//! texel's codes are its `half`s' at either depth.

use ash::vk;
use metal2vulkan::metal::format::{
    alpha_in_red_format, rotated_channels, to_rotated_image, ycbcr_renders, ClearRoute,
    ExtendedRange, WordAttachment,
};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::Format;

/// How Metal converts one clear color channel of a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    /// A channel the format does not have.
    Unused,
    /// Unsigned normalized, of this many bits, through `float`.
    Unorm(u32),
    /// Unsigned normalized, of this many bits, through `float` and `half` (the packed 16-bit
    /// formats).
    HalfUnorm(u32),
    /// Signed normalized, of this many bits.
    Snorm(u32),
    /// An 8-bit sRGB-encoded color channel.
    Srgb,
    /// Unsigned integer, of this many bits.
    Uint(u32),
    /// Signed integer, of this many bits.
    Sint(u32),
    /// IEEE half.
    Half,
    /// IEEE single.
    Float,
    /// An unsigned small float with a 5-bit exponent and this many mantissa bits.
    Ufloat(u32),
    /// A channel of RGB9E5's shared exponent.
    Shared,
    /// A field of this many bits a fill and a clear write as zeros: an RGBX format's X field of a
    /// normalized or integer kind.
    Zero(u32),
}

/// The channels (red, green, blue, alpha) of each color format a pass can render to, or `None`
/// for a format with no Metal clear rule here (a depth, stencil or compressed format).
fn channels(pf: PixelFormat) -> Option<[Channel; 4]> {
    use Channel::*;
    use PixelFormat as P;
    let one = |c| [c, Unused, Unused, Unused];
    let two = |c| [c, c, Unused, Unused];
    let four = |c| [c, c, c, c];
    Some(match pf {
        P::A8Unorm => [Unused, Unused, Unused, Unorm(8)],
        P::R8Unorm => one(Unorm(8)),
        P::R8UnormSRGB => one(Srgb),
        P::R8Snorm => one(Snorm(8)),
        P::R8Uint => one(Uint(8)),
        P::R8Sint => one(Sint(8)),
        P::R16Unorm => one(Unorm(16)),
        P::R16Snorm => one(Snorm(16)),
        P::R16Uint => one(Uint(16)),
        P::R16Sint => one(Sint(16)),
        P::R16Float => one(Half),
        P::RG8Unorm => two(Unorm(8)),
        P::RG8UnormSRGB => two(Srgb),
        P::RG8Snorm => two(Snorm(8)),
        P::RG8Uint => two(Uint(8)),
        P::RG8Sint => two(Sint(8)),
        P::B5G6R5Unorm => [HalfUnorm(5), HalfUnorm(6), HalfUnorm(5), Unused],
        P::A1BGR5Unorm | P::BGR5A1Unorm => [HalfUnorm(5), HalfUnorm(5), HalfUnorm(5), HalfUnorm(1)],
        P::ABGR4Unorm => four(HalfUnorm(4)),
        P::R32Uint => one(Uint(32)),
        P::R32Sint => one(Sint(32)),
        P::R32Float => one(Float),
        P::RG16Unorm => two(Unorm(16)),
        P::RG16Snorm => two(Snorm(16)),
        P::RG16Uint => two(Uint(16)),
        P::RG16Sint => two(Sint(16)),
        P::RG16Float => two(Half),
        P::RGBA8Unorm | P::BGRA8Unorm => four(Unorm(8)),
        // Metal clears `ARGB8Unorm` as `RGBA8Unorm`, its channels rotated.
        P::ARGB8Unorm => four(Unorm(8)),
        P::RGBA8UnormSRGB | P::BGRA8UnormSRGB => [Srgb, Srgb, Srgb, Unorm(8)],
        P::RGBA8Snorm => four(Snorm(8)),
        P::RGBA8Uint => four(Uint(8)),
        P::RGBA8Sint => four(Sint(8)),
        P::RGB10A2Unorm | P::BGR10A2Unorm => [Unorm(10), Unorm(10), Unorm(10), Unorm(2)],
        P::RGB10A2Uint => [Uint(10), Uint(10), Uint(10), Uint(2)],
        P::RG11B10Float => [Ufloat(6), Ufloat(6), Ufloat(5), Unused],
        P::RGB9E5Float => [Shared, Shared, Shared, Unused],
        P::RG32Uint => two(Uint(32)),
        P::RG32Sint => two(Sint(32)),
        P::RG32Float => two(Float),
        P::RGBA16Unorm => four(Unorm(16)),
        P::RGBA16Snorm => four(Snorm(16)),
        P::RGBA16Uint => four(Uint(16)),
        P::RGBA16Sint => four(Sint(16)),
        P::RGBA16Float => four(Half),
        P::RGBA32Uint => four(Uint(32)),
        P::RGBA32Sint => four(Sint(32)),
        P::RGBA32Float => four(Float),
        // The RGBX formats: the twin's red, green and blue; the X field zeros where the twin's
        // alpha is normalized or an integer, the alpha's value where it is a float (three fill
        // colors of each format, beside its RGBA twin; a clear of a whole level by a pass without a
        // draw call).
        P::RGBX8Unorm => [Unorm(8), Unorm(8), Unorm(8), Zero(8)],
        P::RGBX8UnormSRGB => [Srgb, Srgb, Srgb, Zero(8)],
        P::RGBX8Snorm => [Snorm(8), Snorm(8), Snorm(8), Zero(8)],
        P::RGBX8Uint => [Uint(8), Uint(8), Uint(8), Zero(8)],
        P::RGBX8Sint => [Sint(8), Sint(8), Sint(8), Zero(8)],
        P::RGBX16Uint => [Uint(16), Uint(16), Uint(16), Zero(16)],
        P::RGBX16Sint => [Sint(16), Sint(16), Sint(16), Zero(16)],
        P::RGBX16Float => four(Half),
        P::RGBX32Uint => [Uint(32), Uint(32), Uint(32), Zero(32)],
        P::RGBX32Sint => [Sint(32), Sint(32), Sint(32), Zero(32)],
        P::RGBX32Float => four(Float),
        _ => return None,
    })
}

/// The channels (red, green, blue, alpha) color format `pf` has, or `None` for a format a pass
/// cannot render to here.
pub(super) fn has_channels(pf: PixelFormat) -> Option<[bool; 4]> {
    Some(channels(pf)?.map(|c| c != Channel::Unused))
}

/// The texel value Metal's conversion gives one channel: an integer, or the bits of a float.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Texel {
    /// A normalized integer `k` of `max` (`k / max` is the value), or a signed one.
    Norm {
        k: i64,
        max: i64,
    },
    /// An sRGB-encoded 8-bit value.
    Srgb(u32),
    Uint(u32),
    Sint(i32),
    /// A value exactly representable as a `float` (a `half`, small-float or shared-exponent
    /// result, decoded), by its bits.
    Exact(u32),
}

/// The Vulkan clear value of Metal's clear `color` for an attachment of `pf`, whose Vulkan
/// format presents Metal's channels in order (attachment formats do, but an image holding them
/// rotated, `ARGB8Unorm`'s, whose clear value is rotated as its image holds them,
/// `metal::format::to_rotated_image`). An extended-range attachment's codes are by Metal's route
/// `route`. `None` for a format with no clear rule.
pub(super) fn color(
    pf: PixelFormat,
    color: [f64; 4],
    route: ClearRoute,
) -> Option<vk::ClearColorValue> {
    // An extended-range attachment holds its raw texels: the clear is the texel Metal stores.
    if let Some(form) = ExtendedRange::of(pf) {
        return Some(vk::ClearColorValue {
            uint32: form.clear_texel(color, route),
        });
    }
    // A YCbCr attachment's floats, each the clear's `half`: the store encodes them.
    if ycbcr_renders(pf) {
        return Some(vk::ClearColorValue {
            float32: color.map(|c| f16_to_f32(f32_to_f16(c as f32))),
        });
    }
    let channels = channels(pf)?;
    let texels = convert(&channels, color);
    Some(if channels.iter().any(|c| matches!(c, Channel::Uint(_))) {
        let mut uint32 = [0u32; 4];
        for (out, texel) in uint32.iter_mut().zip(texels) {
            if let Texel::Uint(v) = texel {
                *out = v;
            }
        }
        vk::ClearColorValue { uint32 }
    } else if channels.iter().any(|c| matches!(c, Channel::Sint(_))) {
        let mut int32 = [0i32; 4];
        for (out, texel) in int32.iter_mut().zip(texels) {
            if let Texel::Sint(v) = texel {
                *out = v;
            }
        }
        vk::ClearColorValue { int32 }
    } else {
        let mut float32 = [0f32; 4];
        for (out, texel) in float32.iter_mut().zip(texels) {
            *out = match texel {
                Texel::Norm { k, max } => (k as f64 / max as f64) as f32,
                Texel::Srgb(k) => srgb_decode(k),
                Texel::Exact(bits) => f32::from_bits(bits),
                Texel::Uint(_) | Texel::Sint(_) => 0.0,
            };
        }
        if rotated_channels(pf) {
            float32 = to_rotated_image(float32);
        }
        vk::ClearColorValue { float32 }
    })
}

/// [`color`] for an attachment of `pf` whose image is of Vulkan format `format`: an `A8Unorm`
/// image held in `R8_UNORM` (`metal::format::FormatEmulation::AlphaInRed`) takes the clear's alpha
/// in red, as it holds Metal's alpha (Metal's clear stores the color's alpha, probe `a8`, an M3
/// Max under macOS 27.0.1).
pub(super) fn held_color(
    pf: PixelFormat,
    format: Format,
    color: [f64; 4],
    route: ClearRoute,
) -> Option<vk::ClearColorValue> {
    let value = self::color(pf, color, route)?;
    if alpha_in_red_format(pf) != Some(format) {
        return Some(value);
    }
    // SAFETY: `color` gives a normalized format's clear as `float32`.
    let float32 = unsafe { value.float32 };
    Some(vk::ClearColorValue {
        float32: to_rotated_image(float32),
    })
}

/// The Vulkan clear value of Metal's clear `color` for an attachment of `pf` that renders through
/// the view of its word `word` (`metal::format::WordAttachment`): the word of the codes Metal's
/// clear stores ([`color`]'s conversion of each channel; `RGB9E5Float`'s word of the color
/// through `float`, its mantissas truncated, `WordAttachment::output_word`), in the view's one
/// unsigned component. `None` for a format with no clear rule.
pub(super) fn word(
    pf: PixelFormat,
    color: [f64; 4],
    word: WordAttachment,
) -> Option<vk::ClearColorValue> {
    if word == WordAttachment::Rgb9e5 {
        let uint32 = [word.output_word(color.map(|c| c as f32)), 0, 0, 0];
        return Some(vk::ClearColorValue { uint32 });
    }
    let codes = convert(&channels(pf)?, color).map(|t| match t {
        Texel::Norm { k, .. } => u32::try_from(k).unwrap_or(0),
        Texel::Srgb(k) => k,
        _ => 0,
    });
    Some(vk::ClearColorValue {
        uint32: [word.pack(codes), 0, 0, 0],
    })
}

/// Where a `half` or small-float channel of `value`, the Vulkan clear value [`color`] gives an
/// attachment of `pf`, is a NaN, whose bits Vulkan leaves to the device's conversion
/// (`nan_clear.rs`): the unsigned-integer format of the texel's size and the clear value that
/// writes Metal's texel there, its bits from the texel's first byte. `None` for every other
/// clear, which converts exactly.
/// A `half` NaN is the canonical quiet NaN, `0x7e00`, where not `signed`; where `signed` (a
/// level Metal clears keeping them, `nan_clear::keeps_nan_sign`) it keeps the clear value's sign
/// and the top ten bits of its payload, quieted: `sign | 0x7e00 | payload`, the double's
/// mantissa's top ten bits (`0x7ffc...` stores `0x7f00`, `0x7fff...` `0x7fff`,
/// `0xffff...` `0xffff`, the signalling `0x7ff4...` `0x7f00`, `0x7ff0...01` `0x7e00`).
pub(super) fn nan_texel(
    pf: PixelFormat,
    value: &vk::ClearColorValue,
    signed: bool,
) -> Option<(Format, vk::ClearColorValue)> {
    let nan = match signed {
        true => NanBits::Signed,
        false => NanBits::Signless,
    };
    small_float_texel(pf, value, nan)
}

/// The texel Metal's resolve stores from samples that each hold [`nan_texel`]'s texel of
/// `value`, where it averages them: each NaN channel zero, as its resolve counts a NaN sample, in
/// the unsigned-integer format of the texel's size. `None` where `value` holds no `half` or
/// small-float NaN.
pub(super) fn resolved_nan_texel(
    pf: PixelFormat,
    value: &vk::ClearColorValue,
) -> Option<(Format, vk::ClearColorValue)> {
    small_float_texel(pf, value, NanBits::Zero)
}

/// Where a channel of `value`, the Vulkan clear value [`color`] gives an attachment of `pf` whose
/// channels are 32-bit floats, is a NaN: the unsigned-integer format of the texel's size and the
/// clear value that writes the texel's bits, which the device's clear writes too (a `float`
/// converts nothing), for a write of the texel elsewhere (`nan_clear.rs`'s resolves). `None` for
/// another format or a clear without a NaN.
pub(super) fn float_nan_texel(
    pf: PixelFormat,
    value: &vk::ClearColorValue,
) -> Option<(Format, vk::ClearColorValue)> {
    let channels = channels(pf)?;
    // SAFETY: every member of the union is four 32-bit values; any bits are a valid `f32`.
    let float32 = unsafe { value.float32 };
    let used: Vec<f32> = channels
        .iter()
        .zip(float32)
        .filter(|(c, _)| !matches!(c, Channel::Unused))
        .map(|(c, v)| matches!(c, Channel::Float).then_some(v))
        .collect::<Option<_>>()?;
    if !used.iter().any(|v| v.is_nan()) {
        return None;
    }
    let bytes: Vec<u8> = used
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    uint_texel(&bytes)
}

/// What a NaN channel of a [`small_float_texel`] holds.
#[derive(Clone, Copy)]
enum NanBits {
    /// The canonical quiet NaN, without a sign.
    Signless,
    /// The canonical quiet NaN with the clear value's sign, in a `half` channel.
    Signed,
    /// Zero.
    Zero,
}

/// The texel of `value` in `pf`, a format of `half` or small-float channels, each NaN channel as
/// `nan` says ([`nan_texel`]); `None` where no such channel is a NaN.
fn small_float_texel(
    pf: PixelFormat,
    value: &vk::ClearColorValue,
    nan: NanBits,
) -> Option<(Format, vk::ClearColorValue)> {
    let channels = channels(pf)?;
    // SAFETY: every member of the union is four 32-bit values; any bits are a valid `f32`.
    let float32 = unsafe { value.float32 };
    let small_float = |c: &Channel| matches!(c, Channel::Half | Channel::Ufloat(_));
    if !channels
        .iter()
        .zip(float32)
        .any(|(c, v)| small_float(c) && v.is_nan())
    {
        return None;
    }
    let mut bits = 0u64;
    let mut at = 0;
    for (&channel, v) in channels.iter().zip(float32) {
        let (field, width) = match (channel, nan) {
            (Channel::Half, NanBits::Zero) if v.is_nan() => (0, 16),
            (Channel::Ufloat(mantissa), NanBits::Zero) if v.is_nan() => (0, mantissa + 5),
            // The sign and the top ten payload bits, quieted ([`convert`] carries the clear
            // value's in the `float`).
            (Channel::Half, NanBits::Signed) if v.is_nan() => {
                let bits = v.to_bits();
                let kept = ((bits >> 16) & 0x8000) | 0x7e00 | ((bits >> 13) & 0x3ff);
                (u64::from(kept), 16)
            }
            (Channel::Half, _) => (u64::from(f32_to_f16(v)), 16),
            (Channel::Ufloat(mantissa), _) => (ufloat_bits(v, mantissa), mantissa + 5),
            (Channel::Unused, _) => continue,
            // No format mixes small floats with other channels.
            _ => return None,
        };
        bits |= field << at;
        at += width;
    }
    if at % 8 != 0 {
        return None;
    }
    let bytes = bits.to_le_bytes();
    uint_texel(bytes.get(..at as usize / 8)?)
}

/// The unsigned-integer format of `texel`'s size (2, 4, 8 or 16 bytes: `R16_UINT`, `R32_UINT`,
/// `R32G32_UINT`, `R32G32B32A32_UINT`) and the clear value whose conversion into it writes
/// exactly `texel`'s bytes, its little-endian 16- or 32-bit words: a texel whose bits no clear
/// of its own format controls is written so, then copied bit-wise into an image of its format
/// (`nan_clear.rs`, a new multisampled texture's pattern: `first_move::Pattern::Scratch`).
/// `None` for another size.
pub(crate) fn uint_texel(texel: &[u8]) -> Option<(Format, vk::ClearColorValue)> {
    let format = match texel.len() {
        2 => Format::R16_UINT,
        4 => Format::R32_UINT,
        8 => Format::R32G32_UINT,
        16 => Format::R32G32B32A32_UINT,
        _ => return None,
    };
    let mut uint32 = [0; 4];
    match texel {
        [low, high] => uint32[0] = u32::from(u16::from_le_bytes([*low, *high])),
        _ => {
            for (word, bytes) in uint32.iter_mut().zip(texel.as_chunks::<4>().0.iter()) {
                *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            }
        }
    }
    Some((format, vk::ClearColorValue { uint32 }))
}

/// The bits of a small unsigned float of `mantissa` bits (and a 5-bit exponent) holding `v`, which
/// is one exactly ([`ufloat_value`]'s result): NaN is the canonical quiet NaN (the top mantissa
/// bit), as Metal writes it (a Metal probe on the M3 Max and an M4).
fn ufloat_bits(v: f32, mantissa: u32) -> u64 {
    if v.is_nan() {
        return (31 << mantissa) | (1 << (mantissa - 1));
    }
    if v.is_infinite() {
        return 31 << mantissa;
    }
    if v <= 0.0 {
        return 0;
    }
    let steps = 1u64 << mantissa;
    let exponent = binade(v).max(-14);
    // Exact: `v` is a whole number of the binade's spacing.
    let scaled = (f64::from(v) / 2f64.powi(exponent) * steps as f64) as u64;
    match scaled < steps {
        true => scaled,
        false => (((exponent + 15) as u64) << mantissa) | (scaled - steps),
    }
}

/// Each channel's texel under Metal's conversion of `color`.
fn convert(channels: &[Channel; 4], color: [f64; 4]) -> [Texel; 4] {
    let shared = shared_exponent(channels, color);
    let mut out = [Texel::Uint(0); 4];
    for (i, (&channel, &x)) in channels.iter().zip(&color).enumerate() {
        out[i] = match channel {
            Channel::Unused | Channel::Zero(_) => Texel::Uint(0),
            Channel::Unorm(bits) => unorm(x as f32 as f64, bits),
            Channel::HalfUnorm(bits) => unorm(f16_to_f32(f32_to_f16(x as f32)) as f64, bits),
            Channel::Snorm(bits) => snorm(x as f32 as f64, bits),
            Channel::Srgb => Texel::Srgb(srgb_encode(x as f32 as f64)),
            Channel::Uint(bits) => Texel::Uint(uint(x, bits)),
            Channel::Sint(bits) => Texel::Sint(sint(x, bits)),
            // A NaN is the quiet `float` NaN of the double's sign and top payload bits, which
            // `nan_texel` keeps where Metal's clear does; every other value is the `half`'s.
            Channel::Half if x.is_nan() => {
                let bits = x.to_bits();
                let sign = ((bits >> 32) & 0x8000_0000) as u32;
                let payload = ((bits >> 29) & 0x7f_ffff) as u32;
                Texel::Exact(sign | 0x7fc0_0000 | payload)
            }
            Channel::Half => Texel::Exact(f16_to_f32(f32_to_f16(x as f32)).to_bits()),
            // A NaN is the double's sign and top payload bits, quieted: Metal's conversion
            // (`0x7ff0000020000000` clears to `0x7fc00001`), which a cast leaves to
            // the platform.
            Channel::Float if x.is_nan() => {
                let bits = x.to_bits();
                let sign = ((bits >> 32) & 0x8000_0000) as u32;
                Texel::Exact(sign | 0x7fc0_0000 | ((bits >> 29) & 0x7f_ffff) as u32)
            }
            Channel::Float => Texel::Exact((x as f32).to_bits()),
            Channel::Ufloat(mantissa) => Texel::Exact(ufloat_value(x as f32, mantissa).to_bits()),
            Channel::Shared => Texel::Exact((shared[i] as f32).to_bits()),
        };
    }
    out
}

/// `v` as an unsigned normalized integer of `bits`: NaN is 0, clamped, nearest (ties to even).
fn unorm(v: f64, bits: u32) -> Texel {
    let max = (1i64 << bits) - 1;
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    // Exact: a `float` (or `half`) times a 16-bit integer fits a double's mantissa.
    Texel::Norm {
        k: (v * max as f64).round_ties_even() as i64,
        max,
    }
}

/// `v` as a signed normalized integer of `bits`: NaN is 0, clamped to `[-1, 1]`, nearest.
fn snorm(v: f64, bits: u32) -> Texel {
    let max = (1i64 << (bits - 1)) - 1;
    let v = if v.is_nan() { 0.0 } else { v.clamp(-1.0, 1.0) };
    Texel::Norm {
        k: (v * max as f64).round_ties_even() as i64,
        max,
    }
}

/// `x` truncated toward zero and saturated to an unsigned integer of `bits`; NaN is 0.
fn uint(x: f64, bits: u32) -> u32 {
    let max = ((1u64 << bits) - 1) as f64;
    if x.is_nan() || x <= 0.0 {
        0
    } else if x >= max {
        max as u32
    } else {
        x.trunc() as u32
    }
}

/// `x` truncated toward zero and saturated to a signed integer of `bits`; NaN is 0.
fn sint(x: f64, bits: u32) -> i32 {
    let max = ((1i64 << (bits - 1)) - 1) as f64;
    let min = -(1i64 << (bits - 1)) as f64;
    if x.is_nan() {
        0
    } else {
        x.trunc().clamp(min, max) as i32
    }
}

/// The 8-bit sRGB encoding of linear `v`: NaN is 0, clamped, the sRGB curve, nearest.
fn srgb_encode(v: f64) -> u32 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    let s = if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round_ties_even().clamp(0.0, 255.0) as u32
}

/// The linear value whose sRGB encoding is exactly `k / 255`.
fn srgb_decode(k: u32) -> f32 {
    let s = f64::from(k) / 255.0;
    let v = if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    };
    v as f32
}

/// `v` as an IEEE half's bits, rounding to nearest even; NaN is the canonical quiet NaN.
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7e00;
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if v.is_infinite() {
        return sign | 0x7c00;
    }
    let a = f64::from(v.abs());
    // The half nearest `a`, ties to even: scale by the binade's spacing and round.
    let exponent = binade(v).max(-14);
    let spacing = 2f64.powi(exponent - 10);
    let mut steps = (a / spacing).round_ties_even();
    let mut exponent = exponent;
    if steps >= 2048.0 {
        // Rounded up into the next binade.
        exponent += 1;
        steps /= 2.0;
    }
    if exponent > 15 {
        return sign | 0x7c00;
    }
    let steps = steps as u16;
    if steps < 1024 {
        // Subnormal (only at the lowest exponent).
        sign | steps
    } else {
        sign | (((exponent + 15) as u16) << 10) | (steps - 1024)
    }
}

/// An IEEE half's value.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((h >> 10) & 0x1f);
    let mantissa = f64::from(h & 0x3ff);
    let value = match exponent {
        0 => mantissa * 2f64.powi(-24),
        0x1f if mantissa == 0.0 => f64::INFINITY,
        0x1f => return f32::NAN,
        e => (1.0 + mantissa / 1024.0) * 2f64.powi(e - 15),
    };
    (sign * value) as f32
}

/// Metal's small unsigned float of `mantissa` bits (and a 5-bit exponent) for `v`, decoded:
/// negative values are 0, NaN stays NaN, infinity stays, a finite value past the largest is the
/// largest, and the mantissa is truncated.
fn ufloat_value(v: f32, mantissa: u32) -> f32 {
    if v.is_nan() {
        return f32::NAN;
    }
    if v <= 0.0 {
        return 0.0;
    }
    if v.is_infinite() {
        return f32::INFINITY;
    }
    let steps = f64::from(1u32 << mantissa);
    let largest = (2.0 - 1.0 / steps) * 2f64.powi(15);
    let exponent = binade(v).max(-14);
    let v = f64::from(v);
    if v >= largest {
        return largest as f32;
    }
    let spacing = 2f64.powi(exponent) / steps;
    ((v / spacing).floor() * spacing) as f32
}

/// The three channel values of an RGB9E5 clear, decoded (zero where `channels` is not
/// RGB9E5's): NaN and negative values are 0 and values past 65408 are 65408; the exponent is
/// that of the largest channel (at least -16) and every mantissa is truncated.
fn shared_exponent(channels: &[Channel; 4], color: [f64; 4]) -> [f64; 4] {
    const LARGEST: f32 = 65408.0;
    let mut out = [0.0; 4];
    if !channels.contains(&Channel::Shared) {
        return out;
    }
    let mut values = [0.0f32; 4];
    for (value, (&channel, &x)) in values.iter_mut().zip(channels.iter().zip(&color)) {
        if channel == Channel::Shared {
            let x = x as f32;
            *value = if x.is_nan() {
                0.0
            } else {
                x.clamp(0.0, LARGEST)
            };
        }
    }
    let max = values.iter().copied().fold(0.0, f32::max);
    let exponent = binade(max).max(-16);
    // Nine mantissa bits under a shared exponent of `exponent + 1`.
    let spacing = 2f64.powi(exponent + 1 - 9);
    for (o, v) in out.iter_mut().zip(values) {
        *o = (f64::from(v) / spacing).floor() * spacing;
    }
    out
}

/// `floor(log2(|v|))` of a finite `v`, from its bits: exact at every power of two. Zero and the
/// `float` subnormals answer -127, below every binade the callers use.
pub(super) fn binade(v: f32) -> i32 {
    let biased = ((v.to_bits() >> 23) & 0xff) as i32;
    if biased == 0 {
        -127
    } else {
        biased - 127
    }
}

/// The Vulkan clear depth of Metal's `depth` (already checked to lie in `[0, 1]`) for a depth
/// attachment of `pf`: a `float`, truncated to 16 bits for `Depth16Unorm` and given a quarter
/// step above the stored integer (the module docs), and by the same rule, unmeasured, truncated
/// to 24 bits for the 24-bit formats, whose `float` has no room for the quarter step.
pub(super) fn depth(pf: PixelFormat, depth: f64) -> f32 {
    let v = f64::from(depth as f32);
    let (bits, above) = match pf {
        PixelFormat::Depth16Unorm => (16, 0.25),
        PixelFormat::Depth24UnormStencil8 => (24, 0.0),
        _ => return v as f32,
    };
    let max = ((1u64 << bits) - 1) as f64;
    let k = (v * max).trunc();
    if k == max {
        return 1.0;
    }
    ((k + above) / max) as f32
}

/// The Vulkan clear stencil of Metal's `stencil`: its low 8 bits.
pub(super) fn stencil(stencil: u32) -> u32 {
    stencil & 0xff
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use PixelFormat as P;

    /// The bytes of the texel a clear of `color` writes into an attachment of `pf`: each channel
    /// converted as [`convert`] converts it, laid out as Metal lays out `pf`'s texel,
    /// little-endian, the fields from the texel's lowest bit in the order the format's name lists
    /// them (B5G6R5: blue in bits 0 to 4; RGB9E5: the exponent last). `None` for a format with no
    /// clear rule here.
    pub(crate) fn texel_bytes(pf: PixelFormat, color: [f64; 4]) -> Option<Vec<u8>> {
        use PixelFormat as P;
        let channels = channels(pf)?;
        let texels = convert(&channels, color);
        let order: &[usize] = match pf {
            P::BGRA8Unorm | P::BGRA8UnormSRGB | P::BGR10A2Unorm | P::BGR5A1Unorm => &[2, 1, 0, 3],
            P::B5G6R5Unorm => &[2, 1, 0],
            P::A1BGR5Unorm | P::ABGR4Unorm => &[3, 2, 1, 0],
            // Alpha in the lowest byte, then red, green and blue (the format scout's color fill on
            // an M4 under macOS 27.0.1 wrote `ff 40 80 bf` for (0.25, 0.5, 0.75, 1)).
            P::ARGB8Unorm => &[3, 0, 1, 2],
            _ => &[0, 1, 2, 3],
        };
        // RGB9E5's shared exponent: its largest channel's binade (at least -16), which the decoded
        // channels keep, since a truncated mantissa stays in its binade.
        let largest = texels[..3]
            .iter()
            .map(|t| match t {
                Texel::Exact(bits) => f32::from_bits(*bits),
                _ => 0.0,
            })
            .fold(0.0, f32::max);
        let shared = binade(largest).max(-16);
        let mut bits = 0u128;
        let mut at = 0;
        let mut put = |value: u64, width: u32| {
            bits |= u128::from(value & ((1u64 << width) - 1)) << at;
            at += width;
        };
        for &i in order {
            let (value, width) = match (channels[i], texels[i]) {
                (Channel::Unused, _) => continue,
                (Channel::Zero(b), _) => (0, b),
                (
                    Channel::Unorm(b) | Channel::HalfUnorm(b) | Channel::Snorm(b),
                    Texel::Norm { k, .. },
                ) => (k as u64, b),
                (Channel::Srgb, Texel::Srgb(k)) => (u64::from(k), 8),
                (Channel::Uint(b), Texel::Uint(v)) => (u64::from(v), b),
                (Channel::Sint(b), Texel::Sint(v)) => (v as i64 as u64, b),
                (Channel::Half, Texel::Exact(f)) => (u64::from(f32_to_f16(f32::from_bits(f))), 16),
                (Channel::Float, Texel::Exact(f)) => (u64::from(f), 32),
                (Channel::Ufloat(m), Texel::Exact(f)) => (ufloat_bits(f32::from_bits(f), m), m + 5),
                (Channel::Shared, Texel::Exact(f)) => {
                    // Exact: the value is a whole number of the shared exponent's spacing.
                    let spacing = 2f64.powi(shared + 1 - 9);
                    ((f64::from(f32::from_bits(f)) / spacing) as u64, 9)
                }
                _ => return None,
            };
            put(value, width);
        }
        if channels.contains(&Channel::Shared) {
            put((shared + 16) as u64, 5);
        }
        let bytes = at.div_ceil(8) as usize;
        Some(bits.to_le_bytes()[..bytes].to_vec())
    }

    /// A word attachment's clear value ([`word`]) is the little-endian word of the texel Metal's
    /// clear writes ([`texel_bytes`]), for every packed 16-bit layout, the one- and two-channel
    /// 8-bit sRGB ones and `RGB9E5Float`'s, over colors at and beside each layout's ties, NaN,
    /// negative values, values past one and past `RGB9E5Float`'s largest, and tiny ones.
    #[test]
    fn word_clears_are_metals_texels() {
        let values = [
            0.0,
            1.0,
            0.5,
            0.49,
            0.1,
            0.25,
            0.75,
            1.0 / 3.0,
            0.016,
            -0.5,
            2.0,
            f64::NAN,
            15.5 / 31.0,
            7.5 / 15.0,
            31.5 / 63.0,
            70000.0,
            3.0e-6,
            1000.3,
        ];
        for pf in [
            P::B5G6R5Unorm,
            P::A1BGR5Unorm,
            P::ABGR4Unorm,
            P::BGR5A1Unorm,
            P::R8UnormSRGB,
            P::RG8UnormSRGB,
            P::RGB9E5Float,
        ] {
            let layout = WordAttachment::layout(pf).expect("a word layout");
            for (i, &r) in values.iter().enumerate() {
                let n = values.len();
                let color = [
                    r,
                    values[(i + 3) % n],
                    values[(i + 7) % n],
                    values[(i + 11) % n],
                ];
                let got = word(pf, color, layout).expect("a clear value");
                // SAFETY: every member of the union is four 32-bit values; any bits are a `u32`.
                let got = unsafe { got.uint32 };
                let texel = bytes(pf, color);
                let want =
                    (texel.iter().enumerate()).fold(0u32, |w, (i, &b)| w | u32::from(b) << (8 * i));
                assert_eq!(got, [want, 0, 0, 0], "{pf:?} {color:?}");
            }
        }
    }

    /// The texels Metal writes, as little-endian bytes.
    fn bytes(pf: PixelFormat, c: [f64; 4]) -> Vec<u8> {
        texel_bytes(pf, c).unwrap_or_else(|| panic!("no packing for {pf:?}"))
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// Every clear the M4 measured (our Metal probes; the NaNs on an M4 and the M3 Max alike), as
    /// the bytes it wrote.
    #[test]
    fn metal_clears_are_reproduced() {
        let n = f64::NAN;
        let inf = f64::INFINITY;
        let cases: &[(PixelFormat, [f64; 4], &str)] = &[
            (P::RGBA8Unorm, [0.5, 0.25, 1.5, -0.5], "8040ff00"),
            (
                P::RGBA8Unorm,
                [0.5 / 255.0, 1.5 / 255.0, 2.5 / 255.0, 0.501960784],
                "01020380",
            ),
            (P::RGBA8Unorm, [0.0019, 0.0021, 0.998, 0.996], "0001fefe"),
            (P::RGBA8Unorm, [n, inf, -inf, 0.75], "00ff00bf"),
            (P::BGRA8Unorm, [0.25, 0.5, 0.75, 1.0], "bf8040ff"),
            (P::RGBA8UnormSRGB, [0.5, 0.214, 0.0031308, 1.0], "bc7f0aff"),
            (P::RGBA8UnormSRGB, [0.001, 0.01, 0.1, 0.5], "03195980"),
            (
                P::RGBA8UnormSRGB,
                [0.2158605, 0.2122308, 0.214, 0.25],
                "807f7f40",
            ),
            (P::BGRA8UnormSRGB, [0.5, 0.214, 0.0031308, 0.5], "0a7fbc80"),
            (P::RGBA8Snorm, [-1.5, -0.5, 0.5, 1.5 / 127.0], "81c04001"),
            (P::RGBA8Snorm, [-1.0, -0.0039, 0.0039, 0.502], "81000040"),
            (P::RGBA8Snorm, [n, inf, -inf, -0.0], "007f8100"),
            (P::R8Uint, [3.7, 0.0, 0.0, 0.0], "03"),
            (P::R8Uint, [300.0, 0.0, 0.0, 0.0], "ff"),
            (P::R8Uint, [-1.0, 0.0, 0.0, 0.0], "00"),
            (P::R8Uint, [255.5, 0.0, 0.0, 0.0], "ff"),
            (P::R8Uint, [1e10, 0.0, 0.0, 0.0], "ff"),
            (P::R8Uint, [n, 0.0, 0.0, 0.0], "00"),
            (P::R8Uint, [256.0, 0.0, 0.0, 0.0], "ff"),
            (P::R8Uint, [-0.5, 0.0, 0.0, 0.0], "00"),
            (P::R8Sint, [-3.7, 0.0, 0.0, 0.0], "fd"),
            (P::R8Sint, [200.0, 0.0, 0.0, 0.0], "7f"),
            (P::R8Sint, [-200.0, 0.0, 0.0, 0.0], "80"),
            (P::R8Sint, [127.5, 0.0, 0.0, 0.0], "7f"),
            (P::R8Sint, [-128.5, 0.0, 0.0, 0.0], "80"),
            (P::R8Sint, [3.7, 0.0, 0.0, 0.0], "03"),
            (P::R16Uint, [65535.7, 0.0, 0.0, 0.0], "ffff"),
            (P::R16Uint, [70000.0, 0.0, 0.0, 0.0], "ffff"),
            (P::R16Uint, [-2.0, 0.0, 0.0, 0.0], "0000"),
            (P::R16Sint, [-40000.0, 0.0, 0.0, 0.0], "0080"),
            (P::R16Sint, [40000.0, 0.0, 0.0, 0.0], "ff7f"),
            (P::R16Sint, [-2.5, 0.0, 0.0, 0.0], "feff"),
            (P::R32Uint, [4294967295.0, 0.0, 0.0, 0.0], "ffffffff"),
            (P::R32Uint, [4294967296.0, 0.0, 0.0, 0.0], "ffffffff"),
            (P::R32Uint, [1e20, 0.0, 0.0, 0.0], "ffffffff"),
            (P::R32Uint, [-1.0, 0.0, 0.0, 0.0], "00000000"),
            (P::R32Uint, [3.999, 0.0, 0.0, 0.0], "03000000"),
            (P::R32Uint, [2147483648.5, 0.0, 0.0, 0.0], "00000080"),
            (P::R32Uint, [16777217.0, 0.0, 0.0, 0.0], "01000001"),
            (P::R32Uint, [2147483649.0, 0.0, 0.0, 0.0], "01000080"),
            (P::R32Uint, [4294967040.5, 0.0, 0.0, 0.0], "00ffffff"),
            (P::R32Sint, [-2147483648.0, 0.0, 0.0, 0.0], "00000080"),
            (P::R32Sint, [2147483648.0, 0.0, 0.0, 0.0], "ffffff7f"),
            (P::R32Sint, [-2.5, 0.0, 0.0, 0.0], "feffffff"),
            (P::R32Sint, [2.5, 0.0, 0.0, 0.0], "02000000"),
            (P::R32Sint, [-3e9, 0.0, 0.0, 0.0], "00000080"),
            (P::R32Sint, [-16777217.0, 0.0, 0.0, 0.0], "fffffffe"),
            (P::R32Sint, [16777217.0, 0.0, 0.0, 0.0], "01000001"),
            (P::R32Sint, [-2147483520.5, 0.0, 0.0, 0.0], "80000080"),
            (P::R16Float, [0.1, 0.0, 0.0, 0.0], "662e"),
            (P::R16Float, [70000.0, 0.0, 0.0, 0.0], "007c"),
            (P::R16Float, [1e-8, 0.0, 0.0, 0.0], "0000"),
            (P::R16Float, [-0.0, 0.0, 0.0, 0.0], "0080"),
            (P::R16Float, [n, 0.0, 0.0, 0.0], "007e"),
            (P::R16Float, [-n, 0.0, 0.0, 0.0], "007e"),
            (P::RG16Float, [n, 1.0, 0.5, n], "007e003c"),
            (P::RG16Float, [1.0, n, 2.0, 0.25], "003c007e"),
            (P::R16Float, [65520.0, 0.0, 0.0, 0.0], "007c"),
            (P::R16Float, [1.001220703125, 0.0, 0.0, 0.0], "013c"),
            (
                P::R16Float,
                [1.0 + 2f64.powi(-11) + 2f64.powi(-40), 0.0, 0.0, 0.0],
                "003c",
            ),
            (P::R16Float, [1.0 + 2f64.powi(-11), 0.0, 0.0, 0.0], "003c"),
            (
                P::R16Float,
                [1.0 + 3.0 * 2f64.powi(-11), 0.0, 0.0, 0.0],
                "023c",
            ),
            (P::R32Float, [0.1, 0.0, 0.0, 0.0], "cdcccc3d"),
            (P::R32Float, [1e-40, 0.0, 0.0, 0.0], "c2160100"),
            (P::R32Float, [1e39, 0.0, 0.0, 0.0], "0000807f"),
            (P::R32Float, [n, 0.0, 0.0, 0.0], "0000c07f"),
            (P::R32Float, [-n, 0.0, 0.0, 0.0], "0000c0ff"),
            (
                P::R32Float,
                [1.0 + 2f64.powi(-24), 0.0, 0.0, 0.0],
                "0000803f",
            ),
            (
                P::RGBA16Float,
                [0.1, -2.5, 65504.0, 1e-5],
                "662e00c1ff7ba800",
            ),
            (P::RGBA16Float, [-n, -n, -n, -n], "007e007e007e007e"),
            (P::RGBA16Float, [n, 1.0, 0.5, n], "007e003c0038007e"),
            (P::RGBA16Float, [1.0, n, 2.0, 0.25], "003c007e00400034"),
            (P::RG16Snorm, [-1.5, 0.5, 0.0, 0.0], "01800040"),
            (P::RG16Snorm, [-1e-5, 0.99999, 0.0, 0.0], "0000ff7f"),
            (
                P::RGBA16Unorm,
                [0.5, 1.0 / 131070.0, 3.0 / 131070.0, 1.2],
                "008000000100ffff",
            ),
            (P::RGB10A2Unorm, [0.5, 0.25, 1.0, 0.5], "0002f4bf"),
            (P::RGB10A2Unorm, [0.0005, 0.9995, 0.33, 0.17], "01f82f55"),
            (P::BGR10A2Unorm, [0.5, 0.25, 0.75, 0.4], "ff020460"),
            (
                P::RGBA32Float,
                [0.1, -1e39, 1e-40, 3.0],
                "cdcccc3d000080ffc216010000004040",
            ),
            (
                P::RGBA32Uint,
                [1.5, 4294967296.0, -1.0, 7.0],
                "01000000ffffffff0000000007000000",
            ),
            (P::RG11B10Float, [0.1, -1.0, 70000.0, 0.0], "e602c0f7"),
            (P::RG11B10Float, [1e-6, n, inf, 0.0], "01003ff8"),
            (P::RG11B10Float, [n, n, n, n], "e0073ffc"),
            (P::RG11B10Float, [-n, -n, -n, -n], "e0073ffc"),
            (P::RG11B10Float, [n, 1.0, 0.5, n], "e0071e70"),
            (P::RG11B10Float, [1.0, n, 2.0, 0.25], "c0033f80"),
            (P::RG11B10Float, [65000.0, 0.5, 1e-5, 0.0], "be075c01"),
            (
                P::RG11B10Float,
                [
                    1.0 + 2f64.powi(-7),
                    1.0 + 2f64.powi(-6),
                    1.0 + 2f64.powi(-7),
                    0.0,
                ],
                "c00b1e78",
            ),
            (P::RGB9E5Float, [0.1, 0.5, 2.0, 0.0], "0c80008c"),
            (P::RGB9E5Float, [-1.0, 70000.0, 1e-3, 0.0], "00fe03f8"),
            (P::RGB9E5Float, [65408.0, 1e-8, 3.0, 0.0], "ff0100f8"),
            (P::RGB9E5Float, [n, inf, 1.0, 0.0], "00fe03f8"),
            (P::RGB9E5Float, [-n, -n, -n, -n], "00000000"),
            (P::RGB9E5Float, [n, 1.0, 0.5, n], "00000282"),
            (P::RGB9E5Float, [1.0, n, 2.0, 0.25], "8000008c"),
            (P::B5G6R5Unorm, [0.5, 0.25, 0.75, 1.0], "1782"),
            (P::A1BGR5Unorm, [0.5, 0.25, 0.75, 0.5], "2e82"),
            (P::A1BGR5Unorm, [0.5, 0.25, 0.75, 0.49], "2e82"),
            (P::ABGR4Unorm, [0.5, 0.25, 0.75, 0.1], "b184"),
            (P::BGR5A1Unorm, [0.5, 0.25, 0.75, 0.6], "17c1"),
        ];
        for &(pf, c, want) in cases {
            assert_eq!(hex(&bytes(pf, c)), want, "{pf:?} {c:?}");
        }
    }

    /// An `A8Unorm` attachment's clear writes the color's alpha (probe `a8`, an M3 Max under
    /// macOS 27.0.1: a clear of (0.2, 0.4, 0.6, 0.8) stores 204): in `A8_UNORM`'s alpha, and in an
    /// `R8_UNORM` image's red where the image holds Metal's alpha there; another format's image
    /// takes [`color`]'s value as it is.
    #[test]
    fn an_alpha_in_red_clear_takes_the_alpha_in_red() {
        let color = [0.2, 0.4, 0.6, 0.8];
        let route = ClearRoute::ThroughHalf;
        let code = 204.0 / 255.0;
        let held = |pf, format| {
            let value = held_color(pf, format, color, route).expect("a clear rule");
            // SAFETY: a normalized format's clear is `float32`.
            unsafe { value.float32 }
        };
        assert_eq!(held(P::A8Unorm, Format::A8_UNORM), [0.0, 0.0, 0.0, code]);
        assert_eq!(held(P::A8Unorm, Format::R8_UNORM), [code, 0.0, 0.0, 0.0]);
        assert_eq!(
            held(P::R8Unorm, Format::R8_UNORM),
            [51.0 / 255.0, 0.0, 0.0, 0.0]
        );
    }

    /// A texel's unsigned-integer clear writes its bytes: its little-endian words in the format
    /// of its size (`RGBA16Float`'s fresh pattern, half NaNs around a zero, as two 32-bit words).
    #[test]
    fn uint_texels_clear_to_their_bytes() {
        let rgba16 = [0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff];
        let rgba32: Vec<u8> = (1..=16).collect();
        for (texel, format, words) in [
            (&[0x34, 0x12][..], Format::R16_UINT, [0x1234, 0, 0, 0]),
            (&[0xff, 0xff, 0, 0][..], Format::R32_UINT, [0xffff, 0, 0, 0]),
            (&rgba16[..], Format::R32G32_UINT, [0xffff, u32::MAX, 0, 0]),
            (
                &rgba32[..],
                Format::R32G32B32A32_UINT,
                [0x0403_0201, 0x0807_0605, 0x0c0b_0a09, 0x100f_0e0d],
            ),
        ] {
            let (got, value) = uint_texel(texel).unwrap();
            assert_eq!(got, format, "{texel:?}");
            // SAFETY: every member of the union is four 32-bit values.
            assert_eq!(unsafe { value.uint32 }, words, "{texel:?}");
        }
        for size in [0, 1, 3, 6, 12, 32] {
            assert!(uint_texel(&vec![0xff; size]).is_none(), "{size} bytes");
        }
    }

    /// A clear of a `half` or small-float NaN writes Metal's texel through an integer format of its
    /// size (the bytes Metal wrote on the M3 Max and an M4), and no other clear does.
    #[test]
    fn nan_texels_are_metals() {
        let n = f64::NAN;
        let cases: &[(PixelFormat, [f64; 4], Format, &str)] = &[
            (P::R16Float, [n, 0.0, 0.0, 0.0], Format::R16_UINT, "007e"),
            (P::R16Float, [-n, 0.0, 0.0, 0.0], Format::R16_UINT, "007e"),
            (P::RG16Float, [n, 1.0, 0.5, n], Format::R32_UINT, "007e003c"),
            (
                P::RG16Float,
                [1.0, n, 2.0, 0.25],
                Format::R32_UINT,
                "003c007e",
            ),
            (
                P::RGBA16Float,
                [-n, -n, -n, -n],
                Format::R32G32_UINT,
                "007e007e007e007e",
            ),
            (
                P::RGBA16Float,
                [n, 1.0, 0.5, n],
                Format::R32G32_UINT,
                "007e003c0038007e",
            ),
            (
                P::RGBA16Float,
                [1.0, n, 2.0, 0.25],
                Format::R32G32_UINT,
                "003c007e00400034",
            ),
            (P::RG11B10Float, [n, n, n, n], Format::R32_UINT, "e0073ffc"),
            (
                P::RG11B10Float,
                [-n, -n, -n, -n],
                Format::R32_UINT,
                "e0073ffc",
            ),
            (
                P::RG11B10Float,
                [n, 1.0, 0.5, n],
                Format::R32_UINT,
                "e0071e70",
            ),
            (
                P::RG11B10Float,
                [1.0, n, 2.0, 0.25],
                Format::R32_UINT,
                "c0033f80",
            ),
            (
                P::RG11B10Float,
                [1e-6, n, f64::INFINITY, 0.0],
                Format::R32_UINT,
                "01003ff8",
            ),
        ];
        for &(pf, c, format, want) in cases {
            let value = color(pf, c, ClearRoute::ThroughHalf).unwrap();
            let (got, bits) = nan_texel(pf, &value, false).unwrap();
            // SAFETY: `nan_texel` writes `uint32`.
            let words = unsafe { bits.uint32 };
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(got, format, "{pf:?} {c:?}");
            assert_eq!(hex(&bytes[..want.len() / 2]), want, "{pf:?} {c:?}");
            assert!(bytes[want.len() / 2..].iter().all(|&b| b == 0), "{pf:?}");
        }
        for (pf, c) in [
            (P::RGBA16Float, [0.1, -2.5, 65504.0, 1e-5]),
            (P::RG11B10Float, [0.1, -1.0, 70000.0, 0.0]),
            (P::R32Float, [n, 0.0, 0.0, 0.0]),
            (P::RGBA32Float, [-n, n, 1.0, 0.0]),
            (P::RGB9E5Float, [n, 1.0, 0.5, n]),
            (P::RGBA8Unorm, [n, n, n, n]),
            (P::B5G6R5Unorm, [n, n, n, n]),
        ] {
            assert!(
                nan_texel(pf, &color(pf, c, ClearRoute::ThroughHalf).unwrap(), false).is_none(),
                "{pf:?}"
            );
            assert!(
                nan_texel(pf, &color(pf, c, ClearRoute::ThroughHalf).unwrap(), true).is_none(),
                "{pf:?}"
            );
        }
    }

    /// Where Metal keeps a NaN clear's sign (a level of 16 or more texels a side), a `half` NaN is
    /// `0x7e00` with the clear value's sign; the small floats, which have none, and every finite
    /// channel are as without it.
    #[test]
    fn signed_nan_texels_keep_the_sign_of_a_half_nan() {
        let n = f64::NAN;
        let cases: &[(PixelFormat, [f64; 4], &str)] = &[
            (P::RGBA16Float, [-n, -n, -n, -n], "00fe00fe00fe00fe"),
            (P::RGBA16Float, [n, 1.0, 0.5, -n], "007e003c003800fe"),
            (P::RGBA16Float, [n, 1.0, 0.5, n], "007e003c0038007e"),
            (P::R16Float, [-n, 0.0, 0.0, 0.0], "00fe"),
            (P::RG16Float, [-n, -2.5, 0.0, 0.0], "00fe00c1"),
            (P::RG11B10Float, [-n, -n, -n, -n], "e0073ffc"),
        ];
        for &(pf, c, want) in cases {
            let (_, bits) =
                nan_texel(pf, &color(pf, c, ClearRoute::ThroughHalf).unwrap(), true).unwrap();
            // SAFETY: `nan_texel` writes `uint32`.
            let words = unsafe { bits.uint32 };
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(hex(&bytes[..want.len() / 2]), want, "{pf:?} {c:?}");
        }
    }

    /// Metal's resolve of samples each holding a NaN clear's texel stores each NaN channel zero
    /// and every other channel as the texel holds it (an M4: `RGBA16Float` (NaN, 1, 0.5, -NaN)
    /// resolves to `0000 3c00 3800 0000`; `RG11B10Float` (NaN, 1, 0.5) to `701e0000`); a clear
    /// without such a NaN has none.
    #[test]
    fn resolved_nan_texels_zero_each_nan_channel() {
        let n = f64::NAN;
        let cases: &[(PixelFormat, [f64; 4], &str)] = &[
            (P::RGBA16Float, [n, 1.0, 0.5, -n], "0000003c00380000"),
            (P::RGBA16Float, [-n, -n, -n, -n], "0000000000000000"),
            (P::R16Float, [-n, 0.0, 0.0, 0.0], "0000"),
            (P::RG16Float, [-n, f64::INFINITY, 0.0, 0.0], "0000007c"),
            (P::RG11B10Float, [n, 1.0, 0.5, n], "00001e70"),
            (P::RG11B10Float, [n, n, n, n], "00000000"),
        ];
        for &(pf, c, want) in cases {
            let value = color(pf, c, ClearRoute::ThroughHalf).unwrap();
            let (format, bits) = resolved_nan_texel(pf, &value).unwrap();
            assert_eq!(format, nan_texel(pf, &value, false).unwrap().0, "{pf:?}");
            // SAFETY: `resolved_nan_texel` writes `uint32`.
            let words = unsafe { bits.uint32 };
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(hex(&bytes[..want.len() / 2]), want, "{pf:?} {c:?}");
        }
        let finite = color(
            P::RGBA16Float,
            [0.1, 1.0, 0.5, 1.0],
            ClearRoute::ThroughHalf,
        );
        assert!(resolved_nan_texel(P::RGBA16Float, &finite.unwrap()).is_none());
    }

    /// A 32-bit float NaN clears to the double's sign and top payload bits, quieted, as Metal's
    /// conversion (`0x7ff0000020000000` stores `0x7fc00001`, `0xfff8...`
    /// `0xffc00000`, `0x7ffc...` `0x7fe00000`, the signalling `0x7ff0...01` `0x7fc00000`), and
    /// `float_nan_texel` gives those bits; a clear without a NaN, or of another format, none.
    #[test]
    fn float_nan_texels_are_metal_s_conversion() {
        let nan = f64::from_bits;
        let cases: &[(PixelFormat, [f64; 4], &str)] = &[
            (
                P::R32Float,
                [nan(0x7ff0_0000_2000_0000), 0.0, 0.0, 0.0],
                "0100c07f",
            ),
            (
                P::R32Float,
                [nan(0xfff8_0000_0000_0000), 0.0, 0.0, 0.0],
                "0000c0ff",
            ),
            (
                P::R32Float,
                [nan(0x7ffc_0000_0000_0000), 0.0, 0.0, 0.0],
                "0000e07f",
            ),
            (
                P::R32Float,
                [nan(0x7ff0_0000_0000_0001), 0.0, 0.0, 0.0],
                "0000c07f",
            ),
            (
                P::RG32Float,
                [nan(0x7ff8_0000_0000_0000), 1.0, 0.0, 0.0],
                "0000c07f0000803f",
            ),
        ];
        for &(pf, c, want) in cases {
            let value = color(pf, c, ClearRoute::ThroughHalf).unwrap();
            let (format, bits) = float_nan_texel(pf, &value).unwrap();
            let format_for = |size: usize| uint_texel(&vec![0; size]).unwrap().0;
            assert!(format == format_for(want.len() / 2), "{pf:?}");
            // SAFETY: `float_nan_texel` writes `uint32`.
            let words = unsafe { bits.uint32 };
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            assert_eq!(hex(&bytes[..want.len() / 2]), want, "{pf:?} {c:?}");
        }
        let finite = color(P::R32Float, [0.5, 0.0, 0.0, 0.0], ClearRoute::ThroughHalf);
        assert!(float_nan_texel(P::R32Float, &finite.unwrap()).is_none());
        let half = color(P::RGBA16Float, [f64::NAN; 4], ClearRoute::ThroughHalf);
        assert!(float_nan_texel(P::RGBA16Float, &half.unwrap()).is_none());
    }

    /// Where Metal keeps a NaN clear's sign it keeps the top ten bits of the double's payload
    /// too, quieted (an M4: `0x7ffc...` stores `0x7f00`, `0x7fff...` `0x7fff`,
    /// `0xffff...` `0xffff`, the signalling `0x7ff4...` `0x7f00` and `0xfff0...01` `0xfe00`);
    /// elsewhere every such NaN is `0x7e00`.
    #[test]
    fn signed_nan_texels_keep_a_half_nan_s_payload() {
        let nan = f64::from_bits;
        let payloads = [
            nan(0x7ffc_0000_0000_0000),
            nan(0x7fff_ffff_ffff_ffff),
            nan(0xffff_ffff_ffff_ffff),
            nan(0x7ff4_0000_0000_0000),
        ];
        let low = [nan(0xfff0_0000_0000_0001), 0.0, 0.0, 0.0];
        let cases: &[(PixelFormat, [f64; 4], &str, &str)] = &[
            (
                P::RGBA16Float,
                payloads,
                "007fff7fffff007f",
                "007e007e007e007e",
            ),
            (P::R16Float, low, "00fe", "007e"),
            (
                P::RG16Float,
                [nan(0x7ff0_0000_2000_0000), 1.0, 0.0, 0.0],
                "007e003c",
                "007e003c",
            ),
        ];
        for &(pf, c, kept, dropped) in cases {
            let value = color(pf, c, ClearRoute::ThroughHalf).unwrap();
            for (signed, want) in [(true, kept), (false, dropped)] {
                let (_, bits) = nan_texel(pf, &value, signed).unwrap();
                // SAFETY: `nan_texel` writes `uint32`.
                let words = unsafe { bits.uint32 };
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                assert_eq!(hex(&bytes[..want.len() / 2]), want, "{pf:?} {c:?} {signed}");
            }
        }
    }

    /// The Vulkan clear value of each converted texel is one Vulkan's conversion maps back onto
    /// it: an exact value, or one within a hair of the normalized integer.
    #[test]
    fn vulkan_values_land_on_metals_texels() {
        // SAFETY (union reads): each arm reads the member `color` wrote for that format.
        let v = color(
            P::RGBA8Unorm,
            [0.5, 0.25, 1.5, -0.5],
            ClearRoute::ThroughHalf,
        )
        .unwrap();
        let f = unsafe { v.float32 };
        for (x, k) in f.iter().zip([128.0, 64.0, 255.0, 0.0]) {
            assert!((f64::from(*x) * 255.0 - k).abs() < 1e-3, "{f:?}");
        }
        let v = color(
            P::R32Uint,
            [16777217.0, 0.0, 0.0, 0.0],
            ClearRoute::ThroughHalf,
        )
        .unwrap();
        assert_eq!(unsafe { v.uint32 }[0], 16777217);
        let v = color(P::R8Sint, [-3.7, 0.0, 0.0, 0.0], ClearRoute::ThroughHalf).unwrap();
        assert_eq!(unsafe { v.int32 }[0], -3);
        let v = color(P::R16Float, [0.1, 0.0, 0.0, 0.0], ClearRoute::ThroughHalf).unwrap();
        assert_eq!(f32_to_f16(unsafe { v.float32 }[0]), 0x2e66);
        let v = color(
            P::RGBA8UnormSRGB,
            [0.5, 0.214, 0.0031308, 1.0],
            ClearRoute::ThroughHalf,
        )
        .unwrap();
        let f = unsafe { v.float32 };
        for (x, k) in f.iter().take(3).zip([188u32, 127, 10]) {
            assert_eq!(srgb_encode(f64::from(*x)), k, "{f:?}");
        }
        assert!(color(P::Depth32Float, [0.0; 4], ClearRoute::ThroughHalf).is_none());
    }

    /// Metal's clears of an `ARGB8Unorm` attachment, its bytes alpha, red, green and blue (an M4
    /// under macOS 27.0.1, plain and validated alike): the clear value lands on
    /// each byte in the image's red, green, blue and alpha.
    #[test]
    fn argb8_clears_are_metals_bytes_rotated() {
        let cases: [([f64; 4], [u8; 4]); 8] = [
            ([0.1, 0.2, 0.3, 0.4], [0x66, 0x1a, 0x33, 0x4d]),
            ([1.0, 0.0, 0.0, 0.0], [0x00, 0xff, 0x00, 0x00]),
            ([0.0, 0.0, 0.0, 1.0], [0xff, 0x00, 0x00, 0x00]),
            (
                [127.5 / 255.0, 0.5 / 255.0, 254.5 / 255.0, 1.5 / 255.0],
                [0x02, 0x80, 0x01, 0xfe],
            ),
            ([-0.5, 1.5, 0.25, 0.75], [0xbf, 0x00, 0xff, 0x40]),
            ([0.6, 0.4, 0.2, 0.0], [0x00, 0x99, 0x66, 0x33]),
            ([0.004, 0.996, 0.502, 0.498], [0x7f, 0x01, 0xfe, 0x80]),
            ([f64::NAN, 0.5, 0.5, 0.5], [0x80, 0x00, 0x80, 0x80]),
        ];
        for (clear, bytes) in cases {
            // SAFETY (union read): `color` writes `float32` for a normalized format.
            let f = unsafe {
                color(P::ARGB8Unorm, clear, ClearRoute::ThroughHalf)
                    .unwrap()
                    .float32
            };
            for (x, b) in f.iter().zip(bytes) {
                assert!(
                    (f64::from(*x) * 255.0 - f64::from(b)).abs() < 1e-3,
                    "{clear:?}: {f:?}"
                );
            }
        }
    }

    /// Depth goes through `float`, `Depth16Unorm` truncates, stencil keeps 8 bits (probes 1, 2);
    /// a 16-bit clear value maps to Metal's integer whether the device rounds or truncates.
    #[test]
    fn depth_and_stencil_clears_are_metals() {
        assert_eq!(depth(P::Depth32Float, 0.3).to_bits(), 0x3e99999a);
        assert_eq!(depth(P::Depth32FloatStencil8, 0.1).to_bits(), 0x3dcccccd);
        for (x, k) in [
            (0.5, 0x7fff),
            (1.0 / 131070.0, 0),
            (3.0 / 131070.0, 1),
            (0.3, 0x4ccc),
            (1.0, 0xffff),
            (0.1, 0x1999),
        ] {
            // Rounding to nearest and truncating both store `k`.
            let v = f64::from(depth(P::Depth16Unorm, x)) * 65535.0;
            assert_eq!(
                (v.round(), v.trunc()),
                (f64::from(k), f64::from(k)),
                "{x} {v}"
            );
        }
        for (x, k) in [(0x5a, 0x5a), (0x1ff, 0xff), (0xffff_ffff, 0xff), (0x100, 0)] {
            assert_eq!(stencil(x), k);
        }
    }

    /// The clear colors of our probe `bfp.m`, in its order: 2,048 colors of four
    /// channels from its generator (NaNs, specials, `half` values and the midpoints between two
    /// with their neighbours, bfloat midpoints with theirs, and random values), the first 38 of
    /// them the specials in every channel.
    fn bfloat_clear_colors() -> Vec<[f64; 4]> {
        const SPECIALS: [f64; 38] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            65504.0,
            65_519.999_999_999_99,
            65520.0,
            65536.0,
            1e10,
            -1e10,
            3.402_823_466_385_288_6e38,
            1e39,
            -1e39,
            1e-30,
            -1e-30,
            5.960_464_477_539_063e-8,
            2.980_232_238_769_531_2e-8,
            // The probe's `2.9802322387695315e-08`, the same double.
            2.980_232_238_769_531_2e-8,
            6.103_515_625e-5,
            6.097_555_160_522_461e-5,
            1.401_298_464_324_817e-45,
            1.0 + 1.0 / 256.0,
            1.0 + 3.0 / 256.0,
            1.0 + 1.0 / 2048.0,
            1.0 + 3.0 / 2048.0,
            1.0 + 1.0 / 2048.0 + 1.0 / 1_073_741_824.0,
            1.0 / 3.0,
            255.5,
            0.1,
            0.2,
            0.3,
            1e5,
            70000.0,
            100000.0,
            1e-8,
            1e-6,
        ];
        const NANS: [u64; 6] = [
            0x7ff8_0000_0000_0000,
            0xfff8_0000_0000_0000,
            0x7ff0_0000_0000_0001,
            0x7ff4_0000_0000_0000,
            0xfff0_0400_0000_0000,
            0x7ff8_4000_0000_0000,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut value = move || -> f64 {
            let r = next();
            let signed = |v: f64| if (r >> 30) & 1 == 1 { -v } else { v };
            match r % 8 {
                0 => f64::from_bits(NANS[((r >> 8) % 6) as usize]),
                1 | 2 => SPECIALS[((r >> 8) % 38) as usize],
                3 => {
                    // A `half` value, or the midpoint between it and the next, or a neighbour of
                    // the midpoint.
                    let h = (next() & 0x7bff) as u16;
                    let a = f64::from(f16_to_f32(h));
                    let b = f64::from(f16_to_f32(h + 1));
                    let m = (a + b) / 2.0;
                    signed(match (r >> 20) % 3 {
                        0 => a,
                        1 => m,
                        _ if (r >> 24) & 1 == 1 => m.next_up(),
                        _ => m.next_down(),
                    })
                }
                4 => {
                    // A bfloat midpoint, or a neighbour of it.
                    let f = f64::from(f32::from_bits((next() & 0x7f7f_0000) as u32 | 0x8000));
                    signed(match (r >> 20) % 3 {
                        0 => f,
                        1 => f.next_up(),
                        _ => f.next_down(),
                    })
                }
                _ => {
                    let u = (next() >> 11) as f64 / 9_007_199_254_740_992.0;
                    match (r >> 20) % 3 {
                        0 => -2.0 + 5.0 * u,
                        1 => -70000.0 + 140_000.0 * u,
                        // `ldexp(u, e)`: the power of two exact.
                        _ => u * f64::from_bits((1023 + (r >> 32) % 200 - 100) << 52),
                    }
                }
            }
        };
        let mut colors: Vec<[f64; 4]> = (0..2048)
            .map(|_| std::array::from_fn(|_| value()))
            .collect();
        for (color, special) in colors.iter_mut().zip(SPECIALS) {
            *color = [special; 4];
        }
        colors
    }

    /// FNV-1a of `bytes`.
    fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
        bytes.fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
    }

    /// A render pass's `Clear` of a bfloat color attachment (`R16Bfloat`, `RG16Bfloat`,
    /// `RGBA16Bfloat`; the consumer's passes) stores Metal's codes for every color of our probe
    /// `bfp.m` (an Apple M4 under macOS 27.0.1: 2,048 colors, each into a slice
    /// of its own): each channel through `float` and `half`, each to nearest even, then the bfloat
    /// code nearest, ties to even, a NaN `0x7fc0` (`ExtendedRange::clear_texel`). Metal's answer
    /// is FNV-1a of each format's codes, color after color; the colors' own (their bits, as the
    /// probe recorded them) check the generator first.
    #[test]
    fn bfloat_clears_are_metal_s() {
        let colors = bfloat_clear_colors();
        let bits = colors
            .iter()
            .flat_map(|c| c.iter().flat_map(|v| v.to_le_bytes()));
        assert_eq!(fnv(bits), 0xde31_e66e_0a02_d326, "the probe's colors");
        for (pf, channels, metal) in [
            (P::R16Bfloat, 1, 0x787c_c90b_99ff_56ac),
            (P::RG16Bfloat, 2, 0x38a9_4edd_d5c8_b90a),
            (P::RGBA16Bfloat, 4, 0xe219_6f7b_bcab_e264),
        ] {
            let codes: Vec<u16> = colors
                .iter()
                .flat_map(|&c| {
                    // SAFETY: an extended-range attachment's clear value is its `uint32` texel.
                    let uint32 = unsafe {
                        color(pf, c, ClearRoute::ThroughHalf)
                            .expect("a clear rule")
                            .uint32
                    };
                    uint32
                        .into_iter()
                        .take(channels)
                        .map(|w| u16::try_from(w).expect("a code"))
                        .collect::<Vec<_>>()
                })
                .collect();
            let bytes = codes.iter().flat_map(|c| c.to_le_bytes());
            assert_eq!(fnv(bytes), metal, "{pf:?}");
        }
        // A few of Metal's codes: a bfloat tie that rounds up to even, a `half` tie that
        // rounds down, the largest `half` (whose bfloat rounds up), a double whose `float` is the
        // `half` overflow (an infinity), `half`'s least and the tie below it, the signed zero, and
        // a negative NaN.
        for (v, metal) in [
            (1.0 + 3.0 / 256.0, 0x3f82),
            (1.0 + 1.0 / 2048.0, 0x3f80),
            (65504.0, 0x4780),
            (65_519.999_999_999_99, 0x7f80),
            (5.960_464_477_539_063e-8, 0x3380),
            (2.980_232_238_769_531_2e-8, 0x0000),
            (-1e-30, 0x8000),
            (f64::from_bits(0xfff8_0000_0000_0000), 0x7fc0),
        ] {
            // SAFETY: as above.
            let uint32 = unsafe {
                color(P::R16Bfloat, [v; 4], ClearRoute::ThroughHalf)
                    .expect("a clear rule")
                    .uint32
            };
            assert_eq!(uint32[0], metal, "{v:e}");
        }
    }

    /// Each of Metal's routes clears a bfloat attachment by its rule (an M4 under macOS 27.0.1; the
    /// consumer's M3 Max under 26.7 alike): straight to bfloat (`Nearest`) rounds the color's
    /// `float`, subnormals kept, and keeps a NaN's sign and top payload bits; through `half`
    /// (`ThroughHalf`) loses them. The consumer's colors, the ties around `1 + 2^-8`, whose `half`
    /// is a bfloat tie, and the probe's NaNs, whose codes are the M4's.
    #[test]
    fn each_route_clears_bfloat_by_its_rule() {
        let at = |route: ClearRoute, pf: P, c: [f64; 4]| -> [u32; 4] {
            let value = color(pf, c, route).expect("a clear rule");
            // SAFETY: an extended-range attachment's clear value is its `uint32` texel.
            unsafe { value.uint32 }
        };
        let consumers = [-3e38, 1e-40, 70000.0, -2.5];
        let ties = [
            1.0 + 1.0 / 256.0,
            1.0 + 3.0 / 256.0,
            1.0 + 1.0 / 256.0 + 2f64.powi(-23),
            -0.0,
        ];
        for (route, want, tied) in [
            (
                ClearRoute::Straight,
                [0xff62, 0x0001, 0x4789, 0xc020],
                [0x3f80, 0x3f82, 0x3f81, 0x8000],
            ),
            (
                ClearRoute::ThroughHalf,
                [0xff80, 0x0000, 0x7f80, 0xc020],
                [0x3f80, 0x3f82, 0x3f80, 0x8000],
            ),
        ] {
            assert_eq!(at(route, P::RGBA16Bfloat, consumers), want, "{route:?}");
            assert_eq!(at(route, P::RGBA16Bfloat, ties), tied, "{route:?}");
            assert_eq!(
                at(route, P::RG16Bfloat, consumers),
                [want[0], want[1], 0, 0],
                "{route:?}"
            );
            assert_eq!(
                at(route, P::R16Bfloat, consumers),
                [want[0], 0, 0, 0],
                "{route:?}"
            );
        }
        // The probe's NaN doubles (`bcs.m`), and the codes the M4 stored for each in a 16 x 16
        // private `R16Bfloat` target cleared by a pass with no draw call; a 1 x 1 one stored
        // `0x7fc0` for each.
        for (bits, nearest) in [
            (0x7ff8_0000_0000_0000_u64, 0x7fc0),
            (0xfff8_0000_0000_0000, 0xffc0),
            (0x7ff8_4000_0000_0000, 0x7fc2),
            (0xfffc_0000_0000_0000, 0xffe0),
            (0x7ff8_0000_0000_0001, 0x7fc0),
            (0x7fff_ffff_ffff_ffff, 0x7fff),
            (0xffff_ffff_ffff_ffff, 0xffff),
            (0x7ff0_0000_0000_0001, 0x7fc0),
            (0xfff0_0000_0000_0001, 0xffc0),
            (0x7ff4_0000_0000_0000, 0x7fe0),
            (0xfff0_0400_0000_0000, 0xffc0),
            (0x7ff7_ffff_ffff_ffff, 0x7fff),
            (0xfff7_e000_0000_0000, 0xffff),
        ] {
            let nan = f64::from_bits(bits);
            let c = [nan, 1.0, -2.5, 0.5];
            assert_eq!(
                at(ClearRoute::Straight, P::RGBA16Bfloat, c),
                [nearest, 0x3f80, 0xc020, 0x3f00],
                "{bits:#x}"
            );
            assert_eq!(
                at(ClearRoute::ThroughHalf, P::RGBA16Bfloat, c),
                [0x7fc0, 0x3f80, 0xc020, 0x3f00],
                "{bits:#x}"
            );
        }
    }

    /// The half conversion rounds to nearest even at every boundary it meets.
    #[test]
    fn halves_round_to_nearest_even() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(3.0 * 2f32.powi(-25)), 0x0002);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        for h in [
            0x0001u16, 0x03ff, 0x0400, 0x3c00, 0x3c01, 0x7bff, 0x8001, 0xfbff,
        ] {
            assert_eq!(f32_to_f16(f16_to_f32(h)), h);
        }
    }
}
