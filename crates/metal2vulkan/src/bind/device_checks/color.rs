//! The device's checks of six color-write deviations Metal's rule decides (`DriverDeviation`'s
//! `MetalColorWrites`, `MetalFloatColorNans`, `MetalHalfColorNans`, `HalfColorOutputBits`,
//! `MetalSmallFloatColorWrites` and `MetalNorm16ColorWrites`). Each is a set of draws
//! ([`ColorDraw`]): one triangle over a target of [`ColorDraw::WIDTH`] pixels a row, whose fragment
//! writes each pixel's input to every channel of its color output as computed, the form
//! translation writes where the profile takes the deviation, plain or through blending with
//! factors one and zero. The runtime reads the target back; Metal's texels by its rule
//! ([`ColorDraw::expected`], computed here) are Metal's answer.
//!
//! **Metal's rule**, closed from probes on the M3 Max and the M4 (`lower::arith::color_write`
//! writes the same codes in the shader where no deviation is taken):
//!
//! - a unorm channel of `b` bits: `x` clamped to `[0, 1]` (a NaN 0), times `2^b - 1` exactly,
//!   rounded to nearest, a tie up;
//! - a snorm channel: `x` clamped to `[-1, 1]` (a NaN 0), times `2^(b-1) - 1` exactly, rounded to
//!   nearest, a tie away from zero; through blending a NaN stores -1.0's code;
//! - an sRGB channel: `x` clamped to `[0, 1]` (a NaN 0), times 4095 exactly, rounded to nearest, a
//!   tie up, encoded by Metal's thresholds ([`SRGB_RESOLVE_THRESHOLDS`]); an sRGB format's alpha is
//!   unorm;
//! - a half-float channel: `x` rounded toward zero to `half`, a finite value past 65504 stored as
//!   65504, an infinity as itself, a NaN as `0x7e00`; a `half` output's own bits, a NaN's payload
//!   included;
//! - `RG11B10Float`'s unsigned 11- and 10-bit floats: `x` rounded toward zero, a negative value
//!   stored as 0, a finite value past the largest as the largest, an infinity as itself, a NaN as
//!   the quiet NaN.
//!
//! **The draws** of each fact, every input of a unorm, snorm or sRGB channel each code's exact
//! value and each tie between neighbouring codes with two `f32` steps either side, of a
//! half-float or small-float channel each value and each tie with one step either side, beside
//! specials (zeros, ones, the half range's edges, the infinities, NaNs):
//!
//! - `MetalColorWrites`: `R8Unorm`, `R8Snorm`, `R16Unorm`, `R16Snorm`, `RGB10A2Unorm`,
//!   `RGBA8Unorm_sRGB` and `R16Float`, every input, NaNs included.
//! - `MetalFloatColorNans`: `R16Float`, `RG16Float` and `RGBA16Float`, the `float` NaN
//!   `0x7fc00000` (the NaN translation's rounding writes).
//! - `MetalHalfColorNans`: `RGBA16Float`, every `half` NaN widened exactly to a `float` output: the
//!   half's own bits.
//! - `HalfColorOutputBits`: `RGBA16Float`, every `half` through a 16-bit output (where the device
//!   has `storageInputOutput16`, `shaderInt16` and `shaderFloat16`): its own bits.
//! - `MetalSmallFloatColorWrites`: `R16Float` and `RG11B10Float`, every input but a NaN.
//! - `MetalNorm16ColorWrites`: `R16Unorm` and `R16Snorm`, every input, a blended snorm channel's NaN
//!   written as -1.0 (the value translation chooses there).

use std::sync::Arc;

use super::kernel::finish;
use crate::bind::color_resolve::SRGB_RESOLVE_THRESHOLDS;
use crate::bind::sample_planes::merge_vertex_typed_module;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{DriverDeviation, Error};
use crate::float16::f16_bits_to_f32;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule,
    MemAccess, StorageClass, Type, TypedModule,
};
use crate::vk::Format;

/// The color-write deviations the draws check, in the order [`color_write_draws`] takes them.
pub const COLOR_WRITE_FACTS: [DriverDeviation; 6] = [
    DriverDeviation::MetalColorWrites,
    DriverDeviation::MetalFloatColorNans,
    DriverDeviation::MetalHalfColorNans,
    DriverDeviation::HalfColorOutputBits,
    DriverDeviation::MetalSmallFloatColorWrites,
    DriverDeviation::MetalNorm16ColorWrites,
];

/// The fragment output a draw writes through.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ColorOutput {
    /// A `vec4` output, each pixel's input word read as a `float` ([`ColorWriteModules::float`]).
    Float,
    /// A 16-bit `f16vec4` output, each pixel's input word's low 16 bits read as a `half`
    /// ([`ColorWriteModules::half`]).
    Half,
}

/// One draw of a color-write check (the module docs): `format`'s target of [`ColorDraw::WIDTH`]
/// by [`height`](Self::height) pixels, cleared to zero, drawn once with the module of `output`,
/// plain or through blending with factors one and zero (`ONE`, `ZERO`, `ADD`, for color and
/// alpha) where `blended`, every channel written, and read back.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorDraw {
    /// The target's format.
    pub format: Format,
    /// Whether the draw blends with factors one and zero.
    pub blended: bool,
    /// The output it writes through.
    pub output: ColorOutput,
    /// Rows of the target.
    pub height: u32,
    /// Each pixel's input word, row by row: the storage buffer at
    /// [`ColorWriteModules::VALUES_BINDING`].
    pub values: Vec<u32>,
    /// Bytes of one texel of `format`.
    pub texel_bytes: u32,
    /// Metal's texels, row by row, tightly packed.
    pub expected: Vec<u8>,
}

impl ColorDraw {
    /// Pixels of a row of every draw's target.
    pub const WIDTH: u32 = 1024;

    /// Whether `texels`, read back from the target row by row, tightly packed, are Metal's.
    pub fn answers_metal(&self, texels: &[u8]) -> bool {
        texels == self.expected.as_slice()
    }

    /// How `texels`, read back from the target row by row, tightly packed, differ from Metal's, in a
    /// few words for the other answer's statement (`CheckAnswer::NotMetals`): how many texels
    /// differ, and the first, its position, the bytes read and Metal's. `None` for Metal's.
    pub fn difference(&self, texels: &[u8]) -> Option<String> {
        super::texel_difference(texels, &self.expected, self.texel_bytes, Self::WIDTH)
    }
}

/// The modules of the color-write draws: a vertex stage that draws one triangle over the viewport
/// from `VertexIndex` 0 to 2, and the fragment stages of each output (the module docs). Each
/// fragment reads the storage buffer of 4-byte words at set 0, binding
/// [`ColorWriteModules::VALUES_BINDING`], at word `y * WIDTH + x` of its pixel, and writes it to
/// every lane of the color output at `Location` 0.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorWriteModules {
    /// The vertex stage's SPIR-V, entry `main`.
    pub vertex: Arc<[u32]>,
    /// The fragment stage of a `float` output, entry `main`.
    pub float: Arc<[u32]>,
    /// The fragment stage of a 16-bit output, entry `main`, where the device has
    /// `storageInputOutput16`, `shaderInt16` and `shaderFloat16`.
    pub half: Option<Arc<[u32]>>,
    /// What the modules need from the device (within its profile).
    pub requirements: Requirements,
}

impl ColorWriteModules {
    /// The binding of the input words, in set 0.
    pub const VALUES_BINDING: u32 = 0;
}

/// The modules of the color-write draws on `device` (the module docs).
pub fn color_write_modules(device: &Device) -> Result<ColorWriteModules, Error> {
    let (vertex, vertex_needs) = finish(device, merge_vertex_typed_module()?, "color draw vertex")?;
    let (float, float_needs) = finish(
        device,
        fragment_typed_module(ColorOutput::Float)?,
        "color draw fragment",
    )?;
    let profile = device.profile();
    let mut requirements = profile.union_requirements(&vertex_needs, &float_needs);
    let leaf = &profile.fields().leaf;
    let half = match leaf.input_output_16bit && leaf.int16 && leaf.float16 {
        true => {
            let (half, half_needs) = finish(
                device,
                fragment_typed_module(ColorOutput::Half)?,
                "color draw 16-bit fragment",
            )?;
            requirements = profile.union_requirements(&requirements, &half_needs);
            Some(half)
        }
        false => None,
    };
    Ok(ColorWriteModules {
        vertex,
        float,
        half,
        requirements,
    })
}

/// A `u32` constant as a value.
fn word(f: &mut FunctionBuilder<'_>, u32_: TypeId, bits: u32) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(bits)),
    })?;
    f.module().const_value(c)
}

/// The values buffer: `Block { u32 words[]; }` (std430) at buffer slot
/// [`ColorWriteModules::VALUES_BINDING`] of the fragment stage.
fn values_buffer(b: &mut ModuleBuilder, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    let slot = ResourceSlot {
        stage: Stage::Fragment,
        kind: ResourceKind::Buffer,
        index: ColorWriteModules::VALUES_BINDING,
    };
    b.resource(slot, StorageClass::StorageBuffer, block)
}

/// The word of the values buffer pixel `(x, y)` reads (`values` points to the buffer's block):
/// word `y · WIDTH + x`, or word 0 where that lies at or past the buffer's end, so that every
/// invocation's load stays inside the buffer. A helper invocation of a quad that reaches past
/// the draw's last row (a 2x2 quad, llvmpipe's 4x4 blocks) runs at `y` = the target's height,
/// where an unbounded load would read past the buffer, which only robust buffer access makes
/// defined. A covered pixel reads its own word.
fn pixel_word(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    values: Value,
    (x, y): (Value, Value),
) -> BResult<Value> {
    let width = word(f, u32_, ColorDraw::WIDTH)?;
    let row = f.binary(BinOp::IMul, y, width)?;
    let at = f.binary(BinOp::IAdd, row, x)?;
    let len = f.array_length(values, 0)?;
    let inside = f.binary(BinOp::ULessThan, at, len)?;
    let zero = word(f, u32_, 0)?;
    let at = f.select(inside, at, zero)?;
    let p = f.access_chain(values, &[zero, at])?;
    f.load(p, MemAccess::default())
}

/// The fragment stage of `output` as a typed module (the module docs).
fn fragment_typed_module(output: ColorOutput) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
    let (lane, out_ty) = match output {
        ColorOutput::Float => (f32_, vec4),
        ColorOutput::Half => {
            let f16_ = b.ty(Type::Float { width: 16 })?;
            (f16_, b.ty(Type::Vector { elem: f16_, len: 4 })?)
        }
    };
    let values = values_buffer(&mut b, u32_)?;
    let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, vec4)?;
    let color = b.fixed_io(0, IoDir::Output, out_ty, false)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let coord = f.module().global_value(coord)?;
    let coord = f.load(coord, MemAccess::default())?;
    // A pixel's center is at `(x + 0.5, y + 0.5)`: the conversion truncates to its integer.
    let x = f.composite_extract(coord, &[0])?;
    let y = f.composite_extract(coord, &[1])?;
    let x = f.convert(ConvOp::ConvertFToU, x, u32_)?;
    let y = f.convert(ConvOp::ConvertFToU, y, u32_)?;
    let values = f.module().global_value(values)?;
    let w = pixel_word(&mut f, u32_, values, (x, y))?;
    let v = match output {
        ColorOutput::Float => f.convert(ConvOp::Bitcast, w, f32_)?,
        ColorOutput::Half => {
            let u16_ = f.module().ty(Type::Int {
                width: 16,
                signed: false,
            })?;
            let bits = f.convert(ConvOp::UConvert, w, u16_)?;
            f.convert(ConvOp::Bitcast, bits, lane)?
        }
    };
    let texel = f.composite_construct(out_ty, &[v, v, v, v])?;
    let color = f.module().global_value(color)?;
    f.store(color, texel, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

// ---------------------------------------------------------------------------------------------
// Metal's rule.

/// How a channel stores a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel {
    Unorm(u32),
    Snorm(u32),
    Srgb,
    Half,
    /// An unsigned float of a five-bit exponent and this many fraction bits.
    UnsignedFloat(u32),
}

/// A format of the draws' texel: its bytes and each channel's bit offset and encoding.
fn layout(format: Format) -> (u32, &'static [(u32, Channel)]) {
    use Channel::*;
    match format {
        Format::R8_UNORM => (1, &[(0, Unorm(8))]),
        Format::R8_SNORM => (1, &[(0, Snorm(8))]),
        Format::R16_UNORM => (2, &[(0, Unorm(16))]),
        Format::R16_SNORM => (2, &[(0, Snorm(16))]),
        Format::A2B10G10R10_UNORM_PACK32 => (
            4,
            &[
                (0, Unorm(10)),
                (10, Unorm(10)),
                (20, Unorm(10)),
                (30, Unorm(2)),
            ],
        ),
        Format::R8G8B8A8_SRGB => (4, &[(0, Srgb), (8, Srgb), (16, Srgb), (24, Unorm(8))]),
        Format::R16_SFLOAT => (2, &[(0, Half)]),
        Format::R16G16_SFLOAT => (4, &[(0, Half), (16, Half)]),
        Format::R16G16B16A16_SFLOAT => (8, &[(0, Half), (16, Half), (32, Half), (48, Half)]),
        Format::B10G11R11_UFLOAT_PACK32 => (
            4,
            &[
                (0, UnsignedFloat(6)),
                (11, UnsignedFloat(6)),
                (22, UnsignedFloat(5)),
            ],
        ),
        // No draw takes another format.
        _ => (0, &[]),
    }
}

/// The bits of a channel.
fn channel_bits(channel: Channel) -> u32 {
    match channel {
        Channel::Unorm(bits) | Channel::Snorm(bits) => bits,
        Channel::Srgb => 8,
        Channel::Half => 16,
        Channel::UnsignedFloat(mantissa) => mantissa + 5,
    }
}

/// `x` (finite, at least 0) times `max` exactly, rounded to nearest, a tie up.
fn round_product(x: f32, max: u32) -> u32 {
    let bits = x.to_bits();
    let exponent = (bits >> 23) & 0xff;
    let (mantissa, scale) = match exponent {
        0 => (u128::from(bits & 0x7f_ffff), 149u32),
        _ => (u128::from((bits & 0x7f_ffff) | 0x80_0000), 150 - exponent),
    };
    // x * max + 1/2 = (2 * mantissa * max + 2^scale) / 2^(scale + 1), scale at most 150.
    if scale >= 100 {
        return 0;
    }
    let twice = 2 * mantissa * u128::from(max) + (1u128 << scale);
    (twice >> (scale + 1)) as u32
}

/// `x` rounded toward zero to `half`: a finite value past 65504 is 65504, a NaN `0x7e00`.
fn half_toward_zero(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7e00;
    }
    let sign = if x.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = x.abs();
    if a.is_infinite() {
        return sign | 0x7c00;
    }
    if a >= 65504.0 {
        return sign | 0x7bff;
    }
    if a < 2f32.powi(-14) {
        // Exact: a power-of-two scaling of a value below 2^-14.
        return sign | (a * 2f32.powi(24)).floor() as u16;
    }
    let bits = a.to_bits();
    let exponent = ((bits >> 23) & 0xff) as u16 - (127 - 15);
    sign | (exponent << 10) | ((bits >> 13) & 0x3ff) as u16
}

/// `x` rounded toward zero to an unsigned float of `mantissa` fraction bits.
fn unsigned_float_toward_zero(x: f32, mantissa: u32) -> u64 {
    let infinity = 0x1fu64 << mantissa;
    if x.is_nan() {
        return infinity | (1 << (mantissa - 1));
    }
    if x.is_sign_negative() {
        return 0;
    }
    if x.is_infinite() {
        return infinity;
    }
    let a = f64::from(x);
    let m = mantissa as i32;
    if a >= (2.0 - 2f64.powi(-m)) * 32768.0 {
        return infinity - 1;
    }
    if a < 2f64.powi(-14) {
        return (a / 2f64.powi(-14 - m)).floor() as u64;
    }
    let exponent = ((x.to_bits() >> 23) & 0xff) as i32 - 127;
    let k = (a / 2f64.powi(exponent - m)).floor() as u64;
    (((exponent + 15) as u64) << mantissa) | (k - (1 << mantissa))
}

/// Metal's code of `x` in `channel`, written plain or through blending (the module docs).
fn metal_code(channel: Channel, x: f32, blended: bool) -> u64 {
    let unit = |x: f32| if x >= 0.0 { x.min(1.0) } else { 0.0 };
    match channel {
        Channel::Unorm(bits) => u64::from(round_product(unit(x), (1 << bits) - 1)),
        Channel::Snorm(bits) => {
            let max = (1u32 << (bits - 1)) - 1;
            let code = match (x.is_nan(), blended) {
                (true, true) => -i64::from(max),
                (true, false) => 0,
                (false, _) => {
                    let magnitude = i64::from(round_product(x.abs().min(1.0), max));
                    if x < 0.0 {
                        -magnitude
                    } else {
                        magnitude
                    }
                }
            };
            (code as u64) & ((1 << bits) - 1)
        }
        Channel::Srgb => {
            let index = round_product(unit(x), 4095);
            SRGB_RESOLVE_THRESHOLDS.partition_point(|&t| u32::from(t) <= index) as u64
        }
        Channel::Half => u64::from(half_toward_zero(x)),
        Channel::UnsignedFloat(mantissa) => unsigned_float_toward_zero(x, mantissa),
    }
}

/// The `half` an exactly widened `half` `x` is, a NaN's payload included.
fn narrowed(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    match x.is_nan() || x.is_infinite() {
        true => sign | 0x7c00 | ((bits >> 13) & 0x3ff) as u16,
        false => sign | half_toward_zero(x.abs()),
    }
}

/// What a draw's channels hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    /// Metal's code of the input by the channel's rule.
    Metal,
    /// The input's own `half` bits: a `half` widened exactly, or a 16-bit output's.
    OwnBits,
}

// ---------------------------------------------------------------------------------------------
// The inputs.

/// `x` and its neighbours in `f32` steps, `steps` either side.
fn around(x: f32, steps: u32) -> Vec<f32> {
    let mut out = vec![x];
    let (mut down, mut up) = (x, x);
    for _ in 0..steps {
        down = down.next_down();
        up = up.next_up();
        out.extend([down, up]);
    }
    out
}

/// Zeros, ones, halves, the half range's edges, the smallest normal and denormal, large values,
/// the infinities and NaNs.
pub(super) fn specials(nans: bool) -> Vec<f32> {
    let mut out = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.0,
        -2.0,
        1e-45,
        -1e-45,
        1.175_494_2e-38,
        -1.175_494_2e-38,
        1e-10,
        -1e-10,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65504.0,
        65519.0,
        65520.0,
        1e10,
        -1e10,
    ];
    if nans {
        out.extend(
            [0x7fc0_0000u32, 0xffc0_0000, 0x7f80_0001, 0x7fff_ffff]
                .into_iter()
                .map(f32::from_bits),
        );
    }
    out
}

/// Unorm of `bits`: every code's exact value, and every tie with two `f32` steps either side.
pub(super) fn unorm(bits: u32) -> Vec<f32> {
    let max = f64::from((1u32 << bits) - 1);
    let mut out: Vec<f32> = (0..1u32 << bits)
        .map(|k| (f64::from(k) / max) as f32)
        .collect();
    for k in 0..(1u32 << bits) - 1 {
        out.extend(around(((f64::from(k) + 0.5) / max) as f32, 2));
    }
    out
}

/// Snorm of `bits`: every code's exact value, and every tie with two `f32` steps either side.
pub(super) fn snorm(bits: u32) -> Vec<f32> {
    let max = (1i32 << (bits - 1)) - 1;
    let mut out: Vec<f32> = (-max..=max)
        .map(|k| (f64::from(k) / f64::from(max)) as f32)
        .collect();
    for k in -max..max {
        out.extend(around(((f64::from(k) + 0.5) / f64::from(max)) as f32, 2));
    }
    out
}

/// The linear value of sRGB code fraction `c`.
fn srgb_decode(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB: every multiple of 2^-14 in `[0, 1.0078]`, every half in `[0, 1]`, each code's exact
/// linear value, the boundaries between codes and the 12-bit grid's ties, each boundary and tie
/// with two `f32` steps either side.
fn srgb() -> Vec<f32> {
    let mut out: Vec<f32> = (0..=16512).map(|j| j as f32 / 16384.0).collect();
    out.extend((0..=0x3c00u16).map(f16_bits_to_f32));
    out.extend((0..=255).map(|k| srgb_decode(f64::from(k) / 255.0) as f32));
    for k in 0..255 {
        out.extend(around(srgb_decode((f64::from(k) + 0.5) / 255.0) as f32, 2));
    }
    for i in 0..4095 {
        out.extend(around(((f64::from(i) + 0.5) / 4095.0) as f32, 2));
    }
    out
}

/// Half: every finite half, and every tie between neighbouring finite halves with one `f32` step
/// either side, of either sign.
pub(super) fn half() -> Vec<f32> {
    let mut out = Vec::new();
    for bits in 0..0x7c00u16 {
        let a = f64::from(f16_bits_to_f32(bits));
        let b = match bits {
            0x7bff => 65536.0,
            _ => f64::from(f16_bits_to_f32(bits + 1)),
        };
        let tie = ((a + b) / 2.0) as f32;
        for v in around(tie, 1).into_iter().chain([a as f32]) {
            out.extend([v, -v]);
        }
    }
    out
}

/// The value of code `c` of an unsigned float of `mantissa` fraction bits (not a NaN).
fn unsigned_float_value(c: u32, mantissa: u32) -> f64 {
    let exponent = (c >> mantissa) & 0x1f;
    let fraction = f64::from(c & ((1 << mantissa) - 1));
    let m = mantissa as i32;
    match exponent {
        0 => fraction * 2f64.powi(-14 - m),
        0x1f => f64::INFINITY,
        _ => (1.0 + fraction * 2f64.powi(-m)) * 2f64.powi(exponent as i32 - 15),
    }
}

/// An unsigned float of `mantissa` fraction bits: every finite code's value and every tie between
/// neighbouring codes with one `f32` step either side, and their negations.
fn unsigned_float(mantissa: u32) -> Vec<f32> {
    let largest = (0x1f << mantissa) - 1;
    let mut out = Vec::new();
    for c in 0..=largest {
        let a = unsigned_float_value(c, mantissa);
        let b = match c == largest {
            true => 65536.0,
            false => unsigned_float_value(c + 1, mantissa),
        };
        let tie = ((a + b) / 2.0) as f32;
        for v in around(tie, 1).into_iter().chain([a as f32]) {
            out.extend([v, -v]);
        }
    }
    out
}

/// Every `half` NaN, of both signs.
pub(super) fn half_nans() -> Vec<u16> {
    (0x7c01..=0x7fffu16).flat_map(|h| [h, h | 0x8000]).collect()
}

/// A target's inputs and Metal's texels for them: whole rows of [`ColorDraw::WIDTH`] pixels.
pub(super) struct Texels {
    pub(super) height: u32,
    pub(super) values: Vec<u32>,
    pub(super) texel_bytes: u32,
    pub(super) expected: Vec<u8>,
}

/// The texels of `values` (each pixel's word: a `float`'s bits, or a `half`'s where `half_bits`)
/// in `format` by `rule`, written plain or through blending, padded to whole rows with zeros.
pub(super) fn texels(
    format: Format,
    blended: bool,
    half_bits: bool,
    rule: Rule,
    mut values: Vec<u32>,
) -> Texels {
    let width = ColorDraw::WIDTH as usize;
    let height = values.len().div_ceil(width).max(1);
    values.resize(height * width, 0);
    let (texel_bytes, channels) = layout(format);
    let mut expected = Vec::with_capacity(values.len() * texel_bytes as usize);
    for &w in &values {
        let mut texel = 0u64;
        for &(offset, channel) in channels {
            let code = match (rule, half_bits) {
                (_, true) => u64::from(w & 0xffff),
                (Rule::OwnBits, false) => u64::from(narrowed(f32::from_bits(w))),
                (Rule::Metal, false) => metal_code(channel, f32::from_bits(w), blended),
            };
            texel |= (code & ((1 << channel_bits(channel)) - 1)) << offset;
        }
        expected.extend_from_slice(&texel.to_le_bytes()[..texel_bytes as usize]);
    }
    Texels {
        height: height as u32,
        values,
        texel_bytes,
        expected,
    }
}

/// The draw of `values` into `format` through `output`, its texels by `rule`.
fn draw(
    format: Format,
    blended: bool,
    output: ColorOutput,
    rule: Rule,
    values: Vec<u32>,
) -> ColorDraw {
    let t = texels(format, blended, output == ColorOutput::Half, rule, values);
    ColorDraw {
        format,
        blended,
        output,
        height: t.height,
        values: t.values,
        texel_bytes: t.texel_bytes,
        expected: t.expected,
    }
}

/// `float` inputs as words.
pub(super) fn words(values: Vec<f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

/// Draws of `format`'s `values` by Metal's rule through a `float` output, plain and blended,
/// `blended_value` taking each value as the blended draw writes it.
fn plain_and_blended(
    format: Format,
    values: &[f32],
    blended_value: impl Fn(f32) -> f32,
) -> [ColorDraw; 2] {
    let float = ColorOutput::Float;
    [
        draw(format, false, float, Rule::Metal, words(values.to_vec())),
        draw(
            format,
            true,
            float,
            Rule::Metal,
            words(values.iter().map(|&v| blended_value(v)).collect()),
        ),
    ]
}

/// The draws of color-write deviation `fact` (the module docs), or `None` where `fact` is not one
/// of [`COLOR_WRITE_FACTS`].
pub fn color_write_draws(fact: DriverDeviation) -> Option<Vec<ColorDraw>> {
    let with = |mut values: Vec<f32>, nans: bool| {
        values.extend(specials(nans));
        values
    };
    let same = |v: f32| v;
    let draws = match fact {
        DriverDeviation::MetalColorWrites => {
            let cases = [
                (Format::R8_UNORM, with(unorm(8), true)),
                (Format::R8_SNORM, with(snorm(8), true)),
                (Format::R16_UNORM, with(unorm(16), true)),
                (Format::R16_SNORM, with(snorm(16), true)),
                (
                    Format::A2B10G10R10_UNORM_PACK32,
                    with([unorm(10), unorm(2)].concat(), true),
                ),
                (
                    Format::R8G8B8A8_SRGB,
                    with([srgb(), unorm(8)].concat(), true),
                ),
                (Format::R16_SFLOAT, with(half(), true)),
            ];
            cases
                .into_iter()
                .flat_map(|(format, values)| plain_and_blended(format, &values, same))
                .collect()
        }
        DriverDeviation::MetalFloatColorNans => {
            let nan = vec![f32::from_bits(0x7fc0_0000); ColorDraw::WIDTH as usize];
            [
                Format::R16_SFLOAT,
                Format::R16G16_SFLOAT,
                Format::R16G16B16A16_SFLOAT,
            ]
            .into_iter()
            .flat_map(|format| plain_and_blended(format, &nan, same))
            .collect()
        }
        DriverDeviation::MetalHalfColorNans => {
            let widened: Vec<u32> = half_nans()
                .into_iter()
                .map(|h| f16_bits_to_f32(h).to_bits())
                .collect();
            let format = Format::R16G16B16A16_SFLOAT;
            let float = ColorOutput::Float;
            vec![
                draw(format, false, float, Rule::OwnBits, widened.clone()),
                draw(format, true, float, Rule::OwnBits, widened),
            ]
        }
        DriverDeviation::HalfColorOutputBits => {
            let every: Vec<u32> = (0..=u16::MAX).map(u32::from).collect();
            let format = Format::R16G16B16A16_SFLOAT;
            let half = ColorOutput::Half;
            vec![
                draw(format, false, half, Rule::OwnBits, every.clone()),
                draw(format, true, half, Rule::OwnBits, every),
            ]
        }
        DriverDeviation::MetalSmallFloatColorWrites => {
            let cases = [
                (Format::R16_SFLOAT, with(half(), false)),
                (
                    Format::B10G11R11_UFLOAT_PACK32,
                    with([unsigned_float(6), unsigned_float(5)].concat(), false),
                ),
            ];
            cases
                .into_iter()
                .flat_map(|(format, values)| plain_and_blended(format, &values, same))
                .collect()
        }
        DriverDeviation::MetalNorm16ColorWrites => {
            let snorm_nan = |v: f32| if v.is_nan() { -1.0 } else { v };
            let mut draws = Vec::from(plain_and_blended(
                Format::R16_UNORM,
                &with(unorm(16), true),
                same,
            ));
            draws.extend(plain_and_blended(
                Format::R16_SNORM,
                &with(snorm(16), true),
                snorm_nan,
            ));
            draws
        }
        _ => return None,
    };
    Some(draws)
}

#[cfg(test)]
mod tests {
    use super::{
        color_write_draws, fragment_typed_module, metal_code, pixel_word, Channel, ColorDraw,
        ColorOutput, COLOR_WRITE_FACTS,
    };
    use crate::bind::sample_planes::merge_vertex_typed_module;
    use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
    use crate::spirv::builder::ModuleBuilder;
    use crate::spirv::interp::{run, Buffer, Config, Resources};
    use crate::spirv::ir::{
        natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass,
        Type,
    };
    use crate::spirv::testing::{assert_valid, committed_profiles};
    use crate::vk::Format;

    /// A pixel's word stays inside the values buffer for every invocation, a helper invocation
    /// below the draw's last row included: on the interpreter, which traps an access past a
    /// buffer's end, a kernel reading [`pixel_word`] of one row of values at `y` = 1 reads word 0,
    /// and at a covered pixel its own word.
    #[test]
    fn a_pixel_past_the_last_row_reads_inside_the_values() {
        let slot = |index| ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        };
        let mut b = ModuleBuilder::new();
        let u32_ = b
            .ty(Type::Int {
                width: 32,
                signed: false,
            })
            .expect("u32");
        let rta = b
            .ty(Type::RuntimeArray {
                elem: u32_,
                stride: Some(4),
            })
            .expect("array");
        let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
        let block = b
            .ty(Type::Struct {
                members: [rta].into(),
                layout: Some(layout),
            })
            .expect("block");
        let [coords, values, out] = [0, 1, 2].map(|i| {
            b.resource(slot(i), StorageClass::StorageBuffer, block)
                .expect("buffer")
        });
        let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 }).expect("uvec3");
        let id = b
            .builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)
            .expect("id");
        let void = b.ty(Type::Void).expect("void");
        let mut f = b.function(void).expect("function");
        let k = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, v: u32| {
            let c = f.u32_const(v).expect("constant");
            f.module().const_value(c).expect("value")
        };
        let id = f.module().global_value(id).expect("id");
        let id = f.load(id, MemAccess::default()).expect("load");
        let i = f.composite_extract(id, &[0]).expect("x");
        let zero = k(&mut f, 0);
        let two = k(&mut f, 2);
        let coords = f.module().global_value(coords).expect("coords");
        let xi = f.binary(BinOp::IMul, i, two).expect("mul");
        let one = k(&mut f, 1);
        let yi = f.binary(BinOp::IAdd, xi, one).expect("add");
        let px = f.access_chain(coords, &[zero, xi]).expect("chain");
        let py = f.access_chain(coords, &[zero, yi]).expect("chain");
        let x = f.load(px, MemAccess::default()).expect("x");
        let y = f.load(py, MemAccess::default()).expect("y");
        let values = f.module().global_value(values).expect("values");
        let w = pixel_word(&mut f, u32_, values, (x, y)).expect("pixel word");
        let out = f.module().global_value(out).expect("out");
        let p = f.access_chain(out, &[zero, i]).expect("chain");
        f.store(p, w, MemAccess::default()).expect("store");
        f.ret(None).expect("ret");
        let main = f.finish().expect("finish");
        let size = ExecMode::LocalSize { x: 4, y: 1, z: 1 };
        b.entry(ExecModel::GLCompute, main, vec![size])
            .expect("entry");
        let m = b.finalize().expect("module");
        let last = ColorDraw::WIDTH - 1;
        let words = |w: &[u32]| w.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        let buffer = |index, bytes| Buffer {
            slot: Some(slot(index)),
            address: None,
            bytes,
        };
        let row: Vec<u32> = (0..ColorDraw::WIDTH).map(|x| 100 + x).collect();
        let resources = Resources {
            buffers: vec![
                buffer(0, words(&[0, 0, last, 0, 0, 1, last, 1])),
                buffer(1, words(&row)),
                buffer(2, vec![0; 16]),
            ],
            ..Resources::default()
        };
        let config = Config {
            workgroups: [1, 1, 1],
            ..Config::default()
        };
        let outcome = run(&m, &config, resources).expect("no access past the values");
        let read: Vec<u32> = outcome.buffers[2]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| u32::from_le_bytes(w))
            .collect();
        assert_eq!(read, [100, 100 + last, 100, 100]);
    }

    /// The modules are valid on every committed profile, the 16-bit output's where the profile
    /// has 16-bit outputs.
    #[test]
    fn the_color_draw_modules_are_valid_on_the_committed_profiles() {
        let mut sixteen = 0;
        for (_, profile) in committed_profiles().expect("profiles") {
            assert_valid(merge_vertex_typed_module().expect("vertex"), &profile);
            let m = fragment_typed_module(ColorOutput::Float).expect("float fragment");
            assert_valid(m, &profile);
            let leaf = &profile.fields().leaf;
            if leaf.input_output_16bit && leaf.int16 && leaf.float16 {
                let m = fragment_typed_module(ColorOutput::Half).expect("16-bit fragment");
                assert_valid(m, &profile);
                sixteen += 1;
            }
        }
        assert!(sixteen > 0, "no committed profile has 16-bit outputs");
    }

    /// Every fact has draws of whole rows whose texels match their values.
    #[test]
    fn every_color_fact_draws_whole_rows() {
        for fact in COLOR_WRITE_FACTS {
            let draws = color_write_draws(fact).expect("a color fact");
            assert!(!draws.is_empty(), "{fact:?}");
            for d in draws {
                let pixels = (ColorDraw::WIDTH * d.height) as usize;
                assert_eq!(d.values.len(), pixels, "{fact:?} {:?}", d.format);
                assert_eq!(d.expected.len(), pixels * d.texel_bytes as usize);
            }
        }
        assert_eq!(
            color_write_draws(crate::error::DriverDeviation::MetalMinMax),
            None
        );
    }

    /// The rule at its edges: a unorm tie up, a snorm tie away from zero, a blended snorm NaN
    /// -1.0's code, a half rounded toward zero, the half range's end.
    #[test]
    fn the_color_rule_rounds_as_metal() {
        let tie = (0.5f64 / 255.0) as f32;
        assert_eq!(metal_code(Channel::Unorm(8), tie.next_up(), false), 1);
        assert_eq!(metal_code(Channel::Unorm(8), f32::NAN, false), 0);
        assert_eq!(metal_code(Channel::Snorm(8), -1.0, false), 0x81);
        assert_eq!(metal_code(Channel::Snorm(8), f32::NAN, true), 0x81);
        assert_eq!(metal_code(Channel::Snorm(8), f32::NAN, false), 0);
        assert_eq!(metal_code(Channel::Half, 65519.0, false), 0x7bff);
        assert_eq!(metal_code(Channel::Half, 1.0f32.next_up(), false), 0x3c00);
        assert_eq!(metal_code(Channel::Half, f32::NAN, false), 0x7e00);
        assert_eq!(metal_code(Channel::UnsignedFloat(6), -1.0, false), 0);
        assert_eq!(metal_code(Channel::Srgb, 1.0, false), 255);
        let nan = crate::float16::f16_bits_to_f32(0xfd55);
        assert_eq!(super::narrowed(nan), 0xfd55);
        let d = &color_write_draws(crate::error::DriverDeviation::HalfColorOutputBits).unwrap()[0];
        assert_eq!(d.format, Format::R16G16B16A16_SFLOAT);
        assert_eq!(&d.expected[8..16], &[1, 0, 1, 0, 1, 0, 1, 0]);
    }
}
