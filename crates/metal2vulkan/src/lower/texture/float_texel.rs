//! Metal's `float` value of one texel of a normalized or packed format.
//!
//! A read, a 2D gather, or a sample whose sampler reads one texel of one level, with a `float`
//! result, of a color texture slot (or, a depth texture slot, a comparison's never) returns the
//! device's conversion of each channel's code `c`, which Vulkan bounds only loosely: lavapipe
//! multiplies by the reciprocal (some 8-bit codes one ulp high, a 16-bit code one ulp low), and ANV
//! and NVIDIA widen a 5- or 6-bit channel by replicating its bits. Metal answers `c / (2^b - 1)`
//! correctly rounded (measured on an Apple M3 Max, every read and nearest word of the and the
//! committed rows), and reads `RG11B10Float`'s NaN codes, and a half-float texel's NaNs, as
//! `0x7fc00000`.
//!
//! The slot's `SpecKey::TexelFormat` specialization (`metal::texel_format`) names the bound
//! format's class, and one module routine, [`define`]: `fn(x: vec4, flags: u32, channel: u32) ->
//! vec4`, converts the device's value by it, called with the slot's constant and the access's
//! lanes (`half::TexelLanes`), so once a device specializes the module and inlines the call, the
//! other arms fold away:
//!
//! - **The normalized families** (`BicubicTexels::EXACT` without a packed field): the value's
//!   step on the `half` routine's grid (`1/4080`, `1/2032`, `1/65535`, `1/32767`, each a multiple
//!   of its code scale, so the step is the code's), then `n / D` correctly rounded
//!   (`ratio_to_f32`).
//! - **A packed normalized class** (`TexelFormat::PACKED` with scales): each channel's code on
//!   its scale `2^b - 1`, then `c / (2^b - 1)` correctly rounded; a gather's lanes take the
//!   gathered channel's scale.
//! - **A small float** (`PackedTexels::SmallFloat`): the device's value, exact, a NaN Metal's
//!   `0x7fc00000`.
//! - **A half float** (`BicubicTexels::HALF`: `R16Float`, `RG16Float`, `RGBA16Float`,
//!   `RGBX16Float`, and a swizzled small float, whose class keeps the family without the field):
//!   the device's value, the half's exact widening, a NaN Metal's `0x7fc00000`. Metal hands back
//!   every NaN of a half-float texel so in a `float` result, whatever its sign and payload, by a
//!   read through `access::read` or `access::read_write`, of a 2D, 3D or array texture or a texture
//!   buffer, a nearest sample and a gather; where the device widens it with its sign and payload
//!   (NVIDIA, RADV), as `0xffc00000` (ANV) or quieted (lavapipe).
//! - **A YCbCr format** (`PackedTexels::YCbCr2048`, a field without scales): the device's value,
//!   which its view holds as Metal's read exactly.
//! - **Any other class**, and any value that may be filtered (`TexelLanes::Filtered`: a 16-bit
//!   format's filtered value is finer than its code, and Metal's filter of a packed format is not
//!   modelled): the device's value.
//!
//! On a device that samples through Metal's own sampler (`VkProperty::MetalSampler`) the device's
//! value is Metal's, so no module defines the routine there.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{splat, ty};
use crate::lower::arith::texel_round;
use crate::metal::bicubic::BicubicTexels;
use crate::metal::texel_format::TexelFormat;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{FuncId, TypeId, Value as SValue};
use crate::spirv::ir::{BinOp, ConvOp, Type as SType, UnOp};

use super::half::{packed_scales, TEXEL_LANES};

/// An upper estimate of the routine's words, counted once per module that defines it
/// (`the_float_texel_routine_words_bound_its_emission`).
pub const ROUTINE_WORDS: u64 = 1200;

/// An upper estimate of one call's words: the call and its two constant arguments.
pub const CALL_WORDS: u32 = 16;

/// The census name of the routine.
pub const CENSUS_NAME: &str = "texture.float_texel";

/// Metal's `float` NaN (`RG11B10Float`'s NaN codes and a half-float texel's NaNs).
const METAL_NAN: u32 = 0x7fc0_0000;

fn fform() -> Form {
    Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 4,
    }
}

fn uform(lanes: u8) -> Form {
    Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes,
    }
}

/// Define the routine `fn(x: vec4, flags: u32, channel: u32) -> vec4` in `mb` (module docs;
/// `channel` is `half::TexelLanes::code`).
pub fn define(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let f32_ = mb.ty(SType::Float { width: 32 })?;
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let vf = mb.ty(SType::Vector { elem: f32_, len: 4 })?;
    let vu = mb.ty(SType::Vector { elem: u32_, len: 4 })?;
    let mut fb = mb.routine(vf)?;
    let x = fb.param(vf)?;
    let flags = fb.param(u32_)?;
    let channel = fb.param(u32_)?;
    let r = body(&mut fb, x, flags, channel, (f32_, vu))?;
    fb.ret(Some(r))?;
    fb.finish()
}

fn body(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    flags: SValue,
    channel: SValue,
    (f32_, vu): (TypeId, TypeId),
) -> BResult<SValue> {
    let f = fform();
    let u1 = |fb: &mut FunctionBuilder<'_>, v: u32| splat(fb, uform(1), u64::from(v));
    let has = |fb: &mut FunctionBuilder<'_>, bit: u32| -> BResult<SValue> {
        let m = u1(fb, bit)?;
        let z = u1(fb, 0)?;
        let and = fb.binary(BinOp::BitwiseAnd, flags, m)?;
        fb.binary(BinOp::INotEqual, and, z)
    };
    let texel_lanes = u1(fb, TEXEL_LANES)?;
    let single = fb.binary(BinOp::ULessThanEqual, channel, texel_lanes)?;
    let ft = ty(fb, f)?;
    // The normalized families: the value onto the format's grid `D`, then `n / D` to nearest
    // even. A treatment without `EXACT` takes the 16-bit unsigned grid, which keeps every
    // operation of the arm (selected away) within its operands' range.
    let wide = has(fb, BicubicTexels::WIDE16)?;
    let signed = has(fb, BicubicTexels::SNORM)?;
    let (g8, g8s) = (u1(fb, TexelFormat::GRID8)?, u1(fb, TexelFormat::GRID8S)?);
    let (g16, g16s) = (u1(fb, TexelFormat::GRID16)?, u1(fb, TexelFormat::GRID16S)?);
    let narrow = fb.select(signed, g8s, g8)?;
    let broad = fb.select(signed, g16s, g16)?;
    let d = fb.select(wide, broad, narrow)?;
    let exact = has(fb, BicubicTexels::EXACT)?;
    let d = fb.select(exact, d, g16)?;
    let df = fb.convert(ConvOp::ConvertUToF, d, f32_)?;
    let df = fb.composite_construct(ft, &[df; 4])?;
    let neg_df = fb.unary(UnOp::FNegate, df)?;
    let dv = fb.composite_construct(vu, &[d; 4])?;
    let n = texel_round::grid_step(fb, f, x, df, neg_df)?;
    let norm = texel_round::grid_value(fb, uform(4), n, dv, |fb, m, d| {
        texel_round::ratio_to_f32(fb, uform(4), m, d)
    })?;
    // A packed class: each channel's code on its scale, then `c / S` to nearest even; a small
    // or half float's NaN Metal's.
    let (scales, coded, small) = packed_scales(fb, flags, channel)?;
    let sf = fb.convert(ConvOp::ConvertUToF, scales, ft)?;
    let neg_sf = fb.unary(UnOp::FNegate, sf)?;
    let codes = texel_round::grid_step(fb, f, x, sf, neg_sf)?;
    let packed = texel_round::grid_value(fb, uform(4), codes, scales, |fb, m, d| {
        texel_round::ratio_to_f32(fb, uform(4), m, d)
    })?;
    let nan = fb.unary(UnOp::IsNan, x)?;
    let metal_nan = splat(fb, uform(4), u64::from(METAL_NAN))?;
    let metal_nan = fb.convert(ConvOp::Bitcast, metal_nan, ft)?;
    let canonical = fb.select(nan, metal_nan, x)?;
    // The normalized families' grids are a class's own: a packed class's `EXACT` family sums on
    // its field's grids (`half::body`'s `norm_arm`), so a field without scales (a YCbCr format's,
    // whose view holds Metal's value exactly) keeps the device's value.
    let in_field = has(fb, TexelFormat::PACKED_MASK)?;
    let own_grid = fb.unary(UnOp::LogicalNot, in_field)?;
    let exact = fb.binary(BinOp::LogicalAnd, exact, own_grid)?;
    let exact = fb.binary(BinOp::LogicalAnd, exact, single)?;
    let coded = fb.binary(BinOp::LogicalAnd, coded, single)?;
    let half = has(fb, BicubicTexels::HALF)?;
    let small = fb.binary(BinOp::LogicalOr, small, half)?;
    let small = fb.binary(BinOp::LogicalAnd, small, single)?;
    let r = fb.select(exact, norm, x)?;
    let r = fb.select(coded, packed, r)?;
    fb.select(small, canonical, r)
}

/// The `float` texel of image texel `x` (a `vec4` of `f32`), converted by `routine` under the
/// slot's treatment constant `flags` (its `TexelFormat` specialization) and `lanes`, the access's
/// `half::TexelLanes::code` (a constant, or the shader's choice where a runtime sampler's word
/// decides whether a sample reads one texel).
pub fn convert(
    fb: &mut FunctionBuilder<'_>,
    routine: FuncId,
    x: SValue,
    flags: SValue,
    lanes: SValue,
) -> BResult<SValue> {
    // Every call of the routine returns its texel.
    fb.call(routine, &[x, flags, lanes])?
        .ok_or(BuildError::Operand {
            op: crate::spv::Op::FunctionCall,
            rule: OperandRule::Callee,
        })
}
