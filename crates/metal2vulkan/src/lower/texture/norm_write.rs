//! A `float` or `half` texel a function writes to a normalized storage texture, rounded to the
//! code Metal stores.
//!
//! Metal stores a `float` write through `access::write`, `access::read_write`, a texture buffer or
//! an argument buffer's texture into a unorm or snorm format (not an sRGB one) as each channel's
//! value clamped to its range (a NaN 0) times its maximum code `m = 2^b - 1` (`2^(b-1) - 1`
//! signed) exactly, rounded to nearest with a tie away from zero; a packed 16-bit format's channel
//! takes the same exact product, not the `half` a color attachment rounds through first, and its
//! 1-bit channel stores 1 only above one half (an Apple M4 under macOS 27.0.1: every code's value
//! and every tie between codes with two `f32` steps either side, per channel), and a `half` write
//! as its value's (every one of the 65,536 halves, every format, both accesses), which the routine
//! takes widened. A Vulkan driver converts the `f32` it is handed by its own rounding: ANV and
//! lavapipe round the `f32` product with ties to even, NVIDIA's 615.71.09 stores the code below
//! about 40% of the values just past a tie (and RGB10A2's 2-bit alpha takes the top bits of an
//! 8-bit code), so near each tie they store another code.
//!
//! The slot's `SpecKey::TexelFormat` specialization (a heap texture's: the class its handle
//! carries) names the bound format's class (`metal::texel_format`), and one module routine,
//! [`define`]: `fn(x: vec4, flags: u32) -> vec4`, rounds each channel of `x` (Metal's channel
//! order, before any conversion of the storage choice) by it, so once a device specializes the
//! module and inlines the call, a slot's other arms fold away:
//!
//! - **The normalized families** (`BicubicTexels::EXACT` with `UNORM` or `SNORM`, without a packed
//!   field): every channel on the format's maximum code, 255, 127, 65535 or 32767
//!   (`BicubicTexels::WIDE16` for 16 bits).
//! - **A packed normalized class** (`TexelFormat::PACKED` with code scales, beside the family):
//!   each channel on its own scale (`PackedTexels::scales`: 5, 6, 4, 10 and 2 bits, and the 1-bit
//!   alpha of `A1BGR5Unorm` and `BGR5A1Unorm`, whose tie at one half rounds to even, 0, as Metal
//!   stores it).
//! - **Any other class** (a float, half-float or small-float family, an 8-bit sRGB format's, whose
//!   storage choice encodes it, a YCbCr field, `ARGB8Unorm`'s field, whose storage choice rounds
//!   it, and `OTHER`): the texel as computed.
//!
//! Each rounded channel is `color_write`'s rule for a unorm or snorm color channel: the code `c`
//! from the exact product (`color_write::round_scaled_product`, the scale `m + 1` a power of two)
//! and the value written `c / m`, whose product with `m` lies within `2^-5` of `c`, so a device
//! that rounds the product to nearest stores `c` whatever it does at a tie.
//!
//! A device measured to store Metal's codes (`DriverDeviation::MetalNormStorageWrites`) takes no
//! routine; it rounds a 1-bit channel's half as other channels' (RADV), so a written slot of such a
//! format takes `StorageImageClass::OneBitAlpha` there.

use crate::lower::arith::color_write::round_scaled_product;
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::metal::bicubic::BicubicTexels;
use crate::metal::texel_format::TexelFormat;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{FuncId, Value as SValue};
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, Type as SType, UnOp};

use super::half::{packed_scales, TEXEL_LANES};

/// An upper estimate of the routine's words, counted once per module that defines it
/// (`the_norm_write_routine_words_bound_its_emission`).
pub const ROUTINE_WORDS: u64 = 1100;

/// An upper estimate of one call's words: the call and its argument, and a texel of fewer than
/// four lanes widened to four and narrowed back.
pub const CALL_WORDS: u32 = 32;

/// The census name of the routine.
pub const CENSUS_NAME: &str = "texture.norm_write";

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

/// Define the routine `fn(x: vec4, flags: u32) -> vec4` in `mb` (module docs).
pub fn define(mb: &mut ModuleBuilder) -> BResult<FuncId> {
    let f32_ = mb.ty(SType::Float { width: 32 })?;
    let u32_ = mb.ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let vf = mb.ty(SType::Vector { elem: f32_, len: 4 })?;
    let mut fb = mb.routine(vf)?;
    let x = fb.param(vf)?;
    let flags = fb.param(u32_)?;
    let r = body(&mut fb, x, flags)?;
    fb.ret(Some(r))?;
    fb.finish()
}

fn body(fb: &mut FunctionBuilder<'_>, x: SValue, flags: SValue) -> BResult<SValue> {
    let f = fform();
    let u1 = |fb: &mut FunctionBuilder<'_>, v: u32| splat(fb, uform(1), u64::from(v));
    let has = |fb: &mut FunctionBuilder<'_>, bit: u32| -> BResult<SValue> {
        let m = u1(fb, bit)?;
        let z = u1(fb, 0)?;
        let and = fb.binary(BinOp::BitwiseAnd, flags, m)?;
        fb.binary(BinOp::INotEqual, and, z)
    };
    let unorm = has(fb, BicubicTexels::UNORM)?;
    let snorm = has(fb, BicubicTexels::SNORM)?;
    let exact = has(fb, BicubicTexels::EXACT)?;
    let wide = has(fb, BicubicTexels::WIDE16)?;
    // The unpacked families' maximum code, in every lane.
    let (m8, m8s) = (u1(fb, 255)?, u1(fb, 127)?);
    let (m16, m16s) = (u1(fb, 65535)?, u1(fb, 32767)?);
    let narrow = fb.select(snorm, m8s, m8)?;
    let broad = fb.select(snorm, m16s, m16)?;
    let m = fb.select(wide, broad, narrow)?;
    let vu = ty(fb, uform(4))?;
    let m = fb.composite_construct(vu, &[m; 4])?;
    // A packed field's: each channel's own scale (1 in every lane of a field without scales).
    let texel_lanes = u1(fb, TEXEL_LANES)?;
    let (scales, coded, _) = packed_scales(fb, flags, texel_lanes)?;
    let in_field = has(fb, TexelFormat::PACKED_MASK)?;
    let m = fb.select(in_field, scales, m)?;
    // The classes rounded: a normalized family's exact class, without a field or with one of
    // scales.
    let normalized = fb.binary(BinOp::LogicalOr, unorm, snorm)?;
    let rounded = fb.binary(BinOp::LogicalAnd, exact, normalized)?;
    let own = fb.unary(UnOp::LogicalNot, in_field)?;
    let own_or_coded = fb.binary(BinOp::LogicalOr, own, coded)?;
    let rounded = fb.binary(BinOp::LogicalAnd, rounded, own_or_coded)?;
    // Each lane: a NaN 0, clamped to `[0, 1]` (`[-1, 1]` signed), its magnitude's code `c` on
    // the scale `m` from the exact product, written as `c / m` with the value's sign.
    let ft = ty(fb, f)?;
    let mf = fb.convert(ConvOp::ConvertUToF, m, ft)?;
    let one = fsplat(fb, f, 1.0)?;
    let scale = fb.binary(BinOp::FAdd, mf, one)?;
    let zero = fsplat(fb, f, 0.0)?;
    let is_nan = fb.unary(UnOp::IsNan, x)?;
    let v = fb.select(is_nan, zero, x)?;
    let minus_one = fsplat(fb, f, -1.0)?;
    let lo = fb.select(snorm, minus_one, zero)?;
    let v = fb.glsl(GlslOp::NClamp, &[v, lo, one])?;
    let a = fb.glsl(GlslOp::FAbs, &[v])?;
    let c = round_scaled_product(fb, f, a, scale)?;
    let q = fb.binary(BinOp::FDiv, c, mf)?;
    let negative = fb.binary(BinOp::FOrdLessThan, v, zero)?;
    let minus = fb.unary(UnOp::FNegate, q)?;
    let q = fb.select(negative, minus, q)?;
    fb.select(rounded, q, x)
}

/// Texel `x` (a `vec4` of `f32`, Metal's channel order) rounded by `routine` to the codes of the
/// bound format's class `flags` (a slot's `TexelFormat` specialization constant, or a heap
/// handle's class, as a value).
pub fn convert(
    fb: &mut FunctionBuilder<'_>,
    routine: FuncId,
    x: SValue,
    flags: SValue,
) -> BResult<SValue> {
    // Every call of the routine returns its texel.
    fb.call(routine, &[x, flags])?.ok_or(BuildError::Operand {
        op: crate::spv::Op::FunctionCall,
        rule: OperandRule::Callee,
    })
}
