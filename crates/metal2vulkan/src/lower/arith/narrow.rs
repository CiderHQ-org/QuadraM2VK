//! Narrow floats held widened: `half` and `bfloat` values are held in an `f32` that is always
//! exactly a value of the narrow format, on every profile.
//!
//! An operation on narrow floats runs in `f32` and rounds its result back to the format
//! ([`round`]), as Metal's conversion of a `float` to `half` does: to nearest even on the half
//! grid, half denormals kept, infinity from 65520 up, a NaN to `0x7E00`. By default
//! ([`QuantizeForm::Integer`]) that is integer arithmetic on the `f32`'s bits and, below the
//! smallest normal half, an exact `RoundEven` of `x · 2^24`, the same on every device. Where the
//! device's `OpQuantizeToF16` is measured to round as Metal's conversion does
//! (`DriverDeviation::MetalHalfQuantize`), it is that quantize, with the same denormal arm
//! ([`QuantizeForm::Device`]: the module requires the fact). Where the device's conversion to 16
//! bits, of an operand its compiler cannot fold into the instruction computing it, is measured to
//! be Metal's `half(float)` (`DriverDeviation::MetalHalfConversion`) and the profile has
//! `shaderFloat16`, it is that conversion behind such a barrier and its widening
//! ([`QuantizeForm::Converted`]): no quantize and no denormal arithmetic, and the module requires
//! the fact. Where the profile has `shaderFloat16` and the float controls that make the conversion
//! Metal's by the Vulkan spec (`RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve`
//! for 16 bits, and the modes the device's independence ties to them), it is that conversion and
//! its widening, each decorated `NoContraction`, in an entry declaring those modes
//! ([`QuantizeForm::Controlled`], [`controlled_modes`]): exact on every such device whose driver
//! keeps the pair (not NVIDIA's, `DriverWorkaround::HalfConversionPairFolded`), under either
//! policy, with no fact and no note. Elsewhere, under native half rounding
//! (`HalfRoundingPolicy::Native`, the default), the module takes the same forms though neither is
//! measured to be Metal's: the conversion where the profile has `shaderFloat16`, else the quantize,
//! recording `PrecisionNote::NativeHalfConversion` or `NativeHalfQuantize` instead of requiring a
//! fact; `HalfRoundingPolicy::Exact` keeps the integer form. A sum or difference, exact in `f32`
//! below the smallest normal half, keeps its `f32` value there instead
//! ([`round_exact_below_normal`]); a texel's `half` takes the same rounding and keeps the NaN the
//! device decoded ([`round_texel`]) unless the profile leaves NaN bits to the device under native
//! half rounding or the float controls; `bfloat` rounds through an
//! exact integer round-to-nearest-even of the `f32` bits. Where the device's 16-bit `+ - *`, kept
//! apart, is Metal's `half` arithmetic and the profile has `shaderFloat16`, a `half` `+ - *` is
//! that instruction between exact conversions instead, with no rounding here (`ops::f16_binary`):
//! where it is measured to be (`DriverDeviation::MetalHalfArithmetic`), and under the float
//! controls of [`QuantizeForm::Controlled`], which make it so by the Vulkan spec and which the
//! entry then declares ([`computes_in_16_bits`]). For `+ - * /` and square root the double
//! rounding (the exact result to `f32`, then to the narrow format) is innocuous, because `f32`
//! carries at least `2p + 2` significand bits for both formats; a `half` fused multiply-add is not,
//! so it rounds its exact `f32` sum to odd first (`super::fma`), and a `bfloat` one is corrected
//! ([`bf16_fma`]).
//!
//! Every other crossing is exact: [`to_bits`] and [`from_bits`] move between a value and its
//! format bits with integer sequences and, for a half denormal, one exact scaling by `2^±24`
//! between an integer and a normal `f32` (denormals, infinities, and NaN payloads included), but
//! under the float controls, where the profile leaves NaN bits to the device, [`from_bits`] is the
//! device's widening, exact but for a NaN's bits ([`controlled_from_bits`]); an
//! integer converts to `half` through `f32` without double rounding (any integer the `f32` step
//! rounds is at least `2^24`, far above the largest half) and rounds by integer arithmetic, not
//! `OpQuantizeToF16` ([`round_integer`]), an integer converts to `bfloat` through a round-to-odd
//! `f32` ([`int_to_f32_odd`], [`int64_to_f32_odd`]), a `double` narrows through a round-to-odd
//! `f32` ([`f64_to_f32_odd`]), and a `bfloat` fused multiply-add corrects its one inexact case
//! ([`bf16_fma`]), so the final rounding is the only one that counts.

use crate::budget::Meter;
use crate::device::profile::{DriverWorkaround, HalfRoundingRoute, Independence};
use crate::device::DeviceProfile;
use crate::error::{DriverDeviation, Error};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::pipeline::artifact::PrecisionNote;
use crate::spirv::builder::{FunctionBuilder, NarrowFmt, QuantizeForm};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    BinOp, ConstBits, ConstKind, Constant, ConvOp, ExecMode, GlslOp, Inst, Type, TypedModule, UnOp,
    ValueOrigin,
};

/// The unsigned 32-bit integer form with `f`'s lanes.
pub fn u32_form(f: Form) -> Form {
    f.with(Held::Int {
        bits: 32,
        container: 32,
    })
}

fn bin(fb: &mut FunctionBuilder<'_>, op: BinOp, a: Value, b: Value) -> BResult<Value> {
    fb.binary(op, a, b)
}

/// An upper estimate of the SPIR-V words of one [`round`] on form `f`, where the module's
/// [`QuantizeForm`] is not at hand: [`round_words_in`] of [`QuantizeForm::Integer`], the largest.
pub fn round_words(f: Form) -> u32 {
    round_words_in(QuantizeForm::Integer, f)
}

/// An upper estimate of the SPIR-V words of one [`round`] on form `f` in quantize form `q`: for
/// `half`, [`QuantizeForm::Integer`]'s twenty instructions and its twelve constants' share; the
/// device's quantize or conversion ([`QuantizeForm::Device`], [`QuantizeForm::Converted`]), at most
/// six instructions and their constants, the bound of 64 words and 4 per lane, so that the choices
/// weighing it (`fma::complete`) stay as they were on those profiles; a dozen instructions for
/// `bfloat`.
pub fn round_words_in(q: QuantizeForm, f: Form) -> u32 {
    let lanes = u32::from(f.lanes);
    match (f.float_fmt(), q) {
        (Some(FloatFmt::Half), QuantizeForm::Integer) => 200 + 12 * lanes,
        (
            Some(FloatFmt::Half),
            QuantizeForm::Device | QuantizeForm::Converted | QuantizeForm::Controlled,
        ) => 64 + 4 * lanes,
        (Some(FloatFmt::BFloat), _) => 96,
        _ => 0,
    }
}

/// `x` (an `f32` of form `f`) rounded to `f`'s narrow format; the value itself for `float` and
/// `double` forms.
///
/// `half`: Metal's conversion to `half` (to nearest even, half denormals kept, infinity from
/// 65520 up, a NaN to `0x7E00`), in the module's [`QuantizeForm`]. [`QuantizeForm::Integer`]:
/// [`integer`], below the smallest normal half by [`scaled`]. [`QuantizeForm::Device`]: `q`,
/// `OpQuantizeToF16(x)`, is the rounded value wherever it is not a zero. It is a zero exactly when
/// `|x|` rounds below the smallest normal half (or `x` is a zero), and then the result is
/// [`scaled`]'s. A NaN compares unequal to zero and keeps `q`. [`QuantizeForm:: Converted`]:
/// [`converted`]'s alone, which is the same value where the device's conversion is Metal's.
/// [`QuantizeForm::Controlled`]: [`controlled`]'s alone, Metal's value by the Vulkan spec.
pub fn round(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    match f.held {
        Held::Float(FloatFmt::Half) => {
            let wide = f.with(Held::Float(FloatFmt::F32));
            match fb.module().quantize_form() {
                QuantizeForm::Converted => converted(fb, wide, x),
                QuantizeForm::Controlled => controlled(fb, wide, x),
                QuantizeForm::Integer => integer(fb, wide, x, Below::Scaled),
                QuantizeForm::Device => {
                    let q = fb.unary(UnOp::QuantizeToF16, x)?;
                    let zero = fsplat(fb, wide, 0.0)?;
                    let flushed = bin(fb, BinOp::FOrdEqual, q, zero)?;
                    let d = scaled(fb, wide, x)?;
                    fb.select(flushed, d, q)
                }
            }
        }
        Held::Float(FloatFmt::BFloat) => {
            let u = u32_form(f);
            let ut = ty(fb, u)?;
            let b = fb.convert(ConvOp::Bitcast, x, ut)?;
            let r = bf16_round_bits(fb, u, b)?;
            let ft = ty(fb, f)?;
            fb.convert(ConvOp::Bitcast, r, ft)
        }
        _ => Ok(x),
    }
}

/// How profile `p`'s modules round an `f32` to `half` ([`QuantizeForm`]): the device's
/// conversion to 16 bits and back where it is measured to be Metal's `half(float)`
/// (`DriverDeviation::MetalHalfConversion`) and the profile has `shaderFloat16`, else that
/// conversion under the float controls that make it Metal's by the Vulkan spec where the profile
/// has `shaderFloat16` and them ([`QuantizeForm::Controlled`], [`controlled_modes`]), else the
/// device's `OpQuantizeToF16` where it is measured to round as Metal does
/// (`DriverDeviation::MetalHalfQuantize`). Where neither is measured, under native half rounding
/// (`HalfRoundingPolicy::Native`, the default) the same forms, the conversion where the profile has
/// `shaderFloat16` and the quantize elsewhere, each the device's rounding rather than Metal's
/// ([`native_rounding`]); under `HalfRoundingPolicy::Exact`, integer arithmetic, Metal's on every
/// device.
pub fn quantize_form(p: &DeviceProfile) -> QuantizeForm {
    let float16 = p.fields().leaf.float16;
    if p.takes_driver_deviation(DriverDeviation::MetalHalfConversion) && float16 {
        QuantizeForm::Converted
    } else if float16 && controlled_modes(p).is_some() {
        QuantizeForm::Controlled
    } else if p.takes_driver_deviation(DriverDeviation::MetalHalfQuantize) {
        QuantizeForm::Device
    } else if p.fields().half_rounding == HalfRoundingRoute::Metal {
        QuantizeForm::Integer
    } else if float16 {
        QuantizeForm::Converted
    } else {
        QuantizeForm::Device
    }
}

/// The precision note of a module of profile `p` whose `half` roundings take `form` where the
/// profile does not take the deviation that makes the form Metal's: `NativeHalfConversion` for
/// [`QuantizeForm::Converted`] without `DriverDeviation::MetalHalfConversion`,
/// `NativeHalfQuantize` for [`QuantizeForm::Device`] without `DriverDeviation::MetalHalfQuantize`
/// (native half rounding, [`quantize_form`]). `None` where the form is Metal's: the measured forms,
/// whose fact the module then requires, and the integer form.
pub fn native_rounding(p: &DeviceProfile, form: QuantizeForm) -> Option<PrecisionNote> {
    match form {
        QuantizeForm::Converted
            if !p.takes_driver_deviation(DriverDeviation::MetalHalfConversion) =>
        {
            Some(PrecisionNote::NativeHalfConversion)
        }
        QuantizeForm::Device if !p.takes_driver_deviation(DriverDeviation::MetalHalfQuantize) => {
            Some(PrecisionNote::NativeHalfQuantize)
        }
        _ => None,
    }
}

/// What [`integer`] answers for an `|x|` below the smallest normal half, `2^-14`.
#[derive(Clone, Copy)]
enum Below {
    /// [`scaled`]'s rounding to the half denormal grid ([`round`]).
    Scaled,
    /// `x` itself, a value of the format there ([`round_exact_below_normal`]).
    Kept,
}

/// `x` (`f32` lanes of form `wide`) rounded to `half` by integer arithmetic on its bits
/// ([`QuantizeForm::Integer`]), and as `below` states where `|x|` is below the smallest normal
/// half, `2^-14`: the same on every device and in every rounding mode, since no instruction of
/// it rounds inexactly. From `2^-14` up, the half keeps the top 11 of the `f32`'s 24 significand bits, so
/// adding `0xFFF` plus the lowest kept bit and clearing the 13 bits below it rounds to nearest
/// even, once, as [`round_integer`] does; a carry out of the fraction raises the exponent, as it
/// must. From 65520 up (`0x477FF000`: halfway between 65504 and `2^16`, which rounds to the even
/// `2^16`), and for an infinity, the result is the infinity of `x`'s sign. A NaN gives `0x7E00`'s
/// widening, `0x7FC00000`, of either sign and any payload: Metal converts every `float` NaN to
/// `0x7E00` (the M3 Max, fast and safe math, a loaded and a computed `float`, every one of the
/// 2^24 NaN patterns), which the NaN model's `half` arithmetic answers too. The rounding Metal's
/// conversion does, measured on every `f32` (`DriverDeviation::MetalHalfConversion`'s measurements)
/// , on both chips.
fn integer(fb: &mut FunctionBuilder<'_>, wide: Form, x: Value, below: Below) -> BResult<Value> {
    let u = u32_form(wide);
    let ut = ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    let abs_mask = splat(fb, u, 0x7FFF_FFFF)?;
    let mag = bin(fb, BinOp::BitwiseAnd, b, abs_mask)?;
    let k13 = splat(fb, u, 13)?;
    let one = splat(fb, u, 1)?;
    let lsb = bin(fb, BinOp::ShiftRightLogical, b, k13)?;
    let lsb = bin(fb, BinOp::BitwiseAnd, lsb, one)?;
    let half_ulp = splat(fb, u, 0xFFF)?;
    let bias = bin(fb, BinOp::IAdd, half_ulp, lsb)?;
    let r = bin(fb, BinOp::IAdd, b, bias)?;
    let keep = splat(fb, u, 0xFFFF_E000)?;
    let r = bin(fb, BinOp::BitwiseAnd, r, keep)?;
    // From 65520 up: the infinity of the sign, or for a NaN Metal's NaN.
    let below_edge = splat(fb, u, 0x477F_EFFF)?;
    let over = bin(fb, BinOp::UGreaterThan, mag, below_edge)?;
    let sign_mask = splat(fb, u, 0x8000_0000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, b, sign_mask)?;
    let inf = splat(fb, u, 0x7F80_0000)?;
    let signed_inf = bin(fb, BinOp::BitwiseOr, sign, inf)?;
    let nan = bin(fb, BinOp::UGreaterThan, mag, inf)?;
    let metal_nan = splat(fb, u, 0x7FC0_0000)?;
    let big = fb.select(nan, metal_nan, signed_inf)?;
    let r = fb.select(over, big, r)?;
    // Below the smallest normal half: `below`.
    let normal = splat(fb, u, 0x3880_0000)?;
    let small = bin(fb, BinOp::ULessThan, mag, normal)?;
    let below = match below {
        Below::Scaled => {
            let d = scaled(fb, wide, x)?;
            fb.convert(ConvOp::Bitcast, d, ut)?
        }
        Below::Kept => b,
    };
    let r = fb.select(small, below, r)?;
    let ft = ty(fb, wide)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}

/// `x` (`f32` lanes of form `wide`) rounded to the half denormal grid, `RoundEven(x · 2^24) ·
/// 2^-24`: the half of an `|x|` below the smallest normal half. It is exact in every rounding mode:
/// both products scale by a power of two into the normal `f32` range, and `RoundEven` is exact. It
/// rounds to nearest even on the grid (`2^-24`), keeps the sign of a zero, and gives the smallest
/// normal half where `x` rounds up to it. (An `f32` denormal `x` may be flushed by the device's
/// multiply; its half is a signed zero either way.)
fn scaled(fb: &mut FunctionBuilder<'_>, wide: Form, x: Value) -> BResult<Value> {
    let up = fsplat(fb, wide, 16_777_216.0)?;
    let down = fsplat(fb, wide, 1.0 / 16_777_216.0)?;
    let s = bin(fb, BinOp::FMul, x, up)?;
    let k = fb.glsl(GlslOp::RoundEven, &[s])?;
    bin(fb, BinOp::FMul, k, down)
}

/// `x` (`f32` lanes of form `wide`) rounded to `half` by the device's conversion to 16 bits,
/// `OpFConvert`, and widened back, exactly ([`QuantizeForm::Converted`]). The conversion's
/// operand is `x + -0.0`, an `OpFAdd` decorated `NoContraction` ([`BARRIER`]): `x` itself in
/// every case, but a value Metal's compiler cannot fold the conversion into. SPIRV-Cross writes
/// the add as its `[[clang::optnone]]` helper `spvFAdd`, and the conversion of a call's result
/// stays a conversion; without it, Metal's compiler folds the conversion into the instruction
/// computing `x`, and that conversion rounds toward zero (`a1_half_tanh_regimes`, case 27353187:
/// 1, 204 of its 16, 384 halves one ulp toward zero on MoltenVK). Where the profile takes
/// `DriverDeviation::MetalHalfConversion` the conversion behind the barrier is Metal's
/// `half(float)`, measured on every `f32`: to nearest even on the half grid, half denormals kept,
/// infinity from 65520 up, the sign of a zero kept, a NaN to a NaN. That is [`round`]'s value in
/// its other forms, with no quantize, no clamp and no denormal arithmetic: three instructions where
/// they take six to twelve, which Metal's compiler takes far less time over on MoltenVK (corpus row
/// 69a35b6f).
///
/// Under native half rounding (the conversion not measured to be Metal's, recorded as
/// `PrecisionNote::NativeHalfConversion`) the conversion's operand is `x` itself, with no barrier:
/// the barrier keeps a measured conversion Metal's. The barrier cost NVIDIA's compiler about 2% of
/// `A2Xghfc`'s making. NVIDIA's compiler (RTX 3090) drops the pair, barrier or not, and so does
/// lavapipe's where measured, so there the value is not rounded at all (the note's stated bound),
/// and the store's bits saturate what passes `half`'s range ([`f32_bits_to_half_bits`]).
fn converted(fb: &mut FunctionBuilder<'_>, wide: Form, x: Value) -> BResult<Value> {
    let e = fb.module().ty(Type::Float { width: 16 })?;
    let h = match wide.lanes {
        1 => e,
        len => fb.module().ty(Type::Vector { elem: e, len })?,
    };
    let w = ty(fb, wide)?;
    let held = if fb.module().native_half() {
        x
    } else {
        let minus_zero = fsplat(fb, wide, BARRIER)?;
        fb.binary_no_contraction(BinOp::FAdd, x, minus_zero)?
    };
    let n = fb.convert(ConvOp::FConvert, held, h)?;
    fb.convert(ConvOp::FConvert, n, w)
}

/// `x` (`f32` lanes of form `wide`) rounded to `half` by the device's conversion to 16 bits and
/// its widening, each an `OpFConvert` decorated `NoContraction` ([`QuantizeForm::Controlled`]),
/// in an entry that declares [`controlled_modes`]: Metal's `half(float)` on every `f32` by the
/// Vulkan spec. `NoContraction` keeps the pair: on a float operation without it Vulkan assumes
/// `AllowTransform`, under which Mesa's compiler folds the widening of a narrowing into nothing and
/// hands `x` on unrounded. Two instructions where the device's quantize with its denormal arm takes
/// five, which ANV's compiler lowers to eleven a lane.
fn controlled(fb: &mut FunctionBuilder<'_>, wide: Form, x: Value) -> BResult<Value> {
    let e = fb.module().ty(Type::Float { width: 16 })?;
    let h = match wide.lanes {
        1 => e,
        len => fb.module().ty(Type::Vector { elem: e, len })?,
    };
    let w = ty(fb, wide)?;
    let n = fb.convert_no_contraction(x, h)?;
    fb.convert_no_contraction(n, w)
}

/// The execution modes an entry that rounds to `half` in [`QuantizeForm::Controlled`] declares on
/// profile `p`, or `None` where the profile lacks one or its driver folds the pair
/// (`DriverWorkaround::HalfConversionPairFolded`): `RoundingModeRTE`, `DenormPreserve` and
/// `SignedZeroInfNanPreserve` for 16 bits, which make an `OpFConvert` from 32 to 16 bits Metal's
/// `half(float)`. The Vulkan spec ("Precision and Operation of SPIR-V Instructions"): conversions
/// between types are "Correctly rounded", "according to IEEE 754 roundTiesToEven" where the entry
/// declares `RoundingModeRTE`; under `DenormPreserve` "the denormalized result and any denormalized
/// operands must be preserved"; and `OpFConvert` is among the operations that keep signed zeros,
/// infinities and NaNs in an entry declaring `SignedZeroInfNanPreserve` for one of its widths.
/// Then the modes the device's independence ties to them (RuntimeSpirv VUIDs 06289-06292):
/// `RoundingModeRTE` for 64 bits where rounding modes are independent for 32 bits only, and for 32
/// and 64 bits where they are not independent; `DenormPreserve` for 64 bits where denormal modes
/// are independent for 32 bits only. A device whose denormal modes are not independent at all
/// would have to preserve `float` denormals with them, changing how the module's `float`
/// arithmetic reads them (`PrecisionNote::DeviceDenormals`), so the form is not taken there.
pub fn controlled_modes(p: &DeviceProfile) -> Option<Vec<ExecMode>> {
    if p.fields()
        .workarounds
        .contains(&DriverWorkaround::HalfConversionPairFolded)
    {
        return None;
    }
    let fc = p.fields().float_controls;
    let half = fc.f16;
    if !(half.rounding_rte && half.denorm_preserve && half.signed_zero_inf_nan_preserve) {
        return None;
    }
    let mut modes = vec![
        ExecMode::RoundingModeRte { width: 16 },
        ExecMode::DenormPreserve { width: 16 },
        ExecMode::SignedZeroInfNanPreserve { width: 16 },
    ];
    match fc.rounding_independence {
        Independence::All => {}
        Independence::Only32Bit if fc.f64.rounding_rte => {
            modes.push(ExecMode::RoundingModeRte { width: 64 });
        }
        Independence::None if fc.f32.rounding_rte && fc.f64.rounding_rte => {
            modes.push(ExecMode::RoundingModeRte { width: 32 });
            modes.push(ExecMode::RoundingModeRte { width: 64 });
        }
        Independence::Only32Bit | Independence::None => return None,
    }
    match fc.denorm_independence {
        Independence::All => {}
        Independence::Only32Bit if fc.f64.denorm_preserve => {
            modes.push(ExecMode::DenormPreserve { width: 64 });
        }
        Independence::Only32Bit | Independence::None => return None,
    }
    Some(modes)
}

/// The addend of [`converted`]'s barrier: `x + -0.0` is `x` for every `x` (`+0 + -0` is `+0`,
/// `-0 + -0` is `-0`), and a NaN stays a NaN. The add may quiet a signalling NaN; measured on
/// every `f32` (`DriverDeviation::MetalHalfConversion`), the half of each NaN through it has the
/// bits of Metal's own conversion of that NaN.
const BARRIER: f64 = -0.0;

/// Whether module `m` rounds to `half` through [`converted`] or [`controlled`]: an `OpFConvert` of
/// an `f32` to 16 bits whose result another widens back. The module then requires
/// `VkProperty::MetalHalfConversion` ([`QuantizeForm::Converted`]) or declares
/// [`controlled_modes`] ([`QuantizeForm::Controlled`]). A `half` fused multiply-add on the
/// device's 16-bit `Fma` (`super::fma`) converts its operands, each a `half` already, exactly, and
/// widens the `Fma`'s
/// result, which is no such pair. Two passes over the instructions, with one bit per value of the
/// module charged to `meter` while they run.
pub fn rounds_by_conversion(m: &TypedModule, meter: &mut Meter) -> Result<bool, Error> {
    // The width of `v`'s float lanes, if it is a float or a vector of floats.
    let width = |v: Value| -> Option<u8> {
        let ty = m.value(v).ok()?.ty;
        let ty = match m.types().get(ty).ok()? {
            Type::Vector { elem, .. } => *elem,
            _ => ty,
        };
        match m.types().get(ty).ok()? {
            Type::Float { width } => Some(*width),
            _ => None,
        }
    };
    let fconverts = || {
        m.blocks()
            .iter()
            .flat_map(|b| &b.insts)
            .filter_map(|n| match (&n.inst, n.result) {
                (
                    Inst::Convert {
                        op: ConvOp::FConvert,
                        a,
                        ..
                    },
                    Some(r),
                ) => Some((*a, r)),
                _ => None,
            })
    };
    let words = m.values().len().div_ceil(64);
    meter.charge(8 * words)?;
    let mut narrowed = vec![0u64; words];
    let mut any = false;
    for (a, r) in fconverts() {
        if width(a) == Some(32) && width(r) == Some(16) {
            if let Some(w) = narrowed.get_mut(r.index() / 64) {
                *w |= 1 << (r.index() % 64);
                any = true;
            }
        }
    }
    let is_narrowed = |v: Value| {
        narrowed
            .get(v.index() / 64)
            .is_some_and(|w| w & (1 << (v.index() % 64)) != 0)
    };
    let found = any && fconverts().any(|(a, r)| is_narrowed(a) && width(r) == Some(32));
    drop(narrowed);
    meter.release(8 * words);
    Ok(found)
}

/// An upper estimate of the SPIR-V words of one [`round_texel`] on form `f`: [`round`]'s, and the
/// NaN test's four instructions with their two constants.
pub fn round_texel_words(f: Form) -> u32 {
    match f.float_fmt() {
        Some(FloatFmt::Half) => round_words(f) + 40,
        _ => 0,
    }
}

/// Texel value `x` (`f32` lanes of form `f`: the device's decode of an image texel) as a texture
/// access hands it to the function in `f`'s narrow format. A `half` is [`round`]ed, which keeps
/// half subnormals on their grid to nearest even, so a half-float texel, which the device decodes
/// to its exact `f32` value (a half subnormal is a normal `f32`), keeps its value: every subnormal,
/// both zeros, both infinities, and `65504`. A NaN is the device's decode itself, since
/// `OpQuantizeToF16` may return another NaN; it is told by its bits, so the test holds whatever
/// float modes the device compiles the module under. Under native half rounding
/// ([`native_rounding`]) the rounding's own NaN stands instead: the note the module records bounds
/// a NaN's encoding to the device's, so the keep's four instructions a texel are left out, but
/// where the NaN switch asks for Metal's NaN (`NanMinMaxPolicy::Exact`;
/// `ModuleBuilder::texel_nans`), which keeps them. So too under the float controls
/// ([`QuantizeForm::Controlled`]) where the profile leaves a NaN's bits to the device
/// (`ModuleBuilder::native_half_nans`, `NanMinMaxPolicy::Native`): the conversion keeps a NaN a
/// NaN, its bits the device's, as the texel's decode is (`PrecisionNote::DeviceNanEncoding`, which
/// the access records). Another form is [`round`]ed.
pub fn round_texel(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let native = fb.module().native_half() && !fb.module().texel_nans();
    match f.held {
        Held::Float(FloatFmt::Half) if native || fb.module().native_half_nans() => round(fb, f, x),
        Held::Float(FloatFmt::Half) => {
            let r = round(fb, f, x)?;
            let u = u32_form(f);
            let ut = ty(fb, u)?;
            let b = fb.convert(ConvOp::Bitcast, x, ut)?;
            let abs = splat(fb, u, 0x7FFF_FFFF)?;
            let mag = bin(fb, BinOp::BitwiseAnd, b, abs)?;
            let inf = splat(fb, u, 0x7F80_0000)?;
            let nan = bin(fb, BinOp::UGreaterThan, mag, inf)?;
            fb.select(nan, x, r)
        }
        _ => round(fb, f, x),
    }
}

/// [`round`] of an `x` that is a value of `f`'s narrow format whenever it rounds below the
/// smallest normal half: the sum or difference of two values of the format, which is exact in `f32`
/// there (both lie on the `2^-24` grid, and so does a sum below `2^-14`, which `f32` holds exactly
/// in every rounding mode). `half` then needs no denormal arithmetic: below the smallest normal
/// half the result is `x` itself, which is the half denormal, or the zero, `x` already is (with its
/// sign). [`QuantizeForm::Integer`]: [`integer`] with `x` below.
/// [`QuantizeForm::Device`]: `q`, the quantize of `x`, wherever it is not a zero, and `x` where
/// it is. [`QuantizeForm::Converted`]: [`converted`]'s alone, as in [`round`], and
/// [`QuantizeForm::Controlled`] [`controlled`]'s. Other forms round as [`round`] does.
pub fn round_exact_below_normal(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    match f.held {
        Held::Float(FloatFmt::Half) => {
            let wide = f.with(Held::Float(FloatFmt::F32));
            match fb.module().quantize_form() {
                QuantizeForm::Converted => converted(fb, wide, x),
                QuantizeForm::Controlled => controlled(fb, wide, x),
                QuantizeForm::Integer => integer(fb, wide, x, Below::Kept),
                QuantizeForm::Device => {
                    let q = fb.unary(UnOp::QuantizeToF16, x)?;
                    let zero = fsplat(fb, wide, 0.0)?;
                    let flushed = bin(fb, BinOp::FOrdEqual, q, zero)?;
                    fb.select(flushed, x, q)
                }
            }
        }
        _ => round(fb, f, x),
    }
}

/// An upper estimate of the SPIR-V words of one [`round_integer`] on form `f`: twelve instructions
/// and eight constants for `half` (a vector's constants carry their lanes); [`round`]'s for every
/// other form.
pub fn round_integer_words(f: Form) -> u32 {
    match f.float_fmt() {
        Some(FloatFmt::Half) => 136 + 8 * u32::from(f.lanes),
        _ => round_words(f),
    }
}

/// `x` (an `f32` of form `f`: an integer converted to `f32`, so exact wherever its magnitude is
/// below `2^24` and at least `2^24` elsewhere) rounded to `f`'s narrow format; the value itself
/// for `float` and `double` forms, [`round`] for `bfloat`.
///
/// `half`: by integer arithmetic on the `f32` bits, with no `OpQuantizeToF16`, so the answer is the
/// same on every device and in every rounding mode. A nonzero integer below `2^24` is a normal
/// `f32` whose half keeps the top 11 of its 24 significand bits, so adding `0xFFF` plus the lowest
/// kept bit and clearing the 13 bits below it rounds the exact value to nearest even, once; a carry
/// out of the fraction raises the exponent, as it must. Zero stays zero. A result above 65504 is
/// the infinity of the integer's sign: those are exactly the integers from 65520 up (65519 rounds
/// to 65504; 65520, halfway between 65504 and `2^16`, rounds to the even `2^16`), and every integer
/// the `f32` step rounded, which is at least `2^24`. Metal answers every one of these (measured:
/// 2051 is 2052, 4097 is 4096, 65519 is 65504), where NVIDIA's `OpQuantizeToF16` of an integer
/// conversion keeps the integer (driver 615.71.09).
pub fn round_integer(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    match f.held {
        Held::Float(FloatFmt::Half) => {
            let u = u32_form(f);
            let ut = ty(fb, u)?;
            let b = fb.convert(ConvOp::Bitcast, x, ut)?;
            let k13 = splat(fb, u, 13)?;
            let one = splat(fb, u, 1)?;
            let lsb = bin(fb, BinOp::ShiftRightLogical, b, k13)?;
            let lsb = bin(fb, BinOp::BitwiseAnd, lsb, one)?;
            let half_ulp = splat(fb, u, 0xFFF)?;
            let bias = bin(fb, BinOp::IAdd, half_ulp, lsb)?;
            let r = bin(fb, BinOp::IAdd, b, bias)?;
            let keep = splat(fb, u, 0xFFFF_E000)?;
            let r = bin(fb, BinOp::BitwiseAnd, r, keep)?;
            // Past the largest half: the infinity of the sign.
            let abs_mask = splat(fb, u, 0x7FFF_FFFF)?;
            let mag = bin(fb, BinOp::BitwiseAnd, r, abs_mask)?;
            let largest = splat(fb, u, 0x477F_E000)?;
            let over = bin(fb, BinOp::UGreaterThan, mag, largest)?;
            let sign_mask = splat(fb, u, 0x8000_0000)?;
            let sign = bin(fb, BinOp::BitwiseAnd, b, sign_mask)?;
            let inf = splat(fb, u, 0x7F80_0000)?;
            let inf = bin(fb, BinOp::BitwiseOr, sign, inf)?;
            let r = fb.select(over, inf, r)?;
            let ft = ty(fb, f)?;
            fb.convert(ConvOp::Bitcast, r, ft)
        }
        _ => round(fb, f, x),
    }
}

/// Whether module `m` rounds to `half` through `OpQuantizeToF16` ([`round`],
/// [`round_exact_below_normal`], [`round_texel`] in [`QuantizeForm::Device`]): the module then
/// requires `VkProperty::MetalHalfQuantize`.
pub fn quantizes(m: &TypedModule) -> bool {
    m.blocks().iter().any(|b| {
        b.insts.iter().any(|n| {
            matches!(
                n.inst,
                Inst::Unary {
                    op: UnOp::QuantizeToF16,
                    ..
                }
            )
        })
    })
}

/// The `f32` bits `b` (form `u`) rounded to the nearest `bfloat` (ties to even), as `f32` bits
/// with the low half clear. A NaN stays a NaN of its sign (quieted); a finite value that rounds
/// past the largest `bfloat` becomes the infinity of its sign, as the carry into the exponent
/// gives.
fn bf16_round_bits(fb: &mut FunctionBuilder<'_>, u: Form, b: Value) -> BResult<Value> {
    let k16 = splat(fb, u, 16)?;
    let one = splat(fb, u, 1)?;
    let hi_mask = splat(fb, u, 0xFFFF_0000)?;
    let lsb = bin(fb, BinOp::ShiftRightLogical, b, k16)?;
    let lsb = bin(fb, BinOp::BitwiseAnd, lsb, one)?;
    let half_ulp = splat(fb, u, 0x7FFF)?;
    let bias = bin(fb, BinOp::IAdd, half_ulp, lsb)?;
    let r = bin(fb, BinOp::IAdd, b, bias)?;
    let r = bin(fb, BinOp::BitwiseAnd, r, hi_mask)?;
    // NaN: the magnitude bits above the infinity's.
    let abs_mask = splat(fb, u, 0x7FFF_FFFF)?;
    let inf = splat(fb, u, 0x7F80_0000)?;
    let mag = bin(fb, BinOp::BitwiseAnd, b, abs_mask)?;
    let nan = bin(fb, BinOp::UGreaterThan, mag, inf)?;
    let quiet = splat(fb, u, 0x0040_0000)?;
    let q = bin(fb, BinOp::BitwiseOr, b, quiet)?;
    let q = bin(fb, BinOp::BitwiseAnd, q, hi_mask)?;
    fb.select(nan, q, r)
}

/// An upper estimate of the SPIR-V words of one [`to_bits`] or [`from_bits`] on one lane of
/// form `f`: at most 22 instructions of at most six words for half, a shift for bfloat.
pub fn bits_words(f: Form) -> u32 {
    match f.float_fmt() {
        Some(FloatFmt::Half) => 136,
        Some(FloatFmt::BFloat) => 24,
        _ => 0,
    }
}

/// The format bits (zero-extended in `u32` lanes) of narrow value `x` of form `f`. Exact for
/// every value of the format, including denormals, infinities, and NaN payloads (a NaN whose
/// payload does not survive is quieted, never turned into an infinity).
///
/// Bits already known are reused: see `known_bits`; and a lane extract's bits are taken from its
/// vector's when those are known. Each is exactly what the sequence below computes from the value,
/// so the result does not depend on which is taken.
pub fn to_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let Some(fmt) = narrow_fmt(f) else {
        return encode(fb, f, x);
    };
    if let Some(b) = known_bits(fb, fmt, f, x)? {
        return Ok(b);
    }
    let mut lane = None;
    if f.lanes == 1 {
        if let Some((v, i)) = fb.lane_of(x) {
            if let Some(n) = vector_lanes(fb, v)? {
                let vf = Form { lanes: n, ..f };
                if let Some(vb) = known_bits(fb, fmt, vf, v)? {
                    let b = fb.composite_extract(vb, &[i])?;
                    fb.note_bits(fmt, x, b, 0, false);
                    return Ok(b);
                }
                // The lane of one encode of the vector, shared by its lanes' stores in this
                // block, is offered for the lane's own encode.
                let (vb, group) = match fb.encode_of(fmt, v) {
                    Some(e) => e,
                    None => {
                        let vb = encode(fb, vf, v)?;
                        let group = fb.offer_group();
                        fb.note_encode(fmt, v, vb, group);
                        (vb, group)
                    }
                };
                lane = Some((fb.composite_extract(vb, &[i])?, group));
            }
        }
    }
    let b = encode(fb, f, x)?;
    fb.note_bits(fmt, x, b, 0, false);
    if let Some((c, group)) = lane {
        fb.offer(group, b, c);
    }
    Ok(b)
}

/// The format bits of `x` (form `f`, format `fmt`) when they are known without running the
/// sequence, cleared above each lane's 16 bits: the bits `x` was decoded from ([`from_bits`] notes
/// them) or computed with earlier in the open block, a constant's bits computed at translation time
/// ([`constant_bits`]), or, for a vector built by lane inserts ([`note_insert`]), the inserts
/// repeated on the lanes' known bits. `None` otherwise.
fn known_bits(
    fb: &mut FunctionBuilder<'_>,
    fmt: NarrowFmt,
    f: Form,
    x: Value,
) -> BResult<Option<Value>> {
    if let Some((bits, dirty)) = fb.bits_of(fmt, x) {
        return clean(fb, fmt, f, x, bits, dirty).map(Some);
    }
    if let Some(k) = constant_bits(fb, f, x)? {
        return Ok(Some(k));
    }
    if f.lanes > 1 {
        return inserted_bits(fb, fmt, f, x);
    }
    Ok(None)
}

/// `bits` (the noted bits of `x`, dirty in the lanes `dirty` has a bit for) cleared above each
/// lane's 16 bits, noted for `x` as clean in the open block.
fn clean(
    fb: &mut FunctionBuilder<'_>,
    fmt: NarrowFmt,
    f: Form,
    x: Value,
    bits: Value,
    dirty: u32,
) -> BResult<Value> {
    if dirty == 0 {
        return Ok(bits);
    }
    let m = splat(fb, u32_form(f), 0xFFFF)?;
    let b = bin(fb, BinOp::BitwiseAnd, bits, m)?;
    fb.note_bits(fmt, x, b, 0, false);
    Ok(b)
}

/// The most lane inserts [`inserted_bits`] repeats on bits: a bound on its work (a longer chain
/// runs the sequence).
const MAX_INSERTS: usize = 16;

/// Where a lane insert's bits come from.
enum LaneBits {
    /// Noted bits, dirty or not.
    Noted(Value, bool),
    /// A constant's bits.
    Const(u64),
    /// Lane `.2` of a vector's noted bits (dirty in the lanes `.1` has a bit for).
    Of(Value, u32, u32),
}

/// Where the first vector of a chain of lane inserts gets its bits from.
enum BaseBits {
    Noted(Value, u32),
    Undef,
    Const(Vec<u64>),
}

/// The bits of vector `x` built by a chain of lane inserts ([`note_insert`]) from a vector
/// whose bits are known (noted, a constant's, or an `undef`'s, which may be any bits), each
/// inserted lane's bits known (noted, a constant's, or a lane of a vector's noted bits): the
/// same inserts on the bits. Each lane then holds the bits of the lane `x` holds, so they are
/// `x`'s bits. Checked before anything is emitted; `None` (nothing emitted) when a link is
/// unknown.
fn inserted_bits(
    fb: &mut FunctionBuilder<'_>,
    fmt: NarrowFmt,
    f: Form,
    x: Value,
) -> BResult<Option<Value>> {
    let mut steps: Vec<(LaneBits, u32)> = Vec::new();
    let mut at = x;
    let base = loop {
        if !steps.is_empty() {
            if let Some((b, d)) = fb.bits_of(fmt, at) {
                break BaseBits::Noted(b, d);
            }
            if fb.is_undef(at) {
                break BaseBits::Undef;
            }
            if let Some(k) = constant_lanes(fb, f, at)? {
                break BaseBits::Const(k);
            }
        }
        if steps.len() == MAX_INSERTS {
            return Ok(None);
        }
        let Some((v, e, i)) = fb.insert_of(at) else {
            return Ok(None);
        };
        let lane = Form { lanes: 1, ..f };
        let e_bits = if let Some((b, d)) = fb.bits_of(fmt, e) {
            LaneBits::Noted(b, d != 0)
        } else if let Some(k) = constant_lanes(fb, lane, e)? {
            LaneBits::Const(k.first().copied().unwrap_or(0))
        } else if let Some((w, j)) = fb.lane_of(e) {
            match fb.bits_of(fmt, w) {
                Some((wb, wd)) => LaneBits::Of(wb, wd, j),
                None => return Ok(None),
            }
        } else {
            return Ok(None);
        };
        steps.push((e_bits, i));
        at = v;
    };
    let all = u32::MAX >> (32 - u32::from(f.lanes).clamp(1, 32));
    let (mut rb, mut dirty) = match base {
        BaseBits::Noted(b, d) => (b, d),
        BaseBits::Undef => {
            let t = ty(fb, u32_form(f))?;
            (fb.undef(t)?, all)
        }
        BaseBits::Const(k) => (materialize(fb, f, &k)?, 0),
    };
    for (e_bits, i) in steps.into_iter().rev() {
        let (eb, d) = match e_bits {
            LaneBits::Noted(b, d) => (b, d),
            LaneBits::Const(k) => (splat(fb, u32_form(Form { lanes: 1, ..f }), k)?, false),
            LaneBits::Of(wb, wd, j) => (fb.composite_extract(wb, &[j])?, wd >> j & 1 != 0),
        };
        rb = fb.composite_insert(eb, rb, &[i])?;
        let bit = 1u32.checked_shl(i).unwrap_or(0);
        dirty = if d { dirty | bit } else { dirty & !bit };
    }
    fb.note_bits(fmt, x, rb, dirty, false);
    clean(fb, fmt, f, x, rb, dirty).map(Some)
}

/// Note that narrow vector `y` is `v` with lane `index` replaced by `e` (an `f32` scalar), for
/// `inserted_bits`; nothing for any other lane type.
pub fn note_insert(
    fb: &mut FunctionBuilder<'_>,
    y: Value,
    v: Value,
    e: Value,
    index: u32,
) -> BResult<()> {
    if may_be_narrow(fb, e)? {
        fb.note_insert(y, v, e, index);
    }
    Ok(())
}

/// `insertelement` of lane `e` into vector `v` at lane `index`: the `f32` insert, noted for
/// `inserted_bits`. When `e` is a narrow decode (its bits noted by [`from_bits`]) and `v` one too
/// or an `undef`, the same insert on the bits is decoded as a vector first and offered for the
/// `f32` insert: it holds the same lanes (each lane the decode of the bits it was decoded from; an
/// `undef` lane any value), and the insert's bits are noted, so a chain of such inserts decodes
/// once if the module is smaller for it.
pub fn insert(fb: &mut FunctionBuilder<'_>, v: Value, e: Value, index: u32) -> BResult<Value> {
    let mut offered = None;
    if may_be_narrow(fb, e)? {
        if let Some(n) = vector_lanes(fb, v)? {
            offered = decoded_insert(fb, v, e, index, n)?;
        }
    }
    let y = fb.composite_insert(e, v, &[index])?;
    note_insert(fb, y, v, e, index)?;
    if let Some((fmt, rb, dirty, yd)) = offered {
        fb.note_bits(fmt, y, rb, dirty, true);
        let group = fb.offer_group();
        fb.offer(group, y, yd);
    }
    Ok(y)
}

/// The bits insert and its vector decode for [`insert`] when both operands are decodes of known
/// bits (or the vector an `undef`): the format, the inserted bits and their dirty lanes, and the
/// decode. A declined offer leaves nothing behind: its instructions, the constants and vector
/// types only they read leave, and the integer scalars it reads are the lane decode's own.
fn decoded_insert(
    fb: &mut FunctionBuilder<'_>,
    v: Value,
    e: Value,
    index: u32,
    n: u8,
) -> BResult<Option<(NarrowFmt, Value, u32, Value)>> {
    for (fmt, held) in [
        (NarrowFmt::Half, FloatFmt::Half),
        (NarrowFmt::BFloat, FloatFmt::BFloat),
    ] {
        let Some((eb, ed)) = fb.decoded_from(fmt, e) else {
            continue;
        };
        let f = Form {
            held: Held::Float(held),
            lanes: n,
        };
        let all = u32::MAX >> (32 - u32::from(n).clamp(1, 32));
        let (vb, vd) = if let Some(noted) = fb.decoded_from(fmt, v) {
            noted
        } else if fb.is_undef(v) {
            let t = ty(fb, u32_form(f))?;
            (fb.undef(t)?, all)
        } else {
            return Ok(None);
        };
        let rb = fb.composite_insert(eb, vb, &[index])?;
        let bit = 1u32.checked_shl(index).unwrap_or(0);
        let dirty = if ed & 1 != 0 { vd | bit } else { vd & !bit };
        let yd = from_bits(fb, f, rb)?;
        return Ok(Some((fmt, rb, dirty, yd)));
    }
    Ok(None)
}

/// The builder's name of `f`'s narrow format, if it is one.
fn narrow_fmt(f: Form) -> Option<NarrowFmt> {
    match f.held {
        Held::Float(FloatFmt::Half) => Some(NarrowFmt::Half),
        Held::Float(FloatFmt::BFloat) => Some(NarrowFmt::BFloat),
        _ => None,
    }
}

/// Whether `y` may hold a narrow float: an `f32` scalar.
pub fn may_be_narrow(fb: &mut FunctionBuilder<'_>, y: Value) -> BResult<bool> {
    let t = fb.value_ty(y)?;
    Ok(matches!(
        fb.module().types().get(t)?,
        Type::Float { width: 32 }
    ))
}

/// The lane count of `v` when it is a vector.
fn vector_lanes(fb: &mut FunctionBuilder<'_>, v: Value) -> BResult<Option<u8>> {
    let t = fb.value_ty(v)?;
    Ok(match fb.module().types().get(t)? {
        Type::Vector { len, .. } => Some(*len),
        _ => None,
    })
}

/// The format bits of narrow value `x` of form `f`, computed by the exact sequence.
fn encode(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let b = fb.convert(ConvOp::Bitcast, x, ut)?;
    match f.held {
        Held::Float(FloatFmt::BFloat) => {
            let k16 = splat(fb, u, 16)?;
            bin(fb, BinOp::ShiftRightLogical, b, k16)
        }
        _ => f32_bits_to_half_bits(fb, u, b),
    }
}

/// The format bits of `x` (form `f`) when `x` is a plain constant every lane of which is a
/// value of the format, computed at translation time by the transliterations
/// [`half_bits_of_f32`] and [`f32_of_half_bits`] of the device sequences: `None` for anything
/// else (the device sequence then runs). A `null` is zero bits.
pub fn constant_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Option<Value>> {
    match constant_lanes(fb, f, x)? {
        Some(k) => materialize(fb, f, &k).map(Some),
        None => Ok(None),
    }
}

/// The lanes' bits [`constant_bits`] folds `x` to, without emitting anything.
fn constant_lanes(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Option<Vec<u64>>> {
    let ValueOrigin::Const(c) = fb.origin(x)? else {
        return Ok(None);
    };
    let lane = |bits: u64| -> Option<u64> {
        let b = u32::try_from(bits).ok()?;
        match f.held {
            Held::Float(FloatFmt::Half) => {
                let h = half_bits_of_f32(b);
                (f32_of_half_bits(h) == b).then_some(u64::from(h))
            }
            Held::Float(FloatFmt::BFloat) => (b & 0xFFFF == 0).then_some(u64::from(b >> 16)),
            _ => None,
        }
    };
    let consts = fb.module().consts();
    Ok(match &consts.get(c)?.kind {
        ConstKind::Plain(Constant::Scalar { bits, .. }) if f.lanes == 1 => {
            lane(bits.0).map(|b| vec![b])
        }
        ConstKind::Plain(Constant::Null { .. }) => Some(vec![0; usize::from(f.lanes)]),
        ConstKind::Plain(Constant::Composite { parts, .. })
            if parts.len() == usize::from(f.lanes) =>
        {
            parts
                .iter()
                .map(|p| consts.scalar_bits(*p).and_then(|b| lane(b.0)))
                .collect()
        }
        _ => None,
    })
}

/// The `u32` constant of form `u32_form(f)` with lanes `lanes`.
fn materialize(fb: &mut FunctionBuilder<'_>, f: Form, lanes: &[u64]) -> BResult<Value> {
    let u = u32_form(f);
    if f.lanes == 1 {
        return splat(fb, u, lanes.first().copied().unwrap_or(0));
    }
    let e = ty(fb, Form { lanes: 1, ..u })?;
    let mut parts = Vec::with_capacity(lanes.len());
    for b in lanes {
        parts.push(fb.module().constant(Constant::Scalar {
            ty: e,
            bits: ConstBits(*b),
        })?);
    }
    let t = ty(fb, u)?;
    let k = fb.module().constant(Constant::Composite {
        ty: t,
        parts: parts.into(),
    })?;
    fb.module().const_value(k)
}

/// The host transliteration of `f32_bits_to_half_bits` on one lane: the half bits of the
/// `f32` with bits `b` (exact when `b` is a half value).
pub fn half_bits_of_f32(b: u32) -> u32 {
    let a = b & 0x7FFF_FFFF;
    let sign = (b >> 16) & 0x8000;
    let t = a >> 13;
    let bias = if a >= 0x7F80_0000 {
        224 << 10
    } else {
        112 << 10
    };
    let normal = if a.wrapping_sub(0x7F80_0001) < 0x1FFF {
        0x7E00
    } else {
        t.wrapping_sub(bias)
    };
    let r = if a < 0x3880_0000 {
        // `|x| · 2^24`, an integer below 2^10 for a half denormal or zero.
        (f32::from_bits(a) * 16_777_216.0) as u32
    } else {
        normal
    };
    sign | r
}

/// The host transliteration of `half_bits_to_f32_bits` on one lane: the `f32` bits of the half
/// with bits `h` (its low 16 bits).
pub fn f32_of_half_bits(h: u32) -> u32 {
    let em = h & 0x7FFF;
    let bias = if em >= 0x7C00 { 224 << 23 } else { 112 << 23 };
    let r = if em < 0x0400 {
        (em as f32 * (1.0 / 16_777_216.0)).to_bits()
    } else {
        (em << 13).wrapping_add(bias)
    };
    ((h & 0x8000) << 16) | r
}

/// `f32` bits `b` of a half value to its half bits: the magnitude `a` rebiased by `112 << 10` after
/// dropping the 13 bits a half does not have; an infinity or NaN rebiased by `224 << 10`, to the
/// half's all-ones exponent with the payload's top bits (a NaN whose top bits are all zero is
/// quieted, never made infinite); a value below the smallest normal half (a half denormal or zero)
/// is `|x| · 2^24` converted to an integer, which is exact: `|x|` is a multiple of `2^-24` at least
/// `2^-24` (a normal `f32`, so no device flushes it) or zero, and the product is an integer below
/// `2^10`.
///
/// Under native half rounding ([`crate::spirv::builder::ModuleBuilder::native_half`]) the value
/// need not be a half value: the device's compiler may fold [`converted`]'s conversion pair, which
/// carries no barrier there, and hand over `x` unrounded (NVIDIA and lavapipe do). A finite
/// magnitude from `65536` up then rebiases past the half's exponent, to a NaN's code or beyond 16
/// bits, whose bit 17 a store's pairing carries into the next lane; so there the finite code is
/// capped at infinity's, `0x7C00`. Every other value takes the truncation, within the native
/// conversion's bound (`NativeHalfConversion`: one ulp, and 65504 where Metal answers infinity).
fn f32_bits_to_half_bits(fb: &mut FunctionBuilder<'_>, u: Form, b: Value) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: u64| splat(fb, u, k);
    let abs_mask = c(fb, 0x7FFF_FFFF)?;
    let a = bin(fb, BinOp::BitwiseAnd, b, abs_mask)?;
    let k16 = c(fb, 16)?;
    let sign = bin(fb, BinOp::ShiftRightLogical, b, k16)?;
    let sm = c(fb, 0x8000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, sign, sm)?;
    let k13 = c(fb, 13)?;
    let t = bin(fb, BinOp::ShiftRightLogical, a, k13)?;
    let inf = c(fb, 0x7F80_0000)?;
    let is_special = bin(fb, BinOp::UGreaterThanEqual, a, inf)?;
    let (special_bias, bias) = (c(fb, 224 << 10)?, c(fb, 112 << 10)?);
    let bias = fb.select(is_special, special_bias, bias)?;
    let normal = bin(fb, BinOp::ISub, t, bias)?;
    // The cap is the bias less `0x14400`: `0x7C00` for a finite lane, `0x23C00` for an infinity
    // or NaN, whose code (at most `0x7FFF`) it leaves alone.
    let normal = match fb.module().native_half() {
        true => {
            let less = c(fb, 0x14400)?;
            let cap = bin(fb, BinOp::ISub, bias, less)?;
            fb.glsl(GlslOp::UMin, &[normal, cap])?
        }
        false => normal,
    };
    // A NaN whose payload lies below the half's bits: `a` in `(0x7F80_0000, 0x7F80_2000)`.
    let first_nan = c(fb, 0x7F80_0001)?;
    let low = bin(fb, BinOp::ISub, a, first_nan)?;
    let span = c(fb, 0x1FFF)?;
    let lost = bin(fb, BinOp::ULessThan, low, span)?;
    let quiet = c(fb, 0x7E00)?;
    let normal = fb.select(lost, quiet, normal)?;
    // Only a denormal lane converts its magnitude (every other lane converts zero), so no
    // conversion is out of range.
    let min_normal = c(fb, 0x3880_0000)?;
    let is_den = bin(fb, BinOp::ULessThan, a, min_normal)?;
    let z = c(fb, 0)?;
    let small = fb.select(is_den, a, z)?;
    let wide = u.with(Held::Float(FloatFmt::F32));
    let ft = ty(fb, wide)?;
    let af = fb.convert(ConvOp::Bitcast, small, ft)?;
    let scale = fsplat(fb, wide, 16_777_216.0)?;
    let scaled = bin(fb, BinOp::FMul, af, scale)?;
    let ut = ty(fb, u)?;
    let den = fb.convert(ConvOp::ConvertFToU, scaled, ut)?;
    let r = fb.select(is_den, den, normal)?;
    bin(fb, BinOp::BitwiseOr, sign, r)
}

/// The narrow value (an `f32` of form `f`) of format bits `h`, the low 16 bits of each `u32`
/// lane; the bits above them are ignored, so a load need not clear them. Exact for every bit
/// pattern.
///
/// The result's bits are noted: a later [`to_bits`] of the value, wherever the value is, returns
/// `h` (masked to its low 16 bits) without converting back, since the two sequences are inverse on
/// every bit pattern.
///
/// Under native half rounding through the device's conversion ([`native_decode`]) a `half`'s exact
/// decode is offered the device's ([`native_from_bits`]) where no loop holds it
/// (`FunctionBuilder::offer_outside_loops`): NVIDIA's compiler makes a module faster with the
/// device's widening outside loops and slower with it inside one. Under the float controls, where
/// the profile leaves a NaN's bits to the device (`ModuleBuilder::native_half_nans`), a `half` is
/// the device's widening of its bits wherever it is ([`controlled_from_bits`]).
pub fn from_bits(fb: &mut FunctionBuilder<'_>, f: Form, h: Value) -> BResult<Value> {
    if matches!(f.held, Held::Float(FloatFmt::Half)) && fb.module().native_half_nans() {
        let x = controlled_from_bits(fb, f, h)?;
        let all = u32::MAX >> (32 - u32::from(f.lanes).clamp(1, 32));
        fb.note_bits(NarrowFmt::Half, x, h, all, true);
        return Ok(x);
    }
    // The device's decode is emitted first: an offered value precedes the one it stands for.
    let native = match f.held {
        Held::Float(FloatFmt::Half) if native_decode(fb) => Some(native_from_bits(fb, f, h)?),
        _ => None,
    };
    let u = u32_form(f);
    let b = match f.held {
        Held::Float(FloatFmt::BFloat) => {
            let k16 = splat(fb, u, 16)?;
            bin(fb, BinOp::ShiftLeftLogical, h, k16)?
        }
        _ => half_bits_to_f32_bits(fb, u, h)?,
    };
    let ft = ty(fb, f)?;
    let x = fb.convert(ConvOp::Bitcast, b, ft)?;
    if let Some(n) = native {
        fb.offer_outside_loops(x, n);
    }
    if let Some(fmt) = narrow_fmt(f) {
        let all = u32::MAX >> (32 - u32::from(f.lanes).clamp(1, 32));
        fb.note_bits(fmt, x, h, all, true);
    }
    Ok(x)
}

/// Whether the module's `half` decodes may be the device's widening: under native half rounding
/// through the device's conversion ([`QuantizeForm::Converted`] where the profile does not take
/// `DriverDeviation::MetalHalfConversion`), whose note, `NativeHalfConversion`, bounds what the
/// device makes of a `half`: a denormal may be the zero of its sign on a device that flushes 16-bit
/// denormals, and a NaN's encoding is the device's. Every other value widens exactly.
fn native_decode(fb: &mut FunctionBuilder<'_>) -> bool {
    fb.module().native_half() && fb.module().quantize_form() == QuantizeForm::Converted
}

/// The `half` values of bits `h` (the low 16 bits of each `u32` lane of form [`u32_form`]`(f)`)
/// widened by the device: each lane's `UnpackHalf2x16`, whose first component reads the low 16
/// bits.
fn native_from_bits(fb: &mut FunctionBuilder<'_>, f: Form, h: Value) -> BResult<Value> {
    if f.lanes <= 1 {
        let pair = fb.glsl(GlslOp::UnpackHalf2x16, &[h])?;
        return fb.composite_extract(pair, &[0]);
    }
    let mut lanes = Vec::with_capacity(usize::from(f.lanes));
    for i in 0..f.lanes {
        let lane = fb.composite_extract(h, &[u32::from(i)])?;
        let pair = fb.glsl(GlslOp::UnpackHalf2x16, &[lane])?;
        lanes.push(fb.composite_extract(pair, &[0])?);
    }
    let ft = ty(fb, f)?;
    fb.composite_construct(ft, &lanes)
}

/// The `half` values of bits `h` (the low 16 bits of each `u32` lane of form [`u32_form`]`(f)`)
/// widened by the device under the float controls ([`QuantizeForm::Controlled`],
/// [`controlled_modes`], which the entry declares where a module widens so, [`widens_bits`]): each
/// lane's bits reinterpreted as a pair of 16-bit floats (`OpBitcast` maps a lane's lower-ordered
/// bits to the pair's first component) and that first component converted to 32 bits. Every
/// `half` but a NaN widens to its exact value, by the Vulkan spec ("Precision and Operation of
/// SPIR-V Instructions"): a conversion is "Correctly rounded", and every `half` is an `f32`; under
/// `DenormPreserve` "Denormalized values obtained via unpacking an integer into a vector of values
/// with smaller bit width and interpreting those values as floating-point numbers must be
/// preserved"; and `OpFConvert` keeps signed zeros and infinities under
/// `SignedZeroInfNanPreserve`. A NaN widens to a NaN whose bits are the device's (the SPIR-V
/// spec's `SignedZeroInfNanPreserve`: "Bit patterns for NaNs might not be preserved"). Metal widens
/// every `half` NaN to `0x7FC00000` and every other `half` to its value (the M3 Max, every bit
/// pattern, loaded from device and constant memory, safe and fast math), which the NaN switch
/// answers at the widening to `float` (`nan::cast_answer`) on [`half_bits_to_f32_bits`]'s decode.
fn controlled_from_bits(fb: &mut FunctionBuilder<'_>, f: Form, h: Value) -> BResult<Value> {
    let e = fb.module().ty(Type::Float { width: 16 })?;
    let pair = fb.module().ty(Type::Vector { elem: e, len: 2 })?;
    let lane_form = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let w = ty(fb, lane_form)?;
    let widen = |fb: &mut FunctionBuilder<'_>, lane: Value| -> BResult<Value> {
        let halves = fb.convert(ConvOp::Bitcast, lane, pair)?;
        let half = fb.composite_extract(halves, &[0])?;
        fb.convert(ConvOp::FConvert, half, w)
    };
    if f.lanes <= 1 {
        return widen(fb, h);
    }
    let mut lanes = Vec::with_capacity(usize::from(f.lanes));
    for i in 0..f.lanes {
        let lane = fb.composite_extract(h, &[u32::from(i)])?;
        lanes.push(widen(fb, lane)?);
    }
    let ft = ty(fb, f)?;
    fb.composite_construct(ft, &lanes)
}

/// Whether module `m` widens a `half` from its bits by [`controlled_from_bits`]: an `OpFConvert` to
/// 32 bits of a component extracted from an `OpBitcast` to a vector of 16-bit floats. The entry
/// then declares [`controlled_modes`], and the module records `PrecisionNote::NativeNanEncoding`
/// (a NaN's bits the device's). Three passes over the instructions, with two bits per value of
/// the module charged to `meter` while they run.
pub fn widens_bits(m: &TypedModule, meter: &mut Meter) -> Result<bool, Error> {
    // Whether `ty` is a 16-bit float vector, or a 32-bit float.
    let is = |ty, vector16: bool| -> bool {
        match m.types().get(ty) {
            Ok(Type::Vector { elem, .. }) if vector16 => {
                matches!(m.types().get(*elem), Ok(Type::Float { width: 16 }))
            }
            Ok(Type::Float { width: 32 }) => !vector16,
            _ => false,
        }
    };
    let insts = || m.blocks().iter().flat_map(|b| &b.insts);
    let words = m.values().len().div_ceil(64);
    meter.charge(16 * words)?;
    let mut halves = vec![0u64; words];
    let mut lanes = vec![0u64; words];
    let set = |bits: &mut [u64], v: Value| {
        if let Some(w) = bits.get_mut(v.index() / 64) {
            *w |= 1 << (v.index() % 64);
        }
    };
    let has = |bits: &[u64], v: Value| {
        bits.get(v.index() / 64)
            .is_some_and(|w| w & (1 << (v.index() % 64)) != 0)
    };
    for n in insts() {
        if let (
            Inst::Convert {
                op: ConvOp::Bitcast,
                ..
            },
            Some(r),
        ) = (&n.inst, n.result)
        {
            if m.value(r).is_ok_and(|v| is(v.ty, true)) {
                set(&mut halves, r);
            }
        }
    }
    for n in insts() {
        if let (Inst::CompositeExtract { composite, .. }, Some(r)) = (&n.inst, n.result) {
            if has(&halves, *composite) {
                set(&mut lanes, r);
            }
        }
    }
    let found = insts().any(|n| match (&n.inst, n.result) {
        (
            Inst::Convert {
                op: ConvOp::FConvert,
                a,
                ..
            },
            Some(r),
        ) => has(&lanes, *a) && m.value(r).is_ok_and(|v| is(v.ty, false)),
        _ => false,
    });
    drop((halves, lanes));
    meter.release(16 * words);
    Ok(found)
}

/// Whether module `m` computes in 16-bit floats: an `OpFAdd`, `OpFSub` or `OpFMul`, an `OpFmaKHR`
/// or a `GLSL.std.450 Fma` whose result is a 16-bit float or a vector of them, the device's own
/// `half` arithmetic (`super::ops::f16_binary`, `super::fma`). Under the 16-bit float controls
/// ([`QuantizeForm::Controlled`]) the entry then declares [`controlled_modes`], which make that
/// arithmetic Metal's. One pass over the instructions.
pub fn computes_in_16_bits(m: &TypedModule) -> bool {
    let half = |v: Value| -> bool {
        let Ok(ty) = m.value(v).map(|v| v.ty) else {
            return false;
        };
        let ty = match m.types().get(ty) {
            Ok(Type::Vector { elem, .. }) => *elem,
            _ => ty,
        };
        matches!(m.types().get(ty), Ok(Type::Float { width: 16 }))
    };
    m.blocks().iter().flat_map(|b| &b.insts).any(|n| {
        let arithmetic = match &n.inst {
            Inst::Binary { op, .. } => matches!(op, BinOp::FAdd | BinOp::FSub | BinOp::FMul),
            Inst::Fma { .. } => true,
            Inst::Glsl { op, .. } => *op == GlslOp::Fma,
            _ => false,
        };
        arithmetic && n.result.is_some_and(half)
    })
}

/// Half bits `h` (the low 16 bits of each lane) to the `f32` bits of the same value: the
/// magnitude's bits moved up by 13 and rebiased by `112 << 23`, or, for an infinity or NaN, by `224
/// << 23`, which gives the `f32`'s all-ones exponent with the payload moved up (the half exponent
/// 31 moved up is `0x0F80_0000`); a denormal (or zero) `m · 2^-24` computed as the conversion of
/// the integer `m` times `2^-24`, which is exact: `m` is below `2^10`, and the product is a zero or
/// at least `2^-24`, a normal `f32`, so no device flushes it.
fn half_bits_to_f32_bits(fb: &mut FunctionBuilder<'_>, u: Form, h: Value) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: u64| splat(fb, u, k);
    let mag_mask = c(fb, 0x7FFF)?;
    let em = bin(fb, BinOp::BitwiseAnd, h, mag_mask)?;
    let k13 = c(fb, 13)?;
    let t = bin(fb, BinOp::ShiftLeftLogical, em, k13)?;
    let exp_ones = c(fb, 0x7C00)?;
    let is_special = bin(fb, BinOp::UGreaterThanEqual, em, exp_ones)?;
    let (special_bias, bias) = (c(fb, 224 << 23)?, c(fb, 112 << 23)?);
    let bias = fb.select(is_special, special_bias, bias)?;
    let normal = bin(fb, BinOp::IAdd, t, bias)?;
    let wide = u.with(Held::Float(FloatFmt::F32));
    let ft = ty(fb, wide)?;
    let m = fb.convert(ConvOp::ConvertUToF, em, ft)?;
    let scale = fsplat(fb, wide, 1.0 / 16_777_216.0)?;
    let d = bin(fb, BinOp::FMul, m, scale)?;
    let ut = ty(fb, u)?;
    let den = fb.convert(ConvOp::Bitcast, d, ut)?;
    let min_normal = c(fb, 0x0400)?;
    let is_den = bin(fb, BinOp::ULessThan, em, min_normal)?;
    let r = fb.select(is_den, den, normal)?;
    let sm = c(fb, 0x8000)?;
    let sign = bin(fb, BinOp::BitwiseAnd, h, sm)?;
    let k16 = c(fb, 16)?;
    let sign = bin(fb, BinOp::ShiftLeftLogical, sign, k16)?;
    bin(fb, BinOp::BitwiseOr, sign, r)
}

/// `x` (`double` lanes of form `from`) converted to `f32` rounding to odd: the result is
/// truncated toward zero and its lowest bit set when inexact, so one later rounding to a format
/// of at most 22 significand bits is correctly rounded. Independent of the device's rounding
/// mode (the step back toward zero undoes a rounding away).
pub fn f64_to_f32_odd(fb: &mut FunctionBuilder<'_>, from: Form, x: Value) -> BResult<Value> {
    let f32f = from.with(Held::Float(FloatFmt::F32));
    let u = u32_form(from);
    let t32 = ty(fb, f32f)?;
    let ut = ty(fb, u)?;
    let t64 = ty(fb, from)?;
    let r = fb.convert(ConvOp::FConvert, x, t32)?;
    let back = fb.convert(ConvOp::FConvert, r, t64)?;
    let ab = fb.glsl(GlslOp::FAbs, &[back])?;
    let ax = fb.glsl(GlslOp::FAbs, &[x])?;
    let away = bin(fb, BinOp::FOrdGreaterThan, ab, ax)?;
    let rb = fb.convert(ConvOp::Bitcast, r, ut)?;
    let one = splat(fb, u, 1)?;
    let toward = bin(fb, BinOp::ISub, rb, one)?;
    let rb = fb.select(away, toward, rb)?;
    let inexact = bin(fb, BinOp::FUnordNotEqual, back, x)?;
    let odd = bin(fb, BinOp::BitwiseOr, rb, one)?;
    let rb = fb.select(inexact, odd, rb)?;
    fb.convert(ConvOp::Bitcast, rb, t32)
}

/// Unsigned magnitude `m` (`u32` lanes of form `u`) converted to `f32` rounding to odd: the
/// bits below the 24 the significand keeps are folded into a sticky lowest bit before an exact
/// conversion, then scaled back exactly.
pub fn int_to_f32_odd(fb: &mut FunctionBuilder<'_>, u: Form, to: Form, m: Value) -> BResult<Value> {
    let msb = fb.glsl(GlslOp::FindUMsb, &[m])?;
    let k23 = splat(fb, u, 23)?;
    let s = bin(fb, BinOp::ISub, msb, k23)?;
    let z = splat(fb, u, 0)?;
    let s = fb.glsl(GlslOp::SMax, &[s, z])?;
    let t = bin(fb, BinOp::ShiftRightLogical, m, s)?;
    let kept = bin(fb, BinOp::ShiftLeftLogical, t, s)?;
    let lost = bin(fb, BinOp::INotEqual, kept, m)?;
    let one = splat(fb, u, 1)?;
    let sticky = fb.select(lost, one, z)?;
    let t = bin(fb, BinOp::BitwiseOr, t, sticky)?;
    let f32f = to.with(Held::Float(FloatFmt::F32));
    let ft = ty(fb, f32f)?;
    let f = fb.convert(ConvOp::ConvertUToF, t, ft)?;
    fb.glsl(GlslOp::Ldexp, &[f, s])
}

/// Unsigned 64-bit magnitude `m` (`u64` lanes of form `w`) converted to `f32` rounding to odd,
/// as [`int_to_f32_odd`] does for 32 bits: the most significant bit is found from the two
/// halves, the bits below the 24 kept are folded into a sticky bit, and the (at most 24-bit)
/// rest converts exactly and is scaled back exactly.
pub fn int64_to_f32_odd(
    fb: &mut FunctionBuilder<'_>,
    w: Form,
    to: Form,
    m: Value,
) -> BResult<Value> {
    let u = u32_form(w);
    let ut = ty(fb, u)?;
    let wt = ty(fb, w)?;
    let lo = fb.convert(ConvOp::UConvert, m, ut)?;
    let k32w = splat(fb, w, 32)?;
    let hi = bin(fb, BinOp::ShiftRightLogical, m, k32w)?;
    let hi = fb.convert(ConvOp::UConvert, hi, ut)?;
    let mlo = fb.glsl(GlslOp::FindUMsb, &[lo])?;
    let mhi = fb.glsl(GlslOp::FindUMsb, &[hi])?;
    let k32 = splat(fb, u, 32)?;
    let mhi = bin(fb, BinOp::IAdd, mhi, k32)?;
    let z = splat(fb, u, 0)?;
    let hz = bin(fb, BinOp::IEqual, hi, z)?;
    let msb = fb.select(hz, mlo, mhi)?;
    let k23 = splat(fb, u, 23)?;
    let s = bin(fb, BinOp::ISub, msb, k23)?;
    let s = fb.glsl(GlslOp::SMax, &[s, z])?;
    let s64 = fb.convert(ConvOp::UConvert, s, wt)?;
    let t = bin(fb, BinOp::ShiftRightLogical, m, s64)?;
    let kept = bin(fb, BinOp::ShiftLeftLogical, t, s64)?;
    let lost = bin(fb, BinOp::INotEqual, kept, m)?;
    let (one, z64) = (splat(fb, w, 1)?, splat(fb, w, 0)?);
    let sticky = fb.select(lost, one, z64)?;
    let t = bin(fb, BinOp::BitwiseOr, t, sticky)?;
    let f32f = to.with(Held::Float(FloatFmt::F32));
    let ft = ty(fb, f32f)?;
    let f = fb.convert(ConvOp::ConvertUToF, t, ft)?;
    fb.glsl(GlslOp::Ldexp, &[f, s])
}

/// `fma(a, b, c)` of `bfloat` values held in `f32` (form `f`), as an `f32` whose one rounding
/// to `bfloat` ([`round`]) is the correctly rounded fused result.
///
/// `a * b` is exact in `f32` (two 8-bit significands), so `s = Fma(a, b, c)` is the exact sum
/// `a * b + c` rounded once to `f32` in the device's mode, fused or not. Rounding `s` to
/// `bfloat` is then correct unless `s` is exactly a `bfloat` tie (its low 16 bits `0x8000`)
/// while the exact sum is not: only then does the direction of the first rounding decide. That
/// direction is exact: with `big`/`small` the addends of larger/smaller magnitude, `t = s -
/// big` is exact (Sterbenz, or the sum itself is exact), so comparing `small` with `t` says
/// whether the exact sum lies above or below `s`, and `s` moves one `f32` ulp that way before
/// the rounding. The correction applies only when the rounded product `p` is a finite normal
/// `f32`: the product has at most 16 significant bits, so such a `p` is exact (below `2^-126`
/// its gap to `2^-126` exceeds half a denormal ulp). An overflowed (`±inf`) or denormal `p`
/// keeps the device's `s` unchanged, so the result is exact while `a * b` is a normal `f32` and
/// is never made worse otherwise (there `bfloat` follows the device's `float` behavior).
pub fn bf16_fma(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    a: Value,
    b: Value,
    c: Value,
) -> BResult<Value> {
    use crate::spirv::ir::Contraction;
    let s = fb.glsl(GlslOp::Fma, &[a, b, c])?;
    let p = fb.binary_with(BinOp::FMul, a, b, Contraction::Forbidden)?;
    let ap = fb.glsl(GlslOp::FAbs, &[p])?;
    let ac = fb.glsl(GlslOp::FAbs, &[c])?;
    let p_big = bin(fb, BinOp::FOrdGreaterThanEqual, ap, ac)?;
    let big = fb.select(p_big, p, c)?;
    let small = fb.select(p_big, c, p)?;
    let t = fb.binary_with(BinOp::FSub, s, big, Contraction::Forbidden)?;
    let above = bin(fb, BinOp::FOrdGreaterThan, small, t)?;
    let below = bin(fb, BinOp::FOrdLessThan, small, t)?;
    let u = u32_form(f);
    let ut = ty(fb, u)?;
    let sb = fb.convert(ConvOp::Bitcast, s, ut)?;
    let low = splat(fb, u, 0xFFFF)?;
    let low = bin(fb, BinOp::BitwiseAnd, sb, low)?;
    let tie_bits = splat(fb, u, 0x8000)?;
    let tie = bin(fb, BinOp::IEqual, low, tie_bits)?;
    // The correction's premise: `p` is the exact product, a finite normal `f32` (exponent
    // field 1..=254, so `bits(|p|) - 2^23 < 254 * 2^23` unsigned). Otherwise `s` is kept.
    let pb = fb.convert(ConvOp::Bitcast, ap, ut)?;
    let min_normal = splat(fb, u, 0x0080_0000)?;
    let pe = bin(fb, BinOp::ISub, pb, min_normal)?;
    let normal_span = splat(fb, u, 0x7F00_0000)?;
    let p_normal = bin(fb, BinOp::ULessThan, pe, normal_span)?;
    let tie = bin(fb, BinOp::LogicalAnd, tie, p_normal)?;
    let k31 = splat(fb, u, 31)?;
    let sign = bin(fb, BinOp::ShiftRightLogical, sb, k31)?;
    let z = splat(fb, u, 0)?;
    let neg = bin(fb, BinOp::INotEqual, sign, z)?;
    let one = splat(fb, u, 1)?;
    let plus = bin(fb, BinOp::IAdd, sb, one)?;
    let minus = bin(fb, BinOp::ISub, sb, one)?;
    // One ulp toward +inf / -inf (the magnitude grows toward the sign's infinity).
    let up = fb.select(neg, minus, plus)?;
    let down = fb.select(neg, plus, minus)?;
    let nb = fb.select(above, up, sb)?;
    let nb = fb.select(below, down, nb)?;
    let nb = fb.select(tie, nb, sb)?;
    let ft = ty(fb, f)?;
    fb.convert(ConvOp::Bitcast, nb, ft)
}

/// Float resize (`fpext`/`fptrunc`/float `air.convert`) of `x` from `from` to `to`, exact up to
/// the one rounding to `to`'s format.
pub fn resize(fb: &mut FunctionBuilder<'_>, from: Form, to: Form, x: Value) -> BResult<Value> {
    let (Held::Float(src), Held::Float(dst)) = (from.held, to.held) else {
        return Ok(x);
    };
    if src == dst {
        return Ok(x);
    }
    // Narrow or widen the container first; only `double -> f32` rounds there.
    let x = match (src, dst) {
        (FloatFmt::F64, FloatFmt::Half | FloatFmt::BFloat) => f64_to_f32_odd(fb, from, x)?,
        (FloatFmt::F64, _) | (_, FloatFmt::F64) => {
            let t = ty(fb, to)?;
            fb.convert(ConvOp::FConvert, x, t)?
        }
        _ => x,
    };
    // Every value of a narrower format is one of the wider (half and bfloat are not ordered,
    // so between them the destination rounds).
    let widening = matches!(
        (src, dst),
        (
            FloatFmt::Half | FloatFmt::BFloat,
            FloatFmt::F32 | FloatFmt::F64
        ) | (FloatFmt::F32, FloatFmt::F64)
    );
    if widening {
        Ok(x)
    } else {
        round(fb, to, x)
    }
}

/// The layout of an 8-bit float format (`air.convert` sources, carried in `i8` lanes). The
/// names follow LLVM's `APFloat` semantics of the same spellings: `f8e4m3` and `f8e5m2` are
/// IEEE-like (an all-ones exponent is infinity or NaN), `f8e4m3fn` is finite (only
/// `S.1111.111` is NaN).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp8Layout {
    pub exp: u32,
    pub man: u32,
    pub bias: u32,
    pub ieee: bool,
}

impl Fp8Layout {
    pub fn of(f: crate::air::intrinsics::Fp8) -> Fp8Layout {
        use crate::air::intrinsics::Fp8;
        match f {
            Fp8::E4M3 => Fp8Layout {
                exp: 4,
                man: 3,
                bias: 7,
                ieee: true,
            },
            Fp8::E4M3Fn => Fp8Layout {
                exp: 4,
                man: 3,
                bias: 7,
                ieee: false,
            },
            Fp8::E5M2 => Fp8Layout {
                exp: 5,
                man: 2,
                bias: 15,
                ieee: true,
            },
        }
    }

    /// The value of bits `b` (the reference decoding, for tests and constants).
    #[cfg(test)]
    pub fn value(self, b: u8) -> f64 {
        let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = (u32::from(b) >> self.man) & ((1 << self.exp) - 1);
        let m = u32::from(b) & ((1 << self.man) - 1);
        let top = (1 << self.exp) - 1;
        if self.ieee && e == top {
            return if m == 0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            };
        }
        if !self.ieee && e == top && m == (1 << self.man) - 1 {
            return f64::NAN;
        }
        let m = f64::from(m) / f64::from(1u32 << self.man);
        if e == 0 {
            sign * m * 2f64.powi(1 - self.bias as i32)
        } else {
            sign * (1.0 + m) * 2f64.powi(e as i32 - self.bias as i32)
        }
    }
}

/// The value (an `f32` of form `to`'s lanes) of 8-bit float bits `b` (zero-extended in `u32`
/// lanes of form `u`). Exact: every 8-bit float value is an `f32` (and a `half` and a `bfloat`)
/// value.
pub fn fp8_to_f32(
    fb: &mut FunctionBuilder<'_>,
    l: Fp8Layout,
    u: Form,
    to: Form,
    b: Value,
) -> BResult<Value> {
    let c = |fb: &mut FunctionBuilder<'_>, k: u64| splat(fb, u, k);
    let top = u64::from((1u32 << l.exp) - 1);
    let mmax = u64::from((1u32 << l.man) - 1);
    let (k24, k23, kman) = (c(fb, 24)?, c(fb, 23)?, c(fb, u64::from(l.man))?);
    let k80 = c(fb, 0x80)?;
    let sign = bin(fb, BinOp::BitwiseAnd, b, k80)?;
    let sign = bin(fb, BinOp::ShiftLeftLogical, sign, k24)?;
    let e = bin(fb, BinOp::ShiftRightLogical, b, kman)?;
    let et = c(fb, top)?;
    let e = bin(fb, BinOp::BitwiseAnd, e, et)?;
    let mm = c(fb, mmax)?;
    let m = bin(fb, BinOp::BitwiseAnd, b, mm)?;
    let up = c(fb, u64::from(23 - l.man))?;
    let mup = bin(fb, BinOp::ShiftLeftLogical, m, up)?;
    let rebias = c(fb, u64::from(127 - l.bias))?;
    let ne = bin(fb, BinOp::IAdd, e, rebias)?;
    let ne = bin(fb, BinOp::ShiftLeftLogical, ne, k23)?;
    let normal = bin(fb, BinOp::BitwiseOr, ne, mup)?;
    // Denormal: m * 2^(1 - bias - man), normalized from its most significant bit.
    let msb = fb.glsl(GlslOp::FindUMsb, &[m])?;
    let dbias = c(fb, u64::from(127 + 1 - l.bias - l.man))?;
    let de = bin(fb, BinOp::IAdd, msb, dbias)?;
    let de = bin(fb, BinOp::ShiftLeftLogical, de, k23)?;
    let sh = bin(fb, BinOp::ISub, k23, msb)?;
    let k31 = c(fb, 31)?;
    let sh = fb.glsl(GlslOp::UMin, &[sh, k31])?;
    let dm = bin(fb, BinOp::ShiftLeftLogical, m, sh)?;
    let man = c(fb, 0x7F_FFFF)?;
    let dm = bin(fb, BinOp::BitwiseAnd, dm, man)?;
    let den = bin(fb, BinOp::BitwiseOr, de, dm)?;
    let z = c(fb, 0)?;
    let m_zero = bin(fb, BinOp::IEqual, m, z)?;
    let den = fb.select(m_zero, z, den)?;
    let e_zero = bin(fb, BinOp::IEqual, e, z)?;
    let r = fb.select(e_zero, den, normal)?;
    let e_top = bin(fb, BinOp::IEqual, e, et)?;
    let r = if l.ieee {
        let inf = c(fb, 0x7F80_0000)?;
        let special = bin(fb, BinOp::BitwiseOr, inf, mup)?;
        fb.select(e_top, special, r)?
    } else {
        let m_top = bin(fb, BinOp::IEqual, m, mm)?;
        let nan_bits = bin(fb, BinOp::LogicalAnd, e_top, m_top)?;
        let qnan = c(fb, 0x7FC0_0000)?;
        fb.select(nan_bits, qnan, r)?
    };
    let r = bin(fb, BinOp::BitwiseOr, sign, r)?;
    let ft = ty(fb, to.with(Held::Float(FloatFmt::F32)))?;
    fb.convert(ConvOp::Bitcast, r, ft)
}
