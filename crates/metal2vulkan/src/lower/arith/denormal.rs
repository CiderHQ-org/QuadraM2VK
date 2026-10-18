//! How Metal reads an `f32` denormal operand, per operation: the one table of the translation
//! (the denormal-operand model extends it).
//!
//! Metal flushes most `float` operands whose exponent field is zero to a zero of their sign, but
//! not every operation does, and the measured exceptions are per operation and per math mode:
//! precise `asin` answers a denormal operand itself, where `fast::asin` answers `+0` for it (cases
//! 087758cb and cc234732). A lowering that computes an operation itself,
//! rather than leaving it to a device that is Metal's own (`FloatFunctions::Metal`), reads the
//! operand as this table says, before its own special values. The table is decided on the exponent
//! field, so the answer is the same whatever the device's own denormal mode. Only `float` operands
//! are listed: a widened `half` is never an `f32` denormal, and Metal's `bfloat` and `double`
//! environments are not measured.

use crate::device::profile::FloatFunctions;
use crate::device::DeviceProfile;
use crate::error::DriverDeviation;
use crate::lower::arith::fma;
use crate::lower::arith::legal::Form;
use crate::lower::arith::math::FloatFn;
use crate::lower::arith::narrow::u32_form;
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, Contraction, ConvOp, GlslOp};

/// How Metal reads an `f32` denormal operand of one operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// As the zero of its sign.
    Flush,
    /// As itself.
    Keep,
}

/// How Metal reads an `f32` denormal operand of float function `op`, where the table states it
/// (`None` for an operation it does not list yet).
///
/// | operation | operand | Metal's answer for a denormal operand (cases 087758cb, cc234732) |
/// |---|---|---|
/// | `atan` | flushed | `+0`, whatever its sign (as `atan(-0)`) |
/// | `fast::atan` | flushed | the zero of its sign |
/// | `asin` | kept | the operand itself |
/// | `fast::asin` | flushed | `+0`, whatever its sign (as `fast::asin(-0)`) |
/// | `acos`, `fast::acos` | flushed | `π/2` (either reading gives it) |
/// | `atan2`, `fast::atan2` | flushed | as a zero operand |
/// | `sin`, `cos`, `tan` | kept | `sin` and `tan`: the operand itself; `cos`: 1 |
/// | `fast::sin`, `fast::cos`, `fast::tan` | kept | as `sin`, `cos`, and `tan` (case cc234732, where RADV, ANV, and NVIDIA flush it) |
/// | `sinpi`, `cospi`, `tanpi`, and their `fast::` forms | flushed | `sinpi` and `tanpi`: `+0`, whatever its sign; `cospi`: 1 (both modes) |
pub fn operand(op: FloatFn) -> Option<Operand> {
    use FloatFn as F;
    Some(match op {
        F::Asin | F::Sin | F::Cos | F::Tan | F::FastSin | F::FastCos | F::FastTan { .. } => {
            Operand::Keep
        }
        F::Atan
        | F::FastAtan
        | F::FastAsin
        | F::Acos
        | F::FastAcos
        | F::Atan2
        | F::FastAtan2 { .. }
        | F::SinPi
        | F::CosPi
        | F::TanPi
        | F::FastSinPi
        | F::FastCosPi
        | F::FastTanPi => Operand::Flush,
        _ => return None,
    })
}

/// `x` (`float` lanes of form `f`) as operation `op` reads it ([`operand`]): a denormal lane
/// replaced by the zero of its sign where the table flushes it; `x` itself otherwise.
pub fn read(fb: &mut FunctionBuilder<'_>, op: FloatFn, f: Form, x: Value) -> BResult<Value> {
    if operand(op) != Some(Operand::Flush) {
        return Ok(x);
    }
    flush(fb, f, x)
}

/// Whether profile `p`'s device reads the denormals of the operations [`Covered::Deviation`]
/// names as Metal does: an `f32` denormal operand as the zero of its sign, a denormal result
/// flushed to one (`DriverDeviation::MetalFloatDenormals`, a measured driver fact). Elsewhere
/// the module flushes them itself ([`Flush`]).
pub fn device_flushes(p: &DeviceProfile) -> bool {
    p.takes_driver_deviation(DriverDeviation::MetalFloatDenormals)
}

/// Where an operation the device computes reads its `f32` denormals as Metal does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Covered {
    /// On a device with the measured deviation ([`device_flushes`]); elsewhere the module
    /// flushes them itself.
    Deviation,
    /// On no device measured: the module flushes them itself on every profile. `Floor` and
    /// `Ceil` of a denormal give `-1` and `1` on NVIDIA's 615.71.09 and on lavapipe, where Metal
    /// reads the operand as a zero (`floor(-2^-149)` is `-0`, `ceil(2^-149)` is `+0`).
    Never,
}

/// How Metal reads the `f32` denormals of an operation the device computes, and where the
/// device reads them so ([`coverage`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reading {
    /// Where the device's reading is Metal's.
    pub covered: Covered,
    /// A bit per operand Metal reads as the zero of its sign where it is a denormal.
    pub operands: u8,
    /// Whether Metal flushes a denormal result.
    pub result: bool,
}

impl Reading {
    /// `+ - *` and the comparisons (`result` false for a comparison, whose result is a `bool`):
    /// both operands, on the deviation.
    pub const fn binary(result: bool) -> Reading {
        Reading {
            covered: Covered::Deviation,
            operands: 0b11,
            result,
        }
    }
}

/// How the device's instruction for float function `op` (on `float` lanes, where the lowering
/// leaves `op` to it) reads `f32` denormals against Metal, which reads each `float` operand of
/// these as the zero of its sign and flushes a denormal result (the denormal probes,
/// `nan_denormals` and `nan_denormals_fast`, every lane). `None` for a function whose lowering
/// reads its denormals as Metal does itself (`FloatFn::metal_denormals`) or whose reading is not
/// measured.
///
/// | function | operands | result | where the device's reading is Metal's |
/// |---|---|---|---|
/// | `floor`, `ceil` | the operand | integral | no device measured |
/// | `trunc`, `rint` | the operand | integral | the deviation |
/// | `sqrt`, `rsqrt`, `fast::rsqrt` | the operand | never denormal | the deviation |
/// | `dot` | both vectors | flushed | the deviation |
/// | `fma`, `llvm.fmuladd` (`OpFmaKHR`) | all three | flushed | the deviation |
/// | `fast::atan2` | both | never denormal | the deviation |
///
/// `dot` is Metal's fused chain of its products: the device's own `OpDot` where it reaches Metal's
/// `dot`, else the multiply and `OpFmaKHR`s the deviation also covers, or the routines, which read
/// as Metal does; where the profile lacks the deviation, the chain flushes its values between the
/// fused multiply-adds as well.
///
/// `fast::atan2(y, x)` reads a denormal `y` or `x` as the zero of its sign and answers no denormal
/// (an Apple M3 Max under macOS 27.0.1: every pair of 28 patterns, in entries with and without fast
/// math); its quotient is the device's multiply or division, whose operands are those two, and a
/// denormal quotient reaches the fast `atan`, which reads it as a zero. Where native device maths
/// answers it, its denormals are the device's (the caller flushes nothing there).
///
/// `trunc` and `rint` (`Trunc`, `RoundEven`) of a denormal answer the zero of its sign on every
/// venue measured, as Metal does (the probes' words, both modes, all seven venues), but Vulkan
/// states no sign for a flushed denormal, so where the profile lacks the deviation the module
/// flushes the operand itself rather than trust the device's zero.
pub fn coverage(op: FloatFn) -> Option<Reading> {
    use FloatFn as F;
    let (covered, operands, result) = match op {
        F::Floor | F::Ceil => (Covered::Never, 0b1, false),
        F::Trunc | F::Rint => (Covered::Deviation, 0b1, false),
        F::Sqrt | F::Rsqrt | F::FastRsqrt => (Covered::Deviation, 0b1, false),
        F::Dot => (Covered::Deviation, 0b11, true),
        F::Fma | F::MulAdd => (Covered::Deviation, 0b111, true),
        F::FastAtan2 { .. } => (Covered::Deviation, 0b11, false),
        _ => return None,
    };
    Some(Reading {
        covered,
        operands,
        result,
    })
}

/// The `f32` denormals an operation flushes itself where the device may keep them
/// (`ArithStrategy::Denormals`): the operands whose bit is set in `operands`, read by [`flush`]
/// before the operation, and its result where `result`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Flush {
    /// A bit per operand read as Metal reads it.
    pub operands: u8,
    /// Whether a denormal result is flushed.
    pub result: bool,
}

impl Flush {
    /// Nothing flushed.
    pub const NONE: Flush = Flush {
        operands: 0,
        result: false,
    };

    /// Whether operand `i` is flushed.
    pub fn operand(self, i: usize) -> bool {
        i < 8 && self.operands & (1 << i) != 0
    }

    /// Whether anything is flushed.
    pub fn any(self) -> bool {
        self.operands != 0 || self.result
    }

    /// An upper estimate of the SPIR-V words of the flushes on lanes of form `f`: [`flush_words`]
    /// each.
    pub fn words(self, f: Form) -> u32 {
        (self.operands.count_ones() + u32::from(self.result)) * flush_words(f)
    }
}

/// Whether profile `p` computes Metal's `float` product where the exact product lies below
/// `2^-126` and the device's rounding reaches `±2^-126` ([`tiny_product`]): where its functions
/// are to answer as Metal's (`FloatFunctions::Device`, the policy's `FloatMathsPolicy::Exact`).
/// Metal flushes such a product to the zero of its sign, deciding tininess before rounding; every
/// Vulkan device measured off MoltenVK rounds first, those that flush their denormals and those
/// that keep them (the authored tininess cases' rows on RADV, ANV, NVIDIA and lavapipe), and
/// Vulkan states no order. On MoltenVK the multiply is Metal's own (`FloatFunctions::Metal`);
/// under native device maths (`FloatFunctions::Native`, the default) the device's product is
/// taken, which the runtime announces once per device (`float-tininess-after-rounding`).
pub fn checks_tiny_products(p: &DeviceProfile) -> bool {
    p.fields().float_functions == FloatFunctions::Device
}

/// How a check of a `float` product at `±2^-126` decides whether the exact product lies below
/// `2^-126` ([`tiny_product`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TinyCheck {
    /// By one `OpFmaKHR` on the operands' magnitudes, one of them scaled: where the profile's
    /// `float` fused multiply-add is `OpFmaKHR` (`fma::Route::Khr`).
    Fused,
    /// By the operands' bits, in integer operations of Vulkan 1.2's core alone.
    Bits,
}

/// The check of a `float` product at `±2^-126` profile `p` takes, where it checks one
/// ([`checks_tiny_products`]).
pub fn tiny_check(p: &DeviceProfile) -> Option<TinyCheck> {
    if !checks_tiny_products(p) {
        return None;
    }
    Some(match fma::float_route(p) {
        fma::Route::Khr => TinyCheck::Fused,
        _ => TinyCheck::Bits,
    })
}

/// Metal's product of `float` lanes `a` and `b` (form `f`, each read as Metal reads a denormal)
/// from `r`, the device's product of them: the zero of `r`'s sign where `r` is `±2^-126` and the
/// exact product lies below `2^-126` in magnitude, else `r`, decided by `check`. Inline and
/// branchless: the device's compiler inlines any call, so a shared routine would not shrink what
/// it compiles at each product.
///
/// Only a product whose exact value lies within half a unit of `2^-126` rounds to `±2^-126`, and
/// a zero, denormal, infinite or NaN operand gives no such `r` (a denormal operand is read as a
/// zero before the multiply, by the device or [`flush`]). So where `r` is `±2^-126` both operands
/// are normal, at most `1 + 2^-23` in magnitude (the other is at least `2^-126`), and with biased
/// exponent fields `ea` and `eb` and significands `ma` and `mb` (hidden bit set) the exact product
/// is `ma · mb · 2^(ea + eb - 300)`, with `ea + eb` 126, 127 or 128.
///
/// - [`TinyCheck::Fused`]: `OpFmaKHR(|a| · 2^48, |b|, -2^-78)` is `2^48 · (|a · b| - 2^-126)`
///   rounded once (Vulkan's precision table: correctly rounded), negative exactly where the exact
///   product lies below `2^-126`. `|a| · 2^48` is exact and finite, an `OpFMul` decorated
///   `NoContraction` (rounded on its own, as it is exact); the difference is a multiple
///   of `2^-174` (`ea + eb` is at least 126), so scaled it is zero or at least `2^-126`, a normal
///   number no denormal mode changes and rounding keeps the sign of. Elsewhere the lane is not
///   `±2^-126` and the test is not taken.
/// - [`TinyCheck::Bits`]: below `2^-126` where `ea + eb <= 126`, or where `ea + eb == 127` and
///   `ma · mb < 2^47`, which the high word of `OpUMulExtended` tells (`hi < 2^15`).
///
/// The test of `r` and its zero are on `r`'s bits, so the same in every float mode.
pub fn tiny_product(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    [a, b]: [Value; 2],
    r: Value,
    check: TinyCheck,
) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let rb = fb.convert(ConvOp::Bitcast, r, ut)?;
    // `r` is `±2^-126`.
    let magnitude = splat(fb, u, 0x7FFF_FFFF)?;
    let least = splat(fb, u, 0x0080_0000)?;
    let m = fb.binary(BinOp::BitwiseAnd, rb, magnitude)?;
    let at_least = fb.binary(BinOp::IEqual, m, least)?;
    let below = match check {
        TinyCheck::Fused => fused_below(fb, f, [a, b])?,
        TinyCheck::Bits => bits_below(fb, u, [a, b])?,
    };
    let tiny = fb.binary(BinOp::LogicalAnd, at_least, below)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let signed_zero = fb.binary(BinOp::BitwiseAnd, rb, sign)?;
    let out = fb.select(tiny, signed_zero, rb)?;
    fb.convert(ConvOp::Bitcast, out, ft)
}

/// Whether `|a · b|` lies below `2^-126` ([`TinyCheck::Fused`]), for normal `a` and `b` at most
/// `1 + 2^-23` in magnitude whose product rounds to `±2^-126`.
fn fused_below(fb: &mut FunctionBuilder<'_>, f: Form, [a, b]: [Value; 2]) -> BResult<Value> {
    let x = fb.glsl(GlslOp::FAbs, &[a])?;
    let y = fb.glsl(GlslOp::FAbs, &[b])?;
    let scale = fsplat(fb, f, 2f64.powi(48))?;
    // Exact, and `NoContraction` so that no device folds it into another operation.
    let x = fb.binary_with(BinOp::FMul, x, scale, Contraction::Forbidden)?;
    let edge = fsplat(fb, f, -(2f64.powi(-78)))?;
    let d = fb.fma(x, y, edge)?;
    let zero = fsplat(fb, f, 0.0)?;
    fb.binary(BinOp::FOrdLessThan, d, zero)
}

/// Whether the exact product of `a` and `b` lies below `2^-126` ([`TinyCheck::Bits`]), on `u32`
/// lanes of form `u`, for operands whose product rounds to `±2^-126`.
fn bits_below(fb: &mut FunctionBuilder<'_>, u: Form, [a, b]: [Value; 2]) -> BResult<Value> {
    let ut = ty(fb, u)?;
    let ab = fb.convert(ConvOp::Bitcast, a, ut)?;
    let bb = fb.convert(ConvOp::Bitcast, b, ut)?;
    // The exponent fields' sum, in place: at most 2 · 254 · 2^23, below 2^32.
    let field = splat(fb, u, 0x7F80_0000)?;
    let ea = fb.binary(BinOp::BitwiseAnd, ab, field)?;
    let eb = fb.binary(BinOp::BitwiseAnd, bb, field)?;
    let sum = fb.binary(BinOp::IAdd, ea, eb)?;
    let below = splat(fb, u, 126 << 23)?;
    let edge = splat(fb, u, 127 << 23)?;
    let low = fb.binary(BinOp::ULessThanEqual, sum, below)?;
    let at_edge = fb.binary(BinOp::IEqual, sum, edge)?;
    // The significands' product below `2^47`: its high word below `2^15`.
    let fraction = splat(fb, u, 0x007F_FFFF)?;
    let hidden = splat(fb, u, 0x0080_0000)?;
    let mut significand = |x: Value| -> BResult<Value> {
        let m = fb.binary(BinOp::BitwiseAnd, x, fraction)?;
        fb.binary(BinOp::BitwiseOr, m, hidden)
    };
    let (ma, mb) = (significand(ab)?, significand(bb)?);
    let product = fb.binary(BinOp::UMulExtended, ma, mb)?;
    let hi = fb.composite_extract(product, &[1])?;
    let limit = splat(fb, u, 0x8000)?;
    let under = fb.binary(BinOp::ULessThan, hi, limit)?;
    let edge_tiny = fb.binary(BinOp::LogicalAnd, at_edge, under)?;
    fb.binary(BinOp::LogicalOr, low, edge_tiny)
}

/// An upper estimate of the SPIR-V words of one [`tiny_product`] on form `f` by `check`: the test
/// of `r` and the zero, seven instructions; [`TinyCheck::Fused`]'s five more, three constants and
/// the scaling's decoration (three words), [`TinyCheck::Bits`]'s 16 more, six constants and the
/// extended product's struct type; each instruction of at most six words, and the constants' share
/// (a vector's constants carry their lanes).
pub fn tiny_product_words(f: Form, check: TinyCheck) -> u32 {
    let lanes = u32::from(f.lanes);
    match check {
        TinyCheck::Fused => 93 + 7 * (7 + lanes),
        TinyCheck::Bits => 160 + 10 * (7 + lanes),
    }
}

/// An upper estimate of the SPIR-V words of one [`flush`] on form `f`: six instructions of at most
/// six words, and the three masks' share (a vector's constants carry their lanes).
pub fn flush_words(f: Form) -> u32 {
    36 + 4 * u32::from(f.lanes)
}

/// `x` (`float` lanes of form `f`) with each denormal lane replaced by the zero of its sign,
/// decided on the exponent field, so the same on every device and in every float mode.
pub fn flush(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let u = u32_form(f);
    let (ut, ft) = (ty(fb, u)?, ty(fb, f)?);
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let exponent = splat(fb, u, 0x7F80_0000)?;
    let sign = splat(fb, u, 0x8000_0000)?;
    let zero = splat(fb, u, 0)?;
    let e = fb.binary(BinOp::BitwiseAnd, bits, exponent)?;
    let denormal = fb.binary(BinOp::IEqual, e, zero)?;
    let signed_zero = fb.binary(BinOp::BitwiseAnd, bits, sign)?;
    let r = fb.select(denormal, signed_zero, bits)?;
    fb.convert(ConvOp::Bitcast, r, ft)
}
