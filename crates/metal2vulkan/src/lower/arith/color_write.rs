//! A fragment function's float color output rounded to the code Metal stores.
//!
//! Metal converts a float written to a normalized attachment by the exact product with the
//! format's maximum code, rounded to nearest; to an 8-bit sRGB channel by its twelve-bit encode;
//! and to a half-float channel, or one of `RG11B10Float`'s, rounded toward zero
//! (`LoweringInputs::color_writes`). A Vulkan driver rounds the `f32` product, encodes sRGB by its
//! own approximation, and rounds a half to nearest, so near each tie it stores another code.
//! [`round`] computes Metal's code in the shader and writes the value the device stores as exactly
//! that code:
//!
//! - unorm and snorm ([`ChannelWrite::Unorm`], [`ChannelWrite::Snorm`]): NaN is 0 (a blended
//!   snorm NaN is `-1`, whose code is `-m`, as Metal stores it), the value is clamped to
//!   `[0, 1]` (`[-1, 1]`), the code `c` is the exact product with `m = 2^k - 1` rounded to
//!   nearest (`round_mersenne_product`, about fourteen float instructions; its only ties, at `x =
//!   ±0.5`, round to even as Metal's round away from zero), and the value written is `c` times the
//!   `f32` nearest `1 / m`: within two `f32` rounding errors of `c / m`, so its product with `m`
//!   lies within `2^-7` of `c` (`m < 2^16`) and a device that rounds the product to nearest stores
//!   `c`, as Vulkan's conversion does, whatever it does at ties;
//! - sRGB ([`ChannelWrite::Srgb`]): NaN is 0, the value is clamped to `[0, 1]`, the byte is
//!   Metal's twelve-bit encode of the exact product with `4095 = 2^12 - 1`, rounded as a unorm
//!   channel's, and the value written is that byte's linear value: an estimate and one read of
//!   the color table, a buffer of the module's ([`srgb::color_linear`]);
//! - a small float ([`ChannelWrite::FloatTowardZero`]: a half, or an unsigned 11- or 10-bit
//!   float): the value rounded toward zero, by integer operations on the bits, to the `f32` that
//!   holds it exactly: a finite value past the largest is the largest, an infinity stays, a NaN
//!   is the quiet NaN (written as the `f32` NaN `0x7FC00000`), and where unsigned a negative
//!   value is 0;
//! - a small float rounded to nearest even ([`ChannelWrite::FloatNearestEven`], a render
//!   pipeline's half-float channel under Apple's `textureWriteRoundingMode` 2): the same, but the
//!   dropped bits rounded to nearest with ties to even, by integer operations on the bits, and a
//!   finite value from the midpoint past the largest infinity;
//! - a packed 16-bit format's unorm channel ([`ChannelWrite::HalfUnorm`]): the value rounded to a
//!   half to nearest even as above, then as a unorm channel's.
//!
//! A channel without a rule ([`ChannelWrite::Device`] or `None`) is written as computed. A
//! channel the device is measured to convert as Metal does but for a NaN ([`Lane::NanAs`]) is
//! written as computed with a NaN chosen as the value whose store is Metal's.
//!
//! A channel whose rule the draw selects ([`ChannelWrite::Selected`], [`Lane::Selected`]) computes
//! each selectable rule's value and selects one by its code in the attachment's rule word
//! (`pipeline::color_writes::rule_words`): the value as computed for the device's conversion or no
//! rule, the 8-bit unorm channel's rounding, or the half-float channel's.

use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::ops::{fsplat, splat, ty};
use crate::lower::arith::srgb;
use crate::pipeline::color_writes::{HALF, HALF_CODE, RULE_BITS, RULE_MASK, UNORM8_CODE};
use crate::pipeline::lower_inputs::ChannelWrite;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, OperandRule};
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, Type, UnOp};

/// How [`write`] writes one lane of a color output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// Rounded to the code Metal stores by the channel's rule ([`round`]).
    Rounded(ChannelWrite),
    /// Written as computed, a NaN as the `f32` of these bits: the device converts every other
    /// value as Metal does (a measured driver deviation), and stores this NaN, or this value, as
    /// Metal stores the output's NaN.
    NanAs(u32),
    /// Written by the rule its code in the attachment's rule word selects
    /// ([`ChannelWrite::Selected`]): as computed (no rule, or the device's conversion), rounded
    /// as an 8-bit unorm channel, or written as a half-float channel's lane is (`half`).
    Selected {
        /// How a half-float channel's lane is written from this output.
        half: HalfLane,
    },
}

/// How a selected lane ([`Lane::Selected`]) is written where its code selects the half-float
/// channel's rule: as the lane of [`ChannelWrite::FloatTowardZero`]'s half would be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfLane {
    /// As computed: a `half` output, already a half, or a `float` one the device converts as
    /// Metal does, NaN included.
    AsComputed,
    /// Rounded toward zero to a half ([`small_float`]).
    Rounded,
    /// As computed, a NaN as the `f32` of these bits ([`Lane::NanAs`]).
    NanAs(u32),
}

/// An upper estimate of the SPIR-V words of one lane written as `lane`.
pub fn lane_words(lane: Lane) -> u64 {
    match lane {
        Lane::Rounded(rule) => channel_words(rule),
        // The NaN test and select, and the constant.
        Lane::NanAs(_) => 3 * 6,
        // The unorm rounding and the half lane, and each code's shift, mask, test and select.
        Lane::Selected { half } => {
            let half = match half {
                HalfLane::AsComputed => 0,
                HalfLane::Rounded => channel_words(HALF),
                HalfLane::NanAs(_) => 3 * 6,
            };
            channel_words(ChannelWrite::Unorm { bits: 8 }) + half + 2 * 5 * 6
        }
    }
}

/// An upper estimate of the SPIR-V words of one channel's rounding by `rule`.
pub fn channel_words(rule: ChannelWrite) -> u64 {
    match rule {
        // The NaN select, clamp, exact product, scale and sign, their constants included: about
        // 31 instructions and the product's nine `NoContraction` decorations.
        ChannelWrite::Unorm { .. } | ChannelWrite::Snorm { .. } => 36 * 6,
        // The NaN select, clamp, exact product and conversion (about 25 instructions and nine
        // decorations), and the byte's estimate, entry read and choice.
        ChannelWrite::Srgb => 32 * 6 + u64::from(srgb::COLOR_WORDS),
        ChannelWrite::FloatTowardZero { .. } => 32 * 6,
        // The rounding's bias and the denormal's shift clamped beside the toward-zero rule's.
        ChannelWrite::FloatNearestEven { .. } => 36 * 6,
        // The half's rounding, then the unorm channel's.
        ChannelWrite::HalfUnorm { .. } => 2 * 36 * 6,
        // A selected channel's words are its lane's (`Lane::Selected`).
        ChannelWrite::Device | ChannelWrite::BlendedNearestEven { .. } | ChannelWrite::Selected => {
            0
        }
    }
}

/// An upper estimate of the SPIR-V words of rounding an output by `rules`, its extracts and
/// construct included.
pub fn words(rules: &[Option<ChannelWrite>; 4]) -> u64 {
    write_words(&rules.map(|r| r.map(Lane::Rounded)))
}

/// An upper estimate of the SPIR-V words of writing an output by `lanes` ([`write`]), its
/// extracts and construct included.
pub fn write_words(lanes: &[Option<Lane>; 4]) -> u64 {
    let written = lanes.iter().flatten().count() as u64;
    lanes.iter().flatten().map(|&l| lane_words(l)).sum::<u64>()
        + written.saturating_add(1).saturating_mul(6)
}

/// An upper estimate of the SPIR-V words of [`half_bits`]: about 20 instructions on the whole
/// vector, its constants, 16-bit types and their capabilities included (192 words measured on four
/// lanes, two widening conversions with them).
pub const HALF_BITS_WORDS: u64 = 34 * 6;

/// The `half` lanes whose bits hold `x` (float lanes of form `f`, a 32-bit container), each lane
/// a value a half holds exactly (a half widened, or a value [`round`] rounded to one), on a
/// profile whose conversion to 16 bits, of an operand its compiler cannot fold into the
/// instruction computing it, is Metal's `half(float)` (`DriverDeviation::MetalHalfConversion`,
/// `narrow::QuantizeForm::Converted`): that conversion of `x + -0.0` (an `OpFAdd` decorated
/// `NoContraction`, as `narrow` converts), exact for every value but a NaN, whose half the
/// conversion makes `0x7e00`; a NaN lane is [`nan_codes`]'s code instead, chosen as 16-bit
/// integers, and the lanes bitcast to `half` once (`Int16` and `Float16`), so no conversion on
/// the device chooses a NaN's encoding.
pub fn half_bits(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let lanes = |fb: &mut FunctionBuilder<'_>, elem: Type| -> BResult<TypeId> {
        let elem = fb.module().ty(elem)?;
        match f.lanes {
            1 => Ok(elem),
            len => fb.module().ty(Type::Vector { elem, len }),
        }
    };
    let u16_ = lanes(
        fb,
        Type::Int {
            width: 16,
            signed: false,
        },
    )?;
    let h = lanes(fb, Type::Float { width: 16 })?;
    let minus_zero = fsplat(fb, f.with(Held::Float(FloatFmt::F32)), -0.0)?;
    let held = fb.binary_no_contraction(BinOp::FAdd, x, minus_zero)?;
    let converted = fb.convert(ConvOp::FConvert, held, h)?;
    let converted = fb.convert(ConvOp::Bitcast, converted, u16_)?;
    let nan = nan_codes(fb, f, x)?;
    let nan = fb.convert(ConvOp::UConvert, nan, u16_)?;
    let is_nan = fb.unary(UnOp::IsNan, x)?;
    let code = fb.select(is_nan, nan, converted)?;
    fb.convert(ConvOp::Bitcast, code, h)
}

/// The 16-bit `half` NaN codes (in `u32` lanes) of `x` (float lanes of form `f`, a 32-bit
/// container) where a lane is a NaN, by integer operations on the bits: the sign, every exponent
/// bit, and the top ten fraction bits, a widened half NaN's own payload, `0x200` where those are
/// zero (a NaN no half widens to). A lane that is not a NaN gives a code [`half_bits`] does not
/// choose.
pub fn nan_codes(fb: &mut FunctionBuilder<'_>, f: Form, x: Value) -> BResult<Value> {
    let u = f.with(Held::Int {
        bits: 32,
        container: 32,
    });
    let k = |fb: &mut FunctionBuilder<'_>, bits: u32| splat(fb, u, u64::from(bits));
    let ut = ty(fb, u)?;
    let bits = fb.convert(ConvOp::Bitcast, x, ut)?;
    let k16 = k(fb, 16)?;
    let sign = fb.binary(BinOp::ShiftRightLogical, bits, k16)?;
    let sign_bit = k(fb, 0x8000)?;
    let sign = fb.binary(BinOp::BitwiseAnd, sign, sign_bit)?;
    let k13 = k(fb, 13)?;
    let top = fb.binary(BinOp::ShiftRightLogical, bits, k13)?;
    let payload_mask = k(fb, 0x3ff)?;
    let top = fb.binary(BinOp::BitwiseAnd, top, payload_mask)?;
    let zero = k(fb, 0)?;
    let low = fb.binary(BinOp::IEqual, top, zero)?;
    let quiet = k(fb, 0x200)?;
    let payload = fb.select(low, quiet, top)?;
    let infinity = k(fb, 0x7c00)?;
    let special = fb.binary(BinOp::BitwiseOr, infinity, payload)?;
    fb.binary(BinOp::BitwiseOr, sign, special)
}

fn malformed() -> BuildError {
    BuildError::Operand {
        op: crate::spv::Op::ExtInst,
        rule: OperandRule::Arity,
    }
}

/// Output `x` (float lanes of form `f`, a 32-bit container) with each lane `i` rounded by
/// `rules[i]` (see the module docs), an [`Srgb`](ChannelWrite::Srgb) lane through the module's
/// color table `table` ([`srgb::declare_color_table`]).
pub fn round(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    rules: &[Option<ChannelWrite>; 4],
    table: Option<GlobalId>,
) -> BResult<Value> {
    write(
        fb,
        f,
        x,
        &rules.map(|r| r.map(Lane::Rounded)),
        (table, None),
    )
}

/// Output `x` (float lanes of form `f`, a 32-bit container) with each lane `i` written as
/// `lanes[i]` says (see the module docs), an [`Srgb`](ChannelWrite::Srgb) lane through the
/// module's color table `table` ([`srgb::declare_color_table`]), and a selected lane
/// ([`Lane::Selected`]) by its code in the attachment's rule word `word` (a `u32`), which
/// such a lane needs.
pub fn write(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    lanes: &[Option<Lane>; 4],
    (table, word): (Option<GlobalId>, Option<Value>),
) -> BResult<Value> {
    if f.float_fmt().map(FloatFmt::container) != Some(32) || f.lanes == 0 || f.lanes > 4 {
        return Err(malformed());
    }
    let s = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let mut written = Vec::with_capacity(usize::from(f.lanes));
    for (i, lane) in lanes.iter().enumerate().take(usize::from(f.lanes)) {
        let v = if f.lanes == 1 {
            x
        } else {
            fb.composite_extract(x, &[i as u32])?
        };
        let rule = match lane {
            None => None,
            Some(Lane::Rounded(rule)) => Some(rule),
            Some(Lane::NanAs(bits)) => {
                written.push(nan_as(fb, s, v, *bits)?);
                continue;
            }
            Some(Lane::Selected { half }) => {
                let word = word.ok_or_else(malformed)?;
                written.push(selected(fb, s, v, (word, i as u32), *half)?);
                continue;
            }
        };
        written.push(match rule {
            None | Some(ChannelWrite::Device | ChannelWrite::BlendedNearestEven { .. }) => v,
            Some(ChannelWrite::Unorm { bits }) => normalized(fb, s, v, unorm_bits(*bits)?, None)?,
            Some(ChannelWrite::Snorm { bits, blended }) => {
                normalized(fb, s, v, snorm_bits(*bits)?, Some(*blended))?
            }
            Some(ChannelWrite::Srgb) => srgb_channel(fb, s, v, table)?,
            Some(
                &rule @ (ChannelWrite::FloatTowardZero { .. }
                | ChannelWrite::FloatNearestEven { .. }),
            ) => small_float(fb, s, v, rule)?,
            Some(ChannelWrite::HalfUnorm { bits }) => {
                let h = float_nearest_even(fb, s, v, 10, true)?;
                normalized(fb, s, h, unorm_bits(*bits)?, None)?
            }
            // A selected channel is written by its lane's form (`Lane::Selected`).
            Some(ChannelWrite::Selected) => return Err(malformed()),
        });
    }
    if let [one] = written.as_slice() {
        return Ok(*one);
    }
    let t = ty(fb, f.with(Held::Float(FloatFmt::F32)))?;
    fb.composite_construct(t, &written)
}

/// `v` (an `f32` lane of form `s`) written as channel `i`'s code in the rule word `word` selects
/// ([`Lane::Selected`]): rounded as an 8-bit unorm channel (`UNORM8_CODE`), written as a
/// half-float channel's lane is by `half` (`HALF_CODE`), else as computed. Each rule's value is
/// computed and one selected, so the module is the same for every attachment.
fn selected(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    (word, i): (Value, u32),
    half: HalfLane,
) -> BResult<Value> {
    let u = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    };
    let shift = splat(fb, u, u64::from(RULE_BITS * i))?;
    let code = fb.binary(BinOp::ShiftRightLogical, word, shift)?;
    let mask = splat(fb, u, u64::from(RULE_MASK))?;
    let code = fb.binary(BinOp::BitwiseAnd, code, mask)?;
    let unorm = normalized(fb, s, v, unorm_bits(8)?, None)?;
    let unorm_code = splat(fb, u, u64::from(UNORM8_CODE))?;
    let is_unorm = fb.binary(BinOp::IEqual, code, unorm_code)?;
    let out = fb.select(is_unorm, unorm, v)?;
    let half = match half {
        HalfLane::AsComputed => return Ok(out),
        HalfLane::Rounded => small_float(fb, s, v, HALF)?,
        HalfLane::NanAs(bits) => nan_as(fb, s, v, bits)?,
    };
    let half_code = splat(fb, u, u64::from(HALF_CODE))?;
    let is_half = fb.binary(BinOp::IEqual, code, half_code)?;
    fb.select(is_half, half, out)
}

/// `v` (an `f32` lane of form `s`), a NaN written as the `f32` of `bits` ([`Lane::NanAs`]).
fn nan_as(fb: &mut FunctionBuilder<'_>, s: Form, v: Value, bits: u32) -> BResult<Value> {
    let is_nan = fb.unary(UnOp::IsNan, v)?;
    let nan = splat(fb, s, u64::from(bits))?;
    fb.select(is_nan, nan, v)
}

/// The `k` of a unorm channel's maximum code `2^k - 1`.
fn unorm_bits(bits: u8) -> BResult<u32> {
    match bits {
        1..=16 => Ok(u32::from(bits)),
        _ => Err(malformed()),
    }
}

/// The `k` of a snorm channel's maximum code `2^k - 1`.
fn snorm_bits(bits: u8) -> BResult<u32> {
    match bits {
        2..=16 => Ok(u32::from(bits) - 1),
        _ => Err(malformed()),
    }
}

/// `a` (`f32` lanes of form `s`, each in `[0, 1]`) times `2^k - 1` (`k` in `1..=16`), rounded to
/// the nearest integer from the exact product, ties to even, as `f32` lanes holding the integer.
/// The product's exact split is `p - w`: `p = fl(2^k a - a)` with `2^k a` exact, and `w = a + (p -
/// 2^k a)` (Fast2Sum: both steps are exact). `p` rounds to nearest even as `r`, `d = p - r` is
/// exact in `[-1/2, 1/2]`, and `r` moves by one where the product's distance `d - w` from `r` is
/// past a half: up where `w < d - 1/2`, down where `w > d + 1/2`, each threshold exact wherever the
/// comparison can go either way (`|w|` is below `2^(k-23)`). The only tie (`a = 1/2`, `w = 0` and
/// `d = ±1/2`) stays at `RoundEven`'s even integer: `2^(k-1)` for `k >= 2`, which is Metal's
/// rounding away from zero there.
///
/// Every step is decorated `NoContraction`, so no device fuses or reassociates the split, and
/// none relies on the rounding mode: an exhaustive check of every `f32` in `[0, 1]` for every
/// `k` matched the exact product rounded to nearest even under rounding to nearest and toward zero,
/// with and without denormals flushed. About fourteen instructions, where
/// [`round_exact_product_lanes`](crate::lower::arith::math::round_exact_product_lanes) takes about
/// forty and a conversion back.
fn round_mersenne_product(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    a: Value,
    k: u32,
) -> BResult<Value> {
    if !(1..=16).contains(&k) {
        return Err(malformed());
    }
    let scale = fsplat(fb, s, f64::from(1u32 << k))?;
    round_scaled_product(fb, s, a, scale)
}

/// [`round_mersenne_product`] by a scale `scale` (`f32` lanes of form `s`, each `2^k` for a `k`
/// in `1..=16`, a value the shader may compute): `a` times `scale - 1`, rounded to nearest from
/// the exact product, ties to even, by the same instructions, so each lane is exact as the
/// constant scale's is (the split's steps do not depend on `k`'s being constant).
pub(crate) fn round_scaled_product(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    a: Value,
    scale: Value,
) -> BResult<Value> {
    let big = fb.binary_no_contraction(BinOp::FMul, a, scale)?;
    let p = fb.binary_no_contraction(BinOp::FSub, big, a)?;
    let z = fb.binary_no_contraction(BinOp::FSub, p, big)?;
    // The exact product is `p - w`.
    let w = fb.binary_no_contraction(BinOp::FAdd, a, z)?;
    let r = fb.glsl(GlslOp::RoundEven, &[p])?;
    let d = fb.binary_no_contraction(BinOp::FSub, p, r)?;
    let half = fsplat(fb, s, 0.5)?;
    let below = fb.binary_no_contraction(BinOp::FSub, d, half)?;
    let above = fb.binary_no_contraction(BinOp::FAdd, d, half)?;
    let up = fb.binary(BinOp::FOrdLessThan, w, below)?;
    let down = fb.binary(BinOp::FOrdGreaterThan, w, above)?;
    let one = fsplat(fb, s, 1.0)?;
    let r_up = fb.binary_no_contraction(BinOp::FAdd, r, one)?;
    let r_down = fb.binary_no_contraction(BinOp::FSub, r, one)?;
    let c = fb.select(up, r_up, r)?;
    fb.select(down, r_down, c)
}

/// `v` clamped to `[lo, 1]` with NaN taken as `nan`.
fn clamped(fb: &mut FunctionBuilder<'_>, s: Form, v: Value, nan: f64, lo: f64) -> BResult<Value> {
    let is_nan = fb.unary(UnOp::IsNan, v)?;
    let nan = fsplat(fb, s, nan)?;
    let v = fb.select(is_nan, nan, v)?;
    let lo = fsplat(fb, s, lo)?;
    let one = fsplat(fb, s, 1.0)?;
    fb.glsl(GlslOp::NClamp, &[v, lo, one])
}

/// A unorm channel of maximum code `2^k - 1` (`snorm: None`), or a snorm one
/// (`Some(blended)`).
fn normalized(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    k: u32,
    snorm: Option<bool>,
) -> BResult<Value> {
    let (nan, lo) = match snorm {
        None => (0.0, 0.0),
        Some(false) => (0.0, -1.0),
        Some(true) => (-1.0, -1.0),
    };
    let v = clamped(fb, s, v, nan, lo)?;
    // A unorm channel's clamped value is its magnitude.
    let a = match snorm {
        None => v,
        Some(_) => fb.glsl(GlslOp::FAbs, &[v])?,
    };
    let c = round_mersenne_product(fb, s, a, k)?;
    let m = (1u32 << k) - 1;
    let r = fsplat(fb, s, f64::from(1.0 / m as f32))?;
    let q = fb.binary(BinOp::FMul, c, r)?;
    if snorm.is_none() {
        return Ok(q);
    }
    let zero = fsplat(fb, s, 0.0)?;
    let negative = fb.binary(BinOp::FOrdLessThan, v, zero)?;
    let minus = fb.unary(UnOp::FNegate, q)?;
    fb.select(negative, minus, q)
}

/// An sRGB color channel: the linear value of the byte Metal's encode gives, from the color
/// table `table`.
fn srgb_channel(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    table: Option<GlobalId>,
) -> BResult<Value> {
    let Some(table) = table else {
        return Err(malformed());
    };
    let v = clamped(fb, s, v, 0.0, 0.0)?;
    let i = round_mersenne_product(fb, s, v, 12)?;
    let ut = ty(
        fb,
        s.with(Held::Int {
            bits: 32,
            container: 32,
        }),
    )?;
    let i = fb.convert(ConvOp::ConvertFToU, i, ut)?;
    srgb::color_linear(fb, table, i)
}

/// An upper estimate of the SPIR-V words of [`small_float`] by `rule` on `lanes` lanes: one
/// channel's, and each of its constants (at most 18) as a vector of `lanes`.
pub fn small_float_words(rule: ChannelWrite, lanes: u8) -> u64 {
    let composites = if lanes > 1 {
        18 * (3 + u64::from(lanes))
    } else {
        0
    };
    channel_words(rule) + composites + 16
}

/// `x` (`f32` lanes of form `f`) each rounded by the small-float rule `rule`
/// ([`ChannelWrite::FloatTowardZero`] or [`ChannelWrite::FloatNearestEven`]), every lane by the
/// same instructions on the whole vector; another rule is malformed.
pub fn small_float(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    x: Value,
    rule: ChannelWrite,
) -> BResult<Value> {
    if f.float_fmt() != Some(FloatFmt::F32) {
        return Err(malformed());
    }
    match rule {
        ChannelWrite::FloatTowardZero { mantissa, signed } => {
            float_toward_zero(fb, f, x, mantissa, signed)
        }
        ChannelWrite::FloatNearestEven { mantissa, signed } => {
            float_nearest_even(fb, f, x, mantissa, signed)
        }
        _ => Err(malformed()),
    }
}

/// `v` (`f32` lanes of form `s`) rounded to the nearest `half`, ties to even, held in an `f32`: a
/// NaN stays a NaN and a value past the largest finite `half` an infinity (a packed 16-bit
/// format's channel reads it so, `ChannelWrite::HalfUnorm`).
pub(crate) fn half_rounded(fb: &mut FunctionBuilder<'_>, s: Form, v: Value) -> BResult<Value> {
    float_nearest_even(fb, s, v, 10, true)
}

/// `v` (`f32` lanes of form `s`) rounded to nearest even to a float of a five-bit exponent (bias
/// 15) and `mantissa` fraction bits, signed or not, held in an `f32` (see the module docs): `x`'s
/// bits plus `2^(k-1) - 1` and the lowest kept bit, shifted right by the `k` bits dropped, is the
/// kept value rounded to nearest with ties to even.
fn float_nearest_even(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    mantissa: u8,
    signed: bool,
) -> BResult<Value> {
    let m = match mantissa {
        1..=10 => u32::from(mantissa),
        _ => return Err(malformed()),
    };
    let u = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: s.lanes,
    };
    let f = u.with(Held::Float(FloatFmt::F32));
    let k = |fb: &mut FunctionBuilder<'_>, bits: u32| splat(fb, u, u64::from(bits));
    let ut = ty(fb, u)?;
    let ft = ty(fb, f)?;
    let one = k(fb, 1)?;
    // `x` rounded to nearest even at bit `shift`: plus `2^(shift-1) - 1` and the lowest kept bit.
    let round = |fb: &mut FunctionBuilder<'_>, x: Value, shift: Value| -> BResult<Value> {
        let kept = fb.binary(BinOp::ShiftRightLogical, x, shift)?;
        let lsb = fb.binary(BinOp::BitwiseAnd, kept, one)?;
        let below = fb.binary(BinOp::ISub, shift, one)?;
        let half = fb.binary(BinOp::ShiftLeftLogical, one, below)?;
        let bias = fb.binary(BinOp::ISub, half, one)?;
        let x = fb.binary(BinOp::IAdd, x, bias)?;
        let x = fb.binary(BinOp::IAdd, x, lsb)?;
        fb.binary(BinOp::ShiftRightLogical, x, shift)
    };
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let sign_bit = k(fb, 0x8000_0000)?;
    let sign = fb.binary(BinOp::BitwiseAnd, bits, sign_bit)?;
    let abs_mask = k(fb, 0x7FFF_FFFF)?;
    let a = fb.binary(BinOp::BitwiseAnd, bits, abs_mask)?;
    // A normal value: its bits rounded at the format's last fraction bit, a carry into the
    // exponent included.
    let dropped = k(fb, 23 - m)?;
    let normal = round(fb, a, dropped)?;
    let normal = fb.binary(BinOp::ShiftLeftLogical, normal, dropped)?;
    // A denormal one: its multiples of `2^-(14 + m)`, `mant * 2^(e - 150)` for the biased
    // exponent `e < 113` (an `f32` denormal, `e = 0`, is below 2^-126 and rounds to 0), so
    // `mant` rounded at bit `136 - m - e`, that shift held in [1, 31] (a value of 31 rounds to
    // 0 there, and a larger exponent's lane is the normal one's).
    let k23 = k(fb, 23)?;
    let e = fb.binary(BinOp::ShiftRightLogical, a, k23)?;
    let man_mask = k(fb, 0x7F_FFFF)?;
    let man = fb.binary(BinOp::BitwiseAnd, a, man_mask)?;
    let lead = k(fb, 0x80_0000)?;
    let man = fb.binary(BinOp::BitwiseOr, man, lead)?;
    let base = k(fb, 136 - m)?;
    let shift = fb.binary(BinOp::ISub, base, e)?;
    let k31 = k(fb, 31)?;
    let shift = fb.glsl(GlslOp::UClamp, &[shift, one, k31])?;
    let units = round(fb, man, shift)?;
    let units = fb.convert(ConvOp::ConvertUToF, units, ft)?;
    let step = fsplat(fb, f, 2f64.powi(-14 - i32::from(mantissa)))?;
    let denormal = fb.binary(BinOp::FMul, units, step)?;
    let denormal = fb.convert(ConvOp::Bitcast, denormal, ut)?;
    let smallest_normal = k(fb, 0x3880_0000)?;
    let is_normal = fb.binary(BinOp::UGreaterThanEqual, a, smallest_normal)?;
    let mag = fb.select(is_normal, normal, denormal)?;
    // From 2^16, past the largest finite value rounded up (an infinity included): infinity.
    let past = k(fb, 0x4780_0000)?;
    let infinity = k(fb, 0x7F80_0000)?;
    let is_past = fb.binary(BinOp::UGreaterThanEqual, mag, past)?;
    let mag = fb.select(is_past, infinity, mag)?;
    let out = if signed {
        fb.binary(BinOp::BitwiseOr, sign, mag)?
    } else {
        let zero = k(fb, 0)?;
        let negative = fb.binary(BinOp::INotEqual, sign, zero)?;
        fb.select(negative, zero, mag)?
    };
    let is_nan = fb.binary(BinOp::UGreaterThan, a, infinity)?;
    let quiet = k(fb, 0x7FC0_0000)?;
    let out = fb.select(is_nan, quiet, out)?;
    fb.convert(ConvOp::Bitcast, out, ft)
}

/// `v` (`f32` lanes of form `s`) rounded toward zero to a float of a five-bit exponent (bias 15)
/// and `mantissa` fraction bits, signed or not, held in an `f32` (see the module docs).
fn float_toward_zero(
    fb: &mut FunctionBuilder<'_>,
    s: Form,
    v: Value,
    mantissa: u8,
    signed: bool,
) -> BResult<Value> {
    let m = match mantissa {
        1..=10 => u32::from(mantissa),
        _ => return Err(malformed()),
    };
    let u = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: s.lanes,
    };
    let f = u.with(Held::Float(FloatFmt::F32));
    let k = |fb: &mut FunctionBuilder<'_>, bits: u32| splat(fb, u, u64::from(bits));
    let ut = ty(fb, u)?;
    let ft = ty(fb, f)?;
    let bits = fb.convert(ConvOp::Bitcast, v, ut)?;
    let sign_bit = k(fb, 0x8000_0000)?;
    let sign = fb.binary(BinOp::BitwiseAnd, bits, sign_bit)?;
    let abs_mask = k(fb, 0x7FFF_FFFF)?;
    let a = fb.binary(BinOp::BitwiseAnd, bits, abs_mask)?;
    // A normal value: the significand bits below the format's `m` dropped.
    let keep = k(fb, u32::MAX << (23 - m))?;
    let normal = fb.binary(BinOp::BitwiseAnd, a, keep)?;
    // A denormal one: the whole multiples of `2^-(14 + m)` below the value. The value is `mant *
    // 2^(e - 150)` for the biased exponent `e < 113` (an `f32` denormal, `e = 0`, is below
    // 2^-126 and has none), so `mant >> (136 - m - e)` of them.
    let k23 = k(fb, 23)?;
    let e = fb.binary(BinOp::ShiftRightLogical, a, k23)?;
    let man_mask = k(fb, 0x7F_FFFF)?;
    let man = fb.binary(BinOp::BitwiseAnd, a, man_mask)?;
    let lead = k(fb, 0x80_0000)?;
    let man = fb.binary(BinOp::BitwiseOr, man, lead)?;
    let base = k(fb, 136 - m)?;
    let shift = fb.binary(BinOp::ISub, base, e)?;
    let k31 = k(fb, 31)?;
    let shift = fb.glsl(GlslOp::UMin, &[shift, k31])?;
    let units = fb.binary(BinOp::ShiftRightLogical, man, shift)?;
    let units = fb.convert(ConvOp::ConvertUToF, units, ft)?;
    let step = fsplat(fb, f, 2f64.powi(-14 - i32::from(mantissa)))?;
    let denormal = fb.binary(BinOp::FMul, units, step)?;
    let denormal = fb.convert(ConvOp::Bitcast, denormal, ut)?;
    let smallest_normal = k(fb, 0x3880_0000)?;
    let is_normal = fb.binary(BinOp::UGreaterThanEqual, a, smallest_normal)?;
    let mag = fb.select(is_normal, normal, denormal)?;
    // Past the largest finite value (and infinity and NaN, chosen below): the largest.
    let largest = k(fb, 0x4700_0000 | (((1 << m) - 1) << (23 - m)))?;
    let past = fb.binary(BinOp::UGreaterThanEqual, a, largest)?;
    let mag = fb.select(past, largest, mag)?;
    let infinity = k(fb, 0x7F80_0000)?;
    let is_infinite = fb.binary(BinOp::IEqual, a, infinity)?;
    let mag = fb.select(is_infinite, infinity, mag)?;
    let out = if signed {
        fb.binary(BinOp::BitwiseOr, sign, mag)?
    } else {
        let zero = k(fb, 0)?;
        let negative = fb.binary(BinOp::INotEqual, sign, zero)?;
        fb.select(negative, zero, mag)?
    };
    let is_nan = fb.binary(BinOp::UGreaterThan, a, infinity)?;
    let quiet = k(fb, 0x7FC0_0000)?;
    let out = fb.select(is_nan, quiet, out)?;
    fb.convert(ConvOp::Bitcast, out, ft)
}
