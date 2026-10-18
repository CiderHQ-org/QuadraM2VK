//! Metal's level of a sample with an explicit level (`level(x)`), or a kernel's bias (`bias(x)`,
//! the explicit level `x` outside a fragment function), and the level that makes a device's sample
//! read it.
//!
//! **The rule** (Apple M4, macOS 27.0.1; for comparisons of a mipmapped `depth32Float` texture, for
//! colour samples). Metal rounds the level `x` to `half`, to nearest even (a NaN reads 0), clamps
//! it into the sampler's level clamp (whose bounds a constexpr sampler holds as `half`s), and into
//! the texture's levels. With `L = floor(x)` and the fraction truncated onto sixty-fourths, `q =
//! floor(64 (x - L)) / 64`:
//!
//! - a nearest mip filter reads level `L + 1` where `q > 1/2`, else level `L`;
//! - a linear mip filter blends levels `L` and `L + 1` with weights `1 - q` and `q`;
//! - a sampler without a mip filter reads level 0;
//! - every sampler magnifies (its magnification filter) where `L + q` is 0, and minifies
//!   elsewhere.
//!
//! Every lane of the probes reads so: 2,230 levels each (every sixty-fourth from -1 to 5, the
//! neighbourhoods of `L + 1/2` in steps of `1/4096`, every `1/1024` from 1 to 2, the `half` ties
//! around `L + k/64` with their `f32` neighbours, past the last level, the infinities and a NaN)
//! through nearest and linear filters under nearest and linear mip filters, of `r32Float`,
//! `rgba16Float` and `rgba8Unorm` 2D textures of five levels (the level and weight; an 8-bit
//! format's blend itself rounds onto sixteenths of a code, which the device's filter answers under
//! the sampler's note), and of `r32Float` 3D, cube and 2D-array textures; as a bias in a kernel
//! alike; through samplers clamping the level into `[0.75, 2.5]` and `[0.3, 2.7]`; and where a
//! sampler minifies and magnifies with different filters (a 16x16 texture of distinct texels).
//! Metal's compiler builds no kernel that samples with `min_lod_clamp` (its device compiler
//! stops), so a clamp beside a bias is not measured; nor is a texture of one level, where the
//! clamp into its levels would decide the filter of a sampler whose filters differ.
//!
//! **A depth comparison** reads its level by the same rule through a runtime, an argument
//! buffer's, and a constexpr sampler alike (2,048,400 comparisons at 1,707 levels through nearest
//! filters under nearest, linear and no mip filters and eleven clamps, a runtime or argument
//! buffer's sampler's as `f32`s); a linear mip filter blends the two levels' comparisons as an
//! `r8Unorm` texture's codes 0 and 255, rounded once onto sixteenths of a code, ties up
//! (`Depth32Float` and `Depth16Unorm`, one level, both, or neither passing). The shader makes those
//! comparisons itself (`compared`); a filtered comparison left to the device keeps the device's
//! level.
//!
//! **The device** chooses its level from `λ` (Vulkan's explicit `Lod`), clamped into the
//! sampler's `[minLod, maxLod]` (the same level clamp), at its own precision: a nearest mip filter
//! reads `ceil(λ + 1/2) - 1` (rounding `L + 1/2` either way is the implementation's), a linear one
//! blends `floor(λ)` and the next level by `λ`'s fraction to `mipmapPrecisionBits` (at least 6 on
//! the measured venues), and `λ <= 0` magnifies. So the translation hands it `λ` on Metal's grid
//! ([`LevelSnap::of`]): `L + q` for a linear mip filter (exact on a 1/64 grid) and for a sampler
//! without one (whose `maxLod` keeps level 0, and whose sign alone chooses the filter), and `L`,
//! `L + 1/4`, or `L + 1` for a nearest mip filter (no half-way point to round). It clamps `x` as
//! Metal does, and caps it at 64 (past every texture's levels; Metal's largest have 15) so that an
//! infinity's fraction is 0; the device's clamp into the texture's levels is then Metal's. The
//! device's `minLod` raises a `λ` below it: where the sampler's `lod_min_clamp` is on the 1/64 grid
//! that never happens under a linear mip filter, and under a nearest one it never moves `λ` onto or
//! past a half-way point unless the clamp's fraction lies in `[1/2, 33/64)`; a sampler clamping
//! otherwise keeps the device's level ([`LevelSnap::plan`]).
//!
//! **Reached:** a sample with a level or a kernel's bias (no `min_lod_clamp`) of a texture of
//! floating-point texels through a constexpr sampler with normalized coordinates, no anisotropy,
//! no level bias of its own, and a level clamp as above, on a device without
//! `VkProperty::MetalSampler`; on one with it, the device's sample is Metal's own, so the module
//! leaves the level to it and requires the property. A level known when the function compiles takes
//! no snap where it is on Metal's grid already, as `level(0)` is.
//!
//! **A runtime or argument buffer's sampler** holds its clamp as `f32`s, not `half`s (Apple M4 on
//! macOS 27.0.1: all 37,080 lanes of `level(x)` and a kernel's `bias(x)` through eighteen clamps,
//! nearest and linear filters and mip filters). The function does not know the clamp; the
//! `VkSampler` holds it on Metal's grid instead where the sampler takes one filter both ways and no
//! anisotropy (`metal::sampler::clamps_on_grid`): the level a nearest mip filter reads at each
//! bound, or the bound truncated onto sixty-fourths under a linear one. Metal's rule is monotone in
//! the level, so its level clamped by the bounds is the level clamped by those values; the function
//! hands the device the unclamped level on the grid (`by_word`: `L` or `L + 1` under a nearest mip
//! filter, `L + q` under a linear one), and the device's clamp moves it exactly as Metal's moves
//! the level. The sampler's word says whether it takes its clamp so ([`SamplerWord::LEVEL_RULE`]);
//! elsewhere the device keeps the level it is handed. With one filter both ways the level's sign
//! chooses nothing, so a level handed to the device needs no fraction to stay minified.
//!
//! **A level from derivatives or gradients** (a fragment function's implicit level, explicit
//! gradients, an LOD query) is the device's level arithmetic: Vulkan's level of detail, whose
//! scale factor may be approximated, and the rounding above. Metal's level of detail there is not
//! measured, so nothing hands the device Metal's. Nor does anything hand it Metal's level where
//! the rule is not reached: an integer texture's level, and a constexpr sampler's that
//! [`LevelSnap::plan`] leaves (a level bias of its own, a clamp it does not take). Where the level
//! chooses what a sample reads and no other note records the sample, the module records
//! `PrecisionNote::DeviceSampleLevel` ([`device_when`]).

use crate::air::abi::{Coordinates, MipFilter, SamplerState};
use crate::air::ir::{Const, Value};
use crate::error::Error;
use crate::float16::{f16_bits_to_f32, f32_to_f16_bits};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::texel_round::{self, HalfRounding};
use crate::lower::select::{any_of, NoteClause, NoteOn, NoteTest, SelectCx};
use crate::metal::sampler::SamplerWord;
use crate::metal::SamplerMipFilter;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConstBits, Constant, GlslOp, Type as SType, UnOp};

use super::{BiasReach, Lod, SampleOp};

#[cfg(test)]
mod tests;

/// The estimated words of `LevelSnap::emit`: the rounding to `half` (about 40 instructions of 4 to
/// 6 words), the NaN's select, the clamps, the floor, the fraction's grid, the mip filter's level,
/// and their constants.
pub const LEVEL_WORDS: u32 = 400;

/// The cap on a level before its floor: past every texture's levels, and an `f32` whose fraction
/// is 0.
const CAP: f32 = 64.0;

/// How a sample's level is handed to the device (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LevelRule {
    /// A constexpr sampler's, by its state.
    Stated(LevelSnap),
    /// A runtime or argument buffer's sampler's, by its word (`by_word`).
    Word,
}

/// How a sample's level is handed to the device (module docs): the sampler's mip filter and its
/// level clamp, as `f32` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LevelSnap {
    pub mip: MipFilter,
    /// The sampler's `lod_min_clamp`, a `half`'s value.
    pub min: u32,
    /// The sampler's `lod_max_clamp`, a `half`'s value.
    pub max: u32,
}

impl LevelSnap {
    /// The snap of a sample with a level through constexpr sampler `s`, or `None` where the rule
    /// is not measured or the device's clamp would move the level the snap hands it (module
    /// docs): a pixel sampler (which has no mip filter), an anisotropic one, one with a level bias
    /// of its own, one whose level clamp is empty, and one whose `lod_min_clamp` the device's
    /// `minLod` would raise a snapped level to (off the 1/64 grid under a linear mip filter, of a
    /// fraction in `[1/2, 33/64)` under a nearest one).
    pub fn plan(s: &SamplerState) -> Option<LevelSnap> {
        let (min, max) = (s.lod_min_clamp.to_f32(), s.lod_max_clamp.to_f32());
        if s.coordinates != Coordinates::Normalized
            || s.max_anisotropy > 1
            || s.lod_bias.to_f32() != 0.0
            || min.is_nan()
            || max.is_nan()
            || min > max
        {
            return None;
        }
        let fraction = min - min.floor();
        let raised = min > 0.0
            && match s.mip_filter {
                MipFilter::Linear => (fraction * 64.0).fract() != 0.0,
                MipFilter::Nearest => (0.5..33.0 / 64.0).contains(&fraction),
                MipFilter::None => false,
            };
        if raised {
            return None;
        }
        Some(LevelSnap {
            mip: s.mip_filter,
            min: min.to_bits(),
            max: max.to_bits(),
        })
    }

    /// The level `λ` handed to the device for Metal's level `x` (module docs).
    pub fn of(&self, x: f32) -> f32 {
        let (min, max) = (f32::from_bits(self.min), f32::from_bits(self.max));
        let x = if x.is_nan() {
            0.0
        } else {
            f16_bits_to_f32(f32_to_f16_bits(x))
        };
        let x = x.max(min).min(max).clamp(0.0, CAP);
        let l = x.floor();
        let q = ((x - l) * 64.0).floor() / 64.0;
        match self.mip {
            MipFilter::Nearest if q > 0.5 => l + 1.0,
            MipFilter::Nearest if q > 0.0 => l + 0.25,
            MipFilter::Nearest => l,
            MipFilter::Linear | MipFilter::None => l + q,
        }
    }

    /// [`Self::of`] of level `x` (an `f32`) in the function.
    pub(super) fn emit(&self, fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
        let h = half_level(fb, x)?;
        let zero = float(fb, 0.0)?;
        let min = float(fb, f32::from_bits(self.min))?;
        let max = float(fb, f32::from_bits(self.max))?;
        let cap = float(fb, CAP)?;
        let h = fb.glsl(GlslOp::FMax, &[h, min])?;
        let h = fb.glsl(GlslOp::FMin, &[h, max])?;
        let h = fb.glsl(GlslOp::FMax, &[h, zero])?;
        let h = fb.glsl(GlslOp::FMin, &[h, cap])?;
        let l = fb.glsl(GlslOp::Floor, &[h])?;
        let f = fb.binary_no_contraction(BinOp::FSub, h, l)?;
        let (sixty_four, step) = (float(fb, 64.0)?, float(fb, 1.0 / 64.0)?);
        let f = fb.binary_no_contraction(BinOp::FMul, f, sixty_four)?;
        let f = fb.glsl(GlslOp::Floor, &[f])?;
        let q = fb.binary_no_contraction(BinOp::FMul, f, step)?;
        let up = match self.mip {
            MipFilter::Linear | MipFilter::None => q,
            MipFilter::Nearest => {
                let (half, one, quarter) = (float(fb, 0.5)?, float(fb, 1.0)?, float(fb, 0.25)?);
                let next = fb.binary(BinOp::FOrdGreaterThan, q, half)?;
                let within = fb.binary(BinOp::FOrdGreaterThan, q, zero)?;
                let up = fb.select(within, quarter, zero)?;
                fb.select(next, one, up)?
            }
        };
        fb.binary_no_contraction(BinOp::FAdd, l, up)
    }
}

/// Level `x` (an `f32`) rounded to `half` to nearest even, a NaN read as 0 (module docs).
fn half_level(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let form = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let h = texel_round::f32_to_half(fb, form, x, HalfRounding::NearestEven)?;
    let zero = float(fb, 0.0)?;
    let nan = fb.unary(UnOp::IsNan, x)?;
    fb.select(nan, zero, h)
}

/// Level `x` as Metal reads it before the sampler's clamp, on its grid, in whole levels and
/// sixty-fourths `(L, q)` (module docs): rounded to `half`, a NaN 0, kept within `[0, 64]`.
fn on_grid(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<(SValue, SValue)> {
    let h = half_level(fb, x)?;
    let (zero, cap) = (float(fb, 0.0)?, float(fb, CAP)?);
    let h = fb.glsl(GlslOp::FMax, &[h, zero])?;
    let h = fb.glsl(GlslOp::FMin, &[h, cap])?;
    let l = fb.glsl(GlslOp::Floor, &[h])?;
    let f = fb.binary_no_contraction(BinOp::FSub, h, l)?;
    let (sixty_four, step) = (float(fb, 64.0)?, float(fb, 1.0 / 64.0)?);
    let f = fb.binary_no_contraction(BinOp::FMul, f, sixty_four)?;
    let f = fb.glsl(GlslOp::Floor, &[f])?;
    let q = fb.binary_no_contraction(BinOp::FMul, f, step)?;
    Ok((l, q))
}

/// Level `x` (an `f32`) as Metal reads it, on its grid, in whole levels and sixty-fourths `(L,
/// q)` (module docs): rounded to `half` (a NaN 0), clamped into the sampler's clamp `[min, max]`
/// (`f32`s; `None`: no maximum) and into the texture's levels `[0, last]` (`last` an `f32`).
pub(super) fn clamped_on_grid(
    fb: &mut FunctionBuilder<'_>,
    x: SValue,
    (min, max): (SValue, Option<SValue>),
    last: SValue,
) -> BResult<(SValue, SValue)> {
    let h = half_level(fb, x)?;
    let h = fb.glsl(GlslOp::FMax, &[h, min])?;
    let h = match max {
        Some(max) => fb.glsl(GlslOp::FMin, &[h, max])?,
        None => h,
    };
    let zero = float(fb, 0.0)?;
    let h = fb.glsl(GlslOp::FMax, &[h, zero])?;
    let h = fb.glsl(GlslOp::FMin, &[h, last])?;
    let l = fb.glsl(GlslOp::Floor, &[h])?;
    let f = fb.binary_no_contraction(BinOp::FSub, h, l)?;
    let (sixty_four, step) = (float(fb, 64.0)?, float(fb, 1.0 / 64.0)?);
    let f = fb.binary_no_contraction(BinOp::FMul, f, sixty_four)?;
    let f = fb.glsl(GlslOp::Floor, &[f])?;
    let q = fb.binary_no_contraction(BinOp::FMul, f, step)?;
    Ok((l, q))
}

/// The level a nearest mip filter reads at level `x` before the sampler's clamp (module docs):
/// `L + 1` where `q > 1/2`, else `L`, on [`on_grid`]'s `(L, q)`.
pub(super) fn nearest_on_grid(
    fb: &mut FunctionBuilder<'_>,
    (l, q): (SValue, SValue),
) -> BResult<SValue> {
    let (zero, half, one) = (float(fb, 0.0)?, float(fb, 0.5)?, float(fb, 1.0)?);
    let next = fb.binary(BinOp::FOrdGreaterThan, q, half)?;
    let up = fb.select(next, one, zero)?;
    fb.binary_no_contraction(BinOp::FAdd, l, up)
}

/// Whether `word`'s [`SamplerWord::LEVEL_RULE`] fields state the mip filter `mip` and a sampler
/// that takes its clamp on Metal's grid (a `bool`): normalized coordinates, no anisotropy, and
/// both filters nearest or both linear (`metal::sampler::clamps_on_grid`).
fn word_takes(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    mip: SamplerMipFilter,
) -> BResult<SValue> {
    let rule = word_const(fb, SamplerWord::LEVEL_RULE)?;
    let have = fb.binary(BinOp::BitwiseAnd, word, rule)?;
    let nearest = mip.as_raw() << SamplerWord::MIP_FILTER_SHIFT;
    let linear = nearest | SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR;
    let (nearest, linear) = (word_const(fb, nearest)?, word_const(fb, linear)?);
    let a = fb.binary(BinOp::IEqual, have, nearest)?;
    let b = fb.binary(BinOp::IEqual, have, linear)?;
    fb.binary(BinOp::LogicalOr, a, b)
}

/// A runtime or argument buffer's sampler's level of a sample at level `x` (an `f32`), by its
/// word `word` (module docs): the level handed to the device, `L` or `L + 1` where the word states
/// a sampler that takes its clamp on Metal's grid with a nearest mip filter, `L + q` where it
/// states one with a linear mip filter, `x` elsewhere; and the level before the sampler's clamp
/// that a nearest mip filter reads, which a nearest sample's snap reads.
pub(super) fn by_word(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    x: SValue,
) -> BResult<(SValue, SValue)> {
    let (l, q) = on_grid(fb, x)?;
    let nearest = nearest_on_grid(fb, (l, q))?;
    let linear = fb.binary_no_contraction(BinOp::FAdd, l, q)?;
    let takes_linear = word_takes(fb, word, SamplerMipFilter::Linear)?;
    let lambda = fb.select(takes_linear, linear, x)?;
    let takes_nearest = word_takes(fb, word, SamplerMipFilter::Nearest)?;
    Ok((fb.select(takes_nearest, nearest, lambda)?, nearest))
}

/// The level a runtime sampler's nearest sample reads (a `u32`) before the texture's levels clamp
/// it, by its word `word`, where the sample's level before the sampler's clamp reads `nearest`
/// (an `f32`, [`by_word`]'s, or 0 at level 0) under a nearest mip filter: that level clamped into
/// the levels the word's clamp leaves ([`SamplerWord::LEVEL_MIN`], [`SamplerWord::LEVEL_MAX`])
/// under a nearest mip filter, else 0 (a snapped sample's sampler has no mip filter there).
pub(super) fn snap_level(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    nearest: SValue,
) -> BResult<SValue> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let field = |fb: &mut FunctionBuilder<'_>, shift: u32| -> BResult<SValue> {
        let s = word_const(fb, shift)?;
        let v = fb.binary(BinOp::ShiftRightLogical, word, s)?;
        let m = word_const(fb, 0xf)?;
        fb.binary(BinOp::BitwiseAnd, v, m)
    };
    let min = field(fb, SamplerWord::LEVEL_MIN_SHIFT)?;
    let below = field(fb, SamplerWord::LEVEL_MAX_SHIFT)?;
    let last = word_const(fb, crate::metal::sampler::LAST_WORD_LEVEL)?;
    let max = fb.binary(BinOp::ISub, last, below)?;
    let n = fb.convert(crate::spirv::ir::ConvOp::ConvertFToU, nearest, u32_)?;
    let n = fb.glsl(GlslOp::UMax, &[n, min])?;
    let n = fb.glsl(GlslOp::UMin, &[n, max])?;
    let mip = word_const(fb, SamplerWord::MIP_FILTER)?;
    let mip = fb.binary(BinOp::BitwiseAnd, word, mip)?;
    let zero = word_const(fb, 0)?;
    let none = fb.binary(BinOp::IEqual, mip, zero)?;
    fb.select(none, zero, n)
}

/// The level a runtime or argument buffer's sampler's sample reads at level `x` (an `f32`; `None`:
/// level 0, a kernel's sample without a level or bias), by its word `word` (a `u32`), before the
/// texture's levels clamp it (a `u32`, [`snap_level`] of the level a nearest mip filter reads at
/// `x`): under a nearest mip filter the level Metal's rule reads, clamped into the levels the
/// word's clamp leaves, and level 0 without a mip filter. Metal reads a comparison's level so too.
pub(super) fn word_level(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    x: Option<SValue>,
) -> BResult<SValue> {
    let nearest = match x {
        Some(x) => {
            let grid = on_grid(fb, x)?;
            nearest_on_grid(fb, grid)?
        }
        None => float(fb, 0.0)?,
    };
    snap_level(fb, word, nearest)
}

/// The snap of sample `op`'s level `value` (the call's level or bias operand) through static
/// sampler `stat` of a texture of `integer` texels, where the module docs reach it: a constexpr
/// sampler's by its state, a runtime or argument buffer's (`stat` a slot, or `None`) by its word.
/// On a device with `VkProperty::MetalSampler` the device's level is Metal's, and the module
/// requires the property instead.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    stat: Option<(SamplerRef, Coordinates)>,
    (value, integer): (Option<Value>, bool),
) -> Result<Option<LevelRule>, Error> {
    let measured = matches!(
        op,
        SampleOp::Sample {
            lod: Lod::Level(Some(_))
                | Lod::Base {
                    bias: Some(_),
                    min: None,
                },
            min_lod: None,
            ..
        }
    );
    if !measured || integer {
        return Ok(None);
    }
    let constant = match value.and_then(|v| super::select::konst(cx, v)) {
        Some(Const::Null) => Some(0.0),
        Some(Const::Float(bits)) => Some(f32::from_bits(*bits as u32)),
        _ => None,
    };
    let rule = match stat {
        Some((SamplerRef::Constexpr(words), _)) => {
            let Some(snap) = LevelSnap::plan(&super::state_of(words)?) else {
                return Ok(None);
            };
            if constant.is_some_and(|x| snap.of(x).to_bits() == x.to_bits()) {
                return Ok(None);
            }
            LevelRule::Stated(snap)
        }
        // A level on the grid of every rule is a whole level within `[0, 64]`. A comparison
        // through a runtime sampler that reads one texel fetches it from the level its word reads
        // (`fetched::FetchLevel::Word`); a filtered one keeps the device's level.
        Some((SamplerRef::Slot(_), _)) | None => {
            let on_grid = constant.is_some_and(|x| x.fract() == 0.0 && (0.0..=CAP).contains(&x));
            if on_grid || !matches!(op, SampleOp::Sample { dref: None, .. }) {
                return Ok(None);
            }
            LevelRule::Word
        }
    };
    if cx.profile.device_samples() {
        cx.families.sampler.metal_sampler = true;
        return Ok(None);
    }
    Ok(Some(rule))
}

/// Where the level of `op` through `sampler` (`None`: an argument buffer's) is the device's level
/// arithmetic and no other note records it (`PrecisionNote::DeviceSampleLevel`, the module docs),
/// `filtered` whether the sample's filter note records it where its sampler filters
/// (`linear::filter_note`), `snapped` whether [`select`] hands the device Metal's level, `value`
/// the sample's level or bias, and `integer` whether the texture's texels are integers.
///
/// - Never on a device that samples through Metal's own sampler or natively
///   (`DeviceSampling`), nor for a cube, whose level `CubeGradientFootprint` records, nor for a
///   comparison, whose level `DeviceDepthComparison` records where the device compares.
/// - Always for an LOD query, which answers the level.
/// - For a sample at an implicit level whose bias is within the device's limit (a bias past it
///   and hoisted derivatives record `GradFromBias`), with explicit gradients, or at an explicit
///   level that [`select`] does not snap and the device may not read as Metal does
///   ([`reads_metal_level`]): where its sampler chooses a
///   level or a filter by the level (a mip filter, or minification and magnification filters
///   that differ) and the filter note does not hold. A filtered sample's note holds wherever its
///   sampler filters, so through a constexpr sampler that is nearest filters and a nearest mip
///   filter, through a runtime sampler where its word states those, and through an argument
///   buffer's sampler never; a sample no filter note records (an integer texture's) wherever its
///   sampler chooses, an argument buffer's always.
pub(super) fn device_when(
    cx: &SelectCx<'_>,
    op: &SampleOp,
    sampler: Option<SamplerRef>,
    (cube, filtered): (bool, bool),
    (snapped, value, integer): (bool, Option<Value>, bool),
) -> Result<Vec<NoteClause>, Error> {
    let always = || vec![NoteClause::new()];
    if cx.profile.device_samples() || cube {
        return Ok(Vec::new());
    }
    let state = match sampler {
        Some(SamplerRef::Constexpr(words)) => Some(super::state_of(words)?),
        _ => None,
    };
    match op {
        SampleOp::Lod { .. } => return Ok(always()),
        SampleOp::Sample { dref: Some(_), .. } => return Ok(Vec::new()),
        SampleOp::Sample {
            lod:
                Lod::Implicit {
                    reach: BiasReach::Within,
                    ..
                }
                | Lod::Grad(..),
            ..
        } => {}
        SampleOp::Sample {
            lod: Lod::Level(_) | Lod::Base { .. },
            ..
        } if !snapped => {
            let level = constant_level(cx, op, value);
            if level.is_some_and(|x| reads_metal_level(x, state.as_ref(), integer)) {
                return Ok(Vec::new());
            }
        }
        _ => return Ok(Vec::new()),
    }
    Ok(match (sampler, state) {
        (Some(SamplerRef::Constexpr(_)), Some(s)) => {
            let chooses = s.mip_filter != MipFilter::None || s.min_filter != s.mag_filter;
            match chooses && !filtered {
                true => always(),
                false => Vec::new(),
            }
        }
        (Some(SamplerRef::Slot(n)), _) => {
            let on = NoteOn::Sampler(n);
            let nearest_mip = 1 << SamplerWord::MIP_FILTER_SHIFT;
            match filtered {
                true => any_of([NoteTest::equal(on, SamplerWord::LINEAR_FILTER, nearest_mip)]),
                false => any_of([
                    NoteTest::differ(on, SamplerWord::MIP_FILTER, 0),
                    NoteTest::equal(on, SamplerWord::LINEAR, SamplerWord::MIN_LINEAR),
                    NoteTest::equal(on, SamplerWord::LINEAR, SamplerWord::MAG_LINEAR),
                ]),
            }
        }
        _ => match filtered {
            true => Vec::new(),
            false => always(),
        },
    })
}

/// The level sample `op` reads at an explicit level where the function states it: its level
/// `value` (`None`: level 0), or a kernel's sample without a level, at its bias `value` (`None`:
/// level 0); `None` where a clamp of the call's own enters or the value is not a constant.
fn constant_level(cx: &SelectCx<'_>, op: &SampleOp, value: Option<Value>) -> Option<f32> {
    let SampleOp::Sample {
        lod, min_lod: None, ..
    } = op
    else {
        return None;
    };
    match lod {
        Lod::Level(None)
        | Lod::Base {
            bias: None,
            min: None,
        } => Some(0.0),
        Lod::Level(Some(_)) | Lod::Base { min: None, .. } => {
            match value.and_then(|v| super::select::konst(cx, v)) {
                Some(Const::Null) => Some(0.0),
                Some(Const::Float(bits)) => Some(f32::from_bits(*bits as u32)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Whether the device reads Metal's level at the constant explicit level `x` that [`select`] does
/// not snap, through constexpr sampler `state` (`None`: a runtime or argument buffer's), of a
/// texture of `integer` texels:
/// - through a constexpr sampler, where `x` is a whole level that its clamp leaves whole within
///   `[0, 64]`, without a level bias of its own; or, of a texture of float texels, where its snap
///   hands the device `x` itself ([`LevelSnap::of`]);
/// - through a runtime or argument buffer's sampler, of a texture of float texels, where `x` is a
///   whole level within `[0, 64]`: its clamp is on Metal's grid there (`metal::sampler::vk_sampler`).
///   An integer texture's sampler keeps the descriptor's clamp, which may move such a level off it.
fn reads_metal_level(x: f32, state: Option<&SamplerState>, integer: bool) -> bool {
    let whole = |x: f32| x.fract() == 0.0 && (0.0..=CAP).contains(&x);
    match state {
        Some(s) => {
            let (min, max) = (s.lod_min_clamp.to_f32(), s.lod_max_clamp.to_f32());
            let clamps = !min.is_nan() && !max.is_nan() && min <= max;
            let kept =
                s.lod_bias.to_f32() == 0.0 && clamps && whole(x) && whole(x.max(min).min(max));
            let snaps = !integer
                && LevelSnap::plan(s).is_some_and(|snap| snap.of(x).to_bits() == x.to_bits());
            kept || snaps
        }
        None => !integer && whole(x),
    }
}

/// The `u32` constant `v`.
fn word_const(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<SValue> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// The `f32` constant `v`.
fn float(fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<SValue> {
    let ty = fb.module().ty(SType::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v.to_bits())),
    })?;
    fb.module().const_value(c)
}
