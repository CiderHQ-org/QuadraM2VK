//! Metal's anisotropic filter, emulated by the device's samples at explicit levels.
//!
//! **The rule** (an Apple M4 on macOS 27.0.1, a probe rounds ANI-p1-m4 to ANI-p7-m4, our own
//! synthetic textures and gradients; the M3 Max on macOS 26.7 answered every compared footprint
//! identically, EV ANI-p4-m3). For a sample of a 2D plane with gradients `gx`, `gy` in level-0
//! texels through a sampler of `maxAnisotropy` rounded up to a power of two, `a = 2^la`
//! (`metal::sampler::vk_sampler`):
//!
//! - each gradient's length is `max(M + 3m/16, 49M/64 + 21m/32)` of its larger and smaller
//!   component magnitudes `M`, `m`; the major axis is the longer, and the taps step along it
//!   divided by that length (not its Euclidean length);
//! - Metal takes logarithms by a four-segment piecewise-linear `log2` ([`vlog`]): the exponent,
//!   the top two mantissa bits choosing a segment, and the next ten bits its offset, in units of
//!   `2^-15` of `1/256` of a level; the determinant `|gx.x gy.y - gx.y gy.x|` of the gradients
//!   with each component's mantissa truncated to 12 bits gives the minor axis's;
//! - the level, in 256ths, is `max(V(det) - max(V(major), A0) + A0, V(major) - la U) >> 15`
//!   (`A0` the first segment's offset, `U` one level): the minor axis held at a texel of the major
//!   or more, and the major over `a`;
//! - the tap count is the first of 1, 2, 3, 4, 6, 8, 10, 12, 14 and 16 whose `ceil(64 log2 n) /
//!   64` the log ratio `V(major) - A0 - max(V(det) - V(major), 0)` does not pass, at most `a`
//!   (11,090 of 11,096 footprints);
//! - tap `j` of `n` weighs `floor(256 / n)` 256ths, the remainder one each to the middle taps,
//!   and sits `(j - (n - 1) / 2) s0` texels from the centre along the major axis, with the
//!   spacing `s0 = major · round_{1/4096}(g · S)`: `r'` the log ratio taken back through the
//!   chords between `2^(k/4)` and rounded to `1/256` (at most `a`), `g = (r' - 1) / 2` for two
//!   taps and `(1 - κ_n (1 - r'/n)) / n` past two (`κ_n` fitted per count), `S` the power
//!   `2^-λ` of the level past the texture's levels (the level past 0 under no mip filter), in
//!   sixty-fourths with a 6-bit mantissa ([`scale`]);
//! - each tap reads the level by the sampler's mip filter: level 0 without one, the level
//!   `L + 1` past half a level under a nearest one, and levels `L` and `L + 1` blended by the
//!   256ths `256 - q` and `q` under a linear one.
//!
//! **Exactness** (the emulation's residue against the M4's answers): the length, the logarithm,
//! the tap count, the weights and the direction are Metal's; the level is Metal's without
//! anisotropy and off by `1/256` of a level on about 1% of anisotropic samples; the spacing is
//! Metal's at the ratio cap and for two taps (with residue where `r'` lies within `0.07/256` of
//! a rounding tie), and an approximation past two taps below the cap (`κ_n`). Each tap is the
//! device's own filter at the level handed to it, so its footprint and sum are the device's. The
//! lane `runtime_lane::anisotropic_filter` measures it on `rgba8Unorm` at `maxAnisotropy` 2 to 16.
//! A linear filter is within 0.53 of a code of Metal on lavapipe, 0.41 on ANV, 0.91 on RADV (its
//! bilinear filter's own residue) and 0.68 on an RTX 3090, means at most 0.075; the 3090's own
//! filter misses by up to 62. A nearest filter's tap is a texel off Metal's where it sits within
//! the spacing's residue of a texel edge, the same channels on every device: 4 of 5,072 at
//! `maxAnisotropy` 8 or less, 37 to 47 at 16, up to 24.5 codes. `PrecisionNote::AnisotropicFilter`
//! records it.
//!
//! **The form.** Each tap hands the device whole levels under a linear mip filter and blends
//! the two by Metal's 256ths in the shader (two samples a tap): NVIDIA's blend of a fractional
//! level is not Metal's (an RTX 3090: 1.03 codes off without anisotropy), so the device is never
//! asked to blend. A level of 0 above Metal's 0 is handed as `2^-12`, which keeps the device's
//! minification filter and blends nothing a device's mip precision keeps. A nearest mip
//! filter's level is handed whole too: the 3090's anisotropic sampler blends the next level at a
//! fractional explicit level whatever its mipmap mode (EV ANI-l3-nv3090).
//!
//! **The loop**. Tap 0 is written out, and taps 1 to `n - 1` are one loop's body, which asks the
//! driver not to unroll it (`LoopControl` `DontUnroll`): each tap's arithmetic and the sum's order
//! are those of the taps written out one by one, so the answer is the same (the lane's statistics,
//! every sampler alike), and a driver compiles one tap's code instead of sixteen. A guest
//! fragment's variant of 135 emulated samples took 1.49M words instead of 1.79M, and its pipeline
//! 29.0 s to make on an RTX 3090 instead of 38.9 s.
//!
//! **Reach** (`select`): a sample without comparison of a 2D texture or 2D array of `float` or
//! `half` texels, at the fragment's implicit level (with a bias, through Metal's biased minor
//! axis) or with explicit gradients, through a constexpr sampler whose `maxAnisotropy` is above 1
//! or a runtime sampler slot whose word states anisotropy, on a device without
//! `VkProperty::MetalSampler`. A runtime slot's degree is its `SpecKey::SamplerAnisotropy`
//! specialization; where its word states no anisotropy, the sample is the plain one. Elsewhere
//! (a cube, a 3D texture, a comparison, an observed residency, an argument buffer's sampler) the
//! device filters, recorded as `PrecisionNote::DeviceAnisotropicFilter` where the sampler's
//! anisotropy is known.
//!
//! **The variant**. A runtime slot's emulation was about 3,200 words a sample with its taps written
//! out (about 1,900), at every sample it reaches: a guest's fragment function of 337 such samples
//! doubled (2.35M words from 1.18M), whatever samplers its draws bind, and every making of its
//! pipeline hands the driver the whole module. Such a sample states
//! `RuntimeFact::AnisotropicSampler` for its slot and takes the emulation only under the variant
//! naming the slot (`LoweringInputs::anisotropic_samplers`), which the binder asks for wherever
//! the bound sampler's word states anisotropy; elsewhere it is the plain sample, which is
//! Metal's for every word the function is then bound with. Under the variant the sample is the
//! form above, its word's test included, so a slot named or not, each word's answer is the one
//! before.

use crate::air::abi::{Coordinates, MipFilter};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::{NoteOn, NoteTest, SelectCx};
use crate::lower::texture::ImageRef;
use crate::metal::sampler::SamplerWord;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, Contraction, ConvOp, DerivOp, GlslOp, ImageOperands, ImageQuery,
    MemAccess, Type as SType, UnOp,
};

use super::{Lod, SampleOp, Sampling};

/// A sample that takes Metal's anisotropic filter, emulated (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Anisotropic {
    /// The sampler's degree.
    pub degree: Degree,
    /// The sampler's mip filter.
    pub mip: Mip,
}

/// The degree a sampler filters with, as `log2` of its `maxAnisotropy` rounded up to a power of
/// two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Degree {
    /// A constexpr sampler's: 1 to 4.
    Const(u32),
    /// Runtime sampler slot `n`'s: its `SpecKey::SamplerAnisotropy` specialization (0 for an
    /// isotropic sampler, whose sample is the plain one).
    Slot(u32),
}

/// The mip filter a sample reads its levels by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mip {
    /// A constexpr sampler's.
    Const(MipFilter),
    /// A runtime sampler slot's word's `SamplerWord::MIP_FILTER` field.
    Word,
}

/// The most taps Metal's filter takes (at `maxAnisotropy` 16); a degree past it (a runtime
/// sampler's 32) takes 16, its tap counts past 16 not being measured.
pub const MOST_TAPS: u32 = 16;

/// `log2` of [`MOST_TAPS`].
const MOST_LOG: i32 = 4;

/// The piecewise-linear `log2`'s segment offsets, in `2^-15` of `1/256` of a level (the first
/// holds the isotropic path's rounding offset alone).
const A: [i32; 4] = [17401, 2_717_697, 4_924_401, 6_790_136];

/// The segments' slopes per unit of the ten mantissa bits below the segment's two.
const B: [i32; 4] = [2637, 2155, 1822, 1578];

/// One level in [`vlog`]'s units.
const U: i32 = 256 << 15;

/// The tap counts, and the log ratio (in sixty-fourths of a level) each holds up to:
/// `ceil(64 log2 n)`.
const COUNTS: [(i32, i32); 10] = [
    (1, 0),
    (2, 64),
    (3, 102),
    (4, 128),
    (6, 166),
    (8, 192),
    (10, 213),
    (12, 230),
    (14, 244),
    (16, 256),
];

/// The spacing's correction past two taps, `κ_n`, fitted per tap count to the rounds 5 and 6 (an
/// approximation below the ratio cap, exact at it).
const KAPPA: [(i32, f32); 8] = [
    (3, 0.0600),
    (4, 0.0348),
    (6, 0.0255),
    (8, 0.0159),
    (10, 0.0210),
    (12, 0.0392),
    (14, 0.0153),
    (16, 0.0178),
];

/// `2^(k/4)`, `k = 0..=4`: the chords' ends of [`exp2_chord`].
const QUARTERS: [f32; 5] = [1.0, 1.189_207_1, std::f32::consts::SQRT_2, 1.681_792_9, 2.0];

/// The level handed for level 0 where Metal's level is above 0: the device minifies there, and
/// blends nothing of level 1 at any mip precision up to 11 bits.
const MINIFIED_ZERO: f32 = 1.0 / 4096.0;

/// Whether sample `op` of a texture of `shape` (`image`) through static sampler `sampler` (`None`
/// for an argument buffer's) takes Metal's anisotropic filter emulated (the module's reach), with
/// its note recorded: `AnisotropicFilter` where it does, `DeviceAnisotropicFilter` where the
/// sampler's anisotropy is known and the sample, filtered at an implicit level or with
/// gradients, is out of reach; a runtime sampler slot's under its word's anisotropy alone.
/// `integer`: the texel class is an integer one (never filtered); `observed`: the function
/// reads the sample's residency; `other_class`: the slot's variant binds a texture of another
/// class (`LoweringInputs::cross_class`); `swizzled`: the slot swizzles its converted texels
/// (`LoweringInputs::swizzled_storage`).
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    (sampler, image, shape): (Option<SamplerRef>, ImageRef, ImageShape),
    (integer, observed, other_class, swizzled): (bool, bool, bool, bool),
    m: &mut Meter,
) -> Result<Option<Anisotropic>, Error> {
    let metal_sampler = cx.profile.device_samples();
    let SampleOp::Sample { lod, dref, .. } = op else {
        return Ok(None);
    };
    let derived = matches!(
        lod,
        Lod::Implicit { .. } | Lod::Grad(..) | Lod::Hoisted | Lod::CubeGrad(..)
    );
    if integer || !derived {
        return Ok(None);
    }
    if metal_sampler {
        cx.families.sampler.device_sampling = true;
        return Ok(None);
    }
    let (degree, mip) = match sampler {
        Some(SamplerRef::Constexpr(words)) => {
            let state = super::constexpr::state_of(words)?;
            if state.max_anisotropy <= 1 || state.coordinates == Coordinates::Pixel {
                return Ok(None);
            }
            let la = state.max_anisotropy.next_power_of_two().trailing_zeros();
            (Degree::Const(la), Mip::Const(state.mip_filter))
        }
        Some(SamplerRef::Slot(n)) => (Degree::Slot(n), Mip::Word),
        None => return Ok(None),
    };
    let reached = dref.is_none()
        && shape.dim == ImageDim::D2
        && !shape.ms
        && !shape.depth
        && !matches!(lod, Lod::CubeGrad(..))
        && !observed
        && !other_class
        && !swizzled
        && matches!(image, ImageRef::Slot(_));
    let note = match reached {
        true => PrecisionNote::AnisotropicFilter,
        false => PrecisionNote::DeviceAnisotropicFilter,
    };
    match degree {
        Degree::Const(_) => {
            cx.precision.insert(note);
        }
        Degree::Slot(n) => {
            // A reached sample states its slot's fact, and takes the emulation only under the
            // variant naming the slot, which the binder asks for wherever the bound word states
            // anisotropy. Elsewhere it is the plain sample, with no emulation in the module:
            // Metal's sample wherever the word states no anisotropy, the only words a function
            // compiled without the slot is bound with there.
            if reached {
                let slot = super::select::sampler_slot(cx.stage, n);
                cx.facts
                    .insert(crate::pipeline::facts::RuntimeFact::AnisotropicSampler { slot });
                if !cx.inputs.anisotropic_samplers.contains(&slot) {
                    return Ok(None);
                }
            }
            if let Some(s) = cx.families.sampler.slots.get_mut(&n) {
                match reached {
                    true => s.emulated = true,
                    false => s.anisotropy = true,
                }
            }
            let anisotropic = NoteTest::differ(NoteOn::Sampler(n), SamplerWord::ANISOTROPIC, 0);
            cx.note_when(note, vec![[anisotropic].into()], m)?;
        }
    }
    Ok(reached.then_some(Anisotropic { degree, mip }))
}

/// Sample `s` through Metal's anisotropic filter (the module docs), in its result's form: for a
/// runtime sampler slot, where its word states anisotropy, else the plain sample. `grads` are a
/// switch-at-use's hoisted gradients (`Lod::Hoisted`).
pub(super) fn sample(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    a: &Anisotropic,
    grads: Option<(SValue, SValue)>,
) -> BResult<SValue> {
    let SampleOp::Sample {
        lod,
        dref: None,
        min_lod,
        offset,
    } = &s.op
    else {
        return Err(cx.fail("an anisotropic emulation of another access than a plain sample"));
    };
    let coord = cx.operand(fb, s.coord)?;
    // The level-0 size and the level count, and the gradients, in the sample's own position.
    let (si, image, _) = super::emit::sampled_image(cx, fb, s.source)?;
    let img = fb.image(si)?;
    let zero = u32c(fb, 0)?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    let size_lanes = image.plane + u8::from(image.arrayed);
    let levels = fb.query(ImageQuery::Levels, img)?;
    let (dx, dy) = match lod {
        Lod::Grad(dx, dy) => (cx.operand(fb, *dx)?, cx.operand(fb, *dy)?),
        Lod::Hoisted => match grads {
            Some(g) => g,
            None => return Err(cx.fail("a hoisted anisotropic sample without its gradients")),
        },
        Lod::Implicit { bias: None, .. } => (
            fb.derivative(DerivOp::DPdx, coord)?,
            fb.derivative(DerivOp::DPdy, coord)?,
        ),
        // Metal's bias shifts the footprint's minor axis alone.
        Lod::Implicit { bias: Some(b), .. } => {
            let b = cx.operand(fb, *b)?;
            let b = super::emit::metal_bias(fb, b)?;
            super::emit::plane_axis_gradients(fb, coord, b, (size, size_lanes))?
        }
        _ => return Err(cx.fail("an anisotropic emulation of a sample at an explicit level")),
    };
    let min_lod = match min_lod {
        Some(m) => Some(cx.operand(fb, *m)?),
        None => None,
    };
    let mut ops = ImageOperands::default();
    super::emit::offset_operands(cx, fb, offset, s.lanes, &mut ops)?;
    let layer = match s.layer {
        Some(l) => Some(cx.operand(fb, l)?),
        None => None,
    };
    let at = At {
        coord,
        grads: (dx, dy),
        size,
        levels,
        min_lod,
        ops,
        layer,
    };
    let (la, mip) = match (a.degree, a.mip) {
        (Degree::Const(la), Mip::Const(mip)) => {
            let la = i32c(fb, la.min(MOST_LOG as u32) as i32)?;
            let mip = u32c(fb, mip_code(mip))?;
            (la, mip)
        }
        (Degree::Slot(n), Mip::Word) => {
            let Some(&spec) = cx.decl.families.sampler.degrees.get(&n) else {
                return Err(cx.fail(format!("sampler slot {n} has no anisotropy degree")));
            };
            let word = super::emit::sampler_word(cx, fb, super::Pixel::Spec(n))?;
            let field = u32c(fb, SamplerWord::MIP_FILTER_SHIFT)?;
            let mip = fb.binary(BinOp::ShiftRightLogical, word, field)?;
            let three = u32c(fb, 3)?;
            let mip = fb.binary(BinOp::BitwiseAnd, mip, three)?;
            let i = int(fb)?;
            let la = fb.convert(ConvOp::Bitcast, spec, i)?;
            let most = i32c(fb, MOST_LOG)?;
            let la = fb.glsl(GlslOp::SMin, &[la, most])?;
            (la, mip)
        }
        _ => return Err(cx.fail("an anisotropic emulation of a mixed degree and mip filter")),
    };
    let most = match a.degree {
        Degree::Const(la) => 1 << la.min(MOST_LOG as u32),
        Degree::Slot(_) => MOST_TAPS,
    };
    match a.degree {
        Degree::Const(_) => emulated(cx, fb, s, &at, (la, mip), most),
        Degree::Slot(n) => {
            // The plain sample where the bound sampler's word states no anisotropy.
            let word = super::emit::sampler_word(cx, fb, super::Pixel::Spec(n))?;
            let anisotropic = super::emit::word_states_any(fb, word, SamplerWord::ANISOTROPIC)?;
            let plain = Sampling {
                anisotropic: None,
                ..s.clone()
            };
            let emulation: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = emulated(cx, fb, s, &at, (la, mip), most)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let device: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = super::emit::sampled_value(cx, fb, &plain, grads)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let Merged(merged) = fb.if_(cx, anisotropic, emulation, Some(device))?;
            match merged[..] {
                [x] => Ok(x),
                _ => Err(cx.fail("an anisotropic sample without its merged result")),
            }
        }
    }
}

/// What a sample's emulation reads, taken in the sample's own position.
#[derive(Clone, Copy)]
struct At {
    /// The plane coordinate (normalized).
    coord: SValue,
    /// The plane coordinate's gradients (normalized).
    grads: (SValue, SValue),
    /// The level-0 size (`u32` components, the layer count last for an array).
    size: SValue,
    /// The level count (`u32`).
    levels: SValue,
    /// The sample's `min_lod_clamp`.
    min_lod: Option<SValue>,
    /// The sample's offset operands.
    ops: ImageOperands,
    /// The array layer (an integer), for an array.
    layer: Option<SValue>,
}

/// Metal's mip filter code in the sampler word's field: 0 none, 1 nearest, 2 linear.
fn mip_code(mip: MipFilter) -> u32 {
    match mip {
        MipFilter::None => 0,
        MipFilter::Nearest => 1,
        MipFilter::Linear => 2,
    }
}

/// The emulation's result in `s`'s result form: the footprint, then up to `most` taps.
fn emulated(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    at: &At,
    (la, mip): (SValue, SValue),
    most: u32,
) -> BResult<SValue> {
    let f = float(fb)?;
    let v2 = fb.module().ty(SType::Vector { elem: f, len: 2 })?;
    let plane = fb.vector_shuffle(at.size, at.size, &[0, 1])?;
    let texels = fb.convert(ConvOp::ConvertUToF, plane, v2)?;
    let gx = fb.binary(BinOp::FMul, at.grads.0, texels)?;
    let gy = fb.binary(BinOp::FMul, at.grads.1, texels)?;
    let foot = footprint(fb, (gx, gy), (la, mip), at.levels)?;
    let levels = read_levels(fb, &foot, at, mip)?;
    // The taps' weights: `floor(256 / n)` each, the remainder one each to the middle taps.
    let n = foot.n;
    let full = i32c(fb, 256)?;
    let each = fb.binary(BinOp::SDiv, full, n)?;
    let used = fb.binary(BinOp::IMul, each, n)?;
    let rem = fb.binary(BinOp::ISub, full, used)?;
    let lo = fb.binary(BinOp::ISub, n, rem)?;
    let one = i32c(fb, 1)?;
    let lo = fb.binary(BinOp::ShiftRightArithmetic, lo, one)?;
    let hi = fb.binary(BinOp::IAdd, lo, rem)?;
    // The first tap's offset in texels along the major axis: `-(n - 1) s0 / 2`.
    let nf = fb.convert(ConvOp::ConvertSToF, n, f)?;
    let onef = f32c(fb, 1.0)?;
    let span = fb.binary(BinOp::FSub, nf, onef)?;
    let half = f32c(fb, 0.5)?;
    let span = fb.binary(BinOp::FMul, span, half)?;
    let taps = Taps {
        at: *at,
        u: foot.u,
        s0: foot.s0,
        span,
        texels,
        levels,
        weights: (each, lo, hi),
    };
    // Tap 0, then taps 1 to `n - 1` (at most `most`) in a loop of one tap's code that asks the
    // driver not to unroll it: each tap's arithmetic and the sum's order are those of the taps
    // written out one by one, so the answer is the same, and the driver compiles one tap instead
    // of `most`.
    let zero = i32c(fb, 0)?;
    let first = tap(cx, fb, s, &taps, zero)?;
    let i = int(fb)?;
    let sum_ty = fb.value_ty(first)?;
    let sum = fb.local_var(sum_ty, None)?;
    let sum = fb.var_ptr(sum)?;
    fb.store(sum, first, MemAccess::default())?;
    let index = fb.local_var(i, None)?;
    let index = fb.var_ptr(index)?;
    let one = i32c(fb, 1)?;
    fb.store(index, one, MemAccess::default())?;
    let most = i32c(fb, most as i32)?;
    let body: LoopFn<'_, EmitCx<'_>> = Box::new(move |fb, cx, lc| {
        let j = fb.load(index, MemAccess::default())?;
        let past_n = fb.binary(BinOp::SGreaterThanEqual, j, n)?;
        let past_most = fb.binary(BinOp::SGreaterThanEqual, j, most)?;
        let done = fb.binary(BinOp::LogicalOr, past_n, past_most)?;
        let exit: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let t = tap(cx, fb, s, &taps, j)?;
        let kept = fb.load(sum, MemAccess::default())?;
        let total = fb.binary(BinOp::FAdd, kept, t)?;
        fb.store(sum, total, MemAccess::default())?;
        let one = i32c(fb, 1)?;
        let next = fb.binary(BinOp::IAdd, j, one)?;
        fb.store(index, next, MemAccess::default())
    });
    let spec = LoopSpec {
        exit_sel: None,
        dont_unroll: true,
    };
    fb.loop_(cx, spec, body)?;
    let acc = fb.load(sum, MemAccess::default())?;
    let (_, image, _) = super::emit::sampled_image(cx, fb, s.source)?;
    super::emit::texel_value(cx, fb, s, image, acc, None)
}

/// Metal's footprint of one sample (the module docs).
struct Footprint {
    /// The level, in 256ths, held within the texture's levels.
    lambda: SValue,
    /// The tap count (`i32`).
    n: SValue,
    /// The step direction in texels (`vec2`).
    u: SValue,
    /// The spacing in level-0 texels.
    s0: SValue,
}

/// Metal's footprint of gradients `gx`, `gy` (level-0 texels) through a sampler of degree `2^la`
/// (`i32`) and mip filter code `mip` (`u32`) on a texture of `levels` (`u32`) levels.
fn footprint(
    fb: &mut FunctionBuilder<'_>,
    (gx, gy): (SValue, SValue),
    (la, mip): (SValue, SValue),
    levels: SValue,
) -> BResult<Footprint> {
    let f = float(fb)?;
    let i = int(fb)?;
    let v2 = fb.module().ty(SType::Vector { elem: f, len: 2 })?;
    let lx = length(fb, gx)?;
    let ly = length(fb, gy)?;
    let major = fb.glsl(GlslOp::FMax, &[lx, ly])?;
    let x_major = fb.binary(BinOp::FOrdGreaterThanEqual, lx, ly)?;
    let b2 = bool2(fb)?;
    let pick = fb.composite_construct(b2, &[x_major, x_major])?;
    let g = fb.select(pick, gx, gy)?;
    let zf = f32c(fb, 0.0)?;
    let some = fb.binary(BinOp::FOrdGreaterThan, major, zf)?;
    let ux = fb.composite_extract(g, &[0])?;
    let uy = fb.composite_extract(g, &[1])?;
    let ux = fb.binary(BinOp::FDiv, ux, major)?;
    let uy = fb.binary(BinOp::FDiv, uy, major)?;
    let onef = f32c(fb, 1.0)?;
    let ux = fb.select(some, ux, onef)?;
    let uy = fb.select(some, uy, zf)?;
    let u = fb.composite_construct(v2, &[ux, uy])?;
    let none = i32c(fb, -0x4000_0000)?;
    let vmaj = vlog(fb, major)?;
    let vmaj = fb.select(some, vmaj, none)?;
    // The determinant of the gradients, each component's mantissa truncated to 12 bits, rounded
    // as written (no contraction).
    let component = |fb: &mut FunctionBuilder<'_>, v: SValue, c: u32| -> BResult<SValue> {
        let x = fb.composite_extract(v, &[c])?;
        truncated(fb, x)
    };
    let (gxx, gyy) = (component(fb, gx, 0)?, component(fb, gy, 1)?);
    let (gxy, gyx) = (component(fb, gx, 1)?, component(fb, gy, 0)?);
    let p = fb.binary_with(BinOp::FMul, gxx, gyy, Contraction::Forbidden)?;
    let q = fb.binary_with(BinOp::FMul, gxy, gyx, Contraction::Forbidden)?;
    let det = fb.binary_with(BinOp::FSub, p, q, Contraction::Forbidden)?;
    let det = fb.glsl(GlslOp::FAbs, &[det])?;
    let has_det = fb.binary(BinOp::FOrdGreaterThan, det, zf)?;
    let vdet = vlog(fb, det)?;
    let a0 = i32c(fb, A[0])?;
    // The minor axis's level, the major held at a texel or more; the major's past the degree.
    let held = fb.glsl(GlslOp::SMax, &[vmaj, a0])?;
    let vmin = fb.binary(BinOp::ISub, vdet, held)?;
    let vmin = fb.binary(BinOp::IAdd, vmin, a0)?;
    let vmin = fb.select(has_det, vmin, none)?;
    let uc = i32c(fb, U)?;
    let cap = fb.binary(BinOp::IMul, la, uc)?;
    let capped = fb.binary(BinOp::ISub, vmaj, cap)?;
    let lv = fb.glsl(GlslOp::SMax, &[vmin, capped])?;
    let fifteen = i32c(fb, 15)?;
    let lambda = fb.binary(BinOp::ShiftRightArithmetic, lv, fifteen)?;
    // The log ratio, the minor at a texel or more.
    let zi = i32c(fb, 0)?;
    let lmin = fb.binary(BinOp::ISub, vdet, vmaj)?;
    let lmin = fb.glsl(GlslOp::SMax, &[lmin, zi])?;
    let lmin = fb.select(has_det, lmin, zi)?;
    let lr = fb.binary(BinOp::ISub, vmaj, a0)?;
    let lr = fb.binary(BinOp::ISub, lr, lmin)?;
    // The tap count: the first count whose threshold the ratio does not pass, at most `2^la`.
    let one = i32c(fb, 1)?;
    let degree = fb.binary(BinOp::ShiftLeftLogical, one, la)?;
    let mut n = degree;
    for &(count, th) in COUNTS.iter().rev() {
        let th = i32c(fb, th << 17)?;
        let within = fb.binary(BinOp::SLessThanEqual, lr, th)?;
        let count = i32c(fb, count)?;
        n = fb.select(within, count, n)?;
    }
    let n = fb.glsl(GlslOp::SMin, &[n, degree])?;
    // The spacing: `major · round_{1/4096}(g · S)`.
    let lrf = fb.convert(ConvOp::ConvertSToF, lr, f)?;
    let per = f32c(fb, 1.0 / U as f32)?;
    let lrf = fb.binary(BinOp::FMul, lrf, per)?;
    let r = exp2_chord(fb, lrf)?;
    let r = round_to(fb, r, 256.0)?;
    let degf = fb.convert(ConvOp::ConvertSToF, degree, f)?;
    let rp = fb.glsl(GlslOp::FMin, &[r, degf])?;
    let nf = fb.convert(ConvOp::ConvertSToF, n, f)?;
    let half = f32c(fb, 0.5)?;
    let two_g = fb.binary(BinOp::FSub, rp, onef)?;
    let two_g = fb.binary(BinOp::FMul, two_g, half)?;
    let mut kappa = f32c(fb, KAPPA.last().map_or(0.0, |&(_, k)| k))?;
    for &(count, k) in KAPPA.iter().rev().skip(1) {
        let c = i32c(fb, count)?;
        let is = fb.binary(BinOp::IEqual, n, c)?;
        let k = f32c(fb, k)?;
        kappa = fb.select(is, k, kappa)?;
    }
    let share = fb.binary(BinOp::FDiv, rp, nf)?;
    let short = fb.binary(BinOp::FSub, onef, share)?;
    let short = fb.binary(BinOp::FMul, kappa, short)?;
    let many_g = fb.binary(BinOp::FSub, onef, short)?;
    let many_g = fb.binary(BinOp::FDiv, many_g, nf)?;
    let two = i32c(fb, 2)?;
    let is_two = fb.binary(BinOp::IEqual, n, two)?;
    let g = fb.select(is_two, two_g, many_g)?;
    // `S`: the level past 0, less the level past the texture's last under a mip filter.
    let lvs = fb.glsl(GlslOp::SMax, &[lv, zi])?;
    let last = fb.convert(ConvOp::Bitcast, levels, i)?;
    let last = fb.binary(BinOp::ISub, last, one)?;
    let last = fb.glsl(GlslOp::SMax, &[last, zi])?;
    let top = fb.binary(BinOp::IMul, last, uc)?;
    let within = fb.glsl(GlslOp::SMin, &[lvs, top])?;
    let zu = u32c(fb, 0)?;
    let mipped = fb.binary(BinOp::INotEqual, mip, zu)?;
    let lvc = fb.select(mipped, within, zi)?;
    let past = fb.binary(BinOp::ISub, lvs, lvc)?;
    let sc = scale(fb, past)?;
    let gs = fb.binary(BinOp::FMul, g, sc)?;
    let gs = round_to(fb, gs, 4096.0)?;
    let s0 = fb.binary(BinOp::FMul, major, gs)?;
    let several = fb.binary(BinOp::SGreaterThan, n, one)?;
    let s0 = fb.select(several, s0, zf)?;
    // The level within the texture's levels.
    let full = i32c(fb, 256)?;
    let top = fb.binary(BinOp::IMul, last, full)?;
    let lambda = fb.glsl(GlslOp::SClamp, &[lambda, zi, top])?;
    Ok(Footprint { lambda, n, u, s0 })
}

/// Metal's piecewise-linear `log2` of a positive `f32` `x`: `A[s] + B[s] f + 2^23 e`, `e` the
/// unbiased exponent, `s` the top two mantissa bits and `f` the next ten (an `i32`).
fn vlog(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let u = uint(fb)?;
    let i = int(fb)?;
    let bits = fb.convert(ConvOp::Bitcast, x, u)?;
    let field = |fb: &mut FunctionBuilder<'_>, shift: u32, mask: u32| -> BResult<SValue> {
        let sh = u32c(fb, shift)?;
        let v = fb.binary(BinOp::ShiftRightLogical, bits, sh)?;
        let m = u32c(fb, mask)?;
        let v = fb.binary(BinOp::BitwiseAnd, v, m)?;
        fb.convert(ConvOp::Bitcast, v, i)
    };
    let e = field(fb, 23, 255)?;
    let s = field(fb, 21, 3)?;
    let f = field(fb, 11, 1023)?;
    let bias = i32c(fb, 127)?;
    let e = fb.binary(BinOp::ISub, e, bias)?;
    let twenty_three = i32c(fb, 23)?;
    let e = fb.binary(BinOp::ShiftLeftLogical, e, twenty_three)?;
    // A select chain from the last segment down, which is the fallback.
    let mut chosen: Option<(SValue, SValue)> = None;
    for (k, (ak, bk)) in A.into_iter().zip(B).enumerate().rev() {
        let ak = i32c(fb, ak)?;
        let bk = i32c(fb, bk)?;
        chosen = Some(match chosen {
            None => (ak, bk),
            Some((a, b)) => {
                let kc = i32c(fb, k as i32)?;
                let is = fb.binary(BinOp::IEqual, s, kc)?;
                (fb.select(is, ak, a)?, fb.select(is, bk, b)?)
            }
        });
    }
    let Some((a, b)) = chosen else {
        return i32c(fb, 0);
    };
    let bf = fb.binary(BinOp::IMul, b, f)?;
    let v = fb.binary(BinOp::IAdd, a, bf)?;
    fb.binary(BinOp::IAdd, v, e)
}

/// `x` with its mantissa truncated to 12 explicit bits.
fn truncated(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let u = uint(fb)?;
    let f = float(fb)?;
    let bits = fb.convert(ConvOp::Bitcast, x, u)?;
    let keep = u32c(fb, 0xffff_f800)?;
    let bits = fb.binary(BinOp::BitwiseAnd, bits, keep)?;
    fb.convert(ConvOp::Bitcast, bits, f)
}

/// Metal's length of gradient `v` (a `vec2`): `max(M + 3m/16, 49M/64 + 21m/32)`.
fn length(fb: &mut FunctionBuilder<'_>, v: SValue) -> BResult<SValue> {
    let x = fb.composite_extract(v, &[0])?;
    let y = fb.composite_extract(v, &[1])?;
    let x = fb.glsl(GlslOp::FAbs, &[x])?;
    let y = fb.glsl(GlslOp::FAbs, &[y])?;
    let big = fb.glsl(GlslOp::FMax, &[x, y])?;
    let small = fb.glsl(GlslOp::FMin, &[x, y])?;
    let term = |fb: &mut FunctionBuilder<'_>, p: f32, q: f32| -> BResult<SValue> {
        let pc = f32c(fb, p)?;
        let qc = f32c(fb, q)?;
        let a = fb.binary_with(BinOp::FMul, big, pc, Contraction::Forbidden)?;
        let b = fb.binary_with(BinOp::FMul, small, qc, Contraction::Forbidden)?;
        fb.binary_with(BinOp::FAdd, a, b, Contraction::Forbidden)
    };
    let first = term(fb, 1.0, 0.1875)?;
    let second = term(fb, 0.765_625, 0.656_25)?;
    fb.glsl(GlslOp::FMax, &[first, second])
}

/// `2^x` through the chords between `2^(k/4)` (the inverse of [`vlog`]'s segments).
fn exp2_chord(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let f = float(fb)?;
    let i = int(fb)?;
    let e = fb.glsl(GlslOp::Floor, &[x])?;
    let frac = fb.binary(BinOp::FSub, x, e)?;
    let four = f32c(fb, 4.0)?;
    let quarters = fb.binary(BinOp::FMul, frac, four)?;
    let s = fb.convert(ConvOp::ConvertFToS, quarters, i)?;
    let three = i32c(fb, 3)?;
    let s = fb.glsl(GlslOp::SMin, &[s, three])?;
    // A select chain from the last chord down, which is the fallback.
    let mut chosen: Option<(SValue, SValue)> = None;
    for (k, w) in QUARTERS.windows(2).enumerate().rev() {
        let &[a, b] = w else { continue };
        let a = f32c(fb, a)?;
        let b = f32c(fb, b)?;
        chosen = Some(match chosen {
            None => (a, b),
            Some((q0, q1)) => {
                let kc = i32c(fb, k as i32)?;
                let is = fb.binary(BinOp::IEqual, s, kc)?;
                (fb.select(is, a, q0)?, fb.select(is, b, q1)?)
            }
        });
    }
    let Some((q0, q1)) = chosen else {
        return f32c(fb, 1.0);
    };
    let sf = fb.convert(ConvOp::ConvertSToF, s, f)?;
    let quarter = f32c(fb, 0.25)?;
    let start = fb.binary(BinOp::FMul, sf, quarter)?;
    let within = fb.binary(BinOp::FSub, frac, start)?;
    let within = fb.binary(BinOp::FMul, within, four)?;
    let rise = fb.binary(BinOp::FSub, q1, q0)?;
    let rise = fb.binary(BinOp::FMul, within, rise)?;
    let m = fb.binary(BinOp::FAdd, q0, rise)?;
    let ei = fb.convert(ConvOp::ConvertFToS, e, i)?;
    fb.glsl(GlslOp::Ldexp, &[m, ei])
}

/// `2^-λ` of a level `v` in [`vlog`]'s units (`v >= 0`): `λ` in sixty-fourths
/// (`(v + 3 · 2^14) >> 17`), the fraction's power a 6-bit mantissa
/// (`round(64 · 2^(-f/64)) / 64`).
fn scale(fb: &mut FunctionBuilder<'_>, v: SValue) -> BResult<SValue> {
    let f = float(fb)?;
    let round = i32c(fb, 49152)?;
    let v = fb.binary(BinOp::IAdd, v, round)?;
    let seventeen = i32c(fb, 17)?;
    let l64 = fb.binary(BinOp::ShiftRightArithmetic, v, seventeen)?;
    let six = i32c(fb, 6)?;
    let whole = fb.binary(BinOp::ShiftRightArithmetic, l64, six)?;
    let mask = i32c(fb, 63)?;
    let frac = fb.binary(BinOp::BitwiseAnd, l64, mask)?;
    let fracf = fb.convert(ConvOp::ConvertSToF, frac, f)?;
    let step = f32c(fb, -1.0 / 64.0)?;
    let power = fb.binary(BinOp::FMul, fracf, step)?;
    let power = fb.glsl(GlslOp::Exp2, &[power])?;
    let sixty_four = f32c(fb, 64.0)?;
    let m = fb.binary(BinOp::FMul, power, sixty_four)?;
    let half = f32c(fb, 0.5)?;
    let m = fb.binary(BinOp::FAdd, m, half)?;
    let m = fb.glsl(GlslOp::Floor, &[m])?;
    let neg = fb.unary(UnOp::SNegate, whole)?;
    let six_bits = i32c(fb, -6)?;
    let neg = fb.binary(BinOp::IAdd, neg, six_bits)?;
    fb.glsl(GlslOp::Ldexp, &[m, neg])
}

/// `x` rounded to the nearest multiple of `1/steps`, ties up.
fn round_to(fb: &mut FunctionBuilder<'_>, x: SValue, steps: f32) -> BResult<SValue> {
    let k = f32c(fb, steps)?;
    let y = fb.binary(BinOp::FMul, x, k)?;
    let half = f32c(fb, 0.5)?;
    let y = fb.binary(BinOp::FAdd, y, half)?;
    let y = fb.glsl(GlslOp::Floor, &[y])?;
    let inv = f32c(fb, 1.0 / steps)?;
    fb.binary(BinOp::FMul, y, inv)
}

/// The levels each tap reads and their weights: `(first, second, w)`, the tap being
/// `(1 - w) · first + w · second`, `w` 0 where it reads one level.
#[derive(Clone, Copy)]
struct Levels {
    first: SValue,
    second: SValue,
    w: SValue,
}

/// The levels a tap reads at Metal's level `foot.lambda` (256ths), raised to the sample's
/// `min_lod_clamp`, by mip filter code `mip`: level 0 without a mip filter (handed the level, so
/// the device chooses its minification or magnification filter by its sign), the nearer level
/// under a nearest one (`L + 1` past half a level), and levels `L` and `L + 1` by Metal's weights
/// under a linear one. Every level is handed whole, level 0 above Metal's 0 as `MINIFIED_ZERO`.
fn read_levels(
    fb: &mut FunctionBuilder<'_>,
    foot: &Footprint,
    at: &At,
    mip: SValue,
) -> BResult<Levels> {
    let f = float(fb)?;
    let lf = fb.convert(ConvOp::ConvertSToF, foot.lambda, f)?;
    let inv = f32c(fb, 1.0 / 256.0)?;
    let mut lf = fb.binary(BinOp::FMul, lf, inv)?;
    if let Some(m) = at.min_lod {
        lf = fb.glsl(GlslOp::FMax, &[lf, m])?;
    }
    let zf = f32c(fb, 0.0)?;
    let above = fb.binary(BinOp::FOrdGreaterThan, lf, zf)?;
    let whole = fb.glsl(GlslOp::Floor, &[lf])?;
    let q = fb.binary(BinOp::FSub, lf, whole)?;
    let half = f32c(fb, 0.5)?;
    let up = fb.binary(BinOp::FOrdGreaterThan, q, half)?;
    let onef = f32c(fb, 1.0)?;
    let next = fb.binary(BinOp::FAdd, whole, onef)?;
    let nearest = fb.select(up, next, whole)?;
    // Whole levels only: an RTX 3090's anisotropic sampler blends the next level at an explicit
    // fractional level whatever its mipmap mode (EV ANI-l3-nv3090). Level 0 above Metal's 0 is
    // handed as `MINIFIED_ZERO`, which keeps the device's minification filter.
    let minified = f32c(fb, MINIFIED_ZERO)?;
    let nearest_at_zero = fb.binary(BinOp::FOrdEqual, nearest, zf)?;
    let lift = fb.binary(BinOp::LogicalAnd, nearest_at_zero, above)?;
    let nearest = fb.select(lift, minified, nearest)?;
    let at_zero = fb.binary(BinOp::FOrdEqual, whole, zf)?;
    let lift = fb.binary(BinOp::LogicalAnd, at_zero, above)?;
    let first_linear = fb.select(lift, minified, whole)?;
    let one = u32c(fb, 1)?;
    let two = u32c(fb, 2)?;
    let is_nearest = fb.binary(BinOp::IEqual, mip, one)?;
    let is_linear = fb.binary(BinOp::IEqual, mip, two)?;
    let first = fb.select(is_nearest, nearest, lf)?;
    let first = fb.select(is_linear, first_linear, first)?;
    let w = fb.select(is_linear, q, zf)?;
    Ok(Levels {
        first,
        second: next,
        w,
    })
}

/// Everything a tap reads.
#[derive(Clone, Copy)]
struct Taps {
    at: At,
    u: SValue,
    s0: SValue,
    /// `(n - 1) / 2`.
    span: SValue,
    /// The plane's level-0 size as floats.
    texels: SValue,
    levels: Levels,
    /// `floor(256 / n)`, and the taps from `lo` below `hi` that take one more 256th.
    weights: (SValue, SValue, SValue),
}

/// Tap `j`'s weighted texel (`vec4`), `j` an `int`, its samples made in the open block (and in
/// an arm of its own for the second level).
fn tap(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    t: &Taps,
    j: SValue,
) -> BResult<SValue> {
    let f = float(fb)?;
    let jf = fb.convert(ConvOp::ConvertSToF, j, f)?;
    let k = fb.binary(BinOp::FSub, jf, t.span)?;
    let off = fb.binary(BinOp::FMul, k, t.s0)?;
    let step = fb.binary(BinOp::VectorTimesScalar, t.u, off)?;
    let step = fb.binary(BinOp::FDiv, step, t.texels)?;
    let p = fb.binary(BinOp::FAdd, t.at.coord, step)?;
    let p = match t.at.layer {
        Some(l) => {
            let lf = fb.convert(ConvOp::ConvertUToF, l, f)?;
            let v3 = fb.module().ty(SType::Vector { elem: f, len: 3 })?;
            fb.composite_construct(v3, &[p, lf])?
        }
        None => p,
    };
    let (each, lo, hi) = t.weights;
    let from = fb.binary(BinOp::SGreaterThanEqual, j, lo)?;
    let below = fb.binary(BinOp::SLessThan, j, hi)?;
    let middle = fb.binary(BinOp::LogicalAnd, from, below)?;
    let one = i32c(fb, 1)?;
    let zi = i32c(fb, 0)?;
    let extra = fb.select(middle, one, zi)?;
    let w = fb.binary(BinOp::IAdd, each, extra)?;
    let w = fb.convert(ConvOp::ConvertSToF, w, f)?;
    let inv = f32c(fb, 1.0 / 256.0)?;
    let w = fb.binary(BinOp::FMul, w, inv)?;
    let Levels {
        first,
        second,
        w: q,
    } = t.levels;
    let ops = t.at.ops;
    let a = read(cx, fb, s, (p, ops), first)?;
    let zf = f32c(fb, 0.0)?;
    let blends = fb.binary(BinOp::FOrdGreaterThan, q, zf)?;
    let arm: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let b = read(cx, fb, s, (p, ops), second)?;
        let onef = f32c(fb, 1.0)?;
        let keep = fb.binary(BinOp::FSub, onef, q)?;
        let a = fb.binary(BinOp::VectorTimesScalar, a, keep)?;
        let b = fb.binary(BinOp::VectorTimesScalar, b, q)?;
        let x = fb.binary(BinOp::FAdd, a, b)?;
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let one_level: ArmFn<'_, EmitCx<'_>> = Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![a])));
    let Merged(merged) = fb.if_(cx, blends, arm, Some(one_level))?;
    let x = match merged[..] {
        [x] => x,
        _ => return Err(cx.fail("an anisotropic tap without its blended texel")),
    };
    fb.binary(BinOp::VectorTimesScalar, x, w)
}

/// The device's sample of `s`'s texture at `p` with offset operands `ops` at explicit level
/// `lod`, from a sampled image made in the open block.
fn read(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (p, mut ops): (SValue, ImageOperands),
    lod: SValue,
) -> BResult<SValue> {
    let (si, _, _) = super::emit::sampled_image(cx, fb, s.source)?;
    ops.lod = Some(lod);
    fb.sample(si, p, None, false, &ops)
}

fn float(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Float { width: 32 })
}

fn int(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: true,
    })
}

fn uint(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })
}

fn bool2(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    let b = fb.module().ty(SType::Bool)?;
    fb.module().ty(SType::Vector { elem: b, len: 2 })
}

fn f32c(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<SValue> {
    let f = float(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty: f,
        bits: ConstBits(u64::from(x.to_bits())),
    })?;
    fb.module().const_value(c)
}

fn i32c(fb: &mut FunctionBuilder<'_>, x: i32) -> BResult<SValue> {
    let i = int(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty: i,
        bits: ConstBits(u64::from(x as u32)),
    })?;
    fb.module().const_value(c)
}

fn u32c(fb: &mut FunctionBuilder<'_>, x: u32) -> BResult<SValue> {
    let c = fb.u32_const(x)?;
    fb.module().const_value(c)
}

#[cfg(test)]
mod tests;
