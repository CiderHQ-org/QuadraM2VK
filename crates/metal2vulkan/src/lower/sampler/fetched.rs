//! Texels the shader fetches itself where Metal's footprint and level are known: the one path for a
//! comparison's depths ([`super::compared`]) and for a plain gather's or nearest sample's texels
//! ([`Plain`]).
//!
//! **Why.** A device's sample or gather may read a texel otherwise than its fetch does: a denormal
//! `R32_SFLOAT` texel (a 32-bit float format's) reads as zero of its sign in
//! - RADV's gathers and nearest samples;
//! - NVIDIA's gathers;
//! - ANV's nearest samples;
//!
//! and `OpImageFetch` and `OpImageRead` keep it on RADV, ANV, NVIDIA and lavapipe
//! (`runtime_lane::depth_compares`). Metal's sample and gather keep it.
//!
//! **How.** Where Metal's footprint and level are known (`plan`), the shader fetches the texels
//! through `bicubic::footprint`:
//! - a gather's four taps at Metal's quantized position (`TapBase::Quantized`) of the base level,
//!   in gather order (`texels`);
//! - a nearest sample's texel `floor(x)` (`TapBase::Floor`) of the level Metal reads, on each of
//!   its one, two or three plane axes.
//!
//! Each tap is addressed as the sampler's axes say, and a border axis's outside tap reads the
//! border color as the bound format holds it: a constexpr sampler's addressing is the function's
//! own ([`FetchAddress::Modes`]), a runtime sampler's its slot's word's, read in the function
//! ([`FetchAddress::Word`]), and an argument buffer's sampler's its table entry's word's
//! ([`FetchAddress::Entry`]). The fetch sits behind the slot's `TexelFormat` specialization
//! (`branch`): the format classes the caller names take the fetches, and any other the device's own
//! read. A comparison names every class whose border texel is measured; a plain sample or gather
//! the 32-bit float class alone, the only one whose texels a device's sample reads otherwise than
//! its fetch. A heap texture's comparison fetches by the depth class its handle carries instead
//! (`compared::heap`). A sample's fetch arm of a 32-bit float texture through a combined binding is
//! its shape's sampling routine (`sampler::routine`), defined once in the module and called by
//! every sample of the shape with the coordinate, the layer, and the level's operand; a gather's
//! fetches and a blend of two levels stay inline.
//!
//! **The level** is known for:
//! - a gather (the base level);
//! - a sampler without a mip filter (level 0), an implicit level's in a fragment function
//!   included;
//! - under a nearest mip filter, a kernel's level or bias that snaps, or a level known when the
//!   function compiles: `floor` of the level handed to the device, clamped into the texture's
//!   levels;
//! - through a runtime or argument buffer's sampler, a level or a kernel's bias (or neither, level
//!   0): the level its word reads ([`FetchLevel::Word`]), which the caller fetches where the word
//!   states a sampler that reads one texel (`compared::runtime_sample`, `compared::heap`).
//!
//! **Through a word, and of a heap texture**: a plain gather or sample through a runtime sampler
//! slot is addressed by the slot's word ([`FetchAddress::Word`]), and a sample fetches only where
//! the word states one texel ([`word_reach`]), the device's sample elsewhere. A heap texture's
//! gather and sample, through a word ([`FetchAddress::Entry`] for an argument buffer's sampler) or
//! a constexpr sampler, fetch where the class its handle carries is the 32-bit float one
//! ([`heap`]), after the device's access. An arm of a switch-at-use keeps the device's read through
//! a word, so its arms share one emulation.
//!
//! **The device reads the texels** elsewhere, and the module records
//! `PrecisionNote::DeviceTexelDenormals` for them (`sampler::select`):
//! - an implicit level or gradients under a nearest mip filter in a fragment function;
//! - a filtered sample that no emulation reaches, a word's that filters included;
//! - a `min_lod_clamp` or a dynamic offset;
//! - an observed residency, a raw sRGB view, a cube, a switch-at-use's arm through a word.

use crate::air::abi::{AddressMode, Coordinates, Filter, MipFilter};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::air::ir::{Const, ValueId};
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::texture::{ImageRef, OffsetBase, TexelClass, TexelConv, TexelForm};
use crate::metal::bicubic::BicubicTexels;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, FuncId, TypeId, Value as SValue};
use crate::spirv::ir::{BinOp, Constant, ConvOp, GlslOp, ImageQuery, Type as SType};

use super::bicubic::{
    flag, footprint_from, footprint_on, missing_routine, scalar, value, Addressing, At, Bits,
    Footprint, Tap, TapAddress, TapBase, Taps, Texels, Tys,
};
use super::level::LevelSnap;
use super::linear::LinearImage;
use super::routine::{Fetched, Level, Shape};
use super::{BindingKey, CombinedImage, Lod, Offset, Pixel, SampleOp, Sampling, Source};

/// How the shader fetches the texels of a sample or gather (module docs): how its taps are
/// addressed, Metal's constant texel offset, and the level.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fetch {
    pub address: FetchAddress,
    pub offset: Option<Vec<i32>>,
    pub level: FetchLevel,
}

/// How a fetch addresses its taps (`bicubic::Addressing`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FetchAddress {
    /// A constexpr sampler's: each plane axis's addressing, the sampler's border color (which
    /// every axis clamping to zero or to the border reads), and whether the coordinate is in
    /// pixels.
    Modes {
        address: Vec<TapAddress>,
        border: crate::air::abi::BorderColor,
        pixel: bool,
    },
    /// Runtime sampler slot `sampler`'s word on `axes` plane axes: each axis's address mode, the
    /// border color and the coordinates, read in the function from the slot's `SamplerState`
    /// specialization.
    Word { sampler: u32, axes: u8 },
    /// An argument buffer's sampler's word on `axes` plane axes, as [`FetchAddress::Word`], read
    /// in the function from the sampler's table entry.
    Entry { axes: u8 },
}

/// The level the texels are fetched from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FetchLevel {
    /// A level known when the function compiles, clamped into the texture's levels.
    Fixed(u32),
    /// `floor` of the level the sample's level snap hands the device (`Sampling::level`), for the
    /// sample's level or bias operand, clamped into the texture's levels.
    Snapped(ValueId),
    /// The level a runtime or argument buffer's sampler's word reads at the sample's level or a
    /// kernel's bias (`None`: level 0), clamped into the texture's levels (`level::word_level`):
    /// under a nearest mip filter the level Metal's rule reads, clamped into the levels the word's
    /// level clamp leaves; level 0 without a mip filter. Metal reads a comparison's level through
    /// such a sampler as a sample's.
    Word(Option<ValueId>),
    /// A linear mip filter's two levels (`compared::blended`): of the level the constexpr sampler's
    /// snap hands the device for the sample's level or a kernel's bias (`None`: level 0), clamped
    /// into the texture's levels, `L + q`, levels `L` and `L + 1`, whose comparisons blend by `1 -
    /// q` and `q`.
    Blend(LevelSnap, Option<ValueId>),
}

/// The estimated words of a fetch's footprint beyond its taps: the size query, each of two axes'
/// position ([`super::position::POSITION_WORDS`]) and base, the layer's clamp, the border texel,
/// the format's branch and the device's arm, and the level (the levels query, a snapped level's
/// floor and clamp).
pub(super) const FETCH_WORDS: u32 = 2 * super::position::POSITION_WORDS + 320;

/// The estimated words of one fetched tap: its index under its addressing on each axis, the fetch,
/// the border's select, and the component's extraction.
pub(super) const FETCH_TAP_WORDS: u32 = 96;

impl Fetch {
    /// The estimated words of fetching the texels of `op`: the footprint and each tap (a snapped
    /// level's own words are its snap's, `level::LEVEL_WORDS`; a word's level is counted here,
    /// within as many), and for a runtime sampler's word the word's tests and each axis's position
    /// and taps under every mode, as a linear emulation's (`linear::WORD_WORDS`), and Metal's
    /// offset rule on each axis a constant offset moves (`position::OFFSET_RULE_WORDS`).
    pub fn words(&self, op: &SampleOp) -> u32 {
        let (taps, span) = match op {
            SampleOp::Gather { .. } => (4, 2),
            _ => (1, 1),
        };
        let word = match &self.address {
            FetchAddress::Modes { .. } => 0,
            FetchAddress::Word { axes, .. } | FetchAddress::Entry { axes } => {
                super::linear::WORD_WORDS
                    + u32::from(*axes)
                        * (super::position::MODE_POSITION_WORDS
                            + span * super::linear::WORD_TAP_WORDS)
            }
        };
        let offset = self
            .offset
            .as_deref()
            .map_or(0, |o| o.iter().filter(|o| **o != 0).count() as u32)
            * super::position::OFFSET_RULE_WORDS;
        let level = match self.level {
            FetchLevel::Word(_) => super::level::LEVEL_WORDS,
            // The second level's taps and the blend's arithmetic.
            FetchLevel::Blend(..) => super::level::LEVEL_WORDS + taps * FETCH_TAP_WORDS + 64,
            FetchLevel::Fixed(_) | FetchLevel::Snapped(_) => 0,
        };
        FETCH_WORDS + taps * FETCH_TAP_WORDS + word + offset + level
    }
}

/// A plain gather or nearest sample whose texels the shader fetches (module docs): where its
/// texel class is read (a texture slot's `TexelFormat` specialization, or the class a heap
/// texture's handle carries), and how the fetches are made.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Plain {
    pub image: LinearImage,
    pub fetch: Fetch,
}

/// The estimated words of a sample's test that its word states one texel ([`word_reach`]): the
/// filter mask and test, the mip filter's field and its tests, with their constants.
const REACH_WORDS: u32 = 64;

impl Plain {
    /// The estimated words of fetching the texels of `op` ([`Fetch::words`]), a sample's test of
    /// its word ([`word_reach`]), and a heap texture's class from its handle, its element loaded
    /// again in the fetch's arm, an argument buffer's sampler's word from its entry, and the branch
    /// (`linear::HEAP_WORDS`).
    pub fn words(&self, op: &SampleOp) -> u32 {
        let reach = match (&self.fetch.address, op) {
            (FetchAddress::Modes { .. }, _) | (_, SampleOp::Gather { .. }) => 0,
            _ => REACH_WORDS,
        };
        let heap = match (self.image, &self.fetch.address) {
            (LinearImage::Slot(_), _) => 0,
            (LinearImage::Heap, FetchAddress::Entry { .. }) => {
                super::linear::HEAP_WORDS + crate::memory::handles::ENTRY_FIELD_WORDS
            }
            (LinearImage::Heap, _) => super::linear::HEAP_WORDS,
        };
        self.fetch.words(op) + reach + heap
    }

    /// Whether the shader fetches every texel of `op` the device would read where the class is a
    /// 32-bit float one: a gather's, and a sample's through a constexpr sampler, always; a
    /// sample's through a runtime or argument buffer's sampler only where its word states one
    /// texel ([`word_reach`]), the device's sample elsewhere.
    pub fn fetches_every_texel(&self, op: &SampleOp) -> bool {
        matches!(op, SampleOp::Gather { .. })
            || matches!(self.fetch.address, FetchAddress::Modes { .. })
    }
}

/// The fetch of `op`, a gather or a sample without comparison with texel form `form`, of `image`
/// (shape `shape`) through `stat` (or, where `argument`, its argument buffer's sampler), on a
/// device without `VkProperty::MetalSampler` (module docs): a `float` result, a gather of a 2D
/// texture or array, or a sample of a 1D, 2D or 3D texture or array, where its footprint is
/// known: through a constexpr sampler that reads one texel, where [`plan`] knows its level;
/// through a runtime or argument buffer's sampler, addressed by its word ([`plan_word`]), the
/// shader fetching a sample's texel only where the word states one texel ([`word_reach`]). A
/// heap texture's fetch reads the class its handle carries. `operand` is the call's level or bias
/// operand. A device with the property reads the texels by Metal's own sampler, as an unfiltered
/// sample requires nothing of it (`linear::select`) and a gather's snap none (`position::plan`).
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    form: Option<TexelForm>,
    (stat, argument): (Option<(SamplerRef, Coordinates)>, bool),
    (image, shape): (ImageRef, ImageShape),
    operand: Option<crate::air::ir::Value>,
) -> Result<Option<Plain>, Error> {
    let half_gather = half_gather(cx, op, form);
    if form.map(|f| f.conv) != Some(TexelConv::F32) && !half_gather {
        return Ok(None);
    }
    let gather = match op {
        SampleOp::Gather { dref: None, .. } => true,
        SampleOp::Sample { dref: None, .. } => false,
        _ => return Ok(None),
    };
    let axes: u8 = match shape.dim {
        _ if shape.ms => return Ok(None),
        ImageDim::D2 => 2,
        ImageDim::D1 if !gather => 1,
        ImageDim::D3 if !gather => 3,
        _ => return Ok(None),
    };
    if cx.profile.device_samples() {
        cx.families.sampler.device_sampling = true;
        if !half_gather {
            return Ok(None);
        }
    }
    let fetch = match (stat.map(|(r, _)| r), argument) {
        (Some(SamplerRef::Constexpr(words)), _) => {
            let s = super::state_of(words)?;
            if !gather && !super::linear::unfiltered(&s) {
                return Ok(None);
            }
            let modes: Vec<AddressMode> = [s.address_s, s.address_t, s.address_r]
                .into_iter()
                .take(usize::from(axes))
                .collect();
            plan(cx, op, &s, &modes, operand)?
        }
        (Some(SamplerRef::Slot(sampler)), _) => plan_word(op, FetchAddress::Word { sampler, axes }),
        (None, true) => plan_word(op, FetchAddress::Entry { axes }),
        (None, false) => None,
    };
    let image = match image {
        ImageRef::Slot(n) => LinearImage::Slot(n),
        ImageRef::Heap { .. } => LinearImage::Heap,
    };
    cx.families.access.heap_class |= fetch.is_some() && image == LinearImage::Heap;
    Ok(fetch.map(|fetch| Plain { image, fetch }))
}

/// Whether `op` is a gather with a `half` result whose texels the shader fetches where the bound
/// class is a 16-bit float one: where the NaN switch asks for Metal's NaN
/// (`NanMinMaxPolicy::Exact`, the profile's `TexelNanRoute::Metal`) on a device without Metal's
/// own sampler. Metal's gather hands back a 16-bit float texel's own NaN; a device's gather need
/// not (NVIDIA's hands back every NaN as `0x7fffffff`), and its fetch does
/// where its read does.
fn half_gather(cx: &SelectCx<'_>, op: &SampleOp, form: Option<TexelForm>) -> bool {
    use crate::device::profile::TexelNanRoute;
    use crate::device::requirements::VkProperty;
    matches!(op, SampleOp::Gather { dref: None, .. })
        && form.map(|f| f.conv) == Some(TexelConv::F16)
        && cx.profile.fields().texel_nans == TexelNanRoute::Metal
        && !cx
            .profile
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler)
}

/// The format classes (`BicubicTexels` bits) whose texels the shader fetches for plain access
/// `s`: the 32-bit float class, or for a `half` gather ([`half_gather`]) the half-float one.
pub(super) fn classes(s: &Sampling) -> u32 {
    match (&s.op, s.texel.map(|f| f.conv)) {
        (SampleOp::Gather { .. }, Some(TexelConv::F16)) => BicubicTexels::HALF,
        _ => BicubicTexels::LINEAR,
    }
}

/// How the shader fetches the texels of `op` (a gather, a sample through a sampler that reads one
/// texel, its comparison taken out, or a comparison through nearest filters under a linear mip
/// filter, [`FetchLevel::Blend`]) through constexpr sampler `s` on the plane axes of `modes`, or
/// `None` where the device reads them (module docs). `operand` is the call's level or bias
/// operand.
pub(super) fn plan(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    s: &crate::air::abi::SamplerState,
    modes: &[crate::air::abi::AddressMode],
    operand: Option<crate::air::ir::Value>,
) -> Result<Option<Fetch>, Error> {
    let border = s.border_color;
    let constant = |offset: &Option<Offset>| match offset {
        None => Some(None),
        Some(Offset::Const(k)) => Some(Some(k.clone())),
        Some(Offset::Dynamic(..)) => None,
    };
    let (offset, level) = match op {
        SampleOp::Gather { offset, .. } => match constant(offset) {
            Some(o) => (o, FetchLevel::Fixed(0)),
            None => return Ok(None),
        },
        SampleOp::Sample {
            lod,
            min_lod: None,
            offset,
            ..
        } => {
            let Some(o) = constant(offset) else {
                return Ok(None);
            };
            let level = match (s.mip_filter, lod) {
                (MipFilter::None, _) => FetchLevel::Fixed(0),
                // A linear mip filter's two levels, read by a comparison through nearest filters
                // (`compared::blended`).
                (MipFilter::Linear, Lod::Level(v) | Lod::Base { bias: v, min: None })
                    if s.min_filter == Filter::Nearest
                        && s.mag_filter == Filter::Nearest
                        && s.max_anisotropy <= 1 =>
                {
                    match LevelSnap::plan(s) {
                        Some(snap) => FetchLevel::Blend(snap, *v),
                        None => return Ok(None),
                    }
                }
                (MipFilter::Nearest, Lod::Level(None)) => match LevelSnap::plan(s) {
                    Some(snap) => FetchLevel::Fixed(snap.of(0.0) as u32),
                    None => return Ok(None),
                },
                (
                    MipFilter::Nearest,
                    Lod::Level(Some(v))
                    | Lod::Base {
                        bias: Some(v),
                        min: None,
                    },
                ) => {
                    let Some(snap) = LevelSnap::plan(s) else {
                        return Ok(None);
                    };
                    match operand.and_then(|o| super::select::konst(cx, o)) {
                        Some(Const::Null) => FetchLevel::Fixed(snap.of(0.0) as u32),
                        Some(Const::Float(bits)) => {
                            FetchLevel::Fixed(snap.of(f32::from_bits(*bits as u32)) as u32)
                        }
                        _ => FetchLevel::Snapped(*v),
                    }
                }
                _ => return Ok(None),
            };
            (o, level)
        }
        _ => return Ok(None),
    };
    Ok(Some(Fetch {
        address: FetchAddress::Modes {
            address: modes
                .iter()
                .copied()
                .map(super::position::address)
                .collect(),
            border,
            pixel: s.coordinates == Coordinates::Pixel,
        },
        offset,
        level,
    }))
}

/// How the shader fetches the texels of `op` addressed by a sampler's word (`address`, a
/// [`FetchAddress::Word`] or [`FetchAddress::Entry`]): a gather's four taps of the base level; a
/// sample's one texel of the level the word reads at its level or a kernel's bias, or at level 0
/// without either ([`FetchLevel::Word`]), which the caller fetches where the word states a sampler
/// that reads one texel; else (an implicit level or gradients) of level 0, which the caller fetches
/// only where the word states no mip filter (`compared::runtime_sample`, `compared::heap`). `None`
/// for a dynamic offset or a `min_lod_clamp`, whose texels the device reads.
pub(super) fn plan_word(op: &SampleOp, address: FetchAddress) -> Option<Fetch> {
    let (offset, level) = match op {
        SampleOp::Gather { offset, .. } => (offset, FetchLevel::Fixed(0)),
        SampleOp::Sample {
            lod,
            min_lod: None,
            offset,
            ..
        } => match lod {
            Lod::Level(v) | Lod::Base { bias: v, min: None } => (offset, FetchLevel::Word(*v)),
            _ => (offset, FetchLevel::Fixed(0)),
        },
        _ => return None,
    };
    let offset = match offset {
        None => None,
        Some(Offset::Const(k)) => Some(k.clone()),
        Some(Offset::Dynamic(..)) => return None,
    };
    Some(Fetch {
        address,
        offset,
        level,
    })
}

/// The value of `s` from the texels fetched as `f` says through combined binding `key` (one
/// without comparison), where the slot `texture`'s `TexelFormat` specialization states one of the
/// format classes `classes` (`BicubicTexels` bits) and, where `reached`, a runtime sampler's word
/// states a sampler whose sample reads the texel fetched ([`word_reach`]): `fetched` makes it of
/// [`texels`]' texels. Elsewhere `device`'s access of the binding's sampled image makes it. Each
/// arm makes the sampled image it reads, so once a device specializes the module the other arm
/// is dead.
pub(super) fn branch(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (s, key): (&Sampling, BindingKey),
    (texture, f, classes, reached): (u32, &Fetch, u32, bool),
    image: CombinedImage,
    fetched: &dyn Fn(&mut FunctionBuilder<'_>, &[SValue]) -> BResult<SValue>,
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
) -> BResult<SValue> {
    let Some(spec) = crate::lower::texture::texel_format_const(cx, texture) else {
        return Err(cx.fail(format!(
            "texture slot {texture} has no `TexelFormat` specialization"
        )));
    };
    let Some(&(binding, _)) = cx.decl.families.sampler.combined.get(&key) else {
        return Err(cx.fail(format!("combined binding {key:?} is not declared")));
    };
    // A runtime sampler's word, which addresses its taps.
    let word = match &f.address {
        FetchAddress::Word { sampler, .. } => {
            Some(super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?)
        }
        FetchAddress::Modes { .. } => None,
        FetchAddress::Entry { .. } => {
            return Err(cx.fail("a slot's fetch addressed by an argument buffer's sampler"))
        }
    };
    let t = Tys::new(fb)?;
    let measured = flag(fb, &t, spec, classes)?;
    let measured = match (reached, word) {
        (true, Some(w)) => match word_reach(fb, &t, s, f, w)? {
            Some(r) => fb.binary(BinOp::LogicalAnd, measured, r)?,
            None => measured,
        },
        _ => measured,
    };
    // A sample's one texel is fetched by its shape's routine, called with the sample's plain
    // data (`routine::Fetched`); a gather's four, and a blend's two levels, inline.
    let routed = match (Fetched::of(s, key, texture, f), image.texel) {
        (Some(shape), TexelClass::Float) => {
            Some(fetch_routine(cx, fb, s, shape, f, (image, spec))?)
        }
        _ => None,
    };
    let t = &t;
    let fetch: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let x = match &routed {
            Some((r, args)) => {
                let x = fb.call(*r, args)?.ok_or_else(missing_routine)?;
                fetched(fb, &[x])?
            }
            None => {
                let si = binding.sampled_image(fb)?;
                let level = level(cx, fb, t, s, (f.level, word), si)?;
                let Some(taps) = f.taps(s, level, word) else {
                    return Err(cx.fail("a word-addressed fetch without its word"));
                };
                let x = texels(cx, fb, t, s, (si, image), Bits::Spec(spec), &taps)?;
                fetched(fb, &x)?
            }
        };
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let read: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let si = binding.sampled_image(fb)?;
        let x = device(fb, si)?;
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let Merged(merged) = fb.if_(cx, measured, fetch, Some(read))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("fetched texels' branch without its value")),
    }
}

/// Where a sample's fetch addressed by sampler word `word` (a `u32`) reads the texel Metal's
/// sample reads (a `bool`): the word states a sampler that reads one texel (nearest minification
/// and magnification, no linear mip filter, not anisotropic) and, but where the fetch reads the
/// level the word reads ([`FetchLevel::Word`]: a level or a kernel's bias), no mip filter. `None`
/// for a gather, whose four taps are Metal's whatever the word's filters.
fn word_reach(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    s: &Sampling,
    f: &Fetch,
    word: SValue,
) -> BResult<Option<SValue>> {
    if !matches!(s.op, SampleOp::Sample { .. }) {
        return Ok(None);
    }
    let (one, mip) = super::compared::one_texel(fb, t, word)?;
    if matches!(f.level, FetchLevel::Word(_)) {
        return Ok(Some(one));
    }
    let zero = scalar(fb, t.u32_, 0)?;
    let zero = value(fb, zero)?;
    let unmipped = fb.binary(BinOp::IEqual, mip, zero)?;
    Ok(Some(fb.binary(BinOp::LogicalAnd, one, unmipped)?))
}

/// The value of plain fetch `p` of heap texture sampling `s` (image facts `image`): `device`, the
/// device's access already made, in uniform control flow; and where the class the texture's handle
/// carries is the 32-bit float one and, for a sample through a runtime or argument buffer's
/// sampler, its word states one texel ([`word_reach`]), the texel or a gather's component of the
/// four the shader fetches from the element as a slot of that class would ([`texels`]). The fetch
/// arm holds fetches alone, so a divergent handle puts no sample in divergent flow.
pub(super) fn heap(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    image: CombinedImage,
    p: &Plain,
    device: SValue,
) -> BResult<SValue> {
    let handle = match s.source {
        Source::Heap { handle, .. } | Source::HeapSampler { handle, .. } => handle,
        Source::Combined(_) => return Err(cx.fail("a heap texture's fetch of a slot")),
    };
    let t = Tys::new(fb)?;
    let h = cx.operand(fb, handle)?;
    let bits = Bits::Value(crate::lower::texture::heap_texel_class(cx, fb, h)?);
    let word = match (&p.fetch.address, super::emit::word_pixel(s)) {
        (FetchAddress::Modes { .. }, _) => None,
        (_, Some(pixel)) => Some(super::emit::sampler_word(cx, fb, pixel)?),
        (_, None) => return Err(cx.fail("a word-addressed heap fetch without its word")),
    };
    let take = bits.test(fb, &t, classes(s))?;
    let take = match word {
        Some(w) => match word_reach(fb, &t, s, &p.fetch, w)? {
            Some(r) => fb.binary(BinOp::LogicalAnd, take, r)?,
            None => take,
        },
        None => take,
    };
    let t = &t;
    let fetch: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let (si, _, _) = super::emit::sampled_image(cx, fb, s.source)?;
        let level = level(cx, fb, t, s, (p.fetch.level, word), si)?;
        let Some(taps) = p.fetch.taps(s, level, word) else {
            return Err(cx.fail("a word-addressed heap fetch without its word"));
        };
        let x = texels(cx, fb, t, s, (si, image), bits, &taps)?;
        Ok(ArmEnd::FallThrough(vec![plain_value(fb, s, &x)?]))
    });
    let kept: ArmFn<'_, EmitCx<'_>> = Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![device])));
    let Merged(merged) = fb.if_(cx, take, fetch, Some(kept))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a heap texture's fetch without its texel")),
    }
}

/// The value of plain access `s` of its fetched `texels`, as the device's access returns it: a
/// gather's component of each of its four (a `vec4`), a sample's one texel, a depth texture's its
/// first component.
pub(super) fn plain_value(
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    texels: &[SValue],
) -> BResult<SValue> {
    match s.op {
        SampleOp::Gather { component, .. } => {
            let vec4 = Tys::new(fb)?.vec4;
            let mut parts = Vec::with_capacity(texels.len());
            for &texel in texels {
                parts.push(fb.composite_extract(texel, &[component])?);
            }
            fb.composite_construct(vec4, &parts)
        }
        _ => {
            let x = one(fb, texels)?;
            match s.depth {
                true => fb.composite_extract(x, &[0]),
                false => Ok(x),
            }
        }
    }
}

impl Fetch {
    /// The taps of `s` fetched as `self` says from level `level` (`None`: the base level), a
    /// runtime sampler's addressed by its word `word` (a `u32`): a gather's footprint of two taps
    /// per axis from Metal's quantized position, or a nearest sample's one from `floor` of its
    /// position. `None` for a word-addressed fetch without its word.
    pub(super) fn taps<'a>(
        &'a self,
        s: &Sampling,
        level: Option<SValue>,
        word: Option<SValue>,
    ) -> Option<Taps<'a>> {
        self.taps_of(matches!(s.op, SampleOp::Gather { .. }), level, word)
    }

    /// [`taps`](Self::taps) of a gather where `gather`, else of a sample.
    fn taps_of(
        &self,
        gather: bool,
        level: Option<SValue>,
        word: Option<SValue>,
    ) -> Option<Taps<'_>> {
        let addressing = match (&self.address, word) {
            (
                FetchAddress::Modes {
                    address,
                    border,
                    pixel,
                },
                _,
            ) => Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            },
            (FetchAddress::Word { axes, .. } | FetchAddress::Entry { axes }, Some(word)) => {
                Addressing::Word {
                    word,
                    axes: usize::from(*axes),
                }
            }
            (FetchAddress::Word { .. } | FetchAddress::Entry { .. }, None) => return None,
        };
        Some(Taps {
            addressing,
            offset: self.offset.as_deref(),
            span: if gather { &[0, 1] } else { &[0] },
            base: if gather {
                TapBase::Quantized
            } else {
                TapBase::Floor
            },
            level,
        })
    }
}

/// The `vec4` texels of `s` at `taps` (any [`Addressing`]: a constexpr sampler's, or a runtime
/// sampler's word) from sampled image `si` (image facts `image`), a border tap's texel the border
/// color seen through format bits `bits` (the slot's `TexelFormat` specialization, or a heap
/// element's class's): a gather's four in gather order, `x` `(i, j + 1)`, `y` `(i + 1, j + 1)`,
/// `z` `(i + 1, j)` and `w` `(i, j)`, of a span of two taps per axis; else the one of a span of
/// one tap per axis.
pub(super) fn texels(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    s: &Sampling,
    (si, image): (SValue, CombinedImage),
    bits: Bits,
    taps: &Taps<'_>,
) -> BResult<Vec<SValue>> {
    let at = At::of(cx, fb, s)?;
    texels_at(cx, fb, t, &at, (si, image), bits, taps)
}

/// [`texels`] of a sampling whose plain data is `at`.
fn texels_at(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At,
    (si, image): (SValue, CombinedImage),
    bits: Bits,
    taps: &Taps<'_>,
) -> BResult<Vec<SValue>> {
    let (fp, img) = footprint_from(fb, t, at, (si, image), bits, taps)?;
    let Some(sites) = sites(&fp) else {
        return Err(cx.fail("fetched texels' footprint without its taps"));
    };
    let mut out = Vec::with_capacity(sites.len());
    for site in &sites {
        out.push(fp.fetch(fb, img, site)?);
    }
    Ok(out)
}

/// The sites [`texels`] fetches over footprint `fp`: a span of two taps per axis's four in gather
/// order, `(i, j + 1)`, `(i + 1, j + 1)`, `(i + 1, j)` and `(i, j)`, else a span of one tap per
/// axis's one; `None` for another footprint.
fn sites(fp: &Footprint) -> Option<Vec<Vec<Tap>>> {
    let rows: Vec<&[Tap]> = fp.axes.iter().map(|(_, row)| &row[..]).collect();
    match &rows[..] {
        [&[s0, s1], &[t0, t1]] => {
            Some(vec![vec![s0, t1], vec![s1, t1], vec![s1, t0], vec![s0, t0]])
        }
        _ => rows
            .iter()
            .map(|r| match **r {
                [tap] => Some(tap),
                _ => None,
            })
            .collect::<Option<Vec<Tap>>>()
            .map(|site| vec![site]),
    }
}

/// The shared form of plain fetch `p` of sampling `s` in a switch-at-use (SEMANTICS 1012;
/// `shared::shared_value`): the taps once on the named candidate's level-0 size `size` (image facts
/// `image`), fetched in one dispatch over the candidates `c` from each arm's image where the named
/// format's bits `bits` are of the 32-bit float class, and `fetched` makes the value of their
/// texels; elsewhere `device`'s access, dispatched the same way from each arm's sampled image. Only
/// a fetch of the base level is shared (`Sampling::shared_shape`).
pub(super) fn emit_shared(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (size, image): (SValue, CombinedImage),
    p: &Plain,
    (bits, c): (SValue, &super::Candidates<'_>),
    fetched: &dyn Fn(&mut FunctionBuilder<'_>, &[SValue]) -> BResult<SValue>,
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
) -> BResult<SValue> {
    let t = Tys::new(fb)?;
    let bits = Bits::Value(bits);
    let measured = bits.test(fb, &t, BicubicTexels::LINEAR)?;
    let t = &t;
    let fetch: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let Some(taps) = p.fetch.taps(s, None, None) else {
            return Err(cx.fail("a shared fetch without its taps"));
        };
        let fp = footprint_on(cx, fb, t, s, (size, image), bits, &taps)?;
        let Some(sites) = sites(&fp) else {
            return Err(cx.fail("a shared fetch's footprint without its taps"));
        };
        let x = Texels::Candidates(c).fetch(cx, fb, (t, &fp), &sites, &|_, _, v| Ok(v))?;
        Ok(ArmEnd::FallThrough(vec![fetched(fb, &x)?]))
    });
    let read: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
            let si = c.sampled_image(cx, fb, a)?;
            Ok(vec![device(fb, si)?])
        };
        let null = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| {
            let z = fb.module().constant(Constant::Null { ty: t.vec4 })?;
            Ok(vec![value(fb, z)?])
        };
        let [x] = c.each(cx, fb, &read, &null)?[..] else {
            return Err(cx.fail("a shared fetch's device texel"));
        };
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let Merged(merged) = fb.if_(cx, measured, fetch, Some(read))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a shared fetch's branch without its texel")),
    }
}

/// The one texel [`texels`] fetched for a sample.
pub(super) fn one(_: &mut FunctionBuilder<'_>, texels: &[SValue]) -> BResult<SValue> {
    match texels {
        [x] => Ok(*x),
        _ => Err(super::bicubic::missing_tap()),
    }
}

/// The level the texels of `s` are fetched from (a `u32`), or `None` for the base level: a level
/// known when the function compiles, `floor` of the level the sample's snap hands the device, or
/// the level a runtime or argument buffer's sampler's word `word` (a `u32`) reads, clamped into
/// the levels of `si`'s image.
pub(super) fn level(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    s: &Sampling,
    (level, word): (FetchLevel, Option<SValue>),
    si: SValue,
) -> BResult<Option<SValue>> {
    let (level, operand) = match level {
        FetchLevel::Fixed(n) => (Level::Fixed(n), None),
        FetchLevel::Snapped(v) => {
            let Some(super::level::LevelRule::Stated(snap)) = s.level else {
                return Err(cx.fail("a snapped fetch level without its snap"));
            };
            (Level::Snapped(snap), Some(cx.operand(fb, v)?))
        }
        FetchLevel::Word(v) => {
            let x = match v {
                Some(v) => Some(cx.operand(fb, v)?),
                None => None,
            };
            (
                Level::Word {
                    operand: x.is_some(),
                },
                x,
            )
        }
        FetchLevel::Blend(..) => return Err(cx.fail("a blended fetch level outside its blend")),
    };
    level_at(cx, fb, t, (level, operand), word, si)
}

/// [`level`] of level `level` at its operand `operand` (the sample's level or bias, where it
/// reads one), a runtime sampler's by its word `word`.
fn level_at(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    (level, operand): (Level, Option<SValue>),
    word: Option<SValue>,
    si: SValue,
) -> BResult<Option<SValue>> {
    let level = match (level, operand) {
        (Level::Fixed(0), _) => return Ok(None),
        (Level::Fixed(n), _) => {
            let c = scalar(fb, t.u32_, u64::from(n))?;
            value(fb, c)?
        }
        (Level::Snapped(snap), Some(x)) => {
            let l = snap.emit(fb, x)?;
            let l = fb.glsl(GlslOp::Floor, &[l])?;
            fb.convert(ConvOp::ConvertFToU, l, t.u32_)?
        }
        (Level::Snapped(_), None) => {
            return Err(cx.fail("a snapped fetch level without its operand"))
        }
        (Level::Word { .. }, x) => {
            let Some(word) = word else {
                return Err(cx.fail("a word's fetch level without its word"));
            };
            super::level::word_level(fb, word, x)?
        }
    };
    let img = fb.image(si)?;
    let levels = fb.query(ImageQuery::Levels, img)?;
    let one = scalar(fb, t.u32_, 1)?;
    let one = value(fb, one)?;
    let last = fb.binary(BinOp::ISub, levels, one)?;
    Ok(Some(fb.glsl(GlslOp::UMin, &[level, last])?))
}

/// The routine of sample `s`'s fetch arm `shape` (`routine::Fetched`, fetching as `f` from a
/// combined binding of image facts `image` under the slot's `TexelFormat` specialization `spec`),
/// defined at its shape's first call, and its call's arguments: the coordinate and the array
/// layer as the sample states them, and the sample's level or bias where the level reads one,
/// each computed in the sample's own position, and the constant texel offset where a lane is off
/// zero.
fn fetch_routine(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    shape: Fetched,
    f: &Fetch,
    (image, spec): (CombinedImage, ConstId),
) -> BResult<(FuncId, Vec<SValue>)> {
    let at = At::of(cx, fb, s)?;
    let operand = match f.level {
        FetchLevel::Snapped(v) | FetchLevel::Word(Some(v)) => Some(cx.operand(fb, v)?),
        _ => None,
    };
    let mut args = vec![at.coord];
    args.extend(at.layer);
    args.extend(operand);
    if shape.offset.is_some() {
        let Some(o) = f.offset.as_deref() else {
            return Err(cx.fail("a fetch's routine without its offset"));
        };
        args.push(super::emit::const_offset(fb, o)?);
    }
    // The slot's format class, which a function reads as data.
    args.push(fb.module().const_value(spec)?);
    let mut tys = Vec::with_capacity(args.len());
    for &a in &args {
        tys.push(fb.value_ty(a)?);
    }
    let shape = Shape::Fetched(shape);
    if let Some(r) = cx.sampling.get(&shape, &tys) {
        return Ok((r, args));
    }
    let Shape::Fetched(fetched) = &shape else {
        return Err(cx.fail("a fetch's routine of another shape"));
    };
    let mut rcx = cx.for_new_function();
    let made = define(
        &mut rcx,
        fb.module(),
        (fetched, f),
        (&tys, at.parent),
        image,
    );
    cx.adopt(rcx);
    let r = made?;
    cx.sampling.insert(shape, tys, r);
    Ok((r, args))
}

/// Define the routine of fetch arm `shape` (fetching as `f`) in `mb`, its parameters of types
/// `tys` in [`fetch_routine`]'s order, its offset placed by the slot's tests `parent` where `shape`
/// reads them, of a combined binding of image facts `image`: the one texel the arm's code fetched
/// inline, read from `shape`'s binding and specializations and the bound format's bits its last
/// parameter carries.
fn define(
    cx: &mut EmitCx<'_>,
    mb: &mut ModuleBuilder,
    (shape, f): (&Fetched, &Fetch),
    (tys, parent): (&[TypeId], Option<OffsetBase>),
    image: CombinedImage,
) -> BResult<FuncId> {
    let Some(&(binding, _)) = cx.decl.families.sampler.combined.get(&shape.binding) else {
        return Err(cx.fail(format!(
            "combined binding {:?} is not declared",
            shape.binding
        )));
    };
    let f32_ = mb.ty(SType::Float { width: 32 })?;
    let ret = mb.ty(SType::Vector { elem: f32_, len: 4 })?;
    let mut fb = mb.sampling_routine(ret)?;
    let mut params = Vec::with_capacity(tys.len());
    for &ty in tys {
        params.push(fb.param(ty)?);
    }
    // The coordinate, then the layer, the level's operand and the offset, each where the shape
    // takes it, and the format's bits.
    let operand = matches!(
        shape.level,
        Level::Snapped(_) | Level::Word { operand: true }
    );
    let want =
        2 + usize::from(shape.layer) + usize::from(operand) + usize::from(shape.offset.is_some());
    let mut params = params.into_iter();
    let (Some(coord), true) = (params.next(), tys.len() == want) else {
        return Err(cx.fail("a fetch's routine without its parameters"));
    };
    let layer = shape.layer.then(|| params.next()).flatten();
    let operand = operand.then(|| params.next()).flatten();
    let offset = shape
        .offset
        .and_then(|lanes| params.next().map(|v| (v, lanes)));
    let Some(bits) = params.next().map(Bits::Value) else {
        return Err(cx.fail("a fetch's routine without its format's bits"));
    };
    let at = At {
        coord,
        layer,
        parent,
        offset,
    };
    let t = Tys::new(&mut fb)?;
    // A runtime sampler's word, which addresses its tap.
    let word = match &shape.address {
        FetchAddress::Word { sampler, .. } => Some(super::emit::sampler_word(
            cx,
            &mut fb,
            Pixel::Spec(*sampler),
        )?),
        FetchAddress::Modes { .. } => None,
        FetchAddress::Entry { .. } => {
            return Err(cx.fail("a slot's fetch addressed by an argument buffer's sampler"))
        }
    };
    let si = binding.sampled_image(&mut fb)?;
    let level = level_at(cx, &mut fb, &t, (shape.level, operand), word, si)?;
    // The offset is the call's (`at.offset`), not the first caller's constant.
    let Some(taps) = f.taps_of(false, level, word) else {
        return Err(cx.fail("a word-addressed fetch without its word"));
    };
    let taps = Taps {
        offset: None,
        ..taps
    };
    let x = texels_at(cx, &mut fb, &t, &at, (si, image), bits, &taps)?;
    let [x] = x[..] else {
        return Err(cx.fail("a sample's fetch without its one texel"));
    };
    fb.ret(Some(x))?;
    fb.finish()
}
