//! Depth comparisons of one texel per tap that the shader makes itself: a 2D depth texture's or
//! array's `gather_compare`, and its `sample_compare` through a sampler that reads one texel
//! (nearest filters, no mip filter or the nearest level's, no anisotropy), through a constexpr
//! sampler or a runtime sampler slot.
//!
//! **Why.** A device compares a `Depth16Unorm` texel against a reference by its own rule, and
//! Vulkan pins only the reference's clamp into `[0, 1]`: on 9,024 comparisons each, RADV differs
//! from Metal on 71 (a NaN reference compares as 0), ANV on 584 (its rounding threshold lies below
//! the half-way point and moves with the code; a NaN as 0), the RTX 3090 on 474 (its threshold lies
//! up to 0.51 of a step above the half-way point; a NaN as 0), and x86 lavapipe on 1,203 (it
//! compares the clamped reference with the texel as floats; a NaN as 1); MoltenVK on Apple's GPU
//! matches. Metal compares codes: the reference's is its exact product with 65535 rounded to
//! nearest (`linear::compare`'s `reference`), and a NaN fails every ordered test. A `Depth32Float`
//! texel compares as given, denormals included, which x86 lavapipe's float comparison misses (it
//! flushes them; `linear::compare`'s keys do not), and `depth32Float_stencil8` compares as
//! `depth32Float`. `Depth24Unorm_Stencil8` is not an Apple-family format: its comparison, made here
//! as a float format's, is not measured.
//!
//! **How.** The texels are read without comparison through a binding whose sampler does not
//! compare (`BindingKey::compare` false, so a `Depth32Float` texture held with a twin reads its
//! `R32_SFLOAT` image): a gather's four depths, the taps Metal's footprint names, or a nearest
//! sample's one depth. The shader then compares each with the reference through
//! `linear::compare::TapCompare`, the entry point every comparison the shader makes takes. A tap
//! outside a border axis reads the border's depth, which Metal compares as a texel's.
//!
//! **By fetches**. A device's sample or gather may read a denormal `R32_SFLOAT` texel as zero, and
//! its fetch keeps it, so where Metal's footprint and level are known the shader fetches the depths
//! itself through [`super::fetched`]: a gather's four taps and a nearest sample's texel, behind the
//! slot's `TexelFormat` specialization. A format whose border texel is measured (`Depth32Float`'s
//! and `Depth16Unorm`'s classes) takes the fetches, another the device's read. Elsewhere (an
//! implicit level or gradients in a fragment function, a `min_lod_clamp`, a dynamic offset, an
//! observed residency, axes clamping to zero and to a colored border) the device reads the texels,
//! snapped as a plain gather or nearest sample is (`position::plan`), at Metal's level.
//!
//! **A runtime sampler**. Its compare function is its word's `COMPARE` field, the raw
//! `MTLCompareFunction` (`TapCompare::of_word`), read from the slot's `SamplerState`
//! specialization. A comparison gather fetches its four depths as a constexpr sampler's does, its
//! taps addressed by the word (`fetched::FetchAddress::Word`). A `sample_compare` keeps its
//! reference, since whether the bound sampler reads one texel is known only at run time: a branch
//! on the word's filter (`runtime_sample`) compares in the shader where the sampler reads one texel
//! (nearest filters, no linear mip filter, no anisotropy), and elsewhere takes the filtered
//! comparison: `linear`'s emulation, which compares each tap by the same word, where it reaches the
//! sample, else the device's. Where it reads one texel, the shader fetches the texel, addressed by
//! the word, of the level the word reads at the sample's level or a kernel's bias (or level 0
//! without either): level 0 without a mip filter, and under a nearest one the level Metal's rule
//! reads, clamped into the levels the word's clamp leaves (`fetched::FetchLevel::Word`). Metal
//! reads a comparison's level through a runtime or argument buffer's sampler as a sample's, its
//! clamp held as `f32`s. At an implicit level or gradients in a fragment function it fetches level
//! 0's texel where the word states no mip filter, and under a nearest one samples without
//! comparison through the slot's binding without comparison ([`Compared::plain`]), as it does not
//! know the level Metal reads. A dynamic offset or a `min_lod_clamp` leaves the reads to the
//! device. Metal is taken to compare through a runtime sampler as through a constexpr sampler of
//! the same state, as it filters; the runtime lane's `depth_compares` runs both kinds of sampler on
//! each device, Metal's own through MoltenVK. The module records `DeviceDepthComparison` only where
//! the word reaches the device's comparison or level ([`Compared::device_when`]).
//!
//! **A heap texture**. The function does not know the format of an argument buffer's texture, so
//! the binder states its depth class in the texture's handle-table entry (`Depth16Unorm`'s codes,
//! `Depth32Float`'s and `Depth32Float_Stencil8`'s values, none for any other format), which the
//! handle carries (`memory::handles::Lookup::Flagged`). A heap comparison through a constexpr
//! sampler that reads one texel, a runtime sampler slot, or an argument buffer's sampler keeps its
//! reference, and the shader makes both reads (`heap`): the device's comparison, and the depths
//! read through the use's sampler without comparison (a second heap sampler, or the sampler array
//! of that use), which it compares as a slot's, by the class. It takes its own where the handle
//! states a class and, for a runtime or argument buffer's sampler's sample, the sampler's word
//! states one that reads one texel; elsewhere the device's, which the module records
//! (`DeviceDepthComparison`). Every read runs, so no branch puts an access in divergent flow. Where
//! Metal's footprint and level are known, as for a slot, the shader fetches the depths from that
//! element (an argument buffer's sampler's taps addressed by its table entry's word,
//! `fetched::FetchAddress::Entry`), its border texel read as a slot bound in the class's format
//! reads it; a runtime or argument buffer's sampler's sample takes the texel of the level its word
//! reads at its level or a kernel's bias, as a slot's does, and at an implicit level or gradients
//! level 0's where its word states no mip filter, and under a nearest one samples it, as the
//! function does not know the level Metal reads. Elsewhere the device reads the texels, so a
//! `Depth32Float` denormal it may read as zero is recorded (`DeviceTexelDenormals`). A texture
//! bound in a format without a class keeps the device's comparison.
//!
//! **Not reached**, the device comparing: a sample whose dynamic bias may pass the device's limit
//! through a runtime, argument buffer's, or (of a heap texture) constexpr sampler (its second
//! sample), an access whose residency the function observes, a cube, a sample that filters through
//! a constexpr sampler (a linear comparison of a slot is `linear`'s emulation), and a device with
//! `VkProperty::MetalSampler`, whose comparison is Metal's own, which the module then requires.

use crate::air::abi::{CompareFunction, Coordinates, Filter, MipFilter};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::air::ir::ValueId;
use crate::error::Error;
use crate::lower::arith::legal::{Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::texture::ImageRef;
use crate::memory::handles::{ENTRY_FIELD_WORDS, HANDLE_DEPTH_CODES, HANDLE_DEPTH_VALUES};
use crate::metal::bicubic::BicubicTexels;
use crate::metal::sampler::SamplerWord;
use crate::metal::texel_format::texel_format;
use crate::metal::PixelFormat;
use crate::metal::SamplerMipFilter;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, ImageQuery, Type as SType};

use super::bicubic::{scalar, value, Bits, Tys};
use super::fetched::{Fetch, FetchAddress};
use super::linear::compare::TapCompare;
use super::linear::Comparison;
use super::{BiasReach, CombinedImage, Lod, Pixel, SampleOp, Sampling, Source};

/// The census name of the masks a heap handle's depth flags cost every heap operation of a module
/// whose search carries no out-of-bounds alpha (`lower::texture::routine_words`): the heap
/// comparisons the shader makes ask for them.
pub const DEPTH_CLASS_CENSUS: &str = "sampler.heap.depth_class";

/// What says how a comparison's texels compare (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FormatBits {
    /// The texture slot's `TexelFormat` specialization: `BicubicTexels::EXACT` for the codes of
    /// `Depth16Unorm`.
    Slot(u32),
    /// The depth flags heap handle `handle` carries (`memory::handles::HANDLE_DEPTH_CODES` and
    /// `HANDLE_DEPTH_VALUES`).
    Heap(ValueId),
}

/// A comparison the shader makes of the texels read without comparison (module docs): the
/// sampler's compare function (a constexpr sampler's, or a runtime or argument buffer's sampler's
/// word's), the reference, what says how the texels compare, and how the shader fetches them,
/// where it does.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Compared {
    pub function: Comparison,
    pub dref: ValueId,
    pub format: FormatBits,
    pub fetch: Option<Fetch>,
    /// The read without comparison of a comparison that keeps its reference
    /// ([`Compared::keeps_reference`], set by `sampler::select`): a runtime sampler's
    /// `sample_compare`'s slot binding without comparison, through which it samples where the
    /// bound sampler reads one texel, and a heap comparison's element through its use's sampler
    /// without comparison.
    pub plain: Option<Source>,
}

/// The estimated words of a runtime sampler's `sample_compare` beyond its comparison
/// ([`runtime_sample`]): the word's filter test, the branch, the sample without comparison and its
/// sampled image, the depth's extraction, and the device's sampled image in its own arm.
const RUNTIME_WORDS: u32 = 160;

/// The estimated words a runtime sampler's `sample_compare` adds where it fetches its texel
/// ([`runtime_sample`]): the word's mip filter test, its branch, and the second sample without
/// comparison, in the arm the format's test leaves to the device.
const RUNTIME_FETCH_WORDS: u32 = 96;

/// The estimated words a heap comparison adds beyond its comparison ([`heap`]): the handle's two
/// depth flags' tests and the format bits' select, the read without comparison (its sampler's load,
/// or an argument buffer's sampler's element, access chain and load; the sampled image, the access,
/// and its depths' extraction and assembly), an argument buffer's sampler's word, a runtime or
/// argument buffer's sampler's sample's test of the word's filter, and the final select, with their
/// constants.
const HEAP_WORDS: u32 = 256 + 2 * ENTRY_FIELD_WORDS;

/// The format classes whose depths the shader fetches (module docs): those whose border texel is
/// measured, `Depth32Float`'s and `Depth16Unorm`'s among them.
const MEASURED: u32 = BicubicTexels::EXACT | BicubicTexels::LINEAR | BicubicTexels::HALF;

impl Compared {
    /// The estimated words: the reference's code and each tap's test, the fetches where the shader
    /// fetches the texels (a snapped level's own words are its snap's, `level::LEVEL_WORDS`), and a
    /// comparison's that keeps its reference.
    pub fn words(&self, op: &SampleOp) -> u32 {
        let taps = match op {
            SampleOp::Gather { .. } => 4,
            _ if self.blends() => 2,
            _ => 1,
        };
        let fetch = self.fetch.as_ref().map_or(0, |f| f.words(op));
        let kept = match (self.format, self.plain, &self.fetch) {
            (_, None, _) => 0,
            (FormatBits::Heap(_), Some(_), _) => HEAP_WORDS,
            (FormatBits::Slot(_), Some(_), Some(_)) => RUNTIME_WORDS + RUNTIME_FETCH_WORDS,
            (FormatBits::Slot(_), Some(_), None) => RUNTIME_WORDS,
        };
        self.function.words(taps) + fetch + kept
    }

    /// Whether op `op` keeps its reference for the device's comparison beside the shader's
    /// (module docs): a runtime sampler's `sample_compare` of a slot, whose filter only the
    /// sampler's word states (`runtime_sample`), a blend of two levels' comparisons, whose
    /// device arm compares a format whose class is not measured ([`blended`]), and every heap
    /// comparison, whose class only the handle states (`heap`).
    pub fn keeps_reference(&self, op: &SampleOp) -> bool {
        match self.format {
            FormatBits::Heap(_) => true,
            FormatBits::Slot(_) => {
                (self.function == Comparison::Word && matches!(op, SampleOp::Sample { .. }))
                    || self.blends()
            }
        }
    }

    /// Whether the comparison blends two levels' comparisons under a linear mip filter
    /// ([`blended`], `fetched::FetchLevel::Blend`).
    pub fn blends(&self) -> bool {
        self.fetch
            .as_ref()
            .is_some_and(|f| matches!(f.level, super::fetched::FetchLevel::Blend(..)))
    }

    /// Where the device compares a blend's sample ([`blended`]): where the slot's `TexelFormat`
    /// names no measured class.
    pub(super) fn blend_device_when(&self) -> Vec<crate::lower::select::NoteClause> {
        use crate::lower::select::{NoteClause, NoteOn, NoteTest};
        match self.slot() {
            Some(texture) => vec![[NoteTest::equal(NoteOn::Texel(texture), MEASURED, 0)].into()],
            None => vec![NoteClause::new()],
        }
    }

    /// Where the device compares a runtime sampler slot `n`'s `sample_compare` of a slot, whose
    /// filtered comparison is emulation `linear`'s where it is planned ([`runtime_sample`]): where
    /// the word filters and the emulation does not reach it (`linear::device_when`; without an
    /// emulation, wherever the word filters); and where the word reads one texel under a nearest
    /// mip filter but the shader does not fetch it from the level the word reads (an implicit level
    /// or gradients, a dynamic offset, or a `min_lod_clamp`), whose level the device chooses.
    pub(super) fn device_when(
        &self,
        n: u32,
        linear: Option<&super::linear::Linear>,
    ) -> Vec<crate::lower::select::NoteClause> {
        use crate::lower::select::{NoteOn, NoteTest};
        let mut when = match linear {
            Some(l) => super::linear::device_when(l),
            None => super::linear::word_filters(n),
        };
        let leveled = self
            .fetch
            .as_ref()
            .is_some_and(|f| matches!(f.level, super::fetched::FetchLevel::Word(_)));
        if !leveled {
            let one = SamplerWord::MIN_LINEAR
                | SamplerWord::MAG_LINEAR
                | SamplerWord::ANISOTROPIC
                | SamplerWord::MIP_FILTER;
            let nearest = SamplerMipFilter::Nearest.as_raw() << SamplerWord::MIP_FILTER_SHIFT;
            when.push([NoteTest::equal(NoteOn::Sampler(n), one, nearest)].into());
        }
        when
    }

    /// Whether the comparison is a heap texture's (`heap`).
    pub fn on_heap(&self) -> bool {
        matches!(self.format, FormatBits::Heap(_))
    }

    /// Whether the shader's comparison of op `op` reads only texels it fetches (module docs): a
    /// fetched one, but a runtime sampler slot's `sample_compare` and a heap texture's through a
    /// runtime or argument buffer's sampler at an implicit level or gradients, whose texel the
    /// device samples where the word states a nearest mip filter. (Where the word filters, the
    /// device's own comparison reads the texels, `DeviceDepthComparison`.)
    pub fn fetches_every_texel(&self, op: &SampleOp) -> bool {
        let sample = matches!(op, SampleOp::Sample { .. });
        let leveled = self
            .fetch
            .as_ref()
            .is_some_and(|f| matches!(f.level, super::fetched::FetchLevel::Word(_)));
        self.fetch.is_some()
            && (leveled
                || self.blends()
                || match self.format {
                    FormatBits::Slot(_) => !self.keeps_reference(op),
                    FormatBits::Heap(_) => !(sample && self.function == Comparison::Word),
                })
    }

    /// The texture slot whose `TexelFormat` specialization says how the texels compare, where it
    /// is one.
    fn slot(&self) -> Option<u32> {
        match self.format {
            FormatBits::Slot(n) => Some(n),
            FormatBits::Heap(_) => None,
        }
    }
}

/// `op` with its comparison taken out where the shader makes it alone (module docs), and that
/// comparison; else `op` as it is. `stat` is the static sampler (none for an argument buffer's),
/// `image` and `shape` the texture's, and `level` the call's level or bias operand. A runtime
/// sampler's sample of a slot and every comparison of a heap texture keep their reference
/// ([`Compared::keeps_reference`]).
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    op: SampleOp,
    stat: Option<(SamplerRef, Coordinates)>,
    (image, shape): (ImageRef, ImageShape),
    level: Option<crate::air::ir::Value>,
) -> Result<(SampleOp, Option<Compared>), Error> {
    let dref = match &op {
        SampleOp::Sample { dref: Some(r), .. } | SampleOp::Gather { dref: Some(r), .. } => *r,
        _ => return Ok((op, None)),
    };
    if shape.dim != ImageDim::D2 || !shape.depth || shape.ms {
        return Ok((op, None));
    }
    let format = match image {
        ImageRef::Slot(texture) => FormatBits::Slot(texture),
        ImageRef::Heap { handle, .. } => FormatBits::Heap(handle),
    };
    // A static sampler, or an argument buffer's (a heap texture's alone), by its word.
    let sampler = match (stat, format) {
        (Some((sampler, _)), _) => Some(sampler),
        (None, FormatBits::Heap(_)) => None,
        (None, FormatBits::Slot(_)) => return Ok((op, None)),
    };
    let heap = matches!(format, FormatBits::Heap(_));
    let sample = matches!(op, SampleOp::Sample { .. });
    let state = match sampler {
        Some(SamplerRef::Constexpr(words)) => Some(super::state_of(words)?),
        Some(SamplerRef::Slot(_)) | None => None,
    };
    // A dynamic bias's second sample stays the device's comparison where the op keeps its
    // reference.
    let biased = matches!(
        &op,
        SampleOp::Sample {
            lod: Lod::Implicit {
                reach: BiasReach::Either { .. },
                ..
            },
            ..
        }
    );
    // A slot's sample through nearest filters under a linear mip filter blends two levels'
    // comparisons, where Metal's level is known (`fetched::FetchLevel::Blend`).
    let blend = |s: &crate::air::abi::SamplerState| {
        !heap
            && sample
            && s.min_filter == Filter::Nearest
            && s.mag_filter == Filter::Nearest
            && s.mip_filter == MipFilter::Linear
            && s.max_anisotropy <= 1
    };
    let function = match &state {
        // The frontend writes an unspecified compare function as `never`, and no `None`.
        Some(s) => match s.compare {
            CompareFunction::None => return Ok((op, None)),
            _ if sample && !super::linear::unfiltered(s) && !blend(s) => return Ok((op, None)),
            _ if heap && biased => return Ok((op, None)),
            f => Comparison::Function(f),
        },
        None if biased => return Ok((op, None)),
        None => Comparison::Word,
    };
    if cx.profile.device_samples() {
        cx.families.sampler.metal_sampler = true;
        return Ok((op, None));
    }
    let blends = state.as_ref().is_some_and(blend);
    let op = match op {
        // A heap texture's class is known at run time, as is a runtime sampler's filter, and a
        // blend's device arm compares a format whose class is not measured: the op keeps its
        // reference for the device's comparison.
        _ if heap || blends => op,
        SampleOp::Sample { .. } if state.is_none() => op,
        SampleOp::Sample {
            lod,
            min_lod,
            offset,
            ..
        } => SampleOp::Sample {
            lod,
            dref: None,
            min_lod,
            offset,
        },
        SampleOp::Gather {
            component, offset, ..
        } => SampleOp::Gather {
            component,
            dref: None,
            offset,
        },
        other => other,
    };
    // A heap texture's depths are fetched by the class its handle carries (`heap`), an argument
    // buffer's sampler's addressed by its table entry's word.
    let fetch = match (&state, sampler) {
        (Some(s), _) => super::fetched::plan(cx, &op, s, &[s.address_s, s.address_t], level)?,
        (None, Some(SamplerRef::Slot(n))) => super::fetched::plan_word(
            &op,
            FetchAddress::Word {
                sampler: n,
                axes: 2,
            },
        ),
        (None, None) if heap => super::fetched::plan_word(&op, FetchAddress::Entry { axes: 2 }),
        (None, _) => None,
    };
    // A blend whose level the shader does not know (an implicit level or gradients, a
    // `min_lod_clamp`, a dynamic offset, a level clamp the snap does not reach) is the device's.
    let blended = fetch
        .as_ref()
        .is_some_and(|f| matches!(f.level, super::fetched::FetchLevel::Blend(..)));
    if blends && !blended {
        return Ok((op, None));
    }
    let compared = Compared {
        function,
        dref,
        format,
        fetch,
        plain: None,
    };
    Ok((op, Some(compared)))
}

/// The passes of `c` over the depths of `s` (a gather's four, a `vec4`, or a sample's one) that
/// the shader fetches (`c.fetch`, module docs, [`super::fetched::branch`]) where the slot's
/// `TexelFormat` specialization states a measured format, else that `device`'s access of the
/// combined binding's sampled image reads (a `vec4` of the view's sampled type; a sample's first
/// component is its depth).
pub(super) fn fetched(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (c, f): (&Compared, &Fetch),
    image: CombinedImage,
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
) -> BResult<SValue> {
    let gather = matches!(s.op, SampleOp::Gather { .. });
    let (Source::Combined(key), Some(texture)) = (s.source, c.slot()) else {
        return Err(cx.fail("fetched texels of a heap element"));
    };
    let read = |fb: &mut FunctionBuilder<'_>, si: SValue| {
        let x = device(fb, si)?;
        match gather {
            true => Ok(x),
            false => fb.composite_extract(x, &[0]),
        }
    };
    let texel = super::fetched::branch(
        cx,
        fb,
        (s, key),
        (texture, f, MEASURED, false),
        image,
        &depths,
        &read,
    )?;
    emit(cx, fb, s, c, texel)
}

/// The depths of `texels`, each a `vec4` whose first component is its depth: a sample's one, or a
/// gather's four as a `vec4`.
fn depths(fb: &mut FunctionBuilder<'_>, texels: &[SValue]) -> BResult<SValue> {
    let vec4 = Tys::new(fb)?.vec4;
    let mut depths = Vec::with_capacity(texels.len());
    for &texel in texels {
        depths.push(fb.composite_extract(texel, &[0])?);
    }
    match depths[..] {
        [d] => Ok(d),
        _ => fb.composite_construct(vec4, &depths),
    }
}

/// The held value of runtime sampler sample `s` of combined image `image`, which compares as `c`
/// says ([`Compared::keeps_reference`], module docs): a branch on whether the slot's word states a
/// sampler that reads one texel (nearest minification and magnification, no linear mip filter,
/// not anisotropic). There the depth, compared in the shader: where `c.fetch` is planned, the
/// texel the shader fetches through the slot's binding without comparison (`c.plain`) where the
/// slot's format is measured ([`super::fetched::branch`]), of the level the word reads at the
/// sample's level (`fetched::FetchLevel::Word`), or, at an implicit level or gradients, of level
/// 0 where the word states no mip filter; else the first component of `plain`'s sample of that
/// binding's sampled image; elsewhere `filtered`, the filtered comparison's held value. Each arm
/// makes the sampled images it reads.
pub(super) fn runtime_sample<'a, 'e>(
    cx: &mut EmitCx<'e>,
    fb: &mut FunctionBuilder<'_>,
    (s, c, image): (&'a Sampling, &'a Compared, CombinedImage),
    plain: &'a dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
    filtered: ArmFn<'a, EmitCx<'e>>,
) -> BResult<SValue> {
    let (Some(Source::Combined(key)), Some(texture), Some(form)) = (c.plain, c.slot(), s.texel)
    else {
        return Err(cx.fail("a runtime sampler's comparison without its plain binding"));
    };
    let Some(&(binding, _)) = cx.decl.families.sampler.combined.get(&key) else {
        return Err(cx.fail(format!("combined binding {key:?} is not declared")));
    };
    let SamplerRef::Slot(n) = key.sampler else {
        return Err(cx.fail("a branched comparison through a constexpr sampler"));
    };
    let word = super::emit::sampler_word(cx, fb, Pixel::Spec(n))?;
    let t = Tys::new(fb)?;
    let (one, mip) = one_texel(fb, &t, word)?;
    // The one texel of the level the word reads, which the shader fetches; at an implicit level
    // or gradients, only under `MTLSamplerMipFilterNotMipmapped`, level 0's.
    let unmipped = match &c.fetch {
        Some(f) if matches!(f.level, super::fetched::FetchLevel::Word(_)) => Some((f, None)),
        Some(f) => {
            let zero = scalar(fb, t.u32_, 0)?;
            let zero = value(fb, zero)?;
            Some((f, Some(fb.binary(BinOp::IEqual, mip, zero)?)))
        }
        None => None,
    };
    let texel = image.texel;
    let compared: ArmFn<'a, EmitCx<'e>> = Box::new(move |fb, cx| {
        let sampled = |fb: &mut FunctionBuilder<'_>, si: SValue| -> BResult<SValue> {
            let x = plain(fb, si)?;
            fb.composite_extract(x, &[0])
        };
        let depth = match unmipped {
            Some((f, None)) => super::fetched::branch(
                cx,
                fb,
                (s, key),
                (texture, f, MEASURED, false),
                image,
                &depths,
                &sampled,
            )?,
            Some((f, Some(level0))) => {
                let fetch: ArmFn<'_, EmitCx<'e>> = Box::new(|fb, cx| {
                    let d = super::fetched::branch(
                        cx,
                        fb,
                        (s, key),
                        (texture, f, MEASURED, false),
                        image,
                        &depths,
                        &sampled,
                    )?;
                    Ok(ArmEnd::FallThrough(vec![d]))
                });
                let read: ArmFn<'_, EmitCx<'e>> = Box::new(|fb, _| {
                    let si = binding.sampled_image(fb)?;
                    Ok(ArmEnd::FallThrough(vec![sampled(fb, si)?]))
                });
                let Merged(merged) = fb.if_(cx, level0, fetch, Some(read))?;
                match merged[..] {
                    [d] => d,
                    _ => return Err(cx.fail("a fetched comparison's depth without its value")),
                }
            }
            None => {
                let si = binding.sampled_image(fb)?;
                sampled(fb, si)?
            }
        };
        let pass = emit(cx, fb, s, c, depth)?;
        let pass = crate::lower::texture::held_texel(fb, texel, form, pass)?;
        Ok(ArmEnd::FallThrough(vec![pass]))
    });
    let Merged(merged) = fb.if_(cx, one, compared, Some(filtered))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a runtime sampler's comparison without its value")),
    }
}

/// The held value of sample `s` of combined image `image` through a constexpr sampler of nearest
/// filters under a linear mip filter, which compares as `c` says ([`Compared::blends`], module
/// docs): where the slot's `TexelFormat` states a measured class, Metal's blend of two levels'
/// comparisons. With `L + q` the level the sampler's snap hands the device (`level::LevelSnap::of`)
/// clamped into the texture's levels, the shader fetches the texel of level `L` and of level `L +
/// 1` (clamped into the levels) through the slot's binding without comparison (`c.plain`), compares
/// each, and blends the passes by `1 - q` and `q` as Metal does ([`blend_passes`]). Elsewhere
/// `device`'s comparison of the binding's sampled image. Each arm makes the sampled images it
/// reads.
pub(super) fn blended(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (s, c, image): (&Sampling, &Compared, CombinedImage),
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
) -> BResult<SValue> {
    use super::fetched::FetchLevel;
    let (Some(Source::Combined(key)), Some(texture), Some(f), Some(form)) =
        (c.plain, c.slot(), c.fetch.as_ref(), s.texel)
    else {
        return Err(cx.fail("a blended comparison without its plain binding or its fetch"));
    };
    let FetchLevel::Blend(snap, operand) = f.level else {
        return Err(cx.fail("a blended comparison without its levels"));
    };
    let Some(spec) = crate::lower::texture::texel_format_const(cx, texture) else {
        return Err(cx.fail(format!(
            "texture slot {texture} has no `TexelFormat` specialization"
        )));
    };
    let Some(&(binding, _)) = cx.decl.families.sampler.combined.get(&key) else {
        return Err(cx.fail(format!("combined binding {key:?} is not declared")));
    };
    let Source::Combined(own) = s.source else {
        return Err(cx.fail("a blended comparison of a heap element"));
    };
    let Some(&(compares, _)) = cx.decl.families.sampler.combined.get(&own) else {
        return Err(cx.fail(format!("combined binding {own:?} is not declared")));
    };
    let t = Tys::new(fb)?;
    let measured = super::bicubic::flag(fb, &t, spec, MEASURED)?;
    let texel = image.texel;
    let t = &t;
    let fetch: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let si = binding.sampled_image(fb)?;
        let x = match operand {
            Some(v) => cx.operand(fb, v)?,
            None => float(fb, 0.0)?,
        };
        // `L + q`, clamped into the texture's levels: a whole last level leaves `q` 0.
        let lambda = snap.emit(fb, x)?;
        let img = fb.image(si)?;
        let levels = fb.query(ImageQuery::Levels, img)?;
        let one = scalar(fb, t.u32_, 1)?;
        let one = value(fb, one)?;
        let last = fb.binary(BinOp::ISub, levels, one)?;
        let f32_ = fb.module().ty(SType::Float { width: 32 })?;
        let last_f = fb.convert(ConvOp::ConvertUToF, last, f32_)?;
        let lambda = fb.glsl(GlslOp::FMin, &[lambda, last_f])?;
        let whole = fb.glsl(GlslOp::Floor, &[lambda])?;
        let q = fb.binary_no_contraction(BinOp::FSub, lambda, whole)?;
        let l0 = fb.convert(ConvOp::ConvertFToU, whole, t.u32_)?;
        let l1 = fb.binary(BinOp::IAdd, l0, one)?;
        let l1 = fb.glsl(GlslOp::UMin, &[l1, last])?;
        let dref = cx.operand(fb, c.dref)?;
        let test = TapCompare::of(cx, fb, t, (c.function, None), (dref, Bits::Spec(spec)))?;
        let mut passes = [q; 2];
        for (pass, level) in passes.iter_mut().zip([l0, l1]) {
            let Some(taps) = f.taps(s, Some(level), None) else {
                return Err(cx.fail("a blended comparison's taps"));
            };
            let x = super::fetched::texels(cx, fb, t, s, (si, image), Bits::Spec(spec), &taps)?;
            let depth = depths(fb, &x)?;
            *pass = test.pass(fb, depth)?;
        }
        let blend = blend_passes(fb, t, passes, q)?;
        let held = crate::lower::texture::held_texel(fb, texel, form, blend)?;
        Ok(ArmEnd::FallThrough(vec![held]))
    });
    let read: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let si = compares.sampled_image(fb)?;
        let x = device(fb, si)?;
        let held = crate::lower::texture::held_texel(fb, texel, form, x)?;
        Ok(ArmEnd::FallThrough(vec![held]))
    });
    let Merged(merged) = fb.if_(cx, measured, fetch, Some(read))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a blended comparison without its value")),
    }
}

/// Metal's blend of two levels' passes `[p0, p1]` (each `0.0` or `1.0`) by `1 - q` and `q` (`q`
/// on sixty-fourths, `m / 64`), as `blended` takes it: an `r8Unorm` texture's blend of the passes
/// as codes 0 and 255, as a linear comparison sums its taps' (`linear::compare`), its sum `255 (p0
/// (64 - m) + p1 m) / 64` rounded once onto sixteenths of a code, ties up, `n = (255 s + 2) /
/// 4` of `s = p0 (64 - m) + p1 m`, and `n / 4080` correctly rounded to `f32`.
fn blend_passes(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    [p0, p1]: [SValue; 2],
    q: SValue,
) -> BResult<SValue> {
    let sixty_four = float(fb, 64.0)?;
    let m = fb.binary_no_contraction(BinOp::FMul, q, sixty_four)?;
    let rest = fb.binary_no_contraction(BinOp::FSub, sixty_four, m)?;
    let a = fb.binary_no_contraction(BinOp::FMul, p0, rest)?;
    let b = fb.binary_no_contraction(BinOp::FMul, p1, m)?;
    let sum = fb.binary_no_contraction(BinOp::FAdd, a, b)?;
    let sum = fb.convert(ConvOp::ConvertFToU, sum, t.u32_)?;
    let u = |fb: &mut FunctionBuilder<'_>, v: u64| -> BResult<SValue> {
        let c = scalar(fb, t.u32_, v)?;
        value(fb, c)
    };
    let (code, two, grid) = (u(fb, 255)?, u(fb, 2)?, u(fb, 4080)?);
    let n = fb.binary(BinOp::IMul, sum, code)?;
    let n = fb.binary(BinOp::IAdd, n, two)?;
    let n = fb.binary(BinOp::ShiftRightLogical, n, two)?;
    let form = Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    };
    crate::lower::arith::texel_round::ratio_to_f32(fb, form, n, grid)
}

/// Whether sampler word `word` (a `u32`) states a sampler that reads one texel (nearest
/// minification and magnification, no linear mip filter, not anisotropic), and the word's mip
/// filter field.
pub(super) fn one_texel(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    word: SValue,
) -> BResult<(SValue, SValue)> {
    let filters = scalar(
        fb,
        t.u32_,
        u64::from(SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR | SamplerWord::ANISOTROPIC),
    )?;
    let filters = value(fb, filters)?;
    let zero = scalar(fb, t.u32_, 0)?;
    let zero = value(fb, zero)?;
    let filters = fb.binary(BinOp::BitwiseAnd, word, filters)?;
    let nearest = fb.binary(BinOp::IEqual, filters, zero)?;
    let mip = super::bicubic::word_field(
        fb,
        t,
        word,
        SamplerWord::MIP_FILTER_SHIFT,
        SamplerWord::MIP_FILTER >> SamplerWord::MIP_FILTER_SHIFT,
    )?;
    // `MTLSamplerMipFilterLinear`.
    let linear = scalar(fb, t.u32_, 2)?;
    let linear = value(fb, linear)?;
    let one_level = fb.binary(BinOp::INotEqual, mip, linear)?;
    Ok((fb.binary(BinOp::LogicalAnd, nearest, one_level)?, mip))
}

/// The held value of heap comparison `s` (module docs, [`Compared::keeps_reference`]): `device`,
/// the device's comparison of the element (read through `s.source`'s sampler with comparison),
/// where the shader does not compare; else the passes of `c` over the element's depths (a
/// gather's four as a `vec4`, a sample's one), compared by the class the handle carries. The
/// depths are the ones the shader fetches from the sampled image of `c.plain` (the same element
/// through its use's sampler without comparison) where `c.fetch` is planned, as a slot of the
/// class's format would ([`super::fetched::texels`]); else, and for a runtime or argument
/// buffer's sampler's sample whose word states a nearest mip filter, the ones `plain` reads of
/// it. The shader compares where the handle states a class and, for such a sample, the word
/// states a sampler that reads one texel. Every read runs, so no branch puts one in divergent
/// flow.
pub(super) fn heap(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (s, c): (&Sampling, &Compared),
    device: SValue,
    plain: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
) -> BResult<SValue> {
    let (FormatBits::Heap(handle), Some(source)) = (c.format, c.plain) else {
        return Err(cx.fail("a heap comparison without its handle or its plain read"));
    };
    let (si, image, _) = super::emit::sampled_image(cx, fb, source)?;
    let t = Tys::new(fb)?;
    let h = cx.operand(fb, handle)?;
    let zero = scalar(fb, t.u32_, 0)?;
    let zero = value(fb, zero)?;
    let mut flags = [zero; 2];
    for (flag, bit) in flags
        .iter_mut()
        .zip([HANDLE_DEPTH_CODES, HANDLE_DEPTH_VALUES])
    {
        let bit = scalar(fb, t.u32_, u64::from(bit))?;
        let bit = value(fb, bit)?;
        let carried = fb.binary(BinOp::BitwiseAnd, h, bit)?;
        *flag = fb.binary(BinOp::INotEqual, carried, zero)?;
    }
    let [codes, values] = flags;
    // The class's texels compare, and a border tap reads, as a slot bound in its format would:
    // `Depth16Unorm`'s `TexelFormat` (codes) or `Depth32Float`'s (values).
    let mut bits = zero;
    for (class, format) in [
        (values, PixelFormat::Depth32Float),
        (codes, PixelFormat::Depth16Unorm),
    ] {
        let b = scalar(fb, t.u32_, u64::from(texel_format(format).0))?;
        let b = value(fb, b)?;
        bits = fb.select(class, b, bits)?;
    }
    let word = match c.function {
        Comparison::Function(_) => None,
        Comparison::Word => Some(word_of(cx, fb, s)?),
    };
    let one = match (word, &s.op) {
        (Some(w), SampleOp::Sample { .. }) => Some(one_texel(fb, &t, w)?),
        _ => None,
    };
    let texel = match &c.fetch {
        None => plain(fb, si)?,
        Some(f) => {
            let level = super::fetched::level(cx, fb, &t, s, (f.level, word), si)?;
            let Some(taps) = f.taps(s, level, word) else {
                return Err(cx.fail("a word-addressed heap fetch without its word"));
            };
            let x = super::fetched::texels(cx, fb, &t, s, (si, image), Bits::Value(bits), &taps)?;
            let fetched = depths(fb, &x)?;
            match one {
                // The level the word reads (`fetched::FetchLevel::Word`).
                Some(_) if matches!(f.level, super::fetched::FetchLevel::Word(_)) => fetched,
                // At an implicit level or gradients, level 0's texel where the word states no mip
                // filter; under a nearest one, whose level the function does not know, the
                // sample's.
                Some((_, mip)) => {
                    let read = plain(fb, si)?;
                    let none = fb.binary(BinOp::IEqual, mip, zero)?;
                    fb.select(none, fetched, read)?
                }
                None => fetched,
            }
        }
    };
    let passes = passes(cx, fb, (s, c), (texel, word), Bits::Value(bits))?;
    let class = fb.binary(BinOp::LogicalOr, codes, values)?;
    let take = match one {
        Some((one, _)) => fb.binary(BinOp::LogicalAnd, class, one)?,
        None => class,
    };
    fb.select(take, passes, device)
}

/// The passes of `c` over `texel`, the depths the device read for `s`: a gather's four (a
/// `vec4`), or a sample's one. The slot's `TexelFormat` specialization says how they compare.
pub(super) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    c: &Compared,
    texel: SValue,
) -> BResult<SValue> {
    let Some(texture) = c.slot() else {
        return Err(cx.fail("a heap comparison past its own path"));
    };
    let Some(spec) = crate::lower::texture::texel_format_const(cx, texture) else {
        return Err(cx.fail(format!(
            "texture slot {texture} has no `TexelFormat` specialization"
        )));
    };
    let word = match c.function {
        Comparison::Function(_) => None,
        Comparison::Word => Some(word_of(cx, fb, s)?),
    };
    passes(cx, fb, (s, c), (texel, word), Bits::Spec(spec))
}

/// The word of the runtime or argument buffer's sampler of `s` (a `u32`): its slot's
/// `SamplerState` specialization, or its table entry's flags.
fn word_of(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, s: &Sampling) -> BResult<SValue> {
    match super::emit::word_pixel(s) {
        Some(pixel) => super::emit::sampler_word(cx, fb, pixel),
        None => Err(cx.fail("a runtime sampler's comparison without its word")),
    }
}

/// The passes of `c` over `texel` (a gather's four depths, a `vec4`, or a sample's one) of a
/// texture bound in the format whose treatment is `bits`, under a runtime or argument buffer's
/// sampler's word `word` where `c` compares by it.
fn passes(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (s, c): (&Sampling, &Compared),
    (texel, word): (SValue, Option<SValue>),
    bits: Bits,
) -> BResult<SValue> {
    let t = Tys::new(fb)?;
    let dref = cx.operand(fb, c.dref)?;
    let test = TapCompare::of(cx, fb, &t, (c.function, word), (dref, bits))?;
    match s.op {
        SampleOp::Gather { .. } => {
            let mut passes = [texel; 4];
            for (k, pass) in passes.iter_mut().enumerate() {
                let depth = fb.composite_extract(texel, &[k as u32])?;
                *pass = test.pass(fb, depth)?;
            }
            fb.composite_construct(t.vec4, &passes)
        }
        _ => test.pass(fb, texel),
    }
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
