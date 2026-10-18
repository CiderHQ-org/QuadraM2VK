//! The device's own linear filter of an 8-bit unsigned normalized texture, at Metal's footprint
//! step.
//!
//! **What it replaces.** Metal's linear filter is emulated by fetches wherever the bound
//! format's sum is measured (`linear`): exact, at about 574 instructions per sample site, the bulk
//! of a large fragment function's size and of a driver's time to compile it (the breakdown). On a
//! device whose own filter of an 8-bit unsigned normalized texture lands within one code of Metal's
//! once it reads Metal's footprint step, the module hands that class to the device instead, a
//! degrade the owner accepted on 2026-10-10 ("I accept the one color step deviation"), recorded in
//! each artifact and announced with its bound (`PrecisionNote::DeviceUnorm8LinearFilter`).
//!
//! **Metal's footprint step** (`position`'s module docs). Metal reads the texel-space position `X`
//! of a coordinate (wrapped into one period first on a repeating or mirrored axis, then scaled,
//! `position::position`) as `q = floor(256 (X - 1/2) + 1/2)`: the taps `q >> 8` and the one after
//! it, weighed by the step `q & 255` in 256ths. A device reads the position its own way (Vulkan
//! bounds it only by `subTexelPrecisionBits`), so its weights may differ from Metal's by a step.
//! The module hands the device the coordinate of the centre of Metal's step, `X' = q / 256 + 1/2`
//! (`position::AxisSnap::Cell`'s), on each plane axis: a device with eight subtexel bits then reads
//! Metal's taps and weights, and only its blend of them is its own. The position is the cheap one
//! ([`cell`]): the coordinate wrapped into its period by `floor` (a repeating axis's `floor(u)`, a
//! mirrored one's `2 floor(u / 2)`, a constexpr sampler's by its address modes, a runtime sampler
//! slot's by the modes its word states) and scaled, rounded to nearest. Metal rounds the wrap of
//! `-1 < u < 0` up and the product's ties away from zero (`position::position`); where that moves
//! `X` across a step's edge (a coordinate within an `f32` unit of it) the device reads the
//! neighbouring step, a weight 1/256 off. The bound below is measured with this position (the
//! probe's `cell` variant).
//!
//! **The class** ([`CLASSES`]): the slot's `TexelFormat` specialization masked by [`CLASS_MASK`] is
//! `EXACT | UNORM` with the channels of `R8Unorm`, `RG8Unorm`, or `RGBA8Unorm` and `BGRA8Unorm`,
//! the formats the deviation was measured on. Every other class keeps its path: the 16-bit, signed,
//! sRGB, packed, two-plane, float and half-float classes, and `A8Unorm` and `RGBX8Unorm`, which
//! share the family but were not measured.
//!
//! **The reach** ([`planned`]): a sample with a `float` or `half` result of a 2D texture or 2D
//! array slot, without comparison, offset, or pixel coordinates, through a constexpr sampler the
//! emulation reaches (linear filters, no mip filter, no anisotropy) or a runtime sampler slot
//! whose word states such a filter with normalized coordinates and no `MirrorClampToEdge` axis,
//! on a profile that takes `DriverDeviation::Unorm8LinearFilter` (`device::driver_facts`): a
//! device measured within one code, under a policy of `Unorm8LinearFilterPolicy::
//! DeviceWithinOneCode`. An unmeasured device, and every device under the default policy, keep
//! the emulation, exact (on the consumer's fragments this module's arm bought no GPU time and cost
//! module size and making time). A 1D or 3D texture, a cube, a heap element, a comparison and every
//! other sample keep theirs.
//!
//! **The branch** (`linear::emit`, `linear::emit_shared`): the emulation (its routine's call) runs
//! where the class is measured and the native condition ([`coordinate`]'s) does not hold, the
//! device's sample at the selected coordinate elsewhere. The class is data the function reads from
//! its push field, so both arms stay in the module and the branch runs: the native class's gain is
//! GPU time per sample, not words, and a runtime sampler's word tests are specializations as
//! before.
//!
//! **The bound** (the deviation probe, `probes/shb-dev` at the scratch commit `aa944573d` with its
//! `cell` variant, this module's coordinate; our probes compared: 19x13 textures of
//! the four formats, 4,096 coordinates, a third of them near the 1/256 grid, under clamp to edge,
//! repeat, mirrored repeat and clamp to zero, against an Apple M4 on macOS 27.0.1): the largest
//! difference of any channel of a `float` result, in codes (1/255), was 0.849 on NVIDIA's 615.71.09
//! (RTX 3090 and RTX 5080 Laptop GPU alike, its blend's own rounding), 0.031 on RADV and ANV (Mesa
//! 26.2.4 on the Radeon 780M, the Iris Xe of RPL-P and the UHD 770 of RPL-S), and just under 1 on
//! lavapipe (Mesa 26.2.4, LLVM 23.1.1, whose filter answers whole codes); none reached one code. A
//! `half` result is the device's `float` rounded as the class's conversion rounds it: 0.809, 0.125
//! and 0.996 codes on the same lanes. At the coordinate as given the same devices missed by up to
//! 0.919 (NVIDIA), 0.427 (RADV), 0.918 (ANV) and 1.000 (lavapipe).

use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::error::DriverDeviation;
use crate::lower::emit::EmitCx;
use crate::lower::select::{both, NoteClause, NoteOn, NoteTest, SelectCx};
use crate::metal::bicubic::BicubicTexels as B;
use crate::metal::sampler::SamplerWord;
use crate::metal::texel_format::TexelFormat;
use crate::metal::SamplerAddressMode;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConvOp, GlslOp, ImageQuery, Type as SType};

use super::bicubic::{scalar, value, word_field, Bits, TapAddress, Tys};
use super::linear::{Linear, LinearImage, LinearState};
use super::Sampling;

/// The class bits the native class test reads of a slot's `TexelFormat` specialization: the
/// family's bits, the channels present, and the packed field.
pub(crate) const CLASS_MASK: u32 = B::LINEAR
    | B::UNORM
    | B::SNORM
    | B::HALF
    | 0xF << B::PRESENT
    | B::EXACT
    | B::WIDE16
    | B::SRGB
    | B::EXTENDED
    | TexelFormat::PACKED_MASK;

/// The native classes, masked by [`CLASS_MASK`]: `R8Unorm`'s, `RG8Unorm`'s, and `RGBA8Unorm`'s
/// and `BGRA8Unorm`'s (module docs).
pub(crate) const CLASSES: [u32; 3] = [
    B::EXACT | B::UNORM | 0b0001 << B::PRESENT,
    B::EXACT | B::UNORM | 0b0011 << B::PRESENT,
    B::EXACT | B::UNORM | 0b1111 << B::PRESENT,
];

/// The census words [`coordinate`] adds to a linear sample: the class test, a runtime sampler's
/// word tests, the size query, per plane axis its extractions, its period's start under each mode
/// and their selection, and its cell, and the composite, the layer and the select, with their
/// constants.
pub(super) const WORDS: u32 = 2 * 256 + 512;

/// Whether the sample emulation `l` plans (of an image of shape `shape`) takes the device's
/// filter on the native classes (module docs): the profile takes
/// `DriverDeviation::Unorm8LinearFilter`, and the sample is a 2D texture or 2D array slot's,
/// without comparison, texel offset, or pixel coordinates, through a constexpr sampler or a
/// runtime sampler slot. The module then requires `VkProperty::Unorm8LinearFilter`.
pub(super) fn planned(cx: &mut SelectCx<'_>, l: &Linear, shape: ImageShape) -> bool {
    let reach = shape.dim == ImageDim::D2
        && !shape.ms
        && !shape.depth
        && l.cube.is_none()
        && l.compare.is_none()
        && !l.raw
        && !l.exact_only
        && matches!(l.image, LinearImage::Slot(_))
        && match &l.state {
            LinearState::Constant { pixel, .. } => !pixel,
            LinearState::Slot { .. } => true,
            LinearState::Argument { .. } => false,
        }
        && l.offset
            .as_deref()
            .is_none_or(|o| o.iter().all(|k| *k == 0));
    let native = reach
        && cx
            .profile
            .takes_driver_deviation(DriverDeviation::Unorm8LinearFilter);
    if native {
        cx.families.sampler.native_filter = true;
    }
    native
}

/// Where the native arm of emulation `l` runs: the slot's class is a native one ([`CLASSES`]) and,
/// through a runtime sampler slot, its word states the linear filter Metal's rule answers
/// (`SamplerWord::LINEAR_FILTER`), normalized coordinates, and no `MirrorClampToEdge` on either
/// plane axis. Empty where `l` takes no native arm.
pub(super) fn when(l: &Linear) -> Vec<NoteClause> {
    let LinearImage::Slot(texture) = l.image else {
        return Vec::new();
    };
    if !l.native || l.raw {
        return Vec::new();
    }
    let class: Vec<NoteClause> = CLASSES
        .iter()
        .map(|&c| [NoteTest::equal(NoteOn::Texel(texture), CLASS_MASK, c)].into())
        .collect();
    match &l.state {
        LinearState::Constant { .. } => class,
        LinearState::Slot { sampler, .. } => {
            let on = NoteOn::Sampler(*sampler);
            let clamp = SamplerAddressMode::MirrorClampToEdge.as_raw();
            let [s, t, _] = SamplerWord::ADDRESS_SHIFT;
            let word: NoteClause = [
                NoteTest::equal(on, SamplerWord::LINEAR_FILTER, SamplerWord::LINEAR),
                NoteTest::equal(on, SamplerWord::PIXEL_COORDINATES, 0),
                NoteTest::differ(on, 0b111 << s, clamp << s),
                NoteTest::differ(on, 0b111 << t, clamp << t),
            ]
            .into();
            both(&class, &[word])
        }
        LinearState::Argument { .. } => Vec::new(),
    }
}

/// The base level's size of sample `s`'s combined binding (a `u32` vector, its plane axes first),
/// from a sampled image of its own in the open block, and the slot's `TexelFormat`
/// specialization: what [`coordinate`] reads of a slot's own sample.
pub(super) fn slot_facts(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    l: &Linear,
) -> BResult<Option<(SValue, Bits)>> {
    let (true, false, LinearImage::Slot(texture)) = (l.native, l.raw, l.image) else {
        return Ok(None);
    };
    let Some(spec) = crate::lower::texture::texel_format_const(cx, texture) else {
        return Err(cx.fail(format!(
            "texture slot {texture} has no `TexelFormat` specialization"
        )));
    };
    let (si, _, _) = super::emit::sampled_image(cx, fb, s.source)?;
    let img = fb.image(si)?;
    let zero = fb.u32_const(0)?;
    let zero = fb.module().const_value(zero)?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    Ok(Some((size, Bits::Spec(spec))))
}

/// The coordinate the device's arm of emulation `l` samples at, and the native condition (a
/// `bool`): where the class `bits` is native ([`CLASSES`]) and, through a runtime sampler slot,
/// its word states a native sampler ([`when`]), the plane coordinate `plane` (normalized, two
/// lanes) snapped onto the centre of Metal's footprint step on the base level's size `size`
/// (module docs), with `s`'s layer; elsewhere `coord`, the coordinate the device reads otherwise.
/// `None` where `l` takes no native arm.
pub(super) fn coordinate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    l: &Linear,
    (plane, coord): (SValue, SValue),
    (size, bits): (SValue, Bits),
) -> BResult<Option<(SValue, SValue)>> {
    if !l.native || l.raw {
        return Ok(None);
    }
    let t = Tys::new(fb)?;
    let u32_value = |fb: &mut FunctionBuilder<'_>, x: u32| -> BResult<SValue> {
        let c = scalar(fb, t.u32_, u64::from(x))?;
        value(fb, c)
    };
    let class = bits.value(fb)?;
    let mask = u32_value(fb, CLASS_MASK)?;
    let class = fb.binary(BinOp::BitwiseAnd, class, mask)?;
    let mut native = None;
    for c in CLASSES {
        let c = u32_value(fb, c)?;
        let is = fb.binary(BinOp::IEqual, class, c)?;
        native = Some(match native {
            Some(n) => fb.binary(BinOp::LogicalOr, n, is)?,
            None => is,
        });
    }
    let Some(mut native) = native else {
        return Err(cx.fail("a native linear sample without its classes"));
    };
    let word = match &l.state {
        LinearState::Constant { .. } => None,
        LinearState::Slot { sampler, .. } => {
            let word = super::emit::sampler_word(cx, fb, super::Pixel::Spec(*sampler))?;
            let filter = u32_value(fb, SamplerWord::LINEAR_FILTER)?;
            let filter = fb.binary(BinOp::BitwiseAnd, word, filter)?;
            let linear = u32_value(fb, SamplerWord::LINEAR)?;
            let linear = fb.binary(BinOp::IEqual, filter, linear)?;
            native = fb.binary(BinOp::LogicalAnd, native, linear)?;
            let pixel = u32_value(fb, SamplerWord::PIXEL_COORDINATES)?;
            let pixel = fb.binary(BinOp::BitwiseAnd, word, pixel)?;
            let zero = u32_value(fb, 0)?;
            let normalized = fb.binary(BinOp::IEqual, pixel, zero)?;
            native = fb.binary(BinOp::LogicalAnd, native, normalized)?;
            let clamp = u32_value(fb, SamplerAddressMode::MirrorClampToEdge.as_raw())?;
            for shift in SamplerWord::ADDRESS_SHIFT.into_iter().take(2) {
                let mode = word_field(fb, &t, word, shift, 0b111)?;
                let other = fb.binary(BinOp::INotEqual, mode, clamp)?;
                native = fb.binary(BinOp::LogicalAnd, native, other)?;
            }
            Some(word)
        }
        LinearState::Argument { .. } => return Ok(None),
    };
    let mut parts = Vec::with_capacity(2);
    for (a, shift) in (0..2u32).zip(SamplerWord::ADDRESS_SHIFT) {
        let u = fb.composite_extract(plane, &[a])?;
        let n = fb.composite_extract(size, &[a])?;
        let start = match (&l.state, word) {
            (LinearState::Constant { address, .. }, _) => match address.get(a as usize) {
                Some(TapAddress::Repeat) => Some(fb.glsl(GlslOp::Floor, &[u])?),
                Some(TapAddress::Mirror) => Some(mirror_start(fb, &t, u)?),
                Some(TapAddress::Edge | TapAddress::Border) => None,
                None => return Err(cx.fail("a native linear sample without its axis's address")),
            },
            (_, Some(word)) => {
                let mode = word_field(fb, &t, word, shift, 0b111)?;
                let floor = fb.glsl(GlslOp::Floor, &[u])?;
                let twice = mirror_start(fb, &t, u)?;
                let none = f32_value(fb, &t, 0.0)?;
                let repeat = u32_value(fb, SamplerAddressMode::Repeat.as_raw())?;
                let mirror = u32_value(fb, SamplerAddressMode::MirrorRepeat.as_raw())?;
                let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
                let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
                let k = fb.select(mirrors, twice, none)?;
                Some(fb.select(repeats, floor, k)?)
            }
            _ => return Err(cx.fail("a native linear sample without its sampler's word")),
        };
        parts.push(cell(fb, &t, u, n, start)?);
    }
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let ty = fb.module().ty(SType::Vector { elem: f, len: 2 })?;
    let snapped = fb.composite_construct(ty, &parts)?;
    let (snapped, _) = super::emit::layered(cx, fb, s, snapped)?;
    let at = fb.select(native, snapped, coord)?;
    Ok(Some((at, native)))
}

/// The start of the mirrored period of `u` (an `f32`): `2 floor(u / 2)`.
fn mirror_start(fb: &mut FunctionBuilder<'_>, t: &Tys, u: SValue) -> BResult<SValue> {
    let half = f32_value(fb, t, 0.5)?;
    let two = f32_value(fb, t, 2.0)?;
    let h = fb.binary(BinOp::FMul, u, half)?;
    let h = fb.glsl(GlslOp::Floor, &[h])?;
    fb.binary(BinOp::FMul, h, two)
}

/// The normalized coordinate of the centre of the 1/256 footprint step Metal reads coordinate
/// `u` (an `f32`) at, on an axis of size `n` (a `u32`) whose period starts at `start` (an `f32`;
/// `None` on a clamping axis): the position `X = (u - start) N`, `c = floor(256 X) + [frac(256 X)
/// >= 1/2]` (`q + 128` for Metal's step `q`, as `position::AxisSnap::Cell` reads it), and
/// `c / 256 / N`.
fn cell(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    u: SValue,
    n: SValue,
    start: Option<SValue>,
) -> BResult<SValue> {
    let w = match start {
        Some(k) => fb.binary(BinOp::FSub, u, k)?,
        None => u,
    };
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let nf = fb.convert(ConvOp::ConvertUToF, n, f)?;
    let x = fb.binary(BinOp::FMul, w, nf)?;
    let k256 = f32_value(fb, t, 256.0)?;
    let y = fb.binary(BinOp::FMul, x, k256)?;
    let fl = fb.glsl(GlslOp::Floor, &[y])?;
    let r = fb.binary(BinOp::FSub, y, fl)?;
    let half = f32_value(fb, t, 0.5)?;
    let up = fb.binary(BinOp::FOrdGreaterThanEqual, r, half)?;
    let one = f32_value(fb, t, 1.0)?;
    let zero = f32_value(fb, t, 0.0)?;
    let up = fb.select(up, one, zero)?;
    let c = fb.binary(BinOp::FAdd, fl, up)?;
    let step = f32_value(fb, t, 1.0 / 256.0)?;
    let at = fb.binary(BinOp::FMul, c, step)?;
    fb.binary(BinOp::FDiv, at, nf)
}

/// The `f32` constant `x` as a value.
fn f32_value(fb: &mut FunctionBuilder<'_>, _: &Tys, x: f32) -> BResult<SValue> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let c = scalar(fb, f, u64::from(x.to_bits()))?;
    value(fb, c)
}
