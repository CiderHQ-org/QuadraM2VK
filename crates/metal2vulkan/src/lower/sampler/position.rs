//! Metal's texel-space position of a normalized coordinate, and the coordinate that makes a
//! device's gather or nearest sample read the texels Metal reads.
//!
//! **The position** (Apple M3 Max, macOS 26.7, and Apple M4, macOS 27.0.1, byte-identical; the
//! tie, gather, and bicubic probes). Per axis of size `N` and normalized coordinate `u`, Apple's
//! texture unit first wraps `u` into one period, in `f32`, where the axis repeats: `u' = u -
//! floor(u)` under `repeat`, `u' = u - 2 floor(u / 2)` under `mirrored_repeat` (a period of two),
//! the subtraction rounded toward positive infinity (`wrap`; it rounds at all only where `-1 < u <
//! 0`), or exactly where `N` is a power of two (below). Then it scales: `X = u' N` rounded to `f32`
//! with ties away from zero. A clamping axis scales `u` itself. Every filter reads its footprint
//! from `X`: the linear and bicubic filters and a gather the quantized `q = floor(256 X + 1/2) -
//! 128`, a nearest filter `floor(X)`, and the taps then wrap modulo `N` or mirror with period `2N`.
//! The probes measured the wrap on the linear, nearest, gather, and bicubic filters (every lane,
//! both chips), and the ties on the linear, gather, and bicubic filters under `clamp_to_zero`,
//! where a product exactly half a unit in the last place from two `f32`s decides the footprint. The
//! wrap's rounding comes from the position cases (`wrg_sampler_positions` at 13x11, 7x5, and 255x3
//! on the M3, whose outputs the M4's equal byte for byte): rounded up, every linear, nearest,
//! gather, and bicubic result of their repeating and mirrored axes is Metal's; rounded to nearest
//! even, with ties away, or not at all, it misses lanes with `-1 < u < 0` whose wrapped `u'` needs
//! more bits than an `f32` holds.
//!
//! **A power-of-two size wraps exactly**. On a repeating or mirrored axis whose size is a power of
//! two, Metal reads the footprints of the exact `u'`, per axis: the linear-filter probes (M4 and
//! M3, byte-identical) at 2D 16x8 and 4x32 and 3D 16x16x16, 8x4x2 and 16x8x4, and on the 4-texel
//! axis of a 9x4x11 texture whose other axes round up, and the position cases at 16x8, 4x32 and
//! 16x11 (linear, nearest, and gathers, `gather_compare` included; M3 and M4). `wrap` rounds the
//! subtraction down there, which reads the same footprints: `u' N` is then exact, and every step a
//! footprint reads (an integer, or `i + 1/2` plus a multiple of `1/512`) lies on the `1/512` grid,
//! which an `f32` holds exactly below `2^15`, so rounding `u'` down never carries `X` across one.
//! Rounded up, it reads one `1/256` step, sometimes a texel, high where `-1 < u < 0` lands just
//! below a step.
//!
//! Multiplying first and wrapping the taps after (as a Vulkan device does, and as the emulation
//! once did) reads another footprint wherever `u` lies outside `[0, 1)` on a size that is not a
//! power of two and `u N` lands near a step: the product rounds at the magnitude of `u N`, not of
//! `u' N`. At a power-of-two size `u N` is exact, and multiplying first reads Metal's footprints.
//!
//! **Ties away from zero, in integers** ([`position`]). `p = u' N` rounded to nearest even is a tie
//! rounded toward zero exactly when the bits of the exact product below `p`'s last place are
//! `100…0` and `p`'s significand is even; then `X` is the next `f32` away from zero, `p`'s bits
//! plus one. The exact product's low bits are those of `bits(u') * N` (an `OpIMul`, modulo `2^32`):
//! `u'`'s exponent and implicit bit sit at bit 23 and up, so they never reach the bits the test
//! reads. Those are bits `0` to `s`, where `s` is `p`'s biased exponent less `u'`'s (the sign bits
//! cancel, `p` having `u'`'s sign); a shift left by `31 - s` puts them at the top of the word,
//! where the tie reads `0x4000_0000`. This is exact for every finite normal `u'` and every
//! `N < 2^22`, where `s <= 22` and the bits stay below bit 23; Metal's sizes are at most `2^14`. A
//! denormal `u'` may be adjusted wrongly, by one unit in the last place of an `X` below `2^-100`,
//! which moves no footprint. A tie moves a footprint only at a negative position, or at one of
//! `2^14` or more. Where the two neighbours of a tie lie on either side of a footprint's step (an
//! integer, or a gather's `i + 1/2 - 1/512`), the upper neighbour is the step, whose significand is
//! even below `2^14`: the even rounding picks it, as ties away from zero does for a positive
//! position, while at a negative one ties away from zero picks the lower. That is why the probes'
//! deciding lanes were clamped axes' negative coordinates, and a repeating or mirrored axis, whose
//! position is never negative, differs only at sizes of at least `2^13` (mirrored) or `2^14`
//! (repeating). A product past `f32`'s range gives an unspecified `X`, as before (the footprints
//! read no non-finite position: `bicubic`). The test is 12 integer instructions on every device;
//! `OpFmaKHR` for the product's error would take 11, and only where `shaderFmaFloat32` is enabled.
//!
//! **A texel offset on a repeating or mirrored axis** ([`offset_dropped`]). Metal adds a constant
//! texel offset to the tap and then wraps it, as Vulkan places it, but on a repeating or mirrored
//! axis whose level size `n` lies in `[2, 7]` and whose texture's base-level size on that axis is
//! not a power of two: there it drops the offset on that axis. At the base level that is a size of
//! 3, 5, 6, or 7. The offset probes:
//! - round 1, Apple M4 on macOS 27.0.1 and Apple M3 Max on macOS 26.7, byte-identical, each run
//!   with and without Metal's validation layer: every 2D size from 1 to 17 per axis and eight
//!   larger controls, offsets -8 to 7 per axis, nearest and linear samples, gathers of each
//!   component, and a depth texture's gathers; a clamping axis places the offset as Vulkan does
//!   on every size;
//! - round 2, the M4: base levels up to 16384, 2D arrays and 3D textures, per axis;
//! - round 3, the M4: every level of 18 bases at an explicit level, nearest and linear samples
//!   (level sizes 2 and 4 drop it under a base that is not a power of two and keep it under one
//!   that is; sizes 3, 5 and 6 drop it, and 8 and above keep it, under either), and each level
//!   above the base through a view of that one level, which reads exactly as the level does: the
//!   view's parent's base-level size decides.
//!
//! An emulated footprint applies the rule to its taps (`bicubic::axis`), and a snapped axis that
//! repeats or mirrors subtracts the offset the device will add where Metal drops it
//! ([`snapped_axis`]). The base-level size the rule reads is a view's parent's, which a shader
//! cannot query: a view whose first level lies above its parent's base reads the bound view's
//! own, so on an axis of size 2 or 4 under a parent base that is not a power of two (where the
//! view's base is) the rule would keep the offset Metal drops. The binder states those axes in the
//! slot's `SpecKey::OffsetBaseAxes` specialization (0 for every texture that is not such a view),
//! whose tests (`texture::OffsetBase`) the rule reads; a heap element's access reads none, and
//! records `PrecisionNote::ViewBaseTexelOffset` (the residue there).
//!
//! **The snapped coordinate** ([`snapped`]). A device gather or nearest sample at a coordinate
//! near a texel edge reads whichever texel its own arithmetic gives (ANV's gathers read the block
//! one texel low at odd sizes). Where Metal's texel is known, a snapped axis instead hands the
//! device a coordinate half a texel from every edge of it: a gather's `(i0 + 1) / N`, the centre of
//! its 2x2 block, and a nearest sample's `(i + 1/2) / N`, the centre of its texel, with `i0` and
//! `i` from Metal's position. Any device's rounding then reads Metal's block or texel, and the
//! device keeps everything else: the texel conversion, the border, the comparison, the offset
//! (added to the snapped index, then wrapped or clamped, as Metal adds it), and the layer. A
//! gather's `i0` is computed in `f32`, `floor(floor(256 X + 1/2) / 256 - 1/2)`, exact for `|X| <
//! 2^15` (every position of a Metal texture's repeating or mirrored axis, at most `2N`, included).
//! On a clamping axis `i0` is first clamped into `[-10, N + 8]`: past either end, with any offset
//! in Metal's gather range `[-8, 7]`, both texels lie outside the level on the same side, so they
//! read the same edge texels or border as the exact `i0` would, and the coordinate stays small. On
//! a repeating or mirrored axis whose size drops the offset ([`offset_dropped`]), the snapped index
//! is first moved back by the offset, which the device then adds: the device reads the texel or
//! block Metal reads there, the one without the offset. A nearest sample snaps only its repeating
//! and mirrored axes (`i` lies in `[0, N]` or `[0, 2N]` there), and only where it reads the base
//! level: a constexpr sampler's by its state, a runtime sampler's by its word in the function
//! ([`snapped_by_word`]). A nearest cube sample that reads the base level snaps its direction
//! instead, onto the centre of the texel Metal reads on its face (`face::snapped`): there Metal's
//! position comes from its texture unit's reciprocal of the major magnitude, which the module
//! computes correctly rounded where Apple's approximates it (`PrecisionNote::CubeFaceReciprocal`).
//!
//! **A runtime or argument buffer's sampler** names its state in its word, read in the function:
//! its gather of a 2D texture or array snaps behind it ([`gathered_by_word`]), each axis at the
//! position its mode gives ([`position_by_mode`]) or, where the word states pixel coordinates, at
//! the coordinate as given, with `i0` clamped into the wider `[-2N - 10, 2N + 8]`, which holds
//! every position of a repeating or mirrored axis and reads a clamping axis's same edge texels or
//! border. On a `MirrorClampToEdge` axis (which no constexpr sampler states) Metal's position is a
//! clamping axis's, `u N` unwrapped, its taps mirrored once and clamped as Vulkan's
//! `MIRROR_CLAMP_TO_EDGE` mirrors and clamps them, and a texel offset is dropped by the sizes that
//! drop it on a repeating axis (every gather of 13x11, 7x5 and 16x8 textures through a slot's and
//! an argument buffer's runtime sampler, with offsets of 2 and -3, on an Apple M3 Max and an Apple
//! M4 alike), a pixel coordinate's as a normalized one's, and a nearest sample's, a linear one's
//! and a comparison's as a gather's: a nearest sample snaps such an axis as a repeating one
//! ([`snapped_by_word`]), and a pixel word's texel centre moves back by a
//! dropped offset (`emit::pixel_centre`). On a device with Metal's own sampler, only a pixel word's
//! gather is snapped, and a linear pixel word's sample snaps onto its footprint cell
//! ([`celled_by_word`]), as a constexpr pixel sampler's does there (that sample reads Metal's rule
//! at the coordinate as given: our probe's 10,250 lanes of a runtime pixel sampler on both chips).

use crate::air::abi::{AddressMode, Coordinates, Filter, MipFilter, SamplerState};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::texture::OffsetBase;
use crate::metal::SamplerAddressMode;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, ImageQuery, Type as SType};

use super::bicubic::TapAddress;
use super::face::{FaceRoutines, FaceWrap};
use super::{CombinedImage, Lod, SampleOp};

/// The census words of one axis's [`position`]: the conversion of the size, the wrap (a mirrored
/// axis's four instructions, and the rounding's thirteen, which choose up or down by the size), the
/// product, and the ties' twelve integer instructions, with their constants (186 words on a
/// mirrored axis). Every estimate that emits a position adds it per axis.
pub const POSITION_WORDS: u32 = 200;

/// How the texels a snapped plane axis hands the device follow from the position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisSnap {
    /// A gather's 2x2 block: the coordinate `(i0 + 1) / N`.
    Gather,
    /// A nearest sample's texel: the coordinate `(floor(X) + 1/2) / N`.
    Nearest,
    /// A linear sample's footprint cell: the coordinate `(floor(256 X - 127.5) + 128) / 256 / N`,
    /// the centre of the `1/256` step `q` Metal's filter reads `X` as, half a step from either edge
    /// of it.
    Cell,
}

/// What a snap hands the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SnapKind {
    /// The plane coordinate with [`Snap::axes`]' axes snapped.
    Axes(AxisSnap),
    /// A nearest cube sample's direction, onto the centre of the texel Metal reads on its face
    /// (`face::snapped`), Metal's position computed with Metal's reciprocal (`libm::Fun::Recip`)
    /// and the fused multiply-add `fma`, and wrapped on the face axes [`Snap::axes`] repeats
    /// (`face::FaceWrap`).
    Cube(FaceRoutines),
    /// A runtime or argument buffer's sampler's nearest sample of a texture of `axes` plane axes:
    /// where the sampler's word states that it reads one texel of the base level
    /// (`SamplerWord::NEAREST_BASE`), or, for a sample at a level the function knows (`leveled`),
    /// one texel of one level (`SamplerWord::NEAREST_LEVEL`: no mip filter or a nearest one), each
    /// plane axis the word repeats or mirrors is snapped as [`AxisSnap::Nearest`] snaps it, at the
    /// position its word's mode gives ([`position_by_mode`]) on the size of the level it reads
    /// (`level::snap_level`); elsewhere the coordinate is the device's ([`snapped_by_word`]).
    /// [`Snap::axes`] is empty: the word names the modes.
    Word { axes: u8, leveled: bool },
    /// A runtime or argument buffer's sampler's gather of a 2D texture or 2D array: behind the
    /// sampler's word, each plane axis takes [`AxisSnap::Gather`]'s coordinate, the centre of the
    /// 2x2 block Metal reads at its position: the coordinate as given where the word states pixel
    /// coordinates, else [`position_by_mode`] by the axis's mode ([`gathered_by_word`]). On a
    /// device that samples through Metal's own sampler (`metal`) only a pixel coordinate is
    /// snapped: Metal's own gather of a normalized coordinate is Metal's block. [`Snap::axes`] is
    /// empty: the word names the modes.
    WordGather { metal: bool },
    /// On a device that samples through Metal's own sampler, a runtime or argument buffer's
    /// sampler's sample of a texture of `axes` plane axes: where the word states pixel coordinates
    /// and linear min and mag filters, each plane axis takes [`AxisSnap::Cell`]'s coordinate, the
    /// centre of the footprint step Metal reads the coordinate as given at, as a constexpr linear
    /// pixel sampler's sample does there ([`celled_by_word`]); every other word keeps the scaled
    /// coordinate. [`Snap::axes`] is empty.
    WordCell { axes: u8 },
}

/// A gather or nearest sample whose coordinate is snapped onto Metal's texels (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Snap {
    pub kind: SnapKind,
    /// Per plane axis (`s`, then `t`, then `r`), its addressing where the axis is snapped, or
    /// `None` where the device reads the coordinate as given; for a cube its face's `s` and `t`
    /// axes, addressed by the sampler's `s` and `t` addresses.
    pub axes: Vec<Option<TapAddress>>,
    /// The coordinate is in texels (a pixel sampler's): Metal's position `X` is the coordinate as
    /// given, so the snap takes no [`position`].
    pub pixel: bool,
}

impl Snap {
    /// A runtime or argument buffer's sampler's snap behind its word ([`SnapKind::Word`],
    /// [`SnapKind::WordGather`], [`SnapKind::WordCell`]).
    pub fn by_word(&self) -> bool {
        matches!(
            self.kind,
            SnapKind::Word { .. } | SnapKind::WordGather { .. } | SnapKind::WordCell { .. }
        )
    }

    /// The census words: the image and its size query; per snapped plane axis the extractions, the
    /// [`position`], the index and its clamp, and the coordinate, and the composite; or a cube's
    /// `face::snapped` with its reciprocal and two fused multiply-adds, and its wrap.
    pub fn words(&self) -> u32 {
        let position = if self.pixel { 8 } else { POSITION_WORDS };
        let per_axis = position
            + match self.kind {
                SnapKind::Axes(AxisSnap::Gather) => GATHER_AXIS_WORDS,
                SnapKind::Axes(AxisSnap::Nearest) => NEAREST_AXIS_WORDS,
                SnapKind::Axes(AxisSnap::Cell) => CELL_AXIS_WORDS,
                SnapKind::Cube(routines) => {
                    return 32
                        + super::face::SNAP_WORDS
                        + routines.use_words()
                        + self.face_wrap().words();
                }
                SnapKind::Word { axes, leveled } => {
                    return 32
                        + WORD_REACH_WORDS
                        + if leveled { WORD_LEVEL_WORDS } else { 0 }
                        + u32::from(axes)
                            * (POSITION_WORDS
                                + MODE_POSITION_WORDS
                                + NEAREST_AXIS_WORDS
                                + WORD_AXIS_WORDS);
                }
                SnapKind::WordGather { .. } => {
                    return 32
                        + WORD_REACH_WORDS
                        + 2 * (POSITION_WORDS
                            + MODE_POSITION_WORDS
                            + GATHER_AXIS_WORDS
                            + WORD_AXIS_WORDS);
                }
                SnapKind::WordCell { axes } => {
                    return 32
                        + WORD_REACH_WORDS
                        + u32::from(axes) * (8 + CELL_AXIS_WORDS + WORD_AXIS_WORDS);
                }
            };
        let snapped = self.axes.iter().filter(|a| a.is_some()).count() as u32;
        32 + snapped * per_axis + self.axes.len() as u32 * 8
    }

    /// The census words of the offset rule ([`offset_dropped`]) on each axis this snap may move
    /// back, for an access with a texel offset: each snapped plane axis's, and each axis of a word
    /// snap; a cube's access takes no offset.
    pub fn offset_words(&self) -> u32 {
        let axes = match self.kind {
            // A cube's access takes no offset; Metal's own sampler places a pixel cell's.
            SnapKind::Cube(_) | SnapKind::WordCell { .. } => 0,
            SnapKind::Word { axes, .. } => u32::from(axes),
            SnapKind::WordGather { .. } => 2,
            SnapKind::Axes(_) => self.axes.iter().filter(|a| a.is_some()).count() as u32,
        };
        axes * OFFSET_RULE_WORDS
    }

    /// The face axes a cube's snap wraps (`face::FaceWrap`): those its sampler repeats.
    fn face_wrap(&self) -> FaceWrap {
        let axis = |k: usize| self.axes.get(k).copied().flatten();
        FaceWrap::of(axis(0), axis(1))
    }
}

/// A gather axis's words past its position: the two extractions, `i0` (six instructions and their
/// constants), its clamp, and the coordinate.
const GATHER_AXIS_WORDS: u32 = 96;
/// A nearest axis's words past its position: the two extractions, the floor, and the coordinate.
const NEAREST_AXIS_WORDS: u32 = 40;
/// A footprint cell's words past its position: the two extractions, the step (seven instructions
/// and their constants), and the coordinate.
const CELL_AXIS_WORDS: u32 = 96;
/// A word snap's test of the base level (`SamplerWord::NEAREST_BASE`): the mask, the comparison,
/// and their constants.
const WORD_REACH_WORDS: u32 = 24;
/// A word snap's axis words past its position and nearest coordinate: the mode's shift and mask,
/// the three mode tests (`Repeat`, `MirrorRepeat` and `MirrorClampToEdge`), their disjunction and
/// conjunction with the reach, and the select.
const WORD_AXIS_WORDS: u32 = 64;
/// A leveled word snap's level (`level::snap_level`: the clamp's two fields, the conversion, the
/// clamps, and the mip filter's test and select), the texture's levels and their clamp, and the
/// size query of that level; a dynamic level's grid is the level rule's (`level::by_word`) or
/// its own, at most `level::LEVEL_WORDS`.
const WORD_LEVEL_WORDS: u32 = 160 + super::level::LEVEL_WORDS;

/// The tap addressing of address mode `m`.
pub(super) fn address(m: AddressMode) -> TapAddress {
    match m {
        AddressMode::ClampToEdge => TapAddress::Edge,
        AddressMode::Repeat => TapAddress::Repeat,
        AddressMode::MirroredRepeat => TapAddress::Mirror,
        AddressMode::ClampToZero | AddressMode::ClampToBorder => TapAddress::Border,
    }
}

/// The snap of `op` on a texture of `shape` through the static sampler `stat` (`None`: an
/// argument buffer's), where the module docs snap it, on a device without
/// `VkProperty::MetalSampler`:
///
/// - every gather (with or without a comparison or an offset) of a 2D texture or 2D array through
///   a constexpr sampler with normalized coordinates, on both plane axes;
/// - a nearest sample without a comparison through a constexpr sampler with normalized
///   coordinates ([`super::linear::unfiltered`]) of a 1D, 2D, or 3D texture or array that reads
///   the base level (the sampler has no mip filter, or the sample reads level 0 and the sampler
///   neither biases nor clamps it) and has a repeating or mirrored plane axis, on those axes;
/// - such a nearest sample of a cube or cube array that reads the base level, its direction
///   (`face::snapped`, which records `PrecisionNote::CubeFaceReciprocal`);
/// - a sample without a comparison through a runtime or argument buffer's sampler of a 1D, 2D, or
///   3D texture or array, behind the sampler's word ([`SnapKind::Word`]): the axes the word repeats
///   or mirrors where the word reads one texel of the base level, or, for a sample at a level the
///   function knows (an explicit level, or a kernel's bias, without a `min_lod_clamp`) of a texture
///   of floating-point texels, of the level a nearest mip filter reads there.
///
/// Every other gather keeps the device's footprint and records
/// `PrecisionNote::DeviceGatherFootprint` (a cube, a runtime or argument buffer's sampler). On a
/// device with the property, each gather and each such nearest sample is the device's, Metal's own
/// sampler, and the module requires the property.
///
/// A constexpr pixel sampler's coordinate is Metal's position as given, which the device's sampler,
/// reading the coordinate over the size, does not reproduce, its own or Metal's alike. So on every
/// device, the property's included:
///
/// - its gather of a 2D texture or 2D array snaps both plane axes from the coordinate as given;
/// - its linear sample of a 1D, 2D, or 3D texture or array with a floating-point texel (an
///   `integer` one's reads one texel), where nothing emulates it (on a device with the property;
///   elsewhere the emulation reads the coordinate as given, `linear`), snaps every plane axis onto
///   the centre of its footprint cell, unless the sampler is anisotropic.
///
/// A cube's coordinate is a direction through a pixel sampler too, so a cube's snap
/// reads a pixel sampler as a normalized one.
pub(super) fn plan(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    stat: Option<(SamplerRef, Coordinates)>,
    (shape, integer): (ImageShape, bool),
) -> Result<Option<Snap>, Error> {
    let metal_sampler = cx.profile.device_samples();
    // A constexpr sampler's state, and whether its coordinate is in texels (not a cube's).
    let state = match stat {
        Some((SamplerRef::Constexpr(words), c)) => Some((
            super::state_of(words)?,
            c == Coordinates::Pixel && shape.dim != ImageDim::Cube,
        )),
        _ => None,
    };
    // The plane axes of a 1D, 2D, or 3D texture or array, single-sampled.
    let plane = match (shape.dim, shape.ms) {
        (ImageDim::D1, false) => 1,
        (ImageDim::D2, false) => 2,
        (ImageDim::D3, false) => 3,
        _ => 0,
    };
    match op {
        SampleOp::Gather { .. } => match state {
            Some((s, pixel))
                if shape.dim == ImageDim::D2 && !shape.ms && (pixel || !metal_sampler) =>
            {
                Ok(Some(Snap {
                    kind: SnapKind::Axes(AxisSnap::Gather),
                    axes: vec![Some(address(s.address_s)), Some(address(s.address_t))],
                    pixel,
                }))
            }
            // A runtime or argument buffer's sampler's gather, behind its word. Metal's own gather
            // of a normalized coordinate is Metal's block: on a device with the property the word
            // snaps a pixel coordinate alone, and the module requires it.
            None if shape.dim == ImageDim::D2 && !shape.ms => {
                if metal_sampler {
                    cx.families.sampler.metal_sampler = true;
                }
                Ok(Some(Snap {
                    kind: SnapKind::WordGather {
                        metal: metal_sampler,
                    },
                    axes: Vec::new(),
                    pixel: false,
                }))
            }
            _ if metal_sampler => {
                cx.families.sampler.metal_sampler = true;
                Ok(None)
            }
            _ => {
                cx.precision.insert(PrecisionNote::DeviceGatherFootprint);
                Ok(None)
            }
        },
        // A linear pixel sample's footprint cell, where Metal's own sampler filters it.
        SampleOp::Sample { dref: None, .. }
            if state.as_ref().is_some_and(|(s, pixel)| {
                *pixel
                    && plane > 0
                    && metal_sampler
                    && !integer
                    && s.min_filter == Filter::Linear
                    && s.mag_filter == Filter::Linear
            }) =>
        {
            let Some((s, _)) = state else {
                return Ok(None);
            };
            if s.max_anisotropy > 1 {
                return Ok(None);
            }
            Ok(Some(Snap {
                kind: SnapKind::Axes(AxisSnap::Cell),
                axes: [s.address_s, s.address_t, s.address_r]
                    .into_iter()
                    .take(plane)
                    .map(|m| Some(address(m)))
                    .collect(),
                pixel: true,
            }))
        }
        // On a device with the property, a runtime or argument buffer's sampler's sample, a
        // comparison's included, onto its footprint cell where its word states a linear pixel
        // sampler, as a constexpr one's. An integer texture's sample reads one texel whatever the
        // filters say, its pixel sampler's texel centre.
        SampleOp::Sample { .. }
            if plane > 0
                && metal_sampler
                && !integer
                && !matches!(stat, Some((SamplerRef::Constexpr(_), _))) =>
        {
            cx.families.sampler.metal_sampler = true;
            Ok(Some(Snap {
                kind: SnapKind::WordCell { axes: plane as u8 },
                axes: Vec::new(),
                pixel: false,
            }))
        }
        // A runtime or argument buffer's sampler's nearest sample, behind its word: at a level the
        // function knows, on the level a nearest mip filter reads there, where Metal's level rule
        // is measured (`level`: not an integer texture's, nor beside a `min_lod_clamp`).
        SampleOp::Sample {
            dref: None,
            lod,
            min_lod,
            ..
        } if plane > 0 && !matches!(stat, Some((SamplerRef::Constexpr(_), _))) => {
            if metal_sampler {
                cx.families.sampler.metal_sampler = true;
                return Ok(None);
            }
            let leveled = !integer
                && min_lod.is_none()
                && matches!(lod, Lod::Level(_) | Lod::Base { min: None, .. });
            Ok(Some(Snap {
                kind: SnapKind::Word {
                    axes: plane as u8,
                    leveled,
                },
                axes: Vec::new(),
                pixel: false,
            }))
        }
        SampleOp::Sample {
            lod,
            dref: None,
            min_lod,
            ..
        } => {
            // A pixel sampler's nearest sample takes its texel's centre instead (`emit`).
            let Some((s, false)) = state else {
                return Ok(None);
            };
            let axes = match (shape.dim, shape.ms) {
                (ImageDim::D1, false) => 1,
                (ImageDim::D2, false) => 2,
                (ImageDim::D3, false) => 3,
                (ImageDim::Cube, false) => {
                    if !super::linear::unfiltered(&s) || !base_level(&s, lod, min_lod.is_some()) {
                        return Ok(None);
                    }
                    if metal_sampler {
                        cx.families.sampler.metal_sampler = true;
                        return Ok(None);
                    }
                    return cube(cx, &s).map(Some);
                }
                _ => return Ok(None),
            };
            let modes = [s.address_s, s.address_t, s.address_r];
            let wraps =
                |m: &AddressMode| matches!(m, AddressMode::Repeat | AddressMode::MirroredRepeat);
            if !super::linear::unfiltered(&s)
                || !base_level(&s, lod, min_lod.is_some())
                || !modes.iter().take(axes).any(wraps)
            {
                return Ok(None);
            }
            if metal_sampler {
                cx.families.sampler.metal_sampler = true;
                return Ok(None);
            }
            Ok(Some(Snap {
                kind: SnapKind::Axes(AxisSnap::Nearest),
                axes: modes
                    .iter()
                    .take(axes)
                    .map(|m| wraps(m).then(|| address(*m)))
                    .collect(),
                pixel: false,
            }))
        }
        _ => Ok(None),
    }
}

/// The snap of a nearest cube sample through sampler `s` (`face::snapped`), with its face
/// position's routines (`face::plan`, which records `PrecisionNote::CubeFaceReciprocal`) and the
/// addressing of its face's `s` and `t` axes, which wraps a repeating one (`face::FaceWrap`).
fn cube(cx: &mut SelectCx<'_>, s: &SamplerState) -> Result<Snap, Error> {
    Ok(Snap {
        kind: SnapKind::Cube(super::face::plan(cx)?),
        axes: vec![Some(address(s.address_s)), Some(address(s.address_t))],
        pixel: false,
    })
}

/// Whether a sample through sampler `s` with level `lod` (`clamped`: a `min_lod_clamp`) reads the
/// base level: the sampler has no mip filter, or the sample reads level 0 and the sampler neither
/// biases nor clamps it (`footprint::of`'s rule).
fn base_level(s: &SamplerState, lod: &Lod, clamped: bool) -> bool {
    if s.mip_filter == MipFilter::None {
        return true;
    }
    matches!(
        lod,
        Lod::Level(None)
            | Lod::Base {
                bias: None,
                min: None,
            }
    ) && !clamped
        && s.lod_min_clamp.to_f32() == 0.0
        && s.lod_bias.to_f32() == 0.0
}

fn f32_const(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<SValue> {
    let ty = fb.module().ty(SType::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(x.to_bits())),
    })?;
    fb.module().const_value(c)
}

fn u32_const(fb: &mut FunctionBuilder<'_>, x: u32) -> BResult<SValue> {
    let c = fb.u32_const(x)?;
    fb.module().const_value(c)
}

/// The census words of [`offset_dropped`] and of a snapped axis's compensation by it: the
/// subtractions, the comparisons, the mask and their conjunction, the select, the subtraction from
/// the index, and their constants, in the leveled form (the larger).
pub const OFFSET_RULE_WORDS: u32 = 64;

/// Whether Metal drops a constant texel offset on a repeating or mirrored axis whose level size is
/// `n` (a `u32`), a `bool` (module docs): `n` lies in `[2, 7]` and the axis's base-level size
/// `base` (a `u32`; `None`: `n` is the base level's) is not a power of two, `n - 2 <= 5` unsigned
/// and `base & (base - 1) != 0`. At the base level that is `n` of 3, 5, 6, or 7, `n - 3 <= 4`
/// unsigned and `n != 4`. Every other size, and every clamping axis, takes the offset. The
/// base-level size Metal reads is a view's parent's: where `parent` (a `bool`, the axis's
/// `texture::OffsetBase` test) holds, it is not a power of two though the bound view's is, so the
/// offset drops at every `n` in `[2, 7]`.
pub fn offset_dropped(
    fb: &mut FunctionBuilder<'_>,
    n: SValue,
    base: Option<SValue>,
    parent: Option<SValue>,
) -> BResult<SValue> {
    let (base, parent) = match (base, parent) {
        (None, None) => {
            let three = u32_const(fb, 3)?;
            let four = u32_const(fb, 4)?;
            let t = fb.binary(BinOp::ISub, n, three)?;
            let small = fb.binary(BinOp::ULessThanEqual, t, four)?;
            let odd_one = fb.binary(BinOp::INotEqual, n, four)?;
            return fb.binary(BinOp::LogicalAnd, small, odd_one);
        }
        (base, parent) => (base.unwrap_or(n), parent),
    };
    let two = u32_const(fb, 2)?;
    let five = u32_const(fb, 5)?;
    let t = fb.binary(BinOp::ISub, n, two)?;
    let small = fb.binary(BinOp::ULessThanEqual, t, five)?;
    let one = u32_const(fb, 1)?;
    let below = fb.binary(BinOp::ISub, base, one)?;
    let mask = fb.binary(BinOp::BitwiseAnd, base, below)?;
    let zero = u32_const(fb, 0)?;
    let uneven = fb.binary(BinOp::INotEqual, mask, zero)?;
    let uneven = match parent {
        Some(p) => fb.binary(BinOp::LogicalOr, uneven, p)?,
        None => uneven,
    };
    fb.binary(BinOp::LogicalAnd, small, uneven)
}

/// The census words a slot's `SpecKey::OffsetBaseAxes` test adds to [`offset_dropped`] on each
/// plane axis an offset moves, per footprint, snap or centre that places it: the general rule's
/// words where they replace the base level's shorter ones, the disjunction, and their constants,
/// and a share of the slot's tests' declaration (a specialization constant and three
/// `OpSpecConstantOp` tests per axis, `texture::OffsetBase`).
pub const PARENT_RULE_WORDS: u32 = 64;

/// Metal's position `X` (an `f32`, module docs) of normalized coordinate `u` (an `f32`) on an axis
/// of size `n` (a `u32`, below `2^22`) addressed by `address`, with the size as an `f32`.
pub fn position(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    address: TapAddress,
) -> BResult<(SValue, SValue)> {
    let one = u32_const(fb, 1)?;
    let zero = u32_const(fb, 0)?;
    // The wrap into one period, in `f32`, its subtraction rounded up, or down at a power-of-two
    // size.
    let u = match address {
        TapAddress::Edge | TapAddress::Border => u,
        TapAddress::Repeat => {
            let fl = fb.glsl(GlslOp::Floor, &[u])?;
            wrap(fb, u, fl, n, (one, zero))?
        }
        TapAddress::Mirror => {
            let twice = mirror_start(fb, u)?;
            wrap(fb, u, twice, n, (one, zero))?
        }
    };
    scale(fb, u, n, (one, zero))
}

/// The census words [`position_by_mode`] adds to [`position`]'s: both periods' starts, the
/// mode's two tests and selects, and their constants.
pub const MODE_POSITION_WORDS: u32 = 64;

/// [`position`] on an axis whose address mode is `mode`, a `u32` holding `MTLSamplerAddressMode`'s
/// value (a runtime sampler's, read from its `SamplerWord` in the function): the period starts at
/// `floor(u)` under `Repeat` and at `2 floor(u / 2)` under `MirrorRepeat`, and at 0 under every
/// other mode, where the wrap is `u` itself (`u - 0` and its sum with 0 are `u`, so the rounding's
/// step is 0) and the product scales `u`, as a clamping axis's does.
pub fn position_by_mode(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    mode: SValue,
) -> BResult<(SValue, SValue)> {
    let one = u32_const(fb, 1)?;
    let zero = u32_const(fb, 0)?;
    let fl = fb.glsl(GlslOp::Floor, &[u])?;
    let twice = mirror_start(fb, u)?;
    let none = f32_const(fb, 0.0)?;
    let repeat = u32_const(fb, SamplerAddressMode::Repeat.as_raw())?;
    let mirror = u32_const(fb, SamplerAddressMode::MirrorRepeat.as_raw())?;
    let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
    let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
    let k = fb.select(mirrors, twice, none)?;
    let k = fb.select(repeats, fl, k)?;
    let u = wrap(fb, u, k, n, (one, zero))?;
    scale(fb, u, n, (one, zero))
}

/// The start `2 floor(u / 2)` of a mirrored axis's period of two around `u` (an `f32`).
fn mirror_start(fb: &mut FunctionBuilder<'_>, u: SValue) -> BResult<SValue> {
    let half = f32_const(fb, 0.5)?;
    let h = fb.binary(BinOp::FMul, u, half)?;
    let fl = fb.glsl(GlslOp::Floor, &[h])?;
    fb.binary(BinOp::FAdd, fl, fl)
}

/// Metal's position `X` of the wrapped coordinate `u` (an `f32`) on an axis of size `n` (a `u32`):
/// the product rounded with ties away from zero (module docs), with the size as an `f32`;
/// `(one, zero)` are `u32`s.
fn scale(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    (one, zero): (SValue, SValue),
) -> BResult<(SValue, SValue)> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let w = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let nf = fb.convert(ConvOp::ConvertUToF, n, f)?;
    // The product rounded to nearest even, on its own (in the profile's form, never fused into what
    // follows).
    let p = fb.binary_no_contraction(BinOp::FMul, u, nf)?;
    // A tie rounded toward zero: bits `0..=s` of `bits(u) * n` are `0100…0` (module docs).
    let bu = fb.convert(ConvOp::Bitcast, u, w)?;
    let bp = fb.convert(ConvOp::Bitcast, p, w)?;
    let lo = fb.binary(BinOp::IMul, bu, n)?;
    let k23 = u32_const(fb, 23)?;
    let eu = fb.binary(BinOp::ShiftRightLogical, bu, k23)?;
    let ep = fb.binary(BinOp::ShiftRightLogical, bp, k23)?;
    let s = fb.binary(BinOp::ISub, ep, eu)?;
    let k31 = u32_const(fb, 31)?;
    let sh = fb.binary(BinOp::ISub, k31, s)?;
    let top = fb.binary(BinOp::ShiftLeftLogical, lo, sh)?;
    let tie_bits = u32_const(fb, 0x4000_0000)?;
    let tie = fb.binary(BinOp::IEqual, top, tie_bits)?;
    let up = fb.select(tie, one, zero)?;
    let bx = fb.binary(BinOp::IAdd, bp, up)?;
    Ok((fb.convert(ConvOp::Bitcast, bx, f)?, nf))
}

/// `u - k` rounded toward positive infinity, or toward negative infinity where the size `n` (a
/// `u32`) is a power of two, for an `f32` coordinate `u` and the integer `k` (an `f32`) of its
/// period's start (`floor(u)`, or `2 floor(u / 2)` mirrored), with `(1, 0)` as `u32`s (module
/// docs). The difference rounded to nearest even, `s`, is inexact only where `-1 < u < 0` (`k` is
/// `-1`, or `-2` mirrored, and `s` lies in `[1/2, 1]`, or `[1, 2]`), and there `z = s + k` is
/// exact by Sterbenz's lemma; elsewhere `s` is exact and `z` is `u`. So the exact difference
/// `u - k` exceeds `s` exactly when `u > z`, and then the next `f32` up from `s` (which is not
/// negative) is its bits plus one; it lies below `s` exactly when `u < z`, and then the next `f32`
/// down is its bits minus one. The three float operations are `NoContraction`: a device rounds each
/// on its own and never folds `(u - k) + k` to `u`.
fn wrap(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    k: SValue,
    n: SValue,
    (one, zero): (SValue, SValue),
) -> BResult<SValue> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let w = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let s = fb.binary_no_contraction(BinOp::FSub, u, k)?;
    let z = fb.binary_no_contraction(BinOp::FAdd, s, k)?;
    let below = fb.binary(BinOp::FOrdLessThan, z, u)?;
    let above = fb.binary(BinOp::FOrdLessThan, u, z)?;
    let up = fb.select(below, one, zero)?;
    let minus_one = u32_const(fb, u32::MAX)?;
    let down = fb.select(above, minus_one, zero)?;
    // A power of two, `n & (n - 1) == 0`, wraps exactly (module docs).
    let m = fb.binary(BinOp::ISub, n, one)?;
    let m = fb.binary(BinOp::BitwiseAnd, n, m)?;
    let power_of_two = fb.binary(BinOp::IEqual, m, zero)?;
    let step = fb.select(power_of_two, down, up)?;
    let bs = fb.convert(ConvOp::Bitcast, s, w)?;
    let bs = fb.binary(BinOp::IAdd, bs, step)?;
    fb.convert(ConvOp::Bitcast, bs, f)
}

/// A gather's base texel `i0` (an `f32`) at position `x`: `floor(floor(256 x + 1/2) / 256 - 1/2)`,
/// the `q >> 8`, exact for `|x| < 2^15` (module docs).
fn gather_base(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let k256 = f32_const(fb, 256.0)?;
    let y = fb.binary(BinOp::FMul, x, k256)?;
    let half = f32_const(fb, 0.5)?;
    let y = fb.binary(BinOp::FAdd, y, half)?;
    let z = fb.glsl(GlslOp::Floor, &[y])?;
    let step = f32_const(fb, 1.0 / 256.0)?;
    let z = fb.binary(BinOp::FMul, z, step)?;
    let z = fb.binary(BinOp::FSub, z, half)?;
    fb.glsl(GlslOp::Floor, &[z])
}

/// Component `i` of `v`, or `v` itself for a one-lane value.
fn lane(fb: &mut FunctionBuilder<'_>, v: SValue, lanes: u8, i: u32) -> BResult<SValue> {
    if lanes == 1 {
        Ok(v)
    } else {
        fb.composite_extract(v, &[i])
    }
}

/// The snapped axis coordinate (an `f32`) of coordinate `u` (normalized, or in texels where
/// `pixel`, Metal's position as given) on an axis of size `n` (a `u32`) addressed by `address`,
/// for `kind` (module docs), whose access the device moves by the constant texel offset `offset`
/// (an `f32` holding the axis's offset; `None`: none): on a repeating or mirrored axis whose size
/// drops it ([`offset_dropped`], whose `parent` is the axis's `texture::OffsetBase` test), the
/// index is moved back by it first.
pub fn snapped_axis(
    fb: &mut FunctionBuilder<'_>,
    (kind, pixel): (AxisSnap, bool),
    u: SValue,
    n: SValue,
    (address, offset, parent): (TapAddress, Option<SValue>, Option<SValue>),
) -> BResult<SValue> {
    let (x, nf) = if pixel {
        let f = fb.module().ty(SType::Float { width: 32 })?;
        (u, fb.convert(ConvOp::ConvertUToF, n, f)?)
    } else {
        position(fb, u, n, address)?
    };
    let at = match kind {
        AxisSnap::Gather => {
            let i0 = gather_base(fb, x)?;
            let i0 = match address {
                TapAddress::Edge | TapAddress::Border => {
                    let lo = f32_const(fb, -10.0)?;
                    let eight = f32_const(fb, 8.0)?;
                    let hi = fb.binary(BinOp::FAdd, nf, eight)?;
                    fb.glsl(GlslOp::FClamp, &[i0, lo, hi])?
                }
                TapAddress::Repeat | TapAddress::Mirror => i0,
            };
            let one = f32_const(fb, 1.0)?;
            fb.binary(BinOp::FAdd, i0, one)?
        }
        AxisSnap::Nearest => {
            let i = fb.glsl(GlslOp::Floor, &[x])?;
            let half = f32_const(fb, 0.5)?;
            fb.binary(BinOp::FAdd, i, half)?
        }
        // `q + 128 = floor(256 X) + [frac(256 X) >= 1/2]`, as `bicubic::quantize` reads `q`:
        // `256 X`, its floor, and its fraction are exact, where `256 X - 127.5` can round (a
        // negative `X` whose sum carries into the next binade). The centre `(q + 128) / 256` is
        // exact below `|X| = 2^15`.
        AxisSnap::Cell => {
            let k256 = f32_const(fb, 256.0)?;
            let y = fb.binary(BinOp::FMul, x, k256)?;
            let fl = fb.glsl(GlslOp::Floor, &[y])?;
            let r = fb.binary(BinOp::FSub, y, fl)?;
            let half = f32_const(fb, 0.5)?;
            let up = fb.binary(BinOp::FOrdGreaterThanEqual, r, half)?;
            let one = f32_const(fb, 1.0)?;
            let zero = f32_const(fb, 0.0)?;
            let up = fb.select(up, one, zero)?;
            let c = fb.binary(BinOp::FAdd, fl, up)?;
            let step = f32_const(fb, 1.0 / 256.0)?;
            fb.binary(BinOp::FMul, c, step)?
        }
    };
    let at = match (address, offset) {
        (TapAddress::Repeat | TapAddress::Mirror, Some(o)) => {
            let dropped = offset_dropped(fb, n, None, parent)?;
            let none = f32_const(fb, 0.0)?;
            let back = fb.select(dropped, o, none)?;
            fb.binary(BinOp::FSub, at, back)?
        }
        _ => at,
    };
    fb.binary(BinOp::FDiv, at, nf)
}

/// The coordinate `coord` (`lanes` floats; in texels where `snap.pixel`, else normalized) of a
/// sample or gather of sampled image `si` (image facts `image`) snapped by `snap` (module docs) on
/// the base level's size: a normalized coordinate. `offsets` holds, per plane axis, the constant
/// texel offset the device adds to the access (an `f32`, or `None`; [`snapped_axis`]), and
/// `parent` the slot's `SpecKey::OffsetBaseAxes` tests (`texture::OffsetBase`).
pub fn snapped(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    snap: &Snap,
    (coord, offsets, parent): (SValue, &[Option<SValue>], Option<OffsetBase>),
    lanes: u8,
    (si, image): (SValue, CombinedImage),
) -> BResult<SValue> {
    let img = fb.image(si)?;
    let zero = u32_const(fb, 0)?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    snapped_on(cx, fb, snap, (coord, offsets, parent), lanes, (size, image))
}

/// The coordinate `coord` (`lanes` floats) snapped by `snap` (module docs) on the base level's
/// size `size` of an image with facts `image`: a plane coordinate's axes, or a cube's direction
/// on its face's size, the size's first component. `offsets` holds each plane axis's constant
/// texel offset the device adds ([`snapped_axis`]; a cube has none), and `parent` the slot's
/// `SpecKey::OffsetBaseAxes` tests (`texture::OffsetBase`).
pub fn snapped_on(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    snap: &Snap,
    (coord, offsets, parent): (SValue, &[Option<SValue>], Option<OffsetBase>),
    lanes: u8,
    (size, image): (SValue, CombinedImage),
) -> BResult<SValue> {
    let size_lanes = image.plane + u8::from(image.arrayed);
    let kind = match snap.kind {
        SnapKind::Axes(kind) => kind,
        SnapKind::Cube(routines) => {
            let routines = routines.declared(cx)?;
            let n = lane(fb, size, size_lanes, 0)?;
            return super::face::snapped(fb, coord, n, (routines, snap.face_wrap()));
        }
        SnapKind::Word { .. } | SnapKind::WordGather { .. } | SnapKind::WordCell { .. } => {
            return Err(cx.fail("a word snap without its sampler's word"))
        }
    };
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for (a, axis) in (0u32..).zip(&snap.axes).take(usize::from(lanes)) {
        let u = lane(fb, coord, lanes, a)?;
        parts.push(match axis {
            Some(address) => {
                let n = lane(fb, size, size_lanes, a)?;
                let offset = offsets.get(a as usize).copied().flatten();
                let parent = OffsetBase::axis(parent, a);
                snapped_axis(fb, (kind, snap.pixel), u, n, (*address, offset, parent))?
            }
            None => u,
        });
    }
    if let [x] = parts[..] {
        return Ok(x);
    }
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let ty = fb.module().ty(SType::Vector {
        elem: f,
        len: lanes,
    })?;
    fb.composite_construct(ty, &parts)
}

/// The coordinate `coord` (`lanes` normalized floats) of a runtime or argument buffer's sampler's
/// nearest sample, its sampler's word `word`, snapped by [`SnapKind::Word`] on the size `size`
/// (`size_lanes` components) of the level it reads: where the word masked by
/// `SamplerWord::NEAREST_BASE` is 0 (normalized coordinates, nearest filters, no mip filter, no
/// anisotropy: one texel of the base level, whatever the sampler's LOD clamps), or, for a sample at
/// a level the function knows (`leveled`), the word masked by `SamplerWord::NEAREST_LEVEL` is (no
/// mip filter or a nearest one: one texel of the level `level::snap_level` names), each plane axis
/// whose mode the word states as `Repeat`, `MirrorRepeat` or `MirrorClampToEdge` takes the centre
/// `(floor(X) + 1/2) / N` of the texel at Metal's position `X` ([`position_by_mode`]; a
/// mirror-clamped axis's taps the device mirrors once and clamps as Metal does); every other axis,
/// and every axis of any other sampler, keeps its coordinate, the device's as before. A snapped
/// axis whose size drops a constant texel offset (`offsets`, per plane axis an `f32` or `None`;
/// [`offset_dropped`]) moves the centre back by it, as [`snapped_axis`] does: a leveled snap's by
/// the level's size and the base level's, `base` (`size_lanes` components; `None`: the level is the
/// base level), a mirror-clamped axis's as a repeating one's, and by the slot's
/// `SpecKey::OffsetBaseAxes` tests `parent` (`texture::OffsetBase`).
pub fn snapped_by_word(
    fb: &mut FunctionBuilder<'_>,
    (word, leveled): (SValue, bool),
    (coord, offsets, parent): (SValue, &[Option<SValue>], Option<OffsetBase>),
    lanes: u8,
    (size, base, size_lanes): (SValue, Option<SValue>, u8),
) -> BResult<SValue> {
    use crate::metal::sampler::SamplerWord;
    let zero = u32_const(fb, 0)?;
    let reach = if leveled {
        SamplerWord::NEAREST_LEVEL
    } else {
        SamplerWord::NEAREST_BASE
    };
    let mask = u32_const(fb, reach)?;
    let masked = fb.binary(BinOp::BitwiseAnd, word, mask)?;
    let reach = fb.binary(BinOp::IEqual, masked, zero)?;
    let modes = u32_const(fb, 0b111)?;
    let repeat = u32_const(fb, SamplerAddressMode::Repeat.as_raw())?;
    let mirror = u32_const(fb, SamplerAddressMode::MirrorRepeat.as_raw())?;
    let mirror_clamp = u32_const(fb, SamplerAddressMode::MirrorClampToEdge.as_raw())?;
    let half = f32_const(fb, 0.5)?;
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for (a, shift) in (0u32..)
        .zip(SamplerWord::ADDRESS_SHIFT)
        .take(usize::from(lanes))
    {
        let u = lane(fb, coord, lanes, a)?;
        let n = lane(fb, size, size_lanes, a)?;
        let sh = u32_const(fb, shift)?;
        let mode = fb.binary(BinOp::ShiftRightLogical, word, sh)?;
        let mode = fb.binary(BinOp::BitwiseAnd, mode, modes)?;
        let (x, nf) = position_by_mode(fb, u, n, mode)?;
        let i = fb.glsl(GlslOp::Floor, &[x])?;
        let centre = fb.binary(BinOp::FAdd, i, half)?;
        // Only a repeating, mirrored or mirror-clamped axis is snapped, so the sizes' rule alone
        // decides.
        let centre = match offsets.get(a as usize).copied().flatten() {
            Some(o) => {
                let nb = match base {
                    Some(b) => Some(lane(fb, b, size_lanes, a)?),
                    None => None,
                };
                let dropped = offset_dropped(fb, n, nb, OffsetBase::axis(parent, a))?;
                let none = f32_const(fb, 0.0)?;
                let back = fb.select(dropped, o, none)?;
                fb.binary(BinOp::FSub, centre, back)?
            }
            None => centre,
        };
        let centre = fb.binary(BinOp::FDiv, centre, nf)?;
        let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
        let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
        let clamps_once = fb.binary(BinOp::IEqual, mode, mirror_clamp)?;
        let wraps = fb.binary(BinOp::LogicalOr, repeats, mirrors)?;
        let wraps = fb.binary(BinOp::LogicalOr, wraps, clamps_once)?;
        let snaps = fb.binary(BinOp::LogicalAnd, reach, wraps)?;
        parts.push(fb.select(snaps, centre, u)?);
    }
    composite(fb, parts)
}

/// The plane coordinate of `parts` (one `f32` per plane axis).
fn composite(fb: &mut FunctionBuilder<'_>, parts: Vec<SValue>) -> BResult<SValue> {
    if let [x] = parts[..] {
        return Ok(x);
    }
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let ty = fb.module().ty(SType::Vector {
        elem: f,
        len: parts.len() as u8,
    })?;
    fb.composite_construct(ty, &parts)
}

/// Whether sampler word `word` (a `u32`) masked by `mask` holds every bit of `mask`, a `bool`.
fn word_holds(fb: &mut FunctionBuilder<'_>, word: SValue, mask: u32) -> BResult<SValue> {
    let m = u32_const(fb, mask)?;
    let have = fb.binary(BinOp::BitwiseAnd, word, m)?;
    fb.binary(BinOp::IEqual, have, m)
}

/// The coordinate of a runtime or argument buffer's sampler's gather of a 2D texture or array,
/// its sampler's word `word`, snapped by [`SnapKind::WordGather`] on the base level's size `size`
/// (`size_lanes` components): per plane axis the centre `(i0 + 1) / N` of the 2x2 block Metal reads
/// ([`AxisSnap::Gather`]), its base texel `i0` at Metal's position `X`, the coordinate as given
/// (`given`, `lanes` floats) where the word states pixel coordinates, else [`position_by_mode`]'s
/// by the axis's mode: a `MirrorClampToEdge` axis's is a clamping axis's, `u N` unwrapped, whose
/// taps the device mirrors once and clamps as Metal does (Apple M3 Max and M4). `i0` is clamped
/// into `[-2N - 10, 2N + 8]`, which holds every position of a repeating or mirrored axis, and past
/// which, with any offset in Metal's gather range `[-8, 7]`, both texels of a clamping or
/// mirror-clamped axis read the same edge texel or border, as the exact `i0`'s do. A repeating,
/// mirrored or mirror-clamped axis whose size drops a constant texel offset (`offsets`, per plane
/// axis an `f32` or `None`; [`offset_dropped`]) moves the centre back by it, as [`snapped_axis`]
/// does, a pixel word's mirror-clamped axis's too. On a device that samples through
/// Metal's own sampler (`metal`) only a pixel coordinate is snapped, and no offset moves it:
/// Metal's own sampler drops an offset of the normalized centre where it drops the pixel
/// coordinate's; a normalized one is the coordinate the device reads otherwise (`scaled`, `lanes`
/// floats), which Metal's own gather reads at Metal's position.
pub fn gathered_by_word(
    fb: &mut FunctionBuilder<'_>,
    (word, metal): (SValue, bool),
    (given, scaled, offsets, parent): (SValue, SValue, &[Option<SValue>], Option<OffsetBase>),
    lanes: u8,
    (size, size_lanes): (SValue, u8),
) -> BResult<SValue> {
    use crate::metal::sampler::SamplerWord;
    let zero = u32_const(fb, 0)?;
    let mask = u32_const(fb, SamplerWord::PIXEL_COORDINATES)?;
    let p = fb.binary(BinOp::BitwiseAnd, word, mask)?;
    let pixel = fb.binary(BinOp::INotEqual, p, zero)?;
    let modes = u32_const(fb, 0b111)?;
    let repeat = u32_const(fb, SamplerAddressMode::Repeat.as_raw())?;
    let mirror = u32_const(fb, SamplerAddressMode::MirrorRepeat.as_raw())?;
    let mirror_clamp = u32_const(fb, SamplerAddressMode::MirrorClampToEdge.as_raw())?;
    let lo = f32_const(fb, -10.0)?;
    let eight = f32_const(fb, 8.0)?;
    let one = f32_const(fb, 1.0)?;
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for (a, shift) in (0u32..)
        .zip(SamplerWord::ADDRESS_SHIFT)
        .take(usize::from(lanes))
    {
        let u = lane(fb, given, lanes, a)?;
        let device = lane(fb, scaled, lanes, a)?;
        let n = lane(fb, size, size_lanes, a)?;
        let sh = u32_const(fb, shift)?;
        let mode = fb.binary(BinOp::ShiftRightLogical, word, sh)?;
        let mode = fb.binary(BinOp::BitwiseAnd, mode, modes)?;
        let (x, nf) = position_by_mode(fb, u, n, mode)?;
        let x = fb.select(pixel, u, x)?;
        let i0 = gather_base(fb, x)?;
        let twice = fb.binary(BinOp::FAdd, nf, nf)?;
        let hi = fb.binary(BinOp::FAdd, twice, eight)?;
        let low = fb.binary(BinOp::FSub, lo, twice)?;
        let i0 = fb.glsl(GlslOp::FClamp, &[i0, low, hi])?;
        let at = fb.binary(BinOp::FAdd, i0, one)?;
        let at = match offsets.get(a as usize).copied().flatten() {
            // Metal's own sampler places the offset of the normalized coordinate it is handed as
            // Metal places a pixel coordinate's (a pixel word's axis clamps, or mirrors once and
            // clamps, which drops the offset at the sizes a normalized one does).
            Some(_) if metal => at,
            Some(o) => {
                // Metal drops the offset on a repeating, mirrored, or mirror-clamped axis whose
                // size drops it (the mirror clamp's and a pixel word's measured by our probes).
                let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
                let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
                let clamps = fb.binary(BinOp::IEqual, mode, mirror_clamp)?;
                let wraps = fb.binary(BinOp::LogicalOr, repeats, mirrors)?;
                let wraps = fb.binary(BinOp::LogicalOr, wraps, clamps)?;
                let dropped = offset_dropped(fb, n, None, OffsetBase::axis(parent, a))?;
                let dropped = fb.binary(BinOp::LogicalAnd, dropped, wraps)?;
                let none = f32_const(fb, 0.0)?;
                let back = fb.select(dropped, o, none)?;
                fb.binary(BinOp::FSub, at, back)?
            }
            None => at,
        };
        let centre = fb.binary(BinOp::FDiv, at, nf)?;
        parts.push(match metal {
            true => fb.select(pixel, centre, device)?,
            false => centre,
        });
    }
    composite(fb, parts)
}

/// The coordinate of a runtime or argument buffer's sampler's sample on a device that samples
/// through Metal's own sampler, its sampler's word `word`, snapped by [`SnapKind::WordCell`] on the
/// base level's size `size` (`size_lanes` components): where the word states pixel coordinates and
/// linear min and mag filters, each plane axis of the coordinate as given (`given`, `lanes` floats)
/// onto the centre of its footprint step ([`AxisSnap::Cell`]); for every other word the coordinate
/// the device reads otherwise (`scaled`). A pixel sampler clamps every axis or mirrors it once and
/// clamps it (`runtime::sampler`'s rule), and Metal's own sampler places a texel offset of the
/// normalized cell as Metal places the pixel coordinate's (kept on a clamping axis, dropped by a
/// mirror-clamped one's sizes alike), so no offset moves the cell.
pub fn celled_by_word(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    (given, scaled): (SValue, SValue),
    lanes: u8,
    (size, size_lanes): (SValue, u8),
) -> BResult<SValue> {
    use crate::metal::sampler::SamplerWord;
    let reach = word_holds(
        fb,
        word,
        SamplerWord::PIXEL_COORDINATES | SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR,
    )?;
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for a in 0..u32::from(lanes) {
        let x = lane(fb, given, lanes, a)?;
        let device = lane(fb, scaled, lanes, a)?;
        let n = lane(fb, size, size_lanes, a)?;
        let cell = snapped_axis(
            fb,
            (AxisSnap::Cell, true),
            x,
            n,
            (TapAddress::Edge, None, None),
        )?;
        parts.push(fb.select(reach, cell, device)?);
    }
    composite(fb, parts)
}

/// The texel-space centre `centre` (`lanes` floats: `floor(x) + 1/2` of the pixel coordinate `x`)
/// of a runtime or argument buffer's pixel sampler's nearest sample, its sampler's word `word`, on
/// a device that adds a texel offset to the texel it reads before addressing it, moved back by each
/// texel offset (`offsets`, per plane axis an `f32` or `None`) that Metal drops: on an axis the
/// word states `MirrorClampToEdge` whose base level's size `size` (`size_lanes` components) drops
/// it ([`offset_dropped`]; Metal mirrors a pixel coordinate's taps once and clamps them, placing
/// its offset as a repeating axis does). Every other axis keeps its centre: a pixel
/// sampler's other axes clamp, where Metal keeps the offset.
pub fn pixel_centre_by_word(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    (centre, offsets, parent): (SValue, &[Option<SValue>], Option<OffsetBase>),
    lanes: u8,
    (size, size_lanes): (SValue, u8),
) -> BResult<SValue> {
    use crate::metal::sampler::SamplerWord;
    let modes = u32_const(fb, 0b111)?;
    let mirror_clamp = u32_const(fb, SamplerAddressMode::MirrorClampToEdge.as_raw())?;
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for (a, shift) in (0u32..)
        .zip(SamplerWord::ADDRESS_SHIFT)
        .take(usize::from(lanes))
    {
        let c = lane(fb, centre, lanes, a)?;
        parts.push(match offsets.get(a as usize).copied().flatten() {
            Some(o) => {
                let n = lane(fb, size, size_lanes, a)?;
                let sh = u32_const(fb, shift)?;
                let mode = fb.binary(BinOp::ShiftRightLogical, word, sh)?;
                let mode = fb.binary(BinOp::BitwiseAnd, mode, modes)?;
                let clamps_once = fb.binary(BinOp::IEqual, mode, mirror_clamp)?;
                let dropped = offset_dropped(fb, n, None, OffsetBase::axis(parent, a))?;
                let dropped = fb.binary(BinOp::LogicalAnd, dropped, clamps_once)?;
                let none = f32_const(fb, 0.0)?;
                let back = fb.select(dropped, o, none)?;
                fb.binary(BinOp::FSub, c, back)?
            }
            None => c,
        });
    }
    composite(fb, parts)
}

/// The census words [`pixel_centre_by_word`] adds per plane axis that a texel offset moves: the
/// extractions, the mode's shift, mask and test, the offset rule ([`OFFSET_RULE_WORDS`]), the
/// conjunction, the select and the subtraction, and their constants.
pub const PIXEL_OFFSET_WORDS: u32 = OFFSET_RULE_WORDS + 72;

#[cfg(test)]
mod tests;
