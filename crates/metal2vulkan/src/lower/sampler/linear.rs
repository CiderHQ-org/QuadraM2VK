//! Metal's linear filter for a sample with a `half` or `float` result, emulated by fetches.
//!
//! **The filter Metal applies** (Apple M3 Max, macOS 26.7; the probe cases and `f8f81b81`). Per
//! plane axis of a sample through a sampler with linear minification and magnification and no mip
//! filter, the texel-space coordinate `x` (Metal's position of the normalized coordinate on the
//! level-0 size: wrapped into one period first on a repeating or mirrored axis, then scaled with
//! ties away from zero, `position`; or the pixel coordinate) is read as the bicubic filter reads it
//! (`bicubic`): `q = floor(256 (x - 1/2) + 1/2)`, the base texel `i = q >> 8` (plus any constant
//! texel offset) and the fraction's 1/256 steps `k = q & 255`. The taps `i` and `i + 1` weigh
//! `(256 - k) / 256` and `k / 256`, the 2D filter is their product, and each tap is addressed on
//! its own: `clamp_to_edge` clamps its index into the level, `repeat` wraps it, `mirrored_repeat`
//! reflects it ([`TapAddress::Mirror`]), and `clamp_to_zero` and `clamp_to_border` replace an
//! out-of-range tap by the sampler's border color seen through the bound format's channels
//! (`clamp_to_zero` reads the border color too; AIR codes the two as one mode, its border color
//! deciding). The level is the base level (no mip filter: Metal reads level 0 whatever the level,
//! bias, gradients, or derivatives), and a 2D array's layer is clamped into the array. The sum is
//! the bound format's, as `metal::texel_format` states: an 8- or 16-bit normalized format's
//! weighted codes (an 8-bit sRGB format's table steps) summed exactly and rounded half up onto its
//! grid, then `n / D` rounded to nearest even, to `f32` for a `float` result or to `half`; a 32-bit
//! float or half-float format's taps summed through Metal's linear accumulator: each term truncated
//! toward zero onto a grid 38 bits (17 for a half-float format) below the largest present term's
//! tap and weight exponents (`e(v) + e(w)`; a tap of zero weight is absent), the truncated terms
//! summed exactly, and the sum rounded once with ties away from zero, a 32-bit float format's to
//! `f32` (a denormal sum `+0`), then for a `half` result toward zero saturating, and a half-float
//! format's to `half`, which a `float` result widens. A result that is zero is `+0`, whatever its
//! taps' signs (a `-0` texel at a texel centre, and a half-float sum that rounds to zero,
//! included). A normalized format's 2D sum truncates its terms first (below).
//!
//! **A normalized format's 2D sum** (Apple M4 and Apple M3 Max, byte-identical; the 2D and cube
//! probes) weighs an axis whose two taps address one texel `(256, 0)`, as a 3D sample does (below),
//! each tap `w = wx · wy` in units of `2^-16`, and truncates each term onto its grid first, as a 3D
//! sum does: `T = Σ ⌊c · w / 2^s⌋` for the codes `c` on the grid (`s` 4 for an 8-bit format's `16 ·
//! code` and an sRGB step, 8 for a 16-bit format's code, toward negative infinity for a signed
//! format), then the grid step `⌊(T + 2^(15 - s)) / 2^(16 - s)⌋`. The truncation moves an sRGB
//! color lane's sum and a 16-bit format's; an 8-bit format's `16 · code` terms lose nothing, so its
//! sum is the exact one. A float format's 2D sum merges such an axis too, before its accumulator
//! (every 2D channel past the edges of `RGBA16Float`, `RGBA32Float`, `RG11B10Float`, and
//! `RGB9E5Float`).
//!
//! **A 3D texture's filter** (Apple M4, macOS 27.0.1, and Apple M3 Max, macOS 26.7, byte-identical;
//! the 3D probes) reads each of the three axes as above and weighs its eight taps differently from
//! the 2D product:
//! - An axis whose two taps address one texel (clamped onto each other at an edge, reflected onto
//!   each other at a mirrored boundary, a level one texel wide, or both outside a border axis)
//!   weighs them `(256, 0)`.
//! - Each tap's weight is `wx · wy · wz / 256` rounded to nearest, in units of `2^-16`. A tie (a
//!   remainder of 128) rounds up on the `t` axis's first tap in Metal's frame and down on its
//!   partner: the first is tap `j`, but tap `j + 1` where the axis mirrors and its taps run
//!   backwards (the reflected half of the period). The weights sum to `2^16`.
//! - A normalized format's sum truncates each term onto its grid first: `T = Σ ⌊c · w / 2^s⌋` for
//!   the codes `c` on the grid (`s` 4 for an 8-bit format's `16 · code` and an sRGB step, 8 for a
//!   16-bit format's code, toward negative infinity for a signed one), then the grid step `⌊(T +
//!   2^(15 - s)) / 2^(16 - s)⌋`. A float format's sum is the accumulator's under the 16-bit
//!   weights, its non-finite taps those of nonzero product weight `wx · wy · wz`, a tap whose
//!   weight rounds to zero among them (`RG11B10Float`'s infinities).
//!
//! **A cube's filter** (`cube`; the cube probes, Apple M4 and Apple M3 Max, byte-identical) reads
//! the face of the direction's major axis at Metal's face position (`face::position`), quantized as
//! in 2D, its four taps unmerged; a tap past one edge of the face reads the adjacent face's texel,
//! and a tap past both edges a texel that depends on the format (a float format's `x` face's corner
//! texel, a normalized one's made-up texel). It ignores the address modes and the border color, and
//! sums as the 2D filter does. The emulation fetches the taps from the slot's 2D-array view of the
//! faces.
//!
//! **The emulation** reads the slot's `SpecKey::TexelFormat` specialization (the bound format's
//! `BicubicTexels` bits where its conversion is measured, else `0`), fetches the footprint's taps
//! of the base level, and sums them through module routines called with the format's bits and
//! the result type: the linear 1D routines (`bicubic::routines::Span`), 2D ones
//! (`bicubic::routines::Quad`, whose exact routine takes the axes' fractions with a merged axis's
//! at 0, and a cube's taps beside `bicubic::routines::Cube`'s fold and corner), and 3D ones
//! (`bicubic::routines::Block`, which take the merged fractions, and whether the `t` axis's taps
//! run backwards). An `EXACT` format takes the exact
//! routine (the codes' sum in integers, rounded onto the grid and divided exactly). A 32-bit float
//! or half-float format takes the accumulated routine, Metal's accumulator in integer operations
//! on the taps' bits. Either routine's value is the result. A format whose conversion is not
//! measured (`TexelFormat::OTHER`) takes the device's sample instead, which the slot's conversion
//! routine (`texture::half`) converts for a `half` result: the emulation and the device's sample
//! are selected on the specialization, so once a device specializes the module the other is dead.
//! The emulated arm of a sample through a texture slot's combined binding is its shape's sampling
//! routine (`sampler::routine`), defined once in the module and called by every sample of the shape
//! with the coordinate, the layer, a cube's direction, and a comparison's reference; the site keeps
//! the branch and the device's sample. A heap image's emulation is the site's own code.
//!
//! **A runtime sampler**. Metal filters a sample through a sampler bound to a slot or held in an
//! argument buffer exactly as through a constexpr sampler of the same state (Apple M4, macOS
//! 27.0.1: 24 cases, 2D linear under every address mode with normalized and pixel coordinates, 2D
//! nearest, 3D, 1D, and cube, on `RGBA8Unorm` and `RGBA32Float`, every lane bit for bit). So a
//! sample through a runtime sampler slot of the shapes the emulation reaches is planned for it too
//! ([`LinearState::Slot`]), and the slot's `SamplerState` specialization carries the word's
//! [`SamplerWord::LINEAR_STATE`] fields: the branch also tests that the word states the measured
//! filter (`reaches_word`), and the taps follow the word's address modes, border color, and
//! coordinates, computed in the function from the specialization (each tap's index under every
//! mode, selected by the axis's mode; a cube's face wrap by its `s` and `t` modes), so once a
//! device specializes the module the arithmetic folds to the bound sampler's. A `MirrorClampToEdge`
//! axis, which only a runtime sampler states, reflects each tap's index once, `-1 - i` for a
//! negative `i`, and clamps it into the level, under pixel and normalized coordinates alike, its
//! constant texel offset placed as on a repeating axis (Apple M4: every lane of nearest and linear
//! samples, gathers and depth comparisons of textures 1 to 8 texels wide); the emulation reaches it
//! on a 1D or 2D texture, and the device filters it on a 3D texture or a cube, where no probe
//! measured it ([`Linear::mirror_clamp_reached`]). Argument-buffer samplers are not reached: their
//! textures are heap elements, whose format no specialization states; the device filters them,
//! under the notes. A slot under its 8-bit sRGB variant is (Metal's samples of `RGBA8Unorm_sRGB`
//! and `BGRA8Unorm_sRGB` through a runtime sampler are its constexpr sampler's of the same state,
//! bit for bit, in 1D, 2D, and 3D): the emulation sums the raw codes fetched through the slot's raw
//! view (`Linear::fetch`) where the word reaches it; where the word reads one texel the device
//! samples the raw view through the same sampler, and Metal's table decodes the code, as a
//! constexpr nearest sample's raw read does; elsewhere the device samples the slot's own binding,
//! the decoding view, under `PrecisionNote::SrgbTexelDecode` ([`decode_when`]). A cube's is
//! inferred from the same equality on the other formats and from its constexpr sampler's raw
//! emulation; its word that reads one texel samples the decoding view too.
//!
//! **Where it is exact**, on the probes: in 1D and 2D for every measured format (a 16-bit
//! normalized format's 2D sum on every channel of probes, under `clamp_to_edge`, `repeat`,
//! `mirrored_repeat`, and `clamp_to_zero`, with `float` and `half` results, and an 8-bit sRGB
//! format's under `clamp_to_edge`). The accumulator is Metal's on every lane of P2a and
//! P2c, both Metal passes, under `clamp_to_edge` and `clamp_to_zero`, with `float` and `half`
//! results, in 1D and 2D, for the 32-bit float and half-float formats (`R32Float`, `RGBA32Float`,
//! `R16Float`, `RGBA16Float`); the other address modes, an array's layer, an offset, and pixel
//! coordinates move the taps only. An offset on a repeating or mirrored axis moves them as Metal
//! places it, which depends on the axis's size (`position::offset_dropped`, dropped at a base-level
//! size of 3, 5, 6, or 7, added before the wrap at every other). In 3D the emulation is exact on
//! every lane of the probes for the 8- and 16-bit normalized, 8-bit sRGB, and 32-bit float formats,
//! under every address mode, a white border, offsets on clamping axes, and non-finite taps, but for
//! one 32-bit float lane in about 180,000 (a cancellation the accumulator's truncation misses), and
//! a half-float format's result is half a `half` ulp off on about 0.3% of channels; an offset on a
//! repeating or mirrored axis is placed by the same rule per axis (the offset probes measured it on
//! 3D textures), and pixel coordinates are inferred to move the taps only, as in 2D. A cube's is
//! exact on every channel of the cube probes where the major magnitude is 1; elsewhere Apple's
//! approximate reciprocal of the major magnitude can move a position by a 1/256 step
//! (`PrecisionNote::CubeFaceReciprocal`, `cube`). These residues are recorded
//! (`PrecisionNote::HalfSampleFilter`, `PrecisionNote::FloatSampleFilter`), with every sample the
//! emulation does not reach (its selection, `select`).
//!
//! **The packed formats** (an Apple M3 Max on macOS 26.7 and an Apple
//! M4 on macOS 27.0.1, byte-identical; `metal::texel_format`). A packed normalized format sums as
//! an 8-bit format does, each channel on its own 12-bit grid (`16 · 255`, `4 · 1023`, or `4095`,
//! the code widened by replicating its bits), its terms truncated onto `2^-4` of a step: the exact
//! routine, every 1D, 2D, and 3D sample of the probes. A small float sums as a half-float format
//! does, a `float` result the `half` sum widened: every 1D and 2D sample, and in 3D the half-float
//! residue above. A packed normalized format's cube sample is not measured and keeps the device's
//! filter (`TexelFormat::packed_exact`).
//!
//! **The 8-bit 4:2:0 YCbCr format** (byte-identical) sums its converted color's steps `k / 2048` on
//! the grid `2048`, each 2D term truncated by `2^7`
//! (`metal::texel_format::PackedTexels::YCbCr2048`): the exact routine, every channel of the
//! probes. Such a texture has one level (Metal makes it only over an IOSurface plane), and Metal's
//! sample of it through a mip filter is its sample through none, whatever the level, bias or
//! gradient: the emulation reaches a runtime sampler's mip filter on its
//! class (`PackedTexels::one_level`, `reaches_word`). A constexpr sampler with a mip filter keeps
//! the device's filter, noted.
//!
//! On a device whose sampler answers the module's samples such a slot's emulation is its YCbCr
//! variant's alone ([`Linear::exact_only`]), on a profile that takes the variant
//! (`ProfileFields::ycbcr_variant`: under `YCbCrFilterPolicy::Exact`, and on a device that samples
//! through Metal's own sampler also under `SamplingPolicy::Exact`; elsewhere the device filters
//! the slot as any other, `YCbCrFilterPolicy::Device`, noted): the binder binds a two-plane YCbCr
//! format there, whose class is `EXACT` or measured by no family, so only an `EXACT` class takes
//! the emulation, through the exact routine alone, and every other class the device's sample,
//! which is Metal's own filter on that device.
//!
//! **Depth textures and comparisons** (the probe, px3 and px4 probes, Apple M4 on macOS 27.0.1).
//! A depth texture's linear sample is its color class's (`metal::texel_format`: `Depth32Float` an
//! `R32Float`'s, `Depth16Unorm` an `R16Unorm`'s), so the emulation answers it by the class, its
//! value the sum's first channel. A linear comparison is an `r8Unorm` sample of the taps' passes as
//! codes 0 and 255, each tap's depth (a border's included) compared with the reference (`compare`):
//! the emulation compares each tap in the shader and sums the passes through the exact routine
//! under `r8Unorm`'s bits, through a constexpr sampler of a 2D texture or array or a cube or cube
//! array (a cube's corner tap the format's: `Depth32Float`'s the `x` face's texel, `Depth16Unorm`'s
//! the made-up code, compared as a texel).
//!
//! **On a device that samples through Metal's own sampler** (`VkProperty::MetalSampler`: MoltenVK
//! on Apple's GPU) the device's filter is Metal's, so no sample is emulated: the module leaves
//! every filtered sample to the device and requires the property.

use crate::air::abi::{
    AddressMode, BorderColor, CompareFunction, Coordinates, Filter, MipFilter, SamplerState,
};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::air::ir::ValueId;
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::{any_of, both, NoteClause, NoteOn, NoteTest, SelectCx};
use crate::lower::texture::half::HalfForm;
use crate::lower::texture::OffsetBase;
use crate::lower::texture::{half_class, ImageRef, TexelConv, TexelForm};
use crate::metal::bicubic::BicubicTexels;
use crate::metal::sampler::SamplerWord;
use crate::metal::texel_format::{texel_format, PackedTexels, TexelFormat};
use crate::metal::SamplerAddressMode;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, FuncId, TypeId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, Type as SType, UnOp};

use super::bicubic::routines::Routine;
use super::bicubic::{
    self, flag, footprint, footprint_from, footprint_on, scalar, value, Addressing, At, Bits,
    Footprint, Tap, TapAddress, TapBase, Taps, Texels, Tys,
};
use super::face::{FaceRoutines, FaceWrap};
use super::routine::{Emulation, Shape};
use super::{CombinedImage, Lod, Offset, SampleOp, Sampling, Source};
pub use compare::Comparison;
use compare::TapCompare;

pub(super) mod compare;
mod cube;

pub(super) use cube::select_gather;
pub use cube::CubeGather;

/// A linear sample with a `half` result emulated by fetches (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Linear {
    /// The texture whose bound format's treatment the emulation reads: a slot's `TexelFormat`
    /// specialization, or a heap texture's class, which its handle carries.
    pub image: LinearImage,
    /// The sampler's state the taps follow.
    pub state: LinearState,
    /// The constant texel offset Metal applies, per plane axis.
    pub offset: Option<Vec<i32>>,
    /// The sample reads an 8-bit sRGB texture's raw codes, its slot in the `SrgbTexture`
    /// variant: each tap's color lanes become the table's steps (`srgb_raw_tap`), and the exact
    /// routine sums them on the sRGB grids (`BicubicTexels::SRGB`), whatever the device. Through a
    /// runtime sampler slot it does so where the word reaches the emulation, its taps fetched
    /// through [`fetch`](Self::fetch), the slot's raw view, which its device arm samples where the
    /// word reads one texel; the sample's own binding is the decoding view, which its device arm
    /// samples elsewhere.
    pub raw: bool,
    /// A cube's sample (`cube`), with its face position's routines: its taps are fetched from the
    /// faces of the slot's 2D-array view (its raw view where `raw`), its state's addressing is its
    /// face's `s` and `t` axes' (a constexpr sampler's `address`, or a runtime sampler's word's),
    /// of which a repeating one wraps a face coordinate of 1 (`face::FaceWrap`), and its border,
    /// coordinates, and `offset` are not read (Metal reads a cube's other address modes as
    /// clamping, and a cube sample has no offset).
    pub cube: Option<FaceRoutines>,
    /// A depth texture's comparison (`compare`): its sampler's compare function (a runtime
    /// sampler's its word's), under which each tap's depth (a border's included) is compared with
    /// the reference, the passes summed as an `r8Unorm` sample's codes 0 and 255, the result a
    /// scalar.
    pub compare: Option<Comparison>,
    /// The binding a 2D comparison's taps are fetched through, the slot's without comparison
    /// (`BindingKey::compare` false, set by `sampler::select`): the sample's own binding compares,
    /// and through it a `Depth32Float` texture held with a twin reads the twin, whose depth view
    /// ANV clamps into `[0, 1]` (runtime lane `depth_compares`: 30 of 3,392 comparisons of texels
    /// outside it differed). A runtime sampler's raw emulation fetches its taps through the slot's
    /// raw view (`BindingKey::raw`). `None`: the taps are fetched through the sample's binding (a
    /// cube's through its faces' view).
    pub fetch: Option<super::BindingKey>,
    /// The emulation of a slot under its YCbCr variant on a device whose sampler answers the
    /// module's samples (`DeviceProfile::device_samples`): only an `EXACT` class takes it, through
    /// the exact routine alone, and every other class the device's sample, Metal's own filter. The
    /// binder puts a slot in the variant only where it binds a two-plane YCbCr format there
    /// (`bind::plan::slots::ycbcr_variant`), whose class is `EXACT` or measured by no family
    /// (`metal::texel_format`'s `two_plane_family`), never the accumulated routine's: so the
    /// emulated routine, which no such class reads, is not called.
    pub exact_only: bool,
    /// The device's filter answers the native classes (`native`): on a profile that takes
    /// `DriverDeviation::Unorm8LinearFilter`, a 2D slot's sample through a constexpr or runtime
    /// sampler slot hands an 8-bit unsigned normalized class to the device at Metal's footprint
    /// step (`native::coordinate`) where it would otherwise take the emulation.
    pub native: bool,
}

/// The state of the sampler a linear sample's emulation follows (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LinearState {
    /// A constexpr sampler's, known at translation.
    Constant {
        /// The addressing of each plane axis (`s`, then `t` for a 2D texture, then `r` for a 3D
        /// one); none for a cube.
        address: Vec<TapAddress>,
        /// The sampler's border color, which every axis clamping to zero or to the border reads.
        border: BorderColor,
        /// Pixel coordinates: the plane coordinate is already in texels.
        pixel: bool,
    },
    /// The runtime sampler in slot `sampler`, on `axes` plane axes: its `SamplerState`
    /// specialization's word states whether Metal's measured filter applies, and each axis's
    /// address mode, the border color, and the coordinates the emulation reads.
    Slot { sampler: u32, axes: u8 },
    /// An argument buffer's sampler, whose handle `sampler` is the index of its sampler table
    /// entry, on `axes` plane axes: the entry's word states what a slot's does.
    Argument { sampler: ValueId, axes: u8 },
}

/// The texture whose bound format's treatment a linear sample's emulation reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinearImage {
    /// Texture slot `n`, whose `TexelFormat` specialization states the bound format's class.
    Slot(u32),
    /// The sample's heap texture, whose handle carries its class's index
    /// (`memory::handles::HANDLE_TEXEL_CLASS`).
    Heap,
}

impl LinearImage {
    /// The texture slot, or `None` for a heap texture.
    pub(super) fn slot(self) -> Option<u32> {
        match self {
            LinearImage::Slot(n) => Some(n),
            LinearImage::Heap => None,
        }
    }
}

impl Linear {
    /// Whether a runtime or argument buffer's sampler's word states the sampler
    /// ([`LinearState::Slot`], [`LinearState::Argument`]).
    pub(super) fn word_state(&self) -> bool {
        matches!(
            self.state,
            LinearState::Slot { .. } | LinearState::Argument { .. }
        )
    }

    /// Whether the emulation reaches a runtime sampler whose word states `MirrorClampToEdge` on a
    /// plane axis: a 1D or 2D texture's, an array's or not (Metal reflects each tap's index once,
    /// `-1 - i` for a negative `i`, then clamps it into the level, under pixel and normalized
    /// coordinates alike, its constant texel offset placed as on a repeating axis); not a 3D
    /// texture's, whose mirrored `t` axis's tie rule no probe measured
    /// under the mode, nor a cube's.
    pub(super) fn mirror_clamp_reached(&self) -> bool {
        self.cube.is_none() && self.plane_axes() <= 2
    }

    /// The class bits of which one names a family the emulation takes: `EXACT`, `LINEAR`, or
    /// `HALF`; only `EXACT` under [`exact_only`](Self::exact_only).
    pub(super) fn measured(&self) -> u32 {
        match self.exact_only {
            true => BicubicTexels::EXACT,
            false => BicubicTexels::EXACT | BicubicTexels::LINEAR | BicubicTexels::HALF,
        }
    }

    /// The plane axes: 1, 2 (a cube's face's included), or 3.
    pub(super) fn plane_axes(&self) -> usize {
        match &self.state {
            LinearState::Constant { address, .. } if self.cube.is_none() => address.len(),
            LinearState::Constant { .. } => 2,
            LinearState::Slot { axes, .. } | LinearState::Argument { axes, .. } => {
                usize::from(*axes)
            }
        }
    }

    /// The routines the emulation calls, by dimension: the linear 1D, 2D, or 3D routines, or a
    /// cube's 2D ones and its fold and corner.
    pub(super) fn routines(&self) -> &'static [Routine] {
        match (self.cube, self.plane_axes()) {
            (Some(_), _) => &[Routine::Quad, Routine::Cube],
            (None, 1) => &[Routine::Span],
            (None, 2) => &[Routine::Quad],
            (None, _) => &[Routine::Block],
        }
    }

    /// The census words beyond the device's sample: the footprint (each axis's position, which a
    /// pixel coordinate does not take), per tap its addressing and fetch, the calls of the
    /// routines, whose own words the module counts once ([`Routine::words`]), the branch, the
    /// conversion's call, and the select, and a raw sample's tap conversions; for a runtime sampler
    /// the word's tests ([`WORD_WORDS`]), each axis's position under its mode
    /// ([`super::position::MODE_POSITION_WORDS`]), and each axis's taps under every mode
    /// ([`WORD_TAP_WORDS`]); and Metal's offset rule on each axis a constant offset moves
    /// (`position::OFFSET_RULE_WORDS`). The estimate bounds every emitted form
    /// (`the_linear_estimate_bounds_its_emission`, `the_raw_estimate_bounds_its_emission`).
    pub(super) fn words(&self) -> u32 {
        if let Some(face) = self.cube {
            let wrap = match &self.state {
                LinearState::Constant { address, .. } => {
                    FaceWrap::of(address.first().copied(), address.get(1).copied()).words()
                }
                LinearState::Slot { .. } | LinearState::Argument { .. } => {
                    WORD_WORDS + super::face::WORD_WRAP_WORDS
                }
            };
            let compare = self.compare.map_or(0, |c| c.words(4));
            return cube::WORDS
                + face.use_words()
                + wrap
                + compare
                + if self.raw {
                    cube::TAPS * crate::lower::texture::SRGB_TAP_WORDS
                } else {
                    0
                };
        }
        let axes = self.plane_axes() as u32;
        let (base, taps) = match axes {
            1 => (LINE_WORDS, 2),
            2 => (PLANE_WORDS, 4),
            _ => (BLOCK_WORDS, 8),
        };
        let position = match &self.state {
            LinearState::Constant { pixel: true, .. } => 0,
            LinearState::Constant { .. } => axes * super::position::POSITION_WORDS,
            LinearState::Slot { .. } | LinearState::Argument { .. } => {
                WORD_WORDS
                    + axes
                        * (super::position::POSITION_WORDS
                            + super::position::MODE_POSITION_WORDS
                            + 2 * WORD_TAP_WORDS)
            }
        };
        // A heap texture's class from its handle, its sampled image loaded again in the
        // emulation's arm, an argument buffer's sampler's word from its entry, and the branch.
        let heap = match (self.image, &self.state) {
            (LinearImage::Slot(_), _) => 0,
            (LinearImage::Heap, LinearState::Argument { .. }) => {
                HEAP_WORDS + crate::memory::handles::ENTRY_FIELD_WORDS
            }
            (LinearImage::Heap, _) => HEAP_WORDS,
        };
        let compare = self.compare.map_or(0, |c| c.words(taps));
        // Metal's offset rule on each axis an offset moves.
        let offset = self
            .offset
            .as_deref()
            .map_or(0, |o| o.iter().filter(|o| **o != 0).count() as u32)
            * super::position::OFFSET_RULE_WORDS;
        base + position
            + compare
            + offset
            + heap
            + if self.raw {
                taps * crate::lower::texture::SRGB_TAP_WORDS
            } else {
                0
            }
    }

    /// The census words a native class's test and snapped coordinate add at the sample's own site
    /// (`native`): its code is the site's, not its emulation's routine's, so
    /// [`words`](Self::words), which the routine counts, leaves it out.
    pub(super) fn native_words(&self) -> u32 {
        match self.native {
            true => super::native::WORDS,
            false => 0,
        }
    }
}

/// The census words a runtime sampler's emulation adds once per sample: the word's filter test and
/// each axis's mode test (`reaches_word`), each axis's mode and the coordinates' extraction, the
/// border color's selection, and a 3D sample's reflected-axis test, with their constants.
pub(super) const WORD_WORDS: u32 = 448;

/// The census words a runtime sampler's tap adds to a constexpr sampler's: the tap's index under
/// each of its modes (`MirrorClampToEdge`'s included), the mode's tests and selects, and its border
/// test.
pub(super) const WORD_TAP_WORDS: u32 = 160;

/// The census words a heap texture's emulation adds: its class's read
/// (`texture::HEAP_CLASS_WORDS`), its element's access chain, load, and sampled image in the
/// emulation's arm, and the branch and its merge, with their constants.
pub(super) const HEAP_WORDS: u32 = crate::lower::texture::HEAP_CLASS_WORDS + 64;

/// [`Linear::words`] of a 1D sample.
pub(super) const LINE_WORDS: u32 = 320;
/// [`Linear::words`] of a 2D sample.
pub(super) const PLANE_WORDS: u32 = 576;
/// [`Linear::words`] of a 3D sample.
pub(super) const BLOCK_WORDS: u32 = 1152;

/// The plan of sample `op` of `image` (shape `shape`) through `sampler` (`None`: an argument
/// buffer's, chosen at run time, whose handle is `argument`), with texel form `form` (`observed`:
/// the function observes its residency), on a device without `VkProperty::MetalSampler`: the
/// emulation where it reaches the sample, else `None`, the device's filter. The filtered sample's
/// note ([`filter_note`]) is the caller's, once it has settled the plan ([`filter_when`]). On a
/// device with the property the module requires it for each such sample instead
/// ([`super::SamplerPlan::metal_sampler`]).
///
/// The emulation reaches a sample through a constexpr sampler with linear minification and
/// magnification, no mip filter, and no anisotropy, of a 1D, 2D or 3D texture slot (an array
/// or not, a depth texture's included), or a cube or cube array, with a `half` or `float` result, a
/// comparison only of a 2D or cube depth texture or array (each tap compared in the shader,
/// `TapCompare`), a constant texel offset or none, no `min_lod_clamp`, and no observed residency,
/// whose axes do not clamp one to zero and another to a colored border. A sample through a
/// constexpr sampler that reads one texel of one level ([`unfiltered`]) is not filtered: the
/// device's texel is converted exactly, so it neither records a note nor requires the property.
///
/// A sample of the same shapes through a runtime sampler slot is planned for the emulation too
/// ([`LinearState::Slot`]): Metal filters through a runtime sampler as through a constexpr sampler
/// of the same state, and whether the bound sampler's state is the one the
/// emulation reaches (its filter, and for a 3D texture or a cube on each plane axis an address mode
/// other than `MirrorClampToEdge`) is read from the slot's `SamplerState` specialization in the
/// function (`reaches_word`). Under a slot's 8-bit sRGB variant the caller makes the emulation raw,
/// its taps fetched through the slot's raw view (`Linear::fetch`).
///
/// A heap texture's sample of those shapes but a cube's, without comparison, through a
/// constexpr sampler, a runtime sampler slot, or an argument buffer's sampler
/// ([`LinearState::Argument`], whose entry's word states what a slot's does) is planned for the
/// emulation too ([`LinearImage::Heap`]): Metal filters a texture bound through an argument buffer
/// as one bound to a slot (our probe sampled heap textures through its argument buffers'
/// samplers), and the class the emulation reads is the one the handle carries
/// (`AccessPlan::heap_class`): a slot's class for a texture without a swizzle, else
/// `TexelFormat::OTHER`, which the device filters.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    form: Option<TexelForm>,
    (sampler, image, shape): (Option<SamplerRef>, ImageRef, ImageShape),
    (argument, observed): (Option<ValueId>, bool),
) -> Result<Option<Linear>, Error> {
    let SampleOp::Sample {
        lod,
        dref,
        min_lod,
        offset,
    } = op
    else {
        return Ok(None);
    };
    if !matches!(form.map(|f| f.conv), Some(TexelConv::F16 | TexelConv::F32)) {
        return Ok(None);
    }
    let state = match sampler {
        Some(SamplerRef::Constexpr(words)) => Some(super::state_of(words)?),
        Some(SamplerRef::Slot(_)) | None => None,
    };
    if state.as_ref().is_some_and(unfiltered) {
        return Ok(None);
    }
    let slot = match sampler {
        Some(SamplerRef::Slot(n)) => Some(n),
        _ => None,
    };
    // On a device that samples through Metal's own sampler, the device's filter is Metal's, but
    // for a slot under its YCbCr variant; on one that samples natively, the device's filter is the
    // answer, but for a slot under its YCbCr variant where the policy asks for Metal's filter of
    // the format (`ProfileFields::ycbcr_variant`).
    let metal_sampler = cx.profile.device_samples();
    if metal_sampler && !crate::lower::texture::ycbcr_texture(cx, image, shape)? {
        cx.families.sampler.metal_sampler = true;
        return Ok(None);
    }
    // A depth texture's sample is its color class's; its comparison compares each tap in the
    // shader, through a constexpr sampler of a 2D texture or array or a cube or cube array (the
    // frontend writes an unspecified compare function as `never`, and no `None`), or through a
    // runtime sampler slot by its word's function, of a 2D texture or array.
    let compare = match (dref, &state, slot) {
        (None, _, _) => None,
        (Some(_), Some(s), _)
            if matches!(shape.dim, ImageDim::D2 | ImageDim::Cube) && shape.depth =>
        {
            match s.compare {
                CompareFunction::None => return Ok(None),
                f => Some(Comparison::Function(f)),
            }
        }
        (Some(_), None, Some(_)) if shape.dim == ImageDim::D2 && shape.depth && !shape.ms => {
            Some(Comparison::Word)
        }
        (Some(_), _, _) => return Ok(None),
    };
    // A heap texture's class is its handle's: not a cube's, whose faces no heap element views, nor
    // a comparison's, which the heap's own path makes.
    let texture = match image {
        ImageRef::Slot(n) => LinearImage::Slot(n),
        ImageRef::Heap { .. } if shape.dim != ImageDim::Cube && compare.is_none() => {
            LinearImage::Heap
        }
        ImageRef::Heap { .. } => return Ok(None),
    };
    let axes = match shape.dim {
        ImageDim::D1 => 1,
        ImageDim::D2 | ImageDim::Cube => 2,
        ImageDim::D3 => 3,
        ImageDim::Buffer => return Ok(None),
    };
    let clamped = min_lod.is_some() || matches!(lod, Lod::Base { min: Some(_), .. });
    let offset = match offset {
        None => None,
        Some(Offset::Const(k)) => Some(k.clone()),
        Some(Offset::Dynamic(..)) => return Ok(None),
    };
    if clamped || observed {
        return Ok(None);
    }
    let word = match (slot, argument) {
        (Some(sampler), _) => Some(LinearState::Slot {
            sampler,
            axes: axes as u8,
        }),
        (None, Some(sampler)) => Some(LinearState::Argument {
            sampler,
            axes: axes as u8,
        }),
        (None, None) => None,
    };
    let s = match (state, word) {
        (Some(s), _) if reaches(&s) => s,
        (None, Some(state)) => {
            let l = Linear {
                image: texture,
                state,
                offset,
                raw: false,
                cube: match shape.dim {
                    ImageDim::Cube => Some(super::face::plan(cx)?),
                    _ => None,
                },
                compare,
                fetch: None,
                exact_only: metal_sampler && shape.dim != ImageDim::Cube,
                native: false,
            };
            let l = Linear {
                native: super::native::planned(cx, &l, shape),
                ..l
            };
            cx.families
                .sampler
                .bicubic_routines
                .extend(l.routines().iter().copied());
            cx.families.access.heap_class |= texture == LinearImage::Heap;
            return Ok(Some(l));
        }
        _ => return Ok(None),
    };
    if shape.dim == ImageDim::Cube {
        let LinearImage::Slot(texture) = texture else {
            return Ok(None);
        };
        return cube::select(cx, texture, &s, compare);
    }
    let modes: Vec<AddressMode> = [s.address_s, s.address_t, s.address_r]
        .into_iter()
        .take(axes)
        .collect();
    let border = s.border_color;
    let address = modes
        .iter()
        .map(|m| match m {
            AddressMode::ClampToEdge => TapAddress::Edge,
            AddressMode::Repeat => TapAddress::Repeat,
            AddressMode::MirroredRepeat => TapAddress::Mirror,
            AddressMode::ClampToZero | AddressMode::ClampToBorder => TapAddress::Border,
        })
        .collect();
    let l = Linear {
        image: texture,
        state: LinearState::Constant {
            address,
            border,
            pixel: s.coordinates == Coordinates::Pixel,
        },
        offset,
        raw: false,
        cube: None,
        compare,
        fetch: None,
        exact_only: metal_sampler,
        native: false,
    };
    let l = Linear {
        native: super::native::planned(cx, &l, shape),
        ..l
    };
    cx.families
        .sampler
        .bicubic_routines
        .extend(l.routines().iter().copied());
    cx.families.access.heap_class |= texture == LinearImage::Heap;
    Ok(Some(l))
}

/// The note of sample `op` of `image` through `sampler` (`None`: an argument buffer's) with
/// texel form `form`, where a device without `VkProperty::MetalSampler` filters it:
/// `PrecisionNote::HalfSampleFilter` for a `half` result, `PrecisionNote::FloatSampleFilter` for a
/// `float` one (a depth texture's and a comparison's included); `None` for a sample that is not
/// filtered (a constexpr sampler that reads one texel of one level, [`unfiltered`]), one of another
/// result, and every sample on a device with the property but a slot's under its YCbCr variant,
/// whose filter the device's is not, and a slot's whose two-plane YCbCr format the profile leaves
/// to Metal's own sampler. Where it holds is [`filter_when`]'s, for such a slot
/// [`metal_filtered_ycbcr`]'s.
pub(super) fn filter_note(
    cx: &SelectCx<'_>,
    op: &SampleOp,
    form: Option<TexelForm>,
    (sampler, image): (Option<SamplerRef>, ImageRef),
) -> Result<Option<PrecisionNote>, Error> {
    if !matches!(op, SampleOp::Sample { .. }) {
        return Ok(None);
    }
    let note = match form.map(|f| f.conv) {
        Some(TexelConv::F16) => PrecisionNote::HalfSampleFilter,
        Some(TexelConv::F32) => PrecisionNote::FloatSampleFilter,
        _ => return Ok(None),
    };
    if let Some(SamplerRef::Constexpr(words)) = sampler {
        if unfiltered(&super::state_of(words)?) {
            return Ok(None);
        }
    }
    let metal = cx.profile.device_samples();
    let device_metal = metal
        && !crate::lower::texture::ycbcr_slot(cx, image)?
        && metal_filtered_ycbcr(cx, image).is_none();
    Ok((!device_metal).then_some(note))
}

/// On a device that samples through Metal's own sampler (`VkProperty::MetalSampler`) under
/// `YCbCrFilterPolicy::Device` (no `ProfileFields::ycbcr_variant`), the clauses under which a
/// filtered sample of texture slot `image` is not Metal's filter of the format bound there: where
/// that format is two-plane YCbCr (its class's `TexelFormat::PACKED` field one of
/// [`PackedTexels::YCbCr2048`], [`PackedTexels::YCbCr8192`] and
/// [`PackedTexels::YCbCrExtended8192`]), whose view holds Metal's reads exactly but which Metal's
/// sampler then filters as an `RGBA16Float` or `RGBA32Float` texture's texels, not on the format's
/// own grid. The runtime reads the bound format (`SpecKey::BoundTexelFormat`) and announces it as
/// `ycbcr-filter-not-exact`, through a runtime sampler whatever its word states. `None` on any
/// other device or profile, and for a heap's texture, which never takes the variant.
pub(super) fn metal_filtered_ycbcr(cx: &SelectCx<'_>, image: ImageRef) -> Option<Vec<NoteClause>> {
    let ImageRef::Slot(index) = image else {
        return None;
    };
    let fields = cx.profile.fields();
    let metal = fields
        .enabled
        .properties
        .contains(&crate::device::requirements::VkProperty::MetalSampler);
    (metal && !fields.ycbcr_variant).then(|| {
        any_of(
            [
                PackedTexels::YCbCr2048,
                PackedTexels::YCbCr8192,
                PackedTexels::YCbCrExtended8192,
            ]
            .map(|p| NoteTest::equal(NoteOn::Bound(index), TexelFormat::PACKED_MASK, p.class().0)),
        )
    })
}

/// Where the filter note ([`filter_note`]) of a sample through `sampler` (`None`: an argument
/// buffer's) holds, its emulation `linear` as the caller settled it: the clauses under which the
/// specialized module runs an arm that is not Metal's filter.
///
/// - No emulation: the device filters. Through a runtime sampler slot, wherever its word filters
///   ([`word_filters`]); otherwise always.
/// - A raw emulation (an 8-bit sRGB slot's variant): the raw codes' exact sums, in 1D, 2D, and 3D.
///   Through a constexpr sampler it has no device arm, so never; through a runtime sampler slot the
///   device filters the decoding view where the word filters but the emulation does not reach it.
/// - Otherwise the emulation runs where the class names a measured family (for a cube, not a
///   packed normalized format's, `TexelFormat::packed_exact`) and, through a runtime sampler, the
///   word states a filter the emulation reaches (`reaches_word`); the device's arm runs
///   elsewhere. In 3D a half-float and a 32-bit float format's emulated sums keep a residue, so the
///   note holds wherever the class is not normalized.
pub(super) fn filter_when(linear: Option<&Linear>, sampler: Option<SamplerRef>) -> Vec<NoteClause> {
    let always = || vec![NoteClause::new()];
    let Some(l) = linear else {
        return match sampler {
            Some(SamplerRef::Slot(n)) => word_filters(n),
            _ => always(),
        };
    };
    // A heap texture's class is its handle's, which no specialization states.
    let LinearImage::Slot(texture) = l.image else {
        return always();
    };
    if l.raw {
        return device_arm(l, texture, None);
    }
    let class: NoteClause = match l.cube.is_none() && l.plane_axes() == 3 {
        true => [NoteTest::equal(
            NoteOn::Texel(texture),
            BicubicTexels::EXACT,
            0,
        )]
        .into(),
        false => crate::lower::texture::unmeasured_class(texture),
    };
    device_arm(l, texture, Some(class))
}

/// Where the device's arm of emulation `l` runs: the clauses under which the specialized module
/// hands the sample to the device's filter, at the device's position of the coordinate, rather than
/// summing Metal's taps (whose sums [`filter_when`] says where may keep a residue). A raw emulation
/// (an 8-bit sRGB slot's variant) runs it, through a runtime sampler slot, where the word filters
/// but the emulation does not reach it, and through a constexpr sampler never; another runs it
/// where the class names no measured family (for a cube, also a packed normalized format's,
/// `TexelFormat::packed_exact`; under [`Linear::exact_only`], wherever the class is not `EXACT`)
/// and, through a runtime sampler slot, where the word filters but states a filter or an address
/// mode the emulation does not reach (`reaches_word`). A heap texture's runs wherever it is
/// sampled: its class is its handle's, which no specialization states.
pub(super) fn device_when(l: &Linear) -> Vec<NoteClause> {
    let LinearImage::Slot(texture) = l.image else {
        return vec![NoteClause::new()];
    };
    let class = (!l.raw).then(|| match l.exact_only {
        true => [NoteTest::equal(
            NoteOn::Texel(texture),
            BicubicTexels::EXACT,
            0,
        )]
        .into(),
        false => crate::lower::texture::unmeasured_class(texture),
    });
    device_arm(l, texture, class)
}

/// Where a raw emulation `l` through a runtime sampler slot hands its sample to the device's
/// decoding view (`PrecisionNote::SrgbTexelDecode`): where the word does not reach the emulation
/// and filters, since a word that reads one texel samples the slot's raw view ([`Linear::fetch`]);
/// a cube's (no such view) wherever the word does not reach it. Never for another emulation.
pub(super) fn decode_when(l: &Linear) -> Vec<NoteClause> {
    match (&l.state, l.raw, &l.cube) {
        // A word that reads one texel samples the raw view ([`emit`], [`emit_shared`]): the
        // device decodes only where the word filters too.
        (LinearState::Slot { sampler, .. }, true, None) => {
            both(&unreached(l, *sampler), &word_filters(*sampler))
        }
        (LinearState::Slot { sampler, .. }, true, Some(_)) => unreached(l, *sampler),
        _ => Vec::new(),
    }
}

/// The clauses of emulation `l`'s device arm of texture slot `texture`, `class` the clause of the
/// classes it hands the device (none for a raw emulation, whose class the variant fixes;
/// [`filter_when`], [`device_when`]). A runtime sampler's word that does not reach the emulation
/// hands it the classes it takes, so each such clause also tests the texture's class outside
/// `class` (the same condition: the classes `class` names hold anyway): every clause names the
/// texture slot it holds for, which the runtime's announcement names by the format bound there
/// (`ycbcr-filter-not-exact` for a two-plane YCbCr one).
fn device_arm(l: &Linear, texture: u32, class: Option<NoteClause>) -> Vec<NoteClause> {
    let taken: Option<NoteClause> = match class.as_ref().map(|c| c.iter().collect::<Vec<_>>()) {
        Some(tests) => match tests[..] {
            [&test] => Some(
                [NoteTest {
                    equal: !test.equal,
                    ..test
                }]
                .into(),
            ),
            _ => None,
        },
        None => None,
    };
    let mut classes: Vec<NoteClause> = class.into_iter().collect();
    // A cube's device arm also runs on a packed normalized format (`TexelFormat::packed_exact`).
    if l.cube.is_some() && !l.raw {
        let on = NoteOn::Texel(texture);
        classes.push(
            [
                NoteTest::differ(on, BicubicTexels::EXACT, 0),
                NoteTest::differ(on, TexelFormat::PACKED_MASK, 0),
            ]
            .into(),
        );
    }
    match &l.state {
        LinearState::Constant { .. } => classes,
        LinearState::Slot { sampler, .. } => {
            let unreached = unreached(l, *sampler);
            let mut device = match taken {
                Some(taken) => both(&unreached, &[taken]),
                None => unreached,
            };
            device.extend(classes);
            both(&word_filters(*sampler), &device)
        }
        // An argument buffer's sampler samples only heap textures, whose notes hold always.
        LinearState::Argument { .. } => vec![NoteClause::new()],
    }
}

/// Where runtime sampler slot `sampler`'s word does not reach emulation `l` (`reaches_word`): a
/// filter other than Metal's linear one (for a cube, or pixel coordinates); a mip filter, but on
/// a texture slot's class whose textures have one level (`PackedTexels::one_level`; not for a raw
/// emulation, whose class is an sRGB format's, nor a cube); or, where the emulation does not reach
/// it ([`Linear::mirror_clamp_reached`]), `MirrorClampToEdge` on a plane axis.
fn unreached(l: &Linear, sampler: u32) -> Vec<NoteClause> {
    let on = NoteOn::Sampler(sampler);
    let mask = match l.cube {
        Some(_) => SamplerWord::LINEAR_FILTER | SamplerWord::PIXEL_COORDINATES,
        None => SamplerWord::LINEAR_FILTER,
    };
    let mut mipped: NoteClause = [NoteTest::differ(on, SamplerWord::MIP_FILTER, 0)].into();
    if let Some(texture) = one_level_slot(l) {
        mipped.extend(one_level_fields().map(|field| {
            NoteTest::differ(NoteOn::Texel(texture), TexelFormat::PACKED_MASK, field)
        }));
    }
    let mut clauses = vec![
        [NoteTest::differ(
            on,
            mask & !SamplerWord::MIP_FILTER,
            SamplerWord::LINEAR,
        )]
        .into(),
        mipped,
    ];
    if !l.mirror_clamp_reached() {
        let unmeasured = SamplerAddressMode::MirrorClampToEdge.as_raw();
        clauses.extend(any_of(
            SamplerWord::ADDRESS_SHIFT
                .into_iter()
                .take(l.plane_axes())
                .map(|shift| NoteTest::equal(on, 0b111 << shift, unmeasured << shift)),
        ));
    }
    clauses
}

/// The texture slot whose class states whether emulation `l` reaches a runtime sampler's mip
/// filter (`PackedTexels::one_level`): a slot's, but not a raw emulation's, whose class is an 8-bit
/// sRGB format's, nor a cube's.
fn one_level_slot(l: &Linear) -> Option<u32> {
    match (l.image, l.raw, &l.cube) {
        (LinearImage::Slot(texture), false, None) => Some(texture),
        _ => None,
    }
}

/// The values of the [`TexelFormat::PACKED`] field, in place, of the classes whose textures have
/// one level (`PackedTexels::one_level`).
fn one_level_fields() -> impl Iterator<Item = u32> {
    PackedTexels::ALL
        .into_iter()
        .filter(|p| p.one_level())
        .map(|p| p.class().0)
}

/// Where runtime sampler slot `n`'s word states a sampler that filters: a linear minification or
/// magnification filter, a linear mip filter, or anisotropy, any of which [`unfiltered`]'s single
/// texel of one level excludes.
pub(crate) fn word_filters(n: u32) -> Vec<NoteClause> {
    let on = NoteOn::Sampler(n);
    any_of(
        [
            SamplerWord::MIN_LINEAR,
            SamplerWord::MAG_LINEAR,
            2 << SamplerWord::MIP_FILTER_SHIFT,
            SamplerWord::ANISOTROPIC,
        ]
        .map(|bit| NoteTest::differ(on, bit, 0)),
    )
}

/// Whether a sample through sampler `s` reads one texel of one level, its value unfiltered:
/// nearest minification and magnification, no mip filter or the nearest level's, and no
/// anisotropy (a lowered sampler's reduction is the weighted average). Any other sampler filters:
/// linear minification or magnification, a linear mip filter (a blend of two levels), or
/// anisotropy.
pub(crate) fn unfiltered(s: &SamplerState) -> bool {
    s.min_filter == Filter::Nearest
        && s.mag_filter == Filter::Nearest
        && matches!(s.mip_filter, MipFilter::None | MipFilter::Nearest)
        && s.max_anisotropy <= 1
}

/// Whether sampler `s`'s filter is the one measured: linear minification and magnification, no
/// mip filter, and no anisotropy.
fn reaches(s: &SamplerState) -> bool {
    s.min_filter == Filter::Linear
        && s.mag_filter == Filter::Linear
        && s.mip_filter == MipFilter::None
        && s.max_anisotropy <= 1
}

/// The texel of the linear sample `s` (image facts `image`, a combined binding), emulated by `l`
/// (module docs) where the slot's bound format is measured, else `device`'s sample of the
/// binding's sampled image (a `vec4` of `f32`), converted as a sample's for a `half` result: a
/// branch on the slot's `TexelFormat` specialization, so only one of them runs, and once a device
/// specializes the module the other is dead. A raw emulation runs alone through a constexpr
/// sampler, and through a runtime sampler where its word reaches it; elsewhere the device samples
/// the slot's raw view where the word reads one texel, decoded through Metal's table, and the
/// binding's decoding view otherwise. Each arm loads the binding's sampled image, which is consumed
/// in the block that loads it, or a cube's emulation the slot's 2D-array view at `plane`, the
/// direction the device's sample reads (its ties broken as Metal breaks them). A `half` result is
/// the held `half` texel.
pub(super) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (plane, image, dref): (SValue, CombinedImage, Option<SValue>),
    l: &Linear,
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
    native: Option<SValue>,
) -> BResult<SValue> {
    let half = half_result(cx, s)?;
    let LinearImage::Slot(texture) = l.image else {
        return Err(cx.fail("a heap texture's linear sample past its own path"));
    };
    let Some(spec) = crate::lower::texture::texel_format_const(cx, texture) else {
        return Err(cx.fail(format!(
            "texture slot {texture} has no `TexelFormat` specialization"
        )));
    };
    let Source::Combined(key) = s.source else {
        return Err(cx.fail("a linear sample of a heap element"));
    };
    // The combined binding, or its separate texture and sampler: each arm makes the sampled image
    // in its own block, where it is consumed.
    let Some(&(binding, _)) = cx.decl.families.sampler.combined.get(&key) else {
        return Err(cx.fail(format!("combined binding {key:?} is not declared")));
    };
    let t = Tys::new(fb)?;
    let word = state_word(cx, fb, l)?;
    // The emulation is its shape's routine, called with the sample's plain data (`routine`).
    let (routine, args) = emulation(cx, fb, s, (plane, dref), spec)?;
    // A constexpr sampler's raw emulation has no device arm.
    if let (true, None) = (l.raw, word) {
        return fb
            .call(routine, &args)?
            .ok_or_else(bicubic::missing_routine);
    }
    let h = match (half, s.half_texel) {
        (true, true) => match s
            .texel
            .and_then(|form| half_class(cx, &ImageRef::Slot(texture), form))
        {
            Some(h) => Some(HalfArm::Class(h)),
            None => return Err(cx.fail(format!("texture slot {texture} has no `half` conversion"))),
        },
        (true, false) => Some(HalfArm::Device),
        (false, _) => None,
    };
    // A raw emulation's class is its variant's, an 8-bit sRGB format's, whose sum is measured:
    // a runtime sampler's runs where its word reaches it.
    let measured = match (l.raw, word) {
        (true, Some(word)) => reaches_word(fb, &t, l, word, None)?,
        _ => {
            let measured = flag(fb, &t, spec, l.measured())?;
            // A packed normalized format's cube sample is not measured
            // (`TexelFormat::packed_exact`).
            let measured = match l.cube {
                Some(_) => {
                    let exact = flag(fb, &t, spec, BicubicTexels::EXACT)?;
                    let packed = flag(fb, &t, spec, TexelFormat::PACKED_MASK)?;
                    let packed = fb.binary(BinOp::LogicalAnd, exact, packed)?;
                    let other = fb.unary(UnOp::LogicalNot, packed)?;
                    fb.binary(BinOp::LogicalAnd, measured, other)?
                }
                None => measured,
            };
            reached(fb, &t, l, word, (measured, Bits::Spec(spec)))?
        }
    };
    // A native class's sample is the device's, at Metal's footprint step (`native`).
    let measured = without_native(fb, measured, native)?;
    // The emulation: its routine's call.
    let emulated: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let x = fb
            .call(routine, &args)?
            .ok_or_else(bicubic::missing_routine)?;
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    // The device's sample, which reads one texel where a runtime sampler's word states nearest
    // filters: its `half` or `float` texel converts by the class with the sample's lanes. There a
    // 32-bit float texel the shader fetches instead.
    let nearest = s
        .fetched
        .as_ref()
        .filter(|p| p.image.slot() == Some(texture));
    let float = match s.float_texel {
        true => match s
            .texel
            .and_then(|form| crate::lower::texture::float_class(cx, &ImageRef::Slot(texture), form))
        {
            Some(f) => Some(f),
            None => return Err(cx.fail("a `float` sample without its conversion routine")),
        },
        false => None,
    };
    let filtered: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let x = match nearest {
            Some(p) => super::fetched::branch(
                cx,
                fb,
                (s, key),
                (texture, &p.fetch, BicubicTexels::LINEAR, true),
                image,
                &super::fetched::one,
                device,
            )?,
            None => {
                let si = binding.sampled_image(fb)?;
                device(fb, si)?
            }
        };
        // Under the slot's twin variant the device sampled the codes: decoded through the table,
        // exactly where the sampler read one texel.
        let x = match s.srgb_codes {
            true => {
                let four = crate::lower::texture::TexelForm {
                    lanes: 4,
                    conv: crate::lower::texture::TexelConv::F32,
                };
                let color = crate::lower::texture::SRGB_TEXEL;
                crate::lower::texture::srgb_filtered_texel(cx, fb, four, x, color)?
            }
            false => x,
        };
        let x = match (h, float) {
            (Some(HalfArm::Class(h)), _) => {
                let lanes = super::emit::sample_lanes(cx, fb, s)?;
                h.convert(fb, x, HalfForm::Sample, lanes)?
            }
            // The device's texel rounded to nearest even.
            (Some(HalfArm::Device), _) => {
                use crate::lower::arith::legal::{FloatFmt, Form, Held};
                let half = Form {
                    held: Held::Float(FloatFmt::Half),
                    lanes: 4,
                };
                crate::lower::arith::narrow::round_texel(fb, half, x)?
            }
            (None, Some(f)) => {
                let lanes = super::emit::sample_lanes(cx, fb, s)?;
                f.convert(fb, x, lanes)?
            }
            (None, None) => x,
        };
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    // A raw emulation's device arm, where the runtime sampler's word reads one texel (nearest
    // minification and magnification, no linear mip filter, isotropic): the device's sample of the
    // slot's raw view, through the same sampler, reads that texel's code, which Metal's table
    // decodes exactly, as a constexpr nearest sample's raw read does. Every other word samples the
    // decoding view ([`decode_when`]). Under the slot's twin variant the decoding view holds the
    // codes already.
    let device_arm: ArmFn<'_, EmitCx<'_>> = match (l.raw, word, &l.cube, s.srgb_codes) {
        (true, Some(word), None, false) => {
            let (Some(raw_key), Some(form)) = (l.fetch, s.texel) else {
                return Err(cx.fail("a raw sample's device arm without its raw view"));
            };
            let Some(&(raw_binding, _)) = cx.decl.families.sampler.combined.get(&raw_key) else {
                return Err(cx.fail(format!("raw binding {raw_key:?} is not declared")));
            };
            let raw: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let si = raw_binding.sampled_image(fb)?;
                let x = device(fb, si)?;
                let color = crate::lower::texture::SRGB_TEXEL;
                let x = crate::lower::texture::srgb_raw_texel(cx, fb, form, x, color)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            one_texel_arm(word, raw, filtered)
        }
        _ => filtered,
    };
    let Merged(merged) = fb.if_(cx, measured, emulated, Some(device_arm))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a linear sample's branch without its texel")),
    }
}

/// A raw emulation's device arm through runtime sampler word `word`: `raw`'s texel (the device's
/// sample of the slot's raw view, decoded through Metal's table) where the word reads one texel
/// (`compared::one_texel`), else `filtered`'s (the device's sample of the decoding view).
fn one_texel_arm<'a, 'c: 'a>(
    word: SValue,
    raw: ArmFn<'a, EmitCx<'c>>,
    filtered: ArmFn<'a, EmitCx<'c>>,
) -> ArmFn<'a, EmitCx<'c>> {
    Box::new(move |fb, cx| {
        let t = Tys::new(fb)?;
        let (one, _) = super::compared::one_texel(fb, &t, word)?;
        let Merged(merged) = fb.if_(cx, one, raw, Some(filtered))?;
        match merged[..] {
            [x] => Ok(ArmEnd::FallThrough(vec![x])),
            _ => Err(cx.fail("a raw sample's device arm without its texel")),
        }
    })
}

/// The gathered texel of cube gather `s` by `g` (`cube`) of channel `component`, along `plane` (the
/// direction, its ties broken as Metal breaks them), a comparison's against reference `dref`. An
/// integer result's and a raw gather's are the emulation's alone. A `float` or `half` result's, a
/// depth texture's and a comparison's included, branch on the slot's `TexelFormat` specialization:
/// the emulation where the bound format's corner rule is measured (`BicubicTexels::EXACT`,
/// `LINEAR`, or `HALF`, or a `TexelFormat::CORNER` field), else `device`'s gather of the combined
/// binding's sampled image, which each arm makes in its own block; once a device specializes the
/// module the other arm is dead. A raw gather's value is its held texel; any other's is the texel a
/// device gather returns (a `vec4` of the view's sampled type), which the caller converts.
pub(super) fn gather(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (plane, component, dref): (SValue, u32, Option<SValue>),
    g: &CubeGather,
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
) -> BResult<SValue> {
    let t = Tys::new(fb)?;
    let at = (plane, component, dref);
    if g.integer {
        return cube::gather(cx, fb, &t, s, g, at, None);
    }
    let Some(spec) = crate::lower::texture::texel_format_const(cx, g.texture) else {
        return Err(cx.fail(format!(
            "texture slot {} has no `TexelFormat` specialization",
            g.texture
        )));
    };
    if g.raw {
        return cube::gather(cx, fb, &t, s, g, at, Some(spec));
    }
    let Source::Combined(key) = s.source else {
        return Err(cx.fail("a cube gather of a heap element"));
    };
    let Some(&(binding, _)) = cx.decl.families.sampler.combined.get(&key) else {
        return Err(cx.fail(format!("combined binding {key:?} is not declared")));
    };
    let measured = flag(
        fb,
        &t,
        spec,
        BicubicTexels::EXACT
            | BicubicTexels::LINEAR
            | BicubicTexels::HALF
            | TexelFormat::CORNER_MASK,
    )?;
    let t = &t;
    let emulated: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let x = cube::gather(cx, fb, t, s, g, at, Some(spec))?;
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let gathered: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let si = binding.sampled_image(fb)?;
        Ok(ArmEnd::FallThrough(vec![device(fb, si)?]))
    });
    let Merged(merged) = fb.if_(cx, measured, emulated, Some(gathered))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a cube gather's branch without its texel")),
    }
}

/// The texel of the linear sample `s` shared by the arms of a switch-at-use (SEMANTICS 1012):
/// [`emit`]'s branch on the named candidate's format bits `bits` (its `TexelFormat`
/// specialization, which `c`'s arms yield), the emulation on its level-0 size `size` (image facts
/// `image`) with its taps read by `c`'s arms, or `device`'s sample of its sampled image read by
/// `c`'s arms (the null arm's a zero), converted as a sample's for a `half` result by the bits.
pub(super) fn emit_shared(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (size, image, dref): (SValue, CombinedImage, Option<SValue>),
    l: &Linear,
    (bits, c): (SValue, &super::Candidates<'_>),
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
    native: Option<SValue>,
) -> BResult<SValue> {
    let half = half_result(cx, s)?;
    let t = Tys::new(fb)?;
    let bits = Bits::Value(bits);
    let word = state_word(cx, fb, l)?;
    let shared = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, t: &Tys| {
        let Some(taps) = l.taps(word) else {
            return Err(cx.fail("a runtime sampler's linear sample without its word"));
        };
        let fp = footprint_on(cx, fb, t, s, (size, image), bits, &taps)?;
        sum(
            cx,
            fb,
            (t, &fp),
            (l, word),
            (bits, half, dref),
            Texels::Candidates(c),
        )
    };
    // A raw emulation's class is its variant's: a constexpr sampler's has no device arm, and a
    // runtime sampler's runs where its word reaches it.
    let measured = match (l.raw, word) {
        (true, None) => return shared(cx, fb, &t),
        (true, Some(word)) => reaches_word(fb, &t, l, word, None)?,
        (false, _) => {
            let measured = bits.test(fb, &t, l.measured())?;
            reached(fb, &t, l, word, (measured, bits))?
        }
    };
    let measured = without_native(fb, measured, native)?;
    let t = &t;
    let shared = &shared;
    let emulated: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let x = shared(cx, fb, t)?;
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let filtered: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
            let si = c.sampled_image(cx, fb, a)?;
            Ok(vec![device(fb, si)?])
        };
        // A comparison's device sample is a scalar.
        let null = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| {
            let ty = match dref {
                Some(_) => fb.module().ty(SType::Float { width: 32 })?,
                None => t.vec4,
            };
            let z = fb.module().constant(Constant::Null { ty })?;
            Ok(vec![value(fb, z)?])
        };
        let [x] = c.each(cx, fb, &read, &null)?[..] else {
            return Err(cx.fail("a shared linear sample's device texel"));
        };
        let x = match (half, bits) {
            (true, Bits::Value(class)) => {
                crate::lower::texture::half_by_class(cx, fb, x, class, HalfForm::Sample)?
            }
            _ => x,
        };
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    // A raw emulation's device arm reads each arm's raw view where the word reads one texel, as
    // [`emit`]'s does; a null arm's zero decodes to zero.
    let device_arm: ArmFn<'_, EmitCx<'_>> = match (l.raw, word, &l.cube, s.srgb_codes) {
        (true, Some(word), None, false) => {
            let (Some(_), Some(form)) = (l.fetch, s.texel) else {
                return Err(cx.fail("a shared raw sample's device arm without its raw view"));
            };
            let raw: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
                    let si = c.fetch_sampled_image(cx, fb, a)?;
                    Ok(vec![device(fb, si)?])
                };
                let null = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| {
                    let z = fb.module().constant(Constant::Null { ty: t.vec4 })?;
                    Ok(vec![value(fb, z)?])
                };
                let [x] = c.each(cx, fb, &read, &null)?[..] else {
                    return Err(cx.fail("a shared raw sample's device texel"));
                };
                let color = crate::lower::texture::SRGB_TEXEL;
                let x = crate::lower::texture::srgb_raw_texel(cx, fb, form, x, color)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            one_texel_arm(word, raw, filtered)
        }
        _ => filtered,
    };
    let Merged(merged) = fb.if_(cx, measured, emulated, Some(device_arm))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a linear sample's branch without its texel")),
    }
}

/// The texel of the linear sample `s` of a heap texture (image facts `image`), emulated by `l`
/// (module docs) where the class its handle carries names a measured family and, through a runtime
/// or argument buffer's sampler, the word states a filter the emulation reaches, else `device`: the
/// device's sample as the function converts it (its held texel; a depth texture's first component).
/// The device's sample runs before the branch, in the function's own control flow; the emulation's
/// arm loads the heap element again, whose sampled image it consumes in its own block, and fetches
/// its taps from the element's image.
pub(super) fn emit_heap(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    image: CombinedImage,
    l: &Linear,
    device: SValue,
) -> BResult<SValue> {
    let half = half_result(cx, s)?;
    let handle = match s.source {
        Source::Heap { handle, .. } | Source::HeapSampler { handle, .. } => handle,
        Source::Combined(_) => return Err(cx.fail("a heap texture's linear sample of a slot")),
    };
    let t = Tys::new(fb)?;
    let h = cx.operand(fb, handle)?;
    let bits = Bits::Value(crate::lower::texture::heap_texel_class(cx, fb, h)?);
    let word = state_word(cx, fb, l)?;
    let measured = bits.test(
        fb,
        &t,
        BicubicTexels::EXACT | BicubicTexels::LINEAR | BicubicTexels::HALF,
    )?;
    let measured = reached(fb, &t, l, word, (measured, bits))?;
    let t = &t;
    let emulated: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let Some(taps) = l.taps(word) else {
            return Err(cx.fail("a runtime sampler's linear sample without its word"));
        };
        let (si, _, _) = super::emit::sampled_image(cx, fb, s.source)?;
        let (fp, img) = footprint(cx, fb, t, s, (si, image), bits, &taps)?;
        let x = sum(
            cx,
            fb,
            (t, &fp),
            (l, word),
            (bits, half, None),
            Texels::Own(img),
        )?;
        // A depth texture's sample reads the first component.
        let x = match s.depth {
            true => fb.composite_extract(x, &[0])?,
            false => x,
        };
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let kept: ArmFn<'_, EmitCx<'_>> = Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![device])));
    let Merged(merged) = fb.if_(cx, measured, emulated, Some(kept))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a heap texture's linear sample without its texel")),
    }
}

/// The routine of linear sample `s`'s emulated arm (`routine::Emulation`), defined at its shape's
/// first call, and its call's arguments: the coordinate and the array layer as the sample states
/// them, a cube's direction `plane` (its ties broken as Metal breaks them), a comparison's
/// reference `dref`, each computed in the sample's own position, the constant texel offset where a
/// lane is off zero, and the value of the slot's `TexelFormat` specialization `spec`, which a
/// function reads as data, so a routine takes it as its other routines do.
fn emulation(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (plane, dref): (SValue, Option<SValue>),
    spec: ConstId,
) -> BResult<(FuncId, Vec<SValue>)> {
    let Some(e) = Emulation::of(s) else {
        return Err(cx.fail("a linear sample without its emulation's shape"));
    };
    if e.linear.compare.is_some() != dref.is_some() {
        return Err(cx.fail("a linear comparison without its reference"));
    }
    let at = At::of(cx, fb, s)?;
    let mut args = vec![at.coord];
    args.extend(at.layer);
    if e.linear.cube.is_some() {
        args.push(plane);
    }
    args.extend(dref);
    if e.offset.is_some() {
        let Some(o) = s.linear.as_ref().and_then(|l| l.offset.as_deref()) else {
            return Err(cx.fail("a linear sample's routine without its offset"));
        };
        args.push(super::emit::const_offset(fb, o)?);
    }
    args.push(fb.module().const_value(spec)?);
    let mut tys = Vec::with_capacity(args.len());
    for &a in &args {
        tys.push(fb.value_ty(a)?);
    }
    let shape = Shape::Emulation(e);
    if let Some(f) = cx.sampling.get(&shape, &tys) {
        return Ok((f, args));
    }
    let Shape::Emulation(e) = &shape else {
        return Err(cx.fail("a linear sample's routine of another shape"));
    };
    let mut rcx = cx.for_new_function();
    let made = define_emulation(&mut rcx, fb.module(), e, (&tys, at.parent));
    cx.adopt(rcx);
    let f = made?;
    cx.sampling.insert(shape, tys, f);
    Ok((f, args))
}

/// Define the routine of emulated arm `e` (`routine::Emulation`) in `mb`, its parameters of
/// types `tys` in [`emulation`]'s order, its offset placed by the slot's tests `parent` where `e`
/// reads them: the texel the arm's code gave inline, read from `e`'s bindings and specializations
/// and the bound format's bits its last parameter carries.
fn define_emulation(
    cx: &mut EmitCx<'_>,
    mb: &mut ModuleBuilder,
    e: &Emulation,
    (tys, parent): (&[TypeId], Option<OffsetBase>),
) -> BResult<FuncId> {
    let l = &e.linear;
    let Some(&(_, image)) = cx.decl.families.sampler.combined.get(&e.own) else {
        return Err(cx.fail(format!("combined binding {:?} is not declared", e.own)));
    };
    // The taps' binding: a comparison's without comparison, a runtime sampler's raw view, else
    // the sample's own.
    let Some(&(taps, _)) = cx.decl.families.sampler.combined.get(&e.taps) else {
        return Err(cx.fail(format!("combined binding {:?} is not declared", e.taps)));
    };
    // A comparison's value is a scalar, every other the held texel (a `vec4`).
    let f32_ = mb.ty(SType::Float { width: 32 })?;
    let ret = match l.compare {
        Some(_) => f32_,
        None => mb.ty(SType::Vector { elem: f32_, len: 4 })?,
    };
    let mut fb = mb.sampling_routine(ret)?;
    let mut params = Vec::with_capacity(tys.len());
    for &ty in tys {
        params.push(fb.param(ty)?);
    }
    // The coordinate, then the layer, a cube's direction, a comparison's reference and the
    // offset, each where the shape takes it, and the format's bits.
    let want = 2
        + usize::from(e.layer)
        + usize::from(l.cube.is_some())
        + usize::from(l.compare.is_some())
        + usize::from(e.offset.is_some());
    let mut params = params.into_iter();
    let (Some(coord), true) = (params.next(), tys.len() == want) else {
        return Err(cx.fail("a linear sample's routine without its parameters"));
    };
    let layer = e.layer.then(|| params.next()).flatten();
    let plane = l.cube.and_then(|_| params.next());
    let dref = l.compare.and_then(|_| params.next());
    let offset = e.offset.and_then(|lanes| params.next().map(|v| (v, lanes)));
    let Some(bits) = params.next().map(Bits::Value) else {
        return Err(cx.fail("a linear sample's routine without its format's bits"));
    };
    let at = At {
        coord,
        layer,
        parent,
        offset,
    };
    let t = Tys::new(&mut fb)?;
    let word = state_word(cx, &mut fb, l)?;
    let x = match (l.cube, plane) {
        (Some(face), Some(p)) => cube::emulate(
            cx,
            &mut fb,
            &t,
            layer,
            (l, word),
            (face, p),
            (bits, e.half, dref),
        )?,
        _ => {
            let si = taps.sampled_image(&mut fb)?;
            emulate(
                cx,
                &mut fb,
                &t,
                &at,
                (si, image),
                (l, word),
                (bits, e.half, dref),
            )?
        }
    };
    fb.ret(Some(x))?;
    fb.finish()
}

/// `measured` (a `bool`: the emulation runs) where the native condition `native` (a `bool`,
/// `native::coordinate`'s) does not hold: a native class's sample is the device's.
fn without_native(
    fb: &mut FunctionBuilder<'_>,
    measured: SValue,
    native: Option<SValue>,
) -> BResult<SValue> {
    match native {
        Some(n) => {
            let other = fb.unary(UnOp::LogicalNot, n)?;
            fb.binary(BinOp::LogicalAnd, measured, other)
        }
        None => Ok(measured),
    }
}

/// How the device arm of a linear sample with a `half` result converts the device's texel: by
/// the bound format's class, or, where the texel is the device's, its rounding to nearest even.
#[derive(Clone, Copy)]
enum HalfArm {
    Class(crate::lower::texture::HalfClass),
    Device,
}

/// Whether the linear sample `s` has a `half` result.
fn half_result(cx: &mut EmitCx<'_>, s: &Sampling) -> BResult<bool> {
    match s.texel {
        Some(form) => Ok(form.conv == TexelConv::F16),
        None => Err(cx.fail("a linear sample without a texel form")),
    }
}

impl Linear {
    /// The taps the emulation reads: `i` and `i + 1` of each axis, addressed as a constexpr
    /// sampler's state states, or as a runtime sampler's `word` does (`None` without one).
    fn taps(&self, word: Option<SValue>) -> Option<Taps<'_>> {
        let addressing = match (&self.state, word) {
            (
                LinearState::Constant {
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
            (LinearState::Slot { axes, .. } | LinearState::Argument { axes, .. }, Some(word)) => {
                Addressing::Word {
                    word,
                    axes: usize::from(*axes),
                }
            }
            (LinearState::Slot { .. } | LinearState::Argument { .. }, None) => return None,
        };
        Some(Taps {
            addressing,
            offset: self.offset.as_deref(),
            span: &[0, 1],
            base: TapBase::Quantized,
            level: None,
        })
    }
}

/// The word of the runtime sampler `l`'s emulation follows (a `u32`, its slot's `SamplerState`
/// specialization, or an argument buffer's sampler's table entry's), or `None` for a constexpr
/// sampler's.
fn state_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    l: &Linear,
) -> BResult<Option<SValue>> {
    match l.state {
        LinearState::Constant { .. } => Ok(None),
        LinearState::Slot { sampler, .. } => Ok(Some(super::emit::sampler_word(
            cx,
            fb,
            super::Pixel::Spec(sampler),
        )?)),
        LinearState::Argument { sampler, .. } => Ok(Some(super::emit::sampler_word(
            cx,
            fb,
            super::Pixel::Heap(sampler),
        )?)),
    }
}

/// `measured` (a `bool`: the bound format's conversion is measured, its class `bits`), and for a
/// runtime sampler whether its `word` states a sampler the emulation `l` reaches on that class
/// (`reaches_word`).
fn reached(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    l: &Linear,
    word: Option<SValue>,
    (measured, bits): (SValue, Bits),
) -> BResult<SValue> {
    match word {
        Some(word) => {
            let reach = reaches_word(fb, t, l, word, Some(bits))?;
            fb.binary(BinOp::LogicalAnd, measured, reach)
        }
        None => Ok(measured),
    }
}

/// Whether runtime sampler word `word` (a `u32`) states a sampler whose sample `l` the emulation
/// reaches on the bound format's class `bits` (a `bool`): the filter Metal's linear rule answers
/// ([`SamplerWord::LINEAR_FILTER`]), for a cube with normalized coordinates, but with any mip
/// filter on a class whose textures have one level, which Metal's mip filter leaves as it is
/// (`PackedTexels::one_level`; `None` for a raw emulation's, an sRGB format's); and for a cube or a
/// 3D texture, on each plane axis (a cube's face's `s` and `t`) an address mode other than
/// `MirrorClampToEdge`, which no probe measured there ([`Linear::mirror_clamp_reached`]).
fn reaches_word(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    l: &Linear,
    word: SValue,
    bits: Option<Bits>,
) -> BResult<SValue> {
    let u32_value = |fb: &mut FunctionBuilder<'_>, x: u32| -> BResult<SValue> {
        let c = scalar(fb, t.u32_, u64::from(x))?;
        value(fb, c)
    };
    let mask = match l.cube {
        Some(_) => SamplerWord::LINEAR_FILTER | SamplerWord::PIXEL_COORDINATES,
        None => SamplerWord::LINEAR_FILTER,
    };
    let m = u32_value(fb, mask & !SamplerWord::MIP_FILTER)?;
    let want = u32_value(fb, SamplerWord::LINEAR)?;
    let have = fb.binary(BinOp::BitwiseAnd, word, m)?;
    let mut reach = fb.binary(BinOp::IEqual, have, want)?;
    let mip = u32_value(fb, SamplerWord::MIP_FILTER)?;
    let mip = fb.binary(BinOp::BitwiseAnd, word, mip)?;
    let zero = u32_value(fb, 0)?;
    let mut levels = fb.binary(BinOp::IEqual, mip, zero)?;
    if let (Some(bits), Some(_)) = (bits, one_level_slot(l)) {
        let class = bits.value(fb)?;
        let packed = u32_value(fb, TexelFormat::PACKED_MASK)?;
        let field = fb.binary(BinOp::BitwiseAnd, class, packed)?;
        for f in one_level_fields() {
            let f = u32_value(fb, f)?;
            let one = fb.binary(BinOp::IEqual, field, f)?;
            levels = fb.binary(BinOp::LogicalOr, levels, one)?;
        }
    }
    reach = fb.binary(BinOp::LogicalAnd, reach, levels)?;
    if l.mirror_clamp_reached() {
        return Ok(reach);
    }
    let unmeasured = u32_value(fb, SamplerAddressMode::MirrorClampToEdge.as_raw())?;
    for shift in SamplerWord::ADDRESS_SHIFT.into_iter().take(l.plane_axes()) {
        let mode = super::bicubic::word_field(fb, t, word, shift, 0b111)?;
        let measured = fb.binary(BinOp::INotEqual, mode, unmeasured)?;
        reach = fb.binary(BinOp::LogicalAnd, reach, measured)?;
    }
    Ok(reach)
}

/// The emulation's texel (module docs) of sampled image `si` under the bound format's bits `bits`,
/// a `half` result when `half` ([`sum`]), its taps addressed by `l`'s state, a runtime sampler's
/// by its `word`, at the sample's plain data `at`.
fn emulate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At,
    (si, image): (SValue, CombinedImage),
    (l, word): (&Linear, Option<SValue>),
    (bits, half, dref): (Bits, bool, Option<SValue>),
) -> BResult<SValue> {
    let Some(taps) = l.taps(word) else {
        return Err(cx.fail("a runtime sampler's linear sample without its word"));
    };
    let (fp, img) = footprint_from(fb, t, at, (si, image), bits, &taps)?;
    sum(
        cx,
        fb,
        (t, &fp),
        (l, word),
        (bits, half, dref),
        Texels::Own(img),
    )
}

/// The emulation's texel (module docs) over footprint `fp` under the bound format's treatment
/// `bits`, a `half` result when `half`, its taps read from `texels`: the exact routine's for an
/// `EXACT` format, else the emulated routine's, both called with the
/// format's bits, which are the result for every format the emulation answers. A raw sample
/// (`l.raw`) reads an 8-bit sRGB format's raw codes: each tap's color lanes become the table's
/// steps (`srgb_raw_tap`), and the exact routine sums them on the sRGB grids, its flags the
/// specialization's (`UNORM`, `SRGB`, and the format's channels, which its border color reads) with
/// `EXACT`. A comparison (`l.compare`, against reference `dref`) compares each tap's depth
/// (`TapCompare`, a border's included) and sums the passes through the exact routine as
/// `compare::PASS_FORMAT`'s codes 0 and 255, whatever the bound format: its value is the sum's
/// first channel, a scalar.
fn sum(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (t, fp): (&Tys, &Footprint),
    (l, word): (&Linear, Option<SValue>),
    (bits, half, dref): (Bits, bool, Option<SValue>),
    texels: Texels<'_>,
) -> BResult<SValue> {
    // Each axis's taps `i` and `i + 1`.
    let mut pairs: Vec<(SValue, [Tap; 2])> = Vec::with_capacity(fp.axes.len());
    for (k, row) in &fp.axes {
        let &[a, b] = row.as_slice() else {
            return Err(cx.fail("a linear footprint axis without two taps"));
        };
        pairs.push((*k, [a, b]));
    }
    let compare = match (l.compare, dref) {
        (Some(c), Some(r)) => Some(TapCompare::of(cx, fb, t, (c, word), (r, bits))?),
        (None, None) => None,
        _ => return Err(cx.fail("a linear comparison without its reference")),
    };
    let flags = match (&compare, l.raw) {
        (Some(_), _) => {
            let pass = texel_format(compare::PASS_FORMAT).0;
            let pass = scalar(fb, t.u32_, u64::from(pass))?;
            value(fb, pass)?
        }
        (None, true) => {
            let flags = bits.value(fb)?;
            let exact = scalar(fb, t.u32_, u64::from(BicubicTexels::EXACT))?;
            let exact = value(fb, exact)?;
            fb.binary(BinOp::BitwiseOr, flags, exact)?
        }
        (None, false) => bits.value(fb)?,
    };
    let h = bool_value(fb, t, half)?;
    let vec4 = t.vec4;
    let post =
        |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, v: SValue| match (&compare, l.raw) {
            (Some(c), _) => {
                let depth = fb.composite_extract(v, &[0])?;
                let pass = c.pass(fb, depth)?;
                fb.composite_construct(vec4, &[pass; 4])
            }
            (None, true) => crate::lower::texture::srgb_raw_tap(cx, fb, v),
            (None, false) => Ok(v),
        };
    let call = |fb: &mut FunctionBuilder<'_>, f, args: &[SValue]| -> BResult<SValue> {
        fb.call(f, args)?.ok_or_else(bicubic::missing_routine)
    };
    // The routines' arguments: a merged axis's fraction 0 (in 2D a float format's too).
    let (exact, float, args) = match pairs.as_slice() {
        [(k, [x0, x1])] => {
            let Some(line) = cx.decl.families.sampler.linear_span else {
                return Err(cx.fail("a 1D linear sample without its routines"));
            };
            let mut args = texels.fetch(cx, fb, (t, fp), &[[*x0], [*x1]], &post)?;
            args.extend([*k, flags, h]);
            (line.exact, line.float, args)
        }
        [(kx, [x0, x1]), (ky, [y0, y1])] => {
            let Some(quad) = cx.decl.families.sampler.linear_quad else {
                return Err(cx.fail("a 2D linear sample without its routines"));
            };
            let sites = [[*x0, *y0], [*x1, *y0], [*x0, *y1], [*x1, *y1]];
            let mut args = texels.fetch(cx, fb, (t, fp), &sites, &post)?;
            args.extend([
                merged(fb, t, *kx, [*x0, *x1])?,
                merged(fb, t, *ky, [*y0, *y1])?,
                flags,
                h,
            ]);
            (quad.exact, quad.float, args)
        }
        [(kx, [x0, x1]), (ky, [y0, y1]), (kz, [z0, z1])] => {
            let Some(block) = cx.decl.families.sampler.linear_block else {
                return Err(cx.fail("a 3D linear sample without its routines"));
            };
            let mut sites = Vec::with_capacity(8);
            for z in [*z0, *z1] {
                for y in [*y0, *y1] {
                    for x in [*x0, *x1] {
                        sites.push([x, y, z]);
                    }
                }
            }
            let mut args = texels.fetch(cx, fb, (t, fp), &sites, &post)?;
            let kx = merged(fb, t, *kx, [*x0, *x1])?;
            let ky = merged(fb, t, *ky, [*y0, *y1])?;
            let kz = merged(fb, t, *kz, [*z0, *z1])?;
            // A mirrored `t` axis's taps in the reflected half of its period run backwards:
            // there Metal's first tap is the second; a runtime sampler's `t` axis mirrors where its
            // word's mode does.
            let flip = match (&l.state, fp.modes.get(1)) {
                (LinearState::Constant { address, .. }, _)
                    if address.get(1) == Some(&TapAddress::Mirror) =>
                {
                    fb.binary(BinOp::UGreaterThan, y0.0, y1.0)?
                }
                (LinearState::Slot { .. } | LinearState::Argument { .. }, Some(&mode)) => {
                    let mirror = scalar(fb, t.u32_, super::bicubic::MIRROR_REPEAT)?;
                    let mirror = value(fb, mirror)?;
                    let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
                    let back = fb.binary(BinOp::UGreaterThan, y0.0, y1.0)?;
                    fb.binary(BinOp::LogicalAnd, mirrors, back)?
                }
                _ => bool_value(fb, t, false)?,
            };
            args.extend([kx, ky, kz, flip, flags, h]);
            (block.exact, block.float, args)
        }
        _ => return Err(cx.fail("a linear sample without one, two, or three plane axes")),
    };
    if compare.is_some() {
        let e = call(fb, exact, &args)?;
        return fb.composite_extract(e, &[0]);
    }
    if l.raw || l.exact_only {
        return call(fb, exact, &args);
    }
    // One routine runs, by a branch on the class, where both ran and a select took one: a
    // driver that keeps the routines as calls runs only the one the class takes, and one that
    // inlines them folds the branch on the specialization as it folded the select.
    let is_exact = bits.test(fb, t, BicubicTexels::EXACT)?;
    let float_args = args.clone();
    let exact_arm: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let e = fb
            .call(exact, &args)?
            .ok_or_else(bicubic::missing_routine)?;
        Ok(ArmEnd::FallThrough(vec![e]))
    });
    let float_arm: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let f = fb
            .call(float, &float_args)?
            .ok_or_else(bicubic::missing_routine)?;
        Ok(ArmEnd::FallThrough(vec![f]))
    });
    let Merged(merged) = fb.if_(cx, is_exact, exact_arm, Some(float_arm))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a linear sum's branch without its texel")),
    }
}

/// The fraction's steps `k` of an axis whose taps are `a` and `b`, or `0` where both are one
/// texel (an index clamped or reflected onto its partner's, a level one texel wide, or both taps
/// outside a border axis): Metal weighs such a pair `(256, 0)` before it rounds a 3D sample's
/// weights and before it truncates a 2D sample's terms.
pub(super) fn merged(
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    k: SValue,
    [a, b]: [Tap; 2],
) -> BResult<SValue> {
    let mut same = fb.binary(BinOp::IEqual, a.0, b.0)?;
    if let (Some(x), Some(y)) = (a.1, b.1) {
        let both = fb.binary(BinOp::LogicalEqual, x, y)?;
        same = fb.binary(BinOp::LogicalAnd, same, both)?;
    }
    let zero = scalar(fb, t.u32_, 0)?;
    let zero = value(fb, zero)?;
    fb.select(same, zero, k)
}

/// The `bool` constant `b`.
fn bool_value(fb: &mut FunctionBuilder<'_>, t: &Tys, b: bool) -> BResult<SValue> {
    let c = fb.module().constant(Constant::Scalar {
        ty: t.bool_,
        bits: ConstBits(u64::from(b)),
    })?;
    value(fb, c)
}
