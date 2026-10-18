//! Samples of an extended-range texture: Metal's nearest, linear and bicubic filters of a
//! `BGR10_XR`, `BGRA10_XR` or `RGB10A8_2P_XR10` texture (or an `_sRGB` twin), emulated by fetches
//! of the raw texels the slot's variant holds (`LoweringInputs::extended_range`).
//!
//! **Why fetches on every device.** No Vulkan format holds these codes: the image holds the
//! texels' bytes in an unsigned integer format, which no device filters, so the shader reads
//! the taps and filters them itself, on a device that samples through Metal's own sampler too.
//!
//! **Metal's answers** (an Apple M3 Max on macOS 26.7 and an Apple M4 on macOS 27.0.1,
//! byte-identical): a nearest sample is the read of the texel at Metal's position, decoded
//! (`extended_range::decode`, its `half` rounded with ties away from zero). A linear sample's
//! footprint is the other formats' (`linear`, the position quantized to 1/256 of a texel, each tap
//! addressed on its own, a border tap the sampler's border color through the format's channels),
//! and its sum is the exact routine's on the format's grid (`bicubic::routines`,
//! `BicubicTexels::EXTENDED`): each tap's code `16 (c - 384)` (an `_sRGB` color step `±s`, alpha
//! `8 (c - 384)` of its clamped code), each term of the 2D product weights truncated (onto `2^-7`
//! of a color lane's steps, `2^-4` of alpha's), the step rounded half up, divided by the grid
//! exactly, and a `half` result rounded to nearest even but an `_sRGB` color lane's, which rounds
//! the exact `f32` with ties away from zero. Every lane of 4,096 random 2D footprints per format
//! and result type matched. A 2D linear sample of a form but a padded or bfloat one hands the
//! routine each tap's step from its codes (`extended_range::steps`, a border tap's its value on
//! each lane's grid), the integer the routine recovers from the decoded value, so no tap is decoded
//! (`bicubic::routines::define_quad_steps`), and the routine takes the sum's value on the grid as
//! `RN(|n| / 255)` exactly scaled (`texel_round::ratio_by_255_to_f32`, an `_sRGB` color lane's
//! `|n| / 8192` exact), a `half` result rounded from it once (`texel_round::normal_half_bits`): a
//! guest fragment's 49 two-plane samples took 84,961 instructions (143:524 spills:fills) and
//! 17.5-17.6 s to make on the Iris Xe (ANV) with decoded taps and the general quotient, and 46,373
//! (84:204) and 6.6 s so.
//!
//! **The two-plane formats** (an Apple M4 on macOS 27.0.1: every lane of 4,096
//! samples of a 16 x 16 texture per format, filter, address mode and result type): the color
//! lanes are `BGR10_XR`'s (or its twin's), and the alpha byte `b` reads `b / 255` (a nearest
//! sample's `half` rounded with ties away from zero, as the decode rounds it) and filters as an
//! 8-bit unorm code on `1/(16 · 255)`, which is alpha's grid `8 · 510` here: the step
//! `⌊Σ 16 b w / 2^16 + 1/2⌋` of the product weights `w`, its `half` rounded to nearest even.
//!
//! **A runtime sampler** reads its slot's `SamplerState` word: the sample is linear where the word
//! states linear minification and magnification, else nearest; the taps follow the word's address
//! modes, border color, and coordinates, and its level the word's mip filter (below). The binder
//! refuses a bound sampler whose word states anything else (`ResourcePlanner::pair`) but a cubic
//! filter (below).
//!
//! **Levels and layers.** A sample reads the level Metal's rule reads (`level`, the rule measured on
//! the other formats: the level rounded to `half`, clamped into the sampler's clamp and the
//! texture's levels, its fraction truncated onto sixty-fourths): level 0 without a mip filter, the
//! level `L + 1` past half a level under a nearest one, and under a linear one levels `L` and
//! `L + 1` summed as one trilinear footprint, the level its third axis
//! (`bicubic::routines::define_levels`: each level's taps under its own fractions, a nearest
//! filter's texel at full weight, the second level weighing `4q` 256ths, each tap's weight the
//! product rounded as a 3D footprint's, each term truncated as a 2D sum's), where the second level
//! weighs anything, else level `L`'s sample. A 2D array's layer is clamped into the array, as
//! sampling clamps it. Measured (the XR mip probe, an Apple M3 Max on macOS 27.0.1): every lane of
//! 8,192 samples per `BGR10_XR` and `BGRA10_XR` form and `_sRGB` twin, nearest and linear filter,
//! nearest and linear mip filter, `float` and `half` result, of 2D and 2D array textures of five
//! levels (16 x 16 down, random codes) at levels from -0.6 to 5.5 (sixty-fourths, the
//! neighbourhoods of half levels, random) and coordinates up to 0.15 past the edges, through a
//! runtime sampler clamping to the edge, and of the 2D ones under a level clamp of `[0.75, 2.5]`.
//! The level is the explicit level or a kernel's bias (level 0 without either); a fragment
//! function's implicit level, shifted by its bias, and explicit gradients take Vulkan's level of
//! detail, which the shader computes (`log2` of the longer of the coordinate's gradients in texels
//! of the base level, on each plane axis), as Metal's level from derivatives is not measured: the sample records
//! `PrecisionNote::DeviceSampleLevel` (through a runtime sampler where its word states a mip
//! filter). A constexpr sampler's clamp bounds are its `half`s; a runtime sampler's are its `f32`s,
//! which its word states for a nearest mip filter (the levels a nearest mip filter reads at them)
//! and its slot's `SamplerMinLevel` specialization for a linear one (the minimum on the grid), the
//! binder holding the maximum past every texture's last level. A linear mip filter is built for the
//! `BGR10_XR` and `BGRA10_XR` forms and their twins (`ExtendedRange::blends_levels`); another
//! form's is refused. A gather and a bicubic sample read the base level.
//!
//! **3D textures** (the XR mip probe: every lane of 8,192 nearest and linear samples of each
//! `BGR10_XR` and `BGRA10_XR` form and `_sRGB` twin's 3D texture of five levels, 16 x 16 x 16
//! down, under a nearest mip filter, `float` and `half`, through a runtime sampler clamping to the
//! edge; and of 8,192 linear samples of `BGRA10_XR`, its twin and `BGR10_XR` under mirrored repeat
//! and under repeat, coordinates up to 1.3 past the edges). A nearest sample reads the texel at
//! Metal's position on the three axes. A linear one sums the 3D footprint's eight taps through
//! the exact 3D routine under the format's bits (`bicubic::routines::Block`, as the other formats'
//! 3D filter): each tap's weight the product of the axes' weights rounded onto `2^-16`, a tie up on
//! the `t` axis's first tap in Metal's frame (its second where a mirrored axis's taps run
//! backwards), each term truncated as a 2D sum's. The level follows the 2D rules under a nearest
//! mip filter. A linear mip filter's blend of a 3D texture's two levels fits no rule tried (the
//! closest, 284 of 8,192 samples apart), so it is refused, by the lowering for a constexpr sampler
//! and by the binder for a runtime one; so are another form's 3D sample (not measured) and a 3D
//! bicubic sample.
//!
//! **Bicubic samples** (an Apple M3 Max on macOS 26.7, the owned fixture `xr_bicubic`: random codes
//! of each `BGR10_XR`, `BGRA10_XR` and two-plane format and `_sRGB` twin at 16 x 16, 7 x 5, 3 x 3,
//! 2 x 6 and 1 x 1, sampled at each texel's centre and at random coordinates up to 0.15 of the
//! texture past its edges, 2,304 per texture, through constexpr bicubic samplers under clamp to
//! edge, repeat, mirrored repeat, clamp to zero and the opaque-black and opaque-white borders, with
//! pixel coordinates, and with a constant offset under clamp to edge and repeat, and through a
//! runtime sampler whose filters are `Cubic` (a constexpr sampler of its state, as for the other
//! formats, `bicubic::cubic_state`), with `float` results and, under clamp to edge, `half` ones:
//! 3,317,760 lanes): the footprint, the weights, the addressing, a mirrored axis and a level below
//! four texels are the other formats' (`bicubic`), each tap the decoded texel or the border texel
//! through the format's channels. A level below four texels sums as this module's linear filter.
//! Otherwise the sum is the exact routine's (`bicubic::routines::Plane::exact`, under the format's
//! bits): each tap's step weighed by the product of Metal's weights and summed exactly, no term
//! truncated as the linear filter's are, the step rounded half up, then clamped into the steps of
//! the format's codes 0 and 1023 (a color lane's decoded endpoints: `(0 - 384) / 510` and
//! `(1023 - 384) / 510`, an `_sRGB` twin's table steps; alpha's `[0, 1]`), and divided by the grid
//! exactly, its `half` rounded as a linear sample's. The clamp is applied to the routine's result:
//! each rounding is monotonic, so clamping the rounded sum into the rounded endpoints is clamping
//! the step. Every lane but 4,050 matched; each of those is one step from Metal's, its exact sum
//! within 0.014 of a step of a rounding tie, which Metal rounds to the other side. No model of
//! Metal's datapath tried fits, so the sample records `PrecisionNote::BicubicFilterArithmetic`. A
//! bfloat or padded form's bicubic sample, a gather through a bicubic sampler, and a swizzled
//! texture's bicubic sample are not measured, and are refused.
//!
//! The bicubic paragraph above is Metal's reading on these formats, except inferences, each the
//! other formats' bicubic reading (`bicubic`; for a runtime sampler) carried to
//! these formats' decoded taps, which no reading of them covers:
//! - a runtime `Cubic` sampler whose other filter is not `Cubic` (the readings' runtime sampler
//!   had both filters `Cubic`): the other formats' runtime sampler answers as both filters bicubic
//!   whatever its other filter;
//! - a runtime sampler under another address mode than clamp to edge, a border color, or pixel
//!   coordinates (the readings' runtime sampler clamped to the edge at normalized coordinates):
//!   the constexpr samplers of those states were read on these formats, and the other formats'
//!   runtime sampler answers as the constexpr sampler of its state;
//! - a `half` result under another address mode than clamp to edge (the readings' `half` results
//!   were all under clamp to edge, constexpr and runtime): rounded as under clamp to edge;
//! - a sample at an explicit level, with a bias or with gradients, and a fragment function's
//!   sample at its implicit level (the readings are kernels' samples, which select level 0): the
//!   other formats' bicubic filter reads the base level whatever the level asked for;
//! - a texture of more than one level, read at its base level (the readings' textures had one
//!   level), as the other formats' bicubic filter reads it.
//!
//! **The bfloat formats** (an Apple M4 under macOS 27.0.1, every code of `R16Bfloat`, `RG16Bfloat`
//! and `RGBA16Bfloat`; 18, 432 positions of each under every address mode and border color): a
//! sample is nearest under either filter, Metal's linear filter reading the nearest texel, the
//! texel at `floor` of the unsnapped coordinate, so a runtime sampler's word selects no filter, and
//! its `half` result is the `float` result toward zero (saturating at `65504`); a gather of a
//! component reads the four taps of the linear footprint (`TapBase::Quantized`), each decoded, that
//! component of each in Vulkan's order `(x0, y1)`, `(x1, y1)`, `(x1, y0)`, `(x0, y0)`, its `half`
//! result rounded to nearest even.
//!
//! **Gathers and swizzles** (an Apple M4 under macOS 27.0.1: reads, nearest and
//! linear samples under an opaque-white border, and gathers of each component, of an 8 x 4 texture
//! of each `BGR10_XR`, `BGRA10_XR`, padded and bfloat format and `_sRGB` twin, at 540 positions,
//! unswizzled and under four swizzles). An extended-range or padded form's `float` gather is the
//! bfloat form's: the four taps of the linear footprint, each the texel its read gives or the
//! border texel through the format's channels. A swizzled texture's answers are the unswizzled
//! texture's, swizzled: the swizzle applies to the decoded texel, a border texel included, and a
//! gather of a component gathers the component the swizzle names (a `Zero` or `One` 0 or 1 at every
//! tap). The slot's view holds raw codes, so it takes no swizzle: the slot joins
//! `LoweringInputs::swizzled_storage` and the function swizzles under its `TextureSwizzle`
//! specialization. A `BGR10_XR` or `BGRA10_XR` form's (or `_sRGB` twin's) `half` gather rounds each
//! tap as its nearest sample does, with ties away from zero, and every gather reads the base level
//! whatever the sampler's mip filter and clamp (the XR mip probe below: every lane of 8,192 gathers
//! of each such form's 2D and 2D array textures of five levels, through a runtime sampler with a
//! nearest mip filter, and of the 2D ones under a clamp whose minimum is 2). A two-plane form's
//! gather, and a padded form's `half` gather, are not measured.
//!
//! **The routine** (`routine::Extended`). A texture slot's sample is one routine per shape: the
//! slot, form, sampler state, bicubic filter, gather, swizzle, 3D, array, mip filter, result type
//! and the lanes of a constant texel offset. Every sample of the shape calls it with what its site
//! computes (the coordinate, the layer, the level its mip filter reads at, a level from the
//! fragment's derivatives or the sample's gradients taken at the site, and the offset, which the
//! routine places as the footprint places a constant one), and the site keeps the status word, so
//! each answer is the one the code written at the site gave. A sample takes the routine where the
//! profile's driver keeps a `DontInline` call a call (`DriverWorkaround::DontInlineKeepsCalls`),
//! which compiles it once: a guest fragment's 49 two-plane samples, at the defaults on the RTX 5080
//! Laptop GPU, took its variant from 446,777 words made in 7.1 to 9.6 s to 259,809 words made in
//! 2.8 to 4.8 s. Elsewhere its code is written at its site (`Sample::routine`): a driver that
//! inlines every call compiles a copy per sample either way, and on ANV (the Iris Xe) the routine
//! form made the same variant in 18.0 to 18.3 s against the site code's 17.6 to 17.7 s. A heap
//! element's sample (`heap_raw`) is its site's own code.

use crate::air::abi::{BorderColor, MipFilter, SamplerState};
use crate::air::ir::ValueId;
use crate::lower::emit::EmitCx;
use crate::metal::bicubic::BicubicTexels;
use crate::metal::format::extended_srgb::SRGB_DECODE_STEPS;
use crate::metal::format::ExtendedRange;
use crate::metal::sampler::SamplerWord;
use crate::metal::{PixelFormat, SamplerMipFilter, TextureSwizzleChannels};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, FuncId, GlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, ImageQuery};

use super::bicubic::{
    footprint_of_image, missing_routine, scalar, value, Addressing, Bicubic, Bits, Footprint, Tap,
    TapAddress, TapBase, Taps, Texels, Tys,
};
use super::routine::{Extended, Shape};
use super::{CombinedImage, Pixel};

/// A sample of an extended-range texture slot, emulated by fetches of its raw texels (module
/// docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample {
    /// The texture slot, whose variant binds [`form`](Self::form).
    pub texture: u32,
    pub form: ExtendedRange,
    /// The sampler's state the sample follows.
    pub state: State,
    /// Metal's bicubic filter of the decoded texels, a constexpr bicubic sampler's or a runtime
    /// cubic sampler's state (module docs); its addressing is also [`state`](Self::state)'s.
    pub cubic: Option<Bicubic>,
    /// A gather of this component (0 to 3, red first) instead of a sample: a bfloat form's, or
    /// another's but a two-plane form's, with a `float` result, or a `half` one but a padded
    /// form's.
    pub gather: Option<u32>,
    /// The slot is one of `LoweringInputs::swizzled_storage`: the sample swizzles its decoded
    /// texel, a border texel included, and a gather each decoded tap before it takes its
    /// component, under the slot's `TextureSwizzle` specialization.
    pub swizzle: bool,
    /// The constant texel offset Metal applies, per plane axis.
    pub offset: Option<Vec<i32>>,
    /// The offset moves an axis that may repeat or mirror: the sample places it by the slot's
    /// `SpecKey::OffsetBaseAxes` tests (`texture::OffsetBase`), which state where Metal's rule
    /// reads a view's parent's base-level size (`position::offset_dropped`).
    pub offset_base: bool,
    /// The plane coordinate: two floats, three of a 3D texture.
    pub coord: ValueId,
    /// A 2D array texture's layer (an integer), clamped into the array as sampling clamps it.
    pub layer: Option<ValueId>,
    /// A 3D texture: the sample reads three plane axes, `r` the third.
    pub volume: bool,
    /// The level the sample reads by its sampler's mip filter.
    pub level: Level,
    /// The result is `half`, else `float`.
    pub half: bool,
    /// The lowered type of the result (`{T, i8}`, its status word 0).
    pub result: DemoteTy,
    /// The sample calls its shape's routine (`routine::Extended`): where the profile's driver
    /// keeps a `DontInline` call a call (`DriverWorkaround::DontInlineKeepsCalls`), so it
    /// compiles the routine once. Elsewhere the code is written at the site: a driver that
    /// inlines every call (Mesa's) compiles a copy per sample either way, and the routine's
    /// passed offset leaves it more to fold (the module docs).
    pub routine: bool,
}

/// The state of the sampler an extended-range sample follows.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum State {
    /// A constexpr sampler's, known at translation.
    Constant {
        /// Linear minification and magnification, else nearest.
        linear: bool,
        /// The addressing of the `s` and `t` axes.
        address: Vec<TapAddress>,
        /// The border color every axis clamping to zero or to the border reads.
        border: BorderColor,
        /// Pixel coordinates: the plane coordinate is already in texels.
        pixel: bool,
    },
    /// The runtime sampler in slot `sampler`: its `SamplerState` specialization's word states
    /// the filter, the address modes, the border color, and the coordinates.
    Slot { sampler: u32 },
}

/// The level an extended-range sample reads (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// The base level: through a constexpr sampler without a mip filter, and a gather's and a
    /// bicubic sample's.
    Base,
    /// A constexpr sampler's mip filter and level clamp, at the level `at` names.
    Stated { stated: Stated, at: LevelAt },
    /// The mip filter of the runtime sampler in slot `sampler`: its word's, under the levels its
    /// clamp leaves (`SamplerWord::LEVEL_MIN`, `SamplerWord::LEVEL_MAX`), and under a linear one
    /// its slot's `SamplerMinLevel` specialization, at the level `at` names.
    Word { sampler: u32, at: LevelAt },
}

impl Level {
    /// The mip filter the level is read by, without where the level comes from.
    pub(super) fn mip(&self) -> Mip {
        match *self {
            Level::Base => Mip::Base,
            Level::Stated { stated, .. } => Mip::Stated(stated),
            Level::Word { sampler, .. } => Mip::Word(sampler),
        }
    }
}

/// The mip filter an extended-range sample's level is read by ([`Level`] without its operand),
/// as its routine takes it (`routine::Extended`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mip {
    /// The base level.
    Base,
    /// A constexpr sampler's mip filter and level clamp.
    Stated(Stated),
    /// The runtime sampler in this slot's.
    Word(u32),
}

/// A constexpr sampler's mip filter (linear, else nearest) and level clamp, the bounds `half`s'
/// values as `f32` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Stated {
    pub linear: bool,
    pub min: u32,
    pub max: u32,
}

impl Stated {
    /// The mip filter and clamp of constexpr sampler `s`, whose mip filter is nearest or linear,
    /// sampling `form`; `None` where its sample is not built: a linear mip filter of a form whose
    /// levels' blend is not measured (`ExtendedRange::blends_levels`), a level bias of the
    /// sampler's own, and an empty or NaN clamp.
    pub(super) fn of(s: &SamplerState, form: ExtendedRange) -> Option<Stated> {
        let (min, max) = (s.lod_min_clamp.to_f32(), s.lod_max_clamp.to_f32());
        if s.lod_bias.to_f32() != 0.0 || min.is_nan() || max.is_nan() || min > max {
            return None;
        }
        let linear = match s.mip_filter {
            MipFilter::Nearest => false,
            MipFilter::Linear if form.blends_levels() => true,
            _ => return None,
        };
        Some(Stated {
            linear,
            min: min.to_bits(),
            max: max.to_bits(),
        })
    }
}

/// Where a mipmapped extended-range sample's level comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelAt {
    /// An explicit level, or a kernel's bias (`None`: level 0).
    Explicit(Option<ValueId>),
    /// A fragment function's implicit level shifted by its bias, or explicit gradients `(dPdx,
    /// dPdy)`: Vulkan's level of detail, which the shader takes (module docs).
    Derived {
        bias: Option<ValueId>,
        grad: Option<(ValueId, ValueId)>,
    },
}

/// The plane axes a 2D texture's sample reads (a 3D texture's read three: [`Sample::axes`]).
pub(super) const AXES: usize = 2;

/// The census words of a nearest sample's tap beyond its position: the size query, the floors, the
/// tap's addressing, the coordinate, the fetch, and the border's select.
pub(super) const NEAREST_WORDS: u32 = 192;

/// The census words of a runtime sampler's branch: the filter's test and the merge.
pub(super) const BRANCH_WORDS: u32 = 48;

/// The census words of one swizzle of a decoded texel: the word's four fields, their tests, and the
/// selects of each lane's source.
pub(super) const SWIZZLE_WORDS: u32 = 192;

/// The census words of a bicubic sample's clamp: the bounds' constants and the clamp.
const CLAMP_WORDS: u32 = 32;

impl Sample {
    /// The plane axes the sample reads: a 2D texture's two, a 3D texture's three.
    pub(super) fn axes(&self) -> usize {
        match self.volume {
            true => 3,
            false => AXES,
        }
    }

    /// The census name.
    pub(crate) fn name(&self) -> &'static str {
        if self.cubic.is_some() {
            return "sampler.extended_range.bicubic";
        }
        if self.gather.is_some() {
            return "sampler.extended_range.gather";
        }
        match self.state {
            State::Constant { linear: true, .. } => "sampler.extended_range.linear",
            State::Constant { linear: false, .. } => "sampler.extended_range.nearest",
            State::Slot { .. } => "sampler.extended_range.runtime",
        }
    }

    /// The estimated words at the sample's site: the routine's call (`routine::Extended`, whose
    /// own words the module counts once, [`body_words`](Self::body_words)) or, where the sample
    /// takes no routine, its code, a level from derivatives or gradients, and the status
    /// composite.
    pub(crate) fn words(&self) -> u32 {
        let derived = match self.level {
            Level::Stated { at, .. } | Level::Word { at, .. } => {
                matches!(at, LevelAt::Derived { .. })
            }
            Level::Base => false,
        };
        let code = match self.routine {
            true => super::routine::CALL_WORDS,
            false => self.body_words(),
        };
        code + if derived { DERIVED_WORDS } else { 0 } + 16
    }

    /// The estimated words of the sample's code (its routine's body): per filter it takes, each
    /// axis's position (which a pixel coordinate does not take, and which a runtime sampler's word
    /// adds its mode to), the footprint and its taps' addressing (a runtime sampler's under every
    /// mode), the fetches and their decodes, the sums' calls (whose own words the module counts
    /// once, `Routine::words`), and Metal's offset rule on each lane of an offset (with the
    /// slot's parent tests, `position::PARENT_RULE_WORDS`); a runtime sampler's word's tests and
    /// its branch. A bicubic sample's are its filter's (`Bicubic::words`), a decode per tap it
    /// reads (the footprint's 16 and a small level's four inner ones), and the clamp.
    pub(crate) fn body_words(&self) -> u32 {
        let axes = self.axes() as u32;
        let decode = crate::lower::arith::extended_range::DECODE_WORDS;
        if let Some(cubic) = &self.cubic {
            return cubic.words() + 20 * decode + CLAMP_WORDS;
        }
        // A routine places its call's offset on each of its lanes (the site code, each lane off
        // zero).
        let moved = self.offset.as_deref().map_or(0, |o| o.len() as u32);
        let rule = match self.offset_base {
            true => super::position::OFFSET_RULE_WORDS + super::position::PARENT_RULE_WORDS,
            false => super::position::OFFSET_RULE_WORDS,
        };
        let offset = moved * rule;
        // A 2D footprint's four taps, a 3D one's eight, each decoded; a 2D linear filter of a
        // form with grid steps takes each tap's steps and, past a border, the border's.
        let corners = 1 << axes;
        let linear = super::linear::PLANE_WORDS + corners * decode + offset;
        let stepped =
            match !self.volume && crate::lower::arith::extended_range::has_steps(self.form) {
                true => super::linear::PLANE_WORDS + corners * (STEPS + STEP_BORDER_WORDS) + offset,
                false => linear,
            };
        let nearest = NEAREST_WORDS + decode + offset;
        let position = axes * super::position::POSITION_WORDS;
        // A gather's footprint and four decodes, or a bfloat form's nearest sample alone, under
        // its addressing.
        let one = match (self.gather, self.form.bfloat()) {
            (Some(_), _) => Some((linear, 2)),
            (None, true) => Some((nearest, 1)),
            (None, false) => None,
        };
        // A swizzle of the sample's texel, or of each of a gather's four taps.
        let swizzle = match (self.swizzle, self.gather) {
            (false, _) => 0,
            (true, None) => SWIZZLE_WORDS,
            (true, Some(_)) => 4 * SWIZZLE_WORDS,
        };
        if let Some((words, taps)) = one {
            let addressing = match &self.state {
                State::Constant { pixel: true, .. } => 0,
                State::Constant { pixel: false, .. } => position,
                State::Slot { .. } => {
                    super::linear::WORD_WORDS
                        + position
                        + axes * super::position::MODE_POSITION_WORDS
                        + taps * axes * super::linear::WORD_TAP_WORDS
                }
            };
            return words + addressing + swizzle + self.level_words().0;
        }
        let filters = |linear: u32| match &self.state {
            State::Constant {
                linear: true,
                pixel,
                ..
            } => linear + if *pixel { 0 } else { position },
            State::Constant {
                linear: false,
                pixel,
                ..
            } => nearest + if *pixel { 0 } else { position },
            State::Slot { .. } => {
                let word = position + axes * super::position::MODE_POSITION_WORDS;
                let taps = axes * super::linear::WORD_TAP_WORDS;
                super::linear::WORD_WORDS
                    + BRANCH_WORDS
                    + (linear + word + 2 * taps)
                    + (nearest + word + taps)
            }
        };
        let (level, blends) = self.level_words();
        let filters = match blends {
            // The two levels' filters of decoded taps and the one level's, the routine's call and
            // the branch.
            true => 2 * filters(linear) + filters(stepped) + BLEND_CALL_WORDS + BRANCH_WORDS,
            false => filters(stepped),
        };
        filters + level + swizzle
    }

    /// The estimated words of the sample's level ([`Level`]) in its routine: its arithmetic
    /// (`level::LEVEL_WORDS` and a word's tests), and whether it may blend two levels' filters.
    /// A level from derivatives or gradients is its site's ([`words`](Self::words)).
    fn level_words(&self) -> (u32, bool) {
        let blends = match self.level {
            Level::Base => return (0, false),
            Level::Stated { stated, .. } => stated.linear,
            Level::Word { .. } => self.form.blends_levels() && !self.volume,
        };
        let pair = if blends { super::level::LEVEL_WORDS } else { 0 };
        (super::level::LEVEL_WORDS + pair, blends)
    }
}

/// The census words of a tap's grid steps (`extended_range::STEPS_WORDS`).
const STEPS: u32 = crate::lower::arith::extended_range::STEPS_WORDS;

/// The census words of a tap's steps past a border: the grids, the border texel's product and
/// conversion, and the select.
const STEP_BORDER_WORDS: u32 = 32;

/// The census words of a level from derivatives or gradients: the derivatives, the size query
/// and its conversion, the gradients' lengths, their maximum and `log2`, and the bias.
const DERIVED_WORDS: u32 = 128;

/// The census words of the levels' blend's call: its operands and the call.
const BLEND_CALL_WORDS: u32 = 24;

/// What a sample's fetches read: the slot's image (image facts `image`), the format's bits for
/// the exact routine and the border texel ([`flags`]), and how its raw texels decode; and the
/// slot's swizzle word, for a swizzled texture of its variant.
pub(super) struct Texture {
    pub img: SValue,
    pub image: CombinedImage,
    pub decode: Decode,
    pub flags: SValue,
    pub swizzle: Option<SValue>,
    /// The slot's `SpecKey::OffsetBaseAxes` tests, by which a texel offset is placed
    /// (`position::offset_dropped`), where the sample reads them ([`Sample::offset_base`]).
    pub parent: Option<crate::lower::texture::OffsetBase>,
}

/// How a sample's raw texels decode: by one form, a slot's variant's, through the form's sRGB
/// table; or by the form a heap handle's class index names (`sampler::heap_raw`), through the
/// heap's decode routine of each result type (`texture::heap_raw::define`).
#[derive(Clone, Copy)]
pub(super) enum Decode {
    Form {
        form: ExtendedRange,
        table: Option<GlobalId>,
    },
    Heap {
        /// The form's index in `ExtendedRange::HEAP_READ` (a `u32`).
        form: SValue,
        /// Whether it is a bfloat form (a `bool`).
        bfloat: SValue,
        /// The routine of a `float` result, and of a `half` one where the access rounds to it.
        routines: (FuncId, Option<FuncId>),
    },
}

/// Where a sample reads: its plane coordinate (two floats), a 2D array's layer (an integer), the
/// constant texel offset Metal applies, per plane axis (a heap element's), or a routine's call's
/// value of it (`passed`: an `int` or `ivecN` of `N` plane axes, `bicubic::At::offset`), the level
/// its taps are fetched from (a `u32`; `None`: the base level), and whether its result is `half`.
#[derive(Clone, Copy)]
pub(super) struct At<'a> {
    pub coord: SValue,
    pub layer: Option<SValue>,
    pub offset: Option<&'a [i32]>,
    pub passed: Option<(SValue, u8)>,
    pub level: Option<SValue>,
    pub half: bool,
}

/// The texel of extended-range sample `e` in the open block, with its status word 0 (module
/// docs): its shape's routine's value (`routine::Extended`), called with the coordinate, the
/// layer and the level its mip filter reads at, each taken in the sample's own position (a level
/// from derivatives or gradients too, so no implicit derivative crosses a call), or where the
/// sample takes no routine ([`Sample::routine`]) the same code written at the site, its constant
/// offset placed as a constant.
pub(crate) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    e: &Sample,
) -> BResult<SValue> {
    let coord = cx.operand(fb, e.coord)?;
    let layer = match e.layer {
        Some(l) => Some(cx.operand(fb, l)?),
        None => None,
    };
    let level = match e.level {
        Level::Base => None,
        Level::Stated { at, .. } | Level::Word { at, .. } => Some(level_at(cx, fb, e, coord, at)?),
    };
    if !e.routine {
        let r = Extended::of(e);
        let x = sampled(
            cx,
            fb,
            &r,
            (coord, layer, level),
            (e.offset.as_deref(), None),
        )?;
        return crate::lower::texture::with_word(cx, fb, &e.result, x, None);
    }
    // A constant texel offset is the call's value, so the samples at every offset call one
    // routine.
    let offset = match &e.offset {
        Some(k) => Some(super::emit::const_offset(fb, k)?),
        None => None,
    };
    let mut args = vec![coord];
    args.extend(layer);
    args.extend(level);
    args.extend(offset);
    let mut tys = Vec::with_capacity(args.len());
    for &v in &args {
        tys.push(fb.value_ty(v)?);
    }
    let shape = Shape::Extended(Box::new(Extended::of(e)));
    let f = match cx.sampling.get(&shape, &tys) {
        Some(f) => f,
        None => {
            let Shape::Extended(r) = &shape else {
                return Err(cx.fail("an extended-range routine of another shape"));
            };
            let mut rcx = cx.for_new_function();
            let made = define(&mut rcx, fb.module(), r, &tys);
            cx.adopt(rcx);
            let f = made?;
            cx.sampling.insert(shape, tys, f);
            f
        }
    };
    let x = fb.call(f, &args)?.ok_or_else(missing_routine)?;
    crate::lower::texture::with_word(cx, fb, &e.result, x, None)
}

/// The level sample `e` (its coordinate `coord`) reads at, as its sampler's mip filter takes it:
/// its explicit level or a kernel's bias (level 0 without either), or Vulkan's level of detail
/// from the fragment's derivatives or the sample's gradients ([`derived_level`]).
fn level_at(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    e: &Sample,
    coord: SValue,
    at: LevelAt,
) -> BResult<SValue> {
    let t = Tys::new(fb)?;
    match at {
        LevelAt::Explicit(Some(v)) => cx.operand(fb, v),
        LevelAt::Explicit(None) => {
            let z = scalar(fb, t.f32_, 0)?;
            value(fb, z)
        }
        LevelAt::Derived { bias, grad } => {
            let (_, img) = crate::lower::texture::slot_image(cx, fb, e.texture)?;
            derived_level(cx, fb, &t, (coord, e.axes() as u8, img), (bias, grad))
        }
    }
}

/// Define the routine of extended-range sample shape `r` (`routine::Extended`) in `mb`, its
/// parameters of types `tys` in [`emit`]'s order: the swizzled texel the sample's code gave at
/// its site, read from `r`'s slot and its specializations.
fn define(
    cx: &mut EmitCx<'_>,
    mb: &mut ModuleBuilder,
    r: &Extended,
    tys: &[TypeId],
) -> BResult<FuncId> {
    let f32_ = mb.ty(crate::spirv::ir::Type::Float { width: 32 })?;
    let ret = mb.ty(crate::spirv::ir::Type::Vector { elem: f32_, len: 4 })?;
    let mut fb = mb.sampling_routine(ret)?;
    let mut params = Vec::with_capacity(tys.len());
    for &ty in tys {
        params.push(fb.param(ty)?);
    }
    let leveled = r.level != Mip::Base;
    let want = 1 + usize::from(r.layer) + usize::from(leveled) + usize::from(r.offset.is_some());
    if params.len() != want {
        return Err(cx.fail("an extended-range routine without its parameters"));
    }
    let mut params = params.into_iter();
    let Some(coord) = params.next() else {
        return Err(cx.fail("an extended-range routine without its coordinate"));
    };
    let layer = r.layer.then(|| params.next()).flatten();
    let level = leveled.then(|| params.next()).flatten();
    let passed = match r.offset {
        Some(lanes) => params.next().map(|v| (v, lanes)),
        None => None,
    };
    let x = sampled(cx, &mut fb, r, (coord, layer, level), (None, passed))?;
    fb.ret(Some(x))?;
    fb.finish()
}

/// The swizzled texel of extended-range sample shape `r` at coordinate `coord`, `layer` and the
/// level `level` its mip filter reads at (module docs), in the open block: its texel offset a
/// constant at the site (`offset`), or a routine's call's value (`passed`).
fn sampled(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    r: &Extended,
    (coord, layer, level): (SValue, Option<SValue>, Option<SValue>),
    (offset, passed): (Option<&[i32]>, Option<(SValue, u8)>),
) -> BResult<SValue> {
    let (decl, img) = crate::lower::texture::slot_image(cx, fb, r.texture)?;
    let table = crate::lower::texture::extended_table(cx, r.form)?;
    let t = Tys::new(fb)?;
    let flags = scalar(fb, t.u32_, u64::from(flags(r.form)))?;
    let swizzle = match r.swizzle {
        true => match cx.decl.families.texture.swizzle.get(&r.texture).copied() {
            Some(word) => Some(word),
            None => {
                return Err(cx.fail(format!(
                    "texture slot {} has no swizzle constant",
                    r.texture
                )))
            }
        },
        false => None,
    };
    let parent = match r.offset_base {
        true => match cx
            .decl
            .families
            .texture
            .offset_base
            .get(&r.texture)
            .copied()
        {
            Some(base) => Some(base),
            None => {
                return Err(cx.fail(format!(
                    "texture slot {} has no OffsetBaseAxes specialization",
                    r.texture
                )))
            }
        },
        false => None,
    };
    let tex = Texture {
        img,
        image: CombinedImage {
            plane: r.axes() as u8,
            arrayed: r.layer,
            texel: decl.texel,
        },
        decode: Decode::Form {
            form: r.form,
            table,
        },
        flags: value(fb, flags)?,
        swizzle,
        parent,
    };
    let at = At {
        coord,
        layer,
        offset,
        passed,
        level: None,
        half: r.half,
    };
    if let Some(cubic) = &r.cubic {
        return bicubic(cx, fb, &t, &at, &tex, (r.form, cubic));
    }
    // A gather, and a bfloat form's sample, follow no filter.
    if r.gather.is_some() || r.form.bfloat() {
        let addressing = match &r.state {
            State::Constant {
                address,
                border,
                pixel,
                ..
            } => Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            },
            State::Slot { sampler } => {
                let word = super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?;
                Addressing::Word {
                    word,
                    axes: r.axes(),
                }
            }
        };
        return match r.gather {
            // A gather swizzles each tap itself.
            Some(component) => gathered(cx, fb, &t, &at, &tex, addressing, component),
            // A bfloat form's level is one level: its forms' blend is not built.
            None => {
                let level = levels(cx, fb, &t, r, level, &tex)?.map(|l| l.first);
                let at = At { level, ..at };
                let x = nearest(cx, fb, &t, &at, &tex, addressing)?;
                swizzled(fb, &tex, x)
            }
        };
    }
    let x = match levels(cx, fb, &t, r, level, &tex)? {
        None => filters(cx, fb, &t, &at, &tex, &r.state)?,
        Some(Levels { first, blend: None }) => {
            let at = At {
                level: Some(first),
                ..at
            };
            filters(cx, fb, &t, &at, &tex, &r.state)?
        }
        // A linear mip filter's two levels, blended where the second weighs anything.
        Some(Levels {
            first,
            blend: Some((second, weight)),
        }) => {
            let zero = scalar(fb, t.u32_, 0)?;
            let zero = value(fb, zero)?;
            let blends = fb.binary(BinOp::INotEqual, weight, zero)?;
            let Some(blend) = cx.decl.families.sampler.level_blend else {
                return Err(cx.fail("an extended-range sample without its levels' blend"));
            };
            let (t, tex, state) = (&t, &tex, &r.state);
            let half = scalar(fb, t.bool_, u64::from(at.half))?;
            let half = value(fb, half)?;
            let both: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let mut args = Vec::with_capacity(15);
                let mut ks = Vec::with_capacity(4);
                for level in [first, second] {
                    let at = At {
                        level: Some(level),
                        ..at
                    };
                    let [t00, t01, t10, t11, kx, ky] = level_taps(cx, fb, t, &at, tex, state)?;
                    args.extend([t00, t01, t10, t11]);
                    ks.extend([kx, ky]);
                }
                args.extend(ks);
                // The second level's weight in 256ths.
                let two = scalar(fb, t.u32_, 2)?;
                let two = value(fb, two)?;
                let kz = fb.binary(BinOp::ShiftLeftLogical, weight, two)?;
                args.extend([kz, tex.flags, half]);
                let x = fb.call(blend, &args)?.ok_or_else(missing_routine)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let one: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let at = At {
                    level: Some(first),
                    ..at
                };
                let x = filters(cx, fb, t, &at, tex, state)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let Merged(merged) = fb.if_(cx, blends, both, Some(one))?;
            match merged[..] {
                [x] => x,
                _ => return Err(cx.fail("an extended-range sample's levels without their texel")),
            }
        }
    };
    swizzled(fb, &tex, x)
}

/// The filtered texel at `at` of `tex` (unswizzled) by `state`: a constexpr sampler's linear or
/// nearest filter, or the filter a runtime sampler's word states.
fn filters(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    state: &State,
) -> BResult<SValue> {
    match state {
        State::Constant {
            linear,
            address,
            border,
            pixel,
        } => {
            let addressing = Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            };
            match linear {
                true => filtered(cx, fb, t, at, tex, addressing),
                false => nearest(cx, fb, t, at, tex, addressing),
            }
        }
        State::Slot { sampler } => {
            let word = super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?;
            let axes = usize::from(tex.image.plane);
            let addressing = Addressing::Word { word, axes };
            let linear = word_linear(fb, t, word)?;
            let filtered: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = filtered(cx, fb, t, at, tex, addressing)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let nearest: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = nearest(cx, fb, t, at, tex, addressing)?;
                Ok(ArmEnd::FallThrough(vec![x]))
            });
            let Merged(merged) = fb.if_(cx, linear, filtered, Some(nearest))?;
            match merged[..] {
                [x] => Ok(x),
                _ => Err(cx.fail("an extended-range sample's branch without its texel")),
            }
        }
    }
}

/// The levels a mipmapped sample reads (module docs): the level (a `u32`), and under a linear mip
/// filter the next one (a `u32`, the last level past it) with its weight in sixty-fourths (a
/// `u32`; 0 where the sample reads one level).
struct Levels {
    first: SValue,
    blend: Option<(SValue, SValue)>,
}

/// The levels sample shape `r` of `tex` reads by its sampler's mip filter at level `x` (module
/// docs, [`level_at`]), or `None` for the base level ([`Mip::Base`]).
fn levels(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    r: &Extended,
    x: Option<SValue>,
    tex: &Texture,
) -> BResult<Option<Levels>> {
    if r.level == Mip::Base {
        return Ok(None);
    }
    let Some(x) = x else {
        return Err(cx.fail("an extended-range sample's mip filter without its level"));
    };
    let count = fb.query(ImageQuery::Levels, tex.img)?;
    let one = scalar(fb, t.u32_, 1)?;
    let one = value(fb, one)?;
    let last = fb.binary(BinOp::ISub, count, one)?;
    let last_f = fb.convert(ConvOp::ConvertUToF, last, t.f32_)?;
    // A linear mip filter's two levels and the next one's weight in sixty-fourths.
    let pair = |fb: &mut FunctionBuilder<'_>, (l, q): (SValue, SValue)| -> BResult<_> {
        let l0 = fb.convert(ConvOp::ConvertFToU, l, t.u32_)?;
        let l1 = fb.binary(BinOp::IAdd, l0, one)?;
        let l1 = fb.glsl(GlslOp::UMin, &[l1, last])?;
        let k64 = scalar(fb, t.f32_, u64::from(64f32.to_bits()))?;
        let k64 = value(fb, k64)?;
        let w = fb.binary_no_contraction(BinOp::FMul, q, k64)?;
        let w = fb.convert(ConvOp::ConvertFToU, w, t.u32_)?;
        Ok((l0, l1, w))
    };
    let levels = match r.level {
        Mip::Base => return Ok(None),
        Mip::Stated(stated) => {
            let min = scalar(fb, t.f32_, u64::from(stated.min))?;
            let min = value(fb, min)?;
            let max = scalar(fb, t.f32_, u64::from(stated.max))?;
            let max = value(fb, max)?;
            let grid = super::level::clamped_on_grid(fb, x, (min, Some(max)), last_f)?;
            match stated.linear {
                false => {
                    let n = super::level::nearest_on_grid(fb, grid)?;
                    Levels {
                        first: fb.convert(ConvOp::ConvertFToU, n, t.u32_)?,
                        blend: None,
                    }
                }
                true => {
                    let (l0, l1, w) = pair(fb, grid)?;
                    Levels {
                        first: l0,
                        blend: Some((l1, w)),
                    }
                }
            }
        }
        Mip::Word(sampler) => {
            let word = super::emit::sampler_word(cx, fb, Pixel::Spec(sampler))?;
            // No mip filter's level 0, or a nearest one's level within the levels its clamp
            // leaves.
            let near = super::level::word_level(fb, word, Some(x))?;
            let near = fb.glsl(GlslOp::UMin, &[near, last])?;
            // A 3D texture's linear mip filter is the binder's refusal.
            if !r.form.blends_levels() || r.volume {
                return Ok(Some(Levels {
                    first: near,
                    blend: None,
                }));
            }
            // A linear one's: its clamp's minimum on Metal's grid, in sixty-fourths; the binder
            // holds its maximum past every level (`ResourcePlanner::extended_sampler`).
            let Some(min) = cx.decl.families.sampler.min_levels.get(&sampler).copied() else {
                return Err(cx.fail(format!("sampler slot {sampler} has no minimum level")));
            };
            let min = fb.convert(ConvOp::ConvertUToF, min, t.f32_)?;
            let step = scalar(fb, t.f32_, u64::from((1f32 / 64.0).to_bits()))?;
            let step = value(fb, step)?;
            let min = fb.binary_no_contraction(BinOp::FMul, min, step)?;
            let grid = super::level::clamped_on_grid(fb, x, (min, None), last_f)?;
            let (l0, l1, w) = pair(fb, grid)?;
            let mip = scalar(fb, t.u32_, u64::from(SamplerWord::MIP_FILTER))?;
            let mip = value(fb, mip)?;
            let linear = SamplerMipFilter::Linear.as_raw() << SamplerWord::MIP_FILTER_SHIFT;
            let linear = scalar(fb, t.u32_, u64::from(linear))?;
            let linear = value(fb, linear)?;
            let have = fb.binary(BinOp::BitwiseAnd, word, mip)?;
            let linear = fb.binary(BinOp::IEqual, have, linear)?;
            let zero = scalar(fb, t.u32_, 0)?;
            let zero = value(fb, zero)?;
            Levels {
                first: fb.select(linear, l0, near)?,
                blend: Some((l1, fb.select(linear, w, zero)?)),
            }
        }
    };
    Ok(Some(levels))
}

/// Vulkan's level of detail of a sample at coordinate `coord` of `axes` plane axes of image `img`
/// at a level from derivatives or gradients (module docs): `log2` of the longer of the
/// coordinate's two gradients in texels of the base level (the fragment's derivatives, or
/// `grad`), plus `bias`.
fn derived_level(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    (coord, axes, img): (SValue, u8, SValue),
    (bias, grad): (Option<ValueId>, Option<(ValueId, ValueId)>),
) -> BResult<SValue> {
    use crate::spirv::ir::DerivOp;
    let (dx, dy) = match grad {
        Some((dx, dy)) => (cx.operand(fb, dx)?, cx.operand(fb, dy)?),
        None => (
            fb.derivative(DerivOp::DPdx, coord)?,
            fb.derivative(DerivOp::DPdy, coord)?,
        ),
    };
    let zero = scalar(fb, t.u32_, 0)?;
    let zero = value(fb, zero)?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    // The plane axes' sizes (a 2D array's layer count left out).
    let mut scaled = Vec::with_capacity(usize::from(axes));
    for lane in 0..u32::from(axes) {
        let n = fb.composite_extract(size, &[lane])?;
        scaled.push(fb.convert(ConvOp::ConvertUToF, n, t.f32_)?);
    }
    let vec = fb.module().ty(crate::spirv::ir::Type::Vector {
        elem: t.f32_,
        len: axes,
    })?;
    let size = fb.composite_construct(vec, &scaled)?;
    let gx = fb.binary(BinOp::FMul, dx, size)?;
    let gy = fb.binary(BinOp::FMul, dy, size)?;
    let gx = fb.glsl(GlslOp::Length, &[gx])?;
    let gy = fb.glsl(GlslOp::Length, &[gy])?;
    let rho = fb.glsl(GlslOp::FMax, &[gx, gy])?;
    let lambda = fb.glsl(GlslOp::Log2, &[rho])?;
    match bias {
        Some(b) => {
            let b = cx.operand(fb, b)?;
            fb.binary(BinOp::FAdd, lambda, b)
        }
        None => Ok(lambda),
    }
}

/// `x`, a sample's or a gather tap's decoded texel (a `vec4` of `f32`, a border texel included),
/// swizzled by `tex`'s swizzle word where it has one, as plain Metal swizzles a sampled texture's
/// texels after decoding them (an Apple M4 under macOS 27.0.1: each read, nearest
/// and linear sample under an opaque-white border, and gather of each component, of an 8 x 4
/// texture of each extended-range, padded and bfloat format under four swizzles, is the
/// unswizzled texture's, swizzled).
fn swizzled(fb: &mut FunctionBuilder<'_>, tex: &Texture, x: SValue) -> BResult<SValue> {
    match tex.swizzle {
        Some(word) => crate::lower::texture::swizzled_float(fb, x, word),
        None => Ok(x),
    }
}

/// The linear sample at `at` of `tex` under `addressing`: the exact 2D routine's sum of the four
/// taps' decoded values, a merged axis's fraction taken as 0; a 3D texture's, the exact 3D
/// routine's sum of the eight ([`block_taps`]).
pub(super) fn filtered(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    addressing: Addressing<'_>,
) -> BResult<SValue> {
    let (routine, mut args) = match tex.image.plane {
        3 => {
            let Some(block) = cx.decl.families.sampler.linear_block else {
                return Err(cx.fail("an extended-range 3D linear sample without its routines"));
            };
            (block.exact, block_taps(cx, fb, t, at, tex, addressing)?)
        }
        // A form with grid steps sums its taps' steps (`bicubic::routines::define_quad_steps`).
        _ if steps_of(tex).is_some() => {
            let Some(routine) = cx.decl.families.sampler.quad_steps else {
                return Err(cx.fail("an extended-range linear sample without its routines"));
            };
            let taps = quad_taps(cx, fb, t, at, tex, addressing, Tapped::Steps)?;
            (routine, taps.to_vec())
        }
        _ => {
            let Some(quad) = cx.decl.families.sampler.linear_quad else {
                return Err(cx.fail("an extended-range linear sample without its routines"));
            };
            let taps = quad_taps(cx, fb, t, at, tex, addressing, Tapped::Values)?;
            (quad.exact, taps.to_vec())
        }
    };
    args.push(tex.flags);
    let half = fb.module().constant(Constant::Scalar {
        ty: t.bool_,
        bits: ConstBits(u64::from(at.half)),
    })?;
    args.push(value(fb, half)?);
    fb.call(routine, &args)?.ok_or_else(missing_routine)
}

/// The linear footprint at `at` of a 3D texture `tex` under `addressing`, as the exact 3D routine
/// takes it (`bicubic::routines::Block`): its eight taps' decoded values (or the border texel),
/// `s` fastest and `r` slowest, the fractions `kx`, `ky`, `kz` in 256ths, a merged axis's 0, and
/// whether the `t` axis's taps run backwards (a mirrored axis's in the reflected half of its
/// period, where Metal's first tap is the second, as `linear`'s 3D emulation takes it).
fn block_taps(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    addressing: Addressing<'_>,
) -> BResult<Vec<SValue>> {
    let taps = Taps {
        addressing,
        offset: at.offset,
        span: &[0, 1],
        base: TapBase::Quantized,
        level: at.level,
    };
    let fp = footprint_of_image(
        fb,
        t,
        (at.coord, at.layer, at.passed),
        (tex.img, tex.image),
        Bits::Value(tex.flags),
        (&taps, tex.parent),
    )?;
    let [(kx, xs), (ky, ys), (kz, zs)] = &fp.axes[..] else {
        return Err(cx.fail("an extended-range 3D linear sample without three plane axes"));
    };
    let (&[x0, x1], &[y0, y1], &[z0, z1]) = (xs.as_slice(), ys.as_slice(), zs.as_slice()) else {
        return Err(cx.fail("an extended-range 3D linear axis without two taps"));
    };
    let mut out = Vec::with_capacity(14);
    for z in [z0, z1] {
        for y in [y0, y1] {
            for x in [x0, x1] {
                out.push(texel(cx, fb, &fp, tex, &[x, y, z], false)?);
            }
        }
    }
    out.push(super::linear::merged(fb, t, *kx, [x0, x1])?);
    out.push(super::linear::merged(fb, t, *ky, [y0, y1])?);
    out.push(super::linear::merged(fb, t, *kz, [z0, z1])?);
    let back = fb.binary(BinOp::UGreaterThan, y0.0, y1.0)?;
    let flip = match (addressing, fp.modes.get(1)) {
        (Addressing::Modes { address, .. }, _) => match address.get(1) {
            Some(TapAddress::Mirror) => back,
            _ => {
                let no = scalar(fb, t.bool_, 0)?;
                value(fb, no)?
            }
        },
        (Addressing::Word { .. }, Some(&mode)) => {
            let mirror = scalar(fb, t.u32_, super::bicubic::MIRROR_REPEAT)?;
            let mirror = value(fb, mirror)?;
            let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
            fb.binary(BinOp::LogicalAnd, mirrors, back)?
        }
        (Addressing::Word { .. }, None) => {
            return Err(cx.fail("an extended-range 3D sample without its t axis's mode"));
        }
    };
    out.push(flip);
    Ok(out)
}

/// What a footprint's taps hand its sum: each tap's decoded value (or the border texel), or its
/// grid steps ([`step`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tapped {
    Values,
    Steps,
}

/// The linear footprint at `at` of `tex` under `addressing`: its four taps, `(i, j)`, `(i + 1,
/// j)`, `(i, j + 1)`, `(i + 1, j + 1)`, as `tapped` hands them, and the fractions `kx`, `ky` in
/// 256ths, a merged axis's 0.
fn quad_taps(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    addressing: Addressing<'_>,
    tapped: Tapped,
) -> BResult<[SValue; 6]> {
    let taps = Taps {
        addressing,
        offset: at.offset,
        span: &[0, 1],
        base: TapBase::Quantized,
        level: at.level,
    };
    let fp = footprint_of_image(
        fb,
        t,
        (at.coord, at.layer, at.passed),
        (tex.img, tex.image),
        Bits::Value(tex.flags),
        (&taps, tex.parent),
    )?;
    let [(kx, xs), (ky, ys)] = &fp.axes[..] else {
        return Err(cx.fail("an extended-range linear sample without two plane axes"));
    };
    let (&[x0, x1], &[y0, y1]) = (xs.as_slice(), ys.as_slice()) else {
        return Err(cx.fail("an extended-range linear axis without two taps"));
    };
    let (kx, ky) = (*kx, *ky);
    let mut out = Vec::with_capacity(6);
    for site in [[x0, y0], [x1, y0], [x0, y1], [x1, y1]] {
        out.push(match tapped {
            Tapped::Values => texel(cx, fb, &fp, tex, &site, false)?,
            Tapped::Steps => step(cx, fb, &fp, tex, &site)?,
        });
    }
    out.push(super::linear::merged(fb, t, kx, [x0, x1])?);
    out.push(super::linear::merged(fb, t, ky, [y0, y1])?);
    out.try_into()
        .map_err(|_| cx.fail("an extended-range linear footprint without its taps"))
}

/// A level's taps for the levels' blend (`bicubic::routines::define_levels`) at `at` of `tex` by
/// `state`: a linear filter's footprint ([`quad_taps`]), or a nearest filter's texel (decoded, a
/// `float`) at each tap and the fractions 0.
fn level_taps(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    state: &State,
) -> BResult<[SValue; 6]> {
    let at = At { half: false, ..*at };
    let one = |cx: &mut EmitCx<'_>,
               fb: &mut FunctionBuilder<'_>,
               addressing: Addressing<'_>|
     -> BResult<[SValue; 6]> {
        let x = nearest(cx, fb, t, &at, tex, addressing)?;
        let zero = scalar(fb, t.u32_, 0)?;
        let zero = value(fb, zero)?;
        Ok([x, x, x, x, zero, zero])
    };
    match state {
        State::Constant {
            linear,
            address,
            border,
            pixel,
        } => {
            let addressing = Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            };
            match linear {
                true => quad_taps(cx, fb, t, &at, tex, addressing, Tapped::Values),
                false => one(cx, fb, addressing),
            }
        }
        State::Slot { sampler } => {
            let word = super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?;
            let addressing = Addressing::Word { word, axes: AXES };
            let linear = word_linear(fb, t, word)?;
            let at = &at;
            let filtered: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = quad_taps(cx, fb, t, at, tex, addressing, Tapped::Values)?;
                Ok(ArmEnd::FallThrough(x.to_vec()))
            });
            let nearest: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                let x = one(cx, fb, addressing)?;
                Ok(ArmEnd::FallThrough(x.to_vec()))
            });
            let Merged(merged) = fb.if_(cx, linear, filtered, Some(nearest))?;
            merged
                .try_into()
                .map_err(|_| cx.fail("an extended-range level's branch without its taps"))
        }
    }
}

/// Whether runtime sampler word `word` states linear minification and magnification (a `bool`): a
/// mip filter, which the word's level reads, filters nothing within a level.
fn word_linear(fb: &mut FunctionBuilder<'_>, t: &Tys, word: SValue) -> BResult<SValue> {
    let mask = SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR;
    let mask = scalar(fb, t.u32_, u64::from(mask))?;
    let mask = value(fb, mask)?;
    let want = scalar(fb, t.u32_, u64::from(SamplerWord::LINEAR))?;
    let want = value(fb, want)?;
    let have = fb.binary(BinOp::BitwiseAnd, word, mask)?;
    fb.binary(BinOp::IEqual, have, want)
}

/// The bicubic sample `cubic` at `at` of `tex` in `form` (module docs): Metal's bicubic filter of
/// the decoded taps, through the exact routines alone, clamped into the format's endpoints.
fn bicubic(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    (form, cubic): (ExtendedRange, &Bicubic),
) -> BResult<SValue> {
    let span = cubic.span();
    let taps = cubic.taps(span);
    let bits = Bits::Value(tex.flags);
    let fp = footprint_of_image(
        fb,
        t,
        (at.coord, at.layer, at.passed),
        (tex.img, tex.image),
        bits,
        (&taps, tex.parent),
    )?;
    let zero = fb.u32_const(0)?;
    let zero = value(fb, zero)?;
    let size = fb.query(ImageQuery::SizeLod(zero), tex.img)?;
    let x = super::bicubic::filter(
        cx,
        fb,
        (t, &fp, size),
        cubic,
        (bits, at.half),
        Texels::Extended(tex),
    )?;
    let [lo, hi] = clamp_bounds(form, at.half).map(|lanes| {
        let mut parts = Vec::with_capacity(lanes.len());
        for x in lanes {
            parts.push(scalar(fb, t.f32_, u64::from(x.to_bits()))?);
        }
        let k = fb.module().constant(Constant::Composite {
            ty: t.vec4,
            parts: parts.into(),
        })?;
        value(fb, k)
    });
    fb.glsl(GlslOp::NClamp, &[x, lo?, hi?])
}

/// The bounds a bicubic sample of `form` is clamped into, as its result reads them (a `half` one's
/// where `half`): per lane, red to alpha, the values of the steps of the codes 0 and 1023 on the
/// lane's grid (module docs), rounded as the exact routine rounds a step (`f32` correctly; `half`
/// to nearest even, an `_sRGB` color lane's `f32` with ties away from zero). A lane the format
/// lacks is 0 or 1 at every tap, within its bounds.
fn clamp_bounds(form: ExtendedRange, half: bool) -> [[f32; 4]; 2] {
    let color = |code: u32| -> f32 {
        let n = code as i64 - 384;
        let (steps, grid) = match form.srgb() {
            true => {
                // Codes 0 and 1023: the table's entries 384 and 639.
                let at = n.unsigned_abs() as usize;
                let s = SRGB_DECODE_STEPS.get(at).copied().map_or(0, i64::from);
                (if n < 0 { -s } else { s }, 8192)
            }
            false => (16 * n, 16 * 510),
        };
        match (half, form.srgb()) {
            (false, _) => steps as f32 / grid as f32,
            // An `_sRGB` step over 8192 is an exact `f32`.
            (true, true) => half_of(steps, grid, true),
            (true, false) => half_of(steps, grid, false),
        }
    };
    let (lo, hi) = (color(0), color(1023));
    [[lo, lo, lo, 0.0], [hi, hi, hi, 1.0]]
}

/// `n / d` (nonzero, within `half`'s normal range) rounded to `half`, with ties away from zero
/// where `away`, else to even, as an `f32`.
fn half_of(n: i64, d: i64, away: bool) -> f32 {
    let a = n.unsigned_abs();
    let d = d.unsigned_abs();
    // `e` with `2^e <= a / d < 2^(e + 1)`.
    let mut e = i32::try_from(a.ilog2()).unwrap_or(0) - i32::try_from(d.ilog2()).unwrap_or(0);
    let at_least = |e: i32| match e >= 0 {
        true => a >= d << e,
        false => a << -e >= d,
    };
    if !at_least(e) {
        e -= 1;
    }
    // The significand's eleven bits and the remainder: `a · 2^(10 - e) = q · d + r`.
    let (num, den) = match 10 - e >= 0 {
        true => (a << (10 - e), d),
        false => (a, d << (e - 10)),
    };
    let (mut q, r) = (num / den, num % den);
    if 2 * r > den || (2 * r == den && (away || q & 1 == 1)) {
        q += 1;
    }
    let m = q as f32 * 2f32.powi(e - 10);
    if n < 0 {
        -m
    } else {
        m
    }
}

/// The nearest sample at `at` of `tex` under `addressing`: the texel at Metal's position, decoded.
pub(super) fn nearest(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    addressing: Addressing<'_>,
) -> BResult<SValue> {
    let taps = Taps {
        addressing,
        offset: at.offset,
        span: &[0],
        base: TapBase::Floor,
        level: at.level,
    };
    let fp = footprint_of_image(
        fb,
        t,
        (at.coord, at.layer, at.passed),
        (tex.img, tex.image),
        Bits::Value(tex.flags),
        (&taps, tex.parent),
    )?;
    // Each plane axis's one tap (a 3D texture's three).
    let mut site = Vec::with_capacity(fp.axes.len());
    for (_, taps) in &fp.axes {
        let &[x] = taps.as_slice() else {
            return Err(cx.fail("an extended-range nearest axis without its tap"));
        };
        site.push(x);
    }
    // A bfloat form's `half` result is its `float` result's toward zero, border texels included,
    // where its reads and gathers round to nearest even.
    match (at.half, tex.decode) {
        (false, _) => texel(cx, fb, &fp, tex, &site, false),
        (true, Decode::Form { form, .. }) if form.bfloat() => {
            let v = texel(cx, fb, &fp, tex, &site, false)?;
            crate::lower::arith::extended_range::bfloat_sample_half(fb, v)
        }
        (true, Decode::Form { .. }) => texel(cx, fb, &fp, tex, &site, true),
        // A heap handle's form: both roundings of the one raw texel, chosen by the form.
        (true, Decode::Heap { bfloat, .. }) => {
            let (raw, inside) = tap(fb, &fp, tex, &site)?;
            let v = decoded(cx, fb, tex, raw, false)?;
            let v = fp.border(fb, v, inside)?;
            let toward = crate::lower::arith::extended_range::bfloat_sample_half(fb, v)?;
            let rounded = decoded(cx, fb, tex, raw, true)?;
            let rounded = fp.border(fb, rounded, inside)?;
            fb.select(bfloat, toward, rounded)
        }
    }
}

/// The gather `e` of `component` of `tex` under `addressing`: the four taps of the linear
/// footprint, each decoded (rounded to `half` for a `half` result) or the border texel, and
/// swizzled by `tex`'s swizzle word where it has one, that component of each in Vulkan's order
/// `(x0, y1)`, `(x1, y1)`, `(x1, y0)`, `(x0, y0)`: a swizzle's `Zero` or `One` 0 or 1 at every tap,
/// another source that channel's taps.
pub(super) fn gathered(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At<'_>,
    tex: &Texture,
    addressing: Addressing<'_>,
    component: u32,
) -> BResult<SValue> {
    let taps = Taps {
        addressing,
        offset: at.offset,
        span: &[0, 1],
        base: TapBase::Quantized,
        level: at.level,
    };
    let fp = footprint_of_image(
        fb,
        t,
        (at.coord, at.layer, at.passed),
        (tex.img, tex.image),
        Bits::Value(tex.flags),
        (&taps, tex.parent),
    )?;
    let [(_, xs), (_, ys)] = &fp.axes[..] else {
        return Err(cx.fail("an extended-range gather without two plane axes"));
    };
    let (&[x0, x1], &[y0, y1]) = (xs.as_slice(), ys.as_slice()) else {
        return Err(cx.fail("an extended-range gather axis without two taps"));
    };
    let mut lanes = Vec::with_capacity(4);
    for site in [[x0, y1], [x1, y1], [x1, y0], [x0, y0]] {
        let v = texel(cx, fb, &fp, tex, &site, at.half)?;
        let v = swizzled(fb, tex, v)?;
        lanes.push(fb.composite_extract(v, &[component])?);
    }
    fb.composite_construct(t.vec4, &lanes)
}

/// The value of the tap at `site`: the raw texel fetched from the base level, decoded
/// (`extended_range::decode`, rounded to `half` when `half`), or the border texel where the tap
/// lies outside a border axis.
pub(super) fn texel(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    fp: &Footprint,
    tex: &Texture,
    site: &[Tap],
    half: bool,
) -> BResult<SValue> {
    let (raw, inside) = tap(fb, fp, tex, site)?;
    let v = decoded(cx, fb, tex, raw, half)?;
    fp.border(fb, v, inside)
}

/// The grid steps of the tap at `site` (`extended_range::steps`, `i32` bits on four `u32` lanes):
/// the raw texel's, fetched from the base level, or the border texel's (its value times each
/// lane's grid, `extended_range::step_grids`) where the tap lies outside a border axis. The exact
/// sum recovers the same integers from the decoded value ([`texel`]), so a sum of steps is a sum
/// of the decoded taps.
fn step(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    fp: &Footprint,
    tex: &Texture,
    site: &[Tap],
) -> BResult<SValue> {
    let Some((form, table)) = steps_of(tex) else {
        return Err(cx.fail("an extended-range tap's steps of a form without them"));
    };
    let (raw, inside) = tap(fb, fp, tex, site)?;
    let steps = crate::lower::arith::extended_range::steps(fb, form, raw, table)?;
    match (inside, fp.border_texel()) {
        (Some(inside), Some(border)) => {
            let grids = crate::lower::arith::extended_range::step_grids(fb, form)?;
            let ty = fb.value_ty(steps)?;
            let b = fb.binary(BinOp::FMul, border, grids)?;
            let b = fb.convert(ConvOp::ConvertFToS, b, ty)?;
            fb.select(inside, steps, b)
        }
        _ => Ok(steps),
    }
}

/// The form and `_sRGB` table of `tex`'s taps where they have grid steps
/// (`extended_range::has_steps`): a slot's form's, never a heap handle's.
fn steps_of(tex: &Texture) -> Option<(ExtendedRange, Option<GlobalId>)> {
    match tex.decode {
        Decode::Form { form, table } if crate::lower::arith::extended_range::has_steps(form) => {
            Some((form, table))
        }
        _ => None,
    }
}

/// The raw texel of the tap at `site`, fetched from the base level, and whether the tap lies
/// inside every border axis (none where no axis has a border).
fn tap(
    fb: &mut FunctionBuilder<'_>,
    fp: &Footprint,
    tex: &Texture,
    site: &[Tap],
) -> BResult<(SValue, Option<SValue>)> {
    let (at, inside) = fp.site(fb, site)?;
    Ok((fp.read(fb, tex.img, at)?, inside))
}

/// Raw texel `raw` decoded by `tex`'s form (`extended_range::decode`, rounded to `half` when
/// `half`), or by the form its heap handle names, through the heap's routine.
fn decoded(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    tex: &Texture,
    raw: SValue,
    half: bool,
) -> BResult<SValue> {
    match tex.decode {
        Decode::Form { form, table } => {
            crate::lower::arith::extended_range::decode(fb, form, raw, table, half)
        }
        Decode::Heap { form, routines, .. } => {
            let routine = match (half, routines) {
                (false, (f, _)) => f,
                (true, (_, Some(h))) => h,
                (true, (_, None)) => return Err(cx.fail("a raw heap sample without its routine")),
            };
            let bool_ = fb.module().ty(crate::spirv::ir::Type::Bool)?;
            let inside = fb.module().constant(Constant::Scalar {
                ty: bool_,
                bits: ConstBits(1),
            })?;
            let inside = fb.module().const_value(inside)?;
            fb.call(routine, &[raw, form, inside])?
                .ok_or_else(missing_routine)
        }
    }
}

/// The bits of `form` the exact routine and the border texel read: `EXACT` and `EXTENDED` (a
/// padded format's `UNORM` and `RGB10A2Unorm`'s packed field instead), `SRGB` for an `_sRGB` twin,
/// its channels (`BGR10_XR` has no alpha), and Metal's border texels through them
/// (`metal::sampler::border_texel`: a channel the format lacks reads 0, alpha 1). A two-plane
/// texel's alpha byte `b` arrives as `b / 255`, which the routine recovers as the code `16 b` on
/// alpha's grid `8 · 510 = 16 · 255`: its truncated terms `b · w` are exact, so the step is
/// Metal's, `⌊Σ 16 b w / 2^16 + 1/2⌋`.
pub(super) fn flags(form: ExtendedRange) -> u32 {
    use crate::metal::texel_format::PackedTexels;
    let extended = BicubicTexels::EXTENDED;
    // A padded format's lanes sum as `RGB10A2Unorm`'s 10-bit ones (its field's grid `4 · 1023`,
    // each 2D term truncated onto `2^-4` of a step): Metal's linear filter of `R10Unorm_X6` and
    // `RG10Unorm_X12` (an Apple M4 and an Apple M3 Max, byte-identical:
    // every channel of 303,104 samples at 256 x 256, and of 110,592 at each of 1298 x 1298,
    // 300 x 77, 17 x 9 and 1 x 1 under clamp-to-edge, normalized and pixel coordinates). Each tap
    // arrives as `c / 1023`, recovered on the scale 1023.
    let padded = BicubicTexels::UNORM | PackedTexels::Rgb10A2.class().0;
    // `R10Unorm_X6_sRGB`'s red sums as a YCbCr field's on the grid `8192`, each tap its decode
    // step `s / 8192` ([`PackedTexels::YCbCr8192`]: each 2D term `s · wx · wy` truncated by
    // `2^7`, the step rounded half up, a `half` result rounded with ties away from zero), Metal's
    // linear filter of it (an Apple M4 and an Apple M3 Max, byte-identical); `RG10Unorm_X12_sRGB`'s
    // reads and filter are `RG10Unorm_X12`'s.
    let padded_srgb = BicubicTexels::UNORM | PackedTexels::YCbCr8192.class().0;
    let (pf, channels, family) = match form {
        ExtendedRange::Bgr10 { srgb: false } => (PixelFormat::BGR10XR, 0b0111, extended),
        ExtendedRange::Bgr10 { srgb: true } => (PixelFormat::BGR10XRSRGB, 0b0111, extended),
        ExtendedRange::Bgra10 { srgb: false } => (PixelFormat::BGRA10XR, 0b1111, extended),
        ExtendedRange::Bgra10 { srgb: true } => (PixelFormat::BGRA10XRSRGB, 0b1111, extended),
        ExtendedRange::Planes { srgb: false } => (PixelFormat::RGB10A82PXR10, 0b1111, extended),
        ExtendedRange::Planes { srgb: true } => (PixelFormat::RGB10A82PXR10SRGB, 0b1111, extended),
        ExtendedRange::R10X6 => (PixelFormat::R10UnormX6, 0b0001, padded),
        ExtendedRange::Rg10X12 => (PixelFormat::RG10UnormX12, 0b0011, padded),
        ExtendedRange::R10X6Srgb => (PixelFormat::R10UnormX6SRGB, 0b0001, padded_srgb),
        ExtendedRange::Rg10X12Srgb => (PixelFormat::RG10UnormX12SRGB, 0b0011, padded),
        // A bfloat form's samples and gathers read its texels unfiltered: its bits name its
        // channels and border texels alone.
        ExtendedRange::R16Bfloat => (PixelFormat::R16Bfloat, 0b0001, 0),
        ExtendedRange::Rg16Bfloat => (PixelFormat::RG16Bfloat, 0b0011, 0),
        ExtendedRange::Rgba16Bfloat => (PixelFormat::RGBA16Bfloat, 0b1111, 0),
    };
    let srgb = if form.srgb() { BicubicTexels::SRGB } else { 0 };
    let bits = BicubicTexels::EXACT | family | srgb | channels << BicubicTexels::PRESENT;
    BicubicTexels(bits)
        .with_border(pf, TextureSwizzleChannels::DEFAULT)
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A four-channel format's border texels are the colors (a two-plane format's too);
    /// `BGR10_XR`'s alpha reads 1 under each color, as its texels' does.
    #[test]
    fn border_texels_follow_the_channels() {
        let border = |form| flags(form) >> BicubicTexels::BORDER & 0xfff;
        // Channel `c`'s bits per color (transparent black, opaque black, opaque white).
        let four = 0b100 | 0b100 << 3 | 0b100 << 6 | 0b110 << 9;
        let three = 0b100 | 0b100 << 3 | 0b100 << 6 | 0b111 << 9;
        for srgb in [false, true] {
            assert_eq!(border(ExtendedRange::Bgra10 { srgb }), four);
            assert_eq!(border(ExtendedRange::Planes { srgb }), four);
            assert_eq!(border(ExtendedRange::Bgr10 { srgb }), three);
            let f = flags(ExtendedRange::Bgra10 { srgb });
            assert_ne!(f & BicubicTexels::EXACT, 0);
            assert_ne!(f & BicubicTexels::EXTENDED, 0);
            assert_eq!(f & BicubicTexels::SRGB != 0, srgb);
        }
    }
}
