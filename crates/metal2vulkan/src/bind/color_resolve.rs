//! Metal's multisample color resolve as a crate-owned shader.
//!
//! Vulkan's `AVERAGE` resolve leaves the average's rounding, and an sRGB average's space, to the
//! device. Metal's is one rule per channel, which probes closed on an Apple M3 Max and an M4 (macOS
//! 26.7, byte-identical): every pair of 8-bit codes at 2 samples, and hundreds of thousands of 2-
//! and 4-sample texels per format, each matched by the rule ([`ResolveRule`]):
//!
//! - **unorm** of `b` bits (8, 10, 16, and the 2-bit alpha): `(sum + n / 2) / n` over the integer
//!   codes;
//! - **snorm** of `b` bits (8, 16): `floor(sum / n + 1 / 2)` over the codes, the code
//!   `-2^(b-1)` read as `-(2^(b-1) - 1)` (both are -1.0);
//! - **sRGB** (8-bit): each code's 12-bit linear value ([`SRGB_RESOLVE_LINEAR`]), their average
//!   `(sum + n / 2) / n`, and Metal's encode of that 12-bit value ([`SRGB_RESOLVE_THRESHOLDS`]);
//!   the alpha of an sRGB format is unorm;
//! - **32-bit float**: the samples summed in sample order in `f32`, then times `1 / n`, each
//!   input, partial sum and the result a denormal flushed to the zero of its sign, and a NaN
//!   anywhere the canonical `0x7fc00000`.
//!
//! The probe of an M3 Max under macOS 27.0.1 (our own `resolve.swift`: 131072 texels per format
//! and count, every pair of codes of a channel of 6 bits or fewer at 2 samples, and a second draw
//! of the small floats' corners) confirmed those rules and closed the rest, each exact on every
//! channel it measured, at 2 and 4 samples:
//!
//! - **unorm read through a `half`** (the packed 16-bit formats' 4-, 5- and 6-bit channels, and
//!   the 1-bit alpha): each code `c` of `m = 2^b - 1` becomes the `half` nearest `c / m` (ties to
//!   even), times `m` rounded toward zero to a `half`: `c`, or the `half` just below `c` where the
//!   first rounding went down (`half_unorm_deficits`). In units of 2^-11 that is
//!   `Y = 2048 c - s(c)`, `s(c)` the `half` step below `c` (2^(e+1) for `c` in `(2^e, 2^(e+1))`,
//!   2^e for `c = 2^e`) where the deficit holds and 0 elsewhere, and the result is
//!   `floor((sum Y + 1024 n) / (2048 n))`;
//! - **16-bit float**: a NaN counts 0 and an infinity 65536 of its sign. At 2 samples both are
//!   floored to the multiple of 2^(e - 14), `e` the larger exponent (each at least -14, an
//!   infinity's 16), and summed. At 4, samples 0 and 1 are so summed into `A`; then `A` and
//!   samples 2 and 3 are floored to the multiple of 2^(m - 14) and summed, `m` the largest of their
//!   exponents, `A`'s as a two's complement adder normalizes it: never above the larger exponent
//!   of samples 0 and 1 (a carry is not counted), a negative power of two one lower. The sum times
//!   `1 / n` rounds to a `half` with a tie toward +∞ (65520 and above is +∞, below -65520 is -∞),
//!   a zero +0. At 2 samples this is the exact average rounded so, as macOS 26.7 measured;
//! - **`RG11B10Float`**: each channel the 16-bit float rule, then that `half` rounded to nearest
//!   even onto the channel's grid (6 or 5 mantissa bits), a finite result past the largest value
//!   (65024, 64512) that value, and an infinity infinite;
//! - **`RGB9E5Float`**: each channel the 16-bit float rule, then the three `half`s encoded as
//!   `EXT_texture_shared_exponent` states (its `floor(x + 1/2)` roundings).
//!
//! Metal has no texture of more than 4 samples on Apple silicon: at 8 and more the modules apply
//! the integer rules, which no Metal answer measures; the 16-bit float rule, and the two
//! small-float formats' that build on it, are not applied there ([`ResolveRule::resolves`]).
//!
//! **The modules** ([`color_resolve_modules`]): a `Vertex` entry `main` that draws one triangle
//! over the whole viewport from `VertexIndex` 0 to 2 (the sample planes merge's), and a `Fragment`
//! entry `main` that reads each of a pixel's samples from its [`ResolveSource`] at
//! [`ColorResolveModules::SOURCE_BINDING`] of set [`ColorResolveModules::SET`], resolves each
//! channel by its rule, and writes the result to the color output at `Location` 0 as the float that
//! the resolve texture's format stores as the resolved code: `code / (2^b - 1)` for unorm,
//! `code / (2^(b-1) - 1)` for snorm, the code's linear value for sRGB (the attachment encodes it),
//! or `code / 255` where the result is written through the raw twin's view
//! ([`ColorResolveFormat::written_raw`]), and the float's own bits. A pixel whose samples all hold
//! the same bits, as every pixel inside a primitive does where one invocation writes each covered
//! sample, skips the average: there every rule but the 32-bit float's gives back the sample's code
//! (a 16-bit float's NaN and -0 as +0), so the module writes that code's float, after a small
//! float's grid step and a shared exponent's encode as after an average. The 32-bit float's rule
//! always averages, as its sum rounds; so does a format of unorm and snorm channels alone, whose
//! average costs about what comparing the samples does: on a Radeon 780M, comparing first took 8%
//! off an `RGBA8Unorm` 1080p 4x resolve's cost over a plain store where every pixel's samples
//! matched, and added 14% where one pixel in 16 straddled an edge
//! (`runtime_lane::resolve_rounding`'s cost test).
//!
//! - [`ResolveSource::InputAttachment`]: the multisampled attachment as the input attachment of
//!   index 0 (`InputAttachmentIndex 0`), in its own format, read at each sample. A unorm or snorm
//!   channel's code is its float times the code's scale, rounded; an sRGB channel's the code whose
//!   linear value is nearest the float; a float's its bits (a half's as the `float` it reads as).
//! - [`ResolveSource::RawInputAttachment`]: the same attachment viewed in the format's raw twin
//!   ([`ColorResolveFormat::raw_format`]: an 8-bit sRGB format's UNORM one), so an sRGB channel's
//!   float is its code over 255, whose code is read as a unorm channel's, with no table lookup.
//!   Another format's raw twin is itself, and the source reads it as
//!   [`ResolveSource::InputAttachment`] does.
//! - [`ResolveSource::Planes`]: the attachment's sample planes (`bind::PlanesCopy`), a sampled
//!   2D-array view of every layer in the copy format, layer `s` holding sample `s`: each channel's
//!   code is its bits in the texel `(FragCoord.xy, s)` (a half's, unpacked to its `float`), or,
//!   for planes held in `A8_UNORM` itself (`PlanesCopy::Alpha8`), its float's code as an input
//!   attachment's is.
//!
//! Draw it into a render pass whose one color attachment is the resolve texture's single-sampled
//! level and layer in [`ColorResolveFormat::written_format`] (its own format, or its raw twin
//! where the result is written raw), over the render area, one sample, without blending; for
//! [`ResolveSource::InputAttachment`] the pass's subpass also reads the multisampled attachment as
//! its input attachment 0, and for [`ResolveSource::RawInputAttachment`] a view of it in its raw
//! twin.
//!
//! **Synchronization** (consumer): the samples' writes before the draw's reads, and the resolve
//! texture's earlier accesses before its color writes.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, GlslOp, ImageOperands, ImageType,
    ImageUsage, IoDir, MemAccess, StorageClass, TexelMemory, Type, TypedModule, UnOp,
};
use crate::vk::Format;

use super::sample_planes::PlanesCopy;

/// How one channel of a format resolves (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResolveRule {
    /// A unorm code of `bits` bits.
    Unorm { bits: u8 },
    /// A unorm code of `bits` bits that Metal reads through a `half`: a channel of a packed
    /// 16-bit format.
    HalfUnorm { bits: u8 },
    /// A snorm code of `bits` bits.
    Snorm { bits: u8 },
    /// An 8-bit sRGB code.
    Srgb,
    /// A 32-bit float.
    Float32,
    /// A 16-bit float, whose rule Metal's measured resolve closes at 2 and 4 samples
    /// ([`ResolveRule::resolves`]).
    Float16,
    /// An unsigned float of a 5-bit exponent and `mantissa` bits (`RG11B10Float`'s), resolved as
    /// a 16-bit float and rounded onto its own grid.
    SmallFloat { mantissa: u8 },
    /// The 9-bit mantissa of a channel that shares the texel's 5-bit exponent at bits 27 to 31
    /// (`RGB9E5Float`'s), resolved as a 16-bit float and encoded with the texel's other channels.
    SharedExponent,
}

impl ResolveRule {
    /// The channel's bits in a texel (a shared exponent's mantissa alone).
    pub fn bits(self) -> u8 {
        match self {
            ResolveRule::Unorm { bits }
            | ResolveRule::HalfUnorm { bits }
            | ResolveRule::Snorm { bits } => bits,
            ResolveRule::Srgb => 8,
            ResolveRule::Float32 => 32,
            ResolveRule::Float16 => 16,
            ResolveRule::SmallFloat { mantissa } => mantissa + 5,
            ResolveRule::SharedExponent => 9,
        }
    }

    /// Whether the rule gives back the sample where every sample of a pixel holds it: every rule's
    /// but a 32-bit float's, whose sum rounds (`3x` at 4 samples) or overflows (`2x`)
    /// (`resolve_alike`).
    fn keeps_alike(self) -> bool {
        self != ResolveRule::Float32
    }

    /// Whether the rule's average costs about what comparing a pixel's samples costs: a unorm or
    /// snorm channel's few integer operations a sample. A format of such channels alone averages
    /// without comparing first (`resolve_texel`).
    fn averages_cheaply(self) -> bool {
        matches!(self, ResolveRule::Unorm { .. } | ResolveRule::Snorm { .. })
    }

    /// Whether Metal's rule of this channel holds at `samples`: an integer rule's at every count,
    /// a 16-bit float's, and so a small float's, at 2 and 4, the counts Metal measures.
    pub fn resolves(self, samples: u32) -> bool {
        match self {
            ResolveRule::Float16 | ResolveRule::SmallFloat { .. } | ResolveRule::SharedExponent => {
                samples == 2 || samples == 4
            }
            _ => true,
        }
    }
}

/// The codes of a unorm channel of `bits` bits (at most 6) whose `half` Metal reads below the
/// code's value: bit `c` is set where the `half` nearest `c / (2^bits - 1)` (ties to even, none of
/// which a code reaches) lies below it (the module docs).
const fn half_unorm_deficits(bits: u32) -> u64 {
    let m = (1u64 << bits) - 1;
    let mut mask = 0u64;
    let mut c = 1u64;
    while c < m {
        // The binade of c / m: 2^-k <= c / m < 2^(1-k), whose `half` step is 2^(-k-10).
        let mut k = 1;
        while c << k < m {
            k += 1;
        }
        let scaled = c << (k + 10);
        let rem = scaled % m;
        if rem != 0 && 2 * rem < m {
            mask |= 1 << c;
        }
        c += 1;
    }
    mask
}

/// One channel of a format: its rule and where its bits lie in a texel (counted from the
/// texel's least significant bit, its first byte's lowest).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolveChannel {
    pub rule: ResolveRule,
    pub offset: u8,
}

/// A color format whose multisample resolve Metal's measured rules state, by component (R, G, B,
/// A): the formats [`ColorResolveFormat::of`] names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorResolveFormat {
    format: Format,
    channels: [Option<ResolveChannel>; 4],
    /// Whether the result is written through the raw twin's view
    /// ([`ColorResolveFormat::written_raw`]).
    written_raw: bool,
}

/// A channel of `rule` at bit `offset`.
const fn at(rule: ResolveRule, offset: u8) -> Option<ResolveChannel> {
    Some(ResolveChannel { rule, offset })
}

impl ColorResolveFormat {
    /// The resolve of `format`: Metal's 8-, 10- and 16-bit unorm formats with their 2-bit alpha,
    /// its 8- and 16-bit snorm ones, its 8-bit sRGB ones, its 32-bit float ones, its packed 16-bit
    /// unorm ones (each Vulkan layout that holds one, its red and blue twin included), and its
    /// 16-bit float, `RG11B10Float` and `RGB9E5Float` ones, whose rules hold at 2 and 4 samples
    /// ([`ColorResolveFormat::resolves`]), as Vulkan formats; `A8Unorm`'s alpha resolves by the
    /// 8-bit unorm rule, as `R8Unorm`'s red (probe `a8`, an M3 Max under macOS 27.0.1: at every
    /// alpha in 1/32 steps an `A8Unorm` resolve is `RGBA8Unorm`'s alpha). `None` for another format
    /// (the module docs).
    pub fn of(format: Format) -> Option<ColorResolveFormat> {
        use ResolveRule::{Float16 as F16, Float32 as F32, SharedExponent as E9, Srgb as S};
        const H1: ResolveRule = ResolveRule::HalfUnorm { bits: 1 };
        const H4: ResolveRule = ResolveRule::HalfUnorm { bits: 4 };
        const H5: ResolveRule = ResolveRule::HalfUnorm { bits: 5 };
        const H6: ResolveRule = ResolveRule::HalfUnorm { bits: 6 };
        const F11: ResolveRule = ResolveRule::SmallFloat { mantissa: 6 };
        const F10: ResolveRule = ResolveRule::SmallFloat { mantissa: 5 };
        const U8: ResolveRule = ResolveRule::Unorm { bits: 8 };
        const N8: ResolveRule = ResolveRule::Snorm { bits: 8 };
        const U16: ResolveRule = ResolveRule::Unorm { bits: 16 };
        const N16: ResolveRule = ResolveRule::Snorm { bits: 16 };
        const U10: ResolveRule = ResolveRule::Unorm { bits: 10 };
        const U2: ResolveRule = ResolveRule::Unorm { bits: 2 };
        let channels = match format {
            Format::R8_UNORM => [at(U8, 0), None, None, None],
            Format::A8_UNORM => [None, None, None, at(U8, 0)],
            Format::R8G8_UNORM => [at(U8, 0), at(U8, 8), None, None],
            Format::R8G8B8A8_UNORM => [at(U8, 0), at(U8, 8), at(U8, 16), at(U8, 24)],
            Format::B8G8R8A8_UNORM => [at(U8, 16), at(U8, 8), at(U8, 0), at(U8, 24)],
            Format::R8_SRGB => [at(S, 0), None, None, None],
            Format::R8G8_SRGB => [at(S, 0), at(S, 8), None, None],
            Format::R8G8B8A8_SRGB => [at(S, 0), at(S, 8), at(S, 16), at(U8, 24)],
            Format::B8G8R8A8_SRGB => [at(S, 16), at(S, 8), at(S, 0), at(U8, 24)],
            Format::R8_SNORM => [at(N8, 0), None, None, None],
            Format::R8G8_SNORM => [at(N8, 0), at(N8, 8), None, None],
            Format::R8G8B8A8_SNORM => [at(N8, 0), at(N8, 8), at(N8, 16), at(N8, 24)],
            Format::R16_UNORM => [at(U16, 0), None, None, None],
            Format::R16G16_UNORM => [at(U16, 0), at(U16, 16), None, None],
            Format::R16G16B16A16_UNORM => [at(U16, 0), at(U16, 16), at(U16, 32), at(U16, 48)],
            Format::R16_SNORM => [at(N16, 0), None, None, None],
            Format::R16G16_SNORM => [at(N16, 0), at(N16, 16), None, None],
            Format::R16G16B16A16_SNORM => [at(N16, 0), at(N16, 16), at(N16, 32), at(N16, 48)],
            Format::A2B10G10R10_UNORM_PACK32 => [at(U10, 0), at(U10, 10), at(U10, 20), at(U2, 30)],
            Format::A2R10G10B10_UNORM_PACK32 => [at(U10, 20), at(U10, 10), at(U10, 0), at(U2, 30)],
            Format::R32_SFLOAT => [at(F32, 0), None, None, None],
            Format::R32G32_SFLOAT => [at(F32, 0), at(F32, 32), None, None],
            Format::R32G32B32A32_SFLOAT => [at(F32, 0), at(F32, 32), at(F32, 64), at(F32, 96)],
            Format::R16_SFLOAT => [at(F16, 0), None, None, None],
            Format::R16G16_SFLOAT => [at(F16, 0), at(F16, 16), None, None],
            Format::R16G16B16A16_SFLOAT => [at(F16, 0), at(F16, 16), at(F16, 32), at(F16, 48)],
            // Vulkan's packed formats are named from the most significant bit.
            Format::R5G6B5_UNORM_PACK16 => [at(H5, 11), at(H6, 5), at(H5, 0), None],
            Format::B5G6R5_UNORM_PACK16 => [at(H5, 0), at(H6, 5), at(H5, 11), None],
            Format::R5G5B5A1_UNORM_PACK16 => [at(H5, 11), at(H5, 6), at(H5, 1), at(H1, 0)],
            Format::B5G5R5A1_UNORM_PACK16 => [at(H5, 1), at(H5, 6), at(H5, 11), at(H1, 0)],
            Format::A1R5G5B5_UNORM_PACK16 => [at(H5, 10), at(H5, 5), at(H5, 0), at(H1, 15)],
            Format::R4G4B4A4_UNORM_PACK16 => [at(H4, 12), at(H4, 8), at(H4, 4), at(H4, 0)],
            Format::B4G4R4A4_UNORM_PACK16 => [at(H4, 4), at(H4, 8), at(H4, 12), at(H4, 0)],
            Format::B10G11R11_UFLOAT_PACK32 => [at(F11, 0), at(F11, 11), at(F10, 22), None],
            Format::E5B9G9R9_UFLOAT_PACK32 => [at(E9, 0), at(E9, 9), at(E9, 18), None],
            _ => return None,
        };
        Some(ColorResolveFormat {
            format,
            channels,
            written_raw: false,
        })
    }

    /// The Vulkan format.
    pub fn format(self) -> Format {
        self.format
    }

    /// Each component's channel (R, G, B, A), `None` where the format has none.
    pub fn channels(self) -> [Option<ResolveChannel>; 4] {
        self.channels
    }

    /// Whether Metal's rules resolve the format at `samples` ([`ResolveRule::resolves`] of every
    /// channel): a 16-bit float, `RG11B10Float` or `RGB9E5Float` format's at 2 and 4 samples, any
    /// other's at every count.
    pub fn resolves(self, samples: u32) -> bool {
        self.channels
            .iter()
            .flatten()
            .all(|c| c.rule.resolves(samples))
    }

    /// The format of the view [`ResolveSource::RawInputAttachment`] reads: an 8-bit sRGB format's
    /// UNORM twin, whose floats are the codes over 255; any other format itself.
    pub fn raw_format(self) -> Format {
        match self.format {
            Format::R8_SRGB => Format::R8_UNORM,
            Format::R8G8_SRGB => Format::R8G8_UNORM,
            Format::R8G8B8A8_SRGB => Format::R8G8B8A8_UNORM,
            Format::B8G8R8A8_SRGB => Format::B8G8R8A8_UNORM,
            format => format,
        }
    }

    /// The resolve writing its result through a view of the format's raw twin
    /// ([`raw_format`](Self::raw_format)): an sRGB channel's code as an 8-bit unorm code's float,
    /// `code / 255`, which the twin's store keeps as the code, in place of the code's linear value,
    /// which the format's own store encodes. Another format's is itself.
    pub fn written_raw(self) -> ColorResolveFormat {
        ColorResolveFormat {
            written_raw: self.raw_format() != self.format,
            ..self
        }
    }

    /// The format of the view the result is written through: the raw twin's where
    /// [`written_raw`](Self::written_raw), else the format.
    pub fn written_format(self) -> Format {
        match self.written_raw {
            true => self.raw_format(),
            false => self.format,
        }
    }

    /// Whether the channels share an exponent (`RGB9E5Float`'s, at bits 27 to 31).
    fn shared_exponent(self) -> bool {
        self.channels
            .iter()
            .flatten()
            .any(|c| c.rule == ResolveRule::SharedExponent)
    }

    /// Whether a channel is sRGB.
    fn srgb(self) -> bool {
        self.channels
            .iter()
            .flatten()
            .any(|c| c.rule == ResolveRule::Srgb)
    }
}

/// Where the fragment stage reads a pixel's samples (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResolveSource {
    /// The multisampled attachment, as input attachment 0.
    InputAttachment,
    /// The multisampled attachment, as input attachment 0, viewed in its format's raw twin
    /// ([`ColorResolveFormat::raw_format`]).
    RawInputAttachment,
    /// The attachment's sample planes, viewed in their copy format.
    Planes(PlanesCopy),
}

impl ResolveSource {
    /// Whether the source's texels are the attachment's bits, in an unsigned-integer copy format:
    /// sample planes of a bits copy (`PlanesCopy::moves_bits`). Else they are the format's
    /// `float`s.
    fn reads_bits(self) -> bool {
        matches!(self, ResolveSource::Planes(copy) if copy.moves_bits())
    }
}

/// The crate-owned resolve modules of one format, sample count and source on one device,
/// validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColorResolveModules {
    /// The vertex stage: SPIR-V 1.5 words, entry point `main`, no resources.
    pub vertex: Arc<[u32]>,
    /// The fragment stage: SPIR-V 1.5 words, entry point `main`, the source at
    /// [`ColorResolveModules::SOURCE_BINDING`] of set [`ColorResolveModules::SET`], the output
    /// at `Location` 0.
    pub fragment: Arc<[u32]>,
    /// What the two stages need from the device.
    pub requirements: Requirements,
}

impl ColorResolveModules {
    /// Descriptor set of the binding.
    pub const SET: u32 = 0;
    /// The source: an input attachment, or the planes' sampled image.
    pub const SOURCE_BINDING: u32 = 0;
}

/// Metal's 12-bit linear value of each 8-bit sRGB code, `round(decode(k) * 4095)` (probes, the M3
/// Max and the M4 alike).
pub const SRGB_RESOLVE_LINEAR: [u16; 256] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 20, 21, 23, 25, 27, 29, 31, 33, 35, 37, 40,
    42, 45, 48, 50, 53, 56, 59, 62, 66, 69, 72, 76, 79, 83, 87, 91, 95, 99, 103, 107, 112, 116,
    121, 126, 131, 136, 141, 146, 151, 156, 162, 168, 173, 179, 185, 191, 197, 204, 210, 216, 223,
    230, 237, 244, 251, 258, 265, 273, 280, 288, 296, 304, 312, 320, 329, 337, 346, 354, 363, 372,
    381, 390, 400, 409, 419, 428, 438, 448, 458, 469, 479, 490, 500, 511, 522, 533, 544, 555, 567,
    578, 590, 602, 614, 626, 639, 651, 664, 676, 689, 702, 715, 728, 742, 755, 769, 783, 797, 811,
    825, 840, 854, 869, 884, 899, 914, 929, 945, 960, 976, 992, 1008, 1024, 1041, 1057, 1074, 1091,
    1108, 1125, 1142, 1159, 1177, 1195, 1213, 1231, 1249, 1267, 1286, 1304, 1323, 1342, 1361, 1381,
    1400, 1420, 1440, 1459, 1480, 1500, 1520, 1541, 1562, 1582, 1603, 1625, 1646, 1668, 1689, 1711,
    1733, 1755, 1778, 1800, 1823, 1846, 1869, 1892, 1916, 1939, 1963, 1987, 2011, 2035, 2059, 2084,
    2109, 2133, 2159, 2184, 2209, 2235, 2260, 2286, 2312, 2339, 2365, 2392, 2419, 2446, 2473, 2500,
    2527, 2555, 2583, 2611, 2639, 2668, 2696, 2725, 2754, 2783, 2812, 2841, 2871, 2901, 2931, 2961,
    2991, 3022, 3052, 3083, 3114, 3146, 3177, 3209, 3240, 3272, 3304, 3337, 3369, 3402, 3435, 3468,
    3501, 3535, 3568, 3602, 3636, 3670, 3705, 3739, 3774, 3809, 3844, 3879, 3915, 3950, 3986, 4022,
    4059, 4095,
];

/// Metal's encode of a 12-bit linear value: the 8-bit code `c` from
/// `SRGB_RESOLVE_THRESHOLDS[c - 1]` on, code 0 below the first. Probes 1 to 3 observed each of
/// the 3991 values a 2- or 4-sample average reaches. Of the 105 no such average reaches (3923 and
/// above), 97 lie between two observed values of one code and take it; 4069 to 4076, between 254
/// and 255 and reached only at 8 or more samples, take the exact encode rounded half up, 254. The
/// table is the exact encode rounded half up but at 804, 1371, 2121 and 2569, each one code
/// higher.
pub const SRGB_RESOLVE_THRESHOLDS: [u16; 255] = [
    1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 23, 24, 26, 28, 30, 32, 34, 37, 39,
    42, 44, 47, 49, 52, 55, 58, 61, 64, 68, 71, 74, 78, 82, 85, 89, 93, 97, 102, 106, 110, 115,
    119, 124, 129, 134, 139, 144, 149, 154, 160, 165, 171, 177, 183, 189, 195, 201, 207, 214, 220,
    227, 234, 241, 248, 255, 262, 270, 277, 285, 293, 300, 308, 317, 325, 333, 342, 350, 359, 368,
    377, 386, 395, 405, 414, 424, 434, 444, 454, 464, 474, 485, 495, 506, 517, 528, 539, 550, 562,
    573, 585, 597, 609, 621, 633, 645, 658, 670, 683, 696, 709, 722, 736, 749, 763, 777, 790, 804,
    819, 833, 848, 862, 877, 892, 907, 922, 938, 953, 969, 985, 1001, 1017, 1033, 1049, 1066, 1083,
    1100, 1117, 1134, 1151, 1169, 1186, 1204, 1222, 1240, 1259, 1277, 1296, 1314, 1333, 1352, 1371,
    1391, 1410, 1430, 1450, 1470, 1490, 1510, 1531, 1552, 1572, 1593, 1615, 1636, 1657, 1679, 1701,
    1723, 1745, 1767, 1790, 1812, 1835, 1858, 1881, 1904, 1928, 1951, 1975, 1999, 2023, 2048, 2072,
    2097, 2121, 2146, 2172, 2197, 2222, 2248, 2274, 2300, 2326, 2352, 2379, 2406, 2433, 2460, 2487,
    2514, 2542, 2569, 2597, 2626, 2654, 2682, 2711, 2740, 2769, 2798, 2827, 2857, 2886, 2916, 2946,
    2977, 3007, 3038, 3068, 3099, 3130, 3162, 3193, 3225, 3257, 3289, 3321, 3354, 3386, 3419, 3452,
    3485, 3518, 3552, 3586, 3620, 3654, 3688, 3722, 3757, 3792, 3827, 3862, 3897, 3933, 3969, 4005,
    4041, 4077,
];

/// The exact linear value of an 8-bit sRGB code (IEC 61966-2-1).
fn srgb_decode(code: u32) -> f64 {
    let c = f64::from(code) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// The binding symbol of the source in the fragment module ([`ColorResolveModules::SOURCE_BINDING`]):
/// an input attachment.
const SOURCE_INPUT: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::InputAttachment(0),
    index: 0,
};

/// The binding symbol of the planes in the fragment module.
const SOURCE_PLANES: ResourceSlot = ResourceSlot {
    stage: Stage::Fragment,
    kind: ResourceKind::Texture,
    index: 0,
};

/// The scalar and vector types the fragment module uses.
#[derive(Clone, Copy)]
struct Types {
    u32_: TypeId,
    i32_: TypeId,
    f32_: TypeId,
    vec4: TypeId,
}

impl Types {
    fn new(b: &mut ModuleBuilder) -> BResult<Types> {
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let i32_ = b.ty(Type::Int {
            width: 32,
            signed: true,
        })?;
        let f32_ = b.ty(Type::Float { width: 32 })?;
        let vec4 = b.ty(Type::Vector { elem: f32_, len: 4 })?;
        Ok(Types {
            u32_,
            i32_,
            f32_,
            vec4,
        })
    }
}

/// A scalar constant of `ty` with `bits`, as a value.
fn scalar(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })?;
    f.module().const_value(c)
}

/// The `u32` constant `v`.
fn uint(f: &mut FunctionBuilder<'_>, t: Types, v: u32) -> BResult<Value> {
    scalar(f, t.u32_, u64::from(v))
}

/// The `i32` constant `v`.
fn int(f: &mut FunctionBuilder<'_>, t: Types, v: i32) -> BResult<Value> {
    scalar(f, t.i32_, u64::from(v as u32))
}

/// The `f32` constant `v`.
fn float(f: &mut FunctionBuilder<'_>, t: Types, v: f32) -> BResult<Value> {
    scalar(f, t.f32_, u64::from(v.to_bits()))
}

/// A `Private` array of `elem` holding `values` (each a scalar's bits).
fn table(
    b: &mut ModuleBuilder,
    elem: TypeId,
    u32_: TypeId,
    values: impl ExactSizeIterator<Item = u64>,
) -> BResult<GlobalId> {
    let len = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(values.len() as u64),
    })?;
    let ty = b.ty(Type::Array {
        elem,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(values.len());
    for bits in values {
        parts.push(b.constant(Constant::Scalar {
            ty: elem,
            bits: ConstBits(bits),
        })?);
    }
    let init = b.constant(Constant::Composite {
        ty,
        parts: parts.into(),
    })?;
    b.plain_global(StorageClass::Private, ty, Some(init))
}

/// The sRGB tables of a module with an sRGB channel.
#[derive(Clone, Copy)]
struct SrgbTables {
    /// [`SRGB_RESOLVE_LINEAR`].
    linear: GlobalId,
    /// [`SRGB_RESOLVE_THRESHOLDS`].
    thresholds: GlobalId,
    /// Each code's linear value as an `f32`: what the attachment encodes as the code. Declared
    /// only where the module writes an sRGB channel's linear value (not
    /// [`ColorResolveFormat::written_raw`]).
    decoded: Option<GlobalId>,
    /// The `f32` midway between each code's linear value and the next code's: an input
    /// attachment's float is the code of the midpoints at or below it. Declared only where the
    /// module reads an sRGB attachment's floats ([`ResolveSource::InputAttachment`]).
    midpoints: Option<GlobalId>,
}

impl SrgbTables {
    /// The tables of `format`'s module reading `source`: [`SrgbTables::midpoints`] among them
    /// where it reads an sRGB attachment's floats, and [`SrgbTables::decoded`] where it writes
    /// linear values.
    fn declare(
        b: &mut ModuleBuilder,
        t: Types,
        format: ColorResolveFormat,
        source: ResolveSource,
    ) -> BResult<SrgbTables> {
        let linear = table(
            b,
            t.u32_,
            t.u32_,
            SRGB_RESOLVE_LINEAR.iter().map(|&v| u64::from(v)),
        )?;
        let thresholds = table(
            b,
            t.u32_,
            t.u32_,
            SRGB_RESOLVE_THRESHOLDS.iter().map(|&v| u64::from(v)),
        )?;
        let decoded = match format.written_raw {
            true => None,
            false => Some(table(
                b,
                t.f32_,
                t.u32_,
                (0..256).map(|k| u64::from((srgb_decode(k) as f32).to_bits())),
            )?),
        };
        let midpoints = match source == ResolveSource::InputAttachment {
            true => Some(table(
                b,
                t.f32_,
                t.u32_,
                (0..255).map(|k| {
                    let mid = (srgb_decode(k) + srgb_decode(k + 1)) / 2.0;
                    u64::from((mid as f32).to_bits())
                }),
            )?),
            false => None,
        };
        Ok(SrgbTables {
            linear,
            thresholds,
            decoded,
            midpoints,
        })
    }
}

/// The float an sRGB channel's resolved `code` (a `uint`) is written as: its linear value, which
/// the format's store encodes as the code, or, written through the raw twin (`tables` without
/// [`SrgbTables::decoded`]), `code / 255`, which the twin's store keeps as the code, as an 8-bit
/// unorm channel's.
fn srgb_written(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    tables: SrgbTables,
    code: Value,
) -> BResult<Value> {
    match tables.decoded {
        Some(decoded) => element(f, decoded, code),
        None => {
            let x = f.convert(ConvOp::ConvertUToF, code, t.f32_)?;
            let scale = float(f, t, (1.0 / 255.0f64) as f32)?;
            f.binary(BinOp::FMul, x, scale)
        }
    }
}

/// The element of the `Private` array `table` at `index`.
fn element(f: &mut FunctionBuilder<'_>, table: GlobalId, index: Value) -> BResult<Value> {
    let base = f.module().global_value(table)?;
    let at = f.access_chain(base, &[index])?;
    f.load(at, MemAccess::default())
}

/// The cubic in `q = x^(1/4)` that [`srgb_estimate`] evaluates, lowest power first: a least
/// squares fit of `255 (1.055 q^(5/3) - 0.055)`, the sRGB encode of `x` in codes (IEC 61966-2-1),
/// over `q` from `0.0031308^(1/4)` to 1, within 0.12 of it there.
const SRGB_CURVE: [f32; 4] = [-18.61511, 66.77221, 240.6806, -33.919685];

/// An estimate of `255 encode(x)`, the sRGB encode of the linear `x` (from 0 to 1) in codes, within
/// 0.12 of it: `255 · 12.92 x` up to 0.0031308, else [`SRGB_CURVE`] at `sqrt(sqrt(x))`.
fn srgb_estimate(f: &mut FunctionBuilder<'_>, t: Types, x: Value) -> BResult<Value> {
    let knee = float(f, t, 0.0031308)?;
    let low = f.binary(BinOp::FOrdLessThanEqual, x, knee)?;
    let slope = float(f, t, 255.0 * 12.92)?;
    let line = f.binary(BinOp::FMul, x, slope)?;
    let root = f.glsl(GlslOp::Sqrt, &[x])?;
    let q = f.glsl(GlslOp::Sqrt, &[root])?;
    let [c0, c1, c2, c3] = SRGB_CURVE;
    let mut curve = float(f, t, c3)?;
    for c in [c2, c1, c0] {
        let product = f.binary(BinOp::FMul, curve, q)?;
        let c = float(f, t, c)?;
        curve = f.binary(BinOp::FAdd, product, c)?;
    }
    f.select(low, line, curve)
}

/// The count of the 255 ascending entries of `table` that `x` reaches (`at_least(x, entry)`), from
/// `estimate`, a float that rounds to within one of that count: the estimate rounded and held to
/// `[0, 255]`, one lower where `x` does not reach the entry below it, one higher where `x` reaches
/// the entry it names. Exact for any such estimate, whatever its error.
fn corrected_rank(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    table: GlobalId,
    x: Value,
    estimate: Value,
    at_least: BinOp,
) -> BResult<Value> {
    let lowest = float(f, t, 0.0)?;
    let highest = float(f, t, 255.0)?;
    let e = f.glsl(GlslOp::RoundEven, &[estimate])?;
    let e = f.glsl(GlslOp::FClamp, &[e, lowest, highest])?;
    let e = f.convert(ConvOp::ConvertFToU, e, t.u32_)?;
    let zero = uint(f, t, 0)?;
    let one = uint(f, t, 1)?;
    // Entry `e - 1`, where `e` is above 0.
    let above_first = f.binary(BinOp::INotEqual, e, zero)?;
    let lifted = f.glsl(GlslOp::UMax, &[e, one])?;
    let below_at = f.binary(BinOp::ISub, lifted, one)?;
    let below = element(f, table, below_at)?;
    let reaches_below = f.binary(at_least, x, below)?;
    let short = f.unary(UnOp::LogicalNot, reaches_below)?;
    let down = f.binary(BinOp::LogicalAnd, above_first, short)?;
    // Entry `e`, where `e` is below 255.
    let entries = uint(f, t, 255)?;
    let inside = f.binary(BinOp::ULessThan, e, entries)?;
    let last = uint(f, t, 254)?;
    let at_index = f.glsl(GlslOp::UMin, &[e, last])?;
    let at = element(f, table, at_index)?;
    let reaches = f.binary(at_least, x, at)?;
    let up = f.binary(BinOp::LogicalAnd, inside, reaches)?;
    let lower = f.binary(BinOp::ISub, e, one)?;
    let higher = f.binary(BinOp::IAdd, e, one)?;
    let moved = f.select(up, higher, e)?;
    f.select(down, lower, moved)
}

/// The code of `channel` in a sample's `texel` read from `source` (the module docs): a `u32`
/// (unorm, sRGB, float bits) or an `i32` (snorm, at least `-(2^(b-1) - 1)`).
fn code(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    srgb: Option<SrgbTables>,
    channel: ResolveChannel,
    component: u32,
    texel: Value,
    source: ResolveSource,
) -> BResult<Value> {
    let bits = u32::from(channel.rule.bits());
    if source.reads_bits() {
        let offset = u32::from(channel.offset);
        let word = f.composite_extract(texel, &[offset / 32])?;
        let shift = uint(f, t, offset % 32)?;
        let count = uint(f, t, bits)?;
        return match channel.rule {
            ResolveRule::Float32 => Ok(word),
            ResolveRule::Unorm { .. } | ResolveRule::HalfUnorm { .. } | ResolveRule::Srgb => {
                f.bit_field_extract(false, word, shift, count)
            }
            // The half's bits, unpacked to the `float` it is (exactly), as that float's bits.
            ResolveRule::Float16 => {
                let half = f.bit_field_extract(false, word, shift, count)?;
                let pair = f.glsl(GlslOp::UnpackHalf2x16, &[half])?;
                let x = f.composite_extract(pair, &[0])?;
                f.convert(ConvOp::Bitcast, x, t.u32_)
            }
            // A small float's exponent is a half's and its mantissa a half's leading bits: its
            // bits moved up to a half's (the sign 0), unpacked to the `float` it is.
            ResolveRule::SmallFloat { mantissa } => {
                let small = f.bit_field_extract(false, word, shift, count)?;
                let up = uint(f, t, 10 - u32::from(mantissa))?;
                let half = f.binary(BinOp::ShiftLeftLogical, small, up)?;
                let pair = f.glsl(GlslOp::UnpackHalf2x16, &[half])?;
                let x = f.composite_extract(pair, &[0])?;
                f.convert(ConvOp::Bitcast, x, t.u32_)
            }
            // The mantissa times 2^(E - 24), `E` the texel's exponent at bits 27 to 31: exact.
            ResolveRule::SharedExponent => {
                let mantissa = f.bit_field_extract(false, word, shift, count)?;
                let at = uint(f, t, 27)?;
                let five = uint(f, t, 5)?;
                let exponent = f.bit_field_extract(false, word, at, five)?;
                let bias = uint(f, t, 127 - 24)?;
                let field = f.binary(BinOp::IAdd, exponent, bias)?;
                let scale = power_of_two(f, t, field)?;
                let m = f.convert(ConvOp::ConvertUToF, mantissa, t.f32_)?;
                let x = f.binary_no_contraction(BinOp::FMul, m, scale)?;
                f.convert(ConvOp::Bitcast, x, t.u32_)
            }
            ResolveRule::Snorm { .. } => {
                let word = f.convert(ConvOp::Bitcast, word, t.i32_)?;
                let v = f.bit_field_extract(true, word, shift, count)?;
                let floor = int(f, t, -((1i32 << (bits - 1)) - 1))?;
                let below = f.binary(BinOp::SLessThan, v, floor)?;
                f.select(below, floor, v)
            }
        };
    }
    let lane = f.composite_extract(texel, &[component])?;
    match channel.rule {
        ResolveRule::Float32
        | ResolveRule::Float16
        | ResolveRule::SmallFloat { .. }
        | ResolveRule::SharedExponent => f.convert(ConvOp::Bitcast, lane, t.u32_),
        ResolveRule::Srgb if source != ResolveSource::RawInputAttachment => {
            let midpoints = srgb
                .and_then(|tables| tables.midpoints)
                .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
            let estimate = srgb_estimate(f, t, lane)?;
            corrected_rank(f, t, midpoints, lane, estimate, BinOp::FOrdGreaterThanEqual)
        }
        // Read through the raw twin, an sRGB channel's float is its code over 255: an 8-bit unorm
        // code's.
        ResolveRule::Unorm { .. } | ResolveRule::HalfUnorm { .. } | ResolveRule::Srgb => {
            let scale = float(f, t, ((1u64 << bits) - 1) as f32)?;
            let half = float(f, t, 0.5)?;
            let x = f.binary(BinOp::FMul, lane, scale)?;
            let x = f.binary(BinOp::FAdd, x, half)?;
            f.convert(ConvOp::ConvertFToU, x, t.u32_)
        }
        ResolveRule::Snorm { .. } => {
            let scale = float(f, t, ((1u64 << (bits - 1)) - 1) as f32)?;
            let x = f.binary(BinOp::FMul, lane, scale)?;
            let x = f.glsl(GlslOp::Round, &[x])?;
            f.convert(ConvOp::ConvertFToS, x, t.i32_)
        }
    }
}

/// Whether the float bits `x` are a NaN.
fn is_nan(f: &mut FunctionBuilder<'_>, t: Types, x: Value) -> BResult<Value> {
    let magnitude = uint(f, t, 0x7fff_ffff)?;
    let inf = uint(f, t, 0x7f80_0000)?;
    let m = f.binary(BinOp::BitwiseAnd, x, magnitude)?;
    f.binary(BinOp::UGreaterThan, m, inf)
}

/// Whether the float bits `x` are an infinity.
fn is_inf(f: &mut FunctionBuilder<'_>, t: Types, x: Value) -> BResult<Value> {
    let magnitude = uint(f, t, 0x7fff_ffff)?;
    let inf = uint(f, t, 0x7f80_0000)?;
    let m = f.binary(BinOp::BitwiseAnd, x, magnitude)?;
    f.binary(BinOp::IEqual, m, inf)
}

/// The float bits `x` with a denormal flushed to the zero of its sign.
fn flush(f: &mut FunctionBuilder<'_>, t: Types, x: Value) -> BResult<Value> {
    let exponent = uint(f, t, 0x7f80_0000)?;
    let sign = uint(f, t, 0x8000_0000)?;
    let zero = uint(f, t, 0)?;
    let e = f.binary(BinOp::BitwiseAnd, x, exponent)?;
    let tiny = f.binary(BinOp::IEqual, e, zero)?;
    let signed_zero = f.binary(BinOp::BitwiseAnd, x, sign)?;
    f.select(tiny, signed_zero, x)
}

/// Metal's resolve of the 32-bit float bits `samples` (the module docs), as float bits. The sum
/// runs in `f32` on the device, whose add Vulkan rounds correctly; what a device may differ in
/// is settled in integers: a NaN, from an input or from opposite infinities, is flagged and
/// written canonical, and a sum or product whose exponent is zero (a denormal, or a zero a device
/// flushed one to) is the zero of the exact result's sign. Two normal values whose sum is a
/// denormal lie within a factor of two of each other, so their sum is exact and its sign is the
/// larger magnitude's.
fn resolve_float(f: &mut FunctionBuilder<'_>, t: Types, samples: &[Value]) -> BResult<Value> {
    let sign = uint(f, t, 0x8000_0000)?;
    let magnitude = uint(f, t, 0x7fff_ffff)?;
    let exponent = uint(f, t, 0x7f80_0000)?;
    let zero = uint(f, t, 0)?;
    let (first, rest) = samples
        .split_first()
        .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let mut sum = flush(f, t, *first)?;
    let mut nan = is_nan(f, t, sum)?;
    for &sample in rest {
        let x = flush(f, t, sample)?;
        let x_nan = is_nan(f, t, x)?;
        nan = f.binary(BinOp::LogicalOr, nan, x_nan)?;
        let both_inf = {
            let a = is_inf(f, t, sum)?;
            let b = is_inf(f, t, x)?;
            f.binary(BinOp::LogicalAnd, a, b)?
        };
        let signs = f.binary(BinOp::BitwiseXor, sum, x)?;
        let signs = f.binary(BinOp::BitwiseAnd, signs, sign)?;
        let opposite = f.binary(BinOp::INotEqual, signs, zero)?;
        let opposite_inf = f.binary(BinOp::LogicalAnd, both_inf, opposite)?;
        nan = f.binary(BinOp::LogicalOr, nan, opposite_inf)?;
        let a = f.convert(ConvOp::Bitcast, sum, t.f32_)?;
        let b = f.convert(ConvOp::Bitcast, x, t.f32_)?;
        let added = f.binary_no_contraction(BinOp::FAdd, a, b)?;
        let added = f.convert(ConvOp::Bitcast, added, t.u32_)?;
        let e = f.binary(BinOp::BitwiseAnd, added, exponent)?;
        let tiny = f.binary(BinOp::IEqual, e, zero)?;
        let ma = f.binary(BinOp::BitwiseAnd, sum, magnitude)?;
        let mb = f.binary(BinOp::BitwiseAnd, x, magnitude)?;
        let larger_a = f.binary(BinOp::UGreaterThan, ma, mb)?;
        let sign_a = f.binary(BinOp::BitwiseAnd, sum, sign)?;
        let sign_b = f.binary(BinOp::BitwiseAnd, x, sign)?;
        let larger = f.select(larger_a, sign_a, sign_b)?;
        // Equal magnitudes sum to +0 unless both are -0 (round to nearest even).
        let both = f.binary(BinOp::BitwiseAnd, sign_a, sign_b)?;
        let equal = f.binary(BinOp::IEqual, ma, mb)?;
        let signed_zero = f.select(equal, both, larger)?;
        sum = f.select(tiny, signed_zero, added)?;
    }
    let count = samples.len() as f32;
    let scale = float(f, t, 1.0 / count)?;
    let a = f.convert(ConvOp::Bitcast, sum, t.f32_)?;
    let product = f.binary_no_contraction(BinOp::FMul, a, scale)?;
    let product = f.convert(ConvOp::Bitcast, product, t.u32_)?;
    let e = f.binary(BinOp::BitwiseAnd, product, exponent)?;
    let tiny = f.binary(BinOp::IEqual, e, zero)?;
    let sum_sign = f.binary(BinOp::BitwiseAnd, sum, sign)?;
    let resolved = f.select(tiny, sum_sign, product)?;
    let canonical = uint(f, t, 0x7fc0_0000)?;
    f.select(nan, canonical, resolved)
}

/// The resolved output float of `channel` from each sample's `codes` (the module docs).
fn resolve(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    srgb: Option<SrgbTables>,
    channel: ResolveChannel,
    codes: &[Value],
) -> BResult<Value> {
    let n = codes.len() as u32;
    let bits = u32::from(channel.rule.bits());
    match channel.rule {
        ResolveRule::Unorm { .. } => {
            let mut sum = uint(f, t, n / 2)?;
            for &c in codes {
                sum = f.binary(BinOp::IAdd, sum, c)?;
            }
            let count = uint(f, t, n)?;
            let mean = f.binary(BinOp::UDiv, sum, count)?;
            let mean = f.convert(ConvOp::ConvertUToF, mean, t.f32_)?;
            let scale = float(f, t, (1.0 / ((1u64 << bits) - 1) as f64) as f32)?;
            f.binary(BinOp::FMul, mean, scale)
        }
        ResolveRule::Snorm { .. } => {
            // floor((2 sum + n) / 2n), over a numerator made positive by 2n * 2^(b-1).
            let offset = 1i32 << (bits - 1);
            let mut sum = int(f, t, 0)?;
            for &c in codes {
                sum = f.binary(BinOp::IAdd, sum, c)?;
            }
            let two = int(f, t, 2)?;
            let twice = f.binary(BinOp::IMul, sum, two)?;
            let bias = int(f, t, n as i32 + 2 * n as i32 * offset)?;
            let numerator = f.binary(BinOp::IAdd, twice, bias)?;
            let denominator = int(f, t, 2 * n as i32)?;
            let quotient = f.binary(BinOp::SDiv, numerator, denominator)?;
            let offset = int(f, t, offset)?;
            let mean = f.binary(BinOp::ISub, quotient, offset)?;
            let mean = f.convert(ConvOp::ConvertSToF, mean, t.f32_)?;
            let scale = float(f, t, (1.0 / ((1u64 << (bits - 1)) - 1) as f64) as f32)?;
            f.binary(BinOp::FMul, mean, scale)
        }
        ResolveRule::Srgb => {
            let tables = srgb.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
            let mut sum = uint(f, t, n / 2)?;
            for &c in codes {
                let linear = element(f, tables.linear, c)?;
                sum = f.binary(BinOp::IAdd, sum, linear)?;
            }
            let count = uint(f, t, n)?;
            let mean = f.binary(BinOp::UDiv, sum, count)?;
            // The 12-bit value as a linear value, `mean / 4095`, estimates its code.
            let real = f.convert(ConvOp::ConvertUToF, mean, t.f32_)?;
            let unit = float(f, t, 1.0 / 4095.0)?;
            let linear = f.binary(BinOp::FMul, real, unit)?;
            let estimate = srgb_estimate(f, t, linear)?;
            let code = corrected_rank(
                f,
                t,
                tables.thresholds,
                mean,
                estimate,
                BinOp::UGreaterThanEqual,
            )?;
            srgb_written(f, t, tables, code)
        }
        ResolveRule::HalfUnorm { .. } => resolve_half_unorm(f, t, bits, codes),
        ResolveRule::Float32 => {
            let bits = resolve_float(f, t, codes)?;
            f.convert(ConvOp::Bitcast, bits, t.f32_)
        }
        // A shared exponent's channel is encoded with the texel's others after
        // ([`encode_shared_exponent`]).
        ResolveRule::Float16 | ResolveRule::SharedExponent => resolve_half(f, t, codes),
        ResolveRule::SmallFloat { mantissa } => {
            let half = resolve_half(f, t, codes)?;
            small_float(f, t, half, u32::from(mantissa))
        }
    }
}

/// Metal's resolve of the codes of a unorm channel of `bits` bits read through a `half` (the
/// module docs), as the output float `code / (2^bits - 1)`: each code `c` is
/// `Y = 2048 c - s(c)` in units of 2^-11, `s(c)` the `half` step below `c` where
/// `half_unorm_deficits` holds `c` and 0 elsewhere, and the code `(sum Y + 1024 n) / (2048 n)`.
fn resolve_half_unorm(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    bits: u32,
    codes: &[Value],
) -> BResult<Value> {
    let n = codes.len() as u32;
    let deficits = half_unorm_deficits(bits);
    let zero = uint(f, t, 0)?;
    let one = uint(f, t, 1)?;
    let eleven = uint(f, t, 11)?;
    let low = uint(f, t, deficits as u32)?;
    let high = uint(f, t, (deficits >> 32) as u32)?;
    let mut sum = uint(f, t, n * 1024)?;
    for &c in codes {
        let scaled = f.binary(BinOp::ShiftLeftLogical, c, eleven)?;
        let y = if deficits == 0 {
            scaled
        } else {
            // The deficit's bit of `c` (below 64): the mask's word `c / 32`, bit `c % 32`.
            let word = if deficits >> 32 == 0 {
                low
            } else {
                let five = uint(f, t, 5)?;
                let index = f.binary(BinOp::ShiftRightLogical, c, five)?;
                let upper = f.binary(BinOp::INotEqual, index, zero)?;
                f.select(upper, high, low)?
            };
            let thirty_one = uint(f, t, 31)?;
            let bit = f.binary(BinOp::BitwiseAnd, c, thirty_one)?;
            let shifted = f.binary(BinOp::ShiftRightLogical, word, bit)?;
            let flag = f.binary(BinOp::BitwiseAnd, shifted, one)?;
            let deficit = f.binary(BinOp::INotEqual, flag, zero)?;
            // The `half` step below `c` in units of 2^-11: 2^e at `c = 2^e`, else 2^(e+1), `e`
            // its leading bit's (`c | 1`, so a code 0, which has no deficit, names a bit too).
            let odd = f.binary(BinOp::BitwiseOr, c, one)?;
            let lead = f.glsl(GlslOp::FindUMsb, &[odd])?;
            let power = f.binary(BinOp::ShiftLeftLogical, one, lead)?;
            let exact = f.binary(BinOp::IEqual, c, power)?;
            let doubled = f.binary(BinOp::ShiftLeftLogical, power, one)?;
            let step = f.select(exact, power, doubled)?;
            let step = f.select(deficit, step, zero)?;
            f.binary(BinOp::ISub, scaled, step)?
        };
        sum = f.binary(BinOp::IAdd, sum, y)?;
    }
    let denominator = uint(f, t, n * 2048)?;
    let code = f.binary(BinOp::UDiv, sum, denominator)?;
    let code = f.convert(ConvOp::ConvertUToF, code, t.f32_)?;
    let scale = float(f, t, (1.0 / ((1u64 << bits) - 1) as f64) as f32)?;
    f.binary(BinOp::FMul, code, scale)
}

/// The `float` 2^(field - 127) of the biased exponent `field`, a `u32` from 1 to 254.
fn power_of_two(f: &mut FunctionBuilder<'_>, t: Types, field: Value) -> BResult<Value> {
    let shift = uint(f, t, 23)?;
    let bits = f.binary(BinOp::ShiftLeftLogical, field, shift)?;
    f.convert(ConvOp::Bitcast, bits, t.f32_)
}

/// The biased exponent field of the float bits `x`, a `u32`.
fn exponent_field(f: &mut FunctionBuilder<'_>, t: Types, x: Value) -> BResult<Value> {
    let magnitude = uint(f, t, 0x7fff_ffff)?;
    let m = f.binary(BinOp::BitwiseAnd, x, magnitude)?;
    let shift = uint(f, t, 23)?;
    f.binary(BinOp::ShiftRightLogical, m, shift)
}

/// The sum of the `float`s `values`, each floored to the multiple of 2^(e - 14), `e` the biased
/// exponent `field` less 127 (from -14 to 16): `floor(v · 2^(14 - e)) · 2^(e - 14)`, each step
/// exact, and the sum too, a multiple of 2^(e - 14) below 2^(e + 3).
fn floored_sum(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    values: &[Value],
    field: Value,
) -> BResult<Value> {
    let fourteen = uint(f, t, 14)?;
    let step_field = f.binary(BinOp::ISub, field, fourteen)?;
    let step = power_of_two(f, t, step_field)?;
    let both_biases = uint(f, t, 127 + 127 + 14)?;
    let inverse_field = f.binary(BinOp::ISub, both_biases, field)?;
    let inverse = power_of_two(f, t, inverse_field)?;
    let mut sum: Option<Value> = None;
    for &v in values {
        let units = f.binary_no_contraction(BinOp::FMul, v, inverse)?;
        let units = f.glsl(GlslOp::Floor, &[units])?;
        let floored = f.binary_no_contraction(BinOp::FMul, units, step)?;
        sum = Some(match sum {
            None => floored,
            Some(acc) => f.binary_no_contraction(BinOp::FAdd, acc, floored)?,
        });
    }
    sum.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)
}

/// Metal's resolve of 2 or 4 16-bit float samples, each the bits of the `float` it reads as (the
/// module docs), as the `float` of the resolved `half`, which the attachment stores exactly.
///
/// A NaN counts 0 and an infinity 65536 of its sign; each sample's exponent is its `float`'s, at
/// least -14 (an infinity's 16). Every step below is exact in `f32` ([`floored_sum`]): at 2
/// samples the sum is the two floored to the multiple of 2^(e - 14), `e` the larger exponent; at
/// 4, samples 0 and 1 are so summed into `A`, whose exponent is its `float`'s, one lower for a
/// negative power of two (a two's complement adder's), and never above the larger of theirs;
/// then `A` and samples 2 and 3 are floored to the multiple of 2^(m - 14), `m` the largest of the
/// three exponents, and summed. The sum times `1 / n` (exact) rounds to the `half` grid with a tie
/// toward +∞ as `floor(x · (1/u) + 1/2) · u`, `u` the grid's step at `x` (2^-24 below 2^-14,
/// else 2^(e - 10) for `x` in `[2^e, 2^(e+1))`): each step exact, and every nonzero value a
/// normal `float`, which no device flushes. A result of 2^16 or more is an infinity; a zero is
/// +0.
fn resolve_half(f: &mut FunctionBuilder<'_>, t: Types, codes: &[Value]) -> BResult<Value> {
    let sign = uint(f, t, 0x8000_0000)?;
    let zero_bits = uint(f, t, 0)?;
    let big = uint(f, t, 0x4780_0000)?; // 65536.0
    let lowest = uint(f, t, 127 - 14)?;
    let mut samples = Vec::with_capacity(codes.len());
    let mut exponents = Vec::with_capacity(codes.len());
    for &x in codes {
        let nan = is_nan(f, t, x)?;
        let inf = is_inf(f, t, x)?;
        let s = f.binary(BinOp::BitwiseAnd, x, sign)?;
        let signed_big = f.binary(BinOp::BitwiseOr, s, big)?;
        let x = f.select(inf, signed_big, x)?;
        let x = f.select(nan, zero_bits, x)?;
        let field = exponent_field(f, t, x)?;
        exponents.push(f.glsl(GlslOp::UMax, &[field, lowest])?);
        samples.push(f.convert(ConvOp::Bitcast, x, t.f32_)?);
    }
    let sum = match (samples.as_slice(), exponents.as_slice()) {
        (&[a, b], &[ea, eb]) => {
            let e = f.glsl(GlslOp::UMax, &[ea, eb])?;
            floored_sum(f, t, &[a, b], e)?
        }
        (&[a, b, c, d], &[ea, eb, ec, ed]) => {
            let pair = f.glsl(GlslOp::UMax, &[ea, eb])?;
            let first = floored_sum(f, t, &[a, b], pair)?;
            // `A`'s exponent as a two's complement adder normalizes it: a negative power of two
            // (mantissa 0) one lower, a zero's field 0 below every sample's.
            let bits = f.convert(ConvOp::Bitcast, first, t.u32_)?;
            let field = exponent_field(f, t, bits)?;
            let negative = f.binary(BinOp::UGreaterThanEqual, bits, sign)?;
            let fraction = uint(f, t, 0x007f_ffff)?;
            let mantissa = f.binary(BinOp::BitwiseAnd, bits, fraction)?;
            let power = f.binary(BinOp::IEqual, mantissa, zero_bits)?;
            let nonzero = f.binary(BinOp::INotEqual, field, zero_bits)?;
            let lower = f.binary(BinOp::LogicalAnd, negative, power)?;
            let lower = f.binary(BinOp::LogicalAnd, lower, nonzero)?;
            let one = uint(f, t, 1)?;
            let below = f.binary(BinOp::ISub, field, one)?;
            let field = f.select(lower, below, field)?;
            let field = f.glsl(GlslOp::UMin, &[field, pair])?;
            let rest = f.glsl(GlslOp::UMax, &[ec, ed])?;
            let e = f.glsl(GlslOp::UMax, &[field, rest])?;
            floored_sum(f, t, &[first, c, d], e)?
        }
        _ => return Err(crate::spirv::error::BuildError::IdSpaceExhausted),
    };
    let scale = float(f, t, 1.0 / codes.len() as f32)?;
    let x = f.binary_no_contraction(BinOp::FMul, sum, scale)?;
    // The grid's step: 2^(max(e, -14) - 10), from the exponent field of |x|'s bits.
    let bits = f.convert(ConvOp::Bitcast, x, t.u32_)?;
    let magnitude = uint(f, t, 0x7fff_ffff)?;
    let field = exponent_field(f, t, bits)?;
    let field = f.glsl(GlslOp::UMax, &[field, lowest])?;
    let ten = uint(f, t, 10)?;
    let step_field = f.binary(BinOp::ISub, field, ten)?;
    let step = power_of_two(f, t, step_field)?;
    // `1 / u`, a power of two too, so each product is exact.
    let twice_bias = uint(f, t, 254)?;
    let inverse_field = f.binary(BinOp::ISub, twice_bias, step_field)?;
    let inverse = power_of_two(f, t, inverse_field)?;
    let units = f.binary_no_contraction(BinOp::FMul, x, inverse)?;
    let half = float(f, t, 0.5)?;
    let units = f.binary_no_contraction(BinOp::FAdd, units, half)?;
    let units = f.glsl(GlslOp::Floor, &[units])?;
    let rounded = f.binary_no_contraction(BinOp::FMul, units, step)?;
    // 2^16 or more is the infinity of its sign; a zero is +0.
    let rounded_bits = f.convert(ConvOp::Bitcast, rounded, t.u32_)?;
    let rm = f.binary(BinOp::BitwiseAnd, rounded_bits, magnitude)?;
    let past = f.binary(BinOp::UGreaterThanEqual, rm, big)?;
    let rs = f.binary(BinOp::BitwiseAnd, rounded_bits, sign)?;
    let inf_bits = uint(f, t, 0x7f80_0000)?;
    let signed_inf = f.binary(BinOp::BitwiseOr, rs, inf_bits)?;
    let out = f.select(past, signed_inf, rounded_bits)?;
    let is_zero = f.binary(BinOp::IEqual, rm, zero_bits)?;
    let out = f.select(is_zero, zero_bits, out)?;
    f.convert(ConvOp::Bitcast, out, t.f32_)
}

/// `value`, the resolved `half` of an unsigned small float of `mantissa` bits (a `float`, +0 or
/// more, or +∞), rounded as Metal does (the module docs): to nearest even onto the small float's
/// grid (2^-14 its least exponent, as a `half`'s), a finite result past the largest finite value
/// that value, an infinity infinite. Each step is exact.
fn small_float(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    value: Value,
    mantissa: u32,
) -> BResult<Value> {
    let bits = f.convert(ConvOp::Bitcast, value, t.u32_)?;
    let inf_bits = uint(f, t, 0x7f80_0000)?;
    let infinite = f.binary(BinOp::IEqual, bits, inf_bits)?;
    let field = exponent_field(f, t, bits)?;
    let lowest = uint(f, t, 127 - 14)?;
    let field = f.glsl(GlslOp::UMax, &[field, lowest])?;
    let width = uint(f, t, mantissa)?;
    let step_field = f.binary(BinOp::ISub, field, width)?;
    let step = power_of_two(f, t, step_field)?;
    let twice_bias = uint(f, t, 254)?;
    let inverse_field = f.binary(BinOp::ISub, twice_bias, step_field)?;
    let inverse = power_of_two(f, t, inverse_field)?;
    let units = f.binary_no_contraction(BinOp::FMul, value, inverse)?;
    let units = f.glsl(GlslOp::RoundEven, &[units])?;
    let rounded = f.binary_no_contraction(BinOp::FMul, units, step)?;
    let largest = (2.0 - (-(mantissa as f64)).exp2()) * 32768.0;
    let largest = float(f, t, largest as f32)?;
    let rounded = f.glsl(GlslOp::FMin, &[rounded, largest])?;
    let infinity = f.convert(ConvOp::Bitcast, inf_bits, t.f32_)?;
    f.select(infinite, infinity, rounded)
}

/// The resolved `half`s `channels` (red, green, blue: `float`s from +0 to 65408, the largest
/// sample) of a shared-exponent texel, encoded as `EXT_texture_shared_exponent` states and
/// returned as the `float`s the texel then holds, which the attachment stores exactly: each
/// clamped to 65408, the exponent `max(-16, floor(log2(max))) + 16` (one more where the largest
/// channel's mantissa `floor(max / 2^(E - 24) + 1/2)` reaches 512), each mantissa
/// `floor(c / 2^(E - 24) + 1/2)`. Each step is exact in `f32`; an addition of 1/2 that rounds is
/// one to a value below 2^-13, whose floor is 0 either way.
fn encode_shared_exponent(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    channels: [Value; 3],
) -> BResult<[Value; 3]> {
    let largest = float(f, t, 65408.0)?;
    let mut clamped = [channels[0]; 3];
    for (out, &c) in clamped.iter_mut().zip(channels.iter()) {
        *out = f.glsl(GlslOp::FMin, &[c, largest])?;
    }
    let most = f.glsl(GlslOp::FMax, &[clamped[0], clamped[1]])?;
    let most = f.glsl(GlslOp::FMax, &[most, clamped[2]])?;
    // The shared exponent E as the `float` exponent field E + 127 - 16: the largest channel's
    // field, at least 127 - 16.
    let bits = f.convert(ConvOp::Bitcast, most, t.u32_)?;
    let field = exponent_field(f, t, bits)?;
    let lowest = uint(f, t, 127 - 16)?;
    let field = f.glsl(GlslOp::UMax, &[field, lowest])?;
    let half = float(f, t, 0.5)?;
    // The step 2^(E - 24) is the field (E + 111) less 8, its inverse the field 254 less that.
    let scaled = |f: &mut FunctionBuilder<'_>, field: Value| -> BResult<(Value, Value)> {
        let eight = uint(f, t, 8)?;
        let step_field = f.binary(BinOp::ISub, field, eight)?;
        let step = power_of_two(f, t, step_field)?;
        let twice_bias = uint(f, t, 254)?;
        let inverse_field = f.binary(BinOp::ISub, twice_bias, step_field)?;
        let inverse = power_of_two(f, t, inverse_field)?;
        Ok((step, inverse))
    };
    let round = |f: &mut FunctionBuilder<'_>, c: Value, inverse: Value| -> BResult<Value> {
        let units = f.binary_no_contraction(BinOp::FMul, c, inverse)?;
        let units = f.binary_no_contraction(BinOp::FAdd, units, half)?;
        f.glsl(GlslOp::Floor, &[units])
    };
    let (_, inverse) = scaled(f, field)?;
    let top = round(f, most, inverse)?;
    let full = float(f, t, 512.0)?;
    let carried = f.binary(BinOp::FOrdEqual, top, full)?;
    let one = uint(f, t, 1)?;
    let up = f.binary(BinOp::IAdd, field, one)?;
    let field = f.select(carried, up, field)?;
    let (step, inverse) = scaled(f, field)?;
    let mut out = clamped;
    for (o, &c) in out.iter_mut().zip(clamped.iter()) {
        let units = round(f, c, inverse)?;
        *o = f.binary_no_contraction(BinOp::FMul, units, step)?;
    }
    Ok(out)
}

/// The resolved output float of `channel` where every sample holds `code`: what [`resolve`] gives
/// for `n` copies of it, at every count the rule holds at, without the average (the module docs):
///
/// - unorm: `(n c + n / 2) / n = c`, as `floor(n / 2) < n`;
/// - snorm: `floor((2 n c + n) / 2n) = c + floor(1 / 2) = c`;
/// - sRGB: the linear value `L` of `c` averages to `L`, and Metal's encode of `L` is `c` for every
///   code (`SRGB_RESOLVE_THRESHOLDS` against `SRGB_RESOLVE_LINEAR`, which a test checks);
/// - unorm through a `half`: `floor((Y + 1024) / 2048) = c`, as `Y = 2048 c - s(c)` and
///   `0 <= s(c) < 1024`;
/// - 16-bit float, and so the two small floats' and the shared exponent's channels: a `half` `x`
///   lies on the grid of its own exponent, so each floored step, `A = 2x` included, is exact, the
///   sum is `n x` and the rounding keeps `x`; a NaN counts 0, so it and -0 resolve to +0, and an
///   infinity's 65536 rounds to the infinity of its sign. A small float's grid step and the shared
///   exponent's encode then run on it as on an average.
///
/// A 32-bit float has none ([`ResolveRule::keeps_alike`]).
fn resolve_alike(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    srgb: Option<SrgbTables>,
    channel: ResolveChannel,
    code: Value,
) -> BResult<Value> {
    let bits = u32::from(channel.rule.bits());
    let scaled = |f: &mut FunctionBuilder<'_>, x: Value, top: u64| -> BResult<Value> {
        let scale = float(f, t, (1.0 / top as f64) as f32)?;
        f.binary(BinOp::FMul, x, scale)
    };
    let half = |f: &mut FunctionBuilder<'_>| -> BResult<Value> {
        let nan = is_nan(f, t, code)?;
        let magnitude = uint(f, t, 0x7fff_ffff)?;
        let m = f.binary(BinOp::BitwiseAnd, code, magnitude)?;
        let zero = uint(f, t, 0)?;
        let zero_value = f.binary(BinOp::IEqual, m, zero)?;
        let plus_zero = f.binary(BinOp::LogicalOr, nan, zero_value)?;
        let out = f.select(plus_zero, zero, code)?;
        f.convert(ConvOp::Bitcast, out, t.f32_)
    };
    Ok(match channel.rule {
        ResolveRule::Unorm { .. } | ResolveRule::HalfUnorm { .. } => {
            let x = f.convert(ConvOp::ConvertUToF, code, t.f32_)?;
            scaled(f, x, (1u64 << bits) - 1)?
        }
        ResolveRule::Snorm { .. } => {
            let x = f.convert(ConvOp::ConvertSToF, code, t.f32_)?;
            scaled(f, x, (1u64 << (bits - 1)) - 1)?
        }
        ResolveRule::Srgb => {
            let tables = srgb.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
            srgb_written(f, t, tables, code)?
        }
        ResolveRule::Float16 | ResolveRule::SharedExponent => half(f)?,
        ResolveRule::SmallFloat { mantissa } => {
            let x = half(f)?;
            small_float(f, t, x, u32::from(mantissa))?
        }
        ResolveRule::Float32 => return Err(crate::spirv::error::BuildError::IdSpaceExhausted),
    })
}

/// Whether every texel of `texels` (read from `source`) has the first's bits, as a `bool`.
fn alike(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    texels: &[Value],
    source: ResolveSource,
) -> BResult<Value> {
    let uvec4 = f.module().ty(Type::Vector {
        elem: t.u32_,
        len: 4,
    })?;
    let words = |f: &mut FunctionBuilder<'_>, texel: Value| -> BResult<Value> {
        match source.reads_bits() {
            true => Ok(texel),
            false => f.convert(ConvOp::Bitcast, texel, uvec4),
        }
    };
    let (&first, rest) = texels
        .split_first()
        .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let first = words(f, first)?;
    let mut differ: Option<Value> = None;
    for &texel in rest {
        let other = words(f, texel)?;
        let bits = f.binary(BinOp::BitwiseXor, first, other)?;
        differ = Some(match differ {
            None => bits,
            Some(d) => f.binary(BinOp::BitwiseOr, d, bits)?,
        });
    }
    let differ = differ.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let mut any = f.composite_extract(differ, &[0])?;
    for lane in 1..4 {
        let word = f.composite_extract(differ, &[lane])?;
        any = f.binary(BinOp::BitwiseOr, any, word)?;
    }
    let zero = uint(f, t, 0)?;
    f.binary(BinOp::IEqual, any, zero)
}

/// The resolved output floats (R, G, B, A; 0 where the format has no channel) of `format` from
/// each sample's `texels` read from `source` (the module docs).
///
/// A pixel whose samples all hold the same bits (every pixel inside a primitive, where one
/// invocation writes every covered sample) takes [`resolve_alike`], which gives what the rule
/// gives without its average; another takes the rule ([`resolve`]). The two meet before a shared
/// exponent's encode. A format of 32-bit floats, or of unorm and snorm channels alone
/// ([`ResolveRule::averages_cheaply`]), takes the rule without comparing.
fn resolve_texel(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    srgb: Option<SrgbTables>,
    format: ColorResolveFormat,
    texels: &[Value],
    source: ResolveSource,
) -> BResult<[Value; 4]> {
    let zero = float(f, t, 0.0)?;
    let channels: Vec<(u32, ResolveChannel)> = (0u32..)
        .zip(format.channels)
        .filter_map(|(component, channel)| Some((component, channel?)))
        .collect();
    let (&first, rest) = texels
        .split_first()
        .ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let mut firsts = Vec::with_capacity(channels.len());
    for &(component, channel) in &channels {
        firsts.push(code(f, t, srgb, channel, component, first, source)?);
    }
    let averaged = |f: &mut FunctionBuilder<'_>| -> BResult<Vec<Value>> {
        let mut out = Vec::with_capacity(channels.len());
        for (&(component, channel), &first) in channels.iter().zip(&firsts) {
            let mut codes = Vec::with_capacity(texels.len());
            codes.push(first);
            for &texel in rest {
                codes.push(code(f, t, srgb, channel, component, texel, source)?);
            }
            out.push(resolve(f, t, srgb, channel, &codes)?);
        }
        Ok(out)
    };
    let shortcut = channels.iter().all(|(_, c)| c.rule.keeps_alike())
        && channels.iter().any(|(_, c)| !c.rule.averages_cheaply());
    let resolved = if shortcut {
        let same = alike(f, t, texels, source)?;
        let kept: ArmFn<'_, ()> = Box::new(|f, _| {
            let mut out = Vec::with_capacity(channels.len());
            for (&(_, channel), &first) in channels.iter().zip(&firsts) {
                out.push(resolve_alike(f, t, srgb, channel, first)?);
            }
            Ok(ArmEnd::FallThrough(out))
        });
        let average: ArmFn<'_, ()> = Box::new(|f, _| Ok(ArmEnd::FallThrough(averaged(f)?)));
        f.if_(&mut (), same, kept, Some(average))?.0
    } else {
        averaged(f)?
    };
    let mut lanes = [zero; 4];
    for (&(component, _), value) in channels.iter().zip(resolved) {
        lanes[component as usize] = value;
    }
    if format.shared_exponent() {
        let [r, g, b] = encode_shared_exponent(f, t, [lanes[0], lanes[1], lanes[2]])?;
        lanes[..3].copy_from_slice(&[r, g, b]);
    }
    Ok(lanes)
}

/// The fragment stage of `format` at `samples` reading `source`, as a typed module, before
/// serialization: the module [`color_resolve_modules`] serializes. Its binding is the symbol
/// `SOURCE_INPUT` or `SOURCE_PLANES`.
pub fn fragment_typed_module(
    format: ColorResolveFormat,
    samples: u32,
    source: ResolveSource,
) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let t = Types::new(&mut b)?;
    let srgb = if format.srgb() {
        Some(SrgbTables::declare(&mut b, t, format, source)?)
    } else {
        None
    };
    let (image, coord) = match source {
        ResolveSource::InputAttachment | ResolveSource::RawInputAttachment => {
            let ty = b.ty(Type::Image(ImageType {
                sampled_type: t.f32_,
                dim: Dim::SubpassData,
                depth: false,
                arrayed: false,
                multisampled: true,
                usage: ImageUsage::Storage,
                format: crate::spv::ImageFormat::Unknown,
            }))?;
            (
                b.resource(SOURCE_INPUT, StorageClass::UniformConstant, ty)?,
                None,
            )
        }
        ResolveSource::Planes(copy) => {
            let ty = b.ty(Type::Image(ImageType {
                sampled_type: match copy.moves_bits() {
                    true => t.u32_,
                    false => t.f32_,
                },
                dim: Dim::D2,
                depth: false,
                arrayed: true,
                multisampled: false,
                usage: ImageUsage::Sampled,
                format: crate::spv::ImageFormat::Unknown,
            }))?;
            let coord = b.builtin(crate::spv::BuiltIn::FragCoord, IoDir::Input, t.vec4)?;
            (
                b.resource(SOURCE_PLANES, StorageClass::UniformConstant, ty)?,
                Some(coord),
            )
        }
    };
    let output = b.fixed_io(0, IoDir::Output, t.vec4, false)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let image = f.module().global_value(image)?;
    let image = f.load(image, MemAccess::default())?;
    let ivec2 = f.module().ty(Type::Vector {
        elem: t.i32_,
        len: 2,
    })?;
    let uvec3 = f.module().ty(Type::Vector {
        elem: t.u32_,
        len: 3,
    })?;
    let pixel = match coord {
        Some(coord) => {
            // A pixel's center is at `(x + 0.5, y + 0.5)`: the conversion truncates to its integer.
            let coord = f.module().global_value(coord)?;
            let coord = f.load(coord, MemAccess::default())?;
            let x = f.composite_extract(coord, &[0])?;
            let y = f.composite_extract(coord, &[1])?;
            let x = f.convert(ConvOp::ConvertFToU, x, t.u32_)?;
            let y = f.convert(ConvOp::ConvertFToU, y, t.u32_)?;
            Some((x, y))
        }
        None => None,
    };
    let mut texels = Vec::with_capacity(samples as usize);
    for s in 0..samples {
        let texel = match pixel {
            None => {
                let origin = f.module().constant(Constant::Null { ty: ivec2 })?;
                let origin = f.module().const_value(origin)?;
                let sample = int(&mut f, t, s as i32)?;
                let ops = ImageOperands {
                    sample: Some(sample),
                    ..ImageOperands::default()
                };
                f.read(image, origin, &ops, TexelMemory::PRIVATE)?
            }
            Some((x, y)) => {
                let layer = uint(&mut f, t, s)?;
                let at = f.composite_construct(uvec3, &[x, y, layer])?;
                let level = uint(&mut f, t, 0)?;
                let ops = ImageOperands {
                    lod: Some(level),
                    ..ImageOperands::default()
                };
                f.fetch(image, at, &ops)?
            }
        };
        texels.push(texel);
    }
    let lanes = resolve_texel(&mut f, t, srgb, format, &texels, source)?;
    let value = f.composite_construct(t.vec4, &lanes)?;
    let output = f.module().global_value(output)?;
    f.store(output, value, MemAccess::default())?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(ExecModel::Fragment, id, vec![ExecMode::OriginUpperLeft])?;
    b.finalize()
}

/// The fixed binding numbers of the modules' symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ColorResolveModules::SET),
        Symbol::Binding(s) if *s == SOURCE_INPUT || *s == SOURCE_PLANES => {
            Some(ColorResolveModules::SOURCE_BINDING)
        }
        _ => None,
    }
}

/// `m` verified against `device`'s profile (requirements included), serialized, and numbered
/// with the fixed bindings; `what` names it in an internal error. Refused, `Unsupported` naming
/// `feature` and the first capability, where the device lacks what `m` needs.
fn finish(
    device: &Device,
    m: TypedModule,
    feature: &Feature,
    what: &str,
) -> Result<(Arc<[u32]>, Requirements), Error> {
    let profile = device.profile();
    profile
        .check(m.requirements())
        .map_err(|c| Error::Unsupported(Unsupported::needs(feature.clone(), c)))?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the {what} needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok((words.into(), r.requirements))
}

/// The resolve modules of `format` at `samples` (a power of two from 2 to 64 at which
/// [`ColorResolveFormat::resolves`]) reading `source`,
/// on `device`: built with the typed builder (the vertex stage is the sample planes merge's,
/// `bind::sample_planes`), checked by the owned verifier against the device's profile
/// (requirements included), serialized, and numbered with the fixed binding of
/// [`ColorResolveModules`]. Refused, `Unsupported` naming `feature` (what the consumer resolves
/// for) and what the device lacks, where it lacks what the modules need; an internal error for
/// another sample count.
pub fn color_resolve_modules(
    device: &Device,
    format: ColorResolveFormat,
    samples: u32,
    source: ResolveSource,
    feature: Feature,
) -> Result<ColorResolveModules, Error> {
    if !samples.is_power_of_two() || !(2..=64).contains(&samples) || !format.resolves(samples) {
        return Err(Error::Internal(InternalError::new(
            InternalKind::Invariant,
            format!(
                "a color resolve of {:?} at {samples} samples",
                format.format()
            ),
        )));
    }
    let (vertex, vertex_needs) = finish(
        device,
        super::sample_planes::merge_vertex_typed_module()?,
        &feature,
        "color resolve's vertex stage",
    )?;
    let (fragment, fragment_needs) = finish(
        device,
        fragment_typed_module(format, samples, source)?,
        &feature,
        "color resolve",
    )?;
    let requirements = device
        .profile()
        .union_requirements(&vertex_needs, &fragment_needs);
    Ok(ColorResolveModules {
        vertex,
        fragment,
        requirements,
    })
}

#[cfg(test)]
mod tests;
