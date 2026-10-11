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
//! - **16-bit float, at 2 samples only** (EV `RES-hr1-m3`, an M3 Max under macOS 26.7, every one of
//!   77824 channels): the exact average of the two samples, a NaN counting 0 and an infinity 65536
//!   of its sign, rounded to a `half` with a tie toward +∞ (65520 and above is +∞, below -65520 is
//!   -∞), a zero +0. Metal's rule at 4 samples is not closed: [`ColorResolveFormat::resolves`] is
//!   false there.
//!
//! Metal has no texture of more than 4 samples on Apple silicon: at 8 and more the modules apply
//! the same rules, which no Metal answer measures; a half's rule is not applied there. The formats
//! of packed 16-bit, `RG11B10Float` and `RGB9E5Float` channels have no closed rule yet
//! ([`ColorResolveFormat::of`] is `None` for them).
//!
//! **The modules** ([`color_resolve_modules`]): a `Vertex` entry `main` that draws one triangle
//! over the whole viewport from `VertexIndex` 0 to 2 (the sample planes merge's), and a `Fragment`
//! entry `main` that reads each of a pixel's samples from its [`ResolveSource`] at
//! [`ColorResolveModules::SOURCE_BINDING`] of set [`ColorResolveModules::SET`], resolves each
//! channel by its rule, and writes the result to the color output at `Location` 0 as the float
//! that the resolve texture's format stores as the resolved code: `code / (2^b - 1)` for unorm,
//! `code / (2^(b-1) - 1)` for snorm, the code's linear value for sRGB (the attachment encodes it),
//! and the float's own bits.
//!
//! - [`ResolveSource::InputAttachment`]: the multisampled attachment as the input attachment of
//!   index 0 (`InputAttachmentIndex 0`), in its own format, read at each sample. A unorm or snorm
//!   channel's code is its float times the code's scale, rounded; an sRGB channel's the code whose
//!   linear value is nearest the float; a float's its bits (a half's as the `float` it reads as).
//! - [`ResolveSource::Planes`]: the attachment's sample planes (`bind::PlanesCopy`), a sampled
//!   2D-array view of every layer in the copy format, layer `s` holding sample `s`: each channel's
//!   code is its bits in the texel `(FragCoord.xy, s)` (a half's, unpacked to its `float`).
//!
//! Draw it into a render pass whose one color attachment is the resolve texture's single-sampled
//! level and layer in its own format, over the render area, one sample, without blending; for
//! [`ResolveSource::InputAttachment`] the pass's subpass also reads the multisampled attachment as
//! its input attachment 0.
//!
//! **Synchronization** (consumer): the samples' writes before the draw's reads, and the resolve
//! texture's earlier accesses before its color writes.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, GlslOp, ImageOperands, ImageType,
    ImageUsage, IoDir, MemAccess, StorageClass, TexelMemory, Type, TypedModule,
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
    /// A snorm code of `bits` bits.
    Snorm { bits: u8 },
    /// An 8-bit sRGB code.
    Srgb,
    /// A 32-bit float.
    Float32,
    /// A 16-bit float, whose rule Metal's measured resolve closes at 2 samples only
    /// ([`ColorResolveFormat::resolves`]).
    Float16,
}

impl ResolveRule {
    /// The channel's bits in a texel.
    pub fn bits(self) -> u8 {
        match self {
            ResolveRule::Unorm { bits } | ResolveRule::Snorm { bits } => bits,
            ResolveRule::Srgb => 8,
            ResolveRule::Float32 => 32,
            ResolveRule::Float16 => 16,
        }
    }

    /// Whether Metal's rule of this channel holds at `samples`: every count but a 16-bit float's,
    /// which is closed at 2 samples only.
    pub fn resolves(self, samples: u32) -> bool {
        self != ResolveRule::Float16 || samples == 2
    }
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
}

/// A channel of `rule` at bit `offset`.
const fn at(rule: ResolveRule, offset: u8) -> Option<ResolveChannel> {
    Some(ResolveChannel { rule, offset })
}

impl ColorResolveFormat {
    /// The resolve of `format`: Metal's 8-, 10- and 16-bit unorm formats with their 2-bit alpha,
    /// its 8- and 16-bit snorm ones, its 8-bit sRGB ones, its 32-bit float ones, and its 16-bit
    /// float ones, whose rule holds at 2 samples only ([`ColorResolveFormat::resolves`]), as Vulkan
    /// formats. `None` for another format (the module docs).
    pub fn of(format: Format) -> Option<ColorResolveFormat> {
        use ResolveRule::{Float16 as F16, Float32 as F32, Srgb as S};
        const U8: ResolveRule = ResolveRule::Unorm { bits: 8 };
        const N8: ResolveRule = ResolveRule::Snorm { bits: 8 };
        const U16: ResolveRule = ResolveRule::Unorm { bits: 16 };
        const N16: ResolveRule = ResolveRule::Snorm { bits: 16 };
        const U10: ResolveRule = ResolveRule::Unorm { bits: 10 };
        const U2: ResolveRule = ResolveRule::Unorm { bits: 2 };
        let channels = match format {
            Format::R8_UNORM => [at(U8, 0), None, None, None],
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
            _ => return None,
        };
        Some(ColorResolveFormat { format, channels })
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
    /// channel): a 16-bit float format's at 2 samples only, any other's at every count.
    pub fn resolves(self, samples: u32) -> bool {
        self.channels
            .iter()
            .flatten()
            .all(|c| c.rule.resolves(samples))
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
    /// The attachment's sample planes, viewed in their copy format.
    Planes(PlanesCopy),
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
    /// Each code's linear value as an `f32`: what the attachment encodes as the code.
    decoded: GlobalId,
    /// The `f32` midway between each code's linear value and the next code's: an input
    /// attachment's float is the code of the midpoints at or below it.
    midpoints: GlobalId,
}

impl SrgbTables {
    fn declare(b: &mut ModuleBuilder, t: Types) -> BResult<SrgbTables> {
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
        let decoded = table(
            b,
            t.f32_,
            t.u32_,
            (0..256).map(|k| u64::from((srgb_decode(k) as f32).to_bits())),
        )?;
        let midpoints = table(
            b,
            t.f32_,
            t.u32_,
            (0..255).map(|k| {
                let mid = (srgb_decode(k) + srgb_decode(k + 1)) / 2.0;
                u64::from((mid as f32).to_bits())
            }),
        )?;
        Ok(SrgbTables {
            linear,
            thresholds,
            decoded,
            midpoints,
        })
    }
}

/// The element of the `Private` array `table` at `index`.
fn element(f: &mut FunctionBuilder<'_>, table: GlobalId, index: Value) -> BResult<Value> {
    let base = f.module().global_value(table)?;
    let at = f.access_chain(base, &[index])?;
    f.load(at, MemAccess::default())
}

/// The count of the 255 ascending entries of `table` that `x` reaches (`at_least(x, entry)`): a
/// binary search of eight steps.
fn rank(
    f: &mut FunctionBuilder<'_>,
    t: Types,
    table: GlobalId,
    x: Value,
    at_least: BinOp,
) -> BResult<Value> {
    let mut lo = uint(f, t, 0)?;
    for step in [128u32, 64, 32, 16, 8, 4, 2, 1] {
        let less = uint(f, t, step - 1)?;
        let index = f.binary(BinOp::IAdd, lo, less)?;
        let entry = element(f, table, index)?;
        let reached = f.binary(at_least, x, entry)?;
        let size = uint(f, t, step)?;
        let up = f.binary(BinOp::IAdd, lo, size)?;
        lo = f.select(reached, up, lo)?;
    }
    Ok(lo)
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
    if let ResolveSource::Planes(_) = source {
        let offset = u32::from(channel.offset);
        let word = f.composite_extract(texel, &[offset / 32])?;
        let shift = uint(f, t, offset % 32)?;
        let count = uint(f, t, bits)?;
        return match channel.rule {
            ResolveRule::Float32 => Ok(word),
            ResolveRule::Unorm { .. } | ResolveRule::Srgb => {
                f.bit_field_extract(false, word, shift, count)
            }
            // The half's bits, unpacked to the `float` it is (exactly), as that float's bits.
            ResolveRule::Float16 => {
                let half = f.bit_field_extract(false, word, shift, count)?;
                let pair = f.glsl(GlslOp::UnpackHalf2x16, &[half])?;
                let x = f.composite_extract(pair, &[0])?;
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
        ResolveRule::Float32 | ResolveRule::Float16 => f.convert(ConvOp::Bitcast, lane, t.u32_),
        ResolveRule::Unorm { .. } => {
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
        ResolveRule::Srgb => {
            let tables = srgb.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
            rank(f, t, tables.midpoints, lane, BinOp::FOrdGreaterThanEqual)
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
            let code = rank(f, t, tables.thresholds, mean, BinOp::UGreaterThanEqual)?;
            element(f, tables.decoded, code)
        }
        ResolveRule::Float32 => {
            let bits = resolve_float(f, t, codes)?;
            f.convert(ConvOp::Bitcast, bits, t.f32_)
        }
        ResolveRule::Float16 => resolve_half(f, t, codes),
    }
}

/// Metal's resolve of two 16-bit float samples, each the bits of the `float` it reads as (the
/// module docs), as the `float` of the resolved `half`, which the attachment stores exactly.
/// A NaN counts 0 and an infinity 65536 of its sign. The two `half`s' sum in `f32` can round only
/// where their exponents lie more than 13 apart, where it lies far from every tie of the result's
/// `half` grid, so the rounding below sees the exact sum's side of each tie; it and the halving
/// are otherwise exact, and every nonzero sum is a normal `float`, which no device flushes. The
/// average `x` rounds to the `half` grid with a tie toward +∞ as `floor(x · (1/u) + 1/2) · u`,
/// `u` the grid's step at `x` (2^-24 below 2^-14, else 2^(e - 10) for `x` in `[2^e, 2^(e+1))`):
/// each step exact in `f32`. A result of 2^16 or more is an infinity; a zero is +0.
fn resolve_half(f: &mut FunctionBuilder<'_>, t: Types, codes: &[Value]) -> BResult<Value> {
    let sign = uint(f, t, 0x8000_0000)?;
    let zero_bits = uint(f, t, 0)?;
    let big = uint(f, t, 0x4780_0000)?; // 65536.0
    let mut sum: Option<Value> = None;
    for &x in codes {
        let nan = is_nan(f, t, x)?;
        let inf = is_inf(f, t, x)?;
        let s = f.binary(BinOp::BitwiseAnd, x, sign)?;
        let signed_big = f.binary(BinOp::BitwiseOr, s, big)?;
        let x = f.select(inf, signed_big, x)?;
        let x = f.select(nan, zero_bits, x)?;
        let x = f.convert(ConvOp::Bitcast, x, t.f32_)?;
        sum = Some(match sum {
            None => x,
            Some(acc) => f.binary_no_contraction(BinOp::FAdd, acc, x)?,
        });
    }
    let sum = sum.ok_or(crate::spirv::error::BuildError::IdSpaceExhausted)?;
    let scale = float(f, t, 1.0 / codes.len() as f32)?;
    let x = f.binary_no_contraction(BinOp::FMul, sum, scale)?;
    // The grid's step: 2^(max(e, -14) - 10), from the exponent field of |x|'s bits.
    let bits = f.convert(ConvOp::Bitcast, x, t.u32_)?;
    let magnitude = uint(f, t, 0x7fff_ffff)?;
    let m = f.binary(BinOp::BitwiseAnd, bits, magnitude)?;
    let shift = uint(f, t, 23)?;
    let field = f.binary(BinOp::ShiftRightLogical, m, shift)?;
    let lowest = uint(f, t, 127 - 14)?;
    let field = f.glsl(GlslOp::UMax, &[field, lowest])?;
    let ten = uint(f, t, 10)?;
    let step_field = f.binary(BinOp::ISub, field, ten)?;
    let step = f.binary(BinOp::ShiftLeftLogical, step_field, shift)?;
    let step = f.convert(ConvOp::Bitcast, step, t.f32_)?;
    // `1 / u`, a power of two too, so each product is exact.
    let twice_bias = uint(f, t, 254)?;
    let inverse_field = f.binary(BinOp::ISub, twice_bias, step_field)?;
    let inverse = f.binary(BinOp::ShiftLeftLogical, inverse_field, shift)?;
    let inverse = f.convert(ConvOp::Bitcast, inverse, t.f32_)?;
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
        Some(SrgbTables::declare(&mut b, t)?)
    } else {
        None
    };
    let (image, coord) = match source {
        ResolveSource::InputAttachment => {
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
        ResolveSource::Planes(_) => {
            let ty = b.ty(Type::Image(ImageType {
                sampled_type: t.u32_,
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
    let mut lanes = Vec::with_capacity(4);
    for (component, channel) in format.channels.iter().enumerate() {
        let Some(channel) = *channel else {
            lanes.push(float(&mut f, t, 0.0)?);
            continue;
        };
        let mut codes = Vec::with_capacity(texels.len());
        for &texel in &texels {
            codes.push(code(
                &mut f,
                t,
                srgb,
                channel,
                component as u32,
                texel,
                source,
            )?);
        }
        lanes.push(resolve(&mut f, t, srgb, channel, &codes)?);
    }
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

/// The resolve modules of `format` at `samples` (a power of two from 2 to 64) reading `source`,
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
