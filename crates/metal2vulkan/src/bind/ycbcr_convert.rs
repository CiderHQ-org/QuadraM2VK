//! A YCbCr texture's texels as Metal reads them, converted from its IOSurface's two planes.
//!
//! **Metal's answers** (an Apple M4 under macOS 27.0.1; our own probes over CoreVideo surfaces). A
//! texture of a 4:2:0 two-plane format of `w x h` texels made over an IOSurface's plane 0 reads the
//! surface's luma plane (plane 0, a code `Y` a texel) and its chroma plane (plane 1, a Cb and then
//! a Cr code for each 2 x 2 texels, `⌈w / 2⌉ x ⌈h / 2⌉` of them at the plane's rows, whatever the
//! surface's own chroma layout). A code of `b` bits ([`YCbCrCodes`]) is a byte (8 bits) or the high
//! `b` bits of a little-endian 16-bit word (10 bits), or, packed, one of three 10-bit codes of a
//! little-endian 32-bit word (below). Texel `(x, y)` reads, for `YCBCR8_420_2P`
//! over `420v` and `420f` surfaces (every 8-bit code compared), and for `YCBCR10_420_2P` and
//! `RGB10_420_2P` over `x420`, `xf20`, `x422`, `xf22`, `x444` and `xf44` surfaces (random
//! codes), with no YCbCr matrix, BT.601, BT.709 and BT.2020 (500, 505, 520, 525 and 543, every
//! texel of 28 surfaces):
//!
//! - **Chroma**: the four nearest chroma texels, the nearer column and row weighted 3 and the
//!   farther 1 (`9 / 3 / 3 / 1`): column `⌊x / 2⌋` and the one after it for an odd `x`, the one
//!   before it for an even `x`, each clamped to the plane, and rows the same way. Each sum `S`
//!   rounds to a code of `b` bits, `C = (S + 8) >> 4`.
//! - **4:2:2 and 4:4:4 chroma** ([`YCbCrCodes::subsampling`]): a 4:2:2 format's
//!   chroma plane holds a pair for each 2 x 1 texels, `⌈w / 2⌉ x h` of them, a 4:4:4 one's a pair
//!   for each texel, `w x h`. Along an axis the plane does not halve, texel `(x, y)` reads the pair
//!   at its own column (or row) alone, weighted 4: a 4:2:2 texel the nearer of its row's two pairs
//!   weighted 3 and the farther 1, `C = (S + 2) >> 2`, a 4:4:4 texel its own pair's codes. Measured
//!   for `YCBCR8_422_2P`, `YCBCR8_444_2P`, `YCBCR10_422_2P`, `YCBCR10_444_2P` and their `_sRGB`
//!   twins over their own surfaces (`422v`, `422f`, `444v`, `444f`, `x422`, `xf22`, `x444`,
//!   `xf44`), every texel of random planes under no matrix, BT.601, BT.709 and BT.2020.
//! - **Packed codes** ([`YCbCrCodes::packed`]): the `_PACKED` formats' rows hold three 10-bit codes
//!   to a little-endian 32-bit word, at its bits 0, 10 and 20, in order: luma code `x` of a row in
//!   word `⌊x / 3⌋`, the chroma plane's pair `x` as codes `2 x` (Cb) and `2 x + 1` (Cr). Their
//!   texels read as the unpacked 10-bit formats' of the same codes, filter, range and matrix alike.
//!   Measured for `YCBCR10_420_2P_PACKED`, `YCBCR10_422_2P_PACKED`, `YCBCR10_444_2P_PACKED` and
//!   their `_sRGB` twins over `p420`, `pf20`, `p422`, `pf22`, `p444` and `pf44`, every texel of
//!   random planes under no matrix, BT.601, BT.709 and BT.2020.
//! - **Codes** ([`YCbCrOutput::Codes`], `RGB10_420_2P`, `RGB8_420_2P` and their 4:2:2 and 4:4:4
//!   formats): red `Cr`, green `Y` and blue `Cb`, each `code / (2^b - 1)`, alpha 1: an
//!   `RGB10A2Unorm` or an `RGBA8Unorm` texel of the codes. `RGB8_420_2P` over `420v` and `420f`
//!   reads them so (under BT.709 and BT.2020 alike), and `RGB8_422_2P`, `RGB8_444_2P`,
//!   `RGB10_422_2P` and `RGB10_444_2P` over their own surfaces, after their chroma filter (every
//!   texel of 24 surfaces; and our probes under every matrix).
//! - **Converted** ([`YCbCrOutput::Converted`]): the codes counted as 10-bit codes
//!   (`u = 10 - b`), `Cb' = (Cb << u) - 512`, `Cr' = (Cr << u) - 512`; luma video range (`420v`,
//!   `x4..`) `L = 9576 ((Y << u) - 64)`, full range (`420f`, `xf..`)
//!   `L = 256 ⌊(32768 Y + ⌊m / 2⌋) / m⌋` with `m = 2^b - 1`. Red, green and blue are each
//!   `v / 2048`, where `v = clamp((L + kb Cb' + kr Cr' + 2048) >> 12, 0, 2048)` (an arithmetic
//!   shift) with the channel's `(kb, kr)` of the range and matrix ([`ycbcr_read`]); alpha is 1.
//!   Every value lies on the grid of `1 / 2048`, which a half float holds exactly.
//! - **sRGB** ([`YCbCrOutput::Srgb`], the `_sRGB` twins `YCBCR8_420_2P_sRGB` over `420v` and `420f`
//!   and `YCBCR10_420_2P_sRGB` over `x420`, `xf20`, `x422` and `xf44`, as textures and as views of
//!   `YCBCR8_420_2P` and `YCBCR10_420_2P`; every channel, and every one of the 8193 steps below):
//!   the converted sum through the sRGB curve on the grid of `1 / 8192`. With `T = L + kb Cb' + kr
//!   Cr'` as above, the step `q = clamp((T + 512) >> 10, 0, 8192)` is decoded by the curve's knots
//!   at every 16th step, `K[j]` the step nearest `8192 · srgb(16 j / 8192)` (`srgb(c)` is `c /
//!   12.92` up to `0.04045`, `((c + 0.055) / 1.055)^2.4` above), joined by straight lines: with `i
//!   = min(q >> 4, 511)` and `f = q - 16 i`, the step `(16 K[i] + (K[i + 1] - K[i]) f) / 16`
//!   rounded to nearest even, each channel that step `/ 8192`; alpha is 1. A half float does not
//!   hold every step (above `1/4`), so the image is `R32G32B32A32_SFLOAT`.
//!
//! **The module** (one per device and output). One `GLCompute` entry `main` with local size
//! ([`YCbCrConvertModule::LOCAL_SIZE`], [`YCbCrConvertModule::LOCAL_SIZE`], 1), no push constants,
//! and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`YCbCrConvertModule::IMAGE_BINDING`] | the texture's image, a 2D storage image, written: `Rgba16f` of the converted texels, `Rgba32f` of the sRGB ones, or `R32ui` of the codes packed as `A2B10G10R10_UNORM_PACK32` (10 bits) or `R8G8B8A8_UNORM` (8 bits) packs them (a view of that image) |
//! | [`YCbCrConvertModule::SHAPE_BINDING`] | the planes' shape ([`YCbCrConvertModule::shape`]), a storage buffer |
//! | [`YCbCrConvertModule::BYTES_BINDING`] | the surface's bytes, 4 to a word, a storage buffer |
//!
//! Invocation `(x, y)` inside the texture writes texel `(x, y)` as above, from the bytes
//! binding's luma and chroma codes at the shape's offsets and rows. One outside it writes
//! nothing. The consumer dispatches `⌈w / 8⌉ x ⌈h / 8⌉` workgroups and keeps the bytes binding
//! within `maxStorageBufferRange`.
//!
//! **Synchronization** (consumer): the writes of the bytes and the shape (`TRANSFER_WRITE`) →
//! `SHADER_STORAGE_READ` in `COMPUTE_SHADER` before the dispatch; its image writes
//! (`SHADER_STORAGE_WRITE`) → whatever reads the texture after.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::metal::format::ChromaSubsampling;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, Dim, ExecMode, ExecModel, GlslOp,
    ImageOperands, ImageType, ImageUsage, IoDir, LayoutRule, MemAccess, StorageClass, TexelMemory,
    Type, TypedModule,
};

/// The range of a YCbCr surface's codes, which its pixel format names.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum YCbCrRange {
    /// Video range: luma codes 16 to 235 span black to white (CoreVideo's `420v`).
    Video,
    /// Full range: luma codes 0 to 255 (CoreVideo's `420f`).
    Full,
}

/// The YCbCr matrix of a conversion: the surface's `IOSurfaceYCbCrMatrix` as Metal reads it when
/// it makes the texture (none reads as BT.601).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum YCbCrMatrix {
    /// ITU-R BT.601 (`ITU_R_601_4`, or no matrix).
    Bt601,
    /// ITU-R BT.709 (`ITU_R_709_2`).
    Bt709,
    /// ITU-R BT.2020 (`ITU_R_2020`).
    Bt2020,
}

/// What a YCbCr texture's texels read of its codes (the module docs), which picks the module's
/// image.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum YCbCrOutput {
    /// The color converted by the surface's range and matrix, each channel a multiple of
    /// `1 / 2048`: an `R16G16B16A16_SFLOAT` image, written through an `Rgba16f` view.
    Converted,
    /// The codes themselves, red Cr, green Y and blue Cb: an `A2B10G10R10_UNORM_PACK32` image of
    /// 10-bit codes, an `R8G8B8A8_UNORM` one of 8-bit codes, written packed through an `R32ui`
    /// view of it.
    Codes,
    /// The converted color through the sRGB curve, each channel a multiple of `1 / 8192`: an
    /// `R32G32B32A32_SFLOAT` image, written through an `Rgba32f` view.
    Srgb,
}

/// How a two-plane format's planes hold its codes, and what its texels read of them: the layouts
/// the module converts (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct YCbCrCodes {
    /// The bits of each code: 8 or 10.
    pub bits: u32,
    /// The bytes of the unit a plane's row holds its codes in, one after the other: 1, a byte of
    /// one code; 2, a little-endian 16-bit word of one code in its high
    /// [`bits`](Self::bits) bits; or 4, a little-endian 32-bit word of three 10-bit codes at its
    /// bits 0, 10 and 20 ([`packed`](Self::packed)).
    pub bytes: u32,
    /// What a texel reads.
    pub output: YCbCrOutput,
    /// How many luma texels share a chroma pair, which picks the chroma filter (the module
    /// docs).
    pub subsampling: ChromaSubsampling,
    /// Whether the codes are packed (`_PACKED`), three to a 4-byte unit.
    pub packed: bool,
}

impl YCbCrCodes {
    /// The codes of a two-plane format of `format`'s layout, `None` for a layout the module does
    /// not convert: the 4:2:0, 4:2:2 and 4:4:4 formats of unpacked codes, 8- and 10-bit ones read
    /// converted (`YCBCR8_420_2P`, `YCBCR10_444_2P`, ...), through the sRGB curve (their
    /// `_sRGB` twins) or as codes (`RGB8_420_2P`, `RGB10_422_2P`, ...), and the packed 10-bit ones
    /// read converted or through the sRGB curve (`YCBCR10_420_2P_PACKED`,
    /// `YCBCR10_444_2P_PACKED_sRGB`, ...), as Metal reads them (the module docs).
    pub fn of(format: crate::metal::format::YCbCrFormat) -> Option<YCbCrCodes> {
        use crate::metal::format::YCbCrReads;
        let output = match (format.reads, format.bits, format.packed) {
            (YCbCrReads::Converted, 8 | 10, false) | (YCbCrReads::Converted, 10, true) => {
                YCbCrOutput::Converted
            }
            (YCbCrReads::Srgb, 8 | 10, false) | (YCbCrReads::Srgb, 10, true) => YCbCrOutput::Srgb,
            (YCbCrReads::Codes, 8 | 10, false) => YCbCrOutput::Codes,
            _ => return None,
        };
        Some(YCbCrCodes {
            bits: format.bits,
            bytes: match format.packed {
                true => 4,
                false => format.bits.div_ceil(8),
            },
            output,
            subsampling: format.subsampling,
            packed: format.packed,
        })
    }

    /// The largest code, `2^bits - 1`.
    fn max(&self) -> u32 {
        (1 << self.bits) - 1
    }

    /// The codes a unit holds: 3 packed, 1 otherwise.
    fn per_unit(&self) -> u32 {
        match self.packed {
            true => 3,
            false => 1,
        }
    }

    /// The units that hold `n` codes from a row's start: `⌈n / 3⌉` packed, `n` otherwise. A
    /// plane's row of `n` codes is that many units of [`bytes`](Self::bytes) bytes.
    pub fn units(&self, n: u64) -> u64 {
        n.div_ceil(u64::from(self.per_unit()))
    }

    /// The shift that takes a unit to its first code: `8 * bytes - bits` (the high bits of a
    /// byte or a 16-bit word), and 0 packed.
    fn first_shift(&self) -> u32 {
        match self.packed {
            true => 0,
            false => 8 * self.bytes - self.bits,
        }
    }

    /// The bits from one code of a unit to the next: [`bits`](Self::bits) packed, 0 otherwise.
    fn step(&self) -> u32 {
        match self.packed {
            true => self.bits,
            false => 0,
        }
    }
}

/// Where a texture's two planes lie in the bytes the module reads, and how they convert.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct YCbCrPlanes {
    /// The texture's width and height, in luma texels.
    pub size: [u32; 2],
    /// The luma plane's first byte, and the bytes from one of its rows to the next.
    pub luma: [u32; 2],
    /// The chroma plane's first byte, and the bytes from one of its rows to the next.
    pub chroma: [u32; 2],
    pub range: YCbCrRange,
    pub matrix: YCbCrMatrix,
    /// How the planes hold the codes, and what a texel reads.
    pub codes: YCbCrCodes,
}

impl YCbCrPlanes {
    /// The planes of a texture of `size` texels whose luma plane starts at byte `luma[0]` with
    /// rows `luma[1]` bytes apart, and whose chroma plane at `chroma[0]` with rows `chroma[1]`
    /// apart, holding `codes`.
    pub fn new(
        size: [u32; 2],
        luma: [u32; 2],
        chroma: [u32; 2],
        (range, matrix): (YCbCrRange, YCbCrMatrix),
        codes: YCbCrCodes,
    ) -> YCbCrPlanes {
        YCbCrPlanes {
            size,
            luma,
            chroma,
            range,
            matrix,
            codes,
        }
    }

    /// The chroma plane's texels across and down: `⌈w / 2⌉ x ⌈h / 2⌉` for 4:2:0 codes,
    /// `⌈w / 2⌉ x h` for 4:2:2 ones and `w x h` for 4:4:4 ones.
    pub fn chroma_size(&self) -> [u32; 2] {
        let halved = self.codes.subsampling.halved();
        [0, 1].map(|axis| match halved[axis] {
            true => self.size[axis].div_ceil(2),
            false => self.size[axis],
        })
    }
}

/// The `(kb, kr)` of red, green and blue for `range` and `matrix`, in the units of the module
/// docs' conversion (measured, the module docs).
fn coefficients(range: YCbCrRange, matrix: YCbCrMatrix) -> [(i32, i32); 3] {
    match (range, matrix) {
        (YCbCrRange::Video, YCbCrMatrix::Bt601) => [(-9, 13123), (-3218, -6686), (16591, 9)],
        (YCbCrRange::Video, YCbCrMatrix::Bt709) => [(-1, 14743), (-1753, -4383), (17373, 1)],
        (YCbCrRange::Full, YCbCrMatrix::Bt601) => [(-8, 11483), (-2816, -5850), (14518, 8)],
        (YCbCrRange::Full, YCbCrMatrix::Bt709) => [(-1, 12900), (-1534, -3835), (15201, 1)],
        (YCbCrRange::Video, YCbCrMatrix::Bt2020) => [(0, 13806), (-1540, -5349), (17614, -1)],
        (YCbCrRange::Full, YCbCrMatrix::Bt2020) => [(0, 12080), (-1348, -4680), (15412, -1)],
    }
}

/// The luma term of the conversion for the code `y` of `bits` bits (the module docs).
fn luma_term(range: YCbCrRange, bits: u32, y: u32) -> i32 {
    let max = (1u32 << bits) - 1;
    match range {
        YCbCrRange::Video => 9576 * (((y << (10 - bits)) as i32) - 64),
        YCbCrRange::Full => 256 * ((y * 32768 + max / 2) / max) as i32,
    }
}

/// The sRGB curve's knots on the grid of `1 / 8192` (the module docs): `K[j]`, the step nearest
/// `8192 · srgb(16 j / 8192)`, for `j` from 0 to 512 (`the_srgb_knots_are_the_curves`).
const SRGB_KNOTS: [u16; 513] = [
    0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 29, 30, 31,
    33, 34, 36, 37, 39, 41, 42, 44, 46, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 68, 70, 72, 74, 77,
    79, 82, 84, 87, 89, 92, 95, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133,
    137, 140, 144, 147, 151, 154, 158, 161, 165, 169, 173, 176, 180, 184, 188, 192, 197, 201, 205,
    209, 213, 218, 222, 227, 231, 236, 240, 245, 250, 255, 259, 264, 269, 274, 279, 284, 289, 295,
    300, 305, 311, 316, 322, 327, 333, 338, 344, 350, 356, 361, 367, 373, 379, 385, 392, 398, 404,
    410, 417, 423, 430, 436, 443, 450, 456, 463, 470, 477, 484, 491, 498, 505, 512, 519, 527, 534,
    542, 549, 557, 564, 572, 580, 587, 595, 603, 611, 619, 627, 635, 644, 652, 660, 669, 677, 686,
    694, 703, 712, 720, 729, 738, 747, 756, 765, 774, 784, 793, 802, 812, 821, 831, 840, 850, 860,
    870, 879, 889, 899, 909, 920, 930, 940, 950, 961, 971, 982, 992, 1003, 1014, 1025, 1035, 1046,
    1057, 1068, 1080, 1091, 1102, 1113, 1125, 1136, 1148, 1159, 1171, 1183, 1194, 1206, 1218, 1230,
    1242, 1255, 1267, 1279, 1291, 1304, 1316, 1329, 1342, 1354, 1367, 1380, 1393, 1406, 1419, 1432,
    1445, 1458, 1472, 1485, 1499, 1512, 1526, 1539, 1553, 1567, 1581, 1595, 1609, 1623, 1637, 1652,
    1666, 1680, 1695, 1709, 1724, 1739, 1753, 1768, 1783, 1798, 1813, 1828, 1844, 1859, 1874, 1890,
    1905, 1921, 1936, 1952, 1968, 1984, 2000, 2016, 2032, 2048, 2064, 2081, 2097, 2114, 2130, 2147,
    2163, 2180, 2197, 2214, 2231, 2248, 2265, 2282, 2300, 2317, 2335, 2352, 2370, 2387, 2405, 2423,
    2441, 2459, 2477, 2495, 2513, 2532, 2550, 2569, 2587, 2606, 2625, 2643, 2662, 2681, 2700, 2719,
    2738, 2758, 2777, 2796, 2816, 2835, 2855, 2875, 2895, 2914, 2934, 2954, 2975, 2995, 3015, 3035,
    3056, 3076, 3097, 3118, 3138, 3159, 3180, 3201, 3222, 3243, 3265, 3286, 3307, 3329, 3350, 3372,
    3394, 3416, 3437, 3459, 3481, 3504, 3526, 3548, 3570, 3593, 3615, 3638, 3661, 3684, 3706, 3729,
    3752, 3776, 3799, 3822, 3845, 3869, 3892, 3916, 3940, 3964, 3987, 4011, 4035, 4060, 4084, 4108,
    4132, 4157, 4181, 4206, 4231, 4256, 4280, 4305, 4331, 4356, 4381, 4406, 4432, 4457, 4483, 4508,
    4534, 4560, 4586, 4612, 4638, 4664, 4690, 4717, 4743, 4769, 4796, 4823, 4849, 4876, 4903, 4930,
    4957, 4985, 5012, 5039, 5067, 5094, 5122, 5150, 5177, 5205, 5233, 5261, 5290, 5318, 5346, 5375,
    5403, 5432, 5460, 5489, 5518, 5547, 5576, 5605, 5634, 5664, 5693, 5722, 5752, 5782, 5811, 5841,
    5871, 5901, 5931, 5961, 5992, 6022, 6053, 6083, 6114, 6145, 6175, 6206, 6237, 6268, 6300, 6331,
    6362, 6394, 6425, 6457, 6489, 6520, 6552, 6584, 6616, 6649, 6681, 6713, 6746, 6778, 6811, 6844,
    6876, 6909, 6942, 6975, 7009, 7042, 7075, 7109, 7142, 7176, 7210, 7243, 7277, 7311, 7345, 7380,
    7414, 7448, 7483, 7517, 7552, 7587, 7622, 7657, 7692, 7727, 7762, 7797, 7833, 7868, 7904, 7940,
    7975, 8011, 8047, 8083, 8119, 8156, 8192,
];

/// The step of `1 / 8192` Metal reads for the converted sum's step `q` (at most 8192) of an sRGB
/// format (the module docs): the knots joined by straight lines, rounded to nearest even.
fn srgb_step(q: u32) -> u16 {
    let i = (q >> 4).min(511) as usize;
    let f = q - 16 * i as u32;
    let (a, b) = (u32::from(SRGB_KNOTS[i]), u32::from(SRGB_KNOTS[i + 1]));
    let n = 16 * a + (b - a) * f;
    ((n + 7 + ((n >> 4) & 1)) >> 4) as u16
}

/// Metal's red, green and blue of a texel of luma code `y` whose chroma rounded to the 8-bit
/// codes `cb` and `cr` (the module docs), each `v` of the value `v / 2048`: [`ycbcr_read`] of
/// 8-bit codes read converted.
pub fn ycbcr_texel(range: YCbCrRange, matrix: YCbCrMatrix, y: u8, cb: u8, cr: u8) -> [u16; 3] {
    let codes = YCbCrCodes {
        bits: 8,
        bytes: 1,
        output: YCbCrOutput::Converted,
        subsampling: ChromaSubsampling::Chroma420,
        packed: false,
    };
    ycbcr_read(codes, (range, matrix), [y, cb, cr].map(u16::from))
}

/// Metal's red, green and blue of a texel of `codes` whose luma code is `y` and whose chroma
/// rounded to the codes `cb` and `cr` (the module docs): each `v` of the value `v / 2048` for
/// codes read converted, by `range` and `matrix`, and of `v / 8192` for codes read through the
/// sRGB curve; the codes `cr`, `y` and `cb` of `code / (2^bits - 1)` for codes read as codes.
pub fn ycbcr_read(
    codes: YCbCrCodes,
    (range, matrix): (YCbCrRange, YCbCrMatrix),
    [y, cb, cr]: [u16; 3],
) -> [u16; 3] {
    // The converted sums `T = L + kb Cb' + kr Cr'`.
    let sums = || {
        let l = luma_term(range, codes.bits, u32::from(y));
        let up = 10 - codes.bits;
        let (cb, cr) = (i32::from(cb) << up, i32::from(cr) << up);
        let (cb, cr) = (cb - 512, cr - 512);
        coefficients(range, matrix).map(|(kb, kr)| l + kb * cb + kr * cr)
    };
    match codes.output {
        YCbCrOutput::Codes => [cr, y, cb],
        YCbCrOutput::Converted => sums().map(|t| ((t + 2048) >> 12).clamp(0, 2048) as u16),
        YCbCrOutput::Srgb => sums().map(|t| srgb_step(((t + 512) >> 10).clamp(0, 8192) as u32)),
    }
}

/// The crate-owned YCbCr conversion module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct YCbCrConvertModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl YCbCrConvertModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The texture's image: a 2D storage image, written (`Rgba16f` of the converted texels,
    /// `Rgba32f` of the sRGB ones, `R32ui` of the packed codes of either width; the module
    /// docs).
    pub const IMAGE_BINDING: u32 = 0;
    /// The planes' shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 1;
    /// The surface's bytes, 4 to a word: a storage buffer.
    pub const BYTES_BINDING: u32 = 2;
    /// The words of a shape.
    pub const SHAPE_WORDS: u32 = 22;
    /// Invocations of one workgroup along x, and along y: one texel each.
    pub const LOCAL_SIZE: u32 = 8;

    /// The shape's bytes of `planes`.
    pub fn shape(planes: &YCbCrPlanes) -> [u8; Self::SHAPE_WORDS as usize * 4] {
        let k = coefficients(planes.range, planes.matrix);
        let codes = planes.codes;
        let words = [
            planes.size[0],
            planes.size[1],
            planes.luma[0],
            planes.luma[1],
            planes.chroma[0],
            planes.chroma[1],
            u32::from(planes.range == YCbCrRange::Full),
            k[0].0 as u32,
            k[0].1 as u32,
            k[1].0 as u32,
            k[1].1 as u32,
            k[2].0 as u32,
            k[2].1 as u32,
            codes.bytes,
            codes.first_shift(),
            10 - codes.bits,
            codes.max(),
            codes.bits,
            u32::from(codes.subsampling.halved()[0]),
            u32::from(codes.subsampling.halved()[1]),
            codes.per_unit(),
            codes.step(),
        ];
        let mut bytes = [0; Self::SHAPE_WORDS as usize * 4];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The shape's words, in order (written by [`YCbCrConvertModule::shape`]).
mod word {
    pub(super) const WIDTH: u32 = 0;
    pub(super) const HEIGHT: u32 = 1;
    pub(super) const LUMA_OFFSET: u32 = 2;
    pub(super) const LUMA_ROW: u32 = 3;
    pub(super) const CHROMA_OFFSET: u32 = 4;
    pub(super) const CHROMA_ROW: u32 = 5;
    pub(super) const FULL: u32 = 6;
    /// Red's `kb`; red's `kr`, green's and blue's follow.
    pub(super) const COEFFICIENTS: u32 = 7;
    /// The bytes of a unit of codes: 1, 2 or 4 (`YCbCrCodes::bytes`).
    pub(super) const CODE_BYTES: u32 = 13;
    /// The shift that takes a unit to its first code: `8 * bytes - bits`, 0 packed.
    pub(super) const CODE_SHIFT: u32 = 14;
    /// The shift that counts a code as a 10-bit code: `10 - bits`.
    pub(super) const TEN_BIT_SHIFT: u32 = 15;
    /// The largest code, `2^bits - 1`.
    pub(super) const MAX_CODE: u32 = 16;
    /// The bits of a code: 8 or 10.
    pub(super) const CODE_BITS: u32 = 17;
    /// 1 where the chroma plane holds a pair for each two luma columns, 0 for each column.
    pub(super) const HALVED_ACROSS: u32 = 18;
    /// 1 where the chroma plane holds a pair for each two luma rows, 0 for each row.
    pub(super) const HALVED_DOWN: u32 = 19;
    /// The codes a unit holds: 3 packed, 1 otherwise.
    pub(super) const CODES_PER_UNIT: u32 = 20;
    /// The bits from one code of a unit to the next: 10 packed, 0 otherwise.
    pub(super) const CODE_STEP: u32 = 21;
}

/// The binding symbol of the image in the typed module.
const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the shape in the typed module.
const SHAPE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the surface's bytes in the typed module.
const BYTES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
pub(super) fn words_buffer(
    b: &mut ModuleBuilder,
    sym: ResourceSlot,
    u32_: TypeId,
) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

/// The sRGB knots ([`SRGB_KNOTS`]) as a `Private` `u32[513]` initialized to them.
fn knots_table(b: &mut ModuleBuilder, u32_: TypeId) -> BResult<GlobalId> {
    let len = b.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(SRGB_KNOTS.len() as u64),
    })?;
    let arr = b.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(SRGB_KNOTS.len());
    for k in SRGB_KNOTS {
        parts.push(b.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(u64::from(k)),
        })?);
    }
    let init = b.constant(Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    b.plain_global(StorageClass::Private, arr, Some(init))
}

/// A scalar constant of `ty` with `bits`, as a value.
pub(super) fn scalar(f: &mut FunctionBuilder<'_>, ty: TypeId, bits: u64) -> BResult<Value> {
    let c = f.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(bits),
    })?;
    f.module().const_value(c)
}

/// The values the conversion of one texel reads.
#[derive(Clone, Copy)]
struct Cx {
    u32_: TypeId,
    zero: Value,
    one: Value,
    bytes: Value,
}

/// How the planes hold their codes, as the shape's words state (`word::CODE_BYTES` on), each a
/// `u32`.
#[derive(Clone, Copy)]
struct Held {
    /// The bytes of a unit of codes.
    unit: Value,
    /// The codes a unit holds.
    per: Value,
    /// The shift that takes a unit to its first code.
    shift: Value,
    /// The bits from one code of a unit to the next.
    step: Value,
    /// The largest code, which masks one.
    max: Value,
}

impl Cx {
    fn uint(&self, f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
        scalar(f, self.u32_, u64::from(v))
    }

    /// The word of the bytes binding that holds the byte at `at`.
    fn word(&self, f: &mut FunctionBuilder<'_>, at: Value) -> BResult<Value> {
        let two = self.uint(f, 2)?;
        let index = f.binary(BinOp::ShiftRightLogical, at, two)?;
        let p = f.access_chain(self.bytes, &[self.zero, index])?;
        f.load(p, MemAccess::default())
    }

    /// Code `index` of the row whose first byte is at `row`: in unit `⌊index / per⌋`, its
    /// `index mod per`-th code, `bits` bits from `shift + step · (index mod per)` of the unit's
    /// little-endian bytes (a byte, a 16-bit word's high bits, or a packed word's codes at bits
    /// 0, 10 and 20). The unit's bytes are read through the words holding its first and last
    /// byte, so no read passes the unit's bytes.
    fn code(
        &self,
        f: &mut FunctionBuilder<'_>,
        held: Held,
        row: Value,
        index: Value,
    ) -> BResult<Value> {
        let unit_index = f.binary(BinOp::UDiv, index, held.per)?;
        let first = f.binary(BinOp::IMul, unit_index, held.per)?;
        let k = f.binary(BinOp::ISub, index, first)?;
        let offset = f.binary(BinOp::IMul, unit_index, held.unit)?;
        let at = f.binary(BinOp::IAdd, row, offset)?;
        let last = f.binary(BinOp::ISub, held.unit, self.one)?;
        let last = f.binary(BinOp::IAdd, at, last)?;
        let low = self.word(f, at)?;
        let high = self.word(f, last)?;
        // The four bytes from `at`: the first word from its byte `at mod 4` on, and above them
        // the next word's first bytes, shifted by `32 - lane` in two steps so a lane of 0 takes
        // none of them.
        let three = self.uint(f, 3)?;
        let lane = f.binary(BinOp::BitwiseAnd, at, three)?;
        let lane = f.binary(BinOp::ShiftLeftLogical, lane, three)?;
        let low = f.binary(BinOp::ShiftRightLogical, low, lane)?;
        let thirty_one = self.uint(f, 31)?;
        let up = f.binary(BinOp::ISub, thirty_one, lane)?;
        let high = f.binary(BinOp::ShiftLeftLogical, high, up)?;
        let high = f.binary(BinOp::ShiftLeftLogical, high, self.one)?;
        let bytes = f.binary(BinOp::BitwiseOr, low, high)?;
        let step = f.binary(BinOp::IMul, k, held.step)?;
        let shift = f.binary(BinOp::IAdd, held.shift, step)?;
        let code = f.binary(BinOp::ShiftRightLogical, bytes, shift)?;
        f.binary(BinOp::BitwiseAnd, code, held.max)
    }

    /// The step of `1 / 8192` an sRGB format reads for the converted sum's step `q` (at most
    /// 8192), by the knots of `table` ([`srgb_step`]): `i = min(q >> 4, 511)`, `f = q - 16 i`,
    /// and `n = 16 K[i] + (K[i + 1] - K[i]) f` divided by 16 to nearest even.
    fn srgb_step(&self, f: &mut FunctionBuilder<'_>, table: GlobalId, q: Value) -> BResult<Value> {
        let four = self.uint(f, 4)?;
        let last = self.uint(f, 511)?;
        let i = f.binary(BinOp::ShiftRightLogical, q, four)?;
        let i = f.glsl(GlslOp::UMin, &[i, last])?;
        let start = f.binary(BinOp::ShiftLeftLogical, i, four)?;
        let fraction = f.binary(BinOp::ISub, q, start)?;
        let table = f.module().global_value(table)?;
        let p = f.access_chain(table, &[i])?;
        let a = f.load(p, MemAccess::default())?;
        let next = f.binary(BinOp::IAdd, i, self.one)?;
        let p = f.access_chain(table, &[next])?;
        let b = f.load(p, MemAccess::default())?;
        let rise = f.binary(BinOp::ISub, b, a)?;
        let rise = f.binary(BinOp::IMul, rise, fraction)?;
        let base = f.binary(BinOp::ShiftLeftLogical, a, four)?;
        let n = f.binary(BinOp::IAdd, base, rise)?;
        // `(n + 7 + ((n >> 4) & 1)) >> 4`: a remainder of 8 carries where the quotient is odd.
        let quotient = f.binary(BinOp::ShiftRightLogical, n, four)?;
        let odd = f.binary(BinOp::BitwiseAnd, quotient, self.one)?;
        let seven = self.uint(f, 7)?;
        let n = f.binary(BinOp::IAdd, n, seven)?;
        let n = f.binary(BinOp::IAdd, n, odd)?;
        f.binary(BinOp::ShiftRightLogical, n, four)
    }

    /// The two chroma texels of luma column (or row) `p` of a plane `n` texels across, and the
    /// first one's weight (the second's is 4 less it). Where `halved` is 1 (a chroma texel for
    /// each two luma ones): `⌊p / 2⌋` and the next for an odd `p`, weighted 3 and 1; the one
    /// before and `⌊p / 2⌋` for an even `p`, weighted 1 and 3; each clamped to the plane. Where
    /// it is 0: `p` alone, weighted 4.
    fn taps(
        &self,
        f: &mut FunctionBuilder<'_>,
        p: Value,
        n: Value,
        halved: Value,
    ) -> BResult<[Value; 3]> {
        let c = f.binary(BinOp::ShiftRightLogical, p, halved)?;
        let odd = f.binary(BinOp::BitwiseAnd, p, halved)?;
        let a = f.binary(BinOp::IAdd, c, odd)?;
        let last = f.binary(BinOp::ISub, n, self.one)?;
        // `a` is at most `n`: the farther or nearer texel past the plane is its last.
        let past = f.binary(BinOp::UGreaterThan, a, last)?;
        let second = f.select(past, last, a)?;
        let at_start = f.binary(BinOp::IEqual, a, self.zero)?;
        let before = f.binary(BinOp::ISub, a, self.one)?;
        let first = f.select(at_start, self.zero, before)?;
        let is_odd = f.binary(BinOp::IEqual, odd, self.one)?;
        let three = self.uint(f, 3)?;
        let weight = f.select(is_odd, three, self.one)?;
        // A direct axis reads texel `p` (`c`, `a` and `second` are `p`, inside the plane).
        let direct = f.binary(BinOp::IEqual, halved, self.zero)?;
        let first = f.select(direct, c, first)?;
        let four = self.uint(f, 4)?;
        let weight = f.select(direct, four, weight)?;
        Ok([first, second, weight])
    }
}

/// The module of `output` as a typed module, before serialization: the module
/// [`ycbcr_convert_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols `IMAGE`, `SHAPE` and `BYTES` (the module docs).
fn typed_module(output: YCbCrOutput) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let (sampled_type, format) = match output {
        YCbCrOutput::Converted => (f32_, crate::spv::ImageFormat::Rgba16f),
        YCbCrOutput::Srgb => (f32_, crate::spv::ImageFormat::Rgba32f),
        YCbCrOutput::Codes => (u32_, crate::spv::ImageFormat::R32ui),
    };
    let texel_ty = b.ty(Type::Vector {
        elem: sampled_type,
        len: 4,
    })?;
    let image_ty = b.ty(Type::Image(ImageType {
        sampled_type,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Storage,
        format,
    }))?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, image_ty)?;
    let knots = match output {
        YCbCrOutput::Srgb => Some(knots_table(&mut b, u32_)?),
        YCbCrOutput::Converted | YCbCrOutput::Codes => None,
    };
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let bytes = words_buffer(&mut b, BYTES, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = scalar(&mut f, u32_, 0)?;
    let one = scalar(&mut f, u32_, 1)?;
    let bytes = f.module().global_value(bytes)?;
    let cx = Cx {
        u32_,
        zero,
        one,
        bytes,
    };
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = cx.uint(f, index)?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let width = field(&mut f, word::WIDTH)?;
    let height = field(&mut f, word::HEIGHT)?;
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, height)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let luma_offset = field(&mut f, word::LUMA_OFFSET)?;
    let luma_row = field(&mut f, word::LUMA_ROW)?;
    let chroma_offset = field(&mut f, word::CHROMA_OFFSET)?;
    let chroma_row = field(&mut f, word::CHROMA_ROW)?;
    let full = field(&mut f, word::FULL)?;
    let mut k = Vec::with_capacity(6);
    for index in word::COEFFICIENTS..word::COEFFICIENTS + 6 {
        k.push(field(&mut f, index)?);
    }
    let unit = field(&mut f, word::CODE_BYTES)?;
    let shift = field(&mut f, word::CODE_SHIFT)?;
    let up = field(&mut f, word::TEN_BIT_SHIFT)?;
    let max = field(&mut f, word::MAX_CODE)?;
    let bits = field(&mut f, word::CODE_BITS)?;
    let halved_across = field(&mut f, word::HALVED_ACROSS)?;
    let halved_down = field(&mut f, word::HALVED_DOWN)?;
    let per = field(&mut f, word::CODES_PER_UNIT)?;
    let step = field(&mut f, word::CODE_STEP)?;
    let held = Held {
        unit,
        per,
        shift,
        step,
        max,
    };
    let image = f.module().global_value(image)?;
    let converted: ArmFn<'_, ()> = Box::new(move |f, _| {
        // The luma code.
        let row = f.binary(BinOp::IMul, y, luma_row)?;
        let row = f.binary(BinOp::IAdd, luma_offset, row)?;
        let luma = cx.code(f, held, row, x)?;
        // The chroma codes: the four nearest chroma texels, weighted 9 / 3 / 3 / 1, along a
        // direct axis the nearest alone, weighted 4.
        let wide = f.binary(BinOp::IAdd, width, halved_across)?;
        let across = f.binary(BinOp::ShiftRightLogical, wide, halved_across)?;
        let tall = f.binary(BinOp::IAdd, height, halved_down)?;
        let down = f.binary(BinOp::ShiftRightLogical, tall, halved_down)?;
        let [x0, x1, wx] = cx.taps(f, x, across, halved_across)?;
        let [y0, y1, wy] = cx.taps(f, y, down, halved_down)?;
        let four = cx.uint(f, 4)?;
        let wx1 = f.binary(BinOp::ISub, four, wx)?;
        let wy1 = f.binary(BinOp::ISub, four, wy)?;
        let mut sums = [zero, zero];
        for (cy, wy) in [(y0, wy), (y1, wy1)] {
            let row = f.binary(BinOp::IMul, cy, chroma_row)?;
            let row = f.binary(BinOp::IAdd, chroma_offset, row)?;
            for (cx_, wx) in [(x0, wx), (x1, wx1)] {
                let weight = f.binary(BinOp::IMul, wx, wy)?;
                // The pair's Cb, code `2 x` of the row, and its Cr after it.
                let cb = f.binary(BinOp::ShiftLeftLogical, cx_, one)?;
                for (component, sum) in sums.iter_mut().enumerate() {
                    let index = match component {
                        0 => cb,
                        _ => f.binary(BinOp::IAdd, cb, one)?,
                    };
                    let code = cx.code(f, held, row, index)?;
                    let term = f.binary(BinOp::IMul, weight, code)?;
                    *sum = f.binary(BinOp::IAdd, *sum, term)?;
                }
            }
        }
        // `C = (S + 8) >> 4`.
        let eight = cx.uint(f, 8)?;
        let mut chroma = [zero, zero];
        for (c, sum) in chroma.iter_mut().zip(sums) {
            let rounded = f.binary(BinOp::IAdd, sum, eight)?;
            *c = f.binary(BinOp::ShiftRightLogical, rounded, four)?;
        }
        let coord = f.composite_construct(uvec2, &[x, y])?;
        let texel = match output {
            YCbCrOutput::Codes => {
                // `A2B10G10R10_UNORM_PACK32`'s or `R8G8B8A8_UNORM`'s bits, `b` each: red Cr,
                // green Y, blue Cb, and alpha every bit above (3 of 10-bit codes, 255 of 8-bit).
                let blue_at = f.binary(BinOp::ShiftLeftLogical, bits, one)?;
                let alpha_at = f.binary(BinOp::IAdd, blue_at, bits)?;
                let green = f.binary(BinOp::ShiftLeftLogical, luma, bits)?;
                let blue = f.binary(BinOp::ShiftLeftLogical, chroma[0], blue_at)?;
                let packed = f.binary(BinOp::BitwiseOr, chroma[1], green)?;
                let packed = f.binary(BinOp::BitwiseOr, packed, blue)?;
                let ones = cx.uint(f, u32::MAX)?;
                let alpha = f.binary(BinOp::ShiftLeftLogical, ones, alpha_at)?;
                let packed = f.binary(BinOp::BitwiseOr, packed, alpha)?;
                f.composite_construct(texel_ty, &[packed, zero, zero, zero])?
            }
            YCbCrOutput::Converted | YCbCrOutput::Srgb => {
                // Counted as 10-bit codes, less 512.
                let offset = cx.uint(f, 512)?;
                for c in chroma.iter_mut() {
                    let ten = f.binary(BinOp::ShiftLeftLogical, *c, up)?;
                    *c = f.binary(BinOp::ISub, ten, offset)?;
                }
                // The luma term: video range `9576 ((Y << u) - 64)`, full range
                // `256 ⌊(32768 Y + ⌊m / 2⌋) / m⌋` (two's complement words, read as signed
                // below).
                let y10 = f.binary(BinOp::ShiftLeftLogical, luma, up)?;
                let sixty_four = cx.uint(f, 64)?;
                let video = f.binary(BinOp::ISub, y10, sixty_four)?;
                let scale = cx.uint(f, 9576)?;
                let video = f.binary(BinOp::IMul, video, scale)?;
                let fifteen = cx.uint(f, 15)?;
                let wide = f.binary(BinOp::ShiftLeftLogical, luma, fifteen)?;
                let half = f.binary(BinOp::ShiftRightLogical, max, one)?;
                let wide = f.binary(BinOp::IAdd, wide, half)?;
                let quotient = f.binary(BinOp::UDiv, wide, max)?;
                let full_term = f.binary(BinOp::ShiftLeftLogical, quotient, eight)?;
                let is_full = f.binary(BinOp::INotEqual, full, zero)?;
                let l = f.select(is_full, full_term, video)?;
                // Each channel's sum `T = L + kb Cb' + kr Cr'` onto its grid `D`:
                // `clamp((T + D') >> s, 0, D)` with `(D', s, D)` `(2048, 12, 2048)` converted and
                // `(512, 10, 8192)` through the sRGB curve, whose knots then decode the step.
                let (bias, dropped, grid) = match knots {
                    Some(_) => (512, 10, 8192),
                    None => (2048, 12, 2048),
                };
                let round = cx.uint(f, bias)?;
                let shift = cx.uint(f, dropped)?;
                let top = cx.uint(f, grid)?;
                let step = 1.0f32 / grid as f32;
                let step = scalar(f, f32_, u64::from(step.to_bits()))?;
                let mut channels = Vec::with_capacity(4);
                for c in 0..3 {
                    let b_term = f.binary(BinOp::IMul, k[2 * c], chroma[0])?;
                    let r_term = f.binary(BinOp::IMul, k[2 * c + 1], chroma[1])?;
                    let t = f.binary(BinOp::IAdd, l, b_term)?;
                    let t = f.binary(BinOp::IAdd, t, r_term)?;
                    let t = f.binary(BinOp::IAdd, t, round)?;
                    let v = f.binary(BinOp::ShiftRightArithmetic, t, shift)?;
                    let below = f.binary(BinOp::SLessThan, v, zero)?;
                    let v = f.select(below, zero, v)?;
                    let above = f.binary(BinOp::SGreaterThan, v, top)?;
                    let v = f.select(above, top, v)?;
                    let v = match knots {
                        Some(table) => cx.srgb_step(f, table, v)?,
                        None => v,
                    };
                    let v = f.convert(ConvOp::ConvertUToF, v, f32_)?;
                    channels.push(f.binary(BinOp::FMul, v, step)?);
                }
                channels.push(scalar(f, f32_, u64::from(1.0f32.to_bits()))?);
                f.composite_construct(texel_ty, &channels)?
            }
        };
        let image = f.load(image, MemAccess::default())?;
        f.write(
            image,
            coord,
            texel,
            &ImageOperands::default(),
            TexelMemory::PRIVATE,
        )?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, converted, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: YCbCrConvertModule::LOCAL_SIZE,
            y: YCbCrConvertModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(YCbCrConvertModule::SET),
        Symbol::Binding(s) if *s == IMAGE => Some(YCbCrConvertModule::IMAGE_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(YCbCrConvertModule::SHAPE_BINDING),
        Symbol::Binding(s) if *s == BYTES => Some(YCbCrConvertModule::BYTES_BINDING),
        _ => None,
    }
}

/// The YCbCr conversion module of `device` writing texels of `output`: built with the typed
/// builder, checked by the owned verifier against the device's profile (requirements included),
/// serialized, and numbered with the fixed bindings of [`YCbCrConvertModule`].
pub fn ycbcr_convert_module(
    device: &Device,
    output: YCbCrOutput,
) -> Result<YCbCrConvertModule, Error> {
    let profile = device.profile();
    let m = typed_module(output)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the YCbCr conversion module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(YCbCrConvertModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
