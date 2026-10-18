//! Metal's mipmaps as its texture filter computes them, on every device.
//!
//! `generateMipmapsForTexture:` makes each level from the one above it. On an Apple M4 every
//! level, and on an Apple M3 Max every level of a 3D texture and the first level of each of its
//! passes over a 2D chain (below), is the GPU's bilinear (trilinear in 3D) filter of the level
//! above at each texel's centre. Our probes measured it on every mipmappable colour
//! format: 115,086 values of random, extreme and impulse chains of 2D textures, arrays, cubes and
//! 3D textures, halving and not, and 173,505 taps of level widths, heights and depths up to 16383.
//! The M3 Max computes a 2D chain's other levels by its own shader arithmetic (its "tail"). This
//! module computes the filter exactly, and the M3 Max's passes and tail of every 2D chain whose
//! first level is at most 65 texels wide and tall, and of the wider and taller ones whose passes
//! are known (below).
//!
//! **By release** ([`MipmapRule`]). The passes, the tail and the half below are macOS 26.7's
//! ([`MipmapRule::AppleM3MaxPasses`]). Under macOS 27.0.1 the M3 Max makes every level of every
//! chain by its filter of the level above, carrying no field through a half
//! ([`MipmapRule::Filter`]; our own probes: the 679 chains of the 26.7 tables rerun and 182 3D
//! chains of 14 formats at 13 extents, `measured_m3_27.txt`, and 556 chains of random bytes, 2D
//! textures to 16384x64 and 4096x4096, arrays, cubes and 3D textures of 32 formats). Every level of
//! every chain matched, `A1BGR5Unorm`'s and `ABGR4Unorm`'s, the 3D chains of odd widths and heights
//! and the half-float ones among them, but at two 8-bit sRGB near ties (below). Its reciprocal is
//! not separated from an Apple M4's by those chains (levels 41 and 61 texels wide among them); the
//! chain takes the M3 Max's.
//!
//! **The taps.** Along each axis, destination texel `i` of `w` from `W` reads source texels
//! `x0` and `min(x0 + 1, W - 1)` with weights `256 - q` and `q`: the centre's coordinate is
//! computed in `f32`, `u = (i + 0.5) * r` and `c = u * W - 0.5` (each operation rounded to
//! nearest even), `x0 = floor(c)`, and the fraction is rounded half up to `q / 256`. `r` is the
//! GPU's reciprocal of `w`: the M4's is `1 / w` rounded to nearest even, the M3 Max's one unit in
//! the last place off it at 1531 of the sizes up to 16384 (`reciprocal.rs`). An axis of one texel
//! that stays one has one tap. A texel's eight taps weigh `ax ay az` (of `2^24`; a 2D level's
//! depth taps weigh 256 and 0) split along y last: the second row's part of `ax az` (of `2^16`)
//! is `ax az qy` rounded half down to `2^-16`, the first row's the rest of `ax az`. Only a 3D level
//! of a width, height or depth that does not halve has a split that is not exact (an Apple M3 Max
//! under macOS 27.0.1: every weight of a 41x61x5 `RGBA32Float` chain's first
//! computed level, one tap a channel, read back exactly). Each layer of an array or cube is filtered
//! alone.
//!
//! **Whose answer.** The Apple M3 Max is the reference. Its filter (its coordinates, and its
//! half below) gives the first level of each of its passes and every level of a 3D chain; its
//! tail the other levels of a 2D chain, array or cube. Below the first level of a 2D chain whose
//! passes are not known (below) the M4's filter (its coordinates, no half) stands in, which
//! [`MipmapFilterChain::unlike_an_m3`] reports, with the values measured unlike the chain's:
//! four of `A1BGR5Unorm`'s and `ABGR4Unorm`'s in 2D, and in 3D two of the formats summed as
//! half floats (one unit each).
//!
//! **The M3 Max's passes** (the share-mips-m3, share-mips-m3b and share-mips-m3c on macOS 26.7: 2D
//! textures, arrays and cubes from 65 to 16384 texels wide or tall, every level). Each pass starts
//! from a level, its base. While the level below the base would be wider or taller than 32 texels,
//! the pass is the M3 Max's filter of the base, then:
//!
//! - **A texture Metal keeps losslessly compressed** (`MipmapFilterChain::plan`'s `compressed`):
//!   where that filtered level's width and height are even, its tail of the values the filter
//!   kept, or where it is one texel tall (or wide) and its width (or height) is even, its tail
//!   of that level's codes, for one level.
//! - **A texture Metal does not compress**: a strip, narrower or shorter than 16 texels
//!   (`PASS_SIDE`), which Metal never compresses, or one with `allowGPUOptimizedContents` off or a
//!   `ShaderWrite` or `PixelFormatView` usage. Up to five levels of its tail while each level
//!   halves the one above exactly (each side even, or one texel): the values the level above kept
//!   and its codes in turn, the first from kept values, but codes only where the filtered level's
//!   shorter side is a power of two of at most 16 texels (`RUN_CODES_SIDE`). Examples: 4096x1 runs
//!   filter (2048x1), five levels of codes, filter (32x1), tail; an uncompressed 256x256 runs
//!   filter (128x128), kept, codes, kept, codes, kept (4x4), filter (2x2), tail.
//!
//! A format carried through a half reads the kept values wherever the others read codes. The
//! next pass starts from the pass's last level. Otherwise the last pass is the filter of its
//! base and the tail of every level below it. That tail reads the level above's kept values at
//! an even distance below the base where the level above is a square of a power of two and
//! either the filter's or half of the level above it (and at every level of a format carried
//! through a half); the level above's codes everywhere else. A first level at most 65 texels
//! wide and tall is the base of its only pass. Every level of the chains measured fits but at
//! the near ties below, for every format the filter computes (8-bit unorm and 32-bit float
//! formats, every other but `RG11B10Float`, `RG11B10Float`): the passes are known, for a compressed
//! texture where no pass starts from a level more than 128 times wider than tall or taller than
//! wide (`PASS_ASPECT`; 64, 128 measured).
//!
//! **Where the passes are not known.** The chains of compressed textures past `PASS_ASPECT` take
//! an Apple M4's filter below the first level.
//!
//! **Near ties** ([`MipmapFilterChain::near_ties`]). Some roundings miss a few values on an Apple
//! M3 Max, one unit each, at sums a hair from a rounding boundary, by no rule fit yet:
//! - the filter's 8-bit sRGB colour code (a linear sum of 906.5002 steps stored as byte 129, the
//!   chain's 130: 1 of 5,331,270 values of share-mips-m3; on an Apple M4 too; under macOS 27.0.1
//!   one texel in each of 2 of the 588 2D chains of `measured_m3_27.txt`, sums of 906.5002 and
//!   413.5002 steps);
//! - a 16-bit float tail level read from kept values (one unit in the last place: 8 of 538,088;
//!   on an Apple M4 too; share-mips-m3b's 54 of 3,326,656 tail values, every one a level read
//!   from kept values);
//! - the extended-range sRGB formats, which the M3 Max carries through a half under macOS 26.7:
//!   its filter's levels and the levels its tail reads from kept values (their first computed
//!   levels miss at odd weights, on an Apple M4 too). Under macOS 27.0.1, which carries no field
//!   through a half, none missed.
//!
//! The other formats are exact:
//! - the filter's 10-bit grid (`BGR10A2Unorm`'s 193x193 sum of `S = 283.375015` codes, 2^-16 code
//!   past a quarter-code half, stored as 283): each term, a code times its weight, is floored to
//!   `2^-14` code before the sum (below);
//! - `BGR10_XR` and `BGRA10_XR` the same way, a colour's terms floored to `2^-13` code and alpha's
//!   to `2^-15` (2 of 9,216 texels of `BGR10_XR`'s 193x193 level missed the exact sum);
//! - `ABGR4Unorm` and `A1BGR5Unorm`, each code filtered as a wider code (below; 4 of 9,216 texels
//!   of each one's 193x193 level missed each code on its own grid).
//!
//! **The M3 Max's half.** At its filter's levels the M3 Max carries `ABGR4Unorm`,
//! `A1BGR5Unorm` and the extended-range formats through a half float: a normalized field's grid
//! sum `Q` (below; a widened field's of its wide code, `m` times its own) is `v = Q / (den m
//! 2^k)` rounded to a half (nearest even) and stored as `v den` rounded to nearest even; an
//! extended-range colour's `v = (Q / 16 - 384) / 510`, alpha's `(Q / 8 - 384) / 510` (on its 1/8
//! grid), stored as `v 510 + 384` rounded to nearest even; an extended-range sRGB sum's magnitude
//! in steps of `2^-13` is rounded half away from zero to 11 significant bits before its search.
//!
//! **The M3 Max's tail** (115,743 values of squares to 64x64, arrays and cubes, and 2D chains to
//! 61x37, none missed but 4 of `A1BGR5Unorm` and `ABGR4Unorm`). Each axis of destination texel `i`
//! of `w` from `W`: the centre `x = (i + 0.5) W / w` rounded to a half float (nearest even), to an
//! `f32` from a level of 4096 texels or more along the axis (`tail::HALF_CENTRE`), `c = x - 0.5`,
//! `x0 = floor(c)` and `t = c - x0` (a multiple of `2^-11`); source texels `x0` and `min(x0 + 1,
//! W - 1)`. Each field mixes its four values along x and then y, `mix(a, b, t) = a (1 - t) + b
//! t` with `b t` rounded and then the sum (one fused multiply-add), each to a half float for
//! `RGBA16Float`'s family, the formats carried through a half, and to `f32` for the rest
//! (subnormals kept; IEEE's infinities and NaNs, a NaN or an infinity times a weight of 0 a NaN).
//! An axis of one texel reads it alone, with no mix along it: an infinity there stays one (a
//! one-texel tall level of `RG11B10Float`'s 4096x4 strip). The values it reads are the level
//! above's codes as it reads them (`code / den` rounded to the field's precision, an sRGB byte's
//! linear step over 4095, an extended-range code's `(code - 384) / 510` or its signed linear step
//! over 8192, a float's value), or, where its passes say so (above), the level above's unrounded
//! values (the first level's `Q / (den 2^k)` rounded to `f32`, the sRGB step sum over 4095, the
//! 16-bit code's value, the half answers of the formats summed as halves, a float's value). It
//! stores `v den` rounded to nearest even for a normalized or 16-bit code, `v 4095` rounded and
//! searched for an sRGB byte, `v 510 + 384` (alpha's `v` clamped to `[0, 1]`) or `|v| 8192`
//! searched for an extended-range code, an `f32`'s value with those below `2^-126` flushed to their
//! signed zero, a half's value, an unsigned float's value toward zero (the largest finite above
//! it), and `RGB9E5Float`'s shared exponent of the largest channel with each mantissa floored.
//! Every zero is +0.
//!
//! **The sum.** Each field of the texel sums its taps' decoded values times `n`, by its format's
//! rule (`S` exact unless stated, in units of `2^-24`):
//!
//! - **8- and 10-bit unorm, 8-bit snorm, and sRGB alpha**: `S` of the codes (an snorm code of
//!   `-2^(b-1)` read as `-(2^(b-1) - 1)`) rounded half up to a 12-bit grid, `Q = (S 2^(12-b) +
//!   2^23) >> 24`; the code is `Q`'s integer part, one more above a half, and at an exact half
//!   the code Metal's `f32` step gives: `RNE((x + 1/2) / den) den` rounded to nearest even. Each
//!   term `code n` is first floored to `2^(b - 24)` code, `(code n) >> b << b`: a 2D level's `n` is
//!   a multiple of `2^8`, so this floors only the 10-bit codes' terms there, to `2^-14` code.
//! - **Widened: 2-, 4-, 5- and 1-bit unorm** (`RGB10A2Unorm`'s and `BGR10A2Unorm`'s alpha;
//!   `ABGR4Unorm`'s fields, `A1BGR5Unorm`'s): each code as the code of `w` bits its bits replicate
//!   to, `m c` (`m = (2^w - 1) / den`: 341 for 2 bits to 10, 17 for 4 to 8, 33 for 5 to 10, 1023
//!   for 1 to 10), summed by that code's rule above into its grid sum `Q`; the code is the one
//!   nearest `Q / (m 2^k)` (`k = 12 - w`), at an exact half by the code's own `f32` step (the value
//!   `Q / ((2^w - 1) 2^k)` as an `f32`, times `den`, rounded to nearest even). Its first level
//!   keeps `Q / ((2^w - 1) 2^k)`.
//! - **8-bit sRGB colour**: `S` of each byte's twelve-bit linear step (`lower::arith::srgb`'s
//!   table), `(S + 2^23) >> 24` clamped to 4095, then the byte whose encode range holds it.
//! - **16-bit unorm and snorm**: each term floored to 1/256 code, `T = (code n) >> 16`, and the
//!   code `(sum T + 128) >> 8`, clamped (on a halving chain, the box average rounded half up).
//! - **32-bit float**: Metal's accumulator. Subnormal inputs read as zero. With `e` the largest
//!   of the live terms' exponent estimates `msb(M) + z + msb(n) - 24` (the value `M 2^z`), each
//!   term `M n 2^z` is truncated toward zero to `2^(e - 38)` and summed; the sum is rounded half
//!   away from zero to 24 bits, `+0` where it lies below `2^-126` before rounding. A NaN tap
//!   of nonzero weight, or both infinities, gives `0x7fc00000`; one infinity gives itself. A 3D
//!   level sums each depth slice's four terms so, at the slice's own `e`, then truncates each
//!   slice's sum toward zero to `2^(e - 38)` of the larger `e` and adds them (every value of the 3D
//!   `RGBA32Float` and `R16Float` chains, halving and not, on an Apple M3
//!   Max under macOS 27.0.1).
//! - **16-bit float**: the same accumulator at `2^(e - 17)`, rounded half away from zero to a
//!   half, subnormals kept; a NaN gives `0x7e00`.
//! - **`RG11B10Float`**: each field's half-float answer as above (infinities and NaNs as a
//!   half's), its low mantissa bits dropped.
//! - **`RGB9E5Float`**: each channel's half-float answer, then the shared exponent of the largest
//!   (`max(-16, exponent) + 16`, at most 31) and each mantissa floored, at most 511.
//! - **Extended range** (`BGR10_XR`, `BGRA10_XR`): a colour code's `S`, each term floored to
//!   `2^-13` code first, rounded half up to 1/16 code, and at an exact half the code of `RNE((x +
//!   1/2 - 384) / 510)`; alpha clamped to `[384, 894]` first, each term floored to `2^-15` code, on
//!   a 1/8 grid. The `_sRGB` colour codes sum their linear steps
//!   (`lower::arith::extended_range::table`, signed about code 384), each term floored to `2^-9`
//!   step, rounded half up to a step, and take the code whose encode range holds the magnitude
//!   (`layout::XR_SRGB_FLOOR`). `BGR10_XR`'s two
//!   padding bits are set; `BGRA10_XR`'s low six bits of each lane are 0.
//!
//! `B5G6R5Unorm` and `BGR5A1Unorm` are left out: their 5-bit fields miss this rule at a few grid
//! ties (43 and 2 of the values measured), by no rule found.
//!
//! **The consumer** computes the levels on the device in one scratch buffer laid out by
//! [`MipmapFilterChain::plan`]: it copies the first level of every layer in, fills every later
//! level with zeros where the chain says so ([`MipmapFilterChain::zeroed`]: texels narrower than
//! a word), dispatches [`mipmap_filter_module`] of each later level's kernel
//! ([`MipmapFilterLevel::kernel`]) once for the level, each reading the level above (and the
//! values it keeps, where it has them), and copies the later levels back.
//! [`MipmapFilterChain::expected_levels`] is the same chain on the CPU.
//!
//! **The module** (one per [`MipmapFilterKernel`] and device: the level's format and step built
//! in; it needs `shaderInt64`). One `GLCompute` entry `main` with
//! local size ([`MipmapFilterModule::LOCAL_SIZE`], 1, 1), no push constants, and five storage
//! buffers in set 0, each a whole number of 4-byte words:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`MipmapFilterModule::SOURCE_BINDING`] | the level above, as [`MipmapFilterLevel`] lays it |
//! | [`MipmapFilterModule::DESTINATION_BINDING`] | the level the dispatch writes |
//! | [`MipmapFilterModule::PARAMS_BINDING`] | the level's words ([`MipmapFilterLevel::params`]) |
//! | [`MipmapFilterModule::KEPT_IN_BINDING`] | the level above's [`MipmapFilterLevel::kept`] |
//! | [`MipmapFilterModule::KEPT_OUT_BINDING`] | the level's own [`MipmapFilterLevel::kept`] |
//!
//! A level holds each layer's texels in turn, each layer's images, rows and texels tightly
//! packed, in the bytes of the Vulkan format the runtime holds the pixel format in. Invocation
//! `(x, y)` computes destination texel `y * row + x`: a texel narrower than a word is ORed into
//! its word (atomically, the level zeroed first), a wider one stores its words (every word of
//! the level, which needs no zeros), and one past the level's texels writes nothing.
//!
//! **Synchronization** (consumer): the copy in, the fills and the params' upload →
//! `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the first dispatch;
//! each dispatch's `SHADER_STORAGE_WRITE` → `SHADER_STORAGE_READ` before the next; the last →
//! `TRANSFER_READ` before the copies back.

mod fetch;
mod kernel;
mod layout;
mod model;
mod reciprocal;
mod soft;
mod tail;
mod tail_kernel;
#[cfg(test)]
pub(crate) mod tests;

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::{Requirements, VkFeature};
use crate::device::Device;
use crate::error::{Capability, Error, Feature, InternalError, InternalKind, Unsupported};
use crate::metal::PixelFormat;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::vk::Format;
use layout::{axis_tap, header, tie_tables, Layout, Rule, HEADER_WORDS, KEPT_WORDS};
use reciprocal::Reciprocal;

/// How a Metal release makes each level of a chain after the first (the module docs' "By
/// release"). No Metal counterpart: Metal answers as the release it runs on.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MipmapRule {
    /// macOS 26.7 on an Apple M3 Max: its filter, carrying the formats it carries through a half,
    /// at the first level of each of its passes over a 2D chain, array or cube and at every level
    /// of a 3D chain, and its tail at the other levels.
    AppleM3MaxPasses,
    /// macOS 27.0.1 on an Apple M3 Max: its filter of the level above at every level of every
    /// chain, every field at its own precision.
    Filter,
}

/// The crate-owned mipmap filter module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MipmapFilterModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (`shaderInt64` among it).
    pub requirements: Requirements,
}

impl MipmapFilterModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The level above, a storage buffer the dispatch reads; or, in the kernel of the first
    /// later level of a chain that reads its first level in place
    /// ([`MipmapFilterChain::fetch`]), a sampled image: the first level's view in that format,
    /// its one level the first level, a 2D array of every layer (3D for a 3D texture).
    pub const SOURCE_BINDING: u32 = 0;
    /// The level the dispatch writes, a storage buffer zeroed before it where the chain says so
    /// ([`MipmapFilterChain::zeroed`]).
    pub const DESTINATION_BINDING: u32 = 1;
    /// The level's words, a storage buffer the dispatch reads.
    pub const PARAMS_BINDING: u32 = 2;
    /// The values the level above keeps ([`MipmapFilterLevel::kept`]), a storage buffer the
    /// dispatch reads where the level above has them.
    pub const KEPT_IN_BINDING: u32 = 3;
    /// The values the level keeps for the next, a storage buffer the dispatch writes where the
    /// level has them.
    pub const KEPT_OUT_BINDING: u32 = 4;
    /// Invocations of one workgroup, along x: one destination texel each.
    pub const LOCAL_SIZE: u32 = kernel::LOCAL_SIZE;
}

/// One level of a [`MipmapFilterChain`] in its scratch buffer.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MipmapFilterLevel {
    /// Where the level's texels start in the scratch buffer: a multiple of the chain's alignment.
    pub offset: u64,
    /// The level's bytes: every layer's texels, tightly packed, layer after layer.
    pub bytes: u64,
    /// The level's width, height and depth.
    pub extent: [u32; 3],
    /// For every level after the first: where its words start in the scratch buffer (a
    /// multiple of the chain's alignment), and the words, to upload there as little-endian
    /// words and bind at [`MipmapFilterModule::PARAMS_BINDING`] over their bytes: the level's
    /// shape, the format's tie tables and the level's tap tables.
    pub params: Option<(u64, Vec<u32>)>,
    /// For every level after the first: the kernel its dispatch runs
    /// ([`mipmap_filter_module`]).
    pub kernel: Option<MipmapFilterKernel>,
    /// For every level after the first: the workgroups to dispatch along x and y.
    pub groups: [u32; 2],
    /// Where the values this level keeps for the next start in the scratch buffer (a multiple
    /// of the chain's alignment) and their bytes, where the next level reads them: four
    /// 4-byte words a texel, every layer's texels in turn.
    pub kept: Option<(u64, u64)>,
}

/// The scratch buffer and the dispatches of one `generateMipmapsForTexture:` (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MipmapFilterChain {
    /// Every level, the first (which the consumer copies in) first. Level `i` after the first
    /// is one dispatch of [`mipmap_filter_module`] of its kernel: level `i - 1` bound at
    /// [`MipmapFilterModule::SOURCE_BINDING`] and level `i` at
    /// [`MipmapFilterModule::DESTINATION_BINDING`], each over its bytes rounded up to 4, its
    /// params at [`MipmapFilterModule::PARAMS_BINDING`], level `i - 1`'s
    /// [`kept`](MipmapFilterLevel::kept) values at [`MipmapFilterModule::KEPT_IN_BINDING`] and
    /// level `i`'s at [`MipmapFilterModule::KEPT_OUT_BINDING`] (where a level has none, any range
    /// of the scratch buffer: the dispatch does not touch it).
    pub levels: Vec<MipmapFilterLevel>,
    /// Bytes of the scratch buffer.
    pub bytes: u64,
    /// The pixel format of the texture.
    pub pixel_format: PixelFormat,
    /// The layers of every level (an array's slices, a cube's faces; 1 in 3D).
    pub layers: u32,
    /// Whether an Apple M3 Max's answer may differ from the chain's somewhere (the module docs'
    /// "Whose answer"): below the first level of a 2D chain, array or cube whose M3 Max passes
    /// are not known (an Apple M4's filter stands in there: a first level wider or taller than
    /// 65 texels of a compressed texture with a pass from a level more than 128 times wider than
    /// tall or taller than wide), in `A1BGR5Unorm`'s 2D chains and below the first level of
    /// `ABGR4Unorm`'s (a few measured values unexplained), and in a 3D chain of a format summed as
    /// half floats (two); never under [`MipmapRule::Filter`] (the module docs' "By release").
    pub unlike_an_m3: bool,
    /// Whether a level may be one unit from Metal's where its sum lies a hair from a rounding
    /// boundary, by a rounding the chain does not fit yet (the module docs' "Near ties"): a level
    /// of the filter's (any 3D level, the first computed level of any chain, every level under
    /// [`MipmapRule::Filter`]) of a format with an 8-bit sRGB colour field or of an extended-range
    /// sRGB format, one code; or a tail level of a 16-bit float format or an extended-range
    /// sRGB one read from the level above's kept values (one unit in the last place, or one code).
    /// Not a 10-bit field's (`RGB10A2Unorm`, `BGR10A2Unorm`; exact), nor `BGR10_XR`'s and
    /// `BGRA10_XR`'s, nor `ABGR4Unorm`'s and `A1BGR5Unorm`'s.
    pub near_ties: bool,
    /// Whether the consumer fills each later level with zeros before its dispatch: where the
    /// format's texels are narrower than a word, which a dispatch ORs into their words. A wider
    /// texel's dispatch writes every word of its level.
    pub zeroed: bool,
    /// Whether the chain is a 3D texture's: every level the M3 Max's filter.
    volume: bool,
    /// Whether Metal keeps the texture losslessly compressed (`plan`'s `compressed`).
    compressed: bool,
    /// The release's rule (`plan`'s `rule`).
    rule: MipmapRule,
    /// The format of the sampled view the first later level fetches the first level through,
    /// where the chain reads its first level in place (`plan`'s `fetch`): the first level then
    /// has no bytes in the scratch buffer (its offset is 0 and nothing is copied in), and the
    /// first later level's kernel binds the view at
    /// [`MipmapFilterModule::SOURCE_BINDING`]. `None` where the first level is copied in.
    pub fetch: Option<Format>,
}

/// The kernel a level after the first dispatches ([`MipmapFilterLevel::kernel`]): the module
/// [`mipmap_filter_module`] builds for it computes the level's format's rules and its step (the
/// GPU's filter, with the fields the M3 Max carries through a half converted or not, or the M3
/// Max's tail, from the level above's codes or its kept values), keeps the values the next
/// level reads or not, and takes the depth taps of a 3D level or not, each fixed in the
/// module, so that it computes only what the level needs. Levels of one kernel share a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MipmapFilterKernel {
    pixel_format: PixelFormat,
    step: KernelStep,
    keep: bool,
    depth: bool,
    fetch: Option<FetchView>,
}

/// The sampled view a kernel fetches the first level from ([`MipmapFilterChain::fetch`]): its
/// format, and whether it is a 3D view (a 3D texture's) rather than a 2D array of every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct FetchView {
    format: Format,
    volume: bool,
}

/// What a kernel's level computes ([`MipmapFilterKernel`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum KernelStep {
    /// The GPU's filter, the fields the M3 Max carries through a half converted where `half`.
    Sampler { half: bool },
    /// The M3 Max's tail, from the level above's kept values where `read_kept`.
    Tail { read_kept: bool },
}

impl MipmapFilterKernel {
    /// The kernel of a level of `pixel_format` (of `layout`) computed by `step` from a level
    /// above of `big` texels, keeping its values for the next where `keep`. A flag that changes
    /// nothing is left out, so that levels it would not change share a kernel: `half` for a
    /// format the M3 Max carries through no half, the depth taps where the level above is one
    /// texel deep (its second depth tap weighs 0). `fetch`: the first level's view, for the first
    /// later level of a chain that reads it in place.
    fn new(
        pixel_format: PixelFormat,
        layout: &Layout,
        step: Step,
        keep: bool,
        big: [u32; 3],
        fetch: Option<FetchView>,
    ) -> MipmapFilterKernel {
        let step = match step {
            Step::Sampler { half, .. } => KernelStep::Sampler {
                half: half && layout.half,
            },
            Step::Tail { kept } => KernelStep::Tail { read_kept: kept },
        };
        MipmapFilterKernel {
            pixel_format,
            step,
            keep,
            depth: big[2] > 1,
            fetch,
        }
    }

    /// Whether the kernel fetches the first level from its sampled view, bound at
    /// [`MipmapFilterModule::SOURCE_BINDING`] ([`MipmapFilterChain::fetch`]), rather than
    /// reading the level above from a storage buffer there.
    pub fn fetches(&self) -> bool {
        self.fetch.is_some()
    }

    /// What the kernel is built with: its format's texel and records, and its level's step.
    fn spec(&self) -> Option<kernel::Spec> {
        let layout = layout::layout(self.pixel_format)?;
        let (_, tie_at) = tie_tables(&layout)?;
        let records = layout
            .fields
            .iter()
            .zip(tie_at)
            .map(|(f, at)| layout::record(&layout, f, at))
            .collect();
        let (tail, half, read_kept) = match self.step {
            KernelStep::Sampler { half } => (false, half, false),
            KernelStep::Tail { read_kept } => (true, false, read_kept),
        };
        Some(kernel::Spec {
            bytes: layout.bytes,
            records,
            rgb9e5: layout.rgb9e5(),
            tail,
            half,
            keep: self.keep,
            read_kept,
            depth: self.depth,
            fetch: match self.fetch {
                Some(view) => Some(kernel::Fetch {
                    components: fetch::components(view.format)?,
                    volume: view.volume,
                }),
                None => None,
            },
        })
    }
}

/// How a level after the first is computed (the module docs' "Whose answer").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    /// The GPU's filter, its coordinates from `reciprocal`'s GPU, the fields the M3 Max carries
    /// through a half converted where `half` (a level of the M3 Max's own).
    Sampler { reciprocal: Reciprocal, half: bool },
    /// The M3 Max's tail, reading the level above's kept values where `kept`.
    Tail { kept: bool },
}

/// Whether an extent is a square of a power of two (one texel included).
fn square_power_of_two(e: [u32; 3]) -> bool {
    e[0] == e[1] && e[0].is_power_of_two() && e[2] == 1
}

/// The step of level `level` (after the first) of the M3 Max's chain of `extents` of a format of
/// `layout`, of a 3D texture where `volume`, under `rule`: its filter at every level under
/// [`MipmapRule::Filter`]; else its own where its passes are known ([`m3_passes`]), an Apple M4's
/// filter in their place below the first level where they are not.
fn step_of(
    extents: &[[u32; 3]],
    level: usize,
    volume: bool,
    compressed: bool,
    layout: &Layout,
    rule: MipmapRule,
) -> Step {
    let m3 = Step::Sampler {
        reciprocal: Reciprocal::AppleM3Max,
        half: rule == MipmapRule::AppleM3MaxPasses,
    };
    if rule == MipmapRule::Filter || volume || level == 1 {
        return m3;
    }
    match m3_passes(extents, compressed, layout).map(|passes| passes[level]) {
        Some(Pass::Filter) => m3,
        Some(Pass::Kept) => Step::Tail { kept: true },
        Some(Pass::Codes) => Step::Tail { kept: false },
        Some(Pass::Tail { base }) => {
            let (rel, above) = (level - base, extents[level - 1]);
            let halved = rel == 2 || extents[level - 2] == [2 * above[0], 2 * above[1], 1];
            Step::Tail {
                kept: layout.half
                    || (rel.is_multiple_of(2) && square_power_of_two(above) && halved),
            }
        }
        None => Step::Sampler {
            reciprocal: Reciprocal::AppleM4,
            half: false,
        },
    }
}

/// How the M3 Max computes one level after the first of a 2D chain (the module docs' "The M3
/// Max's passes").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// Its filter of the level above: a pass's first level.
    Filter,
    /// Its tail from the values the level above (a pass's first) kept: a pass's second level.
    Kept,
    /// Its tail from the level above's codes: a pass's second level where the pass's first is
    /// one texel tall or wide, or a later level of a strip's pass.
    Codes,
    /// Its tail below level `base`, the level its last pass starts from.
    Tail { base: usize },
}

/// How the M3 Max computes each level of a 2D chain, array or cube of `extents` (index 0 the
/// first level's, unused) of a format of `layout`, its texture losslessly compressed where
/// `compressed` (`MipmapFilterChain::plan`): passes of its filter, each followed by its tail of
/// the values the filter kept where the filtered level halves exactly, while the filtered level
/// would be wider or taller than [`TAIL_EXTENT`], or of the level above's codes where the
/// filtered level is one texel tall (or wide) and of an even width (or height); then its filter and
/// its tail below it. A texture Metal does not compress, among them a strip (a first level narrower
/// or shorter than [`PASS_SIDE`] texels), runs its passes before the last otherwise: its filter,
/// then up to [`STRIP_RUN`] levels of its tail while each level halves the one above exactly (each
/// side even or one texel), from the values the level above kept and its codes in turn, the first
/// from kept values, but from codes only where the filtered level's shorter side is a power of two
/// of at most [`RUN_CODES_SIDE`] texels. A format carried through a half reads kept values wherever
/// the others read codes. `None` where a pass before the last lies outside what was measured: a
/// compressed texture's pass from a level more than [`PASS_ASPECT`] times wider than tall or taller
/// than wide.
fn m3_passes(extents: &[[u32; 3]], compressed: bool, layout: &Layout) -> Option<Vec<Pass>> {
    // A strip, which Metal never compresses, or a texture Metal does not compress runs.
    let runs = !compressed || extents[0][0].min(extents[0][1]) < PASS_SIDE;
    let mut passes = vec![Pass::Filter; extents.len()];
    let mut base = 0;
    while base + 1 < extents.len() {
        let [w, h, _] = extents[base + 1];
        if w <= TAIL_EXTENT && h <= TAIL_EXTENT {
            for pass in &mut passes[base + 2..] {
                *pass = Pass::Tail { base };
            }
            break;
        }
        if runs {
            // The filtered level, then the levels of its run: kept values and codes in turn, or
            // codes only below a filtered level whose shorter side is a power of two of at most
            // `RUN_CODES_SIDE` texels; a format carried through a half reads kept values at every
            // level, as its tail does.
            let side = w.min(h);
            let turns = !(side.is_power_of_two() && side <= RUN_CODES_SIDE);
            let mut last = base + 1;
            while last + 1 < extents.len() && last - base <= STRIP_RUN {
                let [aw, ah, _] = extents[last];
                if !(aw.is_multiple_of(2) || aw == 1) || !(ah.is_multiple_of(2) || ah == 1) {
                    break;
                }
                let kept = layout.half || (turns && (last - base) % 2 == 1);
                passes[last + 1] = if kept { Pass::Kept } else { Pass::Codes };
                last += 1;
            }
            base = last;
            continue;
        }
        let [bw, bh, _] = extents[base];
        if bw.max(bh) > PASS_ASPECT * bw.min(bh) {
            return None;
        }
        let below = base + 2 < extents.len();
        if below && w.is_multiple_of(2) && h.is_multiple_of(2) {
            passes[base + 2] = Pass::Kept;
            base += 2;
        } else if below && w.min(h) == 1 && w.max(h).is_multiple_of(2) {
            passes[base + 2] = if layout.half { Pass::Kept } else { Pass::Codes };
            base += 2;
        } else {
            base += 1;
        }
    }
    Some(passes)
}

/// The widest and tallest level the M3 Max's tail computes below its filter's (the module
/// docs): a chain whose first computed level is at most this wide and tall is its filter and
/// its tail below it.
const TAIL_EXTENT: u32 = 32;

/// The narrowest first level whose passes are the compressed texture's (the module docs' "The
/// M3 Max's passes"): Metal compresses no texture narrower or shorter, whose passes are a
/// strip's.
const PASS_SIDE: u32 = 16;

/// The most levels of its tail a pass before its last runs below its filter's, in a strip or a
/// texture Metal does not compress.
const STRIP_RUN: usize = 5;

/// The longest shorter side, a power of two, of a filtered level whose run reads only codes; a
/// run below any other alternates kept values and codes.
const RUN_CODES_SIDE: u32 = 16;

/// The most a level a pass starts from is wider than tall, or taller than wide, where its
/// passes are known (the module docs' "The M3 Max's passes"; 128 measured).
const PASS_ASPECT: u32 = 128;

/// The words of one later level computed by `step` (the module docs; `layout::header`): its
/// shape, the format's tie tables and the level's tap tables.
fn params(
    layout: &Layout,
    big: [u32; 3],
    small: [u32; 3],
    texels: u32,
    row: u32,
    step: Step,
) -> Option<Vec<u32>> {
    let mut words = vec![0u32; HEADER_WORDS as usize];
    let h = |w: &mut Vec<u32>, at: u32, v: u32| w[at as usize] = v;
    h(&mut words, header::TEXELS, texels);
    h(&mut words, header::ROW, row);
    for a in 0..3 {
        h(&mut words, header::EXTENT + a, small[a as usize]);
        h(&mut words, header::ABOVE + a, big[a as usize]);
    }
    words.extend(tie_tables(layout)?.0);
    for a in 0..3 {
        let at = u32::try_from(words.len()).ok()?;
        h(&mut words, header::TAPS + a as u32, at);
        for i in 0..small[a] {
            let (x0, w) = match step {
                Step::Sampler { reciprocal, .. } => axis_tap(i, big[a], small[a], reciprocal),
                Step::Tail { .. } => tail::axis(i, big[a], small[a]),
            };
            words.push(x0 | (w << 16));
        }
    }
    Some(words)
}

impl MipmapFilterChain {
    /// The chain of `levels` levels of `layers` layers of `pixel_format` from a first level of
    /// `extent` (its depth 1 unless `volume`, a 3D texture's), in a scratch buffer whose bound
    /// ranges start at multiples of `alignment` (`minStorageBufferOffsetAlignment`, at least 16
    /// here, as image copies take) and span at most `max_range` bytes (`maxStorageBufferRange`),
    /// each dispatch at most `max_groups` workgroups along x and y (`maxComputeWorkGroupCount`).
    ///
    /// `fetch` asks to read the first level in place, through a sampled view of that Vulkan
    /// format: the image's own format, or the UNORM twin of an 8-bit sRGB one (the bytes of the
    /// same channels). The chain takes it ([`MipmapFilterChain::fetch`]) where a fetch of that
    /// format recovers the texel's bytes (`fetch.rs`: the 8-bit unorm and snorm
    /// formats, the 10-bit unorm ones, the 4- and 5-bit packed unorm ones, the 16-bit unorm,
    /// snorm and float ones, and the integer carriers of the extended-range formats), and copies
    /// the first level in otherwise. A fetch of a normalized or float format reads the device's
    /// conversion, which Vulkan does not state: the consumer asks for one only on a device
    /// measured to return values whose recovery reads every code back.
    ///
    /// `rule` is the release's ([`MipmapRule`]), which Metal level the consumer answers as.
    ///
    /// `compressed` says whether Metal keeps the texture losslessly compressed: with
    /// `allowGPUOptimizedContents`, a first level 16 texels or more on its shorter side, and a
    /// usage of neither `ShaderWrite` nor `PixelFormatView` (the runtime's measured rule). The
    /// Apple M3 Max's passes over a 2D chain, array or cube depend on it: a compressed texture's
    /// passes keep one level of the tail each, and others' run up to five
    /// ([`MipmapFilterChain::unlike_an_m3`] says where they are not known).
    ///
    /// `None` for a format the filter does not compute (the module docs), and where the
    /// device's limits do not fit: a level past `max_range`, a dispatch past `max_groups`, or
    /// offsets past `u64`. Also `None` for fewer than two levels, no layers, an empty extent, or
    /// an extent past 32768 (the tap tables hold 16 bits).
    pub fn plan(
        pixel_format: PixelFormat,
        extent: [u32; 3],
        layers: u32,
        levels: u32,
        volume: bool,
        compressed: bool,
        rule: MipmapRule,
        alignment: u64,
        max_range: u32,
        max_groups: [u32; 2],
        fetch: Option<Format>,
    ) -> Option<MipmapFilterChain> {
        let layout = layout::layout(pixel_format)?;
        if levels < 2
            || layers == 0
            || extent.contains(&0)
            || extent.iter().any(|&a| a > 32768)
            || !alignment.is_power_of_two()
        {
            return None;
        }
        let alignment = alignment.max(16);
        let fetch = fetch.filter(|&format| {
            fetch::components(format).is_some_and(|components| fetch::fits(&layout, &components))
        });
        let extent_of = |level: u32| extent.map(|axis| (axis >> level.min(31)).max(1));
        let mut placed = Vec::with_capacity(levels as usize);
        let mut at = 0u64;
        for level in 0..levels {
            let e = extent_of(level);
            let texels =
                u64::from(layers).checked_mul(e.iter().map(|&a| u64::from(a)).product())?;
            let bytes = texels.checked_mul(u64::from(layout.bytes))?;
            let texels = u32::try_from(texels).ok()?;
            // A first level read in place has no bytes in the scratch buffer.
            if level == 0 && fetch.is_some() {
                placed.push((0, bytes, e, texels));
                continue;
            }
            if bytes.next_multiple_of(4) > u64::from(max_range) {
                return None;
            }
            placed.push((at, bytes, e, texels));
            at = at.checked_add(bytes)?.checked_next_multiple_of(alignment)?;
        }
        let extents: Vec<[u32; 3]> = placed.iter().map(|p| p.2).collect();
        let step = |level: usize| step_of(&extents, level, volume, compressed, &layout, rule);
        let keeps = |level: usize| {
            level + 1 < placed.len() && matches!(step(level + 1), Step::Tail { kept: true })
        };
        let mut out = Vec::with_capacity(placed.len());
        for (level, &(offset, bytes, extent, texels)) in placed.iter().enumerate() {
            if level == 0 {
                out.push(MipmapFilterLevel {
                    offset,
                    bytes,
                    extent,
                    params: None,
                    kernel: None,
                    groups: [0, 0],
                    kept: None,
                });
                continue;
            }
            let big = placed[level - 1].2;
            let groups = texels.div_ceil(MipmapFilterModule::LOCAL_SIZE);
            let x = groups.min(max_groups[0]);
            if x == 0 {
                return None;
            }
            let y = groups.div_ceil(x);
            if y > max_groups[1] {
                return None;
            }
            let row = x.checked_mul(MipmapFilterModule::LOCAL_SIZE)?;
            let keep = keeps(level);
            let words = params(&layout, big, extent, texels, row, step(level))?;
            let view = fetch
                .filter(|_| level == 1)
                .map(|format| FetchView { format, volume });
            let kernel =
                MipmapFilterKernel::new(pixel_format, &layout, step(level), keep, big, view);
            if 4 * words.len() as u64 > u64::from(max_range) {
                return None;
            }
            let params_at = at;
            at = at
                .checked_add(4 * words.len() as u64)?
                .checked_next_multiple_of(alignment)?;
            let kept = if keep {
                let kept_bytes = u64::from(texels).checked_mul(4 * u64::from(KEPT_WORDS))?;
                if kept_bytes > u64::from(max_range) {
                    return None;
                }
                let kept_at = at;
                at = at
                    .checked_add(kept_bytes)?
                    .checked_next_multiple_of(alignment)?;
                Some((kept_at, kept_bytes))
            } else {
                None
            };
            out.push(MipmapFilterLevel {
                offset,
                bytes,
                extent,
                params: Some((params_at, words)),
                kernel: Some(kernel),
                groups: [x, y],
                kept,
            });
        }
        let half_sums = layout.fields.iter().any(|f| {
            matches!(
                f.rule,
                Rule::Float16 | Rule::Ufloat { .. } | Rule::Rgb9e5 { .. }
            )
        });
        let below_first = levels >= 3;
        let known = m3_passes(&extents, compressed, &layout).is_some();
        let has = |rule: fn(Rule) -> bool| layout.fields.iter().any(|f| rule(f.rule));
        // A format the M3 Max carries through a half (macOS 26.7's) misses a few filtered values a
        // hair from a boundary, and a few tail values read from kept values: only an
        // extended-range sRGB one. The 10-bit grid is exact.
        let half_ties =
            rule == MipmapRule::AppleM3MaxPasses && layout.half && has(|r| r == Rule::XrSrgb);
        let filter_ties = half_ties || has(|r| r == Rule::Srgb8);
        let kept_ties = half_ties || has(|r| r == Rule::Float16);
        let near_ties = (1..placed.len()).any(|level| match step(level) {
            Step::Sampler { .. } => filter_ties,
            Step::Tail { kept } => kept && kept_ties,
        });
        let unlike_an_m3 = match rule {
            // Every level the filter's (the module docs).
            MipmapRule::Filter => false,
            MipmapRule::AppleM3MaxPasses if volume => half_sums,
            MipmapRule::AppleM3MaxPasses => {
                pixel_format == PixelFormat::A1BGR5Unorm
                    || (below_first && (pixel_format == PixelFormat::ABGR4Unorm || !known))
            }
        };
        Some(MipmapFilterChain {
            levels: out,
            bytes: at,
            pixel_format,
            layers,
            unlike_an_m3,
            near_ties,
            zeroed: layout.bytes < 4,
            volume,
            compressed,
            rule,
            fetch,
        })
    }

    /// The format of the sampled view through which `plan`'s `fetch` reads the first level of an
    /// image of Vulkan format `image`: the image's own, or the UNORM twin of an 8-bit sRGB format
    /// (the same bytes, its codes undecoded). `None` where no fetch recovers the texels.
    pub fn fetch_format(image: Format) -> Option<Format> {
        let view = crate::metal::texel_format::srgb_raw_view(image).unwrap_or(image);
        fetch::components(view).map(|_| view)
    }

    /// Whether a fetch of a sampled view of `format` reads the device's conversion of each texel
    /// (a normalized or 16-bit float format: `plan`'s `fetch` there needs a device measured to
    /// return values whose recovery reads every code back) rather than its bits (the
    /// extended-range formats' integer carriers, read in place on any device). `false` too for a
    /// format no fetch reads.
    pub fn fetch_converts(format: Format) -> bool {
        fetch::components(format).is_some_and(|components| !fetch::integer(&components))
    }

    /// The step of level `level` (after the first).
    fn step(&self, level: usize, layout: &Layout) -> Step {
        let extents: Vec<[u32; 3]> = self.levels.iter().map(|l| l.extent).collect();
        step_of(
            &extents,
            level,
            self.volume,
            self.compressed,
            layout,
            self.rule,
        )
    }

    /// Metal's levels of this chain on the CPU, from the bytes of the first level (every layer,
    /// as the scratch buffer holds them): what the copy in and the dispatches leave in the
    /// scratch buffer, every level in order, the first included. `None` for too few bytes.
    pub fn expected_levels(&self, first: &[u8]) -> Option<Vec<Vec<u8>>> {
        let layout = layout::layout(self.pixel_format)?;
        let bytes = usize::try_from(self.levels[0].bytes).ok()?;
        let mut out = vec![first.get(..bytes)?.to_vec()];
        let mut kept: Vec<tail::Kept> = Vec::new();
        for level in 1..self.levels.len() {
            let (big, small) = (self.levels[level - 1].extent, self.levels[level].extent);
            let above = &out[level - 1];
            let (next, keep) = match self.step(level, &layout) {
                Step::Sampler { reciprocal, half } => {
                    model::level(&layout, above, big, small, self.layers, reciprocal, half)
                }
                Step::Tail { kept: read } => tail::level(
                    &layout,
                    above,
                    read.then_some(&kept[..]),
                    big,
                    small,
                    self.layers,
                    layout.half,
                ),
            };
            out.push(next);
            kept = keep;
        }
        Some(out)
    }
}

/// The binding symbol of the level above in the typed module.
const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the first level's sampled view in the typed module of a kernel that
/// fetches it ([`MipmapFilterChain::fetch`]).
const SOURCE_IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the level written in the typed module.
const DESTINATION: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the params in the typed module.
const PARAMS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};
/// The binding symbol of the level above's kept values in the typed module.
const KEPT_IN: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 3,
};
/// The binding symbol of the level's kept values in the typed module.
const KEPT_OUT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 4,
};

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(MipmapFilterModule::SET),
        Symbol::Binding(s) if *s == SOURCE || *s == SOURCE_IMAGE => {
            Some(MipmapFilterModule::SOURCE_BINDING)
        }
        Symbol::Binding(s) if *s == DESTINATION => Some(MipmapFilterModule::DESTINATION_BINDING),
        Symbol::Binding(s) if *s == PARAMS => Some(MipmapFilterModule::PARAMS_BINDING),
        Symbol::Binding(s) if *s == KEPT_IN => Some(MipmapFilterModule::KEPT_IN_BINDING),
        Symbol::Binding(s) if *s == KEPT_OUT => Some(MipmapFilterModule::KEPT_OUT_BINDING),
        _ => None,
    }
}

/// The mipmap filter module of `kernel` on `device`: built with the typed builder, checked by
/// the owned verifier against the device's profile (requirements included), serialized, and
/// numbered with the fixed bindings of [`MipmapFilterModule`]. A device without `shaderInt64` is
/// `Unsupported{MipmapGeneration}` naming it.
pub fn mipmap_filter_module(
    device: &Device,
    kernel: MipmapFilterKernel,
) -> Result<MipmapFilterModule, Error> {
    let profile = device.profile();
    if !profile.fields().leaf.int64 {
        return Err(Unsupported::needs(
            Feature::MipmapGeneration,
            Capability::Feature(VkFeature::ShaderInt64),
        )
        .into());
    }
    let spec = kernel.spec().ok_or_else(|| {
        Error::Internal(InternalError::new(
            InternalKind::Invariant,
            format!("a mipmap filter kernel of {:?}", kernel.pixel_format),
        ))
    })?;
    let m = kernel::typed_module(&spec)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the mipmap filter needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(MipmapFilterModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}
