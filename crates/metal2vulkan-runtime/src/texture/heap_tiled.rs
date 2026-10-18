//! A heap texture over host pages whose elements lie in the heap's bytes as Metal's GPU lays
//! them out: tiled (the consumer's report: a texture placed in a heap over a guest's pages is not
//! linear on Metal).
//!
//! **Metal's layout** (an Apple M4 under macOS 27.0.1, our probes and the M3 probe's L rows: a
//! `Shared` placement heap over host pages, `setAddressRanges:`; each element of each slice, level
//! and depth slice written by `replaceRegion` with values that name it, the span read through the
//! pages; an M3 Max under macOS 26.7 alike on every row of plain formats, the consumer's run of the
//! probe):
//!
//! - **Tiles.** Elements of `B` bytes lie in tiles of at most 16 KiB: 128 x 128 elements of 1
//!   byte, 128 x 64 of 2, 64 x 64 of 4, 64 x 32 of 8, 32 x 32 of 16 (`largest_tile`). A level
//!   of `w x h` elements takes tiles of `min(TW, pot(w)) x min(TH, pot(h))`, `⌈w / tw⌉` a row;
//!   the tiles lie row after row, and a tile's elements in Morton order, x's bit lowest, the
//!   longer side's extra bits on top (`TiledLayout::element`).
//! - **Levels.** Each level lies at the end of the one before. A level both of whose sides are a
//!   full tile or more takes the tiles of level 0's grid (`gx = ⌈w0 / TW⌉`, `gy = ⌈h0 / TH⌉`)
//!   halved `l` times, `⌈gx / 2^l⌉ · ⌈gy / 2^l⌉` of them, plus the whole tiles in the area the
//!   halving's fractions leave (`⌊gx · gy / 4^l − ⌊gx / 2^l⌋ · ⌊gy / 2^l⌋⌋`). The first level
//!   narrower than a tile on either side takes its own sides padded: a side whose other side
//!   is narrower than a tile to a power of two, else to a multiple of the tile; each later
//!   level halves the first one's power-of-two sides. A narrow level's bytes are rounded up to
//!   a multiple of 128 (`level_bytes`).
//! - **Chains.** Each slice of the texture, and each depth slice of a 3D one, holds a chain of
//!   the levels. One chain of a one-level texture is its level; of a mipmapped one, the full
//!   chain's bytes, whatever levels the texture has. An array's, cube's and 3D texture's chains
//!   lie at multiples of 16 KiB from a chain of 16 KiB on, the texture then aligned to 16 KiB;
//!   any other texture is aligned to 128. `heapTextureSizeAndAlignWithDescriptor:` answers the
//!   chains' bytes and that alignment: every uncompressed answer measured (4126 of every plain
//!   format, sRGB, packed, extended-range, depth and stencil formats among them, every type; the
//!   bases of 476 more).
//! - **Block-compressed formats** (an Apple M4 under macOS 27.0.1: every BC, ETC2, EAC and ASTC
//!   format, 815 textures, every element of every slice, level and depth slice;
//!   `tests/heap_tiled_blocks.rs`) lay out a block as an element. Each level's allocation counts
//!   level 0's blocks halved per level, rounding up (`⌈b0 / 2^l⌉`: its full levels' tiles, its
//!   first narrow level's sides). Its tiles count its blocks by a placement of the format's family
//!   (`BlockPlacement`), `⌈e / tw⌉` tiles a row: `Halved` counts them so too (ASTC on the M4);
//!   `Narrow` counts them so while the level before is a tile or more on that side, then halves the
//!   first narrower level's texels (its blocks' whole texels), rounding down (BC, ETC2 and EAC on
//!   the M4: level 3 of a 100 x 36 `BC1_RGBA` chain, 3 x 1 blocks of its own and 4 x 2 halved, lies
//!   in tiles of 4 x 1, where `Halved` would take 4 x 2). An Apple M3 Max under macOS 26.7 (the
//!   same 815 textures, packages `bc-m3pkg` and `bc-m3pkg2` rerun there) answers every texture
//!   alike and places every family as `Halved` where its CPU copies (`replaceRegion`, `getBytes`)
//!   reach the bytes; where its GPU does (a blit from or into a buffer), it counts a level's tiles
//!   a row by the level's own blocks instead, `⌈b / tw⌉` (`GpuRows::Own`), so on 7 of the 815
//!   (levels whose own and halved blocks fill a different number of tiles a row: level 1 of a 513 x
//!   300 `BC1_RGBA` chain, 64 blocks of its own and 65 halved in tiles 64 wide) its GPU and its CPU
//!   copies meet the level's blocks at other bytes, and a CPU copy reads back other blocks than a
//!   blit wrote; the M4's GPU counts them as its CPU copies do (`GpuRows::AsCpu`). The release
//!   makes the difference: the M3 Max under 27.0.1 answers every texture of
//!   `tests/heap_tiled_blocks.rs` as the M4 does, its CPU's spans and its GPU's. The device's Metal
//!   level names each family's placement and its GPU's count (`MetalLevel::heap_blocks`,
//!   `MetalLevel::heap_block_gpu_rows`): the CPU copies place blocks by `TiledLevel::across`, the
//!   kernel's moves by `TiledLevel::gpu_across`, as Metal's own two paths do. A PVRTC texture stays
//!   untiled (`heap-texture-block-compressed-layout`), its answer its device's.
//! - **Lossless compression** (14174 answers of every plain format, type, shape, usage and storage
//!   mode). Metal keeps a texture losslessly compressed where `allowGPUOptimizedContents` holds and
//!   the shorter side of its level 0 is 16 texels or more (16 x 16 compressed; 16 x 8, 6 x 43 and
//!   2048 x 5 not), a colour texture of none of `ShaderWrite`, `PixelFormatView` and `ShaderAtomic`
//!   usage and a depth or stencil texture of any (`metal_compresses`, `metal_keeps_uncompressed`).
//!   Its bytes are no layout of its texels. It answers its chains uncompressed, then each chain's
//!   metadata (`lossless_answer`, `chain_metadata`): for each of its levels, 8 bytes for each 16 x
//!   16 tile of the level, its tiles on each side padded to a power of two, rounded up to 128. A
//!   one-level texture's metadata is its level's (8538 answers, every one). A mipmapped one's
//!   covers the full chain's levels, whatever levels the texture has, while level 0's sides, each
//!   padded to a multiple of 16 and halved per level, have a longer side of 16 or more, each
//!   level's tiles counted over those halved sides (an Apple M4 under macOS 27.0.1, and an M3 Max
//!   under 26.7, alike on every one of their 410356 answers: 2D textures of 12 colour formats and
//!   `Depth16Unorm`, `Depth32Float` and `Stencil8` at every pair of 30 sides from 1 to 1440,
//!   two-slice 2D arrays and cubes of 5 colour formats at 7 sizes, 1 to 10 levels, usages 0x1, 0x5,
//!   0x7 and 0x10005, both storage modes; a 224 x 928 chain's metadata covers its first 6 levels).
//!   Every chain takes it alike: a 3D texture's, one a depth slice of level 0, and a cube array's
//!   (114921 answers on the M4: 3D textures of 3 colour formats at 12 x 12 sides and 6 depths, 2D
//!   arrays of 1, 3 and 6 slices, cubes and cube arrays of 1 and 2 of 7 colour and 4 depth and
//!   stencil formats, every level count, `allowGPUOptimizedContents` off, on and unset for the
//!   depth and stencil ones, which compress at no usage with it off).
//!   It stays untiled (`heap-texture-lossless-layout`).
//! - **Multisample textures** (the consumer's ask; on an Apple M4 under macOS 27.0.1, and the same
//!   package run on an M3 Max under 26.7 (the coordinator's run), which answered every one of its
//!   27648 descriptors and laid out every byte of its 9 textures as the M4; 18 formats, 2 and 4
//!   samples (both answer no to 8), 1 to 3 slices, six usages, `Shared` and `Private`). Metal keeps
//!   a multisample texture losslessly compressed by the rule above over its **sample grid**
//!   (`sample_grid`: `w x 2h` at 2 samples, `2w x 2h` at 4): with `allowGPUOptimizedContents`, the
//!   grid's shorter side 16 or more, and a colour texture of neither `ShaderWrite` nor
//!   `PixelFormatView` usage or a depth or stencil texture of any; the storage mode decides nothing
//!   (every one of 10152 descriptors around the threshold, on both machines). Uncompressed, a
//!   pixel's samples lie side by side as one element of `S·B` bytes (sample `s` at `s·B`), laid out
//!   as a one-level texture's elements above (tiles of 32 x 16 elements of 32 bytes and 16 x 16 of
//!   64), a multisample array's slices as an array's: every sample of 21 textures drawn at the
//!   bytes the rule names (14 on the M4, nothing past their answers written; 7 on both machines).
//!   Compressed, its bytes are no layout of its samples, and its answer is that one's plus, for
//!   each slice, its sample grid's metadata as a one-level texture's above
//!   (`multisample_heap_layout`; every answer of every format but `Depth32Float_Stencil8` measured,
//!   93422 on the M4 and 24576 on the M3 Max, compressed and not, at 128 bytes or, an array's chain
//!   of 16 KiB or more, 16 KiB, a one-slice array's too). A `Depth32Float_Stencil8` texture answers
//!   two planes: its depth plane (`Depth32Float`'s answer, compressed or not) rounded up to 16 KiB,
//!   then its stencil plane (`Stencil8`'s), at the depth plane's alignment (every one of its 4800
//!   answers on the M4 and 3072 on the M3 Max, 1 to 3 slices; before, its device's answer, whose
//!   alignment on MoltenVK was 128 where Metal's is 16 KiB for an array of a 16-KiB depth chain,
//!   the consumer's report), and so does a one-sample one, of every type and level count,
//!   compressed or not (every one of 14552 answers each on the M4 and the M3 Max, 2D, and 12330 on
//!   the M4, 2D arrays, cubes and cube arrays). Over host pages a compressed one lies in memory of
//!   its own and is announced `heap-texture-lossless-layout`; an uncompressed one is refused
//!   (`RuntimeGap::HostHeapUncompressedMultisample`, `heap.rs`). In any
//!   heap one whose image the heap's memory or Metal's answer cannot hold lies in memory of its own
//!   too.
//! - **Not measured**: a two-aspect depth and stencil format's layout (`Depth32Float_Stencil8`'s
//!   answer is, above) and a PVRTC format's; the runtime holds them untiled
//!   (`heap-texture-untiled-layout`), a PVRTC texture at its device's answer.
//!
//! **Held** as the texture's image in memory of its own, a shadow of the heap's bytes
//! (`shadow.rs`, `ShadowTarget::Tiled`): into the image, the texture's bytes copied into a
//! staging buffer, one dispatch of `bind::tiled_texels_module` moving each level's elements into
//! a tight buffer of the image's elements (`heap_plane_layout`), and that buffer copied into
//! the image; back, the heap's bytes copied into the staging buffer again, the image into the
//! tight buffer, the kernel into the staging buffer (each element's bytes; every other byte as
//! the heap held it), and each level's bytes of each chain on into the heap. Each step is
//! ordered after the one before by a full barrier, and the buffers are the texture's for its
//! life. The texture's CPU copies move each element between the heap's bytes and the caller's
//! as laid out (`TiledLayout::element`); a command that writes the bytes two ways merges as
//! any shadow's (`shadow_merge.rs`), the shadow's texels laid out tiled for it
//! (`HeapTiledCopies::record_out`).

use super::super::buffer::Buffer;
use super::super::copy_plan::{block_bytes, block_extent, depth_bytes};
use super::super::degradation::Degradation;
use super::super::error::{internal, Error};
use super::super::heap::SizeAndAlign;
use super::super::image_plan::block_family;
use super::super::metal_level::MetalLevel;
use super::super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::super::row_copies::RowCopies;
use super::super::shadow::{self, ShadowStaging};
use super::super::texture_descriptor::TextureLayout;
use super::super::tracker::Barrier;
use super::super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::super::Device;
use super::copy_units;
use super::heap::HeapPlane;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{TiledLevelMove, TiledTexelsModule};
use metal2vulkan::metal::format::{byte_format, is_depth_stencil, srgb_raw_twin};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{Format, ImageAspectFlags};

/// The bytes of Metal's largest tile, whatever its elements' size.
const TILE_BYTES: u64 = 16 << 10;
/// The multiple an array's, cube's or 3D texture's chains lie at once one holds this many bytes,
/// and its alignment then.
const PAGE: u64 = 16 << 10;
/// The multiple a narrow level's bytes are rounded up to, and any other texture's alignment.
const LINE: u64 = 128;
/// Where a shape lies in the shape buffer, per direction: past its bytes, and a multiple of
/// every `minStorageBufferOffsetAlignment`.
const SHAPE_SLOT: u64 = 1024;
/// The side of the square tile of texels whose compression metadata takes `METADATA_BYTES`.
const METADATA_TILE: u64 = 16;
/// The bytes of compression metadata a `METADATA_TILE` square takes.
const METADATA_BYTES: u64 = 8;
/// The shortest side of a level 0 Metal compresses.
const COMPRESSED_SIDE: u64 = 16;

/// Metal's largest tile of elements of `unit` bytes, elements across and down; `None` for a size
/// Metal has no tile of. Elements of 32 and 64 bytes are a multisample texture's pixels.
pub(crate) fn largest_tile(unit: u64) -> Option<[u64; 2]> {
    Some(match unit {
        1 => [128, 128],
        2 => [128, 64],
        4 => [64, 64],
        8 => [64, 32],
        16 => [32, 32],
        32 => [32, 16],
        64 => [16, 16],
        _ => return None,
    })
}

/// The power of two `v` rounds up to (1 for 0).
fn pot(v: u64) -> u64 {
    v.max(1).next_power_of_two()
}

/// The levels of a full chain whose largest side is `side`.
fn full_levels(side: u64) -> u32 {
    u64::BITS - side.max(1).leading_zeros()
}

/// The bytes a level of `[w, h]` elements of `unit` bytes takes where it is narrower than the
/// `largest` tile on a side: each side padded, to a power of two where the other side is
/// narrower than the tile, else to a multiple of its tile's side; rounded up to 128.
fn narrow_bytes(unit: u64, largest: [u64; 2], [w, h]: [u64; 2]) -> u64 {
    let tile = [largest[0].min(pot(w)), largest[1].min(pot(h))];
    let width = if h < largest[1] {
        pot(w)
    } else {
        w.next_multiple_of(tile[0])
    };
    let height = if w < largest[0] {
        pot(h)
    } else {
        h.next_multiple_of(tile[1])
    };
    (width * height * unit).next_multiple_of(LINE)
}

/// The bytes Metal lays level `level` of a chain out in, whose level 0 is `[w0, h0]` elements of
/// `unit` bytes in tiles of at most `largest`, its levels' elements as `elements` takes them (the
/// module docs).
pub(crate) fn level_bytes(
    unit: u64,
    largest: [u64; 2],
    [w0, h0]: [u64; 2],
    level: u32,
    elements: Elements,
) -> u64 {
    let sides = |k: u32| elements.allocated([w0, h0], k);
    let narrow = |[w, h]: [u64; 2]| w < largest[0] || h < largest[1];
    let here = sides(level);
    if narrow(here) {
        let first = (0..=level).find(|&k| narrow(sides(k))).unwrap_or(level);
        if first == level {
            return narrow_bytes(unit, largest, here);
        }
        let [w, h] = sides(first);
        let halved = level - first;
        return narrow_bytes(
            unit,
            largest,
            [(pot(w) >> halved).max(1), (pot(h) >> halved).max(1)],
        );
    }
    let [gx, gy] = [w0.div_ceil(largest[0]), h0.div_ceil(largest[1])];
    let step = 1u64 << level;
    let whole = gx.div_ceil(step) * gy.div_ceil(step);
    let area = gx * gy - (((gx >> level) * (gy >> level)) << (2 * level));
    (whole + (area >> (2 * level))) * TILE_BYTES
}

/// What a texture's elements are, and how Metal sizes each level's for its tiles and its
/// allocation (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Elements {
    /// A plain format's texels: a level's own, `⌊w0 / 2^l⌋`, for both.
    Texels,
    /// A block-compressed format's blocks of `block` texels, placed as `placement` says, its
    /// GPU's tiles a row counted as `gpu_rows` says.
    Blocks {
        block: [u64; 2],
        placement: BlockPlacement,
        gpu_rows: GpuRows,
    },
}

/// How a level's Metal places a block-compressed texture's blocks in its levels' tiles (the
/// module docs). Each level is allocated alike under both, of level 0's blocks halved per level,
/// rounding up; each lies in tiles of its elements counted as here, `⌈e / tw⌉` tiles a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockPlacement {
    /// Level 0's blocks halved per level, rounding up: `⌈b0 / 2^l⌉`.
    Halved,
    /// Level 0's blocks halved, rounding up, while the level before is a tile or more on that
    /// side; from the first level narrower than a tile, that level's texels (its blocks' whole
    /// texels) halved per level, rounding down, in blocks.
    Narrow,
}

/// How a level's Metal counts a block-compressed level's tiles a row where its GPU reaches the
/// texture (its blits; the runtime's kernel moves, `HeapTiledCopies`), whose CPU copies
/// (`replaceRegion`, `getBytes`; `TiledLayout::element`) count them by the elements its
/// placement counts (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuRows {
    /// As its CPU copies count them (the M4 under 27.0.1).
    AsCpu,
    /// By the level's own blocks, `⌈b / tw⌉` (the M3 Max under 26.7): where that count differs
    /// from its CPU copies', its GPU and its CPU copies meet the level's blocks at other bytes.
    Own,
}

/// The families of block-compressed formats a level's Metal places alike (`MetalLevel`'s rows).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BlockFamily {
    /// BC1 to BC7, linear, sRGB, signed and unsigned.
    Bc,
    /// ETC2 and EAC, linear, sRGB and signed.
    Etc,
    /// ASTC of every block size, LDR, sRGB and HDR.
    Astc,
}

impl BlockFamily {
    /// The family of `pf`; `None` for a format of none (PVRTC's and every plain format).
    pub(crate) fn of(pf: PixelFormat) -> Option<BlockFamily> {
        use PixelFormat as P;
        Some(match block_family(pf) {
            P::BC1RGBA
            | P::BC2RGBA
            | P::BC3RGBA
            | P::BC4RUnorm
            | P::BC5RGUnorm
            | P::BC6HRGBUfloat
            | P::BC7RGBAUnorm => BlockFamily::Bc,
            P::ETC2RGB8 | P::ETC2RGB8A1 | P::EACR11Unorm | P::EACRG11Unorm | P::EACRGBA8 => {
                BlockFamily::Etc
            }
            P::ASTC4x4LDR
            | P::ASTC5x4LDR
            | P::ASTC5x5LDR
            | P::ASTC6x5LDR
            | P::ASTC6x6LDR
            | P::ASTC8x5LDR
            | P::ASTC8x6LDR
            | P::ASTC8x8LDR
            | P::ASTC10x5LDR
            | P::ASTC10x6LDR
            | P::ASTC10x8LDR
            | P::ASTC10x10LDR
            | P::ASTC12x10LDR
            | P::ASTC12x12LDR => BlockFamily::Astc,
            _ => return None,
        })
    }
}

impl Elements {
    /// The texels of an element, across and down.
    fn block(self) -> [u64; 2] {
        match self {
            Elements::Texels => [1, 1],
            Elements::Blocks { block, .. } => block.map(|side| side.max(1)),
        }
    }

    /// Level `level`'s elements as its allocation counts them, of level 0's `[w0, h0]`.
    fn allocated(self, [w0, h0]: [u64; 2], level: u32) -> [u64; 2] {
        let shift = level.min(63);
        match self {
            Elements::Texels => [(w0 >> shift).max(1), (h0 >> shift).max(1)],
            Elements::Blocks { .. } => [w0, h0].map(|side| halved(side, shift)),
        }
    }

    /// Level `level`'s elements as its tiles count them, of level 0's `[w0, h0]` in tiles of at
    /// most `largest`, its own `own`.
    fn tiled(self, [w0, h0]: [u64; 2], level: u32, largest: [u64; 2], own: [u64; 2]) -> [u64; 2] {
        let shift = level.min(63);
        match self {
            Elements::Texels => own,
            Elements::Blocks {
                placement: BlockPlacement::Halved,
                ..
            } => [w0, h0].map(|side| halved(side, shift)),
            Elements::Blocks {
                block,
                placement: BlockPlacement::Narrow,
                ..
            } => [0, 1]
                .map(|axis| narrowed([w0, h0][axis], shift, largest[axis], block[axis].max(1))),
        }
    }

    /// How its GPU counts a level's tiles a row: a plain format's texels as its CPU copies do.
    fn gpu_rows(self) -> GpuRows {
        match self {
            Elements::Texels => GpuRows::AsCpu,
            Elements::Blocks { gpu_rows, .. } => gpu_rows,
        }
    }
}

/// `blocks` halved `shift` times, rounding up.
fn halved(blocks: u64, shift: u32) -> u64 {
    blocks.max(1).div_ceil(1 << shift).max(1)
}

/// A side of `blocks` blocks of `block` texels, at level `shift`, as `BlockPlacement::Narrow`
/// counts it in tiles of `tile` on that side.
fn narrowed(blocks: u64, shift: u32, tile: u64, block: u64) -> u64 {
    let mut side = blocks.max(1);
    let mut k = 0;
    while k < shift && side >= tile {
        side = side.div_ceil(2);
        k += 1;
    }
    ((side * block) >> (shift - k)).max(1).div_ceil(block)
}

/// The Morton index of `(x, y)` in a tile of `2^tx x 2^ty` elements: x's bit lowest, the longer
/// side's extra bits on top.
fn morton(x: u64, y: u64, tx: u32, ty: u32) -> u64 {
    let (mut m, mut k) = (0, 0);
    for i in 0..tx.max(ty) {
        if i < tx {
            m |= ((x >> i) & 1) << k;
            k += 1;
        }
        if i < ty {
            m |= ((y >> i) & 1) << k;
            k += 1;
        }
    }
    m
}

/// One level of `TiledLayout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TiledLevel {
    /// Its elements across and down.
    pub(crate) extent: [u64; 2],
    /// Its depth slices (a 3D level's; 1 for any other).
    pub(crate) depth: u64,
    /// Its tile, elements across and down: powers of two.
    pub(crate) tile: [u64; 2],
    /// Its tiles a row, as Metal's CPU copies count them (`TiledLayout::element`).
    pub(crate) across: u64,
    /// Its tiles a row, as Metal's GPU counts them (`GpuRows`; the kernel's moves): `across`,
    /// but a block-compressed level's own blocks' where its level counts them so.
    pub(crate) gpu_across: u64,
    /// Its first byte from its chain's.
    pub(crate) base: u64,
    /// The bytes Metal lays it out in.
    pub(crate) bytes: u64,
}

/// Metal's layout of a heap texture's elements (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TiledLayout {
    /// The bytes of an element.
    pub(crate) unit: u64,
    /// The texture's levels.
    pub(crate) levels: Vec<TiledLevel>,
    /// Its chains: its slices (six a cube), or a 3D texture's depth slices.
    pub(crate) chains: u64,
    /// A chain's bytes: its level's of a one-level texture, else the full chain's.
    pub(crate) chain: u64,
    /// The levels in a chain's bytes: 1 of a one-level texture, else the full chain's.
    pub(crate) chain_levels: u32,
    /// A 3D texture's: level `l` lies in the chains of its depth slices alone.
    pub(crate) volume: bool,
    /// The bytes from one chain to the next.
    pub(crate) stride: u64,
    /// The bytes Metal answers.
    pub(crate) size: u64,
    /// The alignment Metal answers.
    pub(crate) align: u64,
}

impl TiledLayout {
    /// Metal's layout of a texture of `texture_type` whose level 0 is `extent` texels (its depth
    /// a 3D texture's) in elements of `unit` bytes as `elements` says (texels, or a
    /// block-compressed format's blocks), with `levels` levels and `slices` slices (six a cube);
    /// `None` for an element size Metal has no tile of. A level's elements are its own texels'
    /// blocks; its allocation and its tiles count them as `elements` says (the module docs).
    pub(crate) fn of(
        unit: u64,
        elements: Elements,
        texture_type: TextureType,
        extent: [u64; 3],
        levels: u32,
        slices: u64,
    ) -> Option<TiledLayout> {
        use TextureType as T;
        let largest = largest_tile(unit)?;
        let one_row = matches!(texture_type, T::Type1D | T::Type1DArray);
        let volume = texture_type == T::Type3D;
        let width = extent[0].max(1);
        let height = if one_row { 1 } else { extent[1].max(1) };
        let d0 = if volume { extent[2].max(1) } else { 1 };
        let [bw, bh] = elements.block();
        let [w0, h0] = [width.div_ceil(bw), height.div_ceil(bh)];
        let bytes = |level: u32| level_bytes(unit, largest, [w0, h0], level, elements);
        let mut tiled = Vec::new();
        let mut end = 0;
        for level in 0..levels {
            let shift = level.min(63);
            let [w, h] = [
                (width >> shift).max(1).div_ceil(bw),
                (height >> shift).max(1).div_ceil(bh),
            ];
            let [tw, th] = elements.tiled([w0, h0], level, largest, [w, h]);
            let tile = [largest[0].min(pot(tw)), largest[1].min(pot(th))];
            let taken = bytes(level);
            let across = tw.div_ceil(tile[0]);
            let gpu_across = match elements.gpu_rows() {
                GpuRows::AsCpu => across,
                GpuRows::Own => w.div_ceil(tile[0]),
            };
            tiled.push(TiledLevel {
                extent: [w, h],
                depth: (d0 >> shift).max(1),
                tile,
                across,
                gpu_across,
                base: end,
                bytes: taken,
            });
            end += taken;
        }
        let chain_levels = match levels {
            1 => 1,
            _ => full_levels(width.max(height).max(d0)),
        };
        let chain = match levels {
            1 => end,
            _ => (0..chain_levels).map(bytes).sum(),
        };
        let (stride, align) = match layered(texture_type) && chain >= PAGE {
            true => (chain.next_multiple_of(PAGE), PAGE),
            false => (chain, LINE),
        };
        let chains = if volume { d0 } else { slices.max(1) };
        Some(TiledLayout {
            unit,
            levels: tiled,
            chains,
            chain,
            chain_levels,
            volume,
            stride,
            size: stride * chains,
            align,
        })
    }

    /// The byte at which element `(x, y)` of level `level` of chain `chain` starts, from the
    /// texture's first byte, as Metal's CPU copies place it (`TiledLevel::across`).
    pub(crate) fn element(&self, chain: u64, level: usize, x: u64, y: u64) -> u64 {
        let l = &self.levels[level];
        let [tw, th] = l.tile;
        let tile = (y / th) * l.across + x / tw;
        let inside = morton(x % tw, y % th, tw.trailing_zeros(), th.trailing_zeros());
        chain * self.stride + l.base + (tile * tw * th + inside) * self.unit
    }
}

/// Whether Metal keeps a texture losslessly compressed (the module docs): with
/// `allowGPUOptimizedContents`, a level 0 whose shorter side (`[width, height]`, a 1D texture's
/// height 1; a multisample texture's sample grid, [`sample_grid`]) is 16 texels or more, and
/// `usage` (the effective usage: `Unknown` is every usage) Metal compresses
/// ([`metal_keeps_uncompressed`]).
pub(crate) fn metal_compresses(
    colour: bool,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    [width, height]: [u64; 2],
) -> bool {
    allow_gpu_optimized_contents
        && width.min(height) >= COMPRESSED_SIDE
        && !metal_keeps_uncompressed(colour, usage)
}

/// [`metal_compresses`] of a texture of `layout`: its level 0, a 1D texture's one row high, or a
/// multisample texture's sample grid ([`sample_grid`]; Metal keeps no other sample count's
/// texture compressed here).
pub(crate) fn metal_compresses_layout(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
) -> bool {
    let [width, height, _] = layout.extent.map(u64::from);
    let one_row = matches!(
        layout.texture_type,
        TextureType::Type1D | TextureType::Type1DArray
    );
    let level0 = [width, if one_row { 1 } else { height }];
    let shape = match layout.samples {
        1 => Some(level0),
        samples => sample_grid(samples, level0),
    };
    let colour = !is_depth_stencil(layout.pixel_format);
    shape.is_some_and(|shape| metal_compresses(colour, usage, allow_gpu_optimized_contents, shape))
}

/// Whether Metal keeps a texture of `usage` (the effective usage: `Unknown` is every usage)
/// uncompressed whatever its size and `allowGPUOptimizedContents`: a colour texture (`colour`)
/// of `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage; never a depth or stencil texture,
/// which Metal compresses at every usage (an Apple M4 under macOS 27.0.1: every
/// one of the 256 sets of `ShaderRead`, `ShaderWrite`, `RenderTarget`, 0x8, `PixelFormatView`,
/// `NoConcurrentAccess`, `Video` and `BlockWritesOnly` and 9 sets with `ShaderAtomic`, at 13
/// formats, 15 shapes, both storage modes and the property off, on and unset, each answer the
/// texture's compressed or its plain size as this rule says, 63210 rows; the consumer's M3 Max
/// rows under 26.7 alike). Of the private usages, `NoConcurrentAccess`, `Video` and
/// `BlockWritesOnly` keep nothing uncompressed, alone too.
pub(crate) fn metal_keeps_uncompressed(colour: bool, usage: TextureUsage) -> bool {
    let plain = TextureUsage::SHADER_WRITE
        .union(TextureUsage::PIXEL_FORMAT_VIEW)
        .union(TextureUsage::SHADER_ATOMIC);
    colour && usage.intersects(plain)
}

/// What a new `Private` texture Metal keeps losslessly compressed ([`metal_compresses`]) reads
/// before anything writes it, by pixel format: not zero, but one fixed texel in every texel of each
/// level whose shorter side is 16 texels or more ([`pattern_levels`]), and every sample (an Apple
/// M4 under macOS 27.0.1: every format listed, read by blits, kernels and loads, alike in every
/// read after a written texture or buffer was released; the consumer's M3 Max alike for
/// `RGBA8Unorm`). Each storage field of the texel is all ones but the second from its lowest bit,
/// which is zero (`RGBA8Unorm` reads opaque magenta, `A1BGR5Unorm` opaque yellow), but for
/// `RG11B10Float`'s canonical NaNs and the extended-range formats' own
/// codes; a NaN in a float or depth field. `None` for a format whose new texture reads zero however
/// it is made (block-compressed, 4:2:2 and two-plane formats) or that the probes did not make.
pub(crate) fn fresh_pattern(pf: PixelFormat) -> Option<FreshPattern> {
    use FreshPattern::{Color, DepthStencil};
    use PixelFormat as P;
    // A clear color whose conversion writes the texel exactly (`clear_value::texel_bytes`).
    const MAGENTA: Option<[f64; 4]> = Some([1.0, 0.0, 1.0, 1.0]);
    const YELLOW: Option<[f64; 4]> = Some([1.0, 1.0, 0.0, 1.0]);
    const ONES: Option<[f64; 4]> = Some([4294967295.0, 0.0, 4294967295.0, 4294967295.0]);
    const MINUS_ONE: Option<[f64; 4]> = Some([-1.0, 0.0, -1.0, -1.0]);
    const SNORM8: Option<[f64; 4]> = Some([-1.0 / 127.0, 0.0, -1.0 / 127.0, -1.0 / 127.0]);
    const SNORM16: Option<[f64; 4]> = Some([-1.0 / 32767.0, 0.0, -1.0 / 32767.0, -1.0 / 32767.0]);
    const SHARED: Option<[f64; 4]> = Some([65408.0, 0.0, 65408.0, 1.0]);
    let color = |texel: &'static [u8], color: Option<[f64; 4]>| Some(Color { texel, color });
    match pf {
        P::A8Unorm | P::R8Unorm | P::R8UnormSRGB => color(&[0xff], MAGENTA),
        P::R8Snorm => color(&[0xff], SNORM8),
        P::R8Uint => color(&[0xff], ONES),
        P::R8Sint => color(&[0xff], MINUS_ONE),
        P::RG8Unorm | P::RG8UnormSRGB => color(&[0xff, 0x00], MAGENTA),
        P::RG8Snorm => color(&[0xff, 0x00], SNORM8),
        P::RG8Uint => color(&[0xff, 0x00], ONES),
        P::RG8Sint => color(&[0xff, 0x00], MINUS_ONE),
        P::R16Unorm => color(&[0xff, 0xff], MAGENTA),
        P::R16Snorm => color(&[0xff, 0xff], SNORM16),
        P::R16Uint => color(&[0xff, 0xff], ONES),
        P::R16Sint => color(&[0xff, 0xff], MINUS_ONE),
        P::R16Float => color(&[0xff, 0xff], None),
        P::B5G6R5Unorm => color(&[0x1f, 0xf8], MAGENTA),
        P::A1BGR5Unorm => color(&[0xc1, 0xff], YELLOW),
        P::ABGR4Unorm => color(&[0x0f, 0xff], YELLOW),
        P::BGR5A1Unorm => color(&[0x1f, 0xfc], MAGENTA),
        P::RGBA8Unorm | P::RGBA8UnormSRGB | P::BGRA8Unorm | P::BGRA8UnormSRGB => {
            color(&[0xff, 0x00, 0xff, 0xff], MAGENTA)
        }
        P::RGBA8Snorm => color(&[0xff, 0x00, 0xff, 0xff], SNORM8),
        P::RGBA8Uint => color(&[0xff, 0x00, 0xff, 0xff], ONES),
        P::RGBA8Sint => color(&[0xff, 0x00, 0xff, 0xff], MINUS_ONE),
        P::RGB10A2Unorm | P::BGR10A2Unorm => color(&[0xff, 0x03, 0xf0, 0xff], MAGENTA),
        P::RGB10A2Uint => color(&[0xff, 0x03, 0xf0, 0xff], ONES),
        P::BGR10XR | P::BGR10XRSRGB => color(&[0xff, 0x03, 0xf0, 0xff], None),
        P::RG11B10Float => color(&[0xe0, 0x07, 0x00, 0xfc], None),
        P::RGB9E5Float => color(&[0xff, 0x01, 0xfc, 0xff], SHARED),
        P::RG16Unorm => color(&[0xff, 0xff, 0x00, 0x00], MAGENTA),
        P::RG16Snorm => color(&[0xff, 0xff, 0x00, 0x00], SNORM16),
        P::RG16Uint => color(&[0xff, 0xff, 0x00, 0x00], ONES),
        P::RG16Sint => color(&[0xff, 0xff, 0x00, 0x00], MINUS_ONE),
        P::RG16Float => color(&[0xff, 0xff, 0x00, 0x00], None),
        P::R32Uint => color(&[0xff; 4], ONES),
        P::R32Sint => color(&[0xff; 4], MINUS_ONE),
        P::R32Float => color(&[0xff; 4], None),
        P::RGBA16Unorm => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], MAGENTA),
        P::RGBA16Snorm => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], SNORM16),
        P::RGBA16Uint => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], ONES),
        P::RGBA16Sint => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], MINUS_ONE),
        P::RGBA16Float => color(&[0xff, 0xff, 0, 0, 0xff, 0xff, 0xff, 0xff], None),
        P::RG32Uint => color(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0], ONES),
        P::RG32Sint => color(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0], MINUS_ONE),
        P::RG32Float => color(&[0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0], None),
        P::BGRA10XR | P::BGRA10XRSRGB => color(&[0xc0, 0xff, 0, 0, 0xc0, 0xff, 0x80, 0xdf], None),
        P::RGBA32Uint => color(&RGBA32, ONES),
        P::RGBA32Sint => color(&RGBA32, MINUS_ONE),
        P::RGBA32Float => color(&RGBA32, None),
        P::Depth16Unorm => Some(DepthStencil {
            depth: Some(FreshDepth::Value(1.0)),
            stencil: None,
        }),
        P::Depth32Float => Some(DepthStencil {
            depth: Some(FreshDepth::Nan),
            stencil: None,
        }),
        P::Stencil8 => Some(DepthStencil {
            depth: None,
            stencil: Some(0xff),
        }),
        P::Depth32FloatStencil8 => Some(DepthStencil {
            depth: Some(FreshDepth::Nan),
            stencil: Some(0xff),
        }),
        _ => None,
    }
}

/// The texel of [`fresh_pattern`]'s 128-bit formats: all ones but the second 32-bit field.
const RGBA32: [u8; 16] = [
    0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
];

/// What every texel of a level [`fresh_pattern`] covers reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FreshPattern {
    /// A color format's texel, as Metal's blits read it, and the clear color (Metal's) whose
    /// conversion writes exactly that texel, where one does; `None` where a field is a NaN,
    /// whose bits Vulkan's clears leave to the device (`clear_value.rs`).
    Color {
        texel: &'static [u8],
        color: Option<[f64; 4]>,
    },
    /// A depth or stencil format's depth and stencil, each where the format has the aspect.
    DepthStencil {
        depth: Option<FreshDepth>,
        stencil: Option<u32>,
    },
}

/// The depth [`FreshPattern::DepthStencil`] reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FreshDepth {
    /// A depth a clear writes exactly.
    Value(f64),
    /// A NaN (all ones), which no Vulkan clear or copy writes into a depth image: a clear's depth
    /// lies in `[0, 1]`, and a copy's depth outside it is undefined.
    Nan,
}

/// The levels of a new `Private` texture whose texels read [`fresh_pattern`]: `0..n`, each level
/// whose shorter side (of `[width, height]` at level 0, a 1D texture's height 1; a multisample
/// texture's sample grid, [`sample_grid`]) is 16 texels or more, of a texture Metal compresses
/// ([`metal_compresses`]: a 64 x 64 chain's levels 64, 32 and 16, and none of
/// its levels under 16). 0 where Metal keeps it uncompressed: it reads zero throughout.
pub(crate) fn pattern_levels(
    colour: bool,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    [width, height]: [u64; 2],
    levels: u32,
) -> u32 {
    if !metal_compresses(colour, usage, allow_gpu_optimized_contents, [width, height]) {
        return 0;
    }
    (0..levels)
        .take_while(|&level| {
            let side = |axis: u64| (axis >> level).max(1);
            side(width).min(side(height)) >= COMPRESSED_SIDE
        })
        .count() as u32
}

/// Metal's answer for a texture it keeps losslessly compressed, whose texels uncompressed are
/// laid out as `layout`: its chains, then each chain's metadata over its level 0
/// ([`chain_metadata`], its full chain's levels where it is mipmapped), at the chains'
/// alignment. `None` for a layout of no level.
pub(crate) fn lossless_answer(layout: &TiledLayout) -> Option<SizeAndAlign> {
    let level = layout.levels.first()?;
    let mipmapped = layout.levels.len() > 1;
    Some(with_metadata(
        layout,
        chain_metadata(level.extent, mipmapped),
    ))
}

/// Metal's answer for a `Depth32Float_Stencil8` texture whose depth plane answers `depth`
/// (`Depth32Float`'s answer for the descriptor) and stencil plane `stencil` (`Stencil8`'s): the
/// depth plane rounded up to 16 KiB, then the stencil plane, at the depth plane's alignment
/// (the module docs).
fn depth_stencil_planes(depth: SizeAndAlign, stencil: SizeAndAlign) -> SizeAndAlign {
    SizeAndAlign {
        size: depth.size.next_multiple_of(PAGE) + stencil.size,
        align: depth.align,
    }
}

/// The answer of a texture laid out uncompressed as `layout`, kept compressed with `metadata`
/// bytes for each chain.
fn with_metadata(layout: &TiledLayout, metadata: u64) -> SizeAndAlign {
    SizeAndAlign {
        size: layout.size + layout.chains * metadata,
        align: layout.align,
    }
}

/// The compression metadata of a chain whose level 0 is `[width, height]` texels (a multisample
/// texture's sample grid, [`sample_grid`]), its full chain's levels where it is `mipmapped` (the
/// module docs): for each level, 8 bytes for each 16 x 16 tile of it, its tiles on each side
/// padded to a power of two, rounded up to 128; level 0's alone, or each level of the full
/// chain while level 0's sides, each padded to a multiple of 16 and halved per level, have a
/// longer side of 16 or more, its tiles counted over those halved sides.
fn chain_metadata([width, height]: [u64; 2], mipmapped: bool) -> u64 {
    let padded = [width, height].map(|side| side.max(1).next_multiple_of(METADATA_TILE));
    let longer = padded[0].max(padded[1]);
    (0..u64::BITS)
        .take_while(|&level| level == 0 || (mipmapped && longer >> level >= METADATA_TILE))
        .map(|level| {
            let [tw, th] = padded.map(|side| pot((side >> level).max(1).div_ceil(METADATA_TILE)));
            (tw * th * METADATA_BYTES).next_multiple_of(LINE)
        })
        .sum()
}

/// The sample grid of a multisample texture of `samples` samples whose level is `[width,
/// height]` pixels, as Metal's compression counts it (the module docs): `width x 2·height` at 2
/// samples, `2·width x 2·height` at 4; `None` at any other count, which Apple's GPUs do not make
/// (an M4 and an M3 Max answer no to 8), its layout not measured.
pub(crate) fn sample_grid(samples: u32, [width, height]: [u64; 2]) -> Option<[u64; 2]> {
    match samples {
        2 => Some([width, 2 * height]),
        4 => Some([2 * width, 2 * height]),
        _ => None,
    }
}

/// Whether a texture of `texture_type` lays its chains out at multiples of 16 KiB from a chain
/// of 16 KiB on.
fn layered(texture_type: TextureType) -> bool {
    use TextureType as T;
    matches!(
        texture_type,
        T::Type1DArray
            | T::Type2DArray
            | T::Cube
            | T::CubeArray
            | T::Type3D
            | T::Type2DMultisampleArray
    )
}

/// Why a heap texture over host pages is not laid out as Metal lays it, and the kind its
/// per-texture notice names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Untiled {
    pub(crate) kind: Degradation,
    pub(crate) reason: &'static str,
}

impl Untiled {
    /// A texture Metal may keep losslessly compressed (`heap-texture-lossless-layout`).
    const LOSSLESS: Untiled = Untiled {
        kind: Degradation::HeapTextureLosslessLayout,
        reason: "Metal may keep it losslessly compressed (allowGPUOptimizedContents, 256 texels \
                 or more, and none of ShaderWrite, PixelFormatView and ShaderAtomic usage, or a \
                 depth or stencil format), whose bytes are no layout of its texels",
    };
    /// A multisample texture Metal keeps losslessly compressed (`heap-texture-lossless-layout`),
    /// which a heap over host pages holds in memory of its own (`heap.rs`).
    pub(crate) const LOSSLESS_MULTISAMPLE: Untiled = Untiled {
        kind: Degradation::HeapTextureLosslessLayout,
        reason: "Metal keeps the multisample texture losslessly compressed \
                 (allowGPUOptimizedContents, a sample grid of 16 or more on its shorter side, \
                 and none of ShaderWrite, PixelFormatView and ShaderAtomic usage, or a depth or \
                 stencil format), whose bytes in the heap are no layout of its samples: its \
                 image lies in memory of its own, and the heap's bytes keep what they held",
    };
    /// A block-compressed texture whose placement the device's Metal level does not measure
    /// (`heap-texture-block-compressed-layout`).
    const BLOCK_COMPRESSED: Untiled = Untiled {
        kind: Degradation::HeapTextureBlockCompressedLayout,
        reason: "its format is block-compressed, whose blocks Metal places in each level by a \
                 rule that differs by format family and by machine, not measured for its \
                 family at the device's Metal level",
    };

    /// A texture the runtime holds otherwise than tiled for `reason`
    /// (`heap-texture-untiled-layout`).
    pub(crate) const fn other(reason: &'static str) -> Untiled {
        Untiled {
            kind: Degradation::HeapTextureUntiledLayout,
            reason,
        }
    }
}

/// The reason of a texture whose format's layout in Metal's heaps is not measured.
pub(crate) const UNMEASURED_FORMAT: &str = "its format's layout in Metal's heaps is not measured";

/// The reason of a multisample texture Metal keeps uncompressed, or whose layout is not
/// measured, which no heap over host pages holds (`heap.rs`), and which lies in memory of its
/// own where a heap's memory cannot hold its image.
const MULTISAMPLE_SAMPLES: &str = "Metal lays the multisample texture's samples out in the heap's \
                                   bytes, which no route of the runtime's holds yet, or their \
                                   layout is not measured";

/// What the runtime holds of Metal's layout of a heap texture's bytes (the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HeapLayout {
    /// Its elements as Metal lays them: over host pages a shadow of them
    /// (`HeapTiledCopies`).
    Tiled(TiledLayout),
    /// Not laid out as Metal's: over host pages, announced once per texture.
    Untiled(Untiled),
    /// Laid out as a measured route of its own lays it (a 4:2:2 texture's rows, a two-plane
    /// texture's planes; `heap.rs`).
    Measured,
}

/// Metal's answer for a heap texture where the runtime knows it, and the runtime's layout of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetalHeapLayout {
    pub(crate) answer: Option<SizeAndAlign>,
    pub(crate) layout: HeapLayout,
}

/// Metal's element bytes and texels (across and down) of a format `pf` whose image of `format`
/// holds Metal's bytes of each element in one copy unit, as its CPU copies and blits move them
/// (a colour image of its byte format or its sRGB format's raw twin's, a texel or a
/// block-compressed format's block a unit; a depth image of Metal's depth bytes and no stencil;
/// a stencil aspect's byte); `None` for any other.
/// Whether an image of `format` holding a `pf` texture holds each texel as Metal's `bytes`
/// bytes ([`image_element`]): a copy of Metal's texel's bytes writes that texel.
pub(crate) fn holds_metal_texels(pf: PixelFormat, format: Format, bytes: usize) -> bool {
    image_element(pf, format) == Some((bytes as u64, [1, 1]))
}

fn image_element(pf: PixelFormat, format: Format) -> Option<(u64, [u64; 2])> {
    let units = copy_units(format, pf);
    let element = match pf {
        PixelFormat::Stencil8 => units
            .iter()
            .find(|(aspect, _, _)| *aspect == ImageAspectFlags::STENCIL)
            .filter(|_| units.len() == 1)
            .map(|&(_, bytes, block)| (bytes, block)),
        PixelFormat::Depth16Unorm | PixelFormat::Depth32Float => match units[..] {
            [(ImageAspectFlags::DEPTH, bytes, block)] => {
                (depth_bytes(format) == Some(bytes)).then_some((bytes, block))
            }
            _ => None,
        },
        _ => {
            let bytes_of = |pf: PixelFormat| byte_format(pf).filter(|&own| own == format);
            let same = bytes_of(pf).or_else(|| srgb_raw_twin(pf).and_then(bytes_of));
            match units[..] {
                [(ImageAspectFlags::COLOR, bytes, block)]
                    if same.is_some() && block_bytes(pf) == Some(bytes) =>
                {
                    Some((bytes, block))
                }
                _ => None,
            }
        }
    };
    element.map(|(bytes, block)| (u64::from(bytes), block.map(u64::from)))
}

/// Metal's element bytes of the plain format `pf`, where its layout is measured (the module
/// docs): a colour format's texel, `Depth16Unorm`'s 2, `Depth32Float`'s 4 and `Stencil8`'s 1;
/// `None` for a two-aspect format and any other.
fn metal_element(pf: PixelFormat) -> Option<u64> {
    match pf {
        PixelFormat::Depth16Unorm => Some(2),
        PixelFormat::Depth32Float => Some(4),
        PixelFormat::Stencil8 => Some(1),
        pf if is_depth_stencil(pf) => None,
        pf => block_bytes(pf).map(u64::from),
    }
    .filter(|&unit| largest_tile(unit).is_some())
}

/// Metal's answer for a heap texture of `layout` at the Metal level `level` (the module docs),
/// `usage` and `allow_gpu_optimized_contents` the descriptor's: [`metal_heap_layout`]'s answer,
/// which no image format decides (the format decides only how a heap over host pages holds the
/// texture, so none is given). `None` where the model has none.
pub(crate) fn modelled_size(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    level: MetalLevel,
) -> Option<SizeAndAlign> {
    let no_image = Format::UNDEFINED;
    metal_heap_layout(layout, usage, allow_gpu_optimized_contents, no_image, level).answer
}

/// Metal's answer for a heap texture of `layout` and the runtime's layout of it over host pages
/// (the module docs), its image of `format`, at the device's Metal level `level`: `usage` and
/// `allow_gpu_optimized_contents` the descriptor's, which decide whether Metal may compress it.
pub(crate) fn metal_heap_layout(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
    format: Format,
    level: MetalLevel,
) -> MetalHeapLayout {
    let pf = layout.pixel_format;
    let [width, height, depth] = layout.extent.map(u64::from);
    let slices = u64::from(layout.layers);
    let untiled = |answer, why: Untiled| MetalHeapLayout {
        answer,
        layout: HeapLayout::Untiled(why),
    };
    // Block-compressed formats: Metal's allocations are its blocks' (every BC, ETC2, EAC and ASTC
    // format), alike under both placements, so a placement the level does not measure answers
    // `Halved`'s; PVRTC's are not measured.
    if let Some([bw, bh]) = block_extent(pf) {
        let block = [u64::from(bw), u64::from(bh)];
        let family = BlockFamily::of(pf);
        let placement = family.and_then(|family| level.heap_blocks(family));
        let elements = Elements::Blocks {
            block,
            placement: placement.unwrap_or(BlockPlacement::Halved),
            gpu_rows: level.heap_block_gpu_rows(),
        };
        let unit = block_bytes(pf).map(u64::from);
        let tiled = unit.filter(|_| family.is_some()).and_then(|unit| {
            let ty = layout.texture_type;
            let extent = [width, height, depth];
            TiledLayout::of(unit, elements, ty, extent, layout.levels, slices)
        });
        let answer = tiled.as_ref().map(|tiled| SizeAndAlign {
            size: tiled.size,
            align: tiled.align,
        });
        let (Some(tiled), Some(unit), Some(_)) = (tiled, unit, placement) else {
            return untiled(answer, Untiled::BLOCK_COMPRESSED);
        };
        if image_element(pf, format) != Some((unit, block)) {
            return untiled(
                answer,
                Untiled::other("its image holds other bytes than Metal's blocks"),
            );
        }
        return MetalHeapLayout {
            answer,
            layout: HeapLayout::Tiled(tiled),
        };
    }
    if layout.samples > 1 {
        return multisample_heap_layout(layout, usage, allow_gpu_optimized_contents);
    }
    let texture_type = layout.texture_type;
    let one_row = matches!(texture_type, TextureType::Type1D | TextureType::Type1DArray);
    let level0 = [width, if one_row { 1 } else { height }];
    let colour = !is_depth_stencil(pf);
    let compressed = metal_compresses(colour, usage, allow_gpu_optimized_contents, level0);
    // `Depth32Float_Stencil8`: its two planes, `Depth32Float`'s and `Stencil8`'s answers for the
    // descriptor, the depth plane rounded up to 16 KiB; their layout is not measured, so a heap
    // over host pages holds it untiled as before.
    if pf == PixelFormat::Depth32FloatStencil8 {
        let plane = |pixel_format| {
            let one = TextureLayout {
                pixel_format,
                ..*layout
            };
            metal_heap_layout(&one, usage, allow_gpu_optimized_contents, format, level).answer
        };
        let answer = plane(PixelFormat::Depth32Float)
            .zip(plane(PixelFormat::Stencil8))
            .map(|(depth, stencil)| depth_stencil_planes(depth, stencil));
        return untiled(answer, Untiled::other(UNMEASURED_FORMAT));
    }
    let Some(unit) = metal_element(pf) else {
        return untiled(None, Untiled::other(UNMEASURED_FORMAT));
    };
    let extent = [width, height, depth];
    let tiled = TiledLayout::of(
        unit,
        Elements::Texels,
        texture_type,
        extent,
        layout.levels,
        slices,
    );
    if compressed {
        return untiled(tiled.as_ref().and_then(lossless_answer), Untiled::LOSSLESS);
    }
    let Some(tiled) = tiled else {
        return untiled(None, Untiled::other(UNMEASURED_FORMAT));
    };
    let plain = Some(SizeAndAlign {
        size: tiled.size,
        align: tiled.align,
    });
    if image_element(pf, format) != Some((unit, [1, 1])) {
        return untiled(
            plain,
            Untiled::other("its image holds other bytes than Metal's elements"),
        );
    }
    MetalHeapLayout {
        answer: plain,
        layout: HeapLayout::Tiled(tiled),
    }
}

/// Metal's answer for a multisample heap texture of `layout` (the module docs), and the runtime's
/// layout of it over host pages: `usage` and `allow_gpu_optimized_contents` the descriptor's.
/// Uncompressed, its pixels lie as a one-level texture's elements of all their samples (`S·B`
/// bytes, sample `s` at `s·B`), a multisample array's slices as an array's; compressed, that
/// answer and metadata over its sample grid for each slice. `Depth32Float_Stencil8` answers its
/// two planes: the depth plane (`Depth32Float`'s answer) rounded up to 16 KiB, then the stencil
/// plane (`Stencil8`'s), at the depth plane's alignment. A sample count Apple's GPUs do not make
/// is not measured. Metal's compression decides how a heap over host pages holds it
/// ([`Untiled::LOSSLESS_MULTISAMPLE`], in memory of its own; any other is refused, `heap.rs`).
fn multisample_heap_layout(
    layout: &TextureLayout,
    usage: TextureUsage,
    allow_gpu_optimized_contents: bool,
) -> MetalHeapLayout {
    let pf = layout.pixel_format;
    let [width, height, _] = layout.extent.map(u64::from);
    let samples = layout.samples;
    let refused = |answer| MetalHeapLayout {
        answer,
        layout: HeapLayout::Untiled(Untiled::other(MULTISAMPLE_SAMPLES)),
    };
    let Some(grid) = sample_grid(samples, [width, height]) else {
        return refused(None);
    };
    let colour = !is_depth_stencil(pf);
    let compressed = metal_compresses(colour, usage, allow_gpu_optimized_contents, grid);
    // One plane of elements of `bytes` bytes a sample: its layout's answer, and its metadata
    // where Metal compresses it.
    let plane = |bytes: u64| {
        let unit = u64::from(samples) * bytes;
        let ty = layout.texture_type;
        let slices = u64::from(layout.layers);
        let tiled = TiledLayout::of(unit, Elements::Texels, ty, [width, height, 1], 1, slices)?;
        Some(match compressed {
            true => with_metadata(&tiled, chain_metadata(grid, false)),
            false => SizeAndAlign {
                size: tiled.size,
                align: tiled.align,
            },
        })
    };
    // `Depth32Float_Stencil8`: its depth plane (`Depth32Float`'s answer) rounded up to 16 KiB,
    // then its stencil plane (`Stencil8`'s), at the depth plane's alignment.
    let answer = match pf {
        PixelFormat::Depth32FloatStencil8 => plane(4)
            .zip(plane(1))
            .map(|(depth, stencil)| depth_stencil_planes(depth, stencil)),
        _ => metal_element(pf).and_then(plane),
    };
    if compressed {
        return MetalHeapLayout {
            answer,
            layout: HeapLayout::Untiled(Untiled::LOSSLESS_MULTISAMPLE),
        };
    }
    refused(answer)
}

/// One direction of the kernel's moves: its shape, where it lies in the shape buffer, and the
/// dispatch's workgroups.
struct Move {
    shape: Vec<u8>,
    at: u64,
    groups: [u32; 2],
}

/// A heap texture's moves between its image and its elements tiled in the heap's bytes (the
/// module docs): the tight buffer of the image's elements, the shapes, and the kernel's two sets
/// (into the image, then back).
pub(crate) struct HeapTiledCopies {
    /// Metal's layout of the texture's elements.
    pub(crate) layout: TiledLayout,
    /// The image's elements, each level's from its plane's offset; held for the texture's life.
    /// No command's uses name it: every copy writes the image or the heap's bytes, so the
    /// tracker orders it after every other copy of the texture, and each records a full barrier
    /// first.
    _tight: Buffer,
    tight_raw: vk::Buffer,
    /// The tight buffer's copies to and from the image, one a level.
    regions: Vec<vk::BufferImageCopy>,
    /// The two shapes; held for the texture's life.
    _shapes: Buffer,
    shapes_raw: vk::Buffer,
    moves: [Move; 2],
    kernel: OwnedKernel,
    sets: OwnedSets,
}

impl HeapTiledCopies {
    /// The kernel's levels of a texture of `layout` whose image's elements lie in the tight
    /// buffer as `planes` say (`heap_plane_layout`, one plane a level); `None` where a byte the
    /// moves name lies past `u32`, or the planes are not the levels'.
    fn levels(layout: &TiledLayout, planes: &[HeapPlane]) -> Option<Vec<TiledLevelMove>> {
        if planes.len() != layout.levels.len() {
            return None;
        }
        let narrow = |v: u64| u32::try_from(v).ok();
        layout
            .levels
            .iter()
            .zip(planes)
            .map(|(level, plane)| {
                let images = plane.images;
                if plane.blocks != level.extent || plane.unit != layout.unit {
                    return None;
                }
                Some(TiledLevelMove::new(
                    narrow(layout.unit)?,
                    [narrow(level.extent[0])?, narrow(level.extent[1])?],
                    narrow(images)?,
                    [narrow(level.tile[0])?, narrow(level.tile[1])?],
                    narrow(level.gpu_across)?,
                    narrow(level.base)?,
                    narrow(layout.stride)?,
                    narrow(plane.offset)?,
                    narrow(level.bytes)?,
                ))
            })
            .collect()
    }

    /// The workgroups of a dispatch of `invocations` within `device`'s counts, or `None`.
    fn grid(device: &Device, invocations: u64) -> Option<[u32; 2]> {
        let limits = &device.properties().description.limits;
        let local = u64::from(TiledTexelsModule::LOCAL_SIZE);
        let groups = invocations.div_ceil(local).max(1);
        let across = groups.min(u64::from(limits.max_compute_work_group_count[0]));
        let down = groups.div_ceil(across.max(1));
        let fits = down <= u64::from(limits.max_compute_work_group_count[1]);
        fits.then_some([across as u32, down as u32])
    }

    /// Whether the moves of a texture of `layout`, its tight buffer laid out as `planes` and
    /// `tight_len` bytes long, fit `device`: every byte they name within `u32`, at most
    /// `TiledTexelsModule::MAX_LEVELS` levels, each buffer within `maxStorageBufferRange`, and
    /// each dispatch within `maxComputeWorkGroupCount`.
    pub(crate) fn fit(
        device: &Device,
        layout: &TiledLayout,
        planes: &[HeapPlane],
        tight_len: u64,
    ) -> bool {
        let range = u64::from(
            device
                .properties()
                .description
                .limits
                .max_storage_buffer_range,
        );
        let Some(levels) = Self::levels(layout, planes) else {
            return false;
        };
        let moves = [true, false].iter().all(|&into| {
            let invocations = levels
                .iter()
                .map(|level| u64::from(level.invocations(into)))
                .sum::<u64>();
            u32::try_from(invocations).is_ok() && Self::grid(device, invocations).is_some()
        });
        levels.len() <= TiledTexelsModule::MAX_LEVELS as usize
            && layout.size <= range
            && u32::try_from(layout.size).is_ok()
            && tight_len.next_multiple_of(4) <= range
            && u32::try_from(tight_len).is_ok()
            && moves
    }

    /// The staging buffer of a texture of `layout` whose first byte lies at `at` of the heap's
    /// `VkBuffer`: the texture's bytes, from its first to its last, at its start, of the
    /// texture's `hazard` tracking; each level's bytes of each chain copied back.
    pub(crate) fn staging(
        device: &Device,
        layout: &TiledLayout,
        at: u64,
        hazard: HazardTrackingMode,
    ) -> Result<ShadowStaging, Error> {
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
        let buffer = device.own_buffer(layout.size, options)?;
        let (_, start) = buffer.at(0);
        let back = (0..layout.chains)
            .flat_map(|chain| {
                layout.levels.iter().filter_map(move |level| {
                    // A 3D level lies in the chains of its depth slices alone.
                    let used = !layout.volume || chain < level.depth;
                    let first = chain * layout.stride + level.base;
                    used.then_some(vk::BufferCopy {
                        src_offset: start + first,
                        dst_offset: at + first,
                        size: level.bytes,
                    })
                })
            })
            .collect();
        Ok(ShadowStaging {
            into: vk::BufferCopy {
                src_offset: at,
                dst_offset: start,
                size: layout.size,
            },
            back,
            buffer,
        })
    }

    /// The moves of a texture of `layout` through `staging` (its `staging`),
    /// whose image's elements lie in its tight buffer as `planes` say, `tight_len` bytes, copied
    /// to and from the image by `regions` (offsets in that buffer), of the texture's `hazard`
    /// tracking. Refused as `new_buffer` refuses the tight and shape buffers; the caller checked
    /// that they `fit`.
    pub(crate) fn new(
        device: &Device,
        staging: &ShadowStaging,
        layout: TiledLayout,
        (planes, tight_len): (&[HeapPlane], u64),
        regions: &[vk::BufferImageCopy],
        hazard: HazardTrackingMode,
    ) -> Result<HeapTiledCopies, Error> {
        let levels = Self::levels(&layout, planes)
            .ok_or_else(|| internal("a tiled heap texture's moves past u32"))?;
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, hazard);
        let tight_bytes = tight_len.next_multiple_of(4);
        let tight = device.own_buffer(tight_bytes, options)?;
        let shapes = device.own_buffer(2 * SHAPE_SLOT, options)?;
        let (tight_raw, tight_at) = tight.at(0);
        let (shapes_raw, shapes_at) = shapes.at(0);
        let (staging_raw, staging_at) = (staging.raw(), staging.into.dst_offset);
        let local = TiledTexelsModule::LOCAL_SIZE;
        let mut moves = Vec::new();
        for into in [true, false] {
            let invocations = u64::from(TiledTexelsModule::invocations(&levels, into));
            let groups = Self::grid(device, invocations)
                .ok_or_else(|| internal("a tiled heap texture's moves past the device's grid"))?;
            let shape = TiledTexelsModule::shape(&levels, into, groups[0] * local)
                .ok_or_else(|| internal("a tiled heap texture of more levels than a shape"))?;
            let at = shapes_at + if into { 0 } else { SHAPE_SLOT };
            moves.push(Move { shape, at, groups });
        }
        let [into, back] = <[Move; 2]>::try_from(moves)
            .map_err(|_| internal("a tiled heap texture's moves are two"))?;
        let shape_bytes = 4 * u64::from(TiledTexelsModule::SHAPE_WORDS);
        let kernel = device.owned_kernel(OwnedModule::TiledTexels)?;
        // Each set binds the tight elements, the tiled bytes and the shape, in binding order.
        let sets = OwnedSets::write(
            device,
            &kernel,
            &[&into, &back]
                .iter()
                .map(|step| {
                    vec![
                        (tight_raw, tight_at, tight_bytes),
                        (staging_raw, staging_at, layout.size),
                        (shapes_raw, step.at, shape_bytes),
                    ]
                })
                .collect::<Vec<_>>(),
        )?;
        let regions = regions
            .iter()
            .map(|region| vk::BufferImageCopy {
                buffer_offset: tight_at + region.buffer_offset,
                ..*region
            })
            .collect();
        Ok(HeapTiledCopies {
            layout,
            _tight: tight,
            tight_raw,
            regions,
            _shapes: shapes,
            shapes_raw,
            moves: [into, back],
            kernel,
            sets,
        })
    }

    /// Record the dispatch of the move `into` the image, or back, after writing its shape.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, with every write the dispatch
    /// reads made visible to it; the buffers and the sets are live until it completes.
    unsafe fn dispatch(&self, device: &Vk, command_buffer: vk::CommandBuffer, into: bool) {
        let (step, set) = match into {
            true => (&self.moves[0], self.sets.sets[0]),
            false => (&self.moves[1], self.sets.sets[1]),
        };
        // SAFETY: the caller's contract; the shape fits its slot, and the sets bind the
        // texture's own buffers.
        unsafe {
            device.cmd_update_buffer(command_buffer, self.shapes_raw, step.at, &step.shape);
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(command_buffer, step.groups[0], step.groups[1], 1);
            Barrier::everything().record(device, command_buffer);
        }
    }

    /// Record the move of the texture's `image` from `bytes` (the heap's `VkBuffer`) through
    /// `staging` (`into`), or back, its rows back by `back` (`shadow::record_back`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the image, the buffers and the
    /// sets are live until it completes, and the image rests in `GENERAL`; `back` is the copy's
    /// `ShadowCopy::staged_back`, over `bytes`.
    pub(crate) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        (bytes, staging): (vk::Buffer, &ShadowStaging),
        into: bool,
        back: Option<&RowCopies>,
    ) {
        let general = vk::ImageLayout::GENERAL;
        // SAFETY: the caller's contract; the copies lie in the buffers and the image (planned
        // when the texture was made).
        unsafe {
            Barrier::everything().record(device, command_buffer);
            // Both ways the staging buffer starts as the heap's bytes: back, so the bytes
            // between the elements are the heap's when they are copied back.
            device.cmd_copy_buffer(command_buffer, bytes, staging.raw(), &[staging.into]);
            if !into {
                device.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    general,
                    self.tight_raw,
                    &self.regions,
                );
            }
            self.dispatch(device, command_buffer, into);
            if into {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.tight_raw,
                    image,
                    general,
                    &self.regions,
                );
            } else {
                shadow::record_back(device, command_buffer, (staging, bytes), back);
            }
        }
    }

    /// Record a copy of the texture's `image` into `to` from `at`, laid out as in the heap's
    /// bytes from the texture's first (a merge's kept or changed part, `shadow_merge.rs`): the
    /// move back into `staging`, then its bytes on. A byte no element lies in is the staging
    /// buffer's, the heap's as the last move left it, alike in each copy of one merge.
    ///
    /// # Safety
    ///
    /// As `record`; `to` holds the texture's bytes from `at`, a multiple of 4.
    pub(crate) unsafe fn record_out(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        staging: &ShadowStaging,
        (to, at): (vk::Buffer, u64),
    ) {
        // SAFETY: the caller's contract, as `record`'s.
        unsafe {
            Barrier::everything().record(device, command_buffer);
            device.cmd_copy_image_to_buffer(
                command_buffer,
                image,
                vk::ImageLayout::GENERAL,
                self.tight_raw,
                &self.regions,
            );
            self.dispatch(device, command_buffer, false);
            let region = vk::BufferCopy {
                src_offset: staging.into.dst_offset,
                dst_offset: at,
                size: staging.into.size,
            };
            device.cmd_copy_buffer(command_buffer, staging.raw(), to, &[region]);
        }
    }
}
