//! `texture/heap_tiled`: Metal's layout of a heap texture's elements against the M3 probe's L rows,
//! measured on an Apple M4 under macOS 27.0.1 (an M3 Max under macOS 26.7 alike on every row of a
//! plain format): each row's answered size and alignment, its level bases and chain stride, and,
//! for a plain format, the hash of its span with every element's units holding their numbers, as
//! the probe wrote them; and the lossless compression's answers against the probe's C rows and a
//! second probe's answers.

use crate::metal_level::MetalLevel;
use crate::texture::heap_tiled::{
    fresh_pattern, level_bytes, lossless_answer, metal_compresses, metal_heap_layout,
    pattern_levels, sample_grid, BlockPlacement, Elements, FreshDepth, FreshPattern, GpuRows,
    HeapLayout, TiledLayout, Untiled,
};
use crate::texture_descriptor::TextureLayout;
use metal2vulkan::metal::{PixelFormat, TextureType as T, TextureUsage};
use metal2vulkan::vk::Format;

/// One L row: its id, element bytes, block, type, level-0 extent, levels (0 the full chain) and
/// array length, Metal's size, alignment and chain stride (an array's, cube's or 3D texture's),
/// its level bases, and its span's hash.
struct Row {
    id: &'static str,
    unit: u64,
    block: [u64; 2],
    texture_type: T,
    extent: [u64; 3],
    levels: u32,
    array: u64,
    answer: (u64, u64, Option<u64>),
    bases: &'static [u64],
    span: u64,
}

fn row(
    id: &'static str,
    unit: u64,
    block: [u64; 2],
    texture_type: T,
    extent: [u64; 3],
    (levels, array): (u32, u64),
    answer: (u64, u64, Option<u64>),
    bases: &'static [u64],
    span: u64,
) -> Row {
    Row {
        id,
        unit,
        block,
        texture_type,
        extent,
        levels,
        array,
        answer,
        bases,
        span,
    }
}

/// The probe's L rows, as it states them (`m3probe.m`).
fn rows() -> Vec<Row> {
    vec![
        row(
            "L01",
            4,
            [1, 1],
            T::Type2D,
            [4, 2, 1],
            (1, 1),
            (128, 128, None),
            &[0],
            0x8af0436c15c63d6d,
        ),
        row(
            "L02",
            4,
            [1, 1],
            T::Type2D,
            [4, 4, 1],
            (1, 1),
            (128, 128, None),
            &[0],
            0x3a00051ff68aa595,
        ),
        row(
            "L03",
            4,
            [1, 1],
            T::Type2D,
            [4, 7, 1],
            (1, 1),
            (128, 128, None),
            &[0],
            0xdd6d80dac5721ad9,
        ),
        row(
            "L04",
            4,
            [1, 1],
            T::Type2D,
            [4, 100, 1],
            (1, 1),
            (2048, 128, None),
            &[0],
            0x855c3fdd0a0512da,
        ),
        row(
            "L05",
            4,
            [1, 1],
            T::Type2D,
            [129, 65, 1],
            (0, 1),
            (109568, 128, None),
            &[0, 98304, 106496, 108544, 109056, 109184, 109312, 109440],
            0xa2fba87ea7ff6764,
        ),
        row(
            "L06",
            16,
            [1, 1],
            T::Type2D,
            [129, 65, 1],
            (0, 1),
            (371584, 128, None),
            &[0, 245760, 360448, 368640, 370688, 371200, 371328, 371456],
            0x2492e1d50f408e56,
        ),
        row(
            "L07",
            1,
            [1, 1],
            T::Type2D,
            [129, 65, 1],
            (0, 1),
            (44032, 128, None),
            &[0, 32768, 40960, 43008, 43520, 43648, 43776, 43904],
            0xf5ebb0a44d4ef754,
        ),
        row(
            "L08",
            1,
            [1, 1],
            T::Type2D,
            [37, 328, 1],
            (1, 1),
            (32768, 128, None),
            &[0],
            0x9b11bec5ceb8c87c,
        ),
        row(
            "L09",
            1,
            [1, 1],
            T::Type2D,
            [458, 290, 1],
            (0, 1),
            (300672, 128, None),
            &[
                0, 196608, 278528, 294912, 299008, 300032, 300288, 300416, 300544,
            ],
            0x1eb0ca055317c905,
        ),
        row(
            "L10",
            2,
            [1, 1],
            T::Type2D,
            [100, 37, 1],
            (1, 1),
            (16384, 128, None),
            &[0],
            0xbce013e6e4c96e53,
        ),
        row(
            "L11",
            2,
            [1, 1],
            T::Type2D,
            [383, 256, 1],
            (4, 1),
            (300672, 128, None),
            &[0, 196608, 278528, 294912],
            0x979572030abe7044,
        ),
        row(
            "L12",
            4,
            [1, 1],
            T::Type2D,
            [100, 37, 1],
            (0, 1),
            (43904, 128, None),
            &[0, 32768, 40960, 43008, 43520, 43648, 43776],
            0xa3513495246f96d8,
        ),
        row(
            "L13",
            8,
            [1, 1],
            T::Type2D,
            [300, 20, 1],
            (1, 1),
            (131072, 128, None),
            &[0],
            0x2c070da2053c11e3,
        ),
        row(
            "L14",
            8,
            [1, 1],
            T::Type2D,
            [129, 256, 1],
            (4, 1),
            (579200, 128, None),
            &[0, 393216, 557056, 573440],
            0x629d02556a27e882,
        ),
        row(
            "L15",
            16,
            [1, 1],
            T::Type2D,
            [33, 17, 1],
            (1, 1),
            (32768, 128, None),
            &[0],
            0xe01c04b27f249279,
        ),
        row(
            "L16",
            16,
            [1, 1],
            T::Type2D,
            [576, 49, 1],
            (3, 1),
            (939648, 128, None),
            &[0, 589824, 851968],
            0x2d3542c55254fe77,
        ),
        row(
            "L17",
            4,
            [1, 1],
            T::Type2DArray,
            [100, 37, 1],
            (0, 3),
            (147456, 16384, Some(49152)),
            &[0, 32768, 40960, 43008, 43520, 43648, 43776],
            0x05fcddb2e40428c5,
        ),
        row(
            "L18",
            1,
            [1, 1],
            T::Type2DArray,
            [561, 108, 1],
            (1, 2),
            (262144, 16384, Some(131072)),
            &[0],
            0xb8f4ce1a5624e22f,
        ),
        row(
            "L19",
            4,
            [1, 1],
            T::Cube,
            [40, 40, 1],
            (0, 1),
            (196608, 16384, Some(32768)),
            &[0, 16384, 20480, 21504, 21760, 21888],
            0x792c2854115f5b8a,
        ),
        row(
            "L20",
            4,
            [1, 1],
            T::Type3D,
            [100, 37, 9],
            (0, 1),
            (442368, 16384, Some(49152)),
            &[0, 32768, 40960, 43008, 43520, 43648, 43776],
            0xbe53792d48aaa86c,
        ),
        row(
            "L21",
            1,
            [1, 1],
            T::Type3D,
            [103, 257, 9],
            (3, 1),
            (884736, 16384, Some(98304)),
            &[0, 65536, 81920],
            0x7d553da95cbf9f69,
        ),
        row(
            "L22",
            4,
            [1, 1],
            T::Type1D,
            [5000, 1, 1],
            (1, 1),
            (32768, 128, None),
            &[0],
            0x1a7c65cd83ebe0ec,
        ),
        row(
            "L23",
            4,
            [1, 1],
            T::Type2D,
            [5000, 3, 1],
            (0, 1),
            (180608, 128, None),
            &[
                0, 131072, 163840, 172032, 176128, 178176, 179200, 179712, 179968, 180096, 180224,
                180352, 180480,
            ],
            0x80c6df1ba4aa8059,
        ),
        row(
            "L24",
            1,
            [1, 1],
            T::Type2D,
            [59, 15, 1],
            (4, 1),
            (1792, 128, None),
            &[0, 1024, 1280, 1408],
            0x4b063e858ae06abc,
        ),
        row(
            "L25",
            8,
            [4, 4],
            T::Type2DArray,
            [100, 36, 1],
            (0, 3),
            (17664, 128, Some(5888)),
            &[0, 4096, 5120, 5376, 5504, 5632, 5760],
            0x3b660bcb74cf4178,
        ),
        row(
            "L26",
            16,
            [8, 8],
            T::Type2D,
            [300, 20, 1],
            (0, 1),
            (6144, 128, None),
            &[0, 4096, 5120, 5376, 5504, 5632, 5760, 5888, 6016],
            0x4371b8d24ef9e4f3,
        ),
        row(
            "L27",
            16,
            [4, 4],
            T::Cube,
            [64, 64, 1],
            (0, 1),
            (35328, 128, Some(5888)),
            &[0, 4096, 5120, 5376, 5504, 5632, 5760],
            0x66e8e6b2f1f6e87d,
        ),
        row(
            "L28",
            8,
            [1, 1],
            T::Type2D,
            [64, 64, 1],
            (1, 1),
            (32768, 128, None),
            &[0],
            0x091e7a48780c7645,
        ),
    ]
}

/// The levels of a full chain whose largest side is `side`.
fn full(side: u64) -> u32 {
    u64::BITS - side.leading_zeros()
}

impl Row {
    /// Its slices (six a cube's).
    fn slices(&self) -> u64 {
        match self.texture_type {
            T::Cube => 6 * self.array,
            _ => self.array,
        }
    }

    /// Its levels, the full chain's for 0.
    fn levels(&self) -> u32 {
        let [w, h, d] = self.extent;
        let depth = if self.texture_type == T::Type3D { d } else { 1 };
        match self.levels {
            0 => full(w.max(h).max(depth)),
            levels => levels,
        }
    }

    /// Its layout, of its texels or blocks (a block-compressed row's answer and level bases are
    /// alike under both placements).
    fn layout(&self) -> TiledLayout {
        let (unit, block, ty) = (self.unit, self.block, self.texture_type);
        let elements = match block {
            [1, 1] => Elements::Texels,
            block => Elements::Blocks {
                block,
                placement: BlockPlacement::Halved,
                gpu_rows: GpuRows::AsCpu,
            },
        };
        TiledLayout::of(
            unit,
            elements,
            ty,
            self.extent,
            self.levels(),
            self.slices(),
        )
        .unwrap_or_else(|| panic!("{}: no layout", self.id))
    }
}

/// FNV-1a over `bytes`, from `hash`.
pub(super) fn fnv(bytes: &[u8], mut hash: u64) -> u64 {
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
    }
    hash
}

/// The probe's span hash of a texture laid out as `layout`, `size` bytes: each element of each
/// slice, level and depth slice in turn numbered from 0, its units of `min(B, 4)` bytes (unit `j`
/// of element `e` holding `e · per + j + 1`, `per` its units) written where the layout places
/// it, over passes of 8 or 16 bits for 1- and 2-byte units; the hash chained over the passes.
pub(super) fn span_hash(layout: &TiledLayout, size: u64) -> u64 {
    let unit = layout.unit;
    let u = unit.min(4);
    let per = if unit >= 4 { unit / 4 } else { 1 };
    let mut places = Vec::new();
    let mut element = 0u64;
    let slices = if layout.volume { 1 } else { layout.chains };
    for slice in 0..slices {
        for (k, level) in layout.levels.iter().enumerate() {
            for z in 0..level.depth {
                let chain = if layout.volume { z } else { slice };
                for y in 0..level.extent[1] {
                    for x in 0..level.extent[0] {
                        let at = layout.element(chain, k, x, y);
                        for j in 0..per {
                            places.push((at / u + j, element * per + j + 1));
                        }
                        element += 1;
                    }
                }
            }
        }
    }
    let bits = u64::from(u64::BITS - (element * per + 1).leading_zeros());
    let passes = if u == 4 { 1 } else { bits.div_ceil(8 * u) };
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for pass in 0..passes {
        let mut span = vec![0u8; size as usize];
        for &(place, value) in &places {
            let shifted = value >> (8 * u * pass);
            let bytes = shifted.to_le_bytes();
            let at = (place * u) as usize;
            span[at..at + u as usize].copy_from_slice(&bytes[..u as usize]);
        }
        hash = fnv(&span, hash);
    }
    hash
}

/// Every L row's answer, level bases and chain stride are the layout's.
#[test]
fn the_layout_answers_every_l_row_as_metal() {
    for row in rows() {
        let layout = row.layout();
        let (size, align, stride) = row.answer;
        assert_eq!((layout.size, layout.align), (size, align), "{}", row.id);
        let bases: Vec<u64> = layout.levels.iter().map(|level| level.base).collect();
        assert_eq!(bases, row.bases, "{}", row.id);
        if let Some(stride) = stride {
            assert_eq!(layout.stride, stride, "{}", row.id);
        }
    }
}

/// Every L row of a plain format places each element as Metal does: its span's hash is the
/// probe's.
#[test]
fn the_layout_places_every_plain_l_row_s_elements_as_metal() {
    for row in rows().iter().filter(|row| row.block == [1, 1]) {
        let layout = row.layout();
        let (size, _, _) = row.answer;
        assert_eq!(span_hash(&layout, size), row.span, "{}", row.id);
    }
}

/// The M4's answers for the probe's block-compressed rows L29 to L36: their
/// blocks' allocations are the layout's, a block an element.
#[test]
fn block_compressed_allocations_are_the_layout_s() {
    /// A row: its format, block, element bytes, extent, levels (0 the full chain), and Metal's
    /// size and level bases.
    fn block(
        pf: PixelFormat,
        unit: u64,
        [w, h]: [u64; 2],
        levels: u32,
        (size, bases): (u64, &'static [u64]),
    ) -> (PixelFormat, u64, [u64; 2], u32, u64, &'static [u64]) {
        (pf, unit, [w, h], levels, size, bases)
    }
    let long = &[0, 4096, 5120, 5376, 5504, 5632, 5760];
    let wide = &[0, 8192, 10240, 10752, 10880, 11008, 11136];
    let rows = [
        block(PixelFormat::BC1RGBA, 8, [100, 36], 0, (5888, long)),
        block(PixelFormat::BC1RGBA, 8, [64, 4], 1, (128, &[0])),
        block(PixelFormat::BC1RGBA, 8, [12, 4], 1, (128, &[0])),
        block(PixelFormat::BC1RGBA, 8, [36, 100], 0, (5888, long)),
        block(PixelFormat::BC1RGBA, 8, [4, 12], 1, (128, &[0])),
        block(PixelFormat::ASTC4x4LDR, 16, [100, 36], 0, (11264, wide)),
        block(PixelFormat::BC7RGBAUnorm, 16, [100, 36], 0, (11264, wide)),
        block(PixelFormat::BC1RGBA, 8, [100, 33], 0, (5888, long)),
    ];
    for (pf, unit, [w, h], levels, size, bases) in rows {
        let levels = if levels == 0 { full(w.max(h)) } else { levels };
        let elements = Elements::Blocks {
            block: [4, 4],
            placement: BlockPlacement::Narrow,
            gpu_rows: GpuRows::AsCpu,
        };
        let layout = TiledLayout::of(unit, elements, T::Type2D, [w, h, 1], levels, 1).unwrap();
        let got: Vec<u64> = layout.levels.iter().map(|level| level.base).collect();
        assert_eq!((layout.size, layout.align), (size, 128), "{pf:?} {w}x{h}");
        assert_eq!(got, bases, "{pf:?} {w}x{h}");
    }
}

/// A level's bytes as the heap lane's Metal answers state them (the M3 Max and the
/// M4 alike; `heap_textures.rs`'s shapes, uncompressed).
#[test]
fn level_bytes_are_metal_s() {
    let rgba = [64, 64];
    assert_eq!(level_bytes(4, rgba, [64, 64], 0, Elements::Texels), 16384);
    assert_eq!(level_bytes(4, rgba, [16, 16], 0, Elements::Texels), 1024);
    assert_eq!(level_bytes(4, rgba, [17, 17], 0, Elements::Texels), 4096);
    assert_eq!(level_bytes(4, rgba, [100, 100], 0, Elements::Texels), 65536);
    assert_eq!(
        level_bytes(4, rgba, [512, 128], 0, Elements::Texels),
        262144
    );
    let chain: u64 = (0..9)
        .map(|level| level_bytes(4, rgba, [256, 256], level, Elements::Texels))
        .sum();
    assert_eq!(chain, 349824);
    assert_eq!(
        level_bytes(1, [128, 128], [64, 64], 0, Elements::Texels),
        4096
    );
    assert_eq!(
        level_bytes(2, [128, 64], [64, 64], 0, Elements::Texels),
        8192
    );
}

/// Which textures Metal compresses and what it answers for them (the probe's C rows): a colour
/// texture of neither `ShaderWrite` nor `PixelFormatView` usage with `allowGPUOptimizedContents`
/// and a level 0 of 16 texels or more on its shorter side, a depth or stencil texture of any usage;
/// its chains, then 2 bytes of metadata for each 8 x 8 texels of its level padded to powers of two,
/// rounded up to 128, for each chain.
#[test]
fn lossless_answers_are_metal_s() {
    let read = TextureUsage::SHADER_READ;
    let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let view = TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW;
    let target = TextureUsage::RENDER_TARGET;
    assert!(metal_compresses(true, read, true, [16, 16]));
    assert!(metal_compresses(true, target, true, [64, 64]));
    assert!(!metal_compresses(true, read, true, [16, 8]));
    assert!(!metal_compresses(true, read, true, [6, 43]));
    assert!(!metal_compresses(true, read, true, [2048, 5]));
    assert!(!metal_compresses(true, read, false, [64, 64]));
    assert!(!metal_compresses(true, write, true, [64, 64]));
    assert!(!metal_compresses(true, view, true, [64, 64]));
    assert!(metal_compresses(false, write, true, [64, 64]));
    assert!(!metal_compresses(false, write, true, [8, 8]));
    // (element bytes, type, extent, slices, Metal's size and alignment)
    let rows = [
        (4, T::Type2D, [64, 64, 1], 1, (16512, 128)),
        (4, T::Type2D, [16, 16, 1], 1, (1152, 128)),
        (4, T::Type2D, [100, 37, 1], 1, (33024, 128)),
        (4, T::Type2DArray, [64, 64, 1], 2, (33024, 16384)),
        (1, T::Type2D, [64, 64, 1], 1, (4224, 128)),
        (8, T::Type2D, [64, 64, 1], 1, (32896, 128)),
        (16, T::Type2D, [64, 64, 1], 1, (65664, 128)),
        (2, T::Type2D, [64, 64, 1], 1, (8320, 128)),
        (4, T::Type2D, [17, 17, 1], 1, (4224, 128)),
        (4, T::Type2D, [100, 100, 1], 1, (66048, 128)),
        (4, T::Type2DArray, [33, 33, 1], 3, (49536, 16384)),
        (4, T::Cube, [33, 33, 1], 6, (99072, 16384)),
        (4, T::Type3D, [33, 33, 33], 1, (544896, 16384)),
        (4, T::Type2D, [512, 128, 1], 1, (264192, 128)),
        (4, T::Type2D, [300, 260, 1], 1, (417792, 128)),
    ];
    for (unit, ty, extent, slices, (size, align)) in rows {
        let layout = TiledLayout::of(unit, Elements::Texels, ty, extent, 1, slices).unwrap();
        let answer = lossless_answer(&layout).unwrap();
        assert_eq!(
            (answer.size, answer.align),
            (size, align),
            "{unit} {ty:?} {extent:?}"
        );
    }
}

/// A mipmapped texture Metal compresses answers its full chain uncompressed, then for each chain
/// metadata over the full chain's levels while level 0's sides, padded to multiples of 16 and
/// halved, have a longer side of 16 or more, whatever levels it has (an Apple M4 under macOS
/// 27.0.1, and an M3 Max under 26.7, alike: the consumer's boot's `RGBA16Float` 224 x 928 and 216 x
/// 928 textures of 3 levels among them, and the 7-level 64 x 64 `R32Uint` render target MoltenVK
/// answered 0x5680 for); a 3D texture's chains, one a depth slice of level 0, and a cube array's
/// alike.
#[test]
fn mipmapped_lossless_answers_are_metal_s() {
    // (element bytes, type, extent, levels, slices, Metal's size and alignment)
    let rows = [
        (8, T::Type2D, [224, 928, 1], 3, 1, (0x279700, 128)),
        (8, T::Type2D, [216, 928, 1], 3, 1, (0x279700, 128)),
        (4, T::Type2D, [64, 64, 1], 7, 1, (0x5800, 128)),
        (4, T::Type2D, [1024, 16, 1], 10, 1, (0x15d80, 128)),
        (4, T::Type2D, [16, 1025, 1], 2, 1, (0x2b680, 128)),
        (4, T::Type2D, [513, 17, 1], 10, 1, (0x2b380, 128)),
        (4, T::Type2D, [1440, 1025, 1], 10, 1, (0x8e1680, 128)),
        (4, T::Type2D, [33, 63, 1], 2, 1, (0x5780, 128)),
        (4, T::Type2D, [500, 513, 1], 2, 1, (0x19ac80, 128)),
        (2, T::Type2D, [100, 37, 1], 4, 1, (0x5880, 128)),
        (2, T::Type2D, [129, 500, 1], 6, 1, (0x4c280, 128)),
        (4, T::Type2D, [256, 256, 1], 9, 1, (0x56200, 128)),
        (1, T::Type2D, [1025, 63, 1], 3, 1, (0x2c400, 128)),
        (8, T::Type2D, [1440, 1080, 1], 4, 1, (0x111d600, 128)),
        (4, T::Type2DArray, [100, 100, 1], 7, 2, (0x30600, 16384)),
        (4, T::Type2DArray, [16, 16, 1], 2, 2, (0xe00, 128)),
        (8, T::Type2DArray, [928, 928, 1], 10, 2, (0x135d600, 16384)),
        (4, T::Cube, [216, 216, 1], 8, 6, (0x214200, 16384)),
        (8, T::Cube, [928, 928, 1], 10, 6, (0x3a18200, 16384)),
        (2, T::Type2DArray, [513, 513, 1], 10, 6, (0x6d0200, 16384)),
        (4, T::Type3D, [64, 100, 33], 7, 1, (0x190200, 16384)),
        (4, T::Type3D, [64, 100, 33], 1, 1, (0x10a100, 16384)),
        (8, T::Type3D, [224, 17, 7], 3, 1, (0xa9180, 16384)),
        (4, T::Type3D, [513, 928, 2], 10, 1, (0x655600, 16384)),
        (4, T::Type3D, [16, 16, 16], 5, 1, (0x7000, 128)),
        (4, T::CubeArray, [100, 100, 1], 7, 12, (0x122400, 16384)),
        (8, T::CubeArray, [216, 216, 1], 8, 6, (0x3c4200, 16384)),
        (1, T::CubeArray, [17, 17, 1], 5, 12, (0x5a00, 128)),
    ];
    for (unit, ty, extent, levels, slices, (size, align)) in rows {
        let layout = TiledLayout::of(unit, Elements::Texels, ty, extent, levels, slices).unwrap();
        let answer = lossless_answer(&layout).unwrap();
        assert_eq!(
            (answer.size, answer.align),
            (size, align),
            "{unit} {ty:?} {extent:?} {levels}"
        );
    }
}

/// A one-sample `Depth32Float_Stencil8` texture answers its two planes, `Depth32Float`'s answer
/// rounded up to 16 KiB, then `Stencil8`'s, at the depth plane's alignment, compressed or not,
/// in every type it takes (an Apple M4 under macOS 27.0.1: 12330 answers; 14552 2D answers each,
/// alike).
#[test]
fn depth_stencil_answers_are_their_planes() {
    let rt = TextureUsage::RENDER_TARGET;
    let read = TextureUsage::SHADER_READ;
    // (type, extent, levels, slices, usage, allowGPUOptimizedContents, Metal's answer)
    let rows = [
        (T::Type2D, [4, 4], 1, 1, read, true, (0x4080, 128)),
        (T::Type2D, [64, 64], 7, 1, read | rt, false, (0x9700, 128)),
        (T::Type2D, [64, 64], 7, 1, read | rt, true, (0x9880, 128)),
        (T::Type2D, [100, 33], 3, 1, rt, true, (0xee00, 128)),
        (T::Type2D, [513, 224], 10, 1, read, true, (0x101800, 128)),
        (T::Type2D, [15, 17], 5, 1, read | rt, true, (0x4400, 128)),
        (
            T::Type2DArray,
            [100, 100],
            7,
            3,
            read | rt,
            true,
            (0x64900, 16384),
        ),
        (T::Cube, [64, 64], 7, 6, read | rt, false, (0x38a00, 16384)),
        (T::CubeArray, [216, 216], 8, 12, rt, true, (0x554400, 16384)),
    ];
    for (ty, [w, h], levels, slices, usage, optimized, (size, align)) in rows {
        let what = format!("{ty:?} {w}x{h} {levels} {slices} {usage:?} {optimized}");
        let ds = texture(
            ty,
            PixelFormat::Depth32FloatStencil8,
            [w, h, 1],
            (levels, slices, 1),
        );
        let level = MetalLevel::MacOS27_0_1;
        let format = Format::D32_SFLOAT_S8_UINT;
        let metal = metal_heap_layout(&ds, usage, optimized, format, level);
        let answer = metal.answer.map(|a| (a.size, a.align));
        assert_eq!(answer, Some((size, align)), "{what}");
    }
}

/// A texture layout of `texture_type`, `pixel_format`, `extent`, `levels`, `layers` and
/// `samples`.
fn texture(
    texture_type: T,
    pixel_format: PixelFormat,
    extent: [u32; 3],
    (levels, layers, samples): (u32, u32, u32),
) -> TextureLayout {
    TextureLayout {
        texture_type,
        pixel_format,
        extent,
        levels,
        layers,
        samples,
    }
}

/// How the runtime lays a heap texture out over host pages, by its format, usage and shape: a
/// plain or block-compressed format whose image holds Metal's bytes tiled; a compressed, PVRTC,
/// multisample or two-aspect texture, or one whose image holds other bytes, untiled, each with
/// its kind, and the answers Metal's where measured.
#[test]
fn heap_textures_are_tiled_where_metal_s_layout_is_measured() {
    use crate::degradation::Degradation as D;
    const M4: MetalLevel = MetalLevel::MacOS27_0_1;
    const M3: MetalLevel = MetalLevel::MacOS26_7;
    let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let read = TextureUsage::SHADER_READ;
    let rgba = texture(T::Type2D, PixelFormat::RGBA8Unorm, [64, 64, 1], (1, 1, 1));
    let tiled = metal_heap_layout(&rgba, write, true, Format::R8G8B8A8_UNORM, M4);
    let answer = tiled.answer.unwrap();
    assert_eq!((answer.size, answer.align), (16384, 128));
    assert!(matches!(tiled.layout, HeapLayout::Tiled(_)));
    // Compressed: its answer, and the lossless kind.
    let lossless = metal_heap_layout(&rgba, read, true, Format::R8G8B8A8_UNORM, M4);
    assert_eq!(lossless.answer.map(|a| a.size), Some(16512));
    let HeapLayout::Untiled(untiled) = lossless.layout else {
        panic!("a compressed texture is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureLosslessLayout);
    // An image of other bytes.
    let other = metal_heap_layout(&rgba, write, true, Format::B8G8R8A8_UNORM, M4);
    let HeapLayout::Untiled(untiled) = other.layout else {
        panic!("an image of other bytes is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureUntiledLayout);
    assert_eq!(other.answer, tiled.answer);
    // Block-compressed: Metal's answer, and its kind.
    let bc1 = texture(
        T::Type2DArray,
        PixelFormat::BC1RGBA,
        [100, 36, 1],
        (7, 3, 1),
    );
    let blocks = metal_heap_layout(&bc1, read, true, Format::BC1_RGBA_UNORM_BLOCK, M4);
    assert_eq!(blocks.answer.map(|a| (a.size, a.align)), Some((17664, 128)));
    assert!(matches!(blocks.layout, HeapLayout::Tiled(_)));
    let m3 = metal_heap_layout(&bc1, read, true, Format::BC1_RGBA_UNORM_BLOCK, M3);
    assert_eq!(m3.answer, blocks.answer);
    assert!(matches!(m3.layout, HeapLayout::Tiled(_)));
    let pvrtc = texture(
        T::Type2D,
        PixelFormat::PVRTCRGBA4BPP,
        [64, 64, 1],
        (7, 1, 1),
    );
    for level in [M3, M4] {
        let format = Format::PVRTC1_4BPP_UNORM_BLOCK_IMG;
        let pvrtc = metal_heap_layout(&pvrtc, read, true, format, level);
        assert_eq!(pvrtc.answer, None, "{level}");
        let HeapLayout::Untiled(untiled) = pvrtc.layout else {
            panic!("{level}: a PVRTC texture is tiled");
        };
        assert_eq!(untiled.kind, D::HeapTextureBlockCompressedLayout, "{level}");
    }
    let decoded = metal_heap_layout(&bc1, read, true, Format::R8G8B8A8_UNORM, M4);
    assert_eq!(decoded.answer, blocks.answer);
    let HeapLayout::Untiled(untiled) = decoded.layout else {
        panic!("a block-compressed texture held decoded is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureUntiledLayout);
    // Multisample: Metal's answer, untiled (`multisample_answers_are_metal_s`).
    let ms = texture(
        T::Type2DMultisample,
        PixelFormat::RGBA8Unorm,
        [64, 64, 1],
        (1, 1, 4),
    );
    let multisample = metal_heap_layout(
        &ms,
        TextureUsage::RENDER_TARGET,
        false,
        Format::R8G8B8A8_UNORM,
        M4,
    );
    assert_eq!(
        multisample.answer.map(|a| (a.size, a.align)),
        Some((65536, 128))
    );
    let HeapLayout::Untiled(untiled) = multisample.layout else {
        panic!("a multisample texture is tiled");
    };
    assert_eq!(untiled.kind, D::HeapTextureUntiledLayout);
    // Two-aspect: its depth plane at 16 KiB, then its stencil plane, untiled.
    let ds = texture(
        T::Type2D,
        PixelFormat::Depth32FloatStencil8,
        [64, 64, 1],
        (1, 1, 1),
    );
    let two = metal_heap_layout(&ds, write, false, Format::D32_SFLOAT_S8_UINT, M4);
    assert_eq!(two.answer.map(|a| (a.size, a.align)), Some((0x5000, 128)));
    assert!(matches!(two.layout, HeapLayout::Untiled(_)));
    // A depth texture without GPU-optimized contents: tiled.
    let depth = texture(T::Type2D, PixelFormat::Depth32Float, [64, 64, 1], (1, 1, 1));
    let tiled_depth = metal_heap_layout(&depth, read, false, Format::D32_SFLOAT, M4);
    assert!(matches!(tiled_depth.layout, HeapLayout::Tiled(_)));
}

/// A multisample texture's answers and layout kind (on an Apple M4 under macOS 27.0.1; the M3 Max
/// under 26.7 answered our probe's descriptors alike): compressed by the single-sample rule over
/// its sample grid (`w x 2h` at 2 samples, `2w x 2h` at 4); uncompressed, its pixels elements of
/// all their samples in Metal's tiles, a multisample array's slices an array's; compressed, that
/// plus metadata over the sample grid for each slice. A `Depth32Float_Stencil8` texture answers
/// neither (its device's); a compressed one is held in memory of its own
/// (`Untiled::LOSSLESS_MULTISAMPLE`), any other refused over host pages.
#[test]
fn multisample_answers_are_metal_s() {
    use crate::degradation::Degradation as D;
    let rt = TextureUsage::RENDER_TARGET;
    let rw = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_WRITE;
    let view = TextureUsage::RENDER_TARGET | TextureUsage::PIXEL_FORMAT_VIEW;
    // The sample grid, and the compression threshold over it.
    assert_eq!(sample_grid(2, [16, 8]), Some([16, 16]));
    assert_eq!(sample_grid(4, [8, 8]), Some([16, 16]));
    assert_eq!(sample_grid(8, [64, 64]), None);
    assert!(metal_compresses(true, rt, true, [16, 16]));
    assert!(!metal_compresses(true, rt, true, [8, 32]));
    assert!(!metal_compresses(true, view, true, [128, 128]));
    use PixelFormat as P;
    let (ms, msa) = (T::Type2DMultisample, T::Type2DMultisampleArray);
    // (format, samples, usage, allowGPUOptimizedContents, type, slices, extent, Metal's size and
    // alignment, compressed)
    let rows = [
        (
            P::R8Unorm,
            2,
            rt,
            false,
            ms,
            1,
            [300, 2],
            (2048, 128),
            false,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            false,
            ms,
            1,
            [64, 64],
            (65536, 128),
            false,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            true,
            ms,
            1,
            [64, 64],
            (66048, 128),
            true,
        ),
        (
            P::RGBA8Unorm,
            4,
            rw,
            true,
            ms,
            1,
            [64, 64],
            (65536, 128),
            false,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            false,
            msa,
            2,
            [64, 64],
            (131072, 16384),
            false,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            true,
            msa,
            2,
            [64, 64],
            (132096, 16384),
            true,
        ),
        (
            P::RGBA8Unorm,
            4,
            rt,
            true,
            msa,
            3,
            [16, 16],
            (12672, 128),
            true,
        ),
        (P::RGBA8Unorm, 4, rt, true, ms, 1, [8, 8], (1152, 128), true),
        (
            P::RGBA8Unorm,
            4,
            rt,
            true,
            ms,
            1,
            [7, 7],
            (1024, 128),
            false,
        ),
        (
            P::RGBA8Unorm,
            2,
            rt,
            true,
            ms,
            1,
            [16, 8],
            (1152, 128),
            true,
        ),
        (
            P::RGBA8Unorm,
            2,
            rt,
            true,
            ms,
            1,
            [8, 16],
            (1024, 128),
            false,
        ),
        (
            P::RGBA16Float,
            4,
            rt,
            false,
            ms,
            1,
            [100, 37],
            (196608, 128),
            false,
        ),
        (
            P::RGBA16Float,
            4,
            rt,
            false,
            msa,
            2,
            [9, 9],
            (16384, 128),
            false,
        ),
        (
            P::RGBA32Float,
            4,
            rt,
            false,
            ms,
            1,
            [64, 64],
            (262144, 128),
            false,
        ),
        (
            P::RGBA32Float,
            4,
            rt,
            false,
            msa,
            3,
            [17, 13],
            (98304, 16384),
            false,
        ),
        (
            P::Depth32Float,
            4,
            rw,
            true,
            ms,
            1,
            [64, 64],
            (66048, 128),
            true,
        ),
        (
            P::Depth32Float,
            4,
            rt,
            false,
            ms,
            1,
            [64, 64],
            (65536, 128),
            false,
        ),
        (
            P::Depth16Unorm,
            2,
            rt,
            true,
            msa,
            2,
            [100, 37],
            (66560, 16384),
            true,
        ),
        (P::Stencil8, 4, rt, true, ms, 1, [17, 13], (2176, 128), true),
    ];
    for (pf, samples, usage, optimized, ty, slices, [w, h], (size, align), compressed) in rows {
        let what = format!("{pf:?} x{samples} {usage:?} {optimized} {ty:?} {slices} {w}x{h}");
        let layout = texture(ty, pf, [w, h, 1], (1, slices, samples));
        let format = Format::R8G8B8A8_UNORM;
        let metal = metal_heap_layout(&layout, usage, optimized, format, MetalLevel::MacOS27_0_1);
        let answer = metal.answer.map(|a| (a.size, a.align));
        assert_eq!(answer, Some((size, align)), "{what}");
        let HeapLayout::Untiled(untiled) = metal.layout else {
            panic!("{what}: tiled");
        };
        match compressed {
            true => assert_eq!(untiled, Untiled::LOSSLESS_MULTISAMPLE, "{what}"),
            false => assert_eq!(untiled.kind, D::HeapTextureUntiledLayout, "{what}"),
        }
    }
    // `Depth32Float_Stencil8`: its depth plane rounded up to 16 KiB, then its stencil plane, at
    // the depth plane's alignment (the consumer's 100 x 60 array of 2 samples, 98816 at 16 KiB on
    // the M3 Max), held by the rule as any depth format.
    let ds_rows = [
        (2, false, ms, 1, [8, 8], (16512, 128)),
        (2, false, msa, 1, [100, 37], (81920, 16384)),
        (2, true, ms, 1, [100, 37], (98816, 128)),
        (2, true, msa, 1, [100, 37], (98816, 16384)),
        (2, true, msa, 1, [100, 60], (98816, 16384)),
        (2, true, msa, 2, [100, 37], (181248, 16384)),
        (2, true, msa, 3, [300, 64], (661504, 16384)),
        (2, false, msa, 3, [8, 8], (16768, 128)),
        (2, true, msa, 2, [17, 64], (57600, 16384)),
        (4, false, msa, 2, [8, 8], (16896, 128)),
        (4, true, msa, 2, [8, 8], (17152, 128)),
        (4, false, msa, 3, [300, 64], (1228800, 16384)),
        (4, true, ms, 1, [300, 64], (430080, 128)),
    ];
    for (samples, optimized, ty, slices, [w, h], (size, align)) in ds_rows {
        let what = format!("Depth32Float_Stencil8 x{samples} {optimized} {ty:?} {slices} {w}x{h}");
        let ds = texture(ty, P::Depth32FloatStencil8, [w, h, 1], (1, slices, samples));
        let metal = metal_heap_layout(
            &ds,
            rt,
            optimized,
            Format::D32_SFLOAT_S8_UINT,
            MetalLevel::MacOS27_0_1,
        );
        let answer = metal.answer.map(|a| (a.size, a.align));
        assert_eq!(answer, Some((size, align)), "{what}");
        let lossless = metal.layout == HeapLayout::Untiled(Untiled::LOSSLESS_MULTISAMPLE);
        let grid = sample_grid(samples, [w, h].map(u64::from)).expect("grid");
        assert_eq!(lossless, optimized && grid[0].min(grid[1]) >= 16, "{what}");
    }
    // Eight samples: not measured, no answer, refused over host pages.
    let eight = texture(ms, P::RGBA8Unorm, [64, 64, 1], (1, 1, 8));
    let metal = metal_heap_layout(
        &eight,
        rt,
        true,
        Format::R8G8B8A8_UNORM,
        MetalLevel::MacOS27_0_1,
    );
    assert_eq!(metal.answer, None);
    assert_ne!(
        metal.layout,
        HeapLayout::Untiled(Untiled::LOSSLESS_MULTISAMPLE)
    );
}

/// Each clear color `fresh_pattern` gives converts to its texel exactly, as Metal's conversion
/// does (`clear_value::tests::texel_bytes`, which the runtime's clears match): a clear writes the
/// pattern. Every pattern is of a whole number of bytes a texel, and every format with a pattern
/// but a clear color is one whose pattern holds a NaN or an extended-range code.
#[test]
fn fresh_patterns_clear_colors_write_their_texels() {
    let mut cleared = 0;
    for &pf in PixelFormat::ALL {
        match fresh_pattern(pf) {
            Some(FreshPattern::Color {
                texel,
                color: Some(color),
            }) => {
                let bytes = crate::clear_value::tests::texel_bytes(pf, color);
                assert_eq!(bytes.as_deref(), Some(texel), "{pf:?}: {color:?}");
                cleared += 1;
            }
            Some(FreshPattern::Color { texel, color: None }) => {
                assert!(
                    [2, 4, 8, 16].contains(&texel.len()),
                    "{pf:?}: a NaN or extended-range pattern of {} bytes",
                    texel.len()
                );
            }
            Some(FreshPattern::DepthStencil { depth, stencil }) => {
                assert!(depth.is_some() || stencil.is_some(), "{pf:?}");
                if let Some(FreshDepth::Value(value)) = depth {
                    assert_eq!(value, 1.0, "{pf:?}: Depth16Unorm's all ones");
                }
            }
            None => {}
        }
    }
    assert!(cleared >= 30, "{cleared} formats cleared to their pattern");
}

/// The pattern covers each level whose shorter side is 16 or more of a texture Metal compresses,
/// none of one it keeps uncompressed: the chains and sample grids our probes
/// measured (a 2x texture's grid twice as tall, a 4x one's twice as wide too).
#[test]
fn pattern_levels_are_the_levels_of_16_or_more() {
    let read = TextureUsage::SHADER_READ;
    let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    assert_eq!(pattern_levels(true, read, true, [64, 64], 7), 3);
    assert_eq!(pattern_levels(true, read, true, [64, 64], 1), 1);
    assert_eq!(pattern_levels(true, read, true, [256, 16], 9), 1);
    assert_eq!(pattern_levels(true, read, true, [100, 40], 7), 2);
    assert_eq!(pattern_levels(true, read, true, [1024, 32], 11), 2);
    assert_eq!(pattern_levels(true, read, true, [128, 16], 8), 1);
    assert_eq!(pattern_levels(true, read, true, [16, 256], 9), 1);
    assert_eq!(pattern_levels(true, read, true, [17, 17], 5), 1);
    assert_eq!(pattern_levels(true, read, true, [16, 16], 5), 1);
    assert_eq!(pattern_levels(true, read, true, [15, 15], 4), 0);
    assert_eq!(pattern_levels(true, read, true, [4096, 1], 1), 0);
    assert_eq!(pattern_levels(true, write, true, [64, 64], 7), 0);
    assert_eq!(pattern_levels(true, read, false, [64, 64], 7), 0);
    assert_eq!(pattern_levels(false, write, true, [64, 64], 1), 1);
    let grid = sample_grid(4, [8, 8]).unwrap();
    assert_eq!(
        pattern_levels(true, TextureUsage::RENDER_TARGET, true, grid, 1),
        1
    );
    let rt = TextureUsage::RENDER_TARGET;
    for (samples, [w, h], levels) in [
        (4, [4, 4], 0),
        (4, [7, 7], 0),
        (4, [8, 4], 0),
        (4, [4, 8], 0),
        (4, [9, 8], 1),
        (4, [8, 16], 1),
        (2, [8, 8], 0),
        (2, [16, 8], 1),
        (2, [8, 16], 0),
        (2, [15, 15], 0),
        (2, [16, 16], 1),
    ] {
        let grid = sample_grid(samples, [w, h]).unwrap();
        let got = pattern_levels(true, rt, true, grid, 1);
        assert_eq!(got, levels, "{samples}x {w}x{h}");
    }
}
