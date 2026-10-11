//! A sparse texture's tile map: the words a function reads to answer the residency of a sparse
//! texture where the device has no shader residency.
//!
//! Metal maps a sparse texture's memory a tile at a time (`updateTextureMapping:mode:region:
//! mipLevel:slice:` on an `MTLResourceStateCommandEncoder`), and a function reads whether the
//! texels it touched were mapped (`sparse_color::resident()`). A Vulkan device without
//! `shaderResourceResidency` cannot answer that from the image. The consumer then backs the
//! texture with an ordinary image and keeps, beside it, a storage buffer with a bit per tile of
//! each level and slice that is set while the tile is mapped. A module whose slot has the
//! tile-map arm (`RuntimeFact::SparseTexture { tile_map: true, .. }`) reads it at the slot's
//! `ResourceKind::TileMap` binding. The consumer builds the words with `bind::SparseTileMap`,
//! which mirrors each mapping.
//!
//! **Format** (version 1): little-endian 32-bit words.
//!
//! | word | meaning |
//! |---|---|
//! | 0 | [`FORMAT`] |
//! | 1, 2, 3 | the tile's width, height, and depth in texels (each at least 1) |
//! | 4 | the slice count `S` (array layers; 1 for a 2D or 3D texture) |
//! | 5 | `B`, the bits of one slice |
//! | 6, 7 | 0 |
//! | `8 + 4l + 0..4` | level `l < 16`: its first bit in a slice, its tiles across x, y, z |
//! | [`BITS_WORD`]`..` | the bits: bit `b` is bit `b % 32` of word `BITS_WORD + b / 32` |
//!
//! A level in the texture's mip tail (from Metal's `firstMipmapInTail`) has one tile, the slice's
//! tail bit, which every level of the tail shares: Metal maps and unmaps the tail as one unit. An
//! entry past the texture's last level has one tile at bit 0.
//!
//! **Decoding** ([`bit_index`]): the residency of texel `(x, y, z)` of level `l` and slice `s` is
//! bit `min(s, S - 1)·B + base + (t_z·n_y + t_y)·n_x + t_x` of level `l' = min(l, 15)`, where
//! `(base, n_x, n_y, n_z)` is that level's entry and `t_a = min(c_a / tile_a, n_a - 1)`. Every
//! index is clamped into the words the format states, so a function reads no word past the map
//! even for an access outside the texture (whose residency it answers from the coordinate alone).

/// Word 0: the format's version tag (`"M2T1"` read as a little-endian word).
pub const FORMAT: u32 = u32::from_le_bytes(*b"M2T1");
/// Words 1, 2, 3: the tile's extent in texels.
pub const TILE_WORD: u32 = 1;
/// Word 4: the slice count.
pub const SLICES_WORD: u32 = 4;
/// Word 5: the bits of one slice.
pub const SLICE_BITS_WORD: u32 = 5;
/// The levels the map describes: a Metal texture has at most 15 (16384 texels on a side).
pub const MAX_LEVELS: u32 = 16;
/// The word of level 0's entry; level `l`'s is `LEVELS_WORD + LEVEL_WORDS·l`.
pub const LEVELS_WORD: u32 = 8;
/// The words of one level's entry: its first bit, then its tiles across x, y, and z.
pub const LEVEL_WORDS: u32 = 4;
/// The first word of the bits.
pub const BITS_WORD: u32 = LEVELS_WORD + LEVEL_WORDS * MAX_LEVELS;

/// The extent of level `level` of a texture whose level 0 is `extent`: each axis halved per level,
/// and at least 1.
pub fn level_extent(extent: [u32; 3], level: u32) -> [u32; 3] {
    extent.map(|axis| axis.checked_shr(level).unwrap_or(0).max(1))
}

/// The bit of texel `texel` of level `level` and slice `slice` in a map whose words are `words`,
/// as [the module docs](self) decode it, or `None` where a word the decoding reads is missing (a
/// module's read there is the buffer's contained read).
pub fn bit_index(words: &[u32], level: u32, slice: u32, texel: [u32; 3]) -> Option<u64> {
    let word = |w: u32| words.get(w as usize).copied();
    let slices = word(SLICES_WORD)?;
    let slice_bits = word(SLICE_BITS_WORD)?;
    let entry = LEVELS_WORD + LEVEL_WORDS * level.min(MAX_LEVELS - 1);
    let base = word(entry)?;
    let mut tiles = [0u32; 3];
    let mut counts = [0u32; 3];
    for axis in 0..3 {
        let extent = word(TILE_WORD + axis as u32)?.max(1);
        let count = word(entry + 1 + axis as u32)?;
        counts[axis] = count;
        tiles[axis] = (texel[axis] / extent).min(count.wrapping_sub(1));
    }
    let slice = u64::from(slice.min(slices.wrapping_sub(1)));
    let within = (u64::from(tiles[2]) * u64::from(counts[1]) + u64::from(tiles[1]))
        * u64::from(counts[0])
        + u64::from(tiles[0]);
    Some(slice * u64::from(slice_bits) + u64::from(base) + within)
}

/// Whether bit `bit` of the map `words` is set; `false` past the words.
pub fn bit_set(words: &[u32], bit: u64) -> bool {
    let at = u64::from(BITS_WORD) + bit / 32;
    usize::try_from(at)
        .ok()
        .and_then(|at| words.get(at))
        .is_some_and(|w| (w >> (bit % 32)) & 1 == 1)
}
