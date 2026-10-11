//! Sparse textures' tile maps: the consumer's mirror of Metal's tile mappings, where the device has
//! no shader residency.
//!
//! A function that reads the residency of a sparse texture (`sparse_color::resident()`) answers
//! it from the Vulkan image's residency where the device enables `shaderResourceResidency`. On a
//! device without it the module answers it from a tile map: a storage buffer with a bit per tile
//! of each level and slice, set while Metal has that tile mapped (`pipeline::tile_map` has the
//! word layout). The consumer keeps one [`SparseTileMap`] per sparse texture:
//!
//! - `newTextureWithDescriptor:` of a sparse heap is [`SparseTileMap::new`], with the tile size it
//!   answers the guest (`sparseTileSizeWithTextureType:pixelFormat:sampleCount:`, a whole number
//!   of texels per axis) and the first level of its mip tail (`firstMipmapInTail`); nothing is
//!   mapped;
//! - each `updateTextureMapping:mode:region:mipLevel:slice:` (and each element of
//!   `updateTextureMappings:...`) is [`SparseTileMap::update`] of the same region in tiles, in
//!   the order the encoder records them;
//! - the buffer holds [`SparseTileMap::to_bytes`] as of each command that reads the texture: the
//!   consumer writes the bytes (a buffer update recorded at the resource-state encoder's position
//!   in the queue, with a barrier before the next access) wherever the mapping changed, so a
//!   dispatch or draw reads the mappings Metal's would. An indirect mapping
//!   (`updateTextureMapping:mode:indirectBuffer:indirectBufferOffset:`) reads its region on the
//!   GPU, so the consumer applies it to the map on the GPU, in the same position.
//!
//! The plan binds the buffer where a slot reads it (`WriteSource::TileMap`). The texture itself is
//! an ordinary image whose texels the consumer keeps at zero while their tile is unmapped
//! (cleared when Metal unmaps it, and at creation), so an access of an unmapped texel reads what
//! Metal's reads, and a filter's unmapped taps weigh zero.

use std::fmt;

use crate::metal::TextureType;
use crate::pipeline::tile_map::{
    self as format, BITS_WORD, FORMAT, LEVELS_WORD, LEVEL_WORDS, MAX_LEVELS, SLICES_WORD,
    SLICE_BITS_WORD, TILE_WORD,
};

/// What an update does to its tiles (`MTLSparseTextureMappingMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileMapping {
    /// `MTLSparseTextureMappingModeMap`.
    Map,
    /// `MTLSparseTextureMappingModeUnmap`.
    Unmap,
}

/// Why a tile map cannot be made or updated.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileMapError {
    /// The texture type has no tile map: a sparse texture is 2D, 2D array, or 3D here.
    TextureType { texture_type: TextureType },
    /// An extent, a tile, the slice count, or the level count is 0, or there are more than
    /// 16 levels or more levels than the extent has.
    Shape,
    /// The map would need more bits than a 32-bit index addresses.
    TooLarge,
    /// An update names a level or slice the texture lacks, or a region of tiles past its level.
    Region {
        level: u32,
        slice: u32,
        origin: [u32; 3],
        size: [u32; 3],
    },
}

impl fmt::Display for TileMapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileMapError::TextureType { texture_type } => {
                write!(f, "a {texture_type:?} texture has no tile map")
            }
            TileMapError::Shape => f.write_str(
                "a tile map needs a nonzero extent, tile, slice count, and level count of at \
                 most 16 levels the extent has",
            ),
            TileMapError::TooLarge => f.write_str("the tile map needs more than 2^32 bits"),
            TileMapError::Region {
                level,
                slice,
                origin,
                size,
            } => write!(
                f,
                "tiles {origin:?} + {size:?} of level {level} and slice {slice} are not tiles of \
                 the texture"
            ),
        }
    }
}

impl std::error::Error for TileMapError {}

/// One level's tiles in a slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Level {
    /// Its first bit in a slice (the tail bit for a level in the mip tail).
    base: u32,
    /// Its tiles across x, y, and z (one for a level in the mip tail).
    tiles: [u32; 3],
    /// It is in the mip tail.
    tail: bool,
}

/// A sparse texture's tile map: which tiles Metal has mapped, as the words a module reads
/// ([`to_bytes`](SparseTileMap::to_bytes)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseTileMap {
    tile: [u32; 3],
    slices: u32,
    levels: Vec<Level>,
    slice_bits: u32,
    /// Every word: the header, then the bits.
    words: Vec<u32>,
}

impl SparseTileMap {
    /// The tile map of a sparse texture of `texture_type` whose level 0 is `extent` texels (width,
    /// height, depth: 1 for a 2D texture), with `levels` mipmap levels and `slices` array slices
    /// (1 for a 2D or 3D texture), tiles of `tile` texels, and its mip tail from level
    /// `first_level_in_tail` (`levels` or more for none). Nothing is mapped.
    pub fn new(
        texture_type: TextureType,
        extent: [u32; 3],
        levels: u32,
        slices: u32,
        tile: [u32; 3],
        first_level_in_tail: u32,
    ) -> Result<SparseTileMap, TileMapError> {
        let layered = match texture_type {
            TextureType::Type2D | TextureType::Type3D => false,
            TextureType::Type2DArray => true,
            texture_type => return Err(TileMapError::TextureType { texture_type }),
        };
        let deepest = 32 - extent.iter().copied().max().unwrap_or(0).leading_zeros();
        if extent.contains(&0)
            || tile.contains(&0)
            || slices == 0
            || levels == 0
            || levels > MAX_LEVELS
            || levels > deepest
            || (!layered && slices != 1)
            || (texture_type != TextureType::Type3D && extent[2] != 1)
        {
            return Err(TileMapError::Shape);
        }
        let mut entries = Vec::with_capacity(levels as usize);
        let mut bits: u64 = 0;
        let tail_bit = |bits: u64| u32::try_from(bits).map_err(|_| TileMapError::TooLarge);
        for level in 0..levels {
            if level >= first_level_in_tail {
                // Every level of the tail shares its one bit.
                entries.push(Level {
                    base: tail_bit(bits)?,
                    tiles: [1; 3],
                    tail: true,
                });
                continue;
            }
            let at = crate::pipeline::tile_map::level_extent(extent, level);
            let tiles = [0, 1, 2].map(|axis| at[axis].div_ceil(tile[axis]));
            entries.push(Level {
                base: tail_bit(bits)?,
                tiles,
                tail: false,
            });
            bits += tiles.iter().map(|&n| u64::from(n)).product::<u64>();
        }
        if first_level_in_tail < levels {
            bits += 1;
        }
        let slice_bits = u32::try_from(bits).map_err(|_| TileMapError::TooLarge)?;
        let total = bits * u64::from(slices);
        u32::try_from(total).map_err(|_| TileMapError::TooLarge)?;
        let mut words = vec![0u32; BITS_WORD as usize + total.div_ceil(32) as usize];
        words[0] = FORMAT;
        for axis in 0..3 {
            words[(TILE_WORD + axis as u32) as usize] = tile[axis];
        }
        words[SLICES_WORD as usize] = slices;
        words[SLICE_BITS_WORD as usize] = slice_bits;
        for level in 0..MAX_LEVELS {
            let at = (LEVELS_WORD + LEVEL_WORDS * level) as usize;
            // An entry past the last level is one tile at bit 0: the read stays in the map.
            let (base, tiles) = match entries.get(level as usize) {
                Some(e) => (e.base, e.tiles),
                None => (0, [1; 3]),
            };
            words[at] = base;
            words[at + 1..at + 4].copy_from_slice(&tiles);
        }
        Ok(SparseTileMap {
            tile,
            slices,
            levels: entries,
            slice_bits,
            words,
        })
    }

    /// The tile's extent in texels.
    pub fn tile(&self) -> [u32; 3] {
        self.tile
    }

    /// Map or unmap the tiles `origin .. origin + size` (x, y, z, in tiles) of level `level` and
    /// slice `slice`, as `updateTextureMapping:mode:region:mipLevel:slice:` does. A region of a
    /// level in the mip tail maps or unmaps the slice's whole tail, as Metal's does. An empty
    /// region changes nothing.
    pub fn update(
        &mut self,
        mode: TileMapping,
        level: u32,
        slice: u32,
        origin: [u32; 3],
        size: [u32; 3],
    ) -> Result<(), TileMapError> {
        let refused = TileMapError::Region {
            level,
            slice,
            origin,
            size,
        };
        let Some(entry) = self.levels.get(level as usize).copied() else {
            return Err(refused);
        };
        let inside = (0..3).all(|axis| {
            origin[axis]
                .checked_add(size[axis])
                .is_some_and(|end| end <= entry.tiles[axis] || entry.tail)
        });
        if slice >= self.slices || !inside {
            return Err(refused);
        }
        if size.contains(&0) {
            return Ok(());
        }
        let first = u64::from(slice) * u64::from(self.slice_bits) + u64::from(entry.base);
        let mut set = |bit: u64| {
            let word = BITS_WORD as usize + (bit / 32) as usize;
            let mask = 1u32 << (bit % 32);
            match mode {
                TileMapping::Map => self.words[word] |= mask,
                TileMapping::Unmap => self.words[word] &= !mask,
            }
        };
        if entry.tail {
            set(first);
            return Ok(());
        }
        let [nx, ny, _] = entry.tiles.map(u64::from);
        for z in origin[2]..origin[2] + size[2] {
            for y in origin[1]..origin[1] + size[1] {
                for x in origin[0]..origin[0] + size[0] {
                    set(first + (u64::from(z) * ny + u64::from(y)) * nx + u64::from(x));
                }
            }
        }
        Ok(())
    }

    /// Whether texel `texel` (x, y, z) of level `level` and slice `slice` is in a mapped tile, as
    /// a module reads the map; `false` for a level or slice the texture lacks.
    pub fn is_mapped(&self, level: u32, slice: u32, texel: [u32; 3]) -> bool {
        if level as usize >= self.levels.len() || slice >= self.slices {
            return false;
        }
        format::bit_index(&self.words, level, slice, texel)
            .is_some_and(|bit| format::bit_set(&self.words, bit))
    }

    /// The map's bytes: the words the module reads, little-endian. The buffer bound for the slot
    /// holds exactly these (`byte_len` bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// The map's length in bytes.
    pub fn byte_len(&self) -> u64 {
        self.words.len() as u64 * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2D texture's map: level 0's tiles row-major from bit 0, a level wholly smaller than a
    /// tile one tile, and the bits a module decodes are the ones an update set.
    #[test]
    fn a_region_of_tiles_sets_exactly_its_bits() {
        let mut map =
            SparseTileMap::new(TextureType::Type2D, [256, 128, 1], 2, 1, [64, 64, 1], 2).unwrap();
        // Level 0: 4x2 tiles (bits 0..8); level 1: 128x64, 2x1 tiles (bits 8..10).
        let words = &map.words;
        assert_eq!(words[0], FORMAT);
        assert_eq!(&words[1..6], &[64, 64, 1, 1, 10]);
        assert_eq!(&words[8..12], &[0, 4, 2, 1]);
        assert_eq!(&words[12..16], &[8, 2, 1, 1]);
        map.update(TileMapping::Map, 0, 0, [0, 0, 0], [2, 2, 1])
            .unwrap();
        assert_eq!(map.words[BITS_WORD as usize], 0b11_0011);
        assert!(map.is_mapped(0, 0, [127, 127, 0]));
        assert!(!map.is_mapped(0, 0, [128, 0, 0]));
        assert!(!map.is_mapped(1, 0, [0, 0, 0]));
        map.update(TileMapping::Map, 1, 0, [1, 0, 0], [1, 1, 1])
            .unwrap();
        assert!(map.is_mapped(1, 0, [64, 63, 0]));
        assert!(!map.is_mapped(1, 0, [63, 0, 0]));
        map.update(TileMapping::Unmap, 0, 0, [1, 1, 0], [1, 1, 1])
            .unwrap();
        assert!(!map.is_mapped(0, 0, [64, 64, 0]));
        assert!(map.is_mapped(0, 0, [64, 63, 0]));
        // A region past the level is refused, and changes nothing.
        let before = map.clone();
        assert!(map
            .update(TileMapping::Map, 0, 0, [3, 0, 0], [2, 1, 1])
            .is_err());
        assert_eq!(map, before);
        assert_eq!(map.byte_len(), u64::from(BITS_WORD + 1) * 4);
        assert_eq!(map.to_bytes().len() as u64, map.byte_len());
    }

    /// Slices are `B` bits apart; every level of a slice's mip tail shares one bit, which an update
    /// of any tail level maps; a texel past its level's tiles reads its last tile, as the module's
    /// clamped decoding does.
    #[test]
    fn slices_and_mip_tails_have_bits_of_their_own() {
        let mut map = SparseTileMap::new(
            TextureType::Type2DArray,
            [128, 128, 1],
            4,
            3,
            [64, 64, 1],
            1,
        )
        .unwrap();
        // Level 0: 2x2 tiles (bits 0..4); levels 1..3: the tail, bit 4. B = 5.
        assert_eq!(map.words[SLICE_BITS_WORD as usize], 5);
        map.update(TileMapping::Map, 3, 2, [0, 0, 0], [1, 1, 1])
            .unwrap();
        for level in 1..4 {
            assert!(map.is_mapped(level, 2, [0, 0, 0]));
            assert!(!map.is_mapped(level, 1, [0, 0, 0]));
        }
        assert!(!map.is_mapped(0, 2, [0, 0, 0]));
        map.update(TileMapping::Map, 0, 1, [1, 1, 0], [1, 1, 1])
            .unwrap();
        assert!(map.is_mapped(0, 1, [127, 127, 0]));
        assert!(map.is_mapped(0, 1, [500, 500, 0]));
        assert!(!map.is_mapped(0, 0, [127, 127, 0]));
        assert!(!map.is_mapped(0, 3, [127, 127, 0]));
        let volume =
            SparseTileMap::new(TextureType::Type3D, [64, 32, 32], 1, 1, [16, 16, 16], 1).unwrap();
        // 4x2x2 tiles: tile (1, 0, 1) is bit (1·2 + 0)·4 + 1 = 9.
        let mut volume = volume;
        volume
            .update(TileMapping::Map, 0, 0, [1, 0, 1], [1, 1, 1])
            .unwrap();
        assert_eq!(volume.words[BITS_WORD as usize], 1 << 9);
        assert!(volume.is_mapped(0, 0, [16, 15, 16]));
        assert!(!volume.is_mapped(0, 0, [16, 16, 16]));
    }

    /// A shape the format cannot hold is refused.
    #[test]
    fn a_map_needs_a_sparse_texture_shape() {
        let new = |texture_type, extent, levels, slices, tile| {
            SparseTileMap::new(texture_type, extent, levels, slices, tile, levels)
        };
        assert!(matches!(
            new(TextureType::Cube, [16, 16, 1], 1, 6, [16, 16, 1]),
            Err(TileMapError::TextureType { .. })
        ));
        assert_eq!(
            new(TextureType::Type2D, [16, 16, 1], 6, 1, [16, 16, 1]),
            Err(TileMapError::Shape)
        );
        assert_eq!(
            new(TextureType::Type2D, [16, 16, 1], 1, 2, [16, 16, 1]),
            Err(TileMapError::Shape)
        );
        assert_eq!(
            new(TextureType::Type2D, [16, 16, 1], 1, 1, [0, 16, 1]),
            Err(TileMapError::Shape)
        );
        assert!(new(TextureType::Type2D, [16, 16, 1], 5, 1, [16, 16, 1]).is_ok());
    }
}
