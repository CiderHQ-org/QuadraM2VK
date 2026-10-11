//! What both executors make of a sparse texture's mappings.
//!
//! A case names its mapped texels ([`crate::case::sparse`]); each device divides the texture into
//! its own tiles: the Metal oracle's sparse tiles (`sparseTileSizeWithTextureType:...` at the
//! heap's page size) and the Vulkan candidate's sparse image blocks (`imageGranularity`), with a
//! first level kept in a mip tail (`firstMipmapInTail`, `imageMipTailFirstLod`). Both executors
//! make each mapping through [`device_mappings`] under their own [`SparseTiling`], so a region
//! that is not a whole number of the device's tiles, or that covers part of a level in its mip
//! tail, is refused the same way before anything runs. A case's mip tail is its slice's one level
//! whole (`case::sparse`): a device that keeps that level in its tail maps the tail, and one that
//! does not maps the level's tiles; a region covering a whole level in the tail maps the tail. On
//! a device that makes every mapping, the mapped texels are exactly the regions and tails the
//! case names ([`mapped_texels`]), which is what each executor uploads ([`literal_offset`]).
//!
//! Where the plan answers a slot's residency from a tile map (on every device for a slot whose type
//! has one), the candidate makes the sparse texture as the runtime does: the mapped texels and zero
//! elsewhere ([`mapped_bytes`]), in each block of a sparse resident image that holds a mapped tile
//! where the device makes one, else in an ordinary image, and the tile map of the same mappings
//! ([`tile_map`]) made under [`tile_map_tiling`].

use crate::case::sparse::{sparse_extent, SparseMapping, LITERAL_LEVELS};
use crate::literal::LiteralTexture;

// A case's mip tail is its slice's one level whole (`case::sparse`): a literal with more levels
// needs the tail stated in texels.
const _: () = assert!(LITERAL_LEVELS == 1);

/// How one device divides a sparse texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseTiling {
    /// A tile's extent in texels (x, y, z): Metal's sparse tile, Vulkan's `imageGranularity`.
    pub tile: [u32; 3],
    /// The first mipmap level in the mip tail, or the texture's level count when it has no tail
    /// (Metal's `firstMipmapInTail`, Vulkan's `imageMipTailFirstLod`).
    pub first_level_in_tail: u32,
}

/// Texels `origin .. origin + size` of one level and slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelRegion {
    pub level: u32,
    pub slice: u32,
    pub origin: [u32; 3],
    pub size: [u32; 3],
}

/// One mapping of a sparse texture as a device makes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceMapping {
    /// A region of whole tiles: its texels, and the same region in tiles (Metal maps a region in
    /// tiles; Vulkan binds its texels, each block a whole one or one cut by the level's edge).
    Tiles {
        texels: TexelRegion,
        tile_origin: [u32; 3],
        tile_count: [u32; 3],
    },
    /// The mip tail of slice `slice`, which the device maps as one unit.
    MipTail { slice: u32 },
}

/// The extent of level `level` of a texture whose level 0 is `extent`.
pub fn level_extent(extent: [u32; 3], level: u32) -> [u32; 3] {
    extent.map(|axis| axis.checked_shr(level).unwrap_or(0).max(1))
}

/// `texture`'s level-0 extent and slice count, or why it is not a sparse texture of the case
/// model.
fn extent_of(texture: &LiteralTexture) -> Result<([u32; 3], u32), String> {
    sparse_extent(texture.texture_type, texture.dimensions).ok_or_else(|| {
        format!(
            "texture {} is a {:?} texture, which the case model does not make sparse",
            texture.binding, texture.texture_type
        )
    })
}

/// `texture`'s mappings as a device with `tiling` makes them, in the case's order, or why that
/// device cannot make one of them: a region that covers part of a level in the mip tail, or that
/// is not a whole number of tiles (its origin a multiple of the tile, its end a multiple of it or
/// the level's edge). A mip tail, and a region covering a whole level the device keeps in its
/// tail, map as the tail where the device keeps level 0 there (the tail is then exactly that one
/// level: the case model's textures have one), and a mip tail as level 0's tiles where it does
/// not. `Ok(None)` for a texture that is not sparse.
pub fn device_mappings(
    texture: &LiteralTexture,
    tiling: SparseTiling,
) -> Result<Option<Vec<DeviceMapping>>, String> {
    let Some(sparse) = &texture.sparse else {
        return Ok(None);
    };
    if tiling.tile.contains(&0) {
        return Err(format!(
            "texture {}: the device's sparse tile {:?} is empty",
            texture.binding, tiling.tile
        ));
    }
    let (extent, _) = extent_of(texture)?;
    sparse
        .mappings
        .iter()
        .enumerate()
        .map(|(index, mapping)| {
            let at = format!("texture {} mapping {index}", texture.binding);
            match *mapping {
                SparseMapping::Region {
                    level,
                    slice,
                    origin,
                    size,
                } => {
                    let edge = level_extent(extent, level);
                    if level >= tiling.first_level_in_tail {
                        // The tail is the levels from its first; a literal's is its one level.
                        let whole = origin == [0; 3] && size == edge;
                        let alone =
                            level == tiling.first_level_in_tail && level + 1 == LITERAL_LEVELS;
                        if whole && alone {
                            return Ok(DeviceMapping::MipTail { slice });
                        }
                        return Err(format!(
                            "{at}: texels {origin:?} + {size:?} of level {level} are part of the \
                             device's mip tail (from level {}), which maps only as a whole",
                            tiling.first_level_in_tail
                        ));
                    }
                    let mut tile_origin = [0; 3];
                    let mut tile_count = [0; 3];
                    for axis in 0..3 {
                        let tile = tiling.tile[axis];
                        let end = u64::from(origin[axis]) + u64::from(size[axis]);
                        let whole = end % u64::from(tile) == 0 || end == u64::from(edge[axis]);
                        if size[axis] == 0 || end > u64::from(edge[axis]) {
                            return Err(format!(
                                "{at}: texels {origin:?} + {size:?} are not a region of level \
                                 {level}'s {edge:?}"
                            ));
                        }
                        if origin[axis] % tile != 0 || !whole {
                            return Err(format!(
                                "{at}: texels {origin:?} + {size:?} of level {level} are not \
                                 whole {:?}-texel tiles of the device (each axis must start on a \
                                 tile and end on one or at the level's edge {edge:?})",
                                tiling.tile
                            ));
                        }
                        tile_origin[axis] = origin[axis] / tile;
                        tile_count[axis] = size[axis].div_ceil(tile);
                    }
                    Ok(DeviceMapping::Tiles {
                        texels: TexelRegion {
                            level,
                            slice,
                            origin,
                            size,
                        },
                        tile_origin,
                        tile_count,
                    })
                }
                // The case's tail is its one level whole: the device's tail where it keeps that
                // level in one, else the level's tiles.
                SparseMapping::MipTail { slice } if tiling.first_level_in_tail == 0 => {
                    Ok(DeviceMapping::MipTail { slice })
                }
                SparseMapping::MipTail { slice } => {
                    let size = level_extent(extent, 0);
                    let mut tile_count = [0; 3];
                    for axis in 0..3 {
                        tile_count[axis] = size[axis].div_ceil(tiling.tile[axis]);
                    }
                    Ok(DeviceMapping::Tiles {
                        texels: TexelRegion {
                            level: 0,
                            slice,
                            origin: [0; 3],
                            size,
                        },
                        tile_origin: [0; 3],
                        tile_count,
                    })
                }
            }
        })
        .collect::<Result<Vec<_>, String>>()
        .map(Some)
}

/// The texels `mappings` (of `texture`, made under `tiling`) map, each one level of one slice: a
/// region's own texels, and each level of a slice's mip tail whole. What each executor uploads
/// of the authored bytes, in mapping order.
pub fn mapped_texels(
    texture: &LiteralTexture,
    tiling: SparseTiling,
    mappings: &[DeviceMapping],
) -> Result<Vec<TexelRegion>, String> {
    let (extent, _) = extent_of(texture)?;
    let mut regions = Vec::new();
    for mapping in mappings {
        match *mapping {
            DeviceMapping::Tiles { texels, .. } => regions.push(texels),
            DeviceMapping::MipTail { slice } => {
                regions.extend((tiling.first_level_in_tail..LITERAL_LEVELS).map(|level| {
                    TexelRegion {
                        level,
                        slice,
                        origin: [0; 3],
                        size: level_extent(extent, level),
                    }
                }));
            }
        }
    }
    Ok(regions)
}

/// The byte offset in `texture`'s authored bytes of `region`'s first texel: the literal packs
/// level 0's slices one after another, each its planes (z), each its rows. A copy of the region
/// reads rows of the literal's width and images of its height from there.
pub fn literal_offset(texture: &LiteralTexture, region: &TexelRegion) -> Result<u64, String> {
    let (extent, _) = extent_of(texture)?;
    if region.level != 0 {
        return Err(format!(
            "texture {}: the literal holds level 0 only, not level {}",
            texture.binding, region.level
        ));
    }
    if region.size.contains(&0) {
        return Err(format!(
            "texture {}: an empty region {region:?}",
            texture.binding
        ));
    }
    let texel = texture.format.bytes_per_pixel() as u64;
    let [width, height, depth] = extent.map(u64::from);
    let [x, y, z] = region.origin.map(u64::from);
    let offset = ((u64::from(region.slice) * depth + z) * height + y) * width * texel + x * texel;
    let [w, h, d] = region.size.map(u64::from);
    let last = ((u64::from(region.slice) * depth + z + d - 1) * height + y + h - 1) * width * texel
        + (x + w) * texel;
    if last > texture.bytes.len() as u64 {
        return Err(format!(
            "texture {}: region {region:?} reaches byte {last} of {} authored bytes",
            texture.binding,
            texture.bytes.len()
        ));
    }
    Ok(offset)
}

/// The tiling under which the candidate emulates `texture` from a tile map: Metal's sparse tile at
/// the oracle's 16 KiB page, as apple-m3-max-macos26.7 answers it for 4-byte texels (64x64x1, for a
/// 2D, 2D array, and 3D texture alike), extended to the other texel sizes as the 16 KiB tile whose
/// width is its height or twice it (128x128 for 1 byte, 128x64 for 2, 64x32 for 8, 32x32 for 16);
/// level 0 is in the mip tail where it is narrower or shorter than a tile, as Metal kept it for a
/// 16x16 2D array and a 64x32x32 3D texture and not for a 256x128 or 256x64 one. The tiling decides
/// only which regions the emulation refuses ([`device_mappings`]): the texels it maps are the
/// case's regions and tails either way.
pub fn tile_map_tiling(texture: &LiteralTexture) -> Result<SparseTiling, String> {
    let (extent, _) = extent_of(texture)?;
    let texel = texture.format.bytes_per_pixel();
    let tile = match texel {
        1 => [128, 128, 1],
        2 => [128, 64, 1],
        4 => [64, 64, 1],
        8 => [64, 32, 1],
        16 => [32, 32, 1],
        _ => {
            return Err(format!(
                "texture {}: no 16 KiB tile of {texel}-byte texels",
                texture.binding
            ))
        }
    };
    let tailed = extent[0] < tile[0] || extent[1] < tile[1];
    Ok(SparseTiling {
        tile,
        first_level_in_tail: if tailed { 0 } else { LITERAL_LEVELS },
    })
}

/// The tile map of `texture`'s mappings under `tiling`, as the consumer keeps it: each region's
/// tiles and each slice's mip tail mapped, in the case's order. `Ok(None)` for a texture that is
/// not sparse.
pub fn tile_map(
    texture: &LiteralTexture,
    tiling: SparseTiling,
) -> Result<Option<metal2vulkan::bind::SparseTileMap>, String> {
    use crate::case::TextureType as T;
    use metal2vulkan::bind::{SparseTileMap, TileMapping};
    use metal2vulkan::metal::TextureType as Metal;
    let Some(mappings) = device_mappings(texture, tiling)? else {
        return Ok(None);
    };
    let (extent, slices) = extent_of(texture)?;
    let texture_type = match texture.texture_type {
        T::D2 => Metal::Type2D,
        T::D2Array => Metal::Type2DArray,
        T::D3 => Metal::Type3D,
        other => {
            return Err(format!(
                "texture {}: a {other:?} texture has no tile map",
                texture.binding
            ))
        }
    };
    let error = |e: metal2vulkan::bind::TileMapError| format!("texture {}: {e}", texture.binding);
    let mut map = SparseTileMap::new(
        texture_type,
        extent,
        LITERAL_LEVELS,
        slices,
        tiling.tile,
        tiling.first_level_in_tail,
    )
    .map_err(error)?;
    for mapping in mappings {
        let (level, slice, origin, size) = match mapping {
            DeviceMapping::Tiles {
                texels,
                tile_origin,
                tile_count,
            } => (texels.level, texels.slice, tile_origin, tile_count),
            DeviceMapping::MipTail { slice } => (tiling.first_level_in_tail, slice, [0; 3], [1; 3]),
        };
        map.update(TileMapping::Map, level, slice, origin, size)
            .map_err(error)?;
    }
    Ok(Some(map))
}

/// `texture`'s authored bytes with every texel outside `regions` zero: what the consumer's
/// ordinary image holds for a sparse texture emulated from a tile map, whose unmapped texels read
/// as Metal's do (zero memory).
pub fn mapped_bytes(texture: &LiteralTexture, regions: &[TexelRegion]) -> Result<Vec<u8>, String> {
    let (extent, _) = extent_of(texture)?;
    let texel = texture.format.bytes_per_pixel() as u64;
    let [width, height, _] = extent.map(u64::from);
    let mut bytes = vec![0u8; texture.bytes.len()];
    for region in regions {
        let first = literal_offset(texture, region)?;
        let row = u64::from(region.size[0]) * texel;
        for z in 0..u64::from(region.size[2]) {
            for y in 0..u64::from(region.size[1]) {
                let at = (first + (z * height + y) * width * texel) as usize;
                let end = at + row as usize;
                let (Some(from), Some(to)) = (texture.bytes.get(at..end), bytes.get_mut(at..end))
                else {
                    return Err(format!(
                        "texture {}: region {region:?} leaves the authored bytes",
                        texture.binding
                    ));
                };
                to.copy_from_slice(from);
            }
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::sparse::SparseTextureResource;
    use crate::case::{ResourceRole, TextureFormat, TextureType};

    fn sparse(
        texture_type: TextureType,
        dimensions: [u32; 3],
        mappings: Vec<SparseMapping>,
    ) -> LiteralTexture {
        let layout = crate::literal::texture_layout(texture_type, dimensions, 1).unwrap();
        let texels = layout.width * layout.height * layout.depth * layout.array_layers;
        LiteralTexture {
            binding: 2,
            role: ResourceRole::Input,
            texture_type,
            format: TextureFormat::R32Float,
            dimensions,
            sample_count: 1,
            bytes: vec![0; 4 * texels as usize],
            sparse: Some(SparseTextureResource { mappings }),
            mip_levels: 1,
        }
    }

    fn region(slice: u32, origin: [u32; 3], size: [u32; 3]) -> SparseMapping {
        SparseMapping::Region {
            level: 0,
            slice,
            origin,
            size,
        }
    }

    const NO_TAIL: u32 = LITERAL_LEVELS;

    /// A region of whole tiles maps as those tiles on any device whose tile divides it; the last
    /// tile of an axis may be cut by the level's edge, and the same region is refused where it is
    /// not whole tiles (a 128-texel block cuts the 64-texel tiles a 16 KiB page makes).
    #[test]
    fn a_region_maps_as_whole_tiles_or_is_refused() {
        let left = sparse(
            TextureType::D2,
            [256, 100, 1],
            vec![region(0, [0, 0, 0], [128, 100, 1])],
        );
        let metal = SparseTiling {
            tile: [64, 64, 1],
            first_level_in_tail: NO_TAIL,
        };
        assert_eq!(
            device_mappings(&left, metal).unwrap().unwrap(),
            [DeviceMapping::Tiles {
                texels: TexelRegion {
                    level: 0,
                    slice: 0,
                    origin: [0, 0, 0],
                    size: [128, 100, 1],
                },
                tile_origin: [0, 0, 0],
                tile_count: [2, 2, 1],
            }]
        );
        let vulkan = SparseTiling {
            tile: [128, 128, 1],
            first_level_in_tail: NO_TAIL,
        };
        assert!(device_mappings(&left, vulkan).unwrap().is_some());
        let quarter = sparse(
            TextureType::D2,
            [256, 128, 1],
            vec![region(0, [64, 0, 0], [64, 64, 1])],
        );
        assert!(device_mappings(&quarter, metal).unwrap().is_some());
        let error = device_mappings(&quarter, vulkan).unwrap_err();
        assert!(
            error.contains("not whole [128, 128, 1]-texel tiles"),
            "{error}"
        );
        // A region that ends short of a tile inside the level is not whole tiles either.
        let short = sparse(
            TextureType::D2,
            [256, 128, 1],
            vec![region(0, [0, 0, 0], [100, 128, 1])],
        );
        assert!(device_mappings(&short, metal).is_err());
    }

    /// A case's mip tail is its slice's one level whole on every device: where the device keeps
    /// that level in its tail (Metal's 16 KiB page for a 16x16 texture), it maps the tail; where it
    /// keeps no tail (lavapipe's 128x128 blocks), it maps the level's one block, cut by the level's
    /// edge. The same texels either way. A region covering the whole level maps the tail too, and
    /// one covering part of it is refused there.
    #[test]
    fn a_mip_tail_is_the_whole_level_on_every_device() {
        let small = sparse(
            TextureType::D2Array,
            [16, 16, 2],
            vec![SparseMapping::MipTail { slice: 1 }],
        );
        let tailed = SparseTiling {
            tile: [64, 64, 1],
            first_level_in_tail: 0,
        };
        let untailed = SparseTiling {
            tile: [128, 128, 1],
            first_level_in_tail: NO_TAIL,
        };
        let level = TexelRegion {
            level: 0,
            slice: 1,
            origin: [0; 3],
            size: [16, 16, 1],
        };
        let mappings = device_mappings(&small, tailed).unwrap().unwrap();
        assert_eq!(mappings, [DeviceMapping::MipTail { slice: 1 }]);
        assert_eq!(mapped_texels(&small, tailed, &mappings).unwrap(), [level]);
        let mappings = device_mappings(&small, untailed).unwrap().unwrap();
        assert_eq!(
            mappings,
            [DeviceMapping::Tiles {
                texels: level,
                tile_origin: [0; 3],
                tile_count: [1, 1, 1],
            }]
        );
        assert_eq!(mapped_texels(&small, untailed, &mappings).unwrap(), [level]);
        let whole = sparse(
            TextureType::D2Array,
            [16, 16, 2],
            vec![region(1, [0, 0, 0], [16, 16, 1])],
        );
        assert_eq!(
            device_mappings(&whole, tailed).unwrap().unwrap(),
            [DeviceMapping::MipTail { slice: 1 }]
        );
        assert!(matches!(
            device_mappings(&whole, untailed).unwrap().unwrap()[..],
            [DeviceMapping::Tiles { .. }]
        ));
        let part = sparse(
            TextureType::D2,
            [16, 16, 1],
            vec![region(0, [0, 0, 0], [16, 8, 1])],
        );
        let error = device_mappings(&part, tailed).unwrap_err();
        assert!(error.contains("part of the device's mip tail"), "{error}");
        assert!(device_mappings(&part, untailed).is_err());
    }

    /// The emulation from a tile map: a 16 KiB tile of the texel size, level 0 in the tail where it
    /// is smaller than a tile; the map's bits are exactly the case's mapped texels, a tail
    /// included, and the image's bytes are the authored ones there and zero elsewhere.
    #[test]
    fn a_tile_map_emulation_maps_exactly_the_cases_texels() {
        let left = sparse(
            TextureType::D2,
            [256, 128, 1],
            vec![region(0, [0, 0, 0], [128, 128, 1])],
        );
        let tiling = tile_map_tiling(&left).unwrap();
        assert_eq!(
            tiling,
            SparseTiling {
                tile: [64, 64, 1],
                first_level_in_tail: NO_TAIL
            }
        );
        let map = tile_map(&left, tiling).unwrap().unwrap();
        for (texel, mapped) in [
            ([127, 127, 0], true),
            ([128, 0, 0], false),
            ([5, 7, 0], true),
        ] {
            assert_eq!(map.is_mapped(0, 0, texel), mapped, "{texel:?}");
        }
        let mut left = left;
        left.bytes = (0..left.bytes.len()).map(|i| (i % 251) as u8 + 1).collect();
        let mappings = device_mappings(&left, tiling).unwrap().unwrap();
        let bytes = mapped_bytes(&left, &mapped_texels(&left, tiling, &mappings).unwrap()).unwrap();
        for (x, y) in [(0usize, 0usize), (127, 5), (128, 5), (255, 127)] {
            let at = (y * 256 + x) * 4;
            let want = if x < 128 {
                &left.bytes[at..at + 4]
            } else {
                &[0u8; 4][..]
            };
            assert_eq!(&bytes[at..at + 4], want, "({x}, {y})");
        }
        let small = sparse(
            TextureType::D2Array,
            [16, 16, 3],
            vec![
                SparseMapping::MipTail { slice: 0 },
                SparseMapping::MipTail { slice: 2 },
            ],
        );
        let tiling = tile_map_tiling(&small).unwrap();
        assert_eq!(tiling.first_level_in_tail, 0);
        let map = tile_map(&small, tiling).unwrap().unwrap();
        for (slice, mapped) in [(0, true), (1, false), (2, true)] {
            assert_eq!(
                map.is_mapped(0, slice, [15, 15, 0]),
                mapped,
                "slice {slice}"
            );
        }
        let volume = sparse(
            TextureType::D3,
            [64, 64, 32],
            vec![region(0, [0, 0, 0], [64, 64, 16])],
        );
        let tiling = tile_map_tiling(&volume).unwrap();
        let map = tile_map(&volume, tiling).unwrap().unwrap();
        assert!(map.is_mapped(0, 0, [63, 63, 15]));
        assert!(!map.is_mapped(0, 0, [0, 0, 16]));
        let mut ordinary = volume;
        ordinary.sparse = None;
        assert_eq!(tile_map(&ordinary, tiling), Ok(None));
    }

    /// The upload reads each region from the literal's packing: slices, then planes, then rows.
    #[test]
    fn a_region_is_copied_from_the_literals_packing() {
        let array = sparse(TextureType::D2Array, [4, 3, 2], vec![]);
        let at = |slice, origin| TexelRegion {
            level: 0,
            slice,
            origin,
            size: [1, 1, 1],
        };
        assert_eq!(literal_offset(&array, &at(0, [0, 0, 0])), Ok(0));
        assert_eq!(
            literal_offset(&array, &at(0, [1, 2, 0])),
            Ok(4 * (2 * 4 + 1))
        );
        assert_eq!(
            literal_offset(&array, &at(1, [3, 2, 0])),
            Ok(4 * (12 + 2 * 4 + 3))
        );
        let volume = sparse(TextureType::D3, [4, 3, 2], vec![]);
        assert_eq!(literal_offset(&volume, &at(0, [0, 1, 1])), Ok(4 * (12 + 4)));
        let past = TexelRegion {
            level: 0,
            slice: 1,
            origin: [0, 0, 0],
            size: [4, 3, 1],
        };
        assert!(literal_offset(&volume, &past).is_err());
        // An ordinary texture has no mappings.
        let mut ordinary = sparse(TextureType::D2, [4, 4, 1], vec![]);
        ordinary.sparse = None;
        assert_eq!(
            device_mappings(
                &ordinary,
                SparseTiling {
                    tile: [1, 1, 1],
                    first_level_in_tail: NO_TAIL
                }
            ),
            Ok(None)
        );
    }
}
