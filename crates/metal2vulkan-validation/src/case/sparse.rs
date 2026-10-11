//! A sparse texture's literal: which of its texels are mapped before the work.
//!
//! A texture that declares `sparse` is a sparse texture on both sides: the Metal oracle makes it
//! from a sparse heap and maps exactly its [`SparseMapping`]s, in order, with
//! `MTLResourceStateCommandEncoder` before it runs the case; the engine's candidate creates
//! its image with sparse residency and binds the same regions with `vkQueueBindSparse`. A texel no
//! mapping covers is unmapped. Each executor uploads only the authored bytes of mapped texels: an
//! upload to an unmapped texel is discarded on Metal and undefined on a Vulkan device that is not
//! `residencyNonResidentStrict`.
//!
//! A region names texels, not tiles: the tile size is the device's (Metal's sparse tile of the
//! oracle's page size, Vulkan's `imageGranularity`). Each executor refuses, before it runs
//! anything, a region that is not a whole number of its tiles
//! (`executor_contract::sparse`), so on every device that runs the case the mapped texels are
//! exactly the union of the regions, and the observation means the same everywhere. Regions may
//! not overlap, so the order of the mappings changes nothing either executor observes.
//!
//! A slice's mip tail is its levels from the first one a device keeps in its tail. The case
//! model's textures have one level, so wherever the oracle maps a tail it is the slice's level 0
//! whole, and that is what [`SparseMapping::MipTail`] maps on every device: as the tail where the
//! device keeps level 0 in one, and as that level's tiles where it does not (lavapipe keeps no
//! tail for a single-level texture smaller than a tile, which Metal keeps whole in one). A region
//! that covers a whole level the device keeps in its tail maps as the tail too.
//!
//! The checker refuses what no executor can express whatever its tiles: a texture type, sample
//! count, format, or role the case model does not make sparse, a region outside its level and
//! slice or empty, a level the texture lacks, overlapping regions, a slice's mip tail mapped
//! twice, and a region beside its slice's mip tail (the tail is the whole level). Whether a level
//! is in the mip tail is the device's answer; the executors refuse a region that covers part of a
//! level in the tail.
//!
//! A sparse texture is an input, but the function may declare it writable: the oracle then makes it
//! with shader-write usage and the candidate a storage image, and neither reads it back after the
//! work (its unmapped texels have no contents to read), so a write is observed only by the
//! function's own later reads, a write to an unmapped texel included.

use super::{ResourceRole, TextureFormat, TextureResource, TextureType};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// A sparse texture's mappings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseTextureResource {
    /// What is mapped before the work, in this order. Empty leaves every texel unmapped.
    pub mappings: Vec<SparseMapping>,
}

/// One mapping of a sparse texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SparseMapping {
    /// The texels `origin .. origin + size` (x, y, z) of mipmap level `level` of slice `slice`
    /// (an array layer; 0 for a 2D or 3D texture). Each executor maps it as a whole number of its
    /// tiles: the origin a multiple of the tile, the end a multiple of it or the level's edge.
    Region {
        level: u32,
        slice: u32,
        origin: [u32; 3],
        size: [u32; 3],
    },
    /// The mip tail of slice `slice`: every level from the first one a device keeps in its tail,
    /// which Metal and Vulkan each map as one unit. A texture literal has one level, so this is
    /// the slice's level 0 whole, which a device that keeps no tail maps as its tiles.
    MipTail { slice: u32 },
}

/// The extent of a texture literal's one level in texels (x, y, z) and its slice count, for the
/// texture types the case model makes sparse; `None` for any other type.
pub fn sparse_extent(texture_type: TextureType, dimensions: [u32; 3]) -> Option<([u32; 3], u32)> {
    let [width, height, third] = dimensions;
    match texture_type {
        TextureType::D2 => Some(([width, height, 1], 1)),
        TextureType::D2Array => Some(([width, height, 1], third)),
        TextureType::D3 => Some(([width, height, third], 1)),
        _ => None,
    }
}

/// The levels of every texture literal: the case model's textures have one.
pub const LITERAL_LEVELS: u32 = 1;

/// Whether two boxes `(origin, size)` share a texel.
fn overlaps(a: ([u32; 3], [u32; 3]), b: ([u32; 3], [u32; 3])) -> bool {
    (0..3).all(|axis| {
        let (a0, a1) = (
            u64::from(a.0[axis]),
            u64::from(a.0[axis]) + u64::from(a.1[axis]),
        );
        let (b0, b1) = (
            u64::from(b.0[axis]),
            u64::from(b.0[axis]) + u64::from(b.1[axis]),
        );
        a0 < b1 && b0 < a1
    })
}

/// Check `texture`'s sparse literal, if it declares one.
pub(super) fn validate(texture: &TextureResource, errors: &mut Vec<String>) {
    let Some(sparse) = &texture.sparse else {
        return;
    };
    let label = format!("sparse texture binding {}", texture.binding);
    let Some((extent, slices)) = sparse_extent(texture.texture_type, texture.dimensions) else {
        errors.push(format!(
            "{label}: a sparse texture is 2D, 2D array, or 3D (the case model makes no other \
             type sparse), not {:?}",
            texture.texture_type
        ));
        return;
    };
    if texture.sample_count != 1 {
        errors.push(format!(
            "{label}: a sparse texture has one sample (multisampled sparse textures are not \
             modelled)"
        ));
    }
    if texture.format == TextureFormat::Depth32Float {
        errors.push(format!(
            "{label}: depth formats are not modelled as sparse textures"
        ));
    }
    if texture.role != ResourceRole::Input {
        errors.push(format!(
            "{label}: a sparse texture is an input: its unmapped texels have no contents an \
             executor can upload or read back"
        ));
    }
    let mut tails = BTreeSet::new();
    for (index, mapping) in sparse.mappings.iter().enumerate() {
        let at = format!("{label} mapping {index}");
        match *mapping {
            SparseMapping::Region {
                level,
                slice,
                origin,
                size,
            } => {
                if level >= LITERAL_LEVELS {
                    errors.push(format!(
                        "{at}: level {level}, but the texture has {LITERAL_LEVELS} level"
                    ));
                }
                if slice >= slices {
                    errors.push(format!(
                        "{at}: slice {slice}, but the texture has {slices} slice(s)"
                    ));
                }
                if size.contains(&0) {
                    errors.push(format!("{at}: an empty region {size:?}"));
                }
                let inside = (0..3).all(|axis| {
                    origin[axis]
                        .checked_add(size[axis])
                        .is_some_and(|end| end <= extent[axis])
                });
                if !inside {
                    errors.push(format!(
                        "{at}: texels {origin:?} + {size:?} leave the level's {extent:?}"
                    ));
                }
                let overlapped = sparse.mappings[..index].iter().position(|earlier| {
                    matches!(*earlier, SparseMapping::Region {
                        level: l,
                        slice: s,
                        origin: o,
                        size: z,
                    } if (l, s) == (level, slice) && overlaps((o, z), (origin, size)))
                });
                if let Some(other) = overlapped {
                    errors.push(format!(
                        "{at}: overlaps mapping {other}'s region (mappings may not overlap, so \
                         their order is not observable)"
                    ));
                }
            }
            SparseMapping::MipTail { slice } => {
                if slice >= slices {
                    errors.push(format!(
                        "{at}: slice {slice}, but the texture has {slices} slice(s)"
                    ));
                }
                if !tails.insert(slice) {
                    errors.push(format!("{at}: slice {slice}'s mip tail is mapped twice"));
                }
            }
        }
    }
    // A mip tail is its slice's one level whole, so a region of that slice overlaps it.
    for (index, mapping) in sparse.mappings.iter().enumerate() {
        if let SparseMapping::Region { slice, .. } = *mapping {
            if tails.contains(&slice) {
                errors.push(format!(
                    "{label} mapping {index}: a region of slice {slice}, whose mip tail is mapped \
                     (the tail is the slice's whole level {}, so they overlap)",
                    LITERAL_LEVELS - 1
                ));
            }
        }
    }
}
