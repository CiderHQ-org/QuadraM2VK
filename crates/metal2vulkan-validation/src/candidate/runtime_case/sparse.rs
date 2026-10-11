//! A case's sparse texture through the runtime: made as the Metal oracle makes it, from a sparse
//! heap of 16-KiB pages, its mappings mapped before the work by a resource state encoder, and only
//! the authored bytes of its mapped texels uploaded.
//!
//! A mapping names texels; the runtime maps Metal's tiles, so a region that is not a whole number
//! of them (its origin a multiple of the tile, its end one or the level's edge) does not run
//! (`CaseError::Case`), as every executor refuses one of its own tiles. A slice's mip tail is its
//! level from Metal's `firstMipmapInTail`: the tail where Metal keeps the case's one level in
//! one, and that level's tiles where it does not. A tile maps alone on every device that makes
//! the texture, whatever its blocks; a multisampled texture's mapping that would bind part of one
//! of the device's sparse blocks is refused, typed (`Unsupported(SparseMapping)`).

use super::{texture_descriptor, CaseError};
use crate::case::sparse::{SparseMapping, SparseTextureResource};
use crate::literal::LiteralTexture;
use metal2vulkan::bind::TileMapping;
use metal2vulkan::metal::TextureUsage;
use metal2vulkan_runtime::{
    BlitOption, Device, HeapDescriptor, HeapType, Origin, Region, ResourceOptions, Size, Texture,
};
use std::time::Duration;

/// The longest the mapping and upload may take.
const WAIT: Duration = Duration::from_secs(60);

/// The sparse texture of `literal`, whose mappings `sparse` lists, with `usage`.
pub(super) fn texture_with(
    device: &Device,
    literal: &LiteralTexture,
    sparse: &SparseTextureResource,
    usage: TextureUsage,
) -> Result<Texture, CaseError> {
    let label = format!("sparse texture {}", literal.binding);
    let mut descriptor = texture_descriptor(literal, usage)?;
    descriptor.resource_options = ResourceOptions::private();
    // Room for every tile of each slice, whatever its rows waste, and a tail each.
    let room = (literal.bytes.len() as u64 * 4)
        .max(1 << 20)
        .next_multiple_of(16384);
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Sparse,
            room,
            ResourceOptions::private(),
        ))
        .map_err(CaseError::runtime(format!("{label}: new_heap")))?;
    let texture = heap
        .new_texture(&descriptor)
        .map_err(CaseError::runtime(format!("{label}: new_texture")))?;
    let tile = device.sparse_tile_size(descriptor.texture_type, descriptor.pixel_format, 1);
    let tile = [tile.width, tile.height, tile.depth];
    let extent = [descriptor.width, descriptor.height, descriptor.depth];
    let first_tail = texture.first_mipmap_in_tail().unwrap_or(1);
    // Each mapping as Metal's tiles of level 0 and the texels it uploads.
    let mut updates = Vec::new();
    for (index, mapping) in sparse.mappings.iter().enumerate() {
        let (slice, origin, size) = match *mapping {
            SparseMapping::Region {
                slice,
                origin,
                size,
                ..
            } => (slice, origin.map(u64::from), size.map(u64::from)),
            SparseMapping::MipTail { slice } => (slice, [0; 3], extent),
        };
        let whole = (0..3).all(|axis| {
            let end = origin[axis] + size[axis];
            origin[axis] % tile[axis] == 0 && (end % tile[axis] == 0 || end == extent[axis])
        });
        if !whole {
            return Err(format!(
                "{label} mapping {index}: texels {origin:?} + {size:?} are not whole tiles of \
                 {tile:?}"
            )
            .into());
        }
        let tiles = Region::new_3d(
            origin[0] / tile[0],
            origin[1] / tile[1],
            origin[2] / tile[2],
            (origin[0] + size[0]).div_ceil(tile[0]) - origin[0] / tile[0],
            (origin[1] + size[1]).div_ceil(tile[1]) - origin[1] / tile[1],
            (origin[2] + size[2]).div_ceil(tile[2]) - origin[2] / tile[2],
        );
        // Metal keeps the level whole in its tail: the tail's update, for the whole level only.
        let tiles = if first_tail == 0 {
            if origin != [0; 3] || size != extent {
                return Err(format!(
                    "{label} mapping {index}: texels {origin:?} + {size:?} are part of a level \
                     Metal keeps in its mip tail"
                )
                .into());
            }
            Region::new_2d(0, 0, 0, 0)
        } else {
            tiles
        };
        updates.push((u64::from(slice), tiles, origin, size));
    }
    let queue = device
        .new_command_queue()
        .map_err(CaseError::runtime(format!("{label}: new_command_queue")))?;
    let command_buffer = queue
        .command_buffer()
        .map_err(CaseError::runtime(format!("{label}: command_buffer")))?;
    let mut encoder =
        command_buffer
            .resource_state_command_encoder()
            .map_err(CaseError::runtime(format!(
                "{label}: resource_state_command_encoder"
            )))?;
    for &(slice, tiles, ..) in &updates {
        encoder
            .update_texture_mapping(&texture, TileMapping::Map, tiles, 0, slice)
            .map_err(CaseError::runtime(format!(
                "{label}: update_texture_mapping"
            )))?;
    }
    encoder
        .end_encoding()
        .map_err(CaseError::runtime(format!("{label}: end_encoding")))?;
    // The mapped texels' authored bytes, rows tightly packed as the literal holds them.
    let texel = literal.format.bytes_per_pixel() as u64;
    let row = extent[0] * texel;
    let image = row * extent[1];
    let slice_bytes = image * extent[2];
    let source = device
        .new_buffer(
            (literal.bytes.len() as u64).max(1),
            ResourceOptions::shared(),
        )
        .map_err(CaseError::runtime(format!("{label}: new_buffer")))?;
    let contents = source
        .contents()
        .ok_or_else(|| format!("{label}: a Shared buffer without host bytes"))?;
    // SAFETY: the buffer holds the literal's bytes, and no command uses it yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            literal.bytes.as_ptr(),
            contents.as_ptr(),
            literal.bytes.len(),
        )
    };
    let mut blit = command_buffer
        .blit_command_encoder()
        .map_err(CaseError::runtime(format!("{label}: blit_command_encoder")))?;
    for &(slice, _, origin, size) in &updates {
        let offset = slice * slice_bytes + origin[2] * image + origin[1] * row + origin[0] * texel;
        let per_image = if extent[2] > 1 { image } else { 0 };
        blit.copy_from_buffer_to_texture(
            &source,
            offset,
            row,
            per_image,
            Size::new(size[0], size[1], size[2]),
            &texture,
            slice,
            0,
            Origin::new(origin[0], origin[1], origin[2]),
            BlitOption::None,
        )
        .map_err(CaseError::runtime(format!(
            "{label}: copy_from_buffer_to_texture"
        )))?;
    }
    blit.end_encoding()
        .map_err(CaseError::runtime(format!("{label}: end_encoding")))?;
    command_buffer
        .commit()
        .map_err(CaseError::runtime(format!("{label}: commit")))?;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| format!("{label}: the mapping did not complete in {WAIT:?}"))?;
    if let Some(error) = command_buffer.error() {
        return Err(format!("{label}: the mapping failed: {error:?}").into());
    }
    Ok(texture)
}
