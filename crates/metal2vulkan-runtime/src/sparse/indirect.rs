//! Indirect mapping updates: `updateTextureMapping:mode:indirectBuffer: indirectBufferOffset:`,
//! whose regions are GPU memory the work before the update may write.
//!
//! The update's batch waits for a host step (`CommandBufferShared::host_step`), which runs once
//! everything submitted before it on any queue completed: the host reads the arguments then, as
//! Metal's update reads them when it runs, and the step replaces the update with the mapping
//! they state, which the batch's binding applies in order. A `Shared` or `Managed` buffer is read
//! through its mapping (host-coherent memory, which every batch's end made the device's writes
//! visible to); a `Private` one through a copy into a `Shared` buffer that the device's internal
//! command buffer makes and waits for. The device announces the round trip once
//! (`Degradation::HostReadMappingArguments`).

use super::encoder::SparseUpdate;
use super::mapping::UnitUpdate;
use super::SparseProblem;
use crate::buffer::Buffer;
use crate::degradation::Degradation;
use crate::error::{internal, CommandBufferError, Error};
use crate::texture::Texture;
use crate::values::{Region, ResourceOptions};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::TileMapping;

/// The words of one `MTLMapIndirectArguments`: the region's origin and size in tiles, the level
/// and the slice.
const ARGUMENT_WORDS: u64 = 8;

/// Why an indirect update's host read failed the command buffer.
pub(crate) enum ReadFailure {
    /// The command buffer's error for the arguments read.
    Arguments(CommandBufferError),
    /// The read itself failed.
    Read(Error),
}

impl From<Error> for ReadFailure {
    fn from(error: Error) -> ReadFailure {
        ReadFailure::Read(error)
    }
}

/// The updates an indirect update of `texture` states now: its arguments at `offset` of
/// `buffer`, a count and that many regions, mapped or unmapped by `mode` as one update of them
/// all, through a view naming its texture's own tiles. Arguments the encoder would refuse in a
/// direct update fail the command buffer typed: what Metal's validation stops, and regions past the
/// buffer (`CommandBufferError::SparseMappingArguments`), and regions that bind part of a device
/// unit of a texture whose texels no copy writes (`SparseMappingSplit`).
pub(crate) fn resolve(
    device: &Device,
    texture: &Texture,
    mode: TileMapping,
    buffer: &Buffer,
    offset: u64,
) -> Result<Vec<SparseUpdate>, ReadFailure> {
    let sparse = texture
        .sparse_image()
        .ok_or_else(|| internal("an indirect update of a texture that is not sparse"))?;
    let length = buffer.length();
    let count = read_words(device, buffer, offset, 1)?
        .first()
        .copied()
        .map_or(0, u64::from);
    let bytes = count
        .checked_mul(4 * ARGUMENT_WORDS)
        .and_then(|bytes| bytes.checked_add(4));
    if bytes
        .and_then(|bytes| offset.checked_add(bytes))
        .is_none_or(|end| end > length)
    {
        let problem = SparseProblem::IndirectArguments {
            offset,
            bytes: bytes.unwrap_or(u64::MAX),
            length,
        };
        return Err(refused(problem));
    }
    let words = read_words(device, buffer, offset + 4, count * ARGUMENT_WORDS)?;
    // Through a view, the slices are the view's and the levels its texture's, as a direct
    // update's (the encoder refused a view whose tiles are not its texture's).
    let first = u64::from(texture.record().base_layer);
    let mut units = Vec::new();
    for arguments in words.as_chunks::<{ ARGUMENT_WORDS as usize }>().0.iter() {
        let word = |i: usize| u64::from(arguments[i]);
        let region = Region::new_3d(word(0), word(1), word(2), word(3), word(4), word(5));
        let checked = sparse
            .grid
            .checked_region(word(6), word(7) + first, &region)
            .map_err(refused)?;
        units.extend(checked);
    }
    if !sparse.zeroed && !sparse.grid.check_exact(&units) {
        return Err(ReadFailure::Arguments(
            CommandBufferError::SparseMappingSplit,
        ));
    }
    if units.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![SparseUpdate::Units {
        texture: texture.clone(),
        update: UnitUpdate { mode, units },
    }])
}

fn refused(problem: SparseProblem) -> ReadFailure {
    ReadFailure::Arguments(CommandBufferError::SparseMappingArguments(problem))
}

/// Announce on `device`, once, that the host reads mapping arguments.
pub(crate) fn degrade(device: &Device) {
    device.degrade(Degradation::HostReadMappingArguments, || {
        (
            "`update_texture_mapping_indirect` of a sparse texture".to_owned(),
            "its regions are GPU memory the work before it may write, which the host reads once \
             that work completed"
                .to_owned(),
        )
    });
}

/// `count` little-endian 32-bit words of `buffer` from `offset`, which lie in it, as they stand
/// now: everything that wrote them completed.
fn read_words(
    device: &Device,
    buffer: &Buffer,
    offset: u64,
    count: u64,
) -> Result<Vec<u32>, Error> {
    let bytes = count * 4;
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let mut read = vec![0u8; bytes as usize];
    if let Some(contents) = buffer.device_contents() {
        // SAFETY: the buffer's bytes `offset..offset + bytes` lie in it, host-coherent memory
        // (`Shared` or `Managed` without a host copy) whose device writes completed, made visible
        // to the host by their batch's end; nothing writes them now (the work after waits for
        // this step).
        unsafe {
            std::ptr::copy_nonoverlapping(
                contents.as_ptr().wrapping_add(offset as usize),
                read.as_mut_ptr(),
                read.len(),
            );
        }
    } else {
        let copy = device.own_buffer(bytes, ResourceOptions::shared())?;
        let (from, at) = buffer.at(offset);
        let (into, start) = copy.at(0);
        let region = vk::BufferCopy {
            src_offset: at,
            dst_offset: start,
            size: bytes,
        };
        device.submit_internal(|raw, command_buffer| {
            // SAFETY: the internal command buffer is recording; both buffers are live until the
            // call returns, which waits for the submission, and the region lies in both.
            unsafe {
                raw.cmd_copy_buffer(command_buffer, from, into, &[region]);
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ);
                raw.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        })?;
        let contents = copy
            .contents()
            .ok_or_else(|| internal("a Shared copy without host bytes"))?;
        // SAFETY: the copy holds `bytes` host-coherent bytes, which the waited-for copy wrote and
        // its barrier made visible to the host.
        unsafe {
            std::ptr::copy_nonoverlapping(contents.as_ptr(), read.as_mut_ptr(), read.len());
        }
    }
    Ok(read
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}
