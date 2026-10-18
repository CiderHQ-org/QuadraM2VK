//! Sparse textures over ordinary images, the zeroing of unmapped texels, and tile maps.
//!
//! - **Ordinary images.** On a device that binds no sparse memory for a texture (MoltenVK binds
//!   none), where no function reads the image's own residency (the device does not enable
//!   `shaderResourceResidency`), a sparse texture is an ordinary image in memory of its own: its
//!   units are Metal's own (a tile, a slice's tail), so every mapping update is exact, and its
//!   memory is its image's whatever is mapped.
//! - **Zeroing.** Every sparse texture a copy writes keeps the texels of each unmapped Metal unit
//!   inside memory it holds at zero: an ordinary image's from when it is made, a sparse image's
//!   in each device unit bound while some Metal unit it holds texels of is mapped. A batch zeroes
//!   each device unit it binds and the texels of each Metal unit it unmaps inside a unit that stays
//!   bound ([`Mappings`]), and each blit that writes the texture is followed by the zeroing of the
//!   unmapped Metal units it wrote there (its scrub, `Deferred::SparseScrub`), so a read of an
//!   unmapped texel reads what Metal's reads, zero, and a blit's write to one is dropped as Metal
//!   drops it (a blit into unmapped tiles leaves them reading zero). A sparse
//!   image's device unit that holds no mapped tile reads the device's value, and a write into one
//!   is the device's. A function's writes to a sparse texture never land (the module drops them, as
//!   Metal does), and a render pass into an ordinary image is refused.
//! - **Tile maps.** A function reads a sparse texture's residency from a tile map
//!   (`bind::SparseTileMap`) on every device, at each slot whose type has one and whose observed
//!   accesses each name their texels: a buffer with a bit per tile, which each batch that changes
//!   the texture's mappings rewrites after it applies them, before its work, so a command reads the
//!   mappings Metal's would. Every texture whose type has a map keeps one.
//!
//! A batch's zeroing and map writes are one command buffer ([`Mappings`]), the first of its work
//! in place of its first resource state encoder's commands, after everything submitted before it
//! on every queue, as a bind is (`bind.rs`); the hazard tracker sees it write the textures and
//! maps. A scrub ([`Scrub`]) is a segment of the encoder that wrote, recorded as its batch is
//! submitted.

use super::bind::Moved;
use super::mapping::{MetalUnit, SparseState, TexelBox, VkUnit};
use super::texture::{SparseImage, SparseShape};
use crate::buffer::Buffer;
use crate::copy_plan::ImageRegion;
use crate::device::lock;
use crate::error::{internal, Error};
use crate::exit_guard::{self, Call};
use crate::memory::Placement;
use crate::render_pass::Target;
use crate::resource::ResourceKey;
use crate::texture::{buffer_image_copy, copy_units, Texture};
use crate::tracker::{Access, Use};
use crate::upload::Uploads;
use crate::values::ResourceOptions;
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{SparseTileMap, TileMapping};
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::TextureType;
use metal2vulkan::vk::{ImageAspectFlags, ImageUsageFlags};
use std::collections::{BTreeSet, HashSet};
use std::ops::Range;

/// How a sparse texture's texels are held.
pub(crate) enum Backing {
    /// A sparse resident image, its memory bound by device unit.
    Bound,
    /// An ordinary image in memory of its own, its unmapped tiles kept at zero.
    Emulated { memory: Placement },
}

/// A sparse texture's tile map: the buffer a function reads its residency from, and the map
/// with nothing mapped, which each rewrite starts from.
pub(crate) struct TileMapBuffer {
    pub(crate) buffer: Buffer,
    empty: SparseTileMap,
}

impl TileMapBuffer {
    /// The tile map of a sparse texture of `shape` with `layers` slices, tiles of `tile` texels
    /// and its tail from `first_tail`, on `device`, or `None` where its type has none (a cube or
    /// multisampled texture, which the format does not lay out). Its buffer holds no bytes yet:
    /// [`SparseImage::tile_map_bytes`] are staged when the texture is made, for the device's next
    /// submission to copy in.
    pub(crate) fn new(
        device: &Device,
        shape: &SparseShape,
        layers: u32,
        tile: [u64; 3],
        first_tail: u64,
    ) -> Result<Option<TileMapBuffer>, Error> {
        let Some(empty) = tile_map_layout(shape, layers, tile, first_tail) else {
            return Ok(None);
        };
        let buffer = device.own_buffer(empty.byte_len(), ResourceOptions::private())?;
        Ok(Some(TileMapBuffer { buffer, empty }))
    }
}

/// The tile map, nothing mapped, of a sparse texture of `shape` with `layers` slices, tiles of
/// `tile` texels and its tail from `first_tail`, or `None` where its type has none (a cube or
/// multisampled texture, which the format does not lay out).
fn tile_map_layout(
    shape: &SparseShape,
    layers: u32,
    tile: [u64; 3],
    first_tail: u64,
) -> Option<SparseTileMap> {
    let n = |value: u64| u32::try_from(value).unwrap_or(u32::MAX);
    let slices = match shape.texture_type {
        TextureType::Type2DArray => layers,
        _ => 1,
    };
    SparseTileMap::new(
        shape.texture_type,
        shape.extent.map(n),
        n(shape.levels),
        slices,
        tile.map(n),
        n(first_tail),
    )
    .ok()
}

impl SparseImage {
    /// Whether the texture is an ordinary image whose unmapped tiles the runtime zeroes.
    pub(crate) fn emulated(&self) -> bool {
        matches!(self.backing, Backing::Emulated { .. })
    }

    /// Its tile map, where its type has one.
    pub(crate) fn tile_map(&self) -> Option<&TileMapBuffer> {
        self.tile_map.as_ref()
    }

    /// The bytes of its tile map with `mapped` mapped (none for a texture without one).
    pub(crate) fn tile_map_bytes(&self, mapped: &HashSet<MetalUnit>) -> Vec<u8> {
        let Some(tile_map) = self.tile_map() else {
            return Vec::new();
        };
        let mut map = tile_map.empty.clone();
        for &unit in mapped {
            let (level, slice, at) = match unit {
                MetalUnit::Tile { level, slice, at } => (level, slice, at),
                MetalUnit::Tail { slice } => (self.grid.first_tail(), slice, [0; 3]),
            };
            // The units are the texture's own (an update checked them), so the map takes them.
            let _ = map.update(TileMapping::Map, level, slice, at, [1, 1, 1]);
        }
        map.to_bytes()
    }

    /// Whether device unit `unit` holds memory in `state`: an ordinary image's every unit, a
    /// sparse image's while bound.
    pub(crate) fn held(&self, unit: VkUnit, state: &SparseState) -> bool {
        self.emulated() || state.bound.contains_key(&unit)
    }
}

/// Record into `command_buffer` the copies of zeros that zero `boxes` of `texture`, a sparse
/// texture whose image a copy writes (nothing for another), in the `GENERAL` layout, each box in
/// pieces of at most one Metal tile, whose bytes the zeros hold.
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass; the texture is live and the memory of
/// each box bound; the caller orders the copies after what used the image before and before what
/// uses it after.
pub(crate) unsafe fn record_zeroing(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    texture: &Texture,
    boxes: impl IntoIterator<Item = TexelBox>,
) -> Result<(), Error> {
    let Some(image) = texture.sparse_image().filter(|image| image.zeroed) else {
        return Ok(());
    };
    let raw = texture.raw_image()?;
    let plan = texture.image_plan()?;
    let zeros = &image.zeros;
    let (source, start) = zeros.at(0);
    let copy_units = copy_units(plan.format, plan.pixel_format);
    let tile = image.grid.tile32();
    let mut copies = Vec::new();
    for piece in boxes.into_iter().flat_map(|b| b.pieces(tile)) {
        for &(aspect, bytes, [bw, bh]) in &copy_units {
            let length = u64::from(piece.size[0].div_ceil(bw))
                * u64::from(piece.size[1].div_ceil(bh))
                * u64::from(piece.size[2])
                * u64::from(bytes);
            if length > zeros.length() {
                return Err(internal("a sparse tile larger than its heap's zeros"));
            }
            let region = ImageRegion {
                aspect,
                level: piece.level,
                layer: piece.layer,
                layers: 1,
                origin: piece.origin,
                size: piece.size,
            };
            let mut copy = buffer_image_copy(&region);
            copy.buffer_offset = start;
            copies.push(copy);
        }
    }
    if copies.is_empty() {
        return Ok(());
    }
    // SAFETY: the caller's; the zeros are a live buffer of the device holding every copy's
    // bytes at `start`, and each region lies in the texture's image.
    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            source,
            raw,
            vk::ImageLayout::GENERAL,
            &copies,
        );
    }
    Ok(())
}

/// A barrier from every command before to every command after: all memory writes made
/// available and visible.
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass.
pub(crate) unsafe fn record_full_barrier(device: &ash::Device, command_buffer: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    // SAFETY: the caller's.
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

impl Device {
    /// Whether a function may read a sparse texture's residency from its image (the device
    /// enables `shaderResourceResidency`, so a slot without a tile map takes the sparse arm): an
    /// ordinary image would answer resident everywhere there, so every sparse texture must be a
    /// sparse image.
    pub(crate) fn reads_image_residency(&self) -> bool {
        self.compile_device()
            .profile()
            .enabled()
            .features
            .contains(&VkFeature::ShaderResourceResidency)
    }
}

/// A box of a texture's image a command writes: a level, its layers, and the texels from `origin`
/// of `size`.
#[derive(Clone, Debug)]
pub(crate) struct Written {
    pub(crate) level: u32,
    pub(crate) layers: Range<u32>,
    pub(crate) origin: [u32; 3],
    pub(crate) size: [u32; 3],
}

impl Written {
    /// The box a copy region covers.
    pub(crate) fn of(region: &ImageRegion) -> Written {
        Written {
            level: region.level,
            layers: region.layer..region.layer + region.layers,
            origin: region.origin,
            size: region.size,
        }
    }
}

/// A blit's or a render pass's writes of a sparse texture a copy writes: Metal drops a blit's write
/// to an unmapped tile and a pass's store (a Metal sparse probe on the M3 Max and
/// an M4), so where the command wrote an unmapped tile inside memory the texture holds the runtime
/// zeroes it again right after, recorded as the batch is submitted, with the mappings the batch
/// applied (`Deferred::SparseScrub`). A write into a device unit that holds no memory is the
/// device's (dropped: Vulkan discards a write to an unbound block).
pub(crate) struct Scrub {
    texture: Texture,
    written: Vec<Written>,
}

impl Scrub {
    /// The scrub of `written` of `texture` (made only for one), or `None` for a texture that is
    /// not a sparse texture the runtime zeroes.
    pub(crate) fn of(texture: &Texture, written: impl FnOnce() -> Vec<Written>) -> Option<Scrub> {
        texture
            .sparse_image()
            .filter(|image| image.zeroed)
            .map(|_| Scrub {
                texture: texture.clone(),
                written: written(),
            })
    }

    /// The scrubs of a render pass's writes of `targets` (its attachments, resolve targets and the
    /// textures copied into after it) over its render area `extent` from the origin: Metal drops a
    /// pass's stores into unmapped tiles, of a clear, a draw and a resolve alike (a Metal sparse
    /// probe on the M3 Max and an M4), as a blit's, so each sparse texture the runtime zeroes is
    /// scrubbed after the pass. Inside the pass, the draws see what the draws before them stored,
    /// as in Metal's tile memory.
    pub(crate) fn of_pass<'a>(
        targets: impl IntoIterator<Item = &'a Target>,
        extent: [u32; 2],
    ) -> Vec<Scrub> {
        targets
            .into_iter()
            .filter_map(|target| {
                Scrub::of(&target.texture, || {
                    vec![Written {
                        level: target.level,
                        layers: target.layer..target.layer + target.layers,
                        origin: [0; 3],
                        size: [extent[0], extent[1], 1],
                    }]
                })
            })
            .collect()
    }

    /// Record into `command_buffer` the zeroing of the unmapped units the writes touched inside
    /// memory the texture holds, as mapped now, between barriers from everything before and to
    /// everything after.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass; the texture is live in the
    /// `GENERAL` layout, and its mappings are those the batch the scrub is in applied.
    pub(crate) unsafe fn record(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), Error> {
        let Some(image) = self.texture.sparse_image() else {
            return Ok(());
        };
        let boxes = {
            let state = lock(&image.state);
            let unmapped: BTreeSet<MetalUnit> = self
                .written
                .iter()
                .flat_map(|written| image.grid.units_within(written))
                .filter(|unit| !state.mapped.contains(unit))
                .collect();
            image
                .grid
                .unmapped_boxes(unmapped, |unit| image.held(unit, &state))
        };
        // SAFETY: the caller's; each box lies in memory the texture holds.
        unsafe {
            record_full_barrier(device, command_buffer);
            record_zeroing(device, command_buffer, &self.texture, boxes)?;
            record_full_barrier(device, command_buffer);
        }
        exit_guard::called(Call::CmdPipelineBarrier);
        exit_guard::called(Call::CmdCopyBufferToImage);
        Ok(())
    }
}

/// One texture's work after a batch's updates applied (indexed as the batch's textures): the
/// texels of the device units it bound to new memory, zeroed whole first; the texels of the Metal
/// units it unmapped inside memory the texture still holds, zeroed after the moves' copies; and
/// its tile map's bytes.
struct TextureWork {
    texture: Texture,
    fresh: BTreeSet<TexelBox>,
    unmapped: BTreeSet<TexelBox>,
    map: Vec<u8>,
}

/// One copy of a move's texels, from a box of the batch's texture `from` to one of `to`.
struct MoveCopy {
    from: usize,
    source: TexelBox,
    to: usize,
    destination: TexelBox,
}

/// A batch's work on its sparse textures and tile maps after its updates applied, on the textures
/// the runtime zeroes: each device unit the batch bound to new memory zeroed whole; each move's
/// texels copied from its source tile to its destination, one tile at a time in the batch's order,
/// as Metal's move takes them (a texture the runtime does not zero moves whole device units, their
/// memory with them, `bind.rs`); each Metal unit it unmapped inside memory the texture still holds
/// zeroed; and each changed tile map rewritten.
#[derive(Default)]
pub(crate) struct Mappings {
    work: Vec<TextureWork>,
    copies: Vec<MoveCopy>,
}

impl Mappings {
    /// The work of `textures` once the units `changed` names changed, the device units `fresh`
    /// names were bound to new memory, the steps `moved` moved Metal's units in order, and the
    /// device units `kept` stay bound through the batch for the copies.
    pub(crate) fn of(
        textures: &[&Texture],
        changed: impl IntoIterator<Item = (usize, MetalUnit)>,
        fresh: impl IntoIterator<Item = (usize, VkUnit)>,
        (moved, kept): (&[Moved], &BTreeSet<(usize, VkUnit)>),
    ) -> Mappings {
        let mut touched: Vec<BTreeSet<MetalUnit>> = vec![BTreeSet::new(); textures.len()];
        for (index, unit) in changed {
            touched[index].insert(unit);
        }
        let mut binds: Vec<BTreeSet<VkUnit>> = vec![BTreeSet::new(); textures.len()];
        for (index, unit) in fresh {
            binds[index].insert(unit);
        }
        let images: Vec<Option<&SparseImage>> = textures
            .iter()
            .map(|texture| texture.sparse_image().filter(|image| image.zeroed))
            .collect();
        let mut work = Vec::new();
        for (index, (texture, units)) in textures.iter().zip(touched).enumerate() {
            let (fresh, unmapped, map) = match texture.sparse_image() {
                Some(image) if !units.is_empty() => {
                    let state = lock(&image.state);
                    let (fresh, unmapped) = if image.zeroed {
                        let bound = &binds[index];
                        // Held once the batch's binds apply: what the plan bound (`bind.rs`).
                        let held = |unit: VkUnit| {
                            !bound.contains(&unit)
                                && (image.emulated()
                                    || image.grid.wanted(unit, &state.mapped)
                                    || kept.contains(&(index, unit)))
                        };
                        let unmapped = units
                            .iter()
                            .copied()
                            .filter(|unit| !state.mapped.contains(unit));
                        (
                            image.grid.bound_boxes(bound),
                            image.grid.unmapped_boxes(unmapped, held),
                        )
                    } else {
                        (BTreeSet::new(), BTreeSet::new())
                    };
                    (fresh, unmapped, image.tile_map_bytes(&state.mapped))
                }
                _ => (BTreeSet::new(), BTreeSet::new(), Vec::new()),
            };
            work.push(TextureWork {
                texture: (*texture).clone(),
                fresh,
                unmapped,
                map,
            });
        }
        let mut copies = Vec::new();
        for step in moved {
            let (Some(from), Some(to)) = (images[step.from], images[step.to]) else {
                continue;
            };
            copies.extend(
                from.grid
                    .move_boxes(step.source, &to.grid, step.destination)
                    .into_iter()
                    .map(|(source, destination)| MoveCopy {
                        from: step.from,
                        source,
                        to: step.to,
                        destination,
                    }),
            );
        }
        Mappings { work, copies }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.copies.is_empty()
            && self.work.iter().all(|work| {
                work.fresh.is_empty() && work.unmapped.is_empty() && work.map.is_empty()
            })
    }

    /// What the work uses, for the hazard tracker, in the transfer stage: each texture it zeroes
    /// or copies into written whole, each it only copies from read whole, and each tile map it
    /// rewrites written whole.
    pub(crate) fn uses(&self) -> Result<Vec<Use<ResourceKey>>, Error> {
        let mut reads = vec![false; self.work.len()];
        let mut writes: Vec<bool> = self
            .work
            .iter()
            .map(|work| !work.fresh.is_empty() || !work.unmapped.is_empty())
            .collect();
        for copy in &self.copies {
            reads[copy.from] = true;
            writes[copy.to] = true;
        }
        let mut uses = Vec::new();
        for ((work, read), write) in self.work.iter().zip(reads).zip(writes) {
            if read || write {
                let mut access = vk::AccessFlags2::empty();
                if read {
                    access |= vk::AccessFlags2::TRANSFER_READ;
                }
                if write {
                    access |= vk::AccessFlags2::TRANSFER_WRITE;
                }
                let access = Access::new(vk::PipelineStageFlags2::TRANSFER, access);
                let plan = work.texture.image_plan()?;
                uses.push(
                    work.texture
                        .image_use(0..plan.levels, 0..plan.layers, access, write)?,
                );
            }
            let tile_map = work.texture.sparse_image().and_then(SparseImage::tile_map);
            if let (false, Some(tile_map)) = (work.map.is_empty(), tile_map) {
                let length = tile_map.buffer.length();
                uses.push(crate::blit::writes(&tile_map.buffer, 0..length));
            }
        }
        Ok(uses)
    }

    /// Record the work into `command_buffer`, between barriers from everything before and to
    /// everything after, each step after the one before it: the new units zeroed, each move's
    /// copy in order, the unmapped units zeroed, and each tile map's bytes copied from `uploads`,
    /// the command buffer's arena (inside the scheduler's section, whose entry points hold
    /// copies, not buffer updates).
    ///
    /// # Safety
    ///
    /// `command_buffer` is a command buffer of `device` recording outside a render pass; every
    /// texture is live in the `GENERAL` layout with the batch's binds applied, and the caller
    /// orders the command buffer after everything submitted before the batch on every queue.
    pub(crate) unsafe fn record(
        &self,
        device: &Device,
        uploads: &mut Uploads,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), Error> {
        let raw = device.shared().vk();
        // SAFETY: the caller's.
        unsafe { record_full_barrier(raw, command_buffer) };
        exit_guard::called(Call::CmdPipelineBarrier);
        for work in &self.work {
            // SAFETY: the caller's; each box lies in a unit the batch bound.
            unsafe {
                record_zeroing(
                    raw,
                    command_buffer,
                    &work.texture,
                    work.fresh.iter().copied(),
                )?
            };
            exit_guard::called(Call::CmdCopyBufferToImage);
        }
        for copy in &self.copies {
            // SAFETY: the caller's; each step's writes finish before the next reads them.
            unsafe {
                record_full_barrier(raw, command_buffer);
                record_move_copy(
                    raw,
                    command_buffer,
                    (&self.work[copy.from].texture, copy.source),
                    (&self.work[copy.to].texture, copy.destination),
                )?;
            }
            exit_guard::called(Call::CmdPipelineBarrier);
            exit_guard::called(Call::CmdCopyImage);
        }
        // SAFETY: the caller's.
        unsafe { record_full_barrier(raw, command_buffer) };
        for work in &self.work {
            // SAFETY: the caller's; each box lies in memory the texture holds after the binds.
            unsafe {
                record_zeroing(
                    raw,
                    command_buffer,
                    &work.texture,
                    work.unmapped.iter().copied(),
                )?
            };
            exit_guard::called(Call::CmdCopyBufferToImage);
            let tile_map = work.texture.sparse_image().and_then(SparseImage::tile_map);
            if let (false, Some(tile_map)) = (work.map.is_empty(), tile_map) {
                let bytes = &work.map;
                let upload = uploads.upload(device, bytes, bytes.len() as u64)?;
                let (target, start) = tile_map.buffer.at(0);
                let copy = vk::BufferCopy {
                    src_offset: upload.offset(),
                    dst_offset: start,
                    size: bytes.len() as u64,
                };
                // SAFETY: the caller's; the upload holds the bytes and the map's buffer has
                // room for them; the command buffer holds the arena until it completes.
                unsafe { raw.cmd_copy_buffer(command_buffer, upload.raw(), target, &[copy]) };
                exit_guard::called(Call::CmdCopyBuffer);
            }
        }
        // SAFETY: the caller's.
        unsafe { record_full_barrier(raw, command_buffer) };
        Ok(())
    }
}

/// Record into `command_buffer` the copy of box `source` of texture `from` to box `destination`
/// of texture `to`, both sparse textures the runtime zeroes, of one format, in the `GENERAL`
/// layout, each of the format's aspects; nothing where `from`'s image is not a copy's source.
///
/// # Safety
///
/// `command_buffer` is recording outside a render pass; both textures are live and both boxes
/// lie in their images, of one extent; the caller orders the copy.
unsafe fn record_move_copy(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    (from, source): (&Texture, TexelBox),
    (to, destination): (&Texture, TexelBox),
) -> Result<(), Error> {
    let plan = from.image_plan()?;
    if !plan.usage.contains(ImageUsageFlags::TRANSFER_SRC) {
        return Ok(());
    }
    let offset = |origin: [u32; 3]| vk::Offset3D {
        x: origin[0] as i32,
        y: origin[1] as i32,
        z: origin[2] as i32,
    };
    let subresource = |aspect: ImageAspectFlags, level, layer| vk::ImageSubresourceLayers {
        aspect_mask: aspect.into(),
        mip_level: level,
        base_array_layer: layer,
        layer_count: 1,
    };
    let copies: Vec<vk::ImageCopy> = copy_units(plan.format, plan.pixel_format)
        .into_iter()
        .map(|(aspect, _, _)| vk::ImageCopy {
            src_subresource: subresource(aspect, source.level, source.layer),
            src_offset: offset(source.origin),
            dst_subresource: subresource(aspect, destination.level, destination.layer),
            dst_offset: offset(destination.origin),
            extent: vk::Extent3D {
                width: source.size[0],
                height: source.size[1],
                depth: source.size[2],
            },
        })
        .collect();
    // SAFETY: the caller's; each region lies in its image, which a copy reads (checked) or
    // writes (the runtime zeroes it by copies).
    unsafe {
        device.cmd_copy_image(
            command_buffer,
            from.raw_image()?,
            vk::ImageLayout::GENERAL,
            to.raw_image()?,
            vk::ImageLayout::GENERAL,
            &copies,
        );
    }
    Ok(())
}
