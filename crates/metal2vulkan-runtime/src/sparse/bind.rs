//! A batch's sparse binding, on the scheduler's thread, in its order, inside the submission's
//! section of the exit guard.
//!
//! The batch's updates apply to their textures' Metal state in order (the heap's room counted
//! as they map; a move takes a unit's mapping from one texture to another and counts nothing),
//! then each device unit they touched is bound where some Metal unit it holds texels of is
//! mapped and unbound where none is, as one `vkQueueBindSparse` on the batch's `VkQueue`. The bind
//! waits for the batch's event and binary semaphore waits and for every `VkQueue`'s last
//! submitted value, and signals the queue's timeline one value past its last: the batch's work
//! waits for that value. A bound unit takes its own placement from the sub-allocator; an unbound
//! unit's placement is the command buffer's to release once it completed. Nothing applies when
//! anything fails: the Metal state, the heap's count and the bound units stay as they were.
//!
//! What the updates leave is work for the batch to record before its encoders ([`Mappings`]):
//! each unit bound zeroed whole and each Metal unit unmapped inside a unit still bound zeroed, so
//! a newly mapped tile and a moved one read zero, one of the answers Metal leaves undefined, and
//! an unmapped one zero, Metal's; and each tile map rewritten. An emulated texture binds nothing. A
//! batch whose updates change no device unit makes no bind.

use super::emulated::{Backing, Mappings};
use super::encoder::SparseUpdate;
use super::mapping::{MetalUnit, UnitUpdate, VkUnit};
use super::texture::SparseImage;
use crate::device::lock;
use crate::device::relay::By;
use crate::error::{internal, Error};
use crate::exit_guard::{self, Call};
use crate::heap::Heap;
use crate::memory::{MemoryClass, Placement};
use crate::texture::Texture;
use crate::Device;
use ash::vk;
use metal2vulkan::bind::TileMapping;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Why a batch's binding failed; nothing of it applied.
#[derive(Debug)]
pub(crate) enum BindFailure {
    /// A heap filled inside a device unit of several tiles of a texture the runtime does not
    /// zero.
    Split,
    /// The memory of a unit could not be placed.
    Memory(Error),
    /// `vkQueueBindSparse` failed.
    Vulkan(vk::Result),
}

/// Placements a binding unbound: freed when the command buffer that unbound them completes.
pub(crate) struct Unbound {
    device: Device,
    placements: Vec<Placement>,
}

impl Drop for Unbound {
    fn drop(&mut self) {
        // Once the process is exiting, the memory is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        let mut memory = lock(&shared.memory);
        for placement in &self.placements {
            memory.free(shared.vk(), placement);
        }
    }
}

impl Drop for SparseImage {
    fn drop(&mut self) {
        // The image is gone (`Image`'s drop): its bound units' memory goes with it.
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut memory = lock(&shared.memory);
        for placement in state.bound.values() {
            memory.free(shared.vk(), placement);
        }
        if let Backing::Emulated { memory: placement } = &self.backing {
            memory.free(shared.vk(), placement);
        }
    }
}

/// One Metal unit an update changed, to undo.
struct Changed {
    /// The index of its image among the batch's.
    image: usize,
    unit: MetalUnit,
    /// It was mapped by the update (else unmapped).
    mapped: bool,
    /// Its heap's count changed with it (a map or an unmap, not a move).
    counted: bool,
}

/// One Metal unit a move took, in the batch's order: from unit `source` of image `from` to unit
/// `destination` of image `to`.
#[derive(Clone, Copy)]
pub(crate) struct Moved {
    pub(crate) from: usize,
    pub(crate) source: MetalUnit,
    pub(crate) to: usize,
    pub(crate) destination: MetalUnit,
}

/// The memory a device unit is bound to by a batch.
enum UnitMemory {
    /// Unbound.
    None,
    /// A new placement, its texels zeroed by the batch where the runtime zeroes the texture.
    Fresh(Placement),
    /// The placement of a unit a move vacates, its texels with it (a texture the runtime does not
    /// zero moves whole units).
    Donated(Donor),
}

/// A bound device unit a move vacates whose memory another unit takes: unit `unit` of the
/// batch's image `image`, bound to `memory` at `offset`.
#[derive(Clone, Copy)]
struct Donor {
    image: usize,
    unit: VkUnit,
    memory: vk::DeviceMemory,
    offset: u64,
}

/// One device unit's change: bound to memory, or unbound.
struct UnitChange {
    image: usize,
    unit: VkUnit,
    memory: UnitMemory,
}

impl UnitChange {
    /// The memory and offset it binds (null for an unbind).
    fn bind(&self) -> (vk::DeviceMemory, u64) {
        match &self.memory {
            UnitMemory::None => (vk::DeviceMemory::null(), 0),
            UnitMemory::Fresh(placement) => (placement.memory, placement.offset),
            UnitMemory::Donated(donor) => (donor.memory, donor.offset),
        }
    }
}

/// One update, its textures as indexes into the batch's images.
enum Op<'a> {
    Units {
        image: usize,
        update: &'a UnitUpdate,
    },
    Move {
        from: usize,
        to: usize,
        pairs: &'a [(MetalUnit, MetalUnit)],
    },
}

/// The batch's sparse images, each once, with their textures.
#[derive(Default)]
struct Images<'a> {
    textures: Vec<&'a Texture>,
    images: Vec<&'a SparseImage>,
}

impl<'a> Images<'a> {
    /// The index of `texture`'s image, added when new.
    fn of(&mut self, texture: &'a Texture) -> Result<usize, BindFailure> {
        let image = texture.sparse_image().ok_or(BindFailure::Memory(internal(
            "a sparse update of a texture that is not sparse",
        )))?;
        if let Some(index) = self
            .images
            .iter()
            .position(|known| std::ptr::eq(*known, image))
        {
            return Ok(index);
        }
        self.textures.push(texture);
        self.images.push(image);
        Ok(self.images.len() - 1)
    }
}

/// What a batch's binding did: the placements it unbound, for the command buffer to hold until
/// it completes; whether it bound anything (and so signalled the queue's timeline one value past
/// its last); and the work on its textures and tile maps its updates leave, for the batch to record
/// before its encoders.
pub(crate) struct Bound {
    pub(crate) unbound: Unbound,
    pub(crate) signalled: bool,
    pub(crate) mappings: Mappings,
}

/// Apply `updates` to their textures' Metal state in order, and bind the device units that
/// change on `VkQueue` `queue` of `device`, after `waits` (the batch's semaphores and values; a
/// binary semaphore's value is ignored) and each `VkQueue`'s last submitted value `values`,
/// signalling the queue's timeline at `values[queue] + 1`; where no device unit changes (an
/// emulated texture binds none), nothing is bound or signalled.
pub(crate) fn bind_batch(
    device: &Device,
    queue: usize,
    updates: &[SparseUpdate],
    waits: &[(vk::Semaphore, u64)],
    values: &[u64],
) -> Result<Bound, BindFailure> {
    let mut known = Images::default();
    let mut ops = Vec::new();
    for update in updates {
        ops.push(match update {
            SparseUpdate::Units { texture, update } => Op::Units {
                image: known.of(texture)?,
                update,
            },
            SparseUpdate::Move {
                source,
                destination,
                pairs,
            } => Op::Move {
                from: known.of(source)?,
                to: known.of(destination)?,
                pairs,
            },
            SparseUpdate::Indirect { .. } => {
                let error = internal("an indirect sparse update its host step did not read");
                return Err(BindFailure::Memory(error));
            }
        });
    }
    let Images { textures, images } = known;
    let (changed, moved) = apply(&ops, &images).map_err(|changed| {
        undo(&images, &changed);
        BindFailure::Split
    })?;
    let kept = kept_units(&images, &moved);
    let mut changes = match plan(device, &images, (&changed, &moved), &kept) {
        Ok(changes) => changes,
        Err(failure) => {
            undo(&images, &changed);
            return Err(failure);
        }
    };
    let mappings = Mappings::of(
        &textures,
        changed.iter().map(|change| (change.image, change.unit)),
        changes
            .iter()
            .filter(|change| matches!(change.memory, UnitMemory::Fresh(_)))
            .map(|change| (change.image, change.unit)),
        (&moved, &kept),
    );
    if changes.is_empty() {
        return Ok(Bound {
            unbound: Unbound {
                device: device.clone(),
                placements: Vec::new(),
            },
            signalled: false,
            mappings,
        });
    }
    let shared = device.shared();
    let mut blocks: Vec<Vec<vk::SparseImageMemoryBind>> = vec![Vec::new(); images.len()];
    let mut tails: Vec<Vec<vk::SparseMemoryBind>> = vec![Vec::new(); images.len()];
    for change in &changes {
        let image = images[change.image];
        let (memory, offset) = change.bind();
        match change.unit {
            VkUnit::Block { level, layer, at } => {
                let (origin, size) = image.grid.block_texels(level, at);
                blocks[change.image].push(
                    vk::SparseImageMemoryBind::default()
                        .subresource(vk::ImageSubresource {
                            aspect_mask: image.grid.layout.aspect,
                            mip_level: level,
                            array_layer: layer,
                        })
                        .offset(vk::Offset3D {
                            x: origin[0] as i32,
                            y: origin[1] as i32,
                            z: origin[2] as i32,
                        })
                        .extent(vk::Extent3D {
                            width: size[0],
                            height: size[1],
                            depth: size[2],
                        })
                        .memory(memory)
                        .memory_offset(offset),
                );
            }
            VkUnit::Tail { layer } => tails[change.image].push(
                vk::SparseMemoryBind::default()
                    .resource_offset(
                        image.grid.layout.mip_tail_offset
                            + u64::from(layer) * image.grid.layout.mip_tail_stride,
                    )
                    .size(image.grid.layout.mip_tail_size)
                    .memory(memory)
                    .memory_offset(offset),
            ),
        }
    }
    let raws: Vec<vk::Image> = textures
        .iter()
        .map(|texture| texture.raw_image().unwrap_or(vk::Image::null()))
        .collect();
    let image_binds: Vec<vk::SparseImageMemoryBindInfo<'_>> = blocks
        .iter()
        .zip(&raws)
        .filter(|(binds, _)| !binds.is_empty())
        .map(|(binds, &raw)| {
            vk::SparseImageMemoryBindInfo::default()
                .image(raw)
                .binds(binds)
        })
        .collect();
    let opaque_binds: Vec<vk::SparseImageOpaqueMemoryBindInfo<'_>> = tails
        .iter()
        .zip(&raws)
        .filter(|(binds, _)| !binds.is_empty())
        .map(|(binds, &raw)| {
            vk::SparseImageOpaqueMemoryBindInfo::default()
                .image(raw)
                .binds(binds)
        })
        .collect();
    let mut wait_semaphores: Vec<vk::Semaphore> = waits.iter().map(|&(s, _)| s).collect();
    let mut wait_values: Vec<u64> = waits.iter().map(|&(_, value)| value).collect();
    for (slot, &value) in shared.queues.iter().zip(values) {
        if value > 0 {
            wait_semaphores.push(slot.timeline);
            wait_values.push(value);
        }
    }
    let signal_semaphores = [shared.queues[queue].timeline];
    let signal_values = [values[queue] + 1];
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
        .wait_semaphore_values(&wait_values)
        .signal_semaphore_values(&signal_values);
    let info = vk::BindSparseInfo::default()
        .wait_semaphores(&wait_semaphores)
        .image_binds(&image_binds)
        .image_opaque_binds(&opaque_binds)
        .signal_semaphores(&signal_semaphores)
        .push_next(&mut timeline);
    // SAFETY: every image is live (the command buffer holds its texture) and was made with the
    // sparse flags; every bind lies in its image (whole blocks clipped at the level's edge, a
    // layer's tail) and in its memory (a placement of the unit's bytes at the image's alignment,
    // of a memory type it takes: a new one, or a moved unit's, checked alike); every wait is for
    // a value already submitted; the signal raises the queue's timeline past every value
    // submitted; the queue binds sparse memory (the device's support) and is used under its
    // lock. Where the queue's failures are relayed, one of the bind's loses the device
    // (`device/relay.rs`).
    shared.counts.queue_submitted(0, 0);
    let bound = shared.use_queue(queue, By::Runtime, |vk_queue| unsafe {
        shared
            .vk()
            .queue_bind_sparse(vk_queue, &[info], vk::Fence::null())
    });
    exit_guard::called(Call::QueueBindSparse);
    if let Err(error) = bound {
        free_fresh(device, &changes);
        undo(&images, &changed);
        return Err(BindFailure::Vulkan(error));
    }
    // A moved unit's placement goes to its destination first; its source's unbind then finds
    // nothing to release.
    changes.sort_by_key(|change| matches!(change.memory, UnitMemory::None));
    let mut unbound = Vec::new();
    for change in changes.drain(..) {
        match change.memory {
            UnitMemory::Fresh(placement) => {
                lock(&images[change.image].state)
                    .bound
                    .insert(change.unit, placement);
            }
            UnitMemory::Donated(donor) => {
                let placement = lock(&images[donor.image].state).bound.remove(&donor.unit);
                if let Some(placement) = placement {
                    lock(&images[change.image].state)
                        .bound
                        .insert(change.unit, placement);
                }
            }
            UnitMemory::None => {
                unbound.extend(lock(&images[change.image].state).bound.remove(&change.unit));
            }
        }
    }
    Ok(Bound {
        unbound: Unbound {
            device: device.clone(),
            placements: unbound,
        },
        signalled: true,
        mappings,
    })
}

/// Free the new placements of `changes`, which their bind did not take.
fn free_fresh(device: &Device, changes: &[UnitChange]) {
    let shared = device.shared();
    let mut memory = lock(&shared.memory);
    for change in changes {
        if let UnitMemory::Fresh(placement) = &change.memory {
            memory.free(shared.vk(), placement);
        }
    }
}

/// Apply each update to its textures' Metal state in order: a map takes its heap's room (a heap
/// that fills leaves the rest of that update unmapped), an unmap gives it back, and a move takes
/// each source unit's mapping to its destination unit where that one is unmapped as it moves,
/// one unit at a time in the move's order, leaving the heaps' counts as they are (as Metal's).
/// What changed, and what moved in order; on a split device unit of a texture the runtime does
/// not zero, `Err` with what changed, for the caller to undo.
fn apply(
    ops: &[Op<'_>],
    images: &[&SparseImage],
) -> Result<(Vec<Changed>, Vec<Moved>), Vec<Changed>> {
    let mut changed = Vec::new();
    let mut moved = Vec::new();
    for op in ops {
        match *op {
            Op::Units { image, update } => {
                let sparse = images[image];
                // The update's command buffer holds the texture, and so the heap (the texture
                // object holds it): this hold is never its last.
                let held = sparse.heap.upgrade();
                let Some(heap) = held.as_ref().and_then(Heap::sparse) else {
                    continue;
                };
                let mut state = lock(&sparse.state);
                for &unit in &update.units {
                    match update.mode {
                        TileMapping::Map => {
                            if state.mapped.contains(&unit) {
                                continue;
                            }
                            if !heap.take(sparse.heap_tiles(unit)) {
                                break;
                            }
                            state.mapped.insert(unit);
                        }
                        TileMapping::Unmap => {
                            if !state.mapped.remove(&unit) {
                                continue;
                            }
                            heap.give(sparse.heap_tiles(unit));
                        }
                    }
                    changed.push(Changed {
                        image,
                        unit,
                        mapped: update.mode == TileMapping::Map,
                        counted: true,
                    });
                }
            }
            Op::Move { from, to, pairs } => {
                for &(source, destination) in pairs {
                    let mapped = lock(&images[from].state).mapped.contains(&source);
                    let free = !lock(&images[to].state).mapped.contains(&destination);
                    if !mapped || !free {
                        continue;
                    }
                    lock(&images[from].state).mapped.remove(&source);
                    lock(&images[to].state).mapped.insert(destination);
                    changed.push(Changed {
                        image: from,
                        unit: source,
                        mapped: false,
                        counted: false,
                    });
                    changed.push(Changed {
                        image: to,
                        unit: destination,
                        mapped: true,
                        counted: false,
                    });
                    moved.push(Moved {
                        from,
                        source,
                        to,
                        destination,
                    });
                }
            }
        }
    }
    // Every device unit touched of a texture the runtime does not zero is wholly mapped or
    // wholly unmapped, but where a heap filled inside one.
    for (index, image) in images.iter().enumerate() {
        if image.zeroed {
            continue;
        }
        let state = lock(&image.state);
        let split = touched(image, index, &changed).into_iter().any(|unit| {
            let units = image.grid.metal_units_of(unit);
            let mapped = units
                .iter()
                .filter(|unit| state.mapped.contains(unit))
                .count();
            mapped != 0 && mapped != units.len()
        });
        if split {
            drop(state);
            return Err(changed);
        }
    }
    Ok((changed, moved))
}

/// The device units of `image`, the batch's image `index`, that `changed` touched.
fn touched(image: &SparseImage, index: usize, changed: &[Changed]) -> BTreeSet<VkUnit> {
    changed
        .iter()
        .filter(|change| change.image == index)
        .flat_map(|change| image.grid.vk_units_of(change.unit))
        .collect()
}

/// The bound device units a move's source unit lies in, of a sparse image moving to a texture,
/// both of which the runtime zeroes: they stay bound through the batch, whose work copies the
/// moved texels out of them and then zeroes them; the next batch that changes the texture's
/// mappings unbinds them where no mapped tile holds them.
fn kept_units(images: &[&SparseImage], moved: &[Moved]) -> BTreeSet<(usize, VkUnit)> {
    let mut kept = BTreeSet::new();
    for step in moved {
        let image = images[step.from];
        if !(image.zeroed && images[step.to].zeroed) || image.emulated() {
            continue;
        }
        let state = lock(&image.state);
        kept.extend(
            image
                .grid
                .vk_units_of(step.source)
                .into_iter()
                .filter(|unit| state.bound.contains_key(unit))
                .map(|unit| (step.from, unit)),
        );
    }
    kept
}

/// Undo `changed`, last first: the Metal state and the heaps' counts as they were.
fn undo(images: &[&SparseImage], changed: &[Changed]) {
    for change in changed.iter().rev() {
        let image = images[change.image];
        let mut state = lock(&image.state);
        let held = image.heap.upgrade();
        let heap = held
            .as_ref()
            .and_then(Heap::sparse)
            .filter(|_| change.counted);
        if change.mapped {
            state.mapped.remove(&change.unit);
            if let Some(heap) = heap {
                heap.give(image.heap_tiles(change.unit));
            }
        } else {
            state.mapped.insert(change.unit);
            if let Some(heap) = heap {
                // The room was the image's a moment ago: taking it back cannot fail.
                heap.take(image.heap_tiles(change.unit));
            }
        }
    }
}

/// The bound device units whose memory a move takes to another unit, by the unit that takes it:
/// a texture the runtime does not zero moves whole device units (`UnitGrid::check_move`), each
/// destination unit of a step filled from the source unit at its place, through a chain of
/// steps from the unit first moved. A unit's memory goes once, to its first destination, where
/// it is bound, no mapped tile holds it after the batch, and its placement meets the
/// destination's needs; another destination takes a new placement.
fn donors(
    images: &[&SparseImage],
    moved: &[Moved],
    granularity: u64,
) -> HashMap<(usize, VkUnit), Donor> {
    let mut origins: BTreeMap<(usize, VkUnit), (usize, VkUnit)> = BTreeMap::new();
    let mut claimed = BTreeSet::new();
    for step in moved {
        let (from, to) = (images[step.from], images[step.to]);
        if (from.zeroed && to.zeroed) || from.emulated() || to.emulated() {
            continue;
        }
        let sources = from.grid.vk_units_of(step.source);
        let targets = to.grid.vk_units_of(step.destination);
        if sources.len() != targets.len() {
            continue;
        }
        for (source, target) in sources.into_iter().zip(targets) {
            if let Some(origin) = origins.remove(&(step.from, source)) {
                origins.insert((step.to, target), origin);
            } else if claimed.insert((step.from, source)) {
                origins.insert((step.to, target), (step.from, source));
            }
        }
    }
    let mut donors = HashMap::new();
    for (target, (image, unit)) in origins {
        let source = images[image];
        let state = lock(&source.state);
        if source.grid.wanted(unit, &state.mapped) {
            continue;
        }
        let Some(placement) = state.bound.get(&unit) else {
            continue;
        };
        let needs = unit_requirements(images[target.0], target.1, granularity);
        let fits = placement.size() >= needs.size
            && placement.offset.is_multiple_of(needs.alignment.max(1))
            && needs.memory_type_bits & (1 << placement.type_index()) != 0;
        if fits {
            let donor = Donor {
                image,
                unit,
                memory: placement.memory,
                offset: placement.offset,
            };
            donors.insert(target, donor);
        }
    }
    donors
}

/// What the memory of device unit `unit` of `image` needs, with the device's sparse address
/// space granularity `granularity`.
fn unit_requirements(
    image: &SparseImage,
    unit: VkUnit,
    granularity: u64,
) -> vk::MemoryRequirements {
    vk::MemoryRequirements {
        size: image.grid.unit_bytes(unit).next_multiple_of(granularity),
        alignment: image.grid.layout.block_bytes.max(granularity),
        memory_type_bits: image.grid.layout.memory_type_bits,
    }
}

/// The device units whose binding changes: each touched unit bound where some Metal unit it holds
/// texels of is now mapped and it is not bound, and unbound where none is and it is, but a unit
/// `kept` stays bound through the batch; and each unit a move vacated in an earlier batch (bound,
/// with no mapped tile) unbound. A bound unit takes a new placement, made now, or, filled by a move
/// of a texture the runtime does not zero, the memory of the source unit it moved from, its texels
/// with it. On failure no new placement is kept.
fn plan(
    device: &Device,
    images: &[&SparseImage],
    (changed, moved): (&[Changed], &[Moved]),
    kept: &BTreeSet<(usize, VkUnit)>,
) -> Result<Vec<UnitChange>, BindFailure> {
    let shared = device.shared();
    let granularity = shared.textures.granularity();
    let donors = donors(images, moved, granularity);
    let mut changes: Vec<UnitChange> = Vec::new();
    let mut failure = None;
    'images: for (index, image) in images.iter().enumerate() {
        // An emulated texture's memory is its own: its mappings bind nothing.
        if image.emulated() {
            continue;
        }
        let state = lock(&image.state);
        let mut units = touched(image, index, changed);
        units.extend(
            state
                .bound
                .keys()
                .filter(|&&unit| !image.grid.wanted(unit, &state.mapped)),
        );
        for unit in units {
            let wanted = image.grid.wanted(unit, &state.mapped) || kept.contains(&(index, unit));
            let bound = state.bound.contains_key(&unit);
            if wanted == bound {
                continue;
            }
            let memory = if !wanted {
                UnitMemory::None
            } else if let Some(&donor) = donors.get(&(index, unit)) {
                UnitMemory::Donated(donor)
            } else {
                let requirements = unit_requirements(image, unit, granularity);
                let placed = crate::memory::allocate_image(
                    &shared.memory,
                    shared.vk(),
                    &requirements,
                    MemoryClass::Private,
                );
                exit_guard::called(Call::AllocateMemory);
                match placed {
                    Ok(placement) => UnitMemory::Fresh(placement),
                    Err(error) => {
                        failure = Some(error);
                        break 'images;
                    }
                }
            };
            changes.push(UnitChange {
                image: index,
                unit,
                memory,
            });
        }
    }
    if let Some(error) = failure {
        free_fresh(device, &changes);
        return Err(BindFailure::Memory(error));
    }
    Ok(changes)
}
