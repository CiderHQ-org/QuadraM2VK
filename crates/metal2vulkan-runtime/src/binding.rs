//! From a dispatch's or draw's plan and commands to what the encoder records: the buffer descriptor
//! of every plan write, the command's own data in the command buffer's upload arena, the push holes
//! filled, and the uses the tracker orders. A compute encoder binds one stage's [`ArgumentTable`],
//! a render encoder its vertex and fragment stages'.
//!
//! - **Buffers** a slot binds take exactly their `DescriptorRange`.
//! - **Tables and blobs** (an argument table, a handle table, the constant blob) are uploaded
//!   per command, padded to 16 bytes; a handle table is bound over exactly its planned length
//!   (the shader counts its entries from the range).
//! - **Neutral buffers** are zero regions of the arena, fresh for each command, so a contained
//!   store into a stand-in reaches nothing another command reads.
//! - **Root tables**, one per stage that reads one, are uploaded with a fresh sink's device address
//!   in the sink entry; a stage's push holes take its root table's address and that of the command
//!   buffer's execution status words, each hole in the stage's own push write. A flat variant's
//!   dispatch takes the device address of the region record its preprocess writes
//!   (`Tables::region_record`) in its `ReplayRegionAddress` hole.
//! - **Textures** a slot binds take the `VkImageView` the plan's view asks for (the texture's,
//!   made at its first bind, `binding_view.rs`), in the `GENERAL` layout every runtime image
//!   rests in; a draw's sampled binding of a texture its render pass instance renders takes the
//!   texture's copy (`texture/attached_reads.rs`). **Samplers** take the `VkSampler` of their
//!   state for the write's use and the bind's LOD clamps (`bind::sampler_desc`, `sampler.rs`), its
//!   address modes mapped where the slot samples a texture of an address-mode map
//!   (`address_map.rs`); a combined write checks the view format's filtering against it. A slot
//!   with nothing bound takes the device's neutral image or sampler (`neutral.rs`).
//! - **Texture buffers** a slot binds take a texel buffer view of their buffer's bytes in the
//!   format the plan names, over the range it gives (`buffer_texture.rs`); an unbound slot takes
//!   the device's neutral texel buffer.
//! - **Tessellation factors** (a patch draw's control stage, `tessellation.rs`) take the
//!   encoder's factor buffer over the range the plan gives, as a slot's buffer does.
//! - **Input attachments** (a draw's framebuffer fetch, `framebuffer_fetch.rs`) take the view of
//!   the pass's color attachment of their index, the framebuffer's. **Storage attachments** (its
//!   fetch in the pixel interlock) take a storage view of the attachment's level and layer, or of
//!   its stand-in's (`attachment_stand_in.rs`).
//! - **The heap** (set 1): the device's heap set of the command's layout, which holds every live
//!   texture at its resource ID and each array's neutral at element 0 (`heap_sets.rs`), so a heap
//!   entry writes nothing: the texture it names (made resident, or bound at a slot) is checked as a
//!   slot's view is. An argument sampler takes the `VkSampler` of the listed state for the array's
//!   use, at the element its ID's entry names (`bind::HeapElements`), and a write no handle can
//!   reach is left out. A texture handle table is the device's texture table
//!   (`heap_sets::texture_table`), so a handle reaches every live texture, as Metal's does.
//! - **Function tables** an argument buffer names: a set-1 element takes a listed table's words
//!   buffer whole. On a device without the function-table array's features the stage's snapshot
//!   takes the listed tables' words copied into the arena (`function_table.rs`), which the command
//!   buffer copies again as it commits.
//! - Every other write source names a later step's resource and is refused, typed, before
//!   anything is recorded.

use super::address_map::SamplerMaps;
use super::argument_table::{ArgumentTable, InlineBytes};
use super::buffer::Buffer;
use super::command_buffer::{NotedTable, State};
use super::device::lock;
use super::error::{internal, Error, RuntimeGap};
use super::function_table::{self, TableSnapshot, VisibleFunctionTable};
use super::heap_sets::{Keeper, TextureTable, HEAP_SET};
use super::key_hash::KeyMap;
use super::render_pass::{PassLayout, Target};
use super::render_target;
use super::residency::{Residency, Used};
use super::resource::ResourceKey;
use super::sampler::{OwnedSampler, SamplerState};
use super::texture::{Texture, TextureShared};
use super::tracker::{Access, Use, UseRange};
use super::upload::Upload;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{
    AccessClass, BindingPlan, DescriptorBytes, DescriptorRange, DescriptorWrite,
    FunctionTableWords, HeapElements, HeapSource, ImageSource, ImageViewDesc, IndexBufferRef,
    NeutralResource, PushConstantWrite, PushHole, ResourceUse, RootTable, SamplerDescriptor,
    SamplerSource, UseTarget, ViewSetup, WriteSource,
};
use metal2vulkan::metal::sampler::{OpaqueBlackBorder, SamplerUse};
use metal2vulkan::pipeline::{HandleClass, SlotClass, SlotRef, Stage, StorageAttachmentView};
use std::collections::BTreeSet;
use std::ops::Deref;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};

/// What an encoder binds: each stage's argument table.
pub(super) struct Tables<'a> {
    stages: &'a [(Stage, &'a ArgumentTable)],
    /// The resources the encoder made resident, which the bind state lists as used.
    residency: Option<&'a Residency>,
    /// A render encoder's pass, whose attachments a draw's input and storage attachments read,
    /// and the draw's views of its layers (`render_target::viewed`).
    pass: Option<(&'a PassLayout, ViewSetup)>,
    /// A draw's render pass instance's attachments: a sampled slot binding of a texture that
    /// presents one of its color attachments takes the texture's copy
    /// (`texture/attached_reads.rs`).
    attached: Option<&'a PassLayout>,
    /// The sampler states the bind state lists as argument samplers, in its order.
    argument_samplers: &'a [SamplerState],
    /// Where the plan's heap writes go, when the command writes a heap.
    heap: Option<&'a HeapElements>,
    /// The device's texture table, which the layout's texture handle tables bind.
    texture_table: Option<&'a TextureTable>,
    /// The tables the bind state lists as function tables, in its order.
    function_tables: &'a [VisibleFunctionTable],
    /// A tile dispatch's scratch images, by color index, which its storage attachments read and
    /// write in place of the pass's attachments (`render/tile_dispatch.rs`).
    tile_scratch: Option<&'a [(u32, Target)]>,
    /// The storage view of a render encoder's shader-side stencil's shadow, when the draw tests
    /// the stencil in the shader (`shadow_stencil.rs`): its returned `[[stencil]]`, or, as the
    /// stencil's stand-in, the draw's stencil test.
    shadow: Option<vk::ImageView>,
    /// The storage view of a render encoder's depth stand-in, when the draw tests the depth in
    /// the shader (`depth_stand_in.rs`).
    depth_stand_in: Option<vk::ImageView>,
    /// The storage views of the `half` shadows a render encoder's current instance holds of
    /// its extended-range colors, by color index (`render/extended_shadow.rs`).
    extended_shadows: Vec<(u32, vk::ImageView)>,
    /// A render encoder's tessellation factor buffer and its offset, when it set one
    /// (`tessellation.rs`).
    factors: Option<(&'a Buffer, u64)>,
    /// The storage view of a render encoder's fragment imageblock storage, when the draw's
    /// fragment function declares an explicit imageblock (`fragment_imageblock.rs`).
    imageblock: Option<vk::ImageView>,
    /// The device address of the 32-byte region record a flat variant's dispatch reads, when
    /// the command is one (`BindingPlan::instantiate_dispatch` of a flat plan): a region of the
    /// command buffer's upload arena (`upload.rs`), with the dispatch's
    /// `VkDispatchIndirectCommand` beside it, which a single-dispatch preprocess writes on the
    /// device before the dispatch. Its `PushHole::ReplayRegionAddress` takes it.
    pub(super) region_record: Option<u64>,
    /// A patch draw's patch index buffer and its offset, when the draw has one.
    patch_indices: Option<(&'a Buffer, u64)>,
    /// An indexed patch draw's control-point index buffer and its offset.
    control_point_indices: Option<(&'a Buffer, u64)>,
    /// A render pass's tile threadgroup memory, when a tile dispatch's function reads it
    /// (`render/tile_memory.rs`).
    tile_memory: Option<&'a Buffer>,
    /// The address-mode maps the command's runtime samplers are bound with (`address_map.rs`).
    sampler_maps: SamplerMaps,
    /// A capture draw's addresses (`render/capture.rs`): of its first capture record, and of its
    /// first index where it is indexed. Its `PushHole::CaptureAddress` and
    /// `PushHole::CaptureIndexAddress` take them.
    capture: Option<(u64, Option<u64>)>,
    /// A de-indexed draw's first index's address (`VariantKey::primitive_id_indices`): its
    /// `PushHole::CaptureIndexAddress` takes it.
    indices: Option<u64>,
    /// A draw's slots read through snapshots (`render/snapshot.rs`): each binds its own texture,
    /// not a copy, in the feedback loop layout, and its snapshot at `WriteSource::SnapshotTexels`.
    snapshots: Option<&'a SnapshotViews>,
}

/// The fragment texture slots a draw reads through snapshots, and the storage view of each one's
/// snapshot (`render/snapshot.rs`, `Texture::snapshot_texels`).
pub(super) struct SnapshotViews {
    pub(super) slots: BTreeSet<u32>,
    pub(super) views: Vec<(u32, vk::ImageView)>,
}

impl<'a> Tables<'a> {
    /// The tables of `stages`.
    pub(super) fn new(stages: &'a [(Stage, &'a ArgumentTable)]) -> Tables<'a> {
        Tables {
            stages,
            residency: None,
            pass: None,
            attached: None,
            argument_samplers: &[],
            heap: None,
            texture_table: None,
            function_tables: &[],
            tile_scratch: None,
            shadow: None,
            depth_stand_in: None,
            extended_shadows: Vec::new(),
            factors: None,
            imageblock: None,
            region_record: None,
            patch_indices: None,
            control_point_indices: None,
            tile_memory: None,
            sampler_maps: SamplerMaps::default(),
            capture: None,
            indices: None,
            snapshots: None,
        }
    }

    /// The tables, with a draw's slots read through snapshots and their snapshots' views
    /// (`render/snapshot.rs`).
    pub(super) fn with_snapshots(self, snapshots: &'a SnapshotViews) -> Tables<'a> {
        Tables {
            snapshots: Some(snapshots),
            ..self
        }
    }

    /// Whether `slot` is one a draw reads through a snapshot: its binding takes the texture
    /// itself, in the feedback loop layout, never its copy.
    fn snapshot_slot(&self, slot: SlotRef) -> bool {
        slot.stage == Stage::Fragment
            && slot.class == SlotClass::Texture
            && self
                .snapshots
                .is_some_and(|s| s.slots.contains(&slot.index))
    }

    /// The tables, with a capture draw's addresses: of its first capture `records`, and of its
    /// first index (`indices`) where it is indexed (`render/capture.rs`).
    pub(super) fn with_capture(self, records: u64, indices: Option<u64>) -> Tables<'a> {
        Tables {
            capture: Some((records, indices)),
            ..self
        }
    }

    /// The tables, with a de-indexed draw's first index's address (`Tables::indices`).
    pub(super) fn with_indices(self, address: u64) -> Tables<'a> {
        Tables {
            indices: Some(address),
            ..self
        }
    }

    /// The tables, with the address-mode maps the command's runtime samplers are bound with,
    /// the bind state's (`address_map::sampler_maps`).
    pub(super) fn with_sampler_maps(self, maps: SamplerMaps) -> Tables<'a> {
        Tables {
            sampler_maps: maps,
            ..self
        }
    }

    /// The tables, with a tile dispatch's `scratch` images by color index.
    pub(super) fn with_tile_scratch(self, scratch: &'a [(u32, Target)]) -> Tables<'a> {
        Tables {
            tile_scratch: Some(scratch),
            ..self
        }
    }

    /// The tables, with the render encoder's `pass` and the draw's `views` of its layers.
    pub(super) fn with_pass(self, pass: &'a PassLayout, views: ViewSetup) -> Tables<'a> {
        Tables {
            pass: Some((pass, views)),
            ..self
        }
    }

    /// The tables, with a draw's render pass instance of `instance`'s attachments: a sampled slot
    /// binding of a texture that presents one of its color attachments takes the texture's copy
    /// (`texture/attached_reads.rs`).
    pub(super) fn with_attachments(self, instance: &'a PassLayout) -> Tables<'a> {
        Tables {
            attached: Some(instance),
            ..self
        }
    }

    /// Whether a slot binding of `texture` of the kind `storage` reads it where the draw's
    /// instance renders it: a sampled one of a texture that presents one of its color
    /// attachments (`texture/attached_reads.rs`).
    fn attached(&self, texture: &Texture, storage: bool) -> bool {
        !storage
            && self
                .attached
                .is_some_and(|instance| texture.attached_in(instance))
    }

    /// The tables, with the storage view of the shader-side stencil's `shadow`.
    pub(super) fn with_shadow(self, shadow: vk::ImageView) -> Tables<'a> {
        Tables {
            shadow: Some(shadow),
            ..self
        }
    }

    /// The tables, with the storage view of the depth stand-in `stand_in`.
    pub(super) fn with_depth_stand_in(self, stand_in: vk::ImageView) -> Tables<'a> {
        Tables {
            depth_stand_in: Some(stand_in),
            ..self
        }
    }

    /// The tables, with the storage views of the extended-range colors' `half` `shadows`, by
    /// color index.
    pub(super) fn with_extended_shadows(self, shadows: Vec<(u32, vk::ImageView)>) -> Tables<'a> {
        Tables {
            extended_shadows: shadows,
            ..self
        }
    }

    /// The tables, with the render encoder's tessellation factor buffer and its offset.
    pub(super) fn with_factors(self, buffer: &'a Buffer, offset: u64) -> Tables<'a> {
        Tables {
            factors: Some((buffer, offset)),
            ..self
        }
    }

    /// The tables, with a patch draw's patch index buffer and its offset.
    pub(super) fn with_patch_indices(self, buffer: &'a Buffer, offset: u64) -> Tables<'a> {
        Tables {
            patch_indices: Some((buffer, offset)),
            ..self
        }
    }

    /// The tables, with an indexed patch draw's control-point index buffer and its offset.
    pub(super) fn with_control_point_indices(self, buffer: &'a Buffer, offset: u64) -> Tables<'a> {
        Tables {
            control_point_indices: Some((buffer, offset)),
            ..self
        }
    }

    /// The tables, with the render pass's tile threadgroup memory.
    pub(super) fn with_tile_memory(self, buffer: &'a Buffer) -> Tables<'a> {
        Tables {
            tile_memory: Some(buffer),
            ..self
        }
    }

    /// The tables, with the storage view of the fragment imageblock's storage.
    pub(super) fn with_imageblock(self, imageblock: vk::ImageView) -> Tables<'a> {
        Tables {
            imageblock: Some(imageblock),
            ..self
        }
    }

    /// The tables, with the sampler states the bind state lists as argument samplers and where
    /// the plan's heap writes go.
    pub(super) fn with_heap(
        self,
        samplers: &'a [SamplerState],
        heap: &'a HeapElements,
    ) -> Tables<'a> {
        Tables {
            argument_samplers: samplers,
            heap: Some(heap),
            ..self
        }
    }

    /// The tables, with the device's texture table at the layout's texture handle tables.
    pub(super) fn with_texture_table(self, table: Option<&'a TextureTable>) -> Tables<'a> {
        Tables {
            texture_table: table,
            ..self
        }
    }

    /// The tables, with the visible-function tables the bind state lists as function tables.
    pub(super) fn with_function_tables(self, tables: &'a [VisibleFunctionTable]) -> Tables<'a> {
        Tables {
            function_tables: tables,
            ..self
        }
    }

    /// The tables, with the resources the encoder made resident (`use_resource(s)`).
    pub(super) fn with_residency(self, residency: &'a Residency) -> Tables<'a> {
        Tables {
            residency: Some(residency),
            ..self
        }
    }

    /// The resource the encoder made resident under `token`.
    fn used(&self, token: u32) -> Option<&'a Used> {
        self.residency?.get(token)
    }

    /// The pixel format of the texture bound at texture slot `slot`, if one is: what a specialized
    /// notice names (`Device::degrade_specialized`).
    pub(super) fn bound_format(&self, slot: SlotRef) -> Option<metal2vulkan::metal::PixelFormat> {
        slot_texture(self, slot).map(Texture::pixel_format)
    }

    /// The table of `stage`, if the encoder has one.
    fn of(&self, stage: Stage) -> Option<&'a ArgumentTable> {
        self.stages
            .iter()
            .find(|(of, _)| *of == stage)
            .map(|&(_, table)| table)
    }
}

/// What one command's instantiation gives the binder, a dispatch's or a draw's alike.
pub(super) struct Command<'a> {
    pub(super) descriptor_bytes: &'a [DescriptorBytes],
    pub(super) descriptor_ranges: &'a [DescriptorRange],
    /// Each stage's root table, by the stages its push hole names.
    pub(super) root_tables: Vec<(metal2vulkan::vk::ShaderStageFlags, &'a RootTable)>,
    pub(super) push_holes: &'a [PushHole],
    /// The command's own resources beyond the plan's uses (an index buffer, indirect arguments).
    pub(super) uses: &'a [ResourceUse],
    /// Each Metal stage's constant blob, for its `ConstantBlob` write.
    pub(super) blobs: Vec<(Stage, &'a Arc<[u8]>)>,
    /// An emulated mesh draw's work buffer and records, for its `MeshWork` and `MeshRecords`
    /// writes.
    pub(super) mesh: Option<MeshRegions>,
    /// An emulated tessellation draw's work buffer and records, for its `TessellationWork` and
    /// `TessellationRecords` writes.
    pub(super) tessellation: Option<MeshRegions>,
}

/// Where an emulated mesh or tessellation draw's work buffer and records lie (`render/mesh.rs`,
/// `render/tessellation_scan.rs`), each bound whole.
#[derive(Clone, Copy)]
pub(super) struct MeshRegions {
    pub(super) work: vk::DescriptorBufferInfo,
    pub(super) records: vk::DescriptorBufferInfo,
}

/// The descriptor of a sampler `state` bound with `clamps`: the state's, its LOD clamps replaced
/// by the bind's when it set them, as plain Metal samples with them
/// ([`metal_lod_clamps`](super::sampler::metal_lod_clamps): inverted ones read the upper clamp's
/// level).
pub(super) fn bound_sampler(state: &SamplerState, clamps: Option<[f32; 2]>) -> SamplerDescriptor {
    let mut descriptor = *state.descriptor();
    if let Some(clamps) = clamps {
        [descriptor.lod_min_clamp, descriptor.lod_max_clamp] =
            super::sampler::metal_lod_clamps(clamps);
    }
    descriptor
}

/// One descriptor to write: a buffer's or an image's and sampler's.
pub(super) struct Write {
    pub(super) set: u32,
    pub(super) binding: u32,
    pub(super) element: u32,
    pub(super) ty: vk::DescriptorType,
    pub(super) info: Info,
    /// What keeps what a heap element names while the device's heap set holds it: an argument
    /// sampler's `VkSampler`, a function table's words (`heap_sets.rs`).
    pub(super) held: Option<Keeper>,
}

/// What a descriptor write holds.
#[derive(Clone, Copy)]
pub(super) enum Info {
    Buffer(vk::DescriptorBufferInfo),
    Image(vk::DescriptorImageInfo),
    Texel(vk::BufferView),
}

/// One push-constant write: stages, offset, bytes.
pub(super) type PushWrite = (vk::ShaderStageFlags, u32, Vec<u8>);

/// A push hole's address: the stages of the push write it lies in, its offset, the address.
#[derive(Clone, Copy)]
pub(super) struct Hole {
    stages: vk::ShaderStageFlags,
    offset: u32,
    address: u64,
}

/// What a command records besides its pipeline, its draw or groups, and its descriptor writes
/// ([`bind`]'s).
pub(super) struct Bound {
    /// The push holes' addresses, for [`push_writes`].
    pub(super) holes: Vec<Hole>,
    /// The Vulkan samplers the writes name, which the command buffer holds until it completes.
    pub(super) samplers: Vec<Arc<OwnedSampler>>,
    /// Each sampler table's `(set, binding)` and its region of the arena, which the command
    /// buffer writes again as it commits (`sampler_listing.rs`).
    pub(super) sampler_tables: Vec<((u32, u32), Upload)>,
}

/// The buffer bound at buffer slot `slot`, if any.
fn slot_buffer<'a>(tables: &Tables<'a>, slot: SlotRef) -> Option<&'a (Buffer, u64)> {
    (slot.class == SlotClass::Buffer)
        .then(|| tables.of(slot.stage)?.buffers.get(&slot.index))
        .flatten()
}

/// The inline bytes set at buffer slot `slot`, if any.
fn slot_bytes<'a>(tables: &Tables<'a>, slot: SlotRef) -> Option<&'a InlineBytes> {
    (slot.class == SlotClass::Buffer)
        .then(|| tables.of(slot.stage)?.bytes.get(&slot.index))
        .flatten()
}

/// The texture bound at texture slot `slot`, if any.
fn slot_texture<'a>(tables: &Tables<'a>, slot: SlotRef) -> Option<&'a Texture> {
    (slot.class == SlotClass::Texture)
        .then(|| tables.of(slot.stage)?.textures.get(&slot.index))
        .flatten()
}

/// The tile map of the sparse texture bound at texture slot `slot`, which the plan reads where
/// the slot's shape is sparse: every sparse texture whose type has a tile map keeps one. One
/// without (a view of another type of a cube texture's slices) is refused
/// `NotImplemented(SparseTileMaps)`: the format lays out no cube map.
fn slot_tile_map<'a>(tables: &Tables<'a>, slot: SlotRef) -> Result<&'a Buffer, Error> {
    let image = slot_texture(tables, slot)
        .and_then(Texture::sparse_image)
        .ok_or_else(|| internal("a tile map read of a slot without a sparse texture"))?;
    image
        .tile_map()
        .map(|tile_map| &tile_map.buffer)
        .ok_or(Error::NotImplemented(RuntimeGap::SparseTileMaps))
}

/// The sampler bound at sampler slot `slot`, if any.
fn slot_sampler<'a>(
    tables: &Tables<'a>,
    slot: SlotRef,
) -> Option<&'a (SamplerState, Option<[f32; 2]>)> {
    (slot.class == SlotClass::Sampler)
        .then(|| tables.of(slot.stage)?.samplers.get(&slot.index))
        .flatten()
}

/// `n` rounded up to 16 bytes, at least 16 (the padding contract).
fn padded(n: usize) -> u64 {
    (n as u64).next_multiple_of(16).max(16)
}

/// The descriptors, uploads, and push hole addresses of `command`, a command (by `call`) of
/// `plan` over `tables`, uploading into the command buffer's `state`: the descriptor writes in
/// `writes`, emptied first, so a caller that keeps the vector writes each command's over the last
/// one's storage (a draw's, `LiveBind::writes`).
pub(super) fn bind(
    call: &'static str,
    device: &Device,
    state: &mut State,
    plan: &BindingPlan,
    command: &Command<'_>,
    tables: &Tables<'_>,
    (mut memo, writes): (Option<&mut BindMemo>, &mut Vec<Write>),
) -> Result<Bound, Error> {
    writes.clear();
    writes.reserve(plan.descriptors.len());
    let mut samplers = Vec::new();
    // The sampler tables, whose regions the command buffer writes again as it commits.
    let sampler_keys = plan.sampler_tables();
    let mut sampler_tables = Vec::with_capacity(sampler_keys.len());
    for write in &plan.descriptors {
        let element = match tables.heap {
            Some(heap) => match heap.element(write) {
                Some(element) => element,
                // No handle reaches it.
                None => continue,
            },
            None => write.array_element,
        };
        let ty: vk::DescriptorType = write.ty.into();
        // The device's heap set holds every texture and the neutrals (the module docs): a
        // texture the command names is checked as a slot's view, and nothing is written.
        if tables.heap.is_some() && write.set == HEAP_SET {
            if let WriteSource::HeapEntry { source, view, .. } = &write.source {
                let storage = ty == vk::DescriptorType::STORAGE_IMAGE;
                heap_texture(tables, *source)?.bound_view(call, view, storage)?;
                continue;
            }
            if element == metal2vulkan::bind::NEUTRAL_HEAP_ELEMENT {
                continue;
            }
        }
        let info = match (&write.source, ty) {
            (WriteSource::HeapEntry { .. }, _) => {
                return Err(internal("a heap entry outside the heap set"))
            }
            (WriteSource::HandleTable { bytes_len, .. }, _)
                if sampler_keys.contains(&(write.set, write.binding)) =>
            {
                let (info, region) = sampler_table(device, state, command, write, *bytes_len)?;
                sampler_tables.push(((write.set, write.binding), region));
                Info::Buffer(info)
            }
            (_, ty) => match ty {
                vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER => {
                    Info::Buffer(descriptor(device, state, command, tables, write)?)
                }
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                | vk::DescriptorType::STORAGE_TEXEL_BUFFER => {
                    let storage = ty == vk::DescriptorType::STORAGE_TEXEL_BUFFER;
                    Info::Texel(texel_descriptor(device, command, tables, write, storage)?)
                }
                _ => Info::Image(image_descriptor(
                    call,
                    device,
                    tables,
                    write,
                    ty,
                    (&mut samplers, memo.as_deref_mut()),
                )?),
            },
        };
        // What keeps an argument sampler's `VkSampler` or a function table's words while the
        // heap set holds them.
        let held: Option<Keeper> = match (&write.source, &info) {
            (WriteSource::ArgumentSampler { .. }, Info::Image(image)) => samplers
                .iter()
                .find(|kept| kept.raw() == image.sampler)
                .map(|kept| Box::new(Arc::clone(kept)) as Keeper),
            (WriteSource::FunctionTable { index }, _) => tables
                .function_tables
                .get(*index as usize)
                .map(|table| Box::new(table.words().clone()) as Keeper),
            _ => None,
        };
        writes.push(Write {
            set: write.set,
            binding: write.binding,
            element,
            ty,
            info,
            held,
        });
    }
    let mut roots = Vec::with_capacity(command.root_tables.len());
    for &(stages, table) in &command.root_tables {
        roots.push((stages, root_table(device, state, table)?));
    }
    let mut holes = Vec::with_capacity(command.push_holes.len());
    for hole in command.push_holes {
        holes.push(match *hole {
            PushHole::RootTableAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: roots
                    .iter()
                    .find(|(of, _)| *of == stages)
                    .map(|&(_, address)| address)
                    .ok_or_else(|| internal("a root-table hole without its stage's root table"))?,
            },
            PushHole::ExecutionStatusAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: state
                    .status_address(device)?
                    .ok_or_else(|| internal("a status hole without device addresses"))?,
            },
            // A flat variant's dispatch reads its region record; a device replay's records the
            // runtime does not run.
            PushHole::ReplayRegionAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: tables
                    .region_record
                    .ok_or(Error::NotImplemented(RuntimeGap::CommandReplay))?,
            },
            PushHole::CommandRecordsAddress { .. } => {
                return Err(Error::NotImplemented(RuntimeGap::CommandReplay));
            }
            // A capture draw's first record, and its first index where it is indexed
            // (`Tables::with_capture`).
            PushHole::CaptureAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: tables
                    .capture
                    .map(|(records, _)| records)
                    .ok_or_else(|| internal("a capture hole without capture records"))?,
            },
            PushHole::CaptureIndexAddress { stages, offset } => Hole {
                stages: stages.into(),
                offset,
                address: tables
                    .capture
                    .and_then(|(_, indices)| indices)
                    .or(tables.indices)
                    .ok_or_else(|| internal("a capture index hole without capture indices"))?,
            },
            // Which of a snapshot variant's draws runs: 0 in the plan's bytes, the drawing draw's,
            // and set before each draw as it records (`render/snapshot.rs`).
            PushHole::SnapshotPhase { .. } if tables.snapshots.is_some() => continue,
            PushHole::SnapshotPhase { .. } => {
                return Err(internal(
                    "a snapshot phase hole the runtime did not ask for",
                ));
            }
        });
    }
    Ok(Bound {
        holes,
        samplers,
        sampler_tables,
    })
}

/// The descriptor of sampler table `write` (of `bytes_len` bytes, the plan's) and its region of
/// the command buffer's arena, holding `command`'s bytes.
fn sampler_table(
    device: &Device,
    state: &mut State,
    command: &Command<'_>,
    write: &DescriptorWrite,
    bytes_len: u32,
) -> Result<(vk::DescriptorBufferInfo, Upload), Error> {
    let key = (write.set, write.binding);
    let bytes = command
        .descriptor_bytes
        .iter()
        .find(|d| (d.set, d.binding) == key)
        .map(|d| d.bytes.as_slice())
        .ok_or_else(|| internal("a table write without its bytes"))?;
    if bytes.len() != bytes_len as usize {
        return Err(internal("a sampler table of another length than planned"));
    }
    let region = state.uploads.upload(device, bytes, padded(bytes.len()))?;
    let info = vk::DescriptorBufferInfo::default()
        .buffer(region.raw())
        .offset(region.offset())
        .range(u64::from(bytes_len));
    Ok((info, region))
}

/// The set-1 writes of `plan`'s argument sampler elements for `samplers`, the states a command
/// lists again as its command buffer commits (`sampler_listing.rs`): listed state `i` at its
/// element of each sampler array, made for the array's use as [`bind`] makes it; an element
/// past the list is no handle's, and is not written. With the Vulkan samplers the writes name.
pub(super) fn argument_sampler_writes(
    device: &Device,
    plan: &BindingPlan,
    samplers: &[SamplerState],
) -> Result<(Vec<Write>, Vec<Arc<OwnedSampler>>), Error> {
    let mut writes = Vec::new();
    let mut owned = Vec::new();
    for write in &plan.descriptors {
        let WriteSource::ArgumentSampler {
            index,
            compare,
            integer,
            cube_gradient,
        } = write.source
        else {
            continue;
        };
        let Some(state) = samplers.get(index as usize) else {
            continue;
        };
        if write.set != HEAP_SET {
            return Err(internal("an argument sampler outside the heap set"));
        }
        let element = u32::try_from(state.heap_element())
            .map_err(|_| internal("an argument sampler past every array"))?;
        let desc = metal2vulkan::bind::sampler_desc(
            device.compile_device(),
            state.descriptor(),
            SamplerUse::new(compare, integer).with_cube_gradient(cube_gradient),
            OpaqueBlackBorder::Stated,
            None,
        )
        .map_err(Error::Bind)?;
        let sampler = state.vk_sampler(&desc)?;
        let info = vk::DescriptorImageInfo::default().sampler(sampler.raw());
        writes.push(Write {
            set: write.set,
            binding: write.binding,
            element,
            ty: write.ty.into(),
            info: Info::Image(info),
            held: Some(Box::new(Arc::clone(&sampler)) as Keeper),
        });
        owned.push(sampler);
    }
    Ok((writes, owned))
}

/// Make the device's neutral images, texel buffers and samplers that [`bind`] may write for
/// `plan`: at its unbound texture, sampler and texel slots, its heap elements whose texture it
/// cannot present, and its argument sampler elements past the states listed. So binding a command
/// of the plan later makes none: a host step runs no submission of its own and makes no neutral,
/// which it refuses (`neutral.rs`).
pub(super) fn make_neutrals(device: &Device, plan: &BindingPlan) -> Result<(), Error> {
    for write in &plan.descriptors {
        let ty: vk::DescriptorType = write.ty.into();
        match &write.source {
            WriteSource::Neutral(NeutralResource::TexelBuffer { format }) => {
                let storage = ty == vk::DescriptorType::STORAGE_TEXEL_BUFFER;
                device.neutral_texel_view(*format, storage)?;
            }
            WriteSource::Neutral(kind @ NeutralResource::Image { .. }) => {
                device.neutral_view(*kind)?;
            }
            WriteSource::Neutral(NeutralResource::Sampler { compare })
            | WriteSource::ArgumentSampler { compare, .. } => {
                device.neutral_sampler(*compare)?;
            }
            WriteSource::CombinedImageSampler { image, sampler } => {
                if let ImageSource::Neutral(kind) = image {
                    device.neutral_view(*kind)?;
                }
                if let SamplerSource::Neutral(NeutralResource::Sampler { compare }) = sampler {
                    device.neutral_sampler(*compare)?;
                }
            }
            WriteSource::HeapEntry {
                class: HandleClass::Image(image),
                ..
            } => {
                if let Some(kind) = NeutralResource::for_image(image) {
                    device.neutral_view(kind)?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// The texture heap entry `source` names: made resident by the encoder, or bound at a slot.
fn heap_texture<'a>(tables: &Tables<'a>, source: HeapSource) -> Result<&'a Texture, Error> {
    match source {
        HeapSource::Used(token) => match tables.used(token) {
            Some(Used::Texture(texture)) => Ok(texture),
            _ => Err(internal(
                "a heap entry of a texture the encoder did not use",
            )),
        },
        HeapSource::Bound(slot) => slot_texture(tables, slot)
            .ok_or_else(|| internal("a heap entry of an unbound texture slot")),
    }
}

/// The push writes of `writes` (one region's, or one draw's) with `holes` filled: each hole in
/// the write of its own stages that covers its eight bytes. Its writes' bytes become theirs, in
/// their storage.
pub(super) fn push_writes(
    writes: Vec<PushConstantWrite>,
    holes: &[Hole],
) -> Result<Vec<PushWrite>, Error> {
    let mut filled: Vec<PushWrite> = writes
        .into_iter()
        .map(|write| (write.stages.into(), write.offset, write.bytes))
        .collect();
    fill_holes(&mut filled, holes)?;
    Ok(filled)
}

/// [`push_writes`] of `writes`, written over `filled`: each write's bytes copied into the storage
/// of the write a draw before left at its place, where there is one.
pub(super) fn push_writes_into(
    writes: &[PushConstantWrite],
    holes: &[Hole],
    filled: &mut Vec<PushWrite>,
) -> Result<(), Error> {
    filled.truncate(writes.len());
    for (at, write) in writes.iter().enumerate() {
        let stages = write.stages.into();
        match filled.get_mut(at) {
            Some(kept) => {
                (kept.0, kept.1) = (stages, write.offset);
                kept.2.clear();
                kept.2.extend_from_slice(&write.bytes);
            }
            None => filled.push((stages, write.offset, write.bytes.clone())),
        }
    }
    fill_holes(filled, holes)
}

/// Fill each of `holes` in the write of `filled` of its own stages that covers its eight bytes.
fn fill_holes(filled: &mut [PushWrite], holes: &[Hole]) -> Result<(), Error> {
    for hole in holes {
        let write = filled
            .iter_mut()
            .find(|(stages, start, bytes)| {
                *stages == hole.stages
                    && *start <= hole.offset
                    && hole.offset as usize + 8 <= *start as usize + bytes.len()
            })
            .ok_or_else(|| internal("no push write of its stages covers a push hole"))?;
        let at = (hole.offset - write.1) as usize;
        write.2[at..at + 8].copy_from_slice(&hole.address.to_le_bytes());
    }
    Ok(())
}

/// One descriptor of a pushed set 0: [`Write`]'s, without its set number and keeper.
#[derive(Clone, Copy)]
pub(super) struct SetWrite {
    pub(super) binding: u32,
    pub(super) element: u32,
    pub(super) ty: vk::DescriptorType,
    pub(super) info: Info,
}

impl SetWrite {
    /// Set 0's `writes`, then a variant's `immutable` pushes (`VariantObjects::immutable_pushes`):
    /// what a draw pushes.
    pub(super) fn pushed(writes: &[Write], immutable: &[SetWrite]) -> Vec<SetWrite> {
        let mut pushed = Vec::new();
        Self::push_into(writes, immutable, &mut pushed);
        pushed
    }

    /// [`pushed`](Self::pushed) in `pushed`, emptied first: over the last draw's storage, where
    /// the caller keeps it (`LiveBind::pushed`).
    pub(super) fn push_into(writes: &[Write], immutable: &[SetWrite], pushed: &mut Vec<SetWrite>) {
        pushed.clear();
        let writes = writes.iter().map(|write| SetWrite {
            binding: write.binding,
            element: write.element,
            ty: write.ty,
            info: write.info,
        });
        pushed.extend(writes.chain(immutable.iter().copied()));
    }

    /// Whether it writes what `other` writes.
    pub(super) fn same(&self, other: &SetWrite) -> bool {
        (self.binding, self.element, self.ty) == (other.binding, other.element, other.ty)
            && match (self.info, other.info) {
                (Info::Buffer(a), Info::Buffer(b)) => {
                    (a.buffer, a.offset, a.range) == (b.buffer, b.offset, b.range)
                }
                (Info::Image(a), Info::Image(b)) => {
                    (a.sampler, a.image_view, a.image_layout)
                        == (b.sampler, b.image_view, b.image_layout)
                }
                (Info::Texel(a), Info::Texel(b)) => a == b,
                _ => false,
            }
    }
}

/// `vkCmdPushDescriptorSet` by its core name from Vulkan 1.4 (`api`), else by
/// `VK_KHR_push_descriptor`'s where the device enabled it; none otherwise.
///
/// # Safety
///
/// `device` is live, of `instance`, made with `enabled` enabled.
pub(crate) unsafe fn push_descriptor_command(
    instance: &ash::Instance,
    device: &ash::Device,
    api: metal2vulkan::device::ApiVersion,
    enabled: &metal2vulkan::device::EnableSet,
) -> Option<vk::PFN_vkCmdPushDescriptorSetKHR> {
    let name: &std::ffi::CStr = if api >= (metal2vulkan::device::ApiVersion { major: 1, minor: 4 })
    {
        c"vkCmdPushDescriptorSet"
    } else if enabled
        .extensions
        .contains(&metal2vulkan::device::VkExtension::KhrPushDescriptor)
    {
        c"vkCmdPushDescriptorSetKHR"
    } else {
        return None;
    };
    // SAFETY: the caller's; a command the device does not have is null.
    let command = unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) }?;
    // SAFETY: the device gave the command of this name, and both names share its signature.
    Some(unsafe {
        std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkCmdPushDescriptorSetKHR>(
            command,
        )
    })
}

/// Push `writes` as set 0 of `layout` at `point` into `command_buffer`, in one
/// `vkCmdPushDescriptorSet` (`push`): their infos on the stack up to [`WRITES_AT_ONCE`], else on
/// the heap.
///
/// # Safety
///
/// `command_buffer` is recording, `layout`'s set 0 is a push descriptor set layout, and every
/// buffer, view and sampler written stays live while the command buffer can run.
pub(super) unsafe fn push_set(
    push: vk::PFN_vkCmdPushDescriptorSetKHR,
    command_buffer: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    writes: &[SetWrite],
) {
    let count = writes.len();
    if count <= WRITES_AT_ONCE {
        let mut buffer_infos = [[vk::DescriptorBufferInfo::default()]; WRITES_AT_ONCE];
        let mut image_infos = [[vk::DescriptorImageInfo::default()]; WRITES_AT_ONCE];
        let mut texel_views = [[vk::BufferView::null()]; WRITES_AT_ONCE];
        let mut raw = [vk::WriteDescriptorSet::default(); WRITES_AT_ONCE];
        fill_pushed(
            writes,
            (&mut buffer_infos, &mut image_infos, &mut texel_views),
            &mut raw,
        );
        // SAFETY: the caller's contract; every array `raw` points at is live for the call.
        unsafe { push(command_buffer, point, layout, 0, count as u32, raw.as_ptr()) };
    } else {
        let mut buffer_infos = vec![[vk::DescriptorBufferInfo::default()]; count];
        let mut image_infos = vec![[vk::DescriptorImageInfo::default()]; count];
        let mut texel_views = vec![[vk::BufferView::null()]; count];
        let mut raw = vec![vk::WriteDescriptorSet::default(); count];
        fill_pushed(
            writes,
            (&mut buffer_infos, &mut image_infos, &mut texel_views),
            &mut raw,
        );
        // SAFETY: as above.
        unsafe { push(command_buffer, point, layout, 0, count as u32, raw.as_ptr()) };
    }
}

/// The infos of a push's `writes` into `infos`, one place each, and its writes into `raw`, each
/// pointing at its info.
fn fill_pushed<'a>(
    writes: &[SetWrite],
    infos: (
        &'a mut [[vk::DescriptorBufferInfo; 1]],
        &'a mut [[vk::DescriptorImageInfo; 1]],
        &'a mut [[vk::BufferView; 1]],
    ),
    raw: &mut [vk::WriteDescriptorSet<'a>],
) {
    let (buffer_infos, image_infos, texel_views) = infos;
    for (i, write) in writes.iter().enumerate() {
        match write.info {
            Info::Buffer(info) => buffer_infos[i] = [info],
            Info::Image(info) => image_infos[i] = [info],
            Info::Texel(view) => texel_views[i] = [view],
        }
    }
    let infos = buffer_infos
        .iter()
        .zip(image_infos.iter())
        .zip(texel_views.iter());
    for ((write, ((buffer, image), texel)), raw) in writes.iter().zip(infos).zip(raw) {
        let set = vk::WriteDescriptorSet::default()
            .dst_binding(write.binding)
            .dst_array_element(write.element)
            .descriptor_type(write.ty);
        *raw = match write.info {
            Info::Buffer(_) => set.buffer_info(buffer),
            Info::Image(_) => set.image_info(image),
            Info::Texel(_) => set.texel_buffer_view(texel),
        };
    }
}

/// The descriptor writes [`write_sets`] hands the driver in one call: their infos sit on the stack.
const WRITES_AT_ONCE: usize = 16;

/// Write `writes` into `sets`, the descriptor sets of their layout by set number: in order,
/// [`WRITES_AT_ONCE`] at a time.
///
/// # Safety
///
/// No pending or recording command uses `sets`, and every buffer, view and sampler written stays
/// live while a command that binds the sets can run.
pub(super) unsafe fn write_sets(
    device: &Device,
    sets: &[vk::DescriptorSet],
    writes: &[Write],
) -> Result<(), Error> {
    if writes.iter().any(|write| write.set as usize >= sets.len()) {
        return Err(internal("a descriptor write to a set the layout lacks"));
    }
    for writes in writes.chunks(WRITES_AT_ONCE) {
        let mut buffer_infos = [[vk::DescriptorBufferInfo::default()]; WRITES_AT_ONCE];
        let mut image_infos = [[vk::DescriptorImageInfo::default()]; WRITES_AT_ONCE];
        let mut texel_views = [[vk::BufferView::null()]; WRITES_AT_ONCE];
        for (i, write) in writes.iter().enumerate() {
            match write.info {
                Info::Buffer(info) => buffer_infos[i] = [info],
                Info::Image(info) => image_infos[i] = [info],
                Info::Texel(view) => texel_views[i] = [view],
            }
        }
        let mut raw = [vk::WriteDescriptorSet::default(); WRITES_AT_ONCE];
        for (i, write) in writes.iter().enumerate() {
            let set = vk::WriteDescriptorSet::default()
                .dst_set(sets[write.set as usize])
                .dst_binding(write.binding)
                .dst_array_element(write.element)
                .descriptor_type(write.ty);
            raw[i] = match write.info {
                Info::Buffer(_) => set.buffer_info(&buffer_infos[i]),
                Info::Image(_) => set.image_info(&image_infos[i]),
                Info::Texel(_) => set.texel_buffer_view(&texel_views[i]),
            };
        }
        // SAFETY: the caller's contract; every array `raw` points at is live for the call.
        unsafe {
            device
                .shared()
                .vk()
                .update_descriptor_sets(&raw[..writes.len()], &[])
        };
    }
    Ok(())
}

/// The texel buffer view of `write`, a storage texel buffer's (`storage`) or a uniform one's.
fn texel_descriptor(
    device: &Device,
    command: &Command<'_>,
    tables: &Tables<'_>,
    write: &DescriptorWrite,
    storage: bool,
) -> Result<vk::BufferView, Error> {
    match &write.source {
        WriteSource::TexelBuffer { slot, format, .. } => {
            let texture = slot_texture(tables, *slot)
                .ok_or_else(|| internal("the plan binds an unbound texture slot"))?;
            // A swizzled texture buffer's reads take its swizzle in the function, under the
            // slot's `TextureSwizzle` specialization; the plan refuses one at a slot whose stage
            // does not apply it.
            let texels = texture
                .texels()
                .ok_or_else(|| internal("a texel buffer write of a texture with an image"))?;
            let range = command
                .descriptor_ranges
                .iter()
                .find(|r| (r.set, r.binding) == (write.set, write.binding))
                .ok_or_else(|| internal("a texel buffer write without its range"))?
                .range;
            texels.view(*format, storage, range)
        }
        WriteSource::Neutral(NeutralResource::TexelBuffer { format }) => {
            device.neutral_texel_view(*format, storage)
        }
        other => Err(unexpected(other)),
    }
}

/// The most sampler descriptions a [`BindMemo`] keeps.
const KEPT: usize = 8;

/// The most views a [`BindMemo`] keeps: once full, it lets them all go and starts again. Each
/// is looked up by its key, so an encoder that binds more textures in turn than a short list
/// holds (a page's tiles) finds every one it bound before.
const KEPT_VIEWS: usize = 256;

/// The image views and samplers render encoders' last binds made or found
/// ([`image_descriptor`]), kept by everything each is a function of, so that a bind of the same
/// takes them again without working them out. An encoder that ends gives its memo to its device
/// ([`SpareBindMemos`]), and the next render encoder opened takes it, so a pass that binds the
/// textures and samplers the one before it bound works none out. A memo holds its textures
/// weakly, so it keeps none alive:
///
/// - **A view** of a texture slot is a function of the texture (its storage, image, decoded and
///   depth twins and record are fixed as it is made; its usage shadow is made once and kept), the
///   view's description, whether the binding is a storage one, and its images' views, which an
///   image keeps once made until it lets them all go under its views lock (a rebind's or a
///   storage form's new `VkImage`, a detach's stand-in, a reattach). Each letting go counts in
///   `DeviceShared::views_epoch`: the views are kept while it holds, and all dropped once it
///   moves. A view is kept by its texture's address ([`KeptView`]), which stays the texture's
///   while the memo keeps the view: a texture found there is the one the view is of.
/// - **A sampler's description** (`sampler_desc`'s answer) for a sampler slot is a function of
///   the descriptor the bind maps the slot's state to (its LOD clamps and the address modes its
///   textures map), the facts `sampler_desc` takes beside the device (the use: a comparison, an
///   integer texture, a cube's explicit gradients; the opaque black border; the image's format)
///   and the device. The bind still asks the
///   state for its `VkSampler` of the description (`SamplerState::vk_sampler`), so the state keeps
///   and forgets its samplers as before.
///
/// A sampler's descriptor is compared exactly, a float field by its bits.
#[derive(Default)]
pub(super) struct BindMemo {
    /// The views epoch the views were kept under.
    epoch: u64,
    views: KeyMap<ViewKey, KeptView>,
    samplers: Vec<KeptSampler>,
}

/// What a view a bind made or found is a function of ([`BindMemo`]): a binding of the texture at
/// the address as the view, a storage one where the first flag is set, of a texture the draw's
/// instance renders where the second is (`Tables::attached`).
type ViewKey = (usize, ImageViewDesc, bool, bool);

/// A view a [`BindMemo`] keeps, and its texture, held weakly: a texture that goes leaves its
/// allocation, and so its address, to its weak handles, so no other texture is at the address
/// while the view is kept.
struct KeptView {
    texture: Weak<TextureShared>,
    raw: vk::ImageView,
}

/// The most bind memos a device keeps for its render encoders to take: one per encoder open at
/// once beyond it starts empty.
const SPARE_BIND_MEMOS: usize = 4;

/// A device's bind memos that ended render encoders left, which the render encoders it opens
/// next take ([`BindMemo`]): at most [`SPARE_BIND_MEMOS`], the latest given taken first.
#[derive(Default)]
pub(super) struct SpareBindMemos(Mutex<Vec<BindMemo>>);

impl SpareBindMemos {
    /// The memo a render encoder opening takes: the latest an encoder gave back, else an empty
    /// one.
    pub(super) fn take(&self) -> BindMemo {
        lock(&self.0).pop().unwrap_or_default()
    }

    /// Keep `memo`, an ended render encoder's, for the next encoder to take, where fewer than
    /// [`SPARE_BIND_MEMOS`] are kept; else it goes.
    pub(super) fn give(&self, memo: BindMemo) {
        let mut spare = lock(&self.0);
        if spare.len() < SPARE_BIND_MEMOS {
            spare.push(memo);
        }
    }
}

/// A sampler description a bind worked out ([`BindMemo`]): `descriptor`'s under `facts`.
struct KeptSampler {
    descriptor: SamplerDescriptor,
    facts: SamplerFacts,
    desc: metal2vulkan::vk::SamplerDesc,
}

/// What `sampler_desc` takes of a binding beside its descriptor and the device.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SamplerFacts {
    use_: SamplerUse,
    opaque_black: OpaqueBlackBorder,
    format: Option<metal2vulkan::vk::Format>,
}

/// Whether `a` and `b` are the same descriptor: every field equal, the floats by their bits.
fn same_descriptor(a: &SamplerDescriptor, b: &SamplerDescriptor) -> bool {
    a == b
        && a.lod_min_clamp.to_bits() == b.lod_min_clamp.to_bits()
        && a.lod_max_clamp.to_bits() == b.lod_max_clamp.to_bits()
}

/// Keep `entry` at the front of `kept`, at most [`KEPT`] entries.
fn keep<T>(kept: &mut Vec<T>, entry: T) {
    if kept.len() == KEPT {
        kept.pop();
    }
    kept.insert(0, entry);
}

/// The image and sampler descriptor of `write`, a descriptor of type `ty` that is no buffer, for
/// a command (by `call`) over `tables`; the Vulkan samplers it names go to `samplers`. Views and
/// sampler descriptions are taken from `memo` where it keeps them.
fn image_descriptor(
    call: &'static str,
    device: &Device,
    tables: &Tables<'_>,
    write: &DescriptorWrite,
    ty: vk::DescriptorType,
    (samplers, memo): (&mut Vec<Arc<OwnedSampler>>, Option<&mut BindMemo>),
) -> Result<vk::DescriptorImageInfo, Error> {
    let image = |view: vk::ImageView| {
        vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)
    };
    // A slot a draw reads through a snapshot samples its texture's subresource that the
    // instance's color attachment holds in the feedback loop layout, the layout its accesses
    // there name (`render_target::moved_barriers`; "This layout is valid only for image
    // subresources of images created with the VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT
    // usage flag set", which `Texture::snapshot_color` checked).
    let in_feedback =
        |slot: SlotRef, info: vk::DescriptorImageInfo| match tables.snapshot_slot(slot) {
            true => info.image_layout(vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT),
            false => info,
        };
    // The kept views are dropped once an image of the device let its views go.
    let (mut kept_views, mut kept_samplers) = match memo {
        Some(memo) => {
            let epoch = device.shared().views_epoch.load(Ordering::Acquire);
            if memo.epoch != epoch {
                memo.views.clear();
                memo.epoch = epoch;
            }
            (Some(&mut memo.views), Some(&mut memo.samplers))
        }
        None => (None, None),
    };
    // A texture whose image and usage lack the binding's usage binds its shadow
    // (`texture/usage_shadow.rs`).
    let mut texture_view = |slot: SlotRef, view: &ImageViewDesc, storage: bool| {
        let texture = slot_texture(tables, slot)
            .ok_or_else(|| internal("the plan binds an unbound texture slot"))?;
        let attached = tables.attached(texture, storage) && !tables.snapshot_slot(slot);
        let Some(kept_views) = kept_views.as_deref_mut() else {
            return texture.slot_view(call, view, storage, attached);
        };
        let key = (texture.address(), *view, storage, attached);
        if let Some(kept) = kept_views.get(&key) {
            debug_assert_eq!(kept.texture.as_ptr() as usize, key.0);
            return Ok(kept.raw);
        }
        device.shared().counts.bind_view_worked_out();
        let raw = texture.slot_view(call, view, storage, attached)?;
        if kept_views.len() == KEPT_VIEWS {
            kept_views.clear();
        }
        let texture = texture.downgrade();
        kept_views.insert(key, KeptView { texture, raw });
        Ok::<_, Error>(raw)
    };
    let mut sampler = |slot: SlotRef, use_: SamplerUse, opaque_black, format| {
        let (state, clamps) = slot_sampler(tables, slot)
            .ok_or_else(|| internal("the plan binds an unbound sampler slot"))?;
        // Its address modes as the textures it samples map them.
        let descriptor = tables
            .sampler_maps
            .sampler(slot, bound_sampler(state, *clamps));
        let facts = SamplerFacts {
            use_,
            opaque_black,
            format,
        };
        let kept = kept_samplers.as_deref().and_then(|kept| {
            kept.iter()
                .find(|k| same_descriptor(&k.descriptor, &descriptor) && k.facts == facts)
        });
        let desc = match kept {
            Some(kept) => kept.desc,
            None => {
                let desc = metal2vulkan::bind::sampler_desc(
                    device.compile_device(),
                    &descriptor,
                    use_,
                    opaque_black,
                    format,
                )
                .map_err(Error::Bind)?;
                if let Some(kept) = kept_samplers.as_deref_mut() {
                    let entry = KeptSampler {
                        descriptor,
                        facts,
                        desc,
                    };
                    keep(kept, entry);
                }
                desc
            }
        };
        let owned = state.vk_sampler(&desc)?;
        let raw = owned.raw();
        samplers.push(owned);
        Ok::<_, Error>(raw)
    };
    let neutral_sampler = |kind: &NeutralResource| match *kind {
        NeutralResource::Sampler { compare } => device.neutral_sampler(compare),
        _ => Err(internal("a neutral sampler of a kind that is no sampler")),
    };
    Ok(match &write.source {
        WriteSource::Texture { slot, view } => in_feedback(
            *slot,
            image(texture_view(
                *slot,
                view,
                ty == vk::DescriptorType::STORAGE_IMAGE,
            )?),
        ),
        WriteSource::CombinedImageSampler {
            image: source,
            sampler: sampler_source,
        } => {
            let (view, format, slot) = match source {
                ImageSource::Texture { slot, view } => (
                    texture_view(*slot, view, false)?,
                    Some(view.format),
                    Some(*slot),
                ),
                ImageSource::Neutral(kind) => (device.neutral_view(*kind)?, kind.format(), None),
            };
            let raw = match sampler_source {
                SamplerSource::Slot {
                    slot,
                    compare,
                    integer,
                    cube_gradient,
                    opaque_black,
                } => sampler(
                    *slot,
                    SamplerUse::new(*compare, *integer).with_cube_gradient(*cube_gradient),
                    *opaque_black,
                    format,
                )?,
                SamplerSource::Immutable => vk::Sampler::null(),
                SamplerSource::Neutral(kind) => neutral_sampler(kind)?,
            };
            match slot {
                Some(slot) => in_feedback(slot, image(view).sampler(raw)),
                None => image(view).sampler(raw),
            }
        }
        WriteSource::Sampler {
            slot,
            compare,
            integer,
            cube_gradient,
            opaque_black,
        } => vk::DescriptorImageInfo::default().sampler(sampler(
            *slot,
            SamplerUse::new(*compare, *integer).with_cube_gradient(*cube_gradient),
            *opaque_black,
            None,
        )?),
        WriteSource::Neutral(kind @ NeutralResource::Image { .. }) => {
            image(device.neutral_view(*kind)?)
        }
        WriteSource::Neutral(kind @ NeutralResource::Sampler { .. }) => {
            vk::DescriptorImageInfo::default().sampler(neutral_sampler(kind)?)
        }
        WriteSource::Attachment { color_index } if ty == vk::DescriptorType::INPUT_ATTACHMENT => {
            let (pass, views) = tables
                .pass
                .filter(|(pass, _)| pass.fetch_inputs)
                .ok_or_else(|| internal("an input attachment outside a pass that reads them"))?;
            let color = pass
                .colors
                .get(*color_index as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    internal("an input attachment of a color index without a texture")
                })?;
            // The framebuffer's view of the layers the draw renders.
            let target = render_target::viewed(&color.target, views);
            image(render_target::color_view(&target, true)?)
        }
        // A tile dispatch's scratch, viewed in the staging format.
        WriteSource::TileAttachment {
            color_index,
            format,
            ..
        } if ty == vk::DescriptorType::STORAGE_IMAGE && tables.tile_scratch.is_some() => {
            let scratch = tables
                .tile_scratch
                .and_then(|scratch| scratch.iter().find(|(color, _)| color == color_index))
                .map(|(_, target)| target)
                .ok_or_else(|| internal("a tile attachment without its scratch"))?;
            image(render_target::storage_view_in(scratch, *format)?)
        }
        // The attachment's own image, or its sample planes (`sample_planes.rs`), which the
        // encoder made before the draw.
        // Viewed in `view_alias` where the pipeline reads it so (an 8-bit sRGB attachment's
        // codes, a BGRA-order one's RGBA-order twin), which the encoder made the image or planes
        // mutable to (it held the view to `framebuffer_fetch::storage_view_format`, `render.rs`).
        WriteSource::TileAttachment {
            color_index,
            format,
            view,
            view_alias,
        } if ty == vk::DescriptorType::STORAGE_IMAGE && tables.pass.is_some() => {
            let color = tables
                .pass
                .and_then(|(pass, _)| pass.colors.get(*color_index as usize))
                .and_then(Option::as_ref)
                .filter(|color| color.target.format == *format)
                .ok_or_else(|| internal("a storage attachment the pass does not have"))?;
            let in_format = view_alias.unwrap_or(*format);
            match view {
                StorageAttachmentView::Attachment => {
                    image(render_target::storage_view_in(&color.target, in_format)?)
                }
                StorageAttachmentView::SamplePlanes { layers } => {
                    let planes = color
                        .planes
                        .as_ref()
                        .filter(|planes| planes.layers == *layers && planes.view == in_format)
                        .ok_or_else(|| internal("sample planes the pass does not have"))?;
                    image(planes.draw_view()?)
                }
            }
        }
        // The shader-side stencil's shadow, which the encoder made before the draw.
        WriteSource::ShadowStencil if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables.shadow {
                Some(shadow) => image(shadow),
                None => return Err(internal("a shader-side stencil without its shadow")),
            }
        }
        // The depth stand-in, which the encoder made before the draw.
        WriteSource::DepthStandIn if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables.depth_stand_in {
                Some(stand_in) => image(stand_in),
                None => return Err(internal("a shader-side depth test without its stand-in")),
            }
        }
        // The stencil's stand-in: the pass's shadow of the stencil, which the encoder made before
        // the draw (`shadow_stencil.rs`).
        WriteSource::StencilStandIn if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables.shadow {
                Some(shadow) => image(shadow),
                None => return Err(internal("a shader-side stencil test without its stand-in")),
            }
        }
        // An extended-range color's `half` shadow, which the encoder made before the draw.
        WriteSource::ExtendedShadow { color_index } if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables
                .extended_shadows
                .iter()
                .find(|(color, _)| color == color_index)
            {
                Some(&(_, shadow)) => image(shadow),
                None => return Err(internal("an extended-range color's shadow the pass lacks")),
            }
        }
        // A slot's snapshot, which the encoder made before the draw, in `GENERAL`.
        WriteSource::SnapshotTexels { slot } if ty == vk::DescriptorType::STORAGE_IMAGE => {
            let view = tables
                .snapshots
                .and_then(|s| s.views.iter().find(|(at, _)| at == slot))
                .map(|&(_, view)| view);
            match view {
                Some(view) => image(view),
                None => return Err(internal("a snapshot the draw did not take")),
            }
        }
        // The pass's fragment imageblock storage, which the encoder made before the draw.
        WriteSource::Imageblock if ty == vk::DescriptorType::STORAGE_IMAGE => {
            match tables.imageblock {
                Some(imageblock) => image(imageblock),
                None => return Err(internal("a fragment imageblock without its storage")),
            }
        }
        WriteSource::ArgumentSampler {
            index,
            compare,
            integer,
            cube_gradient,
        } => match tables.argument_samplers.get(*index as usize) {
            Some(state) => {
                let desc = metal2vulkan::bind::sampler_desc(
                    device.compile_device(),
                    state.descriptor(),
                    SamplerUse::new(*compare, *integer).with_cube_gradient(*cube_gradient),
                    OpaqueBlackBorder::Stated,
                    None,
                )
                .map_err(Error::Bind)?;
                let owned = state.vk_sampler(&desc)?;
                let raw = owned.raw();
                samplers.push(owned);
                vk::DescriptorImageInfo::default().sampler(raw)
            }
            // An element past the listed states: no handle names it.
            None => vk::DescriptorImageInfo::default().sampler(device.neutral_sampler(*compare)?),
        },
        _ => return Err(unexpected(&write.source)),
    })
}

/// The refusal of a write source the runtime does not fill.
fn unexpected(source: &WriteSource) -> Error {
    match source {
        WriteSource::Imageblock
        | WriteSource::TileAttachment { .. }
        | WriteSource::ShadowStencil
        | WriteSource::DepthStandIn
        | WriteSource::StencilStandIn
        | WriteSource::ExtendedShadow { .. } => {
            Error::NotImplemented(RuntimeGap::RenderAttachments)
        }
        WriteSource::AccelerationStructure { .. } => {
            Error::NotImplemented(RuntimeGap::AccelerationStructures)
        }
        _ => internal("a descriptor of a type that does not hold its write source"),
    }
}

/// The buffer descriptor of `write`.
fn descriptor(
    device: &Device,
    state: &mut State,
    command: &Command<'_>,
    tables: &Tables<'_>,
    write: &DescriptorWrite,
) -> Result<vk::DescriptorBufferInfo, Error> {
    let key = (write.set, write.binding);
    let bytes = || {
        command
            .descriptor_bytes
            .iter()
            .find(|d| (d.set, d.binding) == key)
            .map(|d| d.bytes.as_slice())
            .ok_or_else(|| internal("a table write without its bytes"))
    };
    let whole = |upload: Upload| {
        vk::DescriptorBufferInfo::default()
            .buffer(upload.raw())
            .offset(upload.offset())
            .range(upload.len())
    };
    // A buffer's range is in the buffer's space, which the bind state states it in
    // (`ArgumentTable::stage_table`, `tessellation.rs`): a heap buffer's starts at its offset in
    // the heap's `VkBuffer`.
    let planned_range = || {
        command
            .descriptor_ranges
            .iter()
            .find(|r| (r.set, r.binding) == key)
            .map(|r| r.range)
            .ok_or_else(|| internal("a buffer write without its range"))
    };
    let ranged = |buffer: &Buffer| {
        let range = planned_range()?;
        let (raw, start) = buffer.space();
        Ok::<_, Error>(
            vk::DescriptorBufferInfo::default()
                .buffer(raw)
                .offset(start + range.offset)
                .range(range.range),
        )
    };
    let mut upload = |bytes: &[u8], len: u64| state.uploads.upload(device, bytes, len);
    Ok(match &write.source {
        WriteSource::Buffer { slot } => {
            let (buffer, _) = slot_buffer(tables, *slot)
                .ok_or_else(|| internal("the plan binds an unbound buffer slot"))?;
            ranged(buffer)?
        }
        // The call's copy of its bytes: the plan's range counts from the region's start.
        WriteSource::InlineBytes { slot } => {
            let bytes = slot_bytes(tables, *slot)
                .ok_or_else(|| internal("the plan binds inline bytes the slot does not hold"))?;
            let range = planned_range()?;
            if range.offset + range.range > bytes.region.len() {
                return Err(internal("an inline-bytes range past the call's region"));
            }
            vk::DescriptorBufferInfo::default()
                .buffer(bytes.region.raw())
                .offset(bytes.region.offset() + range.offset)
                .range(range.range)
        }
        WriteSource::TessellationFactors => {
            let (buffer, _) = tables
                .factors
                .ok_or_else(|| internal("a tessellation factor write without a factor buffer"))?;
            ranged(buffer)?
        }
        // The draw's patch index buffer, or zeros the control stage discards.
        WriteSource::PatchIndices { min_bytes } => match tables.patch_indices {
            Some((buffer, _)) => ranged(buffer)?,
            None => whole(upload(&[], padded(*min_bytes as usize))?),
        },
        // An emulated mesh draw's records and work buffer, whole: the draw's own regions, or the
        // command buffer's records pool (`render/mesh.rs`).
        WriteSource::MeshRecords => {
            command
                .mesh
                .ok_or_else(|| internal("a mesh records write without the draw's dispatch"))?
                .records
        }
        WriteSource::MeshWork => {
            command
                .mesh
                .ok_or_else(|| internal("a mesh work write without the draw's dispatch"))?
                .work
        }
        // An emulated tessellation draw's records and work buffer, whole: the draw's own
        // regions, or the command buffer's records pool (`render/tessellation_scan.rs`).
        WriteSource::TessellationRecords => {
            command
                .tessellation
                .ok_or_else(|| internal("a tessellation records write without the draw's scan"))?
                .records
        }
        WriteSource::TessellationWork => {
            command
                .tessellation
                .ok_or_else(|| internal("a tessellation work write without the draw's scan"))?
                .work
        }
        // An indexed patch draw's control-point index buffer.
        WriteSource::ControlPointIndices => {
            let (buffer, _) = tables.control_point_indices.ok_or_else(|| {
                internal("a control-point index write without a control-point index buffer")
            })?;
            ranged(buffer)?
        }
        // The pass's tile threadgroup memory, every tile's bytes.
        WriteSource::TileMemory { bytes } => {
            let buffer = tables
                .tile_memory
                .ok_or_else(|| internal("a tile memory write without the pass's tile memory"))?;
            let (raw, start) = buffer.space();
            vk::DescriptorBufferInfo::default()
                .buffer(raw)
                .offset(start)
                .range(*bytes)
        }
        WriteSource::ArgumentTable { .. } => {
            let bytes = bytes()?;
            whole(upload(bytes, padded(bytes.len()))?)
        }
        // A texture handle table: the device's, every live texture at its ID (`heap_sets.rs`).
        WriteSource::HandleTable { .. }
            if tables
                .texture_table
                .is_some_and(|table| table.at.contains(&key)) =>
        {
            tables
                .texture_table
                .map(|table| table.info)
                .ok_or_else(|| internal("a texture table gone"))?
        }
        WriteSource::HandleTable { bytes_len, .. } => {
            let bytes = bytes()?;
            if bytes.len() != *bytes_len as usize {
                return Err(internal("a handle table of another length than planned"));
            }
            let region = upload(bytes, padded(bytes.len()))?;
            vk::DescriptorBufferInfo::default()
                .buffer(region.raw())
                .offset(region.offset())
                .range(u64::from(*bytes_len))
        }
        WriteSource::ConstantBlob { stage } => {
            let blob = command
                .blobs
                .iter()
                .find(|(of, _)| of == stage)
                .map(|&(_, blob)| blob)
                .ok_or_else(|| internal("a constant-blob write without its stage's blob"))?;
            whole(upload(blob, padded(blob.len()))?)
        }
        // The command buffer's status words, the ones every command of it writes, on a device
        // that does not address buffers (`status_address` elsewhere).
        WriteSource::ExecutionStatus => whole(state.status_words(device)?),
        // The fragment stage's sRGB color table, the same bytes for every pipeline.
        WriteSource::ColorWriteTable => {
            let table = metal2vulkan::bind::color_write_table();
            whole(upload(table, padded(table.len()))?)
        }
        WriteSource::Neutral(
            NeutralResource::ZeroBuffer { min_bytes } | NeutralResource::Sink { min_bytes },
        ) => whole(upload(&[], padded(*min_bytes as usize))?),
        WriteSource::HeapEntry { .. } => {
            return Err(Error::NotImplemented(RuntimeGap::ResourceHeap))
        }
        // The bound sparse texture's tile map, whole.
        WriteSource::TileMap { slot } => {
            let buffer = slot_tile_map(tables, *slot)?;
            let (raw, start) = buffer.space();
            vk::DescriptorBufferInfo::default()
                .buffer(raw)
                .offset(start)
                .range(buffer.length())
        }
        // The listed tables' words as they stand now, whole, written again as the command buffer
        // commits.
        WriteSource::FunctionTableSnapshot => {
            let layout = function_table::snapshot_layout(device, tables.function_tables)?;
            let bytes = function_table::snapshot_bytes(&layout, tables.function_tables)?;
            let region = state.uploads.upload(device, &bytes, padded(bytes.len()))?;
            let info = vk::DescriptorBufferInfo::default()
                .buffer(region.raw())
                .offset(region.offset())
                .range(layout.byte_len());
            state.snapshot_tables(TableSnapshot::new(
                region,
                layout,
                tables.function_tables.to_vec(),
            ));
            info
        }
        // A listed table's words, whole; an element past the list is the null table. The plan of
        // tables that may pass `maxStorageBufferRange` reads them at their addresses instead where
        // the device has device addresses.
        WriteSource::FunctionTable { index } => match tables.function_tables.get(*index as usize) {
            Some(table) => {
                let len = table.words_len();
                let limit = device
                    .compile_device()
                    .profile()
                    .limits()
                    .max_storage_buffer_range;
                if len > u64::from(limit) {
                    return Err(function_table::past_storage_buffer_range(device));
                }
                let (raw, start) = table.words().space();
                vk::DescriptorBufferInfo::default()
                    .buffer(raw)
                    .offset(start)
                    .range(len)
            }
            None => whole(upload(
                &[],
                padded(FunctionTableWords::byte_len(0) as usize),
            )?),
        },
        other => return Err(unexpected(other)),
    })
}

/// Upload `table` with a fresh sink's device address in its sink entry, at a 32-byte-aligned
/// device address, noted for its command buffer's resident set (`live_buffers.rs`); its device
/// address.
fn root_table(device: &Device, state: &mut State, table: &RootTable) -> Result<u64, Error> {
    let no_address = || internal("a root table on a device without device addresses");
    let sink = state
        .uploads
        .upload(device, &[], u64::from(NeutralResource::SINK_BYTES))?;
    let sink = sink.device_address().ok_or_else(no_address)?;
    let mut bytes = table.bytes.clone();
    let at = RootTable::SINK_HOST_OFFSET as usize;
    bytes
        .get_mut(at..at + 8)
        .ok_or_else(|| internal("a root table without a sink entry"))?
        .copy_from_slice(&sink.to_le_bytes());
    let len = bytes.len() as u64;
    let uploaded = state
        .uploads
        .upload_aligned(device, &bytes, len, RootTable::ENTRY_ALIGNMENT)?;
    let address = uploaded.device_address().ok_or_else(no_address)?;
    let noted = NotedTable {
        upload: uploaded,
        address,
    };
    state.root_table(device, noted, table.resident_ranges())?;
    Ok(address)
}

/// What `command`, a command of `plan` over `tables`, uses, for the tracker: each bound buffer
/// the plan reads or writes, over the bytes the use reaches, each bound texture, over the levels
/// and slices it presents, and a draw's index buffer `index` and argument buffer `indirect`
/// (each with its offset), over the indices and arguments the draw reads (its index buffer's up
/// to where Metal's fetch stops reading it at most, `IndexBufferRef::fetch_end`: positions past
/// there read a zero-padded copy), in the stages the plan names and with the access types of its
/// class. The command's own uploads and the device's neutrals are fresh or never read back, so
/// nothing orders against them; a sampler is no resource.
pub(super) fn command_uses(
    call: &'static str,
    plan: &BindingPlan,
    command: &Command<'_>,
    tables: &Tables<'_>,
    index: Option<(&Buffer, u64)>,
    indirect: Option<(&Buffer, u64)>,
) -> Result<CommandUses, Error> {
    let mut uses = CommandUses::default();
    command_uses_into(call, (plan, command, tables), (index, indirect), &mut uses)?;
    Ok(uses)
}

/// [`command_uses`] in `into`, emptied first: over the last draw's storage, where the caller keeps
/// it (`LiveBind::uses`).
pub(super) fn command_uses_into(
    call: &'static str,
    (plan, command, tables): (&BindingPlan, &Command<'_>, &Tables<'_>),
    (index, indirect): (Option<(&Buffer, u64)>, Option<(&Buffer, u64)>),
    into: &mut CommandUses,
) -> Result<(), Error> {
    let CommandUses { uses, resident } = into;
    uses.clear();
    resident.clear();
    // One use at most a resource use of the plan's or the command's: room for them all at once.
    uses.reserve(plan.uses.len() + command.uses.len());
    // The keys of the buffers made resident the command reaches through the resident set, by
    // address; every other use names its resource (a descriptor, a root table's static, a view).
    // Those it names are gathered once it reaches one so: until then each use names its key.
    let mut reached = BTreeSet::new();
    let mut named = BTreeSet::new();
    let name = |named: &mut BTreeSet<ResourceKey>, reached: &BTreeSet<ResourceKey>, key| {
        if !reached.is_empty() {
            named.insert(key);
        }
    };
    for resource in plan.uses.iter().chain(command.uses) {
        if !resource.access.read && !resource.access.write {
            continue;
        }
        let stages: vk::PipelineStageFlags2 = resource.stages.into();
        if stages.is_empty() {
            return Err(internal("a command's resource use in no stage"));
        }
        let access = |flags: vk::AccessFlags2, used: bool| {
            if used {
                Access::new(stages, flags)
            } else {
                Access::NONE
            }
        };
        let (buffer, offset) = match resource.resource {
            UseTarget::Slot(slot) => match slot.class {
                SlotClass::Buffer => match slot_buffer(tables, slot) {
                    Some((buffer, offset)) => (buffer, *offset),
                    // An unbound slot reads a neutral, fresh for the command.
                    None => continue,
                },
                SlotClass::Texture => {
                    // An unbound slot takes a neutral image; a texture whose image and usage
                    // lack the binding's usage, its shadow (`texture/usage_shadow.rs`), as does
                    // a draw's sampled one its instance renders (`texture/attached_reads.rs`).
                    if let Some(texture) = slot_texture(tables, slot) {
                        let (read, write) = image_access(resource.class)?;
                        let storage = resource.class == AccessClass::StorageImage;
                        let attached =
                            tables.attached(texture, storage) && !tables.snapshot_slot(slot);
                        let used = texture.slot_use(
                            call,
                            storage,
                            attached,
                            (
                                access(read, resource.access.read),
                                access(write, resource.access.write),
                            ),
                        )?;
                        name(&mut named, &reached, used.key);
                        uses.push(used);
                    }
                    continue;
                }
                SlotClass::Sampler => continue,
            },
            UseTarget::IndexBuffer => {
                index.ok_or_else(|| internal("an index buffer use of a draw without one"))?
            }
            UseTarget::Indirect => indirect
                .ok_or_else(|| internal("an indirect use of a command without arguments"))?,
            UseTarget::TileMap(slot) => (slot_tile_map(tables, slot)?, 0),
            UseTarget::Used(token) => match tables.used(token) {
                Some(Used::Buffer(buffer)) => (buffer, 0),
                Some(Used::Heap { bytes, .. }) => (bytes, 0),
                Some(Used::IndirectCommandBuffer(icb)) => (icb.storage(), 0),
                Some(Used::Texture(texture)) => {
                    let (read, write) = image_access(resource.class)?;
                    let used = texture.bound_use(
                        access(read, resource.access.read),
                        access(write, resource.access.write),
                    );
                    name(&mut named, &reached, used.key);
                    uses.push(used);
                    continue;
                }
                None => return Err(internal("a use of a resource the encoder did not use")),
            },
            UseTarget::TessellationFactors => tables
                .factors
                .ok_or_else(|| internal("a tessellation factor use without a factor buffer"))?,
            UseTarget::PatchIndices => tables
                .patch_indices
                .ok_or_else(|| internal("a patch index use of a draw without a buffer"))?,
            UseTarget::ControlPointIndices => tables
                .control_point_indices
                .ok_or_else(|| internal("a control-point index use of a draw without a buffer"))?,
        };
        let range = match resource.range {
            Some(range) => {
                let start = offset.saturating_add(range.offset);
                start..start.saturating_add(range.len)
            }
            None => offset..buffer.length().max(offset),
        };
        let range = match resource.resource {
            UseTarget::IndexBuffer => {
                let end = IndexBufferRef::new(0, buffer.length()).fetch_end(range.start);
                range.start..range.end.min(end).max(range.start)
            }
            _ => range,
        };
        let (read, write) = buffer_access(resource.class)?;
        let (identity, range) = buffer.tracked(range);
        match resource.resource {
            UseTarget::Used(_) => {
                // The first reached: every use before it named its key.
                if reached.is_empty() {
                    named.extend(uses.iter().map(|used: &Use<ResourceKey>| used.key));
                }
                reached.insert(identity.key);
            }
            _ => name(&mut named, &reached, identity.key),
        }
        uses.push(Use {
            key: identity.key,
            tracked: identity.tracked,
            range: UseRange::Bytes(range),
            read: access(read, resource.access.read),
            write: access(write, resource.access.write),
        });
    }
    resident.extend(reached.difference(&named).copied());
    Ok(())
}

/// A dispatch's or draw's uses (`command_uses`), for the tracker, and the keys among them that
/// the command reaches through its root tables' resident set alone: buffers and heaps an encoder
/// made resident, which no use of the command names. The encoder's core takes both
/// (`EncoderCore::command_reaching`, `note_reaching`); a use added later names its key.
#[derive(Clone, Default)]
pub(super) struct CommandUses {
    uses: Vec<Use<ResourceKey>>,
    resident: BTreeSet<ResourceKey>,
}

impl CommandUses {
    /// Add `other`'s uses, as one command's: a key either names stays named, and the keys both
    /// reach through the resident set alone stay so (`render/capture.rs`, whose draws record
    /// as one command).
    pub(super) fn merge(&mut self, other: CommandUses) {
        let named = |uses: &CommandUses| -> BTreeSet<ResourceKey> {
            uses.uses
                .iter()
                .map(|used| used.key)
                .filter(|key| !uses.resident.contains(key))
                .collect()
        };
        let (own, theirs) = (named(self), named(&other));
        self.resident.retain(|key| !theirs.contains(key));
        let reached = other.resident.into_iter().filter(|key| !own.contains(key));
        self.resident.extend(reached);
        self.uses.extend(other.uses);
    }

    /// Add `used`, which names its key.
    pub(super) fn push(&mut self, used: Use<ResourceKey>) {
        self.resident.remove(&used.key);
        self.uses.push(used);
    }

    /// Add each of `uses`, which name their keys.
    pub(super) fn extend(&mut self, uses: impl IntoIterator<Item = Use<ResourceKey>>) {
        for used in uses {
            self.push(used);
        }
    }

    /// The keys the command reaches through the resident set alone.
    pub(super) fn resident(&self) -> &BTreeSet<ResourceKey> {
        &self.resident
    }
}

impl Deref for CommandUses {
    type Target = [Use<ResourceKey>];

    fn deref(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }
}

/// The read and write access types of a texture used as `class`.
fn image_access(class: AccessClass) -> Result<(vk::AccessFlags2, vk::AccessFlags2), Error> {
    use vk::AccessFlags2 as A;
    match class {
        AccessClass::Sampled => Ok((A::SHADER_SAMPLED_READ, A::NONE)),
        AccessClass::StorageImage => Ok((A::SHADER_STORAGE_READ, A::SHADER_STORAGE_WRITE)),
        _ => Err(internal("a texture use of a class no texture slot has")),
    }
}

/// The read and write access types of a buffer used as `class`.
fn buffer_access(class: AccessClass) -> Result<(vk::AccessFlags2, vk::AccessFlags2), Error> {
    use vk::AccessFlags2 as A;
    match class {
        AccessClass::Storage => Ok((A::SHADER_STORAGE_READ, A::SHADER_STORAGE_WRITE)),
        AccessClass::Uniform => Ok((A::UNIFORM_READ, A::NONE)),
        AccessClass::VertexInput => Ok((A::VERTEX_ATTRIBUTE_READ, A::NONE)),
        AccessClass::Index => Ok((A::INDEX_READ, A::NONE)),
        AccessClass::Indirect => Ok((A::INDIRECT_COMMAND_READ, A::NONE)),
        _ => Err(internal("a buffer use of a class no buffer has")),
    }
}
