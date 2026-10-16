//! A sparse case texture's image.
//!
//! The Metal oracle makes a sparse texture from a sparse heap and maps the case's regions and
//! mip tails before the work (`metal::platform::textures`). Here the same texture is an image
//! created with `VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT`, whose
//! regions and tails are bound to device memory by one `vkQueueBindSparse` on the context's queue,
//! in the case's order, and waited for before the upload is recorded. Both executors make the
//! mappings through `executor_contract::sparse::device_mappings`, here under the image's
//! `imageGranularity` and `imageMipTailFirstLod`, so a region that is not a whole number of the
//! image's blocks, or that covers part of a level in its mip tail, is refused, typed
//! (`not_supported{not_enabled}`), before anything is bound (a case's mip tail is its one level
//! whole, bound as that level's blocks where the image keeps no tail); so is a device or queue
//! without what
//! a sparse resident image needs (`sparseBinding`, `sparseResidencyImage2D` or `3D` enabled, a
//! queue with `VK_QUEUE_SPARSE_BINDING_BIT`, and the format's sparse properties). The upload
//! copies only the mapped texels: on a device that is not `residencyNonResidentStrict` a write
//! to an unbound block is undefined.
//!
//! A slot whose plan answers its residency from a tile map is what the runtime makes of it
//! ([`create_tile_mapped_image`]): each block that holds a tile the case maps under Metal's tiles
//! bound and uploaded whole, the unmapped tiles in it zero, where the context makes the sparse
//! image; an ordinary image elsewhere (`plan_resources`).

use super::device::VulkanContext;
use super::enables::ExecutorFeature;
use super::plan_resources::image_support;
use crate::check::admission::{ExecutorError, NotSupported};
use crate::executor_contract::sparse::{
    device_mappings, level_extent, literal_offset, mapped_texels, DeviceMapping, SparseTiling,
    TexelRegion,
};
use crate::literal::LiteralTexture;
use ash::vk;

/// A sparse image with its mapped regions bound, and the copies that upload their texels.
pub(super) struct SparseImage {
    pub(super) image: vk::Image,
    /// The memory every mapped block and tail is bound to; null when nothing is mapped.
    pub(super) memory: vk::DeviceMemory,
    /// The copies of exactly the mapped texels from a staging buffer of the authored bytes.
    pub(super) copies: Vec<vk::BufferImageCopy>,
}

fn not_enabled(what: String) -> ExecutorError {
    NotSupported::NotEnabled(vec![what]).into()
}

/// What the context lacks to create a sparse resident image of `info` and bind it, if anything.
fn missing_support(context: &VulkanContext, info: &vk::ImageCreateInfo<'_>) -> Vec<String> {
    let residency = match info.image_type {
        vk::ImageType::TYPE_3D => ExecutorFeature::SparseResidencyImage3D,
        _ => ExecutorFeature::SparseResidencyImage2D,
    };
    let mut missing: Vec<String> = [ExecutorFeature::SparseBinding, residency]
        .into_iter()
        .filter(|feature| !context.sparse_images.features.contains(feature))
        .map(|feature| format!("{} (a sparse case texture)", feature.name()))
        .collect();
    if !context.sparse_images.queue {
        missing.push("a queue with VK_QUEUE_SPARSE_BINDING_BIT (a sparse case texture)".into());
    }
    missing
}

/// Create the sparse image of `texture` from `info` (with the sparse flags added), bind its
/// mapped regions and tails on the context's queue, wait for the binding, and return the copies
/// that upload its mapped texels.
pub(super) fn create_sparse_image(
    context: &VulkanContext,
    label: &str,
    texture: &LiteralTexture,
    info: vk::ImageCreateInfo<'_>,
) -> Result<SparseImage, ExecutorError> {
    let info = sparse_info(info);
    check_support(context, label, &info)?;
    // SAFETY: `info` is live for the call and the device supports it.
    let image = unsafe { context.device.create_image(&info, None) }
        .map_err(|error| format!("create sparse {label}: {error}"))?;
    match bind(context, label, texture, &info, image) {
        Ok((memory, copies)) => Ok(SparseImage {
            image,
            memory,
            copies,
        }),
        Err(error) => {
            // SAFETY: the image was created above; `bind` waited for or never made a binding,
            // and freed whatever memory it allocated.
            unsafe { context.device.destroy_image(image, None) };
            Err(error)
        }
    }
}

/// `info` with the sparse flags added.
fn sparse_info(info: vk::ImageCreateInfo<'_>) -> vk::ImageCreateInfo<'_> {
    info.flags(
        info.flags | vk::ImageCreateFlags::SPARSE_BINDING | vk::ImageCreateFlags::SPARSE_RESIDENCY,
    )
}

/// Whether the context creates a sparse resident image of `info` (with the sparse flags) and
/// binds it: its features and queue, the image's format properties, and the format's sparse
/// properties of a color aspect; why not, typed, where it does not.
fn check_support(
    context: &VulkanContext,
    label: &str,
    info: &vk::ImageCreateInfo<'_>,
) -> Result<(), ExecutorError> {
    let missing = missing_support(context, info);
    if !missing.is_empty() {
        return Err(NotSupported::NotEnabled(missing).into());
    }
    image_support(context, label, info)?;
    // SAFETY: the physical device belongs to the live instance.
    let formats = unsafe {
        context
            .instance
            .get_physical_device_sparse_image_format_properties(
                context.physical,
                info.format,
                info.image_type,
                info.samples,
                info.usage,
                info.tiling,
            )
    };
    if !formats
        .iter()
        .any(|format| format.aspect_mask.contains(vk::ImageAspectFlags::COLOR))
    {
        return Err(not_enabled(format!(
            "{label}: sparse residency for VkFormat {} with usage {:#x} \
             (vkGetPhysicalDeviceSparseImageFormatProperties lists none)",
            info.format.as_raw(),
            info.usage.as_raw()
        )));
    }
    Ok(())
}

/// Create the sparse image of `texture` from `info` (with the sparse flags added), a slot whose
/// plan answers its residency from a tile map, as the runtime makes it where the device makes a
/// sparse resident image: each of the image's blocks that holds a texel of `regions` (the case's
/// mappings under Metal's tiles, `tile_map_tiling`) bound, and uploaded whole from a staging buffer
/// of the authored bytes with every texel outside `regions` zero
/// ([`mapped_bytes`](crate::executor_contract::sparse::mapped_bytes)): a mapped tile reads its
/// texels, an unmapped tile in a bound block zero, Metal's answer, and a block that holds no mapped
/// tile the device's value, as the runtime's. `Ok(None)` where the context makes no such image (the
/// caller emulates the texture by an ordinary image, as the runtime does on such a device).
pub(super) fn create_tile_mapped_image(
    context: &VulkanContext,
    label: &str,
    texture: &LiteralTexture,
    info: vk::ImageCreateInfo<'_>,
    regions: &[TexelRegion],
) -> Result<Option<SparseImage>, ExecutorError> {
    let info = sparse_info(info);
    if check_support(context, label, &info).is_err() {
        return Ok(None);
    }
    // SAFETY: `info` is live for the call and the device supports it.
    let image = unsafe { context.device.create_image(&info, None) }
        .map_err(|error| format!("create sparse {label}: {error}"))?;
    match bind_tile_mapped(context, label, texture, &info, image, regions) {
        Ok((memory, copies)) => Ok(Some(SparseImage {
            image,
            memory,
            copies,
        })),
        Err(error) => {
            // SAFETY: the image was created above; the binding was waited for or never made,
            // and whatever memory it allocated freed.
            unsafe { context.device.destroy_image(image, None) };
            Err(error)
        }
    }
}

/// One of a sparse image's units the tile-mapped path binds: a block of level 0 of a slice, or a
/// layer's mip tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Unit {
    Block { slice: u32, at: [u32; 3] },
    Tail { layer: u32 },
}

/// Bind each unit of `image` that holds a texel of `regions`, and return the memory and the
/// copies that upload each bound unit whole.
fn bind_tile_mapped(
    context: &VulkanContext,
    label: &str,
    texture: &LiteralTexture,
    info: &vk::ImageCreateInfo<'_>,
    image: vk::Image,
    regions: &[TexelRegion],
) -> Result<(vk::DeviceMemory, Vec<vk::BufferImageCopy>), ExecutorError> {
    let color = color_requirements(context, label, image)?;
    let granularity = color.format_properties.image_granularity;
    let granularity = [granularity.width, granularity.height, granularity.depth].map(|g| g.max(1));
    let single_tail = color
        .format_properties
        .flags
        .contains(vk::SparseImageFormatFlags::SINGLE_MIPTAIL);
    let tail_lod = if color.image_mip_tail_size != 0 {
        color.image_mip_tail_first_lod
    } else {
        info.mip_levels
    };
    let (extent, _) = crate::case::sparse::sparse_extent(texture.texture_type, texture.dimensions)
        .ok_or_else(|| format!("{label} is not a sparse texture type"))?;
    let mut units = std::collections::BTreeSet::new();
    for region in regions {
        if region.level >= tail_lod {
            let layer = if single_tail { 0 } else { region.slice };
            units.insert(Unit::Tail { layer });
            continue;
        }
        let end = [0, 1, 2].map(|axis| region.origin[axis] + region.size[axis]);
        let first = [0, 1, 2].map(|axis| region.origin[axis] / granularity[axis]);
        let last = [0, 1, 2].map(|axis| end[axis].div_ceil(granularity[axis]));
        for z in first[2]..last[2] {
            for y in first[1]..last[1] {
                for x in first[0]..last[0] {
                    let at = [x, y, z];
                    units.insert(Unit::Block {
                        slice: region.slice,
                        at,
                    });
                }
            }
        }
    }
    let level = level_extent(extent, 0);
    let mut mappings = Vec::new();
    let mut uploaded = Vec::new();
    for unit in units {
        match unit {
            Unit::Block { slice, at } => {
                let origin = [0, 1, 2].map(|axis| at[axis] * granularity[axis]);
                let size = [0, 1, 2].map(|axis| granularity[axis].min(level[axis] - origin[axis]));
                let texels = TexelRegion {
                    level: 0,
                    slice,
                    origin,
                    size,
                };
                mappings.push(DeviceMapping::Tiles {
                    texels,
                    tile_origin: at,
                    tile_count: [1, 1, 1],
                });
                uploaded.push(texels);
            }
            Unit::Tail { layer } => {
                mappings.push(DeviceMapping::MipTail { slice: layer });
                let slices = if single_tail {
                    0..info.array_layers
                } else {
                    layer..layer + 1
                };
                uploaded.extend(slices.map(|slice| TexelRegion {
                    level: 0,
                    slice,
                    origin: [0; 3],
                    size: level,
                }));
            }
        }
    }
    // The upload's copies first: nothing can fail once memory is bound.
    let copies = upload_copies(texture, extent, &uploaded).map_err(|e| format!("{label}: {e}"))?;
    let memory = bind_mappings(context, label, image, color, single_tail, &mappings)?;
    Ok((memory, copies))
}

/// The color aspect's sparse memory requirements of `image`.
fn color_requirements(
    context: &VulkanContext,
    label: &str,
    image: vk::Image,
) -> Result<vk::SparseImageMemoryRequirements, ExecutorError> {
    // SAFETY: the image was just created on this device.
    let sparse = unsafe { context.device.get_image_sparse_memory_requirements(image) };
    let color = sparse
        .into_iter()
        .find(|requirements| {
            requirements
                .format_properties
                .aspect_mask
                .contains(vk::ImageAspectFlags::COLOR)
        })
        .ok_or_else(|| format!("{label}: the sparse image has no color aspect's requirements"))?;
    Ok(color)
}

/// The copies that upload `regions` (each of level 0) of `texture`, whose level 0 is `extent`,
/// from a staging buffer of its literal's layout.
fn upload_copies(
    texture: &LiteralTexture,
    extent: [u32; 3],
    regions: &[TexelRegion],
) -> Result<Vec<vk::BufferImageCopy>, String> {
    regions
        .iter()
        .map(|region| {
            let level = level_extent(extent, region.level);
            Ok(vk::BufferImageCopy::default()
                .buffer_offset(literal_offset(texture, region)?)
                .buffer_row_length(level[0])
                .buffer_image_height(level[1])
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: region.level,
                    base_array_layer: region.slice,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D {
                    x: region.origin[0] as i32,
                    y: region.origin[1] as i32,
                    z: region.origin[2] as i32,
                })
                .image_extent(vk::Extent3D {
                    width: region.size[0],
                    height: region.size[1],
                    depth: region.size[2],
                }))
        })
        .collect()
}

/// Bind `texture`'s mappings of `image` and return the memory and the upload's copies.
fn bind(
    context: &VulkanContext,
    label: &str,
    texture: &LiteralTexture,
    info: &vk::ImageCreateInfo<'_>,
    image: vk::Image,
) -> Result<(vk::DeviceMemory, Vec<vk::BufferImageCopy>), ExecutorError> {
    let color = color_requirements(context, label, image)?;
    let granularity = color.format_properties.image_granularity;
    let flags = color.format_properties.flags;
    let has_tail =
        color.image_mip_tail_size != 0 && color.image_mip_tail_first_lod < info.mip_levels;
    let tiling = SparseTiling {
        tile: [granularity.width, granularity.height, granularity.depth],
        first_level_in_tail: if has_tail {
            color.image_mip_tail_first_lod
        } else {
            info.mip_levels
        },
    };
    eprintln!(
        "vulkan-sparse-image {label} format={} granularity={:?} flags={:#x} \
         mip_tail_first_lod={} mip_tail_size={} mip_tail_offset={} mip_tail_stride={}",
        info.format.as_raw(),
        tiling.tile,
        flags.as_raw(),
        color.image_mip_tail_first_lod,
        color.image_mip_tail_size,
        color.image_mip_tail_offset,
        color.image_mip_tail_stride
    );
    let mappings = device_mappings(texture, tiling)
        .map_err(|error| not_enabled(format!("{label}: {error}")))?
        .ok_or_else(|| format!("{label} is not a sparse texture"))?;
    let single_tail = flags.contains(vk::SparseImageFormatFlags::SINGLE_MIPTAIL);
    if single_tail && info.array_layers > 1 {
        let tails = mappings
            .iter()
            .filter(|mapping| matches!(mapping, DeviceMapping::MipTail { .. }))
            .count();
        if tails != 0 && tails != info.array_layers as usize {
            return Err(not_enabled(format!(
                "{label}: the device keeps one mip tail for all {} layers \
                 (VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT), and the case maps {tails} of them",
                info.array_layers
            )));
        }
    }
    // The upload's copies first: nothing can fail once memory is bound.
    let regions =
        mapped_texels(texture, tiling, &mappings).map_err(|error| format!("{label}: {error}"))?;
    let (extent, _) = crate::case::sparse::sparse_extent(texture.texture_type, texture.dimensions)
        .ok_or_else(|| format!("{label} is not a sparse texture type"))?;
    let copies =
        upload_copies(texture, extent, &regions).map_err(|error| format!("{label}: {error}"))?;
    let memory = bind_mappings(context, label, image, color, single_tail, &mappings)?;
    Ok((memory, copies))
}

/// Bind `mappings` of `image` (whose color aspect's sparse requirements are `color`, one tail
/// for all layers where `single_tail`) to memory allocated for them, in order, on the context's
/// queue, and wait for the binding; the memory, null when nothing is mapped.
fn bind_mappings(
    context: &VulkanContext,
    label: &str,
    image: vk::Image,
    color: vk::SparseImageMemoryRequirements,
    single_tail: bool,
    mappings: &[DeviceMapping],
) -> Result<vk::DeviceMemory, ExecutorError> {
    // SAFETY: the image was just created on this device.
    let requirements = unsafe { context.device.get_image_memory_requirements(image) };
    let block = requirements.alignment.max(1);
    // Where each mapping's memory starts, in the case's order; a single tail binds once.
    let mut offsets = Vec::with_capacity(mappings.len());
    let mut size: vk::DeviceSize = 0;
    let mut single_tail_bound = false;
    for mapping in mappings {
        offsets.push(size);
        size += match *mapping {
            DeviceMapping::Tiles { tile_count, .. } => {
                tile_count
                    .iter()
                    .map(|&count| u64::from(count))
                    .product::<u64>()
                    * block
            }
            DeviceMapping::MipTail { .. } if single_tail && single_tail_bound => 0,
            DeviceMapping::MipTail { .. } => {
                single_tail_bound = true;
                color.image_mip_tail_size.next_multiple_of(block)
            }
        };
    }
    let memory = if size == 0 {
        vk::DeviceMemory::null()
    } else {
        let memory_type = context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type);
        // SAFETY: `allocation` is live for the call.
        unsafe { context.device.allocate_memory(&allocation, None) }
            .map_err(|error| format!("allocate the memory of sparse {label}: {error}"))?
    };
    let free = || {
        // SAFETY: the memory was allocated above and nothing waits on it.
        unsafe { context.device.free_memory(memory, None) };
    };
    let mut blocks = Vec::new();
    let mut tails = Vec::new();
    let mut single_tail_bound = false;
    for (mapping, &offset) in mappings.iter().zip(&offsets) {
        match *mapping {
            DeviceMapping::Tiles { texels, .. } => blocks.push(
                vk::SparseImageMemoryBind::default()
                    .subresource(vk::ImageSubresource {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: texels.level,
                        array_layer: texels.slice,
                    })
                    .offset(vk::Offset3D {
                        x: texels.origin[0] as i32,
                        y: texels.origin[1] as i32,
                        z: texels.origin[2] as i32,
                    })
                    .extent(vk::Extent3D {
                        width: texels.size[0],
                        height: texels.size[1],
                        depth: texels.size[2],
                    })
                    .memory(memory)
                    .memory_offset(offset),
            ),
            DeviceMapping::MipTail { .. } if single_tail && single_tail_bound => {}
            DeviceMapping::MipTail { slice } => {
                single_tail_bound = true;
                let layer = if single_tail { 0 } else { u64::from(slice) };
                tails.push(
                    vk::SparseMemoryBind::default()
                        .resource_offset(
                            color.image_mip_tail_offset + layer * color.image_mip_tail_stride,
                        )
                        .size(color.image_mip_tail_size)
                        .memory(memory)
                        .memory_offset(offset),
                );
            }
        }
    }
    if !blocks.is_empty() || !tails.is_empty() {
        let image_binds = [vk::SparseImageMemoryBindInfo::default()
            .image(image)
            .binds(&blocks)];
        let opaque_binds = [vk::SparseImageOpaqueMemoryBindInfo::default()
            .image(image)
            .binds(&tails)];
        let mut bind_info = vk::BindSparseInfo::default();
        if !blocks.is_empty() {
            bind_info = bind_info.image_binds(&image_binds);
        }
        if !tails.is_empty() {
            bind_info = bind_info.image_opaque_binds(&opaque_binds);
        }
        // SAFETY: a fence create info has no pointers.
        let fence = match unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(fence) => fence,
            Err(error) => {
                free();
                return Err(format!("create the binding fence of sparse {label}: {error}").into());
            }
        };
        // SAFETY: every handle is live; the binds lie in the image (`device_mappings`) and the
        // memory (`offsets`); the fence is waited for before anything uses the image.
        let bound = unsafe {
            context
                .device
                .queue_bind_sparse(context.queue, &[bind_info], fence)
        }
        .map_err(|error| format!("bind the memory of sparse {label}: {error}"))
        .and_then(|()| context.wait_for_fence(fence, &format!("Vulkan sparse binding of {label}")));
        // SAFETY: the binding was waited for (a wait returns only once nothing is in flight) or
        // never submitted (a failed `vkQueueBindSparse` leaves every object unaffected).
        unsafe { context.device.destroy_fence(fence, None) };
        if let Err(error) = bound {
            free();
            return Err(error.into());
        }
    }
    Ok(memory)
}
