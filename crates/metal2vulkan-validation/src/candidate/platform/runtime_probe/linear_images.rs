//! Textures over buffers: per format Metal allows on a buffer, a linear image's
//! support, the row pitch the driver chooses, an alias of a buffer's memory at a chosen offset, and
//! the caller-chosen pitch of `DRM_FORMAT_MOD_LINEAR` with an explicit layout.

use super::{
    align_up, device_lists, flag_names, host_coherent_type, image_requirements, memory_types,
    pattern_byte, result_name, MemoryType, Owned, VulkanContext,
};
use ash::vk;
use serde::Serialize;

/// A format Metal allows on a buffer-backed texture: its Metal enumerator, the Vulkan format the
/// product maps it to directly, that format's registry name, and its bytes per texel.
pub(super) struct ProbeFormat {
    pub(super) metal: &'static str,
    pub(super) vulkan: vk::Format,
    pub(super) name: &'static str,
    pub(super) bytes: u32,
}

/// The formats measured: the one-, two- and four-channel 8-bit, and one- and four-channel 16- and
/// 32-bit formats a texture over a buffer most often takes, the display's `BGRA8Unorm`, and an
/// sRGB one.
pub(super) const FORMATS: &[ProbeFormat] = &[
    format(
        "MTLPixelFormatR8Unorm",
        vk::Format::R8_UNORM,
        "VK_FORMAT_R8_UNORM",
        1,
    ),
    format(
        "MTLPixelFormatRG8Unorm",
        vk::Format::R8G8_UNORM,
        "VK_FORMAT_R8G8_UNORM",
        2,
    ),
    format(
        "MTLPixelFormatRGBA8Unorm",
        vk::Format::R8G8B8A8_UNORM,
        "VK_FORMAT_R8G8B8A8_UNORM",
        4,
    ),
    format(
        "MTLPixelFormatRGBA8Unorm_sRGB",
        vk::Format::R8G8B8A8_SRGB,
        "VK_FORMAT_R8G8B8A8_SRGB",
        4,
    ),
    format(
        "MTLPixelFormatBGRA8Unorm",
        vk::Format::B8G8R8A8_UNORM,
        "VK_FORMAT_B8G8R8A8_UNORM",
        4,
    ),
    format(
        "MTLPixelFormatR16Float",
        vk::Format::R16_SFLOAT,
        "VK_FORMAT_R16_SFLOAT",
        2,
    ),
    format(
        "MTLPixelFormatRGBA16Float",
        vk::Format::R16G16B16A16_SFLOAT,
        "VK_FORMAT_R16G16B16A16_SFLOAT",
        8,
    ),
    format(
        "MTLPixelFormatR32Float",
        vk::Format::R32_SFLOAT,
        "VK_FORMAT_R32_SFLOAT",
        4,
    ),
    format(
        "MTLPixelFormatRGBA32Float",
        vk::Format::R32G32B32A32_SFLOAT,
        "VK_FORMAT_R32G32B32A32_SFLOAT",
        16,
    ),
    format(
        "MTLPixelFormatR32Uint",
        vk::Format::R32_UINT,
        "VK_FORMAT_R32_UINT",
        4,
    ),
    format(
        "MTLPixelFormatRGBA32Uint",
        vk::Format::R32G32B32A32_UINT,
        "VK_FORMAT_R32G32B32A32_UINT",
        16,
    ),
];

const fn format(
    metal: &'static str,
    vulkan: vk::Format,
    name: &'static str,
    bytes: u32,
) -> ProbeFormat {
    ProbeFormat {
        metal,
        vulkan,
        name,
        bytes,
    }
}

/// The widths whose row pitch is read: odd ones, where padding shows, and powers of two.
pub(super) const WIDTHS: &[u32] = &[1, 3, 17, 64, 100, 1000, 4097];

/// The height of every probe image.
const HEIGHT: u32 = 4;

/// The width of the alias round trips: odd, so no format's packed row is 16-byte aligned.
const ALIAS_WIDTH: u32 = 17;

/// The row-pitch alignments an explicit layout asks for: the packed row, and rows rounded up to
/// these powers of two (Metal's own alignments lie among them).
pub(super) const PITCH_ALIGNMENTS: &[u64] = &[1, 4, 16, 64, 128, 256];

/// `DRM_FORMAT_MOD_LINEAR`.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

const FORMAT_FEATURES: &[(u64, &str)] = &[
    (0x1, "SAMPLED_IMAGE"),
    (0x2, "STORAGE_IMAGE"),
    (0x4, "STORAGE_IMAGE_ATOMIC"),
    (0x80, "COLOR_ATTACHMENT"),
    (0x100, "COLOR_ATTACHMENT_BLEND"),
    (0x400, "BLIT_SRC"),
    (0x800, "BLIT_DST"),
    (0x1000, "SAMPLED_IMAGE_FILTER_LINEAR"),
    (0x4000, "TRANSFER_SRC"),
    (0x8000, "TRANSFER_DST"),
];

const IMAGE_USAGE: &[(u64, &str)] = &[
    (0x1, "TRANSFER_SRC"),
    (0x2, "TRANSFER_DST"),
    (0x4, "SAMPLED"),
    (0x8, "STORAGE"),
];

/// What the device answered about textures over buffers.
#[derive(Debug, Default, Serialize)]
pub(super) struct LinearTextures {
    /// The device lists `VK_EXT_image_drm_format_modifier`, and the probe's context enabled it.
    drm_format_modifier_listed: bool,
    drm_format_modifier_enabled: bool,
    /// The widths each format's pitches are read at, and the images' height.
    widths: Vec<u32>,
    height: u32,
    formats: Vec<LinearFormat>,
    /// What stopped the step.
    error: Option<String>,
    /// The validation layer's messages during the step.
    pub(super) validation_messages: Vec<String>,
}

/// One format's answers.
#[derive(Debug, Default, Serialize)]
pub(super) struct LinearFormat {
    pub(super) metal: &'static str,
    pub(super) vulkan: &'static str,
    pub(super) bytes_per_texel: u32,
    /// `linearTilingFeatures`.
    pub(super) linear_features: Vec<String>,
    /// Whether a 2D linear image with sampled, storage, and both usages is supported
    /// (`vkGetPhysicalDeviceImageFormatProperties2`).
    pub(super) linear_sampled: bool,
    pub(super) linear_storage: bool,
    pub(super) linear_sampled_and_storage: bool,
    /// The largest linear 2D extent with the probe's usage.
    pub(super) linear_max_extent: Option<[u32; 2]>,
    /// The usage the probe's linear images were created with.
    pub(super) usage: Vec<String>,
    /// The layout the driver chose per width.
    pub(super) pitches: Vec<Pitch>,
    /// A linear image bound into a buffer's memory, round-tripped both ways.
    pub(super) alias: Option<Alias>,
    /// `DRM_FORMAT_MOD_LINEAR` with an explicit layout, where the context enabled the extension.
    pub(super) drm_linear: Option<DrmLinear>,
    /// What stopped this format's measurement.
    pub(super) error: Option<String>,
}

/// A linear image's layout at one width.
#[derive(Debug, Serialize)]
pub(super) struct Pitch {
    pub(super) width: u32,
    /// `width * bytes_per_texel`.
    pub(super) packed: u64,
    /// `vkGetImageSubresourceLayout`.
    pub(super) row_pitch: u64,
    pub(super) offset: u64,
    pub(super) size: u64,
    /// `vkGetImageMemoryRequirements`: the binding offset's alignment, and the memory types.
    pub(super) alignment: u64,
    pub(super) memory_type_bits: u32,
    /// `VkMemoryDedicatedRequirements`: an image that requires its own allocation cannot alias a
    /// buffer's memory.
    pub(super) requires_dedicated: bool,
    pub(super) prefers_dedicated: bool,
}

/// An image bound into the memory of a buffer at a nonzero offset, checked by copies.
#[derive(Debug, Default, Serialize)]
pub(super) struct Alias {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) row_pitch: u64,
    /// Where in the buffer's memory the image is bound (its alignment, the smallest nonzero
    /// offset), and where its texel (0, 0) lies.
    pub(super) binding_offset: u64,
    pub(super) first_texel: u64,
    pub(super) memory_type: Option<u32>,
    /// The memory types both the image and a buffer of the alias's size allow.
    pub(super) common_memory_types: Vec<MemoryType>,
    /// Rows the host wrote into the buffer's memory, read through the image by a copy.
    pub(super) image_reads_buffer: Option<Check>,
    /// Rows a copy wrote into the image, read through the buffer by a copy.
    pub(super) buffer_reads_image: Option<Check>,
    /// The same rows, read by the host from the buffer's memory.
    pub(super) host_reads_image: Option<Check>,
    pub(super) error: Option<String>,
}

/// Bytes compared, and how many differed.
#[derive(Debug, Serialize)]
pub(super) struct Check {
    pub(super) bytes: u64,
    pub(super) mismatched: u64,
}

/// `DRM_FORMAT_MOD_LINEAR`'s answers for one format.
#[derive(Debug, Default, Serialize)]
pub(super) struct DrmLinear {
    /// The format's modifier list holds `DRM_FORMAT_MOD_LINEAR`, with these tiling features.
    pub(super) listed: bool,
    pub(super) modifiers: u32,
    pub(super) tiling_features: Vec<String>,
    /// Whether a 2D image with the modifier is supported with sampled and storage usage.
    pub(super) sampled_and_storage: bool,
    /// The usage the probe's explicit-layout images were created with (empty: none supported).
    pub(super) usage: Vec<String>,
    /// Each explicit layout asked for.
    pub(super) explicit: Vec<Explicit>,
    /// An explicit-layout image at a pitch of the caller's, bound into a buffer's memory.
    pub(super) alias: Option<Alias>,
}

/// One explicit-layout image creation.
#[derive(Debug, Serialize)]
pub(super) struct Explicit {
    pub(super) width: u32,
    /// The row pitch asked for.
    pub(super) requested: u64,
    /// `accepted`, or the result that refused it.
    pub(super) outcome: String,
    /// The row pitch `vkGetImageSubresourceLayout` reports for the accepted image.
    pub(super) row_pitch: Option<u64>,
}

/// The pitches an explicit layout asks for at `width`: the packed row, then the packed row rounded
/// up to each of [`PITCH_ALIGNMENTS`], without repeats, ascending.
pub(super) fn pitch_candidates(width: u32, bytes: u32) -> Vec<u64> {
    let packed = u64::from(width) * u64::from(bytes);
    let mut pitches = PITCH_ALIGNMENTS
        .iter()
        .map(|&alignment| align_up(packed, alignment))
        .collect::<Vec<_>>();
    pitches.sort_unstable();
    pitches.dedup();
    pitches
}

pub(super) fn measure(context: &VulkanContext) -> LinearTextures {
    let mut out = LinearTextures {
        drm_format_modifier_enabled: context
            .runtime_probe
            .is_some_and(|probe| probe.drm_format_modifier),
        widths: WIDTHS.to_vec(),
        height: HEIGHT,
        ..LinearTextures::default()
    };
    match device_lists(context, ash::ext::image_drm_format_modifier::NAME) {
        Ok(listed) => out.drm_format_modifier_listed = listed,
        Err(error) => out.error = Some(error),
    }
    for format in FORMATS {
        let mut answers = LinearFormat {
            metal: format.metal,
            vulkan: format.name,
            bytes_per_texel: format.bytes,
            ..LinearFormat::default()
        };
        if let Err(error) = measure_format(context, format, &mut answers) {
            answers.error = Some(error);
        }
        if out.drm_format_modifier_enabled {
            let drm = answers.drm_linear.insert(DrmLinear::default());
            if let Err(error) = measure_drm_linear(context, format, &answers.pitches, drm) {
                answers.error.get_or_insert(error);
            }
        }
        out.formats.push(answers);
    }
    out.validation_messages = context.take_messages();
    out
}

/// The 2D image format properties of `format` with `tiling` and `usage` (and, for the modifier
/// tiling, `DRM_FORMAT_MOD_LINEAR`), or `None` where the combination is unsupported.
fn image_format_properties(
    context: &VulkanContext,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
) -> Result<Option<vk::ImageFormatProperties>, String> {
    let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(tiling)
        .usage(usage);
    if tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT {
        info = info.push_next(&mut modifier);
    }
    let mut properties = vk::ImageFormatProperties2::default();
    // SAFETY: the physical device belongs to the live instance; the modifier structure is chained
    // only for the modifier tiling, whose extension the device lists.
    match unsafe {
        context
            .instance
            .get_physical_device_image_format_properties2(context.physical, &info, &mut properties)
    } {
        Ok(()) => Ok(Some(properties.image_format_properties)),
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => Ok(None),
        Err(error) => Err(format!(
            "vkGetPhysicalDeviceImageFormatProperties2: {}",
            result_name(error)
        )),
    }
}

/// The first usage, most capable first, that `tiling` supports for `format`, with its properties.
fn chosen_usage(
    context: &VulkanContext,
    format: vk::Format,
    tiling: vk::ImageTiling,
) -> Result<Option<(vk::ImageUsageFlags, vk::ImageFormatProperties)>, String> {
    use vk::ImageUsageFlags as U;
    let transfer = U::TRANSFER_SRC | U::TRANSFER_DST;
    for usage in [
        U::SAMPLED | U::STORAGE | transfer,
        U::SAMPLED | transfer,
        U::STORAGE | transfer,
        U::SAMPLED | U::STORAGE,
        U::SAMPLED,
        U::STORAGE,
    ] {
        if let Some(properties) = image_format_properties(context, format, tiling, usage)? {
            return Ok(Some((usage, properties)));
        }
    }
    Ok(None)
}

fn usage_names(usage: vk::ImageUsageFlags) -> Vec<String> {
    flag_names(u64::from(usage.as_raw()), IMAGE_USAGE)
}

/// A 2D, single-level, single-layer image of `format` at `width` x [`HEIGHT`], starting in
/// [`initial_layout`] of `tiling`.
fn image_info<'a>(
    format: vk::Format,
    width: u32,
    usage: vk::ImageUsageFlags,
    tiling: vk::ImageTiling,
) -> vk::ImageCreateInfo<'a> {
    vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height: HEIGHT,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(tiling)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(initial_layout(tiling))
}

/// The layout a probe image of `tiling` starts in: `PREINITIALIZED` for linear tiling, so its
/// memory's bytes are its texels, and `UNDEFINED` for the modifier tiling, which Vulkan starts in
/// no other layout (`VUID-VkImageCreateInfo-initialLayout-12478`), as the runtime's image at
/// Metal's row pitch does.
fn initial_layout(tiling: vk::ImageTiling) -> vk::ImageLayout {
    if tiling == vk::ImageTiling::LINEAR {
        vk::ImageLayout::PREINITIALIZED
    } else {
        vk::ImageLayout::UNDEFINED
    }
}

fn measure_format(
    context: &VulkanContext,
    format: &ProbeFormat,
    out: &mut LinearFormat,
) -> Result<(), String> {
    // SAFETY: the physical device belongs to the live instance.
    let properties = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.physical, format.vulkan)
    };
    out.linear_features = flag_names(
        u64::from(properties.linear_tiling_features.as_raw()),
        FORMAT_FEATURES,
    );
    let linear = vk::ImageTiling::LINEAR;
    let supported = |usage| image_format_properties(context, format.vulkan, linear, usage);
    out.linear_sampled = supported(vk::ImageUsageFlags::SAMPLED)?.is_some();
    out.linear_storage = supported(vk::ImageUsageFlags::STORAGE)?.is_some();
    out.linear_sampled_and_storage =
        supported(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)?.is_some();
    let Some((usage, limits)) = chosen_usage(context, format.vulkan, linear)? else {
        return Ok(());
    };
    out.usage = usage_names(usage);
    out.linear_max_extent = Some([limits.max_extent.width, limits.max_extent.height]);
    let mut owned = Owned::new(context);
    for &width in WIDTHS {
        if width > limits.max_extent.width || HEIGHT > limits.max_extent.height {
            continue;
        }
        let image = owned
            .image(&image_info(format.vulkan, width, usage, linear))
            .map_err(|error| {
                format!("create a linear image {width} wide: {}", result_name(error))
            })?;
        let layout = subresource_layout(context, image, vk::ImageAspectFlags::COLOR);
        let requirements = image_requirements(context, image);
        out.pitches.push(Pitch {
            width,
            packed: u64::from(width) * u64::from(format.bytes),
            row_pitch: layout.row_pitch,
            offset: layout.offset,
            size: layout.size,
            alignment: requirements.memory.alignment,
            memory_type_bits: requirements.memory.memory_type_bits,
            requires_dedicated: requirements.requires_dedicated,
            prefers_dedicated: requirements.prefers_dedicated,
        });
    }
    drop(owned);
    let transfer = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if usage.contains(transfer) && ALIAS_WIDTH <= limits.max_extent.width {
        out.alias = Some(alias_round_trip(context, format, linear, |owned| {
            owned
                .image(&image_info(format.vulkan, ALIAS_WIDTH, usage, linear))
                .map(|image| (image, vk::ImageAspectFlags::COLOR))
                .map_err(|error| format!("create the linear image: {}", result_name(error)))
        }));
    }
    Ok(())
}

/// `image`'s layout of level 0, layer 0, through `aspect`.
fn subresource_layout(
    context: &VulkanContext,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
) -> vk::SubresourceLayout {
    // SAFETY: the image is a live linear or modifier-tiled image of this device, and `aspect`
    // is the one its tiling reads layouts through.
    unsafe {
        context.device.get_image_subresource_layout(
            image,
            vk::ImageSubresource {
                aspect_mask: aspect,
                mip_level: 0,
                array_layer: 0,
            },
        )
    }
}

/// `DRM_FORMAT_MOD_LINEAR` for `format`: its entry in the modifier list, its support with
/// sampled and storage usage, every explicit layout of [`pitch_candidates`] at each width the
/// linear images took (`linear_pitches`), and an alias round trip at a pitch of the caller's.
fn measure_drm_linear(
    context: &VulkanContext,
    format: &ProbeFormat,
    linear_pitches: &[Pitch],
    out: &mut DrmLinear,
) -> Result<(), String> {
    let modifiers = modifier_list(context, format.vulkan);
    out.modifiers = modifiers.len() as u32;
    let Some(entry) = modifiers
        .iter()
        .find(|entry| entry.drm_format_modifier == DRM_FORMAT_MOD_LINEAR)
    else {
        return Ok(());
    };
    out.listed = true;
    out.tiling_features = flag_names(
        u64::from(entry.drm_format_modifier_tiling_features.as_raw()),
        FORMAT_FEATURES,
    );
    if entry.drm_format_modifier_plane_count != 1 {
        return Err(format!(
            "DRM_FORMAT_MOD_LINEAR has {} planes for {}",
            entry.drm_format_modifier_plane_count, format.name
        ));
    }
    let tiling = vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT;
    out.sampled_and_storage = image_format_properties(
        context,
        format.vulkan,
        tiling,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
    )?
    .is_some();
    let Some((usage, limits)) = chosen_usage(context, format.vulkan, tiling)? else {
        return Ok(());
    };
    out.usage = usage_names(usage);
    let mut owned = Owned::new(context);
    for &width in WIDTHS {
        if width > limits.max_extent.width || HEIGHT > limits.max_extent.height {
            continue;
        }
        for requested in pitch_candidates(width, format.bytes) {
            let layouts = [explicit_layout(requested)];
            let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
                .plane_layouts(&layouts);
            let info = image_info(format.vulkan, width, usage, tiling).push_next(&mut explicit);
            out.explicit.push(match owned.image(&info) {
                Ok(image) => Explicit {
                    width,
                    requested,
                    outcome: "accepted".into(),
                    row_pitch: Some(
                        subresource_layout(
                            context,
                            image,
                            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                        )
                        .row_pitch,
                    ),
                },
                Err(error) => Explicit {
                    width,
                    requested,
                    outcome: result_name(error),
                    row_pitch: None,
                },
            });
        }
    }
    drop(owned);
    // The alias round trip at the smallest accepted pitch of the alias width that differs from
    // the driver's own linear pitch there (a pitch the caller chose), else the smallest accepted.
    let driver = linear_pitches
        .iter()
        .find(|pitch| pitch.width == ALIAS_WIDTH)
        .map(|pitch| pitch.row_pitch);
    let accepted = out
        .explicit
        .iter()
        .filter(|attempt| {
            attempt.width == ALIAS_WIDTH && attempt.row_pitch == Some(attempt.requested)
        })
        .map(|attempt| attempt.requested)
        .collect::<Vec<_>>();
    let chosen = accepted
        .iter()
        .copied()
        .find(|&pitch| Some(pitch) != driver)
        .or_else(|| accepted.first().copied());
    let transfer = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    if let (Some(pitch), true) = (chosen, usage.contains(transfer)) {
        out.alias = Some(alias_round_trip(context, format, tiling, |owned| {
            let layouts = [explicit_layout(pitch)];
            let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
                .plane_layouts(&layouts);
            let info =
                image_info(format.vulkan, ALIAS_WIDTH, usage, tiling).push_next(&mut explicit);
            owned
                .image(&info)
                .map(|image| (image, vk::ImageAspectFlags::MEMORY_PLANE_0_EXT))
                .map_err(|error| {
                    format!("create the explicit-layout image: {}", result_name(error))
                })
        }));
    }
    Ok(())
}

/// The one plane's layout of an explicit `DRM_FORMAT_MOD_LINEAR` image of one level and layer
/// with row pitch `pitch`: at the binding's start, its size and the unused pitches zero (as the
/// explicit layout requires).
fn explicit_layout(pitch: u64) -> vk::SubresourceLayout {
    vk::SubresourceLayout {
        offset: 0,
        size: 0,
        row_pitch: pitch,
        array_pitch: 0,
        depth_pitch: 0,
    }
}

/// `format`'s modifier list (`VkDrmFormatModifierPropertiesListEXT`).
fn modifier_list(
    context: &VulkanContext,
    format: vk::Format,
) -> Vec<vk::DrmFormatModifierPropertiesEXT> {
    let count = {
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        // SAFETY: the physical device belongs to the live instance and lists the extension
        // whose structure is chained; a null array asks for the count.
        unsafe {
            context.instance.get_physical_device_format_properties2(
                context.physical,
                format,
                &mut properties,
            )
        };
        list.drm_format_modifier_count
    };
    let mut entries = vec![vk::DrmFormatModifierPropertiesEXT::default(); count as usize];
    if count != 0 {
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut entries);
        let mut properties = vk::FormatProperties2::default().push_next(&mut list);
        // SAFETY: as above; the array holds `count` entries.
        unsafe {
            context.instance.get_physical_device_format_properties2(
                context.physical,
                format,
                &mut properties,
            )
        };
        let written = list.drm_format_modifier_count as usize;
        entries.truncate(written);
    }
    entries
}

/// Bind the image `make` creates (of `tiling`, with transfer usage, starting in its
/// [`initial_layout`], at [`ALIAS_WIDTH`] x [`HEIGHT`]) into the memory of a buffer at a nonzero
/// offset, the image's own alignment, then check both directions by copies: rows the host wrote
/// through the buffer's mapping read back through the image, and rows a copy wrote into the image
/// read back through the buffer (by a copy, and by the host). An image starting `UNDEFINED` moves
/// to `GENERAL` before the host writes its rows, which a move from `UNDEFINED` need not keep, as
/// the runtime's keep writes the texels back after the first move of its image at Metal's row
/// pitch (`texture/first_move.rs`). The host writes the image's memory only in `PREINITIALIZED` or
/// `GENERAL`, where Vulkan defines host access to a linear resource's memory (a
/// `DRM_FORMAT_MOD_LINEAR` image is one).
fn alias_round_trip(
    context: &VulkanContext,
    format: &ProbeFormat,
    tiling: vk::ImageTiling,
    make: impl FnOnce(&mut Owned<'_>) -> Result<(vk::Image, vk::ImageAspectFlags), String>,
) -> Alias {
    let mut out = Alias {
        width: ALIAS_WIDTH,
        height: HEIGHT,
        ..Alias::default()
    };
    if let Err(error) = alias_into(context, format, initial_layout(tiling), make, &mut out) {
        out.error = Some(error);
    }
    out
}

fn alias_into(
    context: &VulkanContext,
    format: &ProbeFormat,
    initial: vk::ImageLayout,
    make: impl FnOnce(&mut Owned<'_>) -> Result<(vk::Image, vk::ImageAspectFlags), String>,
    out: &mut Alias,
) -> Result<(), String> {
    use vk::BufferUsageFlags as B;
    let mut owned = Owned::new(context);
    let (image, aspect) = make(&mut owned)?;
    let layout = subresource_layout(context, image, aspect);
    let requirements = image_requirements(context, image);
    if requirements.requires_dedicated {
        return Err(
            "the image requires a dedicated allocation: it cannot alias a buffer's memory".into(),
        );
    }
    let image_requirements = requirements.memory;
    let packed = u64::from(ALIAS_WIDTH) * u64::from(format.bytes);
    let height = u64::from(HEIGHT);
    out.row_pitch = layout.row_pitch;
    let offset = image_requirements.alignment.max(1);
    let first_texel = offset + layout.offset;
    out.binding_offset = offset;
    out.first_texel = first_texel;
    let span = offset + image_requirements.size;
    let shared = owned.buffer(
        &vk::BufferCreateInfo::default()
            .size(span)
            .usage(B::TRANSFER_SRC | B::TRANSFER_DST | B::STORAGE_BUFFER),
    )?;
    // SAFETY: the buffer was just created on this device.
    let buffer_requirements = unsafe { context.device.get_buffer_memory_requirements(shared) };
    let bits = image_requirements.memory_type_bits & buffer_requirements.memory_type_bits;
    out.common_memory_types = memory_types(context, bits);
    let memory_type = host_coherent_type(context, bits).ok_or_else(|| {
        format!(
            "no host-visible, host-coherent memory type among the image's {:#x} and a buffer's \
             {:#x}",
            image_requirements.memory_type_bits, buffer_requirements.memory_type_bits
        )
    })?;
    out.memory_type = Some(memory_type);
    let memory = owned.memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(span.max(buffer_requirements.size))
            .memory_type_index(memory_type),
    )?;
    // SAFETY: the buffer and image are this device's and unbound; the memory's type is one both
    // allow, it holds both ranges, and the image's offset is its own alignment.
    unsafe { context.device.bind_buffer_memory(shared, memory, 0) }
        .map_err(|error| format!("bind the buffer: {}", result_name(error)))?;
    // SAFETY: as above.
    unsafe { context.device.bind_image_memory(image, memory, offset) }
        .map_err(|error| format!("bind the image at {offset}: {}", result_name(error)))?;
    let mapped = owned.map(memory)?;
    let color = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    // Where the image keeps no bytes of its memory through its first move, it moves before the
    // host writes them, from `GENERAL` on.
    let first = if initial == vk::ImageLayout::UNDEFINED {
        owned.submit("the runtime probe's first move", |device, command| {
            let moved = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::HOST_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color);
            // SAFETY: the command buffer is recording, and the image is live and bound.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[moved],
                )
            };
        })?;
        vk::ImageLayout::GENERAL
    } else {
        initial
    };
    let row_at = |row: u64| first_texel + row * layout.row_pitch;
    // The buffer's bytes, as the host writes them through its mapping: the image's rows hold
    // pattern 1, everything else 0xee.
    {
        // SAFETY: the mapping covers the whole allocation, and no device work is in flight.
        let bytes = unsafe { std::slice::from_raw_parts_mut(mapped, span as usize) };
        bytes.fill(0xee);
        for row in 0..height {
            for column in 0..packed {
                bytes[(row_at(row) + column) as usize] = pattern_byte(1, row, column);
            }
        }
    }
    let tight = packed * height;
    let usage = B::TRANSFER_SRC | B::TRANSFER_DST;
    let (through_image, through_image_bytes) = owned.host_buffer(tight, usage)?;
    let (upload, upload_bytes) = owned.host_buffer(tight, usage)?;
    let (through_buffer, through_buffer_bytes) = owned.host_buffer(tight, usage)?;
    {
        // SAFETY: the mapping covers the buffer, and no device work is in flight.
        let bytes = unsafe { std::slice::from_raw_parts_mut(upload_bytes, tight as usize) };
        for row in 0..height {
            for column in 0..packed {
                bytes[(row * packed + column) as usize] = pattern_byte(2, row, column);
            }
        }
    }
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D {
            width: ALIAS_WIDTH,
            height: HEIGHT,
            depth: 1,
        });
    let rows = (0..height)
        .map(|row| {
            vk::BufferCopy::default()
                .src_offset(row_at(row))
                .dst_offset(row * packed)
                .size(packed)
        })
        .collect::<Vec<_>>();
    owned.submit("the runtime probe's alias copies", |device, command| {
        let host_written = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(first)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color);
        let read_then_write = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color);
        let image_write_to_buffer_read = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        let to_host = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        let transfer = vk::PipelineStageFlags::TRANSFER;
        let none = vk::DependencyFlags::empty();
        // SAFETY: the command buffer is recording; every object it names is live, the image's
        // memory is bound, and each copy stays inside its resource.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::HOST,
                transfer,
                none,
                &[],
                &[],
                &[host_written],
            );
            device.cmd_copy_image_to_buffer(
                command,
                image,
                vk::ImageLayout::GENERAL,
                through_image,
                &[region],
            );
            device.cmd_pipeline_barrier(
                command,
                transfer,
                transfer,
                none,
                &[],
                &[],
                &[read_then_write],
            );
            device.cmd_copy_buffer_to_image(
                command,
                upload,
                image,
                vk::ImageLayout::GENERAL,
                &[region],
            );
            device.cmd_pipeline_barrier(
                command,
                transfer,
                transfer,
                none,
                &[image_write_to_buffer_read],
                &[],
                &[],
            );
            device.cmd_copy_buffer(command, shared, through_buffer, &rows);
            device.cmd_pipeline_barrier(
                command,
                transfer,
                vk::PipelineStageFlags::HOST,
                none,
                &[to_host],
                &[],
                &[],
            );
        }
    })?;
    let compare = |bytes: &[u8], at: &dyn Fn(u64) -> u64, salt: u8| {
        let mut mismatched = 0;
        for row in 0..height {
            for column in 0..packed {
                if bytes[(at(row) + column) as usize] != pattern_byte(salt, row, column) {
                    mismatched += 1;
                }
            }
        }
        Check {
            bytes: tight,
            mismatched,
        }
    };
    // SAFETY: each mapping covers its allocation, and nothing is in flight any more.
    let (through_image, through_buffer, host) = unsafe {
        (
            std::slice::from_raw_parts(through_image_bytes, tight as usize),
            std::slice::from_raw_parts(through_buffer_bytes, tight as usize),
            std::slice::from_raw_parts(mapped, span as usize),
        )
    };
    let tight_row = |row: u64| row * packed;
    out.image_reads_buffer = Some(compare(through_image, &tight_row, 1));
    out.buffer_reads_image = Some(compare(through_buffer, &tight_row, 2));
    out.host_reads_image = Some(compare(host, &row_at, 2));
    Ok(())
}
