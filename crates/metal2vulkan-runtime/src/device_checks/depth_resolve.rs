//! The check of `RuntimeDriverDefect::DepthStencilResolveFirstSubresource`: a render pass whose
//! depth attachment is layer 1 of a two-layer 4-sample `D16_UNORM` image, cleared by its load to
//! [`PASS`], resolves (`SAMPLE_ZERO`) into layer 1 of a two-layer single-sample image. Before
//! the pass, the multisampled image's layer 0 is cleared to [`OTHER`] and both layers of the
//! resolve image to [`KEPT`]. Metal's answer: layer 0 keeps `KEPT` and layer 1 holds `PASS`. The
//! defect's answers: the resolve reads layer 0 or writes layer 0 (`OTHER` or `PASS` in layer 0
//! with layer 1 kept, or `OTHER` in layer 1). Anything else answered neither.

use super::kit::{CheckError, Kit};
use super::Target;
use ash::vk;
use metal2vulkan::device::{CheckAnswer, VkFeature};

/// The pass's clear of the attachment's layer.
const PASS: f32 = 0.5;

/// The multisampled image's other layer.
const OTHER: f32 = 0.25;

/// The resolve image's layers before the pass.
const KEPT: f32 = 0.75;

/// The images' side, in texels.
const SIDE: u32 = 4;

const FORMAT: vk::Format = vk::Format::D16_UNORM;

/// What one resolved layer held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    Pass,
    Other,
    Kept,
    /// A value of neither, or texels that differ: the codes.
    Neither(u16, u16),
}

/// The check's answer on `target`: not run where the device takes no 4-sample two-layer
/// `D16_UNORM` attachment or no `SAMPLE_ZERO` depth resolve, or its portability subset withholds
/// `multisampleArrayImage`, without which no multisampled image has two layers
/// (`VUID-VkImageCreateInfo-multisampleArrayImage-04460`; MoltenVK's on the Apple GPUs).
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let enabled = target.compile.profile().enabled();
    if enabled.portability_withholds(VkFeature::MultisampleArrayImage) {
        return CheckAnswer::NotRun(
            "the portability subset withholds multisampleArrayImage".to_owned(),
        );
    }
    match unsupported(kit) {
        Ok(None) => {}
        Ok(Some(why)) => return CheckAnswer::NotRun(why),
        Err(error) => return CheckAnswer::Unanswered(error.0),
    }
    match resolved(kit) {
        Ok(layers) => answer(layers),
        Err(error) => CheckAnswer::Unanswered(error.0),
    }
}

/// Why the device cannot run the check, where it cannot.
fn unsupported(kit: &mut Kit<'_>) -> Result<Option<String>, CheckError> {
    kit.physical_query(|instance, physical| {
        let mut resolve = vk::PhysicalDeviceDepthStencilResolveProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut resolve);
        // SAFETY: the physical device is the instance's, at Vulkan 1.2 or later (the structure
        // is core there).
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        if !(resolve.supported_depth_resolve_modes).contains(vk::ResolveModeFlags::SAMPLE_ZERO) {
            return Some("the device resolves no depth by sample 0".to_owned());
        }
        // SAFETY: as above.
        let format = unsafe { instance.get_physical_device_format_properties(physical, FORMAT) };
        let needed = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        if !format.optimal_tiling_features.contains(needed) {
            return Some("D16_UNORM is no copyable depth attachment".to_owned());
        }
        // SAFETY: as above.
        let image = unsafe {
            instance.get_physical_device_image_format_properties(
                physical,
                FORMAT,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                multisampled_usage(),
                vk::ImageCreateFlags::empty(),
            )
        };
        match image {
            Ok(p)
                if p.max_array_layers >= 2
                    && p.sample_counts.contains(vk::SampleCountFlags::TYPE_4) =>
            {
                None
            }
            _ => Some("no two-layer 4-sample D16_UNORM attachment".to_owned()),
        }
    })
}

fn multisampled_usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST
}

/// The answer of the two resolved layers (the module docs).
fn answer(layers: [Held; 2]) -> CheckAnswer {
    match layers {
        [Held::Kept, Held::Pass] => CheckAnswer::Metals,
        [Held::Other | Held::Pass, Held::Kept] | [Held::Kept, Held::Other] => {
            CheckAnswer::NotMetals(format!("the resolved layers held {layers:?}"))
        }
        layers => CheckAnswer::Unanswered(format!("the resolved layers held {layers:?}")),
    }
}

/// `value`'s `D16_UNORM` code.
fn code(value: f32) -> i32 {
    (value * 65535.0).round() as i32
}

/// What a layer of `codes` held.
fn held(codes: &[u16]) -> Held {
    let first = codes.first().copied().unwrap_or_default();
    let last = codes.last().copied().unwrap_or_default();
    if codes.iter().any(|&c| c != first) {
        return Held::Neither(first, last);
    }
    let near = |value: f32| (i32::from(first) - code(value)).abs() <= 2;
    match () {
        () if near(PASS) => Held::Pass,
        () if near(OTHER) => Held::Other,
        () if near(KEPT) => Held::Kept,
        () => Held::Neither(first, last),
    }
}

/// A depth range of `layers` layers from `base`.
fn layers(base: u32, count: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: base,
        layer_count: count,
    }
}

/// What the resolve left in the resolve image's two layers.
fn resolved(kit: &mut Kit<'_>) -> Result<[Held; 2], CheckError> {
    let extent = vk::Extent3D {
        width: SIDE,
        height: SIDE,
        depth: 1,
    };
    let image = |samples, usage| {
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(extent)
            .mip_levels(1)
            .array_layers(2)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    };
    let multisampled = kit.image(&image(vk::SampleCountFlags::TYPE_4, multisampled_usage()))?;
    let resolve_usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let target = kit.image(&image(vk::SampleCountFlags::TYPE_1, resolve_usage))?;
    let view = |image| {
        vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(FORMAT)
            .subresource_range(layers(1, 1))
    };
    let attachment_view = kit.view(&view(multisampled))?;
    let resolve_view = kit.view(&view(target))?;
    let attachments = [
        vk::AttachmentDescription2::default()
            .format(FORMAT)
            .samples(vk::SampleCountFlags::TYPE_4)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        vk::AttachmentDescription2::default()
            .format(FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
    ];
    let depth = vk::AttachmentReference2::default()
        .attachment(0)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH);
    let resolve = vk::AttachmentReference2::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH);
    let mut resolving = vk::SubpassDescriptionDepthStencilResolve::default()
        .depth_resolve_mode(vk::ResolveModeFlags::SAMPLE_ZERO)
        .stencil_resolve_mode(vk::ResolveModeFlags::NONE)
        .depth_stencil_resolve_attachment(&resolve);
    let subpasses = [vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth)
        .push_next(&mut resolving)];
    let attachment_stages = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
        | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    let attachment_access = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    let dependencies = [
        vk::SubpassDependency2::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_stage_mask(attachment_stages)
            .dst_access_mask(attachment_access),
        vk::SubpassDependency2::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(attachment_stages)
            .src_access_mask(attachment_access)
            .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
    ];
    let pass = kit.render_pass(
        &vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies),
    )?;
    let views = [attachment_view, resolve_view];
    let framebuffer = kit.framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(&views)
            .width(SIDE)
            .height(SIDE)
            .layers(1),
    )?;
    let bytes = vec![0u32; (SIDE * SIDE) as usize];
    let read = kit.host_buffer(&bytes, vk::BufferUsageFlags::TRANSFER_DST)?;
    let clears = [
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: PASS,
                stencil: 0,
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        },
    ];
    let begin = vk::RenderPassBeginInfo::default()
        .render_pass(pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: SIDE,
                height: SIDE,
            },
        })
        .clear_values(&clears);
    kit.submit(|device, command| {
        let to_transfer = |image, range| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(range)
        };
        let fresh = [
            to_transfer(multisampled, layers(0, 2)),
            to_transfer(target, layers(0, 2)),
        ];
        let first_to_source = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target)
            .subresource_range(layers(0, 1));
        let clear = |value| vk::ClearDepthStencilValue {
            depth: value,
            stencil: 0,
        };
        let copy = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 2,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: extent,
        };
        let copied = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: recording; every object named is live, each image in the layout its barriers
        // and the pass leave it in.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &fresh,
            );
            device.cmd_clear_depth_stencil_image(
                command,
                multisampled,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear(OTHER),
                &[layers(0, 1)],
            );
            device.cmd_clear_depth_stencil_image(
                command,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear(KEPT),
                &[layers(0, 2)],
            );
            device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
            device.cmd_end_render_pass(command);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[first_to_source],
            );
            device.cmd_copy_image_to_buffer(
                command,
                target,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                read.buffer,
                &[copy],
            );
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[copied],
                &[],
                &[],
            );
        }
    })?;
    let words = kit.read(&read)?;
    let codes: Vec<u16> = words
        .iter()
        .flat_map(|w| [*w as u16, (*w >> 16) as u16])
        .collect();
    let texels = (SIDE * SIDE) as usize;
    Ok([held(&codes[..texels]), held(&codes[texels..])])
}
