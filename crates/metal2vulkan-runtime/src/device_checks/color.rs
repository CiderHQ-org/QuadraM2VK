//! The checks of six color-write deviations Metal's rule decides (`DriverDeviation`'s
//! `MetalColorWrites`, `MetalFloatColorNans`, `MetalHalfColorNans`, `HalfColorOutputBits`,
//! `MetalSmallFloatColorWrites` and `MetalNorm16ColorWrites`): each fact's draws
//! (`metal2vulkan::bind::color_write_draws`), one triangle over a target whose fragment writes each
//! pixel's input as computed, plain and through blending with factors one and zero, read back and
//! compared with Metal's texels. A draw of a format the device does not render (or blend, for a
//! blended one) is left out, as no Metal render target of it is made there; a fact none of whose
//! draws runs is not run, as is `HalfColorOutputBits` on a device without 16-bit outputs.

use super::kit::{CheckError, HostBuffer, Kit};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{
    color_write_draws, color_write_modules, ColorDraw, ColorOutput, ColorWriteModules,
    COLOR_WRITE_FACTS,
};
use metal2vulkan::device::CheckAnswer;
use std::collections::BTreeMap;

/// A pipeline of the draws: its format, blending and output.
type Key = (vk::Format, bool, ColorOutput);

/// What the draws share: the modules' shader objects and the pipelines made so far.
struct Draws {
    vertex: vk::ShaderModule,
    float: vk::ShaderModule,
    half: Option<vk::ShaderModule>,
    pipelines: BTreeMap<Key, (vk::RenderPass, vk::Pipeline)>,
}

/// The six checks' answers on `target`, in [`COLOR_WRITE_FACTS`]' order.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    let modules = match color_write_modules(target.compile) {
        Ok(modules) => modules,
        Err(error) => {
            let why = format!("its modules do not build: {error}");
            return vec![CheckAnswer::Unanswered(why); COLOR_WRITE_FACTS.len()];
        }
    };
    let mut draws = match shaders(kit, &modules) {
        Ok(draws) => draws,
        Err(error) => return vec![CheckAnswer::Unanswered(error.0); COLOR_WRITE_FACTS.len()],
    };
    COLOR_WRITE_FACTS
        .iter()
        .map(|&fact| {
            let Some(fact_draws) = color_write_draws(fact) else {
                return CheckAnswer::Unanswered("the fact has no draws".to_owned());
            };
            let answers: Vec<CheckAnswer> = fact_draws
                .iter()
                .filter_map(|d| drawn(kit, &mut draws, d))
                .collect();
            match answers.is_empty() {
                true => CheckAnswer::NotRun("the device renders none of its draws".to_owned()),
                false => combined(answers),
            }
        })
        .collect()
}

/// One answer of several: the other answer over none, none over Metal's.
pub(super) fn combined(answers: Vec<CheckAnswer>) -> CheckAnswer {
    if let Some(other) = (answers.iter()).find(|a| matches!(a, CheckAnswer::NotMetals(_))) {
        return other.clone();
    }
    answers
        .into_iter()
        .find(|a| *a != CheckAnswer::Metals)
        .unwrap_or(CheckAnswer::Metals)
}

/// The modules' shader objects.
fn shaders(kit: &mut Kit<'_>, modules: &ColorWriteModules) -> Result<Draws, CheckError> {
    Ok(Draws {
        vertex: kit.shader(&modules.vertex)?,
        float: kit.shader(&modules.float)?,
        half: match &modules.half {
            Some(words) => Some(kit.shader(words)?),
            None => None,
        },
        pipelines: BTreeMap::new(),
    })
}

/// The answer of draw `d`, or `None` where the device cannot draw it (the module docs).
fn drawn(kit: &mut Kit<'_>, draws: &mut Draws, d: &ColorDraw) -> Option<CheckAnswer> {
    let format = vk::Format::from_raw(d.format.as_raw());
    let fragment = match d.output {
        ColorOutput::Float => draws.float,
        _ => draws.half?,
    };
    let features = match kit.physical_query(|instance, physical| {
        // SAFETY: the physical device is the instance's.
        unsafe { instance.get_physical_device_format_properties(physical, format) }
    }) {
        Ok(properties) => properties.optimal_tiling_features,
        Err(error) => return Some(CheckAnswer::Unanswered(error.0)),
    };
    let mut needed =
        vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::TRANSFER_SRC;
    if d.blended {
        needed |= vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
    }
    if !features.contains(needed) {
        return None;
    }
    Some(match texels(kit, draws, fragment, format, d) {
        Ok(texels) => match d.difference(&texels) {
            None => CheckAnswer::Metals,
            Some(how) => CheckAnswer::NotMetals(how),
        },
        Err(error) => CheckAnswer::Unanswered(error.0),
    })
}

/// The render pass and pipeline of `key`, made once.
fn pipeline(
    kit: &mut Kit<'_>,
    draws: &mut Draws,
    fragment: vk::ShaderModule,
    key: Key,
    layout: vk::PipelineLayout,
) -> Result<(vk::RenderPass, vk::Pipeline), CheckError> {
    if let Some(&made) = draws.pipelines.get(&key) {
        return Ok(made);
    }
    let (format, blended, _) = key;
    let attachments = [vk::AttachmentDescription2::default()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
    let color = [vk::AttachmentReference2::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR)];
    let subpasses = [vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color)];
    // The pass's color writes before the copy that reads them.
    let dependencies = [vk::SubpassDependency2::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
    let pass = kit.render_pass(
        &vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies),
    )?;
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(draws.vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(blended)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend_state = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
    let dynamic = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend_state)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .render_pass(pass)
        .subpass(0);
    let pipeline = kit.graphics_pipeline(&info)?;
    draws.pipelines.insert(key, (pass, pipeline));
    Ok((pass, pipeline))
}

/// The texels draw `d` leaves in its target of `format`, row by row, tightly packed.
fn texels(
    kit: &mut Kit<'_>,
    draws: &mut Draws,
    fragment: vk::ShaderModule,
    format: vk::Format,
    d: &ColorDraw,
) -> Result<Vec<u8>, CheckError> {
    let (width, height) = (ColorDraw::WIDTH, d.height);
    let values = kit.host_buffer(&d.values, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let (set, layout) = kit.storage_set(&[values], vk::ShaderStageFlags::FRAGMENT)?;
    let (pass, pipeline) = pipeline(kit, draws, fragment, (format, d.blended, d.output), layout)?;
    let image = kit.image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    )?;
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let view = kit.view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range),
    )?;
    let views = [view];
    let framebuffer = kit.framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1),
    )?;
    let bytes = d.expected.len();
    let read: HostBuffer = kit.host_buffer(
        &vec![0; bytes.div_ceil(4)],
        vk::BufferUsageFlags::TRANSFER_DST,
    )?;
    let area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D { width, height },
    };
    let clear = [vk::ClearValue {
        color: vk::ClearColorValue { uint32: [0; 4] },
    }];
    let begin = vk::RenderPassBeginInfo::default()
        .render_pass(pass)
        .framebuffer(framebuffer)
        .render_area(area)
        .clear_values(&clear);
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: width as f32,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    };
    kit.submit(|device, command| {
        let copied = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: recording; every object named is live, the image in the pass's final layout
        // for the copy, the buffer large enough for its tightly packed texels.
        unsafe {
            device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
            device.cmd_set_viewport(command, 0, &[viewport]);
            device.cmd_set_scissor(command, 0, &[area]);
            device.cmd_draw(command, 3, 1, 0, 0);
            device.cmd_end_render_pass(command);
            device.cmd_copy_image_to_buffer(
                command,
                image,
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
    let mut texels: Vec<u8> = (kit.read(&read)?.iter())
        .flat_map(|w| w.to_le_bytes())
        .collect();
    texels.truncate(bytes);
    Ok(texels)
}
