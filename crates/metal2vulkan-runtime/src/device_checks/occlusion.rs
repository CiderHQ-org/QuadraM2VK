//! The check of `RuntimeDriverDefect::OcclusionCountPerColorOutput`: two passes over a
//! [`SIDE`] x [`SIDE`] target of one sample, each one draw of a triangle covering it under a
//! precise occlusion query (`VK_QUERY_CONTROL_PRECISE_BIT`, as a draw under a Counting visibility
//! result is counted, `visibility.rs`): the first with one `R8G8B8A8_UNORM` attachment and a
//! fragment writing it, the second with [`OUTPUTS`] attachments and a fragment writing each
//! (`metal2vulkan::bind::color_outputs_check_fragment`). The fragments read nothing, so every
//! sample passes. Metal's answer: each query counts the target's samples. The defect's answer: the
//! second counts otherwise (ANV counts each sample once per color output). The first counting
//! otherwise answered neither. It does not run where the device enabled no `occlusionQueryPrecise`.

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, Triangle};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{color_outputs_check_fragment, RasterCheck};
use metal2vulkan::device::{CheckAnswer, VkFeature};

/// The target's side, in pixels.
const SIDE: u32 = 4;

/// The color outputs of the second draw.
const OUTPUTS: u32 = 3;

/// The attachments' format.
const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    if !(target.enabled.features).contains(&VkFeature::OcclusionQueryPrecise) {
        return CheckAnswer::NotRun("occlusionQueryPrecise is not enabled".to_owned());
    }
    let samples = u64::from(SIDE * SIDE);
    match counts(kit, target) {
        Ok([one, _]) if one != samples => CheckAnswer::Unanswered(format!(
            "a fragment of one color output counted {one} of {samples} samples"
        )),
        Ok([_, many]) if many == samples => CheckAnswer::Metals,
        Ok([_, many]) => CheckAnswer::NotMetals(format!(
            "a fragment of {OUTPUTS} color outputs counted {many} of {samples} samples"
        )),
        Err(error) => CheckAnswer::Unanswered(error.0),
    }
}

/// A one-subpass pass of `outputs` [`FORMAT`] attachments [`SIDE`] x [`SIDE`], each cleared and
/// left unstored, and its framebuffer.
fn color_pass(
    kit: &mut Kit<'_>,
    outputs: u32,
) -> Result<(vk::RenderPass, vk::Framebuffer), CheckError> {
    let mut views = Vec::new();
    for _ in 0..outputs {
        let (_, view) = raster::image(
            kit,
            FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            (SIDE, SIDE, 1),
            vk::ImageAspectFlags::COLOR,
        )?;
        views.push(view);
    }
    let attachments: Vec<vk::AttachmentDescription2<'_>> = (0..outputs)
        .map(|_| {
            vk::AttachmentDescription2::default()
                .format(FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        })
        .collect();
    let refs: Vec<vk::AttachmentReference2<'_>> = (0..outputs)
        .map(|attachment| {
            vk::AttachmentReference2::default()
                .attachment(attachment)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .aspect_mask(vk::ImageAspectFlags::COLOR)
        })
        .collect();
    let subpasses = [vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&refs)];
    let info = vk::RenderPassCreateInfo2::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    let pass = kit.render_pass(&info)?;
    let framebuffer = kit.framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(&views)
            .width(SIDE)
            .height(SIDE)
            .layers(1),
    )?;
    Ok((pass, framebuffer))
}

/// The pipeline of `vertex` and `fragment` under `layout` for `pass` of `outputs` attachments,
/// every channel written, over the whole target.
fn pipeline(
    kit: &mut Kit<'_>,
    (vertex, fragment): (vk::ShaderModule, vk::ShaderModule),
    layout: vk::PipelineLayout,
    pass: vk::RenderPass,
    outputs: u32,
) -> Result<vk::Pipeline, vk::Result> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewports = [raster::viewport(0, 0, SIDE, SIDE)];
    let scissors = [raster::rect(0, 0, SIDE, SIDE)];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blends: Vec<vk::PipelineColorBlendAttachmentState> = (0..outputs)
        .map(|_| {
            vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        })
        .collect();
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .layout(layout)
        .render_pass(pass)
        .subpass(0);
    kit.graphics_pipeline(&info)
}

/// What the two queries counted: the draw of one color output's, then of [`OUTPUTS`].
fn counts(kit: &mut Kit<'_>, target: &Target<'_>) -> Result<[u64; 2], CheckError> {
    let vertex = raster::vertex_shader(kit, target)?;
    let side = SIDE as f32;
    let covering = Triangle::covering([0.5; 3], (0.0, 0.0, side, side));
    let vertices = raster::vertex_buffer(kit, &[covering])?;
    let (set_layout, set) = kit.descriptor_set(&[SetBinding {
        binding: RasterCheck::VERTICES_BINDING,
        stages: vk::ShaderStageFlags::VERTEX,
        resource: SetResource::Buffer(vertices),
    }])?;
    let layout = kit.layout(&[set_layout], None)?;
    let queries = kit.query_pool(
        &vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(2),
    )?;
    let mut draws = Vec::with_capacity(2);
    for outputs in [1, OUTPUTS] {
        let built = color_outputs_check_fragment(target.compile, outputs);
        let fragment = raster::shader_of(kit, built, "fragment stage")?;
        let (pass, framebuffer) = color_pass(kit, outputs)?;
        let pipeline = pipeline(kit, (vertex, fragment), layout, pass, outputs)?;
        draws.push((pass, framebuffer, pipeline, outputs));
    }
    let clears = [vk::ClearValue::default(); OUTPUTS as usize];
    kit.submit(|device, command| {
        let graphics = vk::PipelineBindPoint::GRAPHICS;
        // SAFETY: recording; every pass, framebuffer and pipeline is the kit's, each pipeline of
        // its pass and of `layout`, whose set 0 is `set`; the query pool holds two queries, reset
        // before either begins.
        unsafe {
            device.cmd_reset_query_pool(command, queries, 0, 2);
            for (query, &(pass, framebuffer, pipeline, outputs)) in (0u32..).zip(&draws) {
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(framebuffer)
                    .render_area(raster::rect(0, 0, SIDE, SIDE))
                    .clear_values(&clears[..outputs as usize]);
                device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(command, graphics, pipeline);
                device.cmd_bind_descriptor_sets(command, graphics, layout, 0, &[set], &[]);
                let precise = vk::QueryControlFlags::PRECISE;
                device.cmd_begin_query(command, queries, query, precise);
                device.cmd_draw(command, 3, 1, 0, 0);
                device.cmd_end_query(command, queries, query);
                device.cmd_end_render_pass(command);
            }
        }
    })?;
    let counted = kit.query_results(queries, 2)?;
    Ok([counted[0], counted[1]])
}
