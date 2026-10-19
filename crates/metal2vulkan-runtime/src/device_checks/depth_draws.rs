//! What the depth checks share (`depth_range.rs`, `depth_denormals.rs`, `fragment_depth.rs`,
//! `colour_copies.rs`): the vertex stage's set of its records, and pipelines of the raster checks'
//! vertex stage (`metal2vulkan::bind::raster_check_vertex`) over a depth target, depth-only or
//! through a fragment stage that writes its depth.

use super::kit::{CheckError, HostBuffer, Kit, SetBinding, SetResource};
use super::raster;
use ash::vk;
use metal2vulkan::bind::RasterCheck;

/// The depth targets' format.
pub(super) const DEPTH: vk::Format = vk::Format::D32_SFLOAT;

/// The depth state of a [`pipeline`].
#[derive(Clone, Copy)]
pub(super) struct DepthState {
    /// The depth test's compare.
    pub(super) compare: vk::CompareOp,
    /// Whether the test writes the depth.
    pub(super) write: bool,
    /// Whether the depth is clamped to the viewport's range (`depthClampEnable`), the primitive
    /// then never clipped at the near and far planes.
    pub(super) clamp: bool,
}

/// The set of the vertex records `vertices` and the pipeline layout of it alone, kept by the kit.
pub(super) fn vertex_set(
    kit: &mut Kit<'_>,
    vertices: HostBuffer,
) -> Result<(vk::DescriptorSet, vk::PipelineLayout), CheckError> {
    let (set_layout, set) = kit.descriptor_set(&[SetBinding {
        binding: RasterCheck::VERTICES_BINDING,
        stages: vk::ShaderStageFlags::VERTEX,
        resource: SetResource::Buffer(vertices),
    }])?;
    let layout = kit.layout(&[set_layout], None)?;
    Ok((set, layout))
}

/// Whether an optimal-tiling [`DEPTH`] image of `usage` beside its attachment's is made on the
/// device.
pub(super) fn takes_depth(
    kit: &mut Kit<'_>,
    usage: vk::ImageUsageFlags,
) -> Result<bool, CheckError> {
    takes_depth_of(kit, DEPTH, usage, 1)
}

/// Whether an optimal-tiling image of the depth `format` and `usage` beside its attachment's is
/// made on the device at `samples`.
pub(super) fn takes_depth_of(
    kit: &mut Kit<'_>,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    samples: u32,
) -> Result<bool, CheckError> {
    let usage = usage | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    raster::takes_samples(kit, format, usage, samples)
}

/// The shader stages of a [`pipeline`] and its layout.
#[derive(Clone, Copy)]
pub(super) struct Stages {
    pub(super) vertex: vk::ShaderModule,
    /// The fragment stage, or none (a depth-only pipeline).
    pub(super) fragment: Option<vk::ShaderModule>,
    pub(super) layout: vk::PipelineLayout,
}

/// The pipeline of `stages` for subpass 0 of `pass`, at `samples` (sample shading on at 1 where
/// more than one, as the runtime's depth copy draws run), its viewport the `width` by `height`
/// target with depth range `[0, 1]`, its scissor dynamic, triangles filled and never culled, its
/// depth state `depth`; the driver's refusal as its `VkResult`.
pub(super) fn pipeline(
    kit: &mut Kit<'_>,
    stages: Stages,
    pass: vk::RenderPass,
    (width, height, samples): (u32, u32, u32),
    depth: DepthState,
) -> Result<vk::Pipeline, vk::Result> {
    let mut shaders = vec![vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(stages.vertex)
        .name(c"main")];
    if let Some(fragment) = stages.fragment {
        shaders.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(c"main"),
        );
    }
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewports = [raster::viewport(0, 0, width, height)];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(depth.clamp)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(raster::sample_count(samples))
        .sample_shading_enable(samples > 1)
        .min_sample_shading(1.0);
    let tests = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(depth.write)
        .depth_compare_op(depth.compare);
    let blend = vk::PipelineColorBlendStateCreateInfo::default();
    let dynamic = [vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&shaders)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&tests)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(stages.layout)
        .render_pass(pass)
        .subpass(0);
    kit.graphics_pipeline(&info)
}

/// The copy of the depth of an image's level 0 and layer 0, `width` by `height`, to or from a
/// buffer of its texels, tight.
pub(super) fn depth_copy(width: u32, height: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width,
            height,
            depth: 1,
        },
    }
}
