//! What the raster checks share: their vertex records and vertex stage
//! (`metal2vulkan::bind::raster_check_vertex`), one-subpass render passes over images the kit
//! makes, and graphics pipelines whose viewport, scissor and depth bias are dynamic.

use super::kit::{CheckError, HostBuffer, Kit};
use super::Target;
use ash::vk;
use metal2vulkan::bind::RasterCheck;
use metal2vulkan::device::CheckAnswer;

/// The answer of `verdict`: Metal's where it holds, else the other answer, with what the device
/// did instead.
pub(super) fn answer(verdict: Result<(), String>) -> CheckAnswer {
    match verdict {
        Ok(()) => CheckAnswer::Metals,
        Err(did) => CheckAnswer::NotMetals(did),
    }
}

/// One triangle the vertex stage draws: each corner's clip position and attribute.
#[derive(Clone, Copy, Debug)]
pub(super) struct Triangle {
    pub(super) corners: [[f32; 4]; 3],
    pub(super) attributes: [[f32; 4]; 3],
}

impl Triangle {
    /// The triangle over the viewport `(x, y, width, height)` of a target, clip `(-1, -1)`,
    /// `(3, -1)` and `(-1, 3)` at depths `z` (`w` 1), each corner's attribute its window
    /// coordinates in the target, so the attribute read at a point is the point's window `x` and
    /// `y`.
    pub(super) fn covering(z: [f32; 3], viewport: (f32, f32, f32, f32)) -> Triangle {
        Triangle::over([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]], z, viewport)
    }

    /// The triangle wholly inside the clip volume of the viewport `(x, y, width, height)`, clip
    /// `(-0.75, -0.75)`, `(0.75, -0.75)` and `(-0.75, 0.75)`, as [`covering`](Self::covering)'s
    /// otherwise. The device clips nothing, so every pixel's fragments are one primitive's:
    /// Apple's GPU draws a triangle it clips as pieces whose shared edges cross pixels, each piece
    /// interpolating from its own corners, so a pixel on such an edge reads a value per piece.
    /// Over a [`CELL`]-square viewport it covers every sample of the pixels of [`COVERED`].
    pub(super) fn inside(z: [f32; 3], viewport: (f32, f32, f32, f32)) -> Triangle {
        Triangle::over([[-0.75, -0.75], [0.75, -0.75], [-0.75, 0.75]], z, viewport)
    }

    /// The triangle of clip corners `clip` at depths `z` over the viewport `(x, y, width,
    /// height)`, each corner's attribute its window coordinates in the target.
    fn over(
        clip: [[f32; 2]; 3],
        z: [f32; 3],
        (x, y, width, height): (f32, f32, f32, f32),
    ) -> Triangle {
        let corner = |i: usize| [clip[i][0], clip[i][1], z[i], 1.0];
        let window = |i: usize| {
            [
                x + (clip[i][0] + 1.0) / 2.0 * width,
                y + (clip[i][1] + 1.0) / 2.0 * height,
                0.0,
                1.0,
            ]
        };
        Triangle {
            corners: [corner(0), corner(1), corner(2)],
            attributes: [window(0), window(1), window(2)],
        }
    }

    /// The triangle of clip corners `corners` (`z` 0.5, `w` 1), every attribute 0.
    pub(super) fn at(corners: [[f32; 2]; 3]) -> Triangle {
        let corner = |i: usize| [corners[i][0], corners[i][1], 0.5, 1.0];
        Triangle {
            corners: [corner(0), corner(1), corner(2)],
            attributes: [[0.0; 4]; 3],
        }
    }
}

/// The side of the square viewport of a [`Triangle::inside`] draw whose pixels a check reads: the
/// triangle's window corners lie 3 and 21 pixels from the viewport's corner along each axis.
pub(super) const CELL: u32 = 24;

/// The pixels, along `x` and along `y` from a [`CELL`]'s corner, whose every sample the
/// [`Triangle::inside`] of the cell covers: inside its edges `x = 3`, `y = 3` and `x + y = 24`,
/// which no sample position of 2 or 4 samples lies on.
pub(super) const COVERED: std::ops::Range<u32> = 3..11;

/// The vertex records of `triangles` ([`RasterCheck::VERTICES_BINDING`]'s), as words: triangle
/// `k` is drawn by 3 vertices from first instance `k`.
pub(super) fn records(triangles: &[Triangle]) -> Vec<u32> {
    triangles
        .iter()
        .flat_map(|t| (0..3).flat_map(move |v| t.corners[v].into_iter().chain(t.attributes[v])))
        .map(f32::to_bits)
        .collect()
}

/// The vertex records of `triangles` in a storage buffer the kit keeps.
pub(super) fn vertex_buffer(
    kit: &mut Kit<'_>,
    triangles: &[Triangle],
) -> Result<HostBuffer, CheckError> {
    kit.host_buffer(&records(triangles), vk::BufferUsageFlags::STORAGE_BUFFER)
}

/// The shared vertex stage's shader module on `target`'s compile device.
pub(super) fn vertex_shader(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
) -> Result<vk::ShaderModule, CheckError> {
    let module = metal2vulkan::bind::raster_check_vertex(target.compile)
        .map_err(|e| CheckError(format!("its vertex stage does not build: {e}")))?;
    kit.shader(&module.spirv)
}

/// The shader module of a check's module as `built` (on the target's compile device); `what` names
/// it in the error where it did not build.
pub(super) fn shader_of(
    kit: &mut Kit<'_>,
    built: Result<RasterCheck, metal2vulkan::error::Error>,
    what: &str,
) -> Result<vk::ShaderModule, CheckError> {
    let module = built.map_err(|e| CheckError(format!("its {what} does not build: {e}")))?;
    kit.shader(&module.spirv)
}

/// The sample count of `samples`.
pub(super) fn sample_count(samples: u32) -> vk::SampleCountFlags {
    vk::SampleCountFlags::from_raw(samples)
}

/// Whether an optimal-tiling image of `format` with `usage` takes `samples` samples on the device.
pub(super) fn takes_samples(
    kit: &mut Kit<'_>,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    samples: u32,
) -> Result<bool, CheckError> {
    let properties = kit.physical_query(|instance, physical| {
        // SAFETY: the physical device is the instance's.
        unsafe {
            instance.get_physical_device_image_format_properties(
                physical,
                format,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::OPTIMAL,
                usage,
                vk::ImageCreateFlags::empty(),
            )
        }
    })?;
    Ok(properties.is_ok_and(|p| p.sample_counts.contains(sample_count(samples))))
}

/// The color format every raster check's pass renders: its writes are masked off, it only gives
/// the pass its sample count.
pub(super) const COLOR: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// The usage of the passes' color images.
pub(super) const COLOR_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::COLOR_ATTACHMENT;

/// The depth attachment of a [`Pass`]: its format, the usage beside the attachment's, the layout
/// the pass leaves it in, and the value its load clears it to.
#[derive(Clone, Copy)]
pub(super) struct DepthTarget {
    pub(super) format: vk::Format,
    pub(super) usage: vk::ImageUsageFlags,
    pub(super) final_layout: vk::ImageLayout,
    pub(super) clear: f32,
}

/// A one-subpass render pass over images the kit made, and its framebuffer.
pub(super) struct Pass {
    pub(super) pass: vk::RenderPass,
    pub(super) framebuffer: vk::Framebuffer,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) samples: u32,
    /// Whether the pass has its [`COLOR`] attachment.
    pub(super) color: bool,
    pub(super) depth: Option<(vk::Image, vk::ImageView)>,
    clears: Vec<vk::ClearValue>,
}

impl Pass {
    /// The render pass's begin info over the whole target.
    pub(super) fn begin(&self) -> vk::RenderPassBeginInfo<'_> {
        vk::RenderPassBeginInfo::default()
            .render_pass(self.pass)
            .framebuffer(self.framebuffer)
            .render_area(rect(0, 0, self.width, self.height))
            .clear_values(&self.clears)
    }
}

/// The rectangle `(x, y, width, height)`.
pub(super) fn rect(x: u32, y: u32, width: u32, height: u32) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D {
            x: x as i32,
            y: y as i32,
        },
        extent: vk::Extent2D { width, height },
    }
}

/// The viewport over the rectangle `(x, y, width, height)`, depth range `[0, 1]`.
pub(super) fn viewport(x: u32, y: u32, width: u32, height: u32) -> vk::Viewport {
    vk::Viewport {
        x: x as f32,
        y: y as f32,
        width: width as f32,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

/// A 2D image of `format` and `usage`, `width` by `height` at `samples`, kept by the kit, and its
/// view of `aspect`.
pub(super) fn image(
    kit: &mut Kit<'_>,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    (width, height, samples): (u32, u32, u32),
    aspect: vk::ImageAspectFlags,
) -> Result<(vk::Image, vk::ImageView), CheckError> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(sample_count(samples))
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = kit.image(&info)?;
    let view = kit.view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresources(aspect)),
    )?;
    Ok((image, view))
}

/// The one level and layer of `aspect`.
pub(super) fn subresources(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: aspect,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// A pass `width` by `height` at `samples`, with a [`COLOR`] attachment where `color` (cleared,
/// its writes masked off by the pipelines) and the depth attachment `depth` where it names one
/// (cleared to its value and stored). The pass ends with a dependency of every later transfer,
/// compute shader and fragment shader read on its attachment writes and its fragment stage's
/// storage writes, so the attachments reach their final layouts before those reads.
pub(super) fn pass(
    kit: &mut Kit<'_>,
    (width, height, samples): (u32, u32, u32),
    color: bool,
    depth: Option<DepthTarget>,
) -> Result<Pass, CheckError> {
    let mut attachments = Vec::new();
    let mut views = Vec::new();
    let mut clears = Vec::new();
    let mut color_refs = Vec::new();
    if color {
        let (_, view) = image(
            kit,
            COLOR,
            COLOR_USAGE,
            (width, height, samples),
            vk::ImageAspectFlags::COLOR,
        )?;
        color_refs.push(
            vk::AttachmentReference2::default()
                .attachment(attachments.len() as u32)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .aspect_mask(vk::ImageAspectFlags::COLOR),
        );
        attachments.push(
            vk::AttachmentDescription2::default()
                .format(COLOR)
                .samples(sample_count(samples))
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        );
        views.push(view);
        clears.push(vk::ClearValue::default());
    }
    let mut depth_ref = None;
    let mut depth_image = None;
    if let Some(d) = depth {
        let (image, view) = image(
            kit,
            d.format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | d.usage,
            (width, height, samples),
            vk::ImageAspectFlags::DEPTH,
        )?;
        depth_ref = Some(
            vk::AttachmentReference2::default()
                .attachment(attachments.len() as u32)
                .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .aspect_mask(vk::ImageAspectFlags::DEPTH),
        );
        attachments.push(
            vk::AttachmentDescription2::default()
                .format(d.format)
                .samples(sample_count(samples))
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(d.final_layout),
        );
        views.push(view);
        clears.push(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: d.clear,
                stencil: 0,
            },
        });
        depth_image = Some((image, view));
    }
    let mut subpass = vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);
    if let Some(r) = depth_ref.as_ref() {
        subpass = subpass.depth_stencil_attachment(r);
    }
    let subpasses = [subpass];
    let dependencies = [vk::SubpassDependency2::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::TRANSFER
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
        )
        .src_access_mask(
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::SHADER_WRITE,
        )
        .dst_access_mask(
            vk::AccessFlags::TRANSFER_READ
                | vk::AccessFlags::SHADER_READ
                | vk::AccessFlags::SHADER_WRITE,
        )];
    let info = vk::RenderPassCreateInfo2::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    let render_pass = kit.render_pass(&info)?;
    let framebuffer = kit.framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&views)
            .width(width)
            .height(height)
            .layers(1),
    )?;
    Ok(Pass {
        pass: render_pass,
        framebuffer,
        width,
        height,
        samples,
        color,
        depth: depth_image,
        clears,
    })
}

/// What a raster check's graphics pipeline holds beside its pass.
#[derive(Clone, Copy)]
pub(super) struct Graphics {
    pub(super) vertex: vk::ShaderModule,
    /// The fragment stage, or none (a depth-only pipeline).
    pub(super) fragment: Option<vk::ShaderModule>,
    pub(super) layout: vk::PipelineLayout,
    /// Whether the depth is tested `ALWAYS` and written.
    pub(super) depth: bool,
    /// Whether the depth bias is enabled, its factors dynamic.
    pub(super) bias: bool,
}

/// The graphics pipeline of `g` for subpass 0 of `pass`, its sample count the pass's, its
/// viewport and scissor dynamic (one each), every color write masked off, triangles filled and
/// never culled; the driver's refusal as its `VkResult`.
pub(super) fn graphics(
    kit: &mut Kit<'_>,
    pass: &Pass,
    g: Graphics,
) -> Result<vk::Pipeline, vk::Result> {
    let mut stages = vec![vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(g.vertex)
        .name(c"main")];
    if let Some(fragment) = g.fragment {
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(c"main"),
        );
    }
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewports = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(g.bias)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(sample_count(pass.samples));
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(g.depth)
        .depth_write_enable(g.depth)
        .depth_compare_op(vk::CompareOp::ALWAYS);
    let blends = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::empty())];
    let colors = match pass.color {
        true => &blends[..],
        false => &blends[..0],
    };
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(colors);
    let mut dynamic = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    if g.bias {
        dynamic.push(vk::DynamicState::DEPTH_BIAS);
    }
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewports)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(g.layout)
        .render_pass(pass.pass)
        .subpass(0);
    kit.graphics_pipeline(&info)
}

/// Record a barrier of host reads on the writes of the stages `src`: a shader stage's shader
/// writes and the transfer stage's transfer writes, each access named only with a stage that
/// makes it.
///
/// # Safety
///
/// `command` is recording, outside a render pass.
pub(super) unsafe fn host_reads(
    device: &ash::Device,
    command: vk::CommandBuffer,
    src: vk::PipelineStageFlags,
) {
    let shaders = vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER;
    let mut writes = vk::AccessFlags::empty();
    if src.intersects(shaders) {
        writes |= vk::AccessFlags::SHADER_WRITE;
    }
    if src.contains(vk::PipelineStageFlags::TRANSFER) {
        writes |= vk::AccessFlags::TRANSFER_WRITE;
    }
    let written = vk::MemoryBarrier::default()
        .src_access_mask(writes)
        .dst_access_mask(vk::AccessFlags::HOST_READ);
    // SAFETY: the caller's.
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            src,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[written],
            &[],
            &[],
        );
    }
}

/// Record a draw of triangle `k` of the vertex records through the bound pipeline, into the
/// region `(x, y, width, height)`: its viewport and scissor.
///
/// # Safety
///
/// `command` is recording inside a pass, a pipeline of [`graphics`] bound with its set.
pub(super) unsafe fn draw(
    device: &ash::Device,
    command: vk::CommandBuffer,
    k: u32,
    (x, y, width, height): (u32, u32, u32, u32),
) {
    // SAFETY: the caller's.
    unsafe {
        device.cmd_set_viewport(command, 0, &[viewport(x, y, width, height)]);
        device.cmd_set_scissor(command, 0, &[rect(x, y, width, height)]);
        device.cmd_draw(command, 3, 1, 0, k);
    }
}

/// The word no draw writes: a NaN whose every bit is set.
pub(super) const UNWRITTEN: u32 = u32::MAX;
