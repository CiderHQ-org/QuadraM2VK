//! The check of `DriverFact::RobustIndexFetchReadsZero`: one indexed indirect draw of
//! [`DRAWN`] indices from an index buffer range of [`INSIDE`], bound as the device's draws bind
//! one (a range of a size where it enables `maintenance5`, else the `VkBuffer`'s end), the memory
//! past the range holding [`JUNK`]. Each vertex marks its `VertexIndex`
//! (`metal2vulkan::bind::index_fetch_check_module`) with the rasterizer off. Metal's answer: the
//! indices inside mark `VERTEX_OFFSET + 1` to `VERTEX_OFFSET + INSIDE`, and every fetch past the
//! range reads 0 and marks `VERTEX_OFFSET` alone. A device that marks the indices inside but not
//! that, or marks another, gives the other answer; one that misses an index inside answered
//! neither.

use super::kit::{CheckError, Kit};
use super::Target;
use ash::vk;
use metal2vulkan::bind::IndexFetchCheck;
use metal2vulkan::device::{CheckAnswer, VkFeature};

/// The indices inside the bound range: `1..=INSIDE`, a whole number of triangles.
const INSIDE: u32 = 18;

/// The indices the draw fetches: twice the range, a whole number of triangles.
const DRAWN: u32 = 36;

/// The draw's vertex offset, added to every index fetched.
const VERTEX_OFFSET: u32 = 100;

/// What the memory past the range holds: its fetch would mark `VERTEX_OFFSET + JUNK`.
const JUNK: u32 = 300;

/// The words of the index buffer's memory.
const MEMORY_WORDS: usize = 64;

/// The check's answer on `target`: not run where the device does not enable
/// `robustBufferAccess2`, which the fact is read under alone, or a vertex stage's storage writes,
/// which the check needs.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let features = &target.enabled.features;
    if !features.contains(&VkFeature::RobustBufferAccess2) {
        return CheckAnswer::NotRun("robustBufferAccess2 is not enabled".to_owned());
    }
    if !features.contains(&VkFeature::VertexPipelineStoresAndAtomics) {
        return CheckAnswer::NotRun("vertexPipelineStoresAndAtomics is not enabled".to_owned());
    }
    match marked(kit, target) {
        Ok(flags) => answer(&flags),
        Err(error) => CheckAnswer::Unanswered(error.0),
    }
}

/// The answer of the flags a draw left (the module docs).
fn answer(flags: &[u32]) -> CheckAnswer {
    let marked: Vec<u32> = (0..flags.len() as u32)
        .filter(|&i| flags[i as usize] != 0)
        .collect();
    let inside: Vec<u32> = (VERTEX_OFFSET + 1..=VERTEX_OFFSET + INSIDE).collect();
    if !inside.iter().all(|i| marked.contains(i)) {
        return CheckAnswer::Unanswered(format!("the draw marked {marked:?}"));
    }
    let mut metals = inside;
    metals.insert(0, VERTEX_OFFSET);
    match marked == metals {
        true => CheckAnswer::Metals,
        false => CheckAnswer::NotMetals(format!("the draw marked {marked:?}")),
    }
}

/// The flags the check's draw leaves on the device.
fn marked(kit: &mut Kit<'_>, target: &Target<'_>) -> Result<Vec<u32>, CheckError> {
    let module = metal2vulkan::bind::index_fetch_check_module(target.compile)
        .map_err(|e| CheckError(format!("its module does not build: {e}")))?;
    let sized = kit.sized_index_bind()?;
    let flags = kit.host_buffer(
        &[0; IndexFetchCheck::FLAGS as usize],
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let mut words: Vec<u32> = (1..=INSIDE).collect();
    words.resize(MEMORY_WORDS, JUNK);
    let usage = vk::BufferUsageFlags::INDEX_BUFFER;
    let indices = match sized {
        Some(_) => kit.host_buffer(&words, usage)?,
        None => kit.host_buffer_in(&words, INSIDE as usize, usage)?,
    };
    let arguments = kit.host_buffer(
        &[DRAWN, 1, 0, VERTEX_OFFSET, 0],
        vk::BufferUsageFlags::INDIRECT_BUFFER,
    )?;
    let shader = kit.shader(&module.spirv)?;
    let (set, layout) = kit.storage_set(&[flags], vk::ShaderStageFlags::VERTEX)?;
    let subpasses =
        [vk::SubpassDescription2::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];
    let pass = kit.render_pass(&vk::RenderPassCreateInfo2::default().subpasses(&subpasses))?;
    let framebuffer = kit.framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .width(1)
            .height(1)
            .layers(1),
    )?;
    let stages = [vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(shader)
        .name(c"main")];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .rasterizer_discard_enable(true)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .rasterization_state(&raster)
        .layout(layout)
        .render_pass(pass)
        .subpass(0);
    let pipeline = kit.graphics_pipeline(&info)?;
    let begin = vk::RenderPassBeginInfo::default()
        .render_pass(pass)
        .framebuffer(framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
        });
    let range = vk::DeviceSize::from(INSIDE * 4);
    kit.submit(|device, command| {
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: recording; every object named is live, and the sized bind is the device's
        // own command, its range inside the buffer.
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
            match sized {
                Some(bind) => bind(command, indices.buffer, 0, range, vk::IndexType::UINT32),
                None => {
                    device.cmd_bind_index_buffer(command, indices.buffer, 0, vk::IndexType::UINT32)
                }
            }
            device.cmd_draw_indexed_indirect(command, arguments.buffer, 0, 1, 20);
            device.cmd_end_render_pass(command);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[],
            );
        }
    })?;
    kit.read(&flags)
}
