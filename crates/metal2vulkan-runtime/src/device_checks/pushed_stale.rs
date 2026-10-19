//! The check of `RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds`: three pipelines
//! of one layout, each a vertex stage of `metal2vulkan::bind::pushed_uniform_check_module` that
//! stores its own word of a uniform buffer, drawn with the rasterizer off in one render pass, in
//! the order the device's draws record: the first pipeline bound, set 0 pushed
//! (`vkCmdPushDescriptorSet`, as the device's draws push it) and drawn, then the second and the
//! third each bound and drawn with no push after the bind. The first and the third read their word
//! at a constant offset, which a driver may load ahead of the draw as constants; the second at an
//! offset of its `VertexIndex`, which it reads through the pushed descriptor itself (ANV's stale
//! read was of such a pipeline, after one of constant reads). Pipeline Layout Compatibility keeps
//! the pushed set for every later pipeline, so Metal's answer is every word stored. The defect's
//! answer: a later pipeline read another word (ANV reads zeros). The first word not stored
//! answered neither. It does not run where the device takes no push command (it binds set 0 alone
//! then) or no vertex stage's storage writes.

use super::kit::{CheckError, Kit, SetBinding, SetResource, Writes};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{pushed_uniform_check_module, DescriptorCheck};
use metal2vulkan::device::{CheckAnswer, VkFeature};

/// The uniform buffer's words (`DescriptorCheck::UNIFORM_WORDS`), each a value no other word
/// holds.
const UNIFORM: [u32; DescriptorCheck::UNIFORM_WORDS as usize] = [
    0x1357_9BDF,
    0x2468_ACE0,
    0x3579_BDF1,
    0x468A_CE02,
    0x579B_DF13,
    0x68AC_E024,
    0x79BD_F135,
    0x8ACE_0246,
];

/// The draws, each storing its word of [`UNIFORM`].
const DRAWS: usize = DescriptorCheck::UNIFORM_READS.len();

/// What the results hold before the draws.
const UNWRITTEN: u32 = u32::MAX;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    if !(target.enabled.features).contains(&VkFeature::VertexPipelineStoresAndAtomics) {
        return CheckAnswer::NotRun("vertexPipelineStoresAndAtomics is not enabled".to_owned());
    }
    let modules: Result<Vec<DescriptorCheck>, _> = (0..DRAWS as u32)
        .map(|draw| pushed_uniform_check_module(target.compile, draw))
        .collect();
    let modules = match modules {
        Ok(modules) => modules,
        Err(error) => return CheckAnswer::Unanswered(format!("its modules do not build: {error}")),
    };
    let stored = match drawn(kit, target, &modules) {
        Ok(Some(stored)) => stored,
        Ok(None) => return CheckAnswer::NotRun("the device takes no push command".to_owned()),
        Err(error) => return CheckAnswer::Unanswered(error.0),
    };
    let want = |draw: usize| UNIFORM[DescriptorCheck::UNIFORM_READS[draw] as usize];
    if stored[0] != want(0) {
        return CheckAnswer::Unanswered(format!("the first draw stored {:#010x}", stored[0]));
    }
    let stale: Vec<String> = (1..DRAWS)
        .filter(|&draw| stored[draw] != want(draw))
        .map(|draw| {
            format!(
                "draw {draw} after its pipeline's bind read {:#010x}, the pushed {:#010x}",
                stored[draw],
                want(draw)
            )
        })
        .collect();
    match stale.is_empty() {
        true => CheckAnswer::Metals,
        false => CheckAnswer::NotMetals(stale.join("; ")),
    }
}

/// The words the draws stored, `None` where the device takes no push command.
fn drawn(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    modules: &[DescriptorCheck],
) -> Result<Option<Vec<u32>>, CheckError> {
    let Some(push) = kit.push_command(target.description.api_version)? else {
        return Ok(None);
    };
    let uniform = kit.host_buffer(&UNIFORM, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
    let results = kit.host_buffer(&[UNWRITTEN; DRAWS], vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let stages = vk::ShaderStageFlags::VERTEX;
    let bindings = [
        SetBinding {
            binding: DescriptorCheck::INPUT_BINDING,
            stages,
            resource: SetResource::UniformBuffer(uniform),
        },
        SetBinding {
            binding: DescriptorCheck::OUTPUT_BINDING,
            stages,
            resource: SetResource::Buffer(results),
        },
    ];
    let set_layout = kit.set_layout(&bindings, true)?;
    let layout = kit.layout(&[set_layout], None)?;
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
    let mut pipelines = Vec::with_capacity(DRAWS);
    for module in modules {
        let shader = kit.shader(&module.spirv)?;
        let stages = [vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader)
            .name(c"main")];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::POINT_LIST);
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
        pipelines.push(kit.graphics_pipeline(&info)?);
    }
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
    let writes = Writes::of(&bindings);
    kit.submit(|device, command| {
        let writes = writes.writes(vk::DescriptorSet::null());
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        let graphics = vk::PipelineBindPoint::GRAPHICS;
        // SAFETY: recording; every pipeline is of `layout`, whose set 0's layout is a push
        // layout of `bindings`, every resource live.
        unsafe {
            device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
            for (draw, &pipeline) in pipelines.iter().enumerate() {
                device.cmd_bind_pipeline(command, graphics, pipeline);
                if draw == 0 {
                    push(
                        command,
                        graphics,
                        layout,
                        0,
                        writes.len() as u32,
                        writes.as_ptr(),
                    );
                }
                device.cmd_draw(command, 1, 1, 0, 0);
            }
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
    Ok(Some(kit.read(&results)?))
}
