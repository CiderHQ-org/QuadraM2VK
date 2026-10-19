//! The check of `RuntimeDriverDefect::PushedBufferSizesUnset`, settled as the device is made: it
//! decides whether the device takes the push command at all (`device/mod.rs`), so every pipeline
//! layout's kind. Two dispatches of `metal2vulkan::bind::array_length_check_module` in one
//! command buffer, each storing the runtime array length of its sized buffer: the first with set 0
//! bound (`vkCmdBindDescriptorSets`), its sized buffer [`BOUND_WORDS`] words, the second with set 0
//! pushed (`vkCmdPushDescriptorSet`, as the device's draws and dispatches push it), its sized
//! buffer [`PUSHED_WORDS`] words. Metal's answer: each its own length. The defect's answer: the
//! pushed dispatch read another length, none or the bound one (MoltenVK's pushed set never fills
//! the sizes a bind left), which the device then avoids by binding set 0 alone. The bound dispatch
//! reading another length answered neither. It pushes with the driver's own command, whether or
//! not the device takes it (`kit::Kit::driver_push_command`), and does not run where the driver
//! has none.

use super::kit::{CheckError, HostBuffer, Kit, SetBinding, SetResource, Writes};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{array_length_check_module, DescriptorCheck};
use metal2vulkan::device::CheckAnswer;

/// The bound dispatch's sized buffer's words.
const BOUND_WORDS: u32 = 23;

/// The pushed dispatch's sized buffer's words: an odd count no default size reads as, and not the
/// bound one.
const PUSHED_WORDS: u32 = 37;

/// What a result word holds before its dispatch.
const UNWRITTEN: u32 = u32::MAX;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let module = match array_length_check_module(target.compile) {
        Ok(module) => module,
        Err(error) => {
            return CheckAnswer::Unanswered(format!("its kernel does not build: {error}"))
        }
    };
    match lengths(kit, target, &module) {
        Ok(None) => CheckAnswer::NotRun("the driver has no push command".to_owned()),
        Ok(Some([bound, _])) if bound != BOUND_WORDS => CheckAnswer::Unanswered(format!(
            "the bound buffer's length read {bound:#x}, its {BOUND_WORDS} words"
        )),
        Ok(Some([_, UNWRITTEN])) => {
            CheckAnswer::Unanswered("the pushed dispatch wrote no length".to_owned())
        }
        Ok(Some([_, PUSHED_WORDS])) => CheckAnswer::Metals,
        Ok(Some([_, pushed])) => CheckAnswer::NotMetals(format!(
            "the pushed buffer's length read {pushed}, its {PUSHED_WORDS} words (the buffer \
             bound before it {BOUND_WORDS})"
        )),
        Err(error) => CheckAnswer::Unanswered(error.0),
    }
}

/// The bindings of one dispatch: the sized buffer and its result.
fn bindings(sized: HostBuffer, result: HostBuffer) -> [SetBinding; 2] {
    let stages = vk::ShaderStageFlags::COMPUTE;
    [
        SetBinding {
            binding: DescriptorCheck::INPUT_BINDING,
            stages,
            resource: SetResource::Buffer(sized),
        },
        SetBinding {
            binding: DescriptorCheck::OUTPUT_BINDING,
            stages,
            resource: SetResource::Buffer(result),
        },
    ]
}

/// The lengths the bound and the pushed dispatch read, `None` where the driver has no push
/// command.
fn lengths(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    module: &DescriptorCheck,
) -> Result<Option<[u32; 2]>, CheckError> {
    let Some(push) = kit.driver_push_command(target.description.api_version)? else {
        return Ok(None);
    };
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let mut buffers = Vec::with_capacity(2);
    for words in [BOUND_WORDS, PUSHED_WORDS] {
        let sized = kit.host_buffer(&vec![0; words as usize], storage)?;
        let result = kit.host_buffer(&[UNWRITTEN], storage)?;
        buffers.push((sized, result));
    }
    let (bound, pushed) = (
        bindings(buffers[0].0, buffers[0].1),
        bindings(buffers[1].0, buffers[1].1),
    );
    let (bound_layout, set) = kit.descriptor_set(&bound)?;
    let bound_layout = kit.layout(&[bound_layout], None)?;
    let pushed_layout = kit.set_layout(&pushed, true)?;
    let pushed_layout = kit.layout(&[pushed_layout], None)?;
    let shader = kit.shader(&module.spirv)?;
    let bound_pipeline = kit.compute_pipeline(shader, bound_layout)?;
    let pushed_pipeline = kit.compute_pipeline(shader, pushed_layout)?;
    let writes = Writes::of(&pushed);
    kit.submit(|device, command| {
        let writes = writes.writes(vk::DescriptorSet::null());
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        let compute = vk::PipelineBindPoint::COMPUTE;
        // SAFETY: recording, outside a render pass; each pipeline is of its layout, the bound
        // one's set 0 `set`, the pushed one's a push layout of `pushed`, every resource live.
        unsafe {
            device.cmd_bind_pipeline(command, compute, bound_pipeline);
            device.cmd_bind_descriptor_sets(command, compute, bound_layout, 0, &[set], &[]);
            device.cmd_dispatch(command, 1, 1, 1);
            device.cmd_bind_pipeline(command, compute, pushed_pipeline);
            let count = writes.len() as u32;
            push(command, compute, pushed_layout, 0, count, writes.as_ptr());
            device.cmd_dispatch(command, 1, 1, 1);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[],
            );
        }
    })?;
    let bound = kit
        .read(&buffers[0].1)?
        .first()
        .copied()
        .unwrap_or(UNWRITTEN);
    let pushed = kit
        .read(&buffers[1].1)?
        .first()
        .copied()
        .unwrap_or(UNWRITTEN);
    Ok(Some([bound, pushed]))
}
