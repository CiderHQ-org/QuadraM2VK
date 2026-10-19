//! The sweeps of three deviations measured on every input of a 32-bit domain (`DriverDeviation`'s
//! `MetalHalfConversion`, `MetalMinMax` and `MetalHalfArithmetic`): each a module of the device's
//! form and the exact form (`metal2vulkan::bind::Sweep`, `SweepKernel`) that counts on the GPU the
//! inputs where the two differ, in each math mode the device has. The structured sample runs
//! first, as an early out; then the whole domain, one chunk of `SweepKernel::CHUNK` inputs a
//! submission, so the queue never waits behind more than one chunk: 1 + `SweepKernel::CHUNKS`
//! dispatches a sweep and mode. Only a whole sweep with no input differing answers Metal's; the
//! first dispatch that finds one ends it, and its answer names the inputs found so far
//! (`SweepKernel::difference`).

use super::color::combined;
use super::kit::{CheckError, HostBuffer, Kit, SetBinding, SetResource};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{MathMode, Sweep, SweepKernel};
use metal2vulkan::device::CheckAnswer;

/// The three sweeps' answers on `target`, in [`Sweep::ALL`]'s order.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    Sweep::ALL
        .iter()
        .map(|&sweep| match sweep.unsupported(target.compile) {
            Some(why) => CheckAnswer::NotRun(why.to_owned()),
            None => combined(
                MathMode::of(target.compile)
                    .into_iter()
                    .map(|mode| swept(kit, target, sweep, mode))
                    .collect(),
            ),
        })
        .collect()
}

/// The answer of `sweep` in math mode `mode`.
fn swept(kit: &mut Kit<'_>, target: &Target<'_>, sweep: Sweep, mode: MathMode) -> CheckAnswer {
    let kernel = match sweep.kernel(target.compile, mode) {
        Ok(kernel) => kernel,
        Err(error) => {
            return CheckAnswer::Unanswered(format!("its module does not build: {error}"));
        }
    };
    match differing(kit, &kernel) {
        Ok(None) => CheckAnswer::Metals,
        Ok(Some((how, dispatches))) => CheckAnswer::NotMetals(format!(
            "{how}, in the {mode:?} math mode, after {dispatches} of {} dispatches",
            SweepKernel::CHUNKS + 1
        )),
        Err(error) => CheckAnswer::Unanswered(error.0),
    }
}

/// The objects of one sweep's dispatches.
struct Sweeping {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set: vk::DescriptorSet,
    result: HostBuffer,
}

/// How `kernel`'s forms differ (`SweepKernel::difference`) and after how many dispatches, or
/// `None` where they differ on no input of the domain.
fn differing(kit: &mut Kit<'_>, kernel: &SweepKernel) -> Result<Option<(String, u32)>, CheckError> {
    let listed: &[u32] = match kernel.sample.is_empty() {
        true => &[0],
        false => &kernel.sample,
    };
    let list = kit.host_buffer(listed, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let result = kit.host_buffer(
        &SweepKernel::RESULT_START,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let (set_layout, set) = kit.descriptor_set(&[
        SetBinding {
            binding: SweepKernel::LIST_BINDING,
            stages: vk::ShaderStageFlags::COMPUTE,
            resource: SetResource::Buffer(list),
        },
        SetBinding {
            binding: SweepKernel::RESULT_BINDING,
            stages: vk::ShaderStageFlags::COMPUTE,
            resource: SetResource::Buffer(result),
        },
    ])?;
    let push = (vk::ShaderStageFlags::COMPUTE, SweepKernel::PUSH_BYTES);
    let layout = kit.layout(&[set_layout], Some(push))?;
    let shader = kit.shader(&kernel.spirv)?;
    let pipeline = kit.compute_pipeline(shader, layout)?;
    let sweeping = Sweeping {
        pipeline,
        layout,
        set,
        result,
    };
    let mut dispatches = 0;
    if kernel.sample_workgroups() > 0 {
        dispatches += 1;
        let found = dispatch(
            kit,
            &sweeping,
            kernel.sample_push(),
            kernel.sample_workgroups(),
        )?;
        if let Some(how) = found {
            return Ok(Some((how, dispatches)));
        }
    }
    for chunk in 0..SweepKernel::CHUNKS {
        dispatches += 1;
        let push = SweepKernel::chunk_push(chunk);
        let found = dispatch(kit, &sweeping, push, SweepKernel::chunk_workgroups())?;
        if let Some(how) = found {
            return Ok(Some((how, dispatches)));
        }
    }
    Ok(None)
}

/// One submission of one dispatch of `workgroups` with push bytes `push`, and how the result
/// differs from Metal's answer after it: `Some` once an input differed.
fn dispatch(
    kit: &mut Kit<'_>,
    sweeping: &Sweeping,
    push: [u8; 8],
    workgroups: u32,
) -> Result<Option<String>, CheckError> {
    let Sweeping {
        pipeline,
        layout,
        set,
        result,
    } = *sweeping;
    kit.submit(|device, command| {
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: recording, outside a render pass; every object named is live, and the push
        // bytes are the layout's range.
        unsafe {
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(command, layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
            device.cmd_dispatch(command, workgroups, 1, 1);
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
    let words = kit.read(&result)?;
    Ok(SweepKernel::difference(&words))
}
