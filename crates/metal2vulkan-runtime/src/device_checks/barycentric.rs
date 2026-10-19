//! The check of `DriverWorkaround::BothBarycentricFormsRejected`: two graphics pipelines are made
//! and never drawn, one whose fragment stage reads `BaryCoordKHR` and `BaryCoordNoPerspKHR`
//! (`metal2vulkan::bind::barycentric_pair_check_fragment`) and its control, which reads
//! `BaryCoordKHR` alone.
//!
//! Both made is no defect: Metal's answer (Metal compiles a fragment function declaring both
//! forms). The pair refused by the driver while the control is made is the defect. A refusal of
//! the control, or one for want of memory, answers neither.
//!
//! A check provokes no logged error, and a refused pipeline is one wherever the driver logs its
//! failures. MoltenVK converts every shader through SPIRV-Cross, whose MSL backend refuses both
//! forms in one shader, and logs the refusal as an error: there the check does not run, and
//! MoltenVK's identity holds the defect at every release (the translator's tables). No other
//! driver is known to refuse the pair; one found to would be listed the same way.

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, Graphics, Triangle};
use super::Target;
use ash::vk;
use metal2vulkan::bind::RasterCheck;
use metal2vulkan::device::{CheckAnswer, VkFeature};
use metal2vulkan::vk::DriverId;

/// The check's answer on `target`: not run where the device lacks `fragmentShaderBarycentric`, nor
/// on MoltenVK (the module docs).
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    if !(target.enabled.features).contains(&VkFeature::FragmentShaderBarycentric) {
        return vec![CheckAnswer::NotRun(
            "fragmentShaderBarycentric is not enabled".to_owned(),
        )];
    }
    if target.description.driver_id == Some(DriverId::MOLTENVK) {
        return vec![CheckAnswer::NotRun(
            "MoltenVK's converter refuses both forms in one shader, logging an error; its identity \
             holds the defect"
                .to_owned(),
        )];
    }
    let made = |kit: &mut Kit<'_>, both| made(kit, target, both);
    let answer = match (made(kit, false), made(kit, true)) {
        (Err(error), _) | (_, Err(error)) => CheckAnswer::Unanswered(error.0),
        (Ok(Err(control)), _) => {
            CheckAnswer::Unanswered(format!("the driver refused the control: {control:?}"))
        }
        (Ok(Ok(())), Ok(Ok(()))) => raster::answer(Ok(())),
        (Ok(Ok(())), Ok(Err(memory)))
            if matches!(
                memory,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
            ) =>
        {
            CheckAnswer::Unanswered(format!("the pair's pipeline failed: {memory:?}"))
        }
        (Ok(Ok(())), Ok(Err(refused))) => raster::answer(Err(format!(
            "the driver refused the pipeline reading both forms: {refused:?}"
        ))),
    };
    vec![answer]
}

/// Whether the driver makes the pipeline of the pair (`both`) or of its control: `Ok(Err)` its
/// refusal.
fn made(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    both: bool,
) -> Result<Result<(), vk::Result>, CheckError> {
    let vertex = raster::vertex_shader(kit, target)?;
    let fragment = raster::shader_of(
        kit,
        metal2vulkan::bind::barycentric_pair_check_fragment(target.compile, both),
        "fragment stage",
    )?;
    let pass = raster::pass(kit, (1, 1, 1), true, None)?;
    // The vertex stage's records, which the layout must bind though nothing draws.
    let vertices =
        raster::vertex_buffer(kit, &[Triangle::covering([0.5; 3], (0.0, 0.0, 1.0, 1.0))])?;
    let (set_layout, _) = kit.descriptor_set(&[SetBinding {
        binding: RasterCheck::VERTICES_BINDING,
        stages: vk::ShaderStageFlags::VERTEX,
        resource: SetResource::Buffer(vertices),
    }])?;
    let layout = kit.layout(&[set_layout], None)?;
    let made = raster::graphics(
        kit,
        &pass,
        Graphics {
            vertex,
            fragment: Some(fragment),
            layout,
            depth: false,
            bias: false,
        },
    );
    Ok(made.map(|_| ()))
}
