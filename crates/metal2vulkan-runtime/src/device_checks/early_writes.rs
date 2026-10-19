//! The check of `DriverWorkaround::EarlyTestsDropMultisampledImageWrites`: at 4 samples, and at 2
//! where the device renders 2, one triangle over a [`SIDE`]-square pass through a fragment stage
//! that declares early fragment tests and stores 1 at its pixel's word of a buffer and at its
//! pixel's texel of an `R32ui` storage image, both 0 before the pass
//! (`metal2vulkan::bind::early_tests_check_fragment`).
//!
//! Every word and texel 1 is no defect: Metal's answer (a function declaring early tests whose
//! image writes land). Every word 1 and every texel 0 is the defect: the stage ran and its image
//! writes were dropped. Anything else answers neither.

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, Graphics, Triangle, COLOR, COLOR_USAGE};
use super::Target;
use ash::vk;
use metal2vulkan::bind::RasterCheck;
use metal2vulkan::device::{CheckAnswer, VkFeature};

/// The pass's side, in pixels.
const SIDE: u32 = 8;

/// The storage image's format.
const IMAGE: vk::Format = vk::Format::R32_UINT;

/// What one pass's stores left.
#[derive(Debug, PartialEq, Eq)]
enum Left {
    Both,
    BufferAlone,
    Neither(String),
}

/// The check's answer on `target`: not run where the device lacks a fragment stage's storage
/// writes.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    if !target
        .enabled
        .features
        .contains(&VkFeature::FragmentStoresAndAtomics)
    {
        return vec![CheckAnswer::NotRun(
            "fragmentStoresAndAtomics is not enabled".to_owned(),
        )];
    }
    let mut passes = Vec::new();
    for samples in [4, 2] {
        match raster::takes_samples(kit, COLOR, COLOR_USAGE, samples) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(error) => return vec![CheckAnswer::Unanswered(error.0)],
        }
        match stores(kit, target, samples) {
            Ok((words, texels)) => passes.push((samples, left(&words, &texels))),
            Err(error) => return vec![CheckAnswer::Unanswered(error.0)],
        }
    }
    vec![answer(&passes)]
}

/// What a pass's words and texels show (the module docs).
fn left(words: &[u32], texels: &[u32]) -> Left {
    let all = |values: &[u32], v: u32| values.iter().all(|&w| w == v);
    match () {
        () if all(words, 1) && all(texels, 1) => Left::Both,
        () if all(words, 1) && all(texels, 0) => Left::BufferAlone,
        () => Left::Neither(format!(
            "the stores left words {words:?} and texels {texels:?}"
        )),
    }
}

/// The answer of every pass (the module docs): Metal's answer is no defect.
fn answer(passes: &[(u32, Left)]) -> CheckAnswer {
    if passes.is_empty() {
        return CheckAnswer::Unanswered("no pass of 2 or 4 samples ran".to_owned());
    }
    if let Some((samples, Left::Neither(why))) =
        passes.iter().find(|(_, l)| matches!(l, Left::Neither(_)))
    {
        return CheckAnswer::Unanswered(format!("at {samples} samples {why}"));
    }
    raster::answer(match passes.iter().find(|(_, l)| *l == Left::BufferAlone) {
        None => Ok(()),
        Some((samples, _)) => Err(format!(
            "at {samples} samples the stage's image writes were dropped, its buffer writes kept"
        )),
    })
}

/// The buffer's words and the image's texels a pass of `samples` samples leaves.
fn stores(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    samples: u32,
) -> Result<(Vec<u32>, Vec<u32>), CheckError> {
    let pixels = (SIDE * SIDE) as usize;
    let vertex = raster::vertex_shader(kit, target)?;
    let fragment = raster::shader_of(
        kit,
        metal2vulkan::bind::early_tests_check_fragment(target.compile, SIDE),
        "fragment stage",
    )?;
    let whole = (0.0, 0.0, SIDE as f32, SIDE as f32);
    let vertices = raster::vertex_buffer(kit, &[Triangle::covering([0.5; 3], whole)])?;
    let words = kit.host_buffer(&vec![0; pixels], vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let texels = kit.host_buffer(&vec![u32::MAX; pixels], vk::BufferUsageFlags::TRANSFER_DST)?;
    let usage = vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let (image, view) = raster::image(
        kit,
        IMAGE,
        usage,
        (SIDE, SIDE, 1),
        vk::ImageAspectFlags::COLOR,
    )?;
    let pass = raster::pass(kit, (SIDE, SIDE, samples), true, None)?;
    let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let (set_layout, set) = kit.descriptor_set(&[
        SetBinding {
            binding: RasterCheck::VERTICES_BINDING,
            stages,
            resource: SetResource::Buffer(vertices),
        },
        SetBinding {
            binding: RasterCheck::RESULTS_BINDING,
            stages,
            resource: SetResource::Buffer(words),
        },
        SetBinding {
            binding: RasterCheck::IMAGE_BINDING,
            stages,
            resource: SetResource::StorageImage(view),
        },
    ])?;
    let layout = kit.layout(&[set_layout], None)?;
    let pipeline = raster::graphics(
        kit,
        &pass,
        Graphics {
            vertex,
            fragment: Some(fragment),
            layout,
            depth: false,
            bias: false,
        },
    )?;
    let range = raster::subresources(vk::ImageAspectFlags::COLOR);
    kit.submit(|device, command| {
        let to_general = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
        let cleared = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(range);
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
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width: SIDE,
                height: SIDE,
                depth: 1,
            },
        };
        // SAFETY: recording; every object named is live; the clear and the copy lie outside the
        // pass, the barrier ordering the draw's image writes after the clear, and the pass's
        // dependency the copy after them.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_general],
            );
            device.cmd_clear_color_image(
                command,
                image,
                vk::ImageLayout::GENERAL,
                &vk::ClearColorValue::default(),
                &[range],
            );
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[cleared],
            );
            device.cmd_begin_render_pass(command, &pass.begin(), vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
            raster::draw(device, command, 0, (0, 0, SIDE, SIDE));
            device.cmd_end_render_pass(command);
            device.cmd_copy_image_to_buffer(
                command,
                image,
                vk::ImageLayout::GENERAL,
                texels.buffer,
                &[copy],
            );
            raster::host_reads(
                device,
                command,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::TRANSFER,
            );
        }
    })?;
    Ok((kit.read(&words)?, kit.read(&texels)?))
}
