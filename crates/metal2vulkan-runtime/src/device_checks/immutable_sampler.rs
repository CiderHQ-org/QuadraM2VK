//! The check of `RuntimeDriverDefect::ImmutableSamplersLostAfterCopies`: one submission copies
//! texels into a staging image (`vkCmdCopyBufferToImage`), copies that into a sampled image
//! (`vkCmdCopyImage`, as a texture's shadow copy goes), and then dispatches
//! `metal2vulkan::bind::sampler_clamp_check_module` in the same command buffer over a layout
//! that bakes a clamp-to-edge sampler into its combined image sampler binding, set 0 pushed where
//! the device pushes it (as its dispatches do) and bound elsewhere. The kernel samples the 4 x 1
//! image of [`TEXELS`] at the coordinates of [`COORDINATES`], inside and outside `[0, 1]`.
//! Metal's answer is the clamped texels. The defect's answer: an outside sample reads another
//! texel (lavapipe 26.0.8 addresses them as Repeat). An inside sample read wrong answered
//! neither.

use super::fetched::color_range;
use super::kit::{CheckError, Kit, SetBinding, SetResource, Writes};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{sampler_clamp_check_module, DescriptorCheck};
use metal2vulkan::device::CheckAnswer;

/// The image's texels, `R32_SFLOAT`.
const TEXELS: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

/// The coordinates sampled, and the texel each reads clamped to the edge: four inside `[0, 1]`,
/// four outside (each of which Repeat reads as another texel).
const COORDINATES: [(f32, f32); 8] = [
    (0.125, 1.0),
    (0.375, 2.0),
    (0.625, 3.0),
    (0.875, 4.0),
    (-0.375, 1.0),
    (-0.125, 1.0),
    (1.125, 4.0),
    (1.375, 4.0),
];

/// What the results hold before the dispatch.
const UNWRITTEN: u32 = u32::MAX;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let module = match sampler_clamp_check_module(target.compile) {
        Ok(module) => module,
        Err(error) => {
            return CheckAnswer::Unanswered(format!("its kernel does not build: {error}"))
        }
    };
    let read = match sampled(kit, target, &module) {
        Ok(read) => read,
        Err(error) => return CheckAnswer::Unanswered(error.0),
    };
    let reads: Vec<f32> = read.iter().map(|&w| f32::from_bits(w)).collect();
    let (inside, outside) = COORDINATES.split_at(4);
    let wrong = |pairs: &[(f32, f32)], reads: &[f32]| -> Vec<String> {
        (pairs.iter().zip(reads))
            .filter(|((_, want), read)| read.to_bits() != want.to_bits())
            .map(|((u, want), read)| format!("{u} read {read}, clamped {want}"))
            .collect()
    };
    let inside_wrong = wrong(inside, &reads[..4]);
    if !inside_wrong.is_empty() {
        return CheckAnswer::Unanswered(format!("inside [0, 1]: {}", inside_wrong.join(", ")));
    }
    let outside_wrong = wrong(outside, &reads[4..]);
    match outside_wrong.is_empty() {
        true => CheckAnswer::Metals,
        false => CheckAnswer::NotMetals(format!(
            "after the copies, outside [0, 1]: {}",
            outside_wrong.join(", ")
        )),
    }
}

/// The kernel's samples, as `f32` bits.
fn sampled(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    module: &DescriptorCheck,
) -> Result<Vec<u32>, CheckError> {
    let push = kit.push_command(target.description.api_version)?;
    let words: Vec<u32> = TEXELS.iter().map(|t| t.to_bits()).collect();
    let texels = kit.host_buffer(&words, vk::BufferUsageFlags::TRANSFER_SRC)?;
    let coordinates: Vec<u32> = COORDINATES.iter().map(|(u, _)| u.to_bits()).collect();
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let coordinates = kit.host_buffer(&coordinates, storage)?;
    let results = kit.host_buffer(&[UNWRITTEN; COORDINATES.len()], storage)?;
    let extent = vk::Extent3D {
        width: TEXELS.len() as u32,
        height: 1,
        depth: 1,
    };
    let image = |usage| {
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    };
    let transfers = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    let staging = kit.image(&image(transfers))?;
    let sampled_usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
    let shadow = kit.image(&image(sampled_usage))?;
    let view = kit.view(
        &vk::ImageViewCreateInfo::default()
            .image(shadow)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R32_SFLOAT)
            .subresource_range(color_range()),
    )?;
    let edge = vk::SamplerAddressMode::CLAMP_TO_EDGE;
    let sampler = kit.sampler(
        &vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(edge)
            .address_mode_v(edge)
            .address_mode_w(edge)
            .min_lod(0.0)
            .max_lod(0.0),
    )?;
    let stages = vk::ShaderStageFlags::COMPUTE;
    let bindings = [
        SetBinding {
            binding: DescriptorCheck::IMAGE_BINDING,
            stages,
            resource: SetResource::Immutable(view, sampler),
        },
        SetBinding {
            binding: DescriptorCheck::INPUT_BINDING,
            stages,
            resource: SetResource::Buffer(coordinates),
        },
        SetBinding {
            binding: DescriptorCheck::OUTPUT_BINDING,
            stages,
            resource: SetResource::Buffer(results),
        },
    ];
    let set_layout = kit.set_layout(&bindings, push.is_some())?;
    let set = match push {
        Some(_) => None,
        None => Some(kit.set_of(set_layout, &bindings)?),
    };
    let layout = kit.layout(&[set_layout], None)?;
    let shader = kit.shader(&module.spirv)?;
    let pipeline = kit.compute_pipeline(shader, layout)?;
    let writes = Writes::of(&bindings);
    kit.submit(|device, command| {
        use vk::ImageLayout as L;
        let barrier = |image, old, new, src, dst| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())
        };
        let (none, write, read) = (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        let fresh = [
            barrier(staging, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, none, write),
            barrier(shadow, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, none, write),
        ];
        let staged = barrier(
            staging,
            L::TRANSFER_DST_OPTIMAL,
            L::TRANSFER_SRC_OPTIMAL,
            write,
            read,
        );
        let shader_read = vk::AccessFlags::SHADER_READ;
        let copied = barrier(
            shadow,
            L::TRANSFER_DST_OPTIMAL,
            L::SHADER_READ_ONLY_OPTIMAL,
            write,
            shader_read,
        );
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let upload = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: layers,
            image_offset: vk::Offset3D::default(),
            image_extent: extent,
        };
        let copy = vk::ImageCopy {
            src_subresource: layers,
            src_offset: vk::Offset3D::default(),
            dst_subresource: layers,
            dst_offset: vk::Offset3D::default(),
            extent,
        };
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        let writes = writes.writes(vk::DescriptorSet::null());
        let transfer = vk::PipelineStageFlags::TRANSFER;
        let compute = vk::PipelineBindPoint::COMPUTE;
        let no_flags = vk::DependencyFlags::empty();
        // SAFETY: recording, outside a render pass; every object named is live, each image in
        // the layout its barriers leave it in, the pipeline of `layout`, whose set 0's layout is
        // of `bindings`.
        unsafe {
            let top = vk::PipelineStageFlags::TOP_OF_PIPE;
            device.cmd_pipeline_barrier(command, top, transfer, no_flags, &[], &[], &fresh);
            let dst = L::TRANSFER_DST_OPTIMAL;
            device.cmd_copy_buffer_to_image(command, texels.buffer, staging, dst, &[upload]);
            device.cmd_pipeline_barrier(command, transfer, transfer, no_flags, &[], &[], &[staged]);
            let src = L::TRANSFER_SRC_OPTIMAL;
            device.cmd_copy_image(command, staging, src, shadow, dst, &[copy]);
            let stage = vk::PipelineStageFlags::COMPUTE_SHADER;
            device.cmd_pipeline_barrier(command, transfer, stage, no_flags, &[], &[], &[copied]);
            device.cmd_bind_pipeline(command, compute, pipeline);
            match (push, set) {
                (Some(push), _) => {
                    push(
                        command,
                        compute,
                        layout,
                        0,
                        writes.len() as u32,
                        writes.as_ptr(),
                    );
                }
                (None, Some(set)) => {
                    device.cmd_bind_descriptor_sets(command, compute, layout, 0, &[set], &[]);
                }
                (None, None) => {}
            }
            device.cmd_dispatch(command, 1, 1, 1);
            let host = vk::PipelineStageFlags::HOST;
            device.cmd_pipeline_barrier(command, stage, host, no_flags, &[written], &[], &[]);
        }
    })?;
    kit.read(&results)
}
