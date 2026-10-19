//! The check of `RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`: for each 16-bit
//! pattern of [`PATTERNS`], a 4-sample `R16_UINT` image cleared to it (`vkCmdClearColorImage`,
//! exact for an integer format) is copied (`vkCmdCopyImage`) into a 4-sample `R16_SFLOAT` color
//! attachment of no other view format, the copy the runtime makes of a NaN clear where the
//! attachment takes no unsigned view (`nan_clear.rs`), that copied on into a second 4-sample
//! `R16_UINT` image, and every sample of the last fetched
//! (`metal2vulkan::bind::texel_fetch_check_module`), all in one submission. Metal's copies keep
//! every bit: each sample holds its pattern. The defect's answer: a NaN pattern's samples changed
//! (RADV's copy into the float image exports each NaN as `0xfe00`) while the other patterns' were
//! kept. Another pattern changed answered neither. It does not run where the device takes no such
//! images.

use super::fetched::color_range;
use super::kit::{CheckError, HostBuffer, Kit, SetBinding, SetResource};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{texel_fetch_check_module, TexelFetch, TexelFetchCheck};
use metal2vulkan::device::CheckAnswer;

/// The patterns copied: first [`NUMBERS`] that are no NaN (1.0, -0.0, the least denormal and
/// infinity), whose copy a device that converts NaNs keeps, then NaNs of both signs, quiet and
/// signalling, with and without payloads, the converted NaN `0xfe00` among them.
const PATTERNS: [u32; 12] = [
    0x3C00, 0x8000, 0x0001, 0x7C00, 0x7E00, 0xFE00, 0x7E01, 0xFC01, 0x7C01, 0x7D55, 0x7FFF, 0xFFFF,
];

/// How many of [`PATTERNS`] lead that are no NaN.
const NUMBERS: usize = 4;

/// The samples of the images.
const SAMPLES: u32 = 4;

/// What the fetch's buffer holds before it: no fetched sample's red reads it.
const UNWRITTEN: u32 = u32::MAX;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    match unsupported(kit) {
        Ok(Some(why)) => return CheckAnswer::NotRun(why),
        Ok(None) => {}
        Err(error) => return CheckAnswer::Unanswered(error.0),
    }
    let fetch = TexelFetch::new(true, SAMPLES);
    let module = match texel_fetch_check_module(target.compile, fetch) {
        Ok(module) => module,
        Err(error) => return CheckAnswer::Unanswered(format!("its fetch does not build: {error}")),
    };
    match copied(kit, &module) {
        Ok(read) => answer(&read),
        Err(error) => CheckAnswer::Unanswered(error.0),
    }
}

/// The usage of each image: a color attachment's, as the runtime's multisampled targets are,
/// with copies and sampling.
fn usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
}

/// Why the device cannot run the check, where it cannot: no 4-sample `R16_UINT` or `R16_SFLOAT`
/// image of [`usage`].
fn unsupported(kit: &mut Kit<'_>) -> Result<Option<String>, CheckError> {
    let formats = [
        (vk::Format::R16_UINT, "R16_UINT"),
        (vk::Format::R16_SFLOAT, "R16_SFLOAT"),
    ];
    for (format, name) in formats {
        let properties = kit.physical_query(|instance, physical| {
            // SAFETY: `physical` is the instance's.
            unsafe {
                instance.get_physical_device_image_format_properties(
                    physical,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    usage(),
                    vk::ImageCreateFlags::empty(),
                )
            }
        })?;
        let samples = vk::SampleCountFlags::from_raw(SAMPLES);
        if !properties.is_ok_and(|p| p.sample_counts.contains(samples)) {
            return Ok(Some(format!("no {SAMPLES}-sample {name} image")));
        }
    }
    Ok(None)
}

/// Each pattern's fetched words.
fn copied(kit: &mut Kit<'_>, module: &TexelFetchCheck) -> Result<Vec<Vec<u32>>, CheckError> {
    let side = TexelFetchCheck::SIDE;
    let image = |format| {
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: side,
                height: side,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::from_raw(SAMPLES))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage())
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    };
    let shader = kit.shader(&module.spirv)?;
    let mut copies: Vec<([vk::Image; 3], HostBuffer, vk::DescriptorSet)> = Vec::new();
    let mut set_layout = None;
    for _ in PATTERNS {
        let scratch = kit.image(&image(vk::Format::R16_UINT))?;
        let target = kit.image(&image(vk::Format::R16_SFLOAT))?;
        let back = kit.image(&image(vk::Format::R16_UINT))?;
        let view = kit.view(
            &vk::ImageViewCreateInfo::default()
                .image(back)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R16_UINT)
                .subresource_range(color_range()),
        )?;
        let unwritten = vec![UNWRITTEN; module.words() as usize];
        let texels = kit.host_buffer(&unwritten, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        let stages = vk::ShaderStageFlags::COMPUTE;
        let (layout, set) = kit.descriptor_set(&[
            SetBinding {
                binding: TexelFetchCheck::IMAGE_BINDING,
                stages,
                resource: SetResource::SampledImage(
                    view,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ),
            },
            SetBinding {
                binding: TexelFetchCheck::TEXELS_BINDING,
                stages,
                resource: SetResource::Buffer(texels),
            },
        ])?;
        set_layout.get_or_insert(layout);
        copies.push(([scratch, target, back], texels, set));
    }
    let Some(set_layout) = set_layout else {
        return Ok(Vec::new());
    };
    // Every set's layout is defined alike, so each is compatible with the first's.
    let layout = kit.layout(&[set_layout], None)?;
    let pipeline = kit.compute_pipeline(shader, layout)?;
    let workgroups = module.workgroups();
    kit.submit(|device, command| {
        for (&pattern, &(images, _, set)) in PATTERNS.iter().zip(&copies) {
            // SAFETY: recording, outside a render pass; the images, set, layout and pipeline are
            // the kit's, live.
            unsafe {
                record(
                    device,
                    command,
                    pattern,
                    images,
                    set,
                    (layout, pipeline),
                    workgroups,
                )
            };
        }
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: recording.
        unsafe {
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
    (copies.iter())
        .map(|(_, texels, _)| kit.read(texels))
        .collect()
}

/// Record the clear of `scratch` to `pattern`, its copy into `target`, that one's into `back`, and
/// the fetch of `back` through `set`.
///
/// # Safety
///
/// `command` is recording outside a render pass; every object named is live, `pipeline` a
/// compute pipeline of `layout` whose set 0 is compatible with `set`'s.
unsafe fn record(
    device: &ash::Device,
    command: vk::CommandBuffer,
    pattern: u32,
    [scratch, target, back]: [vk::Image; 3],
    set: vk::DescriptorSet,
    (layout, pipeline): (vk::PipelineLayout, vk::Pipeline),
    workgroups: u32,
) {
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
    use vk::ImageLayout as L;
    let fresh = [
        barrier(scratch, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, none, write),
        barrier(target, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, none, write),
        barrier(back, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, none, write),
    ];
    let cleared = barrier(
        scratch,
        L::TRANSFER_DST_OPTIMAL,
        L::TRANSFER_SRC_OPTIMAL,
        write,
        read,
    );
    let landed = barrier(
        target,
        L::TRANSFER_DST_OPTIMAL,
        L::TRANSFER_SRC_OPTIMAL,
        write,
        read,
    );
    let copied = barrier(
        back,
        L::TRANSFER_DST_OPTIMAL,
        L::SHADER_READ_ONLY_OPTIMAL,
        write,
        vk::AccessFlags::SHADER_READ,
    );
    let clear = vk::ClearColorValue {
        uint32: [pattern, 0, 0, 0],
    };
    let layers = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let side = TexelFetchCheck::SIDE;
    let copy = vk::ImageCopy {
        src_subresource: layers,
        src_offset: vk::Offset3D::default(),
        dst_subresource: layers,
        dst_offset: vk::Offset3D::default(),
        extent: vk::Extent3D {
            width: side,
            height: side,
            depth: 1,
        },
    };
    let transfer = vk::PipelineStageFlags::TRANSFER;
    // SAFETY: the caller's.
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            transfer,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &fresh,
        );
        device.cmd_clear_color_image(
            command,
            scratch,
            L::TRANSFER_DST_OPTIMAL,
            &clear,
            &[color_range()],
        );
        device.cmd_pipeline_barrier(
            command,
            transfer,
            transfer,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[cleared],
        );
        device.cmd_copy_image(
            command,
            scratch,
            L::TRANSFER_SRC_OPTIMAL,
            target,
            L::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
        device.cmd_pipeline_barrier(
            command,
            transfer,
            transfer,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[landed],
        );
        device.cmd_copy_image(
            command,
            target,
            L::TRANSFER_SRC_OPTIMAL,
            back,
            L::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
        device.cmd_pipeline_barrier(
            command,
            transfer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[copied],
        );
        device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[set],
            &[],
        );
        device.cmd_dispatch(command, workgroups, 1, 1);
    }
}

/// The answer of each pattern's fetched words `read` (the module docs): each sample's red
/// component, the first of its four words.
fn answer(read: &[Vec<u32>]) -> CheckAnswer {
    let mut changed = Vec::new();
    for (&pattern, words) in PATTERNS.iter().zip(read) {
        let reds: Vec<u32> = words.chunks(4).map(|w| w[0]).collect();
        if let Some(at) = reds.iter().position(|&r| r == UNWRITTEN) {
            return CheckAnswer::Unanswered(format!("the fetch left sample {at} unwritten"));
        }
        let differing: Vec<u32> = reds.iter().copied().filter(|&r| r != pattern).collect();
        let Some(&first) = differing.first() else {
            continue;
        };
        let what = format!(
            "{} of {} samples of {pattern:#06x} read otherwise, the first {first:#06x}",
            differing.len(),
            reds.len()
        );
        if PATTERNS[..NUMBERS].contains(&pattern) {
            return CheckAnswer::Unanswered(format!("a pattern of no NaN changed: {what}"));
        }
        changed.push(what);
    }
    match changed.is_empty() {
        true => CheckAnswer::Metals,
        false => CheckAnswer::NotMetals(changed.join("; ")),
    }
}
