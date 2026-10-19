//! The check of `RuntimeDriverDefect::MultisampledDepthStencilColourCopies`, settled as the device
//! is made: it decides whether the runtime copies a multisampled depth or stencil aspect through
//! maintenance8's depth-stencil/colour copies, by draws, or refuses the copy
//! (`depth_stencil_scratch.rs`, `sample_copy.rs`, `nan_clear.rs`, `shadow_stencil.rs`), each read
//! as the device is assembled.
//!
//! One submission of the copies the runtime makes of a separate depth through its combined scratch
//! image (`depth_stencil_scratch::colour_twin`): a 4-sample `D32_SFLOAT_S8_UINT` image [`SIDE`]
//! pixels square, cleared to 1 and drawn whole by a fragment stage that writes a different normal
//! depth at each sample ([`written`], `metal2vulkan::bind::depth_writes_check_fragment`); its depth
//! aspect copied (`vkCmdCopyImage`) into a 4-sample `R32_SFLOAT` colour image cleared to
//! [`COLOUR_CLEAR`], and that copied back into the depth aspect of a second `D32_SFLOAT_S8_UINT`
//! image cleared to [`SECOND_CLEAR`]. Every sample of the three is then fetched
//! (`depth_samples_check_kernel`). Metal's answer, as Metal's blits copy a multisampled texture's
//! every sample: the colour image's samples are the first depth's, and the second depth's the
//! colour image's, bit for bit. The defect's answer otherwise (lavapipe copies sample 0 alone
//! between formats of different texel sizes, the other samples keeping their clears). A first
//! depth whose samples are not all different, or one still cleared, answers neither.
//!
//! The stencil aspect's copies are not run: NVIDIA's lose the device on some of its GPUs
//! (`RuntimeDriverDefect::MultisampledStencilColourCopies`, on the known-driver list). The check
//! does not run where the known-driver list names the defect for the device (MoltenVK's), where
//! the device enabled no maintenance8 or no `sampleRateShading` (which the draw needs), or takes no
//! such images.

use super::depth_draws::{self, DepthState, Stages};
use super::fragment_depth::fetch_set;
use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, Triangle, UNWRITTEN};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{depth_samples_check_kernel, depth_writes_check_fragment, RasterCheck};
use metal2vulkan::device::{CheckAnswer, VkFeature};
use metal2vulkan::error::RuntimeDriverDefect;
use std::collections::BTreeSet;

/// The images' side, in pixels: one workgroup of the stored depths kernel.
const SIDE: u32 = RasterCheck::DEPTH_SAMPLES_LOCAL_SIZE;

/// The samples of every image.
const SAMPLES: u32 = RasterCheck::SAMPLE_SLOTS;

/// The samples of each image.
const SLOTS: u32 = SIDE * SIDE * SAMPLES;

/// The combined depth/stencil format, whose depth aspect is copied.
const DEPTH: vk::Format = vk::Format::D32_SFLOAT_S8_UINT;

/// The colour format the depth aspect is copied through.
const COLOUR: vk::Format = vk::Format::R32_SFLOAT;

/// The depth the first image clears to before its draw.
const FIRST_CLEAR: f32 = 1.0;

/// The colour image's clear before the copy into it.
const COLOUR_CLEAR: f32 = 2.0;

/// The second depth image's clear before the copy into it.
const SECOND_CLEAR: f32 = 0.75;

/// The depth the draw writes at slot `slot` (`4 (y · SIDE + x) + s`): a different normal value in
/// `[0.25, 0.3)` at each, its mantissa's low bits set.
fn written(slot: u32) -> u32 {
    0x3e80_0000 + slot * 0x1235
}

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let defect = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
    if let Some(row) = target.description.listed_driver_defects().get(&defect) {
        return CheckAnswer::NotRun(format!("the known-driver list names it: {row}"));
    }
    let features = &target.enabled.features;
    for (feature, name) in [
        (VkFeature::Maintenance8, "maintenance8"),
        (VkFeature::SampleRateShading, "sampleRateShading"),
    ] {
        if !features.contains(&feature) {
            return CheckAnswer::NotRun(format!("{name} is not enabled"));
        }
    }
    let [first, colour, second] = match copied(kit, target) {
        Ok(Some(fetched)) => fetched,
        Ok(None) => {
            return CheckAnswer::NotRun(
                "no 4-sample sampled D32_SFLOAT_S8_UINT depth or R32_SFLOAT colour image that \
                 copies both ways"
                    .to_owned(),
            )
        }
        Err(error) => return CheckAnswer::Unanswered(error.0),
    };
    if [&first, &colour, &second]
        .iter()
        .any(|words| words.contains(&UNWRITTEN))
    {
        return CheckAnswer::Unanswered("the fetch left a sample unwritten".to_owned());
    }
    let different = first.iter().collect::<BTreeSet<_>>().len();
    if different != first.len() || first.contains(&FIRST_CLEAR.to_bits()) {
        return CheckAnswer::Unanswered(format!(
            "the draw left {different} different values in the first depth's {} samples, or one \
             at its clear",
            first.len()
        ));
    }
    let mut otherwise = Vec::new();
    for (what, from, to) in [
        ("the depth into the colour image", &first, &colour),
        ("the colour image into the depth", &colour, &second),
    ] {
        let off: Vec<usize> = (0..from.len()).filter(|&i| from[i] != to[i]).collect();
        if let Some(&slot) = off.first() {
            let (pixel, sample) = (slot as u32 / SAMPLES, slot as u32 % SAMPLES);
            otherwise.push(format!(
                "the copy of {what} left {} of {SLOTS} samples otherwise, the first sample \
                 {sample} of pixel ({}, {}): {:#010x} for {:#010x}",
                off.len(),
                pixel % SIDE,
                pixel / SIDE,
                to[slot],
                from[slot]
            ));
        }
    }
    match otherwise.is_empty() {
        true => CheckAnswer::Metals,
        false => CheckAnswer::NotMetals(otherwise.join("; ")),
    }
}

/// An image of `format` and `usage` at [`SAMPLES`], [`SIDE`] square, kept by the kit, and its
/// view of `aspect`.
fn image(
    kit: &mut Kit<'_>,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> Result<(vk::Image, vk::ImageView), CheckError> {
    raster::image(kit, format, usage, (SIDE, SIDE, SAMPLES), aspect)
}

/// The depth aspect's sampled view of the combined `image`, kept by the kit.
fn depth_view(kit: &mut Kit<'_>, image: vk::Image) -> Result<vk::ImageView, CheckError> {
    kit.view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(DEPTH)
            .subresource_range(raster::subresources(vk::ImageAspectFlags::DEPTH)),
    )
}

/// The one-subpass pass of the combined depth `view` alone, its depth cleared to [`FIRST_CLEAR`]
/// and stored, left in `TRANSFER_SRC_OPTIMAL` behind a dependency of later transfer reads on its
/// writes, and its framebuffer.
fn first_pass(
    kit: &mut Kit<'_>,
    view: vk::ImageView,
) -> Result<(vk::RenderPass, vk::Framebuffer), CheckError> {
    let attachment = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    let attachments = [vk::AttachmentDescription2::default()
        .format(DEPTH)
        .samples(raster::sample_count(SAMPLES))
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
    let depth_ref = vk::AttachmentReference2::default()
        .attachment(0)
        .layout(attachment)
        .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL);
    let subpasses = [vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_ref)];
    let dependencies = [vk::SubpassDependency2::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
    let pass = kit.render_pass(
        &vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies),
    )?;
    let views = [view];
    let framebuffer = kit.framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(&views)
            .width(SIDE)
            .height(SIDE)
            .layers(1),
    )?;
    Ok((pass, framebuffer))
}

/// The copy of level 0 and layer 0 of the `from` aspect into the `to` aspect, [`SIDE`] square.
fn copy(from: vk::ImageAspectFlags, to: vk::ImageAspectFlags) -> vk::ImageCopy {
    let layers = |aspect_mask| vk::ImageSubresourceLayers {
        aspect_mask,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    vk::ImageCopy {
        src_subresource: layers(from),
        src_offset: vk::Offset3D::default(),
        dst_subresource: layers(to),
        dst_offset: vk::Offset3D::default(),
        extent: vk::Extent3D {
            width: SIDE,
            height: SIDE,
            depth: 1,
        },
    }
}

/// The fetched samples of the first depth, the colour image and the second depth, slot by slot
/// (the module docs); `None` where the device takes no such images.
fn copied(kit: &mut Kit<'_>, target: &Target<'_>) -> Result<Option<[Vec<u32>; 3]>, CheckError> {
    use vk::ImageUsageFlags as U;
    let depth_usage = U::DEPTH_STENCIL_ATTACHMENT | U::SAMPLED | U::TRANSFER_SRC | U::TRANSFER_DST;
    let colour_usage = U::COLOR_ATTACHMENT | U::SAMPLED | U::TRANSFER_SRC | U::TRANSFER_DST;
    if !depth_draws::takes_depth_of(kit, DEPTH, depth_usage, SAMPLES)?
        || !raster::takes_samples(kit, COLOUR, colour_usage, SAMPLES)?
    {
        return Ok(None);
    }
    let vertex = raster::vertex_shader(kit, target)?;
    let fragment = raster::shader_of(
        kit,
        depth_writes_check_fragment(target.compile, SIDE),
        "fragment stage",
    )?;
    let kernel = raster::shader_of(
        kit,
        depth_samples_check_kernel(target.compile, SIDE),
        "stored depths kernel",
    )?;
    let both = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
    let (first, attached) = image(kit, DEPTH, depth_usage, both)?;
    let first_view = depth_view(kit, first)?;
    let (colour, colour_view) = image(kit, COLOUR, colour_usage, vk::ImageAspectFlags::COLOR)?;
    let (second, _) = image(kit, DEPTH, depth_usage, both)?;
    let second_view = depth_view(kit, second)?;
    let (pass, framebuffer) = first_pass(kit, attached)?;
    let covering = Triangle::covering([0.5; 3], (0.0, 0.0, SIDE as f32, SIDE as f32));
    let vertices = raster::vertex_buffer(kit, &[covering])?;
    let storage = vk::BufferUsageFlags::STORAGE_BUFFER;
    let depths: Vec<u32> = (0..SLOTS).map(written).collect();
    let depths = kit.host_buffer(&depths, storage)?;
    let draw_stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let (draw_layout, draw_set) = kit.descriptor_set(&[
        SetBinding {
            binding: RasterCheck::VERTICES_BINDING,
            stages: draw_stages,
            resource: SetResource::Buffer(vertices),
        },
        SetBinding {
            binding: RasterCheck::PARAMS_BINDING,
            stages: draw_stages,
            resource: SetResource::Buffer(depths),
        },
    ])?;
    let unwritten = vec![UNWRITTEN; SLOTS as usize];
    let sampled = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    let mut fetch = |view| -> Result<_, CheckError> {
        let words = kit.host_buffer(&unwritten, storage)?;
        let (set_layout, set) = fetch_set(kit, SetResource::Buffer(words), view, sampled)?;
        Ok((words, set_layout, set))
    };
    let fetches = [fetch(first_view)?, fetch(colour_view)?, fetch(second_view)?];
    let draw_layout = kit.layout(&[draw_layout], None)?;
    let read_layout = kit.layout(&[fetches[0].1], None)?;
    let stages = Stages {
        vertex,
        fragment: Some(fragment),
        layout: draw_layout,
    };
    let state = DepthState {
        compare: vk::CompareOp::ALWAYS,
        write: true,
        clamp: false,
    };
    let draw = depth_draws::pipeline(kit, stages, pass, (SIDE, SIDE, SAMPLES), state)?;
    let read = kit.compute_pipeline(kernel, read_layout)?;
    let clears = [vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue {
            depth: FIRST_CLEAR,
            stencil: 0,
        },
    }];
    kit.submit(|device, command| {
        use vk::AccessFlags as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags as S;
        let barrier = |image, aspect, (old, new), (src, dst)| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(raster::subresources(aspect))
        };
        let colour_aspect = vk::ImageAspectFlags::COLOR;
        let fresh = [(colour, colour_aspect), (second, both)].map(|(image, aspect)| {
            barrier(
                image,
                aspect,
                (L::UNDEFINED, L::TRANSFER_DST_OPTIMAL),
                (A::empty(), A::TRANSFER_WRITE),
            )
        });
        let cleared = vk::MemoryBarrier::default()
            .src_access_mask(A::TRANSFER_WRITE)
            .dst_access_mask(A::TRANSFER_WRITE);
        let stored = vk::MemoryBarrier::default()
            .src_access_mask(A::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(A::TRANSFER_READ);
        let colour_copied = barrier(
            colour,
            colour_aspect,
            (L::TRANSFER_DST_OPTIMAL, L::TRANSFER_SRC_OPTIMAL),
            (A::TRANSFER_WRITE, A::TRANSFER_READ),
        );
        let to_read = [
            (first, both, L::TRANSFER_SRC_OPTIMAL, A::empty()),
            (colour, colour_aspect, L::TRANSFER_SRC_OPTIMAL, A::empty()),
            (second, both, L::TRANSFER_DST_OPTIMAL, A::TRANSFER_WRITE),
        ]
        .map(|(image, aspect, old, src)| {
            barrier(image, aspect, (old, sampled), (src, A::SHADER_READ))
        });
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(pass)
            .framebuffer(framebuffer)
            .render_area(raster::rect(0, 0, SIDE, SIDE))
            .clear_values(&clears);
        let colour_clear = vk::ClearColorValue {
            float32: [COLOUR_CLEAR; 4],
        };
        let second_clear = vk::ClearDepthStencilValue {
            depth: SECOND_CLEAR,
            stencil: 0,
        };
        let no_flags = vk::DependencyFlags::empty();
        let depth_aspect = vk::ImageAspectFlags::DEPTH;
        let into_colour = copy(depth_aspect, colour_aspect);
        let into_depth = copy(colour_aspect, depth_aspect);
        // SAFETY: recording; every object named is the kit's and live. Each image moves through
        // the layouts its barriers name, the pass leaves the first in `TRANSFER_SRC_OPTIMAL`
        // behind its dependency of transfer reads and the barrier after it, each copy reads and writes images in the
        // transfer layouts it names behind the barriers on their writes, and the fetches read
        // the three in `SHADER_READ_ONLY_OPTIMAL` behind the last barrier.
        unsafe {
            device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, draw);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                draw_layout,
                0,
                &[draw_set],
                &[],
            );
            device.cmd_set_scissor(command, 0, &[raster::rect(0, 0, SIDE, SIDE)]);
            device.cmd_draw(command, 3, 1, 0, 0);
            device.cmd_end_render_pass(command);
            // Beside the pass's dependency, which alone let MoltenVK 1.4.1 read a 4-sample pass's
            // depth before the pass stored it, where the submission wrote timestamps.
            device.cmd_pipeline_barrier(
                command,
                S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS,
                S::TRANSFER,
                no_flags,
                &[stored],
                &[],
                &[],
            );
            device.cmd_pipeline_barrier(
                command,
                S::TOP_OF_PIPE,
                S::TRANSFER,
                no_flags,
                &[],
                &[],
                &fresh,
            );
            let dst = L::TRANSFER_DST_OPTIMAL;
            let colour_range = [raster::subresources(colour_aspect)];
            device.cmd_clear_color_image(command, colour, dst, &colour_clear, &colour_range);
            let depth_range = [raster::subresources(both)];
            device.cmd_clear_depth_stencil_image(command, second, dst, &second_clear, &depth_range);
            device.cmd_pipeline_barrier(
                command,
                S::TRANSFER,
                S::TRANSFER,
                no_flags,
                &[cleared],
                &[],
                &[],
            );
            let src = L::TRANSFER_SRC_OPTIMAL;
            device.cmd_copy_image(command, first, src, colour, dst, &[into_colour]);
            device.cmd_pipeline_barrier(
                command,
                S::TRANSFER,
                S::TRANSFER,
                no_flags,
                &[],
                &[],
                &[colour_copied],
            );
            device.cmd_copy_image(command, colour, src, second, dst, &[into_depth]);
            device.cmd_pipeline_barrier(
                command,
                S::TRANSFER,
                S::COMPUTE_SHADER,
                no_flags,
                &[],
                &[],
                &to_read,
            );
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, read);
            for &(_, _, set) in &fetches {
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    read_layout,
                    0,
                    &[set],
                    &[],
                );
                device.cmd_dispatch(command, 1, 1, SAMPLES);
            }
            raster::host_reads(device, command, S::COMPUTE_SHADER);
        }
    })?;
    let [(first, ..), (colour, ..), (second, ..)] = fetches;
    Ok(Some([
        kit.read(&first)?,
        kit.read(&colour)?,
        kit.read(&second)?,
    ]))
}
