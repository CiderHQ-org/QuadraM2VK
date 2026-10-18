//! The stored depth read three ways: whether a draw's stored depth is what its
//! device's own depth test and a copy see, where the step's compute readback of it did not land.
//!
//! On the M3 Max (macOS 26.7, MoltenVK 1.4.1) three cases at one sample (`clamp`,
//! `bias_neg`, `bias_both_clamp`) read back the readback buffer's fill as the stored depth at
//! every covered sample, where the M4 read the clamped `FragCoord.z`. Each of [`CASES`] here (those
//! three and `flat`, which read back as the M4's) runs [`REPEATS`] times at one sample:
//!
//! - first the step's own measurement again ([`super::run_case`]): its plain form's report;
//! - then the plain draw (the depth attachment in the subpass under `ALWAYS` with writes, cleared
//!   to 1), and its stored depth read three ways, each into a buffer filled with [`MARK`] (no
//!   depth, no `FragCoord.z`, not all-ones), so a read that did not land shows as the fill:
//!   1. **fetched**: the step's compute readback through the depth aspect's sampled view;
//!   2. **copied**: the depth aspect copied to a buffer (`vkCmdCopyImageToBuffer`);
//!   3. **tested**: a draw of the same triangles under the same state with the depth test
//!      `EQUAL` and no writes, in a pass that loads the stored depth, over a color cleared to
//!      [`MISSED`]: the samples where the device's own depth test finds the stored depth equal to
//!      the draw's depth again, as an ordinary draw that tests against it does.
//!
//! The draws' fragment shader is the step's; every submission waits through the hang protocol.

use super::super::{Owned, VulkanContext};
use super::{
    pipeline, range, readback_set_layout, render_pass, run_case, unwritten, Case, Form, FormReport,
    Shaders, CASES as ALL_CASES, SIDE, UNWRITTEN,
};
use ash::vk;
use serde::Serialize;

/// The cases, by the step's names: one that read back on the M3 Max as on the M4, and the three
/// that did not.
const CASES: [&str; 4] = ["flat", "clamp", "bias_neg", "bias_both_clamp"];

/// How many times each case runs.
const REPEATS: u32 = 3;

/// The fill of a buffer a read lands in: no depth and no `FragCoord.z` a case gives, and not
/// all-ones.
const MARK: u32 = 0xDEAD_BEEF;

/// The color the tested draw's target is cleared to: no `FragCoord.z` a case gives.
const MISSED: f32 = -1000.0;

/// One run of one case.
#[derive(Debug, Serialize)]
pub(super) struct StoredRead {
    case: &'static str,
    repeat: u32,
    /// The step's own measurement of the case at one sample, run again: its plain form's report.
    original: Option<FormReport>,
    /// The samples the plain draw covered (its `FragCoord.z` written).
    covered: u32,
    /// The stored depth through the sampled view, as the step's readback reads it.
    fetched: ReadReport,
    /// The stored depth's depth aspect copied to a buffer.
    copied: ReadReport,
    /// Covered samples where the tested draw's `EQUAL` test passed.
    equal_passed: u32,
    /// Samples outside the plain draw's coverage where it passed.
    equal_passed_uncovered: u32,
    /// What stopped the run.
    error: Option<String>,
}

/// One read of the stored depth over the covered samples.
#[derive(Debug, Default, Serialize)]
struct ReadReport {
    /// Samples still holding [`MARK`]: the read did not land.
    unlanded: u32,
    /// Samples read as `0xFFFFFFFF`.
    all_ones: u32,
    /// Samples equal to the plain draw's `FragCoord.z` clamped as Metal clamps it.
    equal_clamped: u32,
    /// The first differing samples: index, `FragCoord.z`'s bits, the read bits.
    first_differences: Vec<[u32; 3]>,
}

/// Every case of [`CASES`] the device can draw, [`REPEATS`] times each.
pub(super) fn measure(
    context: &VulkanContext,
    shaders: &Shaders,
    format: vk::Format,
) -> Vec<StoredRead> {
    let mut out = Vec::new();
    for name in CASES {
        let Some(case) = ALL_CASES.iter().find(|c| c.name == name) else {
            continue;
        };
        for repeat in 0..REPEATS {
            let original = run_case(context, shaders, (format, 1, false), case)
                .ok()
                .and_then(|(forms, _)| forms.into_iter().find(|f| f.form == Form::Plain.name()));
            let mut report = StoredRead {
                case: case.name,
                repeat,
                original,
                covered: 0,
                fetched: ReadReport::default(),
                copied: ReadReport::default(),
                equal_passed: 0,
                equal_passed_uncovered: 0,
                error: None,
            };
            if let Err(error) = run(context, shaders, format, case, &mut report) {
                report.error = Some(error);
            }
            out.push(report);
        }
    }
    out
}

/// A host-visible buffer of `words` words with `usage`, each [`MARK`].
fn marked(
    owned: &mut Owned<'_>,
    words: usize,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, *mut u32), String> {
    let (buffer, mapped) = owned.host_buffer((words * 4) as u64, usage)?;
    let mapped = mapped.cast::<u32>();
    // SAFETY: the mapping holds `words` words, host-coherent, and nothing on the device uses the
    // buffer yet.
    unsafe { std::slice::from_raw_parts_mut(mapped, words) }.fill(MARK);
    Ok((buffer, mapped))
}

/// One run of `case` (see the module docs), its reads into `report`.
fn run(
    context: &VulkanContext,
    shaders: &Shaders,
    format: vk::Format,
    case: &Case,
    report: &mut StoredRead,
) -> Result<(), String> {
    let mut owned = Owned::new(context);
    let words = (SIDE * SIDE) as usize;
    let one = vk::SampleCountFlags::TYPE_1;
    let depth_aspects = match format {
        vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::DEPTH,
    };
    let image_info = |format: vk::Format, usage: vk::ImageUsageFlags| {
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: SIDE,
                height: SIDE,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(one)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
    };
    let bytes: Vec<u8> = case
        .vertices
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let (vertices, mapped) =
        owned.host_buffer(bytes.len() as u64, vk::BufferUsageFlags::VERTEX_BUFFER)?;
    // SAFETY: the mapping holds the buffer's bytes, and nothing on the device uses it yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped, bytes.len()) };
    // The plain draw's color and the depth both draws use; the tested draw's color.
    let color = owned.bound_image(&image_info(
        vk::Format::R32_SFLOAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT,
    ))?;
    let color_view = owned.view(color, vk::Format::R32_SFLOAT, vk::ImageAspectFlags::COLOR)?;
    let depth = owned.bound_image(&image_info(
        format,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC,
    ))?;
    let depth_view = owned.view(depth, format, depth_aspects)?;
    let stored_view = owned.view(depth, format, vk::ImageAspectFlags::DEPTH)?;
    let tested_color = owned.bound_image(&image_info(
        vk::Format::R32_SFLOAT,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    ))?;
    let tested_view = owned.view(
        tested_color,
        vk::Format::R32_SFLOAT,
        vk::ImageAspectFlags::COLOR,
    )?;
    let plain_pass = render_pass(&mut owned, format, one, Form::Plain)?;
    let plain_framebuffer = owned.framebuffer(plain_pass, &[color_view, depth_view], SIDE)?;
    let tested_pass = render_pass(&mut owned, format, one, Form::Tested)?;
    let tested_framebuffer = owned.framebuffer(tested_pass, &[tested_view, depth_view], SIDE)?;
    // The buffers.
    let (seen, seen_words) = unwritten(&mut owned, words)?;
    let (seen_again, _) = unwritten(&mut owned, words)?;
    let (fetched, fetched_words) = marked(&mut owned, words, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let (copied, copied_words) = marked(&mut owned, words, vk::BufferUsageFlags::TRANSFER_DST)?;
    let (tested, tested_words) = marked(&mut owned, words, vk::BufferUsageFlags::TRANSFER_DST)?;
    // The fragment stage's sets (one per draw) and the readback's.
    let fragment_binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let fragment_set_layout = owned.descriptor_set_layout(&fragment_binding)?;
    let fragment_layout = owned.plain_layout(fragment_set_layout)?;
    let fragment_sets = owned.sets(
        fragment_set_layout,
        &[vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(2)],
        2,
    )?;
    let (readback_layout, readback_set) = readback_set_layout(&mut owned)?;
    let seen_info = [vk::DescriptorBufferInfo::default()
        .buffer(seen)
        .range(vk::WHOLE_SIZE)];
    let seen_again_info = [vk::DescriptorBufferInfo::default()
        .buffer(seen_again)
        .range(vk::WHOLE_SIZE)];
    let fetched_info = [vk::DescriptorBufferInfo::default()
        .buffer(fetched)
        .range(vk::WHOLE_SIZE)];
    let image_info_read = [vk::DescriptorImageInfo::default()
        .image_view(stored_view)
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(fragment_sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&seen_info),
        vk::WriteDescriptorSet::default()
            .dst_set(fragment_sets[1])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&seen_again_info),
        vk::WriteDescriptorSet::default()
            .dst_set(readback_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&fetched_info),
        vk::WriteDescriptorSet::default()
            .dst_set(readback_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&image_info_read),
    ];
    // SAFETY: the sets, buffers and view are live objects of this device; nothing uses the sets.
    unsafe { owned.device().update_descriptor_sets(&writes, &[]) };
    // The pipelines.
    let vertex = owned.module(&shaders.vertex, "stored depth vertex")?;
    let fragment = owned.module(Shaders::of(&shaders.fragment, 1), "stored depth fragment")?;
    let readback = owned.compute_pipeline(
        Shaders::of(&shaders.readback, 1),
        "stored depth readback",
        readback_layout,
    )?;
    let plain = pipeline(
        &mut owned,
        (vertex, fragment),
        fragment_layout,
        plain_pass,
        one,
        (Form::Plain, false),
        case,
    )?;
    let equal = pipeline(
        &mut owned,
        (vertex, fragment),
        fragment_layout,
        tested_pass,
        one,
        (Form::Tested, false),
        case,
    )?;
    let area = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: SIDE,
            height: SIDE,
        },
    };
    let copy = |aspect: vk::ImageAspectFlags| {
        [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: SIDE,
                height: SIDE,
                depth: 1,
            })]
    };
    let depth_copy = copy(vk::ImageAspectFlags::DEPTH);
    let color_copy = copy(vk::ImageAspectFlags::COLOR);
    let draw_count = case.vertices.len() as u32;
    owned.submit("the stored depth's reads", |device, command| {
        let plain_clears = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let tested_clears = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [MISSED, 0.0, 0.0, 0.0],
            },
        }];
        let draw = |pass, framebuffer, clears: &[vk::ClearValue], bound, set| {
            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(pass)
                .framebuffer(framebuffer)
                .render_area(area)
                .clear_values(clears);
            // SAFETY: the command buffer is recording; every object it names is live.
            unsafe {
                device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, bound);
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    fragment_layout,
                    0,
                    &[set],
                    &[],
                );
                device.cmd_bind_vertex_buffers(command, 0, &[vertices], &[0]);
                device.cmd_draw(command, draw_count, 1, 0, 0);
                device.cmd_end_render_pass(command);
            }
        };
        let depth_barrier = |old, new, src_stage, src_access, dst_stage, dst_access| {
            let barrier = [vk::ImageMemoryBarrier::default()
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(depth)
                .subresource_range(range(depth_aspects))];
            // SAFETY: as above.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barrier,
                );
            }
        };
        // The plain draw; its pass's dependency makes its depth visible to the readback.
        draw(
            plain_pass,
            plain_framebuffer,
            &plain_clears,
            plain,
            fragment_sets[0],
        );
        // 1. fetched.
        // SAFETY: as above.
        unsafe {
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, readback);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                readback_layout,
                0,
                &[readback_set],
                &[],
            );
            device.cmd_dispatch(command, SIDE / 8, SIDE / 8, 1);
        }
        // 2. copied.
        depth_barrier(
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
        );
        // SAFETY: as above.
        unsafe {
            device.cmd_copy_image_to_buffer(
                command,
                depth,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                copied,
                &depth_copy,
            );
        }
        // 3. tested.
        depth_barrier(
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        draw(
            tested_pass,
            tested_framebuffer,
            &tested_clears,
            equal,
            fragment_sets[1],
        );
        // SAFETY: as above. The tested pass's dependency makes its color visible to the copy and
        // leaves it in `TRANSFER_SRC_OPTIMAL`.
        unsafe {
            device.cmd_copy_image_to_buffer(
                command,
                tested_color,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                tested,
                &color_copy,
            );
            let to_host = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &to_host,
                &[],
                &[],
            );
        }
    })?;
    let read = |mapped: *mut u32| -> Vec<u32> {
        // SAFETY: the mapping holds `words` host-coherent words, and the submission that wrote
        // them completed, its writes made available to the host by the barrier.
        unsafe { std::slice::from_raw_parts(mapped, words) }.to_vec()
    };
    let reads = Reads {
        seen: read(seen_words),
        fetched: read(fetched_words),
        copied: read(copied_words),
        tested: read(tested_words),
    };
    let bounds = match case.clamp {
        true => (
            case.range[0].min(case.range[1]),
            case.range[0].max(case.range[1]),
        ),
        false => (0.0, 1.0),
    };
    classify(&reads, bounds, report);
    Ok(())
}

/// One run's words, `SIDE²` each: the plain draw's `FragCoord.z`, the stored depth fetched and
/// copied, and the tested draw's color.
struct Reads {
    seen: Vec<u32>,
    fetched: Vec<u32>,
    copied: Vec<u32>,
    tested: Vec<u32>,
}

/// Count `reads` into `report`, `FragCoord.z` clamped to `(low, high)` as Metal clamps it.
fn classify(reads: &Reads, (low, high): (f32, f32), report: &mut StoredRead) {
    let missed = MISSED.to_bits();
    for (index, &z) in reads.seen.iter().enumerate() {
        let color = reads.tested[index];
        let passed = color != missed && color != MARK;
        if z == UNWRITTEN {
            report.equal_passed_uncovered += u32::from(passed);
            continue;
        }
        report.covered += 1;
        report.equal_passed += u32::from(passed);
        let clamped = f32::from_bits(z).clamp(low, high).to_bits();
        for (path, words) in [
            (&mut report.fetched, &reads.fetched),
            (&mut report.copied, &reads.copied),
        ] {
            let value = words[index];
            match value {
                MARK => path.unlanded += 1,
                UNWRITTEN => path.all_ones += 1,
                _ if value == clamped => path.equal_clamped += 1,
                _ => {}
            }
            if value != clamped && path.first_differences.len() < 4 {
                path.first_differences.push([index as u32, z, value]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read that did not land shows as the fill, one that read all-ones as such, and the
    /// tested draw's color counts where it is neither the clear nor the fill.
    #[test]
    fn a_run_s_reads_are_told_apart() {
        let bits = |v: f32| v.to_bits();
        let reads = Reads {
            seen: vec![bits(0.5), bits(1.5), bits(0.25), UNWRITTEN],
            fetched: vec![bits(0.5), MARK, UNWRITTEN, MARK],
            copied: vec![bits(0.5), bits(1.0), bits(0.75), MARK],
            tested: vec![bits(0.5), MISSED.to_bits(), bits(0.25), bits(0.3)],
        };
        let mut report = StoredRead {
            case: "synthetic",
            repeat: 0,
            original: None,
            covered: 0,
            fetched: ReadReport::default(),
            copied: ReadReport::default(),
            equal_passed: 0,
            equal_passed_uncovered: 0,
            error: None,
        };
        classify(&reads, (0.0, 1.0), &mut report);
        assert_eq!(report.covered, 3);
        assert_eq!(report.equal_passed, 2);
        assert_eq!(report.equal_passed_uncovered, 1);
        assert_eq!(
            (
                report.fetched.unlanded,
                report.fetched.all_ones,
                report.fetched.equal_clamped
            ),
            (1, 1, 1)
        );
        // 1.5 clamps to 1.0, which the copy holds; 0.25 is not 0.75.
        assert_eq!(
            (
                report.copied.unlanded,
                report.copied.all_ones,
                report.copied.equal_clamped
            ),
            (0, 0, 2)
        );
        assert_eq!(
            report.copied.first_differences,
            [[2, bits(0.25), bits(0.75)]]
        );
    }
}
