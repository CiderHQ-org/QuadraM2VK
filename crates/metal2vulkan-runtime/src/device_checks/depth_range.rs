//! The check of `RuntimeDriverDefect::NormalizedDepthPastUnitRange`, settled as the device is
//! made: it decides whether `set_viewport` refuses a depth range past `[0, 1]` that the device
//! takes by the vertex stage's compensation (`render_state::depth_ranges_refused`). One depth-only
//! draw with the depth clamped (`depthClampEnable`, as the compensation draws) of a triangle
//! sloped in x whose normalized depth runs from 0.5 to 2 over its clip corners, as the
//! compensation draws a viewport of `[0.5, 2]` over a triangle at depth `0.25 + 0.25 x`: across
//! an [`SIDE`] x [`SIDE`] `D32_SFLOAT` target its depth is `0.5 + 3 (x + 0.5) / 32` at column
//! `x`'s centres, 0.828125 at column 3 and past 1 from column 5 on. Metal's answer: each sample
//! below 1 within [`TOLERANCE`] of its plane's depth, every one past it 1. The defect's answer: a
//! sample below 1 further off (Metal's rasterizer on an M3 Max, macOS 27.0.1, a probe: 0.8282051
//! for 0.828125 at column 3, every column below 1 off by 1.1e-5 or more). The slope decides it:
//! there a triangle from 0.75 to 1.75 drew every depth exactly. A sample past 1 not stored as 1,
//! or none stored, answered neither.
//! It does not run where the device enabled no `depthClamp` or `shaderClipDistance`, which the
//! compensation needs (the range is refused there whatever the driver draws).

use super::depth_draws::{self, DepthState, Stages, DEPTH};
use super::kit::{CheckError, Kit};
use super::raster::{self, DepthTarget, Triangle};
use super::Target;
use ash::vk;
use metal2vulkan::device::{CheckAnswer, VkFeature};

/// The target's side, in pixels.
const SIDE: u32 = 8;

/// How far a sample below 1 may lie from its plane's depth: 2^-17, 128 units in the last place
/// below 1, where the defect's reads at this slope lie 1.1e-5 to 1e-4 off.
const TOLERANCE: f32 = 1.0 / 131_072.0;

/// What the depths read before the draw's store reaches them.
const UNWRITTEN: u32 = u32::MAX;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The plane's depth at the centre of column `x`.
fn plane(x: u32) -> f32 {
    0.5 + 3.0 * (x as f32 + 0.5) / 32.0
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let features = &target.enabled.features;
    for (feature, name) in [
        (VkFeature::DepthClamp, "depthClamp"),
        (VkFeature::ShaderClipDistance, "shaderClipDistance"),
    ] {
        if !features.contains(&feature) {
            return CheckAnswer::NotRun(format!("{name} is not enabled"));
        }
    }
    match depth_draws::takes_depth(kit, vk::ImageUsageFlags::TRANSFER_SRC) {
        Ok(true) => {}
        Ok(false) => return CheckAnswer::NotRun("no D32_SFLOAT depth target".to_owned()),
        Err(error) => return CheckAnswer::Unanswered(error.0),
    }
    let depths = match drawn(kit, target) {
        Ok(depths) => depths,
        Err(error) => return CheckAnswer::Unanswered(error.0),
    };
    let mut off = Vec::new();
    for (i, &bits) in depths.iter().enumerate() {
        let (x, y) = (i as u32 % SIDE, i as u32 / SIDE);
        let read = f32::from_bits(bits);
        let want = plane(x);
        if bits == UNWRITTEN {
            return CheckAnswer::Unanswered(format!("the draw stored no depth at ({x}, {y})"));
        }
        if want >= 1.0 {
            if read != 1.0 {
                return CheckAnswer::Unanswered(format!(
                    "the depth past 1 at ({x}, {y}) read {read}, not clamped to 1"
                ));
            }
            continue;
        }
        if (read - want).abs() > TOLERANCE {
            off.push((x, y, read, want));
        }
    }
    match off.first() {
        None => CheckAnswer::Metals,
        Some(&(x, y, read, want)) => CheckAnswer::NotMetals(format!(
            "{} of the samples below 1 lie off their plane's depth, the first at ({x}, {y}) {read} \
             for {want}",
            off.len()
        )),
    }
}

/// The depths the draw stored, row by row.
fn drawn(kit: &mut Kit<'_>, target: &Target<'_>) -> Result<Vec<u32>, CheckError> {
    let vertex = raster::vertex_shader(kit, target)?;
    let sloped = Triangle {
        corners: [
            [-1.0, -1.0, 0.5, 1.0],
            [3.0, -1.0, 2.0, 1.0],
            [-1.0, 3.0, 0.5, 1.0],
        ],
        attributes: [[0.0; 4]; 3],
    };
    let vertices = raster::vertex_buffer(kit, &[sloped])?;
    let (set, layout) = depth_draws::vertex_set(kit, vertices)?;
    let depth = DepthTarget {
        format: DEPTH,
        usage: vk::ImageUsageFlags::TRANSFER_SRC,
        final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        clear: 0.0,
    };
    let pass = raster::pass(kit, (SIDE, SIDE, 1), false, Some(depth))?;
    let Some((image, _)) = pass.depth else {
        return Err(CheckError("the pass has no depth attachment".to_owned()));
    };
    let state = DepthState {
        compare: vk::CompareOp::ALWAYS,
        write: true,
        clamp: true,
    };
    let stages = Stages {
        vertex,
        fragment: None,
        layout,
    };
    let pipeline = depth_draws::pipeline(kit, stages, pass.pass, (SIDE, SIDE, 1), state)?;
    let unwritten = vec![UNWRITTEN; (SIDE * SIDE) as usize];
    let depths = kit.host_buffer(&unwritten, vk::BufferUsageFlags::TRANSFER_DST)?;
    kit.submit(|device, command| {
        let graphics = vk::PipelineBindPoint::GRAPHICS;
        let copy = depth_draws::depth_copy(SIDE, SIDE);
        let written = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);
        // SAFETY: recording; the pass, its framebuffer, the pipeline (of `layout`, whose set 0
        // is `set`) and the buffer are the kit's. The pass leaves its depth in
        // `TRANSFER_SRC_OPTIMAL` behind a dependency of later transfers on its writes.
        unsafe {
            device.cmd_begin_render_pass(command, &pass.begin(), vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, graphics, pipeline);
            device.cmd_bind_descriptor_sets(command, graphics, layout, 0, &[set], &[]);
            device.cmd_set_scissor(command, 0, &[raster::rect(0, 0, SIDE, SIDE)]);
            device.cmd_draw(command, 3, 1, 0, 0);
            device.cmd_end_render_pass(command);
            let source = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            device.cmd_copy_image_to_buffer(command, image, source, depths.buffer, &[copy]);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[written],
                &[],
                &[],
            );
        }
    })?;
    kit.read(&depths)
}
