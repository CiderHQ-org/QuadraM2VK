//! What a compositor's reads of its own pass's attachment cost the GPU, on the device the cfg
//! selects, without the validation layer: report only, in release builds, to weigh the copies a
//! draw that samples its pass's attachment takes (`texture/attached_reads.rs`, `shadow.rs`) before
//! and after a change to them.
//!
//! The workload is a window server's composite as the consumer's guest replays it, over a
//! [`WIDTH`]x[`HEIGHT`] `RGBA8Unorm` target of its own image: [`PASSES`] command buffers a frame,
//! each one render pass that loads and stores the target, of [`STEPS`] steps, each a fill of a
//! window's rectangle through its scissor, a memory barrier over the render targets and a draw
//! that reads the target at its own pixels over that rectangle (`tob_read`); then, in every other
//! pass, a barrier and a draw over a window that samples the target at other pixels
//! (`a4_fragment_select_sample`). The windows cover more than a fifth of the target, so no own-pixel
//! read takes a snapshot (`render/snapshot.rs`). The frame waits for its last command buffer.
//!
//! [`WARM`] frames run untimed, then [`TIMED`] frames are printed as one `ATTACHMENT-READ-COST`
//! line: the GPU time per frame (each command buffer's GPU end less its start, summed), the wall
//! time per frame from the first pass's opening through the last command buffer's completion, and
//! the device's work counts per frame: reads through copies, own-pixel reads, splits, snapshots,
//! shadow copies, the copies in of what changed within a rectangle, and the texels each encoder's
//! first copy in of an image's shadow moves.
//!

use super::attachment_reads::{floats, function};
use super::{open_device, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{BarrierScope, PixelFormat, PrimitiveType, RenderStages, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor};
use metal2vulkan_runtime::{
};
use std::time::Instant;

/// The target's size, a 1080p framebuffer's.
const WIDTH: u64 = 1920;
const HEIGHT: u64 = 1080;
/// The command buffers of a frame, each one pass.
const PASSES: u64 = 4;
/// The fill, barrier and own-pixel read steps of a pass.
const STEPS: u64 = 3;
/// The untimed and the timed frames.
const WARM: usize = 10;
const TIMED: usize = 60;

/// Step `step` of pass `pass`'s window: 960x720 pixels, at an offset of its own.
fn window(pass: u64, step: u64) -> [u64; 4] {
    let (x, y) = (
        (pass * 211 + step * 97) % 960,
        (pass * 53 + step * 131) % 360,
    );
    [x, y, x + 960, y + 720]
}

/// The corners, in normalized device coordinates, of the target's pixels `[x0, x1) x [y0, y1)`.
fn corners([x0, y0, x1, y1]: [u64; 4]) -> Vec<u8> {
    let x = |p: u64| p as f32 / WIDTH as f32 * 2.0 - 1.0;
    let y = |p: u64| 1.0 - p as f32 / HEIGHT as f32 * 2.0;
    floats([x(x0), y(y1), x(x1), y(y0)])
}

    let mut d = RenderPipelineDescriptor::new(function(device, "tob_rect", "tob_renders.ll"));
    d.fragment = Some(function(device, name, fixture));
    device
        .new_render_pipeline_state(&d, &PipelineOptions::default())
        .unwrap()
}

/// Draw over `rect`, through a scissor of it.
fn draw_over(encoder: &mut RenderCommandEncoder, [x0, y0, x1, y1]: [u64; 4]) {
    let rect = ScissorRect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    };
    encoder.set_scissor_rect(rect).unwrap();
    encoder
        .set_vertex_bytes(&corners([x0, y0, x1, y1]), 0)
        .unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
}

fn barrier(encoder: &mut RenderCommandEncoder) {
    encoder.memory_barrier_with_scope(
        BarrierScope::RENDER_TARGETS,
        RenderStages::FRAGMENT,
        RenderStages::FRAGMENT,
    );
}

/// The workload's pipelines and sampler.
struct Draws {
    fill: RenderPipelineState,
    read: RenderPipelineState,
    sample: RenderPipelineState,
    sampler: SamplerState,
}

/// One frame over `target` on `queue` (the module docs): its command buffers, committed, the last
/// waited for.
fn frame(queue: &CommandQueue, target: &Texture, draws: &Draws) -> Vec<CommandBuffer> {
    let mut d = RenderPassDescriptor::new();
    let color = &mut d.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let mut committed = Vec::new();
    for pass in 0..PASSES {
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&d).unwrap();
        for step in 0..STEPS {
            let rect = window(pass, step);
            encoder.set_render_pipeline_state(&draws.fill).unwrap();
            let red = (pass * STEPS + step) as f32 / (PASSES * STEPS) as f32;
            encoder
                .set_fragment_bytes(&floats([red, 0.5, 0.0, 1.0]), 0)
                .unwrap();
            draw_over(&mut encoder, rect);
            barrier(&mut encoder);
            encoder.set_render_pipeline_state(&draws.read).unwrap();
            encoder.set_fragment_texture(Some(target), 0).unwrap();
            draw_over(&mut encoder, rect);
        }
        if pass % 2 == 1 {
            barrier(&mut encoder);
            encoder.set_render_pipeline_state(&draws.sample).unwrap();
            encoder.set_fragment_texture(Some(target), 0).unwrap();
            encoder.set_fragment_texture(Some(target), 1).unwrap();
            encoder
                .set_fragment_sampler_state(Some(&draws.sampler), 0)
                .unwrap();
            draw_over(&mut encoder, window(pass, STEPS));
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        committed.push(command_buffer);
    }
    let last = committed.last().unwrap();
    last.wait_until_completed_for(WAIT).unwrap();
    for command_buffer in &committed {
        assert_eq!(command_buffer.error(), None);
    }
    committed
}

/// The work counts the module docs name, per frame over `frames` frames.
fn per_frame(before: &WorkCounts, after: &WorkCounts, frames: usize) -> String {
    let per = |a: u64, b: u64| (a - b) as f64 / frames as f64;
    format!(
        "reads through copies {:.1}, own-pixel reads {:.1}, splits {:.1}, snapshotted {:.1}, \
         shadow copies {:.1}, bounded {:.1}, image first-copy texels {:.0}",
        per(
            after.attachment_reads_through_copies,
            before.attachment_reads_through_copies
        ),
        per(after.own_pixel_reads, before.own_pixel_reads),
        per(after.attachment_read_splits, before.attachment_read_splits),
        per(
            after.reading_draws_snapshotted,
            before.reading_draws_snapshotted
        ),
        per(after.shadow_copies, before.shadow_copies),
        per(after.bounded_shadow_copies, before.bounded_shadow_copies),
        per(
            after.shadow_texels_first_copied_in_images,
            before.shadow_texels_first_copied_in_images
        ),
    )
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn attachment_read_costs() {
    let device = open_device(TEST_DEVICE, DeviceOptions::default()).unwrap();
    let draws = Draws {
        sample: state(
            &device,
            "a4_fragment_select_sample",
            "a4_fragment_select_sample.ll",
        ),
        sampler: device.new_sampler_state(&SamplerDescriptor::new()).unwrap(),
    };
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, WIDTH, HEIGHT, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = ResourceOptions::private();
    let target = device.new_texture(&d).unwrap();
    let queue = device.new_command_queue().unwrap();
    for _ in 0..WARM {
        frame(&queue, &target, &draws);
    }
    let before = device.work_counts();
    let (mut gpu, started) = (0.0, Instant::now());
    for _ in 0..TIMED {
        for command_buffer in frame(&queue, &target, &draws) {
            gpu +=
                command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        }
    }
    let wall = started.elapsed().as_secs_f64();
    let after = device.work_counts();
    println!(
        "ATTACHMENT-READ-COST on {}: {WIDTH}x{HEIGHT} target of its own image, {PASSES} passes of \
         {STEPS} steps: GPU {:.3} ms/frame, wall {:.3} ms/frame; {}",
        device.name(),
        gpu * 1e3 / TIMED as f64,
        wall * 1e3 / TIMED as f64,
        per_frame(&before, &after, TIMED)
    );
}
