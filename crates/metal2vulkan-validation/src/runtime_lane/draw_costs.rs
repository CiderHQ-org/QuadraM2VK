//! What encoding a stream of draws costs on the device the cfg selects, without the validation
//! layer (the consumer's per-draw encode): report only, in release builds, to weigh the runtime's
//! draw path before and after a change to it.
//!
//! The workload is a browser's scrolling, as the consumer (macOS guests over the runtime) encodes
//! it: many draws a pass and few state changes, each draw a textured rectangle of its own inline
//! vertex bytes (`tob_rect`, `tob_sample`). Each stream encodes [`WARM`] command buffers untimed,
//! then [`TIMED`], each one render pass of [`DRAWS`] draws, and is printed as a `DRAW-COST` line:
//! the draws a second the host encodes (from the encoder's creation through `end_encoding`, over
//! every timed command buffer), the 10th, 50th and 90th percentiles of a command buffer's encode
//! time per draw, the median of the encoding thread's own CPU time per draw over the same span
//! (below the wall time where encoding waits on something, a fence or a pool, rather than
//! working), and the median time from `commit` until `wait_until_completed` returns.
//!
//! - **`unchanged`**: every draw the same, nothing set between draws.
//! - **`bytes`**: each draw its own rectangle's inline bytes (`setVertexBytes`).
//! - **`textures`**: each draw its own bytes and one of [`TEXTURES`] textures of one format.
//! - **`pipelines`**: as `textures`, the pipeline state switched between two (the second
//!   blends) every [`SWITCH`] draws.
//!
//! `draw_costs` measures the streams in turn; `draw_costs_<stream>` measures one stream alone, in
//! a process of its own, for a profile of that stream.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system allocator as a
//! consumer's process does: the counting one costs the draw path about a sixth of its time.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    CommandBufferStatus, CommandQueue, Device, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState, StoreAction, Texture,
    TextureDescriptor,
};
use std::time::{Duration, Instant};

/// The untimed and the timed command buffers of each stream.
const WARM: usize = 5;
const TIMED: usize = 20;
/// The draws of a command buffer's pass.
const DRAWS: usize = 500;
/// The textures the `textures` and `pipelines` streams take in turn, and their side.
const TEXTURES: usize = 16;
const TEXTURE_SIDE: u64 = 64;
/// The target's side.
const TARGET_SIDE: u64 = 256;
/// The draws between the `pipelines` stream's switches.
const SWITCH: usize = 50;

/// The streams (the module docs).
#[derive(Clone, Copy, Debug)]
enum Stream {
    Unchanged,
    Bytes,
    Textures,
    Pipelines,
}

impl Stream {
    fn name(self) -> &'static str {
        match self {
            Stream::Unchanged => "unchanged",
            Stream::Bytes => "bytes",
            Stream::Textures => "textures",
            Stream::Pipelines => "pipelines",
        }
    }
}

/// The device, its queue, and what the streams draw with.
struct Workload {
    device: Device,
    queue: CommandQueue,
    /// The plain pipeline state, and the one that blends.
    pipelines: [RenderPipelineState; 2],
    sampler: SamplerState,
    textures: Vec<Texture>,
    target: Texture,
}

impl Workload {
    fn new(device: Device) -> Workload {
        let ll = engine_dir().join("tob_renders.ll");
        let library = device
            .new_library_with_air(&llvm_assemble(&ll).unwrap())
            .unwrap();
        let function = |name| {
            StageSource::Function(
                library
                    .new_function_with_name(name)
                    .unwrap()
                    .function_descriptor(),
            )
        };
        let pipeline = |blends: bool| {
            let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
            descriptor.fragment = Some(function("tob_sample"));
            let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
            color.blending_enabled = blends;
            descriptor.color_attachments[0] = Some(color);
            device
                .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        };
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let texture = |n: usize| {
            let mut descriptor = TextureDescriptor::texture_2d(
                PixelFormat::RGBA8Unorm,
                TEXTURE_SIDE,
                TEXTURE_SIDE,
                false,
            );
            descriptor.usage = TextureUsage::SHADER_READ;
            descriptor.resource_options = ResourceOptions::shared();
            let texture = device.new_texture(&descriptor).unwrap();
            let texel = [n as u8 * 16, 255 - n as u8 * 16, 128, 255];
            let texels: Vec<u8> = (0..TEXTURE_SIDE * TEXTURE_SIDE)
                .flat_map(|_| texel)
                .collect();
            let region = Region::new_2d(0, 0, TEXTURE_SIDE, TEXTURE_SIDE);
            texture
                .replace_region(region, 0, 0, &texels, 4 * TEXTURE_SIDE, 0)
                .unwrap();
            texture
        };
        let mut target =
            TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, TARGET_SIDE, TARGET_SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::private();
        Workload {
            queue: device.new_command_queue().unwrap(),
            pipelines: [pipeline(false), pipeline(true)],
            sampler,
            textures: (0..TEXTURES).map(texture).collect(),
            target: device.new_texture(&target).unwrap(),
            device,
        }
    }

    /// Encode one command buffer of `stream`, commit it and wait for it: the wall time its
    /// encoding took, the encoding thread's CPU time over the same span, and the time from its
    /// commit until the wait returned.
    fn run(&self, stream: Stream, corners: &metal2vulkan_runtime::Buffer) -> Run {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::DontCare;
        color.store_action = StoreAction::Store;
        let start = Instant::now();
        let cpu_start = thread_cpu();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder
            .set_render_pipeline_state(&self.pipelines[0])
            .unwrap();
        encoder
            .set_fragment_sampler_state(Some(&self.sampler), 0)
            .unwrap();
        encoder.set_vertex_buffer(Some(corners), 0, 0).unwrap();
        encoder
            .set_fragment_texture(Some(&self.textures[0]), 0)
            .unwrap();
        for draw in 0..DRAWS {
            if !matches!(stream, Stream::Unchanged) {
                // A rectangle of its own: one of a 25 by 20 grid over the target.
                let (x, y) = ((draw % 25) as f32, (draw / 25) as f32);
                let at = |v: f32, n: f32| v * 2.0 / n - 1.0;
                let rect = [
                    at(x, 25.0),
                    at(y, 20.0),
                    at(x + 1.0, 25.0),
                    at(y + 1.0, 20.0),
                ];
                let bytes: Vec<u8> = rect.iter().flat_map(|v| v.to_le_bytes()).collect();
                encoder.set_vertex_bytes(&bytes, 0).unwrap();
            }
            if matches!(stream, Stream::Textures | Stream::Pipelines) {
                let texture = &self.textures[draw % TEXTURES];
                encoder.set_fragment_texture(Some(texture), 0).unwrap();
            }
            if matches!(stream, Stream::Pipelines) && draw % SWITCH == 0 {
                let pipeline = &self.pipelines[(draw / SWITCH) % 2];
                encoder.set_render_pipeline_state(pipeline).unwrap();
            }
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        let encoded = start.elapsed();
        let cpu = thread_cpu().saturating_sub(cpu_start);
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let latency = committed.elapsed();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Run {
            encoded,
            cpu,
            latency,
        }
    }
}

/// One command buffer's times ([`Workload::run`]).
struct Run {
    encoded: Duration,
    cpu: Duration,
    latency: Duration,
}

/// The calling thread's CPU time so far.
pub(super) fn thread_cpu() -> Duration {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the timespec it is given and has no other effect.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID)");
    Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
}

/// The 10th, 50th and 90th percentiles of `values`.
pub(super) fn percentiles(mut values: Vec<f64>) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    [at(10), at(50), at(90)]
}

/// The draw path's costs, printed (the module docs): `streams` measured in turn on one device.
fn measure(streams: &[Stream]) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let name = device.name().to_owned();
    let work = Workload::new(device);
    let corners = [-1.0f32, -1.0, 1.0, 1.0].map(f32::to_le_bytes).concat();
    let corners = buffer_with(&work.device, &corners).unwrap();
    eprintln!(
        "DRAW-COST {name}: {DRAWS} draws a pass, {WARM} untimed and {TIMED} timed command \
         buffers a stream, {TEXTURES} textures {TEXTURE_SIDE}x{TEXTURE_SIDE}, target \
         {TARGET_SIDE}x{TARGET_SIDE}"
    );
    for &stream in streams {
        for _ in 0..WARM {
            work.run(stream, &corners);
        }
        let runs: Vec<Run> = (0..TIMED).map(|_| work.run(stream, &corners)).collect();
        let encoded: Duration = runs.iter().map(|run| run.encoded).sum();
        let rate = (TIMED * DRAWS) as f64 / encoded.as_secs_f64();
        let per_draw = |d: Duration| d.as_secs_f64() * 1e6 / DRAWS as f64;
        let [low, median, high] =
            percentiles(runs.iter().map(|run| per_draw(run.encoded)).collect());
        let [_, cpu, _] = percentiles(runs.iter().map(|run| per_draw(run.cpu)).collect());
        let [_, latency, _] = percentiles(
            runs.iter()
                .map(|run| run.latency.as_secs_f64() * 1e3)
                .collect(),
        );
        eprintln!(
            "DRAW-COST {name}: {}: {rate:.0} draws/s encoding; {median:.2} us a draw (p10 \
             {low:.2}, p90 {high:.2}), {cpu:.2} us of it the thread's CPU; commit to \
             completion {latency:.2} ms",
            stream.name()
        );
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn draw_costs() {
    measure(&[
        Stream::Unchanged,
        Stream::Bytes,
        Stream::Textures,
        Stream::Pipelines,
    ]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn draw_costs_unchanged() {
    measure(&[Stream::Unchanged]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn draw_costs_bytes() {
    measure(&[Stream::Bytes]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn draw_costs_textures() {
    measure(&[Stream::Textures]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn draw_costs_pipelines() {
    measure(&[Stream::Pipelines]);
}
