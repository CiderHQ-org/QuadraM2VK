//! What a window server's frame costs the thread that encodes it, on the device the cfg selects,
//! without the validation layer: report only, in release builds, to weigh the runtime's encode
//! path (a pass's opening, its draws, its end, the commit) before and after a change to it.
//!
//! The workload is a guest window server's frame as the consumer (macOS guests over the runtime)
//! replays it, on one thread: [`BUFFERS`] command buffers committed back to back, each one render
//! pass of [`DRAWS`] indexed draws of six `UInt16` indices, with the state changes a compositor
//! makes between draws. A pass draws, in turn: a textured rectangle from a vertex buffer
//! (`tob_rect`, `tob_sample`; its rectangle at an offset of its own, one of [`TEXTURES`]
//! textures, a linear sampler); another of them; an antialiased mask (`zwd_quad`, `zwd_masked`:
//! its shape as vertex and fragment bytes, a mask texture, the sampler); another mask, of other
//! bytes; a rounded panel (`zwd_panel`: its shape's bytes alone); and a rectangle again. Every
//! pipeline blends premultiplied colors into `BGRA8Unorm_sRGB`. Each pass loads one of
//! [`TARGETS`] window layers and stores it, but every [`RESOLVED`]th, which clears a 4-sample
//! texture and resolves it into its layer. The frame waits for its last command buffer at its end,
//! outside the times below.
//!
//! The device is made as the consumer makes its own: the default policy (the fast forms), GPU
//! times from the host's clock, no execution record, no layer. The `announced` runs add every
//! precision kind to `DeviceOptions::announce_each`, delivered to a sink that counts each kind and
//! keeps its last text, as the consumer's does.
//!
//! [`WARM`] frames run untimed (every pipeline is made), then [`TIMED`] frames are printed as
//! `FRAME-COST` lines: the encoding thread's CPU time and wall time per frame (from the first
//! pass's opening through the last commit); the 50th and 90th percentiles of the wall time of a
//! pass's opening (`render_command_encoder`), of a draw (its state changes and its draw call), of
//! a pass's end (`end_encoding`) and of a commit; where the lib test binary counts its
//! allocations (a build without `--cfg m2v_system_alloc`), the encoding thread's allocations of
//! each; the notices the sink took per frame; and a frame's share of the device's work counts
//! (`Device::work_counts`) the draw and commit paths keep.
//!
//! `frame_costs` measures the plain device then the announced one, each in turn on one device of
//! its own; `frame_costs_plain` and `frame_costs_announced` measure one alone, in a process of its
//! own, for a profile of it (`perf record -g` of the test's process).
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc` for the times, whose lib test binary keeps the system
//! allocator as a consumer's process does (the counting one costs the draw path about a sixth of
//! its time), and in one without it for the allocations.
//!
//! One check runs by default where the allocations are counted:
//! `a_frame_s_draws_allocate_within_their_bound` holds a frame's draws to [`DRAW_ALLOCATIONS`]
//! on the encoding thread, a draw's state changes and its draw call together, on average. It
//! counts, and times nothing.

use super::{buffer_with, loader, open_device, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use crate::worker::alloc;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{
    BlendFactor, DataType, IndexType, PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, CommandQueue, Degradation,
    DegradationCategory, Device, DeviceOptions, Library, LoadAction, MessageSink, PipelineOptions,
    Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, ShaderCacheOptions, StoreAction, Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The untimed and the timed frames of a run.
const WARM: usize = 20;
const TIMED: usize = 100;
/// The command buffers of a frame, each one pass.
const BUFFERS: usize = 67;
/// The draws of a pass.
const DRAWS: usize = 6;
/// Every `RESOLVED`th pass draws into a 4-sample texture resolved into its layer.
const RESOLVED: usize = 4;
/// The window layers the passes draw into, and their side.
const TARGETS: usize = 4;
const TARGET_SIDE: u64 = 512;
/// The sampled textures the rectangles take in turn, and their side.
const TEXTURES: usize = 16;
const TEXTURE_SIDE: u64 = 64;
/// The bytes of `zwd_shapes`' `ZwdShape`: rectangle, fill, outline and shape.
const SHAPE_BYTES: usize = 64;
/// The most allocations a frame's draw makes on the encoding thread on average, its state changes
/// and its draw call together, once every pipeline is made. Measured: 67.5 on RADV (Radeon 780M)
/// and 67.6 on lavapipe; 97.8 on RADV before each texture kept its storage view formats, each
/// encoder its pipeline state's routing, and the format table its entries.
#[cfg(not(m2v_system_alloc))]
const DRAW_ALLOCATIONS: f64 = 72.0;

/// The pipelines of one raster sample count: the rectangle's, the mask's and the panel's.
struct Pipelines {
    rect: RenderPipelineState,
    masked: RenderPipelineState,
    panel: RenderPipelineState,
}

/// The device, its queue, and what the frames draw with.
struct Workload {
    device: Device,
    queue: CommandQueue,
    /// One sample, and four.
    pipelines: [Pipelines; 2],
    sampler: SamplerState,
    textures: Vec<Texture>,
    mask: Texture,
    /// The window layers, and the 4-sample texture a resolved pass draws into.
    targets: Vec<Texture>,
    multisampled: Texture,
    /// Six `UInt16` indices, 0 to 5, and the rectangles' corners, one `float4` a draw of a frame.
    indices: Buffer,
    rects: Buffer,
}

/// A function of `library`, its constant 0 (`zwd_variant`) 0.
fn function(library: &Library, name: &str) -> StageSource {
    let constants = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::UInt, 0u32.to_le_bytes().to_vec()),
    );
    StageSource::Function(
        library
            .new_function_with_name_constant_values(name, &constants)
            .unwrap()
            .function_descriptor(),
    )
}

/// A pipeline state of `vertex` and `fragment` blending premultiplied colors into
/// `BGRA8Unorm_sRGB`, rasterizing `samples` samples.
fn pipeline(
    device: &Device,
    vertex: StageSource,
    fragment: StageSource,
    samples: u32,
) -> RenderPipelineState {
    let mut descriptor = RenderPipelineDescriptor::new(vertex);
    descriptor.fragment = Some(fragment);
    descriptor.raster_sample_count = samples;
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::BGRA8UnormSRGB);
    color.blending_enabled = true;
    color.source_rgb_blend_factor = BlendFactor::One;
    color.source_alpha_blend_factor = BlendFactor::One;
    color.destination_rgb_blend_factor = BlendFactor::OneMinusSourceAlpha;
    color.destination_alpha_blend_factor = BlendFactor::OneMinusSourceAlpha;
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A `TEXTURE_SIDE` square `BGRA8Unorm` texture, every texel `texel`.
fn sampled(device: &Device, texel: [u8; 4]) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, TEXTURE_SIDE, TEXTURE_SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let texels: Vec<u8> = (0..TEXTURE_SIDE * TEXTURE_SIDE)
        .flat_map(|_| texel)
        .collect();
    let region = Region::new_2d(0, 0, TEXTURE_SIDE, TEXTURE_SIDE);
    texture
        .replace_region(region, 0, 0, &texels, 4 * TEXTURE_SIDE, 0)
        .unwrap();
    texture
}

/// A `TARGET_SIDE` square `BGRA8Unorm_sRGB` render target of `samples` samples.
fn target(device: &Device, samples: u64) -> Texture {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8UnormSRGB, TARGET_SIDE, TARGET_SIDE, false);
    if samples > 1 {
        descriptor.texture_type = TextureType::Type2DMultisample;
        descriptor.sample_count = samples;
    }
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::private();
    device.new_texture(&descriptor).unwrap()
}

/// `ZwdShape`'s bytes: draw `n`'s box in a pass of the target's size, a fill and an outline.
fn shape(n: usize) -> [u8; SHAPE_BYTES] {
    let x = (n % 8) as f32 * 60.0;
    let y = (n / 8 % 8) as f32 * 60.0;
    let words: [f32; 16] = [
        x,
        y,
        x + 48.0,
        y + 32.0,
        0.5,
        0.25,
        0.125,
        0.5,
        0.25,
        0.25,
        0.25,
        0.25,
        6.0,
        TARGET_SIDE as f32,
        TARGET_SIDE as f32,
        0.0,
    ];
    let mut bytes = [0u8; SHAPE_BYTES];
    for (at, word) in words.iter().enumerate() {
        bytes[4 * at..4 * at + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

impl Workload {
    fn new(device: Device) -> Workload {
        let library = |name: &str| {
            device.new_library_with_air(&llvm_assemble(&engine_dir().join(name)).unwrap())
        };
        let tob = library("tob_renders.ll").unwrap();
        let zwd = library("zwd_shapes.ll").unwrap();
        let pipelines = [1, 4].map(|samples| Pipelines {
            rect: pipeline(
                &device,
                function(&tob, "tob_rect"),
                function(&tob, "tob_sample"),
                samples,
            ),
            masked: pipeline(
                &device,
                function(&zwd, "zwd_quad"),
                function(&zwd, "zwd_masked"),
                samples,
            ),
            panel: pipeline(
                &device,
                function(&zwd, "zwd_quad"),
                function(&zwd, "zwd_panel"),
                samples,
            ),
        });
        let mut sampler = SamplerDescriptor::new();
        sampler.min_filter = SamplerMinMagFilter::Linear;
        sampler.mag_filter = SamplerMinMagFilter::Linear;
        let sampler = device.new_sampler_state(&sampler).unwrap();
        let textures = (0..TEXTURES)
            .map(|n| sampled(&device, [n as u8 * 16, 128, 255 - n as u8 * 16, 255]))
            .collect();
        let indices: Vec<u8> = (0u16..6).flat_map(u16::to_le_bytes).collect();
        // A rectangle a draw: a 25 by 20 grid of them over the target.
        let rects: Vec<u8> = (0..BUFFERS * DRAWS)
            .flat_map(|n| {
                let (x, y) = ((n % 25) as f32, (n / 25 % 20) as f32);
                let at = |v: f32, cells: f32| v * 2.0 / cells - 1.0;
                [
                    at(x, 25.0),
                    at(y, 20.0),
                    at(x + 1.0, 25.0),
                    at(y + 1.0, 20.0),
                ]
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        Workload {
            queue: device.new_command_queue().unwrap(),
            pipelines,
            sampler,
            textures,
            mask: sampled(&device, [255, 255, 255, 128]),
            targets: (0..TARGETS).map(|_| target(&device, 1)).collect(),
            multisampled: target(&device, 4),
            indices: buffer_with(&device, &indices).unwrap(),
            rects: buffer_with(&device, &rects).unwrap(),
            device,
        }
    }

    /// Pass `pass`'s descriptor: its layer loaded and stored, or, for a resolved pass, the
    /// 4-sample texture cleared and resolved into the layer.
    fn pass(&self, pass: usize) -> RenderPassDescriptor {
        let mut descriptor = RenderPassDescriptor::new();
        let color = &mut descriptor.color_attachments[0];
        let layer = self.targets[pass % TARGETS].clone();
        if pass % RESOLVED == RESOLVED - 1 {
            color.texture = Some(self.multisampled.clone());
            color.resolve_texture = Some(layer);
            color.load_action = LoadAction::Clear;
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = StoreAction::MultisampleResolve;
        } else {
            color.texture = Some(layer);
            color.load_action = LoadAction::Load;
            color.store_action = StoreAction::Store;
        }
        descriptor
    }

    /// Draw `draw` of pass `pass` (the module docs), its state changes first.
    fn draw(&self, encoder: &mut RenderCommandEncoder, pass: usize, draw: usize) {
        let pipelines = &self.pipelines[usize::from(pass % RESOLVED == RESOLVED - 1)];
        let n = pass * DRAWS + draw;
        match draw {
            0 | 1 | 5 => {
                if draw != 1 {
                    encoder.set_render_pipeline_state(&pipelines.rect).unwrap();
                    encoder
                        .set_fragment_sampler_state(Some(&self.sampler), 0)
                        .unwrap();
                }
                encoder
                    .set_vertex_buffer(Some(&self.rects), 16 * n as u64, 0)
                    .unwrap();
                encoder
                    .set_fragment_texture(Some(&self.textures[n % TEXTURES]), 0)
                    .unwrap();
            }
            2 | 3 => {
                if draw == 2 {
                    encoder
                        .set_render_pipeline_state(&pipelines.masked)
                        .unwrap();
                    encoder.set_fragment_texture(Some(&self.mask), 0).unwrap();
                    encoder
                        .set_fragment_sampler_state(Some(&self.sampler), 0)
                        .unwrap();
                }
                let shape = shape(n);
                encoder.set_vertex_bytes(&shape, 0).unwrap();
                encoder.set_fragment_bytes(&shape, 0).unwrap();
            }
            _ => {
                encoder.set_render_pipeline_state(&pipelines.panel).unwrap();
                let shape = shape(n);
                encoder.set_vertex_bytes(&shape, 0).unwrap();
                encoder.set_fragment_bytes(&shape, 0).unwrap();
            }
        }
        encoder
            .draw_indexed_primitives(
                PrimitiveType::Triangle,
                6,
                IndexType::UInt16,
                &self.indices,
                0,
                1,
                0,
                0,
            )
            .unwrap();
    }

    /// Encode and commit one frame, then wait for its last command buffer: what each step cost.
    fn frame(&self) -> Frame {
        let mut frame = Frame::default();
        let mut last: Option<CommandBuffer> = None;
        let cpu = thread_cpu();
        let start = Instant::now();
        for pass in 0..BUFFERS {
            let command_buffer = self.queue.command_buffer().unwrap();
            let descriptor = self.pass(pass);
            let (mut encoder, open) =
                step(|| command_buffer.render_command_encoder(&descriptor).unwrap());
            frame.open.push(open);
            for draw in 0..DRAWS {
                let ((), drawn) = step(|| self.draw(&mut encoder, pass, draw));
                frame.draws.push(drawn);
            }
            let ((), ended) = step(|| encoder.end_encoding().unwrap());
            frame.end.push(ended);
            let ((), committed) = step(|| command_buffer.commit().unwrap());
            frame.commit.push(committed);
            last = Some(command_buffer);
        }
        frame.wall = start.elapsed();
        frame.cpu = thread_cpu().saturating_sub(cpu);
        let last = last.unwrap();
        last.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(last.status(), CommandBufferStatus::Completed);
        frame
    }
}

/// One step's wall time, and the encoding thread's allocations in it where they are counted.
#[derive(Clone, Copy)]
struct Step {
    wall: Duration,
    allocations: Option<u64>,
}

/// Run `body` as one step.
fn step<R>(body: impl FnOnce() -> R) -> (R, Step) {
    let allocations = alloc::thread_allocations();
    let start = Instant::now();
    let result = body();
    let wall = start.elapsed();
    let allocations = allocations.map(|before| alloc::thread_allocations().unwrap() - before);
    (result, Step { wall, allocations })
}

/// A frame's steps of one kind ([`Frame`]'s `open`, `draws`, `end` or `commit`).
type StepsOf = fn(&Frame) -> &[Step];

/// One frame's steps and times ([`Workload::frame`]).
#[derive(Default)]
struct Frame {
    open: Vec<Step>,
    draws: Vec<Step>,
    end: Vec<Step>,
    commit: Vec<Step>,
    wall: Duration,
    cpu: Duration,
}

/// The calling thread's CPU time so far.
fn thread_cpu() -> Duration {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the timespec it is given and has no other effect.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID)");
    Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
}

/// The 50th and 90th percentiles of `values`.
fn percentiles(mut values: Vec<f64>) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    [at(50), at(90)]
}

/// The kinds announced, each with its count and last text: the consumer's sink.
type Heard = Arc<Mutex<BTreeMap<Degradation, (u64, String)>>>;

/// A device as the consumer makes its own (the module docs), announcing every precision kind at
/// every occurrence where `announced`, into `heard`.
fn device(cache: &ScratchCache, announced: bool, heard: &Heard) -> Device {
    let mut options = DeviceOptions::default();
    options.shader_cache = ShaderCacheOptions::in_directory(cache.path());
    options.loader = loader();
    options.gpu_times_from_host = true;
    let sink = Arc::clone(heard);
    options.messages = Some(MessageSink::new(move |message| {
        if let Some(notice) = &message.degradation {
            let mut heard = sink.lock().unwrap();
            let entry = heard.entry(notice.degradation).or_default();
            entry.0 += 1;
            entry.1.clone_from(&message.text);
        }
    }));
    if announced {
        options.announce_each = Degradation::ALL
            .iter()
            .copied()
            .filter(|kind| kind.category() == DegradationCategory::Precision)
            .collect();
    }
    open_device(TEST_DEVICE, options).unwrap()
}

/// The frame's costs, printed (the module docs): a run of each of `runs` (whether it announces)
/// in turn, each on a device of its own.
fn measure(runs: &[bool]) {
    for &announced in runs {
        let cache = ScratchCache::new().unwrap();
        let heard = Heard::default();
        let work = Workload::new(device(&cache, announced, &heard));
        let label = match announced {
            true => "announced",
            false => "plain",
        };
        let name = work.device.name().to_owned();
        for _ in 0..WARM {
            work.frame();
        }
        let notices_before: u64 = heard.lock().unwrap().values().map(|(n, _)| n).sum();
        let counts_before = work.device.work_counts();
        let frames: Vec<Frame> = (0..TIMED).map(|_| work.frame()).collect();
        let counts = work.device.work_counts();
        let notices: u64 = heard.lock().unwrap().values().map(|(n, _)| n).sum();
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let [cpu, cpu_high] = percentiles(frames.iter().map(|f| ms(f.cpu)).collect());
        let [wall, wall_high] = percentiles(frames.iter().map(|f| ms(f.wall)).collect());
        eprintln!(
            "FRAME-COST {name} {label}: {BUFFERS} command buffers of one pass of {DRAWS} indexed \
             draws a frame, every {RESOLVED}th pass resolved, {WARM} untimed and {TIMED} timed \
             frames: the encoding thread's CPU {cpu:.2} ms a frame (p90 {cpu_high:.2}), wall \
             {wall:.2} ms (p90 {wall_high:.2}); {:.1} us of CPU a draw over the frame; {:.1} \
             notices a frame",
            cpu * 1e3 / (BUFFERS * DRAWS) as f64,
            (notices - notices_before) as f64 / TIMED as f64,
        );
        let steps: [(&str, StepsOf); 4] = [
            ("pass open", |f| &f.open),
            ("draw", |f| &f.draws),
            ("pass end", |f| &f.end),
            ("commit", |f| &f.commit),
        ];
        for (step, of) in steps {
            let all: Vec<Step> = frames.iter().flat_map(|f| of(f).iter().copied()).collect();
            let [p50, p90] = percentiles(all.iter().map(|s| s.wall.as_secs_f64() * 1e6).collect());
            let allocations = match all.iter().map(|s| s.allocations).sum::<Option<u64>>() {
                Some(total) => format!("{:.2} allocations each", total as f64 / all.len() as f64),
                None => "allocations not counted".to_owned(),
            };
            eprintln!(
                "FRAME-COST {name} {label}: {step}: {p50:.2} us (p90 {p90:.2}); {allocations}"
            );
        }
        let per_frame = |count: fn(&WorkCounts) -> u64| {
            (count(&counts) - count(&counts_before)) as f64 / TIMED as f64
        };
        eprintln!(
            "FRAME-COST {name} {label}: a frame's work: {:.0} bind shapes made, {:.0} draws \
             settled, {:.0} pipelines looked up, {:.0} draw states recorded, {:.0} argument \
             tables retained, {:.1} command pools made, {:.1} timeline waits, {:.1} addressable \
             buffers made",
            per_frame(|c| c.draw_shapes_made),
            per_frame(|c| c.draws_settled),
            per_frame(|c| c.graphics_pipelines_looked_up),
            per_frame(|c| c.draw_states_recorded),
            per_frame(|c| c.argument_tables_retained),
            per_frame(|c| c.command_pools_made),
            per_frame(|c| c.timeline_waits),
            per_frame(|c| c.addressable_buffers_made),
        );
        if announced {
            for (kind, (count, _)) in heard.lock().unwrap().iter() {
                eprintln!(
                    "FRAME-COST {name} {label}: heard {} {count} times",
                    kind.code()
                );
            }
        }
    }
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs() {
    measure(&[false, true]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_plain() {
    measure(&[false]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_announced() {
    measure(&[true]);
}

/// A frame's draws allocate at most [`DRAW_ALLOCATIONS`] each on average on the encoding thread,
/// once every pipeline is made: what the draw path keeps of what it found before (each texture's
/// storage view formats, the pipeline state's routing into the pass's colors, the settles of its
/// latest bind shapes, the format table) is not made again at each draw.
#[cfg(not(m2v_system_alloc))]
#[test]
fn a_frame_s_draws_allocate_within_their_bound() {
    let cache = ScratchCache::new().unwrap();
    let heard = Heard::default();
    let work = Workload::new(device(&cache, false, &heard));
    // Every pipeline made, and every pool piece done, before the frame that counts.
    for _ in 0..WARM {
        work.frame();
    }
    let frame = work.frame();
    let allocations: u64 = frame.draws.iter().map(|s| s.allocations.unwrap()).sum();
    let each = allocations as f64 / frame.draws.len() as f64;
    eprintln!(
        "FRAME-COST {}: {each:.2} allocations a draw, at most {DRAW_ALLOCATIONS} kept",
        work.device.name()
    );
    assert!(
        each <= DRAW_ALLOCATIONS,
        "{each:.2} allocations a draw, more than {DRAW_ALLOCATIONS}"
    );
}
