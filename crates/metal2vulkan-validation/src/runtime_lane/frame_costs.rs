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
//! keeps its last text, as the consumer's does. The `checked` runs give each command buffer a
//! submission check of its own before its commit (`CommandBuffer::set_submission_check`), as the
//! consumer commits every command buffer: a check that notes the submission under a lock and
//! lets it run, as the consumer's does while nothing is detached. The device's scheduler thread
//! then runs each check and submits each batch, so the commit hands its batches off to it.
//!
//! [`WARM`] frames run untimed (every pipeline is made), the first of them reported alone (its CPU
//! and wall time, its blocks, its commits' median and longest wall time), then [`TIMED`] frames are
//! printed as `FRAME-COST` lines: the encoding thread's CPU time and wall time per frame (from the
//! first pass's opening through the last commit), and how often it blocked, inside its commits too,
//! and was preempted meanwhile (on Linux, its voluntary and involuntary context switches, from
//! `getrusage`: a commit that waits for the scheduler's lock blocks); the 50th and 90th percentiles
//! of the wall time of a pass's opening (`render_command_encoder`), of a draw (its state changes
//! and its draw call), of a pass's end (`end_encoding`) and of a commit; where the lib test binary
//! counts its allocations (a build without `--cfg m2v_system_alloc`), the encoding thread's
//! allocations of each; where the kernel lets a thread count its own (Linux), the user-space
//! instructions the encoding thread retired in each, on average and their 50th percentile, and in
//! a frame's steps together (steady where the times are not, so a change shows run to run),
//! counted in [`COUNTED`] frames of their own after the timed ones: each read of the counter is a
//! system call, which a timed frame's CPU time would hold; the notices the sink took per frame;
//! and a frame's share of the device's work counts (`Device::work_counts`) the draw and commit
//! paths keep, the command buffers begun and submitted among them. Then the CPU time of the
//! process's other threads over the timed frames, their waits included, per command buffer (on
//! Linux, from each thread's `schedstat`), with its user-space part (from its `stat`, in clock
//! ticks) and how often they slept (their voluntary context switches, from their `status`: the
//! scheduler's are the wakes it took, a commit's among them): the runtime's scheduler, completion,
//! queue-wait and handler threads by name, and the rest (the driver's) together; with their
//! allocations per command buffer where they are counted, and the submission checks run per frame.
//!
//! `frame_costs` measures the plain device, the checked one, then the announced one, each in turn
//! on one device of its own; `frame_costs_plain`, `frame_costs_checked` and
//! `frame_costs_announced` measure one alone, in a process of its own, for a profile of it
//! (`perf record -g` of the test's process).
//!
//! `frame_costs_over_buffers` measures the plain device with window layers that are textures over
//! `Shared` buffers at tight rows (2048 bytes, which RADV and ANV render in place, as linear images
//! over the buffers' memory) in place of private textures, their passes loaded and stored, loaded
//! and stored `DontCare`, and cleared and stored `DontCare`, each on a device of its own: what a
//! pass that does not store such a layer costs, rendered through the image its texture keeps so
//! that the buffer keeps its bytes. Its lines also give a frame's wall time through its last
//! command buffer's completion, the GPU's work included.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc` for the times, whose lib test binary keeps the system
//! allocator as a consumer's process does (the counting one costs the draw path about a sixth of
//! its time), and in one without it for the allocations.
//!
//! Three checks run by default where the allocations are counted:
//! `a_frame_s_draws_and_pass_ends_allocate_within_their_bounds` holds a frame's draws to
//! [`DRAW_ALLOCATIONS`] on the encoding thread, a draw's state changes and its draw call
//! together, and its passes' ends to [`PASS_END_ALLOCATIONS`], each on average;
//! `a_frame_s_pass_opens_allocate_within_their_bound` its pass opens to [`OPEN_ALLOCATIONS`]; and
//! `a_frame_s_commits_allocate_within_their_bound` holds a checked frame's commits to
//! [`COMMIT_ALLOCATIONS`] on the encoding thread and to [`SUBMIT_ALLOCATIONS`] on the others
//! (the scheduler's checks and submissions, the completion), per command buffer on average.
//! Another runs by default on every device,
//! `a_frame_s_passes_and_commits_begin_their_command_buffers_within_their_bound`: each pass, from
//! its opening through its end, begins at most [`PASS_BEGINS`] command buffers, and each commit
//! none but the barrier command buffers it records to keep, on a device that keeps them
//! (`Device::work_counts`). They count, and time nothing.

use super::{buffer_with, loader, open_device, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use crate::worker::alloc;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::error::RuntimeDriverDefect;
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
    SamplerState, ShaderCacheOptions, StoreAction, SubmissionCheck, Texture, TextureDescriptor,
    WorkCounts,
};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The untimed and the timed frames of a run, and the frames whose instructions it counts.
const WARM: usize = 20;
const TIMED: usize = 100;
const COUNTED: usize = 20;
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
/// and its draw call together, once every pipeline is made. Measured: 24.5 on RADV (Radeon 780M)
/// and ANV (Iris Xe), 24.7 on lavapipe; 67.5 on RADV before a draw compared its
/// pipeline key and bind shape in place, held what it retains by reference, kept its small lists,
/// descriptor writes and push bytes on the stack, and found its pass colors' routing kept; 97.8
/// before each texture kept its storage view formats, each encoder its pipeline state's routing,
/// and the format table its entries.
#[cfg(not(m2v_system_alloc))]
const DRAW_ALLOCATIONS: f64 = 27.0;
/// The most allocations a frame's pass open (`render_command_encoder`) makes on the encoding
/// thread on average, once every pipeline is made. Measured: 13.0 on RADV (Radeon 780M), on
/// lavapipe and on ANV (Iris Xe); 18.0 on RADV and ANV before the device kept each pass shape's
/// compatible render pass.
#[cfg(not(m2v_system_alloc))]
const OPEN_ALLOCATIONS: f64 = 14.0;
/// The most allocations a frame's pass end (`end_encoding`) makes on the encoding thread on
/// average, once every pipeline is made. Measured: 36.6 on RADV (Radeon 780M) and ANV (Iris Xe),
/// 38.6 on lavapipe, where an encoder's holds reach its command buffer as one entry, which starts
/// the command buffer's list; 35.6 and 37.6 where the first encoder's list became the command
/// buffer's as it was; 52.1 on RADV before a pass end retained what its commands use as one object
/// and stopped making the lists it found empty.
#[cfg(not(m2v_system_alloc))]
const PASS_END_ALLOCATIONS: f64 = 40.0;
/// The most command buffers a pass begins, from its opening through its end, once every pipeline
/// is made: its primary and its draws' secondary.
const PASS_BEGINS: u64 = 2;
/// The most allocations a checked frame's commit makes on the encoding thread on average, once
/// every pipeline is made. Measured: 4.7 to 5.0 on RADV (Radeon 780M), 5.0 on lavapipe and on ANV
/// (Iris Xe); 19.0 on RADV before a commit listed only what it reaches among the live buffers and
/// the scheduler kept its order's lines between passes.
#[cfg(not(m2v_system_alloc))]
const COMMIT_ALLOCATIONS: f64 = 6.0;
/// The most allocations the process's other threads make per command buffer of a checked frame:
/// its check and its batch's planning and submission on the scheduler thread, its completion and
/// its handlers, which varies with how the threads take turns. Measured: 14.8 to 16.3 on RADV
/// (Radeon 780M), 16.8 to 17.5 on lavapipe, 15.7 to 16.8 on ANV (Iris Xe); 46.9 on RADV before the
/// hazard tracker planned without per-resource allocations, and 21.2 before the scheduler kept its
/// order's lines between passes.
#[cfg(not(m2v_system_alloc))]
const SUBMIT_ALLOCATIONS: f64 = 20.0;

/// One run's device, commits and layers: whether the device announces every precision kind,
/// whether each command buffer commits with a submission check ([`Checks`]), and what the window
/// layers are.
#[derive(Clone, Copy)]
struct Run {
    announced: bool,
    checked: bool,
    layers: Layers,
}

impl Run {
    const PLAIN: Run = Run {
        announced: false,
        checked: false,
        layers: Layers::Private,
    };
    const CHECKED: Run = Run {
        announced: false,
        checked: true,
        layers: Layers::Private,
    };
    const ANNOUNCED: Run = Run {
        announced: true,
        checked: false,
        layers: Layers::Private,
    };

    /// The plain device's run with layers over buffers, its passes loading and storing them so.
    const fn over_buffers(load: LoadAction, store: StoreAction) -> Run {
        Run {
            layers: Layers::OverBuffers(load, store),
            ..Run::PLAIN
        }
    }

    fn label(self) -> String {
        let device = match (self.announced, self.checked) {
            (false, false) => "plain",
            (false, true) => "checked",
            (true, false) => "announced",
            (true, true) => "announced checked",
        };
        match self.layers {
            Layers::Private => device.to_owned(),
            Layers::OverBuffers(load, store) => {
                format!("{device}, layers over buffers, {load:?} load, {store:?} store")
            }
        }
    }
}

/// What the window layers are, and how a pass that is not resolved loads and stores its layer.
#[derive(Clone, Copy, Debug)]
enum Layers {
    /// Private textures, loaded and stored.
    Private,
    /// Textures over `Shared` buffers at tight rows, loaded and stored by these.
    OverBuffers(LoadAction, StoreAction),
}

/// The submissions the checks of a checked run saw: each the encoders it named, noted under a
/// lock, as the consumer's check notes each submission it lets run.
#[derive(Default)]
struct Checks {
    seen: Mutex<(u64, Range<usize>)>,
}

impl Checks {
    /// A command buffer's check: note the submission, and let it run.
    fn check(self: &Arc<Self>) -> SubmissionCheck {
        let checks = Arc::clone(self);
        Arc::new(move |submission| {
            let mut seen = checks.seen.lock().unwrap();
            seen.0 += 1;
            seen.1 = submission.encoders();
            None
        })
    }

    /// The checks run so far.
    fn count(&self) -> u64 {
        self.seen.lock().unwrap().0
    }
}

/// The pipelines of one raster sample count: the rectangle's, the mask's and the panel's.
struct Pipelines {
    rect: RenderPipelineState,
    masked: RenderPipelineState,
    panel: RenderPipelineState,
}

/// The device, its queue, and what the frames draw with.
struct Workload {
    device: Device,
    layers: Layers,
    /// The buffers the layers lie over, where they do.
    layer_buffers: Vec<Buffer>,
    queue: CommandQueue,
    /// A checked run's submissions checked.
    checks: Option<Arc<Checks>>,
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

/// A `TARGET_SIDE` square `BGRA8Unorm_sRGB` render target over a new `Shared` buffer at tight
/// rows, and the buffer.
fn target_over_buffer(device: &Device) -> (Texture, Buffer) {
    let rows = 4 * TARGET_SIDE;
    let buffer = device
        .new_buffer(rows * TARGET_SIDE, ResourceOptions::shared())
        .unwrap();
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::BGRA8UnormSRGB, TARGET_SIDE, TARGET_SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = buffer.resource_options();
    (buffer.new_texture(&descriptor, 0, rows).unwrap(), buffer)
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
    fn new(device: Device, checked: bool, layers: Layers) -> Workload {
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
        let (targets, layer_buffers) = match layers {
            Layers::Private => (
                (0..TARGETS).map(|_| target(&device, 1)).collect(),
                Vec::new(),
            ),
            Layers::OverBuffers(..) => (0..TARGETS).map(|_| target_over_buffer(&device)).unzip(),
        };
        Workload {
            layers,
            layer_buffers,
            queue: device.new_command_queue().unwrap(),
            checks: checked.then(Arc::default),
            pipelines,
            sampler,
            textures,
            mask: sampled(&device, [255, 255, 255, 128]),
            targets,
            multisampled: target(&device, 4),
            indices: buffer_with(&device, &indices).unwrap(),
            rects: buffer_with(&device, &rects).unwrap(),
            device,
        }
    }

    /// Pass `pass`'s descriptor: its layer loaded and stored (by the layers' actions where they
    /// lie over buffers), or, for a resolved pass, the 4-sample texture cleared and resolved into
    /// the layer.
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
            let (load, store) = match self.layers {
                Layers::Private => (LoadAction::Load, StoreAction::Store),
                Layers::OverBuffers(load, store) => (load, store),
            };
            color.texture = Some(layer);
            color.load_action = load;
            color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
            color.store_action = store;
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
        self.frame_reading(Reads::default())
    }

    /// [`frame`](Self::frame), reading `reads` around each step too (so its times are not the
    /// plain frame's).
    fn frame_reading(&self, reads: Reads) -> Frame {
        let mut frame = Frame::default();
        let mut last: Option<CommandBuffer> = None;
        let counts = || reads.work_counts.then(|| self.device.work_counts());
        let switches = thread_switches();
        let cpu = thread_cpu();
        let start = Instant::now();
        let mut begun = [Begun::default(); 4];
        for pass in 0..BUFFERS {
            let command_buffer = self.queue.command_buffer().unwrap();
            let descriptor = self.pass(pass);
            let before = counts();
            let (mut encoder, open) = step(reads.instructions, || {
                command_buffer.render_command_encoder(&descriptor).unwrap()
            });
            frame.open.push(open);
            if let Some(before) = &before {
                begun[0].add(Begun::since(&self.device, before));
            }
            for draw in 0..DRAWS {
                let before = counts();
                let ((), drawn) = step(reads.instructions, || self.draw(&mut encoder, pass, draw));
                frame.draws.push(drawn);
                if let Some(before) = &before {
                    begun[1].add(Begun::since(&self.device, before));
                }
            }
            let before = counts();
            let ((), ended) = step(reads.instructions, || encoder.end_encoding().unwrap());
            frame.end.push(ended);
            if let Some(before) = &before {
                begun[2].add(Begun::since(&self.device, before));
            }
            // The consumer makes each command buffer's check before its commit call.
            let check = self.checks.as_ref().map(Checks::check);
            let before = counts();
            let switches = thread_switches();
            let ((), committed) = step(reads.instructions, || {
                if let Some(check) = check {
                    command_buffer.set_submission_check(check).unwrap();
                }
                command_buffer.commit().unwrap()
            });
            if let Some((before, after)) = switches.zip(thread_switches()) {
                let blocked = after.since(before).voluntary;
                *frame.commit_blocks.get_or_insert(0) += blocked;
            }
            frame.commit.push(committed);
            if let Some(before) = &before {
                begun[3].add(Begun::since(&self.device, before));
            }
            last = Some(command_buffer);
        }
        frame.begun = begun;
        frame.wall = start.elapsed();
        frame.cpu = thread_cpu().saturating_sub(cpu);
        frame.switches = switches
            .zip(thread_switches())
            .map(|(before, after)| after.since(before));
        let last = last.unwrap();
        last.wait_until_completed_for(WAIT).unwrap();
        frame.done = start.elapsed();
        assert_eq!(last.status(), CommandBufferStatus::Completed);
        frame
    }
}

/// One step's wall time, and the encoding thread's allocations and user-space instructions in it
/// where they are counted.
#[derive(Clone, Copy)]
struct Step {
    wall: Duration,
    allocations: Option<u64>,
    instructions: Option<u64>,
}

/// Run `body` as one step, counting its instructions where `counted`.
fn step<R>(counted: bool, body: impl FnOnce() -> R) -> (R, Step) {
    let allocations = alloc::thread_allocations();
    let instructions = counted.then(instructions::read).flatten();
    let start = Instant::now();
    let result = body();
    let wall = start.elapsed();
    let instructions = instructions.map(|before| instructions::read().unwrap() - before);
    let allocations = allocations.map(|before| alloc::thread_allocations().unwrap() - before);
    (
        result,
        Step {
            wall,
            allocations,
            instructions,
        },
    )
}

/// The calling thread's count of the instructions it retired in user space, where the kernel lets
/// a thread count its own (Linux's `perf_event_open`, `kernel.perf_event_paranoid` at most 2):
/// steady where a step's time is not, so a change to the encode path shows in it run to run. On a
/// hybrid CPU (Intel's P and E cores) it counts on the P cores alone (`cpu_core`), where a run is
/// pinned (`taskset`).
#[cfg(target_os = "linux")]
mod instructions {
    use std::cell::Cell;

    /// `struct perf_event_attr` as its first published size (`PERF_ATTR_SIZE_VER0`).
    #[repr(C)]
    struct Attr {
        kind: u32,
        size: u32,
        config: u64,
        sample_period: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events: u32,
        bp_type: u32,
        config1: u64,
    }

    /// `PERF_TYPE_HARDWARE`, `PERF_COUNT_HW_INSTRUCTIONS`, and the `exclude_kernel` and
    /// `exclude_hv` flags.
    const HARDWARE: u32 = 0;
    const INSTRUCTIONS: u64 = 1;
    const EXCLUDE_KERNEL_HV: u64 = (1 << 5) | (1 << 6);

    thread_local! {
        /// The thread's counter: its descriptor, or -1 where none opened; opened at its first read.
        static COUNTER: Cell<Option<i32>> = const { Cell::new(None) };
    }

    /// The counter's descriptor for the calling thread, opened at its first call.
    fn counter() -> i32 {
        COUNTER.with(|cell| {
            if let Some(fd) = cell.get() {
                return fd;
            }
            // A hybrid CPU's P cores' PMU, in the config's upper half; none elsewhere.
            let pmu = std::fs::read_to_string("/sys/bus/event_source/devices/cpu_core/type")
                .ok()
                .and_then(|text| text.trim().parse::<u64>().ok())
                .unwrap_or(0);
            let attr = Attr {
                kind: HARDWARE,
                size: std::mem::size_of::<Attr>() as u32,
                config: INSTRUCTIONS | (pmu << 32),
                sample_period: 0,
                sample_type: 0,
                read_format: 0,
                flags: EXCLUDE_KERNEL_HV,
                wakeup_events: 0,
                bp_type: 0,
                config1: 0,
            };
            // SAFETY: `attr` is a valid `perf_event_attr` of its stated size, live for the call;
            // the calling thread (0), any CPU (-1), no group (-1), no flags.
            let fd = unsafe {
                libc::syscall(
                    libc::SYS_perf_event_open,
                    &attr as *const Attr,
                    0,
                    -1,
                    -1,
                    0,
                )
            };
            let fd = i32::try_from(fd).unwrap_or(-1);
            cell.set(Some(fd));
            fd
        })
    }

    /// The instructions the calling thread retired in user space since its counter opened.
    pub(super) fn read() -> Option<u64> {
        let fd = counter();
        if fd < 0 {
            return None;
        }
        let mut count = 0u64;
        // SAFETY: `count` is 8 writable bytes, the size of the counter's value.
        let read = unsafe { libc::read(fd, (&mut count as *mut u64).cast(), 8) };
        (read == 8).then_some(count)
    }
}

/// Elsewhere none is counted.
#[cfg(not(target_os = "linux"))]
mod instructions {
    pub(super) fn read() -> Option<u64> {
        None
    }
}

/// What a frame reads around each step beside its wall time ([`Workload::frame_reading`]): none
/// in a timed frame, as each read costs the encoding thread.
#[derive(Clone, Copy, Default)]
struct Reads {
    /// The device's work counts: the command buffers each step began ([`Frame::begun`]).
    work_counts: bool,
    /// The encoding thread's user-space instructions in each ([`Step::instructions`]).
    instructions: bool,
}

/// The command buffers the device began, and of those the barrier command buffers it recorded to
/// keep, in a step a frame that read the work counts ran ([`Workload::frame_reading`]).
#[derive(Clone, Copy, Default)]
struct Begun {
    all: u64,
    kept: u64,
}

impl Begun {
    /// What `device` began since `before`, its work counts then.
    fn since(device: &Device, before: &WorkCounts) -> Begun {
        let now = device.work_counts();
        Begun {
            all: now.command_buffers_begun - before.command_buffers_begun,
            kept: now.barrier_buffers_kept - before.barrier_buffers_kept,
        }
    }

    fn add(&mut self, other: Begun) {
        self.all += other.all;
        self.kept += other.kept;
    }
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
    /// Through the last command buffer's completion.
    done: Duration,
    /// The command buffers its passes' openings, their draws, their ends and its commits began,
    /// in that order, where it read the work counts ([`Workload::frame_reading`]); none
    /// elsewhere.
    begun: [Begun; 4],
    /// The encoding thread's context switches over the same span as `cpu`, where they are read.
    switches: Option<Switches>,
    /// Its voluntary ones inside its commits.
    commit_blocks: Option<u64>,
}

/// A thread's context switches: the voluntary ones (it blocked: on a lock, a condition variable, a
/// sleep, I/O) and the involuntary ones (it was preempted).
#[derive(Clone, Copy, Default)]
struct Switches {
    voluntary: u64,
    involuntary: u64,
}

impl Switches {
    /// The switches from `before` to these.
    fn since(self, before: Switches) -> Switches {
        Switches {
            voluntary: self.voluntary.saturating_sub(before.voluntary),
            involuntary: self.involuntary.saturating_sub(before.involuntary),
        }
    }

    /// A thread's switches so far, from its `status` text.
    fn of_status(status: &str) -> Option<Switches> {
        let field = |name: &str| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(name)?.trim().parse().ok())
        };
        Some(Switches {
            voluntary: field("voluntary_ctxt_switches:")?,
            involuntary: field("nonvoluntary_ctxt_switches:")?,
        })
    }
}

/// The calling thread's context switches so far ([`Switches`]), from `getrusage`: a system call,
/// which a frame makes twice a commit, outside its times. `None` off Linux.
#[cfg(target_os = "linux")]
fn thread_switches() -> Option<Switches> {
    // SAFETY: `rusage` is plain integers, for which zeros are a value.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `getrusage` writes the struct it is given and has no other effect.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) };
    (rc == 0).then(|| Switches {
        voluntary: u64::try_from(usage.ru_nvcsw).unwrap_or(0),
        involuntary: u64::try_from(usage.ru_nivcsw).unwrap_or(0),
    })
}

/// The calling thread's context switches so far: not read off Linux.
#[cfg(not(target_os = "linux"))]
fn thread_switches() -> Option<Switches> {
    None
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

/// A thread's CPU time: all of it, from its `schedstat` (nanoseconds), and the user-space part,
/// from its `stat` (`utime`, in clock ticks: 10 ms on a common kernel, a coarse share); and how
/// often it slept, its voluntary context switches, from its `status`.
#[derive(Clone, Copy, Default)]
struct Cpu {
    all: Duration,
    user: Duration,
    sleeps: u64,
}

/// The process's other threads' CPU time so far ([`Cpu`]), each by its name (cut to 15 bytes, as
/// Linux keeps it): every thread but the calling one. `None` off Linux.
fn other_threads_cpu() -> Option<BTreeMap<String, Cpu>> {
    // `<pid>/task/<tid>`: the calling thread's own.
    let own = std::fs::read_link("/proc/thread-self").ok()?;
    // SAFETY: `sysconf` reads a constant of the system and has no other effect.
    let ticks = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    let tick = Duration::from_secs(1) / u32::try_from(ticks).ok().filter(|&t| t > 0)?;
    let mut threads = BTreeMap::new();
    for task in std::fs::read_dir("/proc/self/task").ok()? {
        let task = task.ok()?.path();
        if task.file_name() == own.file_name() {
            continue;
        }
        // A thread that ended meanwhile is left out.
        let (Ok(name), Ok(schedstat), Ok(stat), Ok(status)) = (
            std::fs::read_to_string(task.join("comm")),
            std::fs::read_to_string(task.join("schedstat")),
            std::fs::read_to_string(task.join("stat")),
            std::fs::read_to_string(task.join("status")),
        ) else {
            continue;
        };
        let nanos: u64 = schedstat.split_whitespace().next()?.parse().ok()?;
        // The fields after the name's closing parenthesis start at the third, the state;
        // `utime` is the fourteenth.
        let utime: u32 = stat
            .rsplit_once(')')?
            .1
            .split_whitespace()
            .nth(11)?
            .parse()
            .ok()?;
        let sleeps = Switches::of_status(&status)?.voluntary;
        let cpu: &mut Cpu = threads.entry(name.trim().to_owned()).or_default();
        cpu.all += Duration::from_nanos(nanos);
        cpu.user += tick * utime;
        cpu.sleeps += sleeps;
    }
    Some(threads)
}

/// The groups of the process's other threads [`other_threads_cpu`] reports: the runtime's by the
/// prefix of their names, and the rest.
const THREAD_GROUPS: [(&str, &str); 4] = [
    ("scheduler", "metal2vulkan-sc"),
    ("completion", "metal2vulkan-co"),
    ("queue waits", "metal2vulkan-qu"),
    ("handlers", "metal2vulkan-ha"),
];

/// What the other threads spent between `before` and `after` ([`other_threads_cpu`]), by
/// [`THREAD_GROUPS`] and the rest, `other`: a thread that ended meanwhile is left out, one that
/// started counts from 0.
fn spent_by_group(
    before: &BTreeMap<String, Cpu>,
    after: &BTreeMap<String, Cpu>,
) -> Vec<(&'static str, Cpu)> {
    let mut groups: Vec<(&'static str, Cpu)> = THREAD_GROUPS
        .iter()
        .map(|(group, _)| (*group, Cpu::default()))
        .chain([("other", Cpu::default())])
        .collect();
    for (name, now) in after {
        let then = before.get(name).copied().unwrap_or_default();
        let at = THREAD_GROUPS
            .iter()
            .position(|(_, prefix)| name.starts_with(prefix))
            .unwrap_or(THREAD_GROUPS.len());
        groups[at].1.all += now.all.saturating_sub(then.all);
        groups[at].1.user += now.user.saturating_sub(then.user);
        groups[at].1.sleeps += now.sleeps.saturating_sub(then.sleeps);
    }
    groups
}

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

/// The frame's costs, printed (the module docs): a run of each of `runs` in turn, each on a device
/// of its own.
fn measure(runs: &[Run]) {
    for &run in runs {
        let cache = ScratchCache::new().unwrap();
        let heard = Heard::default();
        let work = Workload::new(
            device(&cache, run.announced, &heard),
            run.checked,
            run.layers,
        );
        let label = run.label();
        let name = work.device.name().to_owned();
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let first = work.frame();
        let [commit, _] = percentiles(
            first
                .commit
                .iter()
                .map(|s| s.wall.as_secs_f64() * 1e6)
                .collect(),
        );
        let longest = first
            .commit
            .iter()
            .map(|s| s.wall)
            .max()
            .unwrap_or_default();
        eprintln!(
            "FRAME-COST {name} {label}: the first frame: the encoding thread's CPU {:.2} ms, wall \
             {:.2} ms, {}; its commits {commit:.2} us (longest {:.2})",
            ms(first.cpu),
            ms(first.wall),
            blocks([&first]),
            longest.as_secs_f64() * 1e6,
        );
        for _ in 1..WARM {
            work.frame();
        }
        let notices_before: u64 = heard.lock().unwrap().values().map(|(n, _)| n).sum();
        let counts_before = work.device.work_counts();
        let checks_before = work.checks.as_ref().map_or(0, |checks| checks.count());
        let threads_before = other_threads_cpu();
        let allocations_before = others_allocations();
        let frames: Vec<Frame> = (0..TIMED).map(|_| work.frame()).collect();
        let allocations = others_allocations();
        let threads = other_threads_cpu();
        let checks = work.checks.as_ref().map_or(0, |checks| checks.count());
        let counts = work.device.work_counts();
        let notices: u64 = heard.lock().unwrap().values().map(|(n, _)| n).sum();
        let reads = Reads {
            instructions: true,
            ..Reads::default()
        };
        let counted: Vec<Frame> = (0..COUNTED).map(|_| work.frame_reading(reads)).collect();
        let [cpu, cpu_high] = percentiles(frames.iter().map(|f| ms(f.cpu)).collect());
        let [wall, wall_high] = percentiles(frames.iter().map(|f| ms(f.wall)).collect());
        let [done, done_high] = percentiles(frames.iter().map(|f| ms(f.done)).collect());
        eprintln!(
            "FRAME-COST {name} {label}: {BUFFERS} command buffers of one pass of {DRAWS} indexed \
             draws a frame, every {RESOLVED}th pass resolved, {WARM} untimed and {TIMED} timed \
             frames: the encoding thread's CPU {cpu:.2} ms a frame (p90 {cpu_high:.2}), wall \
             {wall:.2} ms (p90 {wall_high:.2}), through completion {done:.2} ms (p90 \
             {done_high:.2}), {} a frame; {:.1} us of CPU a draw over the frame; {:.1} notices \
             a frame",
            blocks(&frames),
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
            let read: Vec<Step> = counted.iter().flat_map(|f| of(f).iter().copied()).collect();
            let instructions = match read.iter().map(|s| s.instructions).sum::<Option<u64>>() {
                Some(total) => {
                    let [p50, _] = percentiles(
                        read.iter()
                            .filter_map(|s| s.instructions)
                            .map(|n| n as f64)
                            .collect(),
                    );
                    format!(
                        "{:.0} user-space instructions each (p50 {p50:.0})",
                        total as f64 / read.len() as f64
                    )
                }
                None => "instructions not counted".to_owned(),
            };
            eprintln!(
                "FRAME-COST {name} {label}: {step}: {p50:.2} us (p90 {p90:.2}); {allocations}; \
                 {instructions}"
            );
        }
        let instructions: Option<u64> = counted
            .iter()
            .flat_map(|f| f.open.iter().chain(&f.draws).chain(&f.end).chain(&f.commit))
            .map(|s| s.instructions)
            .sum();
        if let Some(total) = instructions {
            eprintln!(
                "FRAME-COST {name} {label}: the encoding thread's user-space instructions in a \
                 frame's steps, over {COUNTED} frames of their own: {:.0} a frame, {:.0} a pass",
                total as f64 / COUNTED as f64,
                total as f64 / (COUNTED * BUFFERS) as f64,
            );
        }
        let per_frame = |count: fn(&WorkCounts) -> u64| {
            (count(&counts) - count(&counts_before)) as f64 / TIMED as f64
        };
        eprintln!(
            "FRAME-COST {name} {label}: a frame's work: {:.0} bind shapes made, {:.0} draws \
             settled, {:.0} pipelines looked up, {:.0} draw states recorded, {:.0} argument \
             tables retained, {:.1} command pools made, {:.1} timeline waits, {:.1} addressable \
             buffers made, {:.1} queue submissions, {:.1} commits handed off, {:.1} submission \
             checks",
            per_frame(|c| c.draw_shapes_made),
            per_frame(|c| c.draws_settled),
            per_frame(|c| c.graphics_pipelines_looked_up),
            per_frame(|c| c.draw_states_recorded),
            per_frame(|c| c.argument_tables_retained),
            per_frame(|c| c.command_pools_made),
            per_frame(|c| c.timeline_waits),
            per_frame(|c| c.addressable_buffers_made),
            per_frame(|c| c.queue_submissions),
            per_frame(|c| c.commits_handed_off),
            (checks - checks_before) as f64 / TIMED as f64,
        );
        eprintln!(
            "FRAME-COST {name} {label}: a frame's command buffers: {:.1} begun ({:.2} a pass), \
             {:.1} of them barrier command buffers recorded to keep; {:.1} submitted ({:.2} a \
             pass) in {:.1} vkQueueSubmit calls",
            per_frame(|c| c.command_buffers_begun),
            per_frame(|c| c.command_buffers_begun) / BUFFERS as f64,
            per_frame(|c| c.barrier_buffers_kept),
            per_frame(|c| c.command_buffers_submitted),
            per_frame(|c| c.command_buffers_submitted) / BUFFERS as f64,
            per_frame(|c| c.queue_submissions),
        );
        let buffers = (TIMED * BUFFERS) as f64;
        let threads = match (&threads_before, &threads) {
            (Some(before), Some(after)) => spent_by_group(before, after)
                .iter()
                .map(|(group, spent)| {
                    let us = |d: Duration| d.as_secs_f64() * 1e6 / buffers;
                    format!(
                        "{group} {:.1} us ({:.1} user, {:.2} sleeps)",
                        us(spent.all),
                        us(spent.user),
                        spent.sleeps as f64 / buffers
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
            _ => "not measured".to_owned(),
        };
        let allocations = match (allocations_before, allocations) {
            (Some(before), Some(after)) => {
                format!("{:.2} allocations", (after - before) as f64 / buffers)
            }
            _ => "allocations not counted".to_owned(),
        };
        eprintln!(
            "FRAME-COST {name} {label}: the other threads a command buffer: {threads}; \
             {allocations}"
        );
        if run.announced || work.layer_buffers.len() == TARGETS {
            for (kind, (count, _)) in heard.lock().unwrap().iter() {
                eprintln!(
                    "FRAME-COST {name} {label}: heard {} {count} times",
                    kind.code()
                );
            }
        }
    }
}

/// How often the encoding thread blocked (in its commits too) and was preempted in `frames`, on
/// average, where it was read.
fn blocks<'a>(frames: impl IntoIterator<Item = &'a Frame>) -> String {
    let (mut count, mut total, mut in_commits) = (0u64, Switches::default(), 0);
    for frame in frames {
        let (Some(switches), Some(commits)) = (frame.switches, frame.commit_blocks) else {
            return "blocks not read".to_owned();
        };
        count += 1;
        total.voluntary += switches.voluntary;
        total.involuntary += switches.involuntary;
        in_commits += commits;
    }
    let each = |n: u64| n as f64 / count.max(1) as f64;
    format!(
        "blocked {:.2} times ({:.2} in commits), preempted {:.2}",
        each(total.voluntary),
        each(in_commits),
        each(total.involuntary)
    )
}

/// The allocations the process's threads but this one made so far, where they are counted.
fn others_allocations() -> Option<u64> {
    Some(alloc::process_allocations()? - alloc::thread_allocations()?)
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs() {
    measure(&[Run::PLAIN, Run::CHECKED, Run::ANNOUNCED]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_plain() {
    measure(&[Run::PLAIN]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_checked() {
    measure(&[Run::CHECKED]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_announced() {
    measure(&[Run::ANNOUNCED]);
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn frame_costs_over_buffers() {
    measure(&[
        Run::over_buffers(LoadAction::Load, StoreAction::Store),
        Run::over_buffers(LoadAction::Load, StoreAction::DontCare),
        Run::over_buffers(LoadAction::Clear, StoreAction::DontCare),
    ]);
}

/// One frame of a plain device, once every pipeline is made and every pool piece done, its
/// allocations counted; and the device's name.
#[cfg(not(m2v_system_alloc))]
fn counted() -> (String, Frame) {
    let cache = ScratchCache::new().unwrap();
    let heard = Heard::default();
    let work = Workload::new(device(&cache, false, &heard), false, Layers::Private);
    for _ in 0..WARM {
        work.frame();
    }
    let frame = work.frame();
    (work.device.name().to_string(), frame)
}

/// The allocations each step of `steps` of `frame` made on the encoding thread, on average.
#[cfg(not(m2v_system_alloc))]
fn allocations_each(frame: &Frame, steps: StepsOf) -> f64 {
    let steps = steps(frame);
    let allocations: u64 = steps.iter().map(|s| s.allocations.unwrap()).sum();
    allocations as f64 / steps.len() as f64
}

/// A frame's draws allocate at most [`DRAW_ALLOCATIONS`] each on average on the encoding thread,
/// once every pipeline is made: what the draw path keeps of what it found before (each texture's
/// storage view formats, the pipeline state's routing into the pass's colors, the settles of its
/// latest bind shapes and the shapes themselves, the format table) is not made again at each
/// draw, and what a draw retains, binds and records it holds and builds without allocating where
/// the object or the list allows. Its passes' ends allocate at most [`PASS_END_ALLOCATIONS`] each:
/// a pass end retains what its commands use as one object, and makes no tile grid where a draw
/// reaches every tile, no sample planes' steps in a pass without planes, and no scrub's list for a
/// texture that is not sparse.
#[cfg(not(m2v_system_alloc))]
#[test]
fn a_frame_s_draws_and_pass_ends_allocate_within_their_bounds() {
    let (device, frame) = counted();
    let each = allocations_each(&frame, |f| &f.draws);
    eprintln!("FRAME-COST {device}: {each:.2} allocations a draw, at most {DRAW_ALLOCATIONS} kept");
    assert!(
        each <= DRAW_ALLOCATIONS,
        "{each:.2} allocations a draw, more than {DRAW_ALLOCATIONS}"
    );
    let each = allocations_each(&frame, |f| &f.end);
    eprintln!(
        "FRAME-COST {device}: {each:.2} allocations a pass end, at most {PASS_END_ALLOCATIONS} kept"
    );
    assert!(
        each <= PASS_END_ALLOCATIONS,
        "{each:.2} allocations a pass end, more than {PASS_END_ALLOCATIONS}"
    );
}

/// A frame's passes each begin at most [`PASS_BEGINS`] command buffers on the encoding thread,
/// from the opening through the end, once every pipeline is made; and its commits begin none but
/// the barrier command buffers they record to keep (`command_buffer/kept_barriers.rs`), where the
/// device keeps them: every device whose driver reads no command buffer after its timeline's
/// signal (`RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal`), where a commit records
/// its batch's two barrier command buffers, before its pass and after it.
#[test]
fn a_frame_s_passes_and_commits_begin_their_command_buffers_within_their_bound() {
    let cache = ScratchCache::new().unwrap();
    let heard = Heard::default();
    let work = Workload::new(device(&cache, false, &heard), false, Layers::Private);
    // Every pipeline made, and every pool piece done, before the frame that counts.
    for _ in 0..WARM {
        work.frame();
    }
    let frame = work.frame_reading(Reads {
        work_counts: true,
        ..Reads::default()
    });
    let [open, draws, end, commit] = frame.begun;
    let passes = BUFFERS as u64;
    let late_reads = (work.device.properties().driver_defects)
        .contains(&RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal);
    eprintln!(
        "FRAME-COST {}: command buffers begun a pass: {:.2} opening, {:.2} drawing, {:.2} ending, \
         {:.2} committing ({:.2} of them barrier command buffers recorded to keep), the bound \
         {PASS_BEGINS} a pass; the driver reads command buffers after their signal: {late_reads}",
        work.device.name(),
        open.all as f64 / passes as f64,
        draws.all as f64 / passes as f64,
        end.all as f64 / passes as f64,
        commit.all as f64 / passes as f64,
        commit.kept as f64 / passes as f64,
    );
    let pass = open.all + draws.all + end.all;
    assert!(
        pass <= PASS_BEGINS * passes,
        "{pass} command buffers begun by {passes} passes, more than {PASS_BEGINS} each"
    );
    let own = commit.all - commit.kept;
    match late_reads {
        true => assert_eq!(own, 2 * passes, "a commit's two barrier command buffers"),
        false => assert_eq!(
            own, 0,
            "{own} command buffers begun by {passes} commits, not kept"
        ),
    }
}

/// A frame's pass opens (`render_command_encoder`) allocate at most [`OPEN_ALLOCATIONS`] each on
/// average on the encoding thread, once every pipeline is made: an encoder takes the compatible
/// render pass the device keeps for its shape instead of making one.
#[cfg(not(m2v_system_alloc))]
#[test]
fn a_frame_s_pass_opens_allocate_within_their_bound() {
    let (device, frame) = counted();
    let each = allocations_each(&frame, |f| &f.open);
    eprintln!(
        "FRAME-COST {device}: {each:.2} allocations a pass open, at most {OPEN_ALLOCATIONS} kept"
    );
    assert!(
        each <= OPEN_ALLOCATIONS,
        "{each:.2} allocations a pass open, more than {OPEN_ALLOCATIONS}"
    );
}

/// A checked frame's commits allocate at most [`COMMIT_ALLOCATIONS`] each on average on the
/// encoding thread, and the process's other threads at most [`SUBMIT_ALLOCATIONS`] per command
/// buffer (its check, its batch's planning and submission, its completion), once every pipeline is
/// made. The other threads' count is the process's: the test needs a process of its own, as
/// nextest runs each test in.
#[cfg(not(m2v_system_alloc))]
#[test]
fn a_frame_s_commits_allocate_within_their_bound() {
    let cache = ScratchCache::new().unwrap();
    let heard = Heard::default();
    let work = Workload::new(device(&cache, false, &heard), true, Layers::Private);
    // Every pipeline made, and every pool piece done, before the frames that count.
    for _ in 0..WARM {
        work.frame();
    }
    const FRAMES: usize = 4;
    let before = others_allocations().unwrap();
    let frames: Vec<Frame> = (0..FRAMES).map(|_| work.frame()).collect();
    let others = others_allocations().unwrap() - before;
    let commits: u64 = (frames.iter().flat_map(|frame| &frame.commit))
        .map(|s| s.allocations.unwrap())
        .sum();
    let buffers = (FRAMES * BUFFERS) as f64;
    let (each, other) = (commits as f64 / buffers, others as f64 / buffers);
    eprintln!(
        "FRAME-COST {}: {each:.2} allocations a commit, at most {COMMIT_ALLOCATIONS} kept; \
         {other:.2} on the other threads a command buffer, at most {SUBMIT_ALLOCATIONS} kept",
        work.device.name()
    );
    assert!(
        each <= COMMIT_ALLOCATIONS,
        "{each:.2} allocations a commit, more than {COMMIT_ALLOCATIONS}"
    );
    assert!(
        other <= SUBMIT_ALLOCATIONS,
        "{other:.2} allocations on the other threads a command buffer, more than \
         {SUBMIT_ALLOCATIONS}"
    );
}
