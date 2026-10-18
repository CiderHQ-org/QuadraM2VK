//! What capture replay (`render/capture.rs`) costs a pass, on the device the cfg selects, without
//! the validation layer.
//!
//! Each shape is a pass that loads and stores a `BGR10_XR` color attachment, under a pipeline of
//! the owned fixture `rch_vertex_math` (identity matrices: each position is the buffer's) whose
//! color's write mask is none of its channels, in two modes:
//!
//! - **exact**: the shape's draws, then a triangle off the target. Its capture replay is ready
//!   (the shape runs until a pass announces nothing), so each pass draws its captures and bins
//!   their primitives before it, and its load rewrites the tiles they reach.
//! - **every tile**: the shape's draws, then a line off the target, which no capture replays: the
//!   pass rewrites every tile its scissors reach and announces `unreached-tiles-rewritten`.
//!
//! The shapes:
//!
//! - `cells`: a 384x256 target and two scissored cells (0, 0, 166, 166 and 166, 0, 166, 166) of
//!   four small rects each, each rect an indexed (`uint16`) triangle list of 6 indices over its 4
//!   vertices: an atlas pass of eight draws.
//! - `heavy`: a 1024x1024 target and four draws of 3 x 2^14 vertices (3 x 2^10 on a CPU ICD) of
//!   zero-area triangles, as the `vertex_cost` lane spreads them: the capture shades each vertex
//!   again.
//!
//! Each sample is one command buffer of [`PASSES`] passes of one mode, the modes alternating, 20
//! samples a mode after 3 warm ones (`CAPTURE-COST`, a pass's share of each): the host's time to
//! encode it, the command buffer's GPU time (MoltenVK's GPU times of a render pass are not the
//! pass's), and the host's time from the commit to the completion; medians and
//! fastest. Measures; asserts only that each mode took its path.

use super::degrade_notices::counts;
use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{ColorWriteMask, IndexType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, CommandQueue, Degradation, Device, LoadAction, PipelineOptions, RenderCommandEncoder,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, StoreAction,
    TextureDescriptor,
};
use std::time::{Duration, Instant};

/// The passes of each sample's command buffer.
const PASSES: usize = 16;

/// The kind a pass announces where its draws' tile reach is not captured.
const UNREACHED_TILES: Degradation = Degradation::UnreachedTilesRewritten;

/// The vertices of each `heavy` draw: 2^14 triangles on a GPU.
#[cfg(m2v_gpu)]
const HEAVY_VERTICES: u32 = 3 << 14;

/// The vertices of each `heavy` draw on a CPU ICD, which shades them on the CPU: 2^10 triangles.
#[cfg(not(m2v_gpu))]
const HEAVY_VERTICES: u32 = 3 << 10;

/// The `cells` shape's scissored cells.
const CELLS: [ScissorRect; 2] = [
    ScissorRect {
        x: 0,
        y: 0,
        width: 166,
        height: 166,
    },
    ScissorRect {
        x: 166,
        y: 0,
        width: 166,
        height: 166,
    },
];

/// A pass's tail draw: a triangle off the target (exact), or a line off it (every tile).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Exact,
    EveryTile,
}

/// One shape's target side, positions, indices, and draws.
struct Shape {
    name: &'static str,
    extent: [u64; 2],
    positions: Vec<[f32; 4]>,
    /// The `cells` rects' indices (a list of two triangles over 4 vertices), where it draws them.
    indices: Option<Vec<u16>>,
}

/// The off-target vertices each shape's positions end in: a triangle's three, then a line's two.
const OFF: [[f32; 4]; 5] = [[-2.0, -2.0, 0.5, 1.0]; 5];

/// The clip-space position of window pixel `(x, y)` of a target of `extent` under Metal's default
/// viewport.
fn clip([w, h]: [u64; 2], x: f32, y: f32) -> [f32; 4] {
    [2.0 * x / w as f32 - 1.0, 1.0 - 2.0 * y / h as f32, 0.5, 1.0]
}

/// The `cells` shape: four 24x24 rects in each cell, each from its own 4 vertices.
fn cells() -> Shape {
    let extent = [384, 256];
    let mut positions = Vec::new();
    for cell in CELLS {
        for r in 0..4u64 {
            let x = (cell.x + 12 + 70 * (r % 2)) as f32;
            let y = (cell.y + 12 + 70 * (r / 2)) as f32;
            for (dx, dy) in [(0.0, 0.0), (24.0, 0.0), (0.0, 24.0), (24.0, 24.0)] {
                positions.push(clip(extent, x + dx, y + dy));
            }
        }
    }
    positions.extend(OFF);
    Shape {
        name: "cells",
        extent,
        positions,
        indices: Some(vec![0, 1, 2, 2, 1, 3]),
    }
}

/// The `heavy` shape: triangle `k`'s three vertices at one point of the target, spread by a
/// multiplicative hash of `k`.
fn heavy() -> Shape {
    let mut positions = Vec::with_capacity(HEAVY_VERTICES as usize + OFF.len());
    for k in 0..HEAVY_VERTICES / 3 {
        let h = k.wrapping_mul(2654435761);
        let x = f32::from(h as u16) / 32767.5 - 1.0;
        let y = f32::from((h >> 16) as u16) / 32767.5 - 1.0;
        positions.extend([[x, y, 0.5, 1.0]; 3]);
    }
    positions.extend(OFF);
    Shape {
        name: "heavy",
        extent: [1024, 1024],
        positions,
        indices: None,
    }
}

/// The fixture's uniforms: identity model, view and projection matrices.
fn identity() -> Vec<u8> {
    let one = |i: usize| (0..4).map(move |j| if i == j { 1.0f32 } else { 0.0 });
    (0..3)
        .flat_map(|_| (0..4).flat_map(one))
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// A shape's objects on one device.
struct Bench {
    queue: CommandQueue,
    state: RenderPipelineState,
    positions: Buffer,
    uniforms: Buffer,
    indices: Option<Buffer>,
    pass: RenderPassDescriptor,
    shape: Shape,
}

impl Bench {
    fn new(device: &Device, state: &RenderPipelineState, shape: Shape) -> Bench {
        let bytes: Vec<u8> = shape
            .positions
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let indices = shape.indices.as_ref().map(|indices| {
            let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
            buffer_with(device, &bytes).unwrap()
        });
        let [w, h] = shape.extent;
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::BGR10XR, w, h, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::private();
        let target = device.new_texture(&descriptor).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target);
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        Bench {
            queue: device.new_command_queue().unwrap(),
            state: state.clone(),
            positions: buffer_with(device, &bytes).unwrap(),
            uniforms: buffer_with(device, &identity()).unwrap(),
            indices,
            pass,
            shape,
        }
    }

    /// The draws of one pass in `mode`.
    fn draws(&self, e: &mut RenderCommandEncoder, mode: Mode) {
        e.set_render_pipeline_state(&self.state).unwrap();
        e.set_vertex_buffer(Some(&self.positions), 0, 0).unwrap();
        e.set_vertex_buffer(Some(&self.uniforms), 0, 1).unwrap();
        match &self.indices {
            Some(indices) => {
                for (c, cell) in CELLS.into_iter().enumerate() {
                    e.set_scissor_rect(cell).unwrap();
                    for r in 0..4 {
                        let base = 4 * (4 * c + r) as i32;
                        e.draw_indexed_primitives(
                            PrimitiveType::Triangle,
                            6,
                            IndexType::UInt16,
                            indices,
                            0,
                            1,
                            base,
                            0,
                        )
                        .unwrap();
                    }
                }
            }
            None => {
                for _ in 0..4 {
                    e.draw_primitives(PrimitiveType::Triangle, 0, HEAVY_VERTICES, 1, 0)
                        .unwrap();
                }
            }
        }
        let off = (self.shape.positions.len() - OFF.len()) as u32;
        match mode {
            Mode::Exact => e.draw_primitives(PrimitiveType::Triangle, off, 3, 1, 0),
            Mode::EveryTile => e.draw_primitives(PrimitiveType::Line, off + 3, 2, 1, 0),
        }
        .unwrap();
    }

    /// One command buffer of `passes` passes in `mode`: the host's encode time, the GPU time and
    /// the host's time from commit to completion, in milliseconds a pass.
    fn sample(&self, mode: Mode, passes: usize) -> [f64; 3] {
        let began = Instant::now();
        let command_buffer = self.queue.command_buffer().unwrap();
        for _ in 0..passes {
            let mut e = command_buffer.render_command_encoder(&self.pass).unwrap();
            self.draws(&mut e, mode);
            e.end_encoding().unwrap();
        }
        let encoded = began.elapsed().as_secs_f64();
        let committed = Instant::now();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let wall = committed.elapsed().as_secs_f64();
        assert_eq!(command_buffer.error(), None, "{} {mode:?}", self.shape.name);
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        let per = |s: f64| s * 1e3 / passes as f64;
        [per(encoded), per(gpu), per(wall)]
    }
}

/// Each shape's passes in both modes (the module docs).
#[test]
fn capture_replay_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        options.announce_each = [UNREACHED_TILES].into_iter().collect();
    })
    .unwrap();
    let bitcode = llvm_assemble(&engine_dir().join("rch_vertex_math.ll")).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut state = RenderPipelineDescriptor::new(function("rch_vertex_math"));
    state.fragment = Some(function("rch_vertex_math_color"));
    let mut none = ColorAttachmentDescriptor::new(PixelFormat::BGR10XR);
    none.write_mask = ColorWriteMask::NONE;
    state.color_attachments[0] = Some(none);
    let state = device
        .new_render_pipeline_state(&state, &PipelineOptions::default())
        .unwrap();
    let announced = || counts(&messages.take_notices()).contains_key(&UNREACHED_TILES);
    for shape in [cells(), heavy()] {
        let bench = Bench::new(&device, &state, shape);
        let name = bench.shape.name;
        // Until the shape's capture forms and pipelines are made.
        announced();
        let began = Instant::now();
        loop {
            bench.sample(Mode::Exact, 1);
            if !announced() {
                break;
            }
            assert!(began.elapsed() < WAIT, "{name}: no capture within {WAIT:?}");
            std::thread::sleep(Duration::from_millis(20));
        }
        let mut times = [Vec::new(), Vec::new()];
        let tallied = device.work_counts();
        for k in 0..23 {
            for (m, mode) in [Mode::Exact, Mode::EveryTile].into_iter().enumerate() {
                let before = device.work_counts().capture_reach_passes;
                let sample = bench.sample(mode, PASSES);
                let captured = device.work_counts().capture_reach_passes - before;
                let every = announced();
                match mode {
                    Mode::Exact => assert!(captured == PASSES as u64 && !every, "{name} {k}"),
                    Mode::EveryTile => assert!(captured == 0 && every, "{name} {k}"),
                }
                if k >= 3 {
                    times[m].push(sample);
                }
            }
        }
        let after = device.work_counts();
        let passes = after.capture_reach_passes - tallied.capture_reach_passes;
        eprintln!(
            "CAPTURE-TALLY {name}: {passes} captured passes found {} of the {} tiles their \
             scissors bound; {} at their bound, {} narrower",
            after.capture_reach_tiles_found - tallied.capture_reach_tiles_found,
            after.capture_reach_tiles_bound - tallied.capture_reach_tiles_bound,
            after.capture_reach_bound_passes - tallied.capture_reach_bound_passes,
            after.capture_reach_narrower_passes - tallied.capture_reach_narrower_passes,
        );
        let stat = |samples: &[[f64; 3]], i: usize| {
            let mut v: Vec<f64> = samples.iter().map(|s| s[i]).collect();
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[0])
        };
        for (m, mode) in ["exact", "every-tile"].into_iter().enumerate() {
            let (encode, encode_fast) = stat(&times[m], 0);
            let (gpu, gpu_fast) = stat(&times[m], 1);
            let (wall, wall_fast) = stat(&times[m], 2);
            eprintln!(
                "CAPTURE-COST {name} {mode} a pass ({PASSES} passes a command buffer): encode \
                 median {encode:.4} ms, fastest {encode_fast:.4} ms; GPU median {gpu:.4} ms, \
                 fastest {gpu_fast:.4} ms; commit to completion median {wall:.4} ms, fastest \
                 {wall_fast:.4} ms"
            );
        }
    }
    messages.take_notices();
}
