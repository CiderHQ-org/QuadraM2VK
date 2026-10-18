//! The GPU cost of a vertex-heavy draw (what an invariant position costs), on the device the cfg
//! selects, without the validation layer.
//!
//! The owned fixture `rch_vertex_math`'s vertex function (a position from a buffer at its vertex
//! id through a model, a view and a projection matrix, a normalized normal, a texture
//! coordinate) draws 3 x 2^20 vertices (3 x 2^14 on a CPU ICD) of zero-area triangles, each
//! triangle's three vertices one point, four draws to a pass: every vertex is shaded and no
//! fragment runs. Each pass is one command buffer, timed by its GPU start and end over 20 passes
//! after 3 warm ones (`VERTEX-COST`: the median and the fastest, and the median per vertex), and
//! by the host's time from its commit to its completion (MoltenVK's GPU times of a render pass
//! measured as about 11 us for these passes on an Apple M4: not the pass's).
//! Measures; asserts only that each pass completes.

use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    ClearColor, LoadAction, PipelineOptions, RenderPassDescriptor, ResourceOptions, StoreAction,
    TextureDescriptor,
};
use std::time::Instant;

/// The vertices of each draw: 2^20 triangles on a GPU.
#[cfg(m2v_gpu)]
const VERTICES: u32 = 3 << 20;

/// The vertices of each draw on a CPU ICD, which shades them on the CPU: 2^14 triangles.
#[cfg(not(m2v_gpu))]
const VERTICES: u32 = 3 << 14;

/// The draws of each pass.
const DRAWS: u32 = 4;

/// A column-major `float4x4`'s bytes.
fn matrix(columns: [[f32; 4]; 4]) -> Vec<u8> {
    columns
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// The fixture's uniforms: a model turning 0.3 radians about z and scaling by 0.9, a view 3
/// units back, and a 60-degree perspective projection of aspect 1 from 0.1 to 100.
fn uniforms() -> Vec<u8> {
    let (s, c) = 0.3f32.sin_cos();
    let model = [
        [0.9 * c, 0.9 * s, 0.0, 0.0],
        [-0.9 * s, 0.9 * c, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -3.0, 1.0],
    ];
    let f = 1.0 / (std::f32::consts::FRAC_PI_6).tan();
    let (near, far) = (0.1f32, 100.0f32);
    let projection = [
        [f, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ];
    [model, view, projection]
        .into_iter()
        .flat_map(matrix)
        .collect()
}

/// The positions: triangle `k`'s three vertices at one point of the unit square, spread by a
/// multiplicative hash of `k`.
fn positions() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(VERTICES as usize * 16);
    for k in 0..VERTICES / 3 {
        let h = k.wrapping_mul(2654435761);
        let x = f32::from(h as u16) / 32767.5 - 1.0;
        let y = f32::from((h >> 16) as u16) / 32767.5 - 1.0;
        for _ in 0..3 {
            for v in [x, y, 0.5, 1.0] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    bytes
}

/// Passes of four draws of [`VERTICES`] vertices through `rch_vertex_math` (the module docs).
#[test]
fn vertex_heavy_draw_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
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
    let pf = PixelFormat::RGBA8Unorm;
    let mut state = RenderPipelineDescriptor::new(function("rch_vertex_math"));
    state.fragment = Some(function("rch_vertex_math_color"));
    state.color_attachments[0] = Some(ColorAttachmentDescriptor::new(pf));
    let state = device
        .new_render_pipeline_state(&state, &PipelineOptions::default())
        .unwrap();
    let positions = buffer_with(&device, &positions()).unwrap();
    let uniforms = buffer_with(&device, &uniforms()).unwrap();
    let mut target = TextureDescriptor::texture_2d(pf, 512, 512, false);
    target.usage = TextureUsage::RENDER_TARGET;
    target.resource_options = ResourceOptions::private();
    let target = device.new_texture(&target).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target);
    color.load_action = LoadAction::Clear;
    color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 1.0);
    color.store_action = StoreAction::Store;
    let queue = device.new_command_queue().unwrap();
    let (mut times, mut walls) = (Vec::new(), Vec::new());
    for k in 0..23 {
        let began = Instant::now();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
        encoder.set_vertex_buffer(Some(&uniforms), 0, 1).unwrap();
        for _ in 0..DRAWS {
            encoder
                .draw_primitives(PrimitiveType::Triangle, 0, VERTICES, 1, 0)
                .unwrap();
        }
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let wall = began.elapsed().as_secs_f64();
        assert_eq!(command_buffer.error(), None, "pass {k}");
        let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
        if k >= 3 {
            times.push(gpu * 1e3);
            walls.push(wall * 1e3);
        }
    }
    times.sort_by(f64::total_cmp);
    walls.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    let wall = walls[walls.len() / 2];
    eprintln!(
        "VERTEX-COST rch_vertex_math {DRAWS} x {VERTICES} vertices: median {median:.3} ms, \
         fastest {:.3} ms, {:.3} ns a vertex; host wall median {wall:.3} ms, fastest {:.3} ms",
        times[0],
        median * 1e6 / f64::from(DRAWS * VERTICES),
        walls[0]
    );
    messages.take_notices();
}
