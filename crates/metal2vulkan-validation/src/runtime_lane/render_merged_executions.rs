//! Executions of render commands a kernel writes, merged into one draw (the runtime's
//! `render/executions/gpu.rs`, `bind::DrawReplay::merged`), on the device the cfg selects, under
//! the validation layer. `icb_merged_draws`' kernels encode command `tid` from `draws[tid]` (its
//! vertex start, vertex count, instance count and base instance) with its own mark as vertex
//! buffer 0; its vertex function places vertex `vid` of instance `iid` at `xy[vid]` moved down
//! `iid` rows and carries the mark, `vid`, `iid` and the base instance flat into a `RGBA32Uint`
//! target, and writes no memory, and its fragment function reads no buffer. They draw as Metal's
//! draws of the commands do:
//!
//! - points, one pixel a vertex of an instance: each command's own vertices and instances, a
//!   command of no vertex or no instance (or past what the kernel wrote) drawing nothing, a later
//!   command over an earlier one's pixel winning, all in one merged draw, the host reading no
//!   record;
//! - points, lines (a trailing vertex drawing nothing) and triangles (trailing vertices drawing
//!   nothing): the same pixels as the host step, which a barrier the CPU set and cleared sends
//!   them to.

use super::buffer_with;
use super::degrade_notices::{assert_made_once, counts, on_device};
use super::indirect_host_steps::{complete, HOST_STEPS};
use super::render_executions::{forced_with, icb};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::RenderStages;
use metal2vulkan::metal::{IndirectCommandType, PixelFormat, ResourceUsage, TextureUsage};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, Library, LoadAction, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size, StoreAction, Texture,
    TextureDescriptor,
};

/// The target's width and height, and the rows an instance moves its vertices down.
const W: u64 = 64;
const H: u64 = 48;
const ROW: f32 = 8.0;

/// A command's draw: vertex start, vertex count, instance count, base instance.
type Draw = [u32; 4];

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let bitcode = llvm_assemble(&engine_dir().join("icb_merged_draws.ll")).unwrap();
    device.new_library_with_air(&bitcode).unwrap()
}

/// The writer of `kind` (`points`, `lines` or `triangles`).
fn writer(device: &Device, library: &Library, kind: &str) -> ComputePipelineState {
    let function = library
        .new_function_with_name(&format!("icb_merged_{kind}"))
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The commands' state, made with `supportIndirectCommandBuffers`.
fn state(device: &Device, library: &Library) -> RenderPipelineState {
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function("icb_merged_vertex"));
    descriptor.fragment = Some(function("icb_merged_fragment"));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Uint));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Draw commands of three vertex buffers that inherit the encoder's state.
fn inheriting() -> IndirectCommandBufferDescriptor {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::DRAW;
    descriptor.inherit_pipeline_state = true;
    descriptor.max_vertex_buffer_bind_count = 3;
    descriptor.max_fragment_buffer_bind_count = 0;
    descriptor
}

fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}

/// What one run draws: the writer's inputs and the target.
struct Scene {
    draws: Buffer,
    marks: Buffer,
    frame: Buffer,
    xy: Buffer,
    target: Texture,
}

impl Scene {
    /// `draws` for the kernel, each command's mark 100 plus its index, and the vertices at `xy`
    /// (pixel positions).
    fn new(device: &Device, draws: &[Draw], xy: &[[f32; 2]]) -> Scene {
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Uint, W, H, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        Scene {
            draws: buffer_with(device, &words(draws.iter().flatten().copied())).unwrap(),
            marks: buffer_with(device, &words((0..draws.len() as u32).map(|c| 100 + c))).unwrap(),
            frame: buffer_with(device, &floats([W as f32, H as f32, ROW, 0.0])).unwrap(),
            xy: buffer_with(device, &floats(xy.iter().flatten().copied())).unwrap(),
            target: device.new_texture(&target).unwrap(),
        }
    }

    /// Encode and run, in one command buffer of a new queue: `writer` over `threads` threads
    /// writing `icb`, then a pass over the target that executes commands `0..count` of it.
    fn run(
        &self,
        device: &Device,
        (writer, state): (&ComputePipelineState, &RenderPipelineState),
        (icb, threads, count): (&IndirectCommandBuffer, u64, u64),
    ) {
        let arguments = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(writer).unwrap();
        for (index, buffer) in [&arguments, &self.draws, &self.marks, &self.frame, &self.xy]
            .into_iter()
            .enumerate()
        {
            encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
        }
        encoder
            .use_resource(icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        let grid = Size::new(threads, 1, 1);
        encoder.dispatch_threads(grid, grid).unwrap();
        encoder.end_encoding().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        for buffer in [&self.marks, &self.frame, &self.xy] {
            encoder
                .use_resource(
                    buffer.as_resource(),
                    ResourceUsage::READ,
                    RenderStages::VERTEX,
                )
                .unwrap();
        }
        encoder.execute_commands_in_buffer(icb, 0..count).unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
    }

    /// The target's pixels, row by row.
    fn pixels(&self) -> Vec<[u32; 4]> {
        let mut bytes = vec![0u8; (16 * W * H) as usize];
        self.target
            .get_bytes(&mut bytes, 16 * W, 0, Region::new_2d(0, 0, W, H), 0, 0)
            .unwrap();
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|p| {
                std::array::from_fn(|i| u32::from_le_bytes(p[4 * i..4 * i + 4].try_into().unwrap()))
            })
            .collect()
    }
}

/// The merged draws and the GPU form's commands across `run`, and the records the host read.
fn counted(device: &Device, run: impl FnOnce()) -> (u64, u64, u64) {
    let before = device.work_counts();
    run();
    let after = device.work_counts();
    (
        after.merged_draw_commands - before.merged_draw_commands,
        after.gpu_form_commands - before.gpu_form_commands,
        after.indirect_records_read - before.indirect_records_read,
    )
}

/// One point a pixel: vertex `v` at the center of pixel `(v mod W, 0)`.
fn point_row() -> Vec<[f32; 2]> {
    (0..W as u32).map(|v| [v as f32 + 0.5, 0.5]).collect()
}

/// What Metal's draws of points `draws` (command `c` marked `100 + c`) leave at each pixel, in
/// command order: vertex `v` of instance `i` at `(v, i · ROW)`.
fn drawn_points(draws: &[Draw]) -> Vec<[u32; 4]> {
    let mut out = vec![[0; 4]; (W * H) as usize];
    for (c, &[start, count, instances, base]) in draws.iter().enumerate() {
        for i in base..base + instances {
            for v in start..start + count {
                let at = (i as u64 * ROW as u64) * W + v as u64;
                out[at as usize] = [100 + c as u32, v, i, base];
            }
        }
    }
    out
}

/// Points of commands a kernel writes draw in one merged draw, the host reading no record: each
/// command its own vertices from its vertex start, in each of its instances from its base
/// instance, with its own mark; a command of no vertex or no instance draws nothing, and so do
/// the commands past those the kernel wrote; where two commands draw one pixel, the later wins.
#[test]
fn merged_points_draw_each_command_s_own_vertices_and_instances() {
    on_device(|device, messages| {
        let library = library(device);
        let writer = writer(device, &library, "points");
        let state = state(device, &library);
        let draws: Vec<Draw> = vec![
            [0, 5, 1, 0],
            [8, 3, 2, 1],
            [4, 0, 3, 0],
            [20, 4, 0, 0],
            [9, 2, 1, 1],
            [30, 6, 3, 2],
        ];
        let scene = Scene::new(device, &draws, &point_row());
        let icb = icb(device, &inheriting(), 8);
        let (merged, gpu, read) = counted(device, || {
            scene.run(device, (&writer, &state), (&icb, draws.len() as u64, 8));
        });
        assert_eq!(scene.pixels(), drawn_points(&draws));
        assert_eq!((merged, gpu, read), (8, 8, 0), "drawn in one merged draw");
        assert_made_once(messages, &[], &forced_with(device, &[]));
    });
}

/// Vertices for lines and triangles: three-pixel-wide shapes, each vertex at its own place, so
/// a primitive of vertices `3k`, `3k + 1`, `3k + 2` covers its own pixels.
fn shape_vertices() -> Vec<[f32; 2]> {
    (0..48u32)
        .map(|v| {
            let (k, corner) = (v / 3, v % 3);
            let x = 1.0 + 4.0 * (k % 15) as f32;
            let y = 1.0 + 4.0 * (k / 15) as f32;
            match corner {
                0 => [x, y],
                1 => [x + 3.0, y],
                _ => [x, y + 3.0],
            }
        })
        .collect()
}

/// Points, lines and triangles of commands a kernel writes draw in one merged draw what the host
/// step draws of them (a barrier the CPU set and cleared in a command after the writer's sends
/// the same commands there): pixel for pixel, a list's trailing vertices drawing nothing. The
/// merged draws announce nothing; the host steps announce theirs.
#[test]
fn merged_draws_draw_what_the_host_step_draws() {
    on_device(|device, messages| {
        let library = library(device);
        let state = state(device, &library);
        let draws: Vec<Draw> = vec![
            [0, 7, 1, 0],
            [9, 6, 2, 1],
            [3, 4, 1, 0],
            [15, 2, 0, 0],
            [21, 5, 3, 0],
        ];
        let kinds = [
            ("points", point_row()),
            ("lines", shape_vertices()),
            ("triangles", shape_vertices()),
        ];
        let mut outputs: Vec<Vec<[u32; 4]>> = Vec::new();
        for host_step in [false, true] {
            for (kind, xy) in &kinds {
                let writer = writer(device, &library, kind);
                let scene = Scene::new(device, &draws, xy);
                let icb = icb(device, &inheriting(), 6);
                if host_step {
                    let command = icb.indirect_render_command_at(5).unwrap();
                    command.set_barrier().unwrap();
                    command.clear_barrier().unwrap();
                }
                let (merged, gpu, read) = counted(device, || {
                    scene.run(device, (&writer, &state), (&icb, draws.len() as u64, 6));
                });
                match host_step {
                    false => assert_eq!((merged, gpu, read), (6, 6, 0), "{kind}: merged"),
                    true => assert!(merged == 0 && gpu == 0 && read >= 6, "{kind}: {read}"),
                }
                outputs.push(scene.pixels());
            }
            if !host_step {
                assert_made_once(messages, &[], &forced_with(device, &[]));
            }
        }
        let host_notices = counts(&messages.take_notices());
        assert!(
            host_notices.contains_key(&HOST_STEPS[0]),
            "{host_notices:?}"
        );
        let (merged, host) = outputs.split_at(kinds.len());
        for ((kind, _), (merged, host)) in kinds.iter().zip(merged.iter().zip(host)) {
            assert!(
                merged.iter().any(|p| p[0] != 0),
                "{kind}: the commands drew"
            );
            assert_eq!(
                merged, host,
                "{kind}: the merged draw against the host step"
            );
        }
    });
}
