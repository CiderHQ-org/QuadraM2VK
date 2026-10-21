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
//!   them to;
//! - into a target over a buffer whose rows the device lays out otherwise (a shadowed texture,
//!   copied back after the pass): the kernel's triangles in one merged draw, and triangle strips
//!   the CPU wrote one indirect draw per command, the same pixels as the host step's there;
//! - after a rebind between the commit and the first submission, of a buffer over host pages
//!   the commands' vertices lie in or of one they never use: recorded again at the rebind, the
//!   new pages' points, the command buffer completing.

use super::degrade_notices::{assert_made_once, counts, on_device};
use super::host_memory::Seen;
use super::indirect_host_steps::{complete, HOST_STEPS};
use super::rebind_reach::{buffer_len, buffer_over, pages};
use super::rebinds::rebind;
use super::render_executions::{forced_with, icb};
use super::{buffer_with, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::RenderStages;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, ResourceUsage, TextureUsage,
};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, ComputePipelineState, Degradation,
    Device, EventRef, IndirectCommandBuffer, IndirectCommandBufferDescriptor, Library, LoadAction,
    PipelineOptions, Region, RenderPassDescriptor, RenderPipelineState, ResourceOptions, Size,
    StoreAction, Texture, TextureDescriptor,
};
use std::sync::Arc;

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

/// What one run draws: the writer's inputs and the target, and the buffer it lies over where it
/// is a texture over one.
struct Scene {
    draws: Buffer,
    marks: Buffer,
    frame: Buffer,
    xy: Buffer,
    target: Texture,
    _over: Option<Buffer>,
}

impl Scene {
    /// `draws` for the kernel, each command's mark 100 plus its index, and the vertices at `xy`
    /// (pixel positions).
    fn new(device: &Device, draws: &[Draw], xy: &[[f32; 2]]) -> Scene {
        Scene::with_rows(device, draws, xy, None)
    }

    /// [`Scene::new`], the target a texture over a `Shared` buffer whose rows lie `rows` bytes
    /// apart where given.
    fn with_rows(device: &Device, draws: &[Draw], xy: &[[f32; 2]], rows: Option<u64>) -> Scene {
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA32Uint, W, H, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        let (target, over) = match rows {
            Some(rows) => {
                let over = device
                    .new_buffer(rows * H, ResourceOptions::shared())
                    .unwrap();
                (over.new_texture(&target, 0, rows).unwrap(), Some(over))
            }
            None => (device.new_texture(&target).unwrap(), None),
        };
        Scene {
            draws: buffer_with(device, &words(draws.iter().flatten().copied())).unwrap(),
            marks: buffer_with(device, &words((0..draws.len() as u32).map(|c| 100 + c))).unwrap(),
            frame: buffer_with(device, &floats([W as f32, H as f32, ROW, 0.0])).unwrap(),
            xy: buffer_with(device, &floats(xy.iter().flatten().copied())).unwrap(),
            target,
            _over: over,
        }
    }

    /// Write command `c` of `icb` on the CPU, as the kernels write one: its mark, the frame and
    /// the vertices as vertex buffers 0 to 2, and a draw of `primitive_type` of `draw`.
    fn write(
        &self,
        icb: &IndirectCommandBuffer,
        c: u64,
        primitive_type: PrimitiveType,
        draw: Draw,
    ) {
        let command = icb.indirect_render_command_at(c).unwrap();
        command
            .set_vertex_buffer(Some(&self.marks), 4 * c, 0)
            .unwrap();
        command.set_vertex_buffer(Some(&self.frame), 0, 1).unwrap();
        command.set_vertex_buffer(Some(&self.xy), 0, 2).unwrap();
        let [start, count, instances, base] = draw.map(u64::from);
        command
            .draw_primitives(primitive_type, start, count, instances, base)
            .unwrap();
    }

    /// Encode ([`encode`](Self::encode)) and run, in one command buffer of a new queue.
    fn run(
        &self,
        device: &Device,
        pipelines: (Option<&ComputePipelineState>, &RenderPipelineState),
        commands: (&IndirectCommandBuffer, u64, u64),
    ) {
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        self.encode(device, &command_buffer, pipelines, commands);
        complete(&command_buffer);
    }

    /// Encode into `command_buffer`: a compute encoder that makes `icb` resident for writing,
    /// `writer`, where given, writing it over `threads` threads, then a pass over the target
    /// that executes commands `0..count` of it.
    fn encode(
        &self,
        device: &Device,
        command_buffer: &CommandBuffer,
        (writer, state): (Option<&ComputePipelineState>, &RenderPipelineState),
        (icb, threads, count): (&IndirectCommandBuffer, u64, u64),
    ) {
        let arguments = buffer_with(device, &icb.gpu_resource_id().0.to_le_bytes()).unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        if let Some(writer) = writer {
            encoder.set_compute_pipeline_state(writer).unwrap();
            for (index, buffer) in [&arguments, &self.draws, &self.marks, &self.frame, &self.xy]
                .into_iter()
                .enumerate()
            {
                encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
            }
            let grid = Size::new(threads, 1, 1);
            encoder.dispatch_threads(grid, grid).unwrap();
        }
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
            scene.run(
                device,
                (Some(&writer), &state),
                (&icb, draws.len() as u64, 8),
            );
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
                    scene.run(
                        device,
                        (Some(&writer), &state),
                        (&icb, draws.len() as u64, 6),
                    );
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

/// GPU-written commands run on the GPU in a pass whose target is a texture over a buffer whose
/// rows the device lays out otherwise (a shadowed texture: its shadow image renders, and is
/// copied back to the buffer after the pass): the kernel's triangles in one merged draw, and
/// triangle strips the CPU wrote (a compute encoder made the buffer resident for writing) one
/// indirect draw per command, neither read by the host. Each draws, pixel for pixel, what the host
/// step draws into such a target (a barrier the CPU set and cleared sends the same commands
/// there).
#[test]
fn gpu_forms_draw_into_a_shadowed_target_what_the_host_step_draws() {
    on_device(|device, messages| {
        let library = library(device);
        let state = state(device, &library);
        let triangles = writer(device, &library, "triangles");
        let draws: Vec<Draw> = vec![
            [0, 7, 1, 0],
            [9, 6, 2, 1],
            [3, 4, 1, 0],
            [15, 2, 0, 0],
            [21, 5, 3, 0],
        ];
        // Rows of 1280 bytes for 64 texels of 16: past the 1024 a linear image of them lays.
        let rows = Some(W * 16 + 256);
        let mut outputs = Vec::new();
        for host_step in [false, true] {
            for merged in [true, false] {
                let scene = Scene::with_rows(device, &draws, &shape_vertices(), rows);
                let icb = icb(device, &inheriting(), 6);
                if !merged {
                    for (c, draw) in draws.iter().enumerate() {
                        scene.write(&icb, c as u64, PrimitiveType::TriangleStrip, *draw);
                    }
                }
                if host_step {
                    let command = icb.indirect_render_command_at(5).unwrap();
                    command.set_barrier().unwrap();
                    command.clear_barrier().unwrap();
                }
                let writer = merged.then_some(&triangles);
                let copies = device.work_counts().shadow_copies;
                let (merges, gpu, read) = counted(device, || {
                    scene.run(device, (writer, &state), (&icb, draws.len() as u64, 6));
                });
                let what = format!("merged {merged}, host step {host_step}");
                assert!(
                    device.work_counts().shadow_copies > copies,
                    "{what}: the target is shadowed"
                );
                match (host_step, merged) {
                    (false, true) => assert_eq!((merges, gpu, read), (6, 6, 0), "{what}"),
                    (false, false) => assert_eq!((merges, gpu, read), (0, 6, 0), "{what}"),
                    (true, _) => assert!(merges == 0 && gpu == 0 && read >= 6, "{what}: {read}"),
                }
                outputs.push(scene.pixels());
            }
            if !host_step {
                let shadowed = [Degradation::ShadowedTexture];
                assert_made_once(messages, &shadowed, &forced_with(device, &[]));
            }
        }
        let host_notices = counts(&messages.take_notices());
        assert!(
            host_notices.contains_key(&HOST_STEPS[0]),
            "{host_notices:?}"
        );
        let (gpu, host) = outputs.split_at(2);
        for (merged, (gpu, host)) in [true, false].iter().zip(gpu.iter().zip(host)) {
            assert!(gpu.iter().any(|p| p[0] != 0), "merged {merged}: drawn");
            assert_eq!(
                gpu, host,
                "merged {merged}: the GPU form against the host step"
            );
        }
    });
}

/// The points `draws` of the kernel's commands draw over `scene`, committed behind an event wait
/// and first submitted only after `rebind` ran: the commit records them in one merged draw, and
/// the rebind records them again (`command_buffer/reencode.rs`), once, over where the buffers lie
/// then; the command buffer completes with a clean report and announces nothing.
fn rebound_run(
    device: &Device,
    messages: &super::Messages,
    scene: &Scene,
    draws: &[Draw],
    rebind: impl FnOnce(),
) {
    let library = library(device);
    let writer = writer(device, &library, "points");
    let state = state(device, &library);
    let icb = icb(device, &inheriting(), 8);
    let queue = device.new_command_queue().unwrap();
    let event = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    let pipelines = (Some(&writer), &state);
    let (merged, gpu, read) = counted(device, || {
        scene.encode(
            device,
            &command_buffer,
            pipelines,
            (&icb, draws.len() as u64, 8),
        );
        command_buffer.commit().unwrap();
    });
    assert_eq!((merged, gpu, read), (8, 8, 0), "the commit's merged draw");
    let before = device.work_counts();
    rebind();
    let after = device.work_counts();
    assert_eq!(
        after.segments_recorded_again,
        before.segments_recorded_again + 1,
        "the rebind recorded the merged draw again"
    );
    assert_eq!(after.reencodes_refused, before.reencodes_refused);
    assert_eq!(
        after.merged_draw_commands - before.merged_draw_commands,
        8,
        "in one merged draw again"
    );
    assert_eq!(after.indirect_records_read, before.indirect_records_read);
    event.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    assert_made_once(messages, &[], &forced_with(device, &[]));
}

/// A GPU-written execution the commit recorded on the GPU alone is recorded again by a rebind
/// between the commit and the first submission (an event wait holds the command buffer), as
/// Metal resolves what its commands reach as it submits the command buffer. A rebind of a buffer
/// over host pages the commands never use moves the live buffers the preprocess searches, and the
/// draws draw as without it; a rebind of the buffer the commands' vertices lie in, from pages
/// holding vertices off the target to pages holding the row of points, draws the new pages'
/// points, pixel for pixel. Neither fails the command buffer (`EncodedUseRebound`) or announces
/// anything.
#[test]
fn a_rebind_before_the_first_submission_records_the_gpu_form_again_over_the_new_pages() {
    on_device(|device, messages| {
        let draws: Vec<Draw> = vec![[0, 5, 1, 0], [8, 3, 2, 1], [30, 6, 3, 2]];
        let points =
            |xy: &[[f32; 2]]| pages(buffer_len(), 0, &floats(xy.iter().flatten().copied()));
        let seen = Arc::new(Seen::default());
        let Some(unused) = buffer_over(device, points(&[[0.0; 2]]), &seen) else {
            return;
        };
        let scene = Scene::new(device, &draws, &point_row());
        rebound_run(device, messages, &scene, &draws, || {
            let seen = Arc::new(Seen::default());
            rebind(&unused, points(&[[1.0; 2]]), &seen).unwrap();
        });
        assert_eq!(
            scene.pixels(),
            drawn_points(&draws),
            "an unused buffer rebound"
        );

        let off: Vec<[f32; 2]> = point_row().iter().map(|_| [-4.0, -4.0]).collect();
        let xy = buffer_over(device, points(&off), &seen).unwrap();
        let scene = Scene {
            xy,
            ..Scene::new(device, &draws, &point_row())
        };
        rebound_run(device, messages, &scene, &draws, || {
            let seen = Arc::new(Seen::default());
            rebind(&scene.xy, points(&point_row()), &seen).unwrap();
        });
        assert_eq!(scene.pixels(), drawn_points(&draws), "the vertices rebound");
        eprintln!(
            "ICB-REBIND {}: GPU forms recorded again at rebinds before the first submission drew \
             the new pages",
            device.name()
        );
    });
}
