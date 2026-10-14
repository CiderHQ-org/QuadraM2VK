//! Device tests of a draw's encoding work on the device the cfg selects, under the validation
//! layer (the consumer's per-draw encode): each holds a piece of work the runtime once repeated at
//! every draw, whatever changed since the draw before, to what changed, through the device's work
//! counts (`Device::work_counts`, not part of the API). Deterministic: they count, and time
//! nothing; and every draw's pixels are checked, so what a draw binds stays exact.
//!
//! - **Settles.** A draw whose bind state is its encoder's last draw's makes no bind shape and
//!   settles no variant (`WorkCounts::draw_shapes_made`, `draws_settled`); nor does one that
//!   binds another texture of the same type, format and usage, whose slot keeps its shape; one of
//!   another shape (a texture of another format, a slot's binding of another class) finds its
//!   shape among those the state settled lately, or makes it and settles it: draws alternating
//!   between two shapes make and settle each once, and a pass that draws as the one before it
//!   makes and settles none.
//! - **Pipelines.** A draw of a pipeline state whose last draw found its `VkPipeline` made, of
//!   the same variant and an equal key, finds that entry again without looking the key up
//!   (`WorkCounts::graphics_pipelines_looked_up`), states switched between included.
//! - **Recorded states.** A draw records into its secondary command buffer only the state
//!   commands that set what changed since the draws before it there
//!   (`WorkCounts::draw_states_recorded`): of an unchanged state, none but its own descriptor set;
//!   switching pipelines, not the dynamic states both make dynamic.
//! - **Retains.** An argument table that binds the objects its encoder last retained it with is
//!   not retained again (`WorkCounts::argument_tables_retained`): one whose objects changed is,
//!   and an object the program let go of after its draw lives until the command buffer
//!   completes.
//!
//! The pipeline is the owned fixture `tob_renders`'s `tob_rect` (a rectangle of the corners at
//! vertex buffer 0) and `tob_sample` (texture 0 through sampler 0, where its pixel lies).

use super::degrade_notices::on_device;
use super::{buffer_with, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBuffer, CommandBufferStatus, Device, LoadAction, PipelineOptions,
    Region, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState, ResourceOptions,
    SamplerState, StoreAction, Texture, TextureDescriptor, WorkCounts,
};

/// The target's side, in pixels: one column a draw.
const SIDE: u64 = 8;
/// The sampled textures' side.
const TEXTURE_SIDE: u64 = 4;
/// Red and green, and the target's clear, blue.
const RED: [u8; 4] = [255, 0, 0, 255];
const GREEN: [u8; 4] = [0, 255, 0, 255];
const CLEAR: [u8; 4] = [0, 0, 255, 255];

/// What a lane test draws with: the pipeline, a nearest sampler, and the target.
struct Scene {
    pipeline: RenderPipelineState,
    sampler: SamplerState,
    target: Texture,
}

/// A pipeline state of `tob_rect` and `tob_sample` into an `RGBA8Unorm` color attachment, which
/// `blends` (Metal's default factors, which store the fragment's color) or not.
fn pipeline(device: &Device, blends: bool) -> RenderPipelineState {
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
    let mut descriptor = RenderPipelineDescriptor::new(function("tob_rect"));
    descriptor.fragment = Some(function("tob_sample"));
    let mut color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
    color.blending_enabled = blends;
    descriptor.color_attachments[0] = Some(color);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

impl Scene {
    fn new(device: &Device) -> Scene {
        let pipeline = pipeline(device, false);
        let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
        let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        target.usage = TextureUsage::RENDER_TARGET;
        target.resource_options = ResourceOptions::shared();
        Scene {
            pipeline,
            sampler,
            target: device.new_texture(&target).unwrap(),
        }
    }

    /// A render encoder of a new command buffer of `device`'s that clears the target to
    /// [`CLEAR`], its pipeline and sampler set.
    fn encoder(&self, device: &Device) -> (CommandBuffer, RenderCommandEncoder) {
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(self.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&self.pipeline).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&self.sampler), 0)
            .unwrap();
        (command_buffer, encoder)
    }

    /// Each pixel of the target, by row, once `command_buffer` completed.
    fn pixels(&self, command_buffer: &CommandBuffer) -> Vec<[u8; 4]> {
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(command_buffer.error(), None);
        let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
        self.target
            .get_bytes(
                &mut bytes,
                4 * SIDE,
                0,
                Region::new_2d(0, 0, SIDE, SIDE),
                0,
                0,
            )
            .unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect()
    }
}

/// A `TEXTURE_SIDE` square texture of `format`, every texel `bytes`.
fn solid(device: &Device, format: PixelFormat, bytes: [u8; 4]) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(format, TEXTURE_SIDE, TEXTURE_SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let texture = device.new_texture(&descriptor).unwrap();
    let texels: Vec<u8> = (0..TEXTURE_SIDE * TEXTURE_SIDE)
        .flat_map(|_| bytes)
        .collect();
    let region = Region::new_2d(0, 0, TEXTURE_SIDE, TEXTURE_SIDE);
    texture
        .replace_region(region, 0, 0, &texels, 4 * TEXTURE_SIDE, 0)
        .unwrap();
    texture
}

/// The corners of the target's column `column`, as `tob_rect` reads them.
fn column(column: u64) -> Vec<u8> {
    let x = |c: u64| c as f32 * 2.0 / SIDE as f32 - 1.0;
    [x(column), -1.0, x(column + 1), 1.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// Draw the rectangle of the vertex buffer or bytes bound at index 0.
fn draw(encoder: &mut RenderCommandEncoder) {
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
}

/// `after`'s count of `count` less `before`'s.
fn grew(before: &WorkCounts, after: &WorkCounts, count: fn(&WorkCounts) -> u64) -> u64 {
    count(after) - count(before)
}

/// Draws of an unchanged state after the first make no bind shape, settle no variant and retain
/// no table: the first draw's work is the whole draw's. Each draw is recorded, its execution of
/// the first's shape, and the target is the texture's.
#[test]
fn draws_of_an_unchanged_state_settle_and_retain_it_once() {
    const DRAWS: usize = 16;
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let corners = buffer_with(
            device,
            &[-1.0f32, -1.0, 1.0, 1.0].map(f32::to_le_bytes).concat(),
        )
        .unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_vertex_buffer(Some(&corners), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        draw(&mut encoder);
        let first = device.work_counts();
        for _ in 1..DRAWS {
            draw(&mut encoder);
        }
        let rest = device.work_counts();
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        assert_eq!(grew(&first, &rest, |c| c.argument_tables_retained), 0);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
        let executions = command_buffer.executions();
        assert_eq!(executions.len(), DRAWS);
        assert!(executions
            .iter()
            .all(|e| e.shape() == executions[0].shape()));
    });
}

/// Draws that each bind another texture of the same type, format and usage, and other inline
/// bytes of the same length, one column each: none past the first makes a bind shape (its
/// texture slot keeps its shape) or settles, and each retains the fragment table, whose texture
/// changed, alone. Each column is its own draw's texture's.
#[test]
fn draws_of_other_objects_of_one_shape_make_and_settle_nothing() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, RED),
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
        ];
        let (command_buffer, mut encoder) = scene.encoder(device);
        let mut first = None;
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder
                .set_fragment_texture(Some(&textures[(c % 2) as usize]), 0)
                .unwrap();
            draw(&mut encoder);
            first.get_or_insert_with(|| device.work_counts());
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        let later = SIDE - 1;
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        assert_eq!(grew(&first, &rest, |c| c.argument_tables_retained), later);
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = [RED, GREEN][at % 2];
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws that bind textures of two formats in turn make and settle each bind shape once: the state
/// keeps the settles of its latest shapes, among which a draw finds its shape. Before, each draw
/// made its shape, and before that settled it again. Each column is its own draw's texture's.
#[test]
fn draws_of_another_shape_settle_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        // Green either way: `BGRA8Unorm`'s bytes are blue, green, red, alpha.
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
            solid(device, PixelFormat::BGRA8Unorm, GREEN),
        ];
        let (command_buffer, mut encoder) = scene.encoder(device);
        let before = device.work_counts();
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder
                .set_fragment_texture(Some(&textures[(c % 2) as usize]), 0)
                .unwrap();
            draw(&mut encoder);
        }
        let after = device.work_counts();
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&before, &after, |c| c.draw_shapes_made), 2);
        assert_eq!(grew(&before, &after, |c| c.draws_settled), 2);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        let executions = command_buffer.executions();
        assert_eq!(executions.len(), SIDE as usize);
        assert_ne!(executions[0].shape(), executions[1].shape());
        assert_eq!(executions[0].shape(), executions[2].shape());
    });
}

/// A pass that draws as the one before it, each of its encoder's draws of a bind state a draw of
/// the pass before had, makes no bind shape and settles nothing: its first draw, whose encoder
/// keeps no shape, and its draw of the other shape find theirs among those the state settled.
/// Before, each pass's encoder made its shapes again. Every column is drawn.
#[test]
fn a_pass_that_draws_as_the_one_before_makes_no_shape() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let textures = [
            solid(device, PixelFormat::RGBA8Unorm, GREEN),
            solid(device, PixelFormat::BGRA8Unorm, GREEN),
        ];
        let pass = || {
            let (command_buffer, mut encoder) = scene.encoder(device);
            for c in 0..SIDE {
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                encoder
                    .set_fragment_texture(Some(&textures[(c % 2) as usize]), 0)
                    .unwrap();
                draw(&mut encoder);
            }
            encoder.end_encoding().unwrap();
            command_buffer
        };
        let first = pass();
        let pixels = scene.pixels(&first);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        let before = device.work_counts();
        let second = pass();
        let after = device.work_counts();
        assert_eq!(grew(&before, &after, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&before, &after, |c| c.draws_settled), 0);
        let pixels = scene.pixels(&second);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        let (first, second) = (first.executions(), second.executions());
        assert_eq!(second.len(), SIDE as usize);
        assert_eq!(first[0].shape(), second[0].shape());
        assert_eq!(first[1].shape(), second[1].shape());
    });
}

/// Draws whose vertex buffer slot 0 binds a buffer and inline bytes in turn, a binding of another
/// class at the slot each draw, a slot's class being of its shape, make and settle each shape
/// once (the state keeps the settles of its latest shapes, among which a draw finds its shape).
/// Every column is drawn, from the buffer's corners or the bytes'.
#[test]
fn draws_that_change_a_slot_s_class_make_their_shapes_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let corners: Vec<u8> = (0..SIDE).flat_map(column).collect();
        let corners = buffer_with(device, &corners).unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let before = device.work_counts();
        for c in 0..SIDE {
            match c % 2 {
                0 => encoder
                    .set_vertex_buffer(Some(&corners), 16 * c, 0)
                    .unwrap(),
                _ => encoder.set_vertex_bytes(&column(c), 0).unwrap(),
            }
            draw(&mut encoder);
        }
        let after = device.work_counts();
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&before, &after, |c| c.draw_shapes_made), 2);
        assert_eq!(grew(&before, &after, |c| c.draws_settled), 2);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
        let executions = command_buffer.executions();
        assert_eq!(executions.len(), SIDE as usize);
        assert_ne!(executions[0].shape(), executions[1].shape());
    });
}

/// Draws that switch between two pipeline states at every draw, of the same functions, look
/// neither's `VkPipeline` up by its key once each state's draw found it made: each state's later
/// draws find that entry again (`WorkCounts::graphics_pipelines_looked_up`), and neither makes a
/// shape or settles. A first command buffer, completed, has both pipelines made. Every column
/// is drawn.
#[test]
fn draws_switching_pipeline_states_find_their_pipelines_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pipelines = [scene.pipeline.clone(), pipeline(device, true)];
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let (made, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        for pipeline in &pipelines {
            encoder.set_render_pipeline_state(pipeline).unwrap();
            encoder.set_vertex_bytes(&column(0), 0).unwrap();
            draw(&mut encoder);
        }
        encoder.end_encoding().unwrap();
        scene.pixels(&made);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder
                .set_render_pipeline_state(&pipelines[(c % 2) as usize])
                .unwrap();
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
            if c == 1 {
                first = Some(device.work_counts());
            }
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.graphics_pipelines_looked_up), 0);
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Draw through `encoder` (`draw`), returning the state commands it recorded
/// (`WorkCounts::draw_states_recorded`).
fn recorded_by(device: &Device, encoder: &mut RenderCommandEncoder) -> u64 {
    let before = device.work_counts();
    draw(encoder);
    grew(&before, &device.work_counts(), |c| c.draw_states_recorded)
}

/// Draw once through each of `pipelines` in a command buffer of its own, completed, so each
/// pipeline is made: a later draw records as it draws, not once a making is done.
fn make(device: &Device, scene: &Scene, pipelines: &[RenderPipelineState], texture: &Texture) {
    let (command_buffer, mut encoder) = scene.encoder(device);
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    for pipeline in pipelines {
        encoder.set_render_pipeline_state(pipeline).unwrap();
        encoder.set_vertex_bytes(&column(0), 0).unwrap();
        draw(&mut encoder);
    }
    encoder.end_encoding().unwrap();
    scene.pixels(&command_buffer);
}

/// Draws of an unchanged state into one render pass instance record its state commands once:
/// past the first, a draw records at most its own descriptor set (each draw writes its set 0
/// anew), so no pipeline, push constant, buffer or dynamic state is set again. The first draw
/// records them all, and the target is the texture's.
#[test]
fn draws_of_an_unchanged_state_record_its_commands_once() {
    const DRAWS: usize = 16;
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        make(device, &scene, std::slice::from_ref(&scene.pipeline), &red);
        let corners = [-1.0f32, -1.0, 1.0, 1.0].map(f32::to_le_bytes).concat();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_vertex_bytes(&corners, 0).unwrap();
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let first = recorded_by(device, &mut encoder);
        let later: Vec<u64> = (1..DRAWS)
            .map(|_| recorded_by(device, &mut encoder))
            .collect();
        encoder.end_encoding().unwrap();
        assert!(first > 1, "the first draw recorded {first}");
        assert!(later.iter().all(|&n| n <= 1), "{first}, then {later:?}");
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Draws that switch between two pipeline states of the same layout and dynamic states at every
/// draw, each with other inline bytes, record their pipeline and what changed, never the dynamic
/// states the first draw set: each records fewer state commands than the first. Every column
/// is drawn, through either pipeline.
#[test]
fn draws_switching_pipelines_keep_the_dynamic_states_set() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pipelines = [scene.pipeline.clone(), pipeline(device, true)];
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        make(device, &scene, &pipelines, &red);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let recorded: Vec<u64> = (0..SIDE)
            .map(|c| {
                encoder
                    .set_render_pipeline_state(&pipelines[(c % 2) as usize])
                    .unwrap();
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                recorded_by(device, &mut encoder)
            })
            .collect();
        encoder.end_encoding().unwrap();
        let (first, later) = (recorded[0], &recorded[1..]);
        assert!(later.iter().all(|&n| 1 <= n && n < first), "{recorded:?}");
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// A pipeline state that rasterizes nothing: the owned fixture `rvc_draw_bases`'s vertex
/// function, which returns nothing, as Metal requires there, and writes 16 bytes at element
/// `(iid * 8 & 24) | (vid & 7)` of buffer 0, into an `RGBA8Unorm` color attachment.
fn unrasterized(device: &Device) -> RenderPipelineState {
    let ll = engine_dir().join("rvc_draw_bases.ll");
    let library = device
        .new_library_with_air(&llvm_assemble(&ll).unwrap())
        .unwrap();
    let vertex = library
        .new_function_with_name("draw_bases_in_a_vertex_function")
        .unwrap()
        .function_descriptor();
    let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor.rasterization_enabled = false;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Draw once through `unrasterized`, binding `written` at vertex buffer 0.
fn draw_unrasterized(
    encoder: &mut RenderCommandEncoder,
    unrasterized: &RenderPipelineState,
    written: &Buffer,
) {
    encoder.set_render_pipeline_state(unrasterized).unwrap();
    encoder.set_vertex_buffer(Some(written), 0, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
}

/// A draw after one through a pipeline that rasterizes nothing, in one render pass instance,
/// sets its depth, stencil and blend-constant states again: a pipeline without fragment state
/// holds none of them, so its bind leaves them unknown, and the record of the secondary command
/// buffer (`render/bound.rs`) must not keep them as set. Both pipelines are made first, so every
/// draw records as it draws, into one secondary. Under the validation layer, a draw that left
/// them out reports them unset; every column is drawn.
#[test]
fn a_draw_after_an_unrasterized_draw_sets_its_fragment_states_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let unrasterized = unrasterized(device);
        let written = buffer_with(device, &[0; 512]).unwrap();
        make(device, &scene, std::slice::from_ref(&scene.pipeline), &red);
        let (command_buffer, mut encoder) = scene.encoder(device);
        draw_unrasterized(&mut encoder, &unrasterized, &written);
        encoder.end_encoding().unwrap();
        scene.pixels(&command_buffer);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        for c in 0..SIDE {
            encoder.set_render_pipeline_state(&scene.pipeline).unwrap();
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
            if c == 0 {
                draw_unrasterized(&mut encoder, &unrasterized, &written);
            }
        }
        encoder.end_encoding().unwrap();
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// A texture the program let go of after its draw, the fragment table then binding another, is
/// held until the command buffer completes: its column is its texture's, and the validation
/// layer reports no use of a destroyed object. The columns left of the first draw's keep the
/// clear.
#[test]
fn a_texture_let_go_after_its_draw_lives_until_the_command_buffer_completes() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let (command_buffer, mut encoder) = scene.encoder(device);
        for c in 1..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            if c % 2 == 1 {
                let red = solid(device, PixelFormat::RGBA8Unorm, RED);
                encoder.set_fragment_texture(Some(&red), 0).unwrap();
                draw(&mut encoder);
                // The program's last handle goes: the encoder alone holds it now.
                drop(red);
                encoder.set_fragment_texture(Some(&green), 0).unwrap();
            } else {
                draw(&mut encoder);
            }
        }
        encoder.end_encoding().unwrap();
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = match at as u64 % SIDE {
                0 => CLEAR,
                c if c % 2 == 1 => RED,
                _ => GREEN,
            };
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}
