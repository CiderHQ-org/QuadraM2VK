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
//!   makes and settles none. A draw that changes only values no shape reads (the depth/stencil
//!   state, the stencil references, the blend color) keeps its encoder's last draw's shape, and
//!   looks for none among those settled (`WorkCounts::draw_shapes_found`); nor do draws that
//!   switch between pipeline states whose bind states differ, each starting from its own pipeline
//!   state's last draw's state, which the encoder set aside.
//! - **Zeroed colors.** Draws into a pass of a color their pipeline state lacks, which Metal
//!   zeroes where they cover it, read the colors they zero where their encoder keeps them, compare
//!   them with their bind state's in place, and copy none while those hold (allocations counted on
//!   the drawing thread).
//! - **Argument samplers.** A draw lists the device's live sampler states as its encoder's bind
//!   state keeps them, without taking them again, until a state is made or goes
//!   (`WorkCounts::argument_samplers_taken`).
//! - **Render passes.** A pass that draws as the one before it makes no render pass: its device
//!   keeps the compatible pass and the pass its instance begins (`WorkCounts::render_passes_made`).
//! - **Views.** A pass that binds the textures the pass before it bound works out none of their
//!   views (`WorkCounts::bind_views_worked_out`); a texture made after one of them went works out
//!   its own.
//! - **Stage tables.** A draw reads into its encoder's bind state only the stage tables a call
//!   changed since a draw read them, through the same pipeline state's interface
//! - **Pipelines.** A draw of a pipeline state one of whose last draws found its `VkPipeline`
//!   made, of the same variant and an equal key, finds that entry again without looking the key
//!   up (`WorkCounts::graphics_pipelines_looked_up`), states and keys switched between included.
//! - **Recorded states.** A draw records into its secondary command buffer only the state
//!   commands that set what changed since the draws before it there
//!   (`WorkCounts::draw_states_recorded`): of an unchanged state, none but its own descriptor set;
//!   switching pipelines, not the dynamic states both make dynamic.
//! - **Retains.** An argument table that binds the objects its encoder last retained it with is
//!   not retained again (`WorkCounts::argument_tables_retained`): one whose objects changed is,
//!   and an object the program let go of after its draw lives until the command buffer
//!   completes. Its encoder lists every object a retain took or held already
//!   (`CommandBuffer::ended_encoders`), a slot rebound to one that looks like its last included.
//!
//! The pipeline is the owned fixture `tob_renders`'s `tob_rect` (a rectangle of the corners at
//! vertex buffer 0) and `tob_sample` (texture 0 through sampler 0, where its pixel lies).

use super::degrade_notices::on_device;
use super::{buffer_with, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
};
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
    pipeline_of(device, "tob_sample", blends)
}

/// A pipeline state of `tob_rect` and `tob_renders`'s fragment function `fragment` into an
/// `RGBA8Unorm` color attachment, which `blends` or not ([`pipeline`]).
fn pipeline_of(device: &Device, fragment: &str, blends: bool) -> RenderPipelineState {
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
    descriptor.fragment = Some(function(fragment));
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

/// Draws that change between them only values no bind shape reads, one column each: the stencil
/// references, the blend color, and the depth/stencil state (of a depth test every fragment
/// passes, with and without depth writes). None past the first looks for a bind shape, makes one
/// or settles: each keeps its encoder's last draw's. Each column is the texture's.
#[test]
fn draws_that_change_only_values_no_shape_reads_keep_their_shape() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
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
        descriptor.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        descriptor.depth_format = Some(PixelFormat::Depth32Float);
        let pipeline = device
            .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        let mut depth = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, SIDE, SIDE, false);
        depth.usage = TextureUsage::RENDER_TARGET;
        depth.resource_options = ResourceOptions::private();
        let depth = device.new_texture(&depth).unwrap();
        let states = [false, true].map(|writes| {
            let mut state = DepthStencilDescriptor::default();
            state.depth_compare_function = CompareFunction::LessEqual;
            state.depth_write_enabled = writes;
            device.new_depth_stencil_state(&state).unwrap()
        });
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(scene.target.clone());
        color.load_action = LoadAction::Clear;
        color.store_action = StoreAction::Store;
        color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
        pass.depth_attachment.texture = Some(depth);
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.store_action = StoreAction::DontCare;
        pass.depth_attachment.clear_depth = 1.0;
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&scene.sampler), 0)
            .unwrap();
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder.set_stencil_reference_value(c as u32);
            let level = c as f32 / SIDE as f32;
            encoder.set_blend_color(level, 1.0 - level, level, 1.0);
            encoder
                .set_depth_stencil_state(&states[(c % 2) as usize])
                .unwrap();
            draw(&mut encoder);
            first.get_or_insert_with(|| device.work_counts());
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_found), 0);
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Draws of one encoder take the device's live sampler states to list only where they changed
/// since the listing their encoder's bind state keeps (`WorkCounts::argument_samplers_taken`):
/// none while no sampler state is made or goes; a state made between two draws, and that state
/// gone, each make the next draw take them once. Each column is the texture's.
#[test]
fn draws_take_the_device_s_sampler_states_only_where_they_changed() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let taken = || device.work_counts().argument_samplers_taken;
        let mut counts = Vec::new();
        let mut made = None;
        for c in 0..SIDE {
            // A sampler state made before column 3's draw, gone before column 6's.
            if c == 3 {
                let mut descriptor = SamplerDescriptor::new();
                descriptor.lod_max_clamp = 7.0;
                made = Some(device.new_sampler_state(&descriptor).unwrap());
            }
            if c == 6 {
                made = None;
            }
            let before = taken();
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            draw(&mut encoder);
            counts.push(taken() - before);
        }
        assert!(made.is_none());
        encoder.end_encoding().unwrap();
        // The first draws settle the encoder's state from another's, and may take them.
        assert_eq!(counts[2..], [0, 1, 0, 0, 1, 0], "{counts:?}");
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// Passes that each draw as the one before them, one column a draw, make no render pass past the
/// first's (`WorkCounts::render_passes_made`): their device keeps the compatible pass their draws
/// record against and the pass their instance begins, each by what it is made of. Each pass's
/// target is the texture's.
#[test]
fn a_pass_that_draws_as_the_one_before_makes_no_render_pass() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let made = || device.work_counts().render_passes_made;
        let mut counts = Vec::new();
        for _ in 0..3 {
            let before = made();
            let (command_buffer, mut encoder) = scene.encoder(device);
            encoder.set_fragment_texture(Some(&red), 0).unwrap();
            for c in 0..SIDE {
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                draw(&mut encoder);
            }
            encoder.end_encoding().unwrap();
            let pixels = scene.pixels(&command_buffer);
            assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
            counts.push(made() - before);
        }
        assert_eq!(counts[1..], [0, 0], "{counts:?}");
    });
}

/// Draws read into their encoder's kept bind state only the stage tables a call changed since a
/// draw read them (`WorkCounts::stage_tables_read`): each column's first draw the vertex table,
/// whose bytes it sets, and the fragment table too where it binds another texture; its second
/// draw, with no call before it, none; and every table at the draw after another pipeline state
/// is set, whose interface they are read through. Each second draw's uses are its first's, which
/// its encoder noted already (`WorkCounts::commands_noted_as_before`); so are each first draw's
/// the draw's before it, inline bytes being no use (they lie in the command buffer's upload
/// arena, which hazard tracking leaves alone), another pipeline state of the same stages too, but
/// at the encoder's first draw, which no draw of its noted, and where another texture is bound.
/// Each column is its texture's.
#[test]
fn draws_read_only_the_stage_tables_a_call_changed() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let blending = pipeline(device, true);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let read = || device.work_counts().stage_tables_read;
        let noted = || device.work_counts().commands_noted_as_before;
        let mut counts = Vec::new();
        let mut notes = Vec::new();
        for c in 0..SIDE {
            match c {
                3 => encoder.set_fragment_texture(Some(&green), 0).unwrap(),
                5 => encoder.set_render_pipeline_state(&blending).unwrap(),
                _ => {}
            }
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            let (before, noted_before) = (read(), noted());
            draw(&mut encoder);
            let (first, noted_first) = (read() - before, noted() - noted_before);
            draw(&mut encoder);
            counts.push((first, read() - before - first));
            notes.push((noted_first, noted() - noted_before - noted_first));
        }
        encoder.end_encoding().unwrap();
        let as_before = [
            (0, 1),
            (1, 1),
            (1, 1),
            (0, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
        ];
        assert_eq!(notes, as_before, "{notes:?}");
        // The first draw reads the tables from another encoder's state: all four.
        let reads = [
            (4, 0),
            (1, 0),
            (1, 0),
            (2, 0),
            (1, 0),
            (4, 0),
            (1, 0),
            (1, 0),
        ];
        assert_eq!(counts, reads, "{counts:?}");
        let pixels = scene.pixels(&command_buffer);
        for (i, &pixel) in pixels.iter().enumerate() {
            let want = match i as u64 % SIDE >= 3 {
                true => GREEN,
                false => RED,
            };
            assert_eq!(pixel, want, "pixel {i}: {pixels:?}");
        }
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

/// A pass that binds the two textures the pass before it bound works out neither's view: its
/// encoder takes the views the last one's binds found (`WorkCounts::bind_views_worked_out`).
/// One of them let go of and a texture of another color made, a third pass works out the new
/// texture's view alone, and draws its color. Every column is drawn. The textures are
/// `RGBA8Unorm`, as the target, so each texel's bytes are the pixel its column stores.
#[test]
fn a_pass_that_binds_as_the_one_before_works_out_no_view() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pass = |textures: &[&Texture; 2]| {
            let before = device.work_counts();
            let (command_buffer, mut encoder) = scene.encoder(device);
            for c in 0..SIDE {
                encoder.set_vertex_bytes(&column(c), 0).unwrap();
                encoder
                    .set_fragment_texture(Some(textures[(c % 2) as usize]), 0)
                    .unwrap();
                draw(&mut encoder);
            }
            encoder.end_encoding().unwrap();
            let after = device.work_counts();
            let worked_out = grew(&before, &after, |c| c.bind_views_worked_out);
            (worked_out, scene.pixels(&command_buffer))
        };
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let other = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        let (first, pixels) = pass(&[&green, &other]);
        assert_eq!(first, 2);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        let (second, pixels) = pass(&[&green, &other]);
        assert_eq!(second, 0);
        assert!(pixels.iter().all(|&p| p == GREEN), "{pixels:?}");
        drop(other);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let (third, pixels) = pass(&[&green, &red]);
        assert_eq!(third, 1);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = if at.is_multiple_of(2) { GREEN } else { RED };
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
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

/// Draws that switch at every draw between pipeline states whose bind states differ (`tob_sample`,
/// and `tob_solid`, which reads its color through a uniform view of fragment buffer 0), each
/// binding what it bound before but its column's bytes, find each one's shape kept past the first
/// draw of each: the encoder sets aside the state of the pipeline state it drew before
/// (`LiveBind::select`), so none looks its shape up among those settled
/// (`WorkCounts::draw_shapes_found`), makes one or settles. Before, each draw past the first two
/// looked its shape up, the state its encoder kept having last been read through the other
/// pipeline state's interface. Each column is its own pipeline state's color.
#[test]
fn draws_switching_between_pipeline_states_of_other_states_keep_each_shape() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let pipelines = [
            scene.pipeline.clone(),
            pipeline_of(device, "tob_solid", false),
        ];
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let green: Vec<u8> = [0.0f32, 1.0, 0.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        encoder.set_fragment_bytes(&green, 0).unwrap();
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
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_found), 0);
        assert_eq!(grew(&first, &rest, |c| c.draw_shapes_made), 0);
        assert_eq!(grew(&first, &rest, |c| c.draws_settled), 0);
        let pixels = scene.pixels(&command_buffer);
        for (at, &pixel) in pixels.iter().enumerate() {
            let want = [RED, GREEN][at % 2];
            assert_eq!(pixel, want, "pixel {at}: {pixels:?}");
        }
    });
}

/// Draws of one pipeline state that switch between two keys of its `VkPipeline` at every draw
/// (its depth clip mode, which no pipeline makes dynamic) look neither up by its key once each
/// key's draw found it made: the state keeps the pipelines its last draws found
/// (`WorkCounts::graphics_pipelines_looked_up`). A first command buffer, completed, and the
/// pipeline makings its draws began on the device's pool, settled, have both made. Every column is
/// drawn. Not run on a device without `depthClamp`, which refuses the clamp.
#[test]
fn draws_of_one_state_switching_keys_find_their_pipelines_again() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let modes = [DepthClipMode::Clip, DepthClipMode::Clamp];
        let (made, mut encoder) = scene.encoder(device);
        if encoder.set_depth_clip_mode(DepthClipMode::Clamp).is_err() {
            return;
        }
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        for mode in modes {
            encoder.set_depth_clip_mode(mode).unwrap();
            encoder.set_vertex_bytes(&column(0), 0).unwrap();
            draw(&mut encoder);
        }
        encoder.end_encoding().unwrap();
        scene.pixels(&made);
        pipelines_settled(device);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&red), 0).unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder
                .set_depth_clip_mode(modes[(c % 2) as usize])
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
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}

/// The most allocations a draw into a pass of a color its pipeline state lacks makes beyond a
/// draw of the same state into a pass of its own colors alone, both of a state unchanged but for
/// its column's bytes ([`draws_into_a_pass_color_their_state_lacks_copy_no_color_set`]), on
/// average: none. The colors the draw zeroes are those the encoder kept as the pipeline state was
/// set, which its settings read in place (`RenderCommandEncoder::draw_colors`), and its bind
/// state compares in place (`Reshape::field_from`), copying them only where they changed.
/// Measured: -0.20 on ANV (UHD 770); 0.80 where the settings copied them, 1.80 where the bind
/// state copied them too.
#[cfg(not(m2v_system_alloc))]
const ZEROED_COLORS_EXTRA: f64 = 0.3;
/// [`ZEROED_COLORS_EXTRA`] in a build with debug assertions, where each draw's checks build its
/// state's whole bind shape (`LiveBind::shape`), the colors it zeroes included. Measured: 0.80
/// on lavapipe; 1.80 where the settings copied them, on lavapipe and on ANV (UHD 770).
#[cfg(not(m2v_system_alloc))]
const ZEROED_COLORS_EXTRA_DEBUG: f64 = 1.3;

/// Draws into a pass with a color their pipeline state lacks, which plain Metal zeroes where a
/// draw covers it (`BindState::zeroed_colors`), allocate at most [`ZEROED_COLORS_EXTRA`] each
/// beyond the same draws into a pass of the state's colors alone, past a first command buffer that
/// made the pipelines: an unchanged set of colors to zero is read in place, and copied neither
/// into the draw's settings nor into its bind state again. Each draw binds what the draw before it bound but its column's
/// bytes; the target is the texture's, and the lacked color zero where the draws covered it.
#[cfg(not(m2v_system_alloc))]
#[test]
fn draws_into_a_pass_color_their_state_lacks_copy_no_color_set() {
    use crate::worker::alloc::thread_allocations;
    // Enough draws that what grows by doubling (the command buffer's lists) adds little to each.
    const DRAWS: u64 = 64;
    on_device(|device, _| {
        let scene = Scene::new(device);
        let red = solid(device, PixelFormat::RGBA8Unorm, RED);
        let mut lacked = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
        lacked.usage = TextureUsage::RENDER_TARGET;
        lacked.resource_options = ResourceOptions::shared();
        let lacked = device.new_texture(&lacked).unwrap();
        // The allocations each draw past the first two makes, of a pass of the target and, where
        // given, a second color cleared to `CLEAR`; and the command buffer.
        let draws = |second: Option<&Texture>| {
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut pass = RenderPassDescriptor::new();
            for (at, texture) in [Some(&scene.target), second].into_iter().enumerate() {
                let color = &mut pass.color_attachments[at];
                color.texture = texture.cloned();
                color.load_action = LoadAction::Clear;
                color.store_action = StoreAction::Store;
                color.clear_color = ClearColor::new(0.0, 0.0, 1.0, 1.0);
            }
            let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.set_render_pipeline_state(&scene.pipeline).unwrap();
            encoder
                .set_fragment_sampler_state(Some(&scene.sampler), 0)
                .unwrap();
            encoder.set_fragment_texture(Some(&red), 0).unwrap();
            let mut before = None;
            for at in 0..DRAWS {
                encoder.set_vertex_bytes(&column(at % SIDE), 0).unwrap();
                draw(&mut encoder);
                if at == 1 {
                    before = thread_allocations();
                }
            }
            let each = thread_allocations()
                .zip(before)
                .map(|(after, before)| (after - before) as f64 / (DRAWS - 2) as f64);
            encoder.end_encoding().unwrap();
            (each, command_buffer)
        };
        // Every pipeline made first, in a command buffer of each pass.
        for second in [None, Some(&lacked)] {
            let (_, command_buffer) = draws(second);
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        pipelines_settled(device);
        let (own, command_buffer) = draws(None);
        assert!(scene.pixels(&command_buffer).iter().all(|&p| p == RED));
        let (beside, command_buffer) = draws(Some(&lacked));
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
        let mut bytes = vec![0u8; (4 * SIDE * SIDE) as usize];
        let region = Region::new_2d(0, 0, SIDE, SIDE);
        lacked
            .get_bytes(&mut bytes, 4 * SIDE, 0, region, 0, 0)
            .unwrap();
        assert!(bytes.iter().all(|&b| b == 0), "{bytes:?}");
        let (Some(own), Some(beside)) = (own, beside) else {
            eprintln!("DRAW-WORK {}: allocations not counted", device.name());
            return;
        };
        eprintln!(
            "DRAW-WORK {}: {beside:.2} allocations a draw beside a lacked color, {own:.2} without",
            device.name()
        );
        let bound = match cfg!(debug_assertions) {
            true => ZEROED_COLORS_EXTRA_DEBUG,
            false => ZEROED_COLORS_EXTRA,
        };
        assert!(
            beside - own <= bound,
            "{beside:.2} allocations a draw beside a lacked color, {own:.2} without"
        );
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

/// A fragment table that binds two textures, one slot rebound mid-encoder to a texture the program
/// lets go of after its draw: each draw that changed the slot retains the table, and one that
/// changed nothing does not; the texture lives until the command buffer completes (its column is
/// its own, and the validation layer reports no use of a destroyed object), as the other slot's,
/// held from the first retain, does. The columns left of the first draw's keep the clear.
#[test]
fn a_slot_rebound_to_a_texture_let_go_after_its_draw_holds_it_until_completion() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let green = solid(device, PixelFormat::RGBA8Unorm, GREEN);
        // At slot 1, which the shader does not read: bound, and so held, throughout.
        let unread = solid(device, PixelFormat::RGBA8Unorm, CLEAR);
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&unread), 1).unwrap();
        let mut first = None;
        for c in 1..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            if c % 2 == 1 {
                let red = solid(device, PixelFormat::RGBA8Unorm, RED);
                encoder.set_fragment_texture(Some(&red), 0).unwrap();
                draw(&mut encoder);
                first.get_or_insert_with(|| device.work_counts());
                // The program's last handle goes, and the slot is rebound: the encoder alone
                // holds it now.
                drop(red);
                encoder.set_fragment_texture(Some(&green), 0).unwrap();
            } else {
                // The rebound slot's draw, then one that changed nothing.
                draw(&mut encoder);
                draw(&mut encoder);
            }
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        // After the first draw: each odd column's draw and each even column's first.
        let changed = SIDE - 2;
        assert_eq!(grew(&first, &rest, |c| c.argument_tables_retained), changed);
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

/// Draws that each set other vertex bytes and rebind the fragment table's texture slot 0 in turn
/// to one of two textures that look alike (one format, size, usage and texels), the table also
/// binding a buffer and, at slot 1, a texture the shader does not read: each draw retains the
/// table, holding what its last retain held and taking the rest. The ended encoder lists the
/// target, the buffer and the three textures (`CommandBuffer::ended_encoders`), whether a draw
/// took one or held it already; each column is the textures' color.
#[test]
fn draws_rebinding_a_slot_to_a_texture_that_looks_alike_list_each_object() {
    on_device(|device, _| {
        let scene = Scene::new(device);
        let alike = [
            solid(device, PixelFormat::RGBA8Unorm, RED),
            solid(device, PixelFormat::RGBA8Unorm, RED),
        ];
        // Bound throughout, and read by no draw: held from the first retain on.
        let unread = solid(device, PixelFormat::RGBA8Unorm, CLEAR);
        let unread_buffer = buffer_with(device, &[0; 16]).unwrap();
        let (command_buffer, mut encoder) = scene.encoder(device);
        encoder.set_fragment_texture(Some(&unread), 1).unwrap();
        encoder
            .set_fragment_buffer(Some(&unread_buffer), 0, 0)
            .unwrap();
        let mut first = None;
        for c in 0..SIDE {
            encoder.set_vertex_bytes(&column(c), 0).unwrap();
            encoder
                .set_fragment_texture(Some(&alike[(c % 2) as usize]), 0)
                .unwrap();
            draw(&mut encoder);
            first.get_or_insert_with(|| device.work_counts());
        }
        let (first, rest) = (first.unwrap(), device.work_counts());
        encoder.end_encoding().unwrap();
        assert_eq!(
            grew(&first, &rest, |c| c.argument_tables_retained),
            SIDE - 1
        );
        let ended = command_buffer.ended_encoders();
        assert_eq!(ended.len(), 1);
        let used = &ended[0].used_resources;
        let want = [&scene.target, &alike[0], &alike[1], &unread];
        assert_eq!(used.textures.len(), want.len(), "{:?}", used.textures);
        for (at, texture) in want.into_iter().enumerate() {
            assert!(used.textures.contains(texture), "texture {at} not listed");
        }
        assert_eq!(used.buffers, std::slice::from_ref(&unread_buffer));
        assert!(used.heaps.is_empty(), "{:?}", used.heaps);
        let pixels = scene.pixels(&command_buffer);
        assert!(pixels.iter().all(|&p| p == RED), "{pixels:?}");
    });
}
