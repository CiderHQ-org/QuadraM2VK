//! Constant buffers read through uniform views at bind offsets the device's uniform descriptor
//! alignment does and does not divide, in the vertex, fragment and kernel stages: each read is
//! the bound bytes', whether the slot's `SpecKey::RangeAtBindOffset` specialization reads the
//! Metal offset alone (the range starts at the bind offset: offset 0 or 256 on every device, and
//! inline bytes) or the residual plus it (offsets 4, 20, 36, 260 and 516, which no device's
//! alignment of at least 16 divides, and every offset of a slot a state once found bound at
//! such an offset, `argument_table::AtBindViews`). A slot bound at offsets that move keeps one
//! shape and one pipeline. A state's creation makes its predicted key's residual form, every slot
//! read at its residual, and a draw whose own key is not made draws with it at once while its own
//! is made on the device's pool (`pipeline/render/residual.rs`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are those of the owned
//! fixture `validation/fixtures/public/engine/ubo_bind_offsets.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    BlitOption, Buffer, CommandQueue, Device, LoadAction, Origin, PipelineOptions,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, ScissorRect, Size, StoreAction,
    Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;

/// The target's side, in pixels.
const SIDE: u64 = 4;

/// The constants' floats: float `i` is `i / 255`, whose `RGBA8Unorm` code is `i`.
const FLOATS: usize = 256;

/// Where a draw reads its constants: the fragment's at byte `.0`, the vertex's at byte `.1`, of
/// the constants buffer, or both as inline bytes from those offsets of it.
#[derive(Clone, Copy, Debug)]
enum Bound {
    Buffer(u64, u64),
    Bytes(u64, u64),
}

impl Bound {
    /// The pixel the draw writes: `c[0], c[1], c[2]` at the fragment's offset, `k[1]` at the
    /// vertex's.
    fn pixel(self) -> [u8; 4] {
        let (Bound::Buffer(f, v) | Bound::Bytes(f, v)) = self;
        let code = |at: u64| (at / 4) as u8;
        [code(f), code(f + 4), code(f + 8), code(v + 4)]
    }
}

fn library(device: &Device) -> crate::Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/ubo_bind_offsets.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    device.new_library_with_compile_library(library)
}

/// The fixture's render state, one `RGBA8Unorm` target.
fn state(device: &Device) -> RenderPipelineState {
    let library = library(device);
    let vertex = library
        .new_function_with_name("ubo_offsets_vertex")
        .unwrap();
    let fragment = library
        .new_function_with_name("ubo_offsets_fragment")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

fn constants() -> Vec<u8> {
    (0..FLOATS)
        .flat_map(|i| (i as f32 / 255.0).to_le_bytes())
        .collect()
}

/// The bytes of `buffer`, which no command uses now.
fn bytes_of(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// A pass that clears a new `RGBA8Unorm` target of [`SIDE`] by [`SIDE`] and stores it, and the
/// target.
fn pass(device: &Device) -> (RenderPassDescriptor, Texture) {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    d.usage = TextureUsage::RENDER_TARGET;
    let target = device.new_texture(&d).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Clear;
    color.store_action = StoreAction::Store;
    (pass, target)
}

/// `device`'s work counts once its pipeline pool holds no piece (`WorkCounts::pipeline_pool_busy`):
/// a draw that took its key's residual form does not wait for its own key's making.
fn idle(device: &Device) -> crate::WorkCounts {
    let until = std::time::Instant::now() + WAIT;
    loop {
        let counts = device.work_counts();
        if counts.pipeline_pool_busy == 0 {
            return counts;
        }
        assert!(
            std::time::Instant::now() < until,
            "the device's pipeline pool still holds {} pieces after {WAIT:?}",
            counts.pipeline_pool_busy
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Draw each of `cases` into a target of its own in one command buffer, and check every pixel.
fn draw(device: &Device, queue: &CommandQueue, state: &RenderPipelineState, cases: &[Bound]) {
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    let per = SIDE * SIDE * 4;
    let back = device
        .new_buffer(per * cases.len() as u64, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for (n, case) in cases.iter().enumerate() {
        let (pass, target) = pass(device);
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(state).unwrap();
        match *case {
            Bound::Buffer(f, v) => {
                encoder.set_fragment_buffer(Some(&constants), f, 0).unwrap();
                encoder.set_vertex_buffer(Some(&constants), v, 1).unwrap();
            }
            Bound::Bytes(f, v) => {
                let from = |at: u64| &data[at as usize..at as usize + 16];
                encoder.set_fragment_bytes(from(f), 0).unwrap();
                encoder.set_vertex_bytes(from(v), 1).unwrap();
            }
        }
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &back,
            per * n as u64,
            SIDE * 4,
            per,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let bytes = bytes_of(&back);
    for (case, pixels) in cases.iter().zip(bytes.chunks(per as usize)) {
        for (p, pixel) in pixels.chunks(4).enumerate() {
            assert_eq!(pixel, case.pixel(), "{case:?}: pixel {p}");
        }
    }
}

/// A draw reads its vertex and fragment constants at their bind offsets in either form. Draws
/// whose ranges start at their bind offsets share one pipeline; each other combination of the two
/// slots' forms, on a state of its own (a state's slot found at such an offset once is read in
/// the other form from then on), makes one of its own (the path ran), but for both slots read at
/// their residuals, the creation's residual form. Each making a draw began is made on the device's
/// pool, which is waited for before the pipelines made are counted.
#[test]
fn vertex_and_fragment_constants_read_at_every_bind_offset() {
    let scratch = Scratch::new("uniform-bind-offsets");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let [state, fragment, vertex, both] = [(); 4].map(|()| self::state(&device));
    let made = || idle(&device).variant_pipelines_made;
    let start = made();
    // Offsets every device's alignment divides, and inline bytes: one pipeline, which the state's
    // creation did not make (it made the residual form).
    draw(
        &device,
        &queue,
        &state,
        &[
            Bound::Buffer(0, 0),
            Bound::Buffer(256, 512),
            Bound::Bytes(4, 20),
            Bound::Bytes(260, 36),
        ],
    );
    let aligned = made();
    assert_eq!(aligned - start, 1, "pipelines made");
    // Offsets no device's alignment divides, on one slot, the other, and both, each on a state of
    // its own: a pipeline for each of the first two, and none for both, the creation's residual
    // form, and the states keep them.
    draw(&device, &queue, &fragment, &[Bound::Buffer(4, 0)]);
    draw(&device, &queue, &vertex, &[Bound::Buffer(0, 4)]);
    draw(
        &device,
        &queue,
        &both,
        &[
            Bound::Buffer(4, 20),
            Bound::Buffer(20, 36),
            Bound::Buffer(260, 516),
        ],
    );
    let other = made();
    assert_eq!(other - aligned, 2, "one pipeline per other combination");
    // Offsets some devices' alignments divide, and offsets every device's divides, on slots
    // found at another before: read in the other form, in the pipelines made.
    let moved = [
        Bound::Buffer(16, 32),
        Bound::Buffer(64, 48),
        Bound::Buffer(128, 192),
        Bound::Buffer(0, 0),
        Bound::Buffer(256, 512),
    ];
    draw(&device, &queue, &both, &moved);
    assert_eq!(
        made(),
        other,
        "pipelines made for a state's slots found moved before"
    );
    // On the state of aligned offsets, they are read in either form.
    draw(&device, &queue, &state, &moved);
}

/// A fragment constant slot bound at offsets moving through the multiples of 16 up to 256 in one
/// encoder reads each draw's bytes; a second pass through them makes no shape, settles nothing,
/// and makes and looks up no pipeline. The first makes two shapes at most: where the device's
/// view base alignment divides the first offset, 256, and not the next, which leaves the slot out
/// of those that tell it apart from then on (`argument_table::AtBindViews`). A first command
/// buffer draws the offsets once and completes, and the device's pipeline pool is waited for, so
/// their pipelines are made before the counted passes: a draw of a pipeline its device's pool still
/// makes looks the key up (ANV makes them so while the draws record, and a draw that takes its
/// key's residual form meanwhile does not wait for them).
#[test]
fn constants_bound_at_moving_offsets_keep_one_shape() {
    let scratch = Scratch::new("uniform-moving-offsets");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    // One draw per pixel, each through a scissor of its own.
    let offsets: Vec<u64> = std::iter::once(256)
        .chain((1..16).map(|n| 16 * n))
        .collect();
    let per = SIDE * SIDE * 4;
    // `passes` passes through the offsets in one encoder, each pass's starting work counts, the
    // counts after the last, and the target's bytes once the command buffer completed.
    let run = |passes: usize| {
        let (pass, target) = pass(&device);
        let back = device.new_buffer(per, ResourceOptions::shared()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&constants), 0, 1).unwrap();
        let mut counts = Vec::new();
        for _ in 0..passes {
            counts.push(device.work_counts());
            for (p, &offset) in (0..).zip(&offsets) {
                let (x, y) = (p % SIDE, p / SIDE);
                let rect = ScissorRect {
                    x,
                    y,
                    width: 1,
                    height: 1,
                };
                encoder.set_scissor_rect(rect).unwrap();
                encoder
                    .set_fragment_buffer(Some(&constants), offset, 0)
                    .unwrap();
                encoder
                    .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
                    .unwrap();
            }
        }
        counts.push(device.work_counts());
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            &target,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &back,
            0,
            SIDE * 4,
            per,
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let bytes = bytes_of(&back);
        for (p, (pixel, &offset)) in bytes.chunks(4).zip(&offsets).enumerate() {
            let case = Bound::Buffer(offset, 0);
            assert_eq!(pixel, case.pixel(), "{case:?}: pixel {p}");
        }
        counts
    };
    let made_first = run(1);
    idle(&device);
    let shapes = made_first[1].draw_shapes_made - made_first[0].draw_shapes_made;
    assert!(shapes <= 2, "{shapes} shapes made on the first pass");
    let counts = run(2);
    let (second, after) = (&counts[1], &counts[2]);
    let made = |c: &crate::WorkCounts| {
        [
            c.draw_shapes_made,
            c.draws_settled,
            c.variant_pipelines_made,
            c.graphics_pipelines_looked_up,
        ]
    };
    assert_eq!(
        made(after),
        made(second),
        "shapes made, draws settled, pipelines made and looked up on the second pass"
    );
}

/// A state's creation makes its predicted key's residual form, each slot read at its residual
/// (`pipeline/render/residual.rs`). With the device's pool held, so that a draw waiting for a
/// making there could not complete, a draw binding both slots at offset 0, which every device's
/// alignment divides, and a draw binding the fragment slot at 4, which none divides, both draw at
/// once with the creation's pipeline and read their bytes: their own keys' makings wait in the
/// pool, nothing waiting for them. Once the pool runs they are made, and a later draw of the
/// second key (the state reads the fragment slot at its residual from then on) takes its own.
#[test]
fn draws_take_the_residual_form_while_their_own_keys_are_made() {
    let scratch = Scratch::new("uniform-residual-form");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    let state = state(&device);
    let created = device.work_counts();
    assert_eq!(
        created.creation_pipelines_made - before.creation_pipelines_made,
        1,
        "the creation's residual form"
    );
    let hold = crate::pipeline::pool::held::hold();
    draw(
        &device,
        &queue,
        &state,
        &[Bound::Buffer(0, 0), Bound::Buffer(4, 0)],
    );
    let drawn = device.work_counts();
    drop(hold);
    assert_eq!(
        drawn.residual_form_draws - created.residual_form_draws,
        2,
        "draws that took the residual form"
    );
    assert_eq!(
        drawn.residual_form_makings - created.residual_form_makings,
        2,
        "makings begun that nothing waits for"
    );
    assert_eq!(
        drawn.creation_pipelines_drawn - created.creation_pipelines_drawn,
        1,
        "the creation's pipeline, drawn"
    );
    assert_eq!(
        drawn.variant_pipelines_made, created.variant_pipelines_made,
        "pipelines made while the pool was held"
    );
    let made = idle(&device);
    assert_eq!(
        made.variant_pipelines_made - drawn.variant_pipelines_made,
        2,
        "the draws' own keys, made on the pool"
    );
    draw(&device, &queue, &state, &[Bound::Buffer(0, 0)]);
    let later = device.work_counts();
    assert_eq!(
        later.residual_form_draws, made.residual_form_draws,
        "a later draw of a key made took the residual form"
    );
    assert_eq!(
        later.variant_pipelines_made, made.variant_pipelines_made,
        "pipelines made for a later draw of a key made"
    );
}

/// A slot bound to nothing is read at its residual, its residual 0 (`RangeAtBindOffset` 0, as
/// `bind::plan::slots::range_at_bind_offset` has it): the key of a draw that leaves both constant
/// slots unbound is the creation's, its residual form. With the device's pool held, such a draw
/// takes the creation's pipeline itself, neither through the residual form nor by a making of its
/// own: nothing is made, handed to the pool or begun, and its command buffer completes while the
/// pool is still held. (Its pixels read unbound slots, which Metal leaves undefined: unchecked.)
#[test]
fn a_draw_that_leaves_constant_slots_unbound_takes_the_creations_key() {
    let scratch = Scratch::new("uniform-unbound");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let state = state(&device);
    let created = idle(&device);
    let hold = crate::pipeline::pool::held::hold();
    let (pass, _target) = pass(&device);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    let drawn = device.work_counts();
    assert_eq!(
        drawn.creation_pipelines_drawn - created.creation_pipelines_drawn,
        1,
        "the creation's pipeline, drawn"
    );
    assert_eq!(
        (
            drawn.residual_form_draws - created.residual_form_draws,
            drawn.residual_form_makings - created.residual_form_makings,
            drawn.pipelines_pooled - created.pipelines_pooled,
            drawn.variant_pipelines_made - created.variant_pipelines_made,
        ),
        (0, 0, 0, 0),
        "residual-form draws, their makings, pieces pooled and pipelines made"
    );
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(WAIT)
        .expect("the draw's command buffer waits for no making");
    assert_eq!(command_buffer.error(), None);
    assert_eq!(
        device.work_counts().pipeline_pool_busy,
        0,
        "nothing waits in the pool"
    );
    drop(hold);
}

/// A kernel reads its constants at their bind offset in either form.
#[test]
fn kernel_constants_read_at_every_bind_offset() {
    let scratch = Scratch::new("uniform-bind-offsets-kernel");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let function = library(&device)
        .new_function_with_name("ubo_offsets_kernel")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let data = constants();
    let constants = device
        .new_buffer_with_bytes(&data, ResourceOptions::shared())
        .unwrap();
    // `None`: inline bytes from offset 8.
    let offsets = [
        Some(0),
        Some(4),
        Some(16),
        Some(20),
        Some(64),
        Some(260),
        None,
    ];
    let outputs: Vec<Buffer> = offsets
        .iter()
        .map(|_| device.new_buffer(16, ResourceOptions::shared()).unwrap())
        .collect();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    for (offset, output) in offsets.iter().zip(&outputs) {
        match offset {
            Some(at) => encoder.set_buffer(Some(&constants), *at, 0).unwrap(),
            None => encoder.set_bytes(&data[8..24], 0).unwrap(),
        }
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder
            .dispatch_threadgroups(Size::new(1, 1, 1), Size::new(1, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    for (offset, output) in offsets.iter().zip(&outputs) {
        let at = offset.unwrap_or(8) as usize / 4;
        let words = bytes_of(output);
        let read = |w: usize| f32::from_le_bytes(words[w * 4..w * 4 + 4].try_into().unwrap());
        let want = |i: usize| i as f32 / 255.0;
        assert_eq!(
            [read(0), read(1)],
            [want(at), want(at + 2)],
            "offset {offset:?}"
        );
    }
}
