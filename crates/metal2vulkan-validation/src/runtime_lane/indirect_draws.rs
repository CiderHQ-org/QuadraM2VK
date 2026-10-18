//! Device tests of indirect draws (its third part) on the device the cfg selects, under the
//! validation layer.
//!
//! Each draw is one that a Metal probe measured (an M4 with macOS 27.0.1, each case with and
//! without Metal's API validation), drawn with the owned fixture `rt6_render_state`'s fullscreen
//! triangle (vertices 0 to 2; vertex 3 and up land outside the target) into a 4² target, and
//! compared with the texels Metal stored: the arguments' vertex start, index start, base vertex and
//! the index buffer offset are taken; no instances or no vertices draw nothing; arguments a blit
//! wrote earlier in the command buffer are the ones drawn. The indexed form draws on every device:
//! where nothing bounds its index fetch (no `robustBufferAccess2`) through the index bounds pass.
//! Misaligned argument and index offsets, which Metal's validation stops, draw what plain Metal
//! drew from the offset rounded down; arguments past the buffer, which plain Metal reads past it,
//! are refused typed.
//!
//! An indexed indirect draw whose arguments or indices an earlier draw of the encoder writes (the
//! rasterization-disabled `icb_render_replay_template`) draws as a Metal probe measured (an M3 Max
//! and an M4): behind a buffer barrier after and before the vertex stage, what the earlier draw
//! wrote, through an index bounds pass before the draw's own render pass instance where the device
//! takes one. Without the barrier, which Metal does not order: before any barrier of the encoder,
//! the bytes from before the write, which Metal's draw read in every run of a stress probe,
//! through a copy taken before the pass, announced once (`unordered-draw-bytes`), on
//! every device; after a barrier, where a rewrite since it was read in some runs,
//! the index bounds pass is refused typed.

use super::degrade_notices::{
    assert_made_once, forced, made_render_kinds, shared_in_system_memory, UNORDERED_DRAWS,
};
use super::render_state::{
    quiet, rows, texture, validated, validated_checking_synchronization, Probe, EMPTY, FULL, PASS,
    WHITE,
};
use super::{buffer_with, read_buffer};
use crate::air_library::AirLibrary;
use metal2vulkan::device::VkFeature;
use metal2vulkan::metal::{BarrierScope, IndexType, PixelFormat, PrimitiveType, RenderStages};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    Buffer, ClearColor, Degradation, Device, DrawProblem, DriverFact, Error, Invalid, LoadAction,
    PipelineOptions, Problem, RenderCommandEncoder, RenderPassDescriptor, RenderPipelineState,
    RuntimeGap, Size, StoreAction,
};
use std::collections::BTreeSet;

/// A buffer of little-endian `u32`s.
fn u32s(probe: &Probe, words: &[u32]) -> Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    buffer_with(&probe.device, &bytes).unwrap()
}

/// A buffer of little-endian `u16`s.
fn u16s(probe: &Probe, indices: &[u16]) -> Buffer {
    let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    buffer_with(&probe.device, &bytes).unwrap()
}

/// The probe's pipeline and depth buffer set on `encoder`, as before each indirect draw.
fn prepare(probe: &Probe, encoder: &mut RenderCommandEncoder, held: &mut Vec<Buffer>) {
    encoder
        .set_render_pipeline_state(&probe.pipeline(WHITE))
        .unwrap();
    let depth = probe.depth(0.5, 0.0);
    encoder.set_vertex_buffer(Some(&depth), 0, 0).unwrap();
    held.push(depth);
}

/// Each probe draw stores the texels Metal stored (the module docs).
#[test]
fn indirect_draws_store_metals_texels() {
    let (messages, _cache, probe) = validated();
    let full = rows(&FULL);
    let empty = rows(&EMPTY);
    // MTLDrawPrimitivesIndirectArguments {vertexCount, instanceCount, vertexStart, baseInstance}.
    for (name, words, offset, metal) in [
        ("ind_draw", vec![3, 1, 0, 0], 0, &full),
        ("ind_vertex_start", vec![9, 9, 9, 9, 3, 1, 3, 0], 16, &empty),
        ("ind_no_instances", vec![3, 0, 0, 0], 0, &empty),
        ("ind_no_vertices", vec![0, 1, 0, 0], 0, &empty),
    ] {
        let arguments = u32s(&probe, &words);
        let stored = probe.run(PASS, |p, e, h| {
            prepare(p, e, h);
            e.draw_primitives_indirect(PrimitiveType::Triangle, &arguments, offset)
                .unwrap();
        });
        assert_eq!(&stored.color, metal, "{name}");
    }
    // MTLDrawIndexedPrimitivesIndirectArguments {indexCount, instanceCount, indexStart,
    // baseVertex, baseInstance}, over 16-bit indices.
    let five = 5u32.wrapping_neg();
    for (name, indices, index_offset, words, metal) in [
        (
            "ind_indexed",
            vec![5, 6, 7, 0, 1, 2],
            0,
            vec![3, 1, 3, 0, 0],
            &full,
        ),
        (
            "ind_indexed_base_vertex",
            vec![5, 6, 7, 0, 1, 2],
            0,
            vec![3, 1, 0, five, 0],
            &full,
        ),
        (
            "ind_indexed_buffer_offset",
            vec![5, 6, 7, 0, 1, 2],
            6,
            vec![3, 1, 0, 0, 0],
            &full,
        ),
        (
            "ind_indexed_no_instances",
            vec![0, 1, 2],
            0,
            vec![3, 0, 0, 0, 0],
            &empty,
        ),
    ] {
        let index_buffer = u16s(&probe, &indices);
        let arguments = u32s(&probe, &words);
        let mut answer = None;
        let stored = probe.run(PASS, |p, e, h| {
            prepare(p, e, h);
            answer = Some(e.draw_indexed_primitives_indirect(
                PrimitiveType::Triangle,
                IndexType::UInt16,
                &index_buffer,
                index_offset,
                &arguments,
                0,
            ));
        });
        assert_eq!(answer, Some(Ok(())), "{name}");
        assert_eq!(&stored.color, metal, "{name}");
    }
    // Probe 5's ind_blit_written: arguments a blit wrote earlier in the command buffer.
    let arguments = u32s(&probe, &[0, 0, 0, 0]);
    let written = u32s(&probe, &[3, 1, 0, 0]);
    let color = texture(&probe.device, PixelFormat::RGBA8Unorm, 4);
    let command_buffer = probe.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&written, 0, &arguments, 0, 16)
        .unwrap();
    blit.end_encoding().unwrap();
    let mut descriptor = RenderPassDescriptor::new();
    let target = &mut descriptor.color_attachments[0];
    target.texture = Some(color.clone());
    target.load_action = LoadAction::Clear;
    target.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    target.store_action = StoreAction::Store;
    let mut held = Vec::new();
    let mut e = command_buffer.render_command_encoder(&descriptor).unwrap();
    prepare(&probe, &mut e, &mut held);
    e.draw_primitives_indirect(PrimitiveType::Triangle, &arguments, 0)
        .unwrap();
    e.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
    drop(held);
    assert_eq!(
        super::render_state::texels(&color, 4),
        full,
        "ind_blit_written"
    );
    quiet(probe, &messages);
}

/// Misaligned offsets, which Metal's validation stops, draw what plain Metal drew from the
/// offset rounded down (probes 5 and 5d): arguments `{3, 1, 0, 0}` at offset 2 the full triangle,
/// indexed arguments `{0, 3, 1, 0, 0}` at offset 2 nothing (no indices), and 16-bit indices `0, 1,
/// 2` at offset 1 the full triangle, indirect and direct; read where they lie, each would draw
/// otherwise. Arguments past the buffer, which plain Metal reads past it, are refused typed, and
/// nothing draws.
#[test]
fn indirect_draws_read_misaligned_offsets_as_metal_reads_them() {
    let (messages, _cache, probe) = validated();
    let draw = |call, problem| Err(Error::Invalid(Invalid::new(call, Problem::Draw(problem))));
    let plain = "draw_primitives_indirect";
    let indexed = "draw_indexed_primitives_indirect";
    let full = rows(&FULL);
    let empty = rows(&EMPTY);
    let arguments = u32s(&probe, &[3, 1, 0, 0, 0, 0]);
    let indexed_arguments = u32s(&probe, &[0, 3, 1, 0, 0, 0]);
    let short = u32s(&probe, &[3, 1, 0, 0]);
    let indexed_short = u32s(&probe, &[3, 1, 0, 0, 0]);
    let index_buffer = u16s(&probe, &[0, 1, 2, 0]);
    let indexed_draw = |e: &mut RenderCommandEncoder, index_offset, args: &Buffer, offset| {
        e.draw_indexed_primitives_indirect(
            PrimitiveType::Triangle,
            IndexType::UInt16,
            &index_buffer,
            index_offset,
            args,
            offset,
        )
    };
    let mut answer = None;
    let stored = probe.run(PASS, |p, e, h| {
        prepare(p, e, h);
        answer = Some(e.draw_primitives_indirect(PrimitiveType::Triangle, &arguments, 2));
    });
    assert_eq!(answer, Some(Ok(())), "ind_offset_misaligned");
    assert_eq!(stored.color, full, "ind_offset_misaligned");
    let stored = probe.run(PASS, |p, e, h| {
        prepare(p, e, h);
        answer = Some(indexed_draw(e, 0, &indexed_arguments, 2));
    });
    assert_eq!(answer, Some(Ok(())), "ind_indexed_offset_misaligned");
    assert_eq!(stored.color, empty, "ind_indexed_offset_misaligned");
    let stored = probe.run(PASS, |p, e, h| {
        prepare(p, e, h);
        answer = Some(indexed_draw(e, 1, &indexed_short, 0));
    });
    assert_eq!(answer, Some(Ok(())), "ind_index_offset_misaligned");
    assert_eq!(stored.color, full, "ind_index_offset_misaligned");
    // The direct draw reads its indices so too (as our probe's commands did).
    let stored = probe.run(PASS, |p, e, h| {
        prepare(p, e, h);
        answer = Some(e.draw_indexed_primitives(
            PrimitiveType::Triangle,
            3,
            IndexType::UInt16,
            &index_buffer,
            1,
            1,
            0,
            0,
        ));
    });
    assert_eq!(answer, Some(Ok(())), "index_offset_misaligned");
    assert_eq!(stored.color, full, "index_offset_misaligned");
    // Arguments past the buffer (probe 5), from the given offset and from one rounded down.
    let stored = probe.run(PASS, |p, e, h| {
        prepare(p, e, h);
        assert_eq!(
            e.draw_primitives_indirect(PrimitiveType::Triangle, &short, 4),
            draw(plain, DrawProblem::IndirectBufferRange)
        );
        assert_eq!(
            e.draw_primitives_indirect(PrimitiveType::Triangle, &short, 5),
            draw(plain, DrawProblem::IndirectBufferRange)
        );
        assert_eq!(
            indexed_draw(e, 0, &indexed_short, 4),
            draw(indexed, DrawProblem::IndirectBufferRange)
        );
    });
    assert_eq!(stored.color, empty);
    quiet(probe, &messages);
}

/// The probe's writer: `icb_render_replay_template` with rasterization disabled, whose vertex
/// `vid` writes `params[0] * (vid + 1) + params[1]` to word `vid` of buffer 1, in the probe pass.
fn writer(probe: &Probe) -> RenderPipelineState {
    probe
        .device
        .new_render_pipeline_state(&writer_descriptor(probe), &PipelineOptions::default())
        .unwrap()
}

/// The descriptor of [`writer`]'s state.
fn writer_descriptor(probe: &Probe) -> RenderPipelineDescriptor {
    let bitcode = llvm_assemble(&engine_dir().join("icb_render_replay_template.ll")).unwrap();
    let library = probe.device.new_library_with_air(&bitcode).unwrap();
    let function = library
        .new_function_with_name("icb_render_replay_template")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor
}

/// Whether `device` checks an indexed indirect draw's indices in the index bounds pass
/// (`indirect-index-bounds-pass`): it lacks `robustBufferAccess2`, or its driver is not measured
/// to read zero past an index buffer's bound range (`DriverFact::RobustIndexFetchReadsZero`).
fn bounds_pass(device: &Device) -> bool {
    !(device
        .enabled()
        .features
        .contains(&VkFeature::RobustBufferAccess2)
        && device
            .properties()
            .driver_facts
            .contains(&DriverFact::RobustIndexFetchReadsZero))
}

/// The kinds the steps of an indexed indirect draw's test force beside [`UNORDERED_DRAWS`],
/// where the device has not announced them: the pass's split (a buffer barrier between draws, or
/// a tile dispatch: Vulkan orders neither inside a render pass instance), the index bounds pass
/// where the device takes it ([`bounds_pass`]), the white pipeline's and the writer's makings
/// (and those of `made`), and the `Shared` buffers' system memory.
fn forced_beside(probe: &Probe, made: BTreeSet<Degradation>) -> BTreeSet<Degradation> {
    let device = &probe.device;
    let mut kinds = forced(&[
        (Degradation::RenderPassSplit, true),
        (Degradation::IndirectIndexBoundsPass, bounds_pass(device)),
        (
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(device),
        ),
    ]);
    kinds.extend(probe.made(WHITE));
    kinds.extend(made_render_kinds(device, &writer_descriptor(probe)));
    kinds.extend(made);
    kinds
}

/// What a probe 2 case encodes, in order.
#[derive(Clone, Copy)]
enum Step {
    /// The writer's two vertices into buffer `out` (0 the arguments, 1 the indices, 2 another)
    /// with `params`.
    Write { out: usize, params: [u32; 2] },
    /// `memoryBarrierWithScope:` of buffers after and before the vertex stage.
    Barrier,
    /// The probe's small white triangle (`rt6_small`).
    Small,
    /// The indexed indirect draw of the white fullscreen triangle, from buffer 1 at offset 0 with
    /// buffer 0's arguments at offset 0.
    Draw,
}

/// The writer's params: `[3, 1]` into the arguments' first two words; 16-bit indices `[0, 1, 2,
/// 0]` (words `0x0001_0000` and `0x0000_0002`); `[0, 0]` into the arguments' first two words.
const ARGUMENTS: [u32; 2] = [0xFFFF_FFFE, 5];
const INDICES: [u32; 2] = [0xFFFF_0002, 0x0001_FFFE];
const ZEROS: [u32; 2] = [0, 0];

/// An indexed indirect draw whose arguments or indices an earlier draw of the encoder writes
/// draws as probe 2 measured (the module docs); without a barrier, before any barrier, the bytes
/// from before the write (a probe), announced once; a rewrite after a barrier, where the
/// device takes the index bounds pass, is refused `NotImplemented(IndirectDrawArguments)`.
#[test]
fn an_indexed_indirect_draw_reads_what_a_draw_wrote_behind_a_barrier() {
    let (messages, _cache, probe) = validated();
    let writer = writer(&probe);
    let bounds = bounds_pass(&probe.device);
    let name = probe.device.name().to_owned();
    let none = [0, 0, 0, 0, 0];
    let three = [3, 1, 0, 0, 0];
    let (full, degenerate) = ([0u16, 1, 2, 0], [0u16, 0, 0, 0]);
    let args = |out| Step::Write {
        out,
        params: ARGUMENTS,
    };
    let indices = Step::Write {
        out: 1,
        params: INDICES,
    };
    let zeros = Step::Write {
        out: 0,
        params: ZEROS,
    };
    use Step::{Barrier, Draw, Small};
    // Each case: its name, the arguments and the indices it starts with, its steps, and Metal's
    // texels where the runtime draws it (probe 2: the same on the M3 Max and the M4, plain and
    // validated; without a barrier, the bytes from before the write, on both chips),
    // else `None` where it is refused with the index bounds pass.
    let full_rows = rows(&FULL);
    let empty_rows = rows(&EMPTY);
    let cases = [
        (
            "ix2_args_barrier",
            none,
            full,
            vec![args(0), Barrier, Draw],
            Some(&full_rows),
        ),
        (
            "ix2_indices_barrier",
            three,
            degenerate,
            vec![indices, Barrier, Draw],
            Some(&full_rows),
        ),
        (
            "ix2_both_barrier",
            none,
            degenerate,
            vec![args(0), indices, Barrier, Draw],
            Some(&full_rows),
        ),
        (
            "ix2_args_barrier_small",
            none,
            full,
            vec![args(0), Barrier, Small, Draw],
            Some(&full_rows),
        ),
        (
            "ix2_args_small_barrier",
            none,
            full,
            vec![Small, args(0), Barrier, Draw],
            Some(&full_rows),
        ),
        (
            "ix2_args_barrier_twice",
            none,
            full,
            vec![args(0), Barrier, Draw, zeros, Barrier, Draw],
            Some(&full_rows),
        ),
        (
            "ix2_other_nobarrier",
            three,
            full,
            vec![args(2), Draw],
            Some(&full_rows),
        ),
        (
            "ix2_args_nobarrier",
            none,
            full,
            vec![args(0), Draw],
            Some(&empty_rows),
        ),
        (
            "ix2_indices_nobarrier",
            three,
            degenerate,
            vec![indices, Draw],
            Some(&empty_rows),
        ),
        (
            "ix2_both_nobarrier",
            none,
            degenerate,
            vec![args(0), indices, Draw],
            Some(&empty_rows),
        ),
        (
            "ix2_args_barrier_rewrite",
            none,
            full,
            vec![args(0), Barrier, zeros, Draw],
            None,
        ),
    ];
    for (case, arguments, index_words, steps, metal) in cases {
        if metal.is_none() && !bounds {
            // A race Metal does not order, where the device's draw reads the arguments itself.
            continue;
        }
        let buffers = [
            u32s(&probe, &arguments),
            u16s(&probe, &index_words),
            u32s(&probe, &[0; 16]),
        ];
        let mut answers = Vec::new();
        let stored = probe.run(PASS, |p, e, h| {
            for step in &steps {
                match *step {
                    Step::Write { out, params } => {
                        let params = u32s(p, &params);
                        e.set_render_pipeline_state(&writer).unwrap();
                        e.set_vertex_buffer(Some(&params), 0, 0).unwrap();
                        e.set_vertex_buffer(Some(&buffers[out]), 0, 1).unwrap();
                        e.draw_primitives(PrimitiveType::Point, 0, 2, 1, 0).unwrap();
                        h.push(params);
                    }
                    Step::Barrier => e.memory_barrier_with_scope(
                        BarrierScope::BUFFERS,
                        RenderStages::VERTEX,
                        RenderStages::VERTEX,
                    ),
                    Step::Small => p.draw(e, h, WHITE.vertex("rt6_small"), 0.5, 0.0),
                    Step::Draw => {
                        prepare(p, e, h);
                        answers.push(e.draw_indexed_primitives_indirect(
                            PrimitiveType::Triangle,
                            IndexType::UInt16,
                            &buffers[1],
                            0,
                            &buffers[0],
                            0,
                        ));
                    }
                }
            }
        });
        let at = format!("{case} on {name} (index bounds pass {bounds})");
        match metal {
            Some(metal) => {
                assert!(answers.iter().all(Result::is_ok), "{at}: {answers:?}");
                assert_eq!(&stored.color, metal, "{at}");
            }
            None => {
                let refused = Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments));
                assert_eq!(answers.last(), Some(&refused), "{at}");
                assert_eq!(stored.color, rows(&EMPTY), "{at}");
            }
        }
    }
    // The small white triangle's pipeline beside the writer's and the white one's.
    let small = probe.made(WHITE.vertex("rt6_small"));
    assert_made_once(&messages, &UNORDERED_DRAWS, &forced_beside(&probe, small));
    quiet(probe, &messages);
}

/// A draw that writes the arguments an earlier indexed indirect draw of its pass read, behind no
/// barrier, leaves that draw's texels as Metal's (probe 2's `ix2_args_barrier_twice`: the earlier
/// draw read them first, on the M3 Max and the M4), and writes them; the validation layer, checking
/// synchronization, reports no hazard between the read and the write, which records after an
/// execution dependency from the read where the draw reads its arguments in place (no index bounds
/// pass, [`bounds_pass`]). Twice: in the second pass both pipelines are made, so the writer's draw
/// records into the indirect draw's render pass instance (in the first, a draw whose pipeline is
/// being made may record into an instance of its own, `render/inline_first.rs`).
#[test]
fn a_draw_writes_arguments_an_earlier_draw_read_after_the_read() {
    let (messages, _cache, probe) = validated_checking_synchronization();
    let writer = writer(&probe);
    let indices = u16s(&probe, &[0, 1, 2, 0]);
    for pass in 0..2 {
        let arguments = u32s(&probe, &[3, 1, 0, 0, 0]);
        let stored = probe.run(PASS, |p, e, h| {
            prepare(p, e, h);
            e.draw_indexed_primitives_indirect(
                PrimitiveType::Triangle,
                IndexType::UInt16,
                &indices,
                0,
                &arguments,
                0,
            )
            .unwrap();
            let params = u32s(p, &ZEROS);
            e.set_render_pipeline_state(&writer).unwrap();
            e.set_vertex_buffer(Some(&params), 0, 0).unwrap();
            e.set_vertex_buffer(Some(&arguments), 0, 1).unwrap();
            e.draw_primitives(PrimitiveType::Point, 0, 2, 1, 0).unwrap();
            h.push(params);
        });
        assert_eq!(stored.color, rows(&FULL), "pass {pass}");
        assert_eq!(
            read_buffer(&arguments, 0, 20).unwrap(),
            [0; 20],
            "pass {pass}"
        );
    }
    quiet(probe, &messages);
}

/// The tile writer: the owned fixture `rrs_tile_writer`, whose thread (0, 0) of each tile writes
/// what the probe's writer writes ([`writer`]) from tile buffers 0 (`params`) and 1 (`out`), in
/// a state of the probe pass's color.
fn tile_writer(probe: &Probe) -> RenderPipelineState {
    let bitcode = llvm_assemble(&engine_dir().join("rrs_tile_writer.ll")).unwrap();
    let library = probe.device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("rrs_tile_writer").unwrap();
    let mut descriptor =
        TileRenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.color_attachment_formats[0] = Some(PixelFormat::RGBA8Unorm);
    probe
        .device
        .new_tile_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// An indexed indirect draw whose arguments or indices a tile dispatch of the encoder wrote
/// before it draws the bytes from before the write, which Metal's draw read in every run of
/// a probe (an M3 Max and an M4, plain and validated): through the copy taken before the
/// pass, announced once (`unordered-draw-bytes`), on every device. Metal refuses a memory barrier
/// after the tile stage, and a barrier before the dispatch did not order the write either (the draw
/// read what a draw wrote before the barrier, not the dispatch's rewrite); the runtime's copy
/// before the pass misses that earlier write, so such a draw is refused
/// `NotImplemented(IndirectDrawArguments)`.
#[test]
fn an_indexed_indirect_draw_reads_the_bytes_from_before_a_tile_dispatchs_write() {
    let (messages, _cache, probe) = validated();
    let writer = writer(&probe);
    let tile = tile_writer(&probe);
    let name = probe.device.name().to_owned();
    let none = [0, 0, 0, 0, 0];
    let three = [3, 1, 0, 0, 0];
    let (full, degenerate) = ([0u16, 1, 2, 0], [0u16, 0, 0, 0]);
    #[derive(Clone, Copy)]
    enum TileStep {
        /// The probe's writer into buffer `out` with `params`, then a buffer barrier after and
        /// before the vertex stage.
        WriteBehindBarrier { out: usize, params: [u32; 2] },
        /// The tile writer into buffer `out` with `params`, one thread per tile.
        Tile { out: usize, params: [u32; 2] },
        /// The indexed indirect draw of the white fullscreen triangle.
        Draw,
    }
    use TileStep::{Draw, Tile, WriteBehindBarrier};
    let empty_rows = rows(&EMPTY);
    // Each case: its name, the arguments and indices it starts with, its steps, and Metal's
    // texels where the runtime draws them, else `None` where it is refused.
    let cases = [
        (
            "tw_args",
            none,
            full,
            vec![
                Tile {
                    out: 0,
                    params: ARGUMENTS,
                },
                Draw,
            ],
            Some(&empty_rows),
        ),
        (
            "tw_indices",
            three,
            degenerate,
            vec![
                Tile {
                    out: 1,
                    params: INDICES,
                },
                Draw,
            ],
            Some(&empty_rows),
        ),
        (
            "tw_both",
            none,
            degenerate,
            vec![
                Tile {
                    out: 0,
                    params: ARGUMENTS,
                },
                Tile {
                    out: 1,
                    params: INDICES,
                },
                Draw,
            ],
            Some(&empty_rows),
        ),
        (
            "tw_args_barrier_rewrite",
            none,
            full,
            vec![
                WriteBehindBarrier {
                    out: 0,
                    params: ARGUMENTS,
                },
                Tile {
                    out: 0,
                    params: ZEROS,
                },
                Draw,
            ],
            None,
        ),
    ];
    for (case, arguments, index_words, steps, metal) in cases {
        let buffers = [u32s(&probe, &arguments), u16s(&probe, &index_words)];
        let mut answers = Vec::new();
        let stored = probe.run(PASS, |p, e, h| {
            for step in &steps {
                match *step {
                    WriteBehindBarrier { out, params } => {
                        let params = u32s(p, &params);
                        e.set_render_pipeline_state(&writer).unwrap();
                        e.set_vertex_buffer(Some(&params), 0, 0).unwrap();
                        e.set_vertex_buffer(Some(&buffers[out]), 0, 1).unwrap();
                        e.draw_primitives(PrimitiveType::Point, 0, 2, 1, 0).unwrap();
                        e.memory_barrier_with_scope(
                            BarrierScope::BUFFERS,
                            RenderStages::VERTEX,
                            RenderStages::VERTEX,
                        );
                        h.push(params);
                    }
                    Tile { out, params } => {
                        let params = u32s(p, &params);
                        e.set_render_pipeline_state(&tile).unwrap();
                        e.set_tile_buffer(Some(&params), 0, 0).unwrap();
                        e.set_tile_buffer(Some(&buffers[out]), 0, 1).unwrap();
                        e.dispatch_threads_per_tile(Size::new(1, 1, 1)).unwrap();
                        h.push(params);
                    }
                    Draw => {
                        prepare(p, e, h);
                        answers.push(e.draw_indexed_primitives_indirect(
                            PrimitiveType::Triangle,
                            IndexType::UInt16,
                            &buffers[1],
                            0,
                            &buffers[0],
                            0,
                        ));
                    }
                }
            }
        });
        let at = format!("{case} on {name}");
        match metal {
            Some(metal) => {
                assert!(answers.iter().all(Result::is_ok), "{at}: {answers:?}");
                assert_eq!(&stored.color, metal, "{at}");
            }
            None => {
                let refused = Err(Error::NotImplemented(RuntimeGap::IndirectDrawArguments));
                assert_eq!(answers.last(), Some(&refused), "{at}");
                assert_eq!(stored.color, rows(&EMPTY), "{at}");
            }
        }
    }
    let first = forced_beside(&probe, BTreeSet::new());
    assert_made_once(&messages, &UNORDERED_DRAWS, &first);
    quiet(probe, &messages);
}
