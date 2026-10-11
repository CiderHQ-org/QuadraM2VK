//! Capture replay (`render/capture.rs`) on the device the cfg selects, under the validation layer:
//! a pass's load of an extended-range color rewrites the tiles its draws' primitives reach, as the
//! capture of their own positions bins them.
//!
//! A 128x128 `BGR10_XR` target of zero words, loaded and stored, under a pipeline of the owned
//! fixture `rch_vertex_math` (identity matrices: each position is the buffer's) whose color's
//! write mask is none of its channels, draws a triangle list's triangle in tile (1, 2), and an
//! indexed (`uint16`) triangle strip of two instances whose all-ones index restarts it: its first
//! piece's two triangles over tiles (2, 0) and (3, 0), its second's one in tile (0, 3). Metal's
//! pass rewrites the words of each tile a primitive reaches (bits 30 and 31 ones), and keeps the
//! others. The pass's first run asks for its draws' capture forms, which the device translates on
//! its pool and makes at later passes: it rewrites every tile and announces
//! `unreached-tiles-rewritten` (the device announces the kind at every occurrence). A later run,
//! once every capture form and pipeline is made, rewrites the four tiles alone, Metal's answer, and
//! announces nothing; so does the run after it. The same draws in a pass that loads the color
//! `DontCare` or `Clear` run no capture: no load pass takes the color.
//!
//! An atlas pass as the consumer's draw them does the same over its scissored cells' small rects
//! (an `RGB10A8_2P_XR10` target beside memoryless colors and depth): its first run rewrites every
//! tile the scissors reach, a later run the tiles the rects reach alone.

use super::degrade_notices::counts;
use super::resolves::{lane_with, quiet};
use super::{buffer_with, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::metal::{ColorWriteMask, IndexType, PixelFormat, PrimitiveType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Degradation, LoadAction, PipelineOptions, Region, RenderCommandEncoder,
    RenderPassDescriptor, ResourceOptions, ScissorRect, StoreAction, TextureDescriptor, WorkCounts,
};
use std::time::{Duration, Instant};

/// The target's side, in pixels: four 32x32 tiles a side.
const SIDE: u64 = 128;

/// The kind the pass announces until its draws' tile reach is captured.
const UNREACHED_TILES: Degradation = Degradation::UnreachedTilesRewritten;

/// The tiles the draws' primitives reach, `(x, y)`.
const REACHED: [(u64, u64); 4] = [(1, 2), (2, 0), (3, 0), (0, 3)];

/// The strip's indices: its first piece (vertices 3 to 6), the restart, its second (7 to 9).
const STRIP: [u16; 8] = [3, 4, 5, 6, 0xffff, 7, 8, 9];

/// The clip-space position of window pixel `(x, y)` of the target under Metal's default
/// viewport.
fn clip(x: f32, y: f32) -> [f32; 4] {
    [x / 64.0 - 1.0, 1.0 - y / 64.0, 0.5, 1.0]
}

/// The positions: the list's triangle in tile (1, 2), the strip's first piece's four vertices
/// over tiles (2, 0) and (3, 0), and its second piece's triangle in tile (0, 3).
fn positions() -> Vec<u8> {
    [
        clip(40.0, 70.0),
        clip(50.0, 70.0),
        clip(40.0, 80.0),
        clip(70.0, 8.0),
        clip(70.0, 20.0),
        clip(120.0, 8.0),
        clip(120.0, 20.0),
        clip(5.0, 100.0),
        clip(20.0, 100.0),
        clip(5.0, 115.0),
    ]
    .iter()
    .flatten()
    .flat_map(|v| v.to_le_bytes())
    .collect()
}

/// The fixture's uniforms: identity model, view and projection matrices.
fn identity() -> Vec<u8> {
    let one = |i: usize| (0..4).map(move |j| if i == j { 1.0f32 } else { 0.0 });
    (0..3)
        .flat_map(|_| (0..4).flat_map(one))
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// The target's words once a pass rewrote the tiles `tiles` and kept every other.
fn rewritten(tiles: &[(u64, u64)]) -> Vec<u8> {
    (0..SIDE)
        .flat_map(|y| (0..SIDE).map(move |x| (x, y)))
        .flat_map(|(x, y)| match tiles.contains(&(x / 32, y / 32)) {
            true => 0xc000_0000u32.to_le_bytes(),
            false => [0; 4],
        })
        .collect()
}

/// The capture tally between `before` and `after` (`WorkCounts`): the tiles found and bound, and
/// the passes found at their bound and narrower.
fn tally(before: &WorkCounts, after: &WorkCounts) -> (u64, u64, u64, u64) {
    (
        after.capture_reach_tiles_found - before.capture_reach_tiles_found,
        after.capture_reach_tiles_bound - before.capture_reach_tiles_bound,
        after.capture_reach_bound_passes - before.capture_reach_bound_passes,
        after.capture_reach_narrower_passes - before.capture_reach_narrower_passes,
    )
}

/// Every tile of the target.
fn every_tile() -> Vec<(u64, u64)> {
    (0..4).flat_map(|y| (0..4).map(move |x| (x, y))).collect()
}

/// The pass's first run rewrites every tile and announces `unreached-tiles-rewritten`; a later
/// run, once its draws' capture forms and pipelines are made, rewrites only the tiles their
/// primitives reach and announces nothing (the module docs).
#[test]
fn a_load_pass_rewrites_the_tiles_its_captured_primitives_reach() {
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = [UNREACHED_TILES].into_iter().collect();
    });
    let bitcode = llvm_assemble(&engine_dir().join("rch_vertex_math.ll")).unwrap();
    let library = lane.device.new_library_with_air(&bitcode).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let pf = PixelFormat::BGR10XR;
    let mut state = RenderPipelineDescriptor::new(function("rch_vertex_math"));
    state.fragment = Some(function("rch_vertex_math_color"));
    let mut none = ColorAttachmentDescriptor::new(pf);
    none.write_mask = ColorWriteMask::NONE;
    state.color_attachments[0] = Some(none);
    let state = lane
        .device
        .new_render_pipeline_state(&state, &PipelineOptions::default())
        .unwrap();
    let positions = buffer_with(&lane.device, &positions()).unwrap();
    let uniforms = buffer_with(&lane.device, &identity()).unwrap();
    let indices: Vec<u8> = STRIP.iter().flat_map(|i| i.to_le_bytes()).collect();
    let indices = buffer_with(&lane.device, &indices).unwrap();
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::shared();
    let target = lane.device.new_texture(&descriptor).unwrap();
    let mut pass = RenderPassDescriptor::new();
    let color = &mut pass.color_attachments[0];
    color.texture = Some(target.clone());
    color.load_action = LoadAction::Load;
    color.store_action = StoreAction::Store;
    let zeros = vec![0u8; (4 * SIDE * SIDE) as usize];
    // The pass's draws.
    let draw = |e: &mut RenderCommandEncoder| {
        e.set_render_pipeline_state(&state).unwrap();
        e.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
        e.set_vertex_buffer(Some(&uniforms), 0, 1).unwrap();
        e.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        e.draw_indexed_primitives(
            PrimitiveType::TriangleStrip,
            STRIP.len() as u32,
            IndexType::UInt16,
            &indices,
            0,
            2,
            0,
            0,
        )
        .unwrap();
    };
    // One run of the pass over zero words: the words it leaves, and whether it announced.
    let run = || {
        target
            .replace_region(Region::new_2d(0, 0, SIDE, SIDE), 0, 0, &zeros, 4 * SIDE, 0)
            .unwrap();
        lane.run(&pass, |_, e, _| draw(e)).unwrap();
        let words = lane.read(&target, (0, 0), 4, BlitOption::None);
        let announced = counts(&messages.take_notices()).contains_key(&UNREACHED_TILES);
        (words, announced)
    };
    let before = lane.device.work_counts();
    let (words, announced) = run();
    assert!(announced, "the first run announces");
    assert!(
        words == rewritten(&every_tile()),
        "the first run rewrites every tile"
    );
    // The capture forms are translated on the pool, then made at later runs.
    let began = Instant::now();
    let exact = loop {
        let (words, announced) = run();
        if !announced {
            break words;
        }
        assert!(words == rewritten(&every_tile()), "an announced run");
        assert!(
            began.elapsed() < WAIT,
            "no run's capture was ready within {WAIT:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(exact == rewritten(&REACHED), "the captured run's tiles");
    let tallied = lane.device.work_counts();
    let (words, announced) = run();
    assert!(!announced && words == exact, "the run after it");
    let after = lane.device.work_counts();
    assert!(
        after.capture_reach_passes - before.capture_reach_passes >= 2,
        "{before:?} {after:?}"
    );
    // Its tally: the primitives reach 4 of the 16 tiles its viewport reaches.
    assert_eq!(tally(&tallied, &after), (4, 16, 0, 1));
    // The same draws in a pass that loads the color `DontCare` (a dont-care load's clear,
    // `clear_reach.rs`) or `Clear`: no load pass takes it, so no capture runs.
    for load in [LoadAction::DontCare, LoadAction::Clear] {
        let mut other = RenderPassDescriptor::new();
        let color = &mut other.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = load;
        color.store_action = StoreAction::Store;
        let counted = lane.device.work_counts().capture_reach_passes;
        lane.run(&other, |_, e, _| draw(e)).unwrap();
        let captured = lane.device.work_counts().capture_reach_passes - counted;
        assert_eq!(captured, 0, "a {load:?} load captures nothing");
        messages.take_notices();
    }
    quiet(lane, &messages);
}

/// The atlas passes' height, in pixels: eight tiles.
const ATLAS_HEIGHT: u64 = 256;

/// The atlas passes' scissored cells.
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

/// Each cell's small rects, `[x0, y0, x1, y1]` in pixels: their edges off the tile grid, one
/// rect across x = 64.
const RECTS: [&[[f32; 4]]; 2] = [
    &[
        [10.0, 10.0, 26.0, 26.0],
        [60.0, 40.0, 70.0, 50.0],
        [100.0, 130.0, 120.0, 150.0],
    ],
    &[[180.0, 10.0, 200.0, 20.0], [290.0, 140.0, 300.0, 158.0]],
];

/// The tiles each cell's rects reach, `(x, y)`.
const RECT_TILES: [&[(u64, u64)]; 2] =
    [&[(0, 0), (1, 1), (2, 1), (3, 4)], &[(5, 0), (6, 0), (9, 4)]];

/// A two-plane target's color words of every bit, once a pass rewrote the tiles `tiles` (the
/// top two bits zeroed) and kept every other.
fn two_plane_rewritten([width, height]: [u64; 2], tiles: &[(u64, u64)]) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .flat_map(|(x, y)| match tiles.contains(&(x / 32, y / 32)) {
            true => 0x3fff_ffffu32.to_le_bytes(),
            false => u32::MAX.to_le_bytes(),
        })
        .collect()
}

/// An atlas pass as the consumer's draw them: an `RGB10A8_2P_XR10` color attachment of 256x256 (one
/// cell) or 384x256 (both) whose color words hold every bit, loaded and stored, beside two
/// memoryless `RGBA16Float` colors and a memoryless `Depth32Float` depth, each cleared and not
/// stored; in each cell, under its scissor, its small rects, each an indexed (`uint16`) list of two
/// triangles at its own base vertex, but for the third, a strip of four vertices without indices.
/// Every color's write mask is none of its channels. The first run rewrites every tile the cells'
/// scissors reach and announces `unreached-tiles-rewritten`; a later run, once the capture forms
/// and pipelines are made, rewrites the tiles the rects reach alone, Metal's answer, and announces
/// nothing.
#[test]
fn an_atlas_pass_of_scissored_cells_rewrites_the_tiles_its_rects_reach() {
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = [UNREACHED_TILES].into_iter().collect();
    });
    let bitcode = llvm_assemble(&engine_dir().join("rch_vertex_math.ll")).unwrap();
    let library = lane.device.new_library_with_air(&bitcode).unwrap();
    let function = |name: &str| {
        StageSource::Function(
            library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let pf = PixelFormat::RGB10A82PXR10;
    let side = PixelFormat::RGBA16Float;
    let mut state = RenderPipelineDescriptor::new(function("rch_vertex_math"));
    state.fragment = Some(function("rch_vertex_math_color"));
    for (index, format) in [pf, side, side].into_iter().enumerate() {
        let mut none = ColorAttachmentDescriptor::new(format);
        none.write_mask = ColorWriteMask::NONE;
        state.color_attachments[index] = Some(none);
    }
    state.depth_format = Some(PixelFormat::Depth32Float);
    let state = lane
        .device
        .new_render_pipeline_state(&state, &PipelineOptions::default())
        .unwrap();
    let uniforms = buffer_with(&lane.device, &identity()).unwrap();
    let list: Vec<u8> = [0u16, 1, 2, 2, 1, 3]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    let list = buffer_with(&lane.device, &list).unwrap();
    for (width, cells) in [(256u64, 1usize), (384, 2)] {
        let extent = [width, ATLAS_HEIGHT];
        let at = |x: f32, y: f32| {
            [
                2.0 * x / width as f32 - 1.0,
                1.0 - 2.0 * y / ATLAS_HEIGHT as f32,
                0.5,
                1.0,
            ]
        };
        let positions: Vec<u8> = RECTS[..cells]
            .iter()
            .flat_map(|rects| rects.iter())
            .flat_map(|&[x0, y0, x1, y1]| [at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1)])
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect();
        let positions = buffer_with(&lane.device, &positions).unwrap();
        let mut descriptor = TextureDescriptor::texture_2d(pf, width, ATLAS_HEIGHT, false);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::shared();
        let target = lane.device.new_texture(&descriptor).unwrap();
        let transient = |format: PixelFormat| {
            let mut descriptor = TextureDescriptor::texture_2d(format, width, ATLAS_HEIGHT, false);
            descriptor.usage = TextureUsage::RENDER_TARGET;
            descriptor.resource_options = super::memoryless::memoryless_options();
            lane.device.new_texture(&descriptor).unwrap()
        };
        let mut pass = RenderPassDescriptor::new();
        let color = &mut pass.color_attachments[0];
        color.texture = Some(target.clone());
        color.load_action = LoadAction::Load;
        color.store_action = StoreAction::Store;
        for index in [1, 2] {
            let color = &mut pass.color_attachments[index];
            color.texture = Some(transient(side));
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::DontCare;
        }
        pass.depth_attachment.texture = Some(transient(PixelFormat::Depth32Float));
        pass.depth_attachment.load_action = LoadAction::Clear;
        pass.depth_attachment.clear_depth = 1.0;
        pass.depth_attachment.store_action = StoreAction::DontCare;
        let filled = vec![0xffu8; (4 * width * ATLAS_HEIGHT) as usize];
        // One run of the pass over words of every bit: the words it leaves, and whether it
        // announced.
        let run = || {
            target
                .replace_region(
                    Region::new_2d(0, 0, width, ATLAS_HEIGHT),
                    0,
                    0,
                    &filled,
                    4 * width,
                    0,
                )
                .unwrap();
            lane.run(&pass, |_, e, _| {
                e.set_render_pipeline_state(&state).unwrap();
                e.set_vertex_buffer(Some(&positions), 0, 0).unwrap();
                e.set_vertex_buffer(Some(&uniforms), 0, 1).unwrap();
                let mut rect = 0u32;
                for (cell, rects) in CELLS[..cells].iter().zip(RECTS) {
                    e.set_scissor_rect(*cell).unwrap();
                    for _ in rects {
                        match rect {
                            2 => e.draw_primitives(PrimitiveType::TriangleStrip, 4 * rect, 4, 1, 0),
                            _ => e.draw_indexed_primitives(
                                PrimitiveType::Triangle,
                                6,
                                IndexType::UInt16,
                                &list,
                                0,
                                1,
                                4 * rect as i32,
                                0,
                            ),
                        }
                        .unwrap();
                        rect += 1;
                    }
                }
            })
            .unwrap();
            let words = lane.read(&target, (0, 0), 4, BlitOption::None);
            let announced = counts(&messages.take_notices()).contains_key(&UNREACHED_TILES);
            (words, announced)
        };
        // The host's reach: every tile the cells' scissors reach.
        let last = (CELLS[cells - 1].x + CELLS[cells - 1].width - 1) / 32;
        let scissored: Vec<(u64, u64)> = (0..=165 / 32)
            .flat_map(|y| (0..=last).map(move |x| (x, y)))
            .collect();
        let reached: Vec<(u64, u64)> = RECT_TILES[..cells].concat();
        // The first target's first run asks for the draws' capture forms; the second target's
        // draws take the same ones, already made or being made.
        let mut ran = run();
        assert!(ran.1 || cells > 1, "{width}: the first run announces");
        let began = Instant::now();
        let exact = loop {
            let (words, announced) = ran;
            if !announced {
                break words;
            }
            assert!(
                words == two_plane_rewritten(extent, &scissored),
                "{width}: an announced run rewrites the scissors' tiles"
            );
            assert!(
                began.elapsed() < WAIT,
                "{width}: no run's capture was ready within {WAIT:?}"
            );
            std::thread::sleep(Duration::from_millis(20));
            ran = run();
        };
        assert!(
            exact == two_plane_rewritten(extent, &reached),
            "{width}: the captured run's tiles"
        );
        let tallied = lane.device.work_counts();
        let (words, announced) = run();
        assert!(!announced && words == exact, "{width}: the run after it");
        // Its tally: the rects reach a few of the tiles the scissors reach.
        let counted = (reached.len() as u64, scissored.len() as u64, 0, 1);
        let after = lane.device.work_counts();
        assert_eq!(tally(&tallied, &after), counted, "{width}");
    }
    quiet(lane, &messages);
}
