//! Device tests of the counts of render passes whose first instance did not record inline, by the
//! first check that ruled it out (the runtime's `render/inline_first.rs`,
//! `WorkCounts::passes_not_inline_layouts` and kin), on the device the cfg selects, under the
//! validation layer, through the resolve lane's fixture and helpers (`resolves.rs`): a pass of one
//! sample cleared and stored records inline and counts no check; the same loaded `DontCare` counts
//! its actions; the same into a memoryless texture loaded and stored `DontCare` counts the draw's
//! write of its memoryless color, and cleared, its memoryless color; a pass of four samples
//! resolved counts its layout. A pass kept by its color's actions counts the first action ruled
//! out too: a `DontCare` load, a reach past the render area its colors do not share, a clear to a
//! NaN, a resolve, a store.
//!
//! And a memoryless color loaded and stored `DontCare` beside a stored one: a pass whose draw
//! writes only the stored color records inline, and one whose second draw writes the memoryless
//! color splits before it (`WorkCounts::inline_memoryless_write_splits`) into an instance that
//! records inline too (`WorkCounts::inline_split_instances`), each storing what plain Metal
//! stores.

use super::memoryless::memoryless;
use super::resolves::{color_pass, lane, quiet, Lane, Shape, FULL, ONE, SIZE, WHITE};
use metal2vulkan::metal::{ColorWriteMask, PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, ClearColor, LoadAction, PipelineOptions, RenderPassDescriptor, StoreAction,
    WorkCounts,
};

const RGBA8: PixelFormat = PixelFormat::RGBA8Unorm;

/// The counts of passes whose first instance recorded inline and of those that did not, by
/// check: inline, encoders, layouts, memoryless, actions, buffer textures, attachments, storage,
/// stand-ins, memoryless writes, draws.
fn inline_counts(counts: &WorkCounts) -> [u64; 11] {
    [
        counts.inline_passes,
        counts.passes_not_inline_encoders,
        counts.passes_not_inline_layouts,
        counts.passes_not_inline_memoryless,
        counts.passes_not_inline_actions,
        counts.passes_not_inline_buffer_textures,
        counts.passes_not_inline_attachments,
        counts.passes_not_inline_storage,
        counts.passes_not_inline_stand_ins,
        counts.passes_not_inline_memoryless_writes,
        counts.passes_not_inline_draws,
    ]
}

/// Draw white over `descriptor`'s pass, with a pipeline of `samples`: what that pass added to
/// [`inline_counts`].
fn drawn(lane: &Lane, descriptor: &RenderPassDescriptor, samples: u64) -> [u64; 11] {
    let pipeline = lane.pipeline((FULL, WHITE), &[RGBA8], (None, None), samples);
    let before = inline_counts(&lane.device.work_counts());
    lane.run(descriptor, |l, e, h| l.draw(e, h, &pipeline, 0.5))
        .unwrap();
    let after = inline_counts(&lane.device.work_counts());
    std::array::from_fn(|at| after[at] - before[at])
}

#[test]
fn a_pass_that_does_not_record_inline_counts_the_first_check_that_ruled_it_out() {
    let (messages, _cache, lane) = lane();
    let target = lane.single(RGBA8);
    let msaa = lane.msaa(RGBA8);
    let stored = color_pass(&target, None, StoreAction::Store, (0, 0));
    assert_eq!(
        drawn(&lane, &stored, 1),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "cleared and stored"
    );
    let mut undefined = stored.clone();
    undefined.color_attachments[0].load_action = LoadAction::DontCare;
    assert_eq!(
        drawn(&lane, &undefined, 1),
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "loaded DontCare"
    );
    let scratch = memoryless(
        &lane,
        TextureType::Type2D,
        RGBA8,
        ONE,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    let mut transient = color_pass(&scratch, None, StoreAction::DontCare, (0, 0));
    transient.color_attachments[0].load_action = LoadAction::DontCare;
    assert_eq!(
        drawn(&lane, &transient, 1),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "memoryless, loaded and stored DontCare, drawn"
    );
    let mut cleared = transient.clone();
    cleared.color_attachments[0].load_action = LoadAction::Clear;
    assert_eq!(
        drawn(&lane, &cleared, 1),
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "memoryless, cleared"
    );
    let resolved = color_pass(
        &msaa,
        Some(&target),
        StoreAction::MultisampleResolve,
        (0, 0),
    );
    assert_eq!(
        drawn(&lane, &resolved, 4),
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "four samples resolved"
    );
    drop((target, msaa, scratch));
    quiet(lane, &messages);
}

/// The counts of passes kept from recording inline by a color's actions, then by the first
/// action ruled out: a `DontCare` load, a reach past the render area not shared, a clear to a
/// NaN, a resolve, a store.
fn actions_counts(counts: &WorkCounts) -> [u64; 6] {
    [
        counts.passes_not_inline_actions,
        counts.passes_not_inline_actions_dont_care_load,
        counts.passes_not_inline_actions_unequal_reach,
        counts.passes_not_inline_actions_nan_clear,
        counts.passes_not_inline_actions_resolve,
        counts.passes_not_inline_actions_store,
    ]
}

/// The actions of a pass's color attachments that keep its first instance from recording
/// inline, each pass counted once by the first ruled out (`WorkCounts::passes_not_inline_actions`
/// and its `_dont_care_load` kin), white drawn over each: loaded `DontCare`; cleared over a render
/// area of half the texture's side, which Metal's clear of whole 32-pixel tiles reaches past,
/// beside a second color loaded, whose texels past the render area Metal keeps (where it is
/// alone, the pass records inline over the reach), and a larger texture cleared past a memoryless
/// color's side, which sets the render area; an `RGBA16Float` texture cleared to a NaN,
/// whose bits the device's clear would not keep; resolved, a single-sampled attachment's texels
/// copied into the resolve texture; and stored `DontCare`.
#[test]
fn a_pass_kept_by_a_color_s_actions_counts_the_first_action_ruled_out() {
    let (messages, _cache, lane) = lane();
    let rgba16f = PixelFormat::RGBA16Float;
    let ran = |descriptor: &RenderPassDescriptor, pf: PixelFormat| -> [u64; 6] {
        let colors = descriptor
            .color_attachments
            .iter()
            .filter(|c| c.texture.is_some())
            .count();
        let pipeline = lane.pipeline((FULL, WHITE), &vec![pf; colors], (None, None), 1);
        let before = actions_counts(&lane.device.work_counts());
        lane.run(descriptor, |l, e, h| l.draw(e, h, &pipeline, 0.5))
            .unwrap();
        let after = actions_counts(&lane.device.work_counts());
        std::array::from_fn(|at| after[at] - before[at])
    };
    let target = lane.single(RGBA8);
    let stored = color_pass(&target, None, StoreAction::Store, (0, 0));
    let mut undefined = stored.clone();
    undefined.color_attachments[0].load_action = LoadAction::DontCare;
    assert_eq!(
        ran(&undefined, RGBA8),
        [1, 1, 0, 0, 0, 0],
        "loaded DontCare"
    );
    let mut part = stored.clone();
    (part.render_target_width, part.render_target_height) = (SIZE / 2, SIZE / 2);
    assert_eq!(
        ran(&part, RGBA8),
        [0, 0, 0, 0, 0, 0],
        "cleared past half its side, alone"
    );
    let beside = lane.single(RGBA8);
    let loaded = &mut part.color_attachments[1];
    loaded.texture = Some(beside.clone());
    loaded.load_action = LoadAction::Load;
    loaded.store_action = StoreAction::Store;
    assert_eq!(
        ran(&part, RGBA8),
        [1, 0, 1, 0, 0, 0],
        "cleared past half its side, beside a color loaded"
    );
    // A 64x64 texture cleared over the 4x4 render area of a memoryless color beside it, loaded and
    // stored `DontCare`: the clear reaches 32x32, past the memoryless texture.
    let scratch = memoryless(
        &lane,
        TextureType::Type2D,
        RGBA8,
        ONE,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    let wide = lane.texture(
        RGBA8,
        Shape { size: 64, ..ONE },
        TextureUsage::RENDER_TARGET,
    );
    let mut past = color_pass(&scratch, None, StoreAction::DontCare, (0, 0));
    past.color_attachments[0].load_action = LoadAction::DontCare;
    let cleared = &mut past.color_attachments[1];
    cleared.texture = Some(wide.clone());
    cleared.load_action = LoadAction::Clear;
    cleared.store_action = StoreAction::Store;
    assert_eq!(
        ran(&past, RGBA8),
        [1, 0, 1, 0, 0, 0],
        "cleared past a memoryless color's side"
    );
    let half = lane.single(rgba16f);
    let mut nan = color_pass(&half, None, StoreAction::Store, (0, 0));
    nan.color_attachments[0].clear_color = ClearColor::new(f64::NAN, 0.0, 0.0, 1.0);
    assert_eq!(ran(&nan, rgba16f), [1, 0, 0, 1, 0, 0], "cleared to a NaN");
    let resolve = lane.single(RGBA8);
    let resolved = color_pass(
        &target,
        Some(&resolve),
        StoreAction::MultisampleResolve,
        (0, 0),
    );
    assert_eq!(ran(&resolved, RGBA8), [1, 0, 0, 0, 1, 0], "resolved");
    let unstored = color_pass(&target, None, StoreAction::DontCare, (0, 0));
    assert_eq!(ran(&unstored, RGBA8), [1, 0, 0, 0, 0, 1], "stored DontCare");
    drop((target, beside, scratch, wide, half, resolve));
    quiet(lane, &messages);
}

/// A memoryless color loaded and stored `DontCare` beside a stored one cleared to zero (the module
/// docs): white through a pipeline that writes the stored color alone records inline; white
/// through one that leaves the stored color alone and then through one that writes the memoryless
/// color too splits before the second, which records inline too. Each stores the white plain
/// Metal stores, or the clear, and the second pass's memoryless color takes a backing from the
/// split on.
#[test]
fn a_memoryless_color_records_inline_until_a_draw_writes_it() {
    let (messages, _cache, lane) = lane();
    let stored = lane.single(RGBA8);
    let scratch = memoryless(
        &lane,
        TextureType::Type2D,
        RGBA8,
        ONE,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap();
    // Color 0 the memoryless one, which `rt6_resolve_white` writes, and color 1 the stored one.
    let mut pass = color_pass(&scratch, None, StoreAction::DontCare, (0, 0));
    pass.color_attachments[0].load_action = LoadAction::DontCare;
    let other = &mut pass.color_attachments[1];
    other.texture = Some(stored.clone());
    other.load_action = LoadAction::Clear;
    other.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
    other.store_action = StoreAction::Store;
    let white = lane.pipeline((FULL, WHITE), &[RGBA8, RGBA8], (None, None), 1);
    let function = |name| {
        let function = lane.library.new_function_with_name(name).unwrap();
        StageSource::Function(function.function_descriptor())
    };
    let mut unwritten = RenderPipelineDescriptor::new(function(FULL));
    unwritten.fragment = Some(function(WHITE));
    let mut leaves = ColorAttachmentDescriptor::new(RGBA8);
    leaves.write_mask = ColorWriteMask::NONE;
    unwritten.color_attachments[0] = Some(leaves);
    unwritten.color_attachments[1] = Some(ColorAttachmentDescriptor::new(RGBA8));
    let unwritten = lane
        .device
        .new_render_pipeline_state(&unwritten, &PipelineOptions::default())
        .unwrap();
    let texels = (SIZE * SIZE) as usize;
    let cleared = vec![0u8; 4 * texels];
    // The memoryless color unwritten: one instance, inline.
    let splits = |lane: &Lane| lane.device.work_counts().inline_memoryless_write_splits;
    let later = |lane: &Lane| lane.device.work_counts().inline_split_instances;
    let (before, split_before) = (inline_counts(&lane.device.work_counts()), splits(&lane));
    let later_before = later(&lane);
    lane.run(&pass, |l, e, h| l.draw(e, h, &unwritten, 0.5))
        .unwrap();
    let after = inline_counts(&lane.device.work_counts());
    assert_eq!(
        std::array::from_fn::<u64, 11, _>(|at| after[at] - before[at]),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "beside a memoryless color it leaves alone"
    );
    assert_eq!(splits(&lane), split_before, "no split");
    assert_eq!(later(&lane), later_before, "no instance after a split");
    assert_eq!(
        lane.read(&stored, (0, 0), 4, BlitOption::None),
        cleared,
        "the stored color, cleared"
    );
    // Then a draw that writes it: split before it.
    let before = inline_counts(&lane.device.work_counts());
    lane.run(&pass, |l, e, h| {
        l.draw(e, h, &unwritten, 0.5);
        l.draw(e, h, &white, 0.5);
    })
    .unwrap();
    let after = inline_counts(&lane.device.work_counts());
    assert_eq!(
        std::array::from_fn::<u64, 11, _>(|at| after[at] - before[at]),
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "a later draw writes the memoryless color"
    );
    assert_eq!(splits(&lane), split_before + 1, "a split before it");
    assert_eq!(
        later(&lane),
        later_before + 1,
        "the instance after it, inline"
    );
    assert_eq!(
        lane.read(&stored, (0, 0), 4, BlitOption::None),
        cleared,
        "the stored color, cleared"
    );
    drop((stored, scratch));
    quiet(lane, &messages);
}
