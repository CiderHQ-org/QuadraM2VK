//! Device tests of late resolve targets (the runtime's `render/late_resolve.rs`) on the device the
//! cfg selects, under the validation layer, through the resolve lane's fixture `rt6_resolves` and
//! helpers (`resolves.rs`).
//!
//! What Metal probes measured on an M4 (macOS 27.0.1): the target set last is resolved into, at its
//! slice and level, holding every draw of the pass, those after the call too; the descriptor's
//! resolve texture keeps its texels; a depth and a stencil target are written whether or not the
//! descriptor named one; Metal's validation stops a target of another pixel format or size and an
//! index without an attachment. Where Metal ignores a color target the descriptor named none for,
//! the runtime's setter does what it says and resolves into it (the extension's contract, not
//! Metal's answer).

use super::buffer_with;
use super::resolves::{
    color_pass, device_resolves, drawn_depth, lane, quiet, repeat, Lane, Shape, DEPTHS, FULL,
    MASKED, MSAA, SIZE, STENCILS, WHITE,
};
use metal2vulkan::metal::state::{DepthStencilDescriptor, StencilDescriptor};
use metal2vulkan::metal::{
    BarrierScope, CompareFunction, PixelFormat, RenderStages, StencilOperation, TextureUsage,
};
use metal2vulkan::vk::ResolveModeFlags as Mode;
use metal2vulkan_runtime::{
    Attachment, BlitOption, Error, Invalid, LoadAction, MultisampleDepthResolveFilter,
    MultisampleStencilResolveFilter, Problem, Region, RenderPassDescriptor, RenderPassProblem,
    ResolveTarget, RuntimeGap, StoreAction, Texture, MAX_COLOR_ATTACHMENTS,
};

const RGBA8: PixelFormat = PixelFormat::RGBA8Unorm;
/// What an untouched color texture holds: every byte 0x11.
const UNTOUCHED: u8 = 0x11;

/// A shared single-sample `RGBA8Unorm` texture of `side`², `slices` and `levels`, each byte of
/// every subresource 0x11.
fn untouched(lane: &Lane, side: u64, slices: u64, levels: u64) -> Texture {
    let shape = Shape {
        size: side,
        samples: 1,
        slices,
        levels,
        shared: true,
    };
    let texture = lane.texture(RGBA8, shape, TextureUsage::RENDER_TARGET);
    for slice in 0..slices {
        for level in 0..levels {
            let side = side >> level;
            texture
                .replace_region(
                    Region::new_2d(0, 0, side, side),
                    level,
                    slice,
                    &vec![UNTOUCHED; (4 * side * side) as usize],
                    4 * side,
                    4 * side * side,
                )
                .unwrap();
        }
    }
    texture
}

/// `texture`'s level 0 of slice 0 as white or untouched.
fn white(lane: &Lane, texture: &Texture) -> bool {
    let texels = lane.read(texture, (0, 0), 4, BlitOption::None);
    match texels[0] {
        0xff => {
            assert_eq!(texels, vec![0xff; texels.len()]);
            true
        }
        _ => {
            assert_eq!(texels, vec![UNTOUCHED; texels.len()]);
            false
        }
    }
}

fn invalid(call: &'static str, problem: RenderPassProblem) -> Error {
    Error::Invalid(Invalid::new(call, Problem::RenderPass(problem)))
}

/// A color target resolves the whole pass: set before or after the draws, in a pass split into
/// instances, over the descriptor's (which keeps its texels), the last of two, at a slice and a
/// level, and none at all.
#[test]
fn late_color_targets_resolve_the_whole_pass() {
    let (messages, _cache, lane) = lane();
    let pipeline = lane.pipeline((FULL, WHITE), &[RGBA8], (None, None), 4);
    let msaa = lane.msaa(RGBA8);
    let target = |texture: &Texture| Some(ResolveTarget::new(texture.clone()));
    let msr = StoreAction::MultisampleResolve;

    // Where the descriptor names none, set after the draws of a pass a barrier splits into two
    // instances, each recorded without a resolve, then the store action.
    let b = untouched(&lane, SIZE, 1, 1);
    lane.run(
        &color_pass(&msaa, None, StoreAction::Unknown, (0, 0)),
        |l, e, h| {
            l.draw(e, h, &pipeline, 0.5);
            e.memory_barrier_with_scope(
                BarrierScope::TEXTURES,
                RenderStages::FRAGMENT,
                RenderStages::FRAGMENT,
            );
            l.draw(e, h, &pipeline, 0.5);
            e.set_color_resolve_target(target(&b), 0).unwrap();
            e.set_color_store_action(msr, 0).unwrap();
        },
    )
    .unwrap();
    assert!(white(&lane, &b), "set after the draws");

    // Over the descriptor's, set before the draw.
    let (a, b) = (untouched(&lane, SIZE, 1, 1), untouched(&lane, SIZE, 1, 1));
    lane.run(&color_pass(&msaa, Some(&a), msr, (0, 0)), |l, e, h| {
        e.set_color_resolve_target(target(&b), 0).unwrap();
        l.draw(e, h, &pipeline, 0.5);
    })
    .unwrap();
    assert!(!white(&lane, &a), "the descriptor's target");
    assert!(white(&lane, &b), "set before the draw");

    // The last of two.
    let (b, c) = (untouched(&lane, SIZE, 1, 1), untouched(&lane, SIZE, 1, 1));
    lane.run(
        &color_pass(&msaa, None, StoreAction::Unknown, (0, 0)),
        |l, e, h| {
            e.set_color_resolve_target(target(&b), 0).unwrap();
            l.draw(e, h, &pipeline, 0.5);
            e.set_color_resolve_target(target(&c), 0).unwrap();
            e.set_color_store_action(msr, 0).unwrap();
        },
    )
    .unwrap();
    assert!(!white(&lane, &b), "the first of two");
    assert!(white(&lane, &c), "the last of two");

    // Slice 1 and level 1 of an 8² array of two slices and two levels: only that one.
    let array = untouched(&lane, 2 * SIZE, 2, 2);
    lane.run(
        &color_pass(&msaa, None, StoreAction::Unknown, (0, 0)),
        |l, e, h| {
            let mut at = ResolveTarget::new(array.clone());
            at.slice = 1;
            at.level = 1;
            e.set_color_resolve_target(Some(at), 0).unwrap();
            l.draw(e, h, &pipeline, 0.5);
            e.set_color_store_action(msr, 0).unwrap();
        },
    )
    .unwrap();
    for slice in 0..2 {
        for level in 0..2u64 {
            let texels = lane.read(&array, (level, slice), 4, BlitOption::None);
            let expected = if (slice, level) == (1, 1) {
                0xff
            } else {
                UNTOUCHED
            };
            assert_eq!(
                texels,
                vec![expected; texels.len()],
                "slice {slice} level {level}"
            );
        }
    }

    // None over the descriptor's, stored: nothing resolves.
    let a = untouched(&lane, SIZE, 1, 1);
    lane.run(
        &color_pass(&msaa, Some(&a), StoreAction::Unknown, (0, 0)),
        |l, e, h| {
            l.draw(e, h, &pipeline, 0.5);
            e.set_color_resolve_target(None, 0).unwrap();
            e.set_color_store_action(StoreAction::Store, 0).unwrap();
        },
    )
    .unwrap();
    assert!(!white(&lane, &a), "none over the descriptor's");
    quiet(lane, &messages);
}

/// Depth and stencil targets resolve as the descriptor's do: a depth target over the
/// descriptor's (which keeps its texels), and both aspects of a combined attachment the
/// descriptor names none for.
#[test]
fn late_depth_and_stencil_targets_resolve() {
    let (messages, _cache, lane) = lane();
    let texels = (SIZE * SIZE) as usize;
    let state = |stencil: bool| {
        let mut replace = StencilDescriptor::default();
        replace.stencil_compare_function = CompareFunction::Always;
        replace.depth_stencil_pass_operation = StencilOperation::Replace;
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Always;
        state.depth_write_enabled = true;
        if stencil {
            state.front_face_stencil = replace;
            state.back_face_stencil = replace;
        }
        lane.device.new_depth_stencil_state(&state).unwrap()
    };
    // A single-sample texture of `pf` cleared to depth 0.75 (and stencil 7).
    let cleared = |pf: PixelFormat, stencil: bool| {
        let texture = lane.single(pf);
        let mut clear = RenderPassDescriptor::new();
        clear.depth_attachment.texture = Some(texture.clone());
        clear.depth_attachment.clear_depth = 0.75;
        clear.depth_attachment.store_action = StoreAction::Store;
        if stencil {
            clear.stencil_attachment.texture = Some(texture.clone());
            clear.stencil_attachment.clear_stencil = 7;
            clear.stencil_attachment.store_action = StoreAction::Store;
        }
        lane.run(&clear, |_, _, _| {}).unwrap();
        texture
    };
    let depth_of = |texture: &Texture, option| lane.read(texture, (0, 0), 4, option);
    let msr = StoreAction::MultisampleResolve;

    // Depth over the descriptor's.
    let depth = PixelFormat::Depth32Float;
    let (a, b) = (cleared(depth, false), cleared(depth, false));
    let msaa = lane.msaa(depth);
    let mut pass = color_pass(&lane.msaa(RGBA8), None, StoreAction::DontCare, (0, 0));
    pass.depth_attachment.texture = Some(msaa.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.store_action = msr;
    pass.depth_attachment.resolve_texture = Some(a.clone());
    let pipeline = lane.pipeline((FULL, WHITE), &[RGBA8], (Some(depth), None), 4);
    let always = state(false);
    lane.run(&pass, |l, e, h| {
        e.set_depth_stencil_state(&always).unwrap();
        l.draw(e, h, &pipeline, 0.25);
        e.set_depth_resolve_target(Some(ResolveTarget::new(b.clone())))
            .unwrap();
    })
    .unwrap();
    let none = BlitOption::None;
    assert_eq!(depth_of(&a, none), repeat(&0.75f32.to_le_bytes(), texels));
    assert_eq!(depth_of(&b, none), repeat(&0.25f32.to_le_bytes(), texels));

    // Both aspects, where the descriptor names none, each target set before its store action.
    let combined = PixelFormat::Depth32FloatStencil8;
    let resolve = cleared(combined, true);
    let msaa = lane.msaa(combined);
    let mut pass = color_pass(&lane.msaa(RGBA8), None, StoreAction::DontCare, (0, 0));
    for attachment in [
        &mut pass.depth_attachment.texture,
        &mut pass.stencil_attachment.texture,
    ] {
        *attachment = Some(msaa.clone());
    }
    pass.depth_attachment.store_action = StoreAction::Unknown;
    pass.stencil_attachment.store_action = StoreAction::Unknown;
    let pipeline = lane.pipeline((FULL, WHITE), &[RGBA8], (Some(combined), Some(combined)), 4);
    let replace = state(true);
    lane.run(&pass, |l, e, h| {
        e.set_depth_stencil_state(&replace).unwrap();
        e.set_stencil_reference_value(9);
        l.draw(e, h, &pipeline, 0.25);
        let target = Some(ResolveTarget::new(resolve.clone()));
        e.set_depth_resolve_target(target.clone()).unwrap();
        e.set_stencil_resolve_target(target).unwrap();
        e.set_depth_store_action(msr).unwrap();
        e.set_stencil_store_action(msr).unwrap();
    })
    .unwrap();
    assert_eq!(
        depth_of(&resolve, BlitOption::DepthFromDepthStencil),
        repeat(&0.25f32.to_le_bytes(), texels)
    );
    assert_eq!(
        lane.read(&resolve, (0, 0), 1, BlitOption::StencilFromDepthStencil),
        vec![9; texels]
    );
    quiet(lane, &messages);
}

/// Stencil and depth targets set after the draws of a pass whose aspects resolve in different
/// modes, depth `Min` and stencil `Sample0` (Metal probe case `ds_min_sample0` on an M4: depth 0.1,
/// stencil 20), split into two instances by a barrier, with a color target beside them: each
/// resolves as the descriptor's would. Where the device's resolve modes lack `MIN` the depth is
/// drawn after the pass where the driver's fragment depth is exact (lavapipe), from the
/// attachment's samples, so the attachment is made for shader reads too there. Where neither
/// resolves the pair, the depth setter is refused `NotImplemented(DepthStencilResolve)`, as the
/// descriptor's resolve is (`resolves::depth_and_stencil_resolve_the_samples_metal_picks`), and
/// keeps the target before (none): the stencil resolves alone. A device that resolves the pair only
/// one aspect at a time resolves the second after the pass; no committed profile is one.
#[test]
fn late_targets_resolve_aspects_of_different_modes() {
    let (messages, _cache, lane) = lane();
    let combined = PixelFormat::Depth32FloatStencil8;
    let texels = (SIZE * SIZE) as usize;
    let resolves = device_resolves(&lane.device);
    let drawn =
        drawn_depth(&lane.device)(Mode::MIN) && resolves(None, Some(Mode::SAMPLE_ZERO), true);
    let pair = resolves(Some(Mode::MIN), Some(Mode::SAMPLE_ZERO), true) || drawn;
    let mut replace = StencilDescriptor::default();
    replace.stencil_compare_function = CompareFunction::Always;
    replace.depth_stencil_pass_operation = StencilOperation::Replace;
    let mut state = DepthStencilDescriptor::default();
    state.depth_compare_function = CompareFunction::Always;
    state.depth_write_enabled = true;
    state.front_face_stencil = replace;
    state.back_face_stencil = replace;
    let state = lane.device.new_depth_stencil_state(&state).unwrap();
    let msaa = match drawn {
        true => {
            eprintln!("drawn: the depth Min");
            lane.texture(
                combined,
                MSAA,
                TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            )
        }
        false => lane.msaa(combined),
    };
    let resolve = lane.single(combined);
    let color = untouched(&lane, SIZE, 1, 1);
    let mut pass = color_pass(&lane.msaa(RGBA8), None, StoreAction::Unknown, (0, 0));
    pass.depth_attachment.texture = Some(msaa.clone());
    pass.depth_attachment.load_action = LoadAction::Clear;
    pass.depth_attachment.store_action = StoreAction::Unknown;
    pass.depth_attachment.depth_resolve_filter = MultisampleDepthResolveFilter::Min;
    pass.stencil_attachment.texture = Some(msaa.clone());
    pass.stencil_attachment.load_action = LoadAction::Clear;
    pass.stencil_attachment.store_action = StoreAction::Unknown;
    pass.stencil_attachment.stencil_resolve_filter = MultisampleStencilResolveFilter::Sample0;
    let pipeline = lane.pipeline(
        (FULL, MASKED),
        &[RGBA8],
        (Some(combined), Some(combined)),
        4,
    );
    let msr = StoreAction::MultisampleResolve;
    lane.run(&pass, |l, e, h| {
        e.set_depth_stencil_state(&state).unwrap();
        for sample in 0..4 {
            if sample == 2 {
                e.memory_barrier_with_scope(
                    BarrierScope::TEXTURES,
                    RenderStages::FRAGMENT,
                    RenderStages::FRAGMENT,
                );
            }
            e.set_stencil_reference_value(STENCILS[sample]);
            let mask = buffer_with(&l.device, &(1u32 << sample).to_le_bytes()).unwrap();
            e.set_fragment_buffer(Some(&mask), 0, 0).unwrap();
            l.draw(e, h, &pipeline, DEPTHS[sample]);
            h.push(mask);
        }
        e.set_color_resolve_target(Some(ResolveTarget::new(color.clone())), 0)
            .unwrap();
        e.set_color_store_action(msr, 0).unwrap();
        let target = Some(ResolveTarget::new(resolve.clone()));
        e.set_stencil_resolve_target(target.clone()).unwrap();
        let depth = e.set_depth_resolve_target(target);
        if pair {
            depth.unwrap();
            e.set_depth_store_action(msr).unwrap();
        } else {
            eprintln!("refused: the device's resolve modes lack depth Min with stencil Sample0");
            assert_eq!(
                depth,
                Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve))
            );
            e.set_depth_store_action(StoreAction::DontCare).unwrap();
        }
        e.set_stencil_store_action(msr).unwrap();
    })
    .unwrap();
    assert!(white(&lane, &color), "the color target");
    assert_eq!(
        lane.read(&resolve, (0, 0), 1, BlitOption::StencilFromDepthStencil),
        vec![STENCILS[0] as u8; texels],
        "the stencil target"
    );
    if pair {
        assert_eq!(
            lane.read(&resolve, (0, 0), 4, BlitOption::DepthFromDepthStencil),
            repeat(&0.1f32.to_le_bytes(), texels),
            "the depth target"
        );
    }
    quiet(lane, &messages);
}

/// A target the descriptor's resolve texture would be refused as is refused, typed, and the
/// target set before stands: another pixel format, size, sample count or subresource, an index
/// past the attachments or without one, a target under a store action that does not resolve,
/// and none under one that does.
#[test]
fn late_resolve_targets_are_refused_typed() {
    let (messages, _cache, lane) = lane();
    let pipeline = lane.pipeline((FULL, WHITE), &[RGBA8], (None, None), 4);
    let msaa = lane.msaa(RGBA8);
    let call = "set_color_resolve_target";
    let color = Attachment::Color(0);
    let of = |pf: PixelFormat, side: u64| {
        let shape = Shape {
            size: side,
            samples: 1,
            slices: 1,
            levels: 1,
            shared: false,
        };
        Some(ResolveTarget::new(lane.texture(
            pf,
            shape,
            TextureUsage::RENDER_TARGET,
        )))
    };
    // A slice past an array's (a 2D texture's slice is ignored).
    let past_slice = {
        let pair = Shape {
            size: SIZE,
            samples: 1,
            slices: 2,
            levels: 1,
            shared: false,
        };
        let mut at = ResolveTarget::new(lane.texture(RGBA8, pair, TextureUsage::RENDER_TARGET));
        at.slice = 2;
        Some(at)
    };
    let a = untouched(&lane, SIZE, 1, 1);
    // A pass loading its attachment: a target reaching past it is refused (after a `Clear` load
    // plain Metal's answer is the clear color there).
    let mut loading = color_pass(&msaa, Some(&a), StoreAction::MultisampleResolve, (0, 0));
    loading.color_attachments[0].load_action = LoadAction::Load;
    lane.run(&loading, |l, e, h| {
        l.draw(e, h, &pipeline, 0.5);
        for (target, index, problem) in [
            (
                of(PixelFormat::RGBA16Float, SIZE),
                0,
                RenderPassProblem::ResolveFormat(color),
            ),
            // The sRGB twin resolves, as the descriptor's.
            (
                of(PixelFormat::R32Uint, SIZE),
                0,
                RenderPassProblem::ResolveFormat(color),
            ),
            (
                of(RGBA8, 2 * SIZE),
                0,
                RenderPassProblem::ResolveSize(color),
            ),
            (
                Some(ResolveTarget::new(msaa.clone())),
                0,
                RenderPassProblem::ResolveSamples(color),
            ),
            (
                past_slice.clone(),
                0,
                RenderPassProblem::ResolveSubresource(color),
            ),
            (None, 0, RenderPassProblem::ResolveWithoutTexture(color)),
            (
                of(RGBA8, SIZE),
                1,
                RenderPassProblem::NoAttachment(Attachment::Color(1)),
            ),
        ] {
            assert_eq!(
                e.set_color_resolve_target(target, index),
                Err(invalid(call, problem))
            );
        }
        let past = MAX_COLOR_ATTACHMENTS as u32;
        assert_eq!(
            e.set_color_resolve_target(of(RGBA8, SIZE), past),
            Err(Error::Invalid(Invalid::new(
                call,
                Problem::IndexOutOfRange {
                    index: past,
                    limit: past
                }
            )))
        );
        assert_eq!(
            e.set_depth_resolve_target(None),
            Err(invalid(
                "set_depth_resolve_target",
                RenderPassProblem::NoAttachment(Attachment::Depth)
            ))
        );
    })
    .unwrap();
    assert!(white(&lane, &a), "the descriptor's target stands");

    // A target under a store action that does not resolve.
    lane.run(
        &color_pass(&msaa, None, StoreAction::Store, (0, 0)),
        |_, e, _| {
            assert_eq!(
                e.set_color_resolve_target(of(RGBA8, SIZE), 0),
                Err(invalid(call, RenderPassProblem::ResolveStoreAction(color)))
            );
        },
    )
    .unwrap();
    quiet(lane, &messages);
}
