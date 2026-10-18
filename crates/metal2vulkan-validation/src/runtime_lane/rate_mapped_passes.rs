//! Rate-mapped render passes through the runtime on the device the cfg selects, under the
//! validation layer. The committed public rate-mapped cases run in `renders`' census; this module
//! draws the public case `fragment-positions-in-a-half-quality- rate-mapped-pass`'s pipeline (its
//! generated fullscreen triangle and `fragment_rate_map_position`, which writes each fragment's
//! `[[position]]`) in passes of its own, over a map of a 64-pixel screen at quality 0.5, whose
//! physical size is 33 by 33, and over the identity of that screen, as our probes drew them on an
//! M3 Max and an M4:
//!
//! - Into attachments of 33, 48 and 64 pixels, the default viewport (the screen) shades the 33 by
//!   33 physical pixels and none past them, each at its centre; a viewport over the screen's left
//!   half shades physical columns 0 to 15; a scissor of the whole screen clips nothing, and one
//!   past the screen is clipped to it (probe 16's `sc_past_att_screen`, which Metal's validation
//!   stops: "render pass width(64)"); over the identity, a 16-pixel scissor keeps 16 columns.
//! - Plain Metal runs two passes its validation stops: attachments of 32 and 16 pixels, smaller
//!   than the physical size, take the physical pixels in them (probe 16's `att_32` and
//!   `att_small`), and an array length of 2 past the map's one layer rasterizes by that layer, a
//!   draw of slice 0 shading slice 0 alone (`map1_array_flat`).
//! - The pass is refused as Metal's validation refuses it: an array length past the layers of a
//!   map whose layers differ, and a pass without attachments larger than the screen. A scissor
//!   that cuts into the viewport over a map that is not the identity is `NotImplemented`, as the
//!   region path does not draw it.
//! - A pass over a map of no layers runs its clear and store, and its draws write nothing, as our
//!   probe's cases drew on an M4 wherever the map was made while no map of layers had been
//!   released. Made after one was, Metal draws physical rows 0 to 15 (an M4 and an M3 Max alike),
//!   the runtime nothing, and announces it.

use super::degrade_notices::{
    assert_made_once, counts, made_render_kinds, shared_in_system_memory,
};
use super::rate_maps::descriptor;
use super::renders::map_render;
use super::{buffer_with, validated_device, Messages, ScratchCache, WAIT};
use crate::candidate::runtime_case::{render_pipeline_descriptor, render_pipeline_state};
use crate::gpu::TEST_DEVICE;
use crate::store::CorpusStore;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, ClearColor, CommandBufferStatus, Degradation, Device, Error, Invalid, LoadAction,
    Problem, RasterizationRateMap, Region, RegionRefusal, RenderCommandEncoder,
    RenderPassDescriptor, RenderPassProblem, RenderPipelineState, ResourceOptions, RuntimeGap,
    ScissorRect, StoreAction, Texture, TextureDescriptor, Viewport,
};
use std::path::Path;

/// The public case whose pipeline the passes draw.
const CASE: &str = "fragment-positions-in-a-half-quality-rate-mapped-pass";

/// The maps' screen side.
const SCREEN: u32 = 64;

/// The case's pipeline and its fragment buffer (the screen) on one device.
struct Fixture {
    device: Device,
    state: RenderPipelineState,
    screen: Buffer,
    /// The kinds its state's making announced where the device had not
    /// (`made_render_kinds`), and its `Shared` buffer's system memory.
    made: std::collections::BTreeSet<Degradation>,
}

impl Fixture {
    fn new(device: &Device) -> Fixture {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("corpus");
        let store = CorpusStore::new(&root);
        let case = (0..64)
            .flat_map(|shard| store.read_case_shard(shard).unwrap())
            .find(|case| case.name == CASE)
            .expect("the rate-mapped position case");
        let case = map_render(&root, &store, case).unwrap();
        let screen = [SCREEN as f32; 2]
            .iter()
            .flat_map(|side| side.to_le_bytes())
            .collect::<Vec<u8>>();
        let descriptor = render_pipeline_descriptor(device, &case.inputs()).unwrap();
        let mut made = made_render_kinds(device, &descriptor);
        if shared_in_system_memory(device) {
            made.insert(Degradation::SharedInSystemMemory);
        }
        Fixture {
            state: render_pipeline_state(device, &case.inputs()).unwrap(),
            screen: buffer_with(device, &screen).unwrap(),
            device: device.clone(),
            made,
        }
    }

    /// The map of a screen of `side` pixels a side and these layers' qualities.
    fn map_of(&self, side: u32, layers: &[(f32, f32)]) -> RasterizationRateMap {
        self.device
            .new_rasterization_rate_map(&descriptor(side, side, layers))
            .unwrap()
    }

    /// The map of the screen at `quality` on both axes.
    fn map(&self, quality: f32) -> RasterizationRateMap {
        self.device
            .new_rasterization_rate_map(&descriptor(SCREEN, SCREEN, &[(quality, quality)]))
            .unwrap()
    }

    /// An `RGBA32Float` attachment of `side` pixels a side, the CPU reading it.
    fn target(&self, side: u64) -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, side, side, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        self.device.new_texture(&descriptor).unwrap()
    }

    /// An `RGBA32Float` 2D array attachment of `side` pixels a side and `slices` slices, the CPU
    /// reading it.
    fn target_array(&self, side: u64, slices: u64) -> Texture {
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, side, side, false);
        descriptor.texture_type = TextureType::Type2DArray;
        descriptor.array_length = slices;
        descriptor.usage = TextureUsage::RENDER_TARGET;
        descriptor.resource_options = ResourceOptions::shared();
        self.device.new_texture(&descriptor).unwrap()
    }

    /// Run a pass over `map` into `target`, cleared to -1, its encoder's state set by `state`,
    /// with one draw of the case's triangle.
    fn draw(
        &self,
        map: &RasterizationRateMap,
        target: &Texture,
        state: impl FnOnce(&mut RenderCommandEncoder),
    ) -> Result<(), Error> {
        self.draw_in(&pass(map, target), state)
    }

    /// Run `pass` with its encoder's state set by `state` and one draw of the case's triangle.
    fn draw_in(
        &self,
        pass: &RenderPassDescriptor,
        state: impl FnOnce(&mut RenderCommandEncoder),
    ) -> Result<(), Error> {
        let queue = self.device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(pass)?;
        encoder.set_render_pipeline_state(&self.state)?;
        encoder.set_fragment_buffer(Some(&self.screen), 0, 0)?;
        state(&mut encoder);
        encoder.draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)?;
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        Ok(())
    }
}

/// A pass over `map` into `target`, cleared to -1 and stored.
fn pass(map: &RasterizationRateMap, target: &Texture) -> RenderPassDescriptor {
    let mut pass = plain(target);
    pass.rasterization_rate_map = Some(map.clone());
    pass
}

/// A pass into `target` without a map, cleared to -1 and stored.
fn plain(target: &Texture) -> RenderPassDescriptor {
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Clear;
    attachment.clear_color = ClearColor::new(-1.0, -1.0, -1.0, -1.0);
    attachment.store_action = StoreAction::Store;
    pass
}

/// `fixture`'s attachment of `side` pixels a side holding 0.25 in every channel, so that a
/// pass's clear shows.
fn filled(fixture: &Fixture, side: u64) -> Texture {
    let target = fixture.target(side);
    let bytes: Vec<u8> = (0..side * side * 4)
        .flat_map(|_| 0.25f32.to_le_bytes())
        .collect();
    target
        .replace_region(
            Region::new_2d(0, 0, side, side),
            0,
            0,
            &bytes,
            16 * side,
            16 * side * side,
        )
        .unwrap();
    target
}

/// Whether every channel of every texel of `target` is the passes' clear, -1.
fn cleared(target: &Texture) -> bool {
    let side = target.width();
    let mut bytes = vec![0u8; (side * side * 16) as usize];
    target
        .get_bytes(
            &mut bytes,
            side * 16,
            0,
            Region::new_2d(0, 0, side, side),
            0,
            0,
        )
        .unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .all(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) == -1.0)
}

/// What a draw shaded into `target`'s first slice (`shaded_in`).
fn shaded(target: &Texture) -> (usize, [u64; 4], [[f32; 2]; 2]) {
    shaded_in(target, 0)
}

/// What a draw shaded into `slice` of `target`: the count of pixels it wrote, their bounding box
/// `[x0, y0, x1, y1]`, and the positions written at its first and last corner; zeros where it
/// wrote none.
fn shaded_in(target: &Texture, slice: u64) -> (usize, [u64; 4], [[f32; 2]; 2]) {
    let side = target.width();
    let mut bytes = vec![0u8; (side * side * 16) as usize];
    target
        .get_bytes(
            &mut bytes,
            side * 16,
            0,
            Region::new_2d(0, 0, side, side),
            0,
            slice,
        )
        .unwrap();
    let texel = |x: u64, y: u64| {
        let at = ((y * side + x) * 16) as usize;
        let float =
            |k: usize| f32::from_le_bytes(bytes[at + 4 * k..at + 4 * k + 4].try_into().unwrap());
        [float(0), float(1)]
    };
    let mut count = 0;
    let mut bounds = [u64::MAX, u64::MAX, 0, 0];
    for y in 0..side {
        for x in 0..side {
            if texel(x, y)[0] >= 0.0 {
                count += 1;
                bounds = [
                    bounds[0].min(x),
                    bounds[1].min(y),
                    bounds[2].max(x),
                    bounds[3].max(y),
                ];
            }
        }
    }
    if count == 0 {
        return (0, [0; 4], [[0.0; 2]; 2]);
    }
    let positions = [texel(bounds[0], bounds[1]), texel(bounds[2], bounds[3])];
    (count, bounds, positions)
}

/// Probes 16 and 16b's draws: what a rate-mapped pass shades, and where (the module docs).
#[test]
fn a_rate_mapped_pass_shades_its_physical_pixels_region_by_region() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let fixture = Fixture::new(&device);
    let half = fixture.map(0.5);
    let whole = (33 * 33, [0, 0, 32, 32], [[0.5, 0.5], [32.5, 32.5]]);
    for side in [33, 48, 64] {
        let target = fixture.target(side);
        fixture.draw(&half, &target, |_| {}).unwrap();
        assert_eq!(shaded(&target), whole, "attachments of {side}");
    }
    let target = fixture.target(33);
    fixture
        .draw(&half, &target, |encoder| {
            encoder
                .set_viewport(Viewport::new(0.0, 0.0, 32.0, 64.0, 0.0, 1.0))
                .unwrap();
        })
        .unwrap();
    assert_eq!(
        shaded(&target),
        (16 * 33, [0, 0, 15, 32], [[0.5, 0.5], [15.5, 32.5]]),
        "the left half's viewport"
    );
    let target = fixture.target(33);
    fixture
        .draw(&half, &target, |encoder| {
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, 64, 64))
                .unwrap();
        })
        .unwrap();
    assert_eq!(shaded(&target), whole, "the screen's scissor");
    // A scissor past the screen is clipped to it (probe 16's `sc_past_att_screen`).
    let target = fixture.target(64);
    fixture
        .draw(&half, &target, |encoder| {
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, 100, 64))
                .unwrap();
        })
        .unwrap();
    assert_eq!(shaded(&target), whole, "a scissor past the screen");
    let identity = fixture.map(1.0);
    let target = fixture.target(64);
    fixture
        .draw(&identity, &target, |encoder| {
            encoder
                .set_scissor_rect(ScissorRect::new(0, 0, 16, 64))
                .unwrap();
        })
        .unwrap();
    assert_eq!(
        shaded(&target),
        (16 * 64, [0, 0, 15, 63], [[0.5, 0.5], [15.5, 63.5]]),
        "a scissor over the identity"
    );
    // Attachments smaller than the physical size take the physical pixels in them (probe 16's
    // `att_32` and `att_small`).
    for side in [32, 16] {
        let target = fixture.target(side);
        fixture.draw(&half, &target, |_| {}).unwrap();
        let last = side - 1;
        let centre = last as f32 + 0.5;
        assert_eq!(
            shaded(&target),
            (
                (side * side) as usize,
                [0, 0, last, last],
                [[0.5, 0.5], [centre, centre]]
            ),
            "attachments of {side}"
        );
    }
    // An array length past the map's one layer rasterizes by it, a draw of slice 0 shading that
    // slice alone (`map1_array_flat`).
    let target = fixture.target_array(33, 2);
    let mut layered = pass(&half, &target);
    layered.render_target_array_length = 2;
    fixture.draw_in(&layered, |_| {}).unwrap();
    assert_eq!(shaded_in(&target, 0), whole, "slice 0 past the map's layer");
    assert_eq!(shaded_in(&target, 1).0, 0, "slice 1 past the map's layer");
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// The kind [`a_pass_over_a_map_of_no_layers_draws_nothing`] takes: a draw over a map of no
/// layers made after the device released a map of layers.
pub(super) const ZERO_LAYER_AFTER_RELEASE: [Degradation; 1] =
    [Degradation::ZeroLayerMapAfterRelease];

/// Run `test` with a fixture on a device of its own, as each of the probes' cases ran in a process
/// of its own (a device's released maps of layers decide what its later maps of no layers draw),
/// then fail on any validation message.
fn fresh(test: impl FnOnce(&Fixture, &Messages)) {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let fixture = Fixture::new(&device);
    test(&fixture, &messages);
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// Whether the device announced [`ZERO_LAYER_AFTER_RELEASE`] since `messages` was last read.
fn announced_after_release(messages: &Messages) -> bool {
    counts(&messages.take_notices()).contains_key(&Degradation::ZeroLayerMapAfterRelease)
}

/// Our probe's eleven cases and its release before the map (an M4 with macOS
/// 27.0.1, each in a process of its own, three times, the same bytes each time), each on a device
/// of its own: a pass over a map of no layers made while no map of layers was released runs its
/// clear and store and its draw writes nothing, at 8 pixels with and without a scissor, at 64, a
/// map of 256 pixels over an attachment of 8, beside a live map of one layer, between passes over
/// one, and for a map made before a map of layers was released; and announces nothing. One made
/// after a map of layers was released draws on Metal (rm0draw1's `after_half64`: physical rows 0
/// to 15 at full width), where the runtime draws nothing and announces
/// [`ZERO_LAYER_AFTER_RELEASE`] once. The probe's controls: without a map the draw shades the
/// scissor's 3 by 5 pixels, and all 64 by 64; a map of 8 pixels at half quality shades those 3 by 5
/// pixels too, and one of 64 its 33 by 33 physical pixels.
#[test]
fn a_pass_over_a_map_of_no_layers_draws_nothing() {
    let corner = |encoder: &mut RenderCommandEncoder| {
        encoder
            .set_scissor_rect(ScissorRect::new(0, 0, 3, 5))
            .unwrap();
    };
    let whole = (33 * 33, [0, 0, 32, 32], [[0.5, 0.5], [32.5, 32.5]]);
    // `none8` and `half8`.
    fresh(|fixture, _| {
        let target = filled(fixture, 8);
        fixture.draw_in(&plain(&target), corner).unwrap();
        assert_eq!(
            shaded(&target),
            (15, [0, 0, 2, 4], [[0.5, 0.5], [2.5, 4.5]]),
            "none8"
        );
    });
    fresh(|fixture, _| {
        let target = filled(fixture, 8);
        fixture
            .draw(&fixture.map_of(8, &[(0.5, 0.5)]), &target, corner)
            .unwrap();
        let (count, bounds, _) = shaded(&target);
        assert_eq!((count, bounds), (15, [0, 0, 2, 4]), "half8");
    });
    // `zero8`, `zero8full`, `zero64` and `zero256on8`.
    for (name, screen, side, scissor) in [
        ("zero8", 8, 8, true),
        ("zero8full", 8, 8, false),
        ("zero64", 64, 64, false),
        ("zero256on8", 256, 8, true),
    ] {
        fresh(|fixture, messages| {
            let target = filled(fixture, side);
            fixture
                .draw(&fixture.map_of(screen, &[]), &target, |encoder| {
                    if scissor {
                        corner(encoder);
                    }
                })
                .unwrap();
            assert!(cleared(&target), "{name}");
            assert!(!announced_after_release(messages), "{name}");
        });
    }
    // `none64` and `half64`.
    fresh(|fixture, _| {
        let target = filled(fixture, 64);
        fixture.draw_in(&plain(&target), |_| {}).unwrap();
        assert_eq!(
            shaded(&target),
            (64 * 64, [0, 0, 63, 63], [[0.5, 0.5], [63.5, 63.5]]),
            "none64"
        );
        let target = filled(fixture, 64);
        fixture
            .draw(&fixture.map_of(64, &[(0.5, 0.5)]), &target, |_| {})
            .unwrap();
        assert_eq!(shaded(&target), whole, "half64");
    });
    // `after_half64`: a pass over a map of one layer, the map released, then one of none, which
    // Metal draws (its rows 0 to 15) and the runtime does not: announced, once.
    fresh(|fixture, messages| {
        {
            let half = fixture.map_of(64, &[(0.5, 0.5)]);
            let target = filled(fixture, 64);
            fixture.draw(&half, &target, |_| {}).unwrap();
            assert_eq!(shaded(&target), whole, "after_half64_first");
        }
        for _ in 0..2 {
            let target = filled(fixture, 64);
            fixture
                .draw(&fixture.map_of(64, &[]), &target, |_| {})
                .unwrap();
            assert!(cleared(&target), "after_half64_zero");
        }
        // Beside it, the fixture's making and buffer, and the half map's pass drawn by regions
        // (`rasterization-rate-map`: its axes are not the identity).
        let mut first = fixture.made.clone();
        first.insert(Degradation::RasterizationRateMap);
        assert_made_once(messages, &ZERO_LAYER_AFTER_RELEASE, &first);
    });
    // `beside_half64`: a map of one layer live beside one of none.
    fresh(|fixture, messages| {
        let half = fixture.map_of(64, &[(0.5, 0.5)]);
        let empty = fixture.map_of(64, &[]);
        let target = filled(fixture, 64);
        fixture.draw(&empty, &target, |_| {}).unwrap();
        assert!(cleared(&target), "beside_half64_zero");
        let target = filled(fixture, 64);
        fixture.draw(&half, &target, |_| {}).unwrap();
        assert_eq!(shaded(&target), whole, "beside_half64_half");
        assert!(!announced_after_release(messages), "beside_half64");
    });
    // `seq`: passes over a map of none, one of one layer, and the first again.
    fresh(|fixture, messages| {
        let empty = fixture.map_of(64, &[]);
        let half = fixture.map_of(64, &[(0.5, 0.5)]);
        for (name, over, zero) in [
            ("seq_zero1", &empty, true),
            ("seq_half", &half, false),
            ("seq_zero2", &empty, true),
        ] {
            let target = filled(fixture, 64);
            fixture.draw(over, &target, |_| {}).unwrap();
            match zero {
                true => assert!(cleared(&target), "{name}"),
                false => assert_eq!(shaded(&target), whole, "{name}"),
            }
        }
        assert!(!announced_after_release(messages), "seq");
    });
    // `zero_before_release`: a map of none made before a map of one layer is
    // released draws nothing on Metal after the release too, and announces nothing here.
    fresh(|fixture, messages| {
        let empty = fixture.map_of(64, &[]);
        {
            let half = fixture.map_of(64, &[(0.5, 0.5)]);
            fixture.draw(&half, &filled(fixture, 64), |_| {}).unwrap();
        }
        let target = filled(fixture, 64);
        fixture.draw(&empty, &target, |_| {}).unwrap();
        assert!(cleared(&target), "zero_before_release_zero2");
        assert!(!announced_after_release(messages), "zero_before_release");
    });
}

/// Probes 16, 16b and 16c's refusals, and the region path's (the module docs).
#[test]
fn rate_mapped_passes_are_refused_where_metal_or_the_runtime_refuses_them() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
    let fixture = Fixture::new(&device);
    let half = fixture.map(0.5);
    let queue = device.new_command_queue().unwrap();
    let refused = |pass: &RenderPassDescriptor| {
        let command_buffer = queue.command_buffer().unwrap();
        match command_buffer.render_command_encoder(pass) {
            Err(Error::Invalid(Invalid {
                problem: Problem::RenderPass(problem),
                ..
            })) => problem,
            other => panic!("a refused pass made {other:?}"),
        }
    };
    // Past the layers of a map whose layers differ.
    let differing = device
        .new_rasterization_rate_map(&descriptor(SCREEN, SCREEN, &[(0.5, 0.5), (1.0, 1.0)]))
        .unwrap();
    let mut layered = pass(&differing, &fixture.target_array(33, 3));
    layered.render_target_array_length = 3;
    assert_eq!(
        refused(&layered),
        RenderPassProblem::RateMapArrayLength {
            length: 3,
            layers: 2,
        }
    );
    let mut targetless = RenderPassDescriptor::new();
    targetless.rasterization_rate_map = Some(half.clone());
    targetless.render_target_width = 128;
    targetless.render_target_height = 128;
    targetless.default_raster_sample_count = 1;
    assert_eq!(
        refused(&targetless),
        RenderPassProblem::RateMapTargetSize {
            target: [128, 128],
            screen: [SCREEN; 2],
        }
    );
    targetless.render_target_width = 33;
    targetless.render_target_height = 33;
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .render_command_encoder(&targetless)
        .unwrap()
        .end_encoding()
        .unwrap();

    // A scissor that cuts into the viewport over a map that is not the identity.
    let target = fixture.target(33);
    let refusal = fixture.draw(&half, &target, |encoder| {
        encoder
            .set_scissor_rect(ScissorRect::new(0, 0, 16, 64))
            .unwrap();
    });
    assert_eq!(
        refusal,
        Err(Error::NotImplemented(RuntimeGap::RateMapRegions(
            RegionRefusal::Scissor
        )))
    );
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}
