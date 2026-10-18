//! A signed normalized color attachment's minimum codes through a render pass that loads it and
//! draws, on the device the cfg selects, under the validation layer.
//!
//! Metal's pass, once it draws, writes each 32 x 32 tile a draw reaches back through its channels'
//! values, so a channel's minimum code (-128, or -32768), which reads as -1.0 as the code above it
//! does, is stored as that code (-127, -32767) in every texel of the tile, those no fragment writes
//! too (measured on an Apple M3 Max under macOS 27.0.1). An `RGB9E5Float` attachment's tiles are
//! written back the same way, through their values, so a word whose largest mantissa is below 256
//! under an exponent above 0 is stored normalized (its mantissas shifted up as its exponent steps
//! down; measured there too).
//!
//! Under `TileStorePolicy::Device`, the default, the runtime keeps the device's load and store
//! (docs/FAST-PATHS.md), so the texels keep their codes and words, and a draw in such a pass
//! announces `snorm-minimum-kept` or `shared-exponent-kept`, naming the attachment. Under
//! `TileStorePolicy::Exact` the load pass (`render/unused_bits_load.rs`) rewrites them as Metal's
//! store leaves them before the pass, and nothing is announced. On a driver whose pass stores them
//! as Metal's (`DriverFact::TileStoreIsMetals`: MoltenVK, whose pass is a Metal pass) the texels
//! take Metal's codes and words under either value, with no load pass and nothing announced.

use super::degrade_notices::counts;
use super::resolves::{lane_with, quiet, Lane, Shape, FULL, SIZE, WHITE};
use metal2vulkan::device::{DriverFact, TileStorePolicy};
use metal2vulkan::metal::{ColorWriteMask, PixelFormat, TextureUsage};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime::{
    BlitOption, Degradation, Error, LoadAction, PipelineOptions, Region, RenderPassDescriptor,
    ResourceOptions, StoreAction, Texture, TextureDescriptor,
};

/// The kind a draw in a pass that loads a single-sampled signed normalized color announces.
pub(super) const SNORM_MINIMUM: [Degradation; 1] = [Degradation::SnormMinimumKept];

/// The kind a draw in a pass that loads a single-sampled `RGB9E5Float` color announces.
pub(super) const SHARED_EXPONENT: [Degradation; 1] = [Degradation::SharedExponentKept];

/// The signed normalized formats Metal's tile store rewrites: each one's bytes a texel, the bytes
/// of a channel's minimum code and those of the code above it, which Metal's pass stores.
const FORMATS: [(PixelFormat, u64, &[u8], &[u8]); 6] = [
    (PixelFormat::R8Snorm, 1, &[0x80], &[0x81]),
    (PixelFormat::RG8Snorm, 2, &[0x80], &[0x81]),
    (PixelFormat::RGBA8Snorm, 4, &[0x80], &[0x81]),
    (PixelFormat::R16Snorm, 2, &[0x00, 0x80], &[0x01, 0x80]),
    (PixelFormat::RG16Snorm, 4, &[0x00, 0x80], &[0x01, 0x80]),
    (PixelFormat::RGBA16Snorm, 8, &[0x00, 0x80], &[0x01, 0x80]),
];

/// A pipeline of the fixture's full-screen triangle and white into `pf` under a write mask of
/// none of its channels: its draws write no texel.
fn unwritten(lane: &Lane, pf: PixelFormat) -> RenderPipelineDescriptor {
    let function = |name| {
        StageSource::Function(
            lane.library
                .new_function_with_name(name)
                .unwrap()
                .function_descriptor(),
        )
    };
    let mut descriptor = RenderPipelineDescriptor::new(function(FULL));
    descriptor.fragment = Some(function(WHITE));
    let mut color = ColorAttachmentDescriptor::new(pf);
    color.write_mask = ColorWriteMask::NONE;
    descriptor.color_attachments[0] = Some(color);
    descriptor
}

/// The bytes of a 4² texture of `texel` bytes a texel whose every channel holds `code`.
fn filled(texel: u64, code: &[u8]) -> Vec<u8> {
    code.repeat((texel * SIZE * SIZE) as usize / code.len())
}

/// Write `bytes` (`texel` bytes a texel) to every texel of the 4² texture `t`.
fn upload(t: &Texture, texel: u64, bytes: &[u8]) {
    t.replace_region(
        Region::new_2d(0, 0, SIZE, SIZE),
        0,
        0,
        bytes,
        texel * SIZE,
        0,
    )
    .unwrap();
}

/// A shared 4² render target of `pf` (`texel` bytes a texel) whose every channel holds the
/// minimum code `code`, and its bytes.
fn minimum_codes(lane: &Lane, pf: PixelFormat, texel: u64, code: &[u8]) -> (Texture, Vec<u8>) {
    let shape = Shape {
        size: SIZE,
        samples: 1,
        slices: 1,
        levels: 1,
        shared: true,
    };
    let t = lane.texture(pf, shape, TextureUsage::RENDER_TARGET);
    let bytes = filled(texel, code);
    upload(&t, texel, &bytes);
    (t, bytes)
}

/// A pass that loads `t` by `load` and stores it.
fn pass(t: &Texture, load: LoadAction) -> RenderPassDescriptor {
    let mut descriptor = RenderPassDescriptor::new();
    let attachment = &mut descriptor.color_attachments[0];
    attachment.texture = Some(t.clone());
    attachment.load_action = load;
    attachment.store_action = StoreAction::Store;
    descriptor
}

/// What a drawing pass of the lane's device leaves under `tile_store`: whether the device's store
/// is Metal's (`DriverFact::TileStoreIsMetals`), and whether the load pass rewrites the texels.
fn expected(lane: &Lane, tile_store: TileStorePolicy) -> (bool, bool) {
    let metals = lane
        .device
        .properties()
        .driver_facts
        .contains(&DriverFact::TileStoreIsMetals);
    (metals, tile_store == TileStorePolicy::Exact && !metals)
}

/// The load passes the lane's device recorded so far.
fn load_passes(lane: &Lane) -> u64 {
    lane.device.work_counts().unused_bits_load_passes
}

/// A pass that loads a 4² target of each signed normalized format holding the minimum code in
/// every channel and draws under a write mask of none of its channels, under `tile_store`: where
/// the device's store is Metal's or the load pass rewrites the codes (one load pass), every
/// texel takes the code above and nothing is announced; else every code stays and
/// `snorm-minimum-kept` is announced once (the device asks for it at each occurrence), naming
/// color 0 and its format. A pass that loads the minimum codes again and draws nothing keeps
/// every one, records no load pass and announces nothing, as Metal's keeps every texel.
fn snorm_minimum_codes(tile_store: TileStorePolicy) {
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = SNORM_MINIMUM.into_iter().collect();
        options.policy.tile_store = tile_store;
    });
    let (metals, rewritten) = expected(&lane, tile_store);
    eprintln!("{tile_store:?}: Metal's store {metals}, rewritten {rewritten}");
    for (pf, texel, code, above) in FORMATS {
        let pipeline = lane
            .device
            .new_render_pipeline_state(&unwritten(&lane, pf), &PipelineOptions::default())
            .unwrap();
        let (t, bytes) = minimum_codes(&lane, pf, texel, code);
        let before = load_passes(&lane);
        lane.run(&pass(&t, LoadAction::Load), |l, e, h| {
            l.draw(e, h, &pipeline, 0.5)
        })
        .unwrap();
        let loads = load_passes(&lane) - before;
        let got = lane.read(&t, (0, 0), texel, BlitOption::None);
        let notices = messages.take_notices();
        assert_eq!(
            loads,
            u64::from(rewritten),
            "{pf:?} {tile_store:?}: load passes"
        );
        if metals || rewritten {
            let raised = filled(texel, above);
            assert!(
                got == raised,
                "{pf:?} {tile_store:?}: not Metal's codes: {got:02x?}"
            );
            assert!(
                !counts(&notices).contains_key(&SNORM_MINIMUM[0]),
                "{pf:?} {tile_store:?}: announced where the codes are Metal's"
            );
        } else {
            assert!(got == bytes, "{pf:?}: the codes moved: {got:02x?}");
            assert_eq!(counts(&notices).get(&SNORM_MINIMUM[0]), Some(&1), "{pf:?}");
            let named = notices.iter().any(|m| {
                m.degradation.as_ref().is_some_and(|n| {
                    n.degradation == SNORM_MINIMUM[0]
                        && n.object.contains(&format!("color 0, {pf:?}"))
                })
            });
            assert!(named, "{pf:?}: the notice names no color 0");
        }
        upload(&t, texel, &bytes);
        let before = load_passes(&lane);
        lane.run(&pass(&t, LoadAction::Load), |_, _, _| {}).unwrap();
        assert_eq!(
            load_passes(&lane),
            before,
            "{pf:?}: a pass of no draw's load pass"
        );
        let got = lane.read(&t, (0, 0), texel, BlitOption::None);
        assert!(
            got == bytes,
            "{pf:?}: a pass of no draw moved the codes: {got:02x?}"
        );
        let notices = messages.take_notices();
        assert!(
            !counts(&notices).contains_key(&SNORM_MINIMUM[0]),
            "{pf:?}: a pass of no draw"
        );
    }
    quiet(lane, &messages);
}

/// [`snorm_minimum_codes`] under the default, the device's load and store.
#[test]
fn a_drawing_pass_keeps_a_snorm_minimum_code_and_says_so() {
    snorm_minimum_codes(TileStorePolicy::Device);
}

/// [`snorm_minimum_codes`] under the exact tile store, the load pass's rewrite.
#[test]
fn the_exact_tile_store_raises_a_snorm_minimum_code() {
    snorm_minimum_codes(TileStorePolicy::Exact);
}

/// A pass that loads a 4² `RGB9E5Float` target whose every word is unnormalized (mantissas 1, 2
/// and 3 under the exponent 20: its largest mantissa below 256) and draws under a write mask of
/// none of its channels, under `tile_store`: where the device's store is Metal's or the load pass
/// rewrites the words (one load pass), every word takes Metal's (mantissas 128, 256 and 384 under
/// the exponent 13) and nothing is announced; else every word stays and `shared-exponent-kept` is
/// announced once (the device asks for it at each occurrence), naming color 0 and its format. A
/// pass that loads it and draws nothing announces nothing. A device that renders no
/// `RGB9E5Float` (refusing its texture or pipeline state) takes neither, and says so.
fn shared_exponent_words(tile_store: TileStorePolicy) {
    let pf = PixelFormat::RGB9E5Float;
    let (messages, _cache, lane) = lane_with(|options| {
        options.announce_each = SHARED_EXPONENT.into_iter().collect();
        options.policy.tile_store = tile_store;
    });
    let name = lane.device.name();
    let (metals, rewritten) = expected(&lane, tile_store);
    let mut descriptor = TextureDescriptor::texture_2d(pf, SIZE, SIZE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = ResourceOptions::shared();
    let t = match lane.device.new_texture(&descriptor) {
        Ok(t) => t,
        Err(Error::Unsupported(why)) => {
            println!("SHARED-EXPONENT-SKIP {pf:?} on {name}: {why:?}");
            quiet(lane, &messages);
            return;
        }
        Err(e) => panic!("{pf:?} on {name}: {e:?}"),
    };
    let pipeline = match lane
        .device
        .new_render_pipeline_state(&unwritten(&lane, pf), &PipelineOptions::default())
    {
        Ok(p) => p,
        Err(Error::Unsupported(why)) => {
            println!("SHARED-EXPONENT-SKIP {pf:?} pipeline on {name}: {why:?}");
            quiet(lane, &messages);
            return;
        }
        Err(e) => panic!("{pf:?} pipeline on {name}: {e:?}"),
    };
    let word: u32 = 1 | 2 << 9 | 3 << 18 | 20 << 27;
    let metal: u32 = 128 | 256 << 9 | 384 << 18 | 13 << 27;
    let bytes = word.to_le_bytes().repeat((SIZE * SIZE) as usize);
    t.replace_region(Region::new_2d(0, 0, SIZE, SIZE), 0, 0, &bytes, 4 * SIZE, 0)
        .unwrap();
    let before = load_passes(&lane);
    lane.run(&pass(&t, LoadAction::Load), |l, e, h| {
        l.draw(e, h, &pipeline, 0.5)
    })
    .unwrap();
    let loads = load_passes(&lane) - before;
    let got = lane.read(&t, (0, 0), 4, BlitOption::None);
    let notices = messages.take_notices();
    assert_eq!(
        loads,
        u64::from(rewritten),
        "{pf:?} {tile_store:?} on {name}: load passes"
    );
    if metals || rewritten {
        let normalized = metal.to_le_bytes().repeat((SIZE * SIZE) as usize);
        assert!(
            got == normalized,
            "{pf:?} {tile_store:?} on {name}: not Metal's words: {got:02x?}"
        );
        assert!(
            !counts(&notices).contains_key(&SHARED_EXPONENT[0]),
            "{pf:?} {tile_store:?} on {name}: announced where the words are Metal's"
        );
    } else {
        assert!(
            got == bytes,
            "{pf:?} on {name}: the words moved: {got:02x?}"
        );
        assert_eq!(
            counts(&notices).get(&SHARED_EXPONENT[0]),
            Some(&1),
            "{pf:?} on {name}"
        );
        let named = notices.iter().any(|m| {
            m.degradation.as_ref().is_some_and(|n| {
                n.degradation == SHARED_EXPONENT[0]
                    && n.object.contains(&format!("color 0, {pf:?}"))
            })
        });
        assert!(named, "{pf:?} on {name}: the notice names no color 0");
    }
    lane.run(&pass(&t, LoadAction::Load), |_, _, _| {}).unwrap();
    let notices = messages.take_notices();
    assert!(
        !counts(&notices).contains_key(&SHARED_EXPONENT[0]),
        "{pf:?} on {name}: a pass of no draw"
    );
    quiet(lane, &messages);
}

/// [`shared_exponent_words`] under the default, the device's load and store.
#[test]
fn a_drawing_pass_keeps_an_unnormalized_shared_exponent_word_and_says_so() {
    shared_exponent_words(TileStorePolicy::Device);
}

/// [`shared_exponent_words`] under the exact tile store, the load pass's rewrite.
#[test]
fn the_exact_tile_store_normalizes_a_shared_exponent_word() {
    shared_exponent_words(TileStorePolicy::Exact);
}
