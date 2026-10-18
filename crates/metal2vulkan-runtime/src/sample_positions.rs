//! Sample positions (`setSamplePositions:count:`) and the store action option that names them,
//! `MTLStoreActionOptionCustomSamplePositions`.
//!
//! **Metal**, measured on an M4 with macOS 27.0.1 (Metal probes, each case with and without API
//! validation):
//!
//! - `areProgrammableSamplePositionsSupported` is true. Custom positions take the device's sample
//!   counts above 1 (2 and 4); one position, or a count the device has not, is refused, and so is a
//!   coordinate outside `[0, 1)` (each an assertion of `setSamplePositions:count:`, or of the
//!   encoder's creation). The positions' count must be the pass's samples for Metal's validation;
//!   plain Metal takes fewer and puts the samples past them at the pixel's corner, (0, 0), and so
//!   does the runtime (an M4 with macOS 27.0.1: 2 positions in a pass of 4
//!   samples, each sample's position read through a varying at it). Plain Metal stops at a count of
//!   1 or 3 (`setSamplePositions:count:`), and the runtime keeps refusing more positions than
//!   samples (a Metal probe on an M4: Metal ends the process).
//! - The descriptor keeps the positions as set (`getSamplePositions` answers 0.53 for 0.53),
//!   and the pass samples at each coordinate rounded to the nearest sixteenth of a pixel, a tie
//!   up (0.53 at 0.5; 0.53125, 0.55 and 0.5624 at 0.5625), and at most 15/16 (0.97 and 0.99).
//!   Every sample of a pixel half covered by a triangle lands where the positions put it.
//! - `MTLStoreActionOptionCustomSamplePositions` applies, for Metal's validation, to the store
//!   actions that keep the samples, `Store` and `StoreAndMultisampleResolve`, of a color or depth
//!   attachment; under `DontCare` or `MultisampleResolve` it stops, the descriptor's at the
//!   encoder's creation and a late one at `endEncoding`, and so does the descriptor's without
//!   sample positions (a late one set without them is taken), and options set late over a store
//!   action the descriptor knew ("store actions known at render command encoder init time may not
//!   be mutated"). The option changes nothing on Apple silicon ("Store action options have no
//!   effect on Apple Silicon", Apple's deprecation notice), and plain Metal runs each of those
//!   forms of a color or depth attachment as it runs the pass without the option (Metal probe cases
//!   on an M4: `opt_dontcare`, `opt_late_dontcare`, `opt_store_no_positions`, `sp_options_only`,
//!   `sp_custom_storeopt`, `sp_late_options`, `opt_late_over_known`, `opt_depth_dontcare`): the
//!   runtime does too, and checks only a stencil attachment's, unmeasured, as Metal's validation
//!   does; a depth or stencil attachment's set late over a known store action stays refused.
//!
//! **With a depth or stencil attachment**, measured on the M3 Max (macOS 26.7) and an M4 (macOS
//! 27.0.1), each case with and without API validation, the same (a probe): each sample's depth is
//! the triangle's depth at its position, rounded as above (a plane of depths across a 2x2
//! `Depth32Float` target, at four positions and two, and `Depth16Unorm`'s quantized), and the depth
//! test compares there; the stencil test covers the samples there, of a `Depth32Float_Stencil8`
//! texture and of a `Stencil8` one alone; a memoryless depth texture takes them. The device's own
//! positions are Vulkan's standard ones (4x: (0.375, 0.125), (0.875, 0.375), (0.125, 0.625),
//! (0.625, 0.875); 2x: (0.75, 0.75), (0.25, 0.25)). Positions move rasterization alone: a later
//! pass that loads the depth, at the device's positions or at others, keeps every stored sample,
//! and one that draws at others compares each sample's stored depth with the depth at its new
//! position.
//!
//! **The mechanism.** `VK_EXT_sample_locations`: every pipeline a draw of the pass binds is made
//! with `VkPipelineSampleLocationsStateCreateInfoEXT`, the pass's positions on a grid of one
//! pixel (`pipeline/graphics.rs`), each coordinate already rounded as Metal rounds it, so it
//! lies in the range and on the sub-pixel grid every device with the extension takes exactly
//! (Vulkan guarantees `[0, 0.9375]` and four sub-pixel bits). The pass takes custom positions at
//! the counts the device's `sampleLocationSampleCounts` lists, on a device with
//! `variableSampleLocations` (every committed profile), where no pass needs them at its begin;
//! [`are_programmable_sample_positions_supported`](super::DeviceProperties::are_programmable_sample_positions_supported)
//! answers whether that is every count above 1 the device renders. The store action option keeps
//! nothing more: Vulkan stores each sample by its index, wherever the pass rasterized it.
//!
//! A depth/stencil attachment rasterizes at custom locations where its image is made with
//! `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`
//! (`VUID-vkCmdDraw-sampleLocationsEnable-02689`): every multisampled image of a depth format
//! the runtime makes to render into (a texture's, memoryless too, the separate depth and
//! stencil's scratch, a stand-in) takes it where the device takes custom positions at its count
//! ([`compatible_depth`]). Vulkan then needs the locations only where the image's layout
//! changes (`VkRenderPassSampleLocationsBeginInfoEXT`, `VkSampleLocationsInfoEXT` on a barrier),
//! and the runtime holds every image it makes in `GENERAL` from its first use on, so the pass
//! needs none at its begin.
//!
//! **The pass begins at its positions**. MoltenVK begins a Metal render pass at the positions of
//! the pipeline bound when the Vulkan one begins, and ends and begins it again at a draw whose
//! pipeline's positions differ; a memoryless attachment keeps nothing across that, so its clear was
//! lost and every sample passed a depth test against it (`runtime_lane::sample_positions`'
//! `dz_custom4_memoryless`, the M3 Max and the M4 alike). Each render pass instance of a pass with
//! custom positions binds the pipeline of its first draw, which has them, before it begins
//! (`render.rs`); elsewhere the bind is idle.
//!
//! **A stencil-only attachment**. Vulkan lets no image without a depth aspect take the flag
//! (`S8_UINT`, `VUID-VkImageCreateInfo-flags-01533`), where Metal takes positions with a `Stencil8`
//! texture alone. Such a pass renders its stencil through a scratch image of a combined format made
//! with the flag, its stencil copied in where the pass loads it and back where it stores it, as
//! separate depth and stencil textures are (`depth_stencil_scratch.rs`); the scratch's depth aspect
//! is the pass's only, which no draw tests or writes (Metal stops a draw that would in a pass
//! without a depth attachment).
//!
//! **A caller's image not made for them**. A depth/stencil attachment of one texture whose image
//! lacks the flag (a caller's image) renders at the positions through a stand-in of its format made
//! with it, each aspect copied in where the pass loads it and back where it stores it
//! (`attachment_stand_in.rs`), announced once (`Degradation::PositionsThroughScratch`).
//!
//! **A driver that keeps the depth per sample only through a copy**. NVIDIA's 615.71.09 (measured
//! on the RTX 3090) keeps a depth a pass rendered at custom positions as the triangle's plane and
//! answers each later reader at the reader's own positions: a shader's read of a sample at the
//! standard ones, a later pass at other positions at its own (so its test under `Less` drew nothing
//! where Metal's drew three samples a pixel), with or without layout transitions that carry the
//! locations right after the pass. The pass's own test compares at the custom positions, and a copy
//! recorded after the pass keeps each sample's depth at them. On such a driver
//! (`metal2vulkan::device::DriverFact::PositionedDepthReadAtStandardPositions`) a depth/stencil
//! attachment of one texture renders at custom positions through the stand-in above even where
//! its image was made with the flag, announced the same way; an image of that kind that does not
//! copy both ways is refused (`NotImplemented(SamplePositionsAtBegin)`).
//!
//! **Not built** (`NotImplemented(SamplePositionsAtBegin)`): custom positions in a pass whose
//! depth/stencil attachment is a caller's image not so made, of a format the device does not copy
//! both ways, which the stand-in's copies need; a stencil-only attachment on a device without a
//! combined format that takes them; and on a device without `variableSampleLocations`, which takes
//! positions only at a pass's begin.

use super::error::{invalid, Error, Problem, RuntimeGap};
use super::render_pass::{Attachment, RenderPassProblem};
use super::render_values::{SamplePosition, StoreAction, StoreActionOptions};
use super::Device;
use metal2vulkan::device::{DeviceProfile, VkProperty};
use metal2vulkan::vk::{Format, ImageCreateFlags, SampleCountFlags};

fn problem(call: &'static str, p: RenderPassProblem) -> Error {
    invalid(call, Problem::RenderPass(p))
}

/// A sample position in sixteenths of a pixel, `x` then `y`, each at most 15.
pub(super) type Sixteenths = [u8; 2];

/// Whether `device` takes `positions` in a pass of `samples` samples as [`grid`] takes them,
/// whatever the pass's depth/stencil attachment: positions inside the pixel, at a count the
/// device takes custom positions at, no more than the pass's samples, which it takes them at
/// too, on a device with `variableSampleLocations`.
pub(super) fn taken(device: &Device, positions: &[SamplePosition], samples: u32) -> bool {
    let inside = |v: f32| (0.0..1.0).contains(&v);
    let properties = &device.compile_device().profile().enabled().properties;
    let count = positions.len() as u32;
    !positions.is_empty()
        && positions.iter().all(|p| inside(p.x) && inside(p.y))
        && count <= samples
        && count > 1
        && properties.contains(&VkProperty::SampleLocationSampleCount(count))
        && properties.contains(&VkProperty::SampleLocationSampleCount(samples))
        && properties.contains(&VkProperty::VariableSampleLocations)
}

/// The pass's sample positions on `device`, in sixteenths of a pixel (empty for the device's
/// own), for a pass of `samples` samples; `depth_stencil` is the create flags of its
/// depth/stencil attachment's image, where it has one. Refused as Metal refuses them, and typed
/// where not built (the module docs).
pub(super) fn grid(
    call: &'static str,
    device: &Device,
    positions: &[SamplePosition],
    samples: u32,
    depth_stencil: Option<ImageCreateFlags>,
) -> Result<Vec<Sixteenths>, Error> {
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    let inside = |v: f32| (0.0..1.0).contains(&v);
    if let Some(index) = positions.iter().position(|p| !inside(p.x) || !inside(p.y)) {
        let index = index as u32;
        return Err(problem(call, RenderPassProblem::SamplePosition { index }));
    }
    let count = positions.len() as u64;
    let properties = &device.compile_device().profile().enabled().properties;
    let taken = u32::try_from(count).is_ok_and(|count| {
        count > 1 && properties.contains(&VkProperty::SampleLocationSampleCount(count))
    });
    if !taken {
        return Err(problem(
            call,
            RenderPassProblem::SamplePositionCount { count },
        ));
    }
    // Fewer positions than the pass's samples put the samples past them at the pixel's corner,
    // as plain Metal does (the module docs); more stay refused.
    if count > u64::from(samples) {
        let positions = count;
        return Err(problem(
            call,
            RenderPassProblem::SamplePositionsSamples { positions, samples },
        ));
    }
    // A depth/stencil attachment rasterizes at custom locations only where its image was made
    // for them (`compatible_depth`).
    let incompatible = depth_stencil.is_some_and(|flags| {
        !flags.contains(ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT)
    });
    if incompatible
        || !properties.contains(&VkProperty::VariableSampleLocations)
        || !properties.contains(&VkProperty::SampleLocationSampleCount(samples))
    {
        return Err(Error::NotImplemented(RuntimeGap::SamplePositionsAtBegin));
    }
    let corner = [0, 0];
    Ok(positions
        .iter()
        .map(|p| [sixteenths(p.x), sixteenths(p.y)])
        .chain(std::iter::repeat(corner))
        .take(samples as usize)
        .collect())
}

/// A coordinate in `[0, 1)` as Metal samples at it: the nearest sixteenth, a tie up, at most
/// 15/16.
fn sixteenths(v: f32) -> u8 {
    ((v * 16.0 + 0.5).floor() as u8).min(15)
}

/// Check the stencil attachment's (`which`) store action `options` under `store`: the
/// descriptor's (`positioned`: whether the pass has sample positions), or at `end_encoding` the
/// encoder's (`None`). Refused as Metal's validation refuses them (the module docs).
pub(super) fn check_options(
    call: &'static str,
    which: Attachment,
    options: StoreActionOptions,
    store: StoreAction,
    positioned: Option<bool>,
) -> Result<(), Error> {
    if !options.contains(StoreActionOptions::CUSTOM_SAMPLE_POSITIONS) {
        return Ok(());
    }
    if positioned == Some(false) {
        return Err(problem(
            call,
            RenderPassProblem::StoreActionOptionsWithoutPositions(which),
        ));
    }
    match store {
        StoreAction::Store | StoreAction::StoreAndMultisampleResolve | StoreAction::Unknown => {
            Ok(())
        }
        _ => Err(problem(call, RenderPassProblem::StoreActionOptions(which))),
    }
}

/// The create flag an image of `format` and `samples` takes so that a pass with custom sample
/// positions may attach it as its depth/stencil attachment
/// (`VUID-vkCmdDraw-sampleLocationsEnable-02689`): `SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT` for a
/// multisampled image whose format has a depth aspect (which the flag needs,
/// `VUID-VkImageCreateInfo-flags-01533`), on a device that takes custom positions at its count
/// with `variableSampleLocations`; no flag otherwise.
pub(super) fn compatible_depth(
    profile: &DeviceProfile,
    format: Format,
    samples: SampleCountFlags,
) -> ImageCreateFlags {
    let properties = &profile.enabled().properties;
    let count = samples.as_raw();
    let located = count > 1
        && properties.contains(&VkProperty::VariableSampleLocations)
        && properties.contains(&VkProperty::SampleLocationSampleCount(count));
    match located && format.block().is_some_and(|b| b.depth.is_some()) {
        true => ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT,
        false => ImageCreateFlags::empty(),
    }
}

/// Whether `device` takes custom sample positions, without them at a pass's begin, at every
/// sample count above 1 of `counts`, the sample counts it renders at.
pub(super) fn programmable(profile: &DeviceProfile, counts: SampleCountFlags) -> bool {
    let properties = &profile.enabled().properties;
    properties.contains(&VkProperty::VariableSampleLocations)
        && (1..7).map(|shift| 1u32 << shift).all(|count| {
            !counts.contains(SampleCountFlags::from_raw(count))
                || properties.contains(&VkProperty::SampleLocationSampleCount(count))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal's rounding, measured (the module docs): the nearest sixteenth, a tie up, at most
    /// 15/16.
    #[test]
    fn coordinates_round_as_metal_rounds_them() {
        for (v, expected) in [
            (0.0, 0),
            (0.25, 4),
            (0.5, 8),
            (0.51, 8),
            (0.53, 8),
            (0.53125, 9),
            (0.55, 9),
            (0.5624, 9),
            (0.5625, 9),
            (0.97, 15),
            (0.99, 15),
            (0.999_999_9, 15),
        ] {
            assert_eq!(sixteenths(v), expected, "{v}");
        }
    }

    /// A stencil attachment's option applies to the store actions that keep the samples; the
    /// descriptor's needs positions, and a late one does not.
    #[test]
    fn the_option_applies_where_metal_takes_it() {
        let custom = StoreActionOptions::CUSTOM_SAMPLE_POSITIONS;
        let which = Attachment::Stencil;
        let check = |store, positioned| check_options("t", which, custom, store, positioned);
        for store in [StoreAction::Store, StoreAction::StoreAndMultisampleResolve] {
            assert_eq!(check(store, Some(true)), Ok(()));
            assert_eq!(check(store, None), Ok(()));
        }
        assert_eq!(check(StoreAction::Unknown, Some(true)), Ok(()));
        let options = problem("t", RenderPassProblem::StoreActionOptions(which));
        for store in [StoreAction::DontCare, StoreAction::MultisampleResolve] {
            assert_eq!(check(store, Some(true)), Err(options.clone()));
            assert_eq!(check(store, None), Err(options.clone()));
        }
        let without = problem(
            "t",
            RenderPassProblem::StoreActionOptionsWithoutPositions(which),
        );
        assert_eq!(check(StoreAction::Store, Some(false)), Err(without));
        assert_eq!(
            check_options(
                "t",
                which,
                StoreActionOptions::NONE,
                StoreAction::DontCare,
                Some(false)
            ),
            Ok(())
        );
    }
}
