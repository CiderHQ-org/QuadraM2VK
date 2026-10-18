//! Metal's private fixed-function pipeline state of OpenGL on Metal (Apple SPI of
//! `MTLRenderPipelineDescriptor`): which properties the translation builds, and the refusal of the
//! rest.
//!
//! Measured on an Apple M4 (macOS 27.0.1; owned Objective-C and MSL probes setting each property
//! by key-value coding):
//!
//! - `sampleMask` ANDs the coverage of a pipeline with a fragment function, the bits past the
//!   pipeline's sample count ignored, after the function runs: the function runs for every
//!   rasterized sample, masked or not, its `[[sample_mask]]` reads the rasterized coverage, and the
//!   color, depth and stencil writes and the visibility count then keep only the mask's samples
//!   (ANDed with the function's own `[[sample_mask]]` output and alpha to coverage); under
//!   `[[early_fragment_tests]]` only the colors. A pipeline without a fragment function ignores the
//!   mask: its depth and stencil writes and its visibility count keep every sample. Vulkan's
//!   `pSampleMask` ([`GraphicsState::sample_mask`]) masks before the function runs, so it is
//!   Metal's only where the function cannot tell; elsewhere the function applies the mask itself in
//!   its `SampleMask` output (`LoweringInputs::sample_mask`, [`shader_sample_mask`]), which Vulkan
//!   applies after it runs, and after early tests to the colors alone ([`sample_mask`]).
//! - `clipDistanceEnableMask`, `pointCoordLowerLeft`, `pointSizeOutputVS`, `twoSideEnabled`,
//!   `vertexDepthCompareClampMask`, `depthStencilWriteDisabled`, `sampleCoverageInvert` and
//!   `needsCustomBorderColorSamplers` (render and compute) change nothing Metal draws, at any
//!   value tried: they are kept and read by nothing. A sampler's private custom border color
//!   (`borderColorSPI` 3 with its four values) samples transparent black, the values ignored,
//!   with the flag or without: the raw border color 3 the runtime already
//!   samples so (`sampler_border_color`).
//! - `sampleCoverage`, `alphaTestEnabled`, `logicOperationEnabled`, `pointSmoothEnabled` and
//!   `fragmentDepthCompareClampMask` change what Metal draws; until the translation builds one,
//!   a value other than Metal's default refuses the pipeline `NotImplemented`
//!   ([`Feature::PrivateRenderState`]) naming it.
//! - `textureWriteRoundingMode` 2 rounds a fragment function's texture writes to half-float and
//!   `RG11B10Float` textures to nearest even, which the translation has not built: such a write is
//!   refused naming it (`PrivateRenderField::TextureWriteRounding`), where the function's slot
//!   declares such a format, and at the draw where one is bound (`lower::texture`'s
//!   `mode_rounded_write`). Its color writes are built.
//!
//! [`GraphicsState::sample_mask`]: crate::pipeline::GraphicsState::sample_mask

use crate::error::{Error, Feature, PrivateRenderField, RuntimeInput, Unsupported};
use crate::library::FunctionArtifact;
use crate::pipeline::descriptor::{PrivateRenderState, RenderPipelineDescriptor};
use crate::pipeline::stage_io::StageBuiltin;
use crate::pipeline::GraphicsState;

/// The refusal of `field` at a value the translation has not built.
fn refused(field: PrivateRenderField) -> Error {
    Error::Unsupported(Unsupported::not_implemented(Feature::PrivateRenderState(
        field,
    )))
}

/// `Ok` where every property of `s` is Metal's default or built (the module docs), else the
/// refusal naming the first that is not.
pub(crate) fn check_render(s: &PrivateRenderState) -> Result<(), Error> {
    use PrivateRenderField as F;
    let unbuilt = [
        (s.sample_coverage != 1.0, F::SampleCoverage),
        (s.alpha_test_enabled, F::AlphaTest),
        (s.logic_operation_enabled, F::LogicOperation),
        (s.point_smooth_enabled, F::PointSmooth),
        (
            s.fragment_depth_compare_clamp_mask != 0,
            F::FragmentDepthCompareClamp,
        ),
    ];
    match unbuilt.into_iter().find(|(set, _)| *set) {
        Some((_, field)) => Err(refused(field)),
        None => Ok(()),
    }
}

/// The sample mask Metal applies to the pipeline `d`: the private mask's bits below the sample
/// count, where `d` has a fragment function under enabled rasterization; `None` where that keeps
/// every sample, or `d` has no fragment stage, which ignores the mask.
pub(crate) fn applied_sample_mask(d: &RenderPipelineDescriptor) -> Option<u32> {
    if d.fragment.is_none() || !d.rasterization_enabled {
        return None;
    }
    let all = match d.raster_sample_count {
        n if n >= 32 => u32::MAX,
        n => (1u32 << n) - 1,
    };
    // Metal ignores the bits past the sample count, so the high word is dropped.
    let mask = (d.private.sample_mask as u32) & all;
    (mask != all).then_some(mask)
}

/// Whether the fragment function `a` could tell Vulkan's sample mask, applied before it runs,
/// from Metal's, applied after: it writes memory (Metal runs it for the masked samples too), reads
/// `[[sample_mask]]` (Metal's is the rasterized coverage), or declares `[[early_fragment_tests]]`
/// (Metal's mask then keeps every sample's depth, stencil and visibility count, and only the
/// colors' masked samples unwritten).
fn observes_sample_mask(a: &FunctionArtifact) -> bool {
    let io = &a.interface.io;
    a.interface.writes_memory
        || io.builtins.contains(&StageBuiltin::SampleMaskIn)
        || io.early_fragment_tests
}

/// The private sample mask the fragment function of `d`, compiled without it as `a`, applies
/// itself (`LoweringInputs::sample_mask`): the mask Metal applies, where `a` could tell Vulkan's
/// from it ([`observes_sample_mask`]); `None` elsewhere.
pub(crate) fn shader_sample_mask(
    d: &RenderPipelineDescriptor,
    a: &FunctionArtifact,
) -> Option<u32> {
    applied_sample_mask(d).filter(|_| observes_sample_mask(a))
}

/// [`GraphicsState::sample_mask`] for the pipeline `d`, its fragment function `fragment` and
/// its graphics state `g`, or the refusal where neither Vulkan's mask nor the function's own
/// (`StageIo::sample_mask`) is Metal's. A function that could tell Vulkan's mask from Metal's
/// applies the mask itself, and the pipeline has no Vulkan mask; one compiled without it, or for
/// another mask, is compiled for the pipeline again (`NeedsInput(SampleMask)`). A pipeline whose
/// fragment function writes an attachment itself (a storage attachment, a shadow stencil or an
/// imageblock, none measured under the mask) is refused.
pub(crate) fn sample_mask(
    d: &RenderPipelineDescriptor,
    fragment: Option<&FunctionArtifact>,
    g: &GraphicsState,
) -> Result<Option<u32>, Error> {
    let applied = applied_sample_mask(d);
    let own = fragment.and_then(|f| f.interface.io.sample_mask);
    if own.is_some() && own != applied {
        return Err(Error::NeedsInput(RuntimeInput::SampleMask));
    }
    let Some(mask) = applied else {
        return Ok(None);
    };
    let writes_itself = !g.storage_attachments.is_empty()
        || g.shadow_stencil.is_some()
        || g.fragment_imageblock.is_some();
    if writes_itself {
        return Err(refused(PrivateRenderField::SampleMask));
    }
    match (own, fragment) {
        (Some(_), _) => Ok(None),
        (None, Some(f)) if observes_sample_mask(f) => {
            Err(Error::NeedsInput(RuntimeInput::SampleMask))
        }
        (None, _) => Ok(Some(mask)),
    }
}

/// The text the key of the render pipeline `d` appends for its private state: empty at Metal's
/// defaults and wherever only properties that change nothing differ, so every other pipeline
/// keeps its key.
pub(crate) fn render_key_fields(d: &RenderPipelineDescriptor) -> String {
    match applied_sample_mask(d) {
        Some(mask) => format!(" sample_mask {mask:#x}"),
        None => String::new(),
    }
}
