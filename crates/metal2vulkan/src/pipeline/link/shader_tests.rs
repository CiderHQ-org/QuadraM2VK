//! A render pipeline's in-shader depth test: `GraphicsState::shader_tests` from the fragment
//! stage's `RuntimeFact::ShaderTests`.
//!
//! A fragment function compiled for a draw whose tests may drop fragments
//! (`LoweringInputs::shader_tests`) tests its fragment's depth itself against a stand-in of the
//! depth attachment (`lower::fragment`'s `depth_test`). Link checks the pipeline's state against
//! the function's form:
//!
//! - the form follows the pipeline's samples: a function compiled for a pipeline of more than one
//!   sample (`LoweringInputs::multisampled`) tests its sample of a stand-in that holds each, and
//!   one linked into a pipeline of another kind is `NeedsInput(Multisampled)`;
//! - a multisampled stand-in (or the neutral that stands in for it) is a storage image of the
//!   pipeline's samples (`storageImageSampleCounts`); sample planes are single-sample;
//! - alpha to coverage drops a fragment's samples after the function tested and wrote them:
//!   not implemented;
//! - the binder asks for the function only for a pipeline with a depth attachment: one linked
//!   into a pipeline without one was compiled for another pipeline's state
//!   (`NeedsInput(FramebufferFetch)`);
//! - the stand-in holds 32-bit floats, Metal's `Depth32Float` and `Depth32Float_Stencil8` depth:
//!   another depth format (`Depth16Unorm`, `Depth24Unorm_Stencil8`) stores codes the function's
//!   float depth would have to round to as Metal does, not implemented (refinement 3), refused at
//!   the depth attachment.
//! - a function that runs the stencil test too (`RuntimeFact::ShaderStencilTest`) holds the stencil
//!   in a stand-in of the pipeline's samples (`R8_UINT`, a multisampled image at more than one),
//!   which the consumer copies to and from the stencil aspect, at more than one sample by
//!   maintenance8's image copies: one linked into a pipeline without a stencil attachment was
//!   compiled for another pipeline's state (`NeedsInput(FramebufferFetch)`), and at more than one
//!   sample a device without `maintenance8` is refused at the stencil attachment, naming it.

use crate::device::requirements::{VkFeature, VkProperty};
use crate::device::DeviceProfile;
use crate::error::{Attachment, Capability, Error, Feature, RuntimeInput, Unsupported};
use crate::library::FunctionArtifact;
use crate::lower::fragment::{SHADOW_FORMAT, STAND_IN_FORMAT};
use crate::metal::PixelFormat;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::graphics::{ShaderTestsUse, StandInUse, StandInView};
use crate::pipeline::link::RenderLinkInputs;

/// The in-shader depth test of the fragment stage `fragment` of render inputs `r` on `p` (see
/// the module docs).
pub fn shader_tests(
    r: &RenderLinkInputs,
    fragment: Option<&FunctionArtifact>,
    p: &DeviceProfile,
) -> Result<Option<ShaderTestsUse>, Error> {
    let Some((formatless, multisampled, planes)) = fragment.and_then(|f| {
        f.facts.iter().find_map(|fact| match *fact {
            RuntimeFact::ShaderTests {
                formatless,
                multisampled,
                planes,
            } => Some((formatless, multisampled, planes)),
            _ => None,
        })
    }) else {
        return Ok(None);
    };
    if !r.rasterization_enabled {
        return Ok(None);
    }
    let samples = r.raster_sample_count.max(1);
    if multisampled != (samples > 1) {
        return Err(Error::NeedsInput(RuntimeInput::Multisampled));
    }
    if r.alpha_to_coverage {
        return Err(Unsupported::not_implemented(Feature::FramebufferFetch).into());
    }
    if multisampled && !planes {
        let property = VkProperty::StorageImageSampleCount(samples);
        if !p.fields().enabled.properties.contains(&property) {
            return Err(Unsupported::needs(
                Feature::FramebufferFetch,
                Capability::Property(property),
            )
            .into());
        }
    }
    match r.depth_format {
        None => return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch)),
        Some(PixelFormat::Depth32Float | PixelFormat::Depth32FloatStencil8) => {}
        Some(_) => {
            return Err(
                Error::from(Unsupported::not_implemented(Feature::FramebufferFetch))
                    .at_attachment(Attachment::Depth),
            )
        }
    }
    let view = match planes {
        true => StandInView::SamplePlanes { layers: samples },
        false => StandInView::Image,
    };
    let stencil = match stencil_formatless(fragment) {
        Some(formatless) => {
            if r.stencil_format.is_none() {
                return Err(Error::NeedsInput(RuntimeInput::FramebufferFetch));
            }
            if samples > 1
                && !p
                    .fields()
                    .enabled
                    .features
                    .contains(&VkFeature::Maintenance8)
            {
                return Err(Error::from(Unsupported::needs(
                    Feature::FramebufferFetch,
                    Capability::Feature(VkFeature::Maintenance8),
                ))
                .at_attachment(Attachment::Stencil));
            }
            Some(StandInUse {
                format: SHADOW_FORMAT,
                formatless,
                samples,
                view: StandInView::Image,
            })
        }
        None => None,
    };
    Ok(Some(ShaderTestsUse {
        depth: StandInUse {
            format: STAND_IN_FORMAT,
            formatless,
            samples,
            view,
        },
        stencil,
    }))
}

/// Whether `fragment` runs the stencil test too, and whether it declares the stencil's stand-in
/// without a format: its `RuntimeFact::ShaderStencilTest`.
fn stencil_formatless(fragment: Option<&FunctionArtifact>) -> Option<bool> {
    fragment?.facts.iter().find_map(|fact| match *fact {
        RuntimeFact::ShaderStencilTest { formatless } => Some(formatless),
        _ => None,
    })
}
