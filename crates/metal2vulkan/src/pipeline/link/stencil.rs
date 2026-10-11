//! A render pipeline's shader-side stencil: `GraphicsState::shadow_stencil` from the fragment
//! stage's `RuntimeFact::ShadowStencil`.
//!
//! On a device without `VK_EXT_shader_stencil_export`, a fragment function's `[[stencil]]`
//! output is tested and written by the function itself against a shadow of the stencil
//! (`lower::fragment`'s `stencil`). Link checks the pipeline's state against the function's form:
//!
//! - the form follows the pipeline's samples: a function compiled for a pipeline of more than one
//!   sample (`LoweringInputs::multisampled`) tests each sample of a multisampled shadow, and one
//!   linked into a pipeline of another kind is `NeedsInput(Multisampled)`;
//! - a multisampled shadow (or the neutral that stands in for it) is a storage image of the
//!   pipeline's samples (`storageImageSampleCounts`), and the consumer copies it to and from the
//!   stencil with image copies (`maintenance8`): where the device lacks either, the output is
//!   refused naming the extension or what the form lacks;
//! - alpha to coverage changes the fragment's samples before Metal's stencil test, after the
//!   function's write: refused as needing the extension;
//! - a pipeline without a stencil attachment states no shadow (Metal's output reaches no
//!   attachment): the binder binds a neutral image and the function tests nothing.

use crate::device::requirements::{VkExtension, VkFeature, VkProperty};
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, RuntimeInput, Unsupported};
use crate::library::FunctionArtifact;
use crate::lower::fragment::SHADOW_FORMAT;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::graphics::ShadowStencilUse;
use crate::pipeline::link::RenderLinkInputs;

/// The shader-side stencil of the fragment stage `fragment` of render inputs `r` on `p` (see the
/// module docs).
pub fn shadow_stencil(
    r: &RenderLinkInputs,
    fragment: Option<&FunctionArtifact>,
    p: &DeviceProfile,
) -> Result<Option<ShadowStencilUse>, Error> {
    let Some((formatless, multisampled)) = fragment.and_then(|f| {
        f.facts.iter().find_map(|fact| match *fact {
            RuntimeFact::ShadowStencil {
                formatless,
                multisampled,
            } => Some((formatless, multisampled)),
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
        return Err(Unsupported::needs(
            Feature::StencilExport,
            Capability::Extension(VkExtension::ExtShaderStencilExport),
        )
        .into());
    }
    let attached = r.stencil_format.is_some();
    if samples > 1 {
        let f = p.fields();
        let mut missing = Vec::new();
        let property = VkProperty::StorageImageSampleCount(samples);
        if !f.enabled.properties.contains(&property) {
            missing.push(Capability::Property(property));
        }
        if attached && !f.enabled.features.contains(&VkFeature::Maintenance8) {
            missing.push(Capability::Feature(VkFeature::Maintenance8));
        }
        if !missing.is_empty() {
            return Err(Unsupported::needs_any_of(
                Feature::StencilExport,
                vec![
                    vec![Capability::Extension(VkExtension::ExtShaderStencilExport)],
                    missing,
                ],
            )
            .into());
        }
    }
    Ok(attached.then_some(ShadowStencilUse {
        format: SHADOW_FORMAT,
        formatless,
        samples,
    }))
}
