//! A render pipeline's fragment imageblock: `GraphicsState::fragment_imageblock` from the fragment
//! stage's `RuntimeFact::FragmentImageblock`.
//!
//! The fragment function reads and writes the pass's imageblock storage, one plane of cells per
//! pixel (`ImageblockMechanism::StorageImageInterlock`), and writes it inside its body. Link
//! refuses, as not implemented, the pipelines whose Metal imageblock that plane does not model
//! exactly:
//!
//! - a multisampled pipeline (`rasterSampleCount` above 1): Metal's imageblock is per sample
//!   there;
//! - a pipeline whose fixed function can drop a fragment after the function wrote its imageblock:
//!   alpha to coverage, or a depth or stencil attachment when the function does not declare
//!   `[[early_fragment_tests]]`. The storage already holds the write, and whether Metal keeps an
//!   imageblock write of a fragment its tests or coverage drop is not read yet.

use crate::error::{Error, Feature, Unsupported};
use crate::library::FunctionArtifact;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::graphics::{FragmentImageblock, ImageblockMechanism};
use crate::pipeline::link::RenderLinkInputs;

/// The fragment imageblock of render inputs `r` whose fragment stage is `fragment` (see the
/// module docs).
pub fn fragment_imageblock(
    r: &RenderLinkInputs,
    fragment: Option<&FunctionArtifact>,
) -> Result<Option<FragmentImageblock>, Error> {
    let Some((words, writes, early_tests, master, data_size, alignment)) = fragment.and_then(|f| {
        f.facts.iter().find_map(|fact| match *fact {
            RuntimeFact::FragmentImageblock {
                words,
                writes,
                early_tests,
                master,
                data_size,
                alignment,
            } => Some((words, writes, early_tests, master, data_size, alignment)),
            _ => None,
        })
    }) else {
        return Ok(None);
    };
    let tested = r.depth_format.is_some() || r.stencil_format.is_some();
    let dropped_after_write = writes && (r.alpha_to_coverage || (tested && !early_tests));
    if r.raster_sample_count > 1 || dropped_after_write {
        return Err(Unsupported::not_implemented(Feature::Imageblocks).into());
    }
    Ok(Some(FragmentImageblock {
        mechanism: ImageblockMechanism::StorageImageInterlock,
        words,
        master,
        data_size,
        alignment,
    }))
}
