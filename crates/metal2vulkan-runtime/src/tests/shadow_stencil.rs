//! The shader-side stencil's shadow (`shadow_stencil.rs`).

use super::super::shadow_stencil::copies_break;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::Format;
use std::collections::BTreeSet;

const COPIES: RuntimeDriverDefect = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
const STENCIL_COPIES: RuntimeDriverDefect = RuntimeDriverDefect::MultisampledStencilColourCopies;

/// A driver with a colour-copy defect, every aspect's or the stencil's alone, breaks a shadow of
/// more than one sample over a combined depth/stencil attachment, and no other: a stencil-only
/// attachment's, one sample's, and those of a driver without either defect copy.
#[test]
fn a_driver_with_the_colour_copy_defect_breaks_multisampled_combined_shadows() {
    let broken = BTreeSet::from([COPIES]);
    let stencil = BTreeSet::from([STENCIL_COPIES]);
    let clean = BTreeSet::new();
    for format in [
        Format::D32_SFLOAT_S8_UINT,
        Format::D24_UNORM_S8_UINT,
        Format::D16_UNORM_S8_UINT,
    ] {
        assert_eq!(copies_break(&broken, format, 4), Some(COPIES), "{format:?}");
        assert_eq!(copies_break(&broken, format, 2), Some(COPIES), "{format:?}");
        assert_eq!(copies_break(&broken, format, 1), None, "{format:?}");
        let found = copies_break(&stencil, format, 4);
        assert_eq!(found, Some(STENCIL_COPIES), "{format:?}");
        assert_eq!(copies_break(&stencil, format, 1), None, "{format:?}");
        assert_eq!(copies_break(&clean, format, 4), None, "{format:?}");
    }
    assert_eq!(copies_break(&broken, Format::S8_UINT, 4), None);
    assert_eq!(copies_break(&broken, Format::S8_UINT, 1), None);
}
