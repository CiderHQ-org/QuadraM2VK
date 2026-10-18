//! A device's depth-range facts (`render_state::DepthRanges`), read once at its making.

use super::image_plan::committed;
use crate::render_state::DepthRanges;
use metal2vulkan::device::{VkExtension, VkFeature};

/// Each committed profile's facts are its enabled sets' answers: the native range where it enabled
/// `VK_EXT_depth_range_unrestricted`, `depthClamp`, `depthClipEnable` and `depthClampControl`,
/// the unrestricted range where it enabled the extension. The profiles hold devices of the native
/// range, of the extension alone and of neither, so draws on each take the facts their device's
/// profile has, never another's.
#[test]
fn a_device_s_depth_range_facts_are_its_profile_s() {
    let mut seen = Vec::new();
    for (name, _, device) in committed() {
        let enabled = device.profile().enabled();
        let unrestricted = (enabled.extensions).contains(&VkExtension::ExtDepthRangeUnrestricted);
        let features = [
            VkFeature::DepthClamp,
            VkFeature::DepthClipEnable,
            VkFeature::DepthClampControl,
        ];
        let native = unrestricted && features.iter().all(|f| enabled.features.contains(f));
        let facts = DepthRanges::of(device.profile());
        assert_eq!(
            (facts.native, facts.unrestricted),
            (native, unrestricted),
            "{name}"
        );
        seen.push((native, unrestricted));
    }
    for kind in [(true, true), (false, true), (false, false)] {
        assert!(seen.contains(&kind), "a committed profile of {kind:?}");
    }
}
