//! The `FragDepth` draw's one decision (`depth_copy::drawn`): exact, an announced near-equivalent
//! naming what it changes or may change, or refused.

use crate::depth_copy::{drawn_by, Drawn, UNMEASURED_FLOAT_DEPTH};
use metal2vulkan::device::DriverFact;
use metal2vulkan::vk::{DriverId, Format};
use std::collections::BTreeSet;

/// A 32-bit float depth is drawn on every driver: exact where its fragment depth is measured to
/// store every value unchanged (lavapipe's measured releases), naming the measured changes on
/// MoltenVK and NVIDIA's, and on any other driver (a lavapipe release not measured, as the
/// reporter's Mesa 25.0.7 and a 26.2.5 build) naming what Vulkan leaves to it, where it was
/// refused before. A 16-bit depth is drawn only where its codes are measured kept; a colour
/// format never.
#[test]
fn a_float_depth_is_drawn_on_every_driver_announced_where_not_measured_exact() {
    let none = BTreeSet::new();
    let exact: BTreeSet<_> = [DriverFact::FragmentDepthStoresUnchanged].into();
    let codes: BTreeSet<_> = [DriverFact::FragmentDepthKeepsDepth16Codes].into();
    let lavapipe = Some(DriverId::MESA_LLVMPIPE);
    for format in [Format::D32_SFLOAT, Format::D32_SFLOAT_S8_UINT] {
        assert_eq!(drawn_by(&exact, lavapipe, format), Some(Drawn::Exact));
        assert_eq!(
            drawn_by(&none, lavapipe, format),
            Some(Drawn::Changes(UNMEASURED_FLOAT_DEPTH)),
            "an unmeasured lavapipe"
        );
        for driver in [
            Some(DriverId::MESA_RADV),
            Some(DriverId::INTEL_OPEN_SOURCE_MESA),
            None,
        ] {
            assert_eq!(
                drawn_by(&none, driver, format),
                Some(Drawn::Changes(UNMEASURED_FLOAT_DEPTH)),
                "{driver:?}"
            );
        }
        for driver in [DriverId::MOLTENVK, DriverId::NVIDIA_PROPRIETARY] {
            let Some(Drawn::Changes(changes)) = drawn_by(&none, Some(driver), format) else {
                panic!("{driver:?}: not drawn with its changes");
            };
            assert_ne!(
                changes, UNMEASURED_FLOAT_DEPTH,
                "{driver:?}: its measured changes"
            );
            assert!(changes.contains("changes values"), "{driver:?}: {changes}");
        }
    }
    // The notice names each value Vulkan leaves to an unmeasured driver.
    for named in ["outside [0, 1]", "infinity", "NaN", "denormal", "-0"] {
        assert!(UNMEASURED_FLOAT_DEPTH.contains(named), "{named}");
    }
    for format in [Format::D16_UNORM, Format::D16_UNORM_S8_UINT] {
        assert_eq!(drawn_by(&codes, lavapipe, format), Some(Drawn::Exact));
        assert_eq!(drawn_by(&none, lavapipe, format), None);
        assert_eq!(drawn_by(&exact, Some(DriverId::MOLTENVK), format), None);
    }
    assert_eq!(drawn_by(&exact, lavapipe, Format::R32_SFLOAT), None);
}
