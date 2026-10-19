//! The runtime's driver defects, keyed on the driver's identity, release and GPU
//! (`driver_defects.rs`, and the known-driver list's).

use super::super::driver_defects::{named, of, AFFECTED};
use metal2vulkan::device::DeviceDescription;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::DriverId;
use std::path::PathBuf;

const COPIES: RuntimeDriverDefect = RuntimeDriverDefect::MultisampledDepthStencilColourCopies;
const STENCIL_COPIES: RuntimeDriverDefect = RuntimeDriverDefect::MultisampledStencilColourCopies;
const DEPTHS: RuntimeDriverDefect = RuntimeDriverDefect::NormalizedDepthPastUnitRange;
const COMPRESSION: RuntimeDriverDefect = RuntimeDriverDefect::ImageSizeShortOfCompressionData;
const RESOLVES: RuntimeDriverDefect = RuntimeDriverDefect::DepthStencilResolveFirstSubresource;
const PLANE_VIEWS: RuntimeDriverDefect = RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews;
const CACHE_LOAD: RuntimeDriverDefect = RuntimeDriverDefect::PipelineCacheDataCompiledAtLoad;
const COPY_REWRITES: RuntimeDriverDefect = RuntimeDriverDefect::CompressedImageCopiesRewriteTexels;
const NAN_COPIES: RuntimeDriverDefect = RuntimeDriverDefect::MultisampledCopiesConvertFloatNans;
const LATE_READS: RuntimeDriverDefect = RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal;
const PREPROCESS: RuntimeDriverDefect = RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet;
const PAST_HEIGHT: RuntimeDriverDefect = RuntimeDriverDefect::ExplicitPlaneRowsPastHeight;
const COUNTS: RuntimeDriverDefect = RuntimeDriverDefect::OcclusionCountPerColorOutput;
const FREES: RuntimeDriverDefect = RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork;
const PUSHES: RuntimeDriverDefect = RuntimeDriverDefect::PushedBufferSizesUnset;
const STALE_PUSHES: RuntimeDriverDefect =
    RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds;
const BAKED_SAMPLERS: RuntimeDriverDefect = RuntimeDriverDefect::ImmutableSamplersLostAfterCopies;

/// Every committed profile's driver has its defects at every release and GPU but where a range of
/// the known-driver list ends: MoltenVK the multisampled depth/stencil colour copies, the inexact depths past
/// `[0, 1]`, the crashing 2D views of a 3D image in lazily allocated memory, the pipeline cache
/// compiled at load, the copies of compressed images that rewrite texels, the reads of command
/// buffers after their signal, the frees that fail pending work and the pushed sets' unset buffer
/// sizes; lavapipe the colour copies, the depth/stencil resolve of the
/// first layer and level, the explicit plane's rows past its height and the baked samplers lost
/// after a copy;
/// RADV the multisampled copies that convert float NaNs, and the implicit
/// preprocessing after an execution set on every GPU before 26.2.4, measured clean there on the
/// 780M (Phoenix's and Hawk Point's) and the RX 7900 XTX (the RDNA2 and RDNA3 profiles' 26.2.3
/// take it); ANV
/// the compressed image's short size, the occlusion count per color output and the pushed
/// descriptors a pipeline bind leaves stale on every GPU (Arrow Lake's, not measured, too);
/// NVIDIA's the
/// stencil colour copies on every GPU but the RTX 5080 Laptop GPU at 615.71.09, measured clean (the
/// GTX 1060 and the RTX 5080 desktop GPU, not measured, take them); SwiftShader none.
#[test]
fn the_committed_profiles_drivers_have_their_defects() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut drivers = 0;
    for path in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = path.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let expected = match desc.driver_id {
            Some(DriverId::MOLTENVK) => {
                vec![
                    COPIES,
                    DEPTHS,
                    PLANE_VIEWS,
                    CACHE_LOAD,
                    COPY_REWRITES,
                    LATE_READS,
                    FREES,
                    PUSHES,
                ]
            }
            Some(DriverId::MESA_LLVMPIPE) => vec![COPIES, RESOLVES, PAST_HEIGHT, BAKED_SAMPLERS],
            // From 26.2.4 on, the 780M under Phoenix's `deviceID` and Hawk Point's.
            Some(DriverId::MESA_RADV) if desc.driver_version >= MESA_26_2_4 => vec![NAN_COPIES],
            Some(DriverId::MESA_RADV) => vec![NAN_COPIES, PREPROCESS],
            Some(DriverId::INTEL_OPEN_SOURCE_MESA) => vec![COMPRESSION, COUNTS, STALE_PUSHES],
            Some(DriverId::NVIDIA_PROPRIETARY)
                if (desc.device_id, desc.driver_version) == (0x2c19, NVIDIA_615_71_09) =>
            {
                Vec::new()
            }
            Some(DriverId::NVIDIA_PROPRIETARY) => vec![STENCIL_COPIES],
            _ => Vec::new(),
        };
        let defects: Vec<_> = of(&desc).into_iter().collect();
        assert_eq!(defects, expected, "{}", path.display());
        drivers += usize::from(!expected.is_empty());
    }
    assert!(
        drivers >= 13,
        "lavapipe's three, MoltenVK's two, ANV's four, RADV's three and NVIDIA's two"
    );
}

/// Mesa's `driverVersion` of 26.2.4: `VK_MAKE_API_VERSION(0, 26, 2, 4)`.
const MESA_26_2_4: u32 = (26 << 22) | (2 << 12) | 4;

/// NVIDIA's `driverVersion` of 615.71.09: the release's major, minor and patch in 10, 8 and 6 bits
/// from bit 22, 14 and 6.
const NVIDIA_615_71_09: u32 = (615 << 22) | (71 << 14) | (9 << 6);

/// Lavapipe's baked samplers lost after a copy hold from 26.0.8, the first release measured with
/// them, up to 26.2.4, the first measured without them (and checked Metal's on the device there):
/// an earlier release, 26.2.4 and a later one have none of it, and the row names its releases. A
/// device that runs another driver has none.
#[test]
fn lavapipe_s_baked_samplers_hold_at_their_measured_releases_alone() {
    let mut desc = DeviceDescription::from_khronos_json(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../metal2vulkan-validation/profiles/gate-lavapipe-x86_64.json"),
        )
        .expect("gate-lavapipe-x86_64"),
        None,
    )
    .expect("profile parses");
    let mesa_26_0_8 = (26 << 22) | 8;
    for (release, holds) in [
        (mesa_26_0_8 - 1, false),
        (mesa_26_0_8, true),
        (MESA_26_2_4 - 1, true),
        (MESA_26_2_4, false),
        (MESA_26_2_4 + 1, false),
    ] {
        desc.driver_version = release;
        assert_eq!(of(&desc).contains(&BAKED_SAMPLERS), holds, "{release:#x}");
    }
    desc.driver_version = mesa_26_0_8;
    assert_eq!(
        named(&desc).get(&BAKED_SAMPLERS).map(String::as_str),
        Some("VK_DRIVER_ID_MESA_LLVMPIPE from 26.0.8 before 26.2.4, every GPU")
    );
    desc.driver_id = Some(DriverId::MESA_RADV);
    assert!(!of(&desc).contains(&BAKED_SAMPLERS));
}

/// A check-class defect of every release holds on every device of its driver, at every release
/// and on every GPU; another driver, or none, has none of it.
#[test]
fn a_table_defect_holds_on_every_device_of_its_driver() {
    let mut desc = DeviceDescription::from_khronos_json(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../metal2vulkan-validation/profiles/gate-lavapipe.json"),
        )
        .expect("gate-lavapipe"),
        None,
    )
    .expect("profile parses");
    for (device, release) in [(0, MESA_26_2_4 - 1), (0, MESA_26_2_4), (1, MESA_26_2_4 + 1)] {
        (desc.device_id, desc.driver_version) = (device, release);
        assert!(of(&desc).contains(&RESOLVES), "{device:#x} {release:#x}");
    }
    desc.driver_id = Some(DriverId::MESA_RADV);
    assert!(!of(&desc).contains(&RESOLVES));
    desc.driver_id = None;
    assert!(of(&desc).is_empty());
}

/// The defects a check may settle are the table's, and those no check settles the known-driver
/// list's: a defect is in both only where the list holds it for another driver (the multisampled
/// depth/stencil colour copies, MoltenVK's listed and lavapipe's checked).
#[test]
fn the_table_and_the_list_hold_each_defect_once_per_driver() {
    let mut desc = DeviceDescription::from_khronos_json(
        &std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../metal2vulkan-validation/profiles/moltenvk-m4.json"),
        )
        .expect("moltenvk-m4"),
        None,
    )
    .expect("profile parses");
    for driver in [
        DriverId::MOLTENVK,
        DriverId::MESA_LLVMPIPE,
        DriverId::MESA_RADV,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        DriverId::NVIDIA_PROPRIETARY,
    ] {
        desc.driver_id = Some(driver);
        let listed = desc.listed_driver_defects();
        for affected in AFFECTED.iter().filter(|a| a.driver == driver) {
            assert!(
                !listed.contains_key(&affected.defect),
                "{driver:?} {:?}",
                affected.defect
            );
        }
    }
}

/// NVIDIA's stencil colour-copy defect, measured on the RTX 3090 (`deviceID` 0x2204) at 615.71.09,
/// is every NVIDIA GPU's at every release but the RTX 5080 Laptop GPU's (0x2c19) at 615.71.09,
/// measured clean: the RTX 3070 (0x2484), an Ada GPU (0x2684) and the 5080 Laptop GPU at another
/// release take it; a device that runs another driver has none.
#[test]
fn nvidia_s_colour_copies_are_clean_on_the_identity_measured_alone() {
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/nvidia-rtx5080-laptop.json"),
    )
    .expect("nvidia-rtx5080-laptop");
    let laptop = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    let at = |device, release| {
        let mut desc = laptop.clone();
        (desc.device_id, desc.driver_version) = (device, release);
        of(&desc)
    };
    for (device, release) in [
        (0x2204, NVIDIA_615_71_09),
        (0x2484, NVIDIA_615_71_09),
        (0x2684, NVIDIA_615_71_09),
        (0x2c19, NVIDIA_615_71_09 + (1 << 6)),
    ] {
        assert_eq!(
            at(device, release).into_iter().collect::<Vec<_>>(),
            vec![STENCIL_COPIES],
            "{device:#x} {release:#x}"
        );
    }
    assert!(at(0x2c19, NVIDIA_615_71_09).is_empty());
    let mut radv = laptop.clone();
    radv.driver_id = Some(DriverId::MESA_RADV);
    let radv = of(&radv);
    assert!(!radv.contains(&COPIES) && !radv.contains(&STENCIL_COPIES));
}

/// RADV's implicit preprocessing after an execution set, measured broken on the Radeon 780M
/// (`deviceID` 0x15bf) and the RX 7900 XTX (0x744c) at 26.0.8 and clean on both at 26.2.4, is
/// every RADV GPU's before 26.2.4: the 780M and the RX 7900 XTX at 26.0.8 and 26.2.3 take it, and
/// at 26.2.4 and 26.2.5 neither does; a device that runs another driver has none.
#[test]
fn radv_s_preprocessing_holds_before_the_release_measured_clean() {
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/amd-780m-radv.json"),
    )
    .expect("amd-780m-radv");
    let phoenix = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    let at = |device, release| {
        let mut desc = phoenix.clone();
        (desc.device_id, desc.driver_version) = (device, release);
        of(&desc)
    };
    let mesa_26_0_8 = (26 << 22) | 8;
    for device in [0x15bf, 0x744c] {
        for release in [mesa_26_0_8, MESA_26_2_4 - 1] {
            assert!(
                at(device, release).contains(&PREPROCESS),
                "{device:#x} {release:#x}"
            );
        }
        for release in [MESA_26_2_4, MESA_26_2_4 + 1] {
            assert!(
                !at(device, release).contains(&PREPROCESS),
                "{device:#x} {release:#x}"
            );
        }
    }
    let mut nvidia = phoenix.clone();
    (nvidia.driver_id, nvidia.driver_version) = (Some(DriverId::NVIDIA_PROPRIETARY), mesa_26_0_8);
    assert!(!of(&nvidia).contains(&PREPROCESS));
}

/// Hawk Point's Radeon 780M (`deviceID` 0x1900) is Phoenix's GPU (0x15bf): at RADV 26.2.4 it is a
/// measured identity, so its feature status withholds nothing and it takes Phoenix's driver facts
/// and defects, without the implicit preprocessing measured clean there. At another release, or
/// as a neighbouring `deviceID` (0x1901), it is unmeasured, takes none of the facts a check may
/// settle, and says so in its feature status, naming its own `deviceID`; before 26.2.4 it works
/// the preprocessing around.
#[test]
fn hawk_points_780m_is_phoenixs_measured_identity() {
    use crate::device_checks::FEATURE_STATUS;
    use crate::feature_status_message;
    let (stem, phoenix) = super::capabilities::committed_descriptions()
        .into_iter()
        .find(|(stem, _)| stem == "amd-780m-radv")
        .expect("amd-780m-radv");
    let (_, measured) = super::capabilities::created(&stem, &phoenix);
    let mut hawk_point = phoenix.clone();
    hawk_point.device_id = 0x1900;
    let (_, properties) = super::capabilities::created(&stem, &hawk_point);
    let text = feature_status_message(&properties).text;
    assert!(!text.contains("is not measured"), "{text}");
    assert_eq!(properties.driver_facts, measured.driver_facts);
    assert!(!properties.driver_facts.is_empty());
    assert_eq!(properties.driver_defects, measured.driver_defects);
    assert_eq!(
        of(&hawk_point).into_iter().collect::<Vec<_>>(),
        [NAN_COPIES]
    );
    for (device, release, preprocess) in [
        (0x1900, MESA_26_2_4 + 1, false),
        (0x1900, MESA_26_2_4 - 1, true),
        (0x1901, MESA_26_2_4, false),
    ] {
        let mut other = phoenix.clone();
        (other.device_id, other.driver_version) = (device, release);
        let (_, properties) = super::capabilities::created(&stem, &other);
        let text = feature_status_message(&properties).text;
        assert!(text.contains("is not measured"), "{text}");
        assert!(text.contains(&format!(":{device:04x}")), "{text}");
        assert!(text.ends_with(&format!("[{FEATURE_STATUS}]")), "{text}");
        assert!(
            properties
                .driver_facts
                .is_disjoint(&properties.driver_facts_withheld),
            "{text}"
        );
        assert!(!properties.driver_facts_withheld.is_empty(), "{text}");
        assert_eq!(
            of(&other).contains(&PREPROCESS),
            preprocess,
            "{device:#x} {release:#x}"
        );
    }
}

/// A device whose identity is not measured says so in its feature status, an informational
/// message, never a degrade notice: on every committed profile exactly where it withholds an
/// answer; on the M4's profile at another MoltenVK release, the answers it withholds, the listed
/// answers it takes with the last release measured, and the defects it works around, under the
/// message's id name.
#[test]
fn an_unmeasured_identity_says_so_in_an_informational_message() {
    use crate::device_checks::FEATURE_STATUS;
    use crate::feature_status_message;
    for (stem, desc) in super::capabilities::committed_descriptions() {
        let (_, properties) = super::capabilities::created(&stem, &desc);
        let message = feature_status_message(&properties);
        assert_eq!(message.id_name, FEATURE_STATUS, "{stem}");
        assert!(message.degradation.is_none(), "{stem}");
        let withheld = !properties.driver_facts_withheld.is_empty();
        let text = message.text;
        assert_eq!(text.contains("is not measured"), withheld, "{stem}: {text}");
        assert!(
            !withheld || !properties.description.measured_identity(),
            "{stem}"
        );
    }
    let (stem, mut desc) = super::capabilities::committed_descriptions()
        .into_iter()
        .find(|(stem, _)| stem == "moltenvk-m4")
        .expect("moltenvk-m4");
    desc.driver_version += 1;
    let (_, properties) = super::capabilities::created(&stem, &desc);
    let text = feature_status_message(&properties).text;
    assert!(text.ends_with(&format!("[{FEATURE_STATUS}]")), "{text}");
    assert!(text.contains("SampledFetchRecoversCodes"), "{text}");
    assert!(text.contains("DepthImageKeepsDenormals"), "{text}");
    assert!(text.contains("measured through 1.4.1 on"), "{text}");
    assert!(text.contains("DepthStencilImagesLieApart"), "{text}");
    assert!(!text.contains("metal2vulkan-degrade-"), "{text}");
}
