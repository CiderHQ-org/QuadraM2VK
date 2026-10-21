//! The runtime's driver defects, keyed on the driver's identity and release
//! (`driver_defects.rs`).

use super::super::driver_defects::{among, of, Affected, AFFECTED};
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
const DESTROYS: RuntimeDriverDefect = RuntimeDriverDefect::DestroyedAddressBufferFailsSubmission;
const PUSHES: RuntimeDriverDefect = RuntimeDriverDefect::PushedBufferSizesUnset;
const STALE_PUSHES: RuntimeDriverDefect =
    RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds;
const BAKED_SAMPLERS: RuntimeDriverDefect = RuntimeDriverDefect::ImmutableSamplersLostAfterCopies;
const SET_SUPPORT: RuntimeDriverDefect = RuntimeDriverDefect::SetLayoutSupportUnderstated;
const PART_VIEWS: RuntimeDriverDefect = RuntimeDriverDefect::LazilyAllocatedSubresourceViews;

/// Every committed profile's driver has its defects at every release and GPU but those a probe
/// found clean: MoltenVK the multisampled depth/stencil colour copies, the inexact depths past
/// `[0, 1]`, the crashing 2D views of a 3D image in lazily allocated memory, the pipeline cache
/// compiled at load, the copies of compressed images that rewrite texels, the reads of command
/// buffers after their signal, the frees that fail pending work, the destroys that fail a
/// submission encoding meanwhile, the pushed sets' unset buffer sizes, the set layouts its support
/// query refuses though it runs them and the textureless views of some of a lazily allocated
/// image's levels or layers; lavapipe the colour copies, the depth/stencil resolve of the
/// first layer and level, the explicit plane's rows past its height and the baked samplers lost
/// after a copy;
/// RADV the multisampled copies that convert float NaNs, and the implicit
/// preprocessing after an execution set on every GPU but the 780M at 26.2.4 (Phoenix's and Hawk
/// Point's), measured clean; ANV
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
                    DESTROYS,
                    PUSHES,
                    SET_SUPPORT,
                    PART_VIEWS,
                ]
            }
            Some(DriverId::MESA_LLVMPIPE) => vec![COPIES, RESOLVES, PAST_HEIGHT, BAKED_SAMPLERS],
            // The 780M at 26.2.4, under Phoenix's `deviceID` and Hawk Point's.
            Some(DriverId::MESA_RADV)
                if matches!(desc.device_id, 0x15bf | 0x1900)
                    && desc.driver_version == MESA_26_2_4 =>
            {
                vec![NAN_COPIES]
            }
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

/// A driver has a defect at every release and on every GPU but a measured identity a probe found
/// clean; another driver, or none, has none of it.
#[test]
fn a_clean_identity_alone_lacks_the_defect() {
    // Mesa's `driverVersion`: `VK_MAKE_API_VERSION(0, major, minor, patch)`.
    let mesa = |minor: u32, patch: u32| (26 << 22) | (minor << 12) | patch;
    let table = [Affected {
        defect: COPIES,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[(0, (26 << 22) | (3 << 12))],
    }];
    let lavapipe = Some(DriverId::MESA_LLVMPIPE);
    assert!(among(&table, lavapipe, Some(0), mesa(2, 9)).contains(&COPIES));
    assert!(among(&table, lavapipe, Some(0), mesa(3, 0)).is_empty());
    assert!(among(&table, lavapipe, Some(0), mesa(3, 1)).contains(&COPIES));
    assert!(among(&table, lavapipe, Some(1), mesa(3, 0)).contains(&COPIES));
    assert!(among(&table, Some(DriverId::MESA_RADV), Some(0), mesa(2, 9)).is_empty());
    assert!(among(&table, None, Some(0), mesa(2, 9)).is_empty());
}

/// NVIDIA's stencil colour-copy defect, measured on the RTX 3090 (`deviceID` 0x2204) at 615.71.09,
/// is every NVIDIA GPU's at every release but the RTX 5080 Laptop GPU's (0x2c19) at 615.71.09,
/// measured clean: the RTX 3070 (0x2484), an Ada GPU (0x2684) and the 5080 Laptop GPU at another
/// release take it; a device that runs another driver has none.
#[test]
fn nvidia_s_colour_copies_are_clean_on_the_identity_measured_alone() {
    let nvidia = Some(DriverId::NVIDIA_PROPRIETARY);
    let of = |device, release| among(AFFECTED, nvidia, Some(device), release);
    for (device, release) in [
        (0x2204, NVIDIA_615_71_09),
        (0x2484, NVIDIA_615_71_09),
        (0x2684, NVIDIA_615_71_09),
        (0x2c19, NVIDIA_615_71_09 + (1 << 6)),
    ] {
        assert_eq!(
            of(device, release).into_iter().collect::<Vec<_>>(),
            vec![STENCIL_COPIES],
            "{device:#x} {release:#x}"
        );
    }
    assert!(of(0x2c19, NVIDIA_615_71_09).is_empty());
    let radv = among(
        AFFECTED,
        Some(DriverId::MESA_RADV),
        Some(0x2204),
        NVIDIA_615_71_09,
    );
    assert!(!radv.contains(&COPIES) && !radv.contains(&STENCIL_COPIES));
}

/// RADV's implicit preprocessing after an execution set, measured broken on the Radeon 780M
/// (`deviceID` 0x15bf) and the RX 7900 XTX (0x744c) at 26.0.8, is every RADV GPU's at every release
/// but the 780M's at 26.2.4, measured clean: the 780M at 26.0.8 and 26.2.5 and the RX 7900 XTX at
/// 26.2.4 (not a measured identity) take it; a device that runs another driver has none.
#[test]
fn radv_s_preprocessing_is_clean_on_the_identity_measured_alone() {
    let radv = Some(DriverId::MESA_RADV);
    let of = |device, release| among(AFFECTED, radv, Some(device), release);
    let mesa_26_0_8 = (26 << 22) | 8;
    for (device, release) in [
        (0x15bf, mesa_26_0_8),
        (0x744c, mesa_26_0_8),
        (0x744c, MESA_26_2_4),
        (0x15bf, MESA_26_2_4 + 1),
    ] {
        assert!(
            of(device, release).contains(&PREPROCESS),
            "{device:#x} {release:#x}"
        );
    }
    assert!(!of(0x15bf, MESA_26_2_4).contains(&PREPROCESS));
    let nvidia = among(
        AFFECTED,
        Some(DriverId::NVIDIA_PROPRIETARY),
        Some(0x15bf),
        mesa_26_0_8,
    );
    assert!(!nvidia.contains(&PREPROCESS));
}

/// lavapipe's baked samplers lost after a copy, measured broken at 26.0.8, are every lavapipe
/// release's: 26.2.4, measured clean, has no committed profile to be a measured identity. A device
/// that runs another driver has none.
#[test]
fn lavapipe_s_baked_samplers_are_every_release_s() {
    let lavapipe = Some(DriverId::MESA_LLVMPIPE);
    let of = |release| among(AFFECTED, lavapipe, Some(0), release);
    let mesa_26_0_8 = (26 << 22) | 8;
    for release in [mesa_26_0_8, MESA_26_2_4 - 1, MESA_26_2_4, MESA_26_2_4 + 1] {
        assert!(of(release).contains(&BAKED_SAMPLERS), "{release:#x}");
    }
    let radv = among(AFFECTED, Some(DriverId::MESA_RADV), Some(0), mesa_26_0_8);
    assert!(!radv.contains(&BAKED_SAMPLERS));
}

/// Hawk Point's Radeon 780M (`deviceID` 0x1900) is Phoenix's GPU (0x15bf): at RADV 26.2.4 it is a
/// measured identity, so it sends no unmeasured-driver message and takes Phoenix's driver facts
/// and defects, the implicit preprocessing measured clean among them. At another release, or as a
/// neighbouring `deviceID` (0x1901), it is unmeasured, works the preprocessing around, and says
/// so in the message, naming its own `deviceID`.
#[test]
fn hawk_points_780m_is_phoenixs_measured_identity() {
    use crate::driver_facts::{unmeasured_driver, UNMEASURED_DRIVER};
    let (stem, phoenix) = super::capabilities::committed_descriptions()
        .into_iter()
        .find(|(stem, _)| stem == "amd-780m-radv")
        .expect("amd-780m-radv");
    let (_, measured) = super::capabilities::created(&stem, &phoenix);
    let mut hawk_point = phoenix.clone();
    hawk_point.device_id = 0x1900;
    let (_, properties) = super::capabilities::created(&stem, &hawk_point);
    assert_eq!(unmeasured_driver(&properties), None);
    assert_eq!(properties.driver_facts, measured.driver_facts);
    assert!(!properties.driver_facts.is_empty());
    assert_eq!(properties.driver_defects, measured.driver_defects);
    assert_eq!(
        of(&hawk_point).into_iter().collect::<Vec<_>>(),
        [NAN_COPIES]
    );
    for (device, release) in [
        (0x1900, MESA_26_2_4 + 1),
        (0x1900, MESA_26_2_4 - 1),
        (0x1901, MESA_26_2_4),
    ] {
        let mut other = phoenix.clone();
        (other.device_id, other.driver_version) = (device, release);
        let (_, properties) = super::capabilities::created(&stem, &other);
        let text = unmeasured_driver(&properties).expect("an unmeasured identity");
        assert!(text.contains(&format!("deviceID {device:#x}")), "{text}");
        assert!(text.ends_with(&format!("[{UNMEASURED_DRIVER}]")), "{text}");
        assert!(properties.driver_facts.is_empty(), "{text}");
        assert!(of(&other).contains(&PREPROCESS), "{device:#x} {release:#x}");
    }
}

/// Each clean identity is a measured one: a committed profile of the driver, GPU and release
/// that the device layer takes as measured (`DeviceDescription::measured_identity`).
#[test]
fn every_clean_identity_is_measured() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut measured = Vec::new();
    for path in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = path.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        if desc.measured_identity() {
            measured.push((desc.driver_id, desc.device_id, desc.driver_version));
        }
    }
    for affected in AFFECTED {
        for &(device, release) in affected.clean {
            assert!(
                measured.contains(&(Some(affected.driver), device, release)),
                "{:?} clean on {device:#x} at {release:#x}, which is not measured",
                affected.defect
            );
        }
    }
}

/// A device whose identity is not measured says so in an informational message, never a degrade
/// notice: on every committed profile of a measured identity nothing; on the M4's profile at
/// another MoltenVK release, the identity, the answers it withholds and the defects it assumes,
/// under the message's id name; on SwiftShader, whose driver has nothing measured, nothing.
#[test]
fn an_unmeasured_identity_says_so_in_an_informational_message() {
    use crate::driver_facts::{unmeasured_driver, UNMEASURED_DRIVER};
    for (stem, desc) in super::capabilities::committed_descriptions() {
        let (_, properties) = super::capabilities::created(&stem, &desc);
        let text = unmeasured_driver(&properties);
        let measured = properties.description.measured_identity();
        let nothing = properties.driver_facts_withheld.is_empty()
            && properties.driver_facts.is_empty()
            && properties.driver_defects.is_empty();
        assert_eq!(text.is_some(), !measured && !nothing, "{stem}: {text:?}");
    }
    let (stem, mut desc) = super::capabilities::committed_descriptions()
        .into_iter()
        .find(|(stem, _)| stem == "moltenvk-m4")
        .expect("moltenvk-m4");
    desc.driver_version += 1;
    let (_, properties) = super::capabilities::created(&stem, &desc);
    let text = unmeasured_driver(&properties).expect("an unmeasured MoltenVK release");
    assert!(text.ends_with(&format!("[{UNMEASURED_DRIVER}]")), "{text}");
    assert!(text.contains("DepthImageKeepsDenormals"), "{text}");
    assert!(text.contains("DepthStencilImagesLieApart"), "{text}");
    assert!(!text.contains("metal2vulkan-degrade-"), "{text}");
}

/// `DeviceOptions::assume_unmeasured_identity` takes a device as an identity no measurement names:
/// the description it hands the device ([`identity_as_options`]) answers as another GPU of its
/// driver at its release, so on every committed profile the device's facts, withheld answers and
/// defects are such a GPU's, and it says so in the `metal2vulkan-unmeasured-driver` message where
/// such a GPU does. On the RTX 5080 Laptop GPU's measured identity the message appears only under
/// the option. Off, the description is the device's.
#[test]
fn the_unmeasured_identity_option_runs_another_gpu_s_paths() {
    use crate::device::create::identity_as_options;
    use crate::driver_facts::{unmeasured_driver, UNMEASURED_DRIVER};
    use crate::DeviceOptions;
    let assume = DeviceOptions {
        assume_unmeasured_identity: true,
        ..DeviceOptions::default()
    };
    for (stem, desc) in super::capabilities::committed_descriptions() {
        assert_eq!(
            identity_as_options(desc.clone(), &DeviceOptions::default()),
            desc,
            "{stem}"
        );
        let assumed = identity_as_options(desc.clone(), &assume);
        assert!(assumed.assumed_unmeasured, "{stem}");
        let mut other = desc.clone();
        other.device_id ^= 0x8000_0000;
        let (_, a) = super::capabilities::created(&stem, &assumed);
        let (_, o) = super::capabilities::created(&stem, &other);
        assert_eq!(a.driver_facts, o.driver_facts, "{stem}");
        assert_eq!(a.driver_facts_withheld, o.driver_facts_withheld, "{stem}");
        assert_eq!(a.driver_defects, o.driver_defects, "{stem}");
        assert_eq!(
            unmeasured_driver(&a).is_some(),
            unmeasured_driver(&o).is_some(),
            "{stem}"
        );
    }
    let (stem, laptop) = super::capabilities::committed_descriptions()
        .into_iter()
        .find(|(stem, _)| stem == "nvidia-rtx5080-laptop")
        .expect("nvidia-rtx5080-laptop");
    let (_, measured) = super::capabilities::created(&stem, &laptop);
    assert!(unmeasured_driver(&measured).is_none());
    let (_, assumed) = super::capabilities::created(&stem, &identity_as_options(laptop, &assume));
    let text = unmeasured_driver(&assumed).expect("the option's message");
    assert!(text.ends_with(&format!("[{UNMEASURED_DRIVER}]")), "{text}");
    assert!(!text.contains("metal2vulkan-degrade-"), "{text}");
}

/// A descriptor set layout of at most 1024 descriptors (the least `maxPerSetDescriptors` Vulkan
/// allows) is made without asking the driver; a larger one is made where the driver's
/// `vkGetDescriptorSetLayoutSupport` answers `VK_TRUE`, made and announced where it answers
/// `VK_FALSE` on a driver that runs what it refuses (`SetLayoutSupportUnderstated`), and refused
/// on any other: never made past the driver's answer unannounced.
#[test]
fn a_set_layout_past_the_least_per_set_limit_is_made_as_the_driver_answers() {
    use crate::pipeline::objects::{set_support, SetSupport, LEAST_PER_SET_DESCRIPTORS};
    let unasked = || -> bool { panic!("asked about a layout Vulkan promises") };
    for understated in [false, true] {
        for total in [0, 1, 1023, LEAST_PER_SET_DESCRIPTORS] {
            assert_eq!(
                set_support(total, understated, unasked),
                SetSupport::Supported
            );
        }
        let refused = match understated {
            true => SetSupport::Understated,
            false => SetSupport::Refused,
        };
        for total in [1025, 1212, 65536, 393_216] {
            assert_eq!(
                set_support(total, understated, || true),
                SetSupport::Supported
            );
            assert_eq!(set_support(total, understated, || false), refused);
        }
    }
}
