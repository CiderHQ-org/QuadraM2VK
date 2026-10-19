//! The release-keyed table of measured driver facts (`device::driver_facts`), and the runtime's
//! facts each committed profile's driver has, from it and the known-driver list
//! (`device::listed`).

use crate::device::driver_facts::{
    among, mesa, moltenvk, nvidia, Measured, Standing, MEASURED_DEVICES, NVIDIA_RTX_3090,
};
use crate::device::listed::{listed, Gpus, Releases, LISTED_FACTS};
use crate::device::{DeviceDescription, DriverFact};
use crate::vk::DriverId;
use std::collections::BTreeMap;
use std::path::PathBuf;

const APPLE: u32 = 0x106B;

/// A driver has a fact from the first release measured: through the last as measured, beyond
/// it as taken past the measurement, naming the last; before the first, on another driver, or
/// on a device that names none, it has none. A row of one vendor's GPUs holds on that vendor's
/// alone, a row of one GPU on that GPU alone, and of two rows of one fact the better standing
/// holds.
#[test]
fn a_fact_holds_from_the_first_release_measured_and_past_the_last_as_beyond() {
    let radv = Some(DriverId::MESA_RADV);
    let row = Measured {
        fact: 1u8,
        driver: DriverId::MESA_RADV,
        vendor: None,
        device: None,
        first: mesa(26, 2, 4),
        last: mesa(26, 2, 6),
    };
    let at = |table: &[Measured<u8>], driver, vendor, version| {
        among(table, driver, (vendor, 0x1A06_0209), version)
            .get(&1)
            .copied()
    };
    assert_eq!(at(&[row], radv, 0x1002, mesa(26, 2, 3)), None);
    assert_eq!(
        at(&[row], radv, 0x1002, mesa(26, 2, 4)),
        Some(Standing::Measured)
    );
    assert_eq!(
        at(&[row], radv, 0x1002, mesa(26, 2, 6)),
        Some(Standing::Measured)
    );
    let beyond = Standing::Beyond {
        last: mesa(26, 2, 6),
    };
    assert_eq!(at(&[row], radv, 0x1002, mesa(27, 0, 0)), Some(beyond));
    assert_eq!(
        at(
            &[row],
            Some(DriverId::MESA_LLVMPIPE),
            0x1002,
            mesa(26, 2, 4)
        ),
        None
    );
    assert_eq!(at(&[row], None, 0x1002, mesa(26, 2, 4)), None);
    // A row of Apple's GPUs.
    let apple = Measured {
        fact: 1u8,
        driver: DriverId::MOLTENVK,
        vendor: Some(APPLE),
        device: None,
        first: moltenvk(1, 4, 1),
        last: moltenvk(1, 4, 1),
    };
    let mvk = Some(DriverId::MOLTENVK);
    assert_eq!(at(&[apple], mvk, APPLE, 10401), Some(Standing::Measured));
    assert_eq!(at(&[apple], mvk, 0x1002, 10401), None);
    assert_eq!(
        at(&[apple], mvk, APPLE, moltenvk(1, 4, 2)),
        Some(Standing::Beyond { last: 10401 })
    );
    // A row of one GPU holds on that GPU alone.
    let one_gpu = Measured {
        device: Some(0x1A06_0209),
        ..apple
    };
    assert_eq!(at(&[one_gpu], mvk, APPLE, 10401), Some(Standing::Measured));
    assert_eq!(
        among(&[one_gpu], mvk, (APPLE, 0x1B00_0209), 10401).get(&1),
        None
    );
    // Two rows: the later measurement covers what the earlier left beyond.
    let later = Measured {
        first: moltenvk(1, 4, 2),
        last: moltenvk(1, 4, 2),
        ..apple
    };
    assert_eq!(
        at(&[apple, later], mvk, APPLE, moltenvk(1, 4, 2)),
        Some(Standing::Measured)
    );
    // The encodings, as the committed profiles state their releases.
    assert_eq!(nvidia(615, 71, 9), 2_580_660_800);
    assert_eq!(mesa(26, 2, 4), 109_060_100);
    assert_eq!(moltenvk(1, 4, 1), 10401);
}

/// The deviations a profile takes, read from the routes it derives, each as
/// `DeviceProfile::takes_driver_deviation` answers it too.
fn taken(p: &crate::device::DeviceProfile) -> Vec<crate::error::DriverDeviation> {
    use crate::device::profile::{CubeGradientRoute, FloatFunctions, NanEncoding};
    use crate::device::requirements::VkProperty;
    use crate::error::DriverDeviation as D;
    let f = p.fields();
    let property = |v| f.enabled.properties.contains(&v);
    let routes = (f.cube_gradients.graphics, f.cube_gradients.compute);
    [
        (
            D::SimdgroupLockstep,
            property(VkProperty::SimdgroupLockstep),
        ),
        (
            D::MetalFloatFunctions,
            f.float_functions == FloatFunctions::Metal,
        ),
        (
            D::MetalCanonicalNans,
            f.nan_encoding == NanEncoding::MetalCanonical,
        ),
        (D::MetalSampler, property(VkProperty::MetalSampler)),
        (D::CubeGradientRoute, routes.1 == CubeGradientRoute::Metal),
        (
            D::GraphicsCubeGradientRoute,
            routes.0 == CubeGradientRoute::AppleSwizzled,
        ),
        (
            D::GraphicsCubeGradientTangents,
            routes.0 == CubeGradientRoute::AppleProjected,
        ),
        (D::MetalColorWrites, property(VkProperty::MetalColorWrites)),
        (
            D::MetalFloatColorNans,
            property(VkProperty::MetalFloatColorNans),
        ),
        (
            D::MetalHalfColorNans,
            property(VkProperty::MetalHalfColorNans),
        ),
        (
            D::HalfColorOutputBits,
            property(VkProperty::HalfColorOutputBits),
        ),
        (D::MetalHalfFma, property(VkProperty::MetalHalfFma)),
        (
            D::MetalFloatDenormals,
            property(VkProperty::MetalFloatDenormals),
        ),
        (
            D::MetalHalfQuantize,
            property(VkProperty::MetalHalfQuantize),
        ),
        (
            D::MetalHalfConversion,
            property(VkProperty::MetalHalfConversion),
        ),
        (
            D::MetalCenterVaryings,
            property(VkProperty::MetalCenterVaryings),
        ),
        (
            D::MetalSmallFloatColorWrites,
            property(VkProperty::MetalSmallFloatColorWrites),
        ),
        (
            D::MetalNorm16ColorWrites,
            property(VkProperty::MetalNorm16ColorWrites),
        ),
        (D::MetalMinMax, property(VkProperty::MetalMinMax)),
        (
            D::MetalHalfArithmetic,
            property(VkProperty::MetalHalfArithmetic),
        ),
        (
            D::MetalHalfStorageWrites,
            property(VkProperty::MetalHalfStorageWrites),
        ),
        (
            D::MetalHalfStorageNans,
            property(VkProperty::MetalHalfStorageNans),
        ),
        (
            D::MetalFloatStorageNans,
            property(VkProperty::MetalFloatStorageNans),
        ),
        (
            D::MetalFragmentSimdgroups,
            property(VkProperty::MetalFragmentSimdgroups),
        ),
        (
            D::MetalVertexSimdgroups,
            property(VkProperty::MetalVertexSimdgroups),
        ),
        (
            D::CenterFragCoordDepth,
            property(VkProperty::CenterFragCoordDepth),
        ),
        (
            D::SampleFragCoordDepth,
            property(VkProperty::SampleFragCoordDepth),
        ),
        (
            D::MetalHalfTexelWrites,
            property(VkProperty::MetalHalfTexelWrites),
        ),
        (
            D::MetalNormStorageWrites,
            property(VkProperty::MetalNormStorageWrites),
        ),
        (
            D::Unorm8LinearFilter,
            property(VkProperty::Unorm8LinearFilter),
        ),
        (
            D::MetalDepth16Stores,
            property(VkProperty::MetalDepth16Stores),
        ),
        (
            D::MetalDepth16BiasedStores,
            property(VkProperty::MetalDepth16BiasedStores),
        ),
        (
            D::MetalInterpolation,
            property(VkProperty::MetalInterpolation),
        ),
        (
            D::MetalFirstVertexIndices,
            property(VkProperty::MetalFirstVertexIndices),
        ),
    ]
    .into_iter()
    .inspect(|&(d, taken)| {
        // The public answer agrees with the routes (the gate lanes ask it).
        assert_eq!(p.takes_driver_deviation(d), taken, "{d:?}");
    })
    .filter_map(|(d, taken)| taken.then_some(d))
    .collect()
}

/// The profile takes each deviation its driver's release is measured with, never one a description
/// states: every committed MoltenVK capture (MoltenVK 1.4.1 on Apple's M3 Max and M4 under macOS
/// 27, one identity) takes the nineteen of every Apple GPU measured (the 16-bit color outputs'
/// bits, the `half` fused multiply-add, the `float` denormals and the `half` quantize, the `half`
/// conversion, 4.15, the center varyings, the minimum and maximum family, the `half` arithmetic,
/// the three half-float storage writes, the fragment simdgroups, and the vertex simdgroups, among
/// them) and macOS 27's graphics cube route (the fixup's tangents), the M3 Max's under macOS 26.7
/// (no committed capture) its own graphics cube route instead, the
/// RTX 5080 Laptop GPU's (NVIDIA's 615.71.09) the two color-NaN stores, the `float` denormals and
/// the half-float storage writes and their `0x7fc00000` and the half texel writes' NaN bits, as the
/// RTX 3090 would, with the small float color writes, and another RTX 50 does not, lavapipe's (Mesa
/// 26.0.8 and 26.2.3) the `float` color NaNs, Mesa 26.2.4's on the Radeon 780M (RADV) and on the
/// Iris Xe of RPL-P and the UHD 770 of RPL-S (ANV) the `float` denormals, the `half` quantize and
/// the half-float storage writes, the 780M's also their `0x7fc00000` and the half texel writes' NaN
/// bits, the 780M's and the Iris Xe's also the small float color writes and the 780M's the 16-bit
/// norm ones, macOS 27's Apple identity's and the 780M's the normalized storage writes, the RTX
/// GPUs', the 780M's, the Iris Xe's and the UHD 770's the 8-bit linear filter's bound (lavapipe's
/// at Mesa 26.2.4, past the committed captures' 26.0.8 and 26.2.3) under a policy that accepts it
/// (`Unorm8LinearFilterPolicy::DeviceWithinOneCode`; the default takes it nowhere), every Apple
/// GPU's, the 780M's, the Iris Xe's and the UHD 770's 16-bit unorm depth stores, every Apple GPU's
/// interpolation (Metal's own, MoltenVK handing it the varyings), and no other
/// committed profile takes any. A later MoltenVK release takes them past the measurement, listed in
/// `deviations_beyond` with the last release measured. An earlier release, and MoltenVK on another
/// vendor's GPU, take none, each with its conservative route (the cube routes unmeasured); another
/// driver on Apple's vendor takes none, with Vulkan's cube rule. `InterpolateAtOffsetCentered`
/// follows the driver alone.
#[test]
fn the_profile_takes_the_deviations_its_driver_release_is_measured_with() {
    use crate::device::profile::{CubeGradientRoute, CubeGradients};
    use crate::device::requirements::VkProperty;
    use crate::device::{Device, DevicePolicy, Unorm8LinearFilterPolicy};
    use crate::error::DriverDeviation as D;
    use crate::spirv::testing::is_moltenvk_capture;
    // The nineteen of every Apple GPU measured, and the M3 Max's graphics cube route.
    let apple = vec![
        D::SimdgroupLockstep,
        D::MetalFloatFunctions,
        D::MetalCanonicalNans,
        D::MetalSampler,
        D::CubeGradientRoute,
        D::MetalColorWrites,
        D::HalfColorOutputBits,
        D::MetalHalfFma,
        D::MetalFloatDenormals,
        D::MetalHalfQuantize,
        D::MetalHalfConversion,
        D::MetalCenterVaryings,
        D::MetalMinMax,
        D::MetalHalfArithmetic,
        D::MetalHalfStorageWrites,
        D::MetalHalfStorageNans,
        D::MetalFloatStorageNans,
        D::MetalFragmentSimdgroups,
        D::MetalVertexSimdgroups,
    ];
    // NVIDIA's 615.71.09 on the RTX 3090 and the RTX 5080 Laptop GPU: the color-NaN stores, the
    // `float` denormals, the half-float storage writes and their `0x7fc00000`, and the half texel
    // writes' NaN bits.
    let nvidia_measured = vec![
        D::MetalFloatColorNans,
        D::MetalHalfColorNans,
        D::MetalFloatDenormals,
        D::MetalHalfStorageWrites,
        D::MetalFloatStorageNans,
        D::MetalHalfTexelWrites,
        D::Unorm8LinearFilter,
    ];
    // In `taken`'s order: the graphics route (macOS 26.7's on the M3 Max's identity there, macOS
    // 27's on the identity both chips share) before the color writes, and each chip's per-sample
    // depth last.
    let mut m3_max = apple.clone();
    m3_max.insert(5, D::GraphicsCubeGradientRoute);
    m3_max.push(D::CenterFragCoordDepth);
    let mut m4_measured = apple.clone();
    m4_measured.insert(5, D::GraphicsCubeGradientTangents);
    m4_measured.push(D::CenterFragCoordDepth);
    // The M4's normalized storage writes, measured on it alone.
    m4_measured.push(D::MetalNormStorageWrites);
    // Every Apple GPU's 16-bit unorm depth, unbiased and biased, its interpolation and its
    // first-vertex indices, last in `taken`'s order.
    let mut apple_all = apple.clone();
    for list in [&mut m3_max, &mut m4_measured, &mut apple_all] {
        list.push(D::MetalDepth16Stores);
        list.push(D::MetalDepth16BiasedStores);
        list.push(D::MetalInterpolation);
        list.push(D::MetalFirstVertexIndices);
    }
    let all_of = |d: &DeviceDescription| match d.device_id {
        crate::device::driver_facts::APPLE_M3_MAX => m3_max.clone(),
        crate::device::driver_facts::APPLE_M4 => m4_measured.clone(),
        _ => apple_all.clone(),
    };
    // Under a policy that accepts the 8-bit linear filter's bound, so its rows are taken where
    // measured; the default policy takes it nowhere (checked with the committed profiles).
    let accepting = DevicePolicy {
        unorm8_linear_filter: Unorm8LinearFilterPolicy::DeviceWithinOneCode,
        ..DevicePolicy::default()
    };
    let profile_of =
        |d: DeviceDescription| Device::new(d, accepting).expect("device").profile().clone();
    let centered = |p: &crate::device::DeviceProfile| {
        p.fields()
            .enabled
            .properties
            .contains(&VkProperty::InterpolateAtOffsetCentered)
    };
    let committed = super::committed_profiles();
    assert!(committed.len() > 10);
    for (stem, _) in &committed {
        let d = super::profile(stem);
        let want_all = all_of(&d);
        let p = profile_of(d);
        let moltenvk = is_moltenvk_capture(stem);
        let want = match (moltenvk, stem.as_str()) {
            (true, _) => want_all,
            (false, "nvidia-rtx5080-laptop") => nvidia_measured.clone(),
            (false, "lavapipe" | "gate-lavapipe" | "gate-lavapipe-x86_64") => vec![
                D::MetalFloatColorNans,
                D::MetalHalfStorageWrites,
                D::MetalFloatStorageNans,
            ],
            // The per-sample `FragCoord.z` last, in `taken`'s order. The 780M under Phoenix's
            // `deviceID` and Hawk Point's alike.
            (false, "amd-780m-radv" | "amd-780m-hawkpoint-radv") => vec![
                D::MetalFloatDenormals,
                D::MetalHalfQuantize,
                D::MetalSmallFloatColorWrites,
                D::MetalNorm16ColorWrites,
                D::MetalHalfStorageWrites,
                D::MetalFloatStorageNans,
                D::SampleFragCoordDepth,
                D::MetalHalfTexelWrites,
                D::MetalNormStorageWrites,
                D::Unorm8LinearFilter,
                D::MetalDepth16Stores,
                D::MetalDepth16BiasedStores,
            ],
            (false, "intel-irisxe-rplp-anv") => vec![
                D::MetalFloatDenormals,
                D::MetalHalfQuantize,
                D::MetalSmallFloatColorWrites,
                D::MetalHalfStorageWrites,
                D::Unorm8LinearFilter,
                D::MetalDepth16Stores,
            ],
            (false, "intel-uhd770-rpls-anv") => vec![
                D::MetalFloatDenormals,
                D::MetalHalfQuantize,
                D::MetalSmallFloatColorWrites,
                D::MetalHalfStorageWrites,
                D::Unorm8LinearFilter,
                D::MetalDepth16Stores,
            ],
            (false, _) => Vec::new(),
        };
        assert_eq!(taken(&p), want, "{stem}");
        assert!(p.fields().deviations_beyond.is_empty(), "{stem}");
        assert_eq!(centered(&p), !moltenvk, "{stem}");
        // The default policy keeps Metal's 8-bit linear filter exact on every device.
        let exact = Device::new(super::profile(stem), DevicePolicy::default())
            .expect("device")
            .profile()
            .clone();
        let want: Vec<D> = want
            .into_iter()
            .filter(|&d| d != D::Unorm8LinearFilter)
            .collect();
        assert_eq!(taken(&exact), want, "{stem} under the default policy");
    }
    let m4 = super::profile("moltenvk-m4");
    assert_eq!(
        (m4.driver_id, m4.vendor_id, m4.driver_version),
        (Some(DriverId::MOLTENVK), APPLE, 10401)
    );
    // macOS 27's graphics cube route is the fixup's projected form, its compute route Metal's.
    assert_eq!(
        profile_of(m4.clone()).fields().cube_gradients,
        CubeGradients {
            graphics: CubeGradientRoute::AppleProjected,
            compute: CubeGradientRoute::Metal,
        }
    );
    // MoltenVK's deviceID is the macOS release and the GPU family: the M3 Max's capture under
    // macOS 27 is the M4's identity, and the graphics row names the M3 Max under macOS 26.7, whose
    // capture that one replaced.
    let m3 = super::profile("moltenvk-m3");
    assert_eq!(m3.device_id, crate::device::driver_facts::APPLE_M4);
    let mut m3_macos26 = m3;
    m3_macos26.device_id = crate::device::driver_facts::APPLE_M3_MAX;
    let p = profile_of(m3_macos26);
    assert_eq!(taken(&p), m3_max);
    assert_eq!(
        p.fields().cube_gradients,
        CubeGradients {
            graphics: CubeGradientRoute::AppleSwizzled,
            compute: CubeGradientRoute::Metal,
        }
    );
    let all = m4_measured.clone();
    let unmeasured = CubeGradients {
        graphics: CubeGradientRoute::Unmeasured,
        compute: CubeGradientRoute::Unmeasured,
    };
    // A later release: taken, and listed with the last release measured.
    let mut later = m4.clone();
    later.driver_version = moltenvk(1, 4, 2);
    let p = profile_of(later);
    assert_eq!(taken(&p), all);
    let beyond: BTreeMap<D, u32> = all.iter().map(|&d| (d, 10401)).collect();
    assert_eq!(p.fields().deviations_beyond, beyond);
    assert!(!centered(&p));
    // An earlier release, and another vendor's GPU: none, the cube routes unmeasured.
    let mut earlier = m4.clone();
    earlier.driver_version = moltenvk(1, 4, 0);
    let mut intel = m4.clone();
    intel.vendor_id = 0x8086;
    for (what, d) in [("1.4.0", earlier), ("Intel's vendor", intel)] {
        let p = profile_of(d);
        assert_eq!(taken(&p), Vec::new(), "{what}");
        assert_eq!(p.fields().cube_gradients, unmeasured, "{what}");
        assert!(p.fields().deviations_beyond.is_empty(), "{what}");
        assert!(!centered(&p), "{what}");
    }
    // Another driver on Apple's vendor: none, with Vulkan's rule and a centered offset.
    let mut lavapipe = m4;
    lavapipe.driver_id = Some(DriverId::MESA_LLVMPIPE);
    let p = profile_of(lavapipe);
    assert_eq!(taken(&p), Vec::new());
    let vulkan = CubeGradients {
        graphics: CubeGradientRoute::Vulkan,
        compute: CubeGradientRoute::Vulkan,
    };
    assert_eq!(p.fields().cube_gradients, vulkan);
    assert!(centered(&p));
    // NVIDIA's color-NaN stores and `float` denormals: per GPU (the 3090 and the 5080 Laptop
    // GPU, not another RTX 50), and past 615.71.09 taken with the last release measured; the
    // small float color writes the 3090's alone.
    use crate::device::driver_facts::{NVIDIA_RTX_3090, NVIDIA_RTX_5080_LAPTOP};
    let laptop = super::profile("nvidia-rtx5080-laptop");
    assert_eq!(
        (laptop.device_id, laptop.driver_version),
        (NVIDIA_RTX_5080_LAPTOP, nvidia(615, 71, 9))
    );
    let mut rtx_3090 = laptop.clone();
    rtx_3090.device_id = NVIDIA_RTX_3090;
    let mut desktop = laptop.clone();
    desktop.device_id = 0x2c02;
    let mut later = laptop.clone();
    later.driver_version = nvidia(615, 71, 10);
    let mut rtx_3090_measured = nvidia_measured.clone();
    rtx_3090_measured.insert(3, D::MetalSmallFloatColorWrites);
    assert_eq!(taken(&profile_of(rtx_3090)), rtx_3090_measured);
    assert_eq!(taken(&profile_of(desktop)), Vec::new());
    let p = profile_of(later);
    assert_eq!(taken(&p), nvidia_measured);
    let beyond: BTreeMap<D, u32> = nvidia_measured
        .iter()
        .map(|&d| (d, nvidia(615, 71, 9)))
        .collect();
    assert_eq!(p.fields().deviations_beyond, beyond);
}

const ZERO: DriverFact = DriverFact::RobustIndexFetchReadsZero;
const DENORMALS: DriverFact = DriverFact::DepthImageKeepsDenormals;
const OUTSIDE: DriverFact = DriverFact::DepthImageKeepsValuesOutsideUnitRange;
const FRAGDEPTH: DriverFact = DriverFact::FragmentDepthStoresUnchanged;
const APART: DriverFact = DriverFact::DepthStencilImagesLieApart;
const AVERAGE: DriverFact = DriverFact::ColorResolveAverageIsMetals;
const HIDE: DriverFact = DriverFact::CompressedImagesHideAliasedWrites;
const HOST: DriverFact = DriverFact::HostTransferImagesKeepTheirBytes;
const OPTIMIZED: DriverFact = DriverFact::DepthStencilImagesStayGpuOptimized;
const DEPTH16: DriverFact = DriverFact::FragmentDepthKeepsDepth16Codes;
const FETCH: DriverFact = DriverFact::SampledFetchRecoversCodes;
const RESTART: DriverFact = DriverFact::DynamicRestartWarns;
const FILL: DriverFact = DriverFact::FillCompiledPerCommandPool;
const DONT_CARE: DriverFact = DriverFact::DontCareStoreIsMetals;
const STAMPS: DriverFact = DriverFact::TimestampsWaitForEveryQueue;
const COLOR_BLOCKS: DriverFact = DriverFact::ColorBlockDecodeIsMetals;
const CHANNEL_BLOCKS: DriverFact = DriverFact::ChannelBlockDecodeIsMetals;
const PLANES: DriverFact = DriverFact::PositionedDepthReadAtStandardPositions;
const TESTS_DENORMALS: DriverFact = DriverFact::DepthTestKeepsDenormals;
const TESTS_MAGNITUDES: DriverFact = DriverFact::DepthTestComparesMagnitudes;
const SPARSE_WAIT: DriverFact = DriverFact::FirstSparseResourceWaitsForQueues;
const BC6H: DriverFact = DriverFact::Bc6hBlockDecodeIsMetals;
const BC7: DriverFact = DriverFact::Bc7BlockDecodeIsMetals;
const BC7_RESERVED: DriverFact = DriverFact::Bc7ReservedBlockIsMetals;
const SCARCE_POOLS: DriverFact = DriverFact::TimestampPoolsAreCounterSampleBuffers;
const ONE_OF_METALS: DriverFact = DriverFact::DontCareStoreIsOneOfMetals;
const COPY_BACK: DriverFact = DriverFact::ShadowCopyBackDispatchFaster;
const COPY_IN: DriverFact = DriverFact::ShadowCopyInDispatchFaster;
const ROWS: DriverFact = DriverFact::RowCopiesDispatchFaster;
const LINEAR_UNINITIALIZED: DriverFact = DriverFact::LinearTilingFeaturesUninitialized;
const STALL: DriverFact = DriverFact::SemaphoreOperationsStallSubmits;
const SAMPLED_HIZ: DriverFact = DriverFact::SampledDepthKeepsHiZ;
const MSAA_DEPTH: DriverFact = DriverFact::MultisampledDepthFasterOutOfGeneral;
const TILE_STORE: DriverFact = DriverFact::TileStoreIsMetals;
const FOLLOWS: DriverFact = DriverFact::PointerImportsFollowTheirAddress;
const INTERLOCK_ORDER: DriverFact = DriverFact::InterlockOrdersDraws;

/// Every committed profile's driver has exactly its facts: on a measured identity those its rows of
/// `MEASURED` name, and on every profile those the known-driver list names. A profile of a measured
/// identity: the 780M (RADV 26.2.4, Phoenix's and Hawk Point's alike), the gate's lavapipe 26.0.8
/// (both CPUs), the RPL-P and the RPL-S (ANV 26.2.4), the M3 Max and the M4 (MoltenVK 1.4.1, one
/// identity under macOS 27) and the RTX 5080 Laptop GPU (NVIDIA's 615.71.09). Of `MEASURED`: it
/// reads zero past the bound index range on the 780M, the RPL-S and the RTX 5080 Laptop GPU;
/// lavapipe stores fragment depth unchanged and keeps a 16-bit depth's codes; MoltenVK's color
/// resolve's average is Metal's and its depth test keeps denormals; a sampled fetch recovers its
/// codes on every measured identity; BC1 to BC3 decode as Metal's on the 780M and MoltenVK, BC4 and
/// BC5 on MoltenVK; BC6H on the 780M, the RPL-P, lavapipe 26.0.8, MoltenVK and the RTX 5080 Laptop
/// GPU, BC7 on the same but lavapipe, and BC7's reserved mode on MoltenVK alone. Of the list: depth
/// images keep denormals on the RTX 5080 Laptop GPU, lavapipe and MoltenVK, and values outside `[0,
/// 1]` on MoltenVK; MoltenVK's depth and stencil images lie apart and stay GPU-optimized, its depth
/// test compares magnitudes, it warns of a dynamic restart, compiles its fill per command pool,
/// stores `DONT_CARE` as Metal's own (the Apple9 family), orders its timestamps across queues,
/// makes its timestamp pools of counter sample buffers, reads its linear-tiling attachment features
/// uninitialized and stores a signed normalized minimum code as Metal's tile store does; ANV's
/// compressed images hide aliased writes on every Intel GPU, and on the RPL-P and the RPL-S its
/// host-transfer images keep their bytes, its first sparse resource waits for every queue,
/// `DONT_CARE` keeps one of Metal's two answers, a shadow's copy back and a copy of rows run faster
/// as dispatches, and on the RPL-S a merged shadow's copy in; on the RPL-P a sampled depth keeps
/// its hierarchical depth, a multisampled one renders faster out of `GENERAL`, pointer imports
/// follow their address and the interlock orders consecutive draws; NVIDIA's reads a depth rendered
/// at custom sample positions at the standard ones on every GPU, and on the RTX 5080 Laptop GPU
/// copies rows faster as a dispatch and stalls a submission per thousand semaphore operations;
/// `DONT_CARE` keeps one of Metal's two answers on the 780M; pointer imports follow their address
/// on every RADV GPU and lavapipe from Mesa 26.0.8 on. So the profiles of identities not measured
/// take the list's alone: the GTX 1060 (580.178.04) and the RTX 5080 desktop GPU (another GPU at
/// 615.71.09) NVIDIA's custom-position depth, the ARL (26.2.3) and the Tiger Lake Iris Xe (26.2.2)
/// ANV's hidden aliased writes, the RDNA2 and RDNA3 RADV profiles (26.2.3) the pointer imports,
/// lavapipe 26.2.3 the depth denormals and the pointer imports, and SwiftShader nothing.
#[test]
fn the_committed_profiles_drivers_have_their_facts() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut claims = BTreeMap::new();
    for path in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = path.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let facts: Vec<_> = desc.driver_facts().into_iter().collect();
        if !facts.is_empty() {
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            claims.insert(stem, facts);
        }
    }
    let radv_780m = vec![ZERO, FETCH, COLOR_BLOCKS, BC6H, BC7, ONE_OF_METALS, FOLLOWS];
    let lavapipe_gate = vec![DENORMALS, FRAGDEPTH, DEPTH16, FETCH, BC6H, FOLLOWS];
    let moltenvk = vec![
        DENORMALS,
        OUTSIDE,
        APART,
        AVERAGE,
        OPTIMIZED,
        FETCH,
        RESTART,
        FILL,
        DONT_CARE,
        STAMPS,
        COLOR_BLOCKS,
        CHANNEL_BLOCKS,
        TESTS_DENORMALS,
        TESTS_MAGNITUDES,
        BC6H,
        BC7,
        BC7_RESERVED,
        SCARCE_POOLS,
        LINEAR_UNINITIALIZED,
        TILE_STORE,
    ];
    let expected: BTreeMap<String, Vec<DriverFact>> = [
        ("amd-780m-hawkpoint-radv", radv_780m.clone()),
        ("amd-780m-radv", radv_780m),
        ("amd-rdna2-apu-radv", vec![FOLLOWS]),
        ("amd-rdna3-radv", vec![FOLLOWS]),
        ("gate-lavapipe", lavapipe_gate.clone()),
        ("gate-lavapipe-x86_64", lavapipe_gate),
        ("intel-arl-anv", vec![HIDE]),
        ("intel-irisxe-anv", vec![HIDE]),
        (
            "intel-irisxe-rplp-anv",
            vec![
                HIDE,
                HOST,
                FETCH,
                SPARSE_WAIT,
                BC6H,
                BC7,
                ONE_OF_METALS,
                COPY_BACK,
                ROWS,
                SAMPLED_HIZ,
                MSAA_DEPTH,
                FOLLOWS,
                INTERLOCK_ORDER,
            ],
        ),
        (
            "intel-uhd770-rpls-anv",
            vec![
                ZERO,
                HIDE,
                HOST,
                FETCH,
                SPARSE_WAIT,
                ONE_OF_METALS,
                COPY_BACK,
                COPY_IN,
                ROWS,
            ],
        ),
        ("lavapipe", vec![DENORMALS, FOLLOWS]),
        ("moltenvk-m3", moltenvk.clone()),
        ("moltenvk-m4", moltenvk),
        ("nvidia-pascal-linux", vec![PLANES]),
        ("nvidia-rtx50-linux", vec![PLANES]),
        (
            "nvidia-rtx5080-laptop",
            vec![ZERO, DENORMALS, FETCH, PLANES, BC6H, BC7, ROWS, STALL],
        ),
    ]
    .into_iter()
    .map(|(stem, facts)| (stem.to_owned(), facts))
    .collect();
    assert_eq!(claims, expected);
}

/// A release or GPU not measured is an unmeasured identity: it takes none of `MEASURED`'s facts,
/// only those the known-driver list names for its driver, release and GPU. The M4's identity at
/// MoltenVK 1.4.2 takes MoltenVK's listed answers from 1.4.1 on and its defects at every release;
/// at 1.4.0 the defects alone (its depth images apart and GPU-optimized, its scarce timestamp
/// pools, its uninitialized linear-tiling features); another Apple family at 1.4.1 all but the
/// `DONT_CARE` store, which the list keeps to the Apple9 family measured. The RPL-S at ANV 26.2.5
/// takes ANV's listed facts of the Intel venue's GPUs, another Intel GPU at 26.2.4 the hidden
/// aliased writes alone; the RTX 5080 Laptop GPU at a later release its listed facts, the 780M at
/// RADV 26.2.5 and lavapipe at Mesa 26.0.9 theirs. No driver takes another driver's.
#[test]
fn an_unmeasured_identity_takes_the_list_alone() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let read = |stem: &str| {
        let text = std::fs::read_to_string(dir.join(format!("{stem}.json"))).expect(stem);
        DeviceDescription::from_khronos_json(&text, None).expect("profile parses")
    };
    let mut desc = read("moltenvk-m4");
    assert!(desc.measured_identity());
    assert!(desc.driver_facts().contains(&FETCH));
    let defects = [APART, OPTIMIZED, SCARCE_POOLS, LINEAR_UNINITIALIZED];
    let answers = [
        DENORMALS,
        OUTSIDE,
        RESTART,
        FILL,
        STAMPS,
        TESTS_MAGNITUDES,
        TILE_STORE,
    ];
    desc.driver_version = moltenvk(1, 4, 2);
    assert!(!desc.measured_identity());
    let later: std::collections::BTreeSet<_> = defects
        .into_iter()
        .chain(answers)
        .chain([DONT_CARE])
        .collect();
    assert_eq!(desc.driver_facts(), later);
    desc.driver_version = moltenvk(1, 4, 0);
    assert!(!desc.measured_identity());
    assert_eq!(desc.driver_facts(), defects.into());
    let mut desc = read("moltenvk-m4");
    desc.device_id ^= 1;
    assert!(!desc.measured_identity());
    let other_family: std::collections::BTreeSet<_> = defects.into_iter().chain(answers).collect();
    assert_eq!(desc.driver_facts(), other_family);

    let mut desc = read("intel-uhd770-rpls-anv");
    assert!(desc.measured_identity());
    desc.driver_version += 1;
    assert!(!desc.measured_identity());
    assert_eq!(
        desc.driver_facts(),
        [
            HIDE,
            HOST,
            SPARSE_WAIT,
            ONE_OF_METALS,
            COPY_BACK,
            COPY_IN,
            ROWS
        ]
        .into()
    );
    let mut desc = read("intel-uhd770-rpls-anv");
    desc.device_id ^= 1;
    assert_eq!(desc.driver_facts(), [HIDE].into());

    let mut desc = read("nvidia-rtx5080-laptop");
    assert!(desc.measured_identity());
    desc.driver_version += 1;
    assert!(!desc.measured_identity());
    assert_eq!(desc.driver_facts(), [DENORMALS, PLANES, ROWS, STALL].into());

    for (stem, facts) in [
        ("amd-780m-radv", [ONE_OF_METALS, FOLLOWS]),
        ("gate-lavapipe", [DENORMALS, FOLLOWS]),
    ] {
        let mut desc = read(stem);
        assert!(desc.measured_identity(), "{stem}");
        desc.driver_version += 1;
        assert_eq!(desc.driver_facts(), facts.into(), "{stem}");
    }
}

/// Each measured identity is a GPU and release a venue ran: the committed profiles of the measured
/// GPUs carry exactly the identities listed, so no row names a release or GPU nobody measured. The
/// Radeon 780M's capture under its Hawk Point `deviceID` is a measured identity through Phoenix's.
#[test]
fn the_measured_identities_are_the_committed_profiles() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let mut measured = Vec::new();
    for stem in [
        "amd-780m-hawkpoint-radv",
        "amd-780m-radv",
        "gate-lavapipe",
        "gate-lavapipe-x86_64",
        "intel-irisxe-rplp-anv",
        "intel-uhd770-rpls-anv",
        "moltenvk-m3",
        "moltenvk-m4",
        "nvidia-rtx5080-laptop",
    ] {
        let text = std::fs::read_to_string(dir.join(format!("{stem}.json"))).expect(stem);
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        assert!(desc.measured_identity(), "{stem}");
        measured.push((
            desc.driver_id.expect("driver"),
            desc.vendor_id,
            desc.device_id,
            desc.driver_version,
        ));
    }
    for m in MEASURED_DEVICES {
        let id = (m.driver, m.vendor, m.device, m.release);
        assert!(
            measured.contains(&id) || MEASURED_WITHOUT_PROFILE.contains(&id),
            "{id:?} is measured but no venue's profile names it"
        );
    }
}

/// One GPU that reports more than one `deviceID` takes its measured rows under each: the Radeon
/// 780M of Hawk Point (0x1900) is Phoenix's (0x15bf), so a RADV 26.2.4 description of it is a
/// measured identity with Phoenix's driver facts, deviations and defects, and derives Phoenix's
/// profile, its Function storage among it, and fingerprint. The release a row names still holds
/// exactly: 0x1900 at Mesa 26.2.3 or 26.2.5 is unmeasured, as Phoenix is there, and takes
/// Phoenix's listed facts there. A neighbouring `deviceID` (0x1901), or 0x1900 of another vendor,
/// is another GPU: unmeasured, its own.
#[test]
fn a_gpu_under_another_of_its_device_ids_takes_its_measured_rows() {
    use crate::device::driver_facts::{defect_rows, deviations};
    use crate::device::{Device, DevicePolicy, FunctionStorage};
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let text = std::fs::read_to_string(dir.join("amd-780m-radv.json")).expect("amd-780m-radv");
    let phoenix = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    let profile = |desc: &DeviceDescription| {
        Device::new(desc.clone(), DevicePolicy::default())
            .expect("the device is made")
            .profile()
            .clone()
    };
    let mut hawk_point = phoenix.clone();
    hawk_point.device_id = 0x1900;
    assert_eq!(phoenix.canonical_device_id(), 0x15bf);
    assert_eq!(hawk_point.canonical_device_id(), 0x15bf);
    assert!(hawk_point.measured_identity());
    assert!(!hawk_point.driver_facts().is_empty());
    assert_eq!(hawk_point.driver_facts(), phoenix.driver_facts());
    assert!(hawk_point.driver_facts_withheld().is_empty());
    assert!(!deviations(&hawk_point, true).is_empty());
    assert_eq!(deviations(&hawk_point, true), deviations(&phoenix, true));
    assert_eq!(defect_rows(&hawk_point), defect_rows(&phoenix));
    assert_eq!(
        profile(&hawk_point).fields().function_storage,
        FunctionStorage::Measured(96 * 1024)
    );
    assert_eq!(
        profile(&hawk_point).fingerprint(),
        profile(&phoenix).fingerprint()
    );

    for release in [mesa(26, 2, 3), mesa(26, 2, 5)] {
        let (mut other, mut phoenix_there) = (hawk_point.clone(), phoenix.clone());
        other.driver_version = release;
        phoenix_there.driver_version = release;
        assert!(!other.measured_identity(), "{release:#x}");
        assert_eq!(other.driver_facts(), phoenix_there.driver_facts());
        assert!(!other.driver_facts().contains(&FETCH), "{release:#x}");
        assert!(!other.driver_facts_withheld().is_empty(), "{release:#x}");
        assert_eq!(
            other.driver_facts_withheld(),
            phoenix_there.driver_facts_withheld()
        );
        assert_eq!(deviations(&other, true), deviations(&phoenix_there, true));
    }

    let mut neighbour = hawk_point.clone();
    neighbour.device_id = 0x1901;
    let mut other_vendor = hawk_point.clone();
    other_vendor.vendor_id = 0x8086;
    // The pointer imports are listed on every RADV GPU of AMD's.
    for (desc, device, facts) in [
        (&neighbour, 0x1901, vec![FOLLOWS]),
        (&other_vendor, 0x1900, vec![]),
    ] {
        assert_eq!(desc.canonical_device_id(), device);
        assert!(!desc.measured_identity(), "{device:#x}");
        assert_eq!(
            desc.driver_facts().into_iter().collect::<Vec<_>>(),
            facts,
            "{device:#x}"
        );
    }
    assert!(!neighbour.driver_facts_withheld().is_empty());
    assert!(deviations(&neighbour, true).len() < deviations(&phoenix, true).len());
    assert_eq!(
        profile(&neighbour).fields().function_storage,
        FunctionStorage::Assumed(16 * 1024)
    );
}

/// The identities measured whose venue has no committed profile: the RTX 3090 (gem10) at NVIDIA's
/// 615.71.09, lavapipe at Mesa 26.2.4 (the Linux venues' own lavapipe), and the M3 Max under macOS
/// 26.7 at MoltenVK 1.4.1 (its venue's capture is now macOS 27's, the M4's identity).
const MEASURED_WITHOUT_PROFILE: [(DriverId, u32, u32, u32); 3] = [
    (
        DriverId::NVIDIA_PROPRIETARY,
        0x10de,
        NVIDIA_RTX_3090,
        nvidia(615, 71, 9),
    ),
    (DriverId::MESA_LLVMPIPE, 0x10005, 0, mesa(26, 2, 4)),
    (
        DriverId::MOLTENVK,
        APPLE,
        crate::device::driver_facts::APPLE_M3_MAX,
        moltenvk(1, 4, 1),
    ),
];

/// A driver listed to lay depth and stencil images apart from their memory is listed to keep every
/// `float` in a depth image at each release and GPU measured with it, so it holds no `Depth32Float`
/// texture with a depth twin there and no heap texture's twin is kept apart on it. A driver listed
/// otherwise would reach the twin's `heap-depth-stencil-apart` route, which then needs a shadow of
/// its own.
#[test]
fn a_driver_that_lays_depth_images_apart_holds_no_depth_twin() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let text = std::fs::read_to_string(dir.join("moltenvk-m4.json")).expect("moltenvk-m4");
    let base = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    let apart: Vec<_> = LISTED_FACTS.iter().filter(|e| e.fact == APART).collect();
    assert!(
        !apart.is_empty(),
        "no driver is listed to lay depth images apart"
    );
    for entry in apart {
        for &device in entry.evidence.gpus {
            let mut desc = base.clone();
            desc.driver_id = Some(entry.on.driver);
            desc.vendor_id = entry.on.vendor.unwrap_or(APPLE);
            desc.device_id = device;
            desc.driver_version = entry.evidence.last;
            let facts = listed(LISTED_FACTS, &desc);
            assert!(
                facts.contains_key(&DENORMALS) && facts.contains_key(&OUTSIDE),
                "{:?} on {device:#x} at {:#x} lays depth images apart but may lose a depth \
                 texel: {facts:?}",
                entry.on.driver,
                entry.evidence.last
            );
        }
    }
}

/// NVIDIA's 615.71.09 stalls a submission about once in every thousand semaphore operations on
/// the two GPUs measured, the RTX 5080 Laptop GPU and the RTX 3090, which take the fact from that
/// release on; another GPU of the driver (an RTX 5080 desktop GPU), or an earlier release, does
/// not, so it waits for completed timelines as every other driver does. No other driver has it.
#[test]
fn nvidia_s_semaphore_stall_holds_on_the_gpus_measured() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let read = |stem: &str| {
        let text = std::fs::read_to_string(dir.join(format!("{stem}.json"))).expect(stem);
        DeviceDescription::from_khronos_json(&text, None).expect("profile parses")
    };
    let laptop = read("nvidia-rtx5080-laptop");
    assert!(laptop.driver_facts().contains(&STALL));
    let mut rtx_3090 = laptop.clone();
    rtx_3090.device_id = NVIDIA_RTX_3090;
    assert!(rtx_3090.measured_identity());
    assert!(rtx_3090.driver_facts().contains(&STALL));
    let mut desktop = laptop.clone();
    desktop.device_id = 0x2c02;
    assert!(!desktop.driver_facts().contains(&STALL));
    let mut later = laptop.clone();
    later.driver_version = nvidia(615, 71, 10);
    assert!(later.driver_facts().contains(&STALL));
    let mut earlier = laptop.clone();
    earlier.driver_version = nvidia(580, 95, 5);
    assert!(!earlier.driver_facts().contains(&STALL));
    let drivers: Vec<DriverId> = LISTED_FACTS
        .iter()
        .filter(|entry| entry.fact == STALL)
        .map(|entry| entry.on.driver)
        .collect();
    assert_eq!(drivers, [DriverId::NVIDIA_PROPRIETARY]);
}

/// MoltenVK's warning of a pipeline whose restart may be disabled (`MVKPipeline.mm` lines 881-893
/// at 1.4.1) is the diagnostic of a pipeline that makes the restart dynamic with its topology
/// dynamic or a strip or fan, exactly as the driver reports it; no other pipeline's, and no other
/// fact's. It is listed on MoltenVK from 1.4.1 on, on any GPU: 1.4.0 relays it, as does any other
/// driver at 1.4.1's `driverVersion`.
#[test]
fn moltenvk_1_4_1_warns_of_a_pipeline_whose_restart_is_dynamic() {
    use crate::device::DriverDiagnostic;
    use crate::vk::{DynamicState as D, PrimitiveTopology as T};
    let warning = DriverDiagnostic {
        id_name: "mvk-warn",
        id: 0,
        text: "VK_ERROR_FEATURE_NOT_PRESENT: Metal does not support disabling primitive restart.",
    };
    let both = [D::PRIMITIVE_TOPOLOGY, D::PRIMITIVE_RESTART_ENABLE];
    let restart = [D::CULL_MODE, D::PRIMITIVE_RESTART_ENABLE];
    let topology = [D::PRIMITIVE_TOPOLOGY];
    let cases: [(&[D], T, bool); 10] = [
        (&both, T::TRIANGLE_LIST, true),
        (&both, T::LINE_LIST, true),
        (&both, T::POINT_LIST, true),
        (&restart, T::TRIANGLE_STRIP, true),
        (&restart, T::LINE_STRIP, true),
        (&restart, T::TRIANGLE_FAN, true),
        (&restart, T::TRIANGLE_LIST, false),
        (&topology, T::TRIANGLE_STRIP, false),
        (&[], T::TRIANGLE_STRIP, false),
        (&[D::CULL_MODE], T::LINE_STRIP, false),
    ];
    for (dynamic, top, warns) in cases {
        assert_eq!(
            RESTART.pipeline_creation_diagnostic(dynamic, top),
            warns.then_some(warning),
            "{dynamic:?} {top:?}"
        );
        for other in [ZERO, DENORMALS, FETCH] {
            assert_eq!(other.pipeline_creation_diagnostic(dynamic, top), None);
        }
    }
    let entries: Vec<_> = LISTED_FACTS
        .iter()
        .filter(|entry| entry.fact == RESTART)
        .collect();
    assert_eq!(entries.len(), 1);
    let on = entries[0].on;
    assert_eq!(
        (on.driver, on.vendor, on.gpus, on.releases),
        (
            DriverId::MOLTENVK,
            None,
            Gpus::Any,
            Releases::since(moltenvk(1, 4, 1))
        )
    );
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../metal2vulkan-validation/profiles/moltenvk-m3.json"),
    )
    .expect("moltenvk-m3");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
    for (version, has) in [(10400, false), (10401, true), (10402, true)] {
        desc.driver_version = version;
        assert_eq!(desc.driver_facts().contains(&RESTART), has, "{version}");
    }
    desc.driver_version = 10401;
    desc.driver_id = Some(DriverId::MESA_LLVMPIPE);
    assert!(!desc.driver_facts().contains(&RESTART));
}

/// A description leaves out of every format's linear tiling the features its driver reports from
/// memory it has not initialized: on MoltenVK, measured or not, the color attachment, its
/// blending, the blit destination and the depth/stencil attachment
/// (`DriverFact::LinearTilingFeaturesUninitialized`); on every other driver nothing. Every
/// committed profile's formats hold none of them.
#[test]
fn moltenvks_linear_attachment_features_are_left_out() {
    use crate::vk::FormatFeatureFlags2 as F;
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../metal2vulkan-validation/profiles");
    let attachments =
        F::COLOR_ATTACHMENT | F::COLOR_ATTACHMENT_BLEND | F::BLIT_DST | F::DEPTH_STENCIL_ATTACHMENT;
    let mut moltenvk = 0;
    for path in std::fs::read_dir(&dir).expect("validation/profiles is readable") {
        let path = path.expect("directory entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("profile is readable");
        let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("profile parses");
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let unreliable = desc.unreliable_linear_tiling();
        if desc.driver_id == Some(DriverId::MOLTENVK) {
            moltenvk += 1;
            assert_eq!(unreliable, attachments, "{stem}");
            desc.driver_version += 1;
            assert_eq!(
                desc.unreliable_linear_tiling(),
                attachments,
                "{stem} unmeasured"
            );
        } else {
            assert!(unreliable.is_empty(), "{stem}");
        }
        for (format, p) in &desc.formats.formats {
            assert!(
                p.linear_tiling.intersection(unreliable).is_empty(),
                "{stem} {format:?}"
            );
        }
    }
    assert!(moltenvk >= 2, "the MoltenVK profiles are read");
}
