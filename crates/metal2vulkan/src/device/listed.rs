//! The known-driver list: what a driver does where no check on the device settles it, matched on
//! the driver (`VkPhysicalDeviceDriverProperties::driverID`, with its GPU's `vendorID` where an
//! entry names one), a range of its releases (`driverVersion`) and a GPU family (a set of
//! `deviceID`s, each read as [`DeviceDescription::canonical_device_id`]), never on a device's name.
//! An entry ([`Listed`]) is one of two kinds:
//!
//! - **A defect**: the driver does otherwise than Metal, and the runtime works around it. Listed
//!   where it can lose the device, crash the process or hang the GPU, or where one clean run does
//!   not prove a device free of it (aliasing visibility under a compression state, a race, an
//!   uninitialized read). It holds from the first release known to have it (0 where none is known)
//!   and stays open on the affected side: its [`Releases::until`] is set only where a release was
//!   measured with it and a later one without it on the same GPU family, or where the driver's
//!   source shows the fix. A GPU measured clean that was never measured broken is a clean identity:
//!   the entries leave it out at that release alone.
//! - **An answer**: the driver gives Metal's answer, or a cost that chooses between two exact
//!   forms, where a check cannot settle it (its inputs would be invalid usage, a race, a cost no
//!   boot can time). It holds from the first release measured with it and is open-ended: the venue
//!   lanes that measured it run on the releases the venues run, so a release that changes it fails
//!   there. It names every GPU of the driver ([`Gpus::Any`]) only where the driver's source or
//!   structure shows the answer is not the GPU's (MoltenVK handing the operation to Metal, Mesa's
//!   common code, lavapipe), each entry saying which; elsewhere the GPUs measured. Taken past the
//!   last release measured or on a GPU outside those measured, it says so
//!   ([`DeviceDescription::listed_past_evidence`]).
//!
//! A device no entry names follows Vulkan's specification: the exact path. The facts a check may
//! settle are not listed: they are `driver_facts::MEASURED`'s, taken on a measured identity alone,
//! and the runtime's own defect table's.

use crate::device::description::APPLE_VENDOR_ID;
use crate::device::driver_facts::{
    mesa, moltenvk, nvidia, release_name, AMD_RADEON_780M, AMD_VENDOR_ID, APPLE_M3_MAX, APPLE_M4,
    INTEL_IRIS_XE_RPL_P, INTEL_UHD_770_RPL_S, INTEL_VENDOR_ID, NVIDIA_RTX_3090,
    NVIDIA_RTX_5080_LAPTOP, NVIDIA_VENDOR_ID,
};
use crate::device::{DeviceDescription, DriverFact};
use crate::error::RuntimeDriverDefect;
use crate::vk::DriverId;
use std::collections::{BTreeMap, BTreeSet};

/// The releases of a driver an entry holds on: from the `driverVersion` `from` up to, but not
/// including, `until`, the first release known without it (`None`: none is known).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Releases {
    pub(crate) from: u32,
    pub(crate) until: Option<u32>,
}

impl Releases {
    /// Every release of the driver.
    pub(crate) const EVERY: Releases = Releases {
        from: 0,
        until: None,
    };

    /// The release `from` and every later one.
    pub(crate) const fn since(from: u32) -> Releases {
        Releases { from, until: None }
    }

    /// Every release before `until`.
    pub(crate) const fn before(until: u32) -> Releases {
        Releases {
            from: 0,
            until: Some(until),
        }
    }

    /// Whether the release `release` is one of these.
    pub(crate) fn contains(self, release: u32) -> bool {
        release >= self.from && self.until.is_none_or(|until| release < until)
    }
}

/// The GPUs of a driver an entry holds on, by `deviceID` (the one each was measured under,
/// [`DeviceDescription::canonical_device_id`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gpus {
    /// Every GPU the driver runs.
    Any,
    /// The GPUs of these `deviceID`s.
    Only(&'static [u32]),
    /// Every GPU but those of these `deviceID`s.
    Except(&'static [u32]),
    /// The Apple GPUs of one family, on every macOS release: MoltenVK's `deviceID` on an Apple GPU
    /// is `(macOS major << 24) + (macOS minor << 16) + (Mac family << 8) + Apple family`
    /// (`MVKPhysicalDevice::getHighestGPUCapability`, MoltenVK 1.4.1), so the family is its low 16
    /// bits under Apple's `vendorID` ([`APPLE9`]).
    MoltenVkFamily(u16),
}

impl Gpus {
    /// Whether the GPU of `vendor` and canonical `device` is one of these.
    fn contains(self, vendor: u32, device: u32) -> bool {
        match self {
            Gpus::Any => true,
            Gpus::Only(devices) => devices.contains(&device),
            Gpus::Except(devices) => !devices.contains(&device),
            Gpus::MoltenVkFamily(family) => {
                vendor == APPLE_VENDOR_ID && device & 0xffff == u32::from(family)
            }
        }
    }
}

/// The Apple9 GPU family as MoltenVK names it ([`Gpus::MoltenVkFamily`]): Mac family 2, Apple
/// family 9, the M3 and M4 series. The Apple M3 Max ([`APPLE_M3_MAX`]) and the Apple M4
/// ([`APPLE_M4`]) are of it.
pub(crate) const APPLE9: u16 = 0x0209;

/// Where an entry holds: a driver, its GPU's vendor where the entry names one, a GPU family and a
/// range of releases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Listing {
    pub(crate) driver: DriverId,
    /// The GPU vendor (`vendorID`), or `None` where the entry is the driver's on every vendor's.
    pub(crate) vendor: Option<u32>,
    pub(crate) gpus: Gpus,
    pub(crate) releases: Releases,
}

impl Listing {
    /// Whether this listing names the device `description` describes.
    pub(crate) fn names(&self, description: &DeviceDescription) -> bool {
        description.driver_id == Some(self.driver)
            && self.vendor.is_none_or(|v| v == description.vendor_id)
            && (self.gpus).contains(description.vendor_id, description.canonical_device_id())
            && self.releases.contains(description.driver_version)
    }
}

/// What an entry rests on: the last release measured with it, and the GPUs measured (canonical
/// `deviceID`s; lavapipe's 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Evidence {
    pub(crate) last: u32,
    pub(crate) gpus: &'static [u32],
}

impl Evidence {
    /// Whether `description`'s device is within this evidence: at a release up to the last
    /// measured, on a GPU measured.
    fn covers(&self, description: &DeviceDescription) -> bool {
        description.driver_version <= self.last
            && self.gpus.contains(&description.canonical_device_id())
    }
}

/// One entry of the list: the fact `F`, where it holds, and what it rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Listed<F> {
    pub(crate) fact: F,
    pub(crate) on: Listing,
    pub(crate) evidence: Evidence,
}

impl<F> Listed<F> {
    /// The entry as a feature status names it: its driver, releases and GPUs, and where
    /// `description`'s device lies past its evidence, the last release and the GPUs measured
    /// (`measured through <release> on <GPUs>`).
    pub(crate) fn named(&self, description: &DeviceDescription) -> String {
        let Listing {
            driver,
            vendor,
            gpus,
            releases,
        } = self.on;
        let name = |release| release_name(Some(driver), release);
        let releases = match (releases.from, releases.until) {
            (0, None) => "every release".to_owned(),
            (from, None) => format!("from {}", name(from)),
            (0, Some(until)) => format!("before {}", name(until)),
            (from, Some(until)) => format!("from {} before {}", name(from), name(until)),
        };
        let ids = |devices: &[u32]| {
            let ids: Vec<String> = devices.iter().map(|d| format!("{d:04x}")).collect();
            ids.join(", ")
        };
        let of_vendor = vendor.map_or(String::new(), |v| format!(" of vendor {v:04x}"));
        let gpus = match gpus {
            Gpus::Any => format!("every GPU{of_vendor}"),
            Gpus::Only(devices) => format!("GPU {}{of_vendor}", ids(devices)),
            Gpus::Except(devices) => format!("every GPU{of_vendor} but {}", ids(devices)),
            Gpus::MoltenVkFamily(family) => format!("Apple GPU family {family:04x}"),
        };
        let past = match self.evidence.covers(description) {
            true => String::new(),
            false => format!(
                ", measured through {} on {}",
                name(self.evidence.last),
                ids(self.evidence.gpus)
            ),
        };
        format!("{driver:?} {releases} on {gpus}{past}")
    }
}

/// The facts of `table` an entry of which names `description`'s device, each with the first
/// such entry as a feature status names it ([`Listed::named`]).
pub(crate) fn listed<F: Copy + Ord>(
    table: &[Listed<F>],
    description: &DeviceDescription,
) -> BTreeMap<F, String> {
    let mut facts = BTreeMap::new();
    for entry in table.iter().filter(|entry| entry.on.names(description)) {
        facts
            .entry(entry.fact)
            .or_insert_with(|| entry.named(description));
    }
    facts
}

/// The facts of `table` an entry of which names `description`'s device but none covers it in
/// its evidence, each with the evidence of the first that names it.
fn past_evidence<F: Copy + Ord>(
    table: &[Listed<F>],
    description: &DeviceDescription,
) -> BTreeMap<F, Evidence> {
    let naming = || table.iter().filter(|entry| entry.on.names(description));
    let covered: BTreeSet<F> = naming()
        .filter(|entry| entry.evidence.covers(description))
        .map(|entry| entry.fact)
        .collect();
    let mut past = BTreeMap::new();
    for entry in naming().filter(|entry| !covered.contains(&entry.fact)) {
        past.entry(entry.fact).or_insert(entry.evidence);
    }
    past
}

/// What a listed answer a device takes rests on, where the device lies past it
/// ([`DeviceDescription::listed_past_evidence`]): the last release measured with the answer and the
/// GPUs measured.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ListEvidence {
    /// The `driverVersion` of the last release measured with the answer.
    pub last_release: u32,
    /// The `deviceID`s of the GPUs measured with it, as each was measured
    /// ([`DeviceDescription::canonical_device_id`]); 0 for lavapipe, which runs on the CPU.
    pub devices: &'static [u32],
}

impl DeviceDescription {
    /// The runtime driver defects the known-driver list names for this device's driver, release
    /// and GPU (the module docs): those that can lose the device, crash the process or hang the
    /// GPU, or that one clean run does not rule out. Each comes with the entry that names it, as
    /// the device's feature status names it: its driver, releases and GPUs. The runtime adds the
    /// defects of its own table, which a check may settle.
    pub fn listed_driver_defects(&self) -> BTreeMap<RuntimeDriverDefect, String> {
        listed(LISTED_DEFECTS, self)
    }

    /// The listed answers ([`DriverFact`]) this device takes past what they rest on, each with
    /// that evidence: a release newer than the last measured with the answer, or a GPU other than
    /// those measured (the module docs). Empty where every listed answer the device takes was
    /// measured on its GPU at its release or an earlier one. A listed defect
    /// ([`DriverFact::assumed_where_unmeasured`]) is never in it: the list holds a defect wherever
    /// no release is known without it, so taking it past its measurement takes the workaround.
    pub fn listed_past_evidence(&self) -> BTreeMap<DriverFact, ListEvidence> {
        past_evidence(LISTED_FACTS, self)
            .into_iter()
            .filter(|(fact, _)| !fact.assumed_where_unmeasured())
            .map(|(fact, evidence)| {
                let evidence = ListEvidence {
                    last_release: evidence.last,
                    devices: evidence.gpus,
                };
                (fact, evidence)
            })
            .collect()
    }
}

/// An entry of `fact` on `driver` (its GPU's vendor `vendor`), on `gpus` at `releases`, resting
/// on `evidence`.
const fn entry<F>(
    fact: F,
    driver: DriverId,
    vendor: Option<u32>,
    gpus: Gpus,
    releases: Releases,
    evidence: Evidence,
) -> Listed<F> {
    Listed {
        fact,
        on: Listing {
            driver,
            vendor,
            gpus,
            releases,
        },
        evidence,
    }
}

/// Evidence of the last release `last` on the GPUs `gpus`.
const fn seen(last: u32, gpus: &'static [u32]) -> Evidence {
    Evidence { last, gpus }
}

/// The Intel venue's two devices, the Iris Xe of Raptor Lake-P and the UHD 770 of Raptor Lake-S.
const INTEL_VENUE: &[u32] = &[INTEL_IRIS_XE_RPL_P, INTEL_UHD_770_RPL_S];

/// The RTX 3090 and the RTX 5080 Laptop GPU, NVIDIA's two GPUs measured.
const NVIDIA_MEASURED: &[u32] = &[NVIDIA_RTX_3090, NVIDIA_RTX_5080_LAPTOP];

/// MoltenVK's two identities measured: the Apple M3 Max under macOS 26.7, and the Apple M4 and the
/// M3 Max under macOS 27.0.
const APPLE_MEASURED: &[u32] = &[APPLE_M3_MAX, APPLE_M4];

/// The AMD Radeon RX 7900 XTX's `deviceID`, measured on Mesa 26.0.8 (homelab-um780).
const AMD_RX_7900_XTX: u32 = 0x744c;

/// lavapipe's `deviceID`: one CPU device, whatever the CPU.
const LAVAPIPE: &[u32] = &[0];

/// Every listed runtime driver fact ([`DriverFact`]), by fact. What each entry rests on:
///
/// `DepthImageKeepsDenormals` and `DepthImageKeepsValuesOutsideUnitRange`, measured by
/// `runtime_lane::texture_blits::depth_images_keep_what_their_drivers_claim` (at d0c9878a,
/// 2026-10-06): 11 denormals of both signs and 5 values outside `[0, 1]` through `replace_region`
/// into a render target's `Depth32Float` depth image, read back by `get_bytes`, and the first again
/// by `runtime_lane::depth_denormals` (2026-10-10; with `DepthTestKeepsDenormals`, `MEASURED`):
/// - NVIDIA's 615.71.09 keeps every denormal, with `VK_EXT_depth_range_unrestricted`: the GeForce
///   RTX 3090 and the RTX 5080 Laptop GPU. Taken on those GPUs from that release on.
/// - lavapipe keeps every denormal, with the extension: arm64, Mesa 26.0.8 with LLVM 21.1.8, in
///   `scripts/linux-gate` on an M4 mini; x86, Mesa 26.2.4 with LLVM 23.1.1, on an Iris Xe (RPL-P)
///   host. Taken from Mesa 26.0.8 on: lavapipe's depth image is its own memory, whatever the CPU.
/// - MoltenVK 1.4.1 (`driverVersion` 10401) keeps both, without the extension: an Apple M4 mini
///   and the M3 Max. Taken on every Apple GPU from 1.4.1 on: MoltenVK makes `D32_SFLOAT` Metal's
///   own `Depth32Float`, which keeps any `float` (Metal's probe `depthkeep`).
///
/// Measured to keep neither, so not listed: RADV (`MESA_RADV`), Mesa 26.2.4 on the Radeon 780M (no
/// denormal; every value outside `[0, 1]` under the extension), and ANV
/// (`INTEL_OPEN_SOURCE_MESA`), Mesa 26.2.4 on Intel Graphics (RPL-S) and the Iris Xe (RPL-P)
/// without the extension (no denormal, every value outside `[0, 1]` clamped). A driver with the
/// extension is not measured for `DepthImageKeepsValuesOutsideUnitRange`, which only a device
/// without it reads, and no check reaches it: the values a copy would store are invalid usage
/// there (`VUID-vkCmdCopyBufferToImage-pRegions-07931`). Both decide whether a read-only
/// `Depth32Float` texture takes a depth twin, and so its heap size and placement.
///
/// `DepthStencilImagesLieApart`, measured by
/// `runtime_lane::heap_textures::heap_depth_textures_keep_their_bytes_with_their_twins` (two
/// `Depth32Float` textures side by side in a placement heap, then a buffer over the first one's
/// bytes filled): MoltenVK (`MOLTENVK`) 1.4.1 (`driverVersion` 10401), on an Apple M4 mini
/// (2026-10-06): the fill changed none of the first's texels in a `Private` heap. MoltenVK's source
/// states it for every image with a depth or stencil aspect on any GPU
/// (`MVKImagePlane::getMTLTexture` makes the texture from the device, not from the memory's
/// `MTLHeap`: "Metal support for depth/stencil from heaps is flaky"), so it holds at every release
/// and on every GPU. Measured not to have it, so not listed: RADV, Mesa 26.2.4 on the Radeon 780M;
/// ANV, Mesa 26.2.4 on the Iris Xe (RPL-P); NVIDIA's 615.71.09 on the RTX 3090; lavapipe, x86, Mesa
/// 26.0.8 (the fill changed every texel of the first). A visible fill under one compression state
/// does not prove another, so no check settles it.
///
/// `CompressedImagesHideAliasedWrites`, measured by
/// `runtime_lane::heap_textures::a_buffer_write_shows_in_an_overlapping_texture` (a heap's
/// buffer and `RGBA8Unorm` texture over the same bytes; the texture written by a blit, then the
/// buffer by a copy): ANV (`INTEL_OPEN_SOURCE_MESA`) on Intel's GPUs, Mesa 26.2.4, on the Iris Xe
/// (RPL-P) and Intel Graphics (RPL-S) (2026-10-06): the texture's write changed 234 of the buffer's
/// 1024 bytes (compressed), and the buffer's write changed none of the 256 texels in a `Shared`
/// heap. A defect of ANV's compression, held at every release and on every GPU: whether an image
/// is compressed follows its state, so no clean run proves a device free of it. Measured not to
/// have it, so not listed: RADV, Mesa 26.2.4 on the Radeon 780M; NVIDIA's 615.71.09 on the RTX
/// 3090; lavapipe, x86, Mesa 26.0.8; MoltenVK 1.4.1 on an M4 (every texel changed).
///
/// `HostTransferImagesKeepTheirBytes`, measured by our standalone probe (2026-10-06;
/// 8 colour formats, 4 usage sets, 16, 64 and 256 texels square, host-visible and device-local
/// memory, the image at the start of its own allocation and at 64 KiB of a 1 MiB one,
/// `hostImageCopy` enabled): ANV, Mesa 26.2.4, on the same RPL-P and RPL-S: in all 90 cases of each
/// memory kind and placement an optimal image of host-transfer usage read the buffer's fill at
/// every texel, as a linear one did; the 6 more, `RGBA8Unorm_sRGB` with storage usage, are no image
/// the format takes. Neither device lists `VK_EXT_image_compression_control`, the route the ruling
/// ordered first. Taken on those GPUs from that release on. It and
/// `CompressedImagesHideAliasedWrites` decide a heap colour texture's image, and so its heap size.
///
/// `DepthStencilImagesStayGpuOptimized`, measured by
/// `runtime_lane::heap_textures::heap_textures_answer_and_place_as_metal_s_do` (a 64x64
/// `Depth32Float` render target's heap size with `allowGPUOptimizedContents` on and off; the
/// image with the property off is one of `PixelFormatView` usage): MoltenVK 1.4.1 on Apple's GPUs,
/// an M4 mini (2026-10-07) and the M3 Max: 16512 bytes both ways, Metal's answer with the property
/// on, where Metal answers 16384 with it off (every depth and stencil format and shape alike).
/// MoltenVK's source gives a depth-only image no `PixelFormatView` usage, and sets its texture's
/// `allowGPUOptimizedContents` only from create flags (a 2D view of a 3D image, block texel views)
/// it reads after sizing the image, so it holds at every release and on every GPU. Another
/// driver's sizes are its own, not compared with Metal's.
///
/// `DynamicRestartWarns`, `FillCompiledPerCommandPool` and `TimestampsWaitForEveryQueue`: MoltenVK's
/// own code at 1.4.1 (the facts' docs), measured on the Apple M4; taken from 1.4.1 on, on every GPU
/// (the last on Apple's alone, which sample a counter only at a pass's boundaries). Their cost is
/// not one a device times at boot: a submission behind a storm of compiles, a timestamp behind
/// another queue's long work.
///
/// `DontCareStoreIsMetals`, read in MoltenVK 1.4.1's source
/// (`MVKAttachmentDescription::getMTLStoreAction`; the restarts of `MVKCommandEncoder`,
/// `MVKCmdPipelineBarrier` and the draws' compute stages) against Metal's answer on the GPU, and
/// measured by `runtime_lane::dont_care_stores` through MoltenVK 1.4.1 on the M4 ([`APPLE_M4`]) and
/// the M3 Max ([`APPLE_M3_MAX`]), both at 8edef9600: every kept case kept its texels. What Metal
/// keeps is its GPU's tiler's, so it is taken on the Apple9 family measured ([`APPLE9`]), from 1.4.1
/// on; Apple's other GPUs are unmeasured, and announce.
///
/// `PositionedDepthReadAtStandardPositions`, measured by `runtime_lane::sample_positions` and a
/// probe of it (at f1accaf20 and our probe at 3cfc03dab, 2026-10-10): NVIDIA's
/// (`NVIDIA_PROPRIETARY`) 615.71.09 on the GeForce RTX 3090: plane X drawn at four custom positions
/// into a 4x `Depth32Float` read back each sample's depth at the standard positions, in the same
/// command buffer and in a later one, under each layout transition pair after the pass; a pass at
/// four other positions that loaded it drew no sample under `Less` (each stored depth taken at its
/// own positions); the pass's own test under `Less` drew the samples the custom positions put below
/// the clear; and a caller's image rendered through the stand-in, its depth copied back after the
/// pass, read Metal's depths. A defect held at every release and on every NVIDIA GPU: it follows the
/// depth's compression state, which one clean run does not rule out, and the stand-in it selects is
/// exact on every device. Measured not to have it, so not listed: RADV, Mesa 26.2.4 on the Radeon
/// 780M (every read and later pass kept the custom positions' depths); ANV, Mesa 26.2.4 on the Iris
/// Xe (RPL-P); lavapipe, x86, LLVM 21.1.8; and MoltenVK 1.4.1 on the M4, whose
/// `runtime_lane::sample_positions` passed rendering the texture's own image (at 53ae2b964).
///
/// `DepthTestComparesMagnitudes`, measured by a probe of ours (not committed; 2026-10-15) on a
/// runtime that renders every depth in place: 18 values (+-2, +-0.5, the smallest and largest
/// denormal of each sign, +-0, the smallest normal, 0.25, 1, the next `float` above 1, 1.5, +-inf
/// and a quiet NaN) each stored by `replace_region` into a 4x4 render target's `Depth32Float`,
/// read back by `get_bytes`, and tested by a draw at depth 0, 0.5 and 1 under `Less`,
/// `LessEqual`, `Equal`, `GreaterEqual` and `Greater`: MoltenVK 1.4.1 on the M3 Max under macOS
/// 27.0.1, of the M4's identity ([`APPLE_M4`]): every value kept, and 270 of 270 outcomes Metal's
/// magnitude rule (225 of them an IEEE compare's), as Metal's own probe `depthkeep` on the same GPU
/// answers. Taken on every Apple GPU from 1.4.1 on, by construction (the fact's docs): through
/// MoltenVK a GPU's depth test is that GPU's Metal test, where the stand-in would impose the M3
/// Max's rule. Its pass with depth writes over 16 such values (the runtime lane `depth_denormals`'
/// merge, a draw at 0.5 under `Less`) stores each texel as Metal's own pass on the same GPU does
/// (probe `depthstore`): the draw's depth where it passed, the others without their signs. A pass
/// that sets such a state and draws under none (no draw, a draw of no instances or no vertices, or
/// one under a state without depth writes set over it) keeps every bit there, where Metal's strips
/// them; an indirect draw whose arguments draw no instances strips them as Metal's does. Every value
/// that tells the rule from Vulkan's is negative or a NaN, which a copy may not store without
/// `VK_EXT_depth_range_unrestricted`, so no check reaches it on MoltenVK; Vulkan's comparison is of
/// the stored value as it is, so no other driver is listed.
///
/// `FirstSparseResourceWaitsForQueues`, measured by
/// `runtime_lane::first_use::neutrals_and_sparse_textures_wait_for_no_gpu_work` and a plain
/// Vulkan program (2026-10-10): ANV, Mesa 26.2.4 on the Iris Xe (RPL-P) under `i915`, made a
/// 256x256 sparse image in 567 ms behind 0.5 s of copies, and a second in 0.02 ms behind the same
/// (an Iris Xe (RPL-P) host); gdb put the lane's wait in `vkCreateImage`, at `drmSyncobjWait`. Mesa
/// 26.2.4's `anv_trtt_first_bind_init` submits to each queue and waits, the first time a sparse
/// resource is made; under `xe`, ANV binds sparse memory through VM_BIND, whose creation path makes
/// a synchronous ioctl and waits for no queue (read, not measured). Taken on the Intel venue's two
/// devices (both under `i915`) from 26.2.4 on.
///
/// `TimestampPoolsAreCounterSampleBuffers` (the probe): MoltenVK 1.4.1 on the M3 Max and the M4 made
/// 32 timestamp query pools and failed the 33rd, and failed a pool of 8192 timestamps, with
/// `VK_ERROR_OUT_OF_DEVICE_MEMORY`; its source makes each pool one Metal counter sample buffer of
/// the pool's count (`MVKGPUCounterQueryPool::initMTLCounterSampleBuffer`). A defect held at every
/// release and on every GPU: a check would take counter sample buffers other devices of the GPU
/// need.
///
/// `DontCareStoreIsOneOfMetals`: measured by `runtime_lane::dont_care_stores`, every texel of each
/// of its twelve kept cases the earlier one or the stored one, none other: at 66ffe6775, RADV, Mesa
/// 26.2.4, on the Radeon 780M and ANV, Mesa 26.2.4, on the UHD 770 (RPL-S); at dd718a7905, RADV,
/// Mesa 26.0.8, on the RX 7900 XTX, and with its row ANV, Mesa 26.2.4, on the Iris Xe (RPL-P). What
/// a pass leaves follows the GPU's caches, so one run does not prove it and each GPU is taken from
/// the release it was measured on. Not listed: NVIDIA's 615.71.09, whose RTX 3080 left a third
/// answer; and every GPU not measured, which stores with `STORE`.
///
/// `ShadowCopyBackDispatchFaster`, `ShadowCopyInDispatchFaster`, `RowCopiesDispatchFaster`,
/// `SemaphoreOperationsStallSubmits`, `SampledDepthKeepsHiZ` and
/// `MultisampledDepthFasterOutOfGeneral`: cost facts measured on the GPUs and release their docs
/// name, each taken on those GPUs from that release on. No boot can time them: their margins are a
/// few percent on a busy GPU whose clock varies fivefold (the Iris Xe), or a stall once in about a
/// thousand operations.
///
/// `LinearTilingFeaturesUninitialized`, read in MoltenVK 1.4.1's source (`MVKDevice.h` declares
/// `_pixelFormats` before `_metalFeatures`; `MVKPhysicalDevice`'s constructor initializes
/// `_pixelFormats(this)`, whose constructor builds every format's properties through
/// `MVKPixelFormats::setFormatProperties`, before its body calls `initMetalFeatures()`, which
/// clears and then sets `renderLinearTextures`) and measured on the M3 Max (macOS 27.0.1,
/// 2026-10-15): of 40 instances made one after another in one process, 38 reported
/// `R8G8B8A8_UNORM`'s `linearTilingFeatures` as `0x8000dd83` (with `COLOR_ATTACHMENT`,
/// `COLOR_ATTACHMENT_BLEND` and `BLIT_DST`) and 2 as `0x8000d403` (without them), and
/// `vkGetPhysicalDeviceImageFormatProperties` of a linear `COLOR_ATTACHMENT` image succeeded in
/// 38. MoltenVK's code on every GPU, so a defect held at every release and on every GPU.
///
/// `TileStoreIsMetals`, measured by
/// `runtime_lane::snorm_tile_store::a_drawing_pass_keeps_a_snorm_minimum_code_and_says_so` (a
/// 4x4 target of each of the six signed normalized colour formats holding a channel's minimum code
/// in every channel, loaded and drawn over by a full-screen triangle that writes no channel):
/// MoltenVK 1.4.1 on the M3 Max under macOS 27.0.1 ([`APPLE_M4`]'s identity) stored the code
/// above it in every texel, as plain Metal's pass does on that GPU. Its `RGB9E5Float` twin
/// (`a_drawing_pass_keeps_an_unnormalized_shared_exponent_word_and_says_so`: the word `1 | 2 << 9
/// | 3 << 18 | 20 << 27` in every texel) is Metal's by the same construction: MoltenVK renders
/// `E5B9G9R9_UFLOAT_PACK32` as a Metal `RGB9E5Float` attachment, whose pass plain Metal measured
/// normalizing such a word on that GPU. Taken on every Apple GPU from 1.4.1 on, by construction
/// (the fact's docs). Vulkan's load and store keep the bytes, so no other driver is listed.
/// Measured not to have it (every texel kept its code): RADV, Mesa 26.0.8, on the Radeon 780M and
/// the RX 7900 XTX; ANV, Mesa 26.2.4, on the Iris Xe (RPL-P).
///
/// `PointerImportsFollowTheirAddress`: ANV, Mesa 26.2.4 on the Iris Xe of RPL-P under Linux 7.2.9's
/// i915, measured with a probe outside the runtime (2026-10-17): a page of a `memfd` filled with
/// `0xaa`, mapped shared at an address and imported by its pointer as a buffer's memory, read
/// `0xaa` through a copy; another `memfd`'s page of `0xbb` mapped over the address with `MAP_FIXED`,
/// the next copy read `0xbb`; a page of `0xa2` imported and replaced by one of `0xb2` before any
/// submission, the first copy read `0xb2`. ANV imports under `i915` or `xe` by the GPU, so it is
/// taken on the Iris Xe alone. RADV, Mesa 26.2.4 on the Radeon 780M under Linux 7.2.9's amdgpu,
/// measured through the runtime (2026-10-17): a no-copy buffer over two fresh anonymous private
/// pages, read by a blit; fresh anonymous pages mapped over the address with `MAP_FIXED` and
/// `Device::host_range_remapped` told, the next blit read the new pages, and the newer ones after a
/// second remap; remapped between the buffer's making and its first submission, the first blit
/// read the new pages; and Mesa 26.0.8 on the RX 7900 XTX and the Radeon 780M under Linux 7.0.0 the
/// same. Taken on every RADV GPU from 26.0.8 on: RADV imports a pointer as an amdgpu userptr object,
/// whose pages the kernel gets again in each command submission once a remap invalidated them, on
/// every GPU alike (`amdgpu_cs_parser_bos`). lavapipe from Mesa 26.0.8 on by construction, read in
/// 26.0.8's and 26.2.4's `lvp_AllocateMemory`, which binds an import's buffers at the host pointer.
/// A remap under a live import sits beside the device loss of an import over pages since unmapped
/// (ANV's execbuffer fails), so no check runs it. `tests::pointer_residency` in the runtime checks
/// it on a device with it. Every other driver keeps its pointer imports.
///
/// `InterlockOrdersDraws`, measured with no barrier between such draws on any driver (a scratch
/// build of the commit that adds it), by `runtime_lane::interlocked_draw_order` (passes of 84
/// draws, fullscreen and held to bands, of two read-modify-writes of an `R32Uint` color that do
/// not commute, at one sample over 1920x1080 and 64x36 and at four over 960x540 and 64x36, every
/// texel read back) and by `runtime_lane::extended_range_blend_table`'s
/// `a_memoryless_color_2_blends_as_a_stored_color_does` (99 banded blends into a memoryless
/// extended-range color in the interlock, then a fetching copy of it): ANV, Mesa 26.2.4 on the
/// Iris Xe (RPL-P), every texel of every pass right in three runs of the first, every band right
/// in the second, and the `extended_range_blend_table` and `framebuffer_fetch` lanes passing.
/// Measured not to have it: RADV, Mesa 26.2.4 on the Radeon 780M, which passes the first but
/// leaves 19 of the second's 495 bands wrong. An order one run does not prove; taken on the Iris
/// Xe from 26.2.4 on. Every other driver takes the barrier.
pub(crate) const LISTED_FACTS: &[Listed<DriverFact>] = &[
    // NVIDIA's own code, on the GPUs measured.
    entry(
        DriverFact::DepthImageKeepsDenormals,
        DriverId::NVIDIA_PROPRIETARY,
        Some(NVIDIA_VENDOR_ID),
        Gpus::Only(NVIDIA_MEASURED),
        Releases::since(nvidia(615, 71, 9)),
        seen(nvidia(615, 71, 9), NVIDIA_MEASURED),
    ),
    // lavapipe: every GPU, its image the process's memory.
    entry(
        DriverFact::DepthImageKeepsDenormals,
        DriverId::MESA_LLVMPIPE,
        None,
        Gpus::Any,
        Releases::since(mesa(26, 0, 8)),
        seen(mesa(26, 2, 4), LAVAPIPE),
    ),
    // MoltenVK hands the image to Metal as its own `Depth32Float`: every GPU.
    entry(
        DriverFact::DepthImageKeepsDenormals,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::since(moltenvk(1, 4, 1)),
        seen(moltenvk(1, 4, 1), APPLE_MEASURED),
    ),
    // As the denormals: Metal's own `Depth32Float`, every GPU.
    entry(
        DriverFact::DepthImageKeepsValuesOutsideUnitRange,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::since(moltenvk(1, 4, 1)),
        seen(moltenvk(1, 4, 1), APPLE_MEASURED),
    ),
    // A defect of MoltenVK's own code, every GPU and release.
    entry(
        DriverFact::DepthStencilImagesLieApart,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // A defect of ANV's compression, every GPU and release.
    entry(
        DriverFact::CompressedImagesHideAliasedWrites,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Any,
        Releases::EVERY,
        seen(mesa(26, 2, 4), INTEL_VENUE),
    ),
    // The GPUs measured.
    entry(
        DriverFact::HostTransferImagesKeepTheirBytes,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(INTEL_VENUE),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), INTEL_VENUE),
    ),
    // MoltenVK's own sizing of the image, every GPU and release.
    entry(
        DriverFact::DepthStencilImagesStayGpuOptimized,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), APPLE_MEASURED),
    ),
    // MoltenVK's own message: every GPU.
    entry(
        DriverFact::DynamicRestartWarns,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::since(moltenvk(1, 4, 1)),
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // MoltenVK's own fill pipeline per pool: every GPU.
    entry(
        DriverFact::FillCompiledPerCommandPool,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::since(moltenvk(1, 4, 1)),
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // What Metal keeps is its GPU's tiler's: the family measured.
    entry(
        DriverFact::DontCareStoreIsMetals,
        DriverId::MOLTENVK,
        Some(APPLE_VENDOR_ID),
        Gpus::MoltenVkFamily(APPLE9),
        Releases::since(moltenvk(1, 4, 1)),
        seen(moltenvk(1, 4, 1), APPLE_MEASURED),
    ),
    // MoltenVK's own sampling, on every Apple GPU (each samples at a pass's boundaries alone).
    entry(
        DriverFact::TimestampsWaitForEveryQueue,
        DriverId::MOLTENVK,
        Some(APPLE_VENDOR_ID),
        Gpus::Any,
        Releases::since(moltenvk(1, 4, 1)),
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // A defect the stand-in works around exactly, every GPU and release.
    entry(
        DriverFact::PositionedDepthReadAtStandardPositions,
        DriverId::NVIDIA_PROPRIETARY,
        Some(NVIDIA_VENDOR_ID),
        Gpus::Any,
        Releases::EVERY,
        seen(nvidia(615, 71, 9), &[NVIDIA_RTX_3090]),
    ),
    // MoltenVK hands the depth test to Metal: every Apple GPU.
    entry(
        DriverFact::DepthTestComparesMagnitudes,
        DriverId::MOLTENVK,
        Some(APPLE_VENDOR_ID),
        Gpus::Any,
        Releases::since(moltenvk(1, 4, 1)),
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // The Intel venue's devices, both under `i915`.
    entry(
        DriverFact::FirstSparseResourceWaitsForQueues,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(INTEL_VENUE),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), &[INTEL_IRIS_XE_RPL_P]),
    ),
    // A defect of MoltenVK's own pools, every GPU and release.
    entry(
        DriverFact::TimestampPoolsAreCounterSampleBuffers,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), APPLE_MEASURED),
    ),
    // Each GPU measured, from the release it was measured on.
    entry(
        DriverFact::DontCareStoreIsOneOfMetals,
        DriverId::MESA_RADV,
        Some(AMD_VENDOR_ID),
        Gpus::Only(&[AMD_RADEON_780M]),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), &[AMD_RADEON_780M]),
    ),
    entry(
        DriverFact::DontCareStoreIsOneOfMetals,
        DriverId::MESA_RADV,
        Some(AMD_VENDOR_ID),
        Gpus::Only(&[AMD_RX_7900_XTX]),
        Releases::since(mesa(26, 0, 8)),
        seen(mesa(26, 0, 8), &[AMD_RX_7900_XTX]),
    ),
    entry(
        DriverFact::DontCareStoreIsOneOfMetals,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(INTEL_VENUE),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), INTEL_VENUE),
    ),
    // The GPUs measured.
    entry(
        DriverFact::ShadowCopyBackDispatchFaster,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(INTEL_VENUE),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), INTEL_VENUE),
    ),
    // The GPU measured faster (the Iris Xe was not).
    entry(
        DriverFact::ShadowCopyInDispatchFaster,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(&[INTEL_UHD_770_RPL_S]),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), &[INTEL_UHD_770_RPL_S]),
    ),
    // The GPUs measured.
    entry(
        DriverFact::RowCopiesDispatchFaster,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(INTEL_VENUE),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), INTEL_VENUE),
    ),
    entry(
        DriverFact::RowCopiesDispatchFaster,
        DriverId::NVIDIA_PROPRIETARY,
        Some(NVIDIA_VENDOR_ID),
        Gpus::Only(&[NVIDIA_RTX_5080_LAPTOP]),
        Releases::since(nvidia(615, 71, 9)),
        seen(nvidia(615, 71, 9), &[NVIDIA_RTX_5080_LAPTOP]),
    ),
    // A defect of MoltenVK's own code, every GPU and release.
    entry(
        DriverFact::LinearTilingFeaturesUninitialized,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // The GPUs measured.
    entry(
        DriverFact::SemaphoreOperationsStallSubmits,
        DriverId::NVIDIA_PROPRIETARY,
        Some(NVIDIA_VENDOR_ID),
        Gpus::Only(NVIDIA_MEASURED),
        Releases::since(nvidia(615, 71, 9)),
        seen(nvidia(615, 71, 9), NVIDIA_MEASURED),
    ),
    // The GPU measured.
    entry(
        DriverFact::SampledDepthKeepsHiZ,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(&[INTEL_IRIS_XE_RPL_P]),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), &[INTEL_IRIS_XE_RPL_P]),
    ),
    entry(
        DriverFact::MultisampledDepthFasterOutOfGeneral,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(&[INTEL_IRIS_XE_RPL_P]),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), &[INTEL_IRIS_XE_RPL_P]),
    ),
    // MoltenVK renders through Metal's own pass: every Apple GPU.
    entry(
        DriverFact::TileStoreIsMetals,
        DriverId::MOLTENVK,
        Some(APPLE_VENDOR_ID),
        Gpus::Any,
        Releases::since(moltenvk(1, 4, 1)),
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // ANV imports under `i915` or `xe` by the GPU: the GPU measured.
    entry(
        DriverFact::PointerImportsFollowTheirAddress,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(&[INTEL_IRIS_XE_RPL_P]),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), &[INTEL_IRIS_XE_RPL_P]),
    ),
    // An amdgpu userptr object on every GPU: every GPU.
    entry(
        DriverFact::PointerImportsFollowTheirAddress,
        DriverId::MESA_RADV,
        Some(AMD_VENDOR_ID),
        Gpus::Any,
        Releases::since(mesa(26, 0, 8)),
        seen(mesa(26, 2, 4), &[AMD_RADEON_780M, AMD_RX_7900_XTX]),
    ),
    // lavapipe reads through the pointer itself: every CPU.
    entry(
        DriverFact::PointerImportsFollowTheirAddress,
        DriverId::MESA_LLVMPIPE,
        None,
        Gpus::Any,
        Releases::since(mesa(26, 0, 8)),
        seen(mesa(26, 2, 4), LAVAPIPE),
    ),
    // The GPU measured.
    entry(
        DriverFact::InterlockOrdersDraws,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        Some(INTEL_VENDOR_ID),
        Gpus::Only(&[INTEL_IRIS_XE_RPL_P]),
        Releases::since(mesa(26, 2, 4)),
        seen(mesa(26, 2, 4), &[INTEL_IRIS_XE_RPL_P]),
    ),
];

/// Every listed runtime driver defect ([`RuntimeDriverDefect`]), by defect: what the runtime
/// reads through [`DeviceDescription::listed_driver_defects`]. What each entry rests on:
///
/// `MultisampledDepthStencilColourCopies`, MoltenVK (`MOLTENVK`): `MVKCmdCopyImage::encode` copies
/// between a depth/stencil and a colour format through a temporary buffer sized per texel with no
/// sample count, and Metal blits no multisampled texture to or from a buffer, so samples land
/// wrong. Read in 1.4.1's source and measured on 1.4.1 (driverVersion 10401): every release, as
/// the copy asks Metal for a blit it does not allow (lavapipe's, in the runtime's table, is a check's).
///
/// `MultisampledStencilColourCopies`, NVIDIA's (`NVIDIA_PROPRIETARY`): on the RTX 3090 (`deviceID`
/// 0x2204, Ampere's GA10x), `vkCmdCopyImage` between the stencil aspect of a 4-sample
/// `D32_SFLOAT_S8_UINT` image and a 4-sample 8-bit colour image raises a graphics exception (Xid
/// 13, "WIDTH CT Violation" or "3D HEIGHT CT Violation") and loses the device: into the stencil at
/// 4x1, 4x2 and 1x4, out of it at 1x4, whether the colour image is a colour attachment, a storage
/// image or both, while 2x2, the depth aspect, and an `S8_UINT` image copy. Valid usage: the
/// validation layer reports nothing. Measured on 615.71.09 with a bare-Vulkan program (the probe),
/// no release known to fix it. The RTX 5080 Laptop GPU (Blackwell, 0x2c19) on the same release
/// copies them (measured clean), a clean identity at that release alone: every other NVIDIA GPU and
/// release, the RTX 5080 Laptop GPU at another release among them, takes the exact path until a
/// probe finds it clean.
///
/// `ImageSizeShortOfCompressionData`, Mesa's ANV (`INTEL_OPEN_SOURCE_MESA`) on Intel's Xe-LP GPUs:
/// a compressed image's compression data lies last in its memory, at a multiple of 4096 bytes,
/// `size / 256` bytes long (`add_compression_control_buffer` in `anv_image.c`), and the image's
/// size ends there, while the aux map gives each 64-KiB block of texels a 256-byte chunk of it
/// (Mesa's `intel_aux_map.h`) and the GPU writes it back in 64-byte lines. A 4x4 `R8G8B8A8_UNORM`
/// image asks 4112 bytes at 65536; a 1D image bound at byte 4112, written after it in the same
/// command buffer, read its old contents back at bytes 4112 to 4159, while one bound at 4160 or
/// 4352 kept what was written. Measured on 26.2.4 (driverVersion 0x6802004) on Raptor Lake-P's Iris
/// Xe (`deviceID` 0xa7a0; `runtime_lane::texture_blits`, and a bare-Vulkan program) and Raptor
/// Lake-S's UHD 770 (0xa780); read in main's source of 2026-10-07. A stale write-back one clean run
/// does not rule out, so held at every release and on every GPU: the runtime rounds each image's
/// size up to whole 256-byte chunks there (`texture::ImagePlacement` in the runtime), at most 255
/// bytes an image, which a heap's size answers show. Mesa's source pads the data to the map's own
/// granules on Xe-LPG (Meteor Lake, Arrow Lake) and keeps none in memory on Xe-HPG and Xe2, but
/// none of those is measured.
///
/// `PipelineCacheDataCompiledAtLoad`, MoltenVK: `vkCreatePipelineCache` with initial data reads
/// each shader module's converted MSL back (`MVKPipelineCache::readData`) through
/// `MVKShaderLibraryCache::addShaderLibrary`, whose `MVKShaderLibrary` constructor decompresses the
/// MSL and calls `compileLibrary` at once, waiting for Metal's compile: every library in the data
/// is compiled before the call returns, whether a pipeline of the process uses it or not. Read in
/// 1.4.1's source (`MVKShaderModule.mm`); seen by the Vulkan consumer on an M3 Max (1.4.1,
/// driverVersion 10401) as `MTLCompilerService` rebuilding about 115 libraries at every device
/// creation from an 8.9 MB entry of 131 shaders, about 28 s for one process. Nothing branches on
/// it: each pipeline making's cache is seeded from that making's own record
/// (`shader_cache/driver.rs` in the runtime), so a load compiles only the making's own stages,
/// which it converts anyway without one.
///
/// `LazilyAllocatedVolumePlaneViews`, MoltenVK: `MVKImageViewPlane::newMTLTexture` makes a 2D view
/// of a 3D image (the image `VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT`) as a 2D array texture over
/// the image's memory through its placement heap (`getHeapAllocation`), with no check that there is
/// one; `MVKDeviceMemory::ensureMTLHeap` makes none for memory of Metal's memoryless storage, which
/// is what a lazily allocated memory type is, so the view's first command buffer dereferences a
/// null heap and the process takes a `SIGSEGV`
/// (`runtime_lane::memoryless::a_memoryless_3d_depth_plane_renders` on an M4, 1.4.1, driverVersion
/// 10401, where the other types' views pass: they view the memoryless texture itself). Read in
/// 1.4.1's source and main's of 2026-10-05 (`fae55a18`, unchanged). Nothing is refused: such an
/// image takes device-local memory (`ImagePlan::memory_class` in the runtime).
///
/// `CompressedImageCopiesRewriteTexels`, MoltenVK: a copy between a buffer and an image is a Metal
/// blit of the image's Metal texture, which Metal keeps losslessly compressed where its usage
/// allows (`texture/heap_tiled.rs`'s `metal_compresses` in the runtime; an image of neither
/// `STORAGE` usage nor `MUTABLE_FORMAT`), and Metal's blits of a compressed `RG11B10Float` or
/// `RGB9E5Float` texture write other bytes than they read (`bind::CompressedBlit`), so a CPU copy
/// of a `Shared` texture of either format reads back NaNs made quiet and shared exponents
/// renormalized, where Metal's `replaceRegion` and `getBytes` keep the bytes. Measured on 1.4.1
/// (driverVersion 10401) on an M4 under macOS 27.0.1 (`runtime_lane::compressed_blits`). The
/// compression is Metal's, so no MoltenVK release is expected to fix it, but a probe of a later one
/// decides. Nothing is refused: the image of such a texture a CPU copy reaches, and of a device's
/// own one without `allowGPUOptimizedContents` (whose GPU blits keep the bytes on Metal), is made of
/// `PixelFormatView` usage (`compressed_blit::copy_exact_usage` in the runtime), which Metal keeps
/// uncompressed, and the runtime rewrites what its blits write itself (`compressed_blit.rs`). A
/// heap's such texture answers the larger of the two images' sizes, so it is listed, not checked.
///
/// `ReadsCommandBuffersAfterTheirSignal`, MoltenVK: on a device with `MTLSharedEvent` the GPU
/// signals a timeline semaphore (`encodeSignalEvent`), so the host sees the value before the Metal
/// command buffer's completed handlers run, and `MVKCommandEncoder::finishQueries` adds one that
/// calls into every query pool the command buffer reset or began, through raw pointers it does not
/// retain. A command buffer released at the value crashed there (`SIGSEGV` at that handler's load
/// of a released `MVKQueryPool`'s vtable;
/// `runtime_lane::visibility::command_buffers_released_at_completion_leave_nothing_read_later`
/// passed 0 of 8 runs on an M4). MoltenVK signals a fence from the last of those handlers. Read in
/// 1.4.1's source and measured on 1.4.1 (driverVersion 10401). Nothing is refused: a command
/// buffer's last batch signals a fence there too, and the command buffer finishes once both are
/// signaled (`command_buffer/fences.rs` in the runtime). Every other driver ends a command buffer
/// at its value alone: Vulkan makes the value enough, and their fences are signaled with it, so a
/// fence there only added a `vkCreateFence` to the commit path, which NVIDIA's 615.71.09 held for
/// up to 1.9 s while other processes' work ran.
///
/// `ImplicitPreprocessAfterExecutionSet`, Mesa's RADV (`MESA_RADV`):
/// `vkCmdExecuteGeneratedCommandsEXT` of a layout not made for explicit preprocessing dispatches the
/// layout's preprocessing pipeline itself (`radv_prepare_dgc`), and `radv_emit_compute_pipeline`
/// skips a pipeline equal to `emitted_compute_pipeline`. The execution's own commands then bind
/// each sequence's pipeline of the execution set on the GPU, which that field does not see, so the
/// next execution's preprocessing in the command buffer, with nothing dispatched between, runs the
/// shader the execution set bound last with the preprocessing's arguments. A range of two commands,
/// an execution each, faulted (`[gfxhub] page fault`, client `TCP`, permission faults) and timed
/// out the ring (`runtime_lane::generated_executions`, both of its tests that execute, on every
/// run), where explicit preprocessing runs them right. Read in 25.0.0's, 26.0.8's and 26.1.5's
/// source (the field is tracked alike); 26.2.0 tracks a dirty bit instead, which the
/// preprocessing's own bind sets. Measured on 26.0.8 (driverVersion 0x6800008) on the Radeon 780M
/// (`deviceID` 0x15bf) and the RX 7900 XTX (0x744c), and measured clean on 26.2.4 (driverVersion
/// 0x6802004) on both: every RADV GPU before 26.2.4 takes it. Nothing is refused: the runtime
/// preprocesses every execution of the range apart, before the first runs
/// (`generated_execute.rs` in the runtime).
///
/// `ExplicitPlaneRowsPastHeight`, Mesa's lavapipe (`MESA_LLVMPIPE`): a `DRM_FORMAT_MOD_LINEAR`
/// image made for host allocations with an explicit plane at a texture's packed rows, reported at
/// `rows × rowPitch` bytes in its subresource and within its memory requirements, is rendered in
/// blocks of 4 rows: a pass that loads, blends and stores a 658x126 `RGBA16Float` tile (rows of 5264
/// bytes) rewrote rows 126 and 127 past it (signaling NaNs quieted), and one over a 1308x510
/// `BGRA8Unorm` tile (rows of 5232 bytes) wrote rows 510 and 511 past the no-copy buffer's mapping,
/// and the process took a `SIGSEGV` (`runtime_lane::pitched_surfaces`, packed odd-height layouts).
/// Its own linear images report the rounded rows in their subresource. Measured on 26.2.4
/// (llvmpipe, LLVM 23.1.1, 256 bits; gem10); 26.0.8 unprobed, so every release. Nothing is refused:
/// no such image of a height that is no multiple of 4 is made there (`dma_buf.rs`'s
/// `explicit_plane_rows` in the runtime), and the texture takes its shadow.
///
/// `FreedAddressMemoryFailsPendingWork`, MoltenVK: every encoder whose pipeline reads device
/// addresses makes every buffer of `SHADER_DEVICE_ADDRESS` usage resident
/// (`MVKDevice::encodeGPUAddressableBuffers`), in Metal command buffers that retain none of it
/// (`retainedReferences` off where the device keeps its residency set), and `vkFreeMemory` lets go
/// of the memory's Metal objects at once: a Metal command buffer encoded before and not yet run
/// fails `Invalid Resource` (Metal's code 9), its device then lost. Destroying the buffer alone,
/// its memory kept, fails nothing. Read in 1.4.1's source, and measured on 1.4.1 (driverVersion
/// 10401) on an M3 Max with macOS 27.0.1: a bare-Vulkan program freeing an unused buffer's memory
/// while a dispatch through a device address waits behind fills fails at once, and
/// `runtime_lane::freed_memory`. Nothing is refused: the runtime frees such memory once the batches
/// submitted before it completed (`freed_memory.rs` in the runtime), the program's own submissions
/// through `CommandQueue::with_vk_queue` among them (a mark after each use signals a timeline of
/// its own).
pub(crate) const LISTED_DEFECTS: &[Listed<RuntimeDriverDefect>] = &[
    entry(
        RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // Every NVIDIA GPU but the RTX 5080 Laptop GPU, every release.
    entry(
        RuntimeDriverDefect::MultisampledStencilColourCopies,
        DriverId::NVIDIA_PROPRIETARY,
        None,
        Gpus::Except(&[NVIDIA_RTX_5080_LAPTOP]),
        Releases::EVERY,
        seen(nvidia(615, 71, 9), &[NVIDIA_RTX_3090]),
    ),
    // The RTX 5080 Laptop GPU at every release but 615.71.09, its clean identity.
    entry(
        RuntimeDriverDefect::MultisampledStencilColourCopies,
        DriverId::NVIDIA_PROPRIETARY,
        None,
        Gpus::Only(&[NVIDIA_RTX_5080_LAPTOP]),
        Releases::before(nvidia(615, 71, 9)),
        seen(nvidia(615, 71, 9), &[NVIDIA_RTX_3090]),
    ),
    entry(
        RuntimeDriverDefect::MultisampledStencilColourCopies,
        DriverId::NVIDIA_PROPRIETARY,
        None,
        Gpus::Only(&[NVIDIA_RTX_5080_LAPTOP]),
        Releases::since(nvidia(615, 71, 9) + 1),
        seen(nvidia(615, 71, 9), &[NVIDIA_RTX_3090]),
    ),
    entry(
        RuntimeDriverDefect::ImageSizeShortOfCompressionData,
        DriverId::INTEL_OPEN_SOURCE_MESA,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(mesa(26, 2, 4), INTEL_VENUE),
    ),
    entry(
        RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    entry(
        RuntimeDriverDefect::PipelineCacheDataCompiledAtLoad,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), &[APPLE_M3_MAX]),
    ),
    entry(
        RuntimeDriverDefect::CompressedImageCopiesRewriteTexels,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    entry(
        RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
    // Measured broken on 26.0.8 and clean on 26.2.4 on both GPUs.
    entry(
        RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet,
        DriverId::MESA_RADV,
        None,
        Gpus::Any,
        Releases::before(mesa(26, 2, 4)),
        seen(mesa(26, 0, 8), &[AMD_RADEON_780M, AMD_RX_7900_XTX]),
    ),
    entry(
        RuntimeDriverDefect::ExplicitPlaneRowsPastHeight,
        DriverId::MESA_LLVMPIPE,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(mesa(26, 2, 4), LAVAPIPE),
    ),
    entry(
        RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork,
        DriverId::MOLTENVK,
        None,
        Gpus::Any,
        Releases::EVERY,
        seen(moltenvk(1, 4, 1), &[APPLE_M4]),
    ),
];
