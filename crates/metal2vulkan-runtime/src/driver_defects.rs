//! The defects of a device's driver in commands the runtime records for itself, keyed on the
//! driver's identity, release and GPU: `VkPhysicalDeviceDriverProperties`'s `driverID`, and
//! `VkPhysicalDeviceProperties`'s `driverVersion`, which each driver raises with every release
//! (Mesa encodes it as `VK_MAKE_API_VERSION`, MoltenVK as `major * 10000 + minor * 100 + patch`),
//! and `deviceID`. A device's defects are in its properties
//! ([`DeviceProperties::driver_defects`](super::DeviceProperties::driver_defects)), and what a
//! defect breaks is taken another way where an exact one exists, else refused there, typed
//! ([`Needs::RuntimeDriverDefect`]), never run wrong unannounced: only where a refusal would
//! withhold a feature of the families the device claims (a row of the feature table,
//! `families/features.rs`) does it run, announced as a degradation.
//!
//! A device's defects come from two places. The known-driver list
//! ([`DeviceDescription::listed_driver_defects`]) names the defects that can lose the device,
//! crash the process or hang the GPU, and those one clean run does not rule out, by driver,
//! range of releases and GPU family: open on the affected side until a release is known clean. The
//! table here holds the rest, those a check on the device may settle: each holds on every GPU of
//! its driver, at every release or at the releases its evidence names.
//!
//! None is a translation's (`DriverWorkaround` covers those), so none is in the compile profile,
//! a cache key or a digest.
//!
//! [`Needs::RuntimeDriverDefect`]: metal2vulkan::error::Needs::RuntimeDriverDefect

use metal2vulkan::device::{
    decide, release_name, Baseline, DeviceChecks, DeviceDescription, DeviceFact, FactStatus,
    CHECKED,
};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::DriverId;
use std::collections::{BTreeMap, BTreeSet};

/// One driver that has a defect, on every GPU, at `releases`.
pub(super) struct Affected {
    pub(super) defect: RuntimeDriverDefect,
    pub(super) driver: DriverId,
    pub(super) releases: Releases,
}

/// The releases (`driverVersion`) a row holds at: from `from` up to, but not including, `until`,
/// the first release measured without it (`None`: none is).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Releases {
    pub(super) from: u32,
    pub(super) until: Option<u32>,
}

impl Releases {
    /// Every release of the driver.
    pub(super) const EVERY: Releases = Releases {
        from: 0,
        until: None,
    };

    /// Whether the release `release` is one of these.
    pub(super) fn contains(self, release: u32) -> bool {
        release >= self.from && self.until.is_none_or(|until| release < until)
    }
}

/// Mesa's `driverVersion` of release `major.minor.patch` (`VK_MAKE_API_VERSION`).
const fn mesa(major: u32, minor: u32, patch: u32) -> u32 {
    (major << 22) | (minor << 12) | patch
}

/// The defects a check on the device may settle, by driver: the baseline a device takes until
/// its checks answer, on every GPU of the driver, at every release but where a row's evidence
/// bounds its releases. The defects no check settles are on the known-driver list
/// ([`DeviceDescription::listed_driver_defects`]).
///
/// `MultisampledDepthStencilColourCopies`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): `handle_copy_image` in `lvp_execute.c` copies between a
///   depth/stencil and a colour format of different texel sizes (a combined `D32_SFLOAT_S8_UINT`
///   image and `R32_SFLOAT` or `R8_UNORM`) through `texture_map`, which maps sample 0 only, so the
///   other samples are never written. Read in Mesa 26.0.8's source and in main's of 2026-10-05,
///   and measured on 26.0.8 (`runtime_lane::separate_samples`, the render census): known broken
///   through 26.2.x. The device checks it itself as it is made, by the depth aspect's copies of a
///   combined image both ways (`device_checks/colour_copies.rs`); this row is its baseline where
///   the check does not answer, and it is not run where the known-driver list names the defect.
///
/// `NormalizedDepthPastUnitRange`:
/// - MoltenVK (`MOLTENVK`): Metal's rasterizer on Apple's GPUs reads and stores the depths of a
///   sloped triangle whose normalized depths reach past 1 about 2e-4 off (0.968567252 for 0.96875),
///   with or without clip distances and in both depth clip modes, where Metal's own viewport of the
///   range draws the same depths exactly. Measured in Metal on an M4 with macOS 27.0.1 (a probe:
///   the vertex stage writing the compensated depth into a viewport of `[0, 1]`, against a native
///   viewport of `[0.5, 2]`), and through the runtime's compensation on 1.4.1 (driverVersion 10401)
///   on an M3 Max and an M4 (`runtime_lane::viewport_ranges`), and in Metal on the M3 Max with
///   macOS 27.0.1 (a probe of the device check's draw: a triangle whose normalized depth runs from
///   0.5 to 2 read 0.8282051 for 0.828125, where one from 0.75 to 1.75 drew every depth exactly):
///   known broken through 1.4.1. The rasterizer is Metal's, so no MoltenVK release is expected to
///   fix it, but a probe of a later one decides; MoltenVK's other GPUs (Intel and AMD Macs) are
///   unprobed and refused alike where the device's own check does not answer.
///
/// `DepthStencilResolveFirstSubresource`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): `resolve_ds` in `lvp_execute.c` blits the depth/stencil
///   attachment's image into the resolve attachment's with the blit's `box.z` and `level` left
///   0, where `resolve_color` sets them from each view's `base_array_layer` and
///   `base_mip_level`, so a resolve from slice 1 of a multisampled array takes slice 0's samples,
///   and one into slice or level 1 writes slice or level 0. Read in 25.2.0's, 26.0.8's, 26.2.4's
///   and main's source of 2026-10-07, and measured on 26.0.8 and 26.2.4
///   (`runtime_lane::layered_passes`, whose slices read back by resolves all held slice 0's):
///   known broken through 26.2.4. Nothing is refused but a memoryless attachment at another
///   layer (`resolve.rs`).
///
/// `MultisampledCopiesConvertFloatNans`:
/// - Mesa's RADV (`MESA_RADV`): `gfx_or_compute_copy_image` in `radv_meta_copy.c` gives the
///   source's blit surface the destination's format wherever the source is not DCC-compressed or
///   the two formats are DCC-compatible (every pair on GFX11, `radv_dcc_formats_compatible`),
///   and a multisampled copy is then a draw (`radv_gfx_copy_image`) that fetches each sample in
///   that format and exports it: a NaN in a 16-bit float channel lands as `0xfe00`, its sign and
///   payload lost, while a single-sampled copy kept them. Read in 26.2's source (2026-10-09), and
///   measured on 26.2.4 (driverVersion 0x6802004) on the Radeon 780M (`deviceID` 0x15bf;
///   `runtime_lane::renders::multisampled_nan_clears_store_metal_s_texel`: every NaN sample of a 4x
///   `R16Float` and `RGBA16Float` NaN clear, copied from its unsigned-integer scratch, held
///   `0xfe00`: a scratch not mutable to the attachment's format alike): known broken through
///   26.2.4. The path is every RADV GPU's, so every device has it. Nothing is refused: the NaN
///   clear of a multisampled attachment is written through a view of its image in the
///   unsigned-integer format of its texel size (`nan_clear.rs`), which the runtime's multisampled
///   color render targets of such formats take there where it costs the image nothing
///   (`nan_clear::view_candidate`), and is the device's copy, announced (`device-nan-encoding`),
///   where the image takes no such view.
///
/// `OcclusionCountPerColorOutput`:
/// - Mesa's ANV (`INTEL_OPEN_SOURCE_MESA`): a precise occlusion query counts each passing sample
///   once per color output the fragment shader writes, where Vulkan's Sample Counting counts it
///   once; write masks of 0, the query's command buffer and linked pipelines do not change it, and
///   an imprecise query reads nonzero. Measured on 26.2.4 (driverVersion 0x6802004) on the Iris
///   Xe (`deviceID` 0xa7a0; `runtime_lane::color_groups`): known broken through 26.2.4. Nothing
///   is refused, since a refusal would withhold the Mac2 and Metal3 families: the draw runs,
///   announced (`Degradation::VisibilityCountPerColor`).
///
/// `PushedBufferSizesUnset`:
/// - MoltenVK (`MOLTENVK`): a shader's runtime array length of a buffer (`OpArrayLength`) reads
///   the buffer's size from the stage's implicit buffer of sizes, which
///   `MVKCommandEncoderState::bindDescriptorSets` fills for a set of discrete descriptors (a push
///   descriptor set layout is one: `pickArgumentBufferMode` takes no argument buffer for it) and
///   `MVKCommandEncoderState::pushDescriptorSet` never fills: it writes the pushed descriptors
///   alone. A pushed buffer's length then reads 0, or what an earlier bind left there, and the
///   accesses bounded by it go nowhere: on an M3 Max with macOS 27.0.1, kernels that push set 0
///   read the null entry for every resource id (`runtime_lane::resource_ids`), stored no
///   normalized storage write (`runtime_lane::norm_storage_writes`), and wrote nothing through a
///   device without robustness's in-shader bounds (`runtime_lane::executions`), where binding set
///   0 passes each. Read in 1.4.1's source and measured on 1.4.1 (driverVersion 10401): known
///   broken through 1.4.1. Nothing is refused: the device takes no push command there
///   (`device/mod.rs`), and every draw and dispatch binds set 0 from its command buffer's pools,
///   as on a device without the command.
///
/// `PushedDescriptorsStaleAfterPipelineBinds`:
/// - Mesa's ANV (`INTEL_OPEN_SOURCE_MESA`): draws of one layout that push set 0 alike, a
///   pipeline reading a uniform buffer's constants at fixed offsets (its range starting at the
///   bind offset, `SpecKey::RangeAtBindOffset`) and then one reading it at the pushed residual
///   plus the offset through the same descriptor, which the second draw's push leaves out as
///   unchanged, so that no push follows the bind: the second pipeline reads zeros. Pushing set 0
///   whole after the bind reads the bytes, and so does pushing one write that did not change
///   (set 0's first). Measured on 26.2.4 (driverVersion 0x6802004) on the Iris Xe (`deviceID`
///   0xa7a0; `runtime_lane::unorm_depth_bias`, 3 of 3 runs wrong without the whole push, 3 of 3
///   right with it) and on the UHD 770 (`deviceID` 0xa780; the same lane, 3 of 3 runs wrong with
///   no push after the bind, 3 of 3 right with the one write): known broken through 26.2.4.
///   Nothing is refused: the first draw after each pipeline bind pushes at least one write, those
///   that changed or else set 0's first (`render/bound.rs`).
///
/// `ImmutableSamplersLostAfterCopies`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): a compute dispatch recorded after a copy in the same
///   `VkCommandBuffer` samples a combined image sampler binding whose pipeline layout bakes an
///   immutable sampler as Repeat. A kernel's variant for a 4:2:0 texture, whose layout bakes a
///   clamp-to-edge sampler (`runtime_lane::ycbcr_textures`'s
///   `a_constexpr_sampler_takes_a_ycbcr_map_through_one_variant`), read the texels clamped at its
///   first dispatch, recorded in a `VkCommandBuffer` of its own while its pipeline was being made,
///   and Repeat-addressed at every later one, recorded after the texture's shadow copy: the same
///   pipeline, layout and pushed descriptors (with the immutable sampler pushed or not), and the
///   validation layer reported nothing. Measured on 26.0.8 (driverVersion 0x6800008, Ubuntu's
///   26.0.8-1ubuntu0.3 on LLVM 21.1.8): known broken through 26.0.8. Measured clean on 26.2.4
///   (0x6802004, LLVM 23.1.1), the same calls recorded alike, and by the device's own check
///   (`device_checks/immutable_sampler.rs`, answering Metal's on 26.2.4 on x86-64). Mesa 26.1.0's
///   release notes list "lavapipe: rework immutable samplers", likely the fix (unverified: its
///   change was not read). The row holds from 26.0.8, the first release measured with it (earlier
///   ones are unmeasured), up to 26.2.4, the first measured without it; the device's check decides
///   it on every release, the row being its baseline until the check answers. Nothing is
///   refused: an encoder's dispatch of a variant that bakes samplers records in a
///   `VkCommandBuffer` of its own there (`compute.rs`), as one whose pipeline is still being made
///   does (`pending_dispatch.rs`); a host step's dispatch (an indirect dispatch's, an indirect
///   command buffer's execution) records in place. Graphics draws are unmeasured.
pub(super) const AFFECTED: &[Affected] = &[
    Affected {
        defect: RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
        driver: DriverId::MESA_LLVMPIPE,
        releases: Releases::EVERY,
    },
    Affected {
        defect: RuntimeDriverDefect::NormalizedDepthPastUnitRange,
        driver: DriverId::MOLTENVK,
        releases: Releases::EVERY,
    },
    Affected {
        defect: RuntimeDriverDefect::DepthStencilResolveFirstSubresource,
        driver: DriverId::MESA_LLVMPIPE,
        releases: Releases::EVERY,
    },
    Affected {
        defect: RuntimeDriverDefect::MultisampledCopiesConvertFloatNans,
        driver: DriverId::MESA_RADV,
        releases: Releases::EVERY,
    },
    Affected {
        defect: RuntimeDriverDefect::OcclusionCountPerColorOutput,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        releases: Releases::EVERY,
    },
    Affected {
        defect: RuntimeDriverDefect::PushedBufferSizesUnset,
        driver: DriverId::MOLTENVK,
        releases: Releases::EVERY,
    },
    Affected {
        defect: RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        releases: Releases::EVERY,
    },
    Affected {
        defect: RuntimeDriverDefect::ImmutableSamplersLostAfterCopies,
        driver: DriverId::MESA_LLVMPIPE,
        releases: Releases {
            from: mesa(26, 0, 8),
            until: Some(mesa(26, 2, 4)),
        },
    },
];

/// The defects of the device `description` names: its driver's of [`AFFECTED`] at its release,
/// and those the known-driver list names for its driver, release and GPU.
pub(super) fn of(description: &DeviceDescription) -> BTreeSet<RuntimeDriverDefect> {
    named(description).into_keys().collect()
}

/// The defects of the device `description` names ([`of`]), each with the row of [`AFFECTED`] or
/// the entry of the known-driver list that gives it, as a feature status names it.
pub(super) fn named(description: &DeviceDescription) -> BTreeMap<RuntimeDriverDefect, String> {
    let mut defects = description.listed_driver_defects();
    let affected = (AFFECTED.iter()).filter(|a| {
        Some(a.driver) == description.driver_id && a.releases.contains(description.driver_version)
    });
    for a in affected {
        let row = || match a.releases {
            Releases::EVERY => format!("{:?}, every release and GPU", a.driver),
            Releases { from, until } => {
                let name = |v| release_name(Some(a.driver), v);
                let until = until.map_or_else(String::new, |v| format!(" before {}", name(v)));
                format!("{:?} from {}{until}, every GPU", a.driver, name(from))
            }
        };
        defects.entry(a.defect).or_insert_with(row);
    }
    defects
}

/// The defects the device takes: [`of`]'s, each a check decides as its check found it
/// (`metal2vulkan::device::checks`; the description holds no answer where the policy distrusts
/// checks).
pub(super) fn checked(description: &DeviceDescription) -> BTreeSet<RuntimeDriverDefect> {
    taken(description, &description.checks)
}

/// The defects a device of `description` takes where its checks answered `checks`: [`checked`]'s
/// of a description that holds them, as the device's making reads them while its checks answer
/// one by one (`device_checks/kit.rs`).
pub(super) fn taken(
    description: &DeviceDescription,
    checks: &DeviceChecks,
) -> BTreeSet<RuntimeDriverDefect> {
    rows(description, Some(checks))
        .into_iter()
        .filter(|row| row.holds)
        .filter_map(|row| match row.fact {
            DeviceFact::Defect(defect) => Some(defect),
            _ => None,
        })
        .collect()
}

/// Whether a device whose driver has `vkCmdPushDescriptorSet` takes it, its draws and dispatches
/// pushing set 0 with it, where it takes `defects` ([`checked`], [`taken`]): not where its
/// driver's pushed sets lose their buffers' sizes (`RuntimeDriverDefect::PushedBufferSizesUnset`),
/// where every draw and dispatch binds set 0 instead, as on a device without the command. The
/// device is assembled by it (`device/mod.rs`), and its checks push by it as it is made
/// (`device_checks/kit.rs`).
pub(super) fn takes_push_command(defects: &BTreeSet<RuntimeDriverDefect>) -> bool {
    !defects.contains(&RuntimeDriverDefect::PushedBufferSizesUnset)
}

/// The feature status rows of the runtime's defects on `description`'s setup: each defect [`of`]
/// gives it, listed, each defect a check decides, and each the device learned off, its checks
/// taken where `trusted`.
pub(super) fn status_rows(description: &DeviceDescription, trusted: bool) -> Vec<FactStatus> {
    rows(description, trusted.then_some(&description.checks))
}

/// [`status_rows`] by the checks' answers `checks`, none where they are not taken.
fn rows(description: &DeviceDescription, checks: Option<&DeviceChecks>) -> Vec<FactStatus> {
    let listed = named(description);
    let checked = CHECKED.iter().map(|entry| entry.fact);
    let learned = checks.into_iter().flat_map(|checks| checks.learned_facts());
    let defects: BTreeSet<RuntimeDriverDefect> = (listed.keys().copied())
        .chain(checked.chain(learned).filter_map(|fact| match fact {
            DeviceFact::Defect(defect) => Some(defect),
            _ => None,
        }))
        .collect();
    defects
        .into_iter()
        .map(|defect| {
            let baseline = match listed.get(&defect) {
                Some(entry) => Baseline::listed(entry.clone()),
                None => Baseline::unlisted(),
            };
            decide(DeviceFact::Defect(defect), baseline, checks)
        })
        .collect()
}
