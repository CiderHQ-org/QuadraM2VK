//! MoltenVK's Metal argument buffers.
//!
//! Metal always binds through argument buffers. MoltenVK makes a pooled descriptor set a Metal
//! argument buffer only while its setting `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` is on, the
//! default (`MVKEnvironment.h`); with it off (an environment variable, or a layer setting) every
//! set's resources reach Metal as discrete arguments, and two things change:
//!
//! - Metal's compiler crashes (`MTLCompilerService` stops with `EXC_BAD_ACCESS` in its back end) on
//!   the discrete form of large fragment functions whose argument-buffer form it compiles
//!   (measured on an Apple M3 Max and an Apple M4), so their pipelines fail.
//! - The update-after-bind limits fall to the per-stage ones (128 textures), so argument buffers
//!   hold far fewer resources than Metal's tier 2.
//!
//! So an instance the runtime creates asks MoltenVK for argument buffers through
//! `VK_EXT_layer_settings` where the instance lists it ([`setting`]), ahead of the consumer's own
//! settings: MoltenVK applies layer settings over the environment, and in order, so a consumer's
//! own setting wins. Then at device open, on every path (an adopted device too), a MoltenVK device
//! running without them is refused ([`check`], [`DriverSetting::MoltenVkArgumentBuffersOff`]).
//!
//! The state is read from the device's limits, at each release [`READ`] lists: MoltenVK's
//! `initProperties` (`MVKDevice.mm`) reports `maxPerStageDescriptorUpdateAfterBindSampledImages`
//! as 1,000,000 where `isTier2MetalArgumentBuffers()` (argument buffers in use on a tier-2 GPU),
//! else as `maxPerStageDescriptorSampledImages`; and every Apple GPU a Mac has is tier 2, so on
//! one the per-stage value means argument buffers are off. Measured on 1.4.1 on an Apple M3 Max:
//! 1,000,000 (and 256 per stage) with the setting on, 128 and 128 with it off. 1.4.2's
//! `initProperties`, `isTier2MetalArgumentBuffers` and `initMetalFeatures` are 1.4.1's on these
//! (read in its source), and on an Apple M3 Max it reports 1,000,000 with the setting on and is
//! refused with it off (`runtime_lane::degrade_notices`). Not from
//! `pipelineCacheUUID`, whose argument-buffer bit `initPipelineCacheUUID` writes before
//! `initMetalFeatures` sets the flag, so it is 1 either way (read in 1.4.1's source, and measured).
//! Any other release, or another GPU's vendor, or limits of neither shape, is
//! [`ArgumentBuffers::Unread`], announced once
//! (`metal2vulkan-degrade-moltenvk-argument-buffers-unread`), never refused.

use super::create::{LayerSetting, LayerSettingValue};
use crate::error::{DriverSetting, Error};
use metal2vulkan::device::DeviceDescription;
use metal2vulkan::vk::DriverId;
use std::ffi::CStr;

/// MoltenVK's layer name for its settings (`kMVKMoltenVKDriverLayerName`).
pub(crate) const LAYER: &CStr = c"MoltenVK";

/// MoltenVK's setting that makes descriptor sets Metal argument buffers (`MVKConfigMembers.def`:
/// `useMetalArgumentBuffers`, a `VkBool32`).
pub(crate) const SETTING: &CStr = c"MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS";

/// The MoltenVK releases (`driverVersion`, `major * 10000 + minor * 100 + patch`) whose
/// `initProperties` and `initMetalFeatures` were read for the limits the module docs decode.
pub(crate) const READ: &[u32] = &[10401, 10402];

/// `maxPerStageDescriptorUpdateAfterBindSampledImages` with tier-2 argument buffers in use, at
/// each release [`READ`] lists (`1e6` in `initProperties`).
pub(crate) const TIER_2_UPDATE_AFTER_BIND: u32 = 1_000_000;

/// Apple's PCI vendor ID: MoltenVK's `vendorID` on an Apple GPU, each of which a Mac has is a
/// tier-2 argument buffer GPU.
pub(crate) const APPLE: u32 = 0x106b;

/// Whether a device's driver binds through Metal argument buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArgumentBuffers {
    /// Not MoltenVK.
    NotMoltenVk,
    /// MoltenVK, binding through Metal argument buffers.
    On,
    /// MoltenVK, with `MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS` off.
    Off,
    /// MoltenVK, but its state is not read (the module docs).
    Unread,
}

/// The device's facts the state is read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Facts {
    pub(crate) driver: Option<DriverId>,
    /// `driverVersion`.
    pub(crate) release: u32,
    /// `vendorID`.
    pub(crate) vendor: u32,
    /// `maxPerStageDescriptorUpdateAfterBindSampledImages`.
    pub(crate) update_after_bind: u32,
    /// `maxPerStageDescriptorSampledImages`.
    pub(crate) per_stage: u32,
}

impl Facts {
    /// The facts of a device of `description`.
    pub(crate) fn of(description: &DeviceDescription) -> Facts {
        let limits = &description.limits;
        Facts {
            driver: description.driver_id,
            release: description.driver_version,
            vendor: description.vendor_id,
            update_after_bind: limits.max_per_stage_descriptor_update_after_bind_sampled_images,
            per_stage: limits.max_per_stage_descriptor_sampled_images,
        }
    }

    /// Whether the device binds through Metal argument buffers (the module docs).
    pub(crate) fn argument_buffers(self) -> ArgumentBuffers {
        if self.driver != Some(DriverId::MOLTENVK) {
            return ArgumentBuffers::NotMoltenVk;
        }
        if !READ.contains(&self.release) {
            return ArgumentBuffers::Unread;
        }
        if self.update_after_bind == TIER_2_UPDATE_AFTER_BIND {
            ArgumentBuffers::On
        } else if self.vendor == APPLE && self.update_after_bind == self.per_stage {
            ArgumentBuffers::Off
        } else {
            ArgumentBuffers::Unread
        }
    }
}

/// The macOS release from which MoltenVK 1.4.1 makes its device-wide `MTLResidencySet`
/// (`initMetalFeatures` sets `residencySets` from macOS 15 on an Apple GPU, and `initQueues`
/// makes the set where argument buffers are in use; read in 1.4.1's source).
const RESIDENCY_SETS_FROM_MACOS: u32 = 15;

impl Facts {
    /// Whether the device keeps MoltenVK's device-wide residency set: MoltenVK binding through
    /// Metal argument buffers ([`ArgumentBuffers::On`]) on macOS 15 or later. Such a device adds
    /// every memory, buffer and image the driver makes to the set, and removes each one freed,
    /// committing the set at once (`MVKDevice::makeResident`, `removeResidency`).
    pub(crate) fn keeps_residency_set(self) -> bool {
        self.argument_buffers() == ArgumentBuffers::On
            && macos_major().is_some_and(|major| major >= RESIDENCY_SETS_FROM_MACOS)
    }
}

/// The running macOS's major release (`kern.osproductversion`); `None` elsewhere, or unread.
fn macos_major() -> Option<u32> {
    #[cfg(target_os = "macos")]
    {
        let mut text = [0u8; 32];
        let mut length = text.len();
        // SAFETY: a nul-terminated name, and a buffer of `length` bytes the call writes at most.
        let read = unsafe {
            libc::sysctlbyname(
                c"kern.osproductversion".as_ptr(),
                text.as_mut_ptr().cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        if read != 0 {
            return None;
        }
        let version = std::str::from_utf8(text.get(..length)?).ok()?;
        version
            .trim_end_matches('\0')
            .split('.')
            .next()?
            .parse()
            .ok()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// The layer setting that asks MoltenVK for argument buffers, chained ahead of the consumer's.
pub(crate) fn setting() -> LayerSetting {
    LayerSetting::new(
        LAYER.to_owned(),
        SETTING.to_owned(),
        LayerSettingValue::Bool(true),
    )
}

/// Refuse a MoltenVK device running without argument buffers (the module docs).
pub(crate) fn check(state: ArgumentBuffers) -> Result<(), Error> {
    match state {
        ArgumentBuffers::Off => Err(Error::DriverSetting(
            DriverSetting::MoltenVkArgumentBuffersOff,
        )),
        _ => Ok(()),
    }
}

impl super::Device {
    /// Announce `moltenvk-argument-buffers-unread` once where the device is MoltenVK whose state
    /// is not read (the module docs).
    pub(super) fn degrade_argument_buffers_unread(&self) {
        let description = &self.properties().description;
        let facts = Facts::of(description);
        if facts.argument_buffers() != ArgumentBuffers::Unread {
            return;
        }
        let read: Vec<String> = READ.iter().map(u32::to_string).collect();
        self.degrade(
            crate::degradation::Degradation::MoltenVkArgumentBuffersUnread,
            || {
                (
                    format!(
                        "the device's MoltenVK at driverVersion {} (vendorID {:#x})",
                        facts.release, facts.vendor
                    ),
                    format!(
                        "its argument buffers are read at driverVersion {} on Apple GPUs only; \
                         maxPerStageDescriptorUpdateAfterBindSampledImages is {}",
                        read.join(", "),
                        facts.update_after_bind
                    ),
                )
            },
        );
    }
}
