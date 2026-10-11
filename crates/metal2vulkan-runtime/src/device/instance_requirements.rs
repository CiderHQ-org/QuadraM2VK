//! The instance extensions a device extension requires.
//!
//! Enabling a device extension whose registry dependency names an instance extension is valid
//! only on an instance that enables that instance extension
//! (`VUID-vkCreateDevice-ppEnabledExtensionNames-01387`). The validation layer checks it at
//! `vkCreateDevice`; a driver need not, and makes the device or fails as it likes. So where
//! `DeviceOptions::device_extensions` lists such an extension and the instance the runtime would
//! make enables (from `DeviceOptions::instance_extensions` and its own) not every instance
//! extension it requires, the runtime refuses it typed
//! ([`Error::InstanceExtensionRequired`]) before it makes the instance.
//!
//! [`REQUIRES`] is the registry's list, transcribed from `vk.xml` at header 357 (Vulkan 1.4.357):
//! every device extension whose `depends` names an instance extension that Vulkan 1.1 did not
//! make core, with those instance extensions. The runtime's instance is at least Vulkan 1.2
//! (`create::instance_version`), so the core ones hold; in each of the 17 the rest are plain
//! conjuncts, which the registry requires all of. A device extension's dependencies on other
//! device extensions are not checked here.

use crate::error::{Error, InstanceExtensionRequired};
use std::ffi::{CStr, CString};

/// Each device extension of the registry whose dependency names an instance extension Vulkan 1.1
/// did not make core, with every such instance extension, in the registry's order.
const REQUIRES: [(&str, &[&str]); 17] = [
    ("VK_KHR_swapchain", &["VK_KHR_surface"]),
    ("VK_KHR_display_swapchain", &["VK_KHR_display"]),
    ("VK_EXT_debug_marker", &["VK_EXT_debug_report"]),
    (
        "VK_NV_external_memory",
        &["VK_NV_external_memory_capabilities"],
    ),
    (
        "VK_EXT_display_control",
        &["VK_EXT_display_surface_counter"],
    ),
    (
        "VK_KHR_shared_presentable_image",
        &["VK_KHR_get_surface_capabilities2"],
    ),
    ("VK_GGP_frame_token", &["VK_GGP_stream_descriptor_surface"]),
    (
        "VK_EXT_present_timing",
        &["VK_KHR_get_surface_capabilities2"],
    ),
    (
        "VK_AMD_display_native_hdr",
        &["VK_KHR_get_surface_capabilities2"],
    ),
    (
        "VK_EXT_full_screen_exclusive",
        &["VK_KHR_surface", "VK_KHR_get_surface_capabilities2"],
    ),
    (
        "VK_EXT_swapchain_maintenance1",
        &["VK_EXT_surface_maintenance1"],
    ),
    (
        "VK_NV_present_barrier",
        &["VK_KHR_surface", "VK_KHR_get_surface_capabilities2"],
    ),
    (
        "VK_NV_acquire_winrt_display",
        &["VK_EXT_direct_mode_display"],
    ),
    (
        "VK_EXT_device_address_binding_report",
        &["VK_EXT_debug_utils"],
    ),
    (
        "VK_KHR_present_id2",
        &["VK_KHR_get_surface_capabilities2", "VK_KHR_surface"],
    ),
    (
        "VK_KHR_present_wait2",
        &["VK_KHR_get_surface_capabilities2", "VK_KHR_surface"],
    ),
    (
        "VK_KHR_swapchain_maintenance1",
        &["VK_KHR_surface_maintenance1"],
    ),
];

/// Whether an instance enabling `instance_extensions` may enable every one of
/// `device_extensions` on its devices: the refusal names the first of `device_extensions`, in
/// their order, with an instance extension it requires that `instance_extensions` lacks, and the
/// first such instance extension in the registry's order.
pub(crate) fn instance_requirements_met(
    device_extensions: &[CString],
    instance_extensions: &[&CStr],
) -> Result<(), Error> {
    let enabled = |name: &str| {
        instance_extensions
            .iter()
            .any(|enabled| enabled.to_bytes() == name.as_bytes())
    };
    for listed in device_extensions {
        let Some(&(extension, requires)) = REQUIRES
            .iter()
            .find(|(name, _)| listed.as_bytes() == name.as_bytes())
        else {
            continue;
        };
        if let Some(&missing) = requires.iter().find(|&&required| !enabled(required)) {
            return Err(Error::InstanceExtensionRequired(
                InstanceExtensionRequired {
                    extension,
                    instance_extension: missing,
                },
            ));
        }
    }
    Ok(())
}
