//! What a device reports about itself: the Vulkan facts `query` reads of the physical device, from
//! which `Device::assemble` builds the [`DeviceProperties`](crate::DeviceProperties) Metal's device
//! queries answer from (`capabilities.rs`).

use crate::capabilities::DeviceIdentity;
use ash::vk;

/// The kind of physical device, from `VkPhysicalDeviceType`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    /// A discrete GPU.
    Discrete,
    /// A GPU integrated with the host.
    Integrated,
    /// A GPU of a virtual machine.
    Virtual,
    /// A CPU implementation (lavapipe, SwiftShader).
    Cpu,
    /// Any other device.
    Other,
}

impl DeviceKind {
    pub(super) fn of(ty: vk::PhysicalDeviceType) -> DeviceKind {
        match ty {
            vk::PhysicalDeviceType::DISCRETE_GPU => DeviceKind::Discrete,
            vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceKind::Integrated,
            vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceKind::Virtual,
            vk::PhysicalDeviceType::CPU => DeviceKind::Cpu,
            _ => DeviceKind::Other,
        }
    }

    /// Whether a device of the kind shares the host's memory (`hasUnifiedMemory`): an integrated
    /// GPU and a CPU implementation do; a discrete GPU does not, and a virtual GPU or another kind
    /// is not known to.
    pub(crate) fn has_unified_memory(self) -> bool {
        matches!(self, DeviceKind::Integrated | DeviceKind::Cpu)
    }

    /// The order `Device::system_default` prefers kinds in: discrete, integrated, virtual, CPU,
    /// other.
    pub(super) fn preference(self) -> u8 {
        match self {
            DeviceKind::Discrete => 0,
            DeviceKind::Integrated => 1,
            DeviceKind::Virtual => 2,
            DeviceKind::Cpu => 3,
            DeviceKind::Other => 4,
        }
    }
}

/// `registryID` of a device with `uuid`: its two halves, little-endian, exclusive-ored, and 1
/// when that is 0 (Metal never answers 0).
pub(crate) fn registry_id(uuid: &[u8; 16]) -> u64 {
    let half = |at: usize| {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&uuid[at..at + 8]);
        u64::from_le_bytes(bytes)
    };
    match half(0) ^ half(8) {
        0 => 1,
        id => id,
    }
}

/// The identity of `physical`, and its largest memory allocation
/// (`VkPhysicalDeviceVulkan11Properties::maxMemoryAllocationSize`), which the sub-allocator
/// reads.
///
/// # Safety
///
/// `physical` was enumerated from `instance`, created at `instance_api_version`, which is
/// Vulkan 1.2 or later.
pub(super) unsafe fn query(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    instance_api_version: metal2vulkan::vk::ApiVersion,
) -> (DeviceIdentity, u64) {
    let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut v11);
    // SAFETY: the caller's contract; `VkPhysicalDeviceVulkan11Properties` is core in 1.2.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let core = properties.properties;
    let identity = DeviceIdentity {
        name: core
            .device_name_as_c_str()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        kind: DeviceKind::of(core.device_type),
        vendor_id: core.vendor_id,
        device_id: core.device_id,
        device_uuid: v11.device_uuid,
        api_version: metal2vulkan::vk::ApiVersion::from_packed(core.api_version)
            .min(instance_api_version),
        max_viewports: core.limits.max_viewports,
        // SAFETY: the caller's contract.
        gpu_core_count: unsafe { gpu_core_count(instance, physical) },
        sampled_sample_counts: metal2vulkan::vk::SampleCountFlags::from_raw(
            (core.limits.sampled_image_color_sample_counts
                & core.limits.sampled_image_depth_sample_counts)
                .as_raw(),
        ),
    };
    (identity, v11.max_memory_allocation_size)
}

/// `gpuCoreCount` of `physical` as its driver reports it
/// ([`DeviceProperties::gpu_core_count`](crate::DeviceProperties::gpu_core_count)): through the
/// first of NVIDIA's, AMD's and Arm's extensions that the driver lists, each read only where
/// listed (a query of properties enables nothing); `None` where it lists none, or reports 0.
///
/// # Safety
///
/// As [`query`]'s.
unsafe fn gpu_core_count(instance: &ash::Instance, physical: vk::PhysicalDevice) -> Option<u32> {
    // SAFETY: the caller's contract.
    let listed = unsafe { instance.enumerate_device_extension_properties(physical) }.ok()?;
    let has = |name: &std::ffi::CStr| {
        listed
            .iter()
            .any(|extension| extension.extension_name_as_c_str() == Ok(name))
    };
    let nvidia = has(ash::nv::shader_sm_builtins::NAME);
    let amd = has(ash::amd::shader_core_properties::NAME);
    let amd_active = has(ash::amd::shader_core_properties2::NAME);
    let arm = has(ash::arm::shader_core_builtins::NAME);
    let mut sms = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
    let mut cores = vk::PhysicalDeviceShaderCorePropertiesAMD::default();
    let mut active = vk::PhysicalDeviceShaderCoreProperties2AMD::default();
    let mut arm_cores = vk::PhysicalDeviceShaderCoreBuiltinsPropertiesARM::default();
    let mut properties = vk::PhysicalDeviceProperties2::default();
    if nvidia {
        properties = properties.push_next(&mut sms);
    }
    if amd {
        properties = properties.push_next(&mut cores);
    }
    if amd_active {
        properties = properties.push_next(&mut active);
    }
    if arm {
        properties = properties.push_next(&mut arm_cores);
    }
    if !(nvidia || amd || amd_active || arm) {
        return None;
    }
    // SAFETY: the caller's contract; each structure chained is of an extension the device lists.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    let amd_total = cores
        .shader_engine_count
        .saturating_mul(cores.shader_arrays_per_engine_count)
        .saturating_mul(cores.compute_units_per_shader_array);
    [
        (nvidia, sms.shader_sm_count),
        (amd_active, active.active_compute_unit_count),
        (amd, amd_total),
        (arm, arm_cores.shader_core_count),
    ]
    .into_iter()
    .find_map(|(listed, count)| (listed && count > 0).then_some(count))
}

/// The `driverUUID` and `deviceUUID` of `physical`.
///
/// # Safety
///
/// As [`query`]'s.
pub(super) unsafe fn uuids(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> [[u8; 16]; 2] {
    let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut v11);
    // SAFETY: the caller's contract; `VkPhysicalDeviceVulkan11Properties` is core in 1.2.
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    [v11.driver_uuid, v11.device_uuid]
}
