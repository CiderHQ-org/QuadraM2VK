//! The Function storage a thread holds correctly on each device:
//! [`ProfileFields::function_storage`].
//!
//! A thread's private memory, its `alloca`s (a recursion's call stack among them) and the constants
//! it holds as `Private` variables, is storage a driver backs with scratch memory. No Vulkan
//! property states how much of it a thread holds, and past it a driver can answer wrongly or hang
//! with no error: the Radeon 780M and the RX 7900 XTX under RADV over 4096 threads at 128 KiB a
//! thread, the Iris Xe under ANV at 48 KiB, the RTX 5080 at 48 KiB and the RTX 3090 at 192 KiB
//! (both Xid 109). So the amount is measured, per device, by `validation`'s
//! `runtime_lane::private_memory` (a private array of that many bytes a thread, every 64-byte line
//! written and read back in an order no two threads share, over 1, 64, 4096 and 262144 threads,
//! each size cut at 60 s). A device's value is its largest size answered correctly over every
//! thread count below its first size that failed; a size cut at 60 s is not measured, never taken
//! as correct. The table keys each value on the device's identity, its driver, vendor, and device
//! ID, as the driver workarounds are keyed ([`DriverWorkaround`]), and on nothing else.
//!
//! A device the table does not name is assumed to hold [`FLOOR`], 16 KiB a thread: half the
//! least any measured device held (32 KiB), and below every size any of them failed at (48 KiB
//! and up). A stage that keeps private memory on such a device records
//! `PrecisionNote::FunctionStorageAssumed`, so its output is known to rest on the floor, not on a
//! measurement. Past the device's value, measured or assumed, a stage is refused, typed
//! (`Unsupported{PrivateMemory}`, or `Unsupported{Recursion}` for a stage with a call stack,
//! needing `VkLimit::FunctionStorageBytesPerThread`).
//!
//! [`ProfileFields::function_storage`]: crate::device::profile::ProfileFields::function_storage
//! [`DriverWorkaround`]: crate::device::profile::DriverWorkaround

use crate::device::description::{DeviceDescription, APPLE_VENDOR_ID};
use crate::device::profile::FunctionStorage;
use crate::vk;

/// One measured device: its identity, and the most Function storage a thread held correctly.
struct Measured {
    driver: vk::DriverId,
    vendor: u32,
    device: u32,
    bytes: u64,
}

const KIB: u64 = 1024;

/// The devices measured, each with the build and the evidence it was measured on.
const MEASURED: [Measured; 9] = [
    // AMD Radeon 780M (RADV PHOENIX), RADV Mesa 26.2.4-arch3.1 (EV
    // UNI-B3-pm-radv): 64 and 96 KiB correct over 262144 threads; 128 KiB wrong over 4096. RADV
    // Mesa 26.0.8-1ubuntu0.3 (homelab-um780): 4 to 96 KiB correct over 262144 threads.
    Measured {
        driver: vk::DriverId::MESA_RADV,
        vendor: 0x1002,
        device: 0x15bf,
        bytes: 96 * KIB,
    },
    // AMD Radeon RX 7900 XTX (RADV NAVI31), RADV Mesa 26.0.8-1ubuntu0.3 (homelab-um780): 4 to 96
    // KiB correct over 262144 threads; 128 KiB wrong over 4096 (53 to 55 of the 64 threads
    // checked, in three runs).
    Measured {
        driver: vk::DriverId::MESA_RADV,
        vendor: 0x1002,
        device: 0x744c,
        bytes: 96 * KIB,
    },
    // Intel Iris Xe Graphics (RPL-P), ANV Mesa 26.2.4-arch3.1 (EV UNI-B4-pm-anv): 4
    // to 32 KiB correct over 262144 threads; 48 KiB wrong over 4096. The Intel venue's two devices
    // derive one translation key, so each takes the lower of the two values.
    Measured {
        driver: vk::DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: 0x8086,
        device: 0xa7a0,
        bytes: 32 * KIB,
    },
    // Intel Graphics (RPL-S, a UHD 770), ANV Mesa 26.2.4-arch3.1 (EV
    // UNI-B6-pm-rpls): 4 to 256 KiB correct over 262144 threads; 512 KiB refused at its first
    // dispatch (VK_ERROR_UNKNOWN). It takes the RPL-P's lower 32 KiB, its venue's.
    Measured {
        driver: vk::DriverId::INTEL_OPEN_SOURCE_MESA,
        vendor: 0x8086,
        device: 0xa780,
        bytes: 32 * KIB,
    },
    // NVIDIA GeForce RTX 5080 Laptop GPU, NVIDIA 615.71.09 (EV UNI-B4-pm-nv5080): 4
    // to 32 KiB correct over 262144 threads; 48 KiB hung over 4096 threads (no completion within
    // 50 s, Xid 109 CTX SWITCH TIMEOUT).
    Measured {
        driver: vk::DriverId::NVIDIA_PROPRIETARY,
        vendor: 0x10de,
        device: 0x2c19,
        bytes: 32 * KIB,
    },
    // NVIDIA GeForce RTX 3090, NVIDIA 615.71.09 (homelab-gem10): 4 to 128 KiB correct over 262144
    // threads; 192 KiB hung over 4096 threads (no completion within 50 s, Xid 109 CTX SWITCH
    // TIMEOUT).
    Measured {
        driver: vk::DriverId::NVIDIA_PROPRIETARY,
        vendor: 0x10de,
        device: 0x2204,
        bytes: 128 * KIB,
    },
    // Apple M4, MoltenVK 1.4.1 under macOS 27.0.1 (EV UNI-B3-pm-mvkm4): 64 to
    // 192 KiB correct over 262144 threads; 256 KiB refused at pipeline creation
    // (VK_ERROR_INITIALIZATION_FAILED).
    Measured {
        driver: vk::DriverId::MOLTENVK,
        vendor: APPLE_VENDOR_ID,
        device: 0x1b00_0209,
        bytes: 192 * KIB,
    },
    // Apple M3 Max, MoltenVK 1.4.1 under macOS 26.7 (EV UNI-B7-pm-mvkm3): 4 to 192 KiB correct
    // over 262144 threads; 256 KiB refused at pipeline creation (VK_ERROR_INITIALIZATION_FAILED).
    Measured {
        driver: vk::DriverId::MOLTENVK,
        vendor: APPLE_VENDOR_ID,
        device: 0x1a06_0209,
        bytes: 192 * KIB,
    },
    // lavapipe, Mesa 26.0.8-1ubuntu0.3 (LLVM 21.1.8), whose identity (device ID 0) is one on every
    // CPU, so it takes the lower of its measured values. x86_64 on an i9-13900 (EV
    // UNI-B5-pm-lvpx86): 4 to 48 KiB correct over 262144 threads; 64 KiB not measured (pipeline
    // creation passed 35 s, the size cut at 60 s). arm64 in a Mac mini's Linux VM (EV
    // UNI-B3-pm-lvparm): 64 KiB correct over 262144 threads; 96 KiB not measured.
    Measured {
        driver: vk::DriverId::MESA_LLVMPIPE,
        vendor: 0x10005,
        device: 0,
        bytes: 48 * KIB,
    },
];

/// The Function storage a device not measured is assumed to hold, a thread: 16 KiB (the module
/// docs).
pub(crate) const FLOOR: u64 = 16 * KIB;

/// The Function storage a thread of `desc`'s device holds correctly: the measured value of its
/// identity, or the assumed [`FLOOR`] for a device not measured.
pub(crate) fn of(desc: &DeviceDescription) -> FunctionStorage {
    MEASURED
        .iter()
        .find(|d| {
            desc.driver_id == Some(d.driver)
                && desc.vendor_id == d.vendor
                && desc.device_id == d.device
        })
        .map_or(FunctionStorage::Assumed(FLOOR), |d| {
            FunctionStorage::Measured(d.bytes)
        })
}
