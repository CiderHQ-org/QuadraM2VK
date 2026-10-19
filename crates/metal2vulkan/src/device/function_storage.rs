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
//! as correct. The table keys each value on the device's driver, vendor and device ID at every
//! release, as the known-driver list keys its entries (`device::listed`) and the driver
//! workarounds are keyed ([`DriverWorkaround`]), and on nothing else; a GPU that reports more than
//! one device ID is read by the one it was measured under
//! ([`DeviceDescription::canonical_device_id`]). lavapipe's value is its driver's on every CPU:
//! its one device is the process's.
//!
//! A device the table does not name holds [`FLOOR`], 16 KiB a thread: half the least any
//! measured device held (32 KiB), and below every size any of them failed at (48 KiB and up).
//! Private memory within it is taken as held on every device, and records nothing. Past it, on
//! such a device, a vertex or fragment function runs up to [`METAL`], 192 KiB a thread, the most
//! a Metal device was measured to hold, and records `PrecisionNote::FunctionStorageAssumed`: past
//! the floor, unmeasured, it rests on no measurement, and a driver that holds less can answer
//! wrongly or hang the GPU, as the RTX 5080 at 48 KiB and the RTX 3090 at 192 KiB hung when probed.
//! A kernel, an object or a mesh function past the floor there is refused, as is any stage past a
//! measured device's value and a vertex or fragment function past [`METAL`]: typed
//! (`Unsupported{PrivateMemory}`, or `Unsupported{Recursion}` for a stage with a call stack,
//! needing `VkLimit::FunctionStorageBytesPerThread`).
//!
//! [`ProfileFields::function_storage`]: crate::device::profile::ProfileFields::function_storage
//! [`DriverWorkaround`]: crate::device::profile::DriverWorkaround

use crate::device::checks::{decide, Baseline, DeviceFact};
use crate::device::description::{DeviceDescription, APPLE_VENDOR_ID};
use crate::device::listed::{Gpus, Listing, Releases};
use crate::device::profile::FunctionStorage;
use crate::pipeline::sym::Stage;
use crate::vk;

/// One measured device: the devices it names, and the most Function storage a thread held
/// correctly.
struct Measured {
    on: Listing,
    bytes: u64,
}

/// The devices of `driver` on the GPU `gpus` of `vendor`, at every release.
const fn on(driver: vk::DriverId, vendor: u32, gpus: Gpus) -> Listing {
    Listing {
        driver,
        vendor: Some(vendor),
        gpus,
        releases: Releases::EVERY,
    }
}

const KIB: u64 = 1024;

/// The devices measured, each with the build and the evidence it was measured on.
const MEASURED: [Measured; 9] = [
    // AMD Radeon 780M (RADV PHOENIX), RADV Mesa 26.2.4-arch3.1: 64 and 96 KiB correct over 262144
    // threads; 128 KiB wrong over 4096. RADV Mesa 26.0.8-1ubuntu0.3 (homelab-um780): 4 to 96 KiB
    // correct over 262144 threads. Hawk Point's 780M (0x1900) is the same GPU, read as 0x15bf.
    Measured {
        on: on(vk::DriverId::MESA_RADV, 0x1002, Gpus::Only(&[0x15bf])),
        bytes: 96 * KIB,
    },
    // AMD Radeon RX 7900 XTX (RADV NAVI31), RADV Mesa 26.0.8-1ubuntu0.3 (homelab-um780): 4 to 96
    // KiB correct over 262144 threads; 128 KiB wrong over 4096 (53 to 55 of the 64 threads
    // checked, in three runs).
    Measured {
        on: on(vk::DriverId::MESA_RADV, 0x1002, Gpus::Only(&[0x744c])),
        bytes: 96 * KIB,
    },
    // Intel Iris Xe Graphics (RPL-P), ANV Mesa 26.2.4-arch3.1: 4 to 32 KiB correct over 262144
    // threads; 48 KiB wrong over 4096. The Intel venue's two devices derive one translation key, so
    // each takes the lower of the two values.
    Measured {
        on: on(
            vk::DriverId::INTEL_OPEN_SOURCE_MESA,
            0x8086,
            Gpus::Only(&[0xa7a0]),
        ),
        bytes: 32 * KIB,
    },
    // Intel Graphics (RPL-S, a UHD 770), ANV Mesa 26.2.4-arch3.1: 4 to 256 KiB correct over 262144
    // threads; 512 KiB refused at its first dispatch (VK_ERROR_UNKNOWN). It takes the RPL-P's lower
    // 32 KiB, its venue's.
    Measured {
        on: on(
            vk::DriverId::INTEL_OPEN_SOURCE_MESA,
            0x8086,
            Gpus::Only(&[0xa780]),
        ),
        bytes: 32 * KIB,
    },
    // NVIDIA GeForce RTX 5080 Laptop GPU, NVIDIA 615.71.09: 4 to 32 KiB correct over 262144
    // threads; 48 KiB hung over 4096 threads (no completion within 50 s, Xid 109 CTX SWITCH
    // TIMEOUT).
    Measured {
        on: on(
            vk::DriverId::NVIDIA_PROPRIETARY,
            0x10de,
            Gpus::Only(&[0x2c19]),
        ),
        bytes: 32 * KIB,
    },
    // NVIDIA GeForce RTX 3090, NVIDIA 615.71.09 (homelab-gem10): 4 to 128 KiB correct over 262144
    // threads; 192 KiB hung over 4096 threads (no completion within 50 s, Xid 109 CTX SWITCH
    // TIMEOUT).
    Measured {
        on: on(
            vk::DriverId::NVIDIA_PROPRIETARY,
            0x10de,
            Gpus::Only(&[0x2204]),
        ),
        bytes: 128 * KIB,
    },
    // Apple M4, MoltenVK 1.4.1 under macOS 27.0.1: 64 to 192 KiB correct over 262144 threads; 256
    // KiB refused at pipeline creation (VK_ERROR_INITIALIZATION_FAILED).
    Measured {
        on: on(
            vk::DriverId::MOLTENVK,
            APPLE_VENDOR_ID,
            Gpus::Only(&[0x1b00_0209]),
        ),
        bytes: 192 * KIB,
    },
    // Apple M3 Max, MoltenVK 1.4.1 under macOS 26.7: 4 to 192 KiB correct
    // over 262144 threads; 256 KiB refused at pipeline creation (VK_ERROR_INITIALIZATION_FAILED).
    Measured {
        on: on(
            vk::DriverId::MOLTENVK,
            APPLE_VENDOR_ID,
            Gpus::Only(&[0x1a06_0209]),
        ),
        bytes: 192 * KIB,
    },
    // lavapipe, Mesa 26.0.8-1ubuntu0.3 (LLVM 21.1.8), whose identity (device ID 0) is one on every
    // CPU, so it takes the lower of its measured values. x86_64 on an i9-13900: 4 to 48 KiB correct
    // over 262144 threads; 64 KiB not measured (pipeline creation passed 35 s, the size cut at 60
    // s). arm64 in a Mac mini's Linux VM: 64 KiB correct over 262144 threads; 96 KiB not measured.
    Measured {
        on: on(vk::DriverId::MESA_LLVMPIPE, 0x10005, Gpus::Any),
        bytes: 48 * KIB,
    },
];

/// The Function storage a device not measured holds, a thread: 16 KiB (the module docs).
pub(crate) const FLOOR: u64 = 16 * KIB;

/// The most Function storage a thread was measured to hold on Metal: the Apple M3 Max and M4 hold
/// 192 KiB and refuse 256 KiB at pipeline creation (their rows in [`MEASURED`]). A vertex or
/// fragment function on a device not measured may keep this much, past [`FLOOR`] with
/// `PrecisionNote::FunctionStorageAssumed` (the module docs).
pub(crate) const METAL: u64 = 192 * KIB;

/// The Function storage a thread of `desc`'s device holds correctly: the measured value of its
/// identity, or [`FLOOR`] for a device not measured, held by every stage where the device learned
/// it so (`DeviceFact::FunctionStorageFloor`, its checks taken where `trusted`).
pub(crate) fn of(desc: &DeviceDescription, trusted: bool) -> FunctionStorage {
    let unmeasured = || {
        let fact = DeviceFact::FunctionStorageFloor;
        let checks = trusted.then_some(&desc.checks);
        match decide(fact, Baseline::unlisted(), checks).holds {
            true => FunctionStorage::Floor(FLOOR),
            false => FunctionStorage::Assumed(FLOOR),
        }
    };
    MEASURED
        .iter()
        .find(|d| d.on.names(desc))
        .map_or_else(unmeasured, |d| FunctionStorage::Measured(d.bytes))
}

/// The private memory a thread of a `stage` function may keep on a device whose Function storage
/// is `storage`: the device's own value, but [`METAL`] for a vertex or fragment function on a
/// device not measured (`FunctionStorage::Assumed`; not one that learned the floor off), which past
/// [`FLOOR`] rests on no measurement (the module docs).
pub(crate) fn ceiling(storage: FunctionStorage, stage: Stage) -> u64 {
    match (storage, stage) {
        (FunctionStorage::Assumed(_), Stage::Vertex | Stage::Fragment) => METAL,
        _ => storage.bytes(),
    }
}
