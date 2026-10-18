//! GPU generations a measured defect keys on by `deviceID`, where the defect is a generation's
//! and not every GPU its driver runs: the device layer's driver workarounds and the runtime's
//! driver defects share them. And the GPUs that report more than one `deviceID`, which every table
//! keyed on a measured GPU names by one of them ([`DeviceDescription::canonical_device_id`]).

use crate::device::DeviceDescription;

/// The `deviceID`s of NVIDIA's Ampere GA10x GPUs: every GA102, GA103, GA104, GA106 and GA107
/// entry of pci.ids (version 2026.09.03) under vendor 0x10de, their high-definition audio
/// functions left out (114 IDs). 0x2204, the RTX 3090, is the one measured.
pub const NVIDIA_AMPERE_GA10X: &[u32] = &[
    0x2200, 0x2203, 0x2204, 0x2205, 0x2206, 0x2207, 0x2208, 0x220a, 0x220d, 0x2216, 0x222b, 0x222f,
    0x2230, 0x2231, 0x2232, 0x2233, 0x2235, 0x2236, 0x2237, 0x2238, 0x223f, 0x2414, 0x2420, 0x2438,
    0x2460, 0x2480, 0x2482, 0x2483, 0x2484, 0x2486, 0x2487, 0x2488, 0x2489, 0x248a, 0x248c, 0x248d,
    0x248e, 0x249c, 0x249d, 0x249f, 0x24a0, 0x24a4, 0x24ac, 0x24ad, 0x24af, 0x24b0, 0x24b1, 0x24b6,
    0x24b7, 0x24b8, 0x24b9, 0x24ba, 0x24bb, 0x24bf, 0x24c0, 0x24c7, 0x24c8, 0x24c9, 0x24dc, 0x24dd,
    0x24df, 0x24e0, 0x24fa, 0x2501, 0x2503, 0x2504, 0x2505, 0x2507, 0x2508, 0x2509, 0x2520, 0x2521,
    0x2523, 0x252f, 0x2531, 0x2544, 0x2560, 0x2561, 0x2563, 0x2571, 0x2582, 0x2583, 0x2584, 0x25a0,
    0x25a2, 0x25a3, 0x25a4, 0x25a5, 0x25a6, 0x25a7, 0x25a9, 0x25aa, 0x25ab, 0x25ac, 0x25ad, 0x25af,
    0x25b0, 0x25b2, 0x25b5, 0x25b6, 0x25b8, 0x25b9, 0x25ba, 0x25bb, 0x25bc, 0x25bd, 0x25e0, 0x25e2,
    0x25e5, 0x25ec, 0x25ed, 0x25f9, 0x25fa, 0x25fb,
];

/// Each measured GPU that reports more than one `deviceID`, as its `vendorID` and its IDs, the one
/// it was measured under first. A table keyed on a measured GPU names it by that first ID, and a
/// device that reports any of them takes its rows at the releases they name: the driver facts,
/// deviations and defects (`driver_facts`), the Function storage a thread holds
/// (`function_storage`) and the runtime's driver defects.
const ONE_GPU: &[(u32, &[u32])] = &[
    // The AMD Radeon 780M (gfx1103): Phoenix's 0x15bf, measured (the committed capture
    // `amd-780m-radv`), and Hawk Point's 0x1900, whose capture under the same RADV build
    // (`amd-780m-hawkpoint-radv`) differs from Phoenix's in its `deviceID` alone.
    (0x1002, &[0x15bf, 0x1900]),
];

impl DeviceDescription {
    /// The `deviceID` by which the tables keyed on a measured GPU name this device's GPU: the
    /// driver facts, deviations and defects, the Function storage a thread holds, and the
    /// runtime's driver defects. It is the description's own
    /// [`device_id`](DeviceDescription::device_id), but for a GPU that reports more than one
    /// `deviceID`, the one it was measured under: the Radeon 780M reports 0x15bf (Phoenix) and
    /// 0x1900 (Hawk Point), and both name it as 0x15bf. The `driverID`, `vendorID` and
    /// `driverVersion` a measured row names still hold as they are.
    pub fn canonical_device_id(&self) -> u32 {
        ONE_GPU
            .iter()
            .find(|(vendor, ids)| *vendor == self.vendor_id && ids.contains(&self.device_id))
            .map_or(self.device_id, |(_, ids)| ids[0])
    }
}
