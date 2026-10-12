//! The defects of a device's driver in commands the runtime records for itself, keyed on the
//! driver's identity, release and GPU: `VkPhysicalDeviceDriverProperties`'s `driverID`, and
//! `VkPhysicalDeviceProperties`'s `driverVersion`, which each driver raises with every release
//! (Mesa encodes it as `VK_MAKE_API_VERSION`, MoltenVK as `major * 10000 + minor * 100 + patch`),
//! and `deviceID`. A device's defects are in its properties
//! ([`DeviceProperties::driver_defects`](super::DeviceProperties::driver_defects)), and what a
//! defect breaks is taken another way where an exact one exists, else refused there, typed
//! ([`Needs::RuntimeDriverDefect`]), never run wrong.
//!
//! Each defect lists the drivers that have it and, for each, the measured identities a probe found
//! clean: a GPU (`deviceID`) at one release, which the device layer measured
//! (`DeviceDescription::measured_identity`). Every other release and GPU of the driver has it,
//! measured broken or not measured at all, so a release or GPU nobody probed takes the exact path,
//! and one probed clean then needs only its identity in the list, not new code.
//!
//! None is a translation's (`DriverWorkaround` covers those), so none is in the compile profile,
//! a cache key or a digest.
//!
//! [`Needs::RuntimeDriverDefect`]: metal2vulkan::error::Needs::RuntimeDriverDefect

use metal2vulkan::device::DeviceDescription;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::DriverId;
use std::collections::BTreeSet;

/// One driver that has a defect.
pub(super) struct Affected {
    pub(super) defect: RuntimeDriverDefect,
    pub(super) driver: DriverId,
    /// The measured identities a probe found without the defect, each a GPU (`deviceID`) at one
    /// release (`driverVersion`); every other GPU and release of the driver has it.
    pub(super) clean: &'static [(u32, u32)],
}

/// Every known defect, by driver.
///
/// `MultisampledDepthStencilColourCopies`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): `handle_copy_image` in `lvp_execute.c` copies between a
///   depth/stencil and a colour format of different texel sizes (a combined `D32_SFLOAT_S8_UINT`
///   image and `R32_SFLOAT` or `R8_UNORM`) through `texture_map`, which maps sample 0 only, so the
///   other samples are never written. Read in Mesa 26.0.8's source and in main's of 2026-10-05,
///   and measured on 26.0.8 (`runtime_lane::separate_samples`, the render census): known broken
///   through 26.2.x.
/// - MoltenVK (`MOLTENVK`): `MVKCmdCopyImage::encode` copies between a depth/stencil and a colour
///   format through a temporary buffer sized per texel with no sample count, and Metal blits no
///   multisampled texture to or from a buffer, so samples land wrong. Read in 1.4.1's source and
///   measured on 1.4.1 (driverVersion 10401): known broken through 1.4.1.
///
/// `MultisampledStencilColourCopies` (split from the above):
/// - NVIDIA's (`NVIDIA_PROPRIETARY`): on the RTX 3090 (`deviceID` 0x2204, Ampere's GA10x),
///   `vkCmdCopyImage` between the stencil aspect of a 4-sample `D32_SFLOAT_S8_UINT` image and a
///   4-sample 8-bit colour image raises a graphics exception (Xid 13, "WIDTH CT Violation" or "3D
///   HEIGHT CT Violation") and loses the device: into the stencil at 4x1, 4x2 and 1x4, out of it
///   at 1x4, whether the colour image is a colour attachment, a storage image or both, while 2x2,
///   the depth aspect, and an `S8_UINT` image copy. Valid usage: the validation layer reports
///   nothing. Measured on 615.71.09 with a bare-Vulkan program (the EV `RT6-x13-matrix-3090`):
///   known broken through 615.71.09. The RTX 5080 Laptop GPU (Blackwell, 0x2c19) on the same
///   release copies them (measured clean). Every other NVIDIA GPU and release takes the exact path
///   too until a probe finds it clean, where before only the 3090's generation did and Ada and
///   Pascal, unprobed, took the copy.
///
/// `NormalizedDepthPastUnitRange`:
/// - MoltenVK (`MOLTENVK`): Metal's rasterizer on Apple's GPUs reads and stores the depths of a
///   sloped triangle whose normalized depths reach past 1 about 2e-4 off (0.968567252 for
///   0.96875), with or without clip distances and in both depth clip modes, where Metal's own
///   viewport of the range draws the same depths exactly. Measured in Metal on an M4 with macOS
///   27.0.1 (a probe: the vertex stage writing the compensated depth into a viewport of `[0, 1]`,
///   against a native viewport of `[0.5, 2]`), and through the runtime's compensation on 1.4.1
///   (driverVersion 10401) on an M3 Max and an M4 (`runtime_lane::viewport_ranges`): known broken
///   through 1.4.1. The rasterizer is Metal's, so no MoltenVK release is expected to fix it, but a
///   probe of a later one decides; MoltenVK's other GPUs (Intel and AMD Macs) are unprobed and
///   refused alike.
///
/// `ImageSizeShortOfCompressionData`:
/// - Mesa's ANV (`INTEL_OPEN_SOURCE_MESA`) on Intel's Xe-LP GPUs: a compressed
///   image's compression data lies last in its memory, at a multiple of 4096 bytes, `size / 256`
///   bytes long (`add_compression_control_buffer` in `anv_image.c`), and the image's size ends
///   there, while the aux map gives each 64-KiB block of texels a 256-byte chunk of it (Mesa's
///   `intel_aux_map.h`) and the GPU writes it back in 64-byte lines. A 4x4 `R8G8B8A8_UNORM`
///   image asks 4112 bytes at 65536; a 1D image bound at byte 4112, written after it in the same
///   command buffer, read its old contents back at bytes 4112 to 4159, while one bound at 4160 or
///   4352 kept what was written. Measured on 26.2.4 (driverVersion 0x6802004) on Raptor Lake-P's
///   Iris Xe (`deviceID` 0xa7a0; `runtime_lane::texture_blits`, and a bare-Vulkan program, EVs
///   SPH-ccsrepro-ms01 and SPH-ccsrepro2-ms01) and Raptor Lake-S's UHD 770 (0xa780,
///   SPH-ccsrepro-13900, SPH-ccsrepro2-13900); read in main's source of 2026-10-07: known broken
///   through 26.2.4. Nothing is refused: the runtime rounds each image's size up to whole
///   256-byte chunks there (`texture::ImagePlacement`). Every ANV GPU and release takes the
///   rounding, at most 255 bytes an image, until a probe finds it clean. Mesa's source pads the
///   data to the map's own granules on Xe-LPG (Meteor Lake, Arrow Lake) and keeps none in memory on
///   Xe-HPG and Xe2, but none of those is measured, so the Xe-LP list (70 `deviceID`s) that
///   narrowed it before is gone.
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
/// `PipelineCacheDataCompiledAtLoad`:
/// - MoltenVK (`MOLTENVK`): `vkCreatePipelineCache` with initial data reads each shader module's
///   converted MSL back (`MVKPipelineCache::readData`) through
///   `MVKShaderLibraryCache::addShaderLibrary`, whose `MVKShaderLibrary` constructor decompresses
///   the MSL and calls `compileLibrary` at once, waiting for Metal's compile: every library in
///   the data is compiled before the call returns, whether a pipeline of the process uses it or
///   not. Read in 1.4.1's source (`MVKShaderModule.mm`); seen by the Vulkan consumer on an M3 Max
///   (1.4.1, driverVersion 10401) as `MTLCompilerService` rebuilding about 115 libraries at every
///   device creation from an 8.9 MB entry of 131 shaders, about 28 s for one process: known
///   broken through 1.4.1. Nothing is refused, and nothing branches on it any more: each pipeline
///   making's cache is seeded from that making's own record (`shader_cache/driver.rs`), so a load
///   compiles only the making's own stages, which it converts anyway without one.
///
/// `LazilyAllocatedVolumePlaneViews`:
/// - MoltenVK (`MOLTENVK`): `MVKImageViewPlane::newMTLTexture` makes a 2D view of a 3D image
///   (the image `VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT`) as a 2D array texture over the
///   image's memory through its placement heap (`getHeapAllocation`), with no check that there
///   is one; `MVKDeviceMemory::ensureMTLHeap` makes none for memory of Metal's memoryless
///   storage, which is what a lazily allocated memory type is, so the view's first command
///   buffer dereferences a null heap and the process takes a `SIGSEGV`
///   (`runtime_lane::memoryless::a_memoryless_3d_depth_plane_renders` on an M4, 1.4.1,
///   driverVersion 10401, EV RRP-mless1-mvk4, where the other types' views pass: they view the
///   memoryless texture itself). Read in 1.4.1's source and main's of 2026-10-05 (`fae55a18`,
///   unchanged): known broken through 1.4.1. Nothing is refused: such an image takes
///   device-local memory (`ImagePlan::memory_class`).
///
/// `CompressedImageCopiesRewriteTexels`:
/// - MoltenVK (`MOLTENVK`): a copy between a buffer and an image is a Metal blit of the image's
///   Metal texture, which Metal keeps losslessly compressed where its usage allows
///   (`texture/heap_tiled.rs`'s `metal_compresses`; an image of neither `STORAGE` usage nor
///   `MUTABLE_FORMAT`), and Metal's blits of a compressed `RG11B10Float` or `RGB9E5Float`
///   texture write other bytes than they read (`bind::CompressedBlit`), so a CPU copy of a
///   `Shared` texture of either format reads back NaNs made quiet and shared exponents
///   renormalized, where Metal's `replaceRegion` and `getBytes` keep the bytes. Measured on
///   1.4.1 (driverVersion 10401) on an M4 under macOS 27.0.1
///   (`runtime_lane::compressed_blits`, EV FXR-cbl1-mvk4): known broken through 1.4.1. The
///   compression is Metal's, so no MoltenVK release is expected to fix it, but a probe of a
///   later one decides. Nothing is refused: the image of such a texture a CPU copy reaches is
///   made of `PixelFormatView` usage (`compressed_blit::copy_exact_usage`), which Metal keeps
///   uncompressed, and the runtime rewrites what its blits write itself (`compressed_blit.rs`).
///
/// `MultisampledCopiesConvertFloatNans`:
/// - Mesa's RADV (`MESA_RADV`): `gfx_or_compute_copy_image` in `radv_meta_copy.c` gives the
///   source's blit surface the destination's format wherever the source is not DCC-compressed or
///   the two formats are DCC-compatible (every pair on GFX11, `radv_dcc_formats_compatible`),
///   and a multisampled copy is then a draw (`radv_gfx_copy_image`) that fetches each sample in
///   that format and exports it: a NaN in a 16-bit float channel lands as `0xfe00`, its sign and
///   payload lost, while a single-sampled copy kept them. Read in 26.2's source (2026-10-09), and
///   measured on 26.2.4 (driverVersion 0x6802004) on the Radeon 780M (`deviceID` 0x15bf;
///   `runtime_lane::renders::multisampled_nan_clears_store_metal_s_texel`, EVs ZIN-nc7-radv and
///   ZIN-nc8-radv: every NaN sample of a 4x `R16Float` and `RGBA16Float` NaN clear, copied from
///   its unsigned-integer scratch, held `0xfe00`; ZIN-x7a-radv: a scratch not mutable to the
///   attachment's format alike): known broken through 26.2.4. The path is every RADV GPU's, so
///   every device has it. Nothing is refused: the NaN clear of a multisampled attachment is
///   written through a view of its image in the unsigned-integer format of its texel size
///   (`nan_clear.rs`), which the runtime's multisampled color render targets of such formats
///   take there where it costs the image nothing (`nan_clear::view_candidate`), and is the
///   device's copy, announced (`device-nan-encoding`), where the image takes no such view.
///
/// `ReadsCommandBuffersAfterTheirSignal`:
/// - MoltenVK (`MOLTENVK`): on a device with `MTLSharedEvent` the GPU signals a timeline
///   semaphore (`encodeSignalEvent`), so the host sees the value before the Metal command
///   buffer's completed handlers run, and `MVKCommandEncoder::finishQueries` adds one that calls
///   into every query pool the command buffer reset or began, through raw pointers it does not
///   retain. A command buffer released at the value crashed there (`SIGSEGV` at that handler's
///   load of a released `MVKQueryPool`'s vtable;
///   `runtime_lane::visibility::command_buffers_released_at_completion_leave_nothing_read_later`
///   passed 0 of 8 runs on an M4). MoltenVK signals a fence from the last of those handlers. Read
///   in 1.4.1's source and measured on 1.4.1 (driverVersion 10401): known broken through 1.4.1.
///   Nothing is refused: a command buffer's last batch signals a fence there too, and the
///   command buffer finishes once both are signaled (`command_buffer/fences.rs`). Every other
///   driver ends a command buffer at its value alone: Vulkan makes the value enough, and their
///   fences are signaled with it, so a fence there only added a `vkCreateFence` to the commit
///   path, which NVIDIA's 615.71.09 held for up to 1.9 s while other processes' work ran.
///
/// `ImplicitPreprocessAfterExecutionSet`:
/// - Mesa's RADV (`MESA_RADV`): `vkCmdExecuteGeneratedCommandsEXT` of a layout not made for
///   explicit preprocessing dispatches the layout's preprocessing pipeline itself
///   (`radv_prepare_dgc`), and `radv_emit_compute_pipeline` skips a pipeline equal to
///   `emitted_compute_pipeline`. The execution's own commands then bind each sequence's pipeline
///   of the execution set on the GPU, which that field does not see, so the next execution's
///   preprocessing in the command buffer, with nothing dispatched between, runs the shader the
///   execution set bound last with the preprocessing's arguments. A range of two commands, an
///   execution each, faulted (`[gfxhub] page fault`, client `TCP`, permission faults) and timed out
///   the ring (`runtime_lane::generated_executions`, both of its tests that execute, on every run),
///   where explicit preprocessing runs them right. Read in 25.0.0's,
///   26.0.8's and 26.1.5's source (the field is tracked alike); 26.2.0 tracks a dirty bit instead,
///   which the preprocessing's own bind sets. Measured on 26.0.8 (driverVersion 0x6800008) on the
///   Radeon 780M (`deviceID` 0x15bf) and the RX 7900 XTX (0x744c): known broken through 26.0.8.
///   Measured clean on 26.2.4 (driverVersion 0x6802004) on both. Nothing is refused: the runtime
///   preprocesses every execution of the range apart, before the first runs
///   (`generated_execute.rs`). Every RADV GPU and release but the measured identity found clean,
///   the 780M at 26.2.4, takes it; the RX 7900 XTX is no measured identity, so it takes it at
///   every release.
pub(super) const AFFECTED: &[Affected] = &[
    Affected {
        defect: RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::MultisampledDepthStencilColourCopies,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::MultisampledStencilColourCopies,
        driver: DriverId::NVIDIA_PROPRIETARY,
        // The RTX 5080 Laptop GPU at 615.71.09, measured clean.
        clean: &[(0x2c19, (615 << 22) | (71 << 14) | (9 << 6))],
    },
    Affected {
        defect: RuntimeDriverDefect::NormalizedDepthPastUnitRange,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ImageSizeShortOfCompressionData,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::DepthStencilResolveFirstSubresource,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::PipelineCacheDataCompiledAtLoad,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::CompressedImageCopiesRewriteTexels,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::MultisampledCopiesConvertFloatNans,
        driver: DriverId::MESA_RADV,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet,
        driver: DriverId::MESA_RADV,
        // The Radeon 780M at 26.2.4, measured clean.
        clean: &[(0x15bf, (26 << 22) | (2 << 12) | 4)],
    },
];

/// The defects of the driver and device `description` names.
pub(super) fn of(description: &DeviceDescription) -> BTreeSet<RuntimeDriverDefect> {
    among(
        AFFECTED,
        description.driver_id,
        description.device_id,
        description.driver_version,
    )
}

/// The defects of `table` that the driver `driver` has on the device `device` at release
/// `version`: each of the driver's but those a probe found clean on that GPU at that release.
pub(super) fn among(
    table: &[Affected],
    driver: Option<DriverId>,
    device: u32,
    version: u32,
) -> BTreeSet<RuntimeDriverDefect> {
    table
        .iter()
        .filter(|affected| Some(affected.driver) == driver)
        .filter(|affected| !affected.clean.contains(&(device, version)))
        .map(|affected| affected.defect)
        .collect()
}
