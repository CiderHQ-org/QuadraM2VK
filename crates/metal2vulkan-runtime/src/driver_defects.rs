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
//! Each defect lists the drivers that have it and, for each, the measured identities a probe found
//! clean: a GPU (`deviceID`, the one it was measured under for a GPU that reports several:
//! `DeviceDescription::canonical_device_id`) at one release, which the device layer measured
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
///   nothing. Measured on 615.71.09 with a bare-Vulkan program (the probe):
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
///   Iris Xe (`deviceID` 0xa7a0; `runtime_lane::texture_blits`, and a bare-Vulkan program) and
///   Raptor Lake-S's UHD 770 (0xa780); read in main's source of 2026-10-07: known broken
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
///   driverVersion 10401, where the other types' views pass: they view the
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
///   (`runtime_lane::compressed_blits`): known broken through 1.4.1. The
///   compression is Metal's, so no MoltenVK release is expected to fix it, but a probe of a
///   later one decides. Nothing is refused: the image of such a texture a CPU copy reaches, and
///   of a device's own one without `allowGPUOptimizedContents` (whose GPU blits keep the bytes
///   on Metal), is made of `PixelFormatView` usage (`compressed_blit::copy_exact_usage`), which
///   Metal keeps uncompressed, and the runtime rewrites what its blits write itself
///   (`compressed_blit.rs`).
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
///   the 780M at 26.2.4 (Phoenix's 0x15bf, and Hawk Point's 0x1900, the same GPU), takes it; the
///   RX 7900 XTX is no measured identity, so it takes it at every release.
///
/// `ExplicitPlaneRowsPastHeight`:
/// - Mesa's lavapipe (`MESA_LLVMPIPE`): a `DRM_FORMAT_MOD_LINEAR` image made for host allocations
///   with an explicit plane at a texture's packed rows, reported at `rows × rowPitch` bytes in its
///   subresource and within its memory requirements, is rendered in blocks of 4 rows: a pass that
///   loads, blends and stores a 658x126 `RGBA16Float` tile (rows of 5264 bytes) rewrote rows 126
///   and 127 past it (signaling NaNs quieted), and one over a 1308x510 `BGRA8Unorm` tile (rows of
///   5232 bytes) wrote rows 510 and 511 past the no-copy buffer's mapping, and the process took a
///   `SIGSEGV` (`runtime_lane::pitched_surfaces`, packed odd-height layouts). Its own linear images
///   report the rounded rows in their subresource. Measured on 26.2.4 (llvmpipe, LLVM 23.1.1, 256
///   bits; gem10): known broken through 26.2.4; 26.0.8 unprobed. Nothing is refused: no such
///   image of a height that is no multiple of 4 is made there (`dma_buf.rs`'s
///   `explicit_plane_rows`), and the texture takes its shadow.
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
/// `FreedAddressMemoryFailsPendingWork`:
/// - MoltenVK (`MOLTENVK`): every encoder whose pipeline reads device addresses makes every buffer
///   of `SHADER_DEVICE_ADDRESS` usage resident (`MVKDevice::encodeGPUAddressableBuffers`), in
///   Metal command buffers that retain none of it (`retainedReferences` off where the device keeps
///   its residency set), and `vkFreeMemory` lets go of the memory's Metal objects at once: a
///   Metal command buffer encoded before and not yet run fails `Invalid Resource` (Metal's code
///   9), its device then lost. Destroying the buffer alone, its memory kept, fails nothing.
///   Read in 1.4.1's source, and measured on 1.4.1 (driverVersion 10401) on an M3 Max with macOS
///   27.0.1: a bare-Vulkan program freeing an unused buffer's memory while a dispatch through a
///   device address waits behind fills fails at once, and `runtime_lane::freed_memory`: known
///   broken through 1.4.1. Nothing is refused: the runtime frees such memory once the batches
///   submitted before it completed (`freed_memory.rs`), the program's own submissions through
///   `CommandQueue::with_vk_queue` among them (a mark after each use signals a timeline of its
///   own).
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
///   (0x6802004, LLVM 23.1.1), the same calls recorded alike. Mesa 26.1.0's release notes list
///   "lavapipe: rework immutable samplers", likely the fix (unverified: its change was not read).
///   Nothing is refused: an encoder's dispatch of a variant that bakes samplers records in a
///   `VkCommandBuffer` of its own there (`compute.rs`), as one whose pipeline is still being made
///   does (`pending_dispatch.rs`); a host step's dispatch (an indirect dispatch's, an indirect
///   command buffer's execution) records in place. Graphics draws are unmeasured. Every lavapipe
///   release takes it: 26.2.4, measured clean, has no committed profile to be a measured identity.
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
        defect: RuntimeDriverDefect::ExplicitPlaneRowsPastHeight,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::OcclusionCountPerColorOutput,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ImplicitPreprocessAfterExecutionSet,
        driver: DriverId::MESA_RADV,
        // The Radeon 780M at 26.2.4, measured clean.
        clean: &[(0x15bf, (26 << 22) | (2 << 12) | 4)],
    },
    Affected {
        defect: RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::PushedBufferSizesUnset,
        driver: DriverId::MOLTENVK,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::PushedDescriptorsStaleAfterPipelineBinds,
        driver: DriverId::INTEL_OPEN_SOURCE_MESA,
        clean: &[],
    },
    Affected {
        defect: RuntimeDriverDefect::ImmutableSamplersLostAfterCopies,
        driver: DriverId::MESA_LLVMPIPE,
        clean: &[],
    },
];

/// The defects of the driver and device `description` names, its GPU read by the `deviceID` it
/// was measured under (`DeviceDescription::canonical_device_id`).
pub(super) fn of(description: &DeviceDescription) -> BTreeSet<RuntimeDriverDefect> {
    among(
        AFFECTED,
        description.driver_id,
        description.canonical_device_id(),
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
