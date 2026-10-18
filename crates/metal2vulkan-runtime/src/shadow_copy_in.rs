//! A merged shadowed texture's copy in and keep as one compute dispatch, where the device takes
//! them faster so (`DriverFact::ShadowCopyInDispatchFaster`).
//!
//! A shadowed texture over bytes another writer may write while a command buffer runs is merged
//! back (`shadow_merge.rs`): before the encoder's first command that writes through it, its
//! shadow is copied in from the buffer's bytes (`vkCmdCopyBufferToImage`, `shadow.rs`) and then
//! kept, its texels stored row after row in the merge's scratch (the copy back's dispatch on such
//! a device, `shadow_copy_back.rs`), with a barrier between the two. Where both move the same
//! rectangle of a one-region texture copied in place, with no copy of the shadow between them,
//! the encoder records them as one dispatch of `bind::shadow_copy_in_module` at the copy in
//! ([`CopyIn`]): invocation `(x, y)` reads texel `(x0 + x, y0 + y)`'s words from the buffer once,
//! stores them through the shadow's unsigned-integer view, its storage image
//! (`shadow_copy_back::WordView::stores`), and stores them at the texel's place in the kept part;
//! the keep records nothing (`WorkCounts::shadow_copies_in_dispatched`).
//!
//! **Exact.** The shadow holds the bytes the copy writes, and the kept part the words the keep
//! stores: Vulkan states that "Views of compatible formats will have the same mapping between
//! texel coordinates and memory locations irrespective of the format, with only the
//! interpretation of the bit pattern changing" (`VkImageViewCreateInfo`), and the view's
//! unsigned-integer components store their bits unconverted (a 4-byte texel's bytes, one a
//! component of its `R8G8B8A8_UINT` view). The kept part takes the words the
//! shadow takes, from the same read, as the keep after the copy took the shadow's: the merge after
//! the commands compares the texels with them alone.
//!
//! **Synchronization.** The dispatch's uses are a storage read of the buffer's bytes and storage
//! writes of the shadow and of the kept part in `COMPUTE_SHADER` (`ShadowMerge::keep_in_uses`),
//! which the encoder's tracker orders as any command's, before the commands that read the shadow.
//!
//! **Where the two stay.** A device without the fact (its shadows' views do not store), a merge
//! not cut to one rectangle in place, a copy in of another rectangle than the keep's, a copy of
//! the shadow between them, a texel of another width, a shadow whose view takes no storage (a
//! single 32-bit channel's format without storage usage of its own, `shadow_copy_back.rs`), a
//! region past the device's storage buffer range or workgroup counts, a kept part off the
//! storage buffer alignment, a buffer whose `VkBuffer` went, or a kernel the device does not
//! make: each records its copy in and its keep as before.

use super::owned_kernel::{OwnedBinds, OwnedKernel, OwnedModule, OwnedSet};
use super::shadow_copy_back::WordView;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::ShadowCopyInModule;

/// One copy in and keep's dispatch (the module docs): the kernel of its texel's width, its set 0
/// (the shadow's view, the buffer's bytes from the region's first texel and the kept part, as they
/// lie now; pushed where the device pushes, `OwnedBinds`), its shape and its workgroups. The
/// command buffer retains it until it completes.
pub(super) struct CopyIn {
    kernel: OwnedKernel,
    binds: OwnedBinds,
    shape: [u8; 16],
    groups: [u32; 2],
}

impl CopyIn {
    /// The dispatch on `device` that copies `region` of the buffer `buffer`'s `VkBuffer` (its
    /// offset there, its row length, its texels' origin and extent in the shadow) into the shadow
    /// through `words`, and its texels row after row into the kept part at `kept` (its `VkBuffer`
    /// and where the part starts there); `None` where the copy and the keep take it (the module
    /// docs).
    pub(super) fn new(
        device: &Device,
        words: WordView,
        buffer: vk::Buffer,
        region: &vk::BufferImageCopy,
        kept: (vk::Buffer, u64),
    ) -> Option<CopyIn> {
        if buffer == vk::Buffer::null() || !words.stores {
            return None;
        }
        let limits = &device.properties().description.limits;
        let texel = 4 * u64::from(words.texel.words());
        let extent = region.image_extent;
        let (width, height) = (extent.width, extent.height);
        let row = match region.buffer_row_length {
            0 => width,
            row => row,
        };
        let [x0, y0] = [region.image_offset.x, region.image_offset.y].map(|o| o as u32);
        let first = region.buffer_offset;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let offset = first - first % align;
        let row_bytes = u64::from(row) * texel;
        let end = first + u64::from(height.max(1) - 1) * row_bytes + u64::from(width) * texel;
        let range = end - offset;
        let kept_range = u64::from(width) * u64::from(height) * texel;
        let groups = ShadowCopyInModule::groups([width, height]);
        let max_range = u64::from(limits.max_storage_buffer_range);
        let fits = width > 0
            && height > 0
            && extent.depth == 1
            && first.is_multiple_of(4)
            && kept.1.is_multiple_of(align)
            && x0 < 1 << 16
            && y0 < 1 << 16
            && range <= max_range
            && kept_range <= max_range
            && groups[0] <= limits.max_compute_work_group_count[0]
            && groups[1] <= limits.max_compute_work_group_count[1];
        if !fits {
            return None;
        }
        // Each below `maxStorageBufferRange`, a `u32`.
        let row_words = (row_bytes / 4) as u32;
        let first_word = ((first - offset) / 4) as u32;
        let kernel = device
            .owned_kernel(OwnedModule::ShadowCopyIn(words.texel))
            .ok()?;
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: vec![words.view],
            buffers: vec![(buffer, offset, range), (kept.0, kept.1, kept_range)],
        };
        let binds = OwnedBinds::new(device, &kernel, &[set]).ok()?;
        Some(CopyIn {
            kernel,
            binds,
            shape: ShadowCopyInModule::shape([x0, y0], width, row_words, first_word),
            groups,
        })
    }

    /// Record the dispatch into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the shadow, its view, the buffer and
    /// the kept part are live until it completes, and the shadow is in the `GENERAL` layout.
    pub(super) unsafe fn record(&self, device: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract; the set binds the view, the bytes the region's texels
        // lie in and the kept part, and the shape and workgroups stay inside them (checked in
        // `new`).
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            self.binds.bind(device, command_buffer, &self.kernel, 0);
            device.cmd_push_constants(
                command_buffer,
                self.kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.shape,
            );
            device.cmd_dispatch(command_buffer, self.groups[0], self.groups[1], 1);
        }
    }
}
