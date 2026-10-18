//! Interop with the caller's own Vulkan work: the raw handles of the runtime's objects, and what
//! the device keeps for the caller's images and semaphores.
//!
//! - **Raw handles.** Unsafe accessors give the Vulkan objects behind a texture
//!   ([`Texture::vk_image`]), a buffer ([`Buffer::vk_buffer`]), a shared event
//!   ([`SharedEvent::vk_semaphore`]), the device ([`Device::raw`]) and a queue
//!   ([`CommandQueue::with_vk_queue`]). The runtime keeps owning them: the caller never destroys
//!   one, and orders its own use of one against the runtime's through semaphores or completion.
//!   A queue's `VkQueue` is lent under the runtime's lock of it, which every submission takes,
//!   since Vulkan requires a queue's submissions to be externally synchronized.
//! - **The caller's images** (`texture/wrapped.rs`): textures over images the caller made, which
//!   the device's batches move between the caller's layout and `GENERAL`.
//! - **The caller's buffers** (`wrapped_buffer.rs`): buffers over `VkBuffer`s the caller made and
//!   bound to its own memory (an imported dma-buf, for example), `Shared` at a host address the
//!   caller gives, or `Private`.
//! - **The caller's binary semaphores** (`CommandBuffer::encode_wait_for_vk_semaphore` and
//!   `encode_signal_vk_semaphore`): Vulkan has no query for a semaphore's type, so the runtime
//!   refuses what it can tell is no binary semaphore of the device: a null handle, and its own
//!   timeline semaphores (events' and queues'). Every other contract is the caller's.

use super::buffer::ReadApart;
use super::device::relay::By;
use super::event::EventRef;
use super::texture::wrapped::Wrapped;
use super::{Buffer, CommandQueue, Device, SharedEvent, Texture};
use ash::vk;
use std::collections::HashSet;
use std::sync::Mutex;

/// Why a semaphore cannot be one of the caller's binary semaphores.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemaphoreProblem {
    /// A null handle.
    Null,
    /// One of the runtime's own timeline semaphores: an event's ([`SharedEvent::vk_semaphore`])
    /// or a queue's.
    Timeline,
}

/// What a device keeps for interop: the caller's images whose batches move them between layouts,
/// and the timeline semaphores its events hold.
#[derive(Default)]
pub(super) struct Interop {
    pub(super) wrapped: Wrapped,
    timelines: Mutex<HashSet<vk::Semaphore>>,
}

impl Interop {
    /// An event's timeline semaphore was made.
    pub(super) fn timeline_made(&self, semaphore: vk::Semaphore) {
        self.timelines().insert(semaphore);
    }

    /// An event's timeline semaphore is being destroyed.
    pub(super) fn timeline_gone(&self, semaphore: vk::Semaphore) {
        self.timelines().remove(&semaphore);
    }

    fn timelines(&self) -> std::sync::MutexGuard<'_, HashSet<vk::Semaphore>> {
        self.timelines.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// What the runtime can tell of `semaphore` as the caller's binary semaphore on a device whose
/// own timeline semaphores are `queues` (its queues') and `events` (its events').
pub(super) fn semaphore_problem(
    semaphore: vk::Semaphore,
    queues: impl IntoIterator<Item = vk::Semaphore>,
    events: &HashSet<vk::Semaphore>,
) -> Option<SemaphoreProblem> {
    if semaphore == vk::Semaphore::null() {
        Some(SemaphoreProblem::Null)
    } else if events.contains(&semaphore) || queues.into_iter().any(|queue| queue == semaphore) {
        Some(SemaphoreProblem::Timeline)
    } else {
        None
    }
}

impl Device {
    /// Why `semaphore` cannot be the caller's binary semaphore of this device, if the runtime can
    /// tell.
    pub(super) fn binary_semaphore_problem(
        &self,
        semaphore: vk::Semaphore,
    ) -> Option<SemaphoreProblem> {
        let shared = self.shared();
        semaphore_problem(
            semaphore,
            shared.queues.iter().map(|queue| queue.timeline),
            &shared.interop.timelines(),
        )
    }
}

impl Texture {
    /// No Metal counterpart (an interop hook): the texture's `VkImage`, the image of the texture
    /// and of every view of it, made by the runtime or the caller's
    /// ([`Device::new_texture_with_vk_image`](super::Device::new_texture_with_vk_image)). `None`
    /// where the texels are a buffer's bytes the runtime copies into and out of an image of its
    /// own (a texture buffer, a texture over a no-copy buffer's host pages, or a texture over
    /// another buffer that does not alias it), or another image's texels it so copies (a view of
    /// another component layout on a device whose views reinterpret none). A texture with a depth
    /// twin gives its `R32_SFLOAT` image, which holds its texels; its comparisons read the twin,
    /// which a write into the image does not reach.
    ///
    /// A texture over a buffer's bytes whose image aliases them gives a linear image of
    /// `VK_IMAGE_TILING_LINEAR`, bound to the buffer's memory, the runtime's own: such a texture's
    /// rows lie one after another and its image's subresource holds no byte but its texels. A
    /// texture over a no-copy buffer's host pages (imported by their pointer or as dma-bufs) gives
    /// none: an image bound to them would start `VK_IMAGE_LAYOUT_UNDEFINED`, whose first move
    /// Vulkan lets read and write every byte bound to it (it is a shadow, `shadowed-texture`). A
    /// texture placed in a heap over host pages imported as dma-bufs (`HostMemory::with_dma_buf`)
    /// gives an image of `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT` with `DRM_FORMAT_MOD_LINEAR`:
    /// ask its layout of its memory plane (`VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT`, not the color
    /// aspect), and its modifier with `vkGetImageDrmFormatModifierPropertiesEXT`.
    ///
    /// # Safety
    ///
    /// The image stays its owner's: the caller does not destroy one the runtime made or bind its
    /// memory, and uses it only while the texture lives, ordered against the runtime's work on
    /// it. An image the runtime made rests in `VK_IMAGE_LAYOUT_GENERAL`, owned by the device's
    /// queue family, whenever no batch of the runtime uses it, and the caller leaves it so; a
    /// wrapped image follows its layout contract
    /// ([`VkImageLayouts`](super::VkImageLayouts)).
    ///
    /// The texture keeps the image for good: a fetched render target whose image takes
    /// `STORAGE` later no longer does once its image was handed out.
    pub unsafe fn vk_image(&self) -> Option<vk::Image> {
        if self.shadow().is_some() {
            return None;
        }
        self.exported_image().ok()
    }
}

impl Buffer {
    /// The buffer's `VkBuffer`, and where the buffer's byte 0 lies in it: 0, but for a no-copy
    /// buffer bound below an unaligned pointer. No Metal counterpart: an interop hook. A buffer the
    /// runtime placed in its own memory has no `SHADER_DEVICE_ADDRESS` usage of its own (it takes
    /// its address from its memory block's buffer), so the caller asks it for no device address.
    ///
    /// # Safety
    ///
    /// The `VkBuffer` stays the runtime's: the caller does not destroy it, and uses it only while
    /// the buffer lives, ordered against the runtime's work on it.
    pub unsafe fn vk_buffer(&self) -> (vk::Buffer, u64) {
        self.at_for(0, ReadApart::VkBuffer)
    }
}

impl SharedEvent {
    /// The event's timeline semaphore. Its counter is the runtime's signal index, which the
    /// runtime maps to Metal's values, not the event's value.
    ///
    /// # Safety
    ///
    /// The semaphore stays the runtime's: the caller never signals or destroys it, and uses it
    /// only while the event lives.
    pub unsafe fn vk_semaphore(&self) -> vk::Semaphore {
        EventRef::from(self).shared().semaphore()
    }
}

impl CommandQueue {
    /// `f` with the `VkQueue` this queue submits to and its family's index, under the runtime's
    /// lock of that `VkQueue`, which every submission to it takes: the caller's own submissions
    /// and presents (`vkQueuePresentKHR`) go there, externally synchronized as Vulkan requires.
    /// On MoltenVK, where the runtime relays the driver's command-buffer failures, it submits no
    /// work with a fence of its own after `f` under the same lock, so a failure of the caller's
    /// submissions stays the caller's. There the runtime's frees of its memory wait for the work
    /// in flight (`RuntimeDriverDefect::FreedAddressMemoryFailsPendingWork`), the caller's
    /// submissions in `f` among it: that submission of no work also signals a value the use took
    /// before `f` ran, and memory that goes while it is pending is freed once it completed.
    ///
    /// # Safety
    ///
    /// The caller does not keep the handle past `f`, and `f` makes no call into the runtime and
    /// waits for nothing that needs a later submission of the runtime's (the scheduler submits
    /// only under this lock).
    pub unsafe fn with_vk_queue<R>(&self, f: impl FnOnce(vk::Queue, u32) -> R) -> R {
        let shared = self.device().shared();
        let family = shared.queue_family_of(self.vk_queue());
        // Where the queue's failures are relayed, a mark after `f`'s submissions keeps a failure
        // of theirs the caller's (`device/relay.rs`); where freed memory waits for the work in
        // flight, it waits for that mark (`freed_memory.rs`).
        let used = shared.use_queue(self.vk_queue(), By::Program, |queue| f(queue, family));
        if let Some(parking) = shared.parking.get() {
            parking.watch(self.device());
        }
        used
    }
}
