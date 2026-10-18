//! A `Shared` or `Managed` texture's CPU copies on the device's staging queue, a copy engine
//! (`device/staging.rs`): where the device has one and the texture copies none on the host from
//! memory the host caches (NVIDIA: an RTX 3090's host-transfer image lies in resizable BAR,
//! `texture/host_copy.rs`), Metal's `replaceRegion` and `getBytes` of such a texture wait for no
//! GPU work of the program's, as Metal's do.
//!
//! - **Which images** ([`Device::stages`]). A new `Shared` or `Managed` texture's image in memory
//!   of its own, of a color format (a queue of transfers alone copies no depth or stencil aspect,
//!   so a depth texture and its depth twin, keep the host's or the submission's copies), made for
//!   copies both ways, of one sample (Vulkan copies nothing into a multisampled image; a clear
//!   writes its zeros). It is made concurrent among the device's families and the staging queue's,
//!   so its uses change no owner: measured on the RTX 3090, 50 clears and 50 copies of a 1080p
//!   image cost the same either way (0.24 ms and 1.32 ms).
//! - **Made.** It is moved to `GENERAL` and zeroed on the staging queue as it is made, from a
//!   staging buffer of zeros (a copy engine clears nothing), waited for
//!   (`FirstMove::record_alone`): no batch finds its move waiting.
//! - **Copied.** `replace_region` packs the caller's bytes into a staging buffer and copies them
//!   into the image on the staging queue; `get_bytes` copies the image's texels into one there
//!   and lays them out. Each waits for its own submission alone: the copy and one submission's
//!   round trip, whatever the program has in flight. A later batch finds the bytes written (the
//!   submission completed before the call returned); `get_bytes` reads what every command buffer
//!   the host saw complete wrote.
//! - **A race is the program's,** as on the host (`host_copy.rs`): a command in flight meanwhile
//!   reads or writes texels the copy may or may not have moved. A copy runs under the lock a
//!   detach takes, so it never reaches a `VkImage` that went; while the image presents a
//!   stand-in, or a host call's copy into it still waits for a batch (made while it presented
//!   one), the call takes the submission's path.
//! - **With a decoded twin** (`decoded_twin.rs`; a BC1 to BC5 texture on NVIDIA's drivers) too:
//!   `get_bytes` copies its BC image's blocks, which hold its texels, and `replace_region` copies
//!   the blocks into it, then decodes them on the CPU and copies the texels into the twin on the
//!   staging queue at once (`Image::decode_written`): the twin is made to take the staging queue's
//!   copies where its BC image does.

use super::{image_aspects, transfer_barrier, Image, ImagePlan, Staging};
use crate::device::{lock, Device};
use crate::error::Error;
use ash::vk;
use metal2vulkan::metal::format::is_depth_stencil;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::vk::{ImageAspectFlags, ImageUsageFlags};

impl Device {
    /// Whether a new `Shared` or `Managed` texture's image of `plan`, for a texture of
    /// `pixel_format`, copies on the staging queue where it takes no host copies from host-cached
    /// memory (the module docs): the device has a staging queue, the image is of color alone and
    /// not a depth texture's, and it is made for copies both ways.
    pub(super) fn stages(&self, plan: &ImagePlan, pixel_format: PixelFormat) -> bool {
        let both = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        self.has_staging_queue()
            && image_aspects(plan.format) == ImageAspectFlags::COLOR
            && !is_depth_stencil(pixel_format)
            && plan.usage.contains(both)
    }
}

impl Image {
    /// Whether its CPU copies may run on the staging queue: it was made so, and it has no depth
    /// twin (a decoded twin's decode is staged, the module docs).
    pub(super) fn takes_staging_copies(&self) -> bool {
        self.staging_copies && self.twin.is_none()
    }

    /// Whether nothing of a host call's copy into it waits for a batch: no copy staged while it
    /// presented a stand-in (`texture/first_move.rs`).
    fn nothing_staged(&self) -> bool {
        let at: *const Image = self;
        !(self.device.shared().textures.first_moves).texels_wait(at.cast())
    }

    /// Copy `packed`, laid out as `copy` reads it from a buffer, into the image on the staging
    /// queue; the image [takes staging copies](Self::takes_staging_copies). Whether it did: not
    /// while it presents a stand-in.
    pub(super) fn staging_write(
        &self,
        copy: vk::BufferImageCopy,
        packed: &[u8],
    ) -> Result<bool, Error> {
        let detached = lock(&self.detached);
        if detached.is_some() || !self.nothing_staged() {
            return Ok(false);
        }
        let mut staging = Staging::new(&self.device, packed.len() as u64)?;
        staging.bytes().copy_from_slice(packed);
        let (buffer, image) = (staging.raw, self.raw.get());
        self.device.submit_staging(|raw, command_buffer, _| {
            transfer_barrier(raw, command_buffer, true);
            // SAFETY: the staging command buffer is recording; the staging buffer and the image
            // are live (the detach lock keeps the image's own `VkImage` presented), and the image
            // rests in `GENERAL` (moved as it was made), shared with the staging queue.
            unsafe {
                raw.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    &[copy],
                );
            }
            transfer_barrier(raw, command_buffer, false);
        })?;
        drop(detached);
        Ok(true)
    }

    /// Copy the image's texels `copy` reaches into `bytes` on the staging queue, laid out as
    /// `copy` writes them to a buffer; the image [takes staging
    /// copies](Self::takes_staging_copies). Whether it did: not while it presents a stand-in.
    pub(super) fn staging_read(
        &self,
        copy: vk::BufferImageCopy,
        bytes: &mut [u8],
    ) -> Result<bool, Error> {
        let detached = lock(&self.detached);
        if detached.is_some() || !self.nothing_staged() {
            return Ok(false);
        }
        let mut staging = Staging::new(&self.device, bytes.len() as u64)?;
        let (buffer, image) = (staging.raw, self.raw.get());
        self.device.submit_staging(|raw, command_buffer, _| {
            transfer_barrier(raw, command_buffer, true);
            // SAFETY: the staging command buffer is recording; the staging buffer and the image
            // are live (the detach lock keeps the image's own `VkImage` presented), and the image
            // rests in `GENERAL` (moved as it was made), shared with the staging queue.
            unsafe {
                raw.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    buffer,
                    &[copy],
                );
            }
            transfer_barrier(raw, command_buffer, false);
        })?;
        drop(detached);
        bytes.copy_from_slice(staging.bytes());
        Ok(true)
    }
}
