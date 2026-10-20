//! Recycled images: an image the runtime made and placed in memory of its own
//! (`Device::placed_image`) is parked as its texture goes, not destroyed, and the next image of the
//! same making takes it, its `VkImage` and its memory, in place of a new one.
//!
//! - **Why.** MoltenVK 1.4.1 makes each `VkImage`'s Metal texture at its first use, adds it to
//!   its device's residency set and commits the set (`MVKImagePlane::getMTLTexture`,
//!   `MVKDevice::makeResident`), and removes it and commits again as the image is destroyed
//!   (`releaseMTLTexture`, `removeResidency`), each under one lock for the whole device. A guest's
//!   window server makes and releases textures of the same shapes every frame: about 55 commits a
//!   guest frame, the slow ones taking milliseconds and every other thread that makes or frees a
//!   resource meanwhile waiting on the lock. A parked image keeps its Metal texture, resident.
//! - **The same making.** An image is taken only for a making of the same plan (format, extent,
//!   levels, layers, samples, usage, flags, view formats: `ImagePlan`), texture type, memory class
//!   and sharing: its `vkCreateImage` parameters and its placement's memory are what a new one's
//!   would be.
//! - **What a texture sees.** Nothing of the image's last texture: the image is handed over as a
//!   new one is, to be moved from `UNDEFINED` (valid from any layout:
//!   `VUID-VkImageMemoryBarrier-oldLayout-01197`, "oldLayout must be VK_IMAGE_LAYOUT_UNDEFINED or
//!   the current layout"), and its maker's first move writes what a new texture reads: zero, or
//!   the copies in of a shadow, as for an image just made (`texture/first_move.rs`). An image made
//!   to start as its memory lies starts as the parked image's bytes lie, which are memory's as
//!   any placement's are.
//! - **When.** As the last holder of the image goes: every command buffer that used it completed
//!   (they hold it), so no GPU work reaches it. Not an image of a shareable texture, one that took
//!   its storage form, one presenting a stand-in, or one whose memory is not its placement alone.
//! - **Bounded.** At most [`MOST_PARKED`] images and [`MOST_PARKED_BYTES`] of their memory, and
//!   none parked longer than [`PARKED_FOR`]: past them the one parked earliest goes, as a texture's
//!   image went before (destroyed, its memory freed). The memory a parked image holds is the
//!   runtime's, never a program object's: no deallocator, range or use count waits on it.
//!   Counted: `WorkCounts::object_churn` (`images_recycled`, `images_parked`, `images_evicted`).

use super::super::memory::{Memory, MemoryClass, Placement};
use crate::image_plan::ImagePlan;
use crate::key_hash::KeyMap;
use crate::release;
use ash::vk;
use metal2vulkan::metal::TextureType;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// The most images parked at once: none in a build of `--cfg m2v_no_recycle`, the control that
/// measures what recycling changes (each image then goes as it is parked, as before).
pub(super) const MOST_PARKED: usize = if cfg!(m2v_no_recycle) { 0 } else { 32 };
/// The most memory parked images hold.
pub(super) const MOST_PARKED_BYTES: u64 = 128 << 20;
/// How long an image stays parked before it goes.
pub(super) const PARKED_FOR: Duration = Duration::from_secs(1);

/// What fixes an image's making ([`Device::placed_image`](crate::Device)): its parameters and its
/// placement's memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Making {
    pub(super) plan: ImagePlan,
    pub(super) texture_type: TextureType,
    /// The memory class asked.
    pub(super) class: MemoryClass,
    pub(super) families: Vec<u32>,
}

/// A parked image.
struct Parked {
    making: Making,
    raw: vk::Image,
    placement: Placement,
    since: Instant,
}

/// A device's parked images, and the makings of its live recyclable ones.
#[derive(Default)]
pub(crate) struct Recycler {
    /// Each live image `placed_image` made or handed out, by its `VkImage`.
    live: KeyMap<vk::Image, Making>,
    /// Parked images, the earliest first.
    parked: VecDeque<Parked>,
    /// Their memory.
    bytes: u64,
}

/// What a recycler hands back for destruction: images it no longer keeps.
pub(super) type Gone = Vec<(vk::Image, Placement)>;

impl Recycler {
    /// A parked image of `making`, the latest parked such, now live; and the images that stayed
    /// parked past [`PARKED_FOR`] at `now`, to destroy.
    pub(super) fn take(
        &mut self,
        making: &Making,
        now: Instant,
    ) -> (Option<(vk::Image, Placement)>, Gone) {
        let taken = self
            .parked
            .iter()
            .rposition(|parked| parked.making == *making)
            .and_then(|at| self.parked.remove(at))
            .map(|parked| {
                self.bytes -= parked.placement.size();
                self.live.insert(parked.raw, parked.making);
                (parked.raw, parked.placement)
            });
        (taken, self.stale(now))
    }

    /// `raw`, made for `making`, is live.
    pub(super) fn made(&mut self, raw: vk::Image, making: Making) {
        self.live.insert(raw, making);
    }

    /// Whether `raw` is a live image this recycler may park.
    pub(super) fn holds(&self, raw: vk::Image) -> bool {
        self.live.contains_key(&raw)
    }

    /// `raw`, live, goes: parked with `placement` (`Ok`, with the images that go to keep the
    /// bounds), or, where it is not a live image of this recycler's, handed back (`Err`).
    pub(super) fn park(
        &mut self,
        raw: vk::Image,
        placement: Placement,
        now: Instant,
    ) -> Result<Gone, (vk::Image, Placement)> {
        let Some(making) = self.live.remove(&raw) else {
            return Err((raw, placement));
        };
        self.bytes += placement.size();
        self.parked.push_back(Parked {
            making,
            raw,
            placement,
            since: now,
        });
        let mut gone = self.stale(now);
        while self.parked.len() > MOST_PARKED || self.bytes > MOST_PARKED_BYTES {
            let Some(oldest) = self.parked.pop_front() else {
                break;
            };
            self.bytes -= oldest.placement.size();
            gone.push((oldest.raw, oldest.placement));
        }
        Ok(gone)
    }

    /// `raw`, live, goes without parking (its image is not recyclable as it goes).
    pub(super) fn forget(&mut self, raw: vk::Image) {
        self.live.remove(&raw);
    }

    /// The images parked longer than [`PARKED_FOR`] at `now`, out of the recycler.
    fn stale(&mut self, now: Instant) -> Gone {
        let mut gone = Gone::new();
        while let Some(oldest) = self.parked.front() {
            if now.saturating_duration_since(oldest.since) < PARKED_FOR {
                break;
            }
            let Some(oldest) = self.parked.pop_front() else {
                break;
            };
            self.bytes -= oldest.placement.size();
            gone.push((oldest.raw, oldest.placement));
        }
        gone
    }

    /// Every parked image, destroyed and its memory freed, as the device goes.
    pub(crate) fn destroy(&mut self, device: &ash::Device, memory: &mut Memory) {
        self.bytes = 0;
        destroy(
            device,
            memory,
            self.parked.drain(..).map(|p| (p.raw, p.placement)),
        );
    }
}

/// Destroy each image of `gone` and free its memory, as a texture's image goes.
pub(super) fn destroy(
    device: &ash::Device,
    memory: &mut Memory,
    gone: impl IntoIterator<Item = (vk::Image, Placement)>,
) {
    for (raw, placement) in gone {
        // SAFETY: a parked image no command uses (its last holder went), and no view of it lives.
        unsafe { release::object(device, raw) };
        memory.free(device, &placement);
    }
}
