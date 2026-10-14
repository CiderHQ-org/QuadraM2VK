//! The layouts a device's driver refused a texture's `DRM_FORMAT_MOD_LINEAR` image at Metal's row
//! pitch, kept so that it is not asked again (`texture.rs`'s `host_pitched_image`).
//!
//! A texture over a no-copy buffer's host pages whose rows the driver's own linear image lays out
//! otherwise than Metal is tried as a `DRM_FORMAT_MOD_LINEAR` image laid out at Metal's row pitch,
//! which the driver may refuse: it refuses the layout
//! (`VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT`; RADV of Mesa 26.2.4 at a pitch that is
//! not a multiple of 256 bytes), lays the plane out at another pitch than asked (NVIDIA 615.71.09
//! below its own linear image's row), or asks a dedicated allocation of the image, which one bound
//! in a buffer's import is not (RADV at every other pitch). The driver answers the same image the
//! same way while the device lives, so a refusal is kept by everything the image is made of that a
//! texture varies: its format, extent, usage, flags and view formats, and the row pitch. A consumer
//! making the same surface's texture each frame asks the driver once.
//!
//! Not kept by the pitch's class (the largest power of two dividing it) alone: measured
//! (`runtime_lane::surface_texture_costs::pitch_classes`, nine formats, four usages, five widths,
//! two pitches of each class from 16 to 4096 bytes), NVIDIA takes a pitch of any class at least its
//! own linear image's row and refuses one below it, so one class holds both answers.
//!
//! Kept only for a refusal of the image itself, never one that may pass another time (the driver
//! out of memory, which the caller reports as it is); and only for a plan of one slice and depth,
//! whose plane has no array pitch. At most [`KEPT`] refusals are kept; past them the driver is
//! asked again each time.

use crate::device::lock;
use crate::image_plan::ImagePlan;
use metal2vulkan::vk::{Format, ImageCreateFlags, ImageUsageFlags};
use std::collections::HashSet;
use std::sync::Mutex;

/// The most refusals a device keeps.
pub(crate) const KEPT: usize = 256;

/// What a `DRM_FORMAT_MOD_LINEAR` image at a row pitch is made of, of what a texture varies.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PitchedLayout {
    format: Format,
    extent: [u32; 3],
    usage: ImageUsageFlags,
    flags: ImageCreateFlags,
    view_formats: Vec<Format>,
    row_pitch: u64,
}

impl PitchedLayout {
    /// The layout of an image of `plan` with rows `row_pitch` apart; `None` for a plan of several
    /// slices or depths, whose refusals are not kept.
    pub(crate) fn of(plan: &ImagePlan, row_pitch: u64) -> Option<PitchedLayout> {
        (plan.layers == 1 && plan.extent[2] == 1).then(|| PitchedLayout {
            format: plan.format,
            extent: plan.extent,
            usage: plan.usage,
            flags: plan.flags,
            view_formats: plan.view_formats.clone(),
            row_pitch,
        })
    }
}

/// A device's kept refusals of `DRM_FORMAT_MOD_LINEAR` layouts at a row pitch.
#[derive(Debug, Default)]
pub(crate) struct PitchRefusals {
    refused: Mutex<HashSet<PitchedLayout>>,
}

impl PitchRefusals {
    /// Whether the driver refused `layout` before.
    pub(crate) fn refused(&self, layout: &PitchedLayout) -> bool {
        lock(&self.refused).contains(layout)
    }

    /// Keep the driver's refusal of `layout`, while fewer than [`KEPT`] are kept.
    pub(crate) fn keep(&self, layout: PitchedLayout) {
        let mut refused = lock(&self.refused);
        if refused.len() < KEPT {
            refused.insert(layout);
        }
    }
}
