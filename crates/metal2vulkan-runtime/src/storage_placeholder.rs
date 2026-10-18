//! The placeholder images of the color attachments a render pass instance leaves out of its
//! subpass for the pixel interlock (`render_target::PassShape::storage`): what its framebuffer
//! holds for each, in place of the image its draws read and write as a storage image.
//!
//! The render pass of such an instance keeps each left-out color's description, so its framebuffer
//! holds a view for each. Vulkan bars a draw from writing "Memory backing image subresources used
//! as attachments in the current render pass ... in any way other than as an attachment"
//! (VUID-vkCmdDraw-None-06537), and does not say whether an attachment no subpass uses is so used.
//! So the framebuffer holds no view of the image the draws write: it holds a view of a placeholder
//! the device keeps for the attachment's format and sample count, which nothing reads or writes.
//! The description loads and stores nothing (`DONT_CARE`, which an attachment no subpass uses takes
//! in any case: "If an attachment is not used by any subpass, loadOp, storeOp, stencilStoreOp, and
//! stencilLoadOp will be ignored for that attachment, and no load or store ops will be
//! performed"), and its initial and final layouts are `GENERAL`, where the placeholder rests, so
//! the render pass moves no layout of it either (`render_target::final_pass`).
//!
//! - **One per format and sample count**, and another for each further color of them a
//!   framebuffer leaves out (so no two of its attachments alias), made at the first framebuffer
//!   that needs it (`WorkCounts::storage_placeholders_made`), as large as that framebuffer and of
//!   as many layers, in memory of its own where binds wait for queued work, else in a pooled
//!   block (`memory.rs`'s `Fit`). It rests in `GENERAL` like every runtime image: its
//!   move there is first work the device's next submission records (`texture/first_move.rs`), and
//!   no command ever reaches it, so it takes no use. A framebuffer larger, or of more layers,
//!   makes one that covers both in its place. The framebuffers that hold the one before keep it
//!   (`render_target::FramebufferObject`, which its holders keep until their command buffers
//!   complete); the device destroys it at the next placeholder it gives once none holds it, and
//!   every other as it is destroyed itself, idle.
//! - **Not for** a color whose draws read and write sample planes (`sample_planes.rs`), images
//!   apart from its own, nor the planes' anchor, which the subpass keeps (`PassShape::anchor`).
//! - **Counted**: `WorkCounts::storage_placeholder_attachments`, the framebuffer attachments that
//!   take one.

use super::device::lock;
use super::error::{vulkan, Error};
use super::memory::{Memory, MemoryClass, Placement};
use super::resource::ResourceKey;
use super::texture::first_move::{FirstMove, Fresh};
use super::Device;
use crate::{creation, release};
use ash::vk;
use metal2vulkan::vk::Format;
use std::ops::Range;
use std::sync::{Arc, Mutex, Weak};

/// The device's placeholders (the module docs).
#[derive(Default)]
pub(super) struct Placeholders {
    /// The one of each format, sample count and place among a framebuffer's left-out colors of
    /// them.
    kept: Vec<Arc<Placeholder>>,
    /// Those a larger one took the place of, until no framebuffer holds them.
    retired: Vec<Arc<Placeholder>>,
}

impl Placeholders {
    /// Destroy every placeholder, freeing its memory into `memory`. The device is idle, and no
    /// framebuffer is left to hold one.
    pub(super) fn destroy(&mut self, device: &ash::Device, memory: &mut Memory) {
        for placeholder in self.kept.drain(..).chain(self.retired.drain(..)) {
            placeholder.destroy(device, memory);
        }
    }
}

/// One placeholder: its image, the views framebuffers took of it, and its memory.
pub(super) struct Placeholder {
    image: vk::Image,
    format: Format,
    samples: u32,
    /// Its place among a framebuffer's left-out colors of its format and sample count.
    place: usize,
    extent: [u32; 2],
    layers: u32,
    placement: Placement,
    /// Each view, by its first layer and layer count.
    views: Mutex<Vec<((u32, u32), vk::ImageView)>>,
    /// Its first move, until a submission takes it (`texture/first_move.rs`).
    first: Mutex<Option<FirstMove>>,
}

impl Fresh for Placeholder {
    fn take(&self) -> Vec<FirstMove> {
        lock(&self.first).take().into_iter().collect()
    }

    fn give_back(&self, moves: Vec<FirstMove>) {
        let mut first = lock(&self.first);
        if first.is_none() {
            *first = moves.into_iter().next();
        }
    }

    fn rebindable(&self) -> Option<ResourceKey> {
        None
    }
}

impl Placeholder {
    /// Whether it is of `format` at `samples`.
    pub(super) fn is_of(&self, format: Format, samples: u32) -> bool {
        self.format == format && self.samples == samples
    }

    /// Whether it is of `format` at `samples`, at `place` among a framebuffer's left-out colors
    /// of them.
    fn is(&self, format: Format, samples: u32, place: usize) -> bool {
        self.is_of(format, samples) && self.place == place
    }

    /// Whether it covers a framebuffer attachment over `extent` and `layers` from its first.
    fn covers(&self, extent: [u32; 2], layers: u32) -> bool {
        self.extent[0] >= extent[0] && self.extent[1] >= extent[1] && self.layers >= layers
    }

    /// Its view of `layers`, made at the first framebuffer that asks for it.
    fn view(&self, device: &ash::Device, layers: Range<u32>) -> Result<vk::ImageView, Error> {
        let key = (layers.start, layers.len() as u32);
        let mut views = lock(&self.views);
        if let Some((_, view)) = views.iter().find(|(k, _)| *k == key) {
            return Ok(*view);
        }
        let view_type = match key.1 > 1 {
            true => vk::ImageViewType::TYPE_2D_ARRAY,
            false => vk::ImageViewType::TYPE_2D,
        };
        let info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(view_type)
            .format(self.format.into())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: key.0,
                layer_count: key.1,
            });
        // SAFETY: the image is live and bound; the layers are among its own (`covers`).
        let view = unsafe { device.create_image_view(&info, None) }.map_err(vulkan)?;
        views.push((key, view));
        Ok(view)
    }

    /// Destroy its views and image, and free its memory into `memory`. Nothing holds it.
    fn destroy(&self, device: &ash::Device, memory: &mut Memory) {
        for (_, view) in lock(&self.views).drain(..) {
            // SAFETY: no framebuffer holds the placeholder, so no framebuffer of a view is live.
            unsafe { release::object(device, view) };
        }
        // SAFETY: as above; no command reaches the image.
        unsafe { release::object(device, self.image) };
        memory.free(device, &self.placement);
    }
}

impl Device {
    /// A view of the device's placeholder of `format` at `samples` (the module docs) for the
    /// left-out color at `place` among a framebuffer's of them, of `layers` from its first, at
    /// least `extent` large, and the placeholder, which the framebuffer holds: made where the one
    /// kept there does not cover them.
    pub(super) fn storage_placeholder(
        &self,
        (format, samples, place): (Format, u32, usize),
        extent: [u32; 2],
        layers: Range<u32>,
    ) -> Result<(Arc<Placeholder>, vk::ImageView), Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let mut placeholders = lock(&shared.textures.storage_placeholders);
        // Those a larger one took the place of, once no framebuffer holds them: only the device
        // gives the placeholders out, under this lock, so one it alone holds stays so.
        if !placeholders.retired.is_empty() {
            let mut memory = lock(&shared.memory);
            placeholders.retired.retain(|placeholder| {
                let unheld = Arc::strong_count(placeholder) == 1;
                if unheld {
                    placeholder.destroy(raw_device, &mut memory);
                }
                !unheld
            });
        }
        let at = (placeholders.kept.iter()).position(|p| p.is(format, samples, place));
        let found = at
            .map(|at| &placeholders.kept[at])
            .filter(|p| p.covers(extent, layers.end));
        if let Some(placeholder) = found {
            let view = placeholder.view(raw_device, layers)?;
            return Ok((Arc::clone(placeholder), view));
        }
        // One that covers both the one before and this framebuffer.
        let (extent, total) = match at.map(|at| &placeholders.kept[at]) {
            Some(before) => (
                [
                    before.extent[0].max(extent[0]),
                    before.extent[1].max(extent[1]),
                ],
                before.layers.max(layers.end),
            ),
            None => (extent, layers.end),
        };
        let made = Arc::new(self.new_placeholder((format, samples, place), extent, total)?);
        shared.counts.storage_placeholder_made();
        let weak: Weak<Placeholder> = Arc::downgrade(&made);
        let fresh: Weak<dyn Fresh> = weak;
        shared.textures.first_moves.register(fresh);
        let view = made.view(raw_device, layers)?;
        placeholders.kept.push(Arc::clone(&made));
        if let Some(at) = at {
            let before = placeholders.kept.swap_remove(at);
            placeholders.retired.push(before);
        }
        Ok((made, view))
    }

    /// A new placeholder of `format` at `samples`, at `place` among a framebuffer's left-out
    /// colors of them, `extent` large and of `layers`, in memory of its own, its move to
    /// `GENERAL` waiting for the next submission.
    fn new_placeholder(
        &self,
        (format, samples, place): (Format, u32, usize),
        [width, height]: [u32; 2],
        layers: u32,
    ) -> Result<Placeholder, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().image(
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format.into())
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(layers)
                .samples(vk::SampleCountFlags::from_raw(samples))
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        );
        // SAFETY: the device is live; the format is a color attachment's of a pass at `samples`,
        // over a framebuffer of `extent` and `layers`, which the device renders.
        let image = unsafe { creation::image(raw_device, &info) }?;
        // SAFETY: `image` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(image) };
        let requirements = shared.textures.placement().place(requirements);
        let placement = match crate::memory::allocate_image_while_queued(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Private,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the image is unused.
                unsafe { release::object(raw_device, image) };
                return Err(error);
            }
        };
        // SAFETY: the placement meets the image's requirements and is not bound elsewhere.
        if let Err(error) =
            unsafe { raw_device.bind_image_memory(image, placement.memory, placement.offset) }
        {
            lock(&shared.memory).free(raw_device, &placement);
            // SAFETY: the image is unused.
            unsafe { release::object(raw_device, image) };
            return Err(vulkan(error));
        }
        Ok(Placeholder {
            image,
            format,
            samples,
            place,
            extent: [width, height],
            layers,
            placement,
            views: Mutex::new(Vec::new()),
            first: Mutex::new(Some(FirstMove::new(
                image,
                format,
                vk::ImageLayout::UNDEFINED,
                None,
            ))),
        })
    }
}
