//! `generate_mipmaps` of the integer formats, as plain Metal makes them: each level's texels picked
//! from the level above, which copies give exactly on every device ([`PickPlan`] lays the copies
//! out).
//!
//! **The command.** One blit command, between barriers from the commands before it and to the
//! ones after it ([`BlitCommandEncoder::command`]), over a scratch buffer of the runtime's own:
//! the view's first level, over its layers, copied into the scratch, then every later level copied
//! out of it, a column (or a row's texel) a region. The scratch is retained until the command
//! buffer completes. Where the scratch would pass the device's `max_buffer_length`, the call is
//! refused (`NotImplemented(MipmapGeneration)`): a blit's nearest filter puts its sample on the
//! edge between two texels, which the device's precision decides.

use super::blit::BlitCommandEncoder;
use super::error::Error;
use super::mipmap_plan::PickPlan;
use super::sparse::emulated::Written;
use super::texture::{texture_aspects, Texture};
use super::tracker::{Access, Barrier};
use super::values::ResourceOptions;
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::TextureType;

impl BlitCommandEncoder {
    /// Record `texture`'s mipmaps as `plan`'s picks. The caller checked Metal's rules, and
    /// `plan_mipmaps` the image's transfers and texel.
    pub(super) fn pick_mipmaps(
        &mut self,
        texture: &Texture,
        plan: &PickPlan,
        read: Access,
        write: Access,
    ) -> Result<(), Error> {
        let record = texture.record();
        let device = self.core().device().clone();
        if plan.bytes > device.properties().max_buffer_length {
            return Err(Error::Unsupported(Unsupported::not_implemented(
                Feature::MipmapGeneration,
            )));
        }
        let scratch = device.own_buffer(plan.bytes, ResourceOptions::private())?;
        let (raw, at) = scratch.at(0);
        let (base_level, base_layer, layers) =
            (record.base_level, record.base_layer, record.layer_count);
        let all_layers = base_layer..base_layer + layers;
        let uses = [
            texture.image_use(base_level..base_level + 1, all_layers.clone(), read, false)?,
            texture.image_use(
                base_level + 1..base_level + record.level_count,
                all_layers.clone(),
                write,
                true,
            )?,
        ];
        let (image, aspects) = (texture.raw_image()?, texture_aspects(texture.image_plan()?));
        let subresource = |level: u32| vk::ImageSubresourceLayers {
            aspect_mask: aspects,
            mip_level: level,
            base_array_layer: base_layer,
            layer_count: layers,
        };
        let extent = |[width, height, depth]: [u32; 3]| vk::Extent3D {
            width,
            height,
            depth,
        };
        let mut first = record.extent();
        if record.texture_type != TextureType::Type3D {
            first[2] = 1;
        }
        let copy_in = vk::BufferImageCopy {
            buffer_offset: at,
            buffer_row_length: 0,
            buffer_image_height: plan.image_height,
            image_subresource: subresource(base_level),
            image_offset: vk::Offset3D::default(),
            image_extent: extent(first),
        };
        let picks: Vec<vk::BufferImageCopy> = plan
            .picks
            .iter()
            .map(|pick| vk::BufferImageCopy {
                buffer_offset: at + pick.buffer_offset,
                buffer_row_length: pick.row_length,
                buffer_image_height: pick.image_height,
                image_subresource: subresource(pick.level),
                image_offset: vk::Offset3D {
                    x: pick.origin[0] as i32,
                    y: pick.origin[1] as i32,
                    z: pick.origin[2] as i32,
                },
                image_extent: extent(pick.extent),
            })
            .collect();
        self.command(&uses, |device, command_buffer| {
            // SAFETY: the command buffer is recording, outside a render pass; the scratch and the
            // texture are retained until it completes; the image is in the `GENERAL` layout with
            // transfer usages and features (`plan_mipmaps` checked them); every region lies
            // inside the scratch and its level (`plan_picks`), and the picks read what the copy in
            // wrote, past a barrier.
            unsafe {
                device.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    raw,
                    &[copy_in],
                );
                Barrier::everything().record(device, command_buffer);
                if !picks.is_empty() {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        raw,
                        image,
                        vk::ImageLayout::GENERAL,
                        &picks,
                    );
                }
            }
        })?;
        self.core().retain(scratch);
        self.core().retain(texture.clone());
        let written = (1..record.level_count)
            .map(|n| {
                let written = Written {
                    level: base_level + n,
                    layers: all_layers.clone(),
                    origin: [0; 3],
                    size: first.map(|axis| (axis >> n).max(1)),
                };
                (aspects, written)
            })
            .collect();
        self.scrub(texture, written)
    }
}
