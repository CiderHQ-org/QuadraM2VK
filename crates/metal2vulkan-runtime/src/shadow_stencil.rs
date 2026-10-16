//! The shader-side stencil's shadow: what a render encoder keeps for a pipeline whose fragment
//! function tests and writes the stencil itself (`GraphicsState::shadow_stencil`, on a device
//! without `VK_EXT_shader_stencil_export`), as [`ShadowStencilUse`]'s contract asks of a consumer.
//!
//! - **The shadow** is a 2D `R8_UINT` storage image of the pipeline's samples and the stencil
//!   attachment's extent, whose texel `(x, y)` (each sample) is pixel `(x, y)` (that sample), in
//!   `GENERAL` like every runtime image, made at the pass's first such draw and bound where the
//!   plan writes `WriteSource::ShadowStencil`.
//! - **The instances.** Such draws record into render pass instances of their own (`render.rs`):
//!   consecutive ones share an instance, and a draw of another pipeline after them starts the
//!   next. Before such an instance begins, the attachment's stencil aspect is copied into the
//!   shadow, and after it ends, the shadow is copied back, outside the render pass; during it the
//!   pipelines' stencil test is off, so the pass never writes the stencil. An instance that runs
//!   the pass's load actions and clears the stencil clears the shadow to the clear value instead
//!   of copying the stencil in, since the attachment's clear runs only as the instance begins.
//!   A stencil that resolves is resolved by one more instance, of no draws, after the copy back.
//! - **The copies.** Of one sample, through a staging buffer: the stencil aspect (one byte per
//!   texel) into the buffer, then the buffer into the shadow, and back the same way. Of more,
//!   one `vkCmdCopyImage` each way between the stencil aspect and the shadow, which
//!   `VK_KHR_maintenance8` allows (Vulkan copies no multisampled image to a buffer; the pipeline
//!   required maintenance8 then). Each copy runs between barriers from everything before and to
//!   everything after. They cover the render area.
//! - **Without colour copies** where the driver copies the multisampled stencil wrong
//!   ([`copies_break`]): the texels of a combined depth/stencil attachment and of the shadow
//!   differ in size, and a driver with `RuntimeDriverDefect::MultisampledDepthStencilColourCopies`
//!   (`driver_defects.rs` says how each breaks: lavapipe writes sample 0 alone, Ampere GA10x loses
//!   the device) copies between them wrong. Such a shadow of more than one sample is copied by a
//!   kernel that reads the stencil aspect through a sampled view and by stencil draws that write
//!   it back a bit at a time (`stencil_copy.rs`), exact where the attachment's image is sampled
//!   and the device runs them; it is refused where the image is not sampled (`Unsupported`,
//!   `Feature::StencilExport`, naming the defect) and where the device lacks what they need
//!   (`Unsupported`, naming what it lacks).
//! - **The in-shader stencil test's stand-in**: a draw whose fragment function runs the draw's
//!   stencil test itself, beside its depth test (`ShaderTestsUse::stencil`), holds the stencil in
//!   this same shadow, made, filled, copied and refused as above, its refusals naming
//!   `Feature::FramebufferFetch`. Its instances are the shader-side depth test's too
//!   (`depth_stand_in.rs`), so the shadow is filled and copied back around each, as the depth's
//!   stand-in is. A stencil-only `S8_UINT` attachment's texels are the shadow's size, which
//!   lavapipe and the RTX 3090 copy as Metal's stencil reads (`runtime_lane::shader_stencil` at 4
//!   samples, measured), and one sample's copies go through the buffer.
//!
//! [`ShadowStencilUse`]: metal2vulkan::pipeline::ShadowStencilUse

use super::attachment_stand_in;
use super::blit::{reads, writes, TRANSFER};
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::error::{internal, Error};
use super::image_plan::{sample_count, ImagePlan};
use super::render_pass::Target;
use super::render_target;
use super::resource::ResourceKey;
use super::stencil_copy::{self, ShadowPlanes};
use super::texture::Texture;
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::ResourceOptions;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::error::{Feature, RuntimeDriverDefect, Unsupported};
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags};
use std::collections::BTreeSet;

/// The defect, of a driver with `defects`, that copies a shadow of `samples` samples to or from
/// the stencil aspect of an attachment of `format` wrong, if any: more than one sample of a
/// combined depth/stencil format, whose stencil maintenance8's depth-stencil/colour copies move
/// to and from the `R8_UINT` shadow, on a driver with
/// `RuntimeDriverDefect::MultisampledDepthStencilColourCopies` or
/// `MultisampledStencilColourCopies`.
pub(super) fn copies_break(
    defects: &BTreeSet<RuntimeDriverDefect>,
    format: Format,
    samples: u32,
) -> Option<RuntimeDriverDefect> {
    let combined = super::texture::image_aspects(format).contains(ImageAspectFlags::DEPTH);
    if samples <= 1 || !combined {
        return None;
    }
    super::depth_stencil_scratch::copies_defect(defects, ImageAspectFlags::STENCIL)
}

/// A pass's shadow of its stencil, and the buffer its single-sample copies pass through.
#[derive(Clone)]
pub(super) struct ShadowStencil {
    /// The shadow, as the storage target its view is made of.
    shadow: Target,
    /// One sample: one byte per texel of the render area. More: `None`, copied image to image.
    staging: Option<Buffer>,
    /// Where the driver copies the samples wrong ([`copies_break`]): the copies by a kernel and
    /// draws in their place (`stencil_copy.rs`).
    planes: Option<ShadowPlanes>,
    /// The render area, from the origin: what the copies cover.
    area: [u32; 2],
}

impl ShadowStencil {
    /// The shadow of `stencil`, the pass's depth/stencil target, over the render area `area`, of
    /// the format and samples `used` (a `ShadowStencilUse`'s, or, the stencil stand-in's of a
    /// `ShaderTestsUse`, which this shadow is). `stencil` must have a stencil aspect and the use's
    /// samples, and the use's format must be `R8_UINT` (link states neither otherwise). Refused
    /// (`NotImplemented(Render)`) where `stencil`'s image does not copy both ways (a caller's image
    /// without transfer usage). Where the device's driver copies its samples wrong
    /// ([`copies_break`]), the shadow is copied by a kernel and draws (`stencil_copy.rs`), and
    /// refused, typed (naming `feature`, the compile layer's for the use), where those cannot run:
    /// naming the defect where `stencil`'s image is not sampled, and what the device lacks where it
    /// lacks what they need.
    pub(super) fn new(
        maker: &CommandBuffer,
        (format, samples): (Format, u32),
        feature: Feature,
        stencil: &Target,
        area: [u32; 2],
    ) -> Result<ShadowStencil, Error> {
        let device = maker.device();
        if format != Format::R8_UINT
            || samples != stencil.samples
            || !super::texture::image_aspects(stencil.format).contains(ImageAspectFlags::STENCIL)
        {
            return Err(internal(
                "a shader-side stencil whose shadow the pass's stencil cannot be",
            ));
        }
        // The stencil is copied both ways (a caller's image may lack the usage).
        attachment_stand_in::check_copies(stencil)?;
        let defects = &device.properties().driver_defects;
        let broken = copies_break(defects, stencil.format, stencil.samples);
        if let Some(defect) = broken.filter(|_| !matches!(stencil_copy::sampled(stencil), Ok(true)))
        {
            return Err(Error::Unsupported(Unsupported::runtime_driver_defect(
                feature, defect,
            )));
        }
        let [width, height] = stencil.extent;
        let samples = stencil.samples;
        let samples_flag = sample_count(samples)
            .ok_or_else(|| internal("an attachment of a sample count Vulkan lacks"))?;
        let descriptor = TextureDescriptor {
            texture_type: if samples > 1 {
                TextureType::Type2DMultisample
            } else {
                TextureType::Type2D
            },
            pixel_format: PixelFormat::R8Uint,
            width: width.into(),
            height: height.into(),
            sample_count: samples.into(),
            usage: TextureUsage::RENDER_TARGET,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        };
        let plan = ImagePlan {
            pixel_format: PixelFormat::R8Uint,
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::TYPE_2D,
            format: Format::R8_UINT,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: samples_flag,
            usage: ImageUsageFlags::STORAGE
                | ImageUsageFlags::TRANSFER_SRC
                | ImageUsageFlags::TRANSFER_DST,
            view_formats: Vec::new(),
            mapping: FormatMapping::direct(Format::R8_UINT, ImageAspectFlags::COLOR),
            unavailable: TextureUsage::SHADER_READ
                | TextureUsage::SHADER_WRITE
                | TextureUsage::SHADER_ATOMIC
                | TextureUsage::RENDER_TARGET
                | TextureUsage::PIXEL_FORMAT_VIEW,
        };
        let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
        let staging = if samples == 1 {
            let [w, h] = area;
            Some(device.own_buffer(u64::from(w) * u64::from(h), ResourceOptions::private())?)
        } else {
            None
        };
        let shadow = Target {
            texture,
            level: 0,
            layer: 0,
            layers: 1,
            format: Format::R8_UINT,
            samples,
            extent: [width, height],
        };
        let planes = match broken {
            Some(_) => Some(ShadowPlanes::new(device, stencil, &shadow, area, feature)?),
            None => None,
        };
        Ok(ShadowStencil {
            shadow,
            staging,
            planes,
            area,
        })
    }

    /// The storage view the plan's `WriteSource::ShadowStencil` binds.
    pub(super) fn view(&self) -> Result<vk::ImageView, Error> {
        render_target::storage_view(&self.shadow)
    }

    /// A draw's uses of the shadow: the fragment shader reads and writes it.
    pub(super) fn draw_uses(&self) -> Result<Vec<Use<ResourceKey>>, Error> {
        let stage = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let read = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ);
        let write = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE);
        let texture = &self.shadow.texture;
        Ok(vec![
            texture.image_use(0..1, 0..1, read, false)?,
            texture.image_use(0..1, 0..1, write, true)?,
        ])
    }

    /// The copies' uses, both ways, between the shadow and `stencil`.
    pub(super) fn copy_uses(&self, stencil: &Target) -> Result<Vec<Use<ResourceKey>>, Error> {
        if let Some(planes) = &self.planes {
            return planes.uses(stencil, &self.shadow);
        }
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let levels = stencil.level..stencil.level + 1;
        let layers = stencil.layer..stencil.layer + 1;
        let texture = &self.shadow.texture;
        let mut uses = vec![
            texture.image_use(0..1, 0..1, read, false)?,
            texture.image_use(0..1, 0..1, write, true)?,
            stencil
                .texture
                .image_use(levels.clone(), layers.clone(), read, false)?,
            stencil.texture.image_use(levels, layers, write, true)?,
        ];
        if let Some(staging) = &self.staging {
            let length = staging.length();
            uses.push(writes(staging, 0..length));
            uses.push(reads(staging, 0..length));
        }
        Ok(uses)
    }

    /// What the copies go through, which the command buffer holds until it completes.
    pub(super) fn held(&self) -> (Texture, Option<Buffer>, Option<ShadowPlanes>) {
        (
            self.shadow.texture.clone(),
            self.staging.clone(),
            self.planes.clone(),
        )
    }

    /// The copies between the shadow and `stencil`, the pass's depth/stencil target at the end
    /// of its encoder, whose stencil the pass clears to `clear` where it does.
    pub(super) fn copies(
        &self,
        stencil: &Target,
        clear: Option<u32>,
    ) -> Result<ShadowCopies, Error> {
        let [width, height] = self.area;
        Ok(ShadowCopies {
            shadow: self.shadow.texture.raw_image()?,
            attachment: stencil.texture.raw_image()?,
            staging: self.staging.as_ref().map(|staging| staging.at(0)),
            stencil_layers: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::STENCIL,
                mip_level: stencil.level,
                base_array_layer: stencil.layer,
                layer_count: 1,
            },
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            clear,
            planes: self.planes.clone(),
        })
    }
}

/// The recorded form of a pass's copies between its stencil and its shadow
/// ([`ShadowStencil::copies`]).
pub(super) struct ShadowCopies {
    shadow: vk::Image,
    attachment: vk::Image,
    /// The staging buffer and the offset of its range, for a shadow of one sample.
    staging: Option<(vk::Buffer, u64)>,
    stencil_layers: vk::ImageSubresourceLayers,
    extent: vk::Extent3D,
    /// The pass's stencil clear value, where it clears the stencil.
    clear: Option<u32>,
    /// The copies by a kernel and draws, where the driver copies the samples wrong.
    planes: Option<ShadowPlanes>,
}

impl ShadowCopies {
    /// Record, outside a render pass, the shadow's filling before an instance of such draws:
    /// cleared to the stencil clear value where the instance runs the pass's load actions
    /// (`first`) and they clear the stencil, since the attachment's clear runs only as the
    /// instance begins; else the copy of the stencil aspect. Between barriers from everything
    /// before and to everything after.
    ///
    /// # Safety
    ///
    /// `raw` is `device`'s command buffer, recording outside a render pass, which only this
    /// encoder records; the shadow, its staging buffer and the stencil's image live until it
    /// completes and rest in `GENERAL`.
    pub(super) unsafe fn record_in(&self, device: &Vk, raw: vk::CommandBuffer, first: bool) {
        Barrier::everything().record(device, raw);
        match self.clear.filter(|_| first) {
            Some(value) => {
                let colour = vk::ClearColorValue {
                    uint32: [value & 0xff, 0, 0, 0],
                };
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let general = vk::ImageLayout::GENERAL;
                device.cmd_clear_color_image(raw, self.shadow, general, &colour, &[range]);
            }
            None => self.copy(device, raw, true),
        }
        Barrier::everything().record(device, raw);
    }

    /// Record, outside a render pass, the copy of the shadow back into the stencil aspect after
    /// an instance of such draws, between barriers from everything before and to everything
    /// after.
    ///
    /// # Safety
    ///
    /// As for [`ShadowCopies::record_in`].
    pub(super) unsafe fn record_out(&self, device: &Vk, raw: vk::CommandBuffer) {
        Barrier::everything().record(device, raw);
        self.copy(device, raw, false);
        Barrier::everything().record(device, raw);
    }

    /// Record the copy of the render area between the stencil aspect and the shadow: into the
    /// shadow (`inward`) or back.
    ///
    /// # Safety
    ///
    /// As for [`ShadowCopies::record_in`].
    unsafe fn copy(&self, device: &Vk, raw: vk::CommandBuffer, inward: bool) {
        match (&self.planes, inward) {
            (Some(planes), true) => return planes.record_in(device, raw),
            (Some(planes), false) => return planes.record_out(device, raw),
            (None, _) => {}
        }
        let general = vk::ImageLayout::GENERAL;
        let shadow_layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let ((from, from_layers), (to, to_layers)) = if inward {
            (
                (self.attachment, self.stencil_layers),
                (self.shadow, shadow_layers),
            )
        } else {
            (
                (self.shadow, shadow_layers),
                (self.attachment, self.stencil_layers),
            )
        };
        match self.staging {
            Some((buffer, offset)) => {
                let region = |layers| vk::BufferImageCopy {
                    buffer_offset: offset,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: layers,
                    image_offset: vk::Offset3D::default(),
                    image_extent: self.extent,
                };
                device.cmd_copy_image_to_buffer(raw, from, general, buffer, &[region(from_layers)]);
                Barrier::everything().record(device, raw);
                device.cmd_copy_buffer_to_image(raw, buffer, to, general, &[region(to_layers)]);
            }
            None => {
                let region = vk::ImageCopy {
                    src_subresource: from_layers,
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: to_layers,
                    dst_offset: vk::Offset3D::default(),
                    extent: self.extent,
                };
                device.cmd_copy_image(raw, from, general, to, general, &[region]);
            }
        }
    }
}
