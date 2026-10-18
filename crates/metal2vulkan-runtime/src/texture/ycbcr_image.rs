//! The image of a two-plane YCbCr texture over a surface's two planes (`ycbcr.rs`): an image of its
//! sampled and storage usage, a shadow whose copy in converts the planes
//! (`ShadowSource::Converted`). A texture read converted is an `R16G16B16A16_SFLOAT` image of the
//! converted texels, which the conversion writes; an `_sRGB` twin's an `R32G32B32A32_SFLOAT` one
//! (its steps of 1/8192 are no halves above 1/4); one read as codes an `A2B10G10R10_UNORM_PACK32`
//! image of 10-bit codes or an `R8G8B8A8_UNORM` one of 8-bit codes, which the conversion writes
//! packed through an `R32_UINT` view (a store of its bits, where a store of `code / (2^b - 1)`
//! through the image's own format would be the device's conversion to the code).

use super::{image_error, IdUse, Image, Storage, Texture};
use crate::address_map::AddressModeMap;
use crate::binding_view::VkView;
use crate::buffer_texture::Texels;
use crate::device::lock;
use crate::error::{internal, Error};
use crate::image_plan::plan_image;
use crate::shadow::{ShadowBuffer, ShadowCopies, ShadowSource};
use crate::shadow_merge::ScratchSpares;
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use crate::tracker::UseRange;
use crate::values::ResourceOptions;
use crate::view_plan::ViewRecord;
use crate::ycbcr::{YCbCrConversion, YCbCrWriteBack};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{YCbCrCodes, YCbCrOutput, YCbCrPlanes};
use metal2vulkan::device::FormatUsage;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::{vk_format, ycbcr_format, ycbcr_renders, ChromaSubsampling};
use metal2vulkan::metal::{PixelFormat, SamplerAddressMode, TextureType, TextureUsage};
use metal2vulkan::vk::{
    ComponentMapping, Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags,
    ImageUsageFlags, ImageViewType,
};
use std::ops::Range;
use std::sync::Arc;

/// The formats of a texture's image and of the view the conversion writes it through, by what
/// its texels read: the converted texels as halves, every channel a multiple of 1/2048 up to 1,
/// which a half holds exactly; the sRGB twins' as floats, every channel a multiple of 1/8192,
/// which a half does not hold above 1/4; 10-bit codes as `RGB10A2Unorm` texels, alpha 3, and
/// 8-bit codes as `RGBA8Unorm` ones, alpha 255, written as their packed words.
fn image_formats(codes: YCbCrCodes) -> [Format; 2] {
    match (codes.output, codes.bits) {
        (YCbCrOutput::Codes, 8) => [Format::R8G8B8A8_UNORM, Format::R32_UINT],
        (YCbCrOutput::Codes, _) => [Format::A2B10G10R10_UNORM_PACK32, Format::R32_UINT],
        (YCbCrOutput::Srgb, _) => [Format::R32G32B32A32_SFLOAT; 2],
        _ => [Format::R16G16B16A16_SFLOAT; 2],
    }
}

/// The modes Metal samples a two-plane YCbCr texture with, by its codes: a 4:2:0 or 4:2:2 texture
/// samples repeat, mirrored repeat and mirror-clamp-to-edge as clamp-to-edge, a 4:4:4 one as
/// themselves; one read as codes samples clamp-to-zero and clamp-to-border-color as
/// clamp-to-edge too, under every border color, where the others read the border (its alpha 1,
/// as `alpha_reads_one` reads every texel's).
fn ycbcr_modes(codes: YCbCrCodes) -> AddressModeMap {
    let edge = SamplerAddressMode::ClampToEdge;
    let wrapped = match codes.subsampling {
        ChromaSubsampling::Chroma444 => AddressModeMap::IDENTITY,
        _ => AddressModeMap::IDENTITY
            .with(SamplerAddressMode::Repeat, edge)
            .with(SamplerAddressMode::MirrorRepeat, edge)
            .with(SamplerAddressMode::MirrorClampToEdge, edge),
    };
    match codes.output {
        YCbCrOutput::Codes => wrapped
            .with(SamplerAddressMode::ClampToZero, edge)
            .with(SamplerAddressMode::ClampToBorderColor, edge),
        _ => wrapped,
    }
}

impl Device {
    /// The two-plane YCbCr texture of `descriptor` (checked as `layout`) over `texels`, the luma
    /// plane, for Metal's `call`, whose planes `planes` lie in `span` of the texels' buffer: its
    /// image of the texels Metal reads and the conversion (`ycbcr.rs`). It reads alpha 1 and maps
    /// the address modes Metal samples otherwise (`ycbcr_modes`). Refused `NotImplemented` where
    /// the device holds no such image, and as `new_texture` refuses the image or the view the
    /// conversion writes it through.
    pub(crate) fn new_converted_texture(
        &self,
        call: &'static str,
        texels: Texels,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        id_use: IdUse,
        (planes, span): (&YCbCrPlanes, Range<u64>),
    ) -> Result<Texture, Error> {
        let descriptor = TextureDescriptor {
            alpha_reads_one: true,
            address_mode_map: ycbcr_modes(planes.codes),
            ..*descriptor
        };
        let texels = Arc::new(texels);
        let storage = self.converted_storage(call, texels, &descriptor, layout, (planes, span))?;
        let record = ViewRecord::of_descriptor(layout, &descriptor);
        Ok(Texture::made(self, storage, record, None, id_use))
    }

    /// The storage of a two-plane YCbCr texture of `descriptor` (checked as `layout`) over
    /// `texels`, for Metal's `call`, whose planes `planes` lie in `span` of the texels' buffer:
    /// its image and the conversion's shadow (`new_converted_texture`, and a view as the
    /// texture's `_sRGB` twin, `Texture::converted_view`).
    fn converted_storage(
        &self,
        call: &'static str,
        texels: Arc<Texels>,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        (planes, span): (&YCbCrPlanes, Range<u64>),
    ) -> Result<Storage, Error> {
        let pf = layout.pixel_format;
        let gap =
            || Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming_format(pf));
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let plan = plan_image(
            layout,
            descriptor.usage | TextureUsage::SHADER_READ,
            false,
            &facts,
        )
        .map_err(|refusal| image_error(call, refusal))?;
        let [format, written] = image_formats(planes.codes);
        let sampled = facts
            .optimal(format)
            .contains(FormatFeatureFlags2::SAMPLED_IMAGE);
        let storage = |f| {
            facts
                .optimal(f)
                .contains(FormatFeatureFlags2::STORAGE_IMAGE)
        };
        if plan.format != format || !sampled || !storage(written) {
            return Err(gap());
        }
        let mut plan = crate::image_plan::ImagePlan {
            usage: plan.usage | ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE,
            ..plan
        };
        // Written through a view of another format: mutable, viewed as that format too (an image
        // already viewed as any format keeps its empty list), and of extended usage where the
        // image's own format has no storage.
        if written != format {
            let listed = !plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT);
            plan.flags |= ImageCreateFlags::MUTABLE_FORMAT;
            if !storage(format) {
                plan.flags |= ImageCreateFlags::EXTENDED_USAGE;
            }
            if listed || !plan.view_formats.is_empty() {
                if plan.view_formats.is_empty() {
                    plan.view_formats.push(format);
                }
                plan.view_formats.push(written);
            }
        }
        let hazard = texels.options().hazard_tracking_mode()?;
        let image = self.new_image(plan, layout, descriptor, hazard)?;
        // The conversion's set holds it as long as the image (`Image::own_view`).
        let view = image.own_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format: written,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        })?;
        let (raw, at) = texels.buffer.at(span.start);
        let conversion = YCbCrConversion::new(self, (planes, span.clone()), at, view)?;
        let (bytes, tracked) = texels.buffer.tracked(span);
        let copies = ShadowCopies {
            source: ShadowSource::Converted {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                alias: texels.buffer.alias().map(|alias| alias.key),
                conversion: Box::new(conversion),
                pixel_format: pf,
            },
            shadow_key: image.identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            tracked: texels.tracked(),
            // Metal drops a function's writes to the texture.
            drops_writes: true,
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
        };
        Ok(Storage::Shadow {
            image: image.shared(),
            texels,
            copies: Arc::new(copies),
        })
    }
}

impl Texture {
    /// The storage of a view presenting `record` of this texture in its `_sRGB` twin's format
    /// (`view_plan::check_format` admits `YCBCR8_420_2P` as `YCBCR8_420_2P_sRGB` and
    /// `YCBCR10_420_2P` as `YCBCR10_420_2P_sRGB`, as Metal makes them, its usage `ShaderRead`
    /// alone): a converted image of the view's format over the same planes, which reads what
    /// Metal's view reads, a texture of that format over the surface's reads bit for bit. `None`
    /// for a texture whose texels are no conversion of planes. Refused as
    /// a texture of the view's format over the planes is.
    pub(super) fn converted_view(
        &self,
        call: &'static str,
        record: &ViewRecord,
    ) -> Option<Result<Storage, Error>> {
        let Storage::Shadow { texels, copies, .. } = &self.0.storage else {
            return None;
        };
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            return None;
        };
        let pf = record.pixel_format;
        let (planes, span) = conversion.source();
        let Some(codes) = ycbcr_format(pf).and_then(YCbCrCodes::of) else {
            return Some(Err(internal(
                "a converted view of a format no conversion reads",
            )));
        };
        let planes = YCbCrPlanes::new(
            planes.size,
            planes.luma,
            planes.chroma,
            (planes.range, planes.matrix),
            codes,
        );
        let [w, h] = planes.size.map(u64::from);
        let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
        descriptor.usage = record.usage;
        descriptor.swizzle = record.swizzle;
        descriptor.resource_options = texels.options();
        descriptor.alpha_reads_one = true;
        descriptor.address_mode_map = ycbcr_modes(codes);
        let device = self.device();
        Some(
            descriptor
                .check_over_surface(device.properties().texture_rules())
                .map_err(|_| internal("a converted view's descriptor past its texture's"))
                .and_then(|layout| {
                    device.converted_storage(
                        call,
                        Arc::clone(texels),
                        &descriptor,
                        &layout,
                        (&planes, span),
                    )
                }),
        )
    }
}

impl Texture {
    /// The write shadow of a 4:4:4 two-plane YCbCr texture whose writes Metal stores into its
    /// planes (`metal::format::ycbcr_writes_stored`), for Metal's `call`: an `RGBA32Uint` image of
    /// the texture's size, which a function writes each texel to as its red, green and blue bits
    /// and the mark 1 (`StorageImageClass::YCbCrWrite`), and its copies, a clear in and the
    /// encoding of the marked texels' codes back into the planes (`ShadowSource::YCbCrWrite`,
    /// `ycbcr.rs`). Refused (`Unsupported`, `NotImplemented`, `Feature::Format`, naming the
    /// format) for a texture over plane 1 of its surface, whose writes are not measured, and one
    /// whose texels are no conversion of planes; as `new_texture` refuses the image.
    pub(super) fn ycbcr_write_shadow(
        &self,
        call: &'static str,
    ) -> Result<(Image, ShadowCopies), Error> {
        let pf = self.0.record.pixel_format;
        let gap =
            || Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming_format(pf));
        let Storage::Shadow { texels, copies, .. } = &self.0.storage else {
            return Err(gap());
        };
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            return Err(gap());
        };
        if conversion.over_plane_one() {
            return Err(gap());
        }
        let (planes, span) = conversion.source();
        let device = self.device();
        let [width, height] = planes.size;
        let layout = TextureLayout {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::RGBA32Uint,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: 1,
        };
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        let facts = device
            .shared()
            .textures
            .facts(device.compile_device().profile());
        let plan = plan_image(&layout, usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        // Written by the function, read by the encoding, and cleared.
        let needed = ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_DST;
        if plan.format != Format::R32G32B32A32_UINT || !plan.usage.contains(needed) {
            return Err(gap());
        }
        let [w, h] = planes.size.map(u64::from);
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA32Uint, w, h, false);
        descriptor.usage = usage;
        // The device's alone: cleared at every copy in, never read by the CPU.
        descriptor.resource_options = ResourceOptions::private();
        let hazard = texels.options().hazard_tracking_mode()?;
        let image = device.new_image(plan, &layout, &descriptor, hazard)?;
        // The write-back's set holds it as long as the image (`Image::own_view`).
        let view = image.own_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format: Format::R32G32B32A32_UINT,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        })?;
        let (raw, at) = texels.buffer.at(span.start);
        let write_back = YCbCrWriteBack::new(device, (&planes, span.clone()), at, (view, false))?;
        let (bytes, tracked) = texels.buffer.tracked(span);
        let copies = ShadowCopies {
            source: ShadowSource::YCbCrWrite {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                alias: texels.buffer.alias().map(|alias| alias.key),
                image: image.raw.get(),
                write_back: Box::new(write_back),
                pixel_format: pf,
            },
            shadow_key: image.identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            tracked: texels.tracked(),
            drops_writes: false,
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
        };
        Ok((image, copies))
    }
}

impl Texture {
    /// The render shadow of a two-plane YCbCr texture a pass stores as Metal does
    /// (`metal::format::ycbcr_renders`), for Metal's `call`, made at its first need and kept
    /// with the texture: a texture of the texture's format and size over its planes whose image
    /// is `R32G32B32A32_SFLOAT`, the color attachment a pass renders the values it leaves in
    /// each texel into (`FormatEmulation::YCbCrRendered`), and its copies, a clear in and the
    /// store of every texel's codes back into the planes (`ShadowSource::YCbCrWrite` of a
    /// store, `ycbcr.rs`), which leaves the texture's converted texels stale. Refused
    /// (`Unsupported`, `NotImplemented`, `Feature::Format`, naming the format) for a texture
    /// over plane 1 of its surface, one of a size its chroma blocks do not tile, and one whose
    /// texels are no conversion of planes, none of whose stores is measured; as `new_texture`
    /// refuses the image.
    pub(crate) fn ycbcr_render_shadow(&self, call: &'static str) -> Result<Texture, Error> {
        let mut held = lock(&self.0.render_shadow);
        if let Some(shadow) = held.as_ref() {
            return Ok(shadow.clone());
        }
        let shadow = self.new_render_shadow(call)?;
        *held = Some(shadow.clone());
        Ok(shadow)
    }

    /// A new render shadow ([`ycbcr_render_shadow`](Self::ycbcr_render_shadow)).
    fn new_render_shadow(&self, call: &'static str) -> Result<Texture, Error> {
        let pf = self.0.record.pixel_format;
        let gap =
            || Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming_format(pf));
        let Storage::Shadow { texels, copies, .. } = &self.0.storage else {
            return Err(gap());
        };
        let ShadowSource::Converted { conversion, .. } = &copies.source else {
            return Err(gap());
        };
        if !ycbcr_renders(pf) || conversion.over_plane_one() {
            return Err(gap());
        }
        let (planes, span) = conversion.source();
        let [width, height] = planes.size;
        let [across, down] = planes.codes.subsampling.halved();
        if (across && width % 2 != 0) || (down && height % 2 != 0) {
            return Err(gap());
        }
        let device = self.device();
        let profile = device.compile_device().profile();
        let float = TextureLayout {
            texture_type: TextureType::Type2D,
            pixel_format: PixelFormat::RGBA32Float,
            extent: [width, height, 1],
            levels: 1,
            layers: 1,
            samples: 1,
        };
        let usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_WRITE;
        let facts = device.shared().textures.facts(profile);
        let plan = plan_image(&float, usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        // Rendered, read by the store through a view of its bits, and cleared.
        let needed = ImageUsageFlags::COLOR_ATTACHMENT
            | ImageUsageFlags::STORAGE
            | ImageUsageFlags::TRANSFER_DST;
        let mapping =
            vk_format(pf, &[FormatUsage::ColorAttachment].into(), profile).map_err(|_| gap())?;
        let bits = Format::R32G32B32A32_UINT;
        if plan.format != Format::R32G32B32A32_SFLOAT
            || mapping.format != plan.format
            || !plan.usage.contains(needed)
            || !facts
                .optimal(bits)
                .contains(FormatFeatureFlags2::STORAGE_IMAGE)
        {
            return Err(gap());
        }
        // The texture's format, held in floats; viewed as their bits too.
        let plan = crate::image_plan::ImagePlan {
            pixel_format: pf,
            flags: plan.flags | ImageCreateFlags::MUTABLE_FORMAT,
            view_formats: vec![plan.format, bits],
            mapping,
            ..plan
        };
        let layout = TextureLayout {
            pixel_format: pf,
            ..float
        };
        let [w, h] = planes.size.map(u64::from);
        let mut descriptor = TextureDescriptor::texture_2d(pf, w, h, false);
        descriptor.usage = TextureUsage::RENDER_TARGET;
        // The device's alone: cleared at every copy in, never read by the CPU.
        descriptor.resource_options = ResourceOptions::private();
        let hazard = texels.options().hazard_tracking_mode()?;
        let image = device.new_image(plan, &layout, &descriptor, hazard)?;
        // The write-back's set holds it as long as the image (`Image::own_view`).
        let view = image.own_view(&VkView {
            view_type: ImageViewType::TYPE_2D,
            format: bits,
            components: ComponentMapping::IDENTITY,
            aspect: ImageAspectFlags::COLOR,
            base_level: 0,
            level_count: 1,
            base_layer: 0,
            layer_count: 1,
            usage: ImageUsageFlags::STORAGE,
        })?;
        let (raw, at) = texels.buffer.at(span.start);
        let write_back = YCbCrWriteBack::new(device, (&planes, span.clone()), at, (view, true))?;
        let (bytes, tracked) = texels.buffer.tracked(span);
        let copies = ShadowCopies {
            source: ShadowSource::YCbCrWrite {
                buffer: ShadowBuffer::of(&texels.buffer, raw),
                key: bytes.key,
                bytes: tracked,
                alias: texels.buffer.alias().map(|alias| alias.key),
                image: image.raw.get(),
                write_back: Box::new(write_back),
                pixel_format: pf,
            },
            shadow_key: image.identity.key,
            shadow_range: UseRange::Image {
                aspects: vk::ImageAspectFlags::COLOR,
                levels: 0..1,
                layers: 0..1,
            },
            tracked: texels.tracked(),
            drops_writes: false,
            backing: texels.backing(),
            merge_scratch: ScratchSpares::default(),
        };
        let storage = Storage::Shadow {
            image: image.shared(),
            texels: Arc::clone(texels),
            copies: Arc::new(copies),
        };
        let record = ViewRecord::of_descriptor(&layout, &descriptor);
        Ok(Texture::made(device, storage, record, None, IdUse::Runtime))
    }
}
