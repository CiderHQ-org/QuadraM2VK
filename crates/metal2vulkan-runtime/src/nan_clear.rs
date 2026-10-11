//! Clears of a NaN in a `half` or small-float channel, written as Metal writes them.
//!
//! **Metal** writes the canonical quiet NaN: `0x7e00` for a `half` NaN of either sign, `0x7e0`
//! and `0x3f0` for the 11- and 10-bit channels of `RG11B10Float`, the sign dropped (RRS probe 14,
//! an M4 on macOS 27.0.1 and the M3 Max on 26.7, alike, each plain and under the debug layer).
//! **Vulkan** converts a clear value to the attachment's format as the device likes, and the
//! devices measured differ there: RADV writes a `half` NaN as `0xfe00`, and lavapipe an
//! `RG11B10Float` NaN as `0x7c1` and `0x3e1` (lane `renders::clears_store_the_bytes_metal_stores`,
//! EVs RRS-lane29-radv and -lvp262). An integer clear converts nothing.
//!
//! So a color attachment whose `Clear` load holds such a NaN ([`clear_value::nan_texel`]) is
//! cleared before the pass instead: a scratch image of the unsigned-integer format of its texel
//! size (`R16Uint`, `R32Uint` or `RG32Uint`), of its samples and layers
//! (`attachment_stand_in::stand_in_as`), is cleared by `vkCmdClearColorImage` to the texel's
//! bits, which are copied into the attachment bit-wise (formats of one texel size) over the
//! clear's reach (`clear_reach.rs`), and the pass loads it. The scratch image is the pass's own,
//! held until its command buffer completes.
//!
//! **Kept sign**. Where Metal keeps the attachment's texture losslessly compressed, the
//! attachment's level is 16 texels or more a side, and the pass has no draw call
//! ([`keeps_nan_sign`]), its clear keeps a `half` NaN's sign, `0xfe00` for a negative one, in the
//! texel and every sample; elsewhere it drops it (EVs ZIN-zms4-m4, ZIN-zms6-m4 and ZIN-zms8-m4,
//! an M4 under macOS 27.0.1, sizes 1 to 256, 1x and 4x, `R16Float`, `RG16Float` and
//! `RGBA16Float`; any draw call drops it, one wholly off the target, of no area or whose
//! fragments all discard too; the consumer's M3 Max alike at 4x4). The scratch is mutable to the
//! attachment's format (`stand_in_as`), so a device that
//! copies between formats through a view copies so: MoltenVK's copy of a multisampled image
//! between formats that take no such view goes through a buffer, which holds no samples, and
//! left the NaN channels zero (EV ZIN-nc2-mvk4).
//!
//! **Resolves**. Metal's resolve of such a clear in a pass with no draw call stores the clear's
//! texel, its sign kept, where the sign is kept and Metal also keeps the resolve texture compressed
//! (at its level, of the attachment's extent); elsewhere it averages the samples, each NaN counted
//! zero ([`clear_value::resolved_nan_texel`]): a `PixelFormatView` or unoptimized resolve texture,
//! a level under 16 a side, or a later pass that loads the attachment and resolves it, alike (EV
//! ZIN-zms8-m4). No device's resolve gives either (each keeps the NaN, or zeroes it where Metal
//! keeps it: MoltenVK after the copy), so in such a pass the resolve texture's render area is
//! written after the pass's resolve, its copies included, the same way: a scratch image cleared to
//! that texel's bits and copied in. A pass with a draw call resolves by the device
//! (`device-rounded-resolve`).
//!
//! **32-bit floats**. Metal clears a `float` NaN as the double's conversion, its sign and top
//! payload bits kept, at every level (EV ZIN-zms5-m4), as the devices' clears do
//! ([`clear_value::color`] states the bits), so such a clear needs nothing before the pass. Its
//! resolve in a pass with no draw call keeps the clear's texel where the `half` NaN's resolve keeps
//! it, and makes `0x7fc00000` elsewhere (EVs ZIN-zms5-m4 and ZIN-zms7-m4). The exact resolve
//! (`color_resolve.rs`) makes `0x7fc00000` everywhere, so the kept resolve is written after the
//! pass as the `half` one's is ([`clear_value::float_nan_texel`]), on every device alike: where the
//! device's resolve kept the texel already, the write stores the same bits.
//!
//! **Copies that convert NaNs**. A device whose copy into a multisampled float image draws the
//! samples through that image's format (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`:
//! RADV, whose copies store every 16-bit float NaN as `0xfe00`) cannot copy the scratch's bits into
//! a multisampled attachment. There the clear is written through a view of the attachment's image
//! in the unsigned-integer format of its texel size instead: a render pass of that view alone,
//! loaded and stored, in which `vkCmdClearAttachments` clears the clear's reach to the texel's bits
//! (an integer clear converts nothing). A multisampled color render target the runtime makes of
//! such a format takes the view there where it costs the image nothing ([`view_candidate`]), as one
//! of `PixelFormatView` usage does; an image that takes none (a caller's, or one the view would
//! cost) is copied as elsewhere, announced once (`device-nan-encoding`). A resolve's write lands in
//! a single-sampled texture, whose copies keep the bits. The runtime's other copies into such an
//! image, a blit copy between textures and an attachment stand-in's, announce the same
//! ([`degrade_converting_copy`]); a blit copy between two images that both take the unsigned view
//! runs through it instead, exactly (`sample_planes::BitsCopy`).
//!
//! Refused (`NotImplemented(Render)`): such a clear of a memoryless attachment (no image to copy
//! into), of one whose sample planes stand for it (`sample_planes.rs` clears the planes by the
//! device's conversion), of one whose resolve texture reaches past it (`resolve.rs` clears that
//! reach likewise), of a 3D or 1D attachment (`attachment_stand_in::stand_in_as`), and of an
//! attachment whose image takes no transfer writes.

use super::attachment_stand_in::{self, StandInCopies};
use super::blit::TRANSFER;
use super::clear_reach;
use super::clear_value;
use super::command_buffer::CommandBuffer;
use super::degradation::Degradation;
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::framebuffer_fetch::views_in;
use super::image_plan::{compatible, ImagePlan};
use super::render_pass::{PassLayout, Target};
use super::render_target::{self, FramebufferObject, RenderPassObject, ATTACHMENT_STAGES};
use super::render_values::{LoadAction, StoreAction};
use super::resource::ResourceKey;
use super::texture::heap_tiled::metal_compresses;
use super::texture::Texture;
use super::tracker::{Access, Barrier, Use};
use super::Device;
use ash::vk;
use metal2vulkan::bind::PlanesCopy;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::{Format, ImageCreateFlags, ImageUsageFlags, NumericFormat};

/// The clears before a pass of its attachments whose `Clear` load holds a `half` or small-float
/// NaN (the module docs), with the uses they make and the copies into the attachments.
pub(super) struct NanClears {
    /// Each scratch image, its clear value and its layers.
    clears: Vec<(vk::Image, vk::ClearColorValue, u32)>,
    uses: Vec<Use<ResourceKey>>,
    copies: Option<StandInCopies>,
    /// The scratch images, held until the command buffer completes.
    scratch: Vec<Texture>,
    /// The clears through an unsigned-integer view of the attachment (the module docs), and the
    /// uses they make.
    views: Vec<ViewClear>,
    view_uses: Vec<Use<ResourceKey>>,
}

/// A NaN write through a view of the target's image in the unsigned-integer format of its texel
/// size: a render pass of that view alone, loaded and stored (`render_target::reach_pass`), in
/// which `vkCmdClearAttachments` clears `extent` from the origin, over `layers`, to `value`.
struct ViewClear {
    pass: RenderPassObject,
    framebuffer: FramebufferObject,
    extent: [u32; 2],
    layers: u32,
    value: vk::ClearColorValue,
}

impl NanClears {
    /// The clears of `layout`'s attachments, a pass with a draw call where `called`
    /// (`clear_reach::cleared`), each attachment cleared here then loaded in the pass; then,
    /// in a pass without one, the writes of their resolves after the pass (the module docs).
    /// `None` for either when there is none; refused as the module docs say.
    pub(super) fn plan(
        maker: &CommandBuffer,
        layout: &mut PassLayout,
        called: bool,
    ) -> Result<(Option<NanClears>, Option<NanClears>), Error> {
        let mut before = Writes::default();
        let mut after = Writes::default();
        let mut cleared = Vec::new();
        for (index, color) in layout.colors.iter().enumerate() {
            let Some(color) = color.as_ref().filter(|c| c.load == LoadAction::Clear) else {
                continue;
            };
            let pixel_format = color.target.texture.pixel_format();
            let signed = keeps_nan_sign(&color.target, called);
            // The resolve texture's level and slice, where a pass without a draw call resolves
            // the samples.
            let resolves = matches!(
                color.store,
                StoreAction::MultisampleResolve | StoreAction::StoreAndMultisampleResolve
            );
            let to = color
                .resolve
                .as_ref()
                .filter(|_| resolves && !called && color.target.samples > 1)
                .map(|r| r.copy_to.as_ref().unwrap_or(&r.attachment));
            let Some(texel) = clear_value::nan_texel(pixel_format, &color.clear, signed) else {
                // A 32-bit float NaN, which the device clears as Metal does: its resolve written
                // with the clear's texel where Metal keeps it, as below (the module docs).
                let kept = to.filter(|to| signed && keeps_nan_sign(to, called));
                if let (Some(to), Some(texel)) = (
                    kept,
                    clear_value::float_nan_texel(pixel_format, &color.clear),
                ) {
                    after.add(maker, to, texel, layout.extent)?;
                }
                continue;
            };
            let past_resolve = color.resolve.as_ref().is_some_and(|r| r.clear.is_some());
            if color.memoryless || color.planes.is_some() || past_resolve {
                return Err(Error::NotImplemented(RuntimeGap::Render));
            }
            let extent = match clear_reach::stores(color.store) {
                true => clear_reach::cleared(layout, color.target.extent, called),
                false => layout.extent,
            };
            before.add(maker, &color.target, texel, extent)?;
            cleared.push(index);
            let Some(to) = to else {
                continue;
            };
            let texel = match signed && keeps_nan_sign(to, called) {
                true => Some(texel),
                false => clear_value::resolved_nan_texel(pixel_format, &color.clear),
            };
            let texel = texel.ok_or_else(|| internal("a NaN clear without a resolved texel"))?;
            after.add(maker, to, texel, layout.extent)?;
        }
        let before = before.finish()?;
        for index in cleared {
            if let Some(color) = layout.colors.get_mut(index).and_then(Option::as_mut) {
                color.load = LoadAction::Load;
            }
        }
        Ok((before, after.finish()?))
    }

    /// Record the clears and the copies in `core`'s command buffer, between barriers from
    /// everything before and to everything after, and hold the scratch images until it
    /// completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let NanClears {
            clears,
            uses,
            copies,
            scratch,
            views,
            view_uses,
        } = self;
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |device, raw| {
            for (image, value, layers) in &clears {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: *layers,
                };
                // SAFETY: the encoder's command buffer is recording outside a render pass; the
                // scratch image is live until it completes (held below), rests in `GENERAL`,
                // takes transfer writes (`stand_in_as` with copies), and has one level and
                // `layers` layers of a color aspect; an unsigned-integer format takes `uint32`.
                unsafe {
                    device.cmd_clear_color_image(
                        raw,
                        *image,
                        vk::ImageLayout::GENERAL,
                        value,
                        &[range],
                    );
                }
            }
        });
        if let Some(copies) = copies {
            copies.record(core);
        }
        core.retain(scratch);
        if views.is_empty() {
            return;
        }
        core.barrier(Barrier::everything());
        core.command(ATTACHMENT_STAGES, &view_uses, |device, raw| {
            for clear in &views {
                let [width, height] = clear.extent;
                let area = vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                };
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(clear.pass.raw)
                    .framebuffer(clear.framebuffer.raw)
                    .render_area(area);
                let attachments = [vk::ClearAttachment {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    color_attachment: 0,
                    clear_value: vk::ClearValue { color: clear.value },
                }];
                let rects = [vk::ClearRect {
                    rect: area,
                    base_array_layer: 0,
                    layer_count: clear.layers,
                }];
                // SAFETY: the encoder's command buffer is recording outside a render pass; the
                // pass, its framebuffer and the attachment's image are live until it completes
                // (held below, and the texture by the encoder); the image rests in `GENERAL`, the
                // pass's layout; the area is the framebuffer's and the layers its own; the
                // subpass's one color attachment is the one cleared, of an unsigned-integer
                // format, which takes `uint32`.
                unsafe {
                    device.cmd_begin_render_pass(raw, &begin, vk::SubpassContents::INLINE);
                    device.cmd_clear_attachments(raw, &attachments, &rects);
                    device.cmd_end_render_pass(raw);
                }
            }
        });
        core.barrier(Barrier::everything());
        core.retain(views);
    }
}

/// A pass's NaN writes as they are planned ([`NanClears::plan`]): each a scratch image of an
/// unsigned-integer format, cleared to a texel's bits and copied into its target.
#[derive(Default)]
struct Writes {
    /// Each scratch image, its clear value and its layers.
    clears: Vec<(vk::Image, vk::ClearColorValue, u32)>,
    uses: Vec<Use<ResourceKey>>,
    /// Each scratch image, the target it is copied into, and the extent from the origin copied.
    pairs: Vec<(Target, Target, [u32; 2])>,
    /// The writes through an unsigned-integer view of the target, and the uses they make.
    views: Vec<ViewClear>,
    view_uses: Vec<Use<ResourceKey>>,
}

impl Writes {
    /// Write `texel`, an unsigned-integer format of its size and the clear value of its bits
    /// (`clear_value::nan_texel`), into `to` over `extent` from the origin; refused where `to`'s
    /// image takes no copies (`attachment_stand_in::check_copies`). On a device whose
    /// multisampled copies convert float NaNs, a multisampled `to` is written through its image's
    /// view in that format where the image takes one, else by the copy, announced (the module
    /// docs).
    fn add(
        &mut self,
        maker: &CommandBuffer,
        to: &Target,
        (format, value): (metal2vulkan::vk::Format, vk::ClearColorValue),
        extent: [u32; 2],
    ) -> Result<(), Error> {
        let device = maker.device();
        if to.samples > 1 && converts_nans(device) {
            if views_in(to.texture.image_plan()?, format)
                && renders_unsigned(device, format, to.samples)?
            {
                let viewed = Target {
                    format,
                    ..to.clone()
                };
                let (pass, framebuffer, made) =
                    render_target::reach_pass(device, &viewed, false, extent)?;
                self.view_uses.extend(made);
                self.views.push(ViewClear {
                    pass,
                    framebuffer,
                    extent,
                    layers: to.layers,
                    value,
                });
                return Ok(());
            }
            device.degrade(Degradation::DeviceNanEncoding, || {
                (
                    format!(
                        "a multisampled {:?} attachment's NaN clear",
                        to.texture.pixel_format()
                    ),
                    "its image takes no view in the unsigned-integer format of its texel size \
                     that the device renders at its sample count, and the device's copy into a \
                     multisampled float image stores each NaN as it converts it \
                     (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`), not with the \
                     sign and payload Metal's clear stores"
                        .to_owned(),
                )
            });
        }
        attachment_stand_in::check_copies(to)?;
        let scratch = attachment_stand_in::stand_in_as(maker, to, true, format)?;
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        self.uses.push(
            scratch
                .texture
                .image_use(0..1, 0..scratch.layers, write, true)?,
        );
        self.clears
            .push((scratch.texture.raw_image()?, value, scratch.layers));
        self.pairs.push((scratch, to.clone(), extent));
        Ok(())
    }

    /// The writes' clears and copies; `None` when there is none.
    fn finish(self) -> Result<Option<NanClears>, Error> {
        if self.pairs.is_empty() && self.views.is_empty() {
            return Ok(None);
        }
        let pairs: Vec<_> = self
            .pairs
            .iter()
            .map(|(from, to, extent)| (from, to, *extent))
            .collect();
        let copies = match pairs.is_empty() {
            true => None,
            false => Some(
                StandInCopies::plan(&pairs, false)?
                    .ok_or_else(|| internal("NaN writes without copies"))?,
            ),
        };
        let Writes {
            clears,
            uses,
            pairs,
            views,
            view_uses,
        } = self;
        Ok(Some(NanClears {
            clears,
            uses,
            copies,
            scratch: pairs.into_iter().map(|(from, _, _)| from.texture).collect(),
            views,
            view_uses,
        }))
    }
}

/// Whether Metal's NaN clear of `target`, in a pass with a draw call where `called`, keeps a
/// `half` NaN's sign (the module docs): its texture (a view's first texture) is one Metal keeps
/// losslessly compressed (`heap_tiled::metal_compresses`: not over a buffer, of no usage Metal
/// keeps uncompressed, optimized contents allowed), the attachment's level is 16 texels or more
/// a side, counted in pixels at any sample count, and the pass has no draw call. A resolve
/// texture's level of the same keeps the clear's texel through the pass's resolve.
pub(super) fn keeps_nan_sign(target: &Target, called: bool) -> bool {
    if called {
        return false;
    }
    let mut root = &target.texture;
    while let Some(parent) = root.parent_texture() {
        root = parent;
    }
    let level = target.extent.map(u64::from);
    !root.over_buffer()
        && metal_compresses(
            true,
            root.usage(),
            root.allow_gpu_optimized_contents(),
            level,
        )
}

/// Whether `format` is a color format of floats narrower than 32 bits (a `half`'s, a packed
/// small float's): those whose NaN clear the runtime writes ([`clear_value::nan_texel`]), and
/// whose NaNs a device whose multisampled copies convert float NaNs converts
/// (`driver_defects.rs`: RADV's copy exports them through a conversion).
pub(super) fn narrow_float(format: Format) -> bool {
    format.block().is_some_and(|block| {
        matches!(
            block.color,
            Some(NumericFormat::Sfloat | NumericFormat::Ufloat)
        ) && block.component_bits.is_none_or(|bits| bits < 32)
    })
}

/// Whether `device`'s copy into a multisampled image of `format` of `samples` samples converts
/// its NaNs: the device's multisampled copies convert float NaNs
/// (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`), and the format's are among those
/// converted ([`narrow_float`]).
pub(super) fn converting_copy(device: &Device, format: Format, samples: u32) -> bool {
    samples > 1 && narrow_float(format) && converts_nans(device)
}

/// Announce `device-nan-encoding` once for a copy into a multisampled image of `format` of
/// `samples` samples, `what` naming it, where the device's copy converts its NaNs
/// ([`converting_copy`]): they land as the device converts them, where Metal's copy keeps every
/// bit.
pub(super) fn degrade_converting_copy(
    device: &Device,
    format: Format,
    samples: u32,
    what: impl FnOnce() -> String,
) {
    if converting_copy(device, format, samples) {
        device.degrade(Degradation::DeviceNanEncoding, || {
            (
                what(),
                "the device's copy into a multisampled float image stores each NaN of a channel \
                 narrower than 32 bits as it converts it \
                 (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`), where Metal's copy \
                 keeps its sign and payload"
                    .to_owned(),
            )
        });
    }
}

/// Whether `device`'s driver copies into a multisampled float image through its format,
/// converting its NaNs (`RuntimeDriverDefect::MultisampledCopiesConvertFloatNans`).
fn converts_nans(device: &Device) -> bool {
    device
        .properties()
        .driver_defects
        .contains(&RuntimeDriverDefect::MultisampledCopiesConvertFloatNans)
}

/// `plan` mutable with a view in the unsigned-integer format of its texel size, through which a
/// NaN clear writes its samples (the module docs), where the device's multisampled copies convert
/// float NaNs: a multisampled color attachment that is not transient, of a format whose NaN
/// clear the runtime writes, whose view format is of its compatibility class and one the device
/// renders at its sample count ([`renders_unsigned`]). `None` elsewhere, and where the plan
/// takes the view already.
pub(super) fn view_candidate(
    plan: &ImagePlan,
    device: &Device,
) -> Result<Option<ImagePlan>, Error> {
    let samples = plan.samples.as_raw();
    let Some(view) = PlanesCopy::for_format(plan.format).map(PlanesCopy::format) else {
        return Ok(None);
    };
    let candidate = samples > 1
        && converts_nans(device)
        && narrow_float(plan.format)
        && compatible(plan.format, view, false)
        && plan.usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
        && !plan.usage.contains(ImageUsageFlags::TRANSIENT_ATTACHMENT)
        && !views_in(plan, view)
        && renders_unsigned(device, view, samples)?;
    if !candidate {
        return Ok(None);
    }
    let mut candidate = plan.clone();
    candidate.flags |= ImageCreateFlags::MUTABLE_FORMAT;
    if candidate.view_formats.is_empty() {
        candidate.view_formats.push(plan.format);
    }
    candidate.view_formats.push(view);
    Ok(Some(candidate))
}

/// Whether `device` makes an optimal 2D image of `format`, an unsigned-integer format, as a color
/// attachment of `samples` samples (its image format properties): what a render pass through a
/// view in that format at that count needs. The framebuffer's integer sample counts are only
/// the counts every such format has (RADV lists 1 alone there and makes these images at 4).
fn renders_unsigned(device: &Device, format: Format, samples: u32) -> Result<bool, Error> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format.into())
        .extent(vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::from_raw(samples))
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);
    let limits = device.image_limits(&info, &[])?;
    Ok(limits
        .sample_counts
        .contains(vk::SampleCountFlags::from_raw(samples)))
}
