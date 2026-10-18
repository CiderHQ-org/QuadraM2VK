//! The depth stand-in: what a render encoder keeps for a pipeline whose fragment function runs the
//! draw's depth test itself (`GraphicsState::shader_tests`), as [`ShaderTestsUse`]'s contract asks
//! of a consumer, the shader-side stencil's shadow's twin (`shadow_stencil.rs`).
//!
//! - **The stand-in** is an `R32_SFLOAT` image of the depth attachment's extent, in `GENERAL`
//!   like every runtime image, holding pixel `(x, y)`'s depth at texel `(x, y)` as the pipeline's
//!   [`StandInView`] says: a 2D image of the pipeline's samples (single-sample at one), or, on a
//!   device without multisampled storage images, a single-sample 2D array of one layer per
//!   sample. Made at the pass's first such draw and bound where the plan writes
//!   `WriteSource::DepthStandIn`.
//! - **The instances.** Such draws record into render pass instances of their own (`render.rs`):
//!   consecutive ones share an instance, and a draw of another pipeline after them starts the
//!   next. During one the pipelines' depth test is `ALWAYS` without writes, and the stencil test
//!   is off (`render.rs`'s `draw_fixed`), so the pass never changes the depth: the function is
//!   the test. Before each such instance the stand-in is filled, and after it the stand-in is
//!   copied back into the depth aspect, outside the render pass, between barriers from
//!   everything before and to everything after. An instance that runs the pass's load actions
//!   fills it from the depth's load action instead of the attachment: the clear depth where it
//!   clears (the attachment's clear runs only as the instance begins), nothing where Metal's
//!   depth is undefined (`DontCare`), the attachment where it loads. So a memoryless depth, which
//!   renders into its backing in such a pass (`attachment_stand_in.rs`), is read only where an
//!   earlier instance of the pass rendered it. A depth that resolves is resolved by one more
//!   instance, of no draws, after the last copy back.
//! - **The copies.** Of one sample, through a staging buffer of 4 bytes a texel of the render
//!   area: the depth aspect into the buffer, then the buffer into the stand-in, and back the same
//!   way. Of more, in the multisampled form on a device whose maintenance8 depth/colour copies
//!   move a multisampled aspect (`depth_stencil_scratch::colour_copies_refused`: RADV, ANV), one
//!   `vkCmdCopyImage` each way between the depth aspect and the stand-in. Elsewhere (lavapipe,
//!   MoltenVK and Ampere GA10x, whose copies run wrong, a device without maintenance8, and the
//!   sample planes), by the kernel and the draw of `depth_copy.rs`: the depth split, which reads
//!   the depth aspect through a sampled view into the stand-in, and the `FragDepth` draw of
//!   `DepthStencilCopy::DepthFromStandIn` back, exact where the driver's fragment depth stores
//!   every value unchanged (lavapipe), and elsewhere, for a 32-bit float depth, the announced
//!   near-equivalent `drawn-depth-copy` (what MoltenVK's and NVIDIA's are measured to change, or
//!   Vulkan leaves to a driver not measured: values outside `[0, 1]`, denormals, signed zero, NaN,
//!   none of which a depth test stores in the clip mode's range); a 16-bit depth is refused. The
//!   split reads an image with `SAMPLED`, which a multisampled depth/stencil render target the
//!   runtime makes takes on such a device where it costs the image nothing
//!   (`depth_stencil_scratch::sampled_candidate`); a draw whose instance needs the copy in of an
//!   image without it is refused, typed, naming why the copies cannot run.
//!
//! [`ShaderTestsUse`]: metal2vulkan::pipeline::ShaderTestsUse

use super::attachment_stand_in;
use super::blit::{reads, writes, TRANSFER};
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::depth_copy::{self, stand_in_view, DepthDraw, DepthSplit, Drawn};
use super::depth_stencil_scratch::colour_copies_refused;
use super::error::{internal, Error};
use super::image_plan::{sample_count, ImagePlan};
use super::render_pass::{AspectClear, Target};
use super::render_values::LoadAction;
use super::resource::ResourceKey;
use super::stencil_copy::{self, attachment_use};
use super::texture::{image_aspects, Texture};
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::ResourceOptions;
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::DepthStandInForm;
use metal2vulkan::device::VkProperty;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::{StandInUse, StandInView};
use metal2vulkan::vk::{Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags};

/// The feature a refusal of the stand-in's copies names: the compile layer's for the in-shader
/// test (`pipeline::link::shader_tests`).
const FEATURE: Feature = Feature::FramebufferFetch;

/// How a stand-in is copied (the module docs, [`route`]).
struct Route {
    form: DepthStandInForm,
    /// `Ok(true)` image to image, `Ok(false)` through a buffer, or `Err` by the kernel and the
    /// draw, naming why the image copies cannot.
    image_copies: Result<bool, Unsupported>,
    /// Where by the kernel and the draw, what the draw back changes (`depth_copy::drawn`).
    drawn: Option<Drawn>,
}

/// How a stand-in of view `view` of a depth target of Vulkan format `format` and `samples`
/// samples is copied (the module docs); refused where the copies by the kernel and the draw are
/// not measured to run.
fn route(
    device: &Device,
    (view, samples): (StandInView, u32),
    format: Format,
) -> Result<Route, Error> {
    let form = match (view, samples) {
        (StandInView::Image, _) => DepthStandInForm::Multisampled,
        (StandInView::SamplePlanes { layers }, _) if layers == samples && samples > 1 => {
            DepthStandInForm::SamplePlanes
        }
        _ => {
            return Err(internal(
                "a depth stand-in of another view than its samples",
            ))
        }
    };
    let planes = form == DepthStandInForm::SamplePlanes;
    // How the samples move (the module docs): `Err` where by the kernel and the draw, naming why
    // the image copies cannot.
    let image_copies = match (
        samples,
        colour_copies_refused(device, FEATURE, ImageAspectFlags::DEPTH),
    ) {
        (1, _) => Ok(false),
        (_, None) if !planes => Ok(true),
        (_, None) => Err(Unsupported::needs(
            FEATURE,
            Capability::Property(VkProperty::StorageImageSampleCount(samples)),
        )),
        (_, Some(refused)) => Err(refused),
    };
    let drawn = match &image_copies {
        Ok(_) => None,
        Err(refused) => match depth_copy::drawn(device, format) {
            Some(drawn) => Some(drawn),
            None => return Err(Error::Unsupported(refused.clone())),
        },
    };
    Ok(Route {
        form,
        image_copies,
        drawn,
    })
}

/// Make, at a render pipeline state's creation, the pipelines its tested draws' depth stand-in
/// copies by, where they go by the kernel and the draw (the module docs): the draw back and the
/// depth split, each made once per device, as Metal compiles at creation. The caller's state has
/// a depth attachment of Vulkan format `format` and `samples` samples, and a draw of it whose
/// depth test alone can drop fragments runs the variant whose fragment function tests the depth
/// itself (`GraphicsState::make_stand_in_copies`), so these are what its first such draw would
/// otherwise make while it encodes (on MoltenVK two Metal compiles). The stand-in holds the
/// samples as the compile layer's depth test reads them (`lower::fragment::depth_test_form`): as
/// sample planes at more than one sample on a device without multisampled storage images, as one
/// image otherwise. Whatever is refused here is left to that draw, which meets it again.
pub(crate) fn make_copies(device: &Device, samples: u32, format: Format) {
    let planes = samples > 1
        && !device
            .compile_device()
            .profile()
            .images()
            .storage_multisample;
    let view = match planes {
        true => StandInView::SamplePlanes { layers: samples },
        false => StandInView::Image,
    };
    if let Ok(Route {
        form,
        image_copies: Err(_),
        drawn: Some(_),
    }) = route(device, (view, samples), format)
    {
        device.make_stand_in_copies(form, (format, samples), &FEATURE);
    }
}

/// How the stand-in and the depth aspect are copied (the module docs).
#[derive(Clone)]
enum Copies {
    /// One sample: through a buffer of 4 bytes a texel of the render area.
    Buffer(Buffer),
    /// Image to image, by maintenance8's depth/colour copies.
    Image,
    /// The depth split in, where the depth's image is sampled, and the `FragDepth` draw back.
    Drawn {
        split: Option<DepthSplit>,
        back: DepthDraw,
    },
}

/// A pass's depth stand-in, and how it is copied.
#[derive(Clone)]
pub(super) struct DepthStandIn {
    /// The stand-in, as the storage target its view is made of.
    stand_in: Target,
    form: DepthStandInForm,
    copies: Copies,
    /// Why the depth cannot be copied into the stand-in, where it cannot: a draw whose instance
    /// needs the copy in is refused with it.
    no_copy_in: Option<Unsupported>,
    /// The render area, from the origin: what the copies cover.
    area: [u32; 2],
}

impl DepthStandIn {
    /// The stand-in `used` names of `depth`, the pass's depth/stencil target, over the render
    /// area `area`, made at the pass's first draw of such a pipeline (the module docs). `depth`
    /// must have a depth aspect of a 32-bit float, the use's samples and one layer (link and the
    /// binder state no stand-in otherwise). Refused (`NotImplemented(Render)`) where `depth`'s
    /// image does not copy both ways where the copies need it (a caller's image without transfer
    /// usage); where the copies go by the kernel and the draw, refused naming why the device's
    /// copies cannot move the samples where its fragment depth is not measured, and naming
    /// what the device lacks where it lacks what they need.
    pub(super) fn new(
        maker: &CommandBuffer,
        used: &StandInUse,
        depth: &Target,
        area: [u32; 2],
    ) -> Result<DepthStandIn, Error> {
        let device = maker.device();
        let samples = depth.samples;
        if used.format != Format::R32_SFLOAT
            || used.samples != samples
            || depth.layers != 1
            || !image_aspects(depth.format).contains(ImageAspectFlags::DEPTH)
            || !depth_copy::float_depth(depth.format)
        {
            return Err(internal(
                "a shader-side depth test whose stand-in the pass's depth cannot be",
            ));
        }
        let Route {
            form,
            image_copies,
            drawn,
        } = route(device, (used.view, used.samples), depth.format)?;
        let planes = form == DepthStandInForm::SamplePlanes;
        let (layers, image_samples) = match planes {
            true => (samples, 1),
            false => (1, samples),
        };
        if image_copies.is_ok() {
            // Copied by transfers both ways (a caller's image may lack the usage).
            attachment_stand_in::check_copies(depth)?;
        }
        let [width, height] = depth.extent;
        let samples_flag = sample_count(image_samples)
            .ok_or_else(|| internal("an attachment of a sample count Vulkan lacks"))?;
        let descriptor = TextureDescriptor {
            texture_type: match (image_samples > 1, planes) {
                (true, _) => TextureType::Type2DMultisample,
                (false, true) => TextureType::Type2DArray,
                (false, false) => TextureType::Type2D,
            },
            pixel_format: PixelFormat::R32Float,
            width: width.into(),
            height: height.into(),
            array_length: layers.into(),
            sample_count: image_samples.into(),
            usage: TextureUsage::RENDER_TARGET,
            resource_options: ResourceOptions::private(),
            ..TextureDescriptor::default()
        };
        let plan = ImagePlan {
            pixel_format: PixelFormat::R32Float,
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::TYPE_2D,
            format: Format::R32_SFLOAT,
            extent: [width, height, 1],
            levels: 1,
            layers,
            samples: samples_flag,
            usage: ImageUsageFlags::STORAGE
                | ImageUsageFlags::TRANSFER_SRC
                | ImageUsageFlags::TRANSFER_DST,
            view_formats: Vec::new(),
            mapping: FormatMapping::direct(Format::R32_SFLOAT, ImageAspectFlags::COLOR),
            unavailable: TextureUsage::SHADER_READ
                | TextureUsage::SHADER_WRITE
                | TextureUsage::SHADER_ATOMIC
                | TextureUsage::RENDER_TARGET
                | TextureUsage::PIXEL_FORMAT_VIEW,
        };
        let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
        let stand_in = Target {
            texture,
            level: 0,
            layer: 0,
            layers,
            format: Format::R32_SFLOAT,
            samples: image_samples,
            extent: [width, height],
        };
        let mut no_copy_in = None;
        let copies = match (image_copies, drawn) {
            (Ok(false), _) => {
                let [w, h] = area;
                let bytes = 4 * u64::from(w) * u64::from(h);
                Copies::Buffer(device.own_buffer(bytes, ResourceOptions::private())?)
            }
            (Ok(true), _) => Copies::Image,
            (Err(refused), drawn) => {
                if let Some(Drawn::Changes(changes)) = drawn {
                    depth_copy::announce_changes(device, changes, || {
                        format!(
                            "the depth stand-in's copies back into a {samples}-sample {:?} \
                             attachment",
                            depth.format
                        )
                    });
                }
                let back =
                    DepthDraw::from_stand_in(device, &stand_in, form, depth, area, &FEATURE)?;
                let split = match stencil_copy::sampled(depth)? {
                    true => Some(DepthSplit::new(
                        device, depth, &stand_in, form, area, &FEATURE,
                    )?),
                    false => {
                        no_copy_in = Some(refused);
                        None
                    }
                };
                Copies::Drawn { split, back }
            }
        };
        Ok(DepthStandIn {
            stand_in,
            form,
            copies,
            no_copy_in,
            area,
        })
    }

    /// Refuse a draw whose instance, the first of the pass where `first`, fills the stand-in
    /// from the depth `load`s (the module docs) where the depth cannot be copied in: refused
    /// (`Unsupported`) naming why the device's copies cannot move its samples.
    pub(super) fn check_filled(&self, first: bool, load: Option<LoadAction>) -> Result<(), Error> {
        let reads = !first || !matches!(load, Some(LoadAction::Clear | LoadAction::DontCare));
        match (&self.no_copy_in, reads) {
            (Some(refused), true) => Err(Error::Unsupported(refused.clone())),
            _ => Ok(()),
        }
    }

    /// The storage view the plan's `WriteSource::DepthStandIn` binds.
    pub(super) fn view(&self) -> Result<vk::ImageView, Error> {
        stand_in_view(&self.stand_in, self.form)
    }

    /// A draw's uses of the stand-in: the fragment shader reads and writes it.
    pub(super) fn draw_uses(&self) -> Result<Vec<Use<ResourceKey>>, Error> {
        let stage = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let read = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_READ);
        let write = Access::new(stage, vk::AccessFlags2::SHADER_STORAGE_WRITE);
        let layers = 0..self.stand_in.layers;
        let texture = &self.stand_in.texture;
        Ok(vec![
            texture.image_use(0..1, layers.clone(), read, false)?,
            texture.image_use(0..1, layers, write, true)?,
        ])
    }

    /// The copies' uses, both ways, between the stand-in and `depth`, and the stand-in's clear.
    pub(super) fn copy_uses(&self, depth: &Target) -> Result<Vec<Use<ResourceKey>>, Error> {
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let levels = depth.level..depth.level + 1;
        let depth_layers = depth.layer..depth.layer + 1;
        let layers = 0..self.stand_in.layers;
        let texture = &self.stand_in.texture;
        let mut uses = vec![texture.image_use(0..1, layers.clone(), write, true)?];
        match &self.copies {
            Copies::Buffer(_) | Copies::Image => {
                uses.push(texture.image_use(0..1, layers, read, false)?);
                uses.push(depth.texture.image_use(
                    levels.clone(),
                    depth_layers.clone(),
                    read,
                    false,
                )?);
                uses.push(depth.texture.image_use(levels, depth_layers, write, true)?);
            }
            Copies::Drawn { split, .. } => {
                let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
                let fragment = vk::PipelineStageFlags2::FRAGMENT_SHADER;
                if split.is_some() {
                    let sampled = Access::new(compute, vk::AccessFlags2::SHADER_SAMPLED_READ);
                    let stored = Access::new(compute, vk::AccessFlags2::SHADER_STORAGE_WRITE);
                    uses.push(
                        depth
                            .texture
                            .image_use(levels, depth_layers, sampled, false)?,
                    );
                    uses.push(texture.image_use(0..1, layers.clone(), stored, true)?);
                }
                let read = Access::new(fragment, vk::AccessFlags2::SHADER_STORAGE_READ);
                uses.push(texture.image_use(0..1, layers, read, false)?);
                uses.push(attachment_use(depth)?);
            }
        }
        if let Copies::Buffer(staging) = &self.copies {
            let length = staging.length();
            uses.push(writes(staging, 0..length));
            uses.push(reads(staging, 0..length));
        }
        Ok(uses)
    }

    /// What the copies go through, which the command buffer holds until it completes.
    pub(super) fn held(
        &self,
    ) -> (
        Texture,
        Option<Buffer>,
        Option<DepthSplit>,
        Option<DepthDraw>,
    ) {
        let texture = self.stand_in.texture.clone();
        match &self.copies {
            Copies::Buffer(staging) => (texture, Some(staging.clone()), None, None),
            Copies::Image => (texture, None, None, None),
            Copies::Drawn { split, back } => (texture, None, split.clone(), Some(back.clone())),
        }
    }

    /// The copies between the stand-in and `depth`, the pass's depth/stencil target at the end
    /// of its encoder, whose depth the pass loads by `load` and clears to `clear` where it
    /// clears.
    pub(super) fn copies(
        &self,
        depth: &Target,
        load: Option<LoadAction>,
        clear: Option<AspectClear>,
    ) -> Result<StandInCopies, Error> {
        let [width, height] = self.area;
        let fill = match (load, clear) {
            (Some(LoadAction::Clear), Some(AspectClear::Depth(value))) => FirstFill::Clear(value),
            (Some(LoadAction::Clear), _) => {
                return Err(internal("a cleared depth without its clear value"))
            }
            (Some(LoadAction::DontCare), _) => FirstFill::Nothing,
            _ => FirstFill::Copy,
        };
        let (staging, split, back) = match &self.copies {
            Copies::Buffer(staging) => (Some(staging.at(0)), None, None),
            Copies::Image => (None, None, None),
            Copies::Drawn { split, back } => (None, split.clone(), Some(back.clone())),
        };
        Ok(StandInCopies {
            stand_in: self.stand_in.texture.raw_image()?,
            attachment: depth.texture.raw_image()?,
            staging,
            depth_layers: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                mip_level: depth.level,
                base_array_layer: depth.layer,
                layer_count: 1,
            },
            layers: self.stand_in.layers,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            fill,
            split,
            back,
        })
    }
}

/// How the instance that runs the pass's load actions fills the stand-in (the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
enum FirstFill {
    /// The depth clears: to the clear depth.
    Clear(f32),
    /// Metal's depth is undefined (`DontCare`): nothing.
    Nothing,
    /// The depth loads: the copy in.
    Copy,
}

/// The recorded form of a pass's copies between its depth and its stand-in
/// ([`DepthStandIn::copies`]).
pub(super) struct StandInCopies {
    stand_in: vk::Image,
    attachment: vk::Image,
    /// The staging buffer and the offset of its range, for a stand-in of one sample.
    staging: Option<(vk::Buffer, u64)>,
    depth_layers: vk::ImageSubresourceLayers,
    /// The stand-in's layers: its sample planes, or 1.
    layers: u32,
    extent: vk::Extent3D,
    fill: FirstFill,
    /// The split in and the draw back, where the samples move by them.
    split: Option<DepthSplit>,
    back: Option<DepthDraw>,
}

impl StandInCopies {
    /// Record, outside a render pass, the stand-in's filling before an instance of such draws:
    /// from the depth's load action where the instance runs the pass's load actions (`first`),
    /// else the copy of the depth aspect (the module docs). Between barriers from everything
    /// before and to everything after.
    ///
    /// # Safety
    ///
    /// `raw` is `device`'s command buffer, recording outside a render pass, which only this
    /// encoder records; the stand-in, its staging buffer, the depth's image and the copies'
    /// objects live until it completes and rest in `GENERAL`; the encoder refused every draw
    /// whose instance needs a copy in that cannot run ([`DepthStandIn::check_filled`]).
    pub(super) unsafe fn record_in(&self, device: &Vk, raw: vk::CommandBuffer, first: bool) {
        let fill = match first {
            true => self.fill,
            false => FirstFill::Copy,
        };
        Barrier::everything().record(device, raw);
        match fill {
            FirstFill::Clear(value) => {
                let colour = vk::ClearColorValue {
                    float32: [value, 0.0, 0.0, 0.0],
                };
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: self.layers,
                };
                let general = vk::ImageLayout::GENERAL;
                device.cmd_clear_color_image(raw, self.stand_in, general, &colour, &[range]);
            }
            FirstFill::Nothing => {}
            FirstFill::Copy => match &self.split {
                Some(split) => split.record(device, raw),
                None if self.back.is_some() => {}
                None => self.copy(device, raw, true),
            },
        }
        Barrier::everything().record(device, raw);
    }

    /// Record, outside a render pass, the copy of the stand-in back into the depth aspect after
    /// an instance of such draws, between barriers from everything before and to everything
    /// after.
    ///
    /// # Safety
    ///
    /// As for [`StandInCopies::record_in`].
    pub(super) unsafe fn record_out(&self, device: &Vk, raw: vk::CommandBuffer) {
        Barrier::everything().record(device, raw);
        match &self.back {
            Some(back) => back.record(device, raw),
            None => self.copy(device, raw, false),
        }
        Barrier::everything().record(device, raw);
    }

    /// Record the transfer copy of the render area between the depth aspect and the stand-in:
    /// into the stand-in (`inward`) or back; through the staging buffer at one sample.
    ///
    /// # Safety
    ///
    /// As for [`StandInCopies::record_in`].
    unsafe fn copy(&self, device: &Vk, raw: vk::CommandBuffer, inward: bool) {
        let general = vk::ImageLayout::GENERAL;
        let stand_in_layers = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let ((from, from_layers), (to, to_layers)) = if inward {
            (
                (self.attachment, self.depth_layers),
                (self.stand_in, stand_in_layers),
            )
        } else {
            (
                (self.stand_in, stand_in_layers),
                (self.attachment, self.depth_layers),
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
