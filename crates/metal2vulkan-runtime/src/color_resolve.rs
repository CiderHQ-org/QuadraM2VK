//! Metal's exact multisample color resolve, drawn after the pass.
//!
//! Vulkan's `AVERAGE` resolve leaves the average's rounding, and an sRGB average's space, to the
//! device, and of the drivers measured only MoltenVK's from 1.4.1 on Apple's GPUs writes Metal's
//! answer (`DriverFact::ColorResolveAverageIsMetals`, measured at 2 and 4 samples). Every other
//! device resolves a color attachment whose format has Metal's closed rule at its count
//! (`bind::ColorResolveFormat::of` and `resolves`: unorm, snorm, sRGB and 32-bit float formats,
//! and 16-bit float ones at 2 samples) by the crate-owned exact resolve
//! (`bind::color_resolve_modules`), and so does every device at more than 4 samples, where Apple's
//! GPUs have no texture and the rules measured at 2 and 4 are applied unmeasured. A format without
//! a closed rule at its count (half floats past 2 samples, packed 16-bit, `RG11B10Float`,
//! `RGB9E5Float`) resolves by the device's average: Metal's own resolve where the fact holds
//! ([`average_is_metals`]: MoltenVK resolves every such format by Metal's multisample resolve store
//! action), elsewhere a recorded near-equivalent.
//!
//! The choice is made with the attachment's resolve ([`exact`], `resolve.rs`), so the pass's
//! shape (`render_target.rs`) leaves such a resolve out of its subpass from its first instance:
//! the pass's last instance stores the attachment's samples, and after the pass one render pass
//! instance of the resolve's own pass per attachment draws a triangle over the render area that
//! reads each pixel's samples and writes the resolve attachment (the resolve texture's level and
//! slice, or the scratch image copied there after, `resolve.rs`):
//!
//! - **From the sample planes** (`sample_planes.rs`) where they hold the attachment's samples as
//!   the pass ends (its last instance's draws fetch it): read through a sampled 2D-array view in
//!   their copy format. The planes are not merged back into the attachment unless Metal keeps
//!   its samples (`Store`, `StoreAndMultisampleResolve`), no instance of no draws follows to
//!   resolve, and a memoryless attachment the pass only fetches and resolves needs no backing
//!   (`render.rs`). This is the cheaper path on a driver whose average is Metal's too, so such an
//!   attachment resolves from its planes there as well ([`resolve_planes`]): its pass has one
//!   subpass, whose resolve references render pass compatibility ignores (`render_target.rs`).
//! - **From the attachment** otherwise: its level and layer as the resolve pass's input
//!   attachment 0, loaded and stored. Its image needs input attachment usage, which a
//!   multisampled color render target the runtime makes takes where the device may resolve it so
//!   and the usage costs the image nothing (`framebuffer_fetch.rs`), as does a stand-in
//!   (`attachment_stand_in.rs`); an attachment whose image lacks it (a caller's image without it,
//!   or a device where it would change the image's memory requirements) resolves by the device's
//!   average, announced as `device-rounded-resolve`.
//!
//! A layered pass (`renderTargetArrayLength` 2 or more) resolves each of its layers in a render
//! pass instance of its own, through views of that layer. Its resolve from the sample planes, which
//! hold one layer, is `NotImplemented(LayeredPass)`.
//!
//! The resolve's pipelines and passes are made once per device, format, sample count and source,
//! at the first pass that needs them, and destroyed with the device; each pass's descriptor sets
//! and framebuffers are held by its command buffer until it completes. Each resolve runs between
//! barriers from everything before and to everything after. The runtime announces
//! `shader-resolve` (performance) for the extra pass, and `unmeasured-resolve` (precision) at
//! more than 4 samples.

use super::binding_view::VkView;
use super::device::{device_error, lock};
use super::encoder::EncoderCore;
use super::error::{internal, vulkan, Error, RuntimeGap};
use super::exit_guard;
use super::image_plan::ImagePlan;
use super::release;
use super::render_pass::{PassLayout, Target};
use super::render_target::{takes_input, ATTACHMENT_STAGES};
use super::resource::ResourceKey;
use super::sample_planes::{pipeline_layout, set_layout, shader_module};
use super::tracker::{Access, Barrier, Use};
use super::Device;
use crate::shader_cache::driver::Making;
use ash::vk;
use metal2vulkan::bind::{color_resolve_modules, ColorResolveFormat, PlanesCopy, ResolveSource};
use metal2vulkan::device::DriverFact;
use metal2vulkan::error::Feature;
use metal2vulkan::vk::{
    ComponentMapping, Format, ImageAspectFlags, ImageUsageFlags, ImageViewType,
};

/// The most samples of a texture on Apple's GPUs, so the most whose resolve Metal measures.
pub(super) const METAL_SAMPLES: u32 = 4;

/// The exact resolve of a color attachment of `format` at `samples` samples on `device`: its
/// format's rules where the format has them and the device's average is not measured to be
/// Metal's at that count (the module docs). `None` where the device resolves it in the pass.
pub(super) fn exact(device: &Device, format: Format, samples: u32) -> Option<ColorResolveFormat> {
    // Rules that hold at that count: a half's at 2 samples only.
    let rules = ColorResolveFormat::of(format).filter(|rules| rules.resolves(samples))?;
    (samples > 1 && !average_is_metals(device, samples)).then_some(rules)
}

/// Whether `device`'s own `AVERAGE` resolve of a color attachment of `samples` samples writes
/// Metal's resolve, whatever its format: `DriverFact::ColorResolveAverageIsMetals`, at a count
/// Apple's GPUs make (MoltenVK's resolve is Metal's multisample resolve store action).
pub(super) fn average_is_metals(device: &Device, samples: u32) -> bool {
    samples <= METAL_SAMPLES
        && device
            .properties()
            .driver_facts
            .contains(&DriverFact::ColorResolveAverageIsMetals)
}

/// The exact resolve of the color attachment `source`, where the runtime resolves it so: [`exact`]
/// of its format and samples, where its image takes input attachment usage, through which the
/// resolve reads its samples unless its sample planes hold them (the module docs).
pub(super) fn of_attachment(
    device: &Device,
    source: &Target,
) -> Result<Option<ColorResolveFormat>, Error> {
    match exact(device, source.format, source.samples) {
        Some(rules) if takes_input(source)? => Ok(Some(rules)),
        _ => Ok(None),
    }
}

/// Whether a multisampled color attachment of `format` at `samples` that the sample planes may
/// hold as a pass ends resolves from them by the exact resolve whatever its device's average: its
/// format has Metal's rules at that count. Its pass is of one subpass, whose resolve references
/// render pass compatibility ignores, so its last instance may leave the subpass's resolve out
/// (`render_target.rs`).
pub(super) fn from_planes(format: Format, samples: u32) -> bool {
    ColorResolveFormat::of(format).is_some_and(|rules| rules.resolves(samples))
}

/// Resolve each color attachment of `layout` (a pass's, as it ends) whose
/// sample planes hold its samples, its last instance leaving it out of the draws' subpass
/// (`last_storage`), by the exact resolve from them where [`from_planes`] says so: neither merged
/// back for the device's resolve nor resolved in one more instance. Not in a layered pass: the
/// planes hold one layer.
pub(super) fn resolve_planes(layout: &mut PassLayout, last_storage: &[u32]) {
    if layout.layers > 1 {
        return;
    }
    for (index, color) in layout.colors.iter_mut().enumerate() {
        let Some(color) = color else {
            continue;
        };
        if color.planes.is_none() || !last_storage.contains(&(index as u32)) {
            continue;
        }
        let format = color.target.format;
        if let Some(resolve) = &mut color.resolve {
            if resolve.shader.is_none() && from_planes(format, color.target.samples) {
                resolve.shader = ColorResolveFormat::of(format);
            }
        }
    }
}

/// `plan` with input attachment usage, where the runtime may resolve its image by the exact
/// resolve reading it as an input attachment: a multisampled color attachment whose format and
/// samples [`exact`] resolves on `device`. `None` where it has the usage already, or needs none.
pub(super) fn input_candidate(plan: &ImagePlan, device: &Device) -> Option<ImagePlan> {
    let usage = plan.usage;
    let samples = plan.samples.as_raw();
    let candidate = usage.contains(ImageUsageFlags::COLOR_ATTACHMENT)
        && !usage.contains(ImageUsageFlags::INPUT_ATTACHMENT)
        && exact(device, plan.format, samples).is_some();
    candidate.then(|| ImagePlan {
        usage: usage | ImageUsageFlags::INPUT_ATTACHMENT,
        ..plan.clone()
    })
}

/// Where a resolve reads its samples (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Source {
    /// The attachment, as the resolve pass's input attachment 0.
    Attachment,
    /// The attachment's sample planes, viewed in their copy format.
    Planes(PlanesCopy),
}

impl Source {
    fn bind(self) -> ResolveSource {
        match self {
            Source::Attachment => ResolveSource::InputAttachment,
            Source::Planes(copy) => ResolveSource::Planes(copy),
        }
    }

    /// The descriptor type of the source's binding.
    fn descriptor(self) -> vk::DescriptorType {
        match self {
            Source::Attachment => vk::DescriptorType::INPUT_ATTACHMENT,
            Source::Planes(_) => vk::DescriptorType::SAMPLED_IMAGE,
        }
    }
}

/// One resolve's objects, the device's: for one format, sample count and source.
#[derive(Clone, Copy, Debug)]
struct ResolvePipeline {
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
}

impl ResolvePipeline {
    const NULL: ResolvePipeline = ResolvePipeline {
        vertex: vk::ShaderModule::null(),
        fragment: vk::ShaderModule::null(),
        set_layout: vk::DescriptorSetLayout::null(),
        layout: vk::PipelineLayout::null(),
        render_pass: vk::RenderPass::null(),
        pipeline: vk::Pipeline::null(),
    };

    /// Destroy every object.
    ///
    /// # Safety
    ///
    /// No pending command uses them; a null handle is a no-op.
    unsafe fn destroy(&self, raw: &ash::Device) {
        // SAFETY: the caller's contract.
        unsafe {
            release::object(raw, self.pipeline);
            release::object(raw, self.render_pass);
            release::object(raw, self.layout);
            release::object(raw, self.set_layout);
            release::object(raw, self.fragment);
            release::object(raw, self.vertex);
        }
    }
}

/// The resolves' objects one device has made (the module docs): bounded by the formats with
/// closed rules, the sample counts and the planes' copy formats.
#[derive(Default)]
pub(super) struct ColorResolveObjects {
    pipelines: Vec<((ColorResolveFormat, u32, Source), ResolvePipeline)>,
}

impl ColorResolveObjects {
    /// Destroy every object; the device is idle, and is destroyed next.
    pub(super) fn destroy(&self, raw: &ash::Device) {
        for (_, pipeline) in &self.pipelines {
            // SAFETY: the device is idle, so no pending command uses them.
            unsafe { pipeline.destroy(raw) };
        }
    }
}

impl Device {
    /// The resolve's objects for `format` at `samples` from `source`, made on first use.
    fn color_resolve_pipeline(
        &self,
        format: ColorResolveFormat,
        samples: u32,
        source: Source,
    ) -> Result<ResolvePipeline, Error> {
        let key = (format, samples, source);
        // A making's turn before the lock (`budget::helper_turn`): none where it is made.
        let _turn =
            crate::pipeline::budget::helper_turn(&self.shared().color_resolves, |objects| {
                objects.pipelines.iter().any(|(k, _)| *k == key)
            });
        let mut objects = lock(&self.shared().color_resolves);
        if let Some((_, pipeline)) = objects.pipelines.iter().find(|(k, _)| *k == key) {
            return Ok(*pipeline);
        }
        let modules = color_resolve_modules(
            self.compile_device(),
            format,
            samples,
            source.bind(),
            Feature::Format,
        )
        .map_err(device_error)?;
        let raw = self.shared().vk();
        let mut made = ResolvePipeline::NULL;
        let result = (|| {
            made.vertex = shader_module(raw, &modules.vertex)?;
            made.fragment = shader_module(raw, &modules.fragment)?;
            made.set_layout =
                set_layout(raw, &[source.descriptor()], vk::ShaderStageFlags::FRAGMENT)?;
            made.layout = pipeline_layout(raw, made.set_layout)?;
            made.render_pass = resolve_pass(raw, format.format(), samples, source)?;
            let making = Making::new("color resolve")
                .words(&modules.vertex)
                .words(&modules.fragment)
                .text(&key);
            made.pipeline = resolve_pipeline(self, &made, making)?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                objects.pipelines.push((key, made));
                Ok(made)
            }
            Err(error) => {
                // SAFETY: nothing used the objects.
                unsafe { made.destroy(raw) };
                Err(error)
            }
        }
    }
}

/// The resolve's render pass for `format` at `samples` from `source`: the multisampled
/// attachment as input attachment 0 (loaded and stored, `GENERAL`) where it reads it, then the
/// single-sampled resolve attachment, written whole over the render area (`DONT_CARE`, then
/// `STORE`).
fn resolve_pass(
    raw: &ash::Device,
    format: Format,
    samples: u32,
    source: Source,
) -> Result<vk::RenderPass, Error> {
    let general = vk::ImageLayout::GENERAL;
    let samples_attachment = vk::AttachmentDescription::default()
        .format(format.into())
        .samples(vk::SampleCountFlags::from_raw(samples))
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(general)
        .final_layout(general);
    let resolved = vk::AttachmentDescription::default()
        .format(format.into())
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(general)
        .final_layout(general);
    let (attachments, color) = match source {
        Source::Attachment => (vec![samples_attachment, resolved], 1),
        Source::Planes(_) => (vec![resolved], 0),
    };
    let inputs = [vk::AttachmentReference::default()
        .attachment(0)
        .layout(general)];
    let colors = [vk::AttachmentReference::default()
        .attachment(color)
        .layout(general)];
    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors);
    if source == Source::Attachment {
        subpass = subpass.input_attachments(&inputs);
    }
    let subpasses = [subpass];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses);
    // SAFETY: `info` and its arrays are live for the call.
    unsafe { crate::creation::render_pass(raw, &info) }
}

/// The resolve's pipeline in its pass: the triangle over the viewport at one sample, without
/// blending, writing every component. Its record is the one `making` names.
fn resolve_pipeline(
    device: &Device,
    made: &ResolvePipeline,
    making: Making,
) -> Result<vk::Pipeline, Error> {
    let shared = device.shared();
    let raw = shared.vk();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(made.vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(made.fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let attachments = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(made.layout)
        .render_pass(made.render_pass)
        .subpass(0);
    let created = shared
        .make_pipelines(making, |cache| {
            // SAFETY: every pointer in `info` outlives the call; the pipeline cache is the
            // making's own (`shader_cache/driver.rs`).
            unsafe { raw.create_graphics_pipelines(cache, &[info], None) }
        })
        .map_err(|(_, error)| vulkan(error))?;
    created
        .first()
        .copied()
        .ok_or_else(|| internal("the driver made no pipeline"))
}

/// The objects one pass's resolves hold until its command buffer completes: their descriptor
/// sets' pool and their framebuffers.
struct PassObjects {
    device: Device,
    pool: vk::DescriptorPool,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Drop for PassObjects {
    fn drop(&mut self) {
        // Once the process is exiting, the objects are left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let raw = self.device.shared().vk();
        // SAFETY: no pending command uses them (a command buffer retains them until it completes);
        // the sets are freed with their pool, and a null handle is a no-op.
        unsafe {
            for &framebuffer in &self.framebuffers {
                release::object(raw, framebuffer);
            }
            release::object(raw, self.pool);
        }
    }
}

/// One attachment's resolve of one layer, planned.
struct Planned {
    pipeline: ResolvePipeline,
    framebuffer: vk::Framebuffer,
    set: vk::DescriptorSet,
}

/// A pass's exact color resolves, planned before the pass records and recorded after it.
pub(super) struct ShaderResolves {
    resolves: Vec<Planned>,
    extent: [u32; 2],
    uses: Vec<Use<ResourceKey>>,
    objects: PassObjects,
}

/// The single-level view of `target`'s layer `layer` (counted from its first) in `format`,
/// serving `usage`.
fn view(
    target: &Target,
    layer: u32,
    format: Format,
    usage: ImageUsageFlags,
) -> Result<vk::ImageView, Error> {
    // A 1D image's layer is a 1D view (a 3D image's planes take 2D ones).
    let view_type = match target.texture.image_plan()?.image_type {
        metal2vulkan::vk::ImageType::TYPE_1D => ImageViewType::TYPE_1D,
        _ => ImageViewType::TYPE_2D,
    };
    target.texture.image_view(&VkView {
        view_type,
        format,
        components: ComponentMapping::IDENTITY,
        aspect: ImageAspectFlags::COLOR,
        base_level: target.level,
        level_count: 1,
        base_layer: target.layer + layer,
        layer_count: 1,
        usage,
    })
}

impl ShaderResolves {
    /// The exact resolves of `layout`'s color attachments (its final layout) whose last render
    /// pass instance leaves `last_storage` out of its draws' subpass: each from its sample
    /// planes where they hold its samples, else from the attachment (the module docs). `None`
    /// where no attachment resolves so.
    pub(super) fn plan(
        device: &Device,
        layout: &PassLayout,
        last_storage: &[u32],
    ) -> Result<Option<ShaderResolves>, Error> {
        let mut wanted = Vec::new();
        for (index, color) in layout.colors.iter().enumerate() {
            let Some(color) = color else {
                continue;
            };
            let Some(resolve) = &color.resolve else {
                continue;
            };
            let Some(format) = resolve.shader else {
                continue;
            };
            let planes = color
                .planes
                .as_ref()
                .filter(|_| last_storage.contains(&(index as u32)));
            let source = match planes {
                Some(planes) => Source::Planes(planes.copy),
                None => Source::Attachment,
            };
            wanted.push((color, resolve, format, source, planes));
        }
        if wanted.is_empty() {
            return Ok(None);
        }
        // A layered pass resolves each of its layers in a render pass instance of its own. The
        // sample planes hold one layer (`sample_planes.rs`), so such a pass's resolve from them
        // is not built.
        let layers = layout.layers;
        if layers > 1 && wanted.iter().any(|w| w.3 != Source::Attachment) {
            return Err(Error::NotImplemented(RuntimeGap::LayeredPass));
        }
        let raw = device.shared().vk();
        let count = wanted.len() as u32 * layers;
        let mut objects = PassObjects {
            device: device.clone(),
            pool: vk::DescriptorPool::null(),
            framebuffers: Vec::new(),
        };
        let inputs = wanted.iter().filter(|w| w.3 == Source::Attachment).count() as u32 * layers;
        let mut sizes = Vec::new();
        for (ty, n) in [
            (vk::DescriptorType::INPUT_ATTACHMENT, inputs),
            (vk::DescriptorType::SAMPLED_IMAGE, count - inputs),
        ] {
            if n > 0 {
                sizes.push(vk::DescriptorPoolSize::default().ty(ty).descriptor_count(n));
            }
        }
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(count)
            .pool_sizes(&sizes);
        // SAFETY: `info` and its array are live for the call.
        objects.pool = unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
        let general = vk::ImageLayout::GENERAL;
        let mut uses = Vec::new();
        let mut resolves = Vec::with_capacity(wanted.len());
        for (color, resolve, format, source, planes) in wanted {
            let samples = &color.target;
            let into = &resolve.attachment;
            if samples.layers != layers || into.layers != layers {
                return Err(internal("an exact resolve of other layers than its pass"));
            }
            let pipeline = device.color_resolve_pipeline(format, samples.samples, source)?;
            let all = |t: &Target| (t.level..t.level + 1, t.layer..t.layer + layers);
            match (source, planes) {
                (Source::Planes(_), Some(planes)) => uses.push(planes.image_use(
                    Access::new(
                        vk::PipelineStageFlags2::FRAGMENT_SHADER,
                        vk::AccessFlags2::SHADER_SAMPLED_READ,
                    ),
                    false,
                )?),
                (Source::Attachment, _) => {
                    if !takes_input(samples)? {
                        return Err(internal(
                            "an exact resolve of an attachment without input attachment usage",
                        ));
                    }
                    let (levels, on) = all(samples);
                    uses.push(samples.texture.image_use(
                        levels,
                        on,
                        Access::new(
                            vk::PipelineStageFlags2::FRAGMENT_SHADER
                                | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                            vk::AccessFlags2::INPUT_ATTACHMENT_READ
                                | vk::AccessFlags2::COLOR_ATTACHMENT_READ
                                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        ),
                        true,
                    )?);
                }
                (Source::Planes(_), None) => {
                    return Err(internal("a planes resolve without planes"))
                }
            }
            let (levels, on) = all(into);
            uses.push(into.texture.image_use(
                levels,
                on,
                Access::new(
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                ),
                true,
            )?);
            // Each layer of the pass: one set and framebuffer of that layer's views.
            for layer in 0..layers {
                let layouts = [pipeline.set_layout];
                let allocate = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(objects.pool)
                    .set_layouts(&layouts);
                // SAFETY: the pool holds a set per resolve and layer, of its source's descriptor
                // (counted above).
                let set = unsafe { raw.allocate_descriptor_sets(&allocate) }
                    .map_err(vulkan)?
                    .first()
                    .copied()
                    .ok_or_else(|| internal("the driver allocated no set"))?;
                let written = view(into, layer, into.format, ImageUsageFlags::COLOR_ATTACHMENT)?;
                let (read, attachments) = match (source, planes) {
                    (Source::Planes(copy), Some(planes)) => {
                        let read = planes.view(copy.format(), ImageUsageFlags::SAMPLED)?;
                        (read, vec![written])
                    }
                    (Source::Attachment, _) => {
                        let read = view(
                            samples,
                            layer,
                            samples.format,
                            ImageUsageFlags::INPUT_ATTACHMENT,
                        )?;
                        (read, vec![read, written])
                    }
                    (Source::Planes(_), None) => {
                        return Err(internal("a planes resolve without planes"))
                    }
                };
                let info = [vk::DescriptorImageInfo::default()
                    .image_view(read)
                    .image_layout(general)];
                // SAFETY: the set was just allocated and no command uses it; the view is live
                // (the command buffer retains its texture).
                unsafe {
                    raw.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(0)
                            .descriptor_type(source.descriptor())
                            .image_info(&info)],
                        &[],
                    )
                };
                let [width, height] = layout.extent;
                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(pipeline.render_pass)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1);
                // SAFETY: `info` and its array are live for the call; the views are live.
                let framebuffer = unsafe { crate::creation::framebuffer(raw, &info) }?;
                objects.framebuffers.push(framebuffer);
                resolves.push(Planned {
                    pipeline,
                    framebuffer,
                    set,
                });
            }
        }
        Ok(Some(ShaderResolves {
            resolves,
            extent: layout.extent,
            uses,
            objects,
        }))
    }

    /// Record the resolves in `core`'s command buffer after the pass, between barriers from
    /// everything before and to everything after, and hold their objects until the command
    /// buffer completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let ShaderResolves {
            resolves,
            extent,
            uses,
            objects,
        } = self;
        let [width, height] = extent;
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        core.barrier(Barrier::everything());
        core.command(ATTACHMENT_STAGES, &uses, |raw, command_buffer| {
            for resolve in &resolves {
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(resolve.pipeline.render_pass)
                    .framebuffer(resolve.framebuffer)
                    .render_area(area);
                // SAFETY: the encoder's command buffer is recording outside a render pass; the
                // pipeline, framebuffer, set and the images they view are live until it completes
                // (the device keeps the pipeline, this the framebuffer and set, the encoder the
                // textures and planes); every image rests in `GENERAL`.
                unsafe {
                    raw.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
                    raw.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        resolve.pipeline.pipeline,
                    );
                    raw.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        resolve.pipeline.layout,
                        0,
                        &[resolve.set],
                        &[],
                    );
                    raw.cmd_set_viewport(command_buffer, 0, &[viewport]);
                    raw.cmd_set_scissor(command_buffer, 0, &[area]);
                    raw.cmd_draw(command_buffer, 3, 1, 0, 0);
                    raw.cmd_end_render_pass(command_buffer);
                }
            }
        });
        core.barrier(Barrier::everything());
        core.retain(objects);
    }
}
