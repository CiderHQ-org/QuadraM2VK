//! Sample planes: what a consumer keeps for a storage-form attachment of a multisampled pass on a
//! device without multisampled storage images (`StorageAttachmentView::SamplePlanes`).
//!
//! The executor runs the protocol `StorageAttachmentUse` states around its one draw. The
//! attachment's multisampled target ([`Bound::Planes`]) is filled sample by sample by the
//! initializer and read back sample by sample after the draw, as every multisampled target is. Its
//! planes are a single-sample 2D-array image of the use's format with one layer per sample, which
//! the plan's `WriteSource::TileAttachment` binds through a 2D-array storage view of every layer,
//! in `GENERAL` through the draw. Outside the render pass, before it, a compute dispatch of one
//! invocation per sample splits the target into the planes: invocation `(x, y, s)` fetches sample
//! `s` of pixel `(x, y)` (`OpImageFetch` with `Sample`) through the target's unsigned-integer view
//! of its texel size, the view the readback fetches through, and writes it to texel `(x, y)` of
//! layer `s` through the planes' view in that format. After the pass, a sample-rate pass over the
//! target merges the planes back: each invocation fetches layer `SampleId` at its pixel and writes
//! it to its own sample through a color-attachment view of the target in the same unsigned-integer
//! format. Every copy moves a texel's bits through a view of its size, so the planes hold, and the
//! target receives, the bytes unchanged. Nothing is resolved: no case has a memoryless target or a
//! resolve target, so the resolve from the planes that `StorageAttachmentUse` describes is the
//! runtime's. A device that cannot run a copy (the views' format features,
//! `shaderStorageImageExtendedFormats` for an extended view format, color attachments of its
//! samples in the view format, which `framebufferIntegerColorSampleCounts` or the format's own
//! counts give) is `NotSupported::NotEnabled`, never a guess.

use super::super::device::VulkanContext;
use super::super::multisample::{assemble_spirv, multisample_initializer_spvasm};
use super::super::pipeline::create_shader_module;
use super::super::plan_resources::{barrier, create_image};
use super::multisampled::view_format;
use super::{Bound, Target, Transfer};
use crate::candidate::engine::graphics::ColorTarget;
use crate::case::TextureFormat;
use crate::check::admission::{ExecutorError, NotSupported};
use crate::executor_contract::multisample::{sample_read, ReadbackView, SampleRead};
use ash::vk;
use metal2vulkan::device::VkFeature;
use metal2vulkan::pipeline::{StorageAttachmentUse, StorageAttachmentView};
use std::ffi::CString;

/// The planes' accesses inside the fragment shader's critical section.
const SHADER: vk::AccessFlags = vk::AccessFlags::from_raw(
    vk::AccessFlags::SHADER_READ.as_raw() | vk::AccessFlags::SHADER_WRITE.as_raw(),
);

/// One storage-form attachment's planes, the views the draw and the copies take of them and of
/// its target, and the two copies' objects, destroyed after the submission that used them
/// completed (or before any submission). A null handle is one not created yet, which each
/// destroy ignores.
pub(super) struct SamplePlanes<'a> {
    device: &'a ash::Device,
    /// The Metal color index of the attachment.
    pub(super) color_index: u32,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// The 2D-array storage view of every layer in the use's format, which the plan's
    /// `WriteSource::TileAttachment` binds.
    pub(super) view: vk::ImageView,
    /// The 2D-array view of every layer in the unsigned-integer format of the texel size: the
    /// split's storage writes, the merge's fetches.
    words: vk::ImageView,
    /// The target's color-attachment view in that format, which the merge writes.
    merged: vk::ImageView,
    /// The target's image.
    target: vk::Image,
    extent: vk::Extent2D,
    samples: u32,
    set_layouts: [vk::DescriptorSetLayout; 2],
    pipeline_layouts: [vk::PipelineLayout; 2],
    pool: vk::DescriptorPool,
    sets: [vk::DescriptorSet; 2],
    modules: [vk::ShaderModule; 2],
    pipelines: [vk::Pipeline; 2],
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
}

/// The split's index in the pairs of [`SamplePlanes`]; the merge's is [`MERGE`].
const SPLIT: usize = 0;
const MERGE: usize = 1;

impl Drop for SamplePlanes<'_> {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `device` by this run (or is null, which each
        // destroy ignores), is destroyed once, and the submission that used it has completed (or
        // never happened). The pool frees its sets.
        unsafe {
            for pipeline in self.pipelines {
                self.device.destroy_pipeline(pipeline, None);
            }
            for module in self.modules {
                self.device.destroy_shader_module(module, None);
            }
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_descriptor_pool(self.pool, None);
            for layout in self.pipeline_layouts {
                self.device.destroy_pipeline_layout(layout, None);
            }
            for layout in self.set_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device.destroy_image_view(self.merged, None);
            self.device.destroy_image_view(self.words, None);
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// The SPIR-V image format of a storage view of `view`'s format, and whether it is an extended
/// one (`StorageImageExtendedFormats`).
fn storage_format(view: ReadbackView) -> (&'static str, bool) {
    match view {
        ReadbackView::R8Uint => ("R8ui", true),
        ReadbackView::R16Uint => ("R16ui", true),
        ReadbackView::R32Uint => ("R32ui", false),
        ReadbackView::Rg32Uint => ("Rg32ui", true),
        ReadbackView::Rgba32Uint => ("Rgba32ui", false),
    }
}

/// Whether the device can run both copies of a target of `samples` samples through views of
/// `words` (`view`'s format): storage writes, fetches, and color-attachment writes of the format
/// with optimal tiling, the storage capability of its SPIR-V format, and color attachments of
/// `samples` samples in it, which every integer color attachment takes
/// (`framebufferIntegerColorSampleCounts`) or the format itself does.
fn copy_support(
    context: &VulkanContext,
    label: &str,
    view: ReadbackView,
    samples: u32,
) -> Result<(), ExecutorError> {
    let words = view_format(view);
    let mut missing = Vec::new();
    // SAFETY: the physical device belongs to the live instance.
    let features = unsafe {
        context
            .instance
            .get_physical_device_format_properties(context.physical, words)
    }
    .optimal_tiling_features;
    let needed = vk::FormatFeatureFlags::STORAGE_IMAGE
        | vk::FormatFeatureFlags::SAMPLED_IMAGE
        | vk::FormatFeatureFlags::COLOR_ATTACHMENT;
    if !features.contains(needed) {
        missing.push(format!(
            "{label}: its copies' view format {} stored, sampled, and attached with optimal \
             tiling",
            words.as_raw()
        ));
    }
    if storage_format(view).1
        && !context
            .enabled
            .features
            .contains(&VkFeature::ShaderStorageImageExtendedFormats)
    {
        missing.push(format!(
            "{label}: shaderStorageImageExtendedFormats (its split writes VkFormat {})",
            words.as_raw()
        ));
    }
    let mut v12 = vk::PhysicalDeviceVulkan12Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut v12);
    // SAFETY: the physical device is the context's; `properties` is live for the call.
    unsafe {
        context
            .instance
            .get_physical_device_properties2(context.physical, &mut properties)
    };
    let counts = v12.framebuffer_integer_color_sample_counts;
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(words)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);
    let mut image = vk::ImageFormatProperties2::default();
    // SAFETY: the physical device is the context's; `image` is live for the call. A format the
    // device takes no such image of has no sample counts.
    let own = match unsafe {
        context
            .instance
            .get_physical_device_image_format_properties2(context.physical, &info, &mut image)
    } {
        Ok(()) => image.image_format_properties.sample_counts,
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => vk::SampleCountFlags::empty(),
        Err(error) => {
            return Err(format!(
                "{label}: vkGetPhysicalDeviceImageFormatProperties2 of its view format {}: \
                 {error}",
                words.as_raw()
            )
            .into())
        }
    };
    let wanted = vk::SampleCountFlags::from_raw(samples);
    if !counts.contains(wanted) && !own.contains(wanted) {
        missing.push(format!(
            "{label}: color attachments of {samples} samples in its merge's view format {} \
             (framebufferIntegerColorSampleCounts {:#b}, the format's {:#b})",
            words.as_raw(),
            counts.as_raw(),
            own.as_raw()
        ));
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(NotSupported::NotEnabled(missing).into())
    }
}

/// A view of `image` of `view_type` and `format` over `layers` layers, usable as `usage`.
fn create_view(
    context: &VulkanContext,
    label: &str,
    image: vk::Image,
    (view_type, format, layers): (vk::ImageViewType, vk::Format, u32),
    usage: vk::ImageUsageFlags,
) -> Result<vk::ImageView, ExecutorError> {
    let mut restricted = vk::ImageViewUsageCreateInfo::default().usage(usage);
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(view_type)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: layers,
        })
        .push_next(&mut restricted);
    // SAFETY: `info` names the live image.
    unsafe { context.device.create_image_view(&info, None) }
        .map_err(|error| ExecutorError::Failed(format!("create {label} view: {error}")))
}

impl<'a> SamplePlanes<'a> {
    /// The planes of storage-form attachment `used`, whose view is
    /// `StorageAttachmentView::SamplePlanes`, over its multisampled `target` ([`Bound::Planes`],
    /// holding `authored`) of `extent` and `samples` samples. Everything made is destroyed on
    /// failure.
    pub(super) fn create(
        context: &'a VulkanContext,
        used: &StorageAttachmentUse,
        (target, authored): (&Target, &ColorTarget),
        extent: vk::Extent2D,
        samples: u32,
    ) -> Result<SamplePlanes<'a>, ExecutorError> {
        let label = format!("color attachment {}'s sample planes", used.color_index);
        let StorageAttachmentView::SamplePlanes { layers } = used.view else {
            return Err(format!("{label}: the use views the attachment itself").into());
        };
        if layers != samples {
            return Err(format!("{label}: {layers} layers in a pass of {samples} samples").into());
        }
        if target.bound != Bound::Planes {
            return Err(format!("{label}: its target is bound as {:?}", target.bound).into());
        }
        let Transfer::PerSample(readback) = &target.transfer else {
            return Err(format!("{label}: its target is read back by copies").into());
        };
        let read = sample_read(authored.format).map_err(ExecutorError::Failed)?;
        let SampleRead::Color(view) = read else {
            return Err(format!("{label}: the depth format {:?}", authored.format).into());
        };
        copy_support(context, &label, view, samples)?;
        let words = view_format(view);
        let format: vk::Format = used.format.into();
        // The draws' storage view: the attachment's format, or the pipeline's alias of it
        // (`StorageAttachmentUse::view_alias`), to which the mutable planes take a view.
        let draw_format: vk::Format = used.view_format().into();
        let mut planes = SamplePlanes {
            device: &context.device,
            color_index: used.color_index,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            words: vk::ImageView::null(),
            merged: vk::ImageView::null(),
            target: target.image,
            extent,
            samples,
            set_layouts: [vk::DescriptorSetLayout::null(); 2],
            pipeline_layouts: [vk::PipelineLayout::null(); 2],
            pool: vk::DescriptorPool::null(),
            sets: [vk::DescriptorSet::null(); 2],
            modules: [vk::ShaderModule::null(); 2],
            pipelines: [vk::Pipeline::null(); 2],
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
        };
        (planes.image, planes.memory) = create_image(
            context,
            &label,
            &vk::ImageCreateInfo::default()
                .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(layers)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
        )?;
        let array = vk::ImageViewType::TYPE_2D_ARRAY;
        planes.view = create_view(
            context,
            &label,
            planes.image,
            (array, draw_format, layers),
            vk::ImageUsageFlags::STORAGE,
        )?;
        planes.words = create_view(
            context,
            &format!("{label}' words"),
            planes.image,
            (array, words, layers),
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
        )?;
        planes.merged = create_view(
            context,
            &format!("{label}' merge target"),
            target.image,
            (vk::ImageViewType::TYPE_2D, words, 1),
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )?;
        planes.create_layouts(context)?;
        planes.write_sets(context, readback.view)?;
        planes.create_split(context, view)?;
        planes.create_merge(context, words)?;
        Ok(planes)
    }

    /// The two copies' set and pipeline layouts, the pool, and their sets: the split's fetched
    /// target and written planes, the merge's fetched planes.
    fn create_layouts(&mut self, context: &VulkanContext) -> Result<(), ExecutorError> {
        let failed = |what: &str| {
            let what = what.to_owned();
            move |error: vk::Result| {
                ExecutorError::Failed(format!("create the sample planes' {what}: {error}"))
            }
        };
        let split = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let merge = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        for (at, bindings) in [(SPLIT, &split[..]), (MERGE, &merge[..])] {
            // SAFETY: each info is live for its call; every object is destroyed by `Drop`.
            unsafe {
                self.set_layouts[at] = context
                    .device
                    .create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings),
                        None,
                    )
                    .map_err(failed("descriptor set layout"))?;
                let set_layouts = [self.set_layouts[at]];
                self.pipeline_layouts[at] = context
                    .device
                    .create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                        None,
                    )
                    .map_err(failed("pipeline layout"))?;
            }
        }
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 1,
            },
        ];
        // SAFETY: each info is live for its call; the pool is destroyed by `Drop` and frees
        // its sets.
        unsafe {
            self.pool = context
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(2)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(failed("descriptor pool"))?;
            let sets = context
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&self.set_layouts),
                )
                .map_err(failed("descriptor sets"))?;
            self.sets = [sets[SPLIT], sets[MERGE]];
        }
        Ok(())
    }

    /// Write the copies' sets: the target's `fetched` view (its readback view, in `GENERAL`
    /// between the initializer and the merge) and the planes' words.
    fn write_sets(
        &self,
        context: &VulkanContext,
        fetched: vk::ImageView,
    ) -> Result<(), ExecutorError> {
        let general = vk::ImageLayout::GENERAL;
        let target = [vk::DescriptorImageInfo::default()
            .image_view(fetched)
            .image_layout(general)];
        let words = [vk::DescriptorImageInfo::default()
            .image_view(self.words)
            .image_layout(general)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.sets[SPLIT])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&target),
            vk::WriteDescriptorSet::default()
                .dst_set(self.sets[SPLIT])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&words),
            vk::WriteDescriptorSet::default()
                .dst_set(self.sets[MERGE])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&words),
        ];
        // SAFETY: the sets and views are live and no submission uses the sets.
        unsafe { context.device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    /// The split's compute pipeline, through views of `view`'s format ([`split_spvasm`]).
    fn create_split(
        &mut self,
        context: &VulkanContext,
        view: ReadbackView,
    ) -> Result<(), ExecutorError> {
        let spv = assemble_spirv("sample-planes-split", &split_spvasm(view))
            .map_err(ExecutorError::Failed)?;
        self.modules[SPLIT] = create_shader_module(context, &spv, "sample planes split")
            .map_err(ExecutorError::Failed)?;
        let entry = CString::new("main").expect("a static entry name");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(self.modules[SPLIT])
                    .name(&entry),
            )
            .layout(self.pipeline_layouts[SPLIT]);
        // SAFETY: `info` is live for the call; the pipeline is destroyed by `Drop`.
        self.pipelines[SPLIT] = unsafe {
            context
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, error)| format!("create the sample planes' split: {error}"))?[0];
        Ok(())
    }

    /// The merge's render pass over the target's view of `words` (kept in `GENERAL`, loaded and
    /// stored), its framebuffer, and its sample-rate pipeline: the initializer's shader for an
    /// unsigned-integer texel, which fetches layer `SampleId` of the planes at its pixel and writes
    /// it to its sample.
    fn create_merge(
        &mut self,
        context: &VulkanContext,
        words: vk::Format,
    ) -> Result<(), ExecutorError> {
        let general = vk::ImageLayout::GENERAL;
        let samples = vk::SampleCountFlags::from_raw(self.samples);
        let attachments = [vk::AttachmentDescription::default()
            .format(words)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(general)
            .final_layout(general)];
        let references = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(general)];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&references)];
        let failed = |what: &str| {
            let what = what.to_owned();
            move |error: vk::Result| {
                ExecutorError::Failed(format!("create the sample planes' merge {what}: {error}"))
            }
        };
        // SAFETY: each info is live for its call; every object is destroyed by `Drop`.
        unsafe {
            self.render_pass = context
                .device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses),
                    None,
                )
                .map_err(failed("render pass"))?;
            let views = [self.merged];
            self.framebuffer = context
                .device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(self.render_pass)
                        .attachments(&views)
                        .width(self.extent.width)
                        .height(self.extent.height)
                        .layers(1),
                    None,
                )
                .map_err(failed("framebuffer"))?;
        }
        let spv = assemble_spirv(
            "sample-planes-merge",
            &multisample_initializer_spvasm(TextureFormat::Rgba32Uint),
        )
        .map_err(ExecutorError::Failed)?;
        self.modules[MERGE] = create_shader_module(context, &spv, "sample planes merge")
            .map_err(ExecutorError::Failed)?;
        let vertex = CString::new("vertex").expect("a static entry name");
        let fragment = CString::new("fragment").expect("a static entry name");
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.modules[MERGE])
                .name(&vertex),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.modules[MERGE])
                .name(&fragment),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        }];
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        // Every sample of every pixel runs once and writes only its own sample.
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(samples)
            .sample_shading_enable(true)
            .min_sample_shading(1.0);
        let blend = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .layout(self.pipeline_layouts[MERGE])
            .render_pass(self.render_pass)
            .subpass(0);
        // SAFETY: `info` is live for the call; the pipeline is destroyed by `Drop`.
        self.pipelines[MERGE] = unsafe {
            context
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        }
        .map_err(|(_, error)| failed("pipeline")(error))?[0];
        Ok(())
    }

    /// Before the pass: split the target into the planes, and make the planes' writes visible to
    /// the fragment shader's storage reads and writes.
    ///
    /// # Safety
    ///
    /// `command` is recording on the planes' device, outside a render pass, after the barrier
    /// that made the target's initialized samples visible to compute reads in `GENERAL`
    /// (`SampleReaders::record_initialized`, with [`Bound::Planes`]'s use).
    pub(super) unsafe fn record_split(&self, command: vk::CommandBuffer) {
        let color = vk::ImageAspectFlags::COLOR;
        let general = vk::ImageLayout::GENERAL;
        // SAFETY: the caller's contract; every handle is live.
        unsafe {
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    self.image,
                    color,
                    self.samples,
                    (vk::AccessFlags::empty(), vk::ImageLayout::UNDEFINED),
                    (vk::AccessFlags::SHADER_WRITE, general),
                )],
            );
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipelines[SPLIT],
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layouts[SPLIT],
                0,
                &[self.sets[SPLIT]],
                &[],
            );
            // One invocation per sample, one per workgroup: no grid edge to guard.
            self.device
                .cmd_dispatch(command, self.extent.width, self.extent.height, self.samples);
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    self.image,
                    color,
                    self.samples,
                    (vk::AccessFlags::SHADER_WRITE, general),
                    (SHADER, general),
                )],
            );
        }
    }

    /// After the pass: merge the planes back into the target, each sample from its layer, after
    /// the fragment shader's last writes of the planes and the split's reads of the target. The
    /// target stays in `GENERAL`, written as a color attachment, which [`Bound::Planes`]'s use
    /// states for the readback's barrier.
    ///
    /// # Safety
    ///
    /// `command` is recording on the planes' device, outside a render pass, after the pass that
    /// drew with the planes.
    pub(super) unsafe fn record_merge(&self, command: vk::CommandBuffer) {
        let color = vk::ImageAspectFlags::COLOR;
        let general = vk::ImageLayout::GENERAL;
        let attachment =
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
        let barriers = [
            barrier(
                self.image,
                color,
                self.samples,
                (SHADER, general),
                (vk::AccessFlags::SHADER_READ, general),
            ),
            barrier(
                self.target,
                color,
                1,
                (vk::AccessFlags::SHADER_READ, general),
                (attachment, general),
            ),
        ];
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };
        // SAFETY: the caller's contract; every handle is live.
        unsafe {
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
            self.device.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(render_area),
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[MERGE],
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layouts[MERGE],
                0,
                &[self.sets[MERGE]],
                &[],
            );
            self.device.cmd_draw(command, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(command);
        }
    }
}

/// The split's compute shader through views of `view`'s format: workgroup `(x, y, s)` fetches
/// sample `s` of pixel `(x, y)` of the multisampled target (binding 0, an unsigned-integer
/// sampled image) and writes it to texel `(x, y)` of layer `s` of the planes (binding 1, an
/// unsigned-integer 2D-array storage image of the view's SPIR-V format).
fn split_spvasm(view: ReadbackView) -> String {
    let (format, extended) = storage_format(view);
    let capability = if extended {
        "OpCapability StorageImageExtendedFormats\n"
    } else {
        ""
    };
    format!(
        r#"OpCapability Shader
{capability}OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %workgroup %source %planes
OpExecutionMode %main LocalSize 1 1 1
OpDecorate %workgroup BuiltIn WorkgroupId
OpDecorate %source DescriptorSet 0
OpDecorate %source Binding 0
OpDecorate %planes DescriptorSet 0
OpDecorate %planes Binding 1
OpDecorate %planes NonReadable
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v2uint = OpTypeVector %uint 2
%v3uint = OpTypeVector %uint 3
%v4uint = OpTypeVector %uint 4
%source_type = OpTypeImage %uint 2D 0 0 1 1 Unknown
%planes_type = OpTypeImage %uint 2D 0 1 0 2 {format}
%ptr_source = OpTypePointer UniformConstant %source_type
%ptr_planes = OpTypePointer UniformConstant %planes_type
%ptr_v3uint = OpTypePointer Input %v3uint
%workgroup = OpVariable %ptr_v3uint Input
%source = OpVariable %ptr_source UniformConstant
%planes = OpVariable %ptr_planes UniformConstant
%main = OpFunction %void None %fn
%entry = OpLabel
%id = OpLoad %v3uint %workgroup
%x = OpCompositeExtract %uint %id 0
%y = OpCompositeExtract %uint %id 1
%s = OpCompositeExtract %uint %id 2
%xy = OpCompositeConstruct %v2uint %x %y
%loaded = OpLoad %source_type %source
%texel = OpImageFetch %v4uint %loaded %xy Sample %s
%written = OpLoad %planes_type %planes
OpImageWrite %written %id %texel
OpReturn
OpFunctionEnd
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The split of each view format is valid Vulkan SPIR-V, declaring the storage capability of
    /// an extended format and only of one; the merge is the initializer's unsigned-integer
    /// shader, valid as the initializer's tests check.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn each_split_is_vulkan_legal_through_its_view_format() {
        for (view, extended) in [
            (ReadbackView::R8Uint, true),
            (ReadbackView::R16Uint, true),
            (ReadbackView::R32Uint, false),
            (ReadbackView::Rg32Uint, true),
            (ReadbackView::Rgba32Uint, false),
        ] {
            let assembly = split_spvasm(view);
            assert_eq!(
                assembly.contains("StorageImageExtendedFormats"),
                extended,
                "{view:?}"
            );
            let bytes = assemble_spirv("sample-planes-split-test", &assembly).unwrap();
            crate::spirv_tools::validate(&bytes, "sample-planes-split-test")
                .unwrap_or_else(|error| panic!("{view:?}: {error}\n{assembly}"));
        }
        let merge = assemble_spirv(
            "sample-planes-merge-test",
            &multisample_initializer_spvasm(TextureFormat::Rgba32Uint),
        )
        .unwrap();
        crate::spirv_tools::validate(&merge, "sample-planes-merge-test").unwrap();
    }
}
