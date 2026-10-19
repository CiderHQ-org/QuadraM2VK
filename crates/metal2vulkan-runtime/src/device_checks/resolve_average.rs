//! The check of `DriverFact::ColorResolveAverageIsMetals` over the formats Metal's resolve rules
//! close (`metal2vulkan::bind::ColorResolveFormat::of`), in the background: its reader takes
//! it at each resolve it plans (`color_resolve::average_is_metals`), for those formats alone; a
//! format without Metal's rules at its count takes the fact only where the tables hold it too.
//!
//! For each such format at 2 and 4 samples (where its rules hold at that count) that the device
//! renders at that count, one submission: sample planes [`SIDE`] pixels square of random texels,
//! copied in from a buffer, are merged into a multisampled attachment of the format through its
//! bits view (`metal2vulkan::bind::planes_merge_modules`, as the runtime merges planes back),
//! which a pass of no draws then loads and resolves by the device's `AVERAGE` into one image;
//! and the crate's exact resolve (`metal2vulkan::bind::color_resolve_modules` from the planes,
//! Metal's rules) writes a second image. Both are copied out. Metal's answer: every texel of every
//! format alike in both; the other where one differs (RADV's, ANV's, NVIDIA's and lavapipe's
//! averages round an 8-bit tie down). A format the device renders at a count but not through its
//! bits view there answers neither.
//!
//! It does not run where the device enabled no `sampleRateShading` (the merge's), or on a
//! portability-subset device, whose views of another component layout its consumer enables or
//! not (MoltenVK's, whose resolve is Metal's own by construction: the tables decide there).

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster;
use super::Target;
use ash::vk;
use metal2vulkan::bind::{
    color_resolve_modules, planes_merge_modules, ColorResolveFormat, ColorResolveModules,
    PlanesCopy, PlanesMergeModules, ResolveSource,
};
use metal2vulkan::device::{CheckAnswer, VkExtension, VkFeature};
use metal2vulkan::error::Feature;
use metal2vulkan::vk::Format;

/// The planes' and the attachments' side, in pixels.
const SIDE: u32 = 64;

/// The sample counts compared: those Apple's GPUs make.
const COUNTS: [u32; 2] = [2, 4];

/// What a readback holds before its copy.
const UNWRITTEN: u32 = 0xdead_beef;

// The merge and the exact resolve read the planes through one set: its binding is both's.
const _: () = assert!(
    ColorResolveModules::SET == 0
        && ColorResolveModules::SOURCE_BINDING == PlanesMergeModules::PLANES_BINDING
);

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    if !(target.enabled.features).contains(&VkFeature::SampleRateShading) {
        return CheckAnswer::NotRun("sampleRateShading is not enabled".to_owned());
    }
    if (target.enabled.extensions).contains(&VkExtension::KhrPortabilitySubset) {
        return CheckAnswer::NotRun(
            "a portability-subset device's views of another component layout are its \
             consumer's to enable"
                .to_owned(),
        );
    }
    let mut merges = Vec::new();
    let mut compared = 0usize;
    let mut differed = Vec::new();
    for &format in Format::KNOWN {
        let Some(rules) = ColorResolveFormat::of(format) else {
            continue;
        };
        let Some(copy) = PlanesCopy::for_planes(format) else {
            continue;
        };
        for samples in COUNTS {
            if !rules.resolves(samples) {
                continue;
            }
            let case = Case {
                rules,
                copy,
                samples,
            };
            match case.resolved(kit, target, &mut merges) {
                Ok(Resolved::NotRendered) => {}
                Ok(Resolved::Unfilled) => {
                    return CheckAnswer::Unanswered(format!(
                        "{format:?} renders at {samples} samples, but not through its bits view"
                    ))
                }
                Ok(Resolved::Both { device, metals }) => {
                    compared += 1;
                    if let Some(off) = case.differs(&device, &metals) {
                        differed.push(off);
                    }
                }
                Err(error) => return CheckAnswer::Unanswered(error.0),
            }
        }
    }
    match (compared, differed.first()) {
        (0, _) => CheckAnswer::NotRun(
            "the device renders no format of Metal's resolve rules multisampled".to_owned(),
        ),
        (_, None) => CheckAnswer::Metals,
        (_, Some(first)) => CheckAnswer::NotMetals(format!(
            "the device's average differs from Metal's rules at {} of {compared} formats and \
             counts, the first {first}",
            differed.len()
        )),
    }
}

/// What a [`Case`] found.
enum Resolved {
    /// The device renders no attachment of the format at the count: no resolve of it.
    NotRendered,
    /// The device renders the format at the count but not its bits view there.
    Unfilled,
    /// The texels' words, the device's resolve's, then Metal's rules'.
    Both { device: Vec<u32>, metals: Vec<u32> },
}

/// The merge's objects for one copy format and count, made once.
#[derive(Clone, Copy)]
struct Merge {
    copy: PlanesCopy,
    samples: u32,
    pass: vk::RenderPass,
    pipeline: vk::Pipeline,
}

/// One format at one sample count.
struct Case {
    rules: ColorResolveFormat,
    copy: PlanesCopy,
    samples: u32,
}

/// A colour attachment description of `format` at `samples`.
fn attachment(
    format: Format,
    samples: u32,
    (load, store): (vk::AttachmentLoadOp, vk::AttachmentStoreOp),
    (initial, last): (vk::ImageLayout, vk::ImageLayout),
) -> vk::AttachmentDescription2<'static> {
    vk::AttachmentDescription2::default()
        .format(format.into())
        .samples(raster::sample_count(samples))
        .load_op(load)
        .store_op(store)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(initial)
        .final_layout(last)
}

/// A reference to colour attachment `index`.
fn colour_ref(index: u32) -> vk::AttachmentReference2<'static> {
    vk::AttachmentReference2::default()
        .attachment(index)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR)
}

/// A one-subpass pass of `attachments`, its colour attachment 0 and resolve attachment
/// `resolve`, behind a dependency of the earlier colour attachment writes and fragment shader
/// reads, and ahead of one of later colour attachment accesses and transfer reads on its writes.
fn one_pass(
    kit: &mut Kit<'_>,
    attachments: &[vk::AttachmentDescription2<'_>],
    resolve: Option<u32>,
) -> Result<vk::RenderPass, CheckError> {
    use vk::AccessFlags as A;
    use vk::PipelineStageFlags as S;
    let colours = [colour_ref(0)];
    let resolves: Vec<_> = resolve.map(colour_ref).into_iter().collect();
    let mut subpass = vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colours);
    if !resolves.is_empty() {
        subpass = subpass.resolve_attachments(&resolves);
    }
    let subpasses = [subpass];
    let outputs = S::COLOR_ATTACHMENT_OUTPUT;
    let accesses = A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE;
    let dependencies = [
        vk::SubpassDependency2::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(outputs)
            .dst_stage_mask(outputs | S::FRAGMENT_SHADER)
            .src_access_mask(A::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(accesses | A::SHADER_READ),
        vk::SubpassDependency2::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(outputs)
            .dst_stage_mask(outputs | S::TRANSFER)
            .src_access_mask(A::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(accesses | A::TRANSFER_READ),
    ];
    kit.render_pass(
        &vk::RenderPassCreateInfo2::default()
            .attachments(attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies),
    )
}

/// The framebuffer of `pass` over `views`, [`SIDE`] square.
fn framebuffer(
    kit: &mut Kit<'_>,
    pass: vk::RenderPass,
    views: &[vk::ImageView],
) -> Result<vk::Framebuffer, CheckError> {
    kit.framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(views)
            .width(SIDE)
            .height(SIDE)
            .layers(1),
    )
}

/// A [`SIDE`]-square 2D image of `format`, `layers` layers at `samples`, of `usage`, mutable to
/// its other views' formats where `mutable`, kept by the kit.
fn image(
    kit: &mut Kit<'_>,
    format: Format,
    (layers, samples): (u32, u32),
    usage: vk::ImageUsageFlags,
    mutable: bool,
) -> Result<vk::Image, CheckError> {
    let flags = match mutable {
        true => vk::ImageCreateFlags::MUTABLE_FORMAT,
        false => vk::ImageCreateFlags::empty(),
    };
    kit.image(
        &vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(format.into())
            .extent(vk::Extent3D {
                width: SIDE,
                height: SIDE,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(layers)
            .samples(raster::sample_count(samples))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    )
}

/// A view of `image`'s colour aspect in `format`, of `layers` layers (an array view where more
/// than one), kept by the kit.
fn view(
    kit: &mut Kit<'_>,
    image: vk::Image,
    format: Format,
    layers: u32,
) -> Result<vk::ImageView, CheckError> {
    let view_type = match layers {
        1 => vk::ImageViewType::TYPE_2D,
        _ => vk::ImageViewType::TYPE_2D_ARRAY,
    };
    kit.view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format.into())
            .subresource_range(colour_layers(layers)),
    )
}

/// The colour aspect of level 0's first `layers` layers.
fn colour_layers(layers: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: layers,
    }
}

/// The copy between a buffer and level 0's first `layers` layers, tight, [`SIDE`] square.
fn buffer_copy(layers: u32) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: layers,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width: SIDE,
            height: SIDE,
            depth: 1,
        },
    }
}

/// The pipeline of `stages` in subpass 0 of `pass` at `samples` (sample shading on at 1 where
/// more than one), over the [`SIDE`]-square target, writing its colour attachment whole, unblended.
fn pipeline(
    kit: &mut Kit<'_>,
    (vertex, fragment, layout): (vk::ShaderModule, vk::ShaderModule, vk::PipelineLayout),
    pass: vk::RenderPass,
    samples: u32,
) -> Result<vk::Pipeline, vk::Result> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewports = [raster::viewport(0, 0, SIDE, SIDE)];
    let scissors = [raster::rect(0, 0, SIDE, SIDE)];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(raster::sample_count(samples))
        .sample_shading_enable(samples > 1)
        .min_sample_shading(1.0);
    let written = [vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&written);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .layout(layout)
        .render_pass(pass)
        .subpass(0);
    kit.graphics_pipeline(&info)
}

/// The words of `bytes` random bytes from `seed` (a 64-bit xorshift).
fn random_words(bytes: usize, seed: u64) -> Vec<u32> {
    let mut state = seed | 1;
    (0..bytes.div_ceil(4))
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u32
        })
        .collect()
}

impl Case {
    /// The texel's bytes.
    fn texel(&self) -> usize {
        self.rules
            .format()
            .block()
            .map_or(0, |block| block.size as usize)
    }

    /// The device's resolve and Metal's rules' of random texels (the module docs), or why there
    /// are none.
    fn resolved(
        &self,
        kit: &mut Kit<'_>,
        target: &Target<'_>,
        merges: &mut Vec<Merge>,
    ) -> Result<Resolved, CheckError> {
        use vk::ImageUsageFlags as U;
        let (format, samples) = (self.rules.format(), self.samples);
        let bits = self.copy.format();
        let colour = U::COLOR_ATTACHMENT;
        if !raster::takes_samples(kit, format.into(), colour, samples)?
            || !raster::takes_samples(kit, format.into(), colour | U::TRANSFER_SRC, 1)?
        {
            return Ok(Resolved::NotRendered);
        }
        if !raster::takes_samples(kit, bits.into(), colour, samples)?
            || !raster::takes_samples(kit, bits.into(), U::SAMPLED | U::TRANSFER_DST, 1)?
        {
            return Ok(Resolved::Unfilled);
        }
        let written = self.rules.written_raw();
        let exact = color_resolve_modules(
            target.compile,
            written,
            samples,
            ResolveSource::Planes(self.copy),
            Feature::Format,
        )
        .map_err(|e| CheckError(format!("the exact resolve does not build: {e}")))?;
        let texel = self.texel();
        let planes_bytes = (SIDE * SIDE * samples) as usize * texel;
        let seed = (u64::from(format.as_raw() as u32) << 8) | u64::from(samples);
        let upload = kit.host_buffer(
            &random_words(planes_bytes, seed),
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;
        let resolved_words = ((SIDE * SIDE) as usize * texel).div_ceil(4);
        let unwritten = vec![UNWRITTEN; resolved_words];
        let device_words = kit.host_buffer(&unwritten, vk::BufferUsageFlags::TRANSFER_DST)?;
        let metals_words = kit.host_buffer(&unwritten, vk::BufferUsageFlags::TRANSFER_DST)?;
        let planes = image(kit, bits, (samples, 1), U::SAMPLED | U::TRANSFER_DST, false)?;
        let planes_view = view(kit, planes, bits, samples)?;
        let merged = image(kit, format, (1, samples), colour, bits != format)?;
        let merged_bits = view(kit, merged, bits, 1)?;
        let merged_own = view(kit, merged, format, 1)?;
        let by_device = image(kit, format, (1, 1), colour | U::TRANSFER_SRC, false)?;
        let by_device_view = view(kit, by_device, format, 1)?;
        let raw = written.written_format();
        let by_rules = image(kit, format, (1, 1), colour | U::TRANSFER_SRC, raw != format)?;
        let by_rules_view = view(kit, by_rules, raw, 1)?;
        let (set_layout, set) = kit.descriptor_set(&[SetBinding {
            binding: PlanesMergeModules::PLANES_BINDING,
            stages: vk::ShaderStageFlags::FRAGMENT,
            resource: SetResource::SampledImage(
                planes_view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ),
        }])?;
        let layout = kit.layout(&[set_layout], None)?;
        let merge = self.merge(kit, target, merges, layout)?;
        let merge_framebuffer = framebuffer(kit, merge.pass, &[merged_bits])?;
        use vk::AttachmentLoadOp as Load;
        use vk::AttachmentStoreOp as Store;
        use vk::ImageLayout as L;
        let attachment_layout = L::COLOR_ATTACHMENT_OPTIMAL;
        let average_pass = one_pass(
            kit,
            &[
                attachment(
                    format,
                    samples,
                    (Load::LOAD, Store::DONT_CARE),
                    (attachment_layout, attachment_layout),
                ),
                attachment(
                    format,
                    1,
                    (Load::DONT_CARE, Store::STORE),
                    (L::UNDEFINED, L::TRANSFER_SRC_OPTIMAL),
                ),
            ],
            Some(1),
        )?;
        let average_framebuffer = framebuffer(kit, average_pass, &[merged_own, by_device_view])?;
        let rules_pass = one_pass(
            kit,
            &[attachment(
                raw,
                1,
                (Load::DONT_CARE, Store::STORE),
                (L::UNDEFINED, L::TRANSFER_SRC_OPTIMAL),
            )],
            None,
        )?;
        let rules_framebuffer = framebuffer(kit, rules_pass, &[by_rules_view])?;
        let vertex = kit.shader(&exact.vertex)?;
        let fragment = kit.shader(&exact.fragment)?;
        let rules_pipeline = pipeline(kit, (vertex, fragment, layout), rules_pass, 1)?;
        let area = raster::rect(0, 0, SIDE, SIDE);
        kit.submit(|device, command| {
            use vk::AccessFlags as A;
            use vk::PipelineStageFlags as S;
            let barrier = |old, new, src, dst| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(src)
                    .dst_access_mask(dst)
                    .old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(planes)
                    .subresource_range(colour_layers(samples))
            };
            let fresh = barrier(
                L::UNDEFINED,
                L::TRANSFER_DST_OPTIMAL,
                A::empty(),
                A::TRANSFER_WRITE,
            );
            let uploaded = barrier(
                L::TRANSFER_DST_OPTIMAL,
                L::SHADER_READ_ONLY_OPTIMAL,
                A::TRANSFER_WRITE,
                A::SHADER_READ,
            );
            let begin = |pass, framebuffer| {
                vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(framebuffer)
                    .render_area(area)
            };
            let no_flags = vk::DependencyFlags::empty();
            let graphics = vk::PipelineBindPoint::GRAPHICS;
            let inline = vk::SubpassContents::INLINE;
            // SAFETY: recording; every object named is the kit's and live. The planes move
            // through the layouts their barriers name, and are read by both draws behind the
            // second; each pass's dependencies order its attachments' accesses after the earlier
            // passes' writes and the copies out after its own, as do the barriers after the merge
            // and the last pass, and each image is in the layout its pass or copy names.
            unsafe {
                device.cmd_pipeline_barrier(
                    command,
                    S::TOP_OF_PIPE,
                    S::TRANSFER,
                    no_flags,
                    &[],
                    &[],
                    &[fresh],
                );
                let copy = [buffer_copy(samples)];
                let dst = L::TRANSFER_DST_OPTIMAL;
                device.cmd_copy_buffer_to_image(command, upload.buffer, planes, dst, &copy);
                device.cmd_pipeline_barrier(
                    command,
                    S::TRANSFER,
                    S::FRAGMENT_SHADER,
                    no_flags,
                    &[],
                    &[],
                    &[uploaded],
                );
                device.cmd_begin_render_pass(
                    command,
                    &begin(merge.pass, merge_framebuffer),
                    inline,
                );
                device.cmd_bind_pipeline(command, graphics, merge.pipeline);
                device.cmd_bind_descriptor_sets(command, graphics, layout, 0, &[set], &[]);
                device.cmd_draw(command, 3, 1, 0, 0);
                device.cmd_end_render_pass(command);
                // Beside each pass's dependency, which alone let MoltenVK 1.4.1 read a
                // multisampled pass's attachment before the pass stored it, where the submission
                // wrote timestamps.
                let colours = S::COLOR_ATTACHMENT_OUTPUT;
                let stored = |dst| {
                    vk::MemoryBarrier::default()
                        .src_access_mask(A::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(dst)
                };
                let merged = [stored(A::COLOR_ATTACHMENT_READ | A::COLOR_ATTACHMENT_WRITE)];
                device.cmd_pipeline_barrier(command, colours, colours, no_flags, &merged, &[], &[]);
                let averaged = begin(average_pass, average_framebuffer);
                device.cmd_begin_render_pass(command, &averaged, inline);
                device.cmd_end_render_pass(command);
                device.cmd_begin_render_pass(
                    command,
                    &begin(rules_pass, rules_framebuffer),
                    inline,
                );
                device.cmd_bind_pipeline(command, graphics, rules_pipeline);
                device.cmd_bind_descriptor_sets(command, graphics, layout, 0, &[set], &[]);
                device.cmd_draw(command, 3, 1, 0, 0);
                device.cmd_end_render_pass(command);
                let resolved = [stored(A::TRANSFER_READ)];
                device.cmd_pipeline_barrier(
                    command,
                    colours,
                    S::TRANSFER,
                    no_flags,
                    &resolved,
                    &[],
                    &[],
                );
                let src = L::TRANSFER_SRC_OPTIMAL;
                let out = [buffer_copy(1)];
                device.cmd_copy_image_to_buffer(command, by_device, src, device_words.buffer, &out);
                device.cmd_copy_image_to_buffer(command, by_rules, src, metals_words.buffer, &out);
                raster::host_reads(device, command, S::TRANSFER);
            }
        })?;
        Ok(Resolved::Both {
            device: kit.read(&device_words)?,
            metals: kit.read(&metals_words)?,
        })
    }

    /// The merge's objects for the case's copy format and count under `layout`, made the first
    /// time and kept in `merges`.
    fn merge(
        &self,
        kit: &mut Kit<'_>,
        target: &Target<'_>,
        merges: &mut Vec<Merge>,
        layout: vk::PipelineLayout,
    ) -> Result<Merge, CheckError> {
        let (copy, samples) = (self.copy, self.samples);
        if let Some(&made) = merges
            .iter()
            .find(|m| m.copy == copy && m.samples == samples)
        {
            return Ok(made);
        }
        let modules = planes_merge_modules(target.compile, copy)
            .map_err(|e| CheckError(format!("the planes merge does not build: {e}")))?;
        use vk::ImageLayout as L;
        let pass = one_pass(
            kit,
            &[attachment(
                copy.format(),
                samples,
                (
                    vk::AttachmentLoadOp::DONT_CARE,
                    vk::AttachmentStoreOp::STORE,
                ),
                (L::UNDEFINED, L::COLOR_ATTACHMENT_OPTIMAL),
            )],
            None,
        )?;
        let vertex = kit.shader(&modules.vertex)?;
        let fragment = kit.shader(&modules.fragment)?;
        let pipeline = pipeline(kit, (vertex, fragment, layout), pass, samples)?;
        let made = Merge {
            copy,
            samples,
            pass,
            pipeline,
        };
        merges.push(made);
        Ok(made)
    }

    /// Where the device's resolve differs from Metal's rules', described: how many texels, and
    /// the first one's bytes in each. `None` where none does.
    fn differs(&self, device: &[u32], metals: &[u32]) -> Option<String> {
        let texel = self.texel();
        let bytes =
            |words: &[u32]| -> Vec<u8> { words.iter().flat_map(|w| w.to_le_bytes()).collect() };
        let (device, metals) = (bytes(device), bytes(metals));
        let texels = (SIDE * SIDE) as usize;
        let at = |words: &[u8], i: usize| words[i * texel..(i + 1) * texel].to_vec();
        let off: Vec<usize> = (0..texels)
            .filter(|&i| at(&device, i) != at(&metals, i))
            .collect();
        let &first = off.first()?;
        Some(format!(
            "{:?} at {} samples, {} of {texels} texels, texel {first} {:02x?} for Metal's {:02x?}",
            self.rules.format(),
            self.samples,
            off.len(),
            at(&device, first),
            at(&metals, first)
        ))
    }
}
