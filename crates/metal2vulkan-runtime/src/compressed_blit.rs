//! Blits through a texture Metal keeps losslessly compressed.
//!
//! **Metal's answers** (an Apple M4 under macOS 27.0.1, plain; random bytes through every blit
//! direction, every level, slice and texture type, heap textures and views; the package share-cblit
//! on an Apple M3 Max under 26.7, every line of the predicate and the rules identical to the M4's;
//! and the M3 Max under 27.0.1: the rules and the predicate identical to the M4's at every texel of
//! 11 formats' blits and CPU copies). Metal keeps a texture losslessly compressed where
//! `texture/heap_tiled.rs`'s rule holds (`metal_compresses`: `allowGPUOptimizedContents`, the
//! shorter side of level 0 16 texels or more, a colour format of neither `ShaderWrite` nor
//! `PixelFormatView` usage). A GPU blit that reads or writes such a texture (a buffer into it, it
//! into a buffer, a texture into another either way) writes the texels the compression keeps: for
//! `BGRA10_XR` (and its `_sRGB` twin), `RG11B10Float` and `RGB9E5Float` other bytes than it read,
//! by the rule of `bind::CompressedBlit`, at every level of the texture; the 70 other formats the
//! census tried keep their bytes. A texture over a buffer or of an IOSurface is not compressed, and
//! CPU copies (`replace_region`, `get_bytes`) keep the bytes whatever the texture.
//!
//! **Held**: the runtime's copies keep the bytes on every device, so after a blit that reads or
//! writes such a texture the runtime rewrites what the blit wrote, in the same command of the
//! encoder: the written texels copied into a scratch buffer of the encoder, tight, a dispatch of
//! `bind::compressed_blit_module` of each rule that applies (the source's, then the
//! destination's), and the texels copied back: into the destination texture's image, or its
//! buffer's rows (a texture over a buffer), or the destination buffer's rows, a later row keeping
//! the bytes it shares with an earlier one as the copy left them. The rule is idempotent, so on
//! MoltenVK, whose copies of a compressed image may already make the rewrite, it changes nothing
//! more. There a CPU copy, which the runtime makes a Vulkan copy, would make it too
//! (`RuntimeDriverDefect::CompressedImageCopiesRewriteTexels`: an `RG11B10Float` or
//! `RGB9E5Float` image), so the image of a `Shared` or `Managed` texture that
//! takes a rule is made uncompressed there, and so is a device's own texture's without
//! `allowGPUOptimizedContents`, whose blits Metal makes on an uncompressed texture: its image of
//! the texture's own usage would be compressed (`Device::copy_exact_usage`; a heap's texture still
//! takes the compressed image's bytes of the heap).
//!
//! **Not measured**: a texture of many samples (the runtime leaves its bytes), and a region
//! copied onto itself (nothing is recorded for it, as before). Each Metal level states whether
//! its release was measured (`MetalLevel::compressed_blit`; both are).

use super::blit::{reads, writes, BlitCommandEncoder, TRANSFER};
use super::blit_plan::{BufferRange, RowsCopy};
use super::copy_plan::ImageRegion;
use super::error::{internal, Error};
use super::owned_kernel::{OwnedModule, OwnedSets};
use super::texture::heap_tiled::metal_compresses;
use super::texture::{buffer_image_copy, Texture};
use super::texture_descriptor::{TextureDescriptor, TextureLayout};
use super::tracker::{Access, Barrier};
use super::values::{ResourceOptions, StorageMode};
use super::{Buffer, Device, MetalLevel};
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{CompressedBlit, CompressedBlitModule};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::format::is_depth_stencil;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::Format;
use std::ops::Range;

/// The rule of the texels Metal's GPU blits write where `texture` is a side of the blit: its
/// format's (`CompressedBlit::of`) where Metal keeps the texture losslessly compressed (a texture
/// of an image, single-sampled, whose texture, or a view's first texture, `metal_compresses`
/// holds for) and the device's Metal level measured the rule; `None` elsewhere.
pub(super) fn rule_of(texture: &Texture) -> Option<CompressedBlit> {
    if texture.over_buffer() {
        return None;
    }
    let mut root = texture;
    while let Some(parent) = root.parent_texture() {
        root = parent;
    }
    rule_for(
        texture.pixel_format(),
        root.texture_type(),
        [root.width(), root.height()],
        texture.sample_count(),
        (root.usage(), root.allow_gpu_optimized_contents()),
        texture.device().properties().metal_level,
    )
}

/// The rule of Metal's GPU blits of a texture of `pixel_format` whose texture (a view's first) is
/// of `texture_type`, its level 0 `width` by `height`, of `samples` samples and the
/// `(usage, allowGPUOptimizedContents)` of `optimized`, at the Metal `level` (`rule_of`).
fn rule_for(
    pixel_format: PixelFormat,
    texture_type: TextureType,
    [width, height]: [u64; 2],
    samples: u64,
    (usage, allow_gpu_optimized_contents): (TextureUsage, bool),
    level: MetalLevel,
) -> Option<CompressedBlit> {
    let one_row = matches!(texture_type, TextureType::Type1D | TextureType::Type1DArray);
    let level0 = [width, if one_row { 1 } else { height }];
    let colour = !is_depth_stencil(pixel_format);
    let compressed =
        samples == 1 && metal_compresses(colour, usage, allow_gpu_optimized_contents, level0);
    compressed
        .then(|| level.compressed_blit(pixel_format))
        .flatten()
}

/// Whether a driver of `RuntimeDriverDefect::CompressedImageCopiesRewriteTexels` rewrites texels
/// in its copies of a compressed image of `format`: the Vulkan formats MoltenVK makes Metal's
/// `RG11B10Float` and `RGB9E5Float` textures of. `BGRA10_XR`'s images are of a format whose
/// copies keep their bytes.
fn copies_rewrite(format: Format) -> bool {
    matches!(
        format,
        Format::B10G11R11_UFLOAT_PACK32 | Format::E5B9G9R9_UFLOAT_PACK32
    )
}

impl Device {
    /// `usage`, the usage of the image of `format` of a texture of `descriptor` (checked as
    /// `layout`), with `PixelFormatView` where this device's copies of the image would rewrite
    /// texels Metal keeps (`RuntimeDriverDefect::CompressedImageCopiesRewriteTexels`): an image
    /// of a format whose copies rewrite, which the device's Metal keeps compressed (an image of
    /// `usage` with GPU-optimized contents takes a rule, `rule_for`), of a texture whose bytes
    /// Metal keeps through such a copy: a `Shared` or `Managed` texture, which CPU copies read
    /// and write, or one without `allowGPUOptimizedContents`, which Metal keeps uncompressed, so
    /// that its GPU blits keep the bytes too. Metal keeps an image of that usage uncompressed
    /// (MoltenVK makes a `PixelFormatView` Metal texture of a `MUTABLE_FORMAT` image with no list
    /// of formats), so its copies keep the bytes, and the runtime rewrites what a blit writes by
    /// Metal's rule where Metal's texture takes one (`rule_of`, which reads the texture's own
    /// usage and property). `None` elsewhere: a `Private` texture with GPU-optimized contents,
    /// whose bytes no CPU copy reads or writes, keeps the compression.
    pub(super) fn copy_exact_usage(
        &self,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        format: Format,
        usage: TextureUsage,
    ) -> Option<TextureUsage> {
        let properties = self.properties();
        let defect = RuntimeDriverDefect::CompressedImageCopiesRewriteTexels;
        let reachable = matches!(
            descriptor.resource_options.storage_mode(),
            Ok(StorageMode::Shared | StorageMode::Managed)
        );
        let [width, height, _] = layout.extent;
        let compressed = rule_for(
            layout.pixel_format,
            layout.texture_type,
            [u64::from(width), u64::from(height)],
            u64::from(layout.samples),
            (usage, true),
            properties.metal_level,
        );
        let kept = reachable || !descriptor.allow_gpu_optimized_contents;
        let rewrites = properties.driver_defects.contains(&defect)
            && kept
            && copies_rewrite(format)
            && compressed.is_some();
        rewrites.then_some(usage | TextureUsage::PIXEL_FORMAT_VIEW)
    }
}

/// The rules a blit from `source` into `destination` takes, the source's first, each once.
pub(super) fn rules_of(source: &Texture, destination: &Texture) -> Vec<CompressedBlit> {
    let mut rules: Vec<CompressedBlit> = Vec::with_capacity(2);
    for rule in [rule_of(source), rule_of(destination)]
        .into_iter()
        .flatten()
    {
        if !rules.contains(&rule) {
            rules.push(rule);
        }
    }
    rules
}

/// The bytes of one dispatch's band of words and the dispatches a band of `bytes` takes, for
/// texels of `texel` bytes: within `maxStorageBufferRange` and `maxComputeWorkGroupCount[0]`
/// workgroups, a multiple of the storage buffers' offset alignment (and so of the texel).
fn bands(encoder: &mut BlitCommandEncoder, bytes: u64, texel: u64) -> (u64, Vec<Range<u64>>) {
    let device = encoder.core().device().clone();
    let limits = &device.properties().description.limits;
    let align = limits.min_storage_buffer_offset_alignment.max(texel).max(8);
    let per_dispatch = u64::from(CompressedBlitModule::LOCAL_SIZE)
        * u64::from(limits.max_compute_work_group_count[0].max(1))
        * 4;
    let most = u64::from(limits.max_storage_buffer_range).min(per_dispatch);
    let band = (most / align).max(1) * align;
    let ranges = (0..bytes)
        .step_by(band as usize)
        .map(|start| start..(start + band).min(bytes))
        .collect();
    (align, ranges)
}

impl BlitCommandEncoder {
    /// Rewrite `regions` (of the image's levels and layers, in colour) of `texture`'s image,
    /// which the encoder's last command wrote, as Metal's blit writes them by `rules` in turn
    /// (the module docs): one command of the encoder.
    pub(super) fn compressed_image(
        &mut self,
        texture: &Texture,
        regions: &[ImageRegion],
        rules: &[CompressedBlit],
    ) -> Result<(), Error> {
        let Some(texel) = texel_bytes(rules)? else {
            return Ok(());
        };
        let count = |r: &ImageRegion| {
            let [w, h, d] = r.size;
            u64::from(w) * u64::from(h) * u64::from(d) * u64::from(r.layers)
        };
        let bytes: u64 = regions.iter().map(|r| count(r) * texel).sum();
        if bytes == 0 {
            return Ok(());
        }
        let mut at = 0;
        let mut copies = Vec::with_capacity(regions.len());
        let mut uses = Vec::with_capacity(regions.len());
        let access = Access::new(
            TRANSFER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
        );
        for region in regions {
            copies.push((at, *region));
            at += count(region) * texel;
            let levels = region.level..region.level + 1;
            let layers = region.layer..region.layer + region.layers;
            uses.push(texture.image_use(levels, layers, access, true)?);
        }
        let image = texture.raw_image()?;
        let rewrite = self.rewrite(bytes, texel, rules)?;
        let (raw_scratch, base) = rewrite.scratch;
        let regions: Vec<vk::BufferImageCopy> = copies
            .iter()
            .map(|(at, region)| vk::BufferImageCopy {
                buffer_offset: base + at,
                ..buffer_image_copy(region)
            })
            .collect();
        let stages = TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        self.command_in(stages, &uses, |device, raw| {
            // SAFETY: the command buffer is recording, outside a render pass; the image, the
            // scratch and the sets are retained until it completes; the image is in the
            // `GENERAL` layout; the regions lie in the image (the copy's) and the scratch; each
            // step reads what the one before it wrote, past a barrier.
            unsafe {
                device.cmd_copy_image_to_buffer(
                    raw,
                    image,
                    vk::ImageLayout::GENERAL,
                    raw_scratch,
                    &regions,
                );
                rewrite.record(device, raw);
                device.cmd_copy_buffer_to_image(
                    raw,
                    raw_scratch,
                    image,
                    vk::ImageLayout::GENERAL,
                    &regions,
                );
            }
        })?;
        rewrite.retain(self);
        self.core().retain(texture.clone());
        Ok(())
    }

    /// Rewrite `rows` of `buffer`'s bytes (its own offsets, each whole texels, in the order the
    /// encoder's last command wrote them), which hold the texels a blit wrote, as Metal's blit
    /// writes them by `rules` in turn (the module docs): one command of the encoder. Where rows
    /// overlap, the later one keeps the bytes they share, as the copy left them.
    pub(super) fn compressed_rows(
        &mut self,
        buffer: &Buffer,
        rows: &[Range<u64>],
        rules: &[CompressedBlit],
    ) -> Result<(), Error> {
        let Some(texel) = texel_bytes(rules)? else {
            return Ok(());
        };
        let mut tight = Vec::with_capacity(rows.len());
        let mut at = 0;
        for row in rows {
            let size = row.end - row.start;
            if !size.is_multiple_of(texel) {
                return Err(internal("a compressed blit row of part of a texel"));
            }
            tight.push(at..at + size);
            at += size;
        }
        if at == 0 {
            return Ok(());
        }
        let out_of = RowsCopy::new(rows, &tight);
        let back = RowsCopy::new(&tight, rows);
        let rewrite = self.rewrite(at, texel, rules)?;
        let (raw_scratch, scratch_base) = rewrite.scratch;
        // Read as a blit's (`host_residency.rs`): the command buffer holds the import.
        let (raw_buffer, base) = self.core().blit_at(buffer, 0)?;
        let copy = |rows: &[BufferRange], (src, src_base), (dst, dst_base)| {
            let regions: Vec<vk::BufferCopy> = rows
                .iter()
                .map(|row| vk::BufferCopy {
                    src_offset: src_base + row.src,
                    dst_offset: dst_base + row.dst,
                    size: row.size,
                })
                .collect();
            (src, dst, regions)
        };
        let into_scratch = copy(
            &out_of.rows,
            (raw_buffer, base),
            (raw_scratch, scratch_base),
        );
        let into_buffer = copy(&back.rows, (raw_scratch, scratch_base), (raw_buffer, base));
        let span = out_of.src_span.clone();
        let uses = [reads(buffer, span.clone()), writes(buffer, span)];
        let stages = TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        self.command_in(stages, &uses, |device, raw| {
            // SAFETY: the command buffer is recording, outside a render pass; the buffer, the
            // scratch and the sets are retained until it completes; the rows lie in both (the
            // copy's rows, the scratch's tight ones), and no destination row of one copy meets
            // another (`RowsCopy`); each step reads what the one before it wrote, past a barrier.
            unsafe {
                let (src, dst, regions) = &into_scratch;
                device.cmd_copy_buffer(raw, *src, *dst, regions);
                rewrite.record(device, raw);
                let (src, dst, regions) = &into_buffer;
                device.cmd_copy_buffer(raw, *src, *dst, regions);
            }
        })?;
        rewrite.retain(self);
        self.core().retain(buffer.clone());
        Ok(())
    }

    /// A scratch buffer of `bytes` (texels of `texel` bytes) and the dispatches of `rules` over
    /// it, in bands.
    fn rewrite(
        &mut self,
        bytes: u64,
        texel: u64,
        rules: &[CompressedBlit],
    ) -> Result<Rewrite, Error> {
        let (align, bands) = bands(self, bytes, texel);
        let device = self.core().device().clone();
        let scratch = device.own_buffer(bytes, ResourceOptions::private())?;
        let (raw_scratch, base) = scratch.at(0);
        if !base.is_multiple_of(align) {
            return Err(internal("a compressed blit scratch off its alignment"));
        }
        let mut passes = Vec::with_capacity(rules.len());
        for &rule in rules {
            let kernel = device.owned_kernel(OwnedModule::CompressedBlit(rule))?;
            let ranges: Vec<Vec<(vk::Buffer, u64, u64)>> = bands
                .iter()
                .map(|band| vec![(raw_scratch, base + band.start, band.end - band.start)])
                .collect();
            let sets = OwnedSets::write(&device, &kernel, &ranges)?;
            let groups = bands
                .iter()
                .map(|band| {
                    let words = (band.end - band.start) / 4;
                    let groups = words.div_ceil(u64::from(CompressedBlitModule::LOCAL_SIZE));
                    u32::try_from(groups).map_err(|_| internal("a compressed blit band past u32"))
                })
                .collect::<Result<Vec<u32>, Error>>()?;
            passes.push(Pass {
                pipeline: kernel.pipeline,
                layout: kernel.layout,
                sets,
                groups,
            });
        }
        Ok(Rewrite {
            scratch: (raw_scratch, base),
            buffer: scratch,
            passes,
        })
    }
}

/// The texel bytes of `rules`, one size for all (a copy's two sides share their unit); `None`
/// for no rule.
fn texel_bytes(rules: &[CompressedBlit]) -> Result<Option<u64>, Error> {
    let Some(first) = rules.first() else {
        return Ok(None);
    };
    let texel = first.texel_bytes();
    if rules.iter().any(|rule| rule.texel_bytes() != texel) {
        return Err(internal("compressed blit rules of two texel sizes"));
    }
    Ok(Some(u64::from(texel)))
}

/// One rule's dispatches over the scratch's bands.
struct Pass {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    sets: OwnedSets,
    groups: Vec<u32>,
}

/// The scratch a rewrite holds the texels in, and its passes.
struct Rewrite {
    scratch: (vk::Buffer, u64),
    buffer: Buffer,
    passes: Vec<Pass>,
}

impl Rewrite {
    /// Record, after the copy into the scratch, each pass's dispatches between barriers.
    ///
    /// # Safety
    ///
    /// `raw` is recording outside a render pass, and the scratch and the sets live until it
    /// completes.
    unsafe fn record(&self, device: &Vk, raw: vk::CommandBuffer) {
        // SAFETY: the caller's contract.
        unsafe {
            for pass in &self.passes {
                Barrier::everything().record(device, raw);
                device.cmd_bind_pipeline(raw, vk::PipelineBindPoint::COMPUTE, pass.pipeline);
                for (set, groups) in pass.sets.sets.iter().zip(&pass.groups) {
                    device.cmd_bind_descriptor_sets(
                        raw,
                        vk::PipelineBindPoint::COMPUTE,
                        pass.layout,
                        0,
                        std::slice::from_ref(set),
                        &[],
                    );
                    device.cmd_dispatch(raw, *groups, 1, 1);
                }
            }
            Barrier::everything().record(device, raw);
        }
    }

    /// Hand the scratch and the sets to the encoder's command buffer.
    fn retain(self, encoder: &mut BlitCommandEncoder) {
        encoder.core().retain(self.buffer);
        for pass in self.passes {
            encoder.core().retain(pass.sets);
        }
    }
}
