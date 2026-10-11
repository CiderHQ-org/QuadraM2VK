//! Decoded twins of BC1 to BC5 textures (the fitted decode).
//!
//! **Why.** Metal decodes a BC1 to BC5 block by rules of its own (`bind::BlockDecode`: 6-bit
//! weights, BC7's; fitted bit for bit on an Apple M4), which Vulkan leaves to the implementation.
//! A driver not measured to decode a format's blocks so (BC1 to BC3
//! `DriverFact::ColorBlockDecodeIsMetals`, BC4 and BC5 `DriverFact::ChannelBlockDecodeIsMetals`)
//! reads its texels up to a few 255ths off Metal's (`compressed-texel-decode`'s measured bounds).
//!
//! **The twin.** On such a device a texture of the format whose image is its own for good (in
//! memory of its own, or bound in a heap's own memory; [`Device::with_decoded_twin`]) keeps its
//! BC image, which every copy, blit and CPU access reads and writes as before, and beside it a
//! decoded twin: an image of the same shape, levels and layers, in memory of its own, holding
//! each texel as Metal decodes it (`BlockDecode::texel_format`: `R8G8B8A8_UNORM` codes for BC1 to
//! BC3, viewed as `R8G8B8A8_SRGB` where an sRGB form's own view reads; `R32_SFLOAT` and
//! `R32G32_SFLOAT` values for BC4 and BC5, which no 16-bit texel holds). Every sampled view a
//! binding takes of the texture, or of a view of it of the same decode (its sRGB twin), presents
//! the twin ([`Texture::decoded_view`]), so reads, samples and gathers read Metal's texels, which
//! the device filters as it filters the twin's format. The twin's format must take sampling, its
//! linear filter and copies on the device, else the texture has none.
//!
//! **Kept in step.** After every write into the BC image (`replace_region`, staged or submitted
//! now; a blit from a buffer, `fillTexture` included; a copy from a texture), the written region
//! is decoded into the twin by `bind::block_decode_module` ([`DecodePass`]): its blocks copied
//! into a scratch buffer, decoded, and the texels copied into the twin, in bands of whole block
//! rows of at most [`BAND_BYTES`] of blocks and texels, which bound the scratch: the pass's own,
//! or for a pass the device's next submission takes, the device's ([`PassScratch`]). A new
//! texture's twin starts as its zero blocks decode (the zeros: BC1 opaque black, every other format
//! zero); a heap's texture, which holds what its bytes held, has its whole twin decoded before any
//! command reads it (a staged pass, `TexelCopy`).
//!
//! **Host writes decode on the CPU** (off the caller's thread). A `replace_region` whose blocks go
//! on the host or the staging queue queues the region's decode on the CPU
//! (`BlockDecode::decode_into`, the kernel's rules) for the device's decode thread, which copies
//! its texels into the twin the same way ([`Image::decode_written`], `twin_decode.rs`, announced
//! `cpu-block-decode`). A batch that may read the texture waits at its submission for that; one
//! submitted before the call and not yet complete makes the call wait for its decode, so it reads
//! Metal's texels of the new blocks. For that the twin is made as its BC image's CPU copies go:
//! with host-transfer usage, or shared with the staging queue, and moved to `GENERAL` there alone
//! as it is made, its texels started as the zero blocks' decode the same way, a constant fill
//! queued with the texture's making ([`Texture::start_decoded_twin`]). The runtime stages no GPU
//! decode of such a twin of its own: one in flight could read the old blocks before a later write
//! and copy their texels into the twin after that write's CPU decode. A blit's decode
//! (`decode_twin`) is the caller's command's, and covers only what it wrote, as Metal's write of
//! those texels would.
//!
//! **Without a twin**, announced `compressed-texel-decode` as before: a texture over the caller's
//! host pages or laid out apart from a heap's bytes (tiled, shadowed), a sparse texture, a
//! texture over an IOSurface or the caller's image, one whose twin's format the device lacks;
//! and a view of another decode (`BC4_RUnorm` as `BC4_RSnorm`), which reads the BC image.
//!
//! **Cost**, announced once per device (`decoded-block-twin`, performance): the twin's memory,
//! 4 to 8 times the blocks' (a BC1 texel's half byte against its 4-byte twin), and a decode pass
//! after each write. A `VkImage` handed out (`Texture::vk_image`) is the BC image: what the
//! caller writes into it outside the runtime's calls leaves the twin as it was.

use super::super::blit::{BlitCommandEncoder, TRANSFER};
use super::super::copy_plan::ImageRegion;
use super::super::degradation::Degradation;
use super::super::device::{lock, DebugMessage, MessageKind, MessageSeverity};
use super::super::error::{internal, Error};
use super::super::exit_guard::{self, Call};
use super::super::image_plan::ImagePlan;
use super::super::owned_kernel::{OwnedModule, OwnedSets};
use super::super::resource::Identity;
use super::super::resource::ResourceKey;
use super::super::texture_descriptor::{TextureDescriptor, TextureLayout};
use super::super::tracker::{Access, Barrier, Use, UseRange};
use super::super::values::{HazardTrackingMode, ResourceOptions};
use super::super::{Buffer, Device};
use super::first_move::{FirstMove, Pattern, Patterned, TexelCopy};
use super::twin_decode::{self, Job};
use super::{
    binding_view, block_decode_fact, buffer_image_copy, texture_aspects, view_error, view_shadow,
    Image, Owner, Start, Storage, Texture, ViewRecord,
};
use ash::vk;
use metal2vulkan::bind::{BlockDecode, BlockDecodeModule, ImageViewDesc};
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags,
};
use std::sync::{Arc, Weak};

/// The most bytes of texels one band of a decode pass holds: its scratch's texels (and fewer
/// blocks), within `maxStorageBufferRange`.
pub(crate) const BAND_BYTES: u64 = 32 << 20;

/// A BC1 to BC5 image's decoded twin (the module docs).
pub(super) struct Decoded {
    /// The twin: `decode`'s texels, of the BC image's shape, levels and layers.
    pub(super) image: Image,
    /// How the BC image's blocks decode.
    pub(super) decode: BlockDecode,
    /// It was made moved alone (`Start::Moved`): its texels start as the CPU's decode of the zero
    /// blocks, written as the texture is made ([`Texture::start_decoded_twin`]).
    started_on_the_cpu: bool,
}

impl Decoded {
    /// The twin's format for a binding's view of the BC image in `format`: the twin's own, or
    /// for an sRGB form's own view its `R8G8B8A8_SRGB` view; `None` for a format of another
    /// decode, whose view reads the BC image.
    fn view_format(&self, format: Format) -> Option<Format> {
        use BlockDecode as D;
        use Format as F;
        match (self.decode, format) {
            (D::Bc1, F::BC1_RGBA_SRGB_BLOCK)
            | (D::Bc2, F::BC2_SRGB_BLOCK)
            | (D::Bc3, F::BC3_SRGB_BLOCK) => Some(F::R8G8B8A8_SRGB),
            (D::Bc1, F::BC1_RGBA_UNORM_BLOCK)
            | (D::Bc2, F::BC2_UNORM_BLOCK)
            | (D::Bc3, F::BC3_UNORM_BLOCK)
            | (D::Bc4Unorm, F::BC4_UNORM_BLOCK)
            | (D::Bc4Snorm, F::BC4_SNORM_BLOCK)
            | (D::Bc5Unorm, F::BC5_UNORM_BLOCK)
            | (D::Bc5Snorm, F::BC5_SNORM_BLOCK) => Some(self.decode.texel_format()),
            _ => None,
        }
    }

    /// The bytes of memory placed for the twin.
    fn placed(&self) -> Option<u64> {
        placed(&self.image)
    }

    /// The twin's texels' bytes and its blocks', every level and layer.
    fn bytes(&self) -> (u64, u64) {
        let plan = self.image.plan();
        let decode = self.decode;
        (0..plan.levels)
            .map(|level| {
                let [w, h, d] = plan.extent.map(|axis| u64::from((axis >> level).max(1)));
                let layers = u64::from(plan.layers);
                let texels = w * h * d * layers * u64::from(decode.texel_bytes());
                let blocks = w.div_ceil(4) * h.div_ceil(4) * d * layers;
                (texels, blocks * u64::from(decode.block_bytes()))
            })
            .fold((0, 0), |(t, b), (texels, blocks)| (t + texels, b + blocks))
    }
}

/// The bytes of memory the runtime placed `image` in, where it is memory of its own.
fn placed(image: &Image) -> Option<u64> {
    match &image.owner {
        Owner::Runtime { placement, .. } => Some(placement.size()),
        _ => None,
    }
}

/// The plan of the decoded twin of a BC image of `plan` whose blocks decode by `decode`: the
/// image's type, shape, levels, layers and cube compatibility, of the decoded texels' format
/// (mutable for an sRGB view of BC1 to BC3's codes; `BlockDecode::texel_pixel_format`'s), sampled
/// and copied into. `routed`: the BC image's CPU copies go on the host or the staging queue, which
/// the twin starts on and takes the CPU's decodes through ([`Image::decode_written`]), copied both
/// ways as the staging queue's images are (`Device::stages`).
pub(crate) fn twin_plan(plan: &ImagePlan, decode: BlockDecode, routed: bool) -> ImagePlan {
    let format = decode.texel_format();
    let kept = ImageCreateFlags::CUBE_COMPATIBLE | ImageCreateFlags::CREATE_2D_ARRAY_COMPATIBLE;
    let (flags, view_formats) = match format {
        Format::R8G8B8A8_UNORM => (
            (plan.flags & kept) | ImageCreateFlags::MUTABLE_FORMAT,
            vec![Format::R8G8B8A8_UNORM, Format::R8G8B8A8_SRGB],
        ),
        _ => (plan.flags & kept, Vec::new()),
    };
    let copies = match routed {
        true => ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST,
        false => ImageUsageFlags::TRANSFER_DST,
    };
    ImagePlan {
        // Its copies' units are its texels' (`copy_units`), not the BC image's blocks.
        pixel_format: decode.texel_pixel_format(),
        flags,
        image_type: plan.image_type,
        format,
        extent: plan.extent,
        levels: plan.levels,
        layers: plan.layers,
        samples: plan.samples,
        usage: ImageUsageFlags::SAMPLED | copies,
        view_formats,
        mapping: FormatMapping::direct(format, ImageAspectFlags::COLOR),
        unavailable: plan.unavailable,
    }
}

impl Device {
    /// How a texture of `layout` decodes into a twin on this device: a BC1 to BC5 format of one
    /// sample whose decode the driver is not measured to make as Metal does; `None` elsewhere.
    fn twin_decode(&self, layout: &TextureLayout) -> Option<BlockDecode> {
        let decode = BlockDecode::of(layout.pixel_format)?;
        let fact = block_decode_fact(layout.pixel_format)?;
        let measured = self.properties().driver_facts.contains(&fact);
        (!measured && layout.samples == 1).then_some(decode)
    }

    /// `image`, a new texture's image of `layout` made with `descriptor`, its own for good, with
    /// its decoded twin where the texture takes one (the module docs), announced
    /// (`decoded-block-twin`); without one, `compressed-texel-decode` where the device's decode
    /// is not measured. `fresh`: the image holds the zeros of a new texture, which the twin starts
    /// as (where the image takes host or staging copies, by the decode the caller stages once the
    /// image is shared, [`Texture::start_decoded_twin`]); else the caller decodes the whole twin
    /// once the image is shared ([`Texture::decode_whole`]).
    pub(super) fn with_decoded_twin(
        &self,
        mut image: Image,
        layout: &TextureLayout,
        descriptor: &TextureDescriptor,
        fresh: bool,
    ) -> Result<Image, Error> {
        let Some(decode) = self.twin_decode(layout) else {
            self.degrade_block_decode(layout);
            return Ok(image);
        };
        // A twin starts, and takes the CPU's decodes, as the BC image's CPU copies go: on the
        // host or the staging queue where they do (then made as a `Shared` or `Managed`
        // texture's image is), else through submissions.
        let routed = image.host_copies || image.staging_copies;
        let plan = twin_plan(image.plan(), decode, routed);
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let needs = FormatFeatureFlags2::SAMPLED_IMAGE
            | FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR
            | FormatFeatureFlags2::TRANSFER_DST;
        if !facts.optimal(plan.format).contains(needs) {
            self.degrade_block_decode(layout);
            return Ok(image);
        }
        let mut own = *descriptor;
        if !routed {
            own.resource_options = ResourceOptions::private();
        }
        // Its uses are the image's (the twin shares its resource identity): its own goes unused.
        // A routed twin is moved to `GENERAL` alone as it is made, so a CPU decode may write it
        // from then on; its texels start as the CPU's decode of the zero blocks, written on the
        // host or the staging queue as the texture is made (`start_decoded_twin`): 6 to 10 ms for
        // 2048 x 2048 (EVs XSH-mk1-anv, XSH-mk1-3090). No GPU decode the runtime stages of its
        // own is in flight to race a later write's CPU decode.
        let identity = Identity::new(HazardTrackingMode::Default, false);
        let start = match routed {
            true => Start::Moved,
            false => Start::Zero,
        };
        let twin =
            self.new_image_starting(plan, layout, &own, identity, start, Device::runtime_owner)?;
        let moved = twin.host_copies || twin.staging_copies;
        // A twin that takes neither copy while its blocks take one could take no CPU decode
        // behind work in flight ([`Image::decode_written`]): the blocks' CPU copies then go
        // through submissions too, in order with the twin's decodes. No measured device makes
        // such a twin (its formats take the copies its blocks take).
        if routed && !moved {
            image.host_copies = false;
            image.staging_copies = false;
        }
        // A new BC1 texture's zero blocks are three-color blocks of black, alpha 1; every other
        // format's decode to zero, which the twin's first move writes.
        if fresh && decode == BlockDecode::Bc1 && !routed {
            let plan = twin.plan();
            let black = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            };
            let patterned = Patterned::new(Pattern::Color(black), plan.levels, plan.levels);
            let first = FirstMove::new(
                twin.raw.get(),
                plan.format,
                vk::ImageLayout::UNDEFINED,
                false,
            );
            *lock(&twin.first_move) = Some(first.patterned(patterned));
        }
        let decoded = Decoded {
            image: twin,
            decode,
            started_on_the_cpu: routed && fresh,
        };
        self.degrade_decoded_twin(layout, &decoded, placed(&image));
        image.decoded = Some(Box::new(decoded));
        Ok(image)
    }

    /// Announce `decoded-block-twin` for a texture of `layout` made with `decoded`, its cost: the
    /// twin's texels' bytes and the memory placed for it, beside the blocks' bytes and the memory
    /// placed for its image (`image`, where the runtime placed it in memory of its own).
    fn degrade_decoded_twin(&self, layout: &TextureLayout, decoded: &Decoded, image: Option<u64>) {
        self.degrade(Degradation::DecodedBlockTwin, || {
            let [width, height, depth] = layout.extent;
            let object = format!(
                "a {:?} {:?} texture of {width}x{height}x{depth}",
                layout.pixel_format, layout.texture_type
            );
            let (texels, blocks) = decoded.bytes();
            let fact = block_decode_fact(layout.pixel_format)
                .map_or_else(String::new, |fact| format!(" (DriverFact::{fact:?})"));
            let placed =
                |bytes: Option<u64>| bytes.map_or_else(String::new, |n| format!(", {n} placed"));
            let why = format!(
                "the device's driver is not measured to decode the format's blocks as Metal \
                 does{fact}: its samples read a decoded twin of {texels} bytes of {:?} texels{} \
                 beside its {blocks} bytes of blocks{}, decoded after each write into it",
                decoded.decode.texel_format(),
                placed(decoded.placed()),
                placed(image),
            );
            (object, why)
        });
    }
}

impl Texture {
    /// The `VkImageView` of `desc` (a sampled view; `call`'s) of the texture's decoded twin, where
    /// its image (`image`) has one and the texture's format decodes as the twin's (the module
    /// docs): made at its first bind, then kept with the twin. `None` elsewhere (an image with
    /// one presents no stand-in: its detach is refused, `stand_in.rs`).
    pub(super) fn decoded_view(
        &self,
        call: &'static str,
        image: &Image,
        desc: &ImageViewDesc,
    ) -> Result<Option<vk::ImageView>, Error> {
        let Some(decoded) = image.decoded.as_deref() else {
            return Ok(None);
        };
        let same = BlockDecode::of(self.0.record.pixel_format) == Some(decoded.decode);
        let Some(format) = decoded.view_format(desc.format).filter(|_| same) else {
            return Ok(None);
        };
        let mut twin = *desc;
        twin.format = format;
        let view = binding_view(decoded.image.plan(), &self.0.record, &twin, false)
            .map_err(|refusal| view_error(call, refusal))?;
        decoded.image.view(&view).map(Some)
    }

    /// Announce `compressed-texel-decode` for a view presenting `record` over `storage`, of a
    /// texture whose image has a decoded twin the view does not present: a view of another
    /// decode (`BC4_RUnorm` as `BC4_RSnorm`), or a shadowed view, which reads the blocks through
    /// the device's decode.
    pub(super) fn degrade_view_decode(&self, storage: &Storage, record: &ViewRecord) {
        let Storage::Image(image) = storage else {
            return;
        };
        if view_shadow::truth(image).decoded.is_none() {
            return;
        }
        let decode = BlockDecode::of(record.pixel_format);
        if image
            .decoded
            .as_ref()
            .is_some_and(|d| Some(d.decode) == decode)
        {
            return;
        }
        let device = self.device();
        device.degrade_device_decode(record.pixel_format, record.texture_type, record.extent());
    }

    /// The image the texture's texels are held in where it has a decoded twin: its image, or a
    /// shadowed view's texture's (`view_shadow.rs`), whose twin a write through the view decodes
    /// into once its copy back is made. `None` for a texture without one.
    fn decoding_image(&self) -> Option<Arc<Image>> {
        let image = view_shadow::truth(self.image().ok()?);
        image.decoded.is_some().then(|| Arc::clone(image))
    }

    /// Decode the texture's whole image into its decoded twin before any command reads it: a
    /// heap's texture, which holds what its bytes held. Staged for the device's next submission
    /// as a host call's copy is (`TexelCopy`), or past the bound made now. Nothing for a texture
    /// without one.
    pub(super) fn decode_whole(&self) -> Result<(), Error> {
        let Some(image) = self.decoding_image() else {
            return Ok(());
        };
        image.stage_decode(&whole_regions(image.plan()))
    }

    /// Start the texture's decoded twin where it was made moved alone (`Start::Moved`,
    /// [`Device::with_decoded_twin`]): the decode of a new texture's zero blocks, every level's,
    /// queued for the device's decode thread (`twin_decode.rs`), which fills the twin with it on
    /// the host or the staging queue as its copies go. Nothing for any other texture.
    pub(super) fn start_decoded_twin(&self) -> Result<(), Error> {
        let Some(image) = self.decoding_image() else {
            return Ok(());
        };
        if !image
            .decoded
            .as_deref()
            .is_some_and(|d| d.started_on_the_cpu)
        {
            return Ok(());
        }
        for region in whole_regions(image.plan()) {
            image.queue_twin_decode(&region, None)?;
        }
        Ok(())
    }
}

/// The texel every texel of a zero block decodes to, as `decode` writes it: `None` where they
/// differ (no format so far), when the zero blocks are decoded instead.
fn zero_texel(decode: BlockDecode) -> Option<Vec<u8>> {
    let texels = decode.decode(&vec![0; decode.block_bytes() as usize], [4, 4, 1]);
    let size = decode.texel_bytes() as usize;
    let first = texels.get(..size)?;
    texels
        .chunks_exact(size)
        .all(|texel| texel == first)
        .then(|| first.to_vec())
}

/// The copies of `band`'s texels, laid out tight in rows of `row` bytes, in parts of whole block
/// rows of at least `twin_decode::ROWS_PER_PART` texel rows, at most `twin_decode::THREADS`, each
/// reading its rows at its offset: disjoint texels, which host copies may write at once.
fn host_parts(band: &ImageRegion, row: usize) -> Vec<vk::BufferImageCopy> {
    let height = band.size[1];
    let parts = (height / twin_decode::ROWS_PER_PART).clamp(1, twin_decode::THREADS as u32);
    let per = height.div_ceil(parts).next_multiple_of(4);
    (0..height)
        .step_by(per as usize)
        .map(|y| {
            let part = ImageRegion {
                origin: [band.origin[0], band.origin[1] + y, band.origin[2]],
                size: [band.size[0], per.min(height - y), band.size[2]],
                ..*band
            };
            vk::BufferImageCopy {
                buffer_offset: (y as usize * row) as u64,
                ..buffer_image_copy(&part)
            }
        })
        .collect()
}

/// The part of `region` that is texel rows `y` to `y + height` of its image `image` (its layer,
/// or its depth slice in a volume).
fn band_region(region: &ImageRegion, image: u32, y: u32, height: u32) -> ImageRegion {
    let [x, top, z] = region.origin;
    let (layer, z) = match region.layers > 1 {
        true => (region.layer + image, z),
        false => (region.layer, z + image),
    };
    ImageRegion {
        layer,
        layers: 1,
        origin: [x, top + y, z],
        size: [region.size[0], height, 1],
        ..*region
    }
}

/// Every texel of an image of `plan`: one region a level, of all its layers.
fn whole_regions(plan: &ImagePlan) -> Vec<ImageRegion> {
    let volume = plan.image_type == ImageType::TYPE_3D;
    (0..plan.levels)
        .map(|level| {
            let [w, h, d] = plan.extent.map(|axis| (axis >> level).max(1));
            ImageRegion {
                aspect: ImageAspectFlags::COLOR,
                level,
                layer: 0,
                layers: plan.layers,
                origin: [0, 0, 0],
                size: [w, h, if volume { d } else { 1 }],
            }
        })
        .collect()
}

impl Image {
    /// Stage the decode of `regions` of the image into its decoded twin for the device's next
    /// submission (`TexelCopy::decode_only`), what the image holds there already: a heap's
    /// bytes. Past the bound, made now. Nothing for an image without a twin. (no longer a host
    /// call's copy's, which the CPU decodes, [`decode_written`](Self::decode_written).)
    pub(super) fn stage_decode(self: &Arc<Self>, regions: &[ImageRegion]) -> Result<(), Error> {
        let firsts = &self.device.shared().textures.first_moves;
        let Some(pass) = self.decode_pass(regions, PassScratch::Device)? else {
            return Ok(());
        };
        let staged = TexelCopy::decode_only(pass);
        let keeper: Weak<Image> = Arc::downgrade(self);
        if firsts.register_texels(keeper, staged) {
            self.submit_copy(|_, _| {})?;
        }
        Ok(())
    }

    /// The decode into the image's decoded twin of `region`, which a host call just wrote into
    /// the image on the host or the staging queue as `packed` (its blocks, laid out as
    /// [`buffer_image_copy`] reads them): queued for the device's decode thread, which decodes it
    /// on the CPU and copies the texels into the twin as the twin's copies go
    /// (`twin_decode.rs`; announced `cpu-block-decode`). The first batch that may read the
    /// texture waits at its submission for it (every batch, where the texture is untracked);
    /// the call waits for nothing: a batch submitted before it reads the twin as the thread's
    /// copies stand when it runs, as Metal's unsynchronized CPU write leaves the ordering to the
    /// program. Always on the CPU: a decode staged for the GPU could run while a later write's CPU
    /// decode copies into the twin, read the old blocks before it and write their texels after it
    /// (lavapipe under a lane's load, master `7d426aff`). Nothing for an image without a twin.
    pub(super) fn decode_written(
        self: &Arc<Self>,
        region: &ImageRegion,
        packed: Vec<u8>,
    ) -> Result<(), Error> {
        let Some(decoded) = self.decoded.as_deref() else {
            return Ok(());
        };
        let host = decoded.image.takes_host_copies();
        self.queue_twin_decode(region, Some(packed))?;
        let device = &self.device;
        let object = || {
            let plan = self.plan();
            let [width, height, depth] = plan.extent;
            format!(
                "a {:?} texture of {width}x{height}x{depth}",
                plan.pixel_format
            )
        };
        device.degrade(Degradation::CpuBlockDecode, || {
            let route = match host {
                true => "on the host",
                false => "on the staging queue",
            };
            let why = format!(
                "a host call's write into it decodes the written blocks on the CPU, on the \
                 device's decode thread, and copies the texels into its decoded twin {route}; \
                 the first command buffer that may read it waits at its submission for that"
            );
            (object(), why)
        });
        Ok(())
    }

    /// Queue the decode of `region` into the image's twin for the device's decode thread
    /// (`blocks`, the written blocks; `None`, a new texture's zero blocks): the scheduler holds
    /// each batch that may read the texture until it finished (every batch, where the texture is
    /// untracked: no footprint names it). The call waits for nothing but room in the queue.
    fn queue_twin_decode(
        self: &Arc<Self>,
        region: &ImageRegion,
        blocks: Option<Vec<u8>>,
    ) -> Result<(), Error> {
        let shared = self.device.shared();
        let decodes = &shared.textures.twin_decodes;
        decodes.start()?;
        let (key, tracked) = (self.identity.key, self.identity.tracked);
        shared.scheduler.decode_queued(key, tracked);
        decodes.queue(Job {
            image: Arc::clone(self),
            region: *region,
            blocks,
        });
        Ok(())
    }

    /// On the device's decode thread: decode `blocks` (`region`'s; `None`, the zero blocks) into
    /// the twin ([`write_decoded`](Self::write_decoded)), count a write's decode, and let the
    /// batches that wait for it go. A failure is said on the device's message sink as an error
    /// (`metal2vulkan-twin-decode-failed`): the call that queued it has returned.
    pub(super) fn finish_twin_decode(&self, region: &ImageRegion, blocks: Option<&[u8]>) {
        let shared = self.device.shared();
        let result = match self.decoded.as_deref() {
            Some(decoded) => self.write_decoded(decoded, region, blocks),
            None => Ok(false),
        };
        match result {
            Ok(_) if blocks.is_some() => shared.counts.cpu_block_decode(),
            Ok(_) => {}
            Err(error) => {
                let plan = self.plan();
                let message = DebugMessage {
                    severity: MessageSeverity::Error,
                    kind: MessageKind {
                        general: true,
                        ..MessageKind::default()
                    },
                    id_name: "metal2vulkan-twin-decode-failed".to_owned(),
                    id: 0,
                    text: format!(
                        "metal2vulkan: the decoded twin of a {:?} texture of {:?} kept its \
                         earlier texels: its CPU decode failed ({error})",
                        plan.pixel_format, plan.extent
                    ),
                    degradation: None,
                };
                match shared.instance.sink() {
                    Some(sink) => sink.deliver(&message),
                    None => eprintln!("{}", message.text),
                }
            }
        }
        let identity = &self.identity;
        shared
            .scheduler
            .decode_finished(identity.key, identity.tracked);
    }

    /// On the device's decode thread: a start a later queued write overwrites whole, skipped (its
    /// texels are never read); the batches that wait for it go once that write finished too.
    pub(super) fn skip_twin_decode(&self) {
        let identity = &self.identity;
        let scheduler = &self.device.shared().scheduler;
        scheduler.decode_finished(identity.key, identity.tracked);
    }

    /// Decode `blocks`, `region`'s (`None`: the zero blocks, a constant fill), on the CPU and copy
    /// the texels into the twin of `decoded` (the image's) as the twin's copies go, in bands of at
    /// most `twin_decode::BAND_BYTES` of texels, each decoded on up to `twin_decode::THREADS`
    /// threads: whether on the host (else on the staging queue). An internal error where the twin
    /// takes neither copy, or took none.
    fn write_decoded(
        &self,
        decoded: &Decoded,
        region: &ImageRegion,
        blocks: Option<&[u8]>,
    ) -> Result<bool, Error> {
        let twin = &decoded.image;
        let host = twin.takes_host_copies();
        // The image's CPU copies go on the host or the staging queue only where its twin's do
        // ([`Device::with_decoded_twin`]), and a twin presents no stand-in.
        if !host && !twin.takes_staging_copies() {
            return Err(internal(
                "a decoded twin that takes no CPU copy its blocks took",
            ));
        }
        let decode = decoded.decode;
        let block = decode.block_bytes() as usize;
        let texel = decode.texel_bytes() as usize;
        let [width, height, depth] = region.size;
        let (across, down) = (width.div_ceil(4) as usize, height.div_ceil(4) as usize);
        // Block rows a band, each 4 texel rows of `width`.
        let rows = (twin_decode::BAND_BYTES / (4 * width as usize * texel)).max(1);
        let fill = blocks.is_none().then(|| zero_texel(decode));
        for image in 0..depth * region.layers {
            for first in (0..down).step_by(rows) {
                let count = rows.min(down - first);
                let y = first as u32 * 4;
                let h = (count as u32 * 4).min(height - y);
                let mut texels = vec![0; width as usize * h as usize * texel];
                match (blocks, &fill) {
                    (Some(blocks), _) => {
                        let at = (image as usize * down + first) * across * block;
                        let band = &blocks[at..at + count * across * block];
                        twin_decode::decode_band(decode, band, [width, h], &mut texels);
                    }
                    (None, Some(Some(texel))) => {
                        for out in texels.chunks_exact_mut(texel.len()) {
                            out.copy_from_slice(texel);
                        }
                    }
                    (None, _) => {
                        let zeros = vec![0; count * across * block];
                        twin_decode::decode_band(decode, &zeros, [width, h], &mut texels);
                    }
                }
                let band = band_region(region, image, y, h);
                let written = match host {
                    true => {
                        let row = width as usize * texel;
                        let copies = host_parts(&band, row);
                        twin.host_write_disjoint(&copies, &texels, twin_decode::THREADS)?
                    }
                    false => twin.staging_write(buffer_image_copy(&band), &texels)?,
                };
                if !written {
                    return Err(internal(
                        "a decoded twin that took no CPU copy its blocks took",
                    ));
                }
            }
        }
        Ok(host)
    }

    /// The pass that decodes `regions` (of the image's levels and layers) of the image into its
    /// decoded twin (the module docs), its scratch (as `scratch` says) and sets made; `None` for an
    /// image without one, or regions of no texel.
    pub(super) fn decode_pass(
        &self,
        regions: &[ImageRegion],
        scratch: PassScratch,
    ) -> Result<Option<DecodePass>, Error> {
        let Some(decoded) = self.decoded.as_deref() else {
            return Ok(None);
        };
        let decode = decoded.decode;
        let device = &self.device;
        let limits = &device.properties().description.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(16);
        let most = u64::from(limits.max_storage_buffer_range).min(BAND_BYTES);
        let texel = u64::from(decode.texel_bytes());
        let block = u64::from(decode.block_bytes());
        let steps: Vec<ImageRegion> = regions
            .iter()
            .flat_map(|region| bands(region, texel, block, most))
            .collect();
        if steps.is_empty() {
            return Ok(None);
        }
        let images = |r: &ImageRegion| u64::from(r.size[2]) * u64::from(r.layers);
        let blocks_of = |r: &ImageRegion| {
            let [w, h, _] = r.size.map(u64::from);
            w.div_ceil(4) * h.div_ceil(4) * images(r) * u64::from(decode.block_bytes())
        };
        let texels_of = |r: &ImageRegion| {
            let [w, h, _] = r.size.map(u64::from);
            w * h * images(r) * texel
        };
        // Each step's blocks at the scratch's start and its texels at the alignment after them:
        // at most `most` bytes together (`bands`), so within `most` and the alignment.
        let layouts: Vec<(u64, u64, u64)> = steps
            .iter()
            .map(|step| {
                let blocks = blocks_of(step);
                (blocks, blocks.div_ceil(align) * align, texels_of(step))
            })
            .collect();
        let needed = (layouts.iter())
            .map(|&(_, texels_at, texels)| texels_at + texels)
            .max()
            .unwrap_or(0);
        let (scratch, raw_scratch, base, bytes) = match scratch {
            PassScratch::Own => {
                let own = device.own_buffer(needed, ResourceOptions::private())?;
                let (raw, base) = own.at(0);
                (Some(own), raw, base, needed)
            }
            PassScratch::Device => {
                if needed > most + align {
                    return Err(internal(
                        "a block decode's step past the device's decode scratch",
                    ));
                }
                let firsts = &device.shared().textures.first_moves;
                (None, firsts.decode_scratch(device, most + align)?, 0, 0)
            }
        };
        // Each step's shape, written by the host at its slot of the shapes' buffer: a shape's
        // bytes rounded up to the alignment.
        let shape_bytes = u64::from(BlockDecodeModule::SHAPE_WORDS) * 4;
        let slot = shape_bytes.div_ceil(align) * align;
        let shapes = device.write_combined_buffer(steps.len() as u64 * slot)?;
        let (raw_shapes, shapes_base) = shapes.at(0);
        if !base.is_multiple_of(align) || !shapes_base.is_multiple_of(align) {
            return Err(internal("a block decode's scratch off its alignment"));
        }
        let contents = (shapes.contents())
            .ok_or_else(|| internal("a Shared block decode shapes' buffer without contents"))?;
        let mut groups = Vec::with_capacity(steps.len());
        for (at, step) in steps.iter().enumerate() {
            let [w, h, _] = step.size;
            let d = step.size[2] * step.layers;
            let shape = BlockDecodeModule::shape([w, h, d]);
            // SAFETY: the shapes' buffer holds `steps.len()` slots of `slot` bytes, each at least
            // a shape's, in host-coherent memory nothing else uses yet.
            unsafe {
                let to = contents.as_ptr().add(at * slot as usize);
                std::ptr::copy_nonoverlapping(shape.as_ptr(), to, shape.len());
            }
            let local = BlockDecodeModule::LOCAL_SIZE;
            groups.push([w.div_ceil(local), h.div_ceil(local), d]);
        }
        let kernel = device.owned_kernel(OwnedModule::BlockDecode(decode))?;
        let ranges: Vec<Vec<(vk::Buffer, u64, u64)>> = (layouts.iter())
            .zip(0u64..)
            .map(|(&(blocks, texels_at, texels), at)| {
                vec![
                    (raw_shapes, shapes_base + at * slot, shape_bytes),
                    (raw_scratch, base, blocks),
                    (raw_scratch, base + texels_at, texels),
                ]
            })
            .collect();
        let sets = OwnedSets::write(device, &kernel, &ranges)?;
        let steps = steps
            .iter()
            .zip(&layouts)
            .zip(&sets.sets)
            .zip(groups)
            .map(|(((region, &(_, texels_at, _)), &set), groups)| Step {
                blocks: vk::BufferImageCopy {
                    buffer_offset: base,
                    ..buffer_image_copy(region)
                },
                texels: vk::BufferImageCopy {
                    buffer_offset: base + texels_at,
                    ..buffer_image_copy(region)
                },
                set,
                groups,
            })
            .collect();
        device.shared().counts.block_decode_pass();
        Ok(Some(DecodePass {
            image: self.raw.get(),
            twin: decoded.image.raw.get(),
            scratch: raw_scratch,
            pipeline: kernel.pipeline,
            layout: kernel.layout,
            steps,
            bytes,
            held: Held {
                _scratch: scratch,
                _shapes: shapes,
                _sets: sets,
            },
        }))
    }

    /// The uses of a command that decodes `regions` of the image into its twin: a write of each
    /// (the twin shares the image's resource identity), as a copy reads and writes.
    fn decode_uses(&self, regions: &[ImageRegion]) -> Vec<Use<ResourceKey>> {
        let access = Access::new(
            TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
        );
        let volume = self.plan().image_type == ImageType::TYPE_3D;
        regions
            .iter()
            .map(|region| {
                let layers = match volume {
                    true => 0..1,
                    false => region.layer..region.layer + region.layers,
                };
                let range = UseRange::Image {
                    aspects: texture_aspects(self.plan()),
                    levels: region.level..region.level + 1,
                    layers,
                };
                Use::writes(self.identity.key, self.identity.tracked, range, access)
            })
            .collect()
    }
}

/// `region` (of texels of `texel` bytes, in blocks of `block` bytes of 4 x 4 texels) in steps
/// whose blocks and texels take at most `most` bytes together (before, its texels alone): whole,
/// else runs of its images (layers or depth slices), else each image in bands of whole block rows.
/// A block row past `most` (16384 texels of 16 bytes and their blocks are 1 MiB and 64 KiB) is a
/// step alone.
pub(crate) fn bands(region: &ImageRegion, texel: u64, block: u64, most: u64) -> Vec<ImageRegion> {
    let [w, h, d] = region.size;
    let count = u64::from(d) * u64::from(region.layers);
    let across = u64::from(w).div_ceil(4);
    // A row of blocks and its four rows of texels.
    let row = across * block + 4 * u64::from(w) * texel;
    let image = across * u64::from(h).div_ceil(4) * block + u64::from(w) * u64::from(h) * texel;
    if count == 0 || w == 0 || h == 0 {
        return Vec::new();
    }
    if image * count <= most {
        return vec![*region];
    }
    // The `i`-th image alone, of `n` from it.
    let images = |i: u32, n: u32| {
        let mut one = *region;
        match d > 1 {
            true => {
                one.origin[2] += i;
                one.size[2] = n;
            }
            false => {
                one.layer += i;
                one.layers = n;
            }
        }
        one
    };
    let total = d.max(region.layers);
    if image <= most {
        let per = u32::try_from((most / image).max(1)).unwrap_or(u32::MAX);
        return (0..total)
            .step_by(per as usize)
            .map(|i| images(i, per.min(total - i)))
            .collect();
    }
    let rows = u32::try_from((most / row).max(1) * 4).unwrap_or(u32::MAX);
    (0..total)
        .flat_map(|i| {
            let one = images(i, 1);
            (0..h).step_by(rows as usize).map(move |y| {
                let mut band = one;
                band.origin[1] += y;
                band.size[1] = rows.min(h - y);
                band
            })
        })
        .collect()
}

/// One band of a decode pass: its blocks copied out of the BC image, its set, its workgroups and
/// its texels copied into the twin.
struct Step {
    blocks: vk::BufferImageCopy,
    texels: vk::BufferImageCopy,
    set: vk::DescriptorSet,
    groups: [u32; 3],
}

/// What a decode pass holds until the command that records it completed: its own scratch
/// (`None` in the device's, which the device keeps), its shapes and its sets.
struct Held {
    _scratch: Option<Buffer>,
    _shapes: Buffer,
    _sets: OwnedSets,
}

/// The scratch buffer a decode pass's bands go through ([`Image::decode_pass`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassScratch {
    /// One of its own, held until the command that records it completed: a pass a command
    /// buffer records, or an internal submission's own.
    Own,
    /// The device's (`FirstMoves::decode_scratch`), made once: a pass a submission takes from the
    /// device's staged work, which runs after every such pass before it. It is no staged byte
    /// (the bound), so a texture's start (`Texture::start_decoded_twin`) allocates nothing.
    Device,
}

/// The decode of written regions of a BC image into its decoded twin (the module docs), made by
/// [`Image::decode_pass`] and recorded by [`DecodePass::record`] after the write.
pub(crate) struct DecodePass {
    image: vk::Image,
    twin: vk::Image,
    scratch: vk::Buffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    steps: Vec<Step>,
    /// The bytes of its own scratch (none in the device's).
    pub(super) bytes: u64,
    held: Held,
}

impl DecodePass {
    /// Record the pass: after every write before it, each band's blocks copied into the
    /// scratch, decoded, and its texels copied into the twin, each step past a barrier, then a
    /// barrier to everything after. Only entry points of the exit guard's table (a staged
    /// copy's runs inside a submission's section).
    ///
    /// # Safety
    ///
    /// `raw` is recording, outside a render pass, on a queue family with compute; the image, the
    /// twin and the pass live until it completes; both images are in the `GENERAL` layout.
    pub(crate) unsafe fn record(&self, device: &ash::Device, raw: vk::CommandBuffer) {
        let general = vk::ImageLayout::GENERAL;
        let compute = vk::PipelineBindPoint::COMPUTE;
        // SAFETY: the caller's contract; each copy lies in its image and in the scratch, and
        // each dispatch reads and writes within its set's ranges (`decode_pass`).
        unsafe {
            Barrier::everything().record(device, raw);
            device.cmd_bind_pipeline(raw, compute, self.pipeline);
            for step in &self.steps {
                let blocks = std::slice::from_ref(&step.blocks);
                device.cmd_copy_image_to_buffer(raw, self.image, general, self.scratch, blocks);
                Barrier::everything().record(device, raw);
                let set = std::slice::from_ref(&step.set);
                device.cmd_bind_descriptor_sets(raw, compute, self.layout, 0, set, &[]);
                let [x, y, z] = step.groups;
                device.cmd_dispatch(raw, x, y, z);
                Barrier::everything().record(device, raw);
                let texels = std::slice::from_ref(&step.texels);
                device.cmd_copy_buffer_to_image(raw, self.scratch, self.twin, general, texels);
                Barrier::everything().record(device, raw);
            }
        }
        for call in [
            Call::CmdPipelineBarrier,
            Call::CmdBindPipeline,
            Call::CmdCopyImageToBuffer,
            Call::CmdBindDescriptorSets,
            Call::CmdDispatch,
            Call::CmdCopyBufferToImage,
        ] {
            exit_guard::called(call);
        }
    }
}

impl BlitCommandEncoder {
    /// After the encoder's last command wrote `regions` (of the image's levels and layers) of
    /// `texture`'s texels: their decode into its decoded twin (the module docs), one command of
    /// the encoder. Nothing for a texture without one.
    pub(crate) fn decode_twin(
        &mut self,
        texture: &Texture,
        regions: &[ImageRegion],
    ) -> Result<(), Error> {
        let Some(image) = texture.decoding_image() else {
            return Ok(());
        };
        let Some(pass) = image.decode_pass(regions, PassScratch::Own)? else {
            return Ok(());
        };
        let uses = image.decode_uses(regions);
        let stages = TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        self.command_in(stages, &uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording, outside a render pass; the
            // image (its twin with it) and the pass are retained until it completes; both images
            // rest in the `GENERAL` layout.
            unsafe { pass.record(device, raw) }
        })?;
        self.core().retain(pass.held);
        self.core().retain(image);
        Ok(())
    }
}
