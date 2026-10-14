//! Shadows: a 2D texture or array over a buffer that no linear image of the device aliases
//! (`linear_plan::aliases`) is an optimal image whose texels the encoders copy from the buffer and
//! back, so that every access Metal orders sees what the other view wrote. A texture buffer whose
//! offset in its buffer's `VkBuffer` misses the device's `minTexelBufferOffsetAlignment`, where
//! Vulkan views no texels, is likewise a staging buffer its bindings view, the texels at its start.
//! The rules below are the same for both shadows, an image or a staging buffer.
//!
//! - **The buffer is the truth.** CPU access through the texture reads and writes the buffer's
//!   bytes (`buffer_texture.rs`), and between encoders the image holds nothing the buffer lacks.
//! - **Copy in** before the first command of an encoder that uses the texture, and again before a
//!   later one when a command of the encoder wrote the bytes since: through the buffer, a texture
//!   buffer, or another shadowed texture over them. A command buffer's encoders run after its
//!   commit, so the first copy of each encoder sees every CPU write made before then.
//! - **Copy back** what a command wrote through the texture before the next command that touches
//!   the bytes some other way, and at the end of the encoder (orchestrator, 2026-10-04: at the
//!   end of the encoder that writes through the texture, not of the command buffer). A copy moves
//!   only the texels' rows, never the bytes between them; a copy back of a one-level, one-slice
//!   texture's image, only the texels inside the rectangle its writers since the last bound their
//!   writes to, where each bounds them ([`ShadowUse::area`]: a pass's attachment it loads, inside
//!   its draws' scissors), as the image's other texels are what the last copy in left.
//! - **Ordering.** A copy is a command of the encoder with uses of both the buffer's bytes and the
//!   image, so the tracker orders it against the encoder's other commands and other encoders'
//!   like any use: the texture and its buffer are one resource to it. In an encoder whose
//!   commands Metal does not order (concurrent dispatches), each copy has a barrier before and
//!   after it.
//! - **Staging.** Vulkan copies texels at multiples of their size of a `VkBuffer`, which a
//!   no-copy buffer's bytes need not start at (`Buffer::base`). Such a 2D texture's copies go
//!   through a staging buffer of its own ([`ShadowStaging`]): its bytes into the staging buffer and
//!   on into the image, or the image into the staging buffer and its rows on into the buffer, as
//!   one command with a barrier between the steps. That staging buffer needs no uses of its own:
//!   every copy through it writes the image or the texture's bytes, so the tracker orders it after
//!   every other copy of the texture. A texture buffer's staging buffer is its shadow, with uses of
//!   its own, as an image's.
//! - **Render passes** (`render/shadows.rs`). A render pass instance is one command: its draws
//!   record apart, where no copy goes, so the copies go before the instance and the uses are its
//!   draws' and its attachments'.
//! - **Heap depth and stencil textures** (`texture/heap.rs`). On a driver that lays depth and
//!   stencil images apart from their memory, a heap's texture of such a format is an image of its
//!   own memory whose truth is the heap's bytes in the texture's range: each aspect's texels
//!   tightly packed, level by level, in planes (`heap_planes`). Its copies are one region per
//!   aspect and level, the region's aspect its own. The rules above are the same, and a blit
//!   encoder follows them too (`blit.rs`): such a texture is an image to its blits, where a texture
//!   over a buffer is its bytes.
//! - **Views of another component layout** (`texture/view_shadow.rs`). On a device whose Vulkan
//!   views reinterpret no image's components, such a view of an image is an image of the view's
//!   format whose truth is the viewed image's texels in the levels and layers it presents
//!   ([`ShadowSource::Image`]). Its copies are image to image (`vkCmdCopyImage` between formats of
//!   one copy unit, which keeps every bit), and the rules above are the same with the image in the
//!   buffer's place, except that a command writing the image two ways (below) is refused: no merge
//!   serves an image.
//! - **Rotated textures** (`rotated.rs`, `texture/rotation.rs`). A rotated texture renders through
//!   an image of its pass's frame whose truth is the buffer's bytes in the frame Metal stores the
//!   texture in, turned: its copies go through a staging buffer of the bytes as laid out and a
//!   tight one of the image's texels, the rotation kernel moving them between the two
//!   ([`ShadowTarget::Rotated`]). The rules above are the same, but for a command writing the bytes
//!   two ways, which is refused: no merge serves a turned image. Its functions read and write an
//!   image of the stored frame, unturned, whose writes Metal drops
//!   ([`ShadowCopies::drops_writes`]): copied in as any shadow, never copied back, and copied in
//!   again after a command that wrote through it.
//! - **Two ways** (`shadow_merge.rs`): one command that writes through a shadowed texture and also
//!   writes its bytes another way (through the buffer, a texture buffer, or another shadowed
//!   texture over them). A copy back would replace the other write's bytes with the shadow's, where
//!   Metal keeps both writes when they touch different bytes, so the shadow is kept before the
//!   command and merged after it: only the bytes the command changed through the texture reach the
//!   buffer. The merge goes before the encoder's next command, whatever it touches, and leaves the
//!   shadow to copy in again at its next use.
//! - **Converted texels** (`ycbcr.rs`): a two-plane YCbCr texture over a surface's two planes is an
//!   image of the texels Metal reads from them ([`ShadowSource::Converted`]). Its copy in is a
//!   conversion: the planes' bytes into a staging buffer of its own and a dispatch that writes
//!   every texel. The rules above are the same with the conversion in the copy's place; nothing
//!   writes the image (Metal drops a function's writes, and a pass into it is refused), so nothing
//!   is copied back.
//! - **YCbCr writes** (`ycbcr.rs`): a 4:4:4 two-plane YCbCr texture whose writes Metal stores into
//!   its planes takes a second shadow at a storage binding, an image of the written texels and a
//!   mark of each ([`ShadowSource::YCbCrWrite`]). Its copy in clears it, so no texel is marked; its
//!   copy back stores the marked texels' codes into the planes, through a staging buffer of their
//!   bytes and a dispatch of the encode kernel, and leaves every other byte as it was. The rules
//!   above are the same: a copy back leaves the texture's converted texels stale, so they convert
//!   again at their next use, and a command writing the planes two ways is refused, as no merge
//!   serves the planes. A pass's color attachment of such a texture renders into its render shadow,
//!   a texture whose shadow is one of these over floats: its copy back stores every texel's codes,
//!   the marks not read.
//! - **Two planes in a heap** (`texture/heap_two_planes.rs`): a heap's one-level 2D two-plane
//!   extended-range texture is an image of its texels whose truth is its two planes in the heap's
//!   bytes, as Metal lays them ([`ShadowTarget::Planes`]): its copies go through a staging buffer
//!   of the bytes as laid out and a tight one of the image's texels, the two-plane kernel moving
//!   them between the two. The rules above are the same, but for a command writing the bytes two
//!   ways, which is refused: no merge serves the planes.
//! - **Tiles in a heap** (`texture/heap_tiled.rs`): a texture of a plain format in a heap over host
//!   pages is an image of its elements whose truth is the heap's bytes, the elements tiled there as
//!   Metal lays them ([`ShadowTarget::Tiled`]): its copies go through a staging buffer of the bytes
//!   as laid out and a tight one of the image's elements, the tiled texel kernel moving them
//!   between the two. The rules above are the same; a command writing the bytes two ways merges as
//!   a buffer's texture does, the shadow's elements copied out tiled, each other byte as the last
//!   copy in left it.

use super::buffer::{Buffer, CallerMemory, WeakBuffer};
use super::error::{Error, RuntimeGap};
use super::resource::ResourceKey;
use super::rotated::{RotatedCopies, Turn};
use super::shadow_merge::ShadowMerge;
use super::texture::heap_tiled::HeapTiledCopies;
use super::texture::heap_two_planes::HeapPlaneCopies;
use super::texture::Texture;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::view_plan::reinterpretation_refused;
use super::ycbcr::{YCbCrConversion, YCbCrWriteBack};
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::PixelFormat;
use std::ops::Range;
use std::sync::Arc;

/// What a shadowed texture's copies need: its truth, its shadow, and the copies between them.
pub(super) struct ShadowCopies {
    /// What the shadow holds a copy of, and the copies.
    pub(super) source: ShadowSource,
    /// The shadow's identity: its image's, or its staging buffer's.
    pub(super) shadow_key: ResourceKey,
    /// Everything the shadow holds.
    pub(super) shadow_range: UseRange,
    /// The texture's hazard tracking, its truth's.
    pub(super) tracked: bool,
    /// Whether what a command writes through the shadow never reaches the truth: a rotated
    /// texture's image its functions read and write, whose writes Metal drops (EV
    /// ZCI2-rotprobe5-m4). Such a shadow is never copied back, and a command writing through it
    /// leaves it to copy in again at its next use.
    pub(super) drops_writes: bool,
}

/// A shadow's truth: what the shadow holds a copy of between an encoder's copies (the module's
/// rules, with the truth in the buffer's place).
pub(super) enum ShadowSource {
    /// A buffer's bytes: a 2D texture over a buffer, a staged texture buffer, a heap's depth or
    /// stencil texture.
    Bytes {
        buffer: ShadowBuffer,
        /// The buffer's identity.
        key: ResourceKey,
        /// The bytes the texture lies in, from its first texel to its last.
        bytes: Range<u64>,
        /// The shadow the texels are copied into and back from.
        target: ShadowTarget,
        /// The key of the host memory the buffer joined (`host_aliases.rs`), whose uses of the
        /// same bytes touch the truth too.
        alias: Option<ResourceKey>,
    },
    /// Another image's texels: a texture view of another component layout on a device whose
    /// Vulkan views reinterpret no image's components (`texture/view_shadow.rs`). The shadow is an
    /// image of the view's format and the image's shape; `regions` copy the levels and layers the
    /// view presents between the two, the same subresources, texel for texel (formats of one copy
    /// unit, whose copies keep every bit).
    Image {
        image: vk::Image,
        /// The image's identity.
        key: ResourceKey,
        /// The image's levels and layers the view presents.
        range: UseRange,
        shadow: vk::Image,
        /// From `image` to `shadow`; the copies back swap the two.
        regions: Vec<vk::ImageCopy>,
    },
    /// A surface's two planes in a buffer's bytes, which the shadow holds the texels Metal reads
    /// of (`ycbcr.rs`): its copy in converts them, and it is never copied back.
    Converted {
        buffer: ShadowBuffer,
        /// The buffer's identity.
        key: ResourceKey,
        /// The bytes the planes lie in, from the first plane's first byte to the last's last.
        bytes: Range<u64>,
        /// As `Bytes`'s.
        alias: Option<ResourceKey>,
        /// The conversion's staging buffer, kernel and dispatch, and the planes it converts
        /// (boxed: the largest variant's fields by far).
        conversion: Box<YCbCrConversion>,
        /// The texture's format, which a refusal names.
        pixel_format: PixelFormat,
    },
    /// A surface's two planes in a buffer's bytes, into which the texels a function writes to a
    /// 4:4:4 two-plane YCbCr texture are stored (`ycbcr.rs`): the shadow is an image of the
    /// written texels' bits and a mark of each, its copy in a clear of every texel, unmarked,
    /// and its copy back the encoding of the marked texels' codes into the planes.
    YCbCrWrite {
        buffer: ShadowBuffer,
        /// The buffer's identity.
        key: ResourceKey,
        /// The bytes the planes lie in, from the first plane's first byte to the last's last.
        bytes: Range<u64>,
        /// As `Bytes`'s.
        alias: Option<ResourceKey>,
        /// The shadow's image, which the copy in clears.
        image: vk::Image,
        /// The copy back's staging buffer, kernel and dispatch (boxed, as `Converted`'s).
        write_back: Box<YCbCrWriteBack>,
        /// The texture's format, which a refusal names.
        pixel_format: PixelFormat,
    },
}

/// The `VkBuffer` a shadow's bytes lie in.
pub(super) enum ShadowBuffer {
    /// One for good.
    Fixed(vk::Buffer),
    /// A buffer over host pages, which a rebind gives other pages (`rebind.rs`): its `VkBuffer`
    /// is read as each copy is recorded. A rebind keeps the bytes' offset in it (a heap's bytes
    /// start at its first byte, and a no-copy buffer with a texture over it keeps its offset in
    /// its page), so the copies' offsets hold. Held weakly: a copy is recorded only for a texture
    /// its encoder names, whose objects keep the buffer (its texels', or its heap's), so a heap's
    /// shadow image the runtime keeps after them keeps no no-copy heap's pages, range or token.
    Rebound(WeakBuffer),
}

impl ShadowBuffer {
    /// The `VkBuffer` `raw` of `buffer`'s bytes: read as each copy is recorded where a rebind
    /// may give the buffer other pages.
    pub(super) fn of(buffer: &Buffer, raw: vk::Buffer) -> ShadowBuffer {
        match buffer.caller_memory() {
            Some(CallerMemory::HostPages) => ShadowBuffer::Rebound(buffer.downgrade()),
            _ => ShadowBuffer::Fixed(raw),
        }
    }

    /// The `VkBuffer` the bytes lie in now: the null handle once a rebound buffer went, which no
    /// copy records (a copy's texture keeps its buffer).
    pub(super) fn raw(&self) -> vk::Buffer {
        match self {
            ShadowBuffer::Fixed(raw) => *raw,
            ShadowBuffer::Rebound(buffer) => buffer
                .upgrade()
                .map_or(vk::Buffer::null(), |buffer| buffer.at(0).0),
        }
    }
}

/// A shadowed texture's shadow (see the module).
pub(super) enum ShadowTarget {
    /// A 2D texture's image: `regions`, one per slice, copy its texels from the buffer, or from
    /// `staging` where Vulkan cannot copy them in place.
    Image {
        image: vk::Image,
        regions: Vec<vk::BufferImageCopy>,
        staging: Option<ShadowStaging>,
        /// The bytes of one of its texels in the buffer, by which a copy back of a part of the
        /// texture ([`ShadowCopy::area`]) offsets its region; `None` where no copy of it is
        /// bounded (a heap's depth or stencil texture, which is no color attachment).
        texel: Option<u64>,
    },
    /// A texture buffer's staging buffer, which its bindings view.
    Buffer(ShadowStaging),
    /// A rotated texture's image of its pass's frame: `staging` holds the bytes as laid out, and
    /// `turn` moves them to and from the image, turned (`rotated.rs`).
    Rotated {
        image: vk::Image,
        staging: ShadowStaging,
        turn: Box<RotatedCopies>,
    },
    /// A heap's two-plane extended-range texture's image: `staging` holds its two planes as
    /// Metal lays them in the heap's bytes, and `planes` moves them to and from the image
    /// (`texture/heap_two_planes.rs`).
    Planes {
        image: vk::Image,
        staging: ShadowStaging,
        planes: Box<HeapPlaneCopies>,
        /// The texture's format, which a refusal names.
        pixel_format: PixelFormat,
    },
    /// A heap texture's image over host pages: `staging` holds its elements as Metal tiles them
    /// in the heap's bytes, and `tiles` moves them to and from the image
    /// (`texture/heap_tiled.rs`).
    Tiled {
        image: vk::Image,
        staging: ShadowStaging,
        tiles: Box<HeapTiledCopies>,
    },
}

/// A shadowed texture's staging buffer (see the module): its bytes, from the first texel to the
/// last, laid out as in the texture's buffer from the staging buffer's start.
pub(super) struct ShadowStaging {
    /// A runtime buffer, held for the texture's life.
    pub(super) buffer: Buffer,
    /// The texture's bytes from its buffer into the staging buffer.
    pub(super) into: vk::BufferCopy,
    /// Each row of texels from the staging buffer back into the texture's buffer.
    pub(super) back: Vec<vk::BufferCopy>,
}

impl ShadowStaging {
    /// The staging buffer's `VkBuffer`, which `into` and `back` address.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.buffer.at(0).0
    }
}

impl ShadowCopies {
    /// Whether the shadow is a heap's two-plane texture's, whose truth is its two planes in the
    /// heap's bytes ([`ShadowTarget::Planes`]).
    pub(super) fn holds_planes(&self) -> bool {
        matches!(self.target(), Some(ShadowTarget::Planes { .. }))
    }

    /// The truth's identity and the range of it the shadow copies.
    pub(super) fn truth(&self) -> (ResourceKey, UseRange) {
        match &self.source {
            ShadowSource::Bytes { key, bytes, .. }
            | ShadowSource::Converted { key, bytes, .. }
            | ShadowSource::YCbCrWrite { key, bytes, .. } => (*key, UseRange::Bytes(bytes.clone())),
            ShadowSource::Image { key, range, .. } => (*key, range.clone()),
        }
    }

    /// Whether its copy in converts its truth's bytes (`ShadowSource::Converted`,
    /// `ShadowSource::YCbCrWrite`) rather than moving them: zero bytes there are not zero
    /// bytes in the shadow.
    pub(super) fn converts(&self) -> bool {
        matches!(
            self.source,
            ShadowSource::Converted { .. } | ShadowSource::YCbCrWrite { .. }
        )
    }

    /// The shadow of a buffer's bytes, or `None` for an image's and for converted texels'.
    pub(super) fn target(&self) -> Option<&ShadowTarget> {
        match &self.source {
            ShadowSource::Bytes { target, .. } => Some(target),
            ShadowSource::Image { .. }
            | ShadowSource::Converted { .. }
            | ShadowSource::YCbCrWrite { .. } => None,
        }
    }

    /// Whether a render pass that draws loads this shadow other than by the inverse of its
    /// store: a rotated texture's frame of rotation 4 (`rotated.rs`).
    pub(super) fn loads_apart(&self) -> bool {
        matches!(self.target(), Some(ShadowTarget::Rotated { turn, .. }) if turn.loads_apart())
    }

    /// Whether `used` touches the texture's truth: its bytes, or its image's levels and layers.
    pub(super) fn meets(&self, used: &Use<ResourceKey>) -> bool {
        let (key, range) = self.truth();
        (used.key == key || Some(used.key) == self.alias()) && used.range.overlaps(&range)
    }

    /// Why a command writing the truth both through this shadow and another way is refused: no
    /// merge serves an image's shadow (`Unsupported`), nor a rotated texture's, whose texels lie
    /// turned in its bytes (`NotImplemented(RotatedTexture)`); `None` for a shadow a merge serves.
    fn unmerged(&self) -> Option<Error> {
        match &self.source {
            ShadowSource::Image { .. } => Some(Error::Unsupported(reinterpretation_refused())),
            ShadowSource::Bytes {
                target: ShadowTarget::Rotated { .. },
                ..
            } => Some(Error::NotImplemented(RuntimeGap::RotatedTexture)),
            // Its texels lie in two planes of its bytes, which no merge reaches.
            ShadowSource::Bytes {
                target: ShadowTarget::Planes { pixel_format, .. },
                ..
            } => Some(Error::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(*pixel_format),
            )),
            ShadowSource::Bytes { .. } => None,
            // Nothing writes converted texels' image (`ycbcr.rs`): no command reaches here.
            // Its writes are encoded into two planes, which no merge reaches.
            ShadowSource::Converted { pixel_format, .. }
            | ShadowSource::YCbCrWrite { pixel_format, .. } => Some(Error::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(*pixel_format),
            )),
        }
    }

    /// Whether two shadowed textures copy some of the same truth.
    fn shares_truth(&self, other: &ShadowCopies) -> bool {
        let ((key, range), (other_key, other_range)) = (self.truth(), other.truth());
        let mine = [Some(key), self.alias()];
        let theirs = [Some(other_key), other.alias()];
        mine.iter()
            .flatten()
            .any(|key| theirs.contains(&Some(*key)))
            && range.overlaps(&other_range)
    }

    /// The key of the host memory the truth's buffer joined (`host_aliases.rs`): a use of it
    /// touches the same bytes as one of the buffer's own.
    fn alias(&self) -> Option<ResourceKey> {
        match &self.source {
            ShadowSource::Bytes { alias, .. }
            | ShadowSource::Converted { alias, .. }
            | ShadowSource::YCbCrWrite { alias, .. } => *alias,
            ShadowSource::Image { .. } => None,
        }
    }

    /// Where the texture's first byte lies in its buffer; `None` for an image's shadow.
    pub(super) fn first_byte(&self) -> Option<u64> {
        match self.target()? {
            ShadowTarget::Image {
                staging: Some(staging),
                ..
            }
            | ShadowTarget::Buffer(staging)
            | ShadowTarget::Rotated { staging, .. }
            | ShadowTarget::Planes { staging, .. }
            | ShadowTarget::Tiled { staging, .. } => Some(staging.into.src_offset),
            // Slice 0's region starts at the first texel.
            ShadowTarget::Image { regions, .. } => {
                Some(regions.first().map_or(0, |region| region.buffer_offset))
            }
        }
    }

    /// Record a copy of the shadow's texels into `to` from `at`, laid out as in the buffer from
    /// the texture's first byte (`shadow_merge.rs`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the shadow and `to` are live until
    /// it completes, `to` holds the texture's bytes from `at`, a multiple of 16, and an image is
    /// in the `GENERAL` layout.
    pub(super) unsafe fn record_out(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        to: vk::Buffer,
        at: u64,
    ) {
        // An image's shadow keeps no bytes, a rotated texture's keeps them turned, and a heap's
        // two-plane texture's in two planes: no merge is made of any (`shadow_merge.rs`,
        // `ShadowCopies::unmerged`).
        let Some(target) = self.target() else {
            return;
        };
        match target {
            ShadowTarget::Rotated { .. } | ShadowTarget::Planes { .. } => {}
            ShadowTarget::Tiled {
                image,
                staging,
                tiles,
            } => {
                // SAFETY: the caller's contract, which the tiles' record takes.
                unsafe { tiles.record_out(device, command_buffer, *image, staging, (to, at)) };
            }
            ShadowTarget::Image { image, regions, .. } => {
                let origin = regions.first().map_or(0, |region| region.buffer_offset);
                let moved: Vec<vk::BufferImageCopy> = regions
                    .iter()
                    .map(|region| vk::BufferImageCopy {
                        buffer_offset: region.buffer_offset - origin + at,
                        ..*region
                    })
                    .collect();
                // SAFETY: the caller's contract; the regions keep their offsets from the first
                // texel, multiples of the texel's size, and `at` is one too.
                unsafe {
                    device.cmd_copy_image_to_buffer(
                        command_buffer,
                        *image,
                        vk::ImageLayout::GENERAL,
                        to,
                        &moved,
                    );
                }
            }
            ShadowTarget::Buffer(staging) => {
                let region = vk::BufferCopy {
                    src_offset: staging.into.dst_offset,
                    dst_offset: at,
                    size: staging.into.size,
                };
                // SAFETY: the caller's contract; the staging buffer holds the texture's bytes.
                unsafe { device.cmd_copy_buffer(command_buffer, staging.raw(), to, &[region]) };
            }
        }
    }
}

/// One step of a shadowed texture's copies (the module's rules), `M` the scratch of a merge
/// (`shadow_merge.rs`; another type in the unit tests).
#[derive(Clone)]
pub(super) enum Step<M> {
    /// From the buffer into the shadow.
    Into,
    /// From the shadow back into the buffer.
    Back,
    /// The shadow kept, before a command that writes its bytes two ways.
    Keep(M),
    /// After that command, the bytes it changed through the texture merged into the buffer's.
    Merge(M),
}

/// One copy between a shadowed texture's buffer and its shadow.
#[derive(Clone)]
pub(super) struct ShadowCopy<M = Arc<ShadowMerge>> {
    pub(super) copies: Arc<ShadowCopies>,
    pub(super) step: Step<M>,
    /// A copy in before a command that leaves the texels in place ([`ShadowUse::kept`]).
    pub(super) kept: bool,
    /// A copy back of the texels inside `[x0, y0, x1, y1)` of a one-level, one-slice texture's
    /// image alone, where the commands since the last copy back wrote through it there and
    /// nowhere else ([`ShadowUse::area`]); `None` for every texel.
    pub(super) area: Option<[u32; 4]>,
}

impl<M> ShadowCopy<M> {
    /// The texels a copy back of a 2D texture's image over a buffer's bytes moves (its every
    /// region's, or its one region's inside [`area`](Self::area)): 0 for every other copy.
    pub(super) fn texels_back(&self) -> u64 {
        let Some(ShadowTarget::Image {
            regions,
            staging,
            texel,
            ..
        }) = self.copies.target()
        else {
            return 0;
        };
        if !matches!(self.step, Step::Back) {
            return 0;
        }
        let texels = |r: &vk::BufferImageCopy| {
            let e = r.image_extent;
            u64::from(e.width) * u64::from(e.height) * u64::from(e.depth)
        };
        // As `record` cuts the region.
        match (self.area, staging, regions.as_slice(), texel) {
            (Some(area), None, [region], Some(texel)) => {
                part_of(region, area, *texel).as_ref().map_or(0, texels)
            }
            _ => regions.iter().map(texels).sum(),
        }
    }
}

impl ShadowCopy {
    /// The stages the copy runs in: a rotated texture's copies dispatch the rotation kernel, a
    /// heap's two-plane texture's the two-plane kernel, and a tiled heap texture's the tiled
    /// texel kernel (its keep too).
    pub(super) fn stages(&self) -> vk::PipelineStageFlags2 {
        let tiled = matches!(self.copies.target(), Some(ShadowTarget::Tiled { .. }));
        let rotated = tiled
            || matches!(
                self.copies.target(),
                Some(ShadowTarget::Rotated { .. } | ShadowTarget::Planes { .. })
            );
        let converted = matches!(self.copies.source, ShadowSource::Converted { .. });
        let encoded = matches!(self.copies.source, ShadowSource::YCbCrWrite { .. });
        match self.step {
            Step::Merge(_) => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Into | Step::Back if rotated => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Keep(_) if tiled => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Into if converted => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Back if encoded => {
                vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Step::Into | Step::Back | Step::Keep(_) => vk::PipelineStageFlags2::TRANSFER,
        }
    }

    /// The copy's uses: a read of one side and a write of the other, in the transfer stage; a
    /// keep's and a merge's as `ShadowMerge` gives them.
    pub(super) fn uses(&self) -> Vec<Use<ResourceKey>> {
        let c = &self.copies;
        let read = Access::new(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let write = Access::new(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let (truth, range) = c.truth();
        if let ShadowSource::Converted { conversion, .. } = &c.source {
            return match &self.step {
                Step::Into => conversion.uses(Use::reads(truth, c.tracked, range, read), c),
                // Nothing writes the image: no copy back, keep or merge is made of it.
                Step::Back | Step::Keep(_) | Step::Merge(_) => Vec::new(),
            };
        }
        if let ShadowSource::YCbCrWrite { write_back, .. } = &c.source {
            return match &self.step {
                // A clear of every texel: nothing of the truth is read.
                Step::Into => vec![Use::writes(
                    c.shadow_key,
                    c.tracked,
                    c.shadow_range.clone(),
                    Access::new(
                        vk::PipelineStageFlags2::CLEAR,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ),
                )],
                Step::Back => write_back.uses(
                    Use {
                        key: truth,
                        tracked: c.tracked,
                        range,
                        read,
                        write,
                    },
                    c,
                ),
                // No merge serves the planes (`unmerged`): no keep or merge is made of it.
                Step::Keep(_) | Step::Merge(_) => Vec::new(),
            };
        }
        match &self.step {
            Step::Into => vec![
                Use::reads(truth, c.tracked, range, read),
                Use::writes(c.shadow_key, c.tracked, c.shadow_range.clone(), write),
            ],
            Step::Back => vec![
                Use::reads(c.shadow_key, c.tracked, c.shadow_range.clone(), read),
                Use::writes(truth, c.tracked, range, write),
            ],
            Step::Keep(merge) => merge.keep_uses(c),
            Step::Merge(merge) => merge.merge_uses(c),
        }
    }

    /// Record the copy into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the buffer and the shadow are live
    /// until it completes, and an image is in the `GENERAL` layout.
    pub(super) unsafe fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        let c = &self.copies;
        let into_shadow = match &self.step {
            Step::Into => true,
            Step::Back => false,
            // SAFETY: the caller's contract, which the merge's records take.
            Step::Keep(merge) => return unsafe { merge.record_keep(device, command_buffer, c) },
            Step::Merge(merge) => return unsafe { merge.record_merge(device, command_buffer, c) },
        };
        let (buffer, target) = match &c.source {
            ShadowSource::Bytes { buffer, target, .. } => (buffer.raw(), target),
            ShadowSource::Converted {
                buffer, conversion, ..
            } => {
                if into_shadow {
                    // SAFETY: the caller's contract, which the conversion's record takes.
                    unsafe { conversion.record(device, command_buffer, buffer.raw()) };
                }
                return;
            }
            ShadowSource::YCbCrWrite {
                buffer,
                image,
                write_back,
                ..
            } => {
                if into_shadow {
                    let range = vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    let zero = vk::ClearColorValue { uint32: [0; 4] };
                    // SAFETY: the caller's contract; the shadow takes transfer writes (made so,
                    // `texture/ycbcr_image.rs`) and the range is its one level and slice.
                    unsafe {
                        device.cmd_clear_color_image(
                            command_buffer,
                            *image,
                            vk::ImageLayout::GENERAL,
                            &zero,
                            &[range],
                        );
                    }
                } else {
                    // SAFETY: the caller's contract, which the write-back's record takes.
                    unsafe { write_back.record(device, command_buffer, buffer.raw()) };
                }
                return;
            }
            ShadowSource::Image {
                image,
                shadow,
                regions,
                ..
            } => {
                let (from, to) = if into_shadow {
                    (*image, *shadow)
                } else {
                    (*shadow, *image)
                };
                let regions: Vec<vk::ImageCopy> = if into_shadow {
                    regions.clone()
                } else {
                    regions
                        .iter()
                        .map(|region| vk::ImageCopy {
                            src_subresource: region.dst_subresource,
                            src_offset: region.dst_offset,
                            dst_subresource: region.src_subresource,
                            dst_offset: region.src_offset,
                            extent: region.extent,
                        })
                        .collect()
                };
                // SAFETY: the caller's contract; the regions lie in both images, which have
                // the same shape and formats of one copy unit (planned when the view was made).
                unsafe {
                    device.cmd_copy_image(
                        command_buffer,
                        from,
                        vk::ImageLayout::GENERAL,
                        to,
                        vk::ImageLayout::GENERAL,
                        &regions,
                    );
                }
                return;
            }
        };
        let (image, regions, staging) = match target {
            ShadowTarget::Image {
                image,
                regions,
                staging,
                texel,
            } => {
                let regions = match (self.area, staging, regions.as_slice(), *texel) {
                    (Some(area), None, [region], Some(texel)) if !into_shadow => {
                        match part_of(region, area, texel) {
                            Some(part) => std::borrow::Cow::Owned(vec![part]),
                            // The commands wrote no texel.
                            None => return,
                        }
                    }
                    _ => std::borrow::Cow::Borrowed(regions.as_slice()),
                };
                (*image, regions, staging.as_ref())
            }
            ShadowTarget::Rotated {
                image,
                staging,
                turn,
            } => {
                let turned = match (into_shadow, self.kept) {
                    (true, false) => Turn::Load,
                    (true, true) => Turn::Keep,
                    (false, _) => Turn::Store,
                };
                // SAFETY: the caller's contract, which the turn's record takes.
                unsafe {
                    turn.record(device, command_buffer, *image, (buffer, staging), turned);
                }
                return;
            }
            ShadowTarget::Planes {
                image,
                staging,
                planes,
                ..
            } => {
                // SAFETY: the caller's contract, which the planes' record takes.
                unsafe {
                    planes.record(
                        device,
                        command_buffer,
                        *image,
                        (buffer, staging),
                        into_shadow,
                    );
                }
                return;
            }
            ShadowTarget::Tiled {
                image,
                staging,
                tiles,
            } => {
                // SAFETY: the caller's contract, which the tiles' record takes.
                unsafe {
                    tiles.record(
                        device,
                        command_buffer,
                        *image,
                        (buffer, staging),
                        into_shadow,
                    );
                }
                return;
            }
            ShadowTarget::Buffer(staging) => {
                // SAFETY: the caller's contract; the copies lie in both buffers (planned when
                // the texture was made), and the staging buffer lives with the texture.
                unsafe {
                    if into_shadow {
                        device.cmd_copy_buffer(
                            command_buffer,
                            buffer,
                            staging.raw(),
                            &[staging.into],
                        );
                    } else {
                        device.cmd_copy_buffer(
                            command_buffer,
                            staging.raw(),
                            buffer,
                            &staging.back,
                        );
                    }
                }
                return;
            }
        };
        let texels = staging.map_or(buffer, ShadowStaging::raw);
        // SAFETY: the caller's contract; the regions lie in the buffers and the image (checked
        // when the texture was made), and the staging buffer lives with the texture.
        unsafe {
            if into_shadow {
                if let Some(staging) = staging {
                    device.cmd_copy_buffer(command_buffer, buffer, texels, &[staging.into]);
                    Barrier::everything().record(device, command_buffer);
                }
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    texels,
                    image,
                    vk::ImageLayout::GENERAL,
                    &regions,
                );
            } else {
                device.cmd_copy_image_to_buffer(
                    command_buffer,
                    image,
                    vk::ImageLayout::GENERAL,
                    texels,
                    &regions,
                );
                if let Some(staging) = staging {
                    Barrier::everything().record(device, command_buffer);
                    device.cmd_copy_buffer(command_buffer, texels, buffer, &staging.back);
                }
            }
        }
    }
}

/// `region` (a one-level, one-slice texture's every texel, `texel` bytes each in the buffer) cut
/// to the texels inside `[x0, y0, x1, y1)`: `None` where none lies there.
pub(super) fn part_of(
    region: &vk::BufferImageCopy,
    [x0, y0, x1, y1]: [u32; 4],
    texel: u64,
) -> Option<vk::BufferImageCopy> {
    let extent = region.image_extent;
    let (x1, y1) = (x1.min(extent.width), y1.min(extent.height));
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let row = match region.buffer_row_length {
        0 => extent.width,
        row => row,
    };
    Some(vk::BufferImageCopy {
        buffer_offset: region.buffer_offset
            + (u64::from(y0) * u64::from(row) + u64::from(x0)) * texel,
        buffer_row_length: row,
        image_offset: vk::Offset3D {
            x: region.image_offset.x + x0 as i32,
            y: region.image_offset.y + y0 as i32,
            z: region.image_offset.z,
        },
        image_extent: vk::Extent3D {
            width: x1 - x0,
            height: y1 - y0,
            depth: extent.depth,
        },
        ..*region
    })
}

/// Whether a command that uses the shadowed textures `shadows` and has `uses` writes the bytes
/// of `shadows[i]` two ways: through it, and through `uses` or another shadowed texture over
/// them (the module's rules).
fn two_ways(shadows: &[ShadowUse], i: usize, uses: &[Use<ResourceKey>]) -> bool {
    // A shadow whose writes Metal drops writes the bytes no way.
    let writes = |shadow: &ShadowUse| shadow.writes && !shadow.copies.drops_writes;
    let Some(shadow) = shadows.get(i).filter(|shadow| writes(shadow)) else {
        return false;
    };
    let bytes_written = uses
        .iter()
        .any(|used| !used.write.is_empty() && shadow.copies.meets(used));
    let shadow_written = shadows.iter().enumerate().any(|(j, other)| {
        j != i
            && writes(other)
            && other.copies.shadow_key != shadow.copies.shadow_key
            && other.copies.shares_truth(&shadow.copies)
    });
    bytes_written || shadow_written
}

/// The shadowed textures among `textures` (bound, made resident, or a pass's) a command with
/// `uses` uses, each once, writing through it when any of its uses writes.
pub(super) fn shadow_uses<'a>(
    textures: impl Iterator<Item = &'a Texture>,
    uses: &[Use<ResourceKey>],
) -> Vec<ShadowUse> {
    // A texture's own shadow, and the one a binding beyond its usage made
    // (`texture/usage_shadow.rs`).
    let mut copies = Vec::new();
    for texture in textures {
        copies.extend(texture.shadow().cloned());
        copies.extend(texture.usage_shadow_copies());
    }
    copies_used(copies.iter(), uses)
}

/// The shadowed textures of `shadows` a command with `uses` uses, each once, writing through it
/// when any of its uses writes.
pub(super) fn copies_used<'a>(
    shadows_of: impl Iterator<Item = &'a Arc<ShadowCopies>>,
    uses: &[Use<ResourceKey>],
) -> Vec<ShadowUse> {
    let mut shadows: Vec<ShadowUse> = Vec::new();
    for copies in shadows_of {
        if shadows
            .iter()
            .any(|shadow| shadow.copies.shadow_key == copies.shadow_key)
        {
            continue;
        }
        let mut used = uses.iter().filter(|used| used.key == copies.shadow_key);
        let Some(first) = used.next() else {
            continue;
        };
        let writes = !first.write.is_empty() || used.any(|other| !other.write.is_empty());
        shadows.push(ShadowUse {
            copies: Arc::clone(copies),
            writes,
            overwrites: false,
            kept: false,
            area: None,
        });
    }
    shadows
}

/// Add `shadow` to `shadows`: once per shadowed texture, writing through it when either use
/// writes (where both write, inside the rectangle bounding both areas), and overwriting it, or
/// leaving its texels in place, only when both do.
pub(super) fn join(shadows: &mut Vec<ShadowUse>, shadow: &ShadowUse) {
    match shadows
        .iter_mut()
        .find(|other| other.copies.shadow_key == shadow.copies.shadow_key)
    {
        Some(other) => {
            other.area = match (other.writes, shadow.writes) {
                (true, true) => bounding(other.area, shadow.area),
                (false, true) => shadow.area,
                (_, false) => other.area,
            };
            other.writes |= shadow.writes;
            other.overwrites &= shadow.overwrites;
            other.kept &= shadow.kept;
        }
        None => shadows.push(ShadowUse {
            copies: Arc::clone(&shadow.copies),
            writes: shadow.writes,
            overwrites: shadow.overwrites,
            kept: shadow.kept,
            area: shadow.area,
        }),
    }
}

/// A shadowed texture an encoder used, and where its image stands.
struct Entry<M> {
    copies: Arc<ShadowCopies>,
    /// The image holds the buffer's bytes as the encoder's commands so far left them.
    current: bool,
    /// A command wrote through the texture since the last copy back.
    dirty: bool,
    /// Where those commands wrote through it, where they bound it ([`ShadowUse::area`]).
    area: Option<[u32; 4]>,
    /// The scratch the image was kept in before a command that writes its bytes two ways, until
    /// the merge after it.
    merge: Option<M>,
}

/// A command's use of a shadowed texture: its copies, and whether it writes through it.
pub(super) struct ShadowUse {
    pub(super) copies: Arc<ShadowCopies>,
    pub(super) writes: bool,
    /// The command replaces every texel of the texture before it reads any (a render pass's
    /// attachment that clears each of its aspects over the whole of a one-level, one-slice
    /// texture): it takes no copy in. A render encoder's one pass is its only command with
    /// attachments, so nothing earlier in the encoder wrote through the texture; where the pass
    /// does not store it either, nothing is copied back, and the buffer keeps its bytes, Metal's
    /// answer (TOB probe 1).
    pub(super) overwrites: bool,
    /// The command leaves the texture's texels in place, but where it overwrites them: a render
    /// pass that draws nothing, of an attachment it renders. A rotated texture's copy in then
    /// turns its texels by the inverse of its store (`rotated::Turn::Keep`), where Metal's pass
    /// loads them otherwise. (An attachment such a pass loads, whose shadow is an image of a
    /// buffer's bytes, takes no use at all: `render/shadows.rs`.)
    pub(super) kept: bool,
    /// Where the command writes through the texture, where it bounds that: the texels inside
    /// `[x0, y0, x1, y1)` of a one-level, one-slice texture's image (a pass's attachment it
    /// loads, which only its draws' scissors change, `render/shadows.rs`); `None` for anywhere.
    pub(super) area: Option<[u32; 4]>,
}

/// The rectangle bounding `a` and `b`, each `[x0, y0, x1, y1)`; `None` (anywhere) where either is.
fn bounding(a: Option<[u32; 4]>, b: Option<[u32; 4]>) -> Option<[u32; 4]> {
    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (a?, b?);
    Some([a0.min(b0), a1.min(b1), a2.max(b2), a3.max(b3)])
}

/// The shadowed textures one encoder used, and the copies they need (the module's rules), `M`
/// the scratch of a merge.
pub(super) struct ShadowSync<M = Arc<ShadowMerge>> {
    entries: Vec<Entry<M>>,
}

impl<M> Default for ShadowSync<M> {
    fn default() -> Self {
        ShadowSync {
            entries: Vec::new(),
        }
    }
}

impl<M: Clone> ShadowSync<M> {
    /// The entry of `copies`, made when the encoder had none.
    fn entry(&mut self, copies: &Arc<ShadowCopies>) -> usize {
        match self
            .entries
            .iter()
            .position(|e| e.copies.shadow_key == copies.shadow_key)
        {
            Some(i) => i,
            None => {
                self.entries.push(Entry {
                    copies: Arc::clone(copies),
                    current: false,
                    dirty: false,
                    area: None,
                    merge: None,
                });
                self.entries.len() - 1
            }
        }
    }

    /// Copy back entry `i` (dirty): its image's bytes replace the buffer's, so every other
    /// texture over them is no longer current. One kept before a command that wrote its bytes
    /// two ways is merged instead, and is then no longer current either: the buffer holds the
    /// other way's writes too. A copy back covers only the area the commands wrote in, where
    /// they bound it: the image's other texels are what the last copy in left, which the
    /// buffer still holds (a write of its bytes another way leaves the image stale, and copies
    /// in again before the next use).
    fn copy_back(&mut self, i: usize, copies: &mut Vec<ShadowCopy<M>>) {
        let entry = &mut self.entries[i];
        let step = match &entry.merge {
            Some(merge) => {
                let merge = merge.clone();
                entry.merge = None;
                entry.current = false;
                Step::Merge(merge)
            }
            None => Step::Back,
        };
        let area = match step {
            Step::Back => entry.area,
            _ => None,
        };
        entry.dirty = false;
        entry.area = None;
        let written = Arc::clone(&entry.copies);
        copies.push(ShadowCopy {
            copies: Arc::clone(&written),
            step,
            kept: false,
            area,
        });
        for (j, other) in self.entries.iter_mut().enumerate() {
            if j != i && other.copies.shares_truth(&written) {
                other.current = false;
            }
        }
    }

    /// The copies to record, in order, before a command that uses the shadowed textures
    /// `shadows` and has `uses` (every use of the command, the textures' among them): merges of
    /// the textures kept before the command before, copies back of what the command touches,
    /// copies in of what it uses, and a keep of each texture it writes two ways, in a scratch
    /// `merge` makes. Refused, changing nothing, as `merge` refuses, and for a command that
    /// writes an image two ways, through a view's shadow of it and another alias, which no merge
    /// serves (`Unsupported`, `Feature::ViewFormatReinterpretation`), or a rotated texture's bytes
    /// two ways (`NotImplemented(RotatedTexture)`).
    pub(super) fn before(
        &mut self,
        shadows: &[ShadowUse],
        uses: &[Use<ResourceKey>],
        mut merge: impl FnMut(&ShadowCopies) -> Result<M, Error>,
    ) -> Result<Vec<ShadowCopy<M>>, Error> {
        // A copy back of an image's shadow would replace the other alias's writes, and only a
        // buffer's bytes laid out as the texture's are merged: such a command is refused before
        // any merge is made (a rotated texture's too).
        for (i, shadow) in shadows.iter().enumerate() {
            if let Some(refused) = shadow.copies.unmerged() {
                if two_ways(shadows, i, uses) {
                    return Err(refused);
                }
            }
        }
        let mut kept = Vec::new();
        for (i, shadow) in shadows.iter().enumerate() {
            if two_ways(shadows, i, uses) {
                kept.push((Arc::clone(&shadow.copies), merge(&shadow.copies)?));
            }
        }
        let mut copies = Vec::new();
        // Merge what the command before wrote two ways; copy back what this one reads or writes
        // another way: through the buffer's bytes, or through another texture over them that is
        // copied in next.
        for i in 0..self.entries.len() {
            if !self.entries[i].dirty {
                continue;
            }
            let entry = &self.entries[i].copies;
            let touched = uses.iter().any(|used| entry.meets(used))
                || shadows.iter().any(|shadow| {
                    shadow.copies.shadow_key != entry.shadow_key
                        && shadow.copies.shares_truth(entry)
                });
            if touched || self.entries[i].merge.is_some() {
                self.copy_back(i, &mut copies);
            }
        }
        for shadow in shadows {
            let i = self.entry(&shadow.copies);
            let overwrites = shadow.overwrites
                && !kept
                    .iter()
                    .any(|(copies, _)| copies.shadow_key == shadow.copies.shadow_key);
            if !self.entries[i].current {
                if !overwrites {
                    copies.push(ShadowCopy {
                        copies: Arc::clone(&shadow.copies),
                        step: Step::Into,
                        kept: shadow.kept,
                        area: None,
                    });
                }
                self.entries[i].current = true;
            }
        }
        for (shadow, merge) in kept {
            let i = self.entry(&shadow);
            self.entries[i].merge = Some(merge.clone());
            copies.push(ShadowCopy {
                copies: shadow,
                step: Step::Keep(merge),
                kept: false,
                area: None,
            });
        }
        Ok(copies)
    }

    /// Note a recorded command that used `shadows` and had `uses`: a write through a texture
    /// leaves its image to copy back; a write of the bytes another way leaves the images over
    /// them stale.
    pub(super) fn after(&mut self, shadows: &[ShadowUse], uses: &[Use<ResourceKey>]) {
        for entry in &mut self.entries {
            // A texture's own uses are its image's, never its bytes'.
            let stale = uses
                .iter()
                .any(|used| !used.write.is_empty() && entry.copies.meets(used));
            if stale {
                entry.current = false;
            }
        }
        for shadow in shadows.iter().filter(|shadow| shadow.writes) {
            let i = self.entry(&shadow.copies);
            // Writes Metal drops are not copied back, and the image no longer holds the bytes.
            let entry = &mut self.entries[i];
            match shadow.copies.drops_writes {
                true => entry.current = false,
                false => {
                    entry.area = match entry.dirty {
                        true => bounding(entry.area, shadow.area),
                        false => shadow.area,
                    };
                    entry.dirty = true;
                }
            }
        }
    }

    /// The copies back before work the encoder records apart (an indirect command buffer's
    /// commands, recorded at commit, with copies of their own): every texture written through
    /// since its last. The encoder then forgets every texture, so a later command copies in what
    /// that work may have written.
    pub(super) fn leave(&mut self) -> Vec<ShadowCopy<M>> {
        let copies = self.finish();
        self.entries.clear();
        copies
    }

    /// The copies back at the end of the encoder: every texture written through since its last
    /// (merged where kept).
    pub(super) fn finish(&mut self) -> Vec<ShadowCopy<M>> {
        let mut copies = Vec::new();
        for i in 0..self.entries.len() {
            if self.entries[i].dirty {
                self.copy_back(i, &mut copies);
            }
        }
        copies
    }
}
