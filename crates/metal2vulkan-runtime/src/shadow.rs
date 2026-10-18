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
//!   commit, so the first copy of each encoder sees every CPU write made before then. A copy in
//!   of a one-level, one-slice texture's image moves only the texels inside the rectangle its
//!   command reads and writes them in, where the command bounds that ([`ShadowUse::loads`]: a
//!   pass's attachment it loads and takes no other way, inside its draws' scissors). The image's
//!   other texels keep whatever it held, which no copy back moves and no command of the encoder
//!   reads: a later command that needs them current takes a copy back of what was written and a
//!   copy in of every texel first, and each encoder copies in again before its first use.
//! - **Copy in what changed.** An image whose bytes changed since its copy in only inside a
//!   rectangle of its texels, by writes [`Within`] it (a render pass instance's to a color
//!   attachment it loads, inside its draws' scissors), or by a copy back of a part of another
//!   shadow laid over the bytes texel for texel as it is ([`ShadowCopies::same_texels`]), takes
//!   a copy in of the texels inside the rectangle bounding those alone
//!   (`WorkCounts::bounded_shadow_copies`): every other texel holds the bytes still. A command
//!   that reads only inside a rectangle and writes none takes those of them inside it, and the
//!   rest stay to copy in before the next command that reads them. Any other write of the bytes
//!   leaves it to copy in as above; it holds no writes of its own, nor a keep.
//! - **Copy back** what a command wrote through the texture before the next command that touches
//!   the bytes some other way, and at the end of the encoder (orchestrator, 2026-10-04: at the
//!   end of the encoder that writes through the texture, not of the command buffer). A copy moves
//!   only the texels' rows, never the bytes between them; a copy back of a one-level, one-slice
//!   texture's image, only the texels inside the rectangle its writers since the last bound their
//!   writes to, where each bounds them ([`ShadowUse::area`]: a pass's attachment it loads, inside
//!   its draws' scissors), as the image's other texels are what the last copy in left. Where the
//!   device runs it faster so, a copy back of such a texture's image is a compute dispatch that
//!   stores the same bytes (`shadow_copy_back.rs`).
//! - **Merged back** ([`ShadowCopies::merges_back`]: a buffer's bytes another writer may write
//!   while a command buffer runs, where a merge serves the shadow). A copy back would write every
//!   texel it moves, those no command wrote too, with what the copy in read, over another writer's
//!   writes to them while the command buffer ran, which Metal keeps: it writes only the texels a
//!   command writes, and another writer writing bytes no command touches while a command buffer
//!   runs is no race under its contract. The CPU writes a `Shared` or `Managed` buffer's bytes and
//!   host pages so (an M3 Max under macOS 27.0.1 kept such a write, made inside a kernel's run,
//!   every time, a `Managed` buffer's too), and another queue's commands an untracked buffer's,
//!   which the program orders (the M3 Max ran another queue's fill of an untracked `Private`
//!   buffer's other texels inside a kernel's or a render pass's run and kept it). A tracked
//!   `Private` buffer's bytes have no such writer: the CPU writes none, and the tracker orders
//!   another queue's commands that touch them before or after the encoder's command buffer, as
//!   Metal does (the M3 Max ran another queue's fill of a tracked buffer after the kernel or the
//!   pass ended, every time), so its texture's shadow is copied back. A merged-back image is kept,
//!   after its copy in, before the first command since the last copy back that writes through it,
//!   and each copy back of it is a merge (`shadow_merge.rs`): only the words the encoder's
//!   commands changed through the texture since the keep reach the bytes. The keep holds the
//!   rectangle the commands write in where they bound it ([`ShadowUse::area`]); a command that
//!   writes outside it takes the merge first, and a keep of the rectangle bounding both. The image
//!   still holds the bytes as the encoder's commands left them. A command that replaces every
//!   texel ([`ShadowUse::overwrites`]) keeps nothing, as its copy back writes only texels it
//!   wrote; that copy back goes before the keep of a later command that writes through the
//!   texture, since the merge after the keep takes only the words changed since it. The keeps'
//!   scratch is the encoder's, one per texture and rectangle.
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
//!   one copy unit, which keeps every bit), or through a buffer of its own where Vulkan copies no
//!   texels between the two formats (an alpha format and another, `A8_UNORM` and `R8_UNORM`:
//!   `image_plan::copies_between`), and the rules above are the same with the image in the
//!   buffer's place, except that a command writing the image two ways (below) is refused: no merge
//!   serves an image. A shadow of one level and slice copied image to image (a 2D texture's own
//!   image's, `texture/usage_shadow.rs`, among them) has its copies in cut to rectangles as a
//!   one-level, one-slice texture's over a buffer has, its texels the image's at their own
//!   coordinates ([`ShadowCopies::cuts`]); its copies back move every texel, so a command that
//!   writes through it takes every texel in first.
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
use super::row_copies::RowCopies;
use super::shadow_copy_back::{CopyBack, WordView};
use super::shadow_merge::{ScratchSpares, ShadowMerge};
use super::texture::heap_tiled::HeapTiledCopies;
use super::texture::heap_two_planes::HeapPlaneCopies;
use super::texture::Texture;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::values::StorageMode;
use super::view_plan::reinterpretation_refused;
use super::ycbcr::{YCbCrConversion, YCbCrWriteBack};
use super::Device;
use crate::device::functions::Vk;
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
    /// texture's image its functions read and write, whose writes Metal drops. Such a shadow is
    /// never copied back, and a command writing through it leaves it to copy in again at its next
    /// use.
    pub(super) drops_writes: bool,
    /// What holds the truth, as the CPU may write it.
    pub(super) backing: Backing,
    /// The scratch buffers its merges are done with, kept for its next ones (`shadow_merge.rs`).
    pub(super) merge_scratch: ScratchSpares,
}

/// What holds a shadow's truth, as the CPU may write it, by which the encoders' first copies in
/// are counted (`WorkCounts::shadow_texels_first_copied_in`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Backing {
    /// An image (a texture's own, or the image a view presents): only the runtime's commands and
    /// calls write it.
    Image,
    /// A buffer of private or memoryless storage the runtime allocated: likewise.
    PrivateBuffer,
    /// A buffer of shared or managed storage the runtime allocated, which the CPU maps.
    MappedBuffer,
    /// Host pages a no-copy buffer imported, or the memory of the caller's `VkBuffer`.
    HostPages,
    /// Host pages that the texture's own image lies in too (an aliasing image, a pitched host
    /// image among them), a shadow of which reads them as the image left them.
    AliasedHostPages,
    /// The bytes of an IOSurface's plane.
    Surface,
}

impl Backing {
    /// What holds the bytes of `buffer`, a texture's over an IOSurface's plane where `surface`.
    pub(super) fn of(buffer: &Buffer, surface: bool) -> Backing {
        let private = matches!(
            buffer.resource_options().storage_mode(),
            Ok(StorageMode::Private | StorageMode::Memoryless)
        );
        match (surface, buffer.caller_memory(), private) {
            (true, _, _) => Backing::Surface,
            (false, Some(_), _) => Backing::HostPages,
            (false, None, true) => Backing::PrivateBuffer,
            (false, None, false) => Backing::MappedBuffer,
        }
    }

    /// The backing of a shadow of a texture whose own image lies in the bytes too.
    pub(super) fn aliased(self) -> Backing {
        match self {
            Backing::HostPages => Backing::AliasedHostPages,
            other => other,
        }
    }
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
        /// The buffer the copies go through where Vulkan copies no texels between the two
        /// images' formats (an alpha format and another, `image_plan::copies_between`).
        through: Option<ImageThrough>,
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
    /// One for good: a buffer's the runtime made, or the caller's own `VkBuffer`; and where the
    /// buffer's bytes end in it.
    Fixed(vk::Buffer, u64),
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
            _ => ShadowBuffer::Fixed(raw, buffer.start() + buffer.length()),
        }
    }

    /// Where the buffer's bytes end in its `VkBuffer` (its byte 0's offset there,
    /// [`Buffer::start`], past its length), which a rebind keeps: a merge in place reads and
    /// writes whole words of them up to it (`shadow_merge.rs`). 0 once a rebound buffer went.
    pub(super) fn end(&self) -> u64 {
        match self {
            ShadowBuffer::Fixed(_, end) => *end,
            ShadowBuffer::Rebound(buffer) => buffer
                .upgrade()
                .map_or(0, |buffer| buffer.start() + buffer.length()),
        }
    }

    /// The `VkBuffer` the bytes lie in now: the null handle once a rebound buffer went (a copy's
    /// texture keeps its buffer), and while it is detached (`detach.rs`), when its `VkBuffer` is
    /// the device's stand-in, which holds none of its bytes. No copy records it: an encoder
    /// records nothing of a command that names detached bytes (`EncoderCore::skips`), and nothing
    /// planned binds it (a copy back's dispatch, `CopyBack::new`; a merge's window,
    /// `ShadowMerge::new`).
    pub(super) fn raw(&self) -> vk::Buffer {
        match self {
            ShadowBuffer::Fixed(raw, _) => *raw,
            ShadowBuffer::Rebound(buffer) => buffer
                .upgrade()
                .and_then(|buffer| buffer.attached_at(0))
                .map_or(vk::Buffer::null(), |(raw, _)| raw),
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
        /// The image's unsigned-integer view, where the device copies back faster by a dispatch
        /// (`shadow_copy_back.rs`): through which its copies back dispatch and its merges read
        /// the texels the commands left (`shadow_merge.rs`); `None` where they copy.
        words: Option<WordView>,
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
/// A runtime buffer the copies between a view's shadow and its image go through
/// ([`ShadowSource::Image`]), held for the shadow's life: each region of the image into it, then
/// out of it into the other image, the same subresources both ways.
pub(super) struct ImageThrough {
    pub(super) buffer: Buffer,
    pub(super) regions: Vec<vk::BufferImageCopy>,
}

pub(super) struct ShadowStaging {
    /// A runtime buffer, held for the texture's life.
    pub(super) buffer: Buffer,
    /// The texture's bytes from its buffer into the staging buffer.
    pub(super) into: vk::BufferCopy,
    /// Each row of texels from the staging buffer back into the texture's buffer: one region for
    /// rows that lie one after another.
    pub(super) back: Vec<vk::BufferCopy>,
}

impl ShadowStaging {
    /// The staging buffer's `VkBuffer`, which `into` and `back` address.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.buffer.at(0).0
    }
}

impl ShadowCopies {
    /// Whether a copy in of the shadow's texels is cut to the rectangle of a part of them
    /// ([`ShadowCopy::area`]): a one-level, one-slice 2D texture's image over a buffer's bytes,
    /// one region copied in place (no staging buffer, [`part_of`]), whose copies back are cut
    /// too; or a shadow of one level and slice of another image, copied image to image
    /// ([`image_part_of`]), whose copies back move every texel ([`cuts_back`](Self::cuts_back)).
    pub(super) fn cuts(&self) -> bool {
        match &self.source {
            ShadowSource::Bytes {
                target:
                    ShadowTarget::Image {
                        regions,
                        staging: None,
                        texel: Some(_),
                        ..
                    },
                ..
            } => regions.len() == 1,
            ShadowSource::Image {
                regions,
                through: None,
                ..
            } => one_image_copy(regions).is_some(),
            _ => false,
        }
    }

    /// Whether a copy back of the shadow's texels is cut to a rectangle as its copy in is: a
    /// buffer's bytes' shadow that [`cuts`](Self::cuts). An image's shadow's copy back moves
    /// every texel, so a command that writes through it takes every texel in first.
    fn cuts_back(&self) -> bool {
        self.cuts() && self.target().is_some()
    }

    /// The texels the shadow holds: its every region's, every slice of each.
    pub(super) fn texels(&self) -> u64 {
        match &self.source {
            ShadowSource::Image { regions, .. } => regions.iter().map(image_copy_texels).sum(),
            ShadowSource::Bytes {
                target: ShadowTarget::Image { regions, .. },
                ..
            } => regions.iter().map(buffer_copy_texels).sum(),
            ShadowSource::Bytes { .. }
            | ShadowSource::Converted { .. }
            | ShadowSource::YCbCrWrite { .. } => 0,
        }
    }

    /// Whether `other`'s texels lie in the truth as this shadow's do, texel for texel: both cut
    /// to rectangles ([`cuts`](Self::cuts)), of one buffer, their one regions starting at one
    /// byte, rows as long, of one extent, of texels of one size; or of one image, their one
    /// regions of one aspect, level and slice, at one offset, of one extent. A change of one's
    /// texels inside a rectangle is then a change of the other's inside it alone.
    pub(super) fn same_texels(&self, other: &ShadowCopies) -> bool {
        let of_image = |copies: &ShadowCopies| match &copies.source {
            ShadowSource::Image {
                key,
                regions,
                through: None,
                ..
            } => one_image_copy(regions).map(|region| (*key, *region)),
            _ => None,
        };
        if let (Some((key, a)), Some((other_key, b))) = (of_image(self), of_image(other)) {
            let (s, t) = (a.src_subresource, b.src_subresource);
            return key == other_key
                && s.aspect_mask == t.aspect_mask
                && s.mip_level == t.mip_level
                && s.base_array_layer == t.base_array_layer
                && a.src_offset == b.src_offset
                && a.extent == b.extent;
        }
        let layout = |copies: &ShadowCopies| match &copies.source {
            ShadowSource::Bytes {
                key,
                target:
                    ShadowTarget::Image {
                        regions,
                        staging: None,
                        texel: Some(texel),
                        ..
                    },
                ..
            } => match regions.as_slice() {
                [region] => Some((*key, *region, *texel)),
                _ => None,
            },
            _ => None,
        };
        let (Some((key, a, texel)), Some((other_key, b, other_texel))) =
            (layout(self), layout(other))
        else {
            return false;
        };
        let row = |r: &vk::BufferImageCopy| match r.buffer_row_length {
            0 => r.image_extent.width,
            row => row,
        };
        key == other_key
            && texel == other_texel
            && a.buffer_offset == b.buffer_offset
            && row(&a) == row(&b)
            && a.image_offset == b.image_offset
            && a.image_extent == b.image_extent
    }

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

    /// Whether what the commands write through the shadow goes back into its truth as a merge
    /// (`shadow_merge.rs`): a buffer's bytes that another writer may write beside the commands'
    /// writes while a command buffer runs, whose writes to texels the commands left Metal keeps;
    /// never copied back over those texels (the module's rules). That is every buffer's bytes but
    /// a tracked `Private` or `Memoryless` buffer's the runtime made: the CPU writes the others
    /// (a `Shared` or `Managed` buffer's, host pages, the caller's own `VkBuffer`'s memory), and
    /// another queue's commands an untracked buffer's, which the tracker leaves to the program. An
    /// image's shadow, a turned one and two planes take no merge ([`unmerged`](Self::unmerged)),
    /// and writes Metal drops go back no way.
    pub(super) fn merges_back(&self) -> bool {
        let bytes = matches!(&self.source, ShadowSource::Bytes { .. });
        let other_writer = self.backing != Backing::PrivateBuffer || !self.tracked;
        bytes && other_writer && !self.drops_writes && self.unmerged().is_none()
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
        device: &Vk,
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
    /// The shadow kept, before a command that writes its bytes two ways, or before the first
    /// command since the last copy back that writes through it where it merges back
    /// ([`ShadowCopies::merges_back`]).
    Keep(M),
    /// After the command that wrote two ways, or in place of the next copy back, the bytes
    /// changed through the texture since the keep merged into the buffer's.
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
    /// nowhere else ([`ShadowUse::area`]), or a copy in of them alone, where the command reads
    /// and writes no other texel of it ([`ShadowUse::loads`]); `None` for every texel.
    pub(super) area: Option<[u32; 4]>,
    /// Why a copy in is made, as `WorkCounts` counts its texels; `None` for every other step.
    pub(super) cause: Option<CopyCause>,
}

/// Why a copy in is made, as `WorkCounts` counts its texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CopyCause {
    /// The encoder's first copy in of the texture: before the first command of the encoder that
    /// uses it, or the first after work it records apart ([`ShadowSync::leave`]). Every later one
    /// follows a change of the truth, or a command that needs texels the image does not hold
    /// (`WorkCounts::shadow_texels_first_copied_in`).
    First,
    /// A later copy in before a command that needs texels outside the rectangle the image holds
    /// (its last copy in's, [`ShadowUse::loads`]), the truth unchanged there or not
    /// (`WorkCounts::shadow_texels_recopied_beyond_held`).
    BeyondHeld,
    /// The truth changed since the image's copy in, other than by writes within a rectangle that
    /// left it stale there alone, for the reason given
    /// (`WorkCounts::shadow_texels_recopied_after_changes`).
    Changed(Lost),
    /// A copy in of the texels writes [`Within`] a rectangle changed since the image's copy in,
    /// [`ShadowCopy::area`] the rectangle bounding them (the module's rules,
    /// `WorkCounts::shadow_texels_bounded_copied_in`).
    Bounded,
}

/// Why a shadow's image no longer holds the truth as the encoder's commands left it, so the next
/// command that uses the texture copies it in again ([`CopyCause::Changed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lost {
    /// Another shadowed texture over the same truth was copied back or merged.
    OtherShadow,
    /// Its own merge after a command that wrote the truth two ways.
    TwoWays,
    /// A command wrote the truth another way (its bytes, or an image's levels and layers).
    Written,
    /// A command wrote the truth within a rectangle ([`Within`]) where that leaves the image
    /// stale nowhere alone: the writes' texture has no shadow, or the image is written, kept, or
    /// its texels are laid out otherwise.
    Within,
    /// A command wrote through it where Metal drops the writes.
    Dropped,
}

impl<M> ShadowCopy<M> {
    /// Whether the copy writes the texture's truth: a copy back or a merge.
    pub(super) fn writes_truth(&self) -> bool {
        matches!(self.step, Step::Back | Step::Merge(_))
    }

    /// The texels a copy in moves, as `record` cuts it, of a 2D texture's image over a buffer's
    /// bytes (as [`texels_in`](Self::texels_in) counts them) or of an image's shadow (its every
    /// region's, or its one region's inside [`area`](Self::area)): 0 for every other copy.
    pub(super) fn texels_copied_in(&self) -> u64 {
        match (&self.step, &self.copies.source) {
            (Step::Into, ShadowSource::Image { regions, .. }) => {
                match (self.area, self.copies.cuts(), one_image_copy(regions)) {
                    (Some(area), true, Some(region)) => image_part_of(region, area)
                        .as_ref()
                        .map_or(0, image_copy_texels),
                    _ => regions.iter().map(image_copy_texels).sum(),
                }
            }
            (Step::Into, _) => self.texels_in(),
            _ => 0,
        }
    }

    /// The texels a copy back of a 2D texture's image over a buffer's bytes moves (its every
    /// region's, or its one region's inside [`area`](Self::area)): 0 for every other copy.
    pub(super) fn texels_back(&self) -> u64 {
        match self.step {
            Step::Back => self.image_texels(),
            _ => 0,
        }
    }

    /// The texels a copy in of a 2D texture's image over a buffer's bytes moves, as
    /// [`texels_back`](Self::texels_back) counts them: 0 for every other copy.
    pub(super) fn texels_in(&self) -> u64 {
        match self.step {
            Step::Into => self.image_texels(),
            _ => 0,
        }
    }

    /// The texels a copy in or back of a 2D texture's image over a buffer's bytes moves, as
    /// `record` cuts its region: 0 for any other shadow.
    fn image_texels(&self) -> u64 {
        let Some(ShadowTarget::Image {
            regions,
            staging,
            texel,
            ..
        }) = self.copies.target()
        else {
            return 0;
        };
        // As `record` cuts the region.
        match (self.area, staging, regions.as_slice(), texel) {
            (Some(area), None, [region], Some(texel)) => part_of(region, area, *texel)
                .as_ref()
                .map_or(0, buffer_copy_texels),
            _ => regions.iter().map(buffer_copy_texels).sum(),
        }
    }
}

/// The texels a copy of `region` moves between a buffer and an image.
fn buffer_copy_texels(region: &vk::BufferImageCopy) -> u64 {
    let e = region.image_extent;
    u64::from(e.width) * u64::from(e.height) * u64::from(e.depth)
}

/// The texels a copy of `region` moves between two images: every slice's.
fn image_copy_texels(region: &vk::ImageCopy) -> u64 {
    let e = region.extent;
    let layers = u64::from(region.src_subresource.layer_count);
    u64::from(e.width) * u64::from(e.height) * u64::from(e.depth) * layers
}

impl ShadowCopy {
    /// The dispatch that takes this copy where it is a copy back that dispatches
    /// (`shadow_copy_back.rs`): a one-slice 2D texture's image over a buffer's bytes, copied in
    /// place, with its integer view, of texels the copy moves, made on `device` over the buffer's
    /// `VkBuffer` now. `None` where the copy takes it (any other copy, or one the dispatch does
    /// not fit).
    pub(super) fn dispatched(&self, device: &Device) -> Option<CopyBack> {
        let ShadowSource::Bytes {
            buffer,
            target:
                ShadowTarget::Image {
                    regions,
                    staging: None,
                    texel: Some(texel),
                    words: Some(words),
                    ..
                },
            ..
        } = &self.copies.source
        else {
            return None;
        };
        let ([region], Step::Back) = (regions.as_slice(), &self.step) else {
            return None;
        };
        let region = match self.area {
            Some(area) => part_of(region, area, *texel)?,
            None => *region,
        };
        CopyBack::new(device, *words, buffer.raw(), &region)
    }

    /// The texels a copy in of a one-region, one-slice 2D texture's image over a buffer's bytes
    /// copied in place moves, as `record` cuts its region (where they lie in the buffer, and in
    /// the image): what a merge's keep may take with it (`ShadowMerge::keep_in`). `None` for any
    /// other copy, or one that moves no texel.
    pub(super) fn copied_region(&self) -> Option<vk::BufferImageCopy> {
        let (
            Step::Into,
            ShadowSource::Bytes {
                target:
                    ShadowTarget::Image {
                        regions,
                        staging: None,
                        texel: Some(texel),
                        ..
                    },
                ..
            },
        ) = (&self.step, &self.copies.source)
        else {
            return None;
        };
        let [region] = regions.as_slice() else {
            return None;
        };
        match self.area {
            Some(area) => part_of(region, area, *texel),
            None => Some(*region),
        }
    }

    /// The uses of the dispatch that takes a copy back ([`dispatched`](Self::dispatched)): a
    /// sampled read of the shadow and a storage write of the buffer's bytes, in the compute
    /// stage.
    pub(super) fn dispatched_uses(&self) -> Vec<Use<ResourceKey>> {
        let c = &self.copies;
        let (truth, range) = c.truth();
        vec![
            Use::reads(
                c.shadow_key,
                c.tracked,
                c.shadow_range.clone(),
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                ),
            ),
            Use::writes(
                truth,
                c.tracked,
                range,
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            ),
        ]
    }

    /// The copy of rows that moves this copy back's rows from its staging buffer into the
    /// buffer's bytes, made on `device` over the buffer's `VkBuffer` now (`row_copies.rs`): a
    /// copy back through a staging buffer, of a texture over a buffer's bytes, a texture
    /// buffer's, a rotated texture's, or a heap's two-plane or tiled texture's. `None` for any
    /// other copy.
    pub(super) fn staged_back(&self, device: &Device) -> Option<RowCopies> {
        let (ShadowSource::Bytes { buffer, target, .. }, Step::Back) =
            (&self.copies.source, &self.step)
        else {
            return None;
        };
        let staging = match target {
            ShadowTarget::Image {
                staging: Some(staging),
                ..
            }
            | ShadowTarget::Buffer(staging)
            | ShadowTarget::Rotated { staging, .. }
            | ShadowTarget::Planes { staging, .. }
            | ShadowTarget::Tiled { staging, .. } => staging,
            ShadowTarget::Image { staging: None, .. } => return None,
        };
        let back = staging.back.clone();
        Some(RowCopies::new(device, staging.raw(), buffer.raw(), back))
    }

    /// The copy's stages where its staged rows go back by `back` ([`staged_back`]): with the
    /// compute stage where they dispatch.
    ///
    /// [`staged_back`]: Self::staged_back
    pub(super) fn stages_with(&self, back: Option<&RowCopies>) -> vk::PipelineStageFlags2 {
        match back {
            Some(rows) if rows.dispatched() => self.stages() | rows.stages(),
            _ => self.stages(),
        }
    }

    /// The copy's uses where its staged rows go back by `back` ([`staged_back`]): where they
    /// dispatch, the write of the buffer's bytes is the dispatch's, and so is the read of a
    /// texture buffer's staging buffer, its shadow; an image's shadow is read by its copy into
    /// the staging buffer.
    ///
    /// [`staged_back`]: Self::staged_back
    pub(super) fn uses_with(&self, back: Option<&RowCopies>) -> Vec<Use<ResourceKey>> {
        let mut uses = self.uses();
        let (Some(rows), Step::Back, ShadowSource::Bytes { target, .. }) =
            (back, &self.step, &self.copies.source)
        else {
            return uses;
        };
        if !rows.dispatched() {
            return uses;
        }
        let shadow_read = matches!(target, ShadowTarget::Buffer(_));
        for used in &mut uses {
            if !used.write.is_empty() {
                used.write = rows.write();
            } else if shadow_read && !used.read.is_empty() {
                used.read = rows.read();
            }
        }
        uses
    }

    /// The stages the copy runs in: a rotated texture's copies dispatch the rotation kernel, a
    /// heap's two-plane texture's the two-plane kernel, a tiled heap texture's the tiled texel
    /// kernel (its keep too), and a keep that dispatches the copy back's kernel
    /// (`ShadowMerge::dispatches`) that kernel.
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
            Step::Keep(ref merge) if tiled || merge.dispatches() => {
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

    /// Record the copy into `command_buffer`, its staged rows back (where it has them) by `back`
    /// ([`staged_back`](Self::staged_back)).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass, after a barrier that orders the copy
    /// after what precedes it; the buffer and the shadow are live until it completes, and an
    /// image is in the `GENERAL` layout; `back` is this copy's `staged_back`, over the buffer's
    /// `VkBuffer` as it is now.
    pub(super) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        back: Option<&RowCopies>,
    ) {
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
                through,
                ..
            } => {
                let (from, to) = if into_shadow {
                    (*image, *shadow)
                } else {
                    (*shadow, *image)
                };
                if let Some(through) = through {
                    let (buffer, _) = through.buffer.at(0);
                    // SAFETY: the caller's contract; the regions lie in both images, which have
                    // the same shape, and in the buffer, which lives with the shadow; the
                    // barrier orders the copy out of the buffer after the copy into it.
                    unsafe {
                        device.cmd_copy_image_to_buffer(
                            command_buffer,
                            from,
                            vk::ImageLayout::GENERAL,
                            buffer,
                            &through.regions,
                        );
                        Barrier::everything().record(device, command_buffer);
                        device.cmd_copy_buffer_to_image(
                            command_buffer,
                            buffer,
                            to,
                            vk::ImageLayout::GENERAL,
                            &through.regions,
                        );
                    }
                    return;
                }
                // A copy in cut to a rectangle moves its texels alone; a copy back moves every
                // texel (`ShadowCopies::cuts_back`).
                let regions: Vec<vk::ImageCopy> = if into_shadow {
                    match (self.area, c.cuts(), one_image_copy(regions)) {
                        (Some(area), true, Some(region)) => match image_part_of(region, area) {
                            Some(part) => vec![part],
                            // The command reads no texel.
                            None => return,
                        },
                        _ => regions.clone(),
                    }
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
                ..
            } => {
                let regions = match (self.area, staging, regions.as_slice(), *texel) {
                    (Some(area), None, [region], Some(texel)) => {
                        match part_of(region, area, texel) {
                            Some(part) => std::borrow::Cow::Owned(vec![part]),
                            // The commands read or wrote no texel.
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
                    turn.record(
                        device,
                        command_buffer,
                        *image,
                        (buffer, staging),
                        turned,
                        back,
                    );
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
                        back,
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
                        back,
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
                        record_back(device, command_buffer, (staging, buffer), back);
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
                    record_back(device, command_buffer, (staging, buffer), back);
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

/// Record the rows of `staging` back into `buffer`: by `back` where the caller made it
/// ([`ShadowCopy::staged_back`]), else as one copy of every row. Rotated textures' and heap
/// textures' copies back end so too (`rotated.rs`, `texture/heap_two_planes.rs`,
/// `texture/heap_tiled.rs`).
///
/// # Safety
///
/// As [`ShadowCopy::record`], after a barrier from the staging buffer's writes.
pub(super) unsafe fn record_back(
    device: &Vk,
    command_buffer: vk::CommandBuffer,
    (staging, buffer): (&ShadowStaging, vk::Buffer),
    back: Option<&RowCopies>,
) {
    // SAFETY: the caller's contract; the rows lie in the staging buffer and the buffer's bytes
    // (planned when the texture was made), and the staging buffer lives with the texture.
    unsafe {
        match back {
            Some(rows) => rows.record(device, command_buffer),
            None => device.cmd_copy_buffer(command_buffer, staging.raw(), buffer, &staging.back),
        }
    }
}

/// The one region of `regions`, a shadow's of another image, where it is one level and one slice
/// of a 2D image: the texels a rectangle of the texture's own names.
fn one_image_copy(regions: &[vk::ImageCopy]) -> Option<&vk::ImageCopy> {
    match regions {
        [region] if region.src_subresource.layer_count == 1 && region.extent.depth == 1 => {
            Some(region)
        }
        _ => None,
    }
}

/// `region` (one level and one slice of an image, copied into the same texels of its shadow) cut
/// to the texels inside `[x0, y0, x1, y1)`: `None` where none lies there.
pub(super) fn image_part_of(
    region: &vk::ImageCopy,
    [x0, y0, x1, y1]: [u32; 4],
) -> Option<vk::ImageCopy> {
    let extent = region.extent;
    let (x1, y1) = (x1.min(extent.width), y1.min(extent.height));
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let moved = |offset: vk::Offset3D| vk::Offset3D {
        x: offset.x + x0 as i32,
        y: offset.y + y0 as i32,
        z: offset.z,
    };
    Some(vk::ImageCopy {
        src_offset: moved(region.src_offset),
        dst_offset: moved(region.dst_offset),
        extent: vk::Extent3D {
            width: x1 - x0,
            height: y1 - y0,
            depth: extent.depth,
        },
        ..*region
    })
}

/// Whether a command writes through `shadow` without replacing every texel, where it merges back
/// ([`ShadowCopies::merges_back`]): kept and merged back (the module's rules), as a command that
/// writes them two ways too ([`two_ways`]) is.
fn merged_back(shadow: &ShadowUse) -> bool {
    shadow.writes && !shadow.overwrites && shadow.copies.merges_back()
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
            loads: None,
        });
    }
    shadows
}

/// Add `shadow` to `shadows`: once per shadowed texture, writing through it when either use
/// writes (where both write, inside the rectangle bounding both areas), reading its texels inside
/// the rectangle bounding where both read them (anywhere where either does not bound that), and
/// overwriting it, or leaving its texels in place, only when both do.
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
            other.loads = bounding(other.loads, shadow.loads);
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
            loads: shadow.loads,
        }),
    }
}

/// Where a shadowed texture's image stands against its truth in an encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Standing {
    /// Not copied in since the encoder met the texture: its next copy in is the encoder's first.
    Unread,
    /// The image holds the buffer's bytes as the encoder's commands so far left them, inside
    /// `held`.
    Current,
    /// The truth changed since the image's copy in, for the reason given.
    Lost(Lost),
}

/// A shadowed texture an encoder used, and where its image stands.
struct Entry<M> {
    copies: Arc<ShadowCopies>,
    standing: Standing,
    /// Where the image holds them when [`Standing::Current`]: the texels inside `[x0, y0, x1, y1)`, all its
    /// last copy in moved ([`ShadowUse::loads`]); `None` for every texel. Its other texels are
    /// whatever the image held before, which no command of the encoder reads and no copy back
    /// moves: a command that needs them current copies back what was written and copies every
    /// texel in first.
    held: Option<[u32; 4]>,
    /// A command wrote through the texture since the last copy back.
    dirty: bool,
    /// Where those commands wrote through it, where they bound it ([`ShadowUse::area`]).
    area: Option<[u32; 4]>,
    /// The scratch the image was kept in before a command that writes its bytes two ways, until
    /// the merge after it, or before the first command since the last copy back that wrote
    /// through a texture that merges back ([`ShadowCopies::merges_back`]), until the next copy
    /// back, which is then a merge.
    merge: Option<M>,
    /// The keep in `merge` is a command's that writes the bytes two ways: merged before the
    /// encoder's next command, after which the image no longer holds the bytes.
    two_ways: bool,
    /// The scratch of the texture's keeps where it merges back, made at the first
    /// command of the encoder that writes through it, and kept in again at each keep after a
    /// merge: each merge's dispatches end with a barrier to everything after them.
    scratch: Option<M>,
    /// When [`Standing::Current`], the rectangle bounding the texels whose bytes changed since the image's
    /// copy in, where only writes [`Within`] it changed them (the module's rules): the next
    /// command that uses the texture copies those in first, or those of them it reads where it
    /// reads only inside a rectangle and writes none, the rest left here. `None` where none
    /// changed.
    stale: Option<[u32; 4]>,
}

impl<M> Entry<M> {
    /// Whether a change of the bytes of `texels`' texels inside a rectangle leaves the image
    /// current but there ([`Entry::stale`]): it holds the bytes, and no write of its own or keep,
    /// and its texels are `texels`' ([`ShadowCopies::same_texels`]).
    fn stale_within(&self, texels: &ShadowCopies) -> bool {
        self.standing == Standing::Current
            && !self.dirty
            && self.merge.is_none()
            && self.copies.same_texels(texels)
    }

    /// Note a change of the truth for `lost`: the image no longer holds it. One not copied in
    /// yet stays so, and one that lost it already keeps the first reason.
    fn lose(&mut self, lost: Lost) {
        if self.standing == Standing::Current {
            self.standing = Standing::Lost(lost);
        }
    }

    /// Note a change of the bytes of `texels`' texels inside `area` (`[x0, y0, x1, y1)`), or
    /// anywhere where `area` is `None`, for `lost`: the image is stale there, or no longer
    /// current.
    fn changed(&mut self, texels: &ShadowCopies, area: Option<[u32; 4]>, lost: Lost) {
        match area.filter(|_| self.stale_within(texels)) {
            Some(area) => {
                self.stale = Some(match self.stale {
                    Some(stale) => [
                        stale[0].min(area[0]),
                        stale[1].min(area[1]),
                        stale[2].max(area[2]),
                        stale[3].max(area[3]),
                    ],
                    None => area,
                });
            }
            None => self.lose(lost),
        }
    }
}

/// A command's writes of a texture's bytes that change none of its texels outside a rectangle:
/// a render pass instance's of a color attachment it loads, which only its draws change, inside
/// their scissors (`render/shadows.rs`).
pub(super) struct Within {
    /// The writes' uses of the bytes.
    pub(super) uses: Vec<Use<ResourceKey>>,
    /// A shadow of the texture, whose texels are the attachment's pixels: an image whose texels
    /// are laid out as its ([`ShadowCopies::same_texels`]) changed inside `area` alone. `None`
    /// where the texture has none, and then every image over the bytes changed anywhere.
    pub(super) texels: Option<Arc<ShadowCopies>>,
    /// The rectangle, `[x0, y0, x1, y1)` of the texels.
    pub(super) area: [u32; 4],
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
    /// answer (a Metal probe on the M3 Max and an M4).
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
    /// Where the command reads the texture's texels, where it bounds that: the texels inside
    /// `[x0, y0, x1, y1)` of a one-level, one-slice texture's image, whose others it reads at
    /// most into the image alone, never into what it writes elsewhere or what a copy back moves
    /// (a pass's attachment it loads and takes in no other way, whose fragments run only inside
    /// its draws' scissors and read the attachment at their own pixel alone,
    /// `render/shadows.rs`); `None` for anywhere. Its copy in moves only those texels.
    pub(super) loads: Option<[u32; 4]>,
}

/// The rectangle bounding `a` and `b`, each `[x0, y0, x1, y1)`; `None` (anywhere) where either is.
fn bounding(a: Option<[u32; 4]>, b: Option<[u32; 4]>) -> Option<[u32; 4]> {
    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (a?, b?);
    Some([a0.min(b0), a1.min(b1), a2.max(b2), a3.max(b3)])
}

/// The texels inside both `a` and `b`, each `[x0, y0, x1, y1)`; `None` where they share none.
fn meet([a0, a1, a2, a3]: [u32; 4], [b0, b1, b2, b3]: [u32; 4]) -> Option<[u32; 4]> {
    let r = [a0.max(b0), a1.max(b1), a2.min(b2), a3.min(b3)];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}

/// The rectangle bounding the texels of `a` outside `b`, each `[x0, y0, x1, y1)`; `None` where `b`
/// covers `a`.
fn outside(a: [u32; 4], b: [u32; 4]) -> Option<[u32; 4]> {
    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (a, b);
    if meet(a, b).is_none() {
        return Some(a);
    }
    let (across, down) = (b0 <= a0 && a2 <= b2, b1 <= a1 && a3 <= b3);
    // Where `b` crosses `a` from side to side, the texels left are the rows above and below it;
    // from top to bottom, the columns beside it. One of the two alone is a rectangle of its own.
    let left = match (across, down) {
        (true, true) => return None,
        (true, false) => (a1 < b1, b3 < a3, [a0, a1, a2, b1], [a0, b3, a2, a3]),
        (false, true) => (a0 < b0, b2 < a2, [a0, a1, b0, a3], [b2, a1, a2, a3]),
        (false, false) => return Some(a),
    };
    match left {
        (true, false, before, _) => Some(before),
        (false, true, _, after) => Some(after),
        _ => Some(a),
    }
}

/// Whether an image holding the buffer's texels inside `held` (`None` every texel) holds those
/// inside `needs` (`None` every texel).
fn holds(held: Option<[u32; 4]>, needs: Option<[u32; 4]>) -> bool {
    match (held, needs) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some([h0, h1, h2, h3]), Some([n0, n1, n2, n3])) => {
            h0 <= n0 && h1 <= n1 && n2 <= h2 && n3 <= h3
        }
    }
}

/// The shadowed textures one encoder used, and the copies they need (the module's rules), `M`
/// the scratch of a merge.
pub(super) struct ShadowSync<M = Arc<ShadowMerge>> {
    entries: Vec<Entry<M>>,
}

impl<M> Default for ShadowSync<M> {
    /// No texture used yet.
    fn default() -> Self {
        ShadowSync {
            entries: Vec::new(),
        }
    }
}

impl<M> ShadowSync<M> {
    /// Whether the encoder used no shadowed texture yet: then a command that uses none takes no
    /// copy, and changes nothing here ([`before`](Self::before), [`after`](Self::after)).
    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many of the textures the encoder used share their truth with one it used before
    /// (another texture over the same bytes): each copies the texels in apart
    /// (`WorkCounts::shadows_sharing_truth`).
    pub(super) fn sharing_truth(&self) -> usize {
        let entries = &self.entries;
        (1..entries.len())
            .filter(|&i| {
                (entries[..i].iter()).any(|earlier| earlier.copies.shares_truth(&entries[i].copies))
            })
            .count()
    }
}

/// What a merge's scratch keeps of a shadow's texels (`shadow_merge.rs`): the rectangle
/// `[x0, y0, x1, y1)` its keep, its copy of the changed texels and its merge are cut to, or `None`
/// for every texel.
pub(super) trait KeptArea {
    fn kept_area(&self) -> Option<[u32; 4]>;
}

impl KeptArea for Arc<ShadowMerge> {
    fn kept_area(&self) -> Option<[u32; 4]> {
        self.area()
    }
}

impl<M: Clone + KeptArea> ShadowSync<M> {
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
                    standing: Standing::Unread,
                    held: None,
                    dirty: false,
                    area: None,
                    merge: None,
                    two_ways: false,
                    scratch: None,
                    stale: None,
                });
                self.entries.len() - 1
            }
        }
    }

    /// Copy back entry `i` (dirty): its image's bytes replace the buffer's, so every other
    /// texture over them is no longer current, or, where the copy back covers a rectangle of
    /// texels laid out as its own, stale there alone (the module's rules). One kept is merged
    /// instead: one kept before a command that wrote its bytes two ways is then no longer current
    /// either, as the buffer holds the other way's writes too, and leaves every other texture
    /// over the bytes no longer current; one kept as it merges back still holds what the
    /// encoder's commands left in them, and its merge changes only words they changed through
    /// it, inside the area they wrote in where they bound it, so it leaves the others stale there
    /// alone, as a copy back of that area does. A copy back covers only the
    /// area the commands wrote in, where they bound it: the image's other texels are what the last
    /// copy in left, which the buffer still holds (a write of its bytes another way leaves the
    /// image stale, and copies in again before the next use).
    fn copy_back(&mut self, i: usize, copies: &mut Vec<ShadowCopy<M>>) {
        let entry = &mut self.entries[i];
        let two_ways = entry.two_ways;
        let step = match &entry.merge {
            Some(merge) => {
                let merge = merge.clone();
                entry.merge = None;
                if two_ways {
                    entry.lose(Lost::TwoWays);
                }
                entry.two_ways = false;
                Step::Merge(merge)
            }
            None => Step::Back,
        };
        let area = match step {
            Step::Back => entry.area,
            _ => None,
        };
        // Where the buffer's bytes may have changed for the other textures over them.
        let changed = match step {
            Step::Merge(_) if two_ways => None,
            _ => entry.area,
        };
        entry.dirty = false;
        entry.area = None;
        let written = Arc::clone(&entry.copies);
        copies.push(ShadowCopy {
            copies: Arc::clone(&written),
            step,
            kept: false,
            area,
            cause: None,
        });
        for (j, other) in self.entries.iter_mut().enumerate() {
            if j != i && other.copies.shares_truth(&written) {
                other.changed(&written, changed, Lost::OtherShadow);
            }
        }
    }

    /// The copies to record, in order, before a command that uses the shadowed textures
    /// `shadows` and has `uses` (every use of the command, the textures' among them): merges of
    /// the textures kept before the command before, copies back of what the command touches,
    /// copies in of what it uses, a keep of each texture it writes through that merges back, where
    /// none is kept since the last copy back, and a keep of each texture it writes two ways, each
    /// in a scratch `merge` makes (told whether the command writes the bytes two ways, and the
    /// rectangle of texels it is to keep, `None` for every texel). Where the texture merges back,
    /// the scratch keeps the rectangle the command writes in, where
    /// it bounds that and the copies are cut to rectangles ([`ShadowUse::area`]); a later command
    /// that writes outside what the encoder's scratch keeps takes a scratch of the rectangle
    /// bounding both, after a merge of what was kept in the last. Refused, changing nothing, as
    /// `merge` refuses, and for a command that
    /// writes an image two ways, through a view's shadow of it and another alias, which no merge
    /// serves (`Unsupported`, `Feature::ViewFormatReinterpretation`), or a rotated texture's bytes
    /// two ways (`NotImplemented(RotatedTexture)`).
    pub(super) fn before(
        &mut self,
        shadows: &[ShadowUse],
        uses: &[Use<ResourceKey>],
        mut merge: impl FnMut(&ShadowCopies, bool, Option<[u32; 4]>) -> Result<M, Error>,
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
        // The textures the command writes through, its bytes another way not, that merge back:
        // kept in a scratch of the encoder's, made at the first such command.
        let mut merged_kept = Vec::new();
        let mut scratches = Vec::new();
        for (i, shadow) in shadows.iter().enumerate() {
            if two_ways(shadows, i, uses) {
                kept.push((
                    Arc::clone(&shadow.copies),
                    merge(&shadow.copies, true, None)?,
                ));
            } else if merged_back(shadow) {
                merged_kept.push(Arc::clone(&shadow.copies));
                // The texels the command writes, where it bounds them and the copies are cut so.
                let writes = shadow.area.filter(|_| shadow.copies.cuts_back());
                let scratch = self
                    .entries
                    .iter()
                    .find(|e| e.copies.shadow_key == shadow.copies.shadow_key)
                    .and_then(|e| e.scratch.as_ref());
                let area = match scratch {
                    Some(scratch) if holds(scratch.kept_area(), writes) => continue,
                    Some(scratch) => bounding(scratch.kept_area(), writes),
                    None => writes,
                };
                scratches.push((
                    Arc::clone(&shadow.copies),
                    merge(&shadow.copies, false, area)?,
                ));
            }
        }
        // A texture kept in a scratch that does not keep what the command writes: merged before
        // its keep in the new one.
        let mut regrown = Vec::new();
        for (copies, scratch) in scratches {
            let i = self.entry(&copies);
            let entry = &mut self.entries[i];
            if entry.merge.is_some() && !entry.two_ways {
                regrown.push(copies.shadow_key);
            }
            entry.scratch = Some(scratch);
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
            let entry = &self.entries[i];
            let regrows = regrown.contains(&entry.copies.shadow_key);
            // Written since its last copy back with nothing kept (by a command that replaced
            // every texel), and kept below: copied back first, as the merge after the keep takes
            // only the words changed since it, which would leave those writes out.
            let unkept = entry.merge.is_none()
                && (merged_kept.iter()).any(|kept| kept.shadow_key == entry.copies.shadow_key);
            if touched || regrows || unkept || (entry.merge.is_some() && entry.two_ways) {
                self.copy_back(i, &mut copies);
            }
        }
        for shadow in shadows {
            // Made here, or for a keep's scratch above: not copied in yet, so its copy in is the
            // encoder's first.
            let i = self.entry(&shadow.copies);
            let two_ways = kept
                .iter()
                .any(|(copies, _)| copies.shadow_key == shadow.copies.shadow_key);
            let overwrites = shadow.overwrites && !two_ways;
            // The texels the command needs as the buffer holds them: those it reads, and those
            // it writes (which a copy back moves), where it bounds both and the copies are cut to
            // a rectangle; every texel where it writes through a shadow whose copies back move
            // every texel, or writes the bytes two ways, as the merge after it compares every
            // texel the keep took.
            let needs = match two_ways || !shadow.copies.cuts() {
                true => None,
                false if shadow.writes && !shadow.copies.cuts_back() => None,
                false if shadow.writes => bounding(shadow.loads, shadow.area),
                false => shadow.loads,
            };
            let entry = &self.entries[i];
            let beyond_held =
                entry.standing == Standing::Current && !overwrites && !holds(entry.held, needs);
            // The image holds the buffer's texels only inside its last copy in's rectangle: what
            // the encoder wrote through it goes back first, then the copy in below moves what
            // this command needs.
            if beyond_held && entry.dirty {
                self.copy_back(i, &mut copies);
            }
            let entry = &mut self.entries[i];
            let cause = match entry.standing {
                _ if beyond_held => Some(CopyCause::BeyondHeld),
                Standing::Current => None,
                Standing::Unread => Some(CopyCause::First),
                Standing::Lost(lost) => Some(CopyCause::Changed(lost)),
            };
            if let Some(cause) = cause {
                if !overwrites {
                    copies.push(ShadowCopy {
                        copies: Arc::clone(&shadow.copies),
                        step: Step::Into,
                        kept: shadow.kept,
                        area: needs,
                        cause: Some(cause),
                    });
                }
                entry.standing = Standing::Current;
                entry.held = needs;
                entry.stale = None;
            }
            // The texels writes within a rectangle changed since the copy in: those alone. A
            // command that reads only the texels inside a rectangle and writes none takes those of
            // them inside it alone, and the rest stay changed, for the next command that reads
            // them: the image holds the buffer's texels everywhere else it held them.
            if let Some(stale) = entry.stale {
                let area = match (shadow.writes, needs) {
                    (false, Some(reads)) => {
                        entry.stale = outside(stale, reads);
                        meet(stale, reads)
                    }
                    _ => {
                        entry.stale = None;
                        Some(stale)
                    }
                };
                if let (false, Some(area)) = (overwrites, area) {
                    copies.push(ShadowCopy {
                        copies: Arc::clone(&shadow.copies),
                        step: Step::Into,
                        kept: shadow.kept,
                        area: Some(area),
                        cause: Some(CopyCause::Bounded),
                    });
                }
            }
            // The command writes every texel before it reads any.
            if overwrites {
                entry.held = None;
            }
        }
        // Kept after its copy in, until the copy back after the command, or after a later one,
        // which is then a merge: a copy back would write the texels no command wrote with what the
        // copy in read, over what another writer may have written since.
        for shadow in merged_kept {
            let i = self.entry(&shadow);
            let entry = &mut self.entries[i];
            let (None, Some(scratch)) = (&entry.merge, &entry.scratch) else {
                continue;
            };
            let scratch = scratch.clone();
            entry.merge = Some(scratch.clone());
            copies.push(ShadowCopy {
                copies: shadow,
                step: Step::Keep(scratch),
                kept: false,
                area: None,
                cause: None,
            });
        }
        for (shadow, merge) in kept {
            let i = self.entry(&shadow);
            self.entries[i].merge = Some(merge.clone());
            self.entries[i].two_ways = true;
            copies.push(ShadowCopy {
                copies: shadow,
                step: Step::Keep(merge),
                kept: false,
                area: None,
                cause: None,
            });
        }
        Ok(copies)
    }

    /// Note a recorded command that used `shadows` and had `uses`, and the writes `within`
    /// (whose uses are not among `uses`): a write through a texture leaves its image to copy
    /// back; a write of the bytes another way leaves the images over them stale, there alone
    /// where it is one of `within` and the image's texels are laid out as its texture's (the
    /// module's rules).
    pub(super) fn after(
        &mut self,
        shadows: &[ShadowUse],
        uses: &[Use<ResourceKey>],
        within: &[Within],
    ) {
        for entry in &mut self.entries {
            // A texture's own uses are its image's, never its bytes'.
            let copies = Arc::clone(&entry.copies);
            let writes = |used: &Use<ResourceKey>| !used.write.is_empty() && copies.meets(used);
            if uses.iter().any(writes) {
                entry.lose(Lost::Written);
                continue;
            }
            for part in within.iter().filter(|part| part.uses.iter().any(writes)) {
                match &part.texels {
                    Some(texels) => entry.changed(texels, Some(part.area), Lost::Within),
                    None => entry.lose(Lost::Within),
                }
            }
        }
        for shadow in shadows.iter().filter(|shadow| shadow.writes) {
            let i = self.entry(&shadow.copies);
            // Writes Metal drops are not copied back, and the image no longer holds the bytes.
            let entry = &mut self.entries[i];
            match shadow.copies.drops_writes {
                true => entry.lose(Lost::Dropped),
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
