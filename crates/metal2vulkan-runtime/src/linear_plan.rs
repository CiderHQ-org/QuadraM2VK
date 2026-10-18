//! Textures over buffers: which pixel formats a buffer can back, where Metal lays a texture out in
//! its buffer, which of Metal's rules a texture over a buffer breaks, and when a linear Vulkan
//! image in the buffer's memory holds exactly Metal's layout.
//!
//! **Metal's rules** (probes 9 to 9f, on an Apple M3 Max, macOS 26.7, and an M4, macOS 27.0.1,
//! which agree; each case run plain and under API validation). The runtime refuses as `Invalid`
//! what Metal asserts on in either run and what its validation alone stops, and as `Nil` what
//! plain Metal answers nil for:
//!
//! - **Formats.** `minimumLinearTextureAlignmentForPixelFormat:` and
//!   `minimumTextureBufferAlignmentForPixelFormat:` are both 16 bytes for every format a buffer
//!   can back: every uncompressed color format, and `Unspecialized`. Compressed formats (the 4:2:2
//!   ones among them) and depth and stencil formats cannot be backed by a buffer, and Metal's
//!   query asserts on them ([`BufferTextureProblem::NotLinear`]). A plain run of
//!   `newTextureWithDescriptor:offset:bytesPerRow:` makes a 4:2:2 texture all the same; the
//!   runtime, which holds no 4:2:2 texture over a buffer, refuses it `Unsupported` (`Format`)
//!   there, not as Metal's refusal. An IOSurface plane of either 4:2:2 format is made (below).
//! - **Types.** `newTextureWithDescriptor:offset:bytesPerRow:` makes 2D textures and texture
//!   buffers; the SPI `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` also
//!   makes 2D arrays. Each has one mip level.
//! - **Options.** The texture's options are the buffer's, whatever the descriptor names (probe
//!   9d, every pair of six buffer and nine descriptor options): Metal's validation alone stops a
//!   storage or CPU cache mode other than the buffer's, or a hazard tracking mode other than the
//!   default and the buffer's, and a plain run takes the buffer's. An IOSurface plane takes its
//!   descriptor's instead (below).
//! - **Pitches.** `bytesPerRow` covers the packed row (`width` × the texel's bytes). A 2D
//!   texture's offset and `bytesPerRow` are multiples of 16 ([`LINEAR_ALIGNMENT`]); an array's
//!   offset is a multiple of 128 ([`ARRAY_ALIGNMENT`]) and its `bytesPerRow` of 16; a texture
//!   buffer's `bytesPerRow` is any length that covers the row. A texture buffer's offset is any:
//!   Metal's validation alone stops one off 16, and a plain run makes the texture, its functions
//!   reading and writing from the offset rounded down to 16 and its blits and the CPU at the
//!   offset (an M4, macOS 27.0.1; `buffer_texture.rs`), so the runtime does too. The SPI's texture
//!   buffer, unmeasured there, keeps the validated rule. An array's `bytesPerImage` is ignored:
//!   each slice starts `bytesPerRow` × `height`, rounded up to 128, after the one before. Metal's
//!   validation alone asserts that it is a multiple of 128; a plain run makes the array at any (16,
//!   32, 64, 80 and 192 measured, probe 9b), so the runtime does too.
//! - **Length.** Validation counts `offset + bytesPerRow × (height × slices − 1) + the packed row`
//!   against the buffer's length: for an array, as if its slices were not rounded. Where an
//!   array's last slice still runs past the length, Metal answers nil when the buffer is at least
//!   16 KiB and its rounded slices end past the length rounded up to 16 KiB (its
//!   `allocatedSize`), and otherwise makes the texture over bytes past the length. The runtime's
//!   buffers have no bytes there, so it refuses that texture as not implemented
//!   ([`BufferRefusal::TexelsPastLength`]).
//!
//! **IOSurface planes** (`Buffer::new_texture_with_iosurface_plane`, the stand-in for
//! `newTextureWithDescriptor:iosurface:plane:` over the caller's memory). Measured on the M3 Max
//! (macOS 26.7) and an M4 (macOS 27.0.1), which agree on every byte but the memory-dependent reads
//! and the overlapping writes below (one process a case, plain twice and
//! under API validation, which flags none of them):
//!
//! - **Made.** A surface without planes takes a `bytesPerRow` that is a multiple of 16 only: Metal
//!   asserts on any other, plain too ("IOSurface texture: bytesPerRow (4) must be aligned to 16
//!   bytes"). A plane of a planar surface (the consumer's form: one plane dictionary, or three for
//!   its '420v' backing, base and offset alike) takes any offset and any `bytesPerRow` that covers
//!   its row, a multiple of the texel's size or not. IOSurface makes no plane whose pitch is below
//!   its row.
//! - **Options** (an M4, macOS 27.0.1, flat and planar surfaces). The texture
//!   takes and reports its descriptor's storage, CPU cache and hazard tracking modes (its default
//!   hazard tracking mode resolved to tracked), whatever the surface's memory or the buffer
//!   standing for it: `Shared` and `Managed`, each CPU cache mode and each hazard tracking mode are
//!   made silently, and read and write the surface's bytes. Metal's validation asserts on `Private`
//!   and `Memoryless` ("IOSurface textures must use MTLStorageModeManaged or
//!   MTLStorageModeShared"); a plain run makes both, and a `Private` one's GPU and CPU read and
//!   write the surface's bytes as a `Shared` one's do, so the runtime makes it too. What a
//!   `Memoryless` one does is unmeasured, so the runtime refuses it as validation does
//!   ([`BufferTextureProblem::Options`]).
//! - **The CPU** (`getBytes`, `replaceRegion`) reads and writes the plane where the surface lays it
//!   out: from its offset, rows `bytesPerRow` apart.
//! - **The GPU** (a blit's reads and writes, a copy to or from another texture, a kernel's
//!   `read()`, `sample()` and `write()`, a render pass's store) reads and writes it from its offset
//!   rounded down to 16, rows `bytesPerRow` rounded down to 16 apart ([`iosurface_gpu_layout`]):
//!   `bytesPerRow` itself where it is a multiple of 16, else other bytes than the CPU's (a plane
//!   at offset 4 starts at byte 0, an NV12 chroma plane at 36 at byte 32).
//! - **Under 16 bytes a row** that pitch is 0. The GPU writes the rows 16 bytes apart, and reads
//!   the first row there, but its reads of every later row are memory-dependent: zeros in a fresh
//!   process, and after a write other bytes, neither the surface's nor the written ones (not the
//!   same on the M3 Max and the M4). The runtime's GPU reads those rows where its writes put them,
//!   a recorded near-equivalent (`Degradation::NarrowSurfaceRows`, [`Placed::narrow`]).
//! - **4:2:2 planes** (an M4 under macOS 27.0.1, plain twice and validated
//!   alike): a texture of either 4:2:2 format over a CoreVideo `2vuy` or `yuvs` surface (no planes,
//!   `bytesPerRow` 64 for 16 x 4) is made, in either format over either surface; `getBytes` reads
//!   the surface's rows, `replaceRegion` writes them leaving the bytes between untouched, and a
//!   blit into a buffer reads them, 2 bytes a texel, unchanged. The runtime holds it as the 4:2:2
//!   textures of its own are held, in the carrier (`chroma_pairs.rs`), 2 bytes a texel, under the
//!   rules above.
//! - **Rows that overlap.** A row of more than 16 bytes whose `bytesPerRow` rounded down to 16 is
//!   below it overlaps the next row on the GPU: a read sees the overlap, and where two rows'
//!   writes meet, one of them wins, by the order of the writes: the earlier row's in a blit, and
//!   in a kernel the later row's on the M4 and others on the M3 Max. The runtime refuses such a
//!   plane of more than one row (`NotImplemented`, `RuntimeGap::OverlappingSurfaceRows`;
//!   [`BufferRefusal::OverlappingRows`]).
//! - **Depth and stencil over a planar surface** (an M4 under macOS 27.0.1, the format scout's
//!   cells: TTG2's scout, `scout4-metal-vs.tsv`). Metal makes, samples or reads, and draws a
//!   `Depth16Unorm`, `Depth32Float`, `Depth32Float_Stencil8`, `Stencil8` or `X32_Stencil8` texture
//!   over a plane of a surface with planes; over a surface without planes it stops one, as every
//!   buffer's texture of a depth or stencil format ([`BufferTextureProblem::NotLinear`]). Over a
//!   plane the texture keeps one aspect's unit a texel (`copy_plan::surface_depth_unit`: its depth,
//!   else its stencil; the stencil of `Depth32Float_Stencil8` is not in the plane), laid out by the
//!   rules above as a colour texel's: its reads, samples and CPU access take the plane's bytes
//!   there. The texture is a shadow of those bytes (`shadow.rs`), whose copies move that aspect
//!   alone. A render pass stores it otherwise, in Metal's tiled layout (`texture/heap_tiled.rs`)
//!   from the plane's offset rounded down to 128: it renders through a tiled twin
//!   (`texture/surface_depth.rs`, which lists what stays the runtime's gap,
//!   `NotImplemented(RuntimeGap::SurfaceDepthStencil)`, with a blit of the stencil of
//!   `Depth32Float_Stencil8`).
//!
//! The runtime holds both layouts: the GPU's ([`Placed::gpu`]) is the texture's texels for every
//! command, the copies of its shadow, an aliasing image and their hazards; the CPU's
//! ([`Placed::cpu`]) is `replace_region`'s and `get_bytes`'s. Both lie in the buffer, else the
//! texture is refused: a plane past the buffer `Invalid` (`PastBuffer`), GPU texels past it
//! `NotImplemented` (`TexelsPastBufferLength`; Metal's lie in the surface's pages past the plane).
//! A surface without planes starts at a multiple of 16 in the buffer, as a surface's base does
//! (`OffsetAlignment` otherwise).
//!
//! **Aliasing**. A 2D texture or array over a buffer is a linear image bound to the buffer's memory
//! where the device's linear image has every feature its optimal image has for the texture's uses
//! (it lacks nothing, [`linear_lack`]), and the image the device made, at the pitch it chose
//! itself, sits exactly on Metal's layout and holds no byte but the texels ([`aliases`]). A
//! texture whose rows or slices lie with the buffer's bytes between them, which Metal leaves as the
//! program writes them, never is one, whatever its usage ([`holding`]): its image would hold those
//! bytes too, which Vulkan lets its layout transitions (its first move among them) and a render
//! pass's load and store operations write. Otherwise it is a shadow image the encoders copy from
//! and to the buffer (`shadow.rs`), whose copies address the buffer's texels alone, which is exact
//! for Metal's defined behaviour.

use super::chroma_pairs::is_four_two_two;
use super::copy_plan::block_extent;
use super::image_plan::{compatible, view_formats, ImageFacts, ImagePlan};
use super::metal_level::MetalLevel;
use super::texture_descriptor::TextureLayout;
use super::values::StorageMode;
use metal2vulkan::error::LinearNeed;
use metal2vulkan::metal::format::is_depth_stencil;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    Format, FormatFeatureFlags2, ImageCreateFlags, ImageUsageFlags, SampleCountFlags,
};

/// The alignment Metal gives every format a buffer can back: of a texture's offset and rows in
/// its buffer (`minimumLinearTextureAlignmentForPixelFormat:`), and of a texture buffer's offset
/// (`minimumTextureBufferAlignmentForPixelFormat:`).
pub(super) const LINEAR_ALIGNMENT: u64 = 16;

/// The alignment of a 2D array's offset in its buffer, and the multiple each of its slices is
/// rounded up to.
pub(super) const ARRAY_ALIGNMENT: u64 = 128;

/// The granularity of a Metal buffer of at least this length: its `allocatedSize` is its length
/// rounded up to it (a shorter buffer's is its length).
pub(super) const METAL_PAGE: u64 = 16384;

/// A texture over a buffer Metal stops.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferTextureProblem {
    /// A pixel format no buffer can back: compressed, depth or stencil, or no format.
    NotLinear {
        /// The descriptor's pixel format.
        pixel_format: PixelFormat,
    },
    /// A texture type the call does not make over a buffer.
    TextureType {
        /// The descriptor's texture type.
        texture_type: TextureType,
    },
    /// More than one mip level.
    MipmapLevelCount {
        /// The descriptor's level count.
        count: u32,
    },
    /// An IOSurface plane's `Memoryless` storage. Metal's validation asserts on it ("IOSurface
    /// textures must use MTLStorageModeManaged or MTLStorageModeShared"); a plain run makes the
    /// texture, but what its GPU and CPU do with it is unmeasured. A texture made through
    /// `newTextureWithDescriptor:offset:bytesPerRow:` takes the buffer's options whatever its
    /// descriptor names, and a plane its descriptor's other modes.
    Options,
    /// A `bytesPerRow` that does not cover the packed row of `row_bytes`.
    RowPitch {
        /// The `bytesPerRow` given.
        bytes_per_row: u64,
        /// A packed row's bytes.
        row_bytes: u64,
    },
    /// A `bytesPerRow` that is not a multiple of `alignment`.
    RowPitchAlignment {
        /// The `bytesPerRow` given.
        bytes_per_row: u64,
        /// The alignment it needs, in bytes.
        alignment: u64,
    },
    /// An offset that is not a multiple of `alignment`: a 2D texture's or array's, an IOSurface
    /// plane's, or the SPI's texture buffer's (`newTextureWithDescriptor:offset:bytesPerRow:`'s
    /// texture buffer takes any).
    OffsetAlignment {
        /// The offset given.
        offset: u64,
        /// The alignment it needs, in bytes.
        alignment: u64,
    },
    /// Validation's count of the texture's bytes, `counted`, past the buffer's `length`.
    PastBuffer {
        /// The bytes validation counts.
        counted: u64,
        /// The buffer's length.
        length: u64,
    },
    /// An array whose slices end past the buffer's `allocatedSize`: Metal answers nil (an
    /// [`Error::Nil`](crate::Error::Nil)).
    PastAllocation {
        /// Where the last slice ends, in bytes from the buffer's start.
        end: u64,
        /// The buffer's `allocatedSize`.
        allocation: u64,
    },
}

/// Whether a buffer can back a texture of `pf` (the module's rule), of a format the device's
/// Metal level describes: not `Invalid`, compressed, depth or stencil.
pub(super) fn is_linear(pf: PixelFormat) -> bool {
    use PixelFormat as P;
    let compressed = block_extent(pf).is_some() || matches!(pf, P::GBGR422 | P::BGRG422);
    pf != P::Invalid && !compressed && !is_depth_stencil(pf)
}

/// Whether a buffer can back a texture of `pf` ([`is_linear`]), or why not.
fn backed(pf: PixelFormat) -> Result<(), BufferTextureProblem> {
    if is_linear(pf) {
        Ok(())
    } else {
        Err(BufferTextureProblem::NotLinear { pixel_format: pf })
    }
}

/// `pf`'s alignment in a buffer ([`LINEAR_ALIGNMENT`]) on a device answering at `level`, or why
/// no buffer can back it: also a format the level describes none of
/// ([`MetalLevel::describes`]), whose alignment queries Metal asserts on, and one it describes as a
/// depth or a stencil format, `AnyDepth` and `AnyStencil` among them, which Vulkan has no format
/// for ("Linear textures do not support depth/stencil pixel formats").
pub(super) fn linear_alignment(
    pf: PixelFormat,
    level: MetalLevel,
) -> Result<u64, BufferTextureProblem> {
    let info = level.pixel_format_info(u64::from(pf.as_raw()));
    if !info.describes_a_format() || info.depth_or_stencil() {
        return Err(BufferTextureProblem::NotLinear { pixel_format: pf });
    }
    backed(pf).map(|()| LINEAR_ALIGNMENT)
}

/// The Metal call that makes a texture over a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferTextureCall {
    /// `newTextureWithDescriptor:offset:bytesPerRow:`: a 2D texture or a texture buffer.
    Texture,
    /// The SPI `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`, which makes
    /// 2D arrays too. Metal lays an array's slices out itself, whatever `bytesPerImage` is.
    Linear,
    /// `newTextureWithDescriptor:iosurface:plane:` over the caller's memory, of a plane of a
    /// planar surface (`planar`) or of a surface without planes (the module's rules).
    IoSurfacePlane { planar: bool },
}

/// A texture over a buffer as a call asks for it.
#[derive(Clone, Copy, Debug)]
pub(super) struct BufferTexture {
    /// The descriptor's checked layout.
    pub(super) layout: TextureLayout,
    /// The bytes of one texel of the pixel format.
    pub(super) texel_bytes: u64,
    /// The descriptor's storage mode, which only an IOSurface plane is held to.
    pub(super) storage: StorageMode,
    pub(super) offset: u64,
    pub(super) bytes_per_row: u64,
    pub(super) call: BufferTextureCall,
}

/// The buffer a texture is made over: its length.
#[derive(Clone, Copy, Debug)]
pub(super) struct BufferFacts {
    pub(super) length: u64,
}

/// Where a texture over a buffer lies in it, as Metal lays it out: `slices` of `rows` rows, each
/// row's `row_bytes` of texels starting `row_pitch` after the one above, and each slice
/// `slice_pitch` after the one before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BufferLayout {
    /// The texture's first byte in the buffer.
    pub(super) offset: u64,
    pub(super) row_pitch: u64,
    pub(super) slice_pitch: u64,
    pub(super) row_bytes: u64,
    pub(super) rows: u64,
    pub(super) slices: u64,
}

impl BufferLayout {
    /// One past the texture's last texel byte in the buffer.
    pub(super) fn end(&self) -> u64 {
        self.offset
            .saturating_add(self.slice_pitch.saturating_mul(self.slices - 1))
            .saturating_add(self.row_pitch.saturating_mul(self.rows - 1))
            .saturating_add(self.row_bytes)
    }
}

/// Where a texture over a buffer lies in it: its texels for the GPU's commands (`gpu`: copies,
/// views, aliases and their hazards) and for the CPU's `replaceRegion` and `getBytes` (`cpu`),
/// which differ only for an IOSurface plane (the module's rules).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Placed {
    pub(super) gpu: BufferLayout,
    pub(super) cpu: BufferLayout,
    /// An IOSurface plane of more than one row under 16 bytes a row, whose rows past the first
    /// Metal's GPU reads as memory-dependent bytes: the runtime's reads them where its writes put
    /// them (the module's rules).
    pub(super) narrow: bool,
}

impl Placed {
    /// One layout for the GPU and the CPU alike: every texture over a buffer but an IOSurface
    /// plane.
    fn same(layout: BufferLayout) -> Placed {
        Placed {
            gpu: layout,
            cpu: layout,
            narrow: false,
        }
    }
}

/// Where Metal's GPU reads and writes the texels of an IOSurface plane laid out as `cpu` for the
/// CPU (the module's rules): from its offset rounded down to 16, rows its pitch rounded down to 16
/// apart, or 16 apart where that is 0 (a plane under 16 bytes a row, [`Placed::narrow`]). A plane
/// of one row lies there whatever its pitch; refused ([`BufferRefusal::OverlappingRows`]) where
/// later rows would overlap earlier ones.
pub(super) fn iosurface_gpu_layout(cpu: &BufferLayout) -> Result<BufferLayout, BufferRefusal> {
    let down = |n: u64| n - n % LINEAR_ALIGNMENT;
    let pitch = down(cpu.row_pitch);
    if cpu.rows > 1 && pitch != 0 && pitch < cpu.row_bytes {
        return Err(BufferRefusal::OverlappingRows);
    }
    // Every texel size divides a multiple of 16, as the shadow's copies need.
    let row_pitch = pitch.max(
        cpu.row_bytes
            .checked_next_multiple_of(LINEAR_ALIGNMENT)
            .unwrap_or(u64::MAX),
    );
    Ok(BufferLayout {
        offset: down(cpu.offset),
        row_pitch,
        slice_pitch: row_pitch.saturating_mul(cpu.rows),
        ..*cpu
    })
}

/// Why a texture over a buffer is not made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferRefusal {
    /// What Metal asserts on or stops under validation (`Invalid`), or answers nil for (`Nil`:
    /// [`BufferTextureProblem::PastAllocation`]).
    Metal(BufferTextureProblem),
    /// An array whose texels run past the buffer's length, which Metal makes over bytes past
    /// the length and the runtime has no bytes for (`NotImplemented`).
    TexelsPastLength,
    /// An IOSurface plane whose rows overlap on Metal's GPU, where a write's bytes depend on the
    /// order it writes rows in (`NotImplemented`; the module's rules).
    OverlappingRows,
}

impl From<BufferTextureProblem> for BufferRefusal {
    fn from(problem: BufferTextureProblem) -> BufferRefusal {
        BufferRefusal::Metal(problem)
    }
}

/// Where `texture` lies in `buffer`, by the module's rules, or why it is not made.
pub(super) fn plan_buffer_texture(
    texture: &BufferTexture,
    buffer: &BufferFacts,
) -> Result<Placed, BufferRefusal> {
    use BufferTextureProblem as P;
    use TextureType as T;
    let layout = &texture.layout;
    // A surface's 4:2:2 plane is made (the module docs).
    let surface = matches!(texture.call, BufferTextureCall::IoSurfacePlane { .. });
    // A format the device's Metal level describes none of was stopped by the descriptor's check
    // before (`TextureDescriptor::check_made`).
    // Over a plane of a surface with planes Metal makes a depth or stencil texture, laid out as
    // a colour texture's in its one aspect's units (the module docs).
    let planar_depth = matches!(
        texture.call,
        BufferTextureCall::IoSurfacePlane { planar: true }
    ) && is_depth_stencil(layout.pixel_format);
    if !(surface && is_four_two_two(layout.pixel_format)) && !planar_depth {
        backed(layout.pixel_format)?;
    }
    let texture_type = layout.texture_type;
    let made = match texture.call {
        BufferTextureCall::Texture => matches!(texture_type, T::Type2D | T::TextureBuffer),
        BufferTextureCall::Linear => {
            matches!(texture_type, T::Type2D | T::TextureBuffer | T::Type2DArray)
        }
        BufferTextureCall::IoSurfacePlane { .. } => texture_type == T::Type2D,
    };
    if !made {
        return Err(P::TextureType { texture_type }.into());
    }
    if layout.levels != 1 {
        return Err(P::MipmapLevelCount {
            count: layout.levels,
        }
        .into());
    }
    if surface && texture.storage == StorageMode::Memoryless {
        return Err(P::Options.into());
    }
    let [width, height, _] = layout.extent.map(u64::from);
    let (offset, bytes_per_row) = (texture.offset, texture.bytes_per_row);
    let row_bytes = width.saturating_mul(texture.texel_bytes);
    if bytes_per_row < row_bytes {
        return Err(P::RowPitch {
            bytes_per_row,
            row_bytes,
        }
        .into());
    }
    if let BufferTextureCall::IoSurfacePlane { planar } = texture.call {
        let cpu = BufferLayout {
            offset,
            row_pitch: bytes_per_row,
            slice_pitch: bytes_per_row.saturating_mul(height),
            row_bytes,
            rows: height,
            slices: 1,
        };
        return plan_iosurface_plane(cpu, planar, buffer.length);
    }
    let array = texture_type == T::Type2DArray;
    let alignment = if array {
        ARRAY_ALIGNMENT
    } else {
        LINEAR_ALIGNMENT
    };
    let any_offset = texture_type == T::TextureBuffer && texture.call == BufferTextureCall::Texture;
    if !any_offset && !offset.is_multiple_of(alignment) {
        return Err(P::OffsetAlignment { offset, alignment }.into());
    }
    if texture_type != T::TextureBuffer && !bytes_per_row.is_multiple_of(LINEAR_ALIGNMENT) {
        return Err(P::RowPitchAlignment {
            bytes_per_row,
            alignment: LINEAR_ALIGNMENT,
        }
        .into());
    }
    let slices = u64::from(layout.layers);
    let image = bytes_per_row.saturating_mul(height);
    let slice_pitch = if array {
        image
            .checked_next_multiple_of(ARRAY_ALIGNMENT)
            .unwrap_or(u64::MAX)
    } else {
        image
    };
    let placed = BufferLayout {
        offset,
        row_pitch: bytes_per_row,
        slice_pitch,
        row_bytes,
        rows: height,
        slices,
    };
    let length = buffer.length;
    // Validation's count: the slices' rows as if contiguous.
    let counted = BufferLayout {
        slice_pitch: image,
        ..placed
    }
    .end();
    if counted > length {
        return Err(P::PastBuffer { counted, length }.into());
    }
    if placed.end() > length {
        let end = offset.saturating_add(slice_pitch.saturating_mul(slices));
        if length >= METAL_PAGE {
            let allocation = length
                .checked_next_multiple_of(METAL_PAGE)
                .unwrap_or(length);
            if end > allocation {
                return Err(P::PastAllocation { end, allocation }.into());
            }
        }
        return Err(BufferRefusal::TexelsPastLength);
    }
    Ok(Placed::same(placed))
}

/// An IOSurface plane laid out as `cpu` in a buffer of `length` bytes, of a planar surface or
/// not (`planar`), by the module's rules; its type, levels, options and row were checked as
/// every texture over a buffer's are.
fn plan_iosurface_plane(
    cpu: BufferLayout,
    planar: bool,
    length: u64,
) -> Result<Placed, BufferRefusal> {
    use BufferTextureProblem as P;
    let alignment = LINEAR_ALIGNMENT;
    if !planar && !cpu.row_pitch.is_multiple_of(alignment) {
        return Err(P::RowPitchAlignment {
            bytes_per_row: cpu.row_pitch,
            alignment,
        }
        .into());
    }
    if !planar && !cpu.offset.is_multiple_of(alignment) {
        return Err(P::OffsetAlignment {
            offset: cpu.offset,
            alignment,
        }
        .into());
    }
    if cpu.end() > length {
        return Err(P::PastBuffer {
            counted: cpu.end(),
            length,
        }
        .into());
    }
    let gpu = iosurface_gpu_layout(&cpu)?;
    if gpu.end() > length {
        return Err(BufferRefusal::TexelsPastLength);
    }
    Ok(Placed {
        gpu,
        cpu,
        narrow: cpu.rows > 1 && cpu.row_pitch < alignment,
    })
}

/// The features of an image's tiling each of its usages relies on: what the runtime's views,
/// samplers, attachments and copies of the texture may use.
const USAGE_FEATURES: [(ImageUsageFlags, FormatFeatureFlags2); 6] = [
    (
        ImageUsageFlags::SAMPLED,
        FormatFeatureFlags2::SAMPLED_IMAGE
            .union(FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR)
            .union(FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_MINMAX),
    ),
    (
        ImageUsageFlags::STORAGE,
        FormatFeatureFlags2::STORAGE_IMAGE
            .union(FormatFeatureFlags2::STORAGE_IMAGE_ATOMIC)
            .union(FormatFeatureFlags2::STORAGE_READ_WITHOUT_FORMAT)
            .union(FormatFeatureFlags2::STORAGE_WRITE_WITHOUT_FORMAT),
    ),
    (
        ImageUsageFlags::COLOR_ATTACHMENT,
        FormatFeatureFlags2::COLOR_ATTACHMENT.union(FormatFeatureFlags2::COLOR_ATTACHMENT_BLEND),
    ),
    (
        ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        FormatFeatureFlags2::DEPTH_STENCIL_ATTACHMENT,
    ),
    (
        ImageUsageFlags::TRANSFER_SRC,
        FormatFeatureFlags2::TRANSFER_SRC,
    ),
    (
        ImageUsageFlags::TRANSFER_DST,
        FormatFeatureFlags2::TRANSFER_DST,
    ),
];

/// The features of an image's tiling a texture of several mip levels relies on beyond its
/// usages': `generate_mipmaps` blits each level from the one above, linearly filtered
/// (`mipmap_plan.rs`).
const MIPMAP_FEATURES: FormatFeatureFlags2 = FormatFeatureFlags2::BLIT_SRC
    .union(FormatFeatureFlags2::BLIT_DST)
    .union(FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR);

/// What a linear image of a plan lacks of all its optimal image can do ([`linear_lack`]):
/// `features` of `format`'s linear tiling, which its optimal tiling has and the plan relies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LinearLack {
    pub(super) format: Format,
    pub(super) features: FormatFeatureFlags2,
}

/// What a linear image of `plan` lacks of all its optimal image can do for the plan's usages:
/// for the image's own format and each format it is viewed as ([`viewed_formats`]), the linear
/// tiling must have every feature of [`USAGE_FEATURES`] the optimal tiling has, and of
/// [`MIPMAP_FEATURES`] for a plan of several levels. A plan of extended usage (a usage its own
/// format lacks, which a view format has) is checked alike: in each format, the optimal tiling has
/// none of the features of a usage that format does not take, so only those it takes are asked of
/// its linear tiling. `None` where the linear image lacks nothing.
pub(super) fn linear_lack(plan: &ImagePlan, facts: &ImageFacts) -> Option<LinearLack> {
    tiling_lack(plan, facts, |format| facts.linear(format))
}

/// What an image of `plan` in another tiling than optimal, whose features in a format are
/// `tiling`'s answer, lacks of all its optimal image can do, as [`linear_lack`] asks it of linear
/// tiling: a `DRM_FORMAT_MOD_LINEAR` image's features are its modifier's (`dma_buf.rs`). `None`
/// where it lacks nothing.
pub(super) fn tiling_lack(
    plan: &ImagePlan,
    facts: &ImageFacts,
    tiling: impl Fn(Format) -> FormatFeatureFlags2,
) -> Option<LinearLack> {
    let needed = USAGE_FEATURES
        .iter()
        .filter(|(usage, _)| plan.usage.contains(*usage))
        .fold(FormatFeatureFlags2::empty(), |needed, &(_, features)| {
            needed.union(features)
        })
        .union(if plan.levels > 1 {
            MIPMAP_FEATURES
        } else {
            FormatFeatureFlags2::empty()
        });
    std::iter::once(plan.format)
        .chain(viewed_formats(plan, facts))
        .find_map(|format| {
            let (optimal, linear) = (facts.optimal(format), tiling(format));
            let features = optimal.intersection(needed).difference(linear);
            (!features.is_empty()).then_some(LinearLack { format, features })
        })
}

/// The formats a plan's image may be viewed as: its list, or, where its views may take any
/// format (a texture of `pixelFormatView` usage: `MUTABLE_FORMAT` without a list), each format
/// the runtime views a texture of a pixel format as for the plan's usages (`view_formats`) that
/// is compatible with the image's, of every pixel format.
pub(super) fn viewed_formats(plan: &ImagePlan, facts: &ImageFacts) -> Vec<Format> {
    if !plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT) || !plan.view_formats.is_empty() {
        return plan.view_formats.clone();
    }
    let formats: std::collections::BTreeSet<Format> = PixelFormat::ALL
        .iter()
        .flat_map(|&pf| view_formats(pf, plan.usage, facts.profile))
        .filter(|&format| format != plan.format && compatible(plan.format, format, true))
        .collect();
    formats.into_iter().collect()
}

/// Whether a linear image of `plan` fits `limits`, the device's image limits for it
/// (`vkGetPhysicalDeviceImageFormatProperties2` with linear tiling), or what it lacks: a device
/// that reports no such image at all reports every limit 0 ([`LinearNeed::Image`]).
pub(super) fn check_linear_limits(
    plan: &ImagePlan,
    limits: &ash::vk::ImageFormatProperties,
) -> Result<(), LinearNeed> {
    if limits.max_mip_levels == 0 {
        return Err(LinearNeed::Image);
    }
    let max = [
        limits.max_extent.width,
        limits.max_extent.height,
        limits.max_extent.depth,
    ];
    if let Some((&required, &available)) = plan.extent.iter().zip(max.iter()).find(|(w, h)| w > h) {
        return Err(LinearNeed::Extent {
            required,
            available,
        });
    }
    if plan.levels > limits.max_mip_levels {
        return Err(LinearNeed::MipLevels {
            required: plan.levels,
            available: limits.max_mip_levels,
        });
    }
    if plan.layers > limits.max_array_layers {
        return Err(LinearNeed::ArrayLayers {
            required: plan.layers,
            available: limits.max_array_layers,
        });
    }
    if !SampleCountFlags::from(limits.sample_counts).contains(plan.samples) {
        return Err(LinearNeed::SampleCount(plan.samples.as_raw()));
    }
    Ok(())
}

/// A linear image the device made for a texture over a buffer, as Vulkan reports it: its memory
/// requirements' alignment and size, and the layout of its first subresource
/// (`vkGetImageSubresourceLayout`), its bytes (`subresource`, `VkSubresourceLayout::size`) among
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LinearImage {
    pub(super) alignment: u64,
    pub(super) size: u64,
    pub(super) offset: u64,
    pub(super) subresource: u64,
    pub(super) row_pitch: u64,
    pub(super) array_pitch: u64,
}

/// Whether `image`, bound where `layout` starts in its buffer's memory (`memory_offset` in the
/// memory block), holds exactly Metal's texels in the `room` bytes from there to the end of the
/// buffer's memory: the binding is a multiple of the image's alignment, its texels start at the
/// binding, its row pitch is Metal's and so is its slice pitch where it has several slices, its
/// subresources hold no byte but the texture's texels ([`texels_alone`]), and its memory stays in
/// the buffer's. A driver may report a pitch other than the one asked of it (NVIDIA does for
/// explicit layouts it does not support), so the image's own report decides.
pub(super) fn aliases(
    layout: &BufferLayout,
    memory_offset: u64,
    room: u64,
    image: &LinearImage,
) -> bool {
    image.alignment != 0
        && memory_offset.is_multiple_of(image.alignment)
        && image.offset == 0
        && image.row_pitch == layout.row_pitch
        && (layout.slices == 1 || image.array_pitch == layout.slice_pitch)
        && texels_alone(layout, image.subresource, image.array_pitch)
        && image.size <= room
}

/// Whether an image over `layout`'s texels, for a texture of Metal `usage`, may hold them: where
/// its rows (and slices) lie one after another, or it has one row (one slice); else why not, where
/// they lie with bytes between them that are not texels (the buffer's, which Metal's writes of the
/// texture never reach), whatever its usage.
///
/// An image whose rows lie `row_pitch` apart holds those bytes in its subresource
/// (`VkSubresourceLayout`: "size includes any extra memory that is required based on rowPitch"),
/// and Vulkan lets two kinds of operation write them, against which no query of the device states
/// otherwise:
///
/// - Every image's layout transitions, its first move into `GENERAL` among them (the
///   Synchronization chapter's Image Layout Transitions): "Image layout transitions may perform
///   read and write accesses on all memory bound to the image subresource range", and "If the old
///   layout is VK_IMAGE_LAYOUT_UNDEFINED, the contents of that range may be discarded". No
///   operation lets an image be read before it leaves `UNDEFINED` or `PREINITIALIZED`, the layouts
///   it is made in, so a texture that only reads takes a transition too. It takes no other: every
///   command uses a runtime image in `GENERAL`, and its barriers are memory barriers, which
///   transition no layout (`tracker.rs`).
/// - A render pass's attachments (the Render Pass chapter's Load Operations, Store Operations and
///   Multisample Resolve Operations): "any writes performed by a load operation (as defined by its
///   access masks) to a given attachment may read and write back any memory locations within the
///   image subresource bound for that attachment", and a store and a resolve operation alike.
///
/// A copy and a shader's read or write of the image touch its texels alone (the Memory
/// Model: "A memory operation must not affect memory at a memory location not within its set of
/// memory locations"). But every image takes a first move, across which the host could keep the
/// bytes only by reading them before it and writing them back after (`texture/first_move.rs`'s
/// `Keep`), undoing a CPU write made to them meanwhile, where Metal keeps every one; and lavapipe's
/// shader writes read the bytes its 4-texel blocks reach past a row's texels and write them back, a
/// signaling NaN's bits quieted. So such a texture is a shadow, whose copies address its texels
/// alone.
pub(super) fn holding(layout: &BufferLayout, usage: TextureUsage) -> Result<(), String> {
    let (rows, slices) = apart(layout);
    let apart = if rows {
        format!(
            "its rows lie {} bytes apart, {} bytes of texels each",
            layout.row_pitch, layout.row_bytes
        )
    } else if slices {
        format!(
            "its slices lie {} bytes apart, {} rows {} bytes apart each",
            layout.slice_pitch, layout.rows, layout.row_pitch
        )
    } else {
        return Ok(());
    };
    let writers = if usage.contains(TextureUsage::RENDER_TARGET) {
        "its layout transitions read and write, and a render pass's load and store operations \
         write back"
    } else {
        "its layout transitions read and write, its first move from the layout it is made in \
         among them"
    };
    Err(format!(
        "{apart}: an image over them holds the bytes between them, which Vulkan lets {writers}"
    ))
}

/// Whether a linear image over `layout`'s texels, whose subresources take `size` bytes each from
/// the first texel of their slice, `array_pitch` apart, holds no byte but the texture's texels:
/// the texels lie with no byte between them ([`holding`]), and its last subresource ends at their
/// last byte or before. A driver may lay rows out past the texture's (lavapipe rounds the height
/// up to 4), where the buffer's later bytes lie (a surface's next plane). A byte of the image's
/// memory past every subresource is no texel's, and no write through the image reaches it:
/// Vulkan's Memory Aliasing has the bytes a write through an image affects be "those overlapped
/// by the image subresources that were written". A driver that reports one array's whole bytes
/// as its first subresource's has the array refused (the texture is a shadow).
pub(super) fn texels_alone(layout: &BufferLayout, size: u64, array_pitch: u64) -> bool {
    let texels = layout.end() - layout.offset;
    let last = array_pitch.saturating_mul(layout.slices - 1);
    apart(layout) == (false, false) && last.saturating_add(size) <= texels
}

/// Whether `layout`'s rows, and its slices, lie with bytes between them that are not texels.
fn apart(layout: &BufferLayout) -> (bool, bool) {
    let rows = layout.rows > 1 && layout.row_pitch != layout.row_bytes;
    let slices =
        layout.slices > 1 && layout.slice_pitch != layout.row_pitch.saturating_mul(layout.rows);
    (rows, slices)
}
