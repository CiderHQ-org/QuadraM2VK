//! Metal's private texture fills: `fillTexture:level:slice:region:bytes:length:`
//! ([`BlitCommandEncoder::fill_texture`]) and `fillTexture:level:slice:region:color:` with and
//! without `pixelFormat:` ([`BlitCommandEncoder::fill_texture_color`]), Apple SPI the runtime
//! takes as plain Metal answers it.
//!
//! **Metal's answer** (an Apple M4 under macOS 27.0.1, our own probes; every byte of every level,
//! slice and aspect read back after one fill over bytes of `0xEE`, 112 formats, one case a
//! process). A fill writes the pattern, cut at a unit's bytes or repeated to them, into every unit
//! of the region, and leaves every other byte:
//! - a **compressed** format's unit is a block: the region's origin and size count blocks, cut at
//!   the level's blocks (74 formats, PVRTC at square power-of-two extents);
//! - every other format's unit is a texel, and the region counts texels, cut at the level: an
//!   ordinary color or depth format's texel (31 formats); a 4:2:2 format's 2 bytes (`GBGR422`,
//!   `BGRG422`, at bytes `2x` and `2x + 1` of a row, which the 4:2:2 copies move); a stencil
//!   format's 1 byte (`Stencil8`, `X32Stencil8`); `Depth32Float_Stencil8`'s 4 depth bytes, its
//!   stencil kept;
//! - a **two-plane extended-range** texel (`RGB10A8_2P_XR10` and its sRGB twin) is its color word
//!   in plane 0, 4 bytes, its alpha kept, at Metal's blit rows (`⌊4 w / 16⌋ · 16` bytes apart):
//!   where a heap's bytes hold the planes (a one-level 2D texture), the fill writes those bytes
//!   there, rows overlapping included. A texture of its own memory (one level, 8 and 12 texels
//!   wide, a 2D array's slice too) is written at its rows when its width is a multiple of 4; at
//!   another width Metal's blit rows move texels between rows and into bytes between them, which
//!   the runtime does not hold, and of a texture of more than one level Metal's blits read back
//!   bytes no rule fits (both refused). A texture over a surface's planes is written at its GPU
//!   rows (plane 0's `bytesPerRow` rounded down to 16, the rows Metal's blits address it at) at any
//!   width, its alpha plane and every byte between the rows kept (`RGB10A8_2P_XR10` over 7 surfaces
//!   and the sRGB twin over one, 16, 33, 37, 61, 64 and the consumer's 384 texels wide, a region, a
//!   whole texture with a 3-byte pattern and a region past the level, every byte of each surface
//!   read back).
//!
//! A **color fill** packs the color in `pixelFormat:` (the texture's own format without it) by
//! the fill's own rules (`fill_value.rs`), which are not a `Clear` load action's, and fills that
//! pattern: the packed texel's bytes, repeated or cut at the texture's texel (12 pairs of formats,
//! 256 colors each: 128 colors packed in each of 70 formats into textures of 18 formats of 1 to 16
//! bytes). Packed in a format whose fills do not follow the color (compressed, 4:2:2, depth,
//! stencil, two-plane, padded 10-bit), Metal's bytes are one constant
//! for every color of a process, bytes it did not write for the color, which differ between
//! processes and machines. (Refused before.) The runtime fills zero bytes there, one constant for
//! every color, and announces it once (`Degradation::UnpackedFillColor`, one of Metal's answers).
//!
//! **Placement** (31 cases of a byte fill of `RGBA8Unorm` textures of each type,
//! every byte of every slice and level read back): a level past the texture's writes nothing; a 1D,
//! 2D or 3D texture ignores the slice, a cube takes the slice's face modulo 6, and an array (1D, 2D
//! or cube) writes nothing at a slice past its own; a 3D texture writes the region's depth images,
//! cut at the level's depth, and every other type one image, ignoring the region's `z` and depth. A
//! texture of another type (multisample, texture buffer) is unmeasured.
//!
//! **`fillBuffer:range:pattern4:`** ([`BlitCommandEncoder::fill_buffer_pattern4`]; 10 ranges of a
//! 256-byte buffer) writes the 4-byte pattern from the range's start, restarting it at each
//! multiple of 16 bytes of the buffer: byte `k` of the range takes the pattern's byte `k - start`
//! modulo 4 before the first such multiple, and byte `k` modulo 4 from it on.
//!
//! **How.** Each fill is copies the encoder already makes: a staging buffer of the pattern's units
//! (4 MiB at most, copied again for a larger region) and `copy_from_buffer_to_texture` into the
//! region's units of each image, so shadows, twins, carriers and two planes take their own paths;
//! a heap's two-plane texture, one copy of the color words into the heap's bytes. A pattern4
//! fill copies its bytes from a staging buffer. Refused (`AppleOnly::FillUnmeasured`): what the
//! probes did not cover.

use super::apple_only::AppleOnly;
use super::blit::{bytes_in, reads, writes, BlitCommandEncoder};
use super::chroma_pairs::{is_four_two_two, TEXEL_BYTES};
use super::clear_value::has_channels;
use super::copy_plan::{block_bytes, block_extent, level_extent, BlitOption};
use super::error::Error;
use super::fill_value::fill_texel;
use super::heap::Heap;
use super::render_values::ClearColor;
use super::replay::BlitCall;
use super::texture::Texture;
use super::two_planes;
use super::values::{Origin, Region, Size};
use super::{Buffer, Degradation};
use ash::vk;
use metal2vulkan::bind::PlaneRows;
use metal2vulkan::metal::format::{is_depth_stencil, padded_only_over_memory, ExtendedRange};
use metal2vulkan::metal::{PixelFormat, TextureType};
use std::ops::Range;

/// The most bytes a fill stages at once: a larger region copies them again.
const STAGING_BYTES: u64 = 4 << 20;

/// What a fill writes into one unit of a format (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FillUnit {
    /// A compressed format's block of `bytes`, `extent` texels: the region counts blocks.
    Blocks { bytes: u32, extent: [u32; 2] },
    /// A texel's `bytes`, which a copy with `option` writes: the region counts texels.
    Texels { bytes: u32, option: BlitOption },
    /// A two-plane extended-range texel's color word, its alpha kept.
    PlaneWords,
}

/// The fill unit of `pf`, `None` for a format the probes did not cover.
fn fill_unit(pf: PixelFormat) -> Option<FillUnit> {
    use PixelFormat as P;
    let texels = |bytes, option| Some(FillUnit::Texels { bytes, option });
    if two_planes::holds_planes(pf) {
        return Some(FillUnit::PlaneWords);
    }
    if let Some(extent) = block_extent(pf) {
        return Some(FillUnit::Blocks {
            bytes: block_bytes(pf)?,
            extent,
        });
    }
    if is_four_two_two(pf) {
        return texels(TEXEL_BYTES, BlitOption::None);
    }
    match pf {
        P::Stencil8 => texels(1, BlitOption::None),
        P::X32Stencil8 => texels(1, BlitOption::StencilFromDepthStencil),
        P::Depth32FloatStencil8 => texels(4, BlitOption::DepthFromDepthStencil),
        P::Depth16Unorm => texels(2, BlitOption::None),
        P::Depth32Float => texels(4, BlitOption::None),
        _ if has_channels(pf).is_some() => texels(block_bytes(pf)?, BlitOption::None),
        // A bfloat texel's words (color fills of single texels and of 4 x 1).
        _ if ExtendedRange::of(pf).is_some_and(|form| {
            form.bfloat()
                || matches!(
                    form,
                    ExtendedRange::Bgr10 { .. } | ExtendedRange::Bgra10 { .. }
                )
        }) =>
        {
            texels(block_bytes(pf)?, BlitOption::None)
        }
        _ => None,
    }
}

/// `pattern` repeated from its start and cut at `bytes`.
fn unit_of(pattern: &[u8], bytes: u32) -> Vec<u8> {
    pattern
        .iter()
        .copied()
        .cycle()
        .take(bytes as usize)
        .collect()
}

/// The units `from..from + count` of an axis of `units`, cut there.
fn cut(from: u64, count: u64, units: u64) -> Range<u64> {
    from.min(units)..from.saturating_add(count).min(units)
}

/// A refusal of what the probes did not cover.
fn unmeasured() -> Error {
    Error::AppleOnly(AppleOnly::FillUnmeasured)
}

/// Whether Metal's color fills of a color packed in `pf` write bytes that do not follow the color
/// (the module docs): a compressed, 4:2:2, depth, stencil, two-plane or padded 10-bit format.
fn unpacked(pf: PixelFormat) -> bool {
    block_extent(pf).is_some()
        || is_four_two_two(pf)
        || is_depth_stencil(pf)
        || two_planes::holds_planes(pf)
        || padded_only_over_memory(pf)
}

/// Where a fill of `region` at `level` and `slice` of `texture` writes, as plain Metal places it
/// (the module docs): the slice and the depth images, `None` where it writes nothing.
fn placement(
    texture: &Texture,
    level: u64,
    slice: u64,
    region: Region,
) -> Result<Option<(u64, Range<u64>)>, Error> {
    let record = texture.record();
    if matches!(
        texture.texture_type(),
        TextureType::Type2DMultisample
            | TextureType::Type2DMultisampleArray
            | TextureType::TextureBuffer
    ) {
        return Err(unmeasured());
    }
    let Some(level) = u32::try_from(level)
        .ok()
        .filter(|&level| level < record.level_count)
    else {
        return Ok(None);
    };
    let slices = u64::from(record.layer_count);
    let (slice, images) = match texture.texture_type() {
        TextureType::Type1D | TextureType::Type2D => (0, 0..1),
        TextureType::Type3D => {
            let depth = u64::from(level_extent(record, level)[2]);
            (0, cut(region.origin.z, region.size.depth, depth))
        }
        TextureType::Cube => (slice % 6, 0..1),
        TextureType::Type1DArray
        | TextureType::Type2DArray
        | TextureType::CubeArray
        | TextureType::Type2DMultisample
        | TextureType::Type2DMultisampleArray
        | TextureType::TextureBuffer => {
            if slice >= slices {
                return Ok(None);
            }
            (slice, 0..1)
        }
    };
    Ok((!images.is_empty()).then_some((slice, images)))
}

impl BlitCommandEncoder {
    /// `fillTexture:level:slice:region:bytes:length:` (Apple SPI): every unit of `region` of the
    /// texture's `level` and `slice` becomes `pattern`, repeated from its start and cut at the
    /// unit's bytes, as plain Metal writes it (`blit_fill.rs`). A compressed format's unit is a
    /// block, and the region's origin and size count blocks; every other format's is a texel (a
    /// 4:2:2 format's 2 bytes, a stencil format's 1, the depth of `Depth32Float_Stencil8`, a
    /// two-plane extended-range texel's color word). The region is cut at the level; an empty one
    /// writes nothing.
    ///
    /// Placed as plain Metal places it (the module docs): a level past the texture's, or a slice
    /// past an array's, writes nothing; a 1D, 2D or 3D texture ignores the slice and a cube takes
    /// it modulo 6; a 3D texture writes the region's depth images, every other type one image.
    ///
    /// Refused: a texture of another device (`Invalid`); `AppleOnly::FillUnmeasured`, what the
    /// probes did not cover: a multisample texture or a texture buffer, an empty pattern, a
    /// format outside the measured families (the module docs), and a two-plane texture other
    /// than a heap's one-level 2D one, one over a surface's planes, or a one-level texture of its
    /// own a multiple of 4 wide.
    pub fn fill_texture(
        &mut self,
        texture: &Texture,
        level: u64,
        slice: u64,
        region: Region,
        pattern: &[u8],
    ) -> Result<(), Error> {
        self.log(|log| {
            BlitCall::FillTexture(
                Box::new((texture.clone(), level, slice, region)),
                log.byte_span(pattern),
            )
        });
        self.fill("fill_texture", texture, (level, slice), region, pattern)
    }

    /// `fillTexture:level:slice:region:color:` (`pixel_format` `None`) and
    /// `fillTexture:level:slice:region:color:pixelFormat:` (Apple SPI): [`fill_texture`] of the
    /// texel `color` packs to in `pixel_format`, the texture's own format without it, as Metal's
    /// fills pack it (`fill_value.rs`), which is not how a `Clear` load action converts it: an sRGB
    /// channel takes the code whose decoded value is nearest, `RG11B10Float` rounds to nearest
    /// even, an integer channel's NaN is 0 or the extreme by its kind, and the extended-range
    /// formats are packed.
    ///
    /// A color packed in a format whose fills do not follow the color (compressed, 4:2:2, depth,
    /// stencil, two-plane, padded 10-bit) fills zero bytes, one constant for every color, as
    /// Metal's fills write one constant of their process's there, and the device announces
    /// `Degradation::UnpackedFillColor` once.
    ///
    /// Refused as [`fill_texture`] refuses; `AppleOnly::FillUnmeasured` for a color packed in a
    /// format whose fills the probes did not cover.
    ///
    /// [`fill_texture`]: BlitCommandEncoder::fill_texture
    pub fn fill_texture_color(
        &mut self,
        texture: &Texture,
        level: u64,
        slice: u64,
        region: Region,
        color: ClearColor,
        pixel_format: Option<PixelFormat>,
    ) -> Result<(), Error> {
        self.log(|_| {
            BlitCall::FillTextureColor(Box::new((
                texture.clone(),
                level,
                slice,
                region,
                color,
                pixel_format,
            )))
        });
        const CALL: &str = "fill_texture_color";
        self.core().check_device(CALL, texture.device())?;
        let packed_in = pixel_format.unwrap_or(texture.pixel_format());
        let channels = [color.red, color.green, color.blue, color.alpha];
        if let Some(pattern) = fill_texel(packed_in, channels) {
            return self.fill(CALL, texture, (level, slice), region, &pattern);
        }
        if !unpacked(packed_in) {
            return Err(unmeasured());
        }
        self.fill(CALL, texture, (level, slice), region, &[0])?;
        let format = texture.pixel_format();
        self.core()
            .device()
            .degrade(Degradation::UnpackedFillColor, || {
                (
                    format!("a color fill of a {format:?} texture, packed in {packed_in:?}"),
                    "Metal's fills there write one constant of its process's for every color; \
                     the runtime fills zero bytes"
                        .to_owned(),
                )
            });
        Ok(())
    }

    /// `fillBuffer:range:pattern4:` (Apple SPI): the bytes of `range` take `pattern`'s 4 bytes,
    /// its low byte first, as plain Metal writes them (the module docs): from the range's start,
    /// the pattern restarting at each multiple of 16 bytes of the buffer, so byte `k` of the buffer
    /// takes the pattern's byte `k` modulo 4 from the range's first such multiple on.
    ///
    /// An empty range fills nothing. Refused (`Invalid`): a range past the buffer, a buffer of
    /// another device.
    pub fn fill_buffer_pattern4(
        &mut self,
        buffer: &Buffer,
        range: Range<u64>,
        pattern: u32,
    ) -> Result<(), Error> {
        self.log(|_| BlitCall::FillBufferPattern4(buffer.clone(), range.clone(), pattern));
        const CALL: &str = "fill_buffer_pattern4";
        self.core().check_device(CALL, buffer.device())?;
        let size = range.end.saturating_sub(range.start);
        if size == 0 {
            return Ok(());
        }
        let range = bytes_in(CALL, range.start, size, buffer.length())?;
        let pattern = pattern.to_le_bytes();
        // The bytes before the first multiple of 16, the pattern from the range's start; then
        // the pattern from that multiple, in copies of at most `STAGING_BYTES` (a multiple of 4,
        // so each starts the pattern again).
        let restart = range.start.next_multiple_of(16).min(range.end);
        let head = restart - range.start;
        let body = range.end - restart;
        let chunk = body.min(STAGING_BYTES);
        let bytes: Vec<u8> = (0..head)
            .map(|k| pattern[(k % 4) as usize])
            .chain(pattern.iter().copied().cycle().take(chunk as usize))
            .collect();
        let staging = self.staged(&bytes)?;
        let (source, from) = staging.at(0);
        // Read as a blit's (`host_residency.rs`): where the range starts in the `VkBuffer`, and
        // each byte past it as far on.
        let (destination, origin) = self.core().blit_at(buffer, range.start)?;
        let mut regions = Vec::new();
        if head > 0 {
            regions.push(vk::BufferCopy {
                src_offset: from,
                dst_offset: origin,
                size: head,
            });
        }
        let mut at = restart;
        while at < range.end {
            let size = (range.end - at).min(chunk);
            regions.push(vk::BufferCopy {
                src_offset: from + head,
                dst_offset: origin + (at - range.start),
                size,
            });
            at += size;
        }
        let length = bytes.len() as u64;
        let uses = [reads(&staging, 0..length), writes(buffer, range)];
        self.command(&uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording; the staging buffer and the
            // buffer are live and retained until it completes; each region lies inside both,
            // and the buffer's regions are disjoint.
            unsafe { device.cmd_copy_buffer(raw, source, destination, &regions) };
        })?;
        self.core().retain(staging);
        self.core().retain(buffer.clone());
        Ok(())
    }

    /// The fill of `pattern` into `region` of `texture`'s `level` and `slice` (the module docs).
    fn fill(
        &mut self,
        call: &'static str,
        texture: &Texture,
        (level, slice): (u64, u64),
        region: Region,
        pattern: &[u8],
    ) -> Result<(), Error> {
        self.core().check_device(call, texture.device())?;
        self.see(texture);
        if pattern.is_empty() {
            return Err(unmeasured());
        }
        let Some((slice, images)) = placement(texture, level, slice, region)? else {
            return Ok(());
        };
        let Size {
            width,
            height,
            depth,
        } = region.size;
        if width == 0 || height == 0 || depth == 0 {
            return Ok(());
        }
        let unit = fill_unit(texture.pixel_format()).ok_or_else(unmeasured)?;
        let level_u32 = u32::try_from(level).map_err(|_| unmeasured())?;
        let [w, h, _] = level_extent(texture.record(), level_u32).map(u64::from);
        let (bytes, extent, option) = match unit {
            FillUnit::Blocks { bytes, extent } => (bytes, extent, BlitOption::None),
            FillUnit::Texels { bytes, option } => (bytes, [1, 1], option),
            FillUnit::PlaneWords => (4, [1, 1], BlitOption::None),
        };
        let [bw, bh] = extent.map(u64::from);
        let x = cut(region.origin.x, width, w.div_ceil(bw));
        let y = cut(region.origin.y, height, h.div_ceil(bh));
        if x.is_empty() || y.is_empty() {
            return Ok(());
        }
        let unit_bytes = unit_of(pattern, bytes);
        if unit == FillUnit::PlaneWords && !texture.over_buffer() {
            if let Some((heap, at)) = texture.heap_planes() {
                return self.plane_heap_fill(texture, (heap, at), w, (x, y), &unit_bytes);
            }
            // Metal's fill rows are its blits' (`⌊4 w / 16⌋ · 16` bytes apart), the texture's own
            // only at a width that is a multiple of 4; of a texture of more than one level, its
            // blits read back bytes no rule fits.
            if texture.heap().is_some() || texture.mipmap_level_count() > 1 || !w.is_multiple_of(4)
            {
                return Err(unmeasured());
            }
        }
        self.fill_rows(
            texture,
            (level, slice, images),
            option,
            (x, y),
            ([bw, bh], [w, h]),
            &unit_bytes,
        )
    }

    /// Copy `unit` into the units `x` by `y` of each depth image `images` of `texture`'s `level`
    /// and `slice`, `block` texels each (cut at the level's `size`), through `option`: copies
    /// from one staging buffer of whole rows of units, in rows of at most `STAGING_BYTES`.
    fn fill_rows(
        &mut self,
        texture: &Texture,
        (level, slice, images): (u64, u64, Range<u64>),
        option: BlitOption,
        (x, y): (Range<u64>, Range<u64>),
        (block, size): ([u64; 2], [u64; 2]),
        unit: &[u8],
    ) -> Result<(), Error> {
        let row_bytes = (x.end - x.start) * unit.len() as u64;
        let per_copy = (STAGING_BYTES / row_bytes).clamp(1, y.end - y.start);
        let staging = self.staging(unit, row_bytes * per_copy)?;
        let left = x.start * block[0];
        let width = (x.end * block[0]).min(size[0]) - left;
        for z in images {
            let mut row = y.start;
            while row < y.end {
                let end = (row + per_copy).min(y.end);
                let top = row * block[1];
                let height = (end * block[1]).min(size[1]) - top;
                self.unlogged(|this| {
                    this.copy_from_buffer_to_texture(
                        &staging,
                        0,
                        row_bytes,
                        row_bytes * (end - row),
                        Size::new(width, height, 1),
                        texture,
                        slice,
                        level,
                        Origin::new(left, top, z),
                        option,
                    )
                })?;
                row = end;
            }
        }
        Ok(())
    }

    /// Write `word` at each texel `x` by `y` of the one-level two-plane `texture` of `width`
    /// texels whose planes lie in `heap`'s bytes from byte `at`, at Metal's blit rows: the
    /// texels' color words, rows that meet merged, in one copy into the heap's bytes. Its alpha
    /// bytes are not written.
    fn plane_heap_fill(
        &mut self,
        texture: &Texture,
        (heap, at): (&Heap, u64),
        width: u64,
        (x, y): (Range<u64>, Range<u64>),
        word: &[u8],
    ) -> Result<(), Error> {
        let Some(bytes) = heap.bytes() else {
            return Err(unmeasured());
        };
        let width = u32::try_from(width).map_err(|_| unmeasured())?;
        let blit = u64::from(PlaneRows::of(width).blit);
        // Each row's words, from byte `at + row * blit + 4 x`; rows ascend, so a row that meets
        // the one before extends it.
        let mut spans: Vec<Range<u64>> = Vec::new();
        for row in y {
            let span = at + row * blit + 4 * x.start..at + row * blit + 4 * x.end;
            match spans.last_mut() {
                Some(last) if span.start <= last.end => last.end = last.end.max(span.end),
                _ => spans.push(span),
            }
        }
        let (Some(first), Some(last)) = (spans.first(), spans.last()) else {
            return Ok(());
        };
        let written = first.start..last.end;
        // Every span starts on a word (the texture's place and `blit` are multiples of 4) and is
        // whole words long, so the staging buffer's words land on them.
        let longest = spans.iter().map(|s| s.end - s.start).max().unwrap_or(0);
        let length = longest.min(STAGING_BYTES);
        let staging = self.staging(word, length)?;
        let (source, from) = staging.at(0);
        let (destination, _) = bytes.at(written.start);
        let mut regions = Vec::new();
        for span in &spans {
            let mut start = span.start;
            while start < span.end {
                let size = (span.end - start).min(length);
                regions.push(vk::BufferCopy {
                    src_offset: from,
                    dst_offset: bytes.at(start).1,
                    size,
                });
                start += size;
            }
        }
        let uses = [reads(&staging, 0..length), writes(bytes, written)];
        self.command(&uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording; the staging buffer and the
            // heap's bytes are live and retained until it completes; each region lies inside
            // both, and the heap's regions are disjoint.
            unsafe { device.cmd_copy_buffer(raw, source, destination, &regions) };
        })?;
        self.core().retain(staging);
        self.core().retain(texture.clone());
        Ok(())
    }

    /// A `Shared` buffer of `length` bytes of `unit` repeated, the source of a fill's copies.
    fn staging(&mut self, unit: &[u8], length: u64) -> Result<Buffer, Error> {
        let bytes: Vec<u8> = unit.iter().copied().cycle().take(length as usize).collect();
        self.staged(&bytes)
    }

    /// A `Shared` buffer of `bytes`, written through its mapping (seen by the device at the
    /// commit's submission).
    fn staged(&mut self, bytes: &[u8]) -> Result<Buffer, Error> {
        let buffer = self
            .core()
            .device()
            .write_combined_buffer(bytes.len() as u64)?;
        let Some(contents) = buffer.contents() else {
            return Err(Error::Vulkan(super::error::VkResultCode(
                vk::Result::ERROR_MEMORY_MAP_FAILED.as_raw(),
            )));
        };
        // SAFETY: the mapping covers the buffer's `bytes.len()` bytes, which nothing else uses
        // yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
        Ok(buffer)
    }
}
