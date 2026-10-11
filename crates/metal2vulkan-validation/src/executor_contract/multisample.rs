//! Multisampled render passes: the one contract by which the Metal oracle and the engine
//! candidate fill and read a pass's attachments when the case declares `raster_sample_count`.
//!
//! - **Layout.** An attachment of `samples` samples holds, for each pixel in row-major order, its
//!   samples in order: byte `((y * width + x) * samples + s) * texel` starts sample `s` of pixel
//!   `(x, y)`, the layout a multisampled texture literal already uses. One sample is the plain
//!   pixel layout, so a single-sample attachment's bytes mean what they always meant.
//! - **Filling.** Each executor's per-sample initializer (the one that fills a multisampled input
//!   texture) writes every sample of an attachment before the case's pass loads it. A stencil
//!   aspect is the exception: neither executor has a per-sample stencil initializer (a fragment
//!   writes no stencil value of its own without `VK_EXT_shader_stencil_export`), so its samples
//!   must all hold one value ([`uniform_stencil`]), and each executor clears it to that value.
//! - **Reading.** No resolve and no copy: Metal copies no multisampled texture to a buffer, and
//!   Vulkan's `vkCmdCopyImageToBuffer` takes no multisampled image. A compute pass reads every
//!   sample of the attachment through an unsigned-integer view of its texel size
//!   ([`ReadbackView`]), so each texel's bits cross unconverted, through a depth view whose
//!   `f32` is bit-cast, or through a stencil view whose unsigned value is the byte, and writes it
//!   as whole 32-bit words in the layout above ([`unpack_words`]).
//! - **One sample is untouched.** A case without `raster_sample_count` is uploaded and read back
//!   as before by both executors ([`AttachmentTransfer::Copy`]): no new command reaches its pass.

use crate::case::{AuthoredCase, TextureFormat};

/// How an executor fills a render pass's attachments before the case's draw and reads them after.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachmentTransfer {
    /// One sample: uploaded by a copy and read back by a copy, the path every case once drew
    /// through.
    Copy,
    /// More than one sample: filled by the per-sample initializer and read back sample by sample
    /// by a compute pass, never resolved.
    PerSample { samples: u32 },
}

impl AttachmentTransfer {
    /// The transfer of a pass of `samples` samples.
    pub fn of(samples: u32) -> AttachmentTransfer {
        match samples {
            0 | 1 => AttachmentTransfer::Copy,
            samples => AttachmentTransfer::PerSample { samples },
        }
    }
}

/// How `case`'s render pass moves its attachments' bytes.
pub fn attachment_transfer(case: &AuthoredCase) -> AttachmentTransfer {
    AttachmentTransfer::of(case.raster_samples())
}

/// The unsigned-integer view of a color attachment's texel size that a per-sample readback reads
/// through: one 32-bit word per texel up to 4 bytes (the texel zero-extended), two for 8 bytes,
/// four for 16. Each API views a color format as another of its size without conversion (Metal's
/// `newTextureViewWithPixelFormat:`, a Vulkan view of a `MUTABLE_FORMAT` image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadbackView {
    R8Uint,
    R16Uint,
    R32Uint,
    Rg32Uint,
    Rgba32Uint,
}

impl ReadbackView {
    /// The words one texel occupies in the readback.
    pub fn words(self) -> usize {
        match self {
            Self::R8Uint | Self::R16Uint | Self::R32Uint => 1,
            Self::Rg32Uint => 2,
            Self::Rgba32Uint => 4,
        }
    }

    /// The bytes one texel holds.
    pub fn texel_bytes(self) -> usize {
        match self {
            Self::R8Uint => 1,
            Self::R16Uint => 2,
            Self::R32Uint => 4,
            Self::Rg32Uint => 8,
            Self::Rgba32Uint => 16,
        }
    }
}

/// How a per-sample readback reads an attachment of `format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleRead {
    /// A color attachment, through the unsigned-integer view of its texel size.
    Color(ReadbackView),
    /// A `Depth32Float` attachment: each sample's `f32`, bit-cast to one word.
    Depth,
    /// An 8-bit stencil aspect: each sample's unsigned value, one word.
    Stencil,
}

impl SampleRead {
    /// The words one texel occupies in the readback.
    pub fn words(self) -> usize {
        match self {
            Self::Color(view) => view.words(),
            Self::Depth | Self::Stencil => 1,
        }
    }

    /// The bytes one texel holds.
    pub fn texel_bytes(self) -> usize {
        match self {
            Self::Color(view) => view.texel_bytes(),
            Self::Depth => 4,
            Self::Stencil => 1,
        }
    }
}

/// How an attachment of `format` is read sample by sample: `Depth32Float` as depth, every other
/// attachment format through the unsigned-integer view of its texel size. A sampled-only format
/// is no attachment's.
pub fn sample_read(format: TextureFormat) -> Result<SampleRead, String> {
    if format.is_depth() {
        return Ok(SampleRead::Depth);
    }
    if format.sampled_only() {
        return Err(format!(
            "{format:?} is sampled-only, which no attachment holds"
        ));
    }
    Ok(SampleRead::Color(match format.bytes_per_pixel() {
        1 => ReadbackView::R8Uint,
        2 => ReadbackView::R16Uint,
        4 => ReadbackView::R32Uint,
        8 => ReadbackView::Rg32Uint,
        16 => ReadbackView::Rgba32Uint,
        other => {
            return Err(format!(
                "{format:?} has {other}-byte texels, which no view reads"
            ))
        }
    }))
}

/// The one value every sample of a multisampled stencil aspect's `bytes` holds, which each
/// executor clears the aspect to (see the module docs); samples of different values are refused.
pub fn uniform_stencil(bytes: &[u8]) -> Result<u8, String> {
    let (&value, rest) = bytes
        .split_first()
        .ok_or("the multisampled stencil aspect holds no samples")?;
    if rest.iter().any(|&sample| sample != value) {
        return Err(
            "a multisampled stencil aspect whose samples differ: no executor initializes a \
             stencil attachment sample by sample, and one of a single value is cleared to it"
                .into(),
        );
    }
    Ok(value)
}

/// The bytes of the readback's 32-bit little-endian `words` (every sample of a `width` by
/// `height` attachment of `samples` samples, read as `read`) in the attachment layout: each
/// texel's low [`SampleRead::texel_bytes`] bytes of its words, in order.
pub fn unpack_words(
    words: &[u8],
    [width, height]: [u32; 2],
    samples: u32,
    read: SampleRead,
) -> Result<Vec<u8>, String> {
    let texels = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(samples as usize))
        .ok_or("the readback's texel count overflows")?;
    let stride = read.words() * 4;
    let texel = read.texel_bytes();
    if words.len() != texels * stride {
        return Err(format!(
            "the readback holds {} bytes, not {texels} texels of {stride} bytes",
            words.len()
        ));
    }
    Ok(words
        .chunks_exact(stride)
        .flat_map(|chunk| &chunk[..texel])
        .copied()
        .collect())
}

/// The bytes a readback of `samples` samples of a `width` by `height` attachment read as `read`
/// writes: whole words per texel.
pub fn readback_bytes(
    [width, height]: [u32; 2],
    samples: u32,
    read: SampleRead,
) -> Result<usize, String> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(samples as usize))
        .and_then(|texels| texels.checked_mul(read.words() * 4))
        .ok_or_else(|| "the readback's size overflows".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every color format is read through the unsigned-integer view of its texel size, and
    /// `Depth32Float` as depth: the view each executor makes of the same attachment.
    #[test]
    fn every_attachment_format_has_one_readback_of_its_texel_size() {
        use TextureFormat as F;
        for format in [
            F::R8Unorm,
            F::Rgba8Unorm,
            F::Rgba8Uint,
            F::Rgba8Sint,
            F::R16Float,
            F::R16Uint,
            F::Rg16Float,
            F::Rg32Float,
            F::Rgba16Float,
            F::Rgba16Uint,
            F::Rgba16Sint,
            F::R32Uint,
            F::R32Sint,
            F::R32Float,
            F::Rgba32Uint,
            F::Rgba32Sint,
            F::Rgba32Float,
            F::Depth32Float,
        ] {
            let read = sample_read(format).unwrap();
            assert_eq!(read.texel_bytes(), format.bytes_per_pixel(), "{format:?}");
            assert!(read.words() * 4 >= read.texel_bytes(), "{format:?}");
            assert_eq!(
                matches!(read, SampleRead::Depth),
                format == F::Depth32Float,
                "{format:?}"
            );
        }
    }

    /// The words of a readback unpack to the attachment layout: each texel's low bytes, pixel
    /// by pixel, sample by sample.
    #[test]
    fn readback_words_unpack_to_the_sample_minor_attachment_layout() {
        // A 2x1 attachment of two samples of one-byte texels: (x, s) holds 0x10 * x + s.
        let words = [0x00u32, 0x01, 0x10, 0x11]
            .iter()
            .flat_map(|word| (word | 0xabcd_0000).to_le_bytes())
            .collect::<Vec<_>>();
        let read = SampleRead::Color(ReadbackView::R8Uint);
        assert_eq!(readback_bytes([2, 1], 2, read).unwrap(), words.len());
        assert_eq!(
            unpack_words(&words, [2, 1], 2, read).unwrap(),
            [0x00, 0x01, 0x10, 0x11]
        );
        // Eight-byte texels keep both words.
        let read = SampleRead::Color(ReadbackView::Rg32Uint);
        let words = (0u8..16).collect::<Vec<_>>();
        assert_eq!(unpack_words(&words, [1, 1], 2, read).unwrap(), words);
        // A readback of another size is refused, never truncated.
        assert!(unpack_words(&words, [1, 1], 4, read).is_err());
        // A stencil sample is the low byte of its word.
        let words = [0x7fu32, 0x80]
            .iter()
            .flat_map(|word| (word | 0xabcd_0000).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            unpack_words(&words, [1, 1], 2, SampleRead::Stencil).unwrap(),
            [0x7f, 0x80]
        );
    }

    /// A multisampled stencil aspect is cleared to its one value; samples of different values,
    /// which no executor initializes, are refused.
    #[test]
    fn a_multisampled_stencil_holds_one_value() {
        assert_eq!(uniform_stencil(&[0x5a; 8]), Ok(0x5a));
        assert!(uniform_stencil(&[0x5a, 0x5a, 0x5b, 0x5a])
            .unwrap_err()
            .contains("samples differ"));
        assert!(uniform_stencil(&[]).is_err());
    }
}
