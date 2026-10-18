//! `generateMipmapsForTexture:` as a chain of Vulkan blits.
//!
//! Metal's rules (API validation, measured on an Apple M3 Max and M4): the texture has more than
//! one level; its format is color-renderable (no compressed, depth or stencil format) and
//! filterable (no integer format); `RenderTarget` usage is not needed. On a texture view the levels
//! and slices are the view's: its first level is the source of the rest. Metal's plain call leaves
//! a texture of one level as it is, of any format, and a depth or stencil texture of any levels
//! ([`leaves_as_is`]), and so does the runtime. Its plain call mipmaps an integer format, which its
//! validation stops, by picking texels ([`plan_picks`]), and so does the runtime.
//!
//! Metal computes each level from the one above it by bilinear filtering at the scaled texel
//! centers, which is the Vulkan blit's linear filter, and its bytes are not the same on every
//! Apple GPU: an M3 Max and an M4 differ by at most one unit in the last place per component,
//! except where a box averages a format's largest and smallest values (`mipmap.rs`). A device's
//! blit chain gives Metal's answer only within that spread, which the runtime checks per pixel
//! format on the device before it records one (`mipmap.rs`). The formats Metal's filter is
//! computed for take no blit where the device runs its kernel (`mipmap_filter.rs`).

use super::image_plan::{holds_srgb_twin, ImageFacts, ImagePlan};
use super::view_plan::ViewRecord;
use metal2vulkan::device::FormatUsage;
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::format::{
    direct_format, is_depth_stencil, sampled_type, srgb_raw_twin, x_channel, ycbcr, ExtendedRange,
};
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::pipeline::SampledType;
use metal2vulkan::vk::{FormatFeatureFlags2, ImageUsageFlags};

/// A `generateMipmapsForTexture:` Metal's API validation stops, and the runtime refuses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MipmapProblem {
    /// A compressed format, which is not color-renderable: plain Metal writes zeros over the
    /// first bytes of the levels below the first and leaves the rest (an M4 under macOS 27.0.1), no
    /// defined answer. A depth or stencil texture, also not color-renderable, is left as it is.
    NotColorRenderable,
}

/// Why mipmaps cannot be generated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum MipmapRefusal {
    Invalid(MipmapProblem),
    Unsupported(Unsupported),
}

/// One blit of the chain: a level from the one above it, over the view's layers, with the
/// levels' extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct MipBlit {
    pub(super) src_level: u32,
    pub(super) dst_level: u32,
    pub(super) base_layer: u32,
    pub(super) layers: u32,
    pub(super) src_extent: [u32; 3],
    pub(super) dst_extent: [u32; 3],
}

/// A texture's mipmaps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum MipmapPlan {
    /// The blit chain of a filterable format: each level from the one above it (none where Metal
    /// leaves the texture as it is).
    Blits(Vec<MipBlit>),
    /// The texel picks of an integer format.
    Picks(PickPlan),
}

/// The mipmaps of an integer format as plain Metal makes them (an M4 under macOS 27.0.1, each
/// twice; its API validation stops them): texel `i` of a level is texel `floor((2i + 1) W / 2w)` of
/// the level above on each axis, `W` and `w` the two levels' sizes on it. A level halves its axes,
/// rounding down, so that is texel `2i + 1` where the level above is more than one texel on the
/// axis, else texel 0, and texel `i` of the `n`th level below the first is texel `s i + s - 1` of
/// the first, `s` 2 to the power of the levels above it more than one texel on the axis.
///
/// The runtime copies the first level of the view's layers into a scratch buffer, its rows the
/// level's width and its layers (or a 3D texture's slices) `image_height` rows apart, then copies
/// each later level's columns out of it in one buffer-to-image copy: a column's texels are
/// `s_y` rows apart and its slices or layers `s_z` slices (or one layer) apart, which a region's
/// `bufferRowLength` and `bufferImageHeight` say. Each texel is copied as its bytes, so the
/// answer is Metal's exactly on every device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PickPlan {
    /// The scratch buffer's bytes.
    pub(super) bytes: u64,
    /// The first level's rows a layer or slice in the scratch buffer (0: its height).
    pub(super) image_height: u32,
    /// The copies into the levels below the first.
    pub(super) picks: Vec<Pick>,
}

/// One region of the copy out of the scratch buffer: a column of a level (or the texels of one
/// of its rows, where the rows are too far apart for one region) over the view's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Pick {
    /// The image's level.
    pub(super) level: u32,
    /// The region's origin in the level.
    pub(super) origin: [u32; 3],
    /// The region's extent: one texel wide.
    pub(super) extent: [u32; 3],
    pub(super) buffer_offset: u64,
    pub(super) row_length: u32,
    pub(super) image_height: u32,
}

/// Whether plain Metal mipmaps `pf` by picking texels ([`PickPlan`]): a color format a shader
/// reads as integers (not an extended-range format, whose floats an integer image holds raw).
pub(super) fn picks(pf: PixelFormat) -> bool {
    !is_depth_stencil(pf)
        && matches!(
            sampled_type(pf),
            Some(SampledType::Uint | SampledType::Sint | SampledType::Uint64)
        )
}

/// Vulkan's bound on a buffer-image copy's `bufferRowLength` in bytes
/// (`VUID-vkCmdCopyBufferToImage-bufferRowLength-00203`).
const MAX_ROW_BYTES: u64 = (1 << 31) - 1;

/// The texel picks of the mipmaps of a texture presenting `record` whose image holds
/// `texel_bytes` a texel ([`PickPlan`]). None where a size does not fit Vulkan's copy fields.
pub(super) fn plan_picks(record: &ViewRecord, texel_bytes: u32) -> Option<PickPlan> {
    let volume = record.texture_type == TextureType::Type3D;
    let extent = |level: u32| {
        let mut e = record.extent().map(|axis| (axis >> level.min(31)).max(1));
        if !volume {
            e[2] = 1;
        }
        e
    };
    // Per axis, the levels above level `n` (of the view) more than one texel on it.
    let steps = |n: u32, axis: usize| (0..n).filter(|&k| extent(k)[axis] > 1).count() as u32;
    let levels = record.level_count;
    let layers = record.layer_count;
    let [width, height, depth] = extent(0).map(u64::from);
    let texel = u64::from(texel_bytes);
    // The scratch's rows a layer or slice: a multiple of every level's row step, so that each
    // region's layer or slice step is a whole number of its rows.
    let stacked = volume || layers > 1;
    let rows = if stacked {
        height.next_multiple_of(1 << steps(levels - 1, 1))
    } else {
        height
    };
    let bytes = width
        .checked_mul(rows)?
        .checked_mul(depth)?
        .checked_mul(u64::from(layers))?
        .checked_mul(texel)?;
    let mut picks = Vec::new();
    for n in 1..levels {
        let [sx, sy, sz] = [0, 1, 2].map(|axis| 1u64 << steps(n, axis));
        let [w, h, d] = extent(n);
        let row_length = sy * width;
        let level = record.base_level + n;
        let start = ((sz - 1) * rows + sy - 1) * width;
        for i in 0..w {
            let column = start + sx * u64::from(i) + sx - 1;
            if row_length * texel <= MAX_ROW_BYTES {
                let image_height = if volume {
                    sz * rows / sy
                } else if stacked {
                    rows / sy
                } else {
                    0
                };
                picks.push(Pick {
                    level,
                    origin: [i, 0, 0],
                    extent: [1, h, d],
                    buffer_offset: column * texel,
                    row_length: u32::try_from(row_length).ok()?,
                    image_height: u32::try_from(image_height).ok()?,
                });
                continue;
            }
            // Rows too far apart for one region's `bufferRowLength`: a region a row, its
            // layers or slices `width * image_height` texels apart.
            let image_height = if volume { sz * rows } else { rows };
            for j in 0..h {
                picks.push(Pick {
                    level,
                    origin: [i, j, 0],
                    extent: [1, 1, d],
                    buffer_offset: (column + u64::from(j) * row_length) * texel,
                    row_length: u32::try_from(width).ok()?,
                    image_height: u32::try_from(image_height).ok()?,
                });
            }
        }
    }
    Some(PickPlan {
        bytes,
        image_height: if stacked {
            u32::try_from(rows).ok()?
        } else {
            0
        },
        picks,
    })
}

/// Whether no blit gives `generate_mipmaps` of `pf`: an extended-range format, whose image holds
/// the raw codes a shader decodes. The runtime computes its mipmaps with Metal's filter
/// (`mipmap_filter.rs`; Metal's API validation takes the call) or refuses them
/// (`NotImplemented(MipmapGeneration)`). Also a YCbCr format (`metal::format::ycbcr`), whose
/// texture Metal makes only over a surface's planes, of one level, which [`leaves_as_is`]; Metal's
/// answer for more levels is not measured. And, until their levels' X fields are rewritten as Metal
/// stores them, the RGBX formats (`metal::format::x_channel`).
pub(super) fn not_implemented(pf: PixelFormat) -> bool {
    ExtendedRange::of(pf).is_some() || ycbcr(pf) || x_channel(pf)
}

/// Whether Metal's plain `generateMipmapsForTexture:` leaves a texture presenting `record` as it
/// is (an M4 under macOS 27.0.1; its validation stops both): a texture of one
/// level, of any format (`mip_single_rgba8u`, `mip_single_d32`, `mip_single_bc1`), and a depth or
/// stencil texture of any levels (`mip_d16_3`, `mip_d32_3`, `mip_s8_3`, `mip_d32s8_3`), an
/// extended-range format's among them.
pub(super) fn leaves_as_is(record: &ViewRecord) -> bool {
    record.level_count <= 1 || is_depth_stencil(record.pixel_format)
}

/// The runtime's refusal of mipmaps of `pf` whatever the texture: a format that is not
/// color-renderable (depth and stencil formats among them, which the runtime leaves as they are,
/// [`leaves_as_is`]). None for a format [`not_implemented`] refuses, whose rules are not
/// measured. (An integer format, which Metal's API validation also stops, is picked:
/// [`picks`].)
pub(super) fn format_problem(pf: PixelFormat) -> Option<MipmapProblem> {
    if not_implemented(pf) {
        return None;
    }
    let block = direct_format(pf).and_then(|f| f.block());
    let color = block.is_some_and(|b| {
        !is_depth_stencil(pf) && b.color.is_some() && super::copy_plan::block_extent(pf).is_none()
    });
    (!color).then_some(MipmapProblem::NotColorRenderable)
}

/// The mipmaps of a texture presenting `record` whose image is `image`, on a device with
/// `facts`: no blit for a texture Metal leaves as it is ([`leaves_as_is`]), the texel picks of an
/// integer format ([`plan_picks`]), else the blit chain.
///
/// Refusals: an extended-range format, which no blit filters (`NotImplemented(MipmapGeneration)`,
/// [`not_implemented`]; the caller computes it with Metal's filter where it can); the runtime's (`MipmapProblem`); an 8-bit sRGB texture held as its UNORM
/// twin (`image_plan::holds_srgb_twin`), whose blit would average the codes
/// (`Unsupported{MipmapGeneration}` needing the sRGB format's `SampledImageFilterLinear`); the
/// image format's lack of linear filtering (`Unsupported{MipmapGeneration}` needing the format's
/// `SampledImageFilterLinear`), or of blits or transfers (`NotImplemented(MipmapGeneration)`: a
/// compute path would not need them); and, for picks, the image's lack of transfers or a texel
/// block of more than one texel or of other bytes than the view's, or sizes past Vulkan's copy
/// fields (`NotImplemented(MipmapGeneration)`).
pub(super) fn plan_mipmaps(
    record: &ViewRecord,
    image: &ImagePlan,
    facts: &ImageFacts<'_>,
) -> Result<MipmapPlan, MipmapRefusal> {
    if leaves_as_is(record) {
        return Ok(MipmapPlan::Blits(Vec::new()));
    }
    if not_implemented(record.pixel_format) {
        return Err(MipmapRefusal::Unsupported(Unsupported::not_implemented(
            Feature::MipmapGeneration,
        )));
    }
    if let Some(problem) = format_problem(record.pixel_format) {
        return Err(MipmapRefusal::Invalid(problem));
    }
    let gap =
        || MipmapRefusal::Unsupported(Unsupported::not_implemented(Feature::MipmapGeneration));
    // An sRGB texture held as its UNORM twin: a blit would average the codes, which Metal's filter
    // does not; it needs the sRGB format's own filter.
    if holds_srgb_twin(image) && srgb_raw_twin(record.pixel_format).is_some() {
        let format = direct_format(record.pixel_format).ok_or_else(gap)?;
        return Err(MipmapRefusal::Unsupported(Unsupported::needs(
            Feature::MipmapGeneration,
            Capability::Format {
                format,
                usage: FormatUsage::SampledImageFilterLinear,
            },
        )));
    }
    let transfers = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    let features = facts.optimal(image.format);
    if picks(record.pixel_format) {
        let copies = FormatFeatureFlags2::TRANSFER_SRC | FormatFeatureFlags2::TRANSFER_DST;
        let texel = image
            .format
            .block()
            .filter(|b| b.texels == 1)
            .map(|b| b.size);
        let view = direct_format(record.pixel_format)
            .and_then(|f| f.block())
            .map(|b| b.size);
        if !features.contains(copies) || !image.usage.contains(transfers) || texel != view {
            return Err(gap());
        }
        let texel = texel.ok_or_else(gap)?;
        return plan_picks(record, texel)
            .map(MipmapPlan::Picks)
            .ok_or_else(gap);
    }
    if !features.contains(FormatFeatureFlags2::SAMPLED_IMAGE_FILTER_LINEAR) {
        return Err(MipmapRefusal::Unsupported(Unsupported::needs(
            Feature::MipmapGeneration,
            Capability::Format {
                format: image.format,
                usage: FormatUsage::SampledImageFilterLinear,
            },
        )));
    }
    if !features.contains(FormatFeatureFlags2::BLIT_SRC | FormatFeatureFlags2::BLIT_DST)
        || !image.usage.contains(transfers)
    {
        return Err(gap());
    }
    let extent = |level: u32| {
        let mut e = record
            .image_extent
            .map(|axis| (axis >> level.min(31)).max(1));
        if record.texture_type != TextureType::Type3D {
            e[2] = 1;
        }
        e
    };
    let blits = (record.base_level + 1..record.base_level + record.level_count)
        .map(|level| MipBlit {
            src_level: level - 1,
            dst_level: level,
            base_layer: record.base_layer,
            layers: record.layer_count,
            src_extent: extent(level - 1),
            dst_extent: extent(level),
        })
        .collect();
    Ok(MipmapPlan::Blits(blits))
}
