//! Two-plane YCbCr textures over an IOSurface's two planes: `YCBCR8_420_2P` (format 500), the
//! texture CoreVideo's `420v` and `420f` surfaces are sampled through, `RGB8_420_2P` (540), the
//! 4:2:2 and 4:4:4 formats `YCBCR8_422_2P`, `YCBCR8_444_2P`, `RGB8_422_2P` and `RGB8_444_2P` (502,
//! 503, 541, 542) over the 8-bit surfaces, `YCBCR10_420_2P` and `RGB10_420_2P` (505 and 543),
//! `YCBCR10_422_2P`, `YCBCR10_444_2P`, `RGB10_422_2P` and `RGB10_444_2P` (506, 507, 544, 545) over
//! the 10-bit ones, the packed `YCBCR10_420_2P_PACKED`, `YCBCR10_422_2P_PACKED` and
//! `YCBCR10_444_2P_PACKED` (508, 509, 510) over the packed ones (`p420` to `pf44`), and the `_sRGB`
//! twins of the converted formats (520, 522, 523, 525, 526, 527, 528, 529, 530), each over a
//! surface whose chroma plane holds its own (`SURFACES`), which Metal makes only over a surface
//! (`newTextureWithDescriptor:iosurface:plane:`; plain creation answers nil,
//! `TextureProblem::NoTextureOfFormat`), or as a view of its converted format.
//!
//! - **What Metal reads** (an Apple M4 under macOS 27.0.1, EVs XRR2-ycc1-m4, XRR2-ycc2-m4 and
//!   TXN2-ycr1-m4; TXN2-ycq1-m4 for 4:2:2 and 4:4:4): each texel from the luma code at its place
//!   and the chroma pairs around it at the format's subsampling: 4:2:0 9/3/3/1 over the four
//!   nearest pairs at the middle of each 2x2 block, 4:2:2 3/1 over the two nearest across, its
//!   row's down, 4:4:4 its own pair (the chroma plane of a surface of finer chroma read as the
//!   format's at its rows), each code a byte, the high 10 bits of a 16-bit word, or one of three
//!   10-bit codes of a 32-bit word, packed (EVs TXN2-ycq1-m4 and TXN2-ycp1-m4). The converted
//!   formats read the color converted by the surface's range (its OSType: `4..v` and `x4..`
//!   video, `4..f` and `xf..` full) and the matrix it held when the texture was made
//!   (`kIOSurfaceYCbCrMatrix`: BT.601 or none, BT.709, BT.2020, the last EV TXN2-ycr3-m4, and
//!   SMPTE 240M, read as BT.709, EV XRR2-ycf3p4-m4): every channel a multiple of 1/2048, and
//!   alpha 1. The code formats (540 to 545) read the codes, red Cr, green Y, blue Cb, each
//!   `code / 1023` or `code / 255`, alpha 1 (EVs XRR2-ycr540-m4 for 540, XRR2-ycr541-m4 for 541,
//!   542, 544 and 545). The `_sRGB` twins read the converted sum
//!   through the sRGB curve, every channel a multiple of 1/8192 (EVs TXN2-ycr1-m4 and
//!   TXN2-ycv1-m4, every one of the curve's 8193 steps), and a `half` read of it rounds with
//!   ties away from zero (`metal2vulkan::bind::ycbcr_read` states the arithmetic). `read()`, a
//!   nearest sample and a gather read those texels; a 4:4:4 texture samples repeat, mirrored
//!   repeat and mirror-clamp-to-edge as those modes, a 4:2:0 or 4:2:2 one as clamp-to-edge (EVs
//!   TXN2-ycb1-m4 and TXN2-ycb2-m4; XRR2-ycr541-m4 and XRR2-ycr541b-m4 for the code formats), and
//!   the borders read alpha 1, but for a texture read as codes, whose clamp-to-zero and
//!   clamp-to-border-color read as clamp-to-edge too.
//! - **The image.** The texture is an image of those texels, which holds every one exactly
//!   (`R16G16B16A16_SFLOAT` of the converted values, `R32G32B32A32_SFLOAT` of the sRGB twins',
//!   16 bytes a texel where a half holds no step of 1/8192 above 1/4,
//!   `A2B10G10R10_UNORM_PACK32` or `R8G8B8A8_UNORM` of the codes, written through an `R32_UINT`
//!   view), and a shadow
//!   of the surface's bytes (`shadow.rs`, `ShadowSource::Converted`): its copy in is a
//!   conversion, the planes' bytes copied into a staging buffer of the texture's and one
//!   dispatch of the conversion kernel
//!   (`bind::ycbcr_convert_module`, `owned_kernel.rs`) writing every texel. The texture reads
//!   alpha 1 (`alpha_reads_one`) and maps those modes to clamp-to-edge (`address_mode_map`), so
//!   a nearest sample and a border read are Metal's exactly.
//! - **When it converts.** Before an encoder's first command that uses the texture, and again
//!   after a command of the encoder that writes the planes' bytes (through the buffer, a texture
//!   over the planes such as a decoder's `R8Unorm` and `RG8Unorm` ones, or a blit): a use of the
//!   bytes like any other, which the tracker orders after the writes of the same and earlier
//!   command buffers. The surface's bytes are the CPU's to write unseen, so the runtime cannot
//!   tell cheaply that they are unchanged since the last conversion, and converts at every
//!   encoder that uses the texture.
//! - **The texels Metal's copies move** are the luma plane's bytes, a code's bytes a texel
//!   (EVs TXN2-ycr2-m4, TXN2-ycr4-m4 for 4:2:2 and 4:4:4 and, for the code formats,
//!   XRR2-ycr540c-m4 and XRR2-ycr541-m4):
//!   `get_bytes`, `replace_region` and a blit to or from a buffer move the luma rows (Metal
//!   keeps the chroma plane), the texture's `Texels` (`buffer_texture.rs`). A packed texture's
//!   `get_bytes` and `replace_region` move nothing, at any bytes per row (EV TXN2-ycp1-m4: the
//!   bytes and both planes as they were, 48 calls of each), and its blits with a buffer, which
//!   move bytes at places that follow no rule found, are refused.
//! - **Plane 1** (EV XRR2-ycp3-m4: each of the 18 unpacked formats above over plane 1 of `422v`,
//!   `422f`, `444v`, `444f`, `x422`, `xf22`, `x444` and `xf44` surfaces, without a matrix and with
//!   BT.709, 7 x 5 to 64 x 64; `read()`, a `half` read and nearest samples, 0 texels off). Over
//!   plane 1 of a surface whose plane 1 is as tall as plane 0 (4:2:2, 4:4:4), Metal reads plane 1's
//!   codes as the luma plane, at plane 1's rows, with chroma codes 0, converted as over plane 0
//!   (the surface's range and matrix); its copies move plane 1's codes. The runtime converts the
//!   same: the luma plane is plane 1, and the chroma plane one row of zeros the conversion's
//!   staging buffer holds past the plane's bytes, every row read from it.
//! - **Views.** A converted texture's view as its `_sRGB` twin (Metal makes it with
//!   `ShaderRead` usage alone) is a texture of the twin over the same planes, with a conversion
//!   of its own: Metal's view reads what a texture of the twin over the surface reads, bit for
//!   bit (EVs TXN2-ycv1-m4 and TXN2-ycq1-m4; `texture/ycbcr_image.rs`).
//! - **Refused** (`Unsupported`, `NotImplemented`, `Feature::Format`, naming what): a surface of
//!   an OSType the format is not measured over (Metal makes 505 and 543 over plane 0 of many
//!   more, EV TXN2-ycr1-m4), or whose chroma plane is smaller than the format's (Metal reads
//!   past it, EV TXN2-ycq1-m4), plane 1 of a 4:2:0 surface (Metal reads its rows past the plane,
//!   into the bytes after it), plane 1 of a packed surface (not measured), a plane past 1, plane 1
//!   at other bytes than the surface's facts name, a matrix other than BT.601, BT.709, BT.2020 or
//!   SMPTE 240M, a texture of another size than the surface's, planes at offsets or rows off 16
//!   bytes, a row shorter than its codes, planes past the buffer, and planes past one storage
//!   range; a packed texture's blit with a buffer (above); a copy between two such textures (Metal
//!   copies both planes), a view in another format than the twin (a twin's view as its converted
//!   format among them, unmeasured), and a pass rendering into one but as **Passes** below say
//!   (Metal converts back into the planes). A function's writes to a 4:2:0 or 4:2:2 format are
//!   dropped, as Metal drops them (`metal::format::writes_dropped`); a 4:4:4 `_sRGB` twin's or
//!   packed format's are refused (the twins' encoding is not found, and the M3 drops the packed
//!   formats' writes where the M4 stores them), as are a texture's over plane 1 (not measured) and
//!   a slot's that reads the texture too.
//! - **Writes** (EVs XRR2-ycw1-m4 to XRR2-ycw4-m4 on an Apple M4 under macOS 27.0.1, XRR2-ycw2-m3
//!   and XRR2-ycw3-m3 on an Apple M3 Max under macOS 26.7, byte for byte alike). Metal stores a
//!   function's writes to `YCBCR8_444_2P`, `YCBCR10_444_2P`, `RGB8_444_2P` and `RGB10_444_2P` (503,
//!   507, 542, 545) into both planes: each written texel's luma code at its place and its Cb and Cr
//!   pair at its own, the codes `metal2vulkan::bind::ycbcr_write` states (the converted formats' by
//!   the surface's range and matrix; the code formats' green, blue and red as codes), and leaves
//!   every other byte. The function writes such a texture through its write shadow (`shadow.rs`,
//!   `ShadowSource::YCbCrWrite`; `texture/ycbcr_image.rs`): an `RGBA32Uint` image of each written
//!   texel's bits and a mark, cleared before an encoder's first command that writes it and again
//!   after the planes' bytes are written another way; its copy back ([`YCbCrWriteBack`]) copies the
//!   planes' bytes into a staging buffer of the texture's, dispatches the encode kernel
//!   (`bind::ycbcr_write_module`), which stores the codes of the marked texels, and copies the
//!   planes' rows of codes back. The texture's converted texels then convert again at their next
//!   use, as after any write of the planes. Exact, so announced nowhere.
//! - **Passes** (EVs XRR2-ycr1-m4, GSM2-yt1-m4 and GSM2-yt2-m4 on an Apple M4 under macOS 27.0.1,
//!   and XRR2's probe on the consumer's Apple M3 Max under macOS 26.7, byte for byte alike). Metal
//!   stores a pass's color attachment of `YCBCR8_420_2P`, `YCBCR8_422_2P` and `YCBCR8_444_2P` (500,
//!   502, 503) over plane 0 into both planes at its store: every texel's luma code, drawn or
//!   not, and each chroma pair's codes of the block of texels it covers, the codes
//!   `metal2vulkan::bind::ycbcr_store` states (`metal::format::ycbcr_renders`). The pass renders
//!   into the texture's render shadow (`texture/ycbcr_image.rs`), an `R32G32B32A32_SFLOAT`
//!   image of the texture's size, cleared at its copy in; its copy back is the write-back above
//!   storing every texel ([`YCbCrWriteBack`] of a store), and the texture's converted texels
//!   convert again at their next use. Exact for a pass that clears the attachment or leaves it
//!   to the pass (`DontCare`) and stores it, a draw writing red, green and blue unblended; a
//!   load, another store action, a resolve, a blend and a partial write mask are refused
//!   (`NotImplemented(Format)`, naming the format, at the pass or the pipeline), as are a
//!   texture over plane 1 and one whose size the chroma blocks do not tile, none measured; the
//!   other formats' passes stay refused (the 10-bit ones' draws match no rule yet). A packed
//!   10-bit format's pipeline is made, as Metal makes it, and its pass refused at its start,
//!   naming the format (`metal::format::ycbcr_pass_refused`): Metal fails the pass's command
//!   buffer, its error and the luma bytes it leaves varying from run to run (EVs GSM2-pk1-m4 and
//!   GSM2-pk2-m4), so no exact form exists.
//! - **Linear filtering** is Metal's, exact: the function emulates Metal's filter of the texels on
//!   their grid of 1/2048 or 1/8192 (`metal2vulkan::metal::texel_format::PackedTexels::YCbCr2048`,
//!   `YCbCr8192`), under MoltenVK through the slot's YCbCr variant, and announces, naming the slot,
//!   any sample the emulation does not reach (`ycbcr-filter-not-exact`). A function that samples
//!   textures through handles reads its sampler from memory and is refused where it can reach the
//!   texture (`MappedTextureSampling`, as the texture's address-mode map is).

use super::buffer::Buffer;
use super::error::{internal, Error};
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSet, OwnedSets};
use super::resource::ResourceKey;
use super::shadow::ShadowCopies;
use super::texture_descriptor::TextureLayout;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::values::ResourceOptions;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{
    YCbCrCodes, YCbCrConvertModule, YCbCrMatrix, YCbCrPlanes, YCbCrRange, YCbCrWriteModule,
};
use metal2vulkan::error::{DiagName, Feature, Unsupported};
use metal2vulkan::metal::format::{ycbcr_format, ChromaSubsampling};
use metal2vulkan::metal::PixelFormat;
use std::ops::Range;

/// The CoreVideo surfaces a texture is made over (the module docs): the OSType, the bits of its
/// codes and whether they are packed, which the format's must be, its range, and how its chroma
/// plane is subsampled. `420v` and `420f` are 8-bit 4:2:0, `422v` and `422f` 4:2:2, `444v` and
/// `444f` 4:4:4; `x420`, `x422` and `x444` 10-bit 4:2:0, 4:2:2 and 4:4:4 in 16-bit words, video
/// range, and `xf20`, `xf22` and `xf44` the same, full range; `p420`, `p422` and `p444`, and
/// `pf20`, `pf22` and `pf44`, the same packed three to a 32-bit word. A format reads a surface
/// whose chroma plane holds its own: a 4:2:0 format a 4:2:2 or 4:4:4 surface's, a 4:2:2 format a
/// 4:4:4 one's, each at the surface's rows.
const SURFACES: [(&[u8; 4], u32, bool, YCbCrRange, ChromaSubsampling); 18] = {
    use ChromaSubsampling::{Chroma420 as C420, Chroma422 as C422, Chroma444 as C444};
    [
        (b"420v", 8, false, YCbCrRange::Video, C420),
        (b"420f", 8, false, YCbCrRange::Full, C420),
        (b"422v", 8, false, YCbCrRange::Video, C422),
        (b"422f", 8, false, YCbCrRange::Full, C422),
        (b"444v", 8, false, YCbCrRange::Video, C444),
        (b"444f", 8, false, YCbCrRange::Full, C444),
        (b"x420", 10, false, YCbCrRange::Video, C420),
        (b"xf20", 10, false, YCbCrRange::Full, C420),
        (b"x422", 10, false, YCbCrRange::Video, C422),
        (b"xf22", 10, false, YCbCrRange::Full, C422),
        (b"x444", 10, false, YCbCrRange::Video, C444),
        (b"xf44", 10, false, YCbCrRange::Full, C444),
        (b"p420", 10, true, YCbCrRange::Video, C420),
        (b"pf20", 10, true, YCbCrRange::Full, C420),
        (b"p422", 10, true, YCbCrRange::Video, C422),
        (b"pf22", 10, true, YCbCrRange::Full, C422),
        (b"p444", 10, true, YCbCrRange::Video, C444),
        (b"pf44", 10, true, YCbCrRange::Full, C444),
    ]
};
/// What the planes' offsets and rows are multiples of: Metal's GPU reads a plane from its offset
/// rounded down to 16, rows `bytesPerRow` rounded down to 16 apart (`linear_plan.rs`), which
/// the conversion is measured at only where the two agree.
const PLANE_ALIGNMENT: u64 = 16;
/// Where the planes' bytes start in the staging buffer, after the shape: a multiple of every
/// `minStorageBufferOffsetAlignment`.
const BYTES_AT: u64 = 256;

/// The facts of an IOSurface a two-plane YCbCr texture over it reads, and a two-plane
/// extended-range one its plane 1 from
/// ([`IoSurfacePlane::ycbcr`](crate::IoSurfacePlane::ycbcr)): Metal's
/// `newTextureWithDescriptor:iosurface:plane:` reads them from the `IOSurfaceRef`, the
/// runtime's stand-in from here.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IoSurfaceYCbCr {
    /// `IOSurfaceGetPixelFormat`: the surface's OSType, as a big-endian four-character code
    /// (`'420v'` is `0x3432_3076`).
    pub pixel_format: u32,
    /// The plane the texture is asked of (`newTextureWithDescriptor:iosurface:plane:`'s
    /// `plane`).
    pub plane: u64,
    /// `IOSurfaceGetWidth` and `IOSurfaceGetHeight`: the luma plane's size.
    pub size: [u64; 2],
    /// Plane 1's first byte in the buffer that stands for the surface:
    /// `IOSurfaceGetBaseAddressOfPlane(surface, 1)` less `IOSurfaceGetBaseAddress`. A two-plane
    /// extended-range texture reads its alpha plane there.
    pub chroma_offset: u64,
    /// `IOSurfaceGetBytesPerRowOfPlane(surface, 1)`.
    pub chroma_bytes_per_row: u64,
    /// The surface's `kIOSurfaceYCbCrMatrix` as the texture is made, `None` without one.
    pub matrix: Option<SurfaceYCbCrMatrix>,
}

impl IoSurfaceYCbCr {
    /// A surface of OSType `pixel_format` and `size`, the texture asked of `plane`, its chroma
    /// plane at `chroma_offset` with rows `chroma_bytes_per_row` apart, without a matrix.
    pub const fn new(
        pixel_format: u32,
        plane: u64,
        size: [u64; 2],
        chroma_offset: u64,
        chroma_bytes_per_row: u64,
    ) -> IoSurfaceYCbCr {
        IoSurfaceYCbCr {
            pixel_format,
            plane,
            size,
            chroma_offset,
            chroma_bytes_per_row,
            matrix: None,
        }
    }

    /// These facts with the surface's `kIOSurfaceYCbCrMatrix`.
    pub const fn with_matrix(mut self, matrix: SurfaceYCbCrMatrix) -> IoSurfaceYCbCr {
        self.matrix = Some(matrix);
        self
    }
}

/// An IOSurface's `kIOSurfaceYCbCrMatrix` attachment.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceYCbCrMatrix {
    /// `ITU_R_601_4`.
    Itu601,
    /// `ITU_R_709_2`.
    Itu709,
    /// `ITU_R_2020`.
    Itu2020,
    /// `SMPTE_240M_1995`, which Metal reads as BT.709 (EV XRR2-ycf3p4-m4).
    Smpte240M,
    /// Any other value.
    Other,
}

impl SurfaceYCbCrMatrix {
    /// The matrix of the attachment's string value (`kCVImageBufferYCbCrMatrix_*`).
    pub fn from_attachment(value: &str) -> SurfaceYCbCrMatrix {
        match value {
            "ITU_R_601_4" => SurfaceYCbCrMatrix::Itu601,
            "ITU_R_709_2" => SurfaceYCbCrMatrix::Itu709,
            "ITU_R_2020" => SurfaceYCbCrMatrix::Itu2020,
            "SMPTE_240M_1995" => SurfaceYCbCrMatrix::Smpte240M,
            _ => SurfaceYCbCrMatrix::Other,
        }
    }
}

/// A refusal of a texture of `pf` over a surface: `NotImplemented`, naming `what`.
fn refused(pf: PixelFormat, what: String) -> Error {
    let named = format!("{} {what}", pf.name());
    Error::Unsupported(Unsupported::not_implemented(Feature::Format).naming(DiagName::new(named)))
}

/// The planes of a two-plane YCbCr texture of `layout` (checked; a format `metal::format::ycbcr`
/// holds) over `surface`, its luma plane at `luma` (offset, bytes per row) of a buffer of
/// `length` bytes on `device`: the planes as the conversion reads them from the span's first
/// byte, and the span, the bytes the planes lie in. Refused as the module docs say.
pub(super) fn surface_planes(
    layout: &TextureLayout,
    luma: [u64; 2],
    surface: &IoSurfaceYCbCr,
    length: u64,
    device: &Device,
) -> Result<(YCbCrPlanes, Range<u64>), Error> {
    let pf = layout.pixel_format;
    let refused = |what: String| refused(pf, what);
    let codes = ycbcr_format(pf)
        .and_then(YCbCrCodes::of)
        .ok_or_else(|| internal("a YCbCr texture of a layout the conversion does not read"))?;
    // The format's chroma plane fits in the surface's: halved along each axis the surface's is.
    let fits = |chroma: ChromaSubsampling| {
        let [ours, its] = [codes.subsampling, chroma].map(ChromaSubsampling::halved);
        ours.iter().zip(its).all(|(&ours, its)| ours || !its)
    };
    let code = || {
        surface
            .pixel_format
            .to_be_bytes()
            .map(|b| b.escape_ascii().to_string())
            .concat()
    };
    let measured = SURFACES.iter().find(|&&(code, bits, packed, _, _)| {
        u32::from_be_bytes(*code) == surface.pixel_format
            && (bits, packed) == (codes.bits, codes.packed)
    });
    let Some(&(_, _, _, range, chroma)) = measured else {
        return Err(refused(format!("over a surface of OSType '{}'", code())));
    };
    let plane_one = match surface.plane {
        0 if fits(chroma) => false,
        0 => return Err(refused(format!("over a surface of OSType '{}'", code()))),
        1 if codes.packed => {
            return Err(refused(format!(
                "of plane 1 of a surface of OSType '{}'",
                code()
            )))
        }
        1 if !chroma.halved()[1] => true,
        1 => {
            return Err(refused(format!(
                "of plane 1 of a surface of OSType '{}', whose rows Metal reads past the plane",
                code()
            )))
        }
        plane => return Err(refused(format!("of plane {plane}"))),
    };
    let matrix = match surface.matrix {
        None | Some(SurfaceYCbCrMatrix::Itu601) => YCbCrMatrix::Bt601,
        Some(SurfaceYCbCrMatrix::Itu709 | SurfaceYCbCrMatrix::Smpte240M) => YCbCrMatrix::Bt709,
        Some(SurfaceYCbCrMatrix::Itu2020) => YCbCrMatrix::Bt2020,
        Some(other) => return Err(refused(format!("of YCbCr matrix {other:?}"))),
    };
    let [width, height, _] = layout.extent.map(u64::from);
    if [width, height] != surface.size || layout.levels != 1 || layout.layers != 1 {
        return Err(refused(format!(
            "of {width}x{height}, {} levels, {} slices, over a surface of {}x{}",
            layout.levels, layout.layers, surface.size[0], surface.size[1]
        )));
    }
    let halved = codes.subsampling.halved();
    let [cw, ch] =
        [(width, halved[0]), (height, halved[1])]
            .map(|(n, halved)| if halved { n.div_ceil(2) } else { n });
    let chroma = [surface.chroma_offset, surface.chroma_bytes_per_row];
    let aligned = luma
        .iter()
        .chain(&chroma)
        .all(|v| v.is_multiple_of(PLANE_ALIGNMENT));
    // A row of `n` codes is `units(n)` units of `bytes` bytes (three packed codes to a unit).
    let unit = u64::from(codes.bytes);
    let [luma_bytes, chroma_bytes] = [width, 2 * cw].map(|n| unit * codes.units(n));
    // Over plane 1, the call's plane is the surface's plane 1, and its codes the luma.
    let chroma_short = !plane_one && chroma[1] < chroma_bytes;
    if !aligned || luma[1] < luma_bytes || chroma_short || (plane_one && luma != chroma) {
        return Err(refused(format!(
            "with planes at {luma:?} and {chroma:?} (offset, bytes per row)"
        )));
    }
    let end = |[offset, row]: [u64; 2], rows: u64, bytes: u64| offset + row * (rows - 1) + bytes;
    let span = match plane_one {
        true => luma[0]..end(luma, height, luma_bytes),
        false => {
            luma[0].min(chroma[0])..end(luma, height, luma_bytes).max(end(chroma, ch, chroma_bytes))
        }
    };
    let limit = u64::from(
        device
            .properties()
            .description
            .limits
            .max_storage_buffer_range,
    )
    .min(u64::from(u32::MAX));
    // Over plane 1 the conversion holds a row of zeros past the span too.
    let zeros = if plane_one { chroma_bytes + 4 } else { 0 };
    if span.end > length || span.end - span.start + zeros > limit - BYTES_AT {
        return Err(refused(format!(
            "with planes over bytes {span:?} of a buffer of {length}"
        )));
    }
    // The span is under `u32::MAX`, so every offset and row in it is.
    let word = |v: u64| v as u32;
    // Over plane 1, chroma codes 0: one row of zeros past the span's bytes, which the
    // conversion holds (`YCbCrConversion::new`), every row read from it.
    let chroma = match plane_one {
        true => [(span.end - span.start).next_multiple_of(4), 0],
        false => [chroma[0] - span.start, chroma[1]],
    };
    let planes = YCbCrPlanes::new(
        [word(width), word(height)],
        [word(luma[0] - span.start), word(luma[1])],
        [word(chroma[0]), word(chroma[1])],
        (range, matrix),
        codes,
    );
    Ok((planes, span))
}

/// The conversion of a two-plane YCbCr texture's planes into its image (the module docs): the
/// copy in of its shadow (`ShadowSource::Converted`).
pub(super) struct YCbCrConversion {
    /// The planes it converts, and the bytes of their buffer they lie in, which a view as the
    /// texture's `_sRGB` twin converts too (`texture/ycbcr_image.rs`).
    source: (YCbCrPlanes, Range<u64>),
    /// The shape at its start, the planes' bytes from `BYTES_AT`; held for the texture's life.
    staging: Buffer,
    /// The staging buffer's identity and its bytes.
    staging_use: (ResourceKey, Range<u64>),
    /// Where the shape lies in the staging buffer's `VkBuffer`.
    shape_at: u64,
    shape: [u8; YCbCrConvertModule::SHAPE_WORDS as usize * 4],
    /// The planes' bytes from the surface's buffer into the staging buffer.
    into: vk::BufferCopy,
    /// Where the zeros a texture over plane 1 reads as its chroma lie in the staging buffer's
    /// `VkBuffer`, and their bytes (0 for every other texture).
    zeros: (u64, u64),
    kernel: OwnedKernel,
    sets: OwnedSets,
    groups: [u32; 2],
}

impl YCbCrConversion {
    /// The conversion of `planes`, whose bytes lie in `bytes` of their buffer, at `at` of its
    /// `VkBuffer`, into `view`, a storage view of the texture's image, on `device`. Refused as
    /// `new_buffer` refuses the staging buffer and as the kernel's objects are refused.
    pub(super) fn new(
        device: &Device,
        (planes, bytes_in): (&YCbCrPlanes, Range<u64>),
        at: u64,
        view: vk::ImageView,
    ) -> Result<YCbCrConversion, Error> {
        let span = bytes_in.end - bytes_in.start;
        let bytes = span.next_multiple_of(4);
        // The chroma bytes the planes read past their span, zeros (a texture over plane 1,
        // whose chroma codes are 0), filled at every conversion.
        let [cw, ch] = planes.chroma_size().map(u64::from);
        let chroma_row = 2 * u64::from(planes.codes.bytes) * cw;
        let [chroma_at, chroma_pitch] = planes.chroma.map(u64::from);
        let chroma_end = chroma_at + chroma_pitch * (ch.max(1) - 1) + chroma_row;
        let zeros = chroma_end.saturating_sub(bytes).next_multiple_of(4);
        let held = bytes + zeros;
        let staging = device.own_buffer(BYTES_AT + held, ResourceOptions::private())?;
        let (raw, start) = staging.at(0);
        let (identity, range) = staging.tracked(0..BYTES_AT + held);
        let shape = YCbCrConvertModule::shape(planes);
        let kernel = device.owned_kernel(OwnedModule::YCbCrConvert(planes.codes.output))?;
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: vec![view],
            buffers: vec![
                (raw, start, shape.len() as u64),
                (raw, start + BYTES_AT, held),
            ],
        };
        let sets = OwnedSets::write_each(device, &[kernel], &[set])?;
        let groups = planes
            .size
            .map(|n| n.div_ceil(YCbCrConvertModule::LOCAL_SIZE));
        let limits = &device.properties().description.limits;
        if groups[0] > limits.max_compute_work_group_count[0]
            || groups[1] > limits.max_compute_work_group_count[1]
        {
            return Err(internal(
                "a YCbCr texture past the device's workgroup counts",
            ));
        }
        Ok(YCbCrConversion {
            source: (*planes, bytes_in),
            staging_use: (identity.key, range),
            shape_at: start,
            shape,
            into: vk::BufferCopy {
                src_offset: at,
                dst_offset: start + BYTES_AT,
                size: span,
            },
            zeros: (start + BYTES_AT + bytes, zeros),
            staging,
            kernel,
            sets,
            groups,
        })
    }

    /// The planes it converts, and the bytes of their buffer they lie in.
    pub(super) fn source(&self) -> (YCbCrPlanes, Range<u64>) {
        (self.source.0, self.source.1.clone())
    }

    /// Whether the texture is over plane 1 of its surface, its chroma codes the zeros the
    /// staging buffer holds past the planes' bytes.
    pub(super) fn over_plane_one(&self) -> bool {
        self.zeros.1 > 0
    }

    /// The conversion's uses: `truth`, the read of the planes' bytes, then the staging buffer's
    /// writes and read, and the write of every texel of the image of `copies`.
    pub(super) fn uses(
        &self,
        truth: Use<ResourceKey>,
        copies: &ShadowCopies,
    ) -> Vec<Use<ResourceKey>> {
        let (key, range) = self.staging_use.clone();
        vec![
            truth,
            Use {
                key,
                tracked: copies.tracked,
                range: UseRange::Bytes(range),
                read: Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
                write: Access::new(
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            },
            Use::writes(
                copies.shadow_key,
                copies.tracked,
                copies.shadow_range.clone(),
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            ),
        ]
    }

    /// Record the conversion of the planes in `source` (the surface's buffer's `VkBuffer` now)
    /// into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; `source`, the staging buffer and
    /// the image are live until it completes, and the image is in the `GENERAL` layout.
    pub(super) unsafe fn record(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        source: vk::Buffer,
    ) {
        let staging = self.staging.at(0).0;
        // SAFETY: the caller's contract; the shape and the copy lie in the staging buffer and
        // the copy in `source` (checked when the texture was made), and the set binds them.
        unsafe {
            device.cmd_update_buffer(command_buffer, staging, self.shape_at, &self.shape);
            device.cmd_copy_buffer(command_buffer, source, staging, &[self.into]);
            let (zeros_at, zeros) = self.zeros;
            crate::word_fill::record(
                self.staging.device(),
                command_buffer,
                staging,
                zeros_at..zeros_at + zeros,
                0,
            );
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &self.sets.sets,
                &[],
            );
            device.cmd_dispatch(command_buffer, self.groups[0], self.groups[1], 1);
        }
    }
}

/// The write-back of a 4:4:4 two-plane YCbCr texture's written texels into its planes (the
/// module docs): the copy back of its write shadow (`ShadowSource::YCbCrWrite`). The planes'
/// bytes go into a staging buffer of the texture's, the encode kernel (`bind::ycbcr_write_module`,
/// `owned_kernel.rs`) stores the codes of every texel the shadow marks written, and the planes'
/// rows of codes go back; the bytes between the rows stay as they were.
pub(super) struct YCbCrWriteBack {
    /// The shape at its start, the planes' bytes from `BYTES_AT`; held for the texture's life.
    staging: Buffer,
    /// The staging buffer's identity and its bytes.
    staging_use: (ResourceKey, Range<u64>),
    /// Where the shape lies in the staging buffer's `VkBuffer`.
    shape_at: u64,
    shape: [u8; YCbCrWriteModule::SHAPE_WORDS as usize * 4],
    /// The planes' bytes from the surface's buffer into the staging buffer.
    into: vk::BufferCopy,
    /// Each row of codes of both planes from the staging buffer back into the surface's buffer.
    back: Vec<vk::BufferCopy>,
    kernel: OwnedKernel,
    sets: OwnedSets,
    groups: [u32; 2],
}

impl YCbCrWriteBack {
    /// The write-back into `planes` (a layout `bind::ycbcr_write` encodes, over plane 0), whose
    /// bytes lie in `bytes` of their buffer, at `at` of its `VkBuffer`, of the written texels in
    /// `view`, a storage view of the write shadow, on `device`; or, `all`, a pass's store of
    /// every texel of its render shadow (a layout `bind::ycbcr_store` encodes), the view's texels'
    /// bits. Refused as `new_buffer` refuses the staging buffer and as the kernel's objects are
    /// refused.
    pub(super) fn new(
        device: &Device,
        (planes, bytes_in): (&YCbCrPlanes, Range<u64>),
        at: u64,
        (view, all): (vk::ImageView, bool),
    ) -> Result<YCbCrWriteBack, Error> {
        let span = bytes_in.end - bytes_in.start;
        // The kernel reads and writes whole words: a row's last may reach past the span's end.
        let held = span.next_multiple_of(4);
        let staging = device.own_buffer(BYTES_AT + held, ResourceOptions::private())?;
        let (raw, start) = staging.at(0);
        let (identity, range) = staging.tracked(0..BYTES_AT + held);
        let shape = YCbCrWriteModule::shape(planes, all);
        let kernel = device.owned_kernel(OwnedModule::YCbCrWrite)?;
        let set = OwnedSet {
            images: Vec::new(),
            storage_images: vec![view],
            buffers: vec![
                (raw, start, shape.len() as u64),
                (raw, start + BYTES_AT, held),
            ],
        };
        let sets = OwnedSets::write_each(device, &[kernel], &[set])?;
        let groups = YCbCrWriteModule::groups(planes);
        let limits = &device.properties().description.limits;
        if groups[0] > limits.max_compute_work_group_count[0]
            || groups[1] > limits.max_compute_work_group_count[1]
        {
            return Err(internal(
                "a YCbCr texture past the device's workgroup counts",
            ));
        }
        // Each plane's rows of codes: the luma plane's a code a texel, the chroma plane's a pair.
        let unit = u64::from(planes.codes.bytes);
        let [width, height] = planes.size.map(u64::from);
        let [chroma_width, chroma_height] = planes.chroma_size().map(u64::from);
        let rows = |[offset, row]: [u32; 2], rows: u64, bytes: u64| {
            (0..rows).map(move |y| {
                let first = u64::from(offset) + y * u64::from(row);
                vk::BufferCopy {
                    src_offset: start + BYTES_AT + first,
                    dst_offset: at + first,
                    size: bytes,
                }
            })
        };
        let back = rows(planes.luma, height, unit * width)
            .chain(rows(planes.chroma, chroma_height, 2 * unit * chroma_width))
            .collect();
        Ok(YCbCrWriteBack {
            staging_use: (identity.key, range),
            shape_at: start,
            shape,
            into: vk::BufferCopy {
                src_offset: at,
                dst_offset: start + BYTES_AT,
                size: span,
            },
            back,
            staging,
            kernel,
            sets,
            groups,
        })
    }

    /// The write-back's uses: `truth`, the read and the write of the planes' bytes, then the
    /// staging buffer's writes and reads, and the read of every texel of the shadow of `copies`.
    pub(super) fn uses(
        &self,
        truth: Use<ResourceKey>,
        copies: &ShadowCopies,
    ) -> Vec<Use<ResourceKey>> {
        let (key, range) = self.staging_use.clone();
        let both = vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        vec![
            truth,
            Use {
                key,
                tracked: copies.tracked,
                range: UseRange::Bytes(range),
                read: Access::new(
                    both,
                    vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
                write: Access::new(
                    both,
                    vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            },
            Use::reads(
                copies.shadow_key,
                copies.tracked,
                copies.shadow_range.clone(),
                Access::new(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
            ),
        ]
    }

    /// Record the write-back into the planes in `source` (the surface's buffer's `VkBuffer`
    /// now) into `command_buffer`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; `source`, the staging buffer and
    /// the shadow are live until it completes, and the shadow is in the `GENERAL` layout.
    pub(super) unsafe fn record(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        source: vk::Buffer,
    ) {
        let staging = self.staging.at(0).0;
        // SAFETY: the caller's contract; the shape, the copy in and the rows lie in the staging
        // buffer and in `source` (checked when the texture was made), and the set binds them.
        unsafe {
            device.cmd_update_buffer(command_buffer, staging, self.shape_at, &self.shape);
            device.cmd_copy_buffer(command_buffer, source, staging, &[self.into]);
            Barrier::everything().record(device, command_buffer);
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.kernel.layout,
                0,
                &self.sets.sets,
                &[],
            );
            device.cmd_dispatch(command_buffer, self.groups[0], self.groups[1], 1);
            Barrier::everything().record(device, command_buffer);
            device.cmd_copy_buffer(command_buffer, staging, source, &self.back);
        }
    }
}
