//! The macOS releases whose Metal the runtime answers as, where Metal's answers differ by release.
//!
//! Metal's answers about pixel formats differ by macOS release: which raw values a texture
//! descriptor's `setPixelFormat:` takes, and which formats `MTLPixelFormatGetInfoForDevice`
//! describes, whose textures Metal's validation stops otherwise. A [`MetalLevel`] is one measured
//! release, and its row of the table ([`MetalLevel::knows`], [`MetalLevel::describes`]) is what
//! that release answered on our own probes. Every check that differs by release reads the row of
//! the device's level, never a release's name.
//!
//! A device answers at one level, fixed when it is made
//! ([`DeviceProperties::metal_level`](crate::DeviceProperties::metal_level)): the level its
//! options name (`DeviceOptions::metal_level`, `RawDevice::metal_level`), else, on a macOS host,
//! the host's own release, or the newest measured release below it, announced once as
//! `metal2vulkan-degrade-metal-level-below-host`; a macOS host below every measured release, or
//! whose release cannot be read, refuses the device ([`Error::Unproven`], since an older Metal's
//! answers are unknown and a newer one's would answer above the host). Off macOS the level is the
//! newest, [`MetalLevel::NEWEST`], announced once as `metal2vulkan-degrade-metal-level-not-macos`:
//! no release of the host's names it.

use crate::error::{Error, Problem, Unproven};
use crate::texture::heap_tiled::{BlockFamily, BlockPlacement, GpuRows};
use crate::Degradation;
use metal2vulkan::bind::{CompressedBlit, MipmapRule};
use metal2vulkan::metal::PixelFormat;
use std::fmt;

/// A macOS release whose Metal answers the runtime gives where releases answer differently (the
/// module docs): which pixel format values its texture descriptor takes ([`knows`](Self::knows))
/// and which formats it describes ([`describes`](Self::describes)). Ordered by release. No Metal
/// counterpart: Metal answers as the release it runs on.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalLevel {
    /// macOS 26.7 (build 25G229), measured on an Apple M3 Max.
    MacOS26_7,
    /// macOS 27.0.1 (build 26A434), measured on an Apple M3 Max. An Apple M4's answers differ where
    /// both were measured in the formats from 630 to 645, which it describes, in `gpuCoreCount` and
    /// in `supportsComputeCompressedTextureWrite`.
    MacOS27_0_1,
}

/// The 12 three-channel formats the macOS 27.0 SDK declares, and `RGB16Bfloat`.
const THREE_CHANNEL: [PixelFormat; 13] = [
    PixelFormat::RGB8Unorm,
    PixelFormat::RGB8Snorm,
    PixelFormat::RGB8Uint,
    PixelFormat::RGB8Sint,
    PixelFormat::RGB16Unorm,
    PixelFormat::RGB16Snorm,
    PixelFormat::RGB16Uint,
    PixelFormat::RGB16Sint,
    PixelFormat::RGB16Float,
    PixelFormat::RGB16Bfloat,
    PixelFormat::RGB32Uint,
    PixelFormat::RGB32Sint,
    PixelFormat::RGB32Float,
];

/// What one level's Metal answers where levels differ: its row of the table.
struct Row {
    /// The release, as macOS names it.
    macos: &'static str,
    /// Its version, for choosing a host's level.
    version: [u32; 3],
    /// The values its texture descriptor does not know: `setPixelFormat:` stops the process
    /// ("not a valid MTLPixelFormat").
    unknown: &'static [PixelFormat],
    /// How its Metal places each family's blocks in a heap texture's levels, where measured.
    heap_blocks: HeapBlocks,
    /// Whether its Metal is measured to write the texels of `CompressedBlit` in a GPU blit
    /// through a losslessly compressed texture (`compressed_blit.rs`); where not, the runtime
    /// keeps the bytes, its device's answer.
    compressed_blits: bool,
    /// The alignment its `heapTextureSizeAndAlignWithDescriptor:` answers, with 0 bytes, for a
    /// texture of `Unspecialized`, `RGBA16Float_XR`, `AnyDepth` or `AnyStencil` of a type with
    /// layers, faces or depth: a 1D or 2D array, a cube, a cube array, a 3D texture or a 2D
    /// multisample array (`texture/heap.rs`'s `no_texture_alignment`).
    no_texture_stack_alignment: u64,
    /// How its Metal's `generateMipmapsForTexture:` makes each level after the first.
    mipmaps: MipmapRule,
}

/// How a level's Metal places a block-compressed texture's blocks in its levels' tiles, per
/// family (`texture/heap_tiled.rs`); `None` where not measured. Where two levels' rows differ,
/// the release makes the difference: an Apple M3 Max places them as each release's row says (and
/// an Apple M4 under 27.0.1 alike).
struct HeapBlocks {
    bc: Option<BlockPlacement>,
    etc: Option<BlockPlacement>,
    astc: Option<BlockPlacement>,
    /// How its GPU counts a level's tiles a row, every family alike.
    gpu_rows: GpuRows,
}

/// macOS 26.7 on an Apple M3 Max: the three-channel formats are values its descriptor does not
/// know (it describes none from 630 on but `AnyDepth` and `AnyStencil`; the table of
/// `pixel_format_info`).
const MACOS_26_7: Row = Row {
    macos: "26.7",
    version: [26, 7, 0],
    unknown: &THREE_CHANNEL,
    // The M4's 815 textures rerun on an Apple M3 Max under 26.7 (packages bc-m3pkg and
    // bc-m3pkg2): every family `Halved` where its CPU copies reach the bytes; its blits count a
    // level's tiles a row by the level's own blocks.
    heap_blocks: HeapBlocks {
        bc: Some(BlockPlacement::Halved),
        etc: Some(BlockPlacement::Halved),
        astc: Some(BlockPlacement::Halved),
        gpu_rows: GpuRows::Own,
    },
    // The package share-cblit on the consumer's M3 Max (25G229): every line of the predicate,
    // the four rules and the clears identical to the M4's.
    compressed_blits: true,
    // Measured plain and validated, every usage and storage mode.
    no_texture_stack_alignment: 16 << 10,
    mipmaps: MipmapRule::AppleM3MaxPasses,
};

/// macOS 27.0.1 on an Apple M3 Max: its descriptor knows every value the runtime names (it
/// describes none from 630 on but `AnyDepth` and `AnyStencil`; the table of
/// `pixel_format_info`). Its heap blocks and compressed blits are its own, an Apple M4's at 27.0.1
/// alike; the heap shapes the M3 Max was asked of every format (22,856 answers) are the model's.
const MACOS_27_0_1: Row = Row {
    macos: "27.0.1",
    version: [27, 0, 1],
    unknown: &[],
    // Every BC, ETC2, EAC and ASTC format, every element of every level of 815 textures on an
    // Apple M4; the M3 Max answers the 809 of `tests/heap_tiled_blocks.rs` alike, each one's size,
    // alignment, CPU span and GPU span.
    heap_blocks: HeapBlocks {
        bc: Some(BlockPlacement::Narrow),
        etc: Some(BlockPlacement::Narrow),
        astc: Some(BlockPlacement::Halved),
        gpu_rows: GpuRows::AsCpu,
    },
    // Each rule matched every texel of its formats' blits through a texture Metal compresses
    // (private, array, heap, texture to texture, a shared texture's blit out), and every texture
    // Metal keeps uncompressed, every CPU copy and 7 other formats kept their bytes; an Apple M4
    // alike.
    compressed_blits: true,
    // Measured on the M3 Max (and an Apple M4 alike): every type but a texture buffer 128, private
    // and shared storage, `ShaderRead` and `ShaderWrite` usage.
    no_texture_stack_alignment: 128,
    // Every level the filter of the level above (`bind::mipmap_filter`'s "By release").
    mipmaps: MipmapRule::Filter,
};

impl MetalLevel {
    /// Every level, oldest first.
    pub const ALL: [MetalLevel; 2] = [MetalLevel::MacOS26_7, MetalLevel::MacOS27_0_1];

    /// The newest level: a device's off macOS, where no host release is there to answer as.
    pub const NEWEST: MetalLevel = MetalLevel::MacOS27_0_1;

    /// The level's row of the table.
    const fn row(self) -> &'static Row {
        match self {
            MetalLevel::MacOS26_7 => &MACOS_26_7,
            MetalLevel::MacOS27_0_1 => &MACOS_27_0_1,
        }
    }

    /// The release, as macOS names it: `26.7`, `27.0.1`.
    pub const fn macos(self) -> &'static str {
        self.row().macos
    }

    /// Whether the release's texture descriptor takes `pf` (`setPixelFormat:` stops the process
    /// on a value it does not know, "not a valid MTLPixelFormat"). `Invalid` is known.
    pub fn knows(self, pf: PixelFormat) -> bool {
        !self.row().unknown.contains(&pf)
    }

    /// Whether the release's Metal describes `pf`: `MTLPixelFormatGetInfoForDevice` answers a
    /// format for it ([`pixel_format_info`](Self::pixel_format_info), the level's measured table,
    /// which holds no row for `Invalid` or for a value the release's descriptor does not know). Its
    /// validation stops every texture of a format it does not describe ("has invalid pixelFormat"),
    /// and its alignment queries assert on one, which the runtime answers alike (`Invalid`,
    /// `TextureProblem::NotAFormat` and `BufferTextureProblem::NotLinear`), and so does its render
    /// pipeline descriptor's validation for an attachment of one ("is not a valid MTLPixelFormat",
    /// `Problem::AttachmentNotAFormat`). An Apple GPU, whose Metal every device answers as,
    /// describes neither `Depth24Unorm_Stencil8` nor `X24_Stencil8` at any level (an M3 Max under
    /// macOS 26.7 and an M4 under 27.0.1 stop a texture, a depth and a stencil attachment of each
    /// as of no format; our probe and the consumer's M3 run), so
    /// `isDepth24Stencil8PixelFormatSupported` answers no.
    pub fn describes(self, pf: PixelFormat) -> bool {
        self.pixel_format_info(u64::from(pf.as_raw()))
            .describes_a_format()
    }

    /// How the release's Metal places a block-compressed texture's blocks of `family` in a heap
    /// texture's levels, where measured (`HeapBlocks`).
    pub(crate) fn heap_blocks(self, family: BlockFamily) -> Option<BlockPlacement> {
        let blocks = &self.row().heap_blocks;
        match family {
            BlockFamily::Bc => blocks.bc,
            BlockFamily::Etc => blocks.etc,
            BlockFamily::Astc => blocks.astc,
        }
    }

    /// How the release's Metal counts a block-compressed heap texture's tiles a row where its GPU
    /// reaches the texture, whose CPU copies count them by the family's placement
    /// (`heap_blocks`).
    pub(crate) fn heap_block_gpu_rows(self) -> GpuRows {
        self.row().heap_blocks.gpu_rows
    }

    /// The texels the release's Metal writes in a GPU blit through a losslessly compressed
    /// texture of `pf` (`compressed_blit.rs`): `CompressedBlit::of(pf)` where the release is
    /// measured, else `None` (the bytes kept).
    pub(crate) fn compressed_blit(self, pf: PixelFormat) -> Option<CompressedBlit> {
        self.row()
            .compressed_blits
            .then(|| CompressedBlit::of(pf))
            .flatten()
    }

    /// The alignment the release's `heapTextureSizeAndAlignWithDescriptor:` answers, with 0
    /// bytes, for a texture of `Unspecialized`, `RGBA16Float_XR`, `AnyDepth` or `AnyStencil` of
    /// a type with layers, faces or depth (`Row::no_texture_stack_alignment`).
    pub(crate) fn no_texture_stack_alignment(self) -> u64 {
        self.row().no_texture_stack_alignment
    }

    /// How the release's `generateMipmapsForTexture:` makes each level of a chain after the
    /// first on its Apple GPU (`MipmapFilterChain::plan`'s rule), which the device's mipmaps
    /// answer ([`BlitCommandEncoder::generate_mipmaps`](crate::BlitCommandEncoder::generate_mipmaps)).
    /// No Metal counterpart: Metal makes its own release's.
    pub fn mipmap_rule(self) -> MipmapRule {
        self.row().mipmaps
    }

    /// The newest level whose release is at or below macOS `version` (major, minor, patch), if
    /// any.
    pub(crate) fn at_or_below(version: [u32; 3]) -> Option<MetalLevel> {
        MetalLevel::ALL
            .into_iter()
            .rev()
            .find(|level| level.row().version <= version)
    }
}

impl fmt::Display for MetalLevel {
    /// `macOS 26.7`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "macOS {}", self.macos())
    }
}

/// The host a device is made on, as the level's choice reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Host {
    /// macOS, of this version (major, minor, patch), or `None` where it cannot be read.
    MacOS(Option<[u32; 3]>),
    /// Any other system.
    Other,
}

impl Host {
    /// The host the process runs on.
    pub(crate) fn this() -> Host {
        if cfg!(target_os = "macos") {
            Host::MacOS(product_version().as_deref().and_then(parse_version))
        } else {
            Host::Other
        }
    }
}

/// The host's macOS release, `kern.osproductversion` (the release's own version, whatever SDK the
/// process was built with), where it reads; `None` off macOS.
fn product_version() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let mut bytes = [0u8; 64];
        let mut length = bytes.len();
        // SAFETY: the name is a NUL-terminated string; `bytes` holds `length` bytes, which the
        // call writes at most and updates to what it wrote; no new value is given.
        let status = unsafe {
            libc::sysctlbyname(
                c"kern.osproductversion".as_ptr(),
                bytes.as_mut_ptr().cast(),
                &mut length,
                std::ptr::null_mut(),
                0,
            )
        };
        if status != 0 {
            return None;
        }
        let text = std::ffi::CStr::from_bytes_until_nul(&bytes[..length.min(bytes.len())]).ok()?;
        text.to_str().ok().map(str::to_owned)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// `major.minor[.patch]`, each a decimal number, as (major, minor, patch).
pub(crate) fn parse_version(text: &str) -> Option<[u32; 3]> {
    let mut parts = text.trim().split('.');
    let mut number = |required: bool| match parts.next() {
        Some(part) => part.parse::<u32>().ok(),
        None if required => None,
        None => Some(0),
    };
    let version = [number(true)?, number(true)?, number(false)?];
    parts.next().is_none().then_some(version)
}

/// The level a device answers at, and how it was chosen (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Chosen {
    /// The device's options name it.
    Asked(MetalLevel),
    /// The macOS host's own release.
    Host(MetalLevel),
    /// The newest measured release below the macOS host's, `host`: announced once when the
    /// device is made.
    Below {
        /// The host's version.
        host: [u32; 3],
        /// The level.
        level: MetalLevel,
    },
    /// The newest, on a host that is not macOS.
    NotMacOS(MetalLevel),
}

impl Chosen {
    /// The level the device answers at.
    pub(crate) fn level(self) -> MetalLevel {
        match self {
            Chosen::Asked(level)
            | Chosen::Host(level)
            | Chosen::Below { level, .. }
            | Chosen::NotMacOS(level) => level,
        }
    }

    /// The kind, object and reason of the device's announcement: for a level below the host's
    /// release, and for the default off macOS, which is no release of the host's.
    pub(crate) fn announcement(self) -> Option<(Degradation, (String, String))> {
        match self {
            Chosen::Below { host, level } => Some((
                Degradation::MetalLevelBelowHost,
                (
                    format!("the device, on macOS {}.{}.{}", host[0], host[1], host[2]),
                    format!(
                        "no Metal level is measured for the host's release: the device answers \
                         as {level}, the newest measured release below it"
                    ),
                ),
            )),
            Chosen::NotMacOS(level) => Some((
                Degradation::MetalLevelNotMacOS,
                (
                    "the device, on a host that is not macOS".to_owned(),
                    format!(
                        "no options name a Metal level and the host has no macOS release to \
                         answer as: the device answers as {level}, the newest measured release; \
                         a consumer whose guest runs another release names it \
                         (`DeviceOptions::metal_level`)"
                    ),
                ),
            )),
            Chosen::Asked(_) | Chosen::Host(_) => None,
        }
    }
}

/// The level of a device made on `host` with `asked` named by its options, or, on a macOS host
/// below every measured release or whose release cannot be read, `Unproven` for `call` (the
/// module docs).
pub(crate) fn choose(
    asked: Option<MetalLevel>,
    host: Host,
    call: &'static str,
) -> Result<Chosen, Error> {
    if let Some(level) = asked {
        return Ok(Chosen::Asked(level));
    }
    match host {
        Host::Other => Ok(Chosen::NotMacOS(MetalLevel::NEWEST)),
        Host::MacOS(version) => {
            let unmeasured = || {
                Error::Unproven(Unproven::new(
                    call,
                    Problem::HostReleaseUnmeasured { macos: version },
                ))
            };
            let host = version.ok_or_else(unmeasured)?;
            let level = MetalLevel::at_or_below(host).ok_or_else(unmeasured)?;
            Ok(if level.row().version == host {
                Chosen::Host(level)
            } else {
                Chosen::Below { host, level }
            })
        }
    }
}
