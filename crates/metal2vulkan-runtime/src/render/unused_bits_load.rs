//! The load pass of an extended-range, padded 10-bit or bfloat color attachment, and under the
//! exact tile store of a signed normalized or `RGB9E5Float` one.
//!
//! - **Metal's answer** (on an Apple M4 under macOS 27.0.1 and on the consumer's Apple M3 Max under
//!   macOS 26.7, alike; on both). An Apple GPU stores a render pass's attachments a 32 x 32 tile at
//!   a time. A pass that loads a color attachment of `BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`,
//!   `R10Unorm_X6`, `RG10Unorm_X12` or one of their `_sRGB` twins and records a draw writes every
//!   texel of each tile it stores back: a channel a fragment writes takes the write, and every
//!   other texel keeps its codes with the bits they leave unused made canonical
//!   (`bind::unused_bits_module`: `BGR10_XR`'s bits 30 and 31 ones, `BGRA10_XR`'s low six bits of
//!   each field zeros and its alpha code clamped to `[384, 894]`, the two-plane color word's top
//!   two bits zeros, a padded word's low six bits zeros, the padded `_sRGB` twins' codes through
//!   Metal's table). That includes the texels no fragment covers, fragments that discard or fail a
//!   test, and channels a write mask leaves out. A pass that loads a bfloat color attachment
//!   (`R16Bfloat`, `RG16Bfloat`, `RGBA16Bfloat`) and draws does so too, its tiles holding `half`
//!   values: every code of each tile it stores back goes through `half` to nearest even and back to
//!   the nearest bfloat code, a NaN to `0x7fc0` (`ExtendedRange::load_code`; on an Apple M4
//!   under macOS 27.0.1, every code of each format: 56,829 move, and only in the tiles the draws'
//!   primitives reach). The tiles it stores are those a draw's primitives reach (the half of a rect
//!   whose edge lies on x = 32 reaches the tiles right of it; a zero-area rect or one off the
//!   target reaches none), within the render area rounded up to 32 texels a side from the origin
//!   and kept within the texture; every tile of it where a color attachment of the pass is cleared
//!   and stored, a draw of nothing (of no vertices, say) too. A pass that draws nothing keeps every
//!   texel.
//! - **The runtime's answer.** Before a pass that loads such an attachment, stores it and records
//!   a draw call or an execution of render commands, ahead of the pass's other steps (a shadow's
//!   or a stand-in's copy in among them), the encoder copies the region's texels into a scratch
//!   buffer, rewrites the words of each tile a [`TileReach`] names there (one dispatch of
//!   `bind::unused_bits_module` of the attachment's form a band of rows), and copies them back.
//!   The texels come from the buffer's bytes of a texture over a buffer, a surface's plane, or a
//!   texture buffer's attachment twin (`Texture::attachment_twin`), a 2D texture or a 2D array
//!   view of one layer (a two-plane format's plane 0 color words alone, `TexelWords::ColorPlane`),
//!   and from the image of any other single-sampled texture the runtime holds as an image of its
//!   own (`TexelWords::Raw`), every layer the pass renders of its level. The pass then loads
//!   Metal's texels, and a write mask, a discard, a test and a texel no draw covers keep them as
//!   Metal's pass does. A pass whose store discards the attachment (`DontCare`, which leaves the
//!   bytes as they were) is left alone.
//! - **A rotated texture's frame**. A rotated texture renders through its twin of the pass's frame
//!   (`rotated.rs`), an image the shadow's copy in fills from the plane's bytes by Metal's load map
//!   (`Turn::Load`, in a pass that draws). Its tiles are the frame's, so the load pass rewrites
//!   that image instead, right after the pass's shadows' copies in ([`record_frames`]) and before
//!   its stand-ins' copies and clears: the frame's texels of each tile, mapped back through the
//!   rotation by the copy out (`Turn::Store`) as the pass's own writes are, which leaves the
//!   plane's words as Metal's pass leaves them, by Metal's tile rule and rotation maps (each
//!   measured on its own).
//! - **The reach** ([`TileReach`], from the pass's `clear_reach::PassReach`). None in a pass whose
//!   draws reach no tile (each draws nothing, of no vertex or instance or through a pipeline that
//!   rasterizes nothing, and no attachment is cleared and stored): the pass keeps every texel, as
//!   Metal's does. Every tile of the region, exact, where an attachment of a pass of one layer is
//!   cleared and stored. Elsewhere the tiles the draws' viewports and scissors reach
//!   (`clear_reach::DrawnArea`; every tile through a rasterization-rate map, an execution of render
//!   commands or a tile dispatch), and `unreached-tiles-rewritten` names the attachments, whose
//!   tiles there no draw reaches are rewritten here where Metal keeps them; but where capture
//!   replay found the tiles the draws' primitives reach (`render/capture.rs`, Metal's, announced by
//!   the command buffer where a primitive's reach was not measured).
//! - **The exact tile store** (`TileStorePolicy::Exact`). The same pass, once it draws, writes a
//!   single-sampled signed normalized or `RGB9E5Float` attachment's tiles back through their
//!   values (an Apple M3 Max under macOS 27.0.1): a channel's minimum code (-128, or -32768) is
//!   stored as the code above it, and a word whose largest mantissa is below 256 under an exponent
//!   above 0 normalized (`bind::unused_bits_module` of the format). Under the exact tile store the
//!   load pass takes such an attachment as it takes an extended-range one, where the driver's pass
//!   does not store them so already (`DriverFact::TileStoreIsMetals`: MoltenVK's pass is a Metal
//!   pass); under the default, `TileStorePolicy::Device`, it leaves them to the device's load and
//!   store, and a draw announces `snorm-minimum-kept` or `shared-exponent-kept`
//!   ([`snorm_minimum_kept`], [`shared_exponent_kept`]), as it does for one the load pass does not
//!   take.
//! - **Not here** (`degrade_unused_bits` announces `extended-range-bits-kept` at each draw): a
//!   multisampled attachment (no copy reaches its samples), a texture held as a shadow of another
//!   image or of a heap's bytes, a sparse one, and one whose image the device does not copy both
//!   ways. A padded format has none of these (it is a buffer's or a surface's bytes, single
//!   sampled).

use super::capture::CapturedReach;
use crate::buffer_texture::Texels;
use crate::clear_reach::{DrawnTiles, PassReach};
use crate::degradation::Degradation;
use crate::device::functions::Vk;
use crate::device::Device;
use crate::encoder::EncoderCore;
use crate::error::{internal, Error};
use crate::owned_kernel::{OwnedModule, OwnedSets};
use crate::render_pass::{ColorTarget, PassLayout, Target};
use crate::render_values::{LoadAction, StoreAction};
use crate::shadow::ShadowTarget;
use crate::texture::Texture;
use crate::tracker::{Access, Barrier, Use, UseRange};
use crate::values::ResourceOptions;
use ash::vk;
use metal2vulkan::bind::{TexelWords, UnusedBitsModule};
use metal2vulkan::device::{DriverFact, TileStorePolicy};
use metal2vulkan::metal::format::ExtendedRange;
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::{ImageType, ImageUsageFlags};

/// The side, in texels, of the tiles Metal stores.
const TILE: u32 = UnusedBitsModule::TILE;

/// The scratch's largest band of rows: 32 MiB.
const MAX_SCRATCH: u64 = 32 << 20;

/// The tiles of a pass's region whose texels the load pass rewrites: those Metal's tile store
/// writes back, the tiles its draws reach (the module docs).
#[derive(Clone, Copy, Debug)]
pub(super) enum TileReach<'a> {
    /// Every tile: Metal's answer where `exact`, else where Metal's are the tiles the draws'
    /// primitives reach, until those are captured.
    Every { exact: bool },
    /// The tiles the draws' viewports and scissors reach (`clear_reach::DrawnTiles`, 32 by 32):
    /// Metal's answer in each tile outside them, every tile inside rewritten.
    Within(&'a DrawnTiles),
    /// The tiles the draws' primitives reach, as capture replay binned them before the pass
    /// (`render/capture.rs`): Metal's answer, but where a primitive's reach was not measured,
    /// where the mask holds every tile and the command buffer announces it as it completes.
    Captured(&'a CapturedReach),
}

/// What fills a load pass's tile mask ([`TileReach::mask`]).
enum Mask {
    /// Every tile.
    Every,
    /// These words.
    Words(Vec<u32>),
    /// The words of capture replay's mask: its `VkBuffer`, offset and bytes.
    Captured((vk::Buffer, u64, u64)),
}

impl<'a> TileReach<'a> {
    /// The tiles the load pass rewrites in a pass whose draws reach `reach`: those `captured`
    /// names where capture replay binned them, else within `tiles` where those bound them; none
    /// where they reach none (`clear_reach.rs`).
    pub(super) fn of(
        reach: PassReach,
        tiles: Option<&'a DrawnTiles>,
        captured: Option<&'a CapturedReach>,
    ) -> Option<TileReach<'a>> {
        match (reach, captured, tiles) {
            (PassReach::Nothing, ..) => None,
            (PassReach::Every, ..) => Some(TileReach::Every { exact: true }),
            (PassReach::Draws, Some(captured), _) => Some(TileReach::Captured(captured)),
            (PassReach::Draws, None, None) => Some(TileReach::Every { exact: false }),
            (PassReach::Draws, None, Some(tiles)) => Some(TileReach::Within(tiles)),
        }
    }

    /// Whether the reach is Metal's (a captured one's where its primitives' reach was measured,
    /// which the command buffer announces otherwise).
    fn exact(self) -> bool {
        match self {
            TileReach::Every { exact } => exact,
            TileReach::Within(_) => false,
            TileReach::Captured(_) => true,
        }
    }

    /// The mask (one bit a tile, `bind::UnusedBitsModule`'s order) of `layers` layers of
    /// `tile_rows` rows of `tiles_per_row` tiles, of `bytes` bytes. A captured mask is binned in
    /// the same order over the render area (`bind::TileReachModule`, of one layer), whose tiles
    /// are the region's: its bytes are the mask's, else the load pass is refused.
    fn mask(
        self,
        tiles_per_row: u64,
        tile_rows: u64,
        (layers, bytes): (u64, u64),
    ) -> Result<Mask, Error> {
        // A pass's tiles are 32 by 32 at one sample, the only count the load pass takes.
        let tiles = match self {
            TileReach::Within(tiles) if tiles.tile() == [TILE, TILE] => tiles,
            TileReach::Captured(captured) => {
                let (_, _, captured_bytes) = captured.mask;
                return match layers == 1 && captured_bytes == bytes {
                    true => Ok(Mask::Captured(captured.mask)),
                    false => Err(internal("a captured tile mask of another grid")),
                };
            }
            _ => return Ok(Mask::Every),
        };
        let per_layer = tiles_per_row * tile_rows;
        let mut words = vec![0u32; (layers * per_layer).div_ceil(32) as usize];
        for layer in 0..layers {
            for ty in 0..tile_rows {
                for tx in 0..tiles_per_row {
                    let reached = match (u32::try_from(tx), u32::try_from(ty)) {
                        (Ok(tx), Ok(ty)) => tiles.contains(tx, ty),
                        _ => true,
                    };
                    if reached {
                        let t = (layer * per_layer + ty * tiles_per_row + tx) as usize;
                        words[t / 32] |= 1 << (t % 32);
                    }
                }
            }
        }
        Ok(Mask::Words(words))
    }
}

/// Record `mask` (of `bytes`) into `buffer` at `offset`.
///
/// # Safety
///
/// `command_buffer` is recording, outside a render pass, and `offset..offset + bytes` lies in
/// `buffer`, a multiple of 4 from a multiple of 4, and is not in use; a mask's words are
/// `bytes / 4` words, and a captured mask's bytes are `bytes` of a live buffer, written before a
/// barrier the command buffer recorded already.
unsafe fn record_mask(
    raw: &Vk,
    command_buffer: vk::CommandBuffer,
    (buffer, offset, bytes): (vk::Buffer, u64, u64),
    mask: &Mask,
) {
    match mask {
        // SAFETY: the caller's contract.
        Mask::Every => unsafe {
            raw.cmd_fill_buffer(command_buffer, buffer, offset, bytes, u32::MAX)
        },
        Mask::Captured((from, at, _)) => {
            let region = vk::BufferCopy {
                src_offset: *at,
                dst_offset: offset,
                size: bytes,
            };
            // SAFETY: the caller's contract.
            unsafe { raw.cmd_copy_buffer(command_buffer, *from, buffer, &[region]) };
        }
        // At most 65536 bytes an update, as `vkCmdUpdateBuffer` takes.
        Mask::Words(words) => {
            for (i, chunk) in words.chunks(1 << 14).enumerate() {
                let bytes: Vec<u8> = chunk.iter().flat_map(|w| w.to_le_bytes()).collect();
                let at = offset + (i as u64) * (4 << 14);
                // SAFETY: the caller's contract; the chunk lies in the mask's bytes.
                unsafe { raw.cmd_update_buffer(command_buffer, buffer, at, &bytes) };
            }
        }
    }
}

/// Where the load pass finds a taken attachment's texels.
enum Source<'a> {
    /// The rows of a buffer's bytes, held as the words named.
    Bytes(&'a Texels, TexelWords),
    /// The texture's own image, its raw texels.
    Image,
    /// A rotated texture's image of the pass's frame, its raw texels, rewritten after the pass's
    /// shadows' copies in ([`record_frames`]).
    Frame,
}

/// A color attachment the load pass takes: its format, the target whose texels it rewrites (the
/// texture's own, where a stand-in renders for it), and where it finds them.
struct Taken<'a> {
    format: PixelFormat,
    target: &'a Target,
    source: Source<'a>,
}

/// The texture's own target of `color`: a stand-in's (`attachment_stand_in.rs`) is the one it
/// stands in for.
fn own(color: &ColorTarget) -> &Target {
    color.stand_in.as_ref().map_or(&color.target, |s| &s.own)
}

/// Whether the load pass rewrites a texture of `texture`'s format: an extended-range, padded or
/// bfloat one's always, and a signed normalized ([`SNORM_MINIMUM`]) or `RGB9E5Float` one's under
/// the exact tile store (`TileStorePolicy::Exact`) on a driver whose store is not Metal's already
/// (`DriverFact::TileStoreIsMetals`).
fn rewrites(texture: &Texture) -> bool {
    let format = texture.pixel_format();
    if ExtendedRange::of(format).is_some() {
        return true;
    }
    let device = texture.device();
    let stored = SNORM_MINIMUM.contains(&format) || format == PixelFormat::RGB9E5Float;
    stored
        && device.compile_device().policy().tile_store == TileStorePolicy::Exact
        && !metals_store(device)
}

/// Whether `device`'s render pass stores a signed normalized minimum code and an `RGB9E5Float`
/// word as Metal's tile store does (`DriverFact::TileStoreIsMetals`: MoltenVK's pass is a Metal
/// pass).
pub(super) fn metals_store(device: &Device) -> bool {
    device
        .properties()
        .driver_facts
        .contains(&DriverFact::TileStoreIsMetals)
}

/// Whether `color` is of a format the load pass rewrites ([`rewrites`]), loaded and not
/// discarded at the store; `late` counts a store action the encoder decides later as storing.
fn loaded(color: &ColorTarget, late: bool) -> bool {
    let discarded = color.store == StoreAction::DontCare && !(late && color.late_store);
    rewrites(&own(color).texture)
        && color.load == LoadAction::Load
        && !color.memoryless
        && !discarded
}

/// The color attachment `color` as the load pass takes it (the module docs), or `None` where it
/// does not.
fn takes(color: &ColorTarget, late: bool) -> Option<Taken<'_>> {
    if !loaded(color, late) {
        return None;
    }
    let target = own(color);
    let texture = &target.texture;
    let format = texture.pixel_format();
    let turned = texture
        .shadow()
        .is_some_and(|copies| matches!(copies.target(), Some(ShadowTarget::Rotated { .. })));
    if target.samples != 1 {
        return None;
    }
    let copied = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    let copies_both_ways = || {
        texture
            .image_plan()
            .is_ok_and(|plan| plan.usage.contains(copied))
    };
    // A rotated texture's frame, which its shadow's copy in fills.
    if turned {
        return copies_both_ways().then_some(Taken {
            format,
            target,
            source: Source::Frame,
        });
    }
    if let Some(texels) = texture.buffer_texels() {
        let words = match ExtendedRange::of(format) {
            Some(ExtendedRange::Planes { .. }) => TexelWords::ColorPlane,
            _ => TexelWords::Raw,
        };
        let flat = matches!(
            texture.texture_type(),
            TextureType::Type2D | TextureType::Type2DArray
        ) && target.layers == 1;
        let texel = u64::from(UnusedBitsModule::texel_bytes(format, words)?);
        let rows_of_texels = texels.layout.row_bytes == u64::from(target.extent[0]) * texel;
        return (flat && rows_of_texels).then_some(Taken {
            format,
            target,
            source: Source::Bytes(texels, words),
        });
    }
    let image = texture.shadow().is_none() && !texture.is_sparse() && copies_both_ways();
    image.then_some(Taken {
        format,
        target,
        source: Source::Image,
    })
}

/// Whether capture replay's reach has a consumer in a pass of `layout` (`render/capture.rs`): the
/// load pass takes one of its color attachments, as the pass begins (`ended` false) or as it ends
/// (`ended`: its store actions set, one still `Unknown` discarding), and no rotated texture's
/// frame among them. A frame's twin is copied out whole (`rotated.rs`), so a tile no primitive
/// reaches would store the texels its copy in loaded where Metal's pass keeps the plane's (a
/// mirror at rotation 4): such a pass keeps the host's reach, announced.
pub(super) fn consumes_capture(layout: &PassLayout, ended: bool) -> bool {
    let mut taken = layout
        .colors
        .iter()
        .flatten()
        .filter(|color| !ended || color.store != StoreAction::Unknown)
        .filter_map(|color| takes(color, !ended))
        .peekable();
    taken.peek().is_some() && taken.all(|taken| !matches!(taken.source, Source::Frame))
}

/// Whether `color` is of an extended-range, padded or bfloat format, loaded and not discarded at
/// the store, where the load pass does not take it (`degrade_unused_bits`).
pub(super) fn untaken(color: &ColorTarget) -> bool {
    ExtendedRange::of(own(color).texture.pixel_format()).is_some()
        && loaded(color, true)
        && takes(color, true).is_none()
}

/// The signed normalized color formats whose minimum code Metal's tile store rewrites, as
/// measured on an Apple M3 Max (`snorm-minimum-kept`).
const SNORM_MINIMUM: [PixelFormat; 6] = [
    PixelFormat::R8Snorm,
    PixelFormat::RG8Snorm,
    PixelFormat::RGBA8Snorm,
    PixelFormat::R16Snorm,
    PixelFormat::RG16Snorm,
    PixelFormat::RGBA16Snorm,
];

/// Whether `color` is a single-sampled attachment of a signed normalized format
/// ([`SNORM_MINIMUM`]), loaded and not discarded at the store, where the load pass does not take
/// it: Metal's pass, once it draws, writes each tile a draw reaches back through the channels'
/// values, so a minimum code (-128, or -32768) is stored as the code above it, where the device's
/// pass keeps the bytes its draws do not write (`degrade_snorm_minimum`, which reads
/// `DriverFact::TileStoreIsMetals`: a driver whose pass stores them as Metal's). The load pass
/// rewrites them under the exact tile store (`TileStorePolicy::Exact`) where it takes the
/// attachment.
pub(super) fn snorm_minimum_kept(color: &ColorTarget) -> bool {
    let target = own(color);
    let discarded = color.store == StoreAction::DontCare && !color.late_store;
    SNORM_MINIMUM.contains(&target.texture.pixel_format())
        && target.samples == 1
        && color.load == LoadAction::Load
        && !color.memoryless
        && !discarded
        && takes(color, true).is_none()
}

/// Whether `color` is a single-sampled `RGB9E5Float` attachment, loaded and not discarded at the
/// store, where the load pass does not take it: Metal's pass, once it draws, writes each tile a
/// draw reaches back through its texels' values, so a word whose largest mantissa is below 256
/// under an exponent above 0 (one no Metal write leaves) is stored normalized, its mantissas
/// shifted up as its exponent steps down, where the device's pass keeps the bytes its draws do
/// not write (`degrade_shared_exponent`, which reads `DriverFact::TileStoreIsMetals`). The load
/// pass rewrites them under the exact tile store (`TileStorePolicy::Exact`) where it takes the
/// attachment.
pub(super) fn shared_exponent_kept(color: &ColorTarget) -> bool {
    let target = own(color);
    let discarded = color.store == StoreAction::DontCare && !color.late_store;
    target.texture.pixel_format() == PixelFormat::RGB9E5Float
        && target.samples == 1
        && color.load == LoadAction::Load
        && !color.memoryless
        && !discarded
        && takes(color, true).is_none()
}

/// The color attachments of a pass's layout that each of a draw's kept-attachment notices names, as
/// `color {index}, {format:?}`: the extended-range or padded ones the load pass does not take
/// ([`untaken`], `degrade_unused_bits`) and, on a driver whose pass does not store as Metal's
/// ([`metals_store`]), the signed normalized ones ([`snorm_minimum_kept`],
/// `degrade_snorm_minimum`) and the `RGB9E5Float` ones ([`shared_exponent_kept`],
/// `degrade_shared_exponent`). They read the layout's colors, their textures' formats, samples,
/// shadows and image plans, and the device's tile store policy and driver facts; all but the
/// layout are fixed at creation, so an encoder decides them once for each layout it records
/// under.
#[derive(Debug, Default, PartialEq)]
pub(super) struct KeptAttachments {
    pub(super) unused_bits: Vec<String>,
    pub(super) snorm_minimum: Vec<String>,
    pub(super) shared_exponent: Vec<String>,
}

impl KeptAttachments {
    /// The kept attachments of `layout`, a pass of `device`'s.
    pub(super) fn of(layout: &PassLayout, device: &Device) -> Self {
        let metals = metals_store(device);
        let mut kept = Self::default();
        for (index, color) in layout.colors.iter().enumerate() {
            let Some(color) = color else {
                continue;
            };
            let name = || format!("color {index}, {:?}", color.target.texture.pixel_format());
            if untaken(color) {
                kept.unused_bits.push(name());
            }
            if !metals && snorm_minimum_kept(color) {
                kept.snorm_minimum.push(name());
            }
            if !metals && shared_exponent_kept(color) {
                kept.shared_exponent.push(name());
            }
        }
        kept
    }
}

/// The texels of one attachment's region: its columns, the rows of each layer, its layers, and
/// the bytes of a texel and of each row in the scratch (the texels', then any padding to a
/// word).
#[derive(Clone, Copy)]
struct Region {
    columns: u64,
    rows: u64,
    layers: u64,
    texel: u64,
    stride: u64,
}

impl Region {
    /// The bytes of each row's texels.
    fn row_bytes(&self) -> u64 {
        self.columns * self.texel
    }
}

/// One band of a region's rows: its first row (counted across the layers) and its rows.
#[derive(Clone, Copy)]
struct Band {
    first_row: u64,
    rows: u64,
}

/// Record, into `core` before the pass of `layout` whose store actions are settled, the load
/// pass of each color attachment it takes over the tiles of `reach` (the module docs), but for a
/// rotated texture's frame ([`record_frames`]), and announce `unreached-tiles-rewritten` where
/// `reach` is not Metal's, naming every attachment either rewrites. The caller calls it only for
/// a pass that recorded a draw call or an execution of render commands, whose draws reach a
/// tile.
pub(super) fn record(
    core: &mut EncoderCore,
    layout: &PassLayout,
    reach: TileReach<'_>,
) -> Result<(), Error> {
    let rewritten: Vec<String> = layout
        .colors
        .iter()
        .enumerate()
        .filter_map(|(index, color)| {
            let taken = takes(color.as_ref()?, false)?;
            let format = taken.target.texture.pixel_format();
            Some(format!("color {index}, {format:?}"))
        })
        .collect();
    record_taken(core, layout, reach, false)?;
    if !rewritten.is_empty() && !reach.exact() {
        core.device()
            .degrade(Degradation::UnreachedTilesRewritten, || {
                (
                    format!(
                        "a render pass that loads {} and draws",
                        rewritten.join(" and ")
                    ),
                    "the pass rewrites every 32x32 tile of its render area, where Metal's \
                     rewrites only the tiles a draw's primitives reach (or every tile where a \
                     color attachment is cleared): in tiles no draw reaches the unused bits are \
                     made canonical and BGRA10_XR alpha codes clamped to [384, 894], a padded \
                     word's low six bits zeroed (an _sRGB twin's code through Metal's table) and, \
                     under the exact tile store, a signed normalized minimum code raised and an \
                     RGB9E5Float word normalized, where Metal keeps those texels' bytes"
                        .to_owned(),
                )
            });
    }
    Ok(())
}

/// Record, into `core` right after the shadows' copies in of the pass of `layout` (and before
/// its stand-ins' copies and clears), the load pass of each rotated texture's frame it takes over
/// the tiles of `reach` (the module docs). [`record`] announces them. The caller calls it only
/// for a pass that recorded a draw call or an execution of render commands, whose draws reach a
/// tile.
pub(super) fn record_frames(
    core: &mut EncoderCore,
    layout: &PassLayout,
    reach: TileReach<'_>,
) -> Result<(), Error> {
    record_taken(core, layout, reach, true)
}

/// Record the load pass of each color attachment of `layout` taken from a rotated texture's
/// frame (`frames`), or of each other one taken.
fn record_taken(
    core: &mut EncoderCore,
    layout: &PassLayout,
    reach: TileReach<'_>,
    frames: bool,
) -> Result<(), Error> {
    for color in layout.colors.iter().flatten() {
        let Some(taken) = takes(color, false) else {
            continue;
        };
        if matches!(taken.source, Source::Frame) != frames {
            continue;
        }
        let target = taken.target;
        let [width, height] = target.extent;
        let words = match taken.source {
            Source::Bytes(_, words) => words,
            Source::Image | Source::Frame => TexelWords::Raw,
        };
        let texel = UnusedBitsModule::texel_bytes(taken.format, words)
            .ok_or_else(|| internal("an unused bits attachment without its words"))?;
        let columns = u64::from(layout.extent[0].next_multiple_of(TILE).min(width));
        let rows = u64::from(layout.extent[1].next_multiple_of(TILE).min(height));
        let texel = u64::from(texel);
        let region = Region {
            columns,
            rows,
            layers: u64::from(target.layers),
            texel,
            stride: (columns * texel).next_multiple_of(4),
        };
        if columns == 0 || rows == 0 || region.layers == 0 {
            continue;
        }
        rewrite(core, &taken, words, region, reach)?;
        core.device()
            .shared()
            .counts
            .unused_bits_load_recorded(columns * rows * region.layers);
    }
    Ok(())
}

/// Rewrite the words of `taken`'s `region`, held as `words`, in the tiles of `reach`, as Metal's
/// tile store leaves them.
fn rewrite(
    core: &mut EncoderCore,
    taken: &Taken<'_>,
    words: TexelWords,
    region: Region,
    reach: TileReach<'_>,
) -> Result<(), Error> {
    let device = core.device().clone();
    let limits = &device.properties().description.limits;
    // A storage buffer's binding offset, and an image copy's buffer offset (a texel's bytes).
    let align = limits.min_storage_buffer_offset_alignment.max(8);
    let per_dispatch =
        u64::from(UnusedBitsModule::LOCAL_SIZE) * u64::from(limits.max_compute_work_group_count[0]);
    let most = MAX_SCRATCH
        .min(u64::from(limits.max_storage_buffer_range))
        .min(4 * per_dispatch);
    let all_rows = region.rows * region.layers;
    let band_rows = (most / region.stride).min(all_rows);
    if band_rows == 0 {
        return Err(internal("an unused bits row past a storage buffer's range"));
    }
    let bands: Vec<Band> = (0..all_rows)
        .step_by(band_rows as usize)
        .map(|first_row| Band {
            first_row,
            rows: band_rows.min(all_rows - first_row),
        })
        .collect();
    let tiles_per_row = region.columns.div_ceil(u64::from(TILE));
    let tiles = region.layers * region.rows.div_ceil(u64::from(TILE)) * tiles_per_row;
    let mask_bytes = 4 * tiles.div_ceil(32);
    let tile_rows = region.rows.div_ceil(u64::from(TILE));
    let mask = reach.mask(tiles_per_row, tile_rows, (region.layers, mask_bytes))?;
    let narrow = |v: u64| u32::try_from(v).map_err(|_| internal("an unused bits region past u32"));
    let shapes: Vec<[u8; 16]> = bands
        .iter()
        .map(|band| {
            Ok(UnusedBitsModule::shape(
                narrow(region.stride / 4)?,
                narrow(band.first_row)?,
                narrow(region.rows)?,
                narrow(tiles_per_row)?,
            ))
        })
        .collect::<Result<_, Error>>()?;
    // The scratch: a band's rows, the mask, then each band's shape.
    let words_bytes = (band_rows * region.stride).next_multiple_of(align);
    let mask_at = words_bytes;
    let shapes_at = mask_at + mask_bytes.next_multiple_of(align);
    let slot = 16u64.next_multiple_of(align);
    let scratch_bytes = shapes_at + slot * bands.len() as u64;
    let scratch = device.own_buffer(scratch_bytes, ResourceOptions::private())?;
    let (raw_scratch, at) = scratch.at(0);
    if !at.is_multiple_of(align) {
        return Err(internal("an unused bits scratch off its alignment"));
    }
    let kernel = device.owned_kernel(OwnedModule::UnusedBits(taken.format, words))?;
    let ranges: Vec<_> = bands
        .iter()
        .enumerate()
        .map(|(b, band)| {
            vec![
                (raw_scratch, at, band.rows * region.stride),
                (raw_scratch, at + mask_at, mask_bytes),
                (raw_scratch, at + shapes_at + slot * b as u64, 16),
            ]
        })
        .collect();
    let sets = OwnedSets::write(&device, &kernel, &ranges)?;
    let groups: Vec<u32> = bands
        .iter()
        .map(|band| {
            narrow(
                (band.rows * region.stride / 4).div_ceil(u64::from(UnusedBitsModule::LOCAL_SIZE)),
            )
        })
        .collect::<Result<_, Error>>()?;
    let access = Access::new(
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
    );
    let target = taken.target;
    let (copies, uses) = match taken.source {
        Source::Bytes(texels, _) => {
            let layout = &texels.layout;
            let (raw_buffer, base) = texels.buffer.at(layout.offset);
            let row_bytes = region.row_bytes();
            let copies: Vec<Vec<vk::BufferCopy>> = bands
                .iter()
                .map(|band| {
                    (0..band.rows)
                        .map(|row| vk::BufferCopy {
                            src_offset: base + (band.first_row + row) * layout.row_pitch,
                            dst_offset: at + row * region.stride,
                            size: row_bytes,
                        })
                        .collect()
                })
                .collect();
            let span =
                layout.offset..layout.offset + (region.rows - 1) * layout.row_pitch + row_bytes;
            let (identity, range) = texels.buffer.tracked(span);
            let uses = vec![Use::writes(
                identity.key,
                identity.tracked,
                UseRange::Bytes(range),
                access,
            )];
            core.retain(texels.buffer.clone());
            (Copies::Bytes(raw_buffer, copies), uses)
        }
        Source::Image | Source::Frame => {
            let texture = &target.texture;
            let image = texture.raw_image()?;
            let deep = texture.image_plan()?.image_type == ImageType::TYPE_3D;
            let regions: Vec<Vec<vk::BufferImageCopy>> = bands
                .iter()
                .map(|band| image_regions(target, region, *band, at, deep))
                .collect();
            let level = target.level..target.level + 1;
            let layers = target.layer..target.layer + target.layers;
            let uses = vec![texture.image_use(level, layers, access, true)?];
            core.retain(texture.clone());
            (Copies::Image(image, regions), uses)
        }
    };
    let descriptor_sets = sets.sets.clone();
    let stages = vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER;
    core.command(stages, &uses, |raw, command_buffer| {
        // SAFETY: the command buffer is recording outside a render pass (the pass records after
        // it), and only the encoder records it; the texels' buffer or image, the scratch and the
        // sets are retained until it completes; every copy lies inside the texels (the target's
        // rows, its level and layers) and the scratch (a band's rows), and every binding inside
        // the scratch; each step reads what the one before it wrote past a barrier, and the
        // scratch's rows are written again only past a barrier after the band before is done
        // with them. The image rests in `GENERAL`.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            record_mask(
                raw,
                command_buffer,
                (raw_scratch, at + mask_at, mask_bytes),
                &mask,
            );
            for (b, shape) in shapes.iter().enumerate() {
                let offset = at + shapes_at + slot * b as u64;
                raw.cmd_update_buffer(command_buffer, raw_scratch, offset, shape);
            }
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
            for (b, (set, groups)) in descriptor_sets.iter().zip(&groups).enumerate() {
                Barrier::everything().record(raw, command_buffer);
                copies.to_scratch(raw, command_buffer, raw_scratch, b);
                Barrier::everything().record(raw, command_buffer);
                raw.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.layout,
                    0,
                    std::slice::from_ref(set),
                    &[],
                );
                raw.cmd_dispatch(command_buffer, *groups, 1, 1);
                Barrier::everything().record(raw, command_buffer);
                copies.back(raw, command_buffer, raw_scratch, b);
            }
            Barrier::everything().record(raw, command_buffer);
        }
    });
    core.retain(scratch);
    core.retain(sets);
    Ok(())
}

/// Each band's copies between the texels and the scratch, into it (and back, swapped).
enum Copies {
    /// Each row of a buffer's bytes.
    Bytes(vk::Buffer, Vec<Vec<vk::BufferCopy>>),
    /// Each layer's rows of an image's level.
    Image(vk::Image, Vec<Vec<vk::BufferImageCopy>>),
}

impl Copies {
    /// Record band `b`'s copy into `scratch`.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording, outside a render pass; the copies lie in their resources,
    /// which are live until it completes, and an image rests in `GENERAL`.
    unsafe fn to_scratch(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        scratch: vk::Buffer,
        b: usize,
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            match self {
                Copies::Bytes(buffer, copies) => {
                    raw.cmd_copy_buffer(command_buffer, *buffer, scratch, &copies[b])
                }
                Copies::Image(image, regions) => raw.cmd_copy_image_to_buffer(
                    command_buffer,
                    *image,
                    vk::ImageLayout::GENERAL,
                    scratch,
                    &regions[b],
                ),
            }
        }
    }

    /// Record band `b`'s copy back from `scratch`.
    ///
    /// # Safety
    ///
    /// As [`to_scratch`](Self::to_scratch).
    unsafe fn back(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        scratch: vk::Buffer,
        b: usize,
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            match self {
                Copies::Bytes(buffer, copies) => {
                    let back: Vec<vk::BufferCopy> = copies[b]
                        .iter()
                        .map(|c| vk::BufferCopy {
                            src_offset: c.dst_offset,
                            dst_offset: c.src_offset,
                            size: c.size,
                        })
                        .collect();
                    raw.cmd_copy_buffer(command_buffer, scratch, *buffer, &back)
                }
                Copies::Image(image, regions) => raw.cmd_copy_buffer_to_image(
                    command_buffer,
                    scratch,
                    *image,
                    vk::ImageLayout::GENERAL,
                    &regions[b],
                ),
            }
        }
    }
}

/// The copies between `target`'s level and the scratch at `at` of `band`'s rows of `region`:
/// one per layer the band meets, a 3D image's depth planes (`deep`) by its z offset, another's
/// array layers by its layer.
fn image_regions(
    target: &Target,
    region: Region,
    band: Band,
    at: u64,
    deep: bool,
) -> Vec<vk::BufferImageCopy> {
    let mut regions = Vec::new();
    let mut row = band.first_row;
    let end = band.first_row + band.rows;
    while row < end {
        let layer = row / region.rows;
        let y = row % region.rows;
        let rows = (region.rows - y).min(end - row);
        let layer = target.layer + layer as u32;
        let (base_array_layer, z) = match deep {
            true => (0, layer as i32),
            false => (layer, 0),
        };
        regions.push(vk::BufferImageCopy {
            buffer_offset: at + (row - band.first_row) * region.stride,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: target.level,
                base_array_layer,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: 0,
                y: y as i32,
                z,
            },
            image_extent: vk::Extent3D {
                width: region.columns as u32,
                height: rows as u32,
                depth: 1,
            },
        });
        row += rows;
    }
    regions
}
