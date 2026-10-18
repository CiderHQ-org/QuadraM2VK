//! `RenderPassDescriptor` (`MTLRenderPassDescriptor`) and the checks a render encoder makes of one
//! when it opens.
//!
//! The descriptor is a plain value with Metal's defaults (measured on an Apple M4 with macOS
//! 27.0.1): a color attachment loads `DontCare`, stores `Store` and clears to opaque black; the
//! depth and stencil attachments load `Clear`, store `DontCare` and clear to 1.0 and 0; the pass
//! sizes, array length and default sample count are 0.
//!
//! [`layout`] checks a descriptor as Metal's API validation does, then states what the encoder
//! renders into ([`PassLayout`]):
//!
//! - **Attachments.** An attachment's texture needs a format of the attachment's kind; its level
//!   and slice lie in the texture (a view counts its own). A color texture without `RenderTarget`
//!   usage, which Metal's validation stops, renders as plain Metal renders it (an M4: cleared,
//!   drawn, loaded and stored as a render target is, and left as it was under a `DontCare` store;
//!   Metal probe case `usage_missing`) through a stand-in image that has it
//!   (`attachment_stand_in.rs`). So does a depth or stencil one (an M4:
//!   `Depth32Float`, `Depth16Unorm`, `Stencil8` and `Depth32Float_Stencil8` of shader-read usage
//!   alone cleared, loaded, drawn and stored as a render target is, and left as they were under a
//!   `DontCare` store): one texture through a stand-in, separate depth and stencil textures through
//!   their combined scratch image (`depth_stencil_scratch.rs`). A memoryless one, one whose format
//!   the device does not copy both ways, and a `Depth32Float` texture held in a color image beside
//!   its depth twin are not built (`NotImplemented(Render)`). Every attachment has one sample
//!   count, which the pipelines drawn must have. Metal renders into the smallest attachment's
//!   extent, further limited by a nonzero `renderTargetWidth` and `renderTargetHeight` (measured:
//!   an 8x8 color and a 4x4 depth attachment, or an 8x8 one under a 4x4 target size, render 4x4). A
//!   pass without attachments needs both target sizes, and rasterizes `defaultRasterSampleCount`
//!   samples (0, the default, is `Invalid`, as Metal ends the process there). Past 16384 on a side
//!   Metal makes its encoder and fails the command buffer from the pass on (measured): the encoder
//!   records into the device's largest framebuffer within it, and the command buffer's commit runs
//!   nothing from the pass on (`PassLayout::fails_command_buffer`).
//! - **Views of another format** render, and are resolved into, in the view's format, as
//!   Metal's do (a probe, an M3 Max and an M4: an sRGB view of an `RGBA8Unorm` texture stores the
//!   sRGB encoding of a clear and a draw and blends in linear, an `R32Uint` view its words' bytes,
//!   a subview its level and slice, an array view each layer of a layered pass): a color view
//!   through an image view in its direct format (`attachment_format`), and the stencil view of a
//!   combined texture as that texture's stencil.
//! - **The depth and stencil attachments** are one Vulkan attachment: one texture, or two views
//!   of one image at one level and slice. A texture whose format has both aspects attached by one
//!   of them only keeps the other's texels (loaded and stored unchanged). Two different textures
//!   render through a scratch image of the combined format (`depth_stencil_scratch.rs`), of
//!   different sizes too, which Metal's validation stops and plain Metal renders over the
//!   smaller. One texture whose image was not made for the pass's custom sample positions (a
//!   caller's image) renders through a stand-in made for them (`attachment_stand_in.rs`,
//!   `sample_positions.rs`).
//! - **Clear values** are converted as Metal converts them (`clear_value.rs`). Metal's validation
//!   stops a clear depth outside `[0, 1]` whenever the depth attachment has a texture, whatever its
//!   load action, and plain Metal clears to the nearer end (measured on an M4), as the runtime
//!   does; a NaN one, unmeasured, is refused.
//! - **Store actions** may be `Unknown` here: the encoder's `set_*_store_action` calls decide
//!   them before `end_encoding`, which begins the pass. A depth attachment's
//!   `CustomSampleDepthStore`, the descriptor's or a late one, stores as `Store` does: Metal stores
//!   the same samples, with custom sample positions and without, at one sample and four (a probe,
//!   the M3 Max and an M4), and stops what it stops of `Store` (a memoryless texture, a resolve
//!   texture, a late one over a known action). Its validation also stops the store action option
//!   `CustomSamplePositions` under it, which Metal without validation takes as it takes it under
//!   `Store`: the runtime gives that answer (rule 7). A color or stencil attachment's is refused
//!   ([`RenderPassProblem::CustomSampleDepthStore`]; Metal's own check, with and without
//!   validation); a late one there, which plain Metal leaves unstored, stores as `DontCare`, but
//!   where the attachment resolves (unmeasured).
//! - **Resolves**: each attachment's resolve texture and filters are checked as Metal checks
//!   them, and the pass resolves at the end of its subpass (`resolve.rs`).
//! - **A rasterization-rate map** is checked as Metal checks it as the encoder opens
//!   (`rate_mapped_pass.rs`): the map's device, the array length against its layers, and the
//!   attachments against its largest physical size, or the target size against its screen.
//! - **Layered passes** (`renderTargetArrayLength` 2 or more; measured on an M4 with macOS 27.0.1):
//!   the pass renders that many layers of every attachment from its slice (a cube's face;
//!   [`PassLayout::layers`]), loading, clearing and storing those layers alone, and each attachment
//!   must hold them ([`RenderPassProblem::ArrayLengthPastSlices`], a 2D texture's one slice too:
//!   plain Metal writes the missing layers into the memory after the texture). A draw whose vertex
//!   function writes `[[render_target_array_index]]` renders the layer it names, a 2D array view's
//!   layer in a layered Vulkan framebuffer; one that writes none renders the first. A length of 0
//!   or 1 renders the slice alone, whatever index a vertex function writes: at a length of 0 plain
//!   Metal draws such a function as at 1, where its debug layer stops the draw ("the
//!   renderPipelineState has a vertex shader that writes to render_target_array_index but
//!   renderTargetArrayLength is set to zero"; a Metal probe on an M4, another on an M3 Max and an
//!   M4). Metal renders an index at or past the length into the first layer, where Vulkan leaves
//!   the layer undefined, and its fragment function reads the index as written in every pass: the
//!   compile layer stores `Layer` as 0 there and hands the fragment function the index written. A
//!   resolve, separate depth and stencil textures and a stand-in for an attachment take every layer
//!   of the pass: layer `i` resolves into slice `resolveSlice + i` and a layer past the resolve
//!   texture's slices is not resolved (a Metal probe on the M3 Max and an M4, `resolve.rs`); a
//!   depth or stencil attachment resolves its first layer alone (a probe). The planner refuses a
//!   tile dispatch there (`NotImplemented(TileShading)`).
//!
//! Later parts of step 6 lift these refusals (`NotImplemented(Render)`): custom sample
//! positions, and a resolve of an attachment without a texture. (A texture buffer renders
//! through a 2D texture of one row over its bytes; a 3D texture's depth planes through 2D views of
//! them, and a 1D texture through a 1D view.) A render area or layer count past the device's
//! framebuffer limits is refused `Unsupported` (`Feature::RenderTargetSize`), naming the limit
//! (`framebuffer_fits`).

use super::attachment_stand_in::{self, Cause, StandIn};
use super::binding_view::views_as;
use super::buffer::Buffer;
use super::chroma_pairs::is_four_two_two;
use super::clear_value;
use super::command_buffer::CommandBuffer;
use super::degradation::Degradation;
use super::depth_stencil_scratch::{self, Separate};
use super::error::{invalid, nil, CommandBufferError, Error, Problem, RuntimeGap};
use super::fresh_unwritten::FreshUnwritten;
use super::image_plan::{block_family, ImagePlan};
use super::linear_attachment;
use super::nan_clear::{self, NanDepth};
use super::rate_map::RasterizationRateMap;
use super::render::ResolveTarget;
use super::render_values::{
    ClearColor, LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter,
    SamplePosition, StoreAction, StoreActionOptions,
};
use super::resolve::{self, DepthStencilResolve, Resolve, Wanted};
use super::sample_planes::Planes;
use super::sample_positions::{self, Sixteenths};
use super::texture::{signed_depth, Texture};
use super::tile_pass::{self, TilePass, TileRefusal};
use super::view_plan::ViewRecord;
use super::Device;
use ash::vk;
use metal2vulkan::device::{DriverFact, VkLimit};
use metal2vulkan::error::{Capability, Feature, Unsupported};
use metal2vulkan::metal::format::{
    aspects, direct_format, vk_format, ycbcr, ycbcr_renders, ClearRoute, ExtendedRange,
};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use std::borrow::Cow;

/// The color attachments a pass has (`MTLRenderPassDescriptor.colorAttachments`).
pub const MAX_COLOR_ATTACHMENTS: usize = 8;

/// `MTLRenderPassColorAttachmentDescriptor`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RenderPassColorAttachmentDescriptor {
    /// `texture`: the texture rendered into; `None` for no color attachment. It needs
    /// `RenderTarget` usage and a color format.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u64,
    /// `slice`: the array slice or cube face rendered into; a layered pass's first layer.
    pub slice: u64,
    /// `depthPlane`: the depth plane of a 3D texture rendered into.
    pub depth_plane: u64,
    /// `resolveTexture`: the single-sample texture a resolving store action writes; `None`
    /// for no resolve.
    pub resolve_texture: Option<Texture>,
    /// `resolveLevel`: the resolve texture's mipmap level.
    pub resolve_level: u64,
    /// `resolveSlice`: the resolve texture's slice or cube face; a layered pass resolves layer
    /// `i` into `resolve_slice + i`.
    pub resolve_slice: u64,
    /// `resolveDepthPlane`: the resolve texture's depth plane, for a 3D texture.
    pub resolve_depth_plane: u64,
    /// `loadAction`: `DontCare` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `Store` by default. `Unknown` leaves it to the encoder's
    /// `set_color_store_action`, which must set it before `end_encoding`. `CustomSampleDepthStore`
    /// is refused (`RenderPassProblem::CustomSampleDepthStore`), as Metal refuses it.
    pub store_action: StoreAction,
    /// `storeActionOptions`: its one option, `CustomSamplePositions`, needs the pass's
    /// [`sample_positions`](RenderPassDescriptor::sample_positions) and a store action that keeps
    /// the samples.
    pub store_action_options: StoreActionOptions,
    /// `clearColor`: opaque black by default, converted to the attachment's format as Metal
    /// converts it.
    pub clear_color: ClearColor,
}

impl Default for RenderPassColorAttachmentDescriptor {
    /// Metal's defaults: no texture, `DontCare` load, `Store` store, opaque black clear.
    fn default() -> RenderPassColorAttachmentDescriptor {
        RenderPassColorAttachmentDescriptor {
            texture: None,
            level: 0,
            slice: 0,
            depth_plane: 0,
            resolve_texture: None,
            resolve_level: 0,
            resolve_slice: 0,
            resolve_depth_plane: 0,
            load_action: LoadAction::DontCare,
            store_action: StoreAction::Store,
            store_action_options: StoreActionOptions::NONE,
            clear_color: ClearColor::default(),
        }
    }
}

/// `MTLRenderPassDepthAttachmentDescriptor`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RenderPassDepthAttachmentDescriptor {
    /// `texture`: the texture rendered into; `None` for no depth attachment. It needs
    /// `RenderTarget` usage and a depth format.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u64,
    /// `slice`: the array slice or cube face rendered into; a layered pass's first layer.
    pub slice: u64,
    /// `depthPlane`: the depth plane of a 3D texture rendered into.
    pub depth_plane: u64,
    /// `resolveTexture`: the single-sample texture a resolving store action writes; `None`
    /// for no resolve.
    pub resolve_texture: Option<Texture>,
    /// `resolveLevel`: the resolve texture's mipmap level.
    pub resolve_level: u64,
    /// `resolveSlice`: the resolve texture's slice or cube face; a layered pass resolves layer
    /// `i` into `resolve_slice + i`.
    pub resolve_slice: u64,
    /// `resolveDepthPlane`: the resolve texture's depth plane, for a 3D texture.
    pub resolve_depth_plane: u64,
    /// `loadAction`: `Clear` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `DontCare` by default. `Unknown` leaves it to the encoder's
    /// `set_depth_store_action`, which must set it before `end_encoding`. `CustomSampleDepthStore`
    /// stores as `Store` does, as Metal stores it.
    pub store_action: StoreAction,
    /// `storeActionOptions`: its one option, `CustomSamplePositions`, needs the pass's
    /// [`sample_positions`](RenderPassDescriptor::sample_positions) and a store action that keeps
    /// the samples.
    pub store_action_options: StoreActionOptions,
    /// `clearDepth`: 1.0 by default. A value outside `[0, 1]` clears to the nearer end, as plain
    /// Metal clears it (Metal probe case `depth_clear_range_load` on an M4: 2.0 stores 1.0, -1.0
    /// stores 0; its validation stops it). A NaN of either sign clears a 32-bit float depth to
    /// `0x7fc00000`, which a draw in the pass tests above every value, and a `Depth16Unorm` to 0,
    /// as plain Metal clears them (our probe `nanclear` on the M3 Max); where the device writes no
    /// such word into the depth, it clears to 1.0, announced (`signed-depth-test`, or
    /// `restricted-depth-upload` on a device that may clamp a depth). A NaN `Clear` of another
    /// depth format, unmeasured, is refused (`RenderPassProblem::ClearDepth`).
    pub clear_depth: f64,
    /// `depthResolveFilter`: `Sample0` by default.
    pub depth_resolve_filter: MultisampleDepthResolveFilter,
}

impl Default for RenderPassDepthAttachmentDescriptor {
    /// Metal's defaults: no texture, `Clear` load, `DontCare` store, a clear depth of 1.
    fn default() -> RenderPassDepthAttachmentDescriptor {
        RenderPassDepthAttachmentDescriptor {
            texture: None,
            level: 0,
            slice: 0,
            depth_plane: 0,
            resolve_texture: None,
            resolve_level: 0,
            resolve_slice: 0,
            resolve_depth_plane: 0,
            load_action: LoadAction::Clear,
            store_action: StoreAction::DontCare,
            store_action_options: StoreActionOptions::NONE,
            clear_depth: 1.0,
            depth_resolve_filter: MultisampleDepthResolveFilter::Sample0,
        }
    }
}

/// `MTLRenderPassStencilAttachmentDescriptor`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct RenderPassStencilAttachmentDescriptor {
    /// `texture`: the texture rendered into; `None` for no stencil attachment. It needs
    /// `RenderTarget` usage and a stencil format.
    pub texture: Option<Texture>,
    /// `level`: the mipmap level rendered into.
    pub level: u64,
    /// `slice`: the array slice or cube face rendered into; a layered pass's first layer.
    pub slice: u64,
    /// `depthPlane`: the depth plane of a 3D texture rendered into.
    pub depth_plane: u64,
    /// `resolveTexture`: the single-sample texture a resolving store action writes; `None`
    /// for no resolve.
    pub resolve_texture: Option<Texture>,
    /// `resolveLevel`: the resolve texture's mipmap level.
    pub resolve_level: u64,
    /// `resolveSlice`: the resolve texture's slice or cube face; a layered pass resolves layer
    /// `i` into `resolve_slice + i`.
    pub resolve_slice: u64,
    /// `resolveDepthPlane`: the resolve texture's depth plane, for a 3D texture.
    pub resolve_depth_plane: u64,
    /// `loadAction`: `Clear` by default.
    pub load_action: LoadAction,
    /// `storeAction`: `DontCare` by default. `Unknown` leaves it to the encoder's
    /// `set_stencil_store_action`, which must set it before `end_encoding`.
    /// `CustomSampleDepthStore` is refused (`RenderPassProblem::CustomSampleDepthStore`), as Metal
    /// refuses it.
    pub store_action: StoreAction,
    /// `storeActionOptions`: its one option, `CustomSamplePositions`, needs the pass's
    /// [`sample_positions`](RenderPassDescriptor::sample_positions) and a store action that keeps
    /// the samples.
    pub store_action_options: StoreActionOptions,
    /// `clearStencil`: 0 by default; its low 8 bits are cleared, as Metal clears them.
    pub clear_stencil: u32,
    /// `stencilResolveFilter`: `Sample0` by default.
    pub stencil_resolve_filter: MultisampleStencilResolveFilter,
}

impl Default for RenderPassStencilAttachmentDescriptor {
    /// Metal's defaults: no texture, `Clear` load, `DontCare` store, a clear stencil of 0.
    fn default() -> RenderPassStencilAttachmentDescriptor {
        RenderPassStencilAttachmentDescriptor {
            texture: None,
            level: 0,
            slice: 0,
            depth_plane: 0,
            resolve_texture: None,
            resolve_level: 0,
            resolve_slice: 0,
            resolve_depth_plane: 0,
            load_action: LoadAction::Clear,
            store_action: StoreAction::DontCare,
            store_action_options: StoreActionOptions::NONE,
            clear_stencil: 0,
            stencil_resolve_filter: MultisampleStencilResolveFilter::Sample0,
        }
    }
}

/// `MTLRenderPassDescriptor`: a value the caller fills, as Metal's mutable descriptor is.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct RenderPassDescriptor {
    /// `colorAttachments`: the color attachments, by index.
    pub color_attachments: [RenderPassColorAttachmentDescriptor; MAX_COLOR_ATTACHMENTS],
    /// `depthAttachment`.
    pub depth_attachment: RenderPassDepthAttachmentDescriptor,
    /// `stencilAttachment`: the same texture as the depth attachment's, or another one
    /// (rendered through a combined scratch image).
    pub stencil_attachment: RenderPassStencilAttachmentDescriptor,
    /// `visibilityResultBuffer`: the buffer the encoder's visibility results are written to.
    pub visibility_result_buffer: Option<Buffer>,
    /// `renderTargetArrayLength`: the layers a layered pass renders (2 or more); 0 or 1 renders
    /// each attachment's slice alone.
    pub render_target_array_length: u64,
    /// `renderTargetWidth`: limits the rendered width when nonzero; a pass without attachments
    /// needs it, at most 16384.
    pub render_target_width: u64,
    /// `renderTargetHeight`: as `render_target_width`, for the height.
    pub render_target_height: u64,
    /// `defaultRasterSampleCount`: the samples a pass without attachments rasterizes. Metal
    /// takes 1, 2 and 4 on every device; of other counts it stops some, 0 (the default) among
    /// them, so a pass without attachments sets it, and fails the command buffer of others.
    pub default_raster_sample_count: u64,
    /// `tileWidth` and `tileHeight`: the tile a tile dispatch runs over, 16x16, 32x16 or 32x32,
    /// or 0 by 0 for the default (`tile_pass.rs`).
    pub tile_width: u64,
    /// `tileHeight`: see [`tile_width`](Self::tile_width).
    pub tile_height: u64,
    /// `imageblockSampleLength`: the bytes of tile memory each sample holds beside the
    /// attachments, which a tile state's imageblock may need (`tile_pass.rs`).
    pub imageblock_sample_length: u64,
    /// `threadgroupMemoryLength`: the threadgroup memory each tile holds for the pass's tile
    /// functions (`tile_pass.rs`).
    pub threadgroup_memory_length: u64,
    /// `rasterizationRateMap`: the map the pass's draws rasterize through, region by region
    /// (`rate_mapped_pass.rs`).
    pub rasterization_rate_map: Option<RasterizationRateMap>,
    /// `setSamplePositions:count:` and `getSamplePositions:count:`: the pass's sample positions,
    /// one per sample, as set (`sample_positions.rs`); empty for the device's own.
    pub sample_positions: Vec<SamplePosition>,
}

impl RenderPassDescriptor {
    /// `renderPassDescriptor`: a descriptor of Metal's defaults.
    pub fn new() -> RenderPassDescriptor {
        RenderPassDescriptor::default()
    }
}

/// One attachment of a render pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Attachment {
    /// The color attachment at this index (`colorAttachments[i]`).
    Color(u32),
    /// The depth attachment (`depthAttachment`).
    Depth,
    /// The stencil attachment (`stencilAttachment`).
    Stencil,
}

/// What a render pass descriptor, or a late store action, did wrong.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RenderPassProblem {
    /// No attachment has a texture and the target width or height is 0: Metal answers nil (an
    /// [`Error::Nil`](crate::Error::Nil)).
    NoTargets,
    /// The attachment's texture has a format of another kind: a depth or stencil format as a
    /// color attachment, a format without depth as the depth attachment, one without stencil as
    /// the stencil attachment.
    AttachmentFormat(Attachment),
    /// The attachment's level lies past its texture's, its slice past an array's or a cube's
    /// slices, or its depth plane past a 3D texture's planes. An index the texture's type does
    /// not have is ignored, as Metal ignores it.
    AttachmentSubresource(Attachment),
    /// The attachments' sample counts differ.
    SampleCounts,
    /// The depth attachment's `Clear` load holds a NaN clear depth, on a depth format whose
    /// store of one is not measured (not `Depth32Float`, `Depth32Float_Stencil8` or
    /// `Depth16Unorm`, which clear as plain Metal clears them; a value outside `[0, 1]` clears to
    /// the nearer end).
    ClearDepth,
    /// A store action set by the encoder over one the descriptor knew (not `Unknown`), or set to
    /// `Unknown`; or store action options set by the encoder over a store action the descriptor
    /// knew (measured: "store actions known at render command encoder init time may not be
    /// mutated"). Over a color attachment's known action, plain Metal keeps that action and
    /// options, and the encoder does.
    StoreActionKnown(Attachment),
    /// A store action set by the encoder for a stencil attachment without a texture, or store
    /// action options for an attachment without one (plain Metal changes nothing for a color or
    /// depth attachment's store action, nor does the encoder).
    NoAttachment(Attachment),
    /// A resolving store action (the descriptor's or the encoder's) without a resolve texture
    /// (measured: "requires resolve texture"), of a depth or stencil attachment, or of a color
    /// attachment set late by `set_color_resolve_target`: the descriptor's or a late store
    /// action of a color attachment resolves nothing there, as plain Metal's.
    ResolveWithoutTexture(Attachment),
    /// A resolve target set late (`set_color_resolve_target` and its depth and stencil twins)
    /// under a store action the descriptor knew that does not resolve (Metal takes `Unknown`;
    /// plain Metal's answer to its private call is unmeasured). The descriptor's resolve texture
    /// under such an action, the descriptor's or a late one, is left untouched and the
    /// attachment stored by the action, as plain Metal does where its validation stops it
    /// ("must be one of MTLStoreActionMultisampleResolve or
    /// MTLStoreActionStoreAndMultisampleResolve"): a color attachment's, a depth or stencil one's.
    ResolveStoreAction(Attachment),
    /// A resolve of a single-sampled depth or stencil attachment, or a color one's target set
    /// late (measured: Metal's validation stops it; a color attachment's resolving store action
    /// copies its texels into the descriptor's resolve texture, as plain Metal's).
    ResolveSourceSamples(Attachment),
    /// A multisampled resolve texture (measured): a depth or stencil one, and a color one of
    /// the attachment's sample count (plain Metal copies the samples, not built). A color one of
    /// another count is left untouched, as plain Metal leaves it.
    ResolveSamples(Attachment),
    /// A resolve texture of another pixel format than the attachment's (measured: Metal's
    /// validation stops it, and plain Metal's answer depends on the pair, `resolve.rs`; a
    /// multisampled color attachment resolves into its format's sRGB twin in its own encoding,
    /// as plain Metal's), or an integer color format (measured: "cannot be a MSAA resolve target";
    /// plain Metal writes values no sample holds, changing from run to run).
    ResolveFormat(Attachment),
    /// The resolve level, slice or depth plane lies past the resolve texture (measured), an
    /// index its type does not have ignored as for an attachment.
    ResolveSubresource(Attachment),
    /// A color resolve level reaches past the attachment, which loads with `Load` or `DontCare`
    /// (measured: "size mismatch"; plain Metal writes what memory held there). A level of another
    /// extent resolves the overlap elsewhere, as plain Metal does.
    ResolveSize(Attachment),
    /// The depth and stencil attachments are one texture whose stencil resolves alone
    /// (`MultisampleResolve`) while its depth is stored: plain Metal overwrites the depth's
    /// samples with a layout pattern (probes), and its validation stops it ("mixing
    /// MTLStoreActionStore and MTLStoreActionMultisampleResolve store actions ... is not allowed").
    /// The depth resolved alone with the stencil stored, and an aspect stored and resolved beside a
    /// stored one, run as Metal runs them.
    DepthStencilResolveMix,
    /// The stencil resolves with `DepthResolvedSample` without a depth attachment, whose filter
    /// picks its sample (measured; with a depth attachment that does not resolve, the filter
    /// picks it as for one that does).
    DepthResolvedSampleWithoutDepthResolve,
    /// A memoryless attachment loaded with `Load` (measured: "Memoryless attachment cannot be
    /// loaded as it have no memory to load from").
    MemorylessLoad(Attachment),
    /// A memoryless attachment stored with `Store` or `StoreAndMultisampleResolve`, the
    /// descriptor's (measured: "Memoryless attachment content cannot be stored in memory") or
    /// the encoder's late one (measured: "storeAction unsupported for Memoryless attachment").
    MemorylessStore(Attachment),
    /// A memoryless resolve texture (measured: "resolveTexture must not be Memoryless").
    MemorylessResolve(Attachment),
    /// A tile size other than 16x16, 32x16 and 32x32, or a tile with one side 0 (measured:
    /// "Invalid tile dimensions (8, 8)"; plain Metal answers its encoder nil, an
    /// [`Error::Nil`](crate::Error::Nil)).
    TileSize {
        /// The tile's width.
        width: u64,
        /// The tile's height.
        height: u64,
    },
    /// A 32x32 tile at 4 samples (measured: "4X MSAA does not support 32 x 32 tiles"), which
    /// Metal's validation stops, an [`Error::Invalid`](crate::Error::Invalid). Plain Metal
    /// faults there: its command buffer fails, or it makes no encoder (the re-probe on an M4), so
    /// the refusal is Metal's answer.
    TileSizeAtSamples {
        /// The pass's sample count.
        samples: u32,
    },
    /// Per-sample storage, the imageblock sample length rounded up to 8 bytes, past 64
    /// (measured: "Per sample storage (72) cannot be greater than (64)"; plain Metal answers its
    /// encoder nil, an [`Error::Nil`](crate::Error::Nil)).
    PerSampleStorage {
        /// The per-sample storage, in bytes.
        bytes: u64,
    },
    /// A threadgroup memory length past 32768 bytes (measured: "Requested threadgroup memory
    /// length exceeds amount supported by device (32768)"; plain Metal answers its encoder nil, an
    /// [`Error::Nil`](crate::Error::Nil)).
    ThreadgroupMemoryLength {
        /// The threadgroup memory length asked for, in bytes.
        length: u64,
    },
    /// Tile memory, the per-sample storage of each pixel of a tile plus the threadgroup memory
    /// length, past 32768 bytes (measured: "Total allocated tile memory (40960) cannot be greater
    /// than (32768)"; plain Metal answers its encoder nil, an [`Error::Nil`](crate::Error::Nil)).
    TileMemory {
        /// The tile memory asked for, in bytes.
        bytes: u64,
    },
    /// A `renderTargetArrayLength` past the layer count of a rasterization-rate map whose layers
    /// differ (measured: "renderTargetArrayLength (2) must be <= rasterizationRateMap.layerCount
    /// (1)", `rate_mapped_pass.rs`; past a map of layers alike, every slice takes its first
    /// layer, as plain Metal rasterizes past one layer).
    RateMapArrayLength {
        /// The pass's `renderTargetArrayLength`.
        length: u64,
        /// The map's layer count.
        layers: u32,
    },
    /// A pass without attachments whose render target size is past the rasterization-rate map's
    /// screen (measured: "renderTargetWidth (128) must be <= rasterizationRateMap.screenSize.width
    /// (64)").
    RateMapTargetSize {
        /// The pass's render target width and height.
        target: [u32; 2],
        /// The map's screen width and height.
        screen: [u32; 2],
    },
    /// The attachment's slice (a cube's face) plus the pass's `renderTargetArrayLength`, 2 or
    /// more, lies past its texture's slices: Metal's validation holds every attachment to the
    /// length, a 2D texture's one slice among them (measured: "slice + renderTargetArrayLength is
    /// 3, but the texture only has 2 slices"). Plain Metal's answer is not defined: it clears and
    /// draws a missing layer into the memory after the texture (a cube's seventh face and a
    /// depth array's fifth slice landed in the next texture of a heap, on an M3 Max and an M4),
    /// and a 2D texture's pass stored another word on a second run.
    ArrayLengthPastSlices {
        /// The attachment.
        attachment: Attachment,
        /// The slice plus the length.
        end: u64,
        /// The texture's slices (a view's own).
        slices: u32,
    },
    /// A sample position whose coordinate lies outside `[0, 1)` (measured: "Provided sample
    /// position x-coodinate (1.000000) at index 0 is not within the range [0,1)").
    SamplePosition {
        /// The position's index.
        index: u32,
    },
    /// Sample positions of a count the device takes none at (measured: "count (1) is not a
    /// supported sample count for custom positions", "the device does not support the number of
    /// samples supplied").
    SamplePositionCount {
        /// The number of positions.
        count: u64,
    },
    /// More sample positions than the pass's samples (measured: Metal's validation stops another
    /// count, "2 custom sample positions were programmed, but this render pass uses 4 rasterizer
    /// sample(s)", and plain Metal ends the process past them; it puts the samples past fewer
    /// positions at the pixel's corner, as the runtime does).
    SamplePositionsSamples {
        /// The number of positions.
        positions: u64,
        /// The pass's sample count.
        samples: u32,
    },
    /// The stencil descriptor's store action options include `CustomSamplePositions` without
    /// sample positions (measured: "includes MTLStoreActionOptionCustomSamplePositions, but no
    /// custom sample positions were specified"; a color or depth attachment's change nothing, as
    /// plain Metal's).
    StoreActionOptionsWithoutPositions(Attachment),
    /// A stencil attachment's store action options that include `CustomSamplePositions` under a
    /// store action that keeps no samples, `DontCare` or `MultisampleResolve`: the descriptor's,
    /// or at `end_encoding` the encoder's late one (measured: "includes
    /// MTLStoreActionOptionCustomSamplePositions, which does not apply to the store action"; a
    /// color or depth attachment's change nothing, as plain Metal's).
    StoreActionOptions(Attachment),
    /// `CustomSampleDepthStore` as a color or stencil attachment's store action: the
    /// descriptor's (measured, with and without API validation: "storeAction is not a valid
    /// MTLStoreAction"), or the encoder's late one of an attachment that resolves (unmeasured).
    /// A late one of an attachment that does not resolve, which Metal leaves unstored without
    /// validation (its validation stops the pass at `endEncoding`), stores as `DontCare`; a depth
    /// attachment's stores as `Store` does.
    CustomSampleDepthStore(Attachment),
    /// A pass without attachments whose `defaultRasterSampleCount`'s low 8 bits are 8 to 15 or
    /// 128 to 255, or are 0 in a count of 256 or more: plain Metal ends the process making its
    /// encoder, on every device.
    DefaultRasterSampleCount {
        /// The `defaultRasterSampleCount` given.
        count: u64,
    },
}

/// One attachment's subresource, checked: the texture, the image level and layer it renders,
/// its Vulkan format, and that level's extent.
#[derive(Clone, Debug)]
pub(super) struct Target {
    pub(super) texture: Texture,
    /// The image's level and layer (a view's are counted from its own first).
    pub(super) level: u32,
    pub(super) layer: u32,
    /// The layers it renders from `layer`: the pass's (`PassLayout::layers`).
    pub(super) layers: u32,
    /// The attachment's Vulkan format: the image's, or a view's of another format
    /// (`attachment_format`).
    pub(super) format: metal2vulkan::vk::Format,
    pub(super) samples: u32,
    pub(super) extent: [u32; 2],
}

/// A color attachment, checked.
#[derive(Clone)]
pub(super) struct ColorTarget {
    pub(super) target: Target,
    pub(super) load: LoadAction,
    /// The store action: the descriptor's, or the encoder's where the descriptor's was
    /// `Unknown`.
    pub(super) store: StoreAction,
    /// The descriptor's store action was `Unknown`: the encoder decides it.
    pub(super) late_store: bool,
    /// The texture is memoryless: the pass neither loads nor stores it.
    pub(super) memoryless: bool,
    /// The clear value: an extended-range attachment's by `ClearRoute::ThroughHalf`, which the
    /// encoder replaces by `straight_clear` as the pass ends where Metal takes that route.
    pub(super) clear: vk::ClearColorValue,
    /// An extended-range attachment's clear value by `ClearRoute::Straight`, Metal's where it
    /// keeps the attachment's level compressed and the pass has no draw call
    /// (`nan_clear::keeps_nan_sign`): the pass's encoder takes it as it ends. `None` for any
    /// other format.
    pub(super) straight_clear: Option<vk::ClearColorValue>,
    /// What the attachment resolves into, when it resolves (`resolve.rs`).
    pub(super) resolve: Option<Resolve>,
    /// The resolve texture's level and slice a single-sampled attachment resolves into: its
    /// texels copied there after the pass (`resolve::color_copy`), the last instance storing
    /// them whatever Metal's store action.
    pub(super) copied_resolve: Option<Target>,
    /// What `target` stands in for, when an image of the pass stands in for the attachment's
    /// own (`attachment_stand_in.rs`; the encoder decides it at a draw).
    pub(super) stand_in: Option<StandIn>,
    /// The attachment's sample planes, where a draw of the pass reads it as them
    /// (`sample_planes.rs`; the encoder makes them at the first such draw).
    pub(super) planes: Option<Planes>,
}

impl std::fmt::Debug for ColorTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // SAFETY: every member of the union is four 32-bit values; any bits are a valid `u32`.
        let clear = unsafe { self.clear.uint32 };
        f.debug_struct("ColorTarget")
            .field("target", &self.target)
            .field("load", &self.load)
            .field("store", &self.store)
            .field("late_store", &self.late_store)
            .field("memoryless", &self.memoryless)
            .field("clear", &clear)
            // SAFETY: as `clear`'s.
            .field(
                "straight_clear",
                &self.straight_clear.map(|c| unsafe { c.uint32 }),
            )
            .field("resolve", &self.resolve)
            .field("copied_resolve", &self.copied_resolve)
            .field("stand_in", &self.stand_in)
            .field("planes", &self.planes)
            .finish()
    }
}

impl ColorTarget {
    /// Whether the pass clears the attachment and resolves it alone (`MultisampleResolve`) into a
    /// resolve texture of another pixel format than the attachment's own (its sRGB twin, the only
    /// other format the runtime resolves into): pixel format identity, not the twin relation.
    /// Where such a pass has no draw call, Metal keeps the samples' earlier texels, its texture
    /// compressed or not (`RGBA8Unorm`, `BGRA8Unorm` and the extended-range
    /// formats, each way, an Apple M4 under macOS 27.0.1), where a resolve into the attachment's
    /// own format leaves the clear in a texture it compresses. The attachment's own texture is its
    /// stand-in's where one stands in for it.
    pub(super) fn clears_and_resolves_across_formats(&self) -> bool {
        let own = self.stand_in.as_ref().map_or(&self.target, |s| &s.own);
        let into = self
            .resolve
            .as_ref()
            .map(|r| r.copy_to.as_ref().unwrap_or(&r.attachment));
        self.load == LoadAction::Clear
            && self.store == StoreAction::MultisampleResolve
            && into.is_some_and(|into| into.texture.pixel_format() != own.texture.pixel_format())
    }
}

/// One aspect of the depth/stencil attachment: Metal's load and store actions and the Vulkan
/// clear value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AspectTarget {
    pub(super) load: LoadAction,
    /// The store action: the descriptor's, or the encoder's where the descriptor's was
    /// `Unknown`.
    pub(super) store: StoreAction,
    /// The descriptor's store action was `Unknown`: the encoder decides it.
    pub(super) late_store: bool,
    /// The store action options: the descriptor's, or the encoder's where the descriptor's
    /// store action was `Unknown` (`sample_positions.rs`; a stencil attachment's are checked).
    pub(super) options: StoreActionOptions,
    /// The texture is memoryless: the pass neither loads nor stores it.
    pub(super) memoryless: bool,
    /// The clear depth, or the clear stencil as a float-free value.
    pub(super) clear: AspectClear,
    /// The descriptor's `Clear` load held a NaN depth, which Metal stores as `0x7fc00000`: the
    /// load is `Load`, and that word is written into the texture over the clear's reach before
    /// the pass (`nan_clear.rs`, [`NanDepth::Written`](super::nan_clear::NanDepth)).
    pub(super) written_nan: bool,
}

/// An aspect's clear value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum AspectClear {
    Depth(f32),
    Stencil(u32),
}

/// The Metal pixel format of the hidden depth a draw without the pass's depth renders against:
/// a 32-bit float (a Metal probe `pipe_depth16_extra`: a write at 0.5, then `Equal` at the next
/// `float` above, drew nothing under a pipeline naming `Depth16Unorm`).
pub(super) const HIDDEN_DEPTH_FORMAT: PixelFormat = PixelFormat::Depth32Float;

/// Announce `depth-denormals` where the pass of `layout` clears a 32-bit float depth to a
/// denormal on a device whose driver may store it or test it otherwise than Metal's
/// (`ImageFacts::denormal_loss`): its depth/stencil attachment's, the program's (the hidden one
/// clears to 1).
pub(super) fn degrade_depth_clear(device: &Device, layout: &PassLayout) {
    let Some(ds) = layout.depth_stencil.as_ref().filter(|ds| !ds.hidden) else {
        return;
    };
    let float_depth = matches!(
        ds.target.format,
        metal2vulkan::vk::Format::D32_SFLOAT | metal2vulkan::vk::Format::D32_SFLOAT_S8_UINT
    );
    let cleared = match ds.depth.as_ref() {
        Some(AspectTarget {
            load: LoadAction::Clear,
            clear: AspectClear::Depth(depth),
            ..
        }) => *depth,
        _ => return,
    };
    let denormal = cleared != 0.0 && cleared.abs() < f32::MIN_POSITIVE;
    if !float_depth || !denormal {
        return;
    }
    let facts = (device.shared().textures).facts(device.compile_device().profile());
    if let Some(loss) = facts.denormal_loss(false) {
        // Where the device stores it, a later pass that loads the depth tests it over the
        // smallest normal (`texture/signed_depth.rs`); this one tests it as the device does.
        if facts.depth_images.denormals {
            let own = ds.stand_in.as_ref().map_or(&ds.target, |s| &s.own);
            own.texture
                .switch_on_signed_depth("render_command_encoder", true);
        }
        device.degrade(Degradation::DepthDenormals, || {
            let format = ds.target.texture.pixel_format();
            let why = format!("a pass clears its depth to the denormal {cleared:e}: {loss}");
            (format!("a {format:?} texture"), why)
        });
    }
}

/// The depth/stencil attachment, checked: one Vulkan attachment for Metal's depth and stencil
/// attachments.
#[derive(Clone, Debug)]
pub(super) struct DepthStencilTarget {
    pub(super) target: Target,
    /// Metal's depth attachment, when it has a texture.
    pub(super) depth: Option<AspectTarget>,
    /// Metal's stencil attachment, when it has a texture.
    pub(super) stencil: Option<AspectTarget>,
    /// The Metal pixel format of the depth the pass renders: its depth attachment's texture's,
    /// [`HIDDEN_DEPTH_FORMAT`] for the hidden depth, `None` without a depth attachment. The
    /// depth bias follows it (`metal::state::depth_bias`), whatever the pipeline names.
    pub(super) depth_format: Option<PixelFormat>,
    /// Aspects whose texels Metal keeps though no attachment names them: the texture's format
    /// has both and one attachment only names it, and it is not memoryless. Loaded and stored
    /// unchanged.
    pub(super) keep_depth: bool,
    pub(super) keep_stencil: bool,
    /// The depth and stencil attachments are different textures: `target` is the scratch image
    /// the pass renders them through (`depth_stencil_scratch.rs`).
    pub(super) separate: Option<Separate>,
    /// What the attachment resolves into, when an aspect resolves (`resolve.rs`).
    pub(super) resolve: Option<DepthStencilResolve>,
    /// What `target` stands in for: a memoryless texture's backing in a split pass
    /// (`attachment_stand_in.rs`; the encoder decides it at a draw), or an image made for the
    /// pass's custom sample positions where the texture's is not (decided with the pass).
    pub(super) stand_in: Option<StandIn>,
    /// Where each aspect resolves as the descriptor or the encoder last named it, and the
    /// descriptor's filters: what `resolve` is made of again when the encoder sets a target
    /// (`render/late_resolve.rs`).
    pub(super) depth_resolve_target: Option<ResolveTarget>,
    pub(super) stencil_resolve_target: Option<ResolveTarget>,
    pub(super) depth_filter: MultisampleDepthResolveFilter,
    pub(super) stencil_filter: MultisampleStencilResolveFilter,
    /// `target` is the pass's hidden depth and stencil (`DepthStencilTarget::hidden`): Metal's
    /// pass has no depth/stencil attachment, and a draw's state uses one.
    pub(super) hidden: bool,
}

impl DepthStencilTarget {
    /// The hidden depth and stencil of a pass without a depth/stencil attachment, rendered into
    /// `target` (`attachment_stand_in::hidden`): what plain Metal tests a draw against where its
    /// state uses depth or the stencil there (Metal probes on an M4): a `float` depth of 1.0 and a
    /// stencil of 0 at the pass's start, kept per sample through the pass and never stored. The
    /// encoder clears `target` before the pass (`render.rs`), so each instance loads it, and none
    /// stores it past the last.
    pub(super) fn hidden(target: Target) -> DepthStencilTarget {
        let aspect = |clear| AspectTarget {
            load: LoadAction::Load,
            store: StoreAction::DontCare,
            late_store: false,
            options: StoreActionOptions::NONE,
            memoryless: false,
            clear,
            written_nan: false,
        };
        DepthStencilTarget {
            target,
            depth: Some(aspect(AspectClear::Depth(1.0))),
            stencil: Some(aspect(AspectClear::Stencil(0))),
            depth_format: Some(HIDDEN_DEPTH_FORMAT),
            keep_depth: false,
            keep_stencil: false,
            separate: None,
            resolve: None,
            stand_in: None,
            depth_resolve_target: None,
            stencil_resolve_target: None,
            depth_filter: MultisampleDepthResolveFilter::default(),
            stencil_filter: MultisampleStencilResolveFilter::default(),
            hidden: true,
        }
    }
}

/// What a render encoder renders into: [`layout`] of its descriptor.
#[derive(Clone, Debug)]
pub(super) struct PassLayout {
    /// The color attachments by Metal index, up to the last with a texture.
    pub(super) colors: Vec<Option<ColorTarget>>,
    pub(super) depth_stencil: Option<DepthStencilTarget>,
    /// The render area, from the origin.
    pub(super) extent: [u32; 2],
    /// The layers the pass renders, of every attachment from its slice and of the framebuffer:
    /// the descriptor's `renderTargetArrayLength` where it is above 1, else 1 (a layered pass,
    /// whose draws choose a layer by `[[render_target_array_index]]`).
    pub(super) layers: u32,
    /// The samples every attachment has (the rasterization samples without attachments).
    pub(super) samples: u32,
    pub(super) visibility: Option<Buffer>,
    /// The draws read the color attachments as input attachments in rasterization order: the
    /// device's framebuffer fetch does (`framebuffer_fetch::through_inputs`), and the pass has a
    /// color attachment.
    pub(super) fetch_inputs: bool,
    /// The pass's tile memory and the encoder's tile size (`tile_pass.rs`).
    pub(super) tile: TilePass,
    /// The pass's sample positions in sixteenths of a pixel, empty for the device's own
    /// (`sample_positions.rs`).
    pub(super) sample_positions: Vec<Sixteenths>,
    /// The error of a pass without attachments whose command buffer Metal fails from it on,
    /// which runs nothing: past 16384 a side (`CommandBufferError::RenderTargetSize`), its
    /// `extent` the device's largest framebuffer's within it, or of a default sample count Metal
    /// fails it at (`CommandBufferError::DefaultRasterSampleCount`, [`pass_default_samples`]).
    pub(super) fails_command_buffer: Option<CommandBufferError>,
}

/// What [`check_target`] reads of an attachment's texture.
#[derive(Clone, Copy, Debug)]
pub(super) struct Facts {
    pub(super) pixel_format: PixelFormat,
    pub(super) texture_type: TextureType,
    pub(super) usage: TextureUsage,
    /// The texture's levels and slices (a view's own), and the extent of its first level.
    pub(super) levels: u32,
    pub(super) slices: u32,
    pub(super) extent: [u32; 3],
    pub(super) samples: u32,
    /// The image's first level and layer the texture presents.
    pub(super) base_level: u32,
    pub(super) base_layer: u32,
}

impl Facts {
    /// The facts of a texture presenting `record`.
    pub(super) fn of(record: &ViewRecord) -> Facts {
        Facts {
            pixel_format: record.pixel_format,
            texture_type: record.texture_type,
            usage: record.usage,
            levels: record.level_count,
            slices: record.layer_count,
            extent: record.extent(),
            samples: record.samples,
            base_level: record.base_level,
            base_layer: record.base_layer,
        }
    }
}

/// The checked subresource of an attachment: the image level, its first layer and the layers it
/// renders from there, and the level's extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Subresource {
    pub(super) level: u32,
    pub(super) layer: u32,
    pub(super) layers: u32,
    pub(super) extent: [u32; 2],
}

/// The layers a pass of `renderTargetArrayLength` `length` renders: the length, but at least 1
/// (Metal's default length 0 renders one).
fn layers_of(length: u64) -> u64 {
    length.max(1)
}

/// Check `which` attachment of a texture of `facts` at `level`, `slice` and `depth_plane` in a
/// pass of `renderTargetArrayLength` `length`, as Metal's API validation does; refusals of what
/// step 6 builds later are `NotImplemented`.
pub(super) fn check_target(
    call: &'static str,
    which: Attachment,
    facts: &Facts,
    (level, slice, depth_plane): (u64, u64, u64),
    length: u64,
) -> Result<Subresource, Error> {
    let problem = |p| invalid(call, Problem::RenderPass(p));
    let kind = aspects(facts.pixel_format).unwrap_or(metal2vulkan::vk::ImageAspectFlags::empty());
    let fits = match which {
        Attachment::Color(_) => kind == metal2vulkan::vk::ImageAspectFlags::COLOR,
        Attachment::Depth => kind.contains(metal2vulkan::vk::ImageAspectFlags::DEPTH),
        Attachment::Stencil => kind.contains(metal2vulkan::vk::ImageAspectFlags::STENCIL),
    };
    // A 4:2:2 texture as a color attachment: plain Metal makes the pass, whose clear stores the
    // clear color in the format's encoding (its validation stops the pass,
    // "not color renderable"). The runtime does not build it: its gap. Likewise a `YCBCR8_420_2P`
    // one, whose pass Metal stores converted back into the surface's two planes (the consumer's
    // report): the runtime's gap. But for a format whose store a rule matches, which renders
    // through the texture's render shadow (`target`, `texture/ycbcr_image.rs`). A packed 10-bit
    // one's pass Metal fails, its planes' bytes and its error varying from run to run
    // (`ycbcr_pass_refused`): no exact form exists, and the refusal names the format.
    if matches!(which, Attachment::Color(_))
        && (is_four_two_two(facts.pixel_format)
            || (ycbcr(facts.pixel_format) && !ycbcr_renders(facts.pixel_format)))
    {
        return Err(Error::Unsupported(
            Unsupported::not_implemented(Feature::Format).naming_format(facts.pixel_format),
        ));
    }
    if !fits {
        return Err(problem(RenderPassProblem::AttachmentFormat(which)));
    }
    // A texture without `RenderTarget` usage renders through a stand-in (`layout`,
    // `depth_stencil`), a color, depth or stencil one alike (the module docs).
    match facts.texture_type {
        TextureType::Type2D
        | TextureType::Type2DArray
        | TextureType::Type2DMultisample
        | TextureType::Type2DMultisampleArray
        | TextureType::Cube
        | TextureType::CubeArray => {}
        // A texture buffer renders through its attachment twin, a 2D texture of one row over
        // its bytes (`Texture::attachment_twin`): Metal renders into one plain, and its
        // validation stops one made with render target usage (a probe).
        TextureType::TextureBuffer => {}
        // A 1D attachment renders one row through a 1D view (a probe: Metal renders one plain, and
        // its validation stops the pass, "texture must not be MTLTextureType1D").
        TextureType::Type1D | TextureType::Type1DArray => {}
        // A 3D attachment renders its depth planes from `depth_plane` (a probe).
        TextureType::Type3D => {
            let (slice, depth_plane) = used_indices(facts, slice, depth_plane);
            return check_planes(which, facts, (level, slice, depth_plane), length)
                .map_err(problem);
        }
    }
    let (slice, depth_plane) = used_indices(facts, slice, depth_plane);
    if level >= u64::from(facts.levels) || slice >= u64::from(facts.slices) || depth_plane != 0 {
        return Err(problem(RenderPassProblem::AttachmentSubresource(which)));
    }
    // Every attachment holds the pass's layers from its slice (measured, probe 12: a second
    // color attachment of fewer slices, and a 2D texture, stop a pass of two or more).
    let end = slice.saturating_add(layers_of(length));
    if end > u64::from(facts.slices) {
        return Err(problem(RenderPassProblem::ArrayLengthPastSlices {
            attachment: which,
            end: slice.saturating_add(length),
            slices: facts.slices,
        }));
    }
    let level = level as u32;
    Ok(Subresource {
        level: facts.base_level + level,
        layer: facts.base_layer + slice as u32,
        layers: (end - slice) as u32,
        extent: [
            (facts.extent[0] >> level).max(1),
            (facts.extent[1] >> level).max(1),
        ],
    })
}

/// The slice and depth plane Metal uses of a texture of `facts` named at `slice` and
/// `depth_plane`, as an attachment or a resolve destination: an index the texture's type does
/// not have is ignored, taken as 0. The slice of a 2D, 2D multisample, 1D or 3D texture (any
/// value: the pass renders slice 0, or the 3D texture's plane), and the depth plane of any but a
/// 3D texture (probes, an M4: 64x64 targets between neighbours, four fills alike, as an attachment
/// and as a resolve texture). A texture buffer's indices are taken as named (not probed).
pub(super) fn used_indices(facts: &Facts, slice: u64, depth_plane: u64) -> (u64, u64) {
    let slices = !matches!(
        facts.texture_type,
        TextureType::Type2D
            | TextureType::Type2DMultisample
            | TextureType::Type1D
            | TextureType::Type3D
    );
    match facts.texture_type {
        TextureType::TextureBuffer => (slice, depth_plane),
        TextureType::Type3D => (0, depth_plane),
        _ if slices => (slice, 0),
        _ => (0, 0),
    }
}

/// The depth planes a pass renders of a 3D attachment `which` of `facts` at `level`, from
/// `depth_plane` (its slice is 0): the pass's layers (`length`), within the level's depth. Metal
/// renders a plane, every layer of a layered pass from a plane, and a plane of another level
/// (a probe, an M3 Max and an M4, API validation passing each); its slice is ignored
/// ([`used_indices`]: the caller passes 0), a plane past the level's depth is
/// `AttachmentSubresource`, and layers past it `ArrayLengthPastSlices`, as for an array's
/// slices.
fn check_planes(
    which: Attachment,
    facts: &Facts,
    (level, slice, depth_plane): (u64, u64, u64),
    length: u64,
) -> Result<Subresource, RenderPassProblem> {
    if level >= u64::from(facts.levels) || slice != 0 {
        return Err(RenderPassProblem::AttachmentSubresource(which));
    }
    let level = level as u32;
    let depth = (facts.extent[2] >> level).max(1);
    if depth_plane >= u64::from(depth) {
        return Err(RenderPassProblem::AttachmentSubresource(which));
    }
    let end = depth_plane.saturating_add(layers_of(length));
    if end > u64::from(depth) {
        return Err(RenderPassProblem::ArrayLengthPastSlices {
            attachment: which,
            end: depth_plane.saturating_add(length),
            slices: depth,
        });
    }
    Ok(Subresource {
        level: facts.base_level + level,
        layer: depth_plane as u32,
        layers: (end - depth_plane) as u32,
        extent: [
            (facts.extent[0] >> level).max(1),
            (facts.extent[1] >> level).max(1),
        ],
    })
}

/// Metal's checks of `which` attachment's `load` (`None` for a late store action) and `store` where
/// its texture is `memoryless` (a Metal probe on an M4): a pass neither loads nor stores one.
/// `DontCare` and `Clear` loads and `DontCare`, `MultisampleResolve` and `Unknown` stores pass.
pub(super) fn check_memoryless(
    call: &'static str,
    which: Attachment,
    memoryless: bool,
    load: Option<LoadAction>,
    store: StoreAction,
) -> Result<(), Error> {
    if !memoryless {
        return Ok(());
    }
    let problem = |p| invalid(call, Problem::RenderPass(p));
    if load == Some(LoadAction::Load) {
        return Err(problem(RenderPassProblem::MemorylessLoad(which)));
    }
    if matches!(
        store,
        StoreAction::Store | StoreAction::StoreAndMultisampleResolve
    ) {
        return Err(problem(RenderPassProblem::MemorylessStore(which)));
    }
    Ok(())
}

/// Whether `texture` is memoryless.
fn memoryless(texture: &Texture) -> bool {
    texture.storage_mode() == super::values::StorageMode::Memoryless
}

/// The largest render target width and height Metal renders: a pass without attachments wider or
/// taller fails its command buffer from the pass on (measured on an M4, by a Metal probe and our
/// probes: the encoder is made, and the command buffer ends with an internal
/// error, with and without API validation; [`past_target_size`]). An attachment is never larger:
/// Metal's textures are at most 16384 wide and high.
const MAX_TARGET_SIZE: u64 = 16384;

/// Whether a pass of attachments of `extents` rendering `extent` is one without attachments past
/// [`MAX_TARGET_SIZE`] on a side, whose command buffer Metal fails from the pass on.
fn past_target_size(extents: &[([u32; 2], u32)], extent: [u32; 2]) -> bool {
    extents.is_empty() && extent.iter().any(|&side| u64::from(side) > MAX_TARGET_SIZE)
}

/// What plain Metal does with a pass without attachments at a `defaultRasterSampleCount`
/// ([`pass_default_samples`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DefaultSamples {
    /// It makes the encoder, and the pass runs.
    Runs,
    /// It ends the process making the encoder.
    Stops,
    /// It makes the encoder, and fails the command buffer from the pass on with
    /// `MTLCommandBufferErrorInternal`.
    FailsCommandBuffer,
}

/// What plain Metal does with a pass without attachments at `defaultRasterSampleCount`
/// `count`, by its low 8 bits (an Apple M4 under macOS 27.0.1, every count from 0 to 256 and 257,
/// 300, 511, 512, 513, 1024, 2^16, 2^31, 2^32, 2^63 and 2^64 - 1, each of them as its low 8 bits):
/// 1 to 7 run (4x4 passes ended at once, so what 3, 5, 6 and 7 rasterize is not measured, and no
/// pipeline of those counts is made), 0 (the descriptor's default), 8 to 15 and 128 to 255 end the
/// process, and 16 to 127 fail the command buffer ("Internal Error (0000010c:Internal Error)" on
/// the M3 Max under macOS 26.7, `0000010d` on the M4). The consumer's run of our probe
/// on the M3 Max under macOS 26.7 answered every case it shares alike, 0 included. Metal's API
/// validation stops every count but 1, 2 and 4.
pub(super) fn pass_default_samples(count: u64) -> DefaultSamples {
    match count & 0xff {
        1..=7 => DefaultSamples::Runs,
        16..=127 => DefaultSamples::FailsCommandBuffer,
        _ => DefaultSamples::Stops,
    }
}

/// The render area and samples of a pass of attachments of `extents` and `samples` (at least
/// one each, or none), under the descriptor's target sizes and default sample count.
pub(super) fn area(
    call: &'static str,
    targets: &[([u32; 2], u32)],
    target_size: [u64; 2],
    default_samples: u64,
) -> Result<([u32; 2], u32), Error> {
    let problem = |p| invalid(call, Problem::RenderPass(p));
    let limit = |axis: usize, value: u32| {
        let size = target_size[axis];
        if size == 0 {
            value
        } else {
            value.min(u32::try_from(size).unwrap_or(u32::MAX))
        }
    };
    let Some(&(first, samples)) = targets.first() else {
        if target_size.contains(&0) {
            return Err(nil(call, Problem::RenderPass(RenderPassProblem::NoTargets)));
        }
        if pass_default_samples(default_samples) == DefaultSamples::Stops {
            return Err(problem(RenderPassProblem::DefaultRasterSampleCount {
                count: default_samples,
            }));
        }
        let samples = u32::try_from(default_samples).unwrap_or(u32::MAX);
        return Ok(([limit(0, u32::MAX), limit(1, u32::MAX)], samples));
    };
    let mut extent = first;
    for &(other, other_samples) in targets {
        if other_samples != samples {
            return Err(problem(RenderPassProblem::SampleCounts));
        }
        extent = [extent[0].min(other[0]), extent[1].min(other[1])];
    }
    Ok(([limit(0, extent[0]), limit(1, extent[1])], samples))
}

/// Whether a framebuffer of `extent` and `layers` fits `max`, the device's `maxFramebufferWidth`,
/// `maxFramebufferHeight` and `maxFramebufferLayers`: `Unsupported` naming the first it passes.
fn framebuffer_fits(extent: [u32; 2], layers: u64, max: [u32; 3]) -> Result<(), Unsupported> {
    let needs = [
        (VkLimit::MaxFramebufferWidth, u64::from(extent[0])),
        (VkLimit::MaxFramebufferHeight, u64::from(extent[1])),
        (VkLimit::MaxFramebufferLayers, layers),
    ];
    for ((limit, required), available) in needs.into_iter().zip(max.map(u64::from)) {
        if required > available {
            return Err(Unsupported::needs(
                Feature::RenderTargetSize,
                Capability::Limit {
                    limit,
                    required,
                    available,
                },
            ));
        }
    }
    Ok(())
}

/// `descriptor` as Metal's pass takes it where its stencil attachment is a texture over a plane
/// of a surface that holds the texture's other aspect (`Depth32Float_Stencil8`'s depth,
/// `Texture::surface_stencil_unheld`): its stencil is not in the plane, and Metal's pass loads it
/// as 0, so a load of it is a clear to 0. Any other descriptor as it is.
pub(super) fn surface_stencil_loads(
    descriptor: &RenderPassDescriptor,
) -> Cow<'_, RenderPassDescriptor> {
    let s = &descriptor.stencil_attachment;
    let unheld = s
        .texture
        .as_ref()
        .is_some_and(Texture::surface_stencil_unheld);
    if !unheld || s.load_action != LoadAction::Load {
        return Cow::Borrowed(descriptor);
    }
    let mut taken = descriptor.clone();
    taken.stencil_attachment.load_action = LoadAction::Clear;
    taken.stencil_attachment.clear_stencil = 0;
    Cow::Owned(taken)
}

/// What a pass of `descriptor` on `device` renders into, checked (the module docs).
pub(super) fn layout(
    call: &'static str,
    maker: &CommandBuffer,
    descriptor: &RenderPassDescriptor,
) -> Result<PassLayout, Error> {
    let device = maker.device();
    let foreign = || invalid(call, Problem::ForeignObject);
    let not_yet = || Error::NotImplemented(RuntimeGap::Render);
    // A depth attachment's `CustomSampleDepthStore` stores as `Store` does (a probe: Metal stores
    // the same samples, with and without custom positions), so the pass is the one of `Store`; a
    // color or stencil attachment's is refused (the module docs).
    let stored;
    let descriptor = match descriptor.depth_attachment.store_action {
        StoreAction::CustomSampleDepthStore => {
            let mut store = descriptor.clone();
            store.depth_attachment.store_action = StoreAction::Store;
            stored = store;
            &stored
        }
        _ => descriptor,
    };
    // One entry each color attachment: room for them all at once, so no color target moves.
    let mut colors: Vec<Option<ColorTarget>> =
        Vec::with_capacity(descriptor.color_attachments.len());
    // Each attachment's extent and samples, held in place: at most every color and the
    // depth/stencil attachment.
    let mut extents = [([0, 0], 0); MAX_COLOR_ATTACHMENTS + 1];
    let mut attached = 0;
    let positioned = Some(!descriptor.sample_positions.is_empty());
    for (index, color) in descriptor.color_attachments.iter().enumerate() {
        let which = Attachment::Color(index as u32);
        if color.store_action == StoreAction::CustomSampleDepthStore {
            // Of an attachment without a texture, unmeasured.
            return Err(match color.texture {
                Some(_) => invalid(
                    call,
                    Problem::RenderPass(RenderPassProblem::CustomSampleDepthStore(which)),
                ),
                None => not_yet(),
            });
        }
        let Some(texture) = &color.texture else {
            // A resolve, or store action options, of an attachment without a texture is
            // unmeasured.
            if color.resolve_texture.is_some()
                || color.store_action.resolves()
                || color.store_action_options != StoreActionOptions::NONE
            {
                return Err(not_yet());
            }
            colors.push(None);
            continue;
        };
        // A two-plane YCbCr attachment's store is matched for a pass that clears it or leaves
        // it to the pass (`DontCare`), stores it, and resolves nothing: a load (its texels then
        // the planes' converted reads), another store and a resolve are not measured.
        if ycbcr_renders(texture.pixel_format())
            && (color.load_action == LoadAction::Load
                || color.store_action != StoreAction::Store
                || color.resolve_texture.is_some()
                || color.store_action_options != StoreActionOptions::NONE)
        {
            return Err(Error::Unsupported(
                Unsupported::not_implemented(Feature::Format).naming_format(texture.pixel_format()),
            ));
        }
        let target = target(
            call,
            device,
            which,
            texture,
            (color.level, color.slice, color.depth_plane),
            descriptor.render_target_array_length,
        )?;
        let memoryless = memoryless(texture);
        // A texture without `RenderTarget` usage renders through a stand-in that has it, copied
        // in where the pass loads it and back where it stores it (`attachment_stand_in.rs`), as
        // plain Metal renders into it (the module docs).
        let attachable = target
            .texture
            .image_plan()?
            .usage
            .contains(metal2vulkan::vk::ImageUsageFlags::COLOR_ATTACHMENT);
        let (target, stand_in) = match attachable {
            true => (target, None),
            false if memoryless => return Err(not_yet()),
            false => {
                attachment_stand_in::check_copies(&target)?;
                let stood = attachment_stand_in::stand_in(maker, &target, true)?;
                let own = StandIn {
                    own: target,
                    cause: Cause::Unattachable,
                    area: None,
                };
                (stood, Some(own))
            }
        };
        check_memoryless(
            call,
            which,
            memoryless,
            Some(color.load_action),
            color.store_action,
        )?;
        // A color attachment's store action options change nothing, as plain Metal's
        // (`sample_positions.rs`). A single-sampled one resolves as a copy of its texels
        // (`resolve::color_copy`).
        // An extended-range attachment's codes by either of Metal's routes: the pass's encoder
        // chooses as it ends, once it knows whether the pass draws (`ColorTarget::straight_clear`).
        let channels = color.clear_color.channels();
        // A word attachment clears to the word of Metal's codes.
        let word = metal2vulkan::metal::format::WordAttachment::of(
            texture.pixel_format(),
            device.compile_device().profile(),
        )
        .filter(|w| w.word_format() == target.format);
        let clear = match word {
            Some(word) => clear_value::word(texture.pixel_format(), channels, word),
            None => clear_value::held_color(
                texture.pixel_format(),
                target.format,
                channels,
                ClearRoute::ThroughHalf,
            ),
        }
        .ok_or_else(not_yet)?;
        let straight_clear = ExtendedRange::of(texture.pixel_format()).and_then(|_| {
            clear_value::color(texture.pixel_format(), channels, ClearRoute::Straight)
        });
        let wanted = Wanted {
            texture: color.resolve_texture.as_ref(),
            level: color.resolve_level,
            slice: color.resolve_slice,
            depth_plane: color.resolve_depth_plane,
            store: color.store_action,
        };
        let (resolve, copied_resolve) = match target.samples {
            1 => (
                None,
                resolve::color_copy(call, device, which, &target, &wanted)?,
            ),
            _ => (
                resolve::color(
                    call,
                    maker,
                    which,
                    &target,
                    &wanted,
                    (color.load_action, clear),
                )?,
                None,
            ),
        };
        extents[attached] = (target.extent, target.samples);
        attached += 1;
        colors.push(Some(ColorTarget {
            target,
            load: color.load_action,
            store: color.store_action,
            late_store: color.store_action == StoreAction::Unknown,
            memoryless,
            clear,
            straight_clear,
            resolve,
            copied_resolve,
            stand_in,
            planes: None,
        }));
    }
    while colors.last().is_some_and(Option::is_none) {
        colors.pop();
    }
    let depth = &descriptor.depth_attachment;
    let stencil = &descriptor.stencil_attachment;
    // A texture's linear image renders in place where every color attachment is linear and the
    // pass has no depth or stencil attachment; where the device renders it only so and the pass
    // breaks that, through an image its texture keeps, copied in and back
    // (`linear_attachment.rs`).
    let beside = match (depth.texture.is_some(), stencil.texture.is_some()) {
        (true, true) => Some("depth and stencil attachments"),
        (true, false) => Some("a depth attachment"),
        (false, true) => Some("a stencil attachment"),
        (false, false) => None,
    };
    linear_attachment::apart(maker, &mut colors, beside)?;
    // A resolve, or store action options, of an attachment without a texture is unmeasured.
    let unattached = |texture: &Option<Texture>,
                      resolve: &Option<Texture>,
                      store: StoreAction,
                      options: StoreActionOptions| {
        texture.is_none()
            && (resolve.is_some() || store.resolves() || options != StoreActionOptions::NONE)
    };
    if unattached(
        &depth.texture,
        &depth.resolve_texture,
        depth.store_action,
        depth.store_action_options,
    ) || unattached(
        &stencil.texture,
        &stencil.resolve_texture,
        stencil.store_action,
        stencil.store_action_options,
    ) {
        return Err(not_yet());
    }
    if stencil.store_action == StoreAction::CustomSampleDepthStore {
        if stencil.texture.is_none() {
            return Err(not_yet());
        }
        return Err(invalid(
            call,
            Problem::RenderPass(RenderPassProblem::CustomSampleDepthStore(
                Attachment::Stencil,
            )),
        ));
    }
    // A depth attachment's store action options change nothing, as plain Metal's
    // (`sample_positions.rs`); a stencil attachment's are checked.
    if stencil.texture.is_some() {
        sample_positions::check_options(
            call,
            Attachment::Stencil,
            stencil.store_action_options,
            stencil.store_action,
            positioned,
        )?;
    }
    let depth_stencil = depth_stencil(call, maker, descriptor)?;
    if let Some(ds) = &depth_stencil {
        extents[attached] = (ds.target.extent, ds.target.samples);
        attached += 1;
    }
    let extents = &extents[..attached];
    if let Some(buffer) = &descriptor.visibility_result_buffer {
        if !buffer.device().same(device) {
            return Err(foreign());
        }
    }
    let (extent, samples) = area(
        call,
        extents,
        [
            descriptor.render_target_width,
            descriptor.render_target_height,
        ],
        descriptor.default_raster_sample_count,
    )?;
    // A pass without attachments past 16384 a side: Metal makes the encoder and fails the command
    // buffer from it on (`CommandBufferError::RenderTargetSize`). Nothing of it runs, so it
    // records into the device's largest framebuffer.
    let oversized = past_target_size(extents, extent);
    let extent = match oversized {
        true => {
            let [width, height, _] = device.shared().max_framebuffer;
            [extent[0].min(width), extent[1].min(height)]
        }
        false => extent,
    };
    // A pass without attachments at a default sample count Metal fails the command buffer at:
    // nothing of it runs either (no pipeline of its count is made, `raster_samples.rs`).
    let failing_samples = extents.is_empty()
        && pass_default_samples(descriptor.default_raster_sample_count)
            == DefaultSamples::FailsCommandBuffer;
    let fails_command_buffer = match (oversized, failing_samples) {
        (true, _) => Some(CommandBufferError::RenderTargetSize),
        (false, true) => Some(CommandBufferError::DefaultRasterSampleCount),
        (false, false) => None,
    };
    // Separate depth and stencil textures of different sizes, against the render area
    // (`depth_stencil_scratch.rs`).
    if let Some(separate) = depth_stencil.as_ref().and_then(|ds| ds.separate.as_ref()) {
        depth_stencil_scratch::check_sizes(separate, extent, samples)?;
    }
    let depth_stencil_flags = match &depth_stencil {
        Some(ds) => Some(ds.target.texture.image_plan()?.flags),
        None => None,
    };
    let sample_positions = sample_positions::grid(
        call,
        device,
        &descriptor.sample_positions,
        samples,
        depth_stencil_flags,
    )?;
    // A framebuffer past the device's limits, refused naming the limit (no owned device's are
    // below Metal's 16384 a side). Every attachment holds the pass's layers (checked above), so
    // only a pass without attachments asks for more layers than its textures' 2048 slices.
    let layers = layers_of(descriptor.render_target_array_length);
    framebuffer_fits(extent, layers, device.shared().max_framebuffer)
        .map_err(Error::Unsupported)?;
    let layers = layers as u32;
    let tile = tile_pass::check(
        [descriptor.tile_width, descriptor.tile_height],
        descriptor.imageblock_sample_length,
        descriptor.threadgroup_memory_length,
        samples,
    )
    .map_err(|refusal| match refusal {
        TileRefusal::Stops(problem) => invalid(call, Problem::RenderPass(problem)),
        TileRefusal::Nil(problem) => nil(call, Problem::RenderPass(problem)),
        TileRefusal::NotMeasured => Error::NotImplemented(RuntimeGap::TileAtSampleCount),
    })?;
    let fetch_inputs = !colors.is_empty()
        && super::framebuffer_fetch::through_inputs(device.compile_device().profile());
    Ok(PassLayout {
        colors,
        depth_stencil,
        extent,
        layers,
        samples,
        visibility: descriptor.visibility_result_buffer.clone(),
        fetch_inputs,
        tile,
        sample_positions,
        fails_command_buffer,
    })
}

/// `which` attachment of `texture` at `(level, slice, depth_plane)` in a pass of
/// `renderTargetArrayLength` `length`, checked.
fn target(
    call: &'static str,
    device: &Device,
    which: Attachment,
    texture: &Texture,
    subresource: (u64, u64, u64),
    length: u64,
) -> Result<Target, Error> {
    if !texture.device().same(device) {
        return Err(invalid(call, Problem::ForeignObject));
    }
    let facts = Facts::of(texture.record());
    let sub = check_target(call, which, &facts, subresource, length)?;
    // A 2D texture over a buffer renders into its aliasing image, or into its shadow, which the
    // encoder copies in before the pass and back after it (`render/shadows.rs`); a texture
    // buffer renders through its attachment twin, such a texture of one row over its bytes, and a
    // rotated texture through its twin of the pass's frame, whose copies turn the texels
    // (`texture/rotation.rs`). A two-plane YCbCr color attachment renders through the texture's
    // render shadow, whose copy back stores every texel's codes into the planes, as Metal stores
    // them (`texture/ycbcr_image.rs`).
    let texture = &match ycbcr_renders(facts.pixel_format) {
        true => texture.ycbcr_render_shadow(call)?,
        false => texture.pass_texture()?,
    };
    let plan = texture.image_plan()?;
    // A 3D attachment is a 2D view of its planes, which its image takes 2D-array compatible
    // (`image_plan.rs`): a device whose portability subset withholds `imageView2DOn3DImage` makes
    // no such image, and a sparse one is never so.
    if facts.texture_type == TextureType::Type3D
        && !plan
            .flags
            .contains(metal2vulkan::vk::ImageCreateFlags::CREATE_2D_ARRAY_COMPATIBLE)
    {
        let profile = device.compile_device().profile();
        if profile
            .enabled()
            .portability_withholds(metal2vulkan::device::VkFeature::ImageView2DOn3DImage)
        {
            return Err(Error::Unsupported(metal2vulkan::error::Unsupported::needs(
                metal2vulkan::error::Feature::LayeredRendering,
                metal2vulkan::error::Capability::Feature(
                    metal2vulkan::device::VkFeature::ImageView2DOn3DImage,
                ),
            )));
        }
        return Err(Error::NotImplemented(RuntimeGap::Render));
    }
    let needed = match which {
        Attachment::Color(_) => metal2vulkan::vk::ImageUsageFlags::COLOR_ATTACHMENT,
        Attachment::Depth | Attachment::Stencil => {
            metal2vulkan::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        }
    };
    // A texture without `RenderTarget` usage whose image lacks the attachment's usage renders
    // through a stand-in, whose image the device makes where it renders the format (`layout`,
    // `depth_stencil`, `attachment_stand_in::stand_in`).
    let stood = !plan.usage.contains(needed) && !facts.usage.contains(TextureUsage::RENDER_TARGET);
    let usage = match which {
        Attachment::Color(_) => metal2vulkan::device::FormatUsage::ColorAttachment,
        _ => metal2vulkan::device::FormatUsage::DepthStencilAttachment,
    };
    if !plan.usage.contains(needed) {
        // A format no attachment represents (an extended-range one) is refused as the format table
        // refuses it, a texture without `RenderTarget` usage too: its stand-in would render in its
        // image's format as a plain attachment of that format renders (an RGBX texture's twin would
        // store the fragment's alpha where Metal stores green, `metal::format::x_channel`).
        let profile = device.compile_device().profile();
        if let Err(metal2vulkan::error::Error::Unsupported(u)) =
            metal2vulkan::metal::format::vk_format(plan.pixel_format, &[usage].into(), profile)
        {
            if u.needs == metal2vulkan::error::Needs::NotImplemented {
                return Err(Error::Unsupported(u));
            }
        }
    }
    if !stood && !plan.usage.contains(needed) {
        // The device lacks the format as an attachment (`new_texture` kept the texture for its
        // other uses).
        return Err(Error::Unsupported(metal2vulkan::error::Unsupported::needs(
            metal2vulkan::error::Feature::Format,
            metal2vulkan::error::Capability::Format {
                format: plan.format,
                usage,
            },
        )));
    }
    // A view of another format renders in that format (`attachment_format`); a texture without
    // `RenderTarget` usage, through its stand-in, in its own (the module docs), which for a depth
    // or stencil attachment must hold the aspect: a `Depth32Float` texture held in an
    // `R32_SFLOAT` image beside a depth twin (`image_plan.rs`) copies into no depth stand-in
    // exactly.
    let holds = |format: metal2vulkan::vk::Format| match which {
        Attachment::Color(_) => true,
        Attachment::Depth => format.block().is_some_and(|b| b.depth.is_some()),
        Attachment::Stencil => format.block().is_some_and(|b| b.stencil.is_some()),
    };
    let format = match stood {
        false => attachment_format(which, plan, facts.pixel_format, device)?,
        true if plan.pixel_format == facts.pixel_format && holds(plan.format) => plan.format,
        true => return Err(Error::NotImplemented(RuntimeGap::Render)),
    };
    Ok(Target {
        texture: texture.clone(),
        level: sub.level,
        layer: sub.layer,
        layers: sub.layers,
        format,
        samples: facts.samples,
        extent: sub.extent,
    })
}

/// Whether `target` is not of a 2D image: planes of a 3D image (a 2D view of them), or a 1D
/// image (a 1D view). The runtime's copies, clears and 2D views of an attachment are of 2D
/// images: those it would make of such a target (a stand-in's or a scratch's copies, a clear
/// before the pass, a storage view, the exact resolve's view) are not built, and the pass takes
/// another route or refuses it `NotImplemented(Render)`.
pub(super) fn not_2d(target: &Target) -> bool {
    target
        .texture
        .image_plan()
        .is_ok_and(|plan| plan.image_type != metal2vulkan::vk::ImageType::TYPE_2D)
}

/// The Vulkan format `which` attachment (or resolve target) of a texture presenting `pixel_format`
/// renders in, over an image of `plan` that has the attachment's usage: the image's own where the
/// texture presents the image's format. A color view of another format renders in that format's
/// encoding, as Metal's does (the sRGB twin, `R32Uint` over `RGBA8Unorm`; Metal probe case
/// `r.view_srgb` on the M3 Max and an M4; a probe): its direct Vulkan format, which the image takes
/// as a view format (a texture with `PixelFormatView` usage, or the sRGB twin Metal views without
/// it) and the device renders (`Unsupported` naming the format where it does not). A depth or
/// stencil view (the stencil view of a combined texture) renders in its image's format, the one the
/// device represents the view's format by; one the device represents otherwise is not built, nor is
/// a color view without a direct format or one the image cannot take (`NotImplemented(Render)`).
/// A color format the device samples but does not render, whose word it renders
/// (`metal::format::WordAttachment`), renders in the word's format where the image takes that
/// view, as the pipeline's fragment function writes it.
pub(super) fn attachment_format(
    which: Attachment,
    plan: &ImagePlan,
    pixel_format: PixelFormat,
    device: &Device,
) -> Result<metal2vulkan::vk::Format, Error> {
    use metal2vulkan::device::FormatUsage;
    let profile = device.compile_device().profile();
    let family = block_family(pixel_format) == block_family(plan.pixel_format);
    // A color format the device samples but does not render renders through the view of its word
    // (`metal::format::WordAttachment`), which a pipeline's fragment function writes, where the
    // image takes that view.
    if let Attachment::Color(_) = which {
        if let Some(word) = metal2vulkan::metal::format::WordAttachment::of(pixel_format, profile) {
            let format = word.word_format();
            if views_as(plan, format, family) {
                return Ok(format);
            }
        }
    }
    if plan.pixel_format == pixel_format {
        return Ok(plan.format);
    }
    let not_built = || Error::NotImplemented(RuntimeGap::Render);
    if let Attachment::Depth | Attachment::Stencil = which {
        let usages: std::collections::BTreeSet<_> = [FormatUsage::DepthStencilAttachment].into();
        return match vk_format(pixel_format, &usages, profile) {
            Ok(mapping) if mapping.format == plan.format => Ok(plan.format),
            _ => Err(not_built()),
        };
    }
    let direct = direct_format(pixel_format).ok_or_else(not_built)?;
    let usage = FormatUsage::ColorAttachment;
    if !profile.supports_format(direct, usage) {
        return Err(Error::Unsupported(metal2vulkan::error::Unsupported::needs(
            metal2vulkan::error::Feature::Format,
            metal2vulkan::error::Capability::Format {
                format: direct,
                usage,
            },
        )));
    }
    if !views_as(plan, direct, family) {
        return Err(not_built());
    }
    Ok(direct)
}

/// Whether the stencil-only attachment `stencil` of `descriptor` renders through a scratch image:
/// the pass takes its custom sample positions (`sample_positions::taken`), and the attachment's
/// image has no depth aspect, so it was not made for them.
fn stencil_positioned(
    device: &Device,
    descriptor: &RenderPassDescriptor,
    stencil: &Target,
) -> Result<bool, Error> {
    let flags = stencil.texture.image_plan()?.flags;
    let depthless = stencil.format.block().is_some_and(|b| b.depth.is_none());
    Ok(depthless
        && !flags
            .contains(metal2vulkan::vk::ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT)
        && sample_positions::taken(device, &descriptor.sample_positions, stencil.samples))
}

/// Whether `target`, the one texture of `descriptor`'s depth/stencil attachment, renders through
/// a stand-in made for the pass's custom sample positions (`attachment_stand_in.rs`): the pass
/// takes them (`sample_positions::taken`), and the attachment's image has a depth aspect, keeps
/// its texels (not memoryless) and copies both ways, but was not made for them (a caller's image,
/// wrapped without the flag), or was on a driver that keeps a depth rendered at them per sample
/// only through a copy (`DriverFact::PositionedDepthReadAtStandardPositions`: its reads, and
/// later passes at other positions, take the depth at their own positions). An image of a format
/// the device does not copy both ways stays refused: without the flag by
/// `sample_positions::grid`, with it on such a driver here.
fn positioned_stand_in(
    device: &Device,
    descriptor: &RenderPassDescriptor,
    target: &Target,
) -> Result<bool, Error> {
    use metal2vulkan::vk::{ImageCreateFlags, ImageUsageFlags};
    let plan = target.texture.image_plan()?;
    let both = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
    let flagged = plan
        .flags
        .contains(ImageCreateFlags::SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT);
    let planes = device
        .properties()
        .driver_facts
        .contains(&DriverFact::PositionedDepthReadAtStandardPositions);
    let positioned = target.format.block().is_some_and(|b| b.depth.is_some())
        && (!flagged || planes)
        && !memoryless(&target.texture)
        && sample_positions::taken(device, &descriptor.sample_positions, target.samples);
    if positioned && flagged && !plan.usage.contains(both) {
        return Err(Error::NotImplemented(RuntimeGap::SamplePositionsAtBegin));
    }
    Ok(positioned && plan.usage.contains(both))
}

/// The depth/stencil attachment of `descriptor`, checked (the module docs).
fn depth_stencil(
    call: &'static str,
    maker: &CommandBuffer,
    descriptor: &RenderPassDescriptor,
) -> Result<Option<DepthStencilTarget>, Error> {
    let device = maker.device();
    let d = &descriptor.depth_attachment;
    let s = &descriptor.stencil_attachment;
    let depth = match &d.texture {
        Some(texture) => {
            let target = target(
                call,
                device,
                Attachment::Depth,
                texture,
                (d.level, d.slice, d.depth_plane),
                descriptor.render_target_array_length,
            )?;
            // A NaN clear depth of a format whose store of one is not measured.
            let measured = matches!(
                texture.pixel_format(),
                PixelFormat::Depth32Float
                    | PixelFormat::Depth32FloatStencil8
                    | PixelFormat::Depth16Unorm
            );
            if d.load_action == LoadAction::Clear && d.clear_depth.is_nan() && !measured {
                return Err(invalid(
                    call,
                    Problem::RenderPass(RenderPassProblem::ClearDepth),
                ));
            }
            check_memoryless(
                call,
                Attachment::Depth,
                memoryless(texture),
                Some(d.load_action),
                d.store_action,
            )?;
            Some(target)
        }
        None => None,
    };
    // A NaN clear depth (`nan_clear.rs`): written before the pass and loaded, or cleared to the
    // value the pass clears it to.
    let nan = match &depth {
        Some(target) if d.load_action == LoadAction::Clear && d.clear_depth.is_nan() => Some(
            nan_clear::nan_depth(device, target, d.store_action, memoryless(&target.texture)),
        ),
        _ => None,
    };
    let written_nan = nan == Some(NanDepth::Written);
    if written_nan {
        if let Some(target) = &depth {
            target.texture.degrade_signed_depth(call, true);
        }
    }
    let depth_load = match written_nan {
        true => LoadAction::Load,
        false => d.load_action,
    };
    // A new texture's depth pattern, a NaN with the sign set, that the pass loads before a write
    // covered it: Metal's test compares its magnitude, so the depth is switched on as a write that
    // may store one switches it, before the pass decides its magnitude stand-in below
    // (`fresh_unwritten.rs`).
    if let Some(target) = depth.as_ref().filter(|_| d.load_action == LoadAction::Load) {
        if FreshUnwritten::loads_signed_depth(device, target)?
            && !target.texture.switch_on_fresh_pattern()
        {
            target.texture.degrade_signed_depth("new_texture", true);
        }
    }
    let clear_depth = match nan {
        Some(NanDepth::Cleared(value)) => value,
        Some(NanDepth::Written) => 1.0,
        None => d.clear_depth,
    };
    let stencil = match &s.texture {
        Some(texture) => {
            let target = target(
                call,
                device,
                Attachment::Stencil,
                texture,
                (s.level, s.slice, s.depth_plane),
                descriptor.render_target_array_length,
            )?;
            check_memoryless(
                call,
                Attachment::Stencil,
                memoryless(texture),
                Some(s.load_action),
                s.store_action,
            )?;
            Some(target)
        }
        None => None,
    };
    let (target, separate) = match (&depth, &stencil) {
        (None, None) => return Ok(None),
        // A stencil-only texture at custom sample positions: its image has no depth aspect, so
        // no custom-location flag, and the pass renders its stencil through a combined scratch
        // image that has it (`sample_positions.rs`).
        (None, Some(t)) if stencil_positioned(device, descriptor, t)? => {
            let separate = Separate {
                depth: None,
                stencil: t.clone(),
            };
            let scratch = depth_stencil_scratch::scratch(
                maker,
                &separate,
                [depth_load, s.load_action],
                [d.store_action, s.store_action],
            )?;
            (scratch, Some(separate))
        }
        (Some(t), None) | (None, Some(t)) => (t.clone(), None),
        (Some(a), Some(b))
            if a.texture.same_storage(&b.texture)
                && (a.level, a.layer, a.format) == (b.level, b.layer, b.format) =>
        {
            (a.clone(), None)
        }
        (Some(a), Some(b)) => {
            let problem = |p| invalid(call, Problem::RenderPass(p));
            // Textures of different sizes render over the smaller, as plain Metal renders them
            // (`depth_stencil_scratch.rs`; checked against the render area in `layout`).
            if a.samples != b.samples {
                return Err(problem(RenderPassProblem::SampleCounts));
            }
            let separate = Separate {
                depth: Some(a.clone()),
                stencil: b.clone(),
            };
            let scratch = depth_stencil_scratch::scratch(
                maker,
                &separate,
                [depth_load, s.load_action],
                [d.store_action, s.store_action],
            )?;
            (scratch, Some(separate))
        }
    };
    // A texture whose image lacks depth/stencil attachment usage (a texture without
    // `RenderTarget` usage, `target`) renders through images that have it, as plain Metal
    // renders it (the module docs): one texture through a stand-in, separate ones through the
    // combined scratch image, each copied to and from it. Each needs copies both ways; a
    // memoryless one, which keeps no texels to copy, is not built.
    let unattachable = |t: &Target| {
        t.texture.image_plan().map(|plan| {
            !plan
                .usage
                .contains(metal2vulkan::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        })
    };
    for t in depth.iter().chain(&stencil) {
        if unattachable(t)? {
            if memoryless(&t.texture) {
                return Err(Error::NotImplemented(RuntimeGap::Render));
            }
            attachment_stand_in::check_copies(t)?;
        }
    }
    // One texture not made for the pass's custom sample positions (a caller's image) renders
    // through a stand-in made for them, each aspect copied in where the pass loads it and back
    // where it stores it (`attachment_stand_in.rs`), as plain Metal renders it at them; so does
    // one without depth/stencil attachment usage.
    // A depth that may hold a negative value or a NaN, which the pass loads, renders over its
    // magnitude stand-in, so the device's depth test answers as Metal's
    // (`texture/signed_depth.rs`): one flag read where the pass loads its depth.
    let signed = depth
        .as_ref()
        .filter(|t| depth_load == LoadAction::Load && t.texture.renders_over_magnitude());
    let cause = match separate.is_none() {
        false => None,
        true if unattachable(&target)? => Some(Cause::Unattachable),
        true if positioned_stand_in(device, descriptor, &target)? => Some(Cause::Positions),
        true if signed.is_some() => Some(Cause::Magnitude),
        true => None,
    };
    // Through another image, the pass tests the depth as it is.
    if let Some(own) = signed.filter(|_| cause != Some(Cause::Magnitude)) {
        let through = match separate.is_some() {
            true => "the scratch image of separate depth and stencil textures",
            false => "a stand-in for the pass's custom sample positions",
        };
        signed_depth::degrade_through(device, own, through);
    }
    let (target, stand_in) = match cause {
        // The image its texture keeps for the queue, which the pass's copies fill first.
        Some(Cause::Magnitude) => {
            let queue = maker.command_queue().id();
            let stood = attachment_stand_in::kept_backing(maker, &target, queue, true)?;
            let own = StandIn {
                own: target,
                cause: Cause::Magnitude,
                area: None,
            };
            (stood, Some(own))
        }
        Some(cause) => {
            let stood = attachment_stand_in::stand_in(maker, &target, true)?;
            let own = StandIn {
                own: target,
                cause,
                area: None,
            };
            (stood, Some(own))
        }
        None => (target, None),
    };
    // The aspects Metal's format of the attached image has: an aspect it has that no
    // attachment names keeps its texels (a stencil view of a combined texture keeps its depth).
    let metal = |t: &Option<Target>| {
        t.as_ref()
            .and_then(|t| t.texture.image_plan().ok())
            .and_then(|plan| aspects(plan.pixel_format))
            .unwrap_or(metal2vulkan::vk::ImageAspectFlags::empty())
    };
    let metal = metal(&depth) | metal(&stencil);
    // A memoryless texture keeps no texels.
    let kept = !depth.iter().chain(&stencil).any(|t| memoryless(&t.texture));
    let resolve = resolve::depth_stencil(
        call,
        maker,
        &target,
        separate.as_ref(),
        Wanted::of(
            depth.is_some(),
            &d.resolve_texture,
            (d.resolve_level, d.resolve_slice, d.resolve_depth_plane),
            d.store_action,
        )
        .as_ref(),
        Wanted::of(
            stencil.is_some(),
            &s.resolve_texture,
            (s.resolve_level, s.resolve_slice, s.resolve_depth_plane),
            s.store_action,
        )
        .as_ref(),
        d.depth_resolve_filter,
        s.stencil_resolve_filter,
    )?;
    Ok(Some(DepthStencilTarget {
        target,
        depth: depth.map(|_| AspectTarget {
            load: depth_load,
            store: d.store_action,
            late_store: d.store_action == StoreAction::Unknown,
            options: d.store_action_options,
            memoryless: d.texture.as_ref().is_some_and(memoryless),
            clear: AspectClear::Depth(clear_value::depth(
                d.texture
                    .as_ref()
                    .map_or(PixelFormat::Depth32Float, Texture::pixel_format),
                clear_depth.clamp(0.0, 1.0),
            )),
            written_nan,
        }),
        stencil: stencil.map(|_| AspectTarget {
            load: s.load_action,
            store: s.store_action,
            late_store: s.store_action == StoreAction::Unknown,
            options: s.store_action_options,
            memoryless: s.texture.as_ref().is_some_and(memoryless),
            clear: AspectClear::Stencil(clear_value::stencil(s.clear_stencil)),
            written_nan: false,
        }),
        depth_format: d.texture.as_ref().map(Texture::pixel_format),
        keep_depth: d.texture.is_none()
            && metal.contains(metal2vulkan::vk::ImageAspectFlags::DEPTH)
            && kept,
        keep_stencil: s.texture.is_none()
            && metal.contains(metal2vulkan::vk::ImageAspectFlags::STENCIL)
            && kept,
        separate,
        resolve,
        stand_in,
        depth_resolve_target: ResolveTarget::named(
            &d.resolve_texture,
            (d.resolve_level, d.resolve_slice, d.resolve_depth_plane),
        ),
        stencil_resolve_target: ResolveTarget::named(
            &s.resolve_texture,
            (s.resolve_level, s.resolve_slice, s.resolve_depth_plane),
        ),
        depth_filter: d.depth_resolve_filter,
        stencil_filter: s.stencil_resolve_filter,
        hidden: false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(pf: PixelFormat, ty: TextureType) -> Facts {
        Facts {
            pixel_format: pf,
            texture_type: ty,
            usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
            levels: 2,
            slices: 3,
            extent: [8, 8, 1],
            samples: 1,
            base_level: 0,
            base_layer: 0,
        }
    }

    fn problem(error: Error) -> RenderPassProblem {
        match error {
            Error::Invalid(invalid) => match invalid.problem {
                Problem::RenderPass(p) => p,
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
    }

    /// A memoryless attachment is neither loaded nor stored (a Metal probe on an M4); the other
    /// actions pass, and every action passes for an attachment that is not memoryless.
    #[test]
    fn memoryless_attachments_are_neither_loaded_nor_stored() {
        let call = "render_command_encoder";
        let which = Attachment::Depth;
        for load in [LoadAction::Clear, LoadAction::DontCare] {
            for store in [
                StoreAction::DontCare,
                StoreAction::MultisampleResolve,
                StoreAction::Unknown,
            ] {
                assert_eq!(
                    check_memoryless(call, which, true, Some(load), store),
                    Ok(())
                );
            }
        }
        let load = check_memoryless(
            call,
            which,
            true,
            Some(LoadAction::Load),
            StoreAction::DontCare,
        );
        assert_eq!(
            load.map_err(problem),
            Err(RenderPassProblem::MemorylessLoad(which))
        );
        for store in [StoreAction::Store, StoreAction::StoreAndMultisampleResolve] {
            assert_eq!(
                check_memoryless(call, which, true, None, store).map_err(problem),
                Err(RenderPassProblem::MemorylessStore(which))
            );
            assert_eq!(
                check_memoryless(call, which, false, Some(LoadAction::Load), store),
                Ok(())
            );
        }
    }

    /// Metal's defaults (probe 1 on an M4).
    #[test]
    fn descriptor_defaults_are_metals() {
        let d = RenderPassDescriptor::new();
        let c = &d.color_attachments[7];
        assert_eq!(
            (c.load_action, c.store_action, c.clear_color),
            (
                LoadAction::DontCare,
                StoreAction::Store,
                ClearColor::default()
            )
        );
        assert_eq!(
            (
                d.depth_attachment.load_action,
                d.depth_attachment.store_action,
                d.depth_attachment.clear_depth
            ),
            (LoadAction::Clear, StoreAction::DontCare, 1.0)
        );
        assert_eq!(
            (
                d.stencil_attachment.load_action,
                d.stencil_attachment.store_action,
                d.stencil_attachment.clear_stencil
            ),
            (LoadAction::Clear, StoreAction::DontCare, 0)
        );
        assert_eq!(
            (
                d.render_target_array_length,
                d.render_target_width,
                d.render_target_height,
                d.default_raster_sample_count
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(
            (
                d.tile_width,
                d.tile_height,
                d.imageblock_sample_length,
                d.threadgroup_memory_length
            ),
            (0, 0, 0, 0)
        );
    }

    /// Level 1 slice 2 of an 8x8 array with two levels is a 4x4 subresource at that level and
    /// layer (probe 1's `level_slice`); a view counts from its own first level and slice.
    #[test]
    fn a_target_is_its_level_and_slice() {
        let f = facts(PixelFormat::RGBA8Unorm, TextureType::Type2DArray);
        let sub = check_target("t", Attachment::Color(0), &f, (1, 2, 0), 0).unwrap();
        assert_eq!(
            (sub.level, sub.layer, sub.layers, sub.extent),
            (1, 2, 1, [4, 4])
        );
        let view = Facts {
            base_level: 3,
            base_layer: 5,
            ..f
        };
        let sub = check_target("t", Attachment::Color(0), &view, (1, 2, 0), 1).unwrap();
        assert_eq!((sub.level, sub.layer), (4, 7));
    }

    /// What Metal's validation refuses of an attachment (probe 1): a format of another kind, a
    /// level or slice past the texture. A color, depth or stencil texture without `RenderTarget`
    /// usage is taken (a stand-in renders it).
    #[test]
    fn invalid_targets_are_refused() {
        let color = facts(PixelFormat::RGBA8Unorm, TextureType::Type2D);
        let read_only = Facts {
            usage: TextureUsage::SHADER_READ,
            ..color
        };
        let c0 = Attachment::Color(0);
        assert!(check_target("t", c0, &read_only, (0, 0, 0), 0).is_ok());
        let depth_read_only = Facts {
            usage: TextureUsage::SHADER_READ,
            ..facts(PixelFormat::Depth32Float, TextureType::Type2D)
        };
        assert!(check_target("t", Attachment::Depth, &depth_read_only, (0, 0, 0), 0).is_ok());
        let stencil_read_only = Facts {
            usage: TextureUsage::SHADER_READ,
            ..facts(PixelFormat::Stencil8, TextureType::Type2D)
        };
        assert!(check_target("t", Attachment::Stencil, &stencil_read_only, (0, 0, 0), 0).is_ok());
        assert_eq!(
            problem(check_target("t", Attachment::Depth, &color, (0, 0, 0), 0).unwrap_err()),
            RenderPassProblem::AttachmentFormat(Attachment::Depth)
        );
        let depth = facts(PixelFormat::Depth32Float, TextureType::Type2D);
        assert_eq!(
            problem(check_target("t", c0, &depth, (0, 0, 0), 0).unwrap_err()),
            RenderPassProblem::AttachmentFormat(c0)
        );
        assert_eq!(
            problem(check_target("t", Attachment::Stencil, &depth, (0, 0, 0), 0).unwrap_err()),
            RenderPassProblem::AttachmentFormat(Attachment::Stencil)
        );
        // A 4:2:2 texture as a color attachment: the runtime's gap, not Metal's refusal.
        for pf in [PixelFormat::GBGR422, PixelFormat::BGRG422] {
            let pairs = facts(pf, TextureType::Type2D);
            assert!(
                matches!(
                    check_target("t", c0, &pairs, (0, 0, 0), 0),
                    Err(Error::Unsupported(u)) if u.feature == Feature::Format
                ),
                "{pf:?}"
            );
        }
        // A level past the texture, and a slice past an array's (an index the type lacks is
        // ignored, `an_index_the_type_lacks_is_ignored`).
        let array = facts(PixelFormat::RGBA8Unorm, TextureType::Type2DArray);
        for (texture, level, slice) in [(&color, 2, 0), (&array, 2, 0), (&array, 0, 3)] {
            assert_eq!(
                problem(check_target("t", c0, texture, (level, slice, 0), 0).unwrap_err()),
                RenderPassProblem::AttachmentSubresource(c0)
            );
        }
        // A 3D texture's planes are those of the level's depth.
        let volume = Facts {
            slices: 1,
            extent: [8, 8, 4],
            ..facts(PixelFormat::RGBA8Unorm, TextureType::Type3D)
        };
        for (level, slice, plane) in [(2, 0, 0), (0, 0, 4), (1, 0, 2)] {
            assert_eq!(
                problem(check_target("t", c0, &volume, (level, slice, plane), 0).unwrap_err()),
                RenderPassProblem::AttachmentSubresource(c0)
            );
        }
        assert_eq!(
            problem(check_target("t", c0, &volume, (0, 0, 3), 2).unwrap_err()),
            RenderPassProblem::ArrayLengthPastSlices {
                attachment: c0,
                end: 5,
                slices: 4,
            }
        );
    }

    /// An index the texture's type does not have is ignored (probes, an M4): a 2D texture's slice,
    /// any, renders slice 0; a 3D texture's slice renders its plane; an array's or a 2D texture's
    /// depth plane is nothing. A layered pass still needs its layers, and a texture buffer's
    /// indices count.
    #[test]
    fn an_index_the_type_lacks_is_ignored() {
        let c0 = Attachment::Color(0);
        let flat = Facts {
            slices: 1,
            ..facts(PixelFormat::RGBA8Unorm, TextureType::Type2D)
        };
        for named in [(0, 1, 0), (0, 5, 0), (0, 0, 1), (1, 5, 3)] {
            let sub = check_target("t", c0, &flat, named, 0).unwrap();
            assert_eq!((sub.level, sub.layer, sub.layers), (named.0 as u32, 0, 1));
        }
        let array = facts(PixelFormat::RGBA8Unorm, TextureType::Type2DArray);
        let sub = check_target("t", c0, &array, (0, 2, 1), 0).unwrap();
        assert_eq!((sub.layer, sub.layers), (2, 1));
        let volume = Facts {
            slices: 1,
            extent: [8, 8, 4],
            ..facts(PixelFormat::RGBA8Unorm, TextureType::Type3D)
        };
        let sub = check_target("t", c0, &volume, (0, 3, 2), 0).unwrap();
        assert_eq!((sub.layer, sub.layers), (2, 1));
        assert_eq!(
            problem(check_target("t", c0, &flat, (0, 1, 0), 2).unwrap_err()),
            RenderPassProblem::ArrayLengthPastSlices {
                attachment: c0,
                end: 2,
                slices: 1,
            }
        );
        let buffer = Facts {
            slices: 1,
            levels: 1,
            ..facts(PixelFormat::RGBA8Unorm, TextureType::TextureBuffer)
        };
        assert_eq!(
            problem(check_target("t", c0, &buffer, (0, 1, 0), 0).unwrap_err()),
            RenderPassProblem::AttachmentSubresource(c0)
        );
    }

    /// A 3D attachment renders the planes of its level from its depth plane, one or a layered
    /// pass's (a probe: `t3d_plane2`, `t3d_level1_plane1`, `t3d_layered_from1`), and a 1D one its
    /// row, as a 2D target of one row (`t1d`, `t1d_array_slice1`).
    #[test]
    fn volumes_and_lines_are_targets() {
        let c0 = Attachment::Color(0);
        let volume = Facts {
            slices: 1,
            extent: [8, 8, 4],
            ..facts(PixelFormat::RGBA8Unorm, TextureType::Type3D)
        };
        for (level, plane, length, want) in [
            (0, 2, 0, (0, 2, 1, [8, 8])),
            (1, 1, 0, (1, 1, 1, [4, 4])),
            (0, 1, 2, (0, 1, 2, [8, 8])),
            (0, 0, 4, (0, 0, 4, [8, 8])),
        ] {
            let sub = check_target("t", c0, &volume, (level, 0, plane), length).unwrap();
            assert_eq!((sub.level, sub.layer, sub.layers, sub.extent), want);
        }
        let line = Facts {
            extent: [16, 1, 1],
            ..facts(PixelFormat::RGBA8Unorm, TextureType::Type1DArray)
        };
        let sub = check_target("t", c0, &line, (0, 1, 0), 0).unwrap();
        assert_eq!((sub.layer, sub.layers, sub.extent), (1, 1, [16, 1]));
    }

    /// A layered pass renders its length's layers of every attachment from its slice, and Metal's
    /// validation stops an attachment without them (a Metal probe on an M4: "slice +
    /// renderTargetArrayLength is 3, but the texture only has 2 slices", a 2D texture's one slice
    /// among them); a length of 0 or 1 renders the slice alone.
    #[test]
    fn a_layered_target_holds_the_pass_layers() {
        let array = Facts {
            slices: 4,
            ..facts(PixelFormat::R32Uint, TextureType::Type2DArray)
        };
        let c0 = Attachment::Color(0);
        for (slice, length, layers) in [(0, 0, 1), (3, 1, 1), (0, 2, 2), (1, 3, 3), (0, 4, 4)] {
            let sub = check_target("t", c0, &array, (0, slice, 0), length).unwrap();
            assert_eq!(
                (sub.layer, sub.layers),
                (slice as u32, layers),
                "{slice} {length}"
            );
        }
        for (slice, length) in [(3, 2), (0, 5), (1, u64::MAX)] {
            assert_eq!(
                problem(check_target("t", c0, &array, (0, slice, 0), length).unwrap_err()),
                RenderPassProblem::ArrayLengthPastSlices {
                    attachment: c0,
                    end: slice.saturating_add(length),
                    slices: 4,
                }
            );
        }
        let plain = Facts {
            slices: 1,
            ..facts(PixelFormat::Depth32Float, TextureType::Type2D)
        };
        assert_eq!(
            problem(check_target("t", Attachment::Depth, &plain, (0, 0, 0), 2).unwrap_err()),
            RenderPassProblem::ArrayLengthPastSlices {
                attachment: Attachment::Depth,
                end: 2,
                slices: 1,
            }
        );
        // A cube's six faces are its slices (probe 12: a pass of 2 from face 4).
        let cube = Facts {
            slices: 6,
            ..facts(PixelFormat::R32Uint, TextureType::Cube)
        };
        let sub = check_target("t", c0, &cube, (0, 4, 0), 2).unwrap();
        assert_eq!((sub.layer, sub.layers), (4, 2));
    }

    /// The render area is the smallest attachment, under nonzero target sizes; without
    /// attachments the target sizes are needed (probe 1), at most 16384 (probe 3).
    #[test]
    fn the_render_area_is_the_smallest_target() {
        assert_eq!(
            area("t", &[([8, 8], 1), ([4, 4], 1)], [0, 0], 0).unwrap(),
            ([4, 4], 1)
        );
        assert_eq!(area("t", &[([8, 8], 1)], [4, 4], 0).unwrap(), ([4, 4], 1));
        assert_eq!(area("t", &[], [4, 4], 1).unwrap(), ([4, 4], 1));
        assert_eq!(area("t", &[], [4, 4], 4).unwrap(), ([4, 4], 4));
        for target_size in [[0, 0], [4, 0]] {
            match area("t", &[], target_size, 1).unwrap_err() {
                Error::Nil(nil) => assert_eq!(
                    nil.problem,
                    Problem::RenderPass(RenderPassProblem::NoTargets)
                ),
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(
            problem(area("t", &[([8, 8], 1), ([8, 8], 4)], [0, 0], 0).unwrap_err()),
            RenderPassProblem::SampleCounts
        );
        // Past 16384 without attachments Metal makes the encoder and fails the command buffer
        // from the pass on (probe 3).
        assert_eq!(
            area("t", &[], [16384, 16384], 1).unwrap(),
            ([16384, 16384], 1)
        );
        assert!(!past_target_size(&[], [16384, 16384]));
        for size in [[16385, 16], [16, 16385]] {
            let (extent, _) = area("t", &[], size, 1).unwrap();
            assert_eq!(u64::from(extent[0]), size[0]);
            assert_eq!(u64::from(extent[1]), size[1]);
            assert!(past_target_size(&[], extent));
            assert!(!past_target_size(&[([16, 16], 1)], extent));
        }
    }
    /// A pass without attachments at each default sample count measured answers as plain Metal
    /// did on an Apple M4 under macOS 27.0.1 (every count from 0 to 256, and the larger ones
    /// below), and on the M3 Max under macOS 26.7 for the counts it ran (the probe's probe, 0
    /// included): 1 to 7 run, 0, 8 to 15 and 128 to 255 end the process, and 16 to 127 fail the
    /// command buffer, by the count's low 8 bits. A stopped count is `Invalid` making the encoder;
    /// a pass with attachments takes its attachments' samples, whatever its default.
    #[test]
    fn a_pass_without_attachments_answers_metal_s_default_sample_counts() {
        use DefaultSamples::{FailsCommandBuffer, Runs, Stops};
        let metal = |count: u64| match count & 0xff {
            1..=7 => Runs,
            16..=127 => FailsCommandBuffer,
            _ => Stops,
        };
        for count in 0..=256 {
            assert_eq!(pass_default_samples(count), metal(count), "{count}");
        }
        let measured = [
            (257, Runs),
            (300, FailsCommandBuffer),
            (511, Stops),
            (512, Stops),
            (513, Runs),
            (1024, Stops),
            (1 << 16, Stops),
            (1 << 31, Stops),
            (1 << 32, Stops),
            (1 << 63, Stops),
            (u64::MAX, Stops),
        ];
        for (count, answer) in measured {
            assert_eq!(pass_default_samples(count), answer, "{count}");
        }
        for count in [0, 8, 15, 128, 255, 512, u64::MAX] {
            assert_eq!(
                problem(area("t", &[], [4, 4], count).unwrap_err()),
                RenderPassProblem::DefaultRasterSampleCount { count },
                "{count}"
            );
        }
        assert_eq!(area("t", &[], [4, 4], 3).unwrap(), ([4, 4], 3));
        assert_eq!(area("t", &[], [4, 4], 16).unwrap(), ([4, 4], 16));
        assert_eq!(area("t", &[([8, 8], 4)], [0, 0], 8).unwrap(), ([8, 8], 4));
    }

    /// A framebuffer past the device's limits is refused naming the first limit it passes.
    #[test]
    fn a_framebuffer_past_the_limits_names_the_limit() {
        let max = [16384, 8192, 2048];
        assert_eq!(framebuffer_fits([16384, 8192], 2048, max), Ok(()));
        let limit = |limit, required, available| {
            Err(Unsupported::needs(
                Feature::RenderTargetSize,
                Capability::Limit {
                    limit,
                    required,
                    available,
                },
            ))
        };
        assert_eq!(
            framebuffer_fits([16385, 8192], 1, max),
            limit(VkLimit::MaxFramebufferWidth, 16385, 16384)
        );
        assert_eq!(
            framebuffer_fits([16, 8193], 1, max),
            limit(VkLimit::MaxFramebufferHeight, 8193, 8192)
        );
        assert_eq!(
            framebuffer_fits([16, 16], 2049, max),
            limit(VkLimit::MaxFramebufferLayers, 2049, 2048)
        );
        assert_eq!(
            framebuffer_fits([16385, 8193], 2049, max),
            limit(VkLimit::MaxFramebufferWidth, 16385, 16384)
        );
    }
}
