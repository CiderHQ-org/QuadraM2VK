//! Multisample resolves: `MTLStoreActionMultisampleResolve` and
//! `MTLStoreActionStoreAndMultisampleResolve` (step 6's fourth part).
//!
//! **Metal**, measured on an M4 with macOS 27.0.1 (with and without API validation):
//!
//! - A color resolve averages each pixel's stored samples: unorm values round to nearest (the
//!   samples 0, 85, 170 and 255 give 128), sRGB ones are averaged in linear space and encoded
//!   again, float and half ones exactly. Integer formats are no resolve target ("cannot be a
//!   MSAA resolve target"). A pixel no draw covered resolves to the clear color.
//! - The resolve writes only the resolve texture's `resolveLevel` and `resolveSlice`, whose
//!   extent is the attachment's (a level of a larger texture serves). The resolve texture needs
//!   no `RenderTarget` usage.
//! - **A level of another extent** (probes, an M4, four fills alike; Metal's validation stops it,
//!   "size mismatch"): the overlap with the attachment resolves. A depth or stencil level keeps its
//!   texels past it; a color level takes the attachment's clear color past it over its whole level
//!   and slice after a `Clear` load, and what memory held after a `Load` or `DontCare` one
//!   (refused, `ResolveSize`). The runtime resolves into a scratch image of the attachment's
//!   extent, clears a color level reaching past it, and copies the overlap.
//! - **A multisampled color resolve texture of another sample count** than the attachment's,
//!   of its extent, is left untouched (probes 13, 13b and 14: 4 into 2, 2 into 4, drawn and
//!   not): the attachment resolves nowhere. One of the attachment's count takes a copy of its
//!   samples, which the runtime does not build (refused, `ResolveSamples`), and a depth or
//!   stencil one a layout pattern (refused).
//! - **A resolve texture of another pixel format** (probes, an M4 with macOS 27.0.1; Metal's
//!   validation stops it, "does not match Resolve PixelFormat"): a multisampled attachment
//!   resolving into its format's sRGB twin keeps its own encoding, a texture or a view alike
//!   (`RGBA8Unorm` into `RGBA8Unorm_sRGB` stores the unorm bytes, the other way the sRGB ones; the
//!   probe's `v_resolve_format_mismatch` on an M3 Max too), and the runtime resolves it in the
//!   attachment's format, the twin among its image's view formats. Across `RGBA8Unorm` and
//!   `BGRA8Unorm` and their twins the rule is one: the attachment's values in its own encoding, in
//!   the destination's channel order (a texture's or a view's; the four twin pairs measured and six
//!   of the eight across the orders); the other channel order is not built
//!   (`NotImplemented(Render)`). Other pairs follow no one rule: `RGBA8Unorm` into `RGBA8Snorm`
//!   writes its bytes (0x80 as 0x81), into `R32Uint`, `R32Float`, `RGBA8Uint` or `RGBA16Float`
//!   nothing, `RGBA16Float` into `RGBA8Unorm` converts, `R32Float` into `R32Uint` writes its bits,
//!   into `RGBA8Unorm` its red with zeros, and `RGBA16Float` into `RG32Float` a pattern of neither;
//!   each stays refused (`ResolveFormat`). An integer format's resolve writes values no sample
//!   holds, which change from run to run (`RGBA8Uint`, `R32Uint`): refused. Measured over every
//!   integer color format (an M4 with macOS 27.0.1, without API validation, which
//!   stops it; the 19 formats, 4 samples, every subset of covered samples over a clear, two runs):
//!   a pixel whose samples differ resolves to a value no sample holds, not sample 0's, mostly
//!   another one each run, and a 32-bit channel takes such a value even where the samples
//!   agree; only an 8- or 16-bit channel (`RGB10A2Uint`'s too) whose samples agree resolves to
//!   their value. No rule that holds for every pixel: refused.
//! - A packed 16-bit attachment the device renders through the view of its word
//!   (`metal::format::WordAttachment`) holds each sample's stored codes, while Metal's resolve
//!   averages its tile's values, finer than the codes: not modelled, refused
//!   `NotImplemented(Format)`.
//! - The depth filters `Sample0`, `Min` and `Max` take sample 0's depth, the smallest and the
//!   largest. The stencil filter `Sample0` takes sample 0's stencil, and `DepthResolvedSample`
//!   the stencil of the sample whose depth the depth filter took: of equal depths, the last
//!   sample after `Min` and the first after `Max` (a probe, an M3 Max and an M4 alike).
//! - `StoreAndMultisampleResolve` keeps the samples: a later pass loads them. One texture's
//!   depth or stencil stored and resolved beside the other aspect stored keeps both aspects'
//!   samples and resolves that aspect alone, the resolve texture's other aspect untouched
//!   (a probe, hazard-tracked textures, an M3 Max and an M4 alike).
//! - **A depth or stencil `StoreAndMultisampleResolve` of an untracked texture** (a heap's by
//!   default, or one made `HazardTrackingMode::Untracked`; a probe, an M3 Max with macOS 26.7 and
//!   an M4 with 27.0.1): Metal's resolve takes the samples the texture held before the pass in most
//!   runs (every probe-16 run on the M4) and the pass's in the others; a tracked texture's, a color
//!   attachment's and `MultisampleResolve` resolve the pass's. Memory- and timing-dependent, inside
//!   one pass where no fence or event reaches: the runtime resolves the pass's samples, as Metal
//!   does for a tracked texture.
//! - **A layered pass** (`renderTargetArrayLength` 2 or more; a Metal probe on an M3 Max with macOS
//!   26.7 and an M4 with 27.0.1, alike, with and without API validation): layer `i` of the pass,
//!   from the attachment's slice, resolves into slice `resolveSlice + i`, and a layer past the
//!   resolve texture's slices is not resolved (a 2D resolve texture takes layer 0 alone), which API
//!   validation accepts. A layer no draw reached resolves its clear.
//!
//! **Refused** (`Invalid`, each measured): a depth or stencil attachment's resolving store action
//! without a resolve texture; a memoryless resolve texture (a Metal probe on an M4); a
//! single-sampled depth or stencil attachment, or a color one's late target, a multisampled resolve
//! texture (but a color one of another sample count, above), another pixel format (but a
//! multisampled color attachment's sRGB twin, above), an integer color format, a level or slice
//! past the resolve texture, and a color level reaching past the attachment after a load other than
//! `Clear`; `DepthResolvedSample` without a depth attachment, or with a separate depth texture
//! stored beside it; and depth and stencil attachments of one texture that mix storing and
//! resolving as plain Metal leaves undetermined ([`mixed`]). A color attachment's resolve without a
//! texture, or a texture under a store action that does not resolve, the descriptor's or the
//! encoder's, stops Metal's validation, and plain Metal resolves nothing and leaves that texture
//! untouched (a Metal probe on an M4), as the runtime does; so does a depth or stencil attachment's
//! texture under such an action, or under `Unknown` still at `endEncoding`, and the attachment is
//! stored by the action (`DontCare` for `Unknown`; an M4 with macOS 27.0.1,
//! color, depth and stencil alike). So does the resolve of a single-sampled color attachment, and
//! plain Metal resolves its one sample (probe 6m's `ref_single_sampled_source`): the runtime copies
//! the attachment's texels into the resolve texture after the pass (`color_copy`), of the
//! attachment's extent.
//!
//! **Plain Metal's answers to what its validation stops** (probes on an M4, four fills alike): an
//! index the resolve texture's type does not have is ignored (`render_pass::used_indices`: a 2D
//! texture's `resolveSlice`, a 3D texture's slice, a depth plane of any other);
//! `DepthResolvedSample` with a depth attachment that does not resolve takes the sample its filter
//! picks, drawn after the pass as for a resolved depth; and one texture's depth resolved alone
//! (`MultisampleResolve`) with its stencil stored resolves the depth and keeps the stencil's
//! samples, the resolve texture's stencil untouched (a probe alike on hazard-tracked textures). One
//! texture's depth stored and resolved beside its stencil resolved alone, which plain Metal answers
//! with a layout pattern of the resolved depth in the depth's samples and its resolve texture
//! (probe 17, an M4), the runtime resolves both and keeps both aspects' samples, as it ran them
//! previously (the stencil's stored too, so MoltenVK runs Metal's valid form): a recorded
//! near-equivalent, announced once per device (`Degradation::MixedDepthStencilResolve`,
//! [`layout_patterned`]).
//!
//! **The mechanism.** The pass resolves at the end of its subpass, in render pass 2 terms:
//!
//! - **Color** through the subpass's resolve attachments. Vulkan averages a resolve of a
//!   normalized or floating-point format (`VK_RESOLVE_MODE_AVERAGE_BIT`) and leaves the
//!   average's rounding and sRGB handling to the device, so a device may differ from Metal by
//!   one unit at a tie (the exact path is a resolve shader).
//! - **Depth and stencil** through `VkSubpassDescriptionDepthStencilResolve` (core in Vulkan
//!   1.2, which every runtime device has): `Sample0`, `Min` and `Max` are `SAMPLE_ZERO`, `MIN`
//!   and `MAX`, and `DepthResolvedSample` after `Sample0` is `SAMPLE_ZERO`. An aspect the pass
//!   does not resolve is `NONE`, and the resolve texture keeps its texels there.
//! - **One aspect at a time.** A device that resolves a combined format's aspects only in one
//!   mode (`independentResolve`), or both or neither (`independentResolveNone`), resolves each
//!   aspect apart: the first in the draws' subpass, the second after the pass in a render pass of
//!   its own (`render_target::later_resolve`), each into a scratch image whose resolved aspect
//!   alone is copied into the resolve texture after the pass, through a staging buffer
//!   (`ResolveCopies`). A resolve's other aspect takes `NONE` where the device allows, else the
//!   same mode, which writes only the scratch image.
//! - **`DepthResolvedSample` after `Min` or `Max`**: the depth resolves in the subpass by `MIN` or
//!   `MAX`, its stencil `NONE` there, and the stencil is drawn after the pass into the same resolve
//!   attachment by `depth_stencil_resolve.rs`, which reads the attachment's samples (both aspects,
//!   which the last instance stores) through sampled views and picks Metal's sample by depth. The
//!   attachment's image must be sampled, and the pass of one layer.
//! - **`Min` or `Max` on a device without the mode** (lavapipe has neither): the depth is drawn
//!   after the pass the same way, as `FragDepth`, where that stores the picked value unchanged
//!   (`depth_copy::copies_exactly`: a 32-bit float depth on a driver measured to,
//!   `DriverFact::FragmentDepthStoresUnchanged`, or a 16-bit one, whose code is kept where
//!   `DriverFact::FragmentDepthKeepsDepth16Codes` holds); its stencil then too after
//!   `DepthResolvedSample`. Where the subpass resolves no aspect, the draws write the resolve
//!   texture (or a scratch image copied there) themselves. Elsewhere the missing mode stays
//!   refused, `NotImplemented(DepthStencilResolve)`.
//! - A resolve texture whose image cannot be an attachment (Metal's texture has no
//!   `RenderTarget` usage) is resolved into a scratch image of the pass, whose render area is
//!   copied into the texture after the pass.
//! - **Layers.** A layered pass resolves every layer in its subpass, into a view of the resolve
//!   texture's slices from `resolveSlice` where they hold the pass's layers, else into a scratch
//!   image of the pass's layers whose layers that fit in the texture's slices are copied into
//!   them after the pass. A depth or stencil attachment resolves its first layer alone, into slice
//!   `resolveSlice`, as Metal does (a probe): through a scratch image of the pass's layers whose
//!   first is copied, and its drawn aspects from the first layer's samples.
//! - **A render pass that resolves depth and stencil from the first layer and level alone**
//!   (lavapipe, `RuntimeDriverDefect::DepthStencilResolveFirstSubresource`): its render pass
//!   resolves a depth/stencil attachment from its image's first layer and level into the resolve
//!   attachment image's first, whatever layer and level the views begin at. There a resolve slice
//!   or level other than the first takes the scratch image, copied there after the pass; and an
//!   attachment that begins at another slice is copied after the pass's instances, the pass's
//!   layers of it, into a scratch image of its format and samples from the first
//!   ([`ResolveSource`]), which an instance of no draws, and each resolve after the pass, resolves
//!   from in its place. A memoryless attachment there, which takes no copy, is not built
//!   (`NotImplemented(DepthStencilResolve)`).
//! - The multisampled attachment of `MultisampleResolve` is stored `DONT_CARE`: Metal leaves its
//!   texels undefined (the recorded near-equivalent of a `DontCare` store).
//!
//! **Not built** (`NotImplemented(DepthStencilResolve)`): a filter the device's resolve modes
//! lack (`supportedDepthResolveModes`, `supportedStencilResolveModes`, or the other aspect's
//! modes where a subpass resolving one aspect must name the same mode for both), as the compile
//! device's description states them (`VkProperty::{DepthResolveMode, StencilResolveMode,
//! IndependentResolveNone, IndependentResolve}`), for which a resolve draw is the exact path; and
//! `DepthResolvedSample` after `Min` or `Max` of an attachment whose image is not sampled, or
//! whose stencil the draws cannot write; and a split resolve (below) with an aspect drawn after
//! the pass.
//!
//! **Split resolves**. Where the aspects resolve into different textures (separate depth and
//! stencil textures, which render through a combined scratch image, `depth_stencil_scratch.rs`, or
//! one combined attachment whose depth and stencil resolve textures differ), each resolve texture
//! is checked against the attachment that names it, and each aspect resolves in a subpass of its
//! own (the first in the draws' subpass, the second in a later pass) into a scratch image of the
//! attachment's format, whose aspect alone is copied into its texture after the pass through a
//! staging buffer, as one aspect at a time does. An aspect the pass does not resolve keeps its
//! resolve texture's texels. Announced as `one-aspect-resolve`.

use super::blit::{reads, writes, TRANSFER};
use super::buffer::Buffer;
use super::color_resolve;
use super::command_buffer::CommandBuffer;
use super::degradation::Degradation;
use super::depth_stencil_scratch::Separate;
use super::encoder::EncoderCore;
use super::error::{internal, invalid, Error, Problem, RuntimeGap};
use super::image_plan::{block_family, sample_count, srgb_twin, ImagePlan};
use super::linear_attachment;
use super::render_pass::{Attachment, Facts, PassLayout, RenderPassProblem, Target};
use super::render_values::{
    LoadAction, MultisampleDepthResolveFilter, MultisampleStencilResolveFilter, StoreAction,
};
use super::resource::ResourceKey;
use super::texture::{copy_units, image_aspects, Texture};
use super::texture_descriptor::TextureDescriptor;
use super::tracker::{Access, Barrier, Use};
use super::values::{ResourceOptions, StorageMode};
use super::Device;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{ColorResolveFormat, DepthPick};
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::error::{Feature, RuntimeDriverDefect, Unsupported};
use metal2vulkan::metal::format::{aspects, sampled_type, ExtendedRange, FormatMapping};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::pipeline::SampledType;
use metal2vulkan::vk::{ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags};

/// A Vulkan resolve mode of a depth or stencil aspect: `NONE` where the aspect does not resolve.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum ResolveMode {
    None,
    SampleZero,
    Min,
    Max,
    /// Drawn after the pass by the pick (`depth_stencil_resolve.rs`), so no subpass resolves it:
    /// as a stencil, the stencil of the sample the pick takes (`DepthResolvedSample` after `Min`
    /// or `Max`), which no Vulkan mode states; as a depth, the pick's depth, where the device has
    /// no `MIN` or `MAX` depth mode and its fragment depth stores the value unchanged.
    Picked(DepthPick),
}

impl ResolveMode {
    /// The mode's Vulkan bit: `NONE` for `ResolveMode::Picked`, which no subpass makes.
    pub(super) fn vk(self) -> vk::ResolveModeFlags {
        match self {
            ResolveMode::None | ResolveMode::Picked(_) => vk::ResolveModeFlags::NONE,
            ResolveMode::SampleZero => vk::ResolveModeFlags::SAMPLE_ZERO,
            ResolveMode::Min => vk::ResolveModeFlags::MIN,
            ResolveMode::Max => vk::ResolveModeFlags::MAX,
        }
    }

    /// The mode a subpass resolves the aspect in: `NONE` for one drawn after the pass.
    fn in_subpass(self) -> ResolveMode {
        match self {
            ResolveMode::Picked(_) => ResolveMode::None,
            mode => mode,
        }
    }
}

/// The depth and stencil resolve modes a device has (`VkPhysicalDeviceDepthStencilResolveProperties`,
/// as the compile device's description states them).
#[derive(Clone, Copy)]
pub(super) struct ResolveModes {
    depth: vk::ResolveModeFlags,
    stencil: vk::ResolveModeFlags,
    /// One aspect may resolve while the other does not.
    independent_none: bool,
    /// The aspects may resolve in different modes.
    independent: bool,
}

impl ResolveModes {
    /// The modes the compile device's description states (its `supportedDepthResolveModes`,
    /// `supportedStencilResolveModes`, `independentResolveNone` and `independentResolve`), so
    /// the runtime resolves by what the compile layer planned for, and a description that
    /// states fewer modes than the physical device has resolves by those alone.
    pub(super) fn of(profile: &DeviceProfile) -> ResolveModes {
        let limits = profile.limits();
        let flags = |modes: metal2vulkan::vk::ResolveModeFlags| {
            vk::ResolveModeFlags::from_raw(modes.as_raw())
        };
        ResolveModes {
            depth: flags(limits.supported_depth_resolve_modes),
            stencil: flags(limits.supported_stencil_resolve_modes),
            independent_none: limits.independent_resolve_none,
            independent: limits.independent_resolve,
        }
    }

    /// Whether the device resolves a format with `both` aspects in `depth` and `stencil` mode
    /// (`NONE` for an aspect not resolved).
    fn allows(&self, depth: ResolveMode, stencil: ResolveMode, both: bool) -> bool {
        let none = ResolveMode::None;
        let supported = (depth == none || self.depth.contains(depth.vk()))
            && (stencil == none || self.stencil.contains(stencil.vk()));
        let paired = !both
            || depth == stencil
            || self.independent
            || (self.independent_none && (depth == none || stencil == none));
        supported && paired
    }
}

/// What an attachment resolves into.
#[derive(Clone, Debug)]
pub(super) struct Resolve {
    /// The pass's resolve attachment: the resolve texture's level and slice, or a scratch
    /// image's.
    pub(super) attachment: Target,
    /// The resolve texture's level and slice, when the pass resolves into a scratch image whose
    /// render area is copied there after the pass.
    pub(super) copy_to: Option<Target>,
    /// The rules of a color resolve the runtime draws after the pass, in place of the subpass's
    /// (`color_resolve.rs`): `None` where the device resolves it, and for depth and stencil.
    pub(super) shader: Option<ColorResolveFormat>,
    /// The clear color (its bits) the resolve texture's level and layers take before the
    /// scratch image's copy, where they reach past the attachment that clears them: plain Metal
    /// writes the attachment's clear color there (the module docs). `None` elsewhere.
    pub(super) clear: Option<[u32; 4]>,
}

/// The depth/stencil attachment's resolve.
#[derive(Clone, Debug)]
pub(super) struct DepthStencilResolve {
    /// Each aspect's mode as Metal's filters state it, `None` for an aspect not resolved, and
    /// `ResolveMode::Picked` for an aspect drawn after the pass (the module docs).
    pub(super) depth_mode: ResolveMode,
    pub(super) stencil_mode: ResolveMode,
    /// The resolves that make it, the first in the draws' subpass and the other in a pass of its
    /// own after the pass (`render_target::later_resolve`): one, or one per aspect on a device
    /// that cannot resolve the pair's modes in one subpass (the module docs); none where every
    /// resolved aspect is drawn.
    pub(super) steps: Vec<ResolveStep>,
    /// The resolve attachment the draws write where no step resolves into one (every resolved
    /// aspect drawn): the resolve texture's level and slice, or a scratch image copied there.
    pub(super) drawn_into: Option<Resolve>,
    /// Its aspects resolve into different textures (separate ones, or two for one combined
    /// texture): each step resolves one aspect into a scratch image copied into its own.
    pub(super) split: bool,
}

impl DepthStencilResolve {
    /// The pick of the aspects drawn after the pass, and which: `None` where none is.
    pub(super) fn picked(&self) -> Option<(DepthPick, ImageAspectFlags)> {
        picked(self.depth_mode, self.stencil_mode)
    }

    /// The resolve attachment the drawn aspects are written into: the first step's, or the
    /// draws' own.
    pub(super) fn drawn_target(&self) -> Option<&Resolve> {
        self.steps
            .first()
            .map(|step| &step.resolve)
            .or(self.drawn_into.as_ref())
    }

    /// Each resolve attachment it writes, with the aspects of the resolve texture written there
    /// (those a scratch image's copy takes): its steps', then the draws' own.
    pub(super) fn written(&self) -> impl Iterator<Item = (&Resolve, ImageAspectFlags)> {
        let drawn = self
            .picked()
            .map_or(ImageAspectFlags::empty(), |(_, aspects)| aspects);
        let steps = self.steps.iter().map(|step| (&step.resolve, step.aspects));
        steps.chain(self.drawn_into.iter().map(move |resolve| (resolve, drawn)))
    }
}

/// One depth/stencil resolve of a subpass: the draws' subpass's, or a later pass's.
#[derive(Clone, Debug)]
pub(super) struct ResolveStep {
    pub(super) resolve: Resolve,
    /// The subpass's modes, `None` for an aspect it does not resolve.
    pub(super) depth: ResolveMode,
    pub(super) stencil: ResolveMode,
    /// The aspects of the resolve texture it writes: those a scratch image's copy takes.
    pub(super) aspects: ImageAspectFlags,
    /// It resolves one aspect into a scratch image, the device resolving the pair's modes in no
    /// one subpass (the module docs).
    pub(super) one_aspect: bool,
}

/// One attachment's resolve as its descriptor states it: the resolve texture with its level,
/// slice and depth plane, and the store action.
pub(super) struct Wanted<'a> {
    pub(super) texture: Option<&'a Texture>,
    pub(super) level: u64,
    pub(super) slice: u64,
    pub(super) depth_plane: u64,
    pub(super) store: StoreAction,
}

impl<'a> Wanted<'a> {
    /// The resolve of an attachment with a texture (`attached`) whose descriptor names
    /// `texture` at `(level, slice, depth_plane)` under `store`; `None` without one.
    pub(super) fn of(
        attached: bool,
        texture: &'a Option<Texture>,
        (level, slice, depth_plane): (u64, u64, u64),
        store: StoreAction,
    ) -> Option<Wanted<'a>> {
        attached.then_some(Wanted {
            texture: texture.as_ref(),
            level,
            slice,
            depth_plane,
            store,
        })
    }
}

/// The refusal of a resolve from or into a texture of `pf` that the runtime does not model: an
/// extended-range format's, whose resolve in Metal matches no average of its samples' values
/// measured. The format's Resolve capability answers it too (`families::formats`).
pub(crate) fn unmodelled_resolve(pf: PixelFormat) -> Option<Unsupported> {
    ExtendedRange::of(pf).map(|_| Unsupported::not_implemented(Feature::Format))
}

fn problem(call: &'static str, p: RenderPassProblem) -> Error {
    invalid(call, Problem::RenderPass(p))
}

/// Whether `which` resolves: its descriptor's resolve texture and store action, checked (a depth or
/// stencil attachment's resolving action needs a texture). An attachment does not resolve without a
/// texture and an action that resolves or is `Unknown`: plain Metal leaves a resolve texture under
/// `Store` or `DontCare` untouched and stores the attachment by the action, and a color
/// attachment's resolving action without a texture resolves nothing (Metal probe cases
/// `ref_resolve_without_texture` and `ref_texture_with_store` on an M4; depth and stencil alike in
/// our probe; its validation stops each).
fn resolves(call: &'static str, which: Attachment, wanted: &Wanted<'_>) -> Result<bool, Error> {
    let resolving = wanted.store.resolves() || wanted.store == StoreAction::Unknown;
    if let Attachment::Color(_) = which {
        return Ok(wanted.texture.is_some() && resolving);
    }
    match wanted.texture {
        None if wanted.store.resolves() => Err(problem(
            call,
            RenderPassProblem::ResolveWithoutTexture(which),
        )),
        None => Ok(false),
        Some(_) => Ok(resolving),
    }
}

/// Whether one texture's depth and stencil store actions mix storing and resolving as Metal
/// leaves undetermined: the stencil resolved alone (`MultisampleResolve`) beside a stored depth,
/// whose samples plain Metal overwrites with a layout pattern of the resolved depth (probes, an M4;
/// its validation stops it). An `Unknown` action decides at its late store action. The other forms
/// are Metal's answers (probe 17 on hazard-tracked textures, an M3 Max and an M4): an aspect stored
/// and resolved (`StoreAndMultisampleResolve`) beside a stored one keeps both aspects' samples and
/// resolves that aspect alone, the resolve texture's other aspect untouched, and the depth resolved
/// alone beside a stored stencil resolves the depth by its filter and keeps the stencil's samples.
pub(super) fn mixed(depth: StoreAction, stencil: StoreAction) -> bool {
    depth == StoreAction::Store && stencil == StoreAction::MultisampleResolve
}

/// Whether one texture's depth and stencil store actions are the depth stored and resolved
/// beside the stencil resolved alone, which Metal's validation stops and plain Metal answers with
/// a layout pattern of the resolved depth in the depth's samples and its resolve texture (a probe,
/// an M4). The runtime, which ran it previously, resolves both and keeps both aspects' samples
/// (`render_target::patterned_aspects`: so MoltenVK runs Metal's valid `StoreAndMultisampleResolve`
/// of both), and announces `Degradation::MixedDepthStencilResolve` (a recorded near-equivalent).
pub(super) fn layout_patterned(depth: StoreAction, stencil: StoreAction) -> bool {
    depth == StoreAction::StoreAndMultisampleResolve && stencil == StoreAction::MultisampleResolve
}

/// The resolve of color attachment `which`, `source`, as `wanted` states it; `None` when it does
/// not resolve. Refused as the module docs say.
/// The attachment's `load` action and `clear` color decide a resolve texture level reaching past
/// it.
pub(super) fn color(
    call: &'static str,
    maker: &CommandBuffer,
    which: Attachment,
    source: &Target,
    wanted: &Wanted<'_>,
    (load, clear): (LoadAction, vk::ClearColorValue),
) -> Result<Option<Resolve>, Error> {
    let device = maker.device();
    if !resolves(call, which, wanted)? {
        return Ok(None);
    }
    let texture = wanted
        .texture
        .ok_or_else(|| internal("a resolve without its texture"))?;
    if matches!(
        sampled_type(source.texture.pixel_format()),
        Some(SampledType::Uint | SampledType::Sint | SampledType::Uint64)
    ) {
        return Err(problem(call, RenderPassProblem::ResolveFormat(which)));
    }
    if let Some(unsupported) = [source.texture.pixel_format(), texture.pixel_format()]
        .into_iter()
        .find_map(unmodelled_resolve)
    {
        return Err(Error::Unsupported(unsupported));
    }
    // An attachment rendered through the view of its word (`metal::format::WordAttachment`) holds
    // each sample's stored codes, while Metal resolves its tile's values: not modelled.
    let word = metal2vulkan::metal::format::WordAttachment::of(
        source.texture.pixel_format(),
        device.compile_device().profile(),
    );
    if word.is_some_and(|w| w.word_format() == source.format) {
        return Err(Error::Unsupported(
            metal2vulkan::error::Unsupported::not_implemented(metal2vulkan::error::Feature::Format),
        ));
    }
    single_sampled(call, which, source)?;
    let Some(destination) = destination(call, device, which, source, texture, wanted)? else {
        return Ok(None);
    };
    // A level reaching past the attachment takes its clear color there; after a `Load` or
    // `DontCare` load plain Metal writes what memory held there (the module docs): refused.
    let [width, height] = destination.target.extent;
    let past = width > source.extent[0] || height > source.extent[1];
    if past && load != LoadAction::Clear {
        return Err(problem(call, RenderPassProblem::ResolveSize(which)));
    }
    let mut resolve =
        destination.attach(maker, source, ImageUsageFlags::COLOR_ATTACHMENT, false)?;
    if past {
        // SAFETY: every member of the union is four 32-bit values; any bits are a valid `u32`.
        resolve.clear = Some(unsafe { clear.uint32 });
    }
    resolve.shader = color_resolve::of_attachment(device, source)?;
    Ok(Some(resolve))
}

/// The resolve of the depth/stencil attachment `source` (one texture for both aspects, one of
/// them alone, or the combined scratch image `separate` textures render through) with the depth
/// and stencil attachments' `depth` and `stencil` and filters; `None` when neither resolves.
/// Refused as the module docs say.
pub(super) fn depth_stencil(
    call: &'static str,
    maker: &CommandBuffer,
    source: &Target,
    separate: Option<&Separate>,
    depth: Option<&Wanted<'_>>,
    stencil: Option<&Wanted<'_>>,
    depth_filter: MultisampleDepthResolveFilter,
    stencil_filter: MultisampleStencilResolveFilter,
) -> Result<Option<DepthStencilResolve>, Error> {
    let device = maker.device();
    let depth_resolves = match depth {
        Some(d) => resolves(call, Attachment::Depth, d)?,
        None => false,
    };
    let stencil_resolves = match stencil {
        Some(s) => resolves(call, Attachment::Stencil, s)?,
        None => false,
    };
    // One texture whose depth and stencil mix storing and resolving as Metal leaves undetermined
    // ([`mixed`]).
    if let (Some(d), Some(s)) = (depth, stencil) {
        if separate.is_none() && mixed(d.store, s.store) {
            return Err(problem(call, RenderPassProblem::DepthStencilResolveMix));
        }
    }
    // `DepthResolvedSample` takes the sample the depth attachment's filter picks whether or not
    // the depth resolves (probes, an M4): refused without a depth attachment, and where a separate
    // depth texture that does not resolve is stored (plain Metal stores a layout pattern of its
    // samples, probe 14) or may be (`Unknown`).
    let unpicked = match depth {
        None => true,
        Some(d) => separate.is_some() && !depth_resolves && d.store != StoreAction::DontCare,
    };
    if stencil_resolves
        && stencil_filter == MultisampleStencilResolveFilter::DepthResolvedSample
        && unpicked
    {
        return Err(problem(
            call,
            RenderPassProblem::DepthResolvedSampleWithoutDepthResolve,
        ));
    }
    if !depth_resolves && !stencil_resolves {
        return Ok(None);
    }
    // Each resolved aspect's destination, checked as Metal checks it against the attachment
    // that names it: a separate texture's own (its format, samples and extent), else `source`.
    let checked = |which: Attachment| match (separate, which) {
        (Some(separate), Attachment::Depth) => separate
            .depth
            .as_ref()
            .ok_or_else(|| internal("a separate depth resolve without its texture")),
        (Some(separate), _) => Ok(&separate.stencil),
        (None, _) => Ok(source),
    };
    let aspect_destination =
        |which: Attachment, resolves: bool, wanted: Option<&Wanted<'_>>| match (resolves, wanted) {
            (true, Some(wanted)) => {
                let texture = wanted
                    .texture
                    .ok_or_else(|| internal("a resolve without its texture"))?;
                let source = checked(which)?;
                single_sampled(call, which, source)?;
                destination(call, device, which, source, texture, wanted)?
                    .ok_or_else(|| internal("a depth or stencil resolve left untouched"))
                    .map(Some)
            }
            _ => Ok(None),
        };
    let depth_destination = aspect_destination(Attachment::Depth, depth_resolves, depth)?;
    let stencil_destination = aspect_destination(Attachment::Stencil, stencil_resolves, stencil)?;
    // Both aspects resolve into one texture's level and slice, or each into its own: separate
    // textures, or one combined texture's aspects resolving into two (the split resolve).
    let same = match (&depth_destination, &stencil_destination) {
        (Some(d), Some(s)) => {
            d.target.texture.same_storage(&s.target.texture)
                && (d.target.level, d.target.layer) == (s.target.level, s.target.layer)
        }
        _ => true,
    };
    let split = separate.is_some() || !same;
    let destination = depth_destination
        .clone()
        .or_else(|| stencil_destination.clone())
        .ok_or_else(|| internal("a depth/stencil resolve without its attachment"))?;
    let modes = ResolveModes::of(device.compile_device().profile());
    // A `Min` or `Max` depth the device has no mode for is drawn after the pass where its
    // fragment depth stores the picked value unchanged (`depth_copy::copies_exactly`).
    let exact_depth = super::depth_copy::copies_exactly(device, source.format)
        && super::depth_copy::copies_exactly(device, destination.target.format);
    let pick_or = |mode: ResolveMode, pick: DepthPick| match exact_depth
        && !modes.depth.contains(mode.vk())
    {
        true => ResolveMode::Picked(pick),
        false => mode,
    };
    let depth_mode = if depth_resolves {
        match depth_filter {
            MultisampleDepthResolveFilter::Sample0 => ResolveMode::SampleZero,
            MultisampleDepthResolveFilter::Min => pick_or(ResolveMode::Min, DepthPick::Min),
            MultisampleDepthResolveFilter::Max => pick_or(ResolveMode::Max, DepthPick::Max),
        }
    } else {
        ResolveMode::None
    };
    let stencil_mode = match (stencil_resolves, stencil_filter) {
        (false, _) => ResolveMode::None,
        (true, MultisampleStencilResolveFilter::Sample0) => ResolveMode::SampleZero,
        // The stencil of the sample the depth filter takes, the depth resolved or not: sample
        // 0's after `Sample0`; after `Min` or `Max` no Vulkan mode states it, so it is drawn
        // after the pass, picking Metal's sample (`depth_stencil_resolve.rs`).
        (true, MultisampleStencilResolveFilter::DepthResolvedSample) => match depth_filter {
            MultisampleDepthResolveFilter::Sample0 => ResolveMode::SampleZero,
            MultisampleDepthResolveFilter::Min => ResolveMode::Picked(DepthPick::Min),
            MultisampleDepthResolveFilter::Max => ResolveMode::Picked(DepthPick::Max),
        },
    };
    let subpass_stencil = stencil_mode.in_subpass();
    // A drawn depth's subpass mode is free, as the draws overwrite it: `NONE`, or `SAMPLE_ZERO`
    // where the device pairs no `NONE` with the stencil's mode (lavapipe), which spares the
    // stencil a step of its own.
    let both =
        image_aspects(source.format).contains(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL);
    let none = ResolveMode::None;
    let subpass_depth = match depth_mode {
        ResolveMode::Picked(_)
            if subpass_stencil != none
                && !modes.allows(none, subpass_stencil, both)
                && modes.allows(ResolveMode::SampleZero, subpass_stencil, both) =>
        {
            ResolveMode::SampleZero
        }
        mode => mode.in_subpass(),
    };
    // Every resolved aspect drawn: no subpass resolves, and the draws write their own resolve
    // attachment.
    let (mut steps, drawn_into) = if split {
        // A split resolve resolves each aspect into a scratch image of `source` in a subpass of
        // its own, and copies that aspect alone into its texture after the pass. An aspect drawn
        // after the pass writes the first step's attachment, which a split resolve does not
        // copy whole: not built.
        if picked(depth_mode, stencil_mode).is_some() {
            return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
        }
        let aspects = [
            (depth_mode, ImageAspectFlags::DEPTH, depth_destination),
            (stencil_mode, ImageAspectFlags::STENCIL, stencil_destination),
        ];
        (split_steps(maker, source, aspects, &modes)?, None)
    } else if subpass_depth == none && subpass_stencil == none {
        let usage = ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        (
            Vec::new(),
            Some(destination.attach(maker, source, usage, false)?),
        )
    } else {
        let steps = steps(
            maker,
            source,
            destination,
            subpass_depth,
            subpass_stencil,
            &modes,
        )?;
        (steps, None)
    };
    // A drawn aspect is written into the first step's resolve attachment, which then holds it
    // too (a scratch image's copy takes it), or into the draws' own.
    if let (Some((_, aspects)), Some(step)) = (picked(depth_mode, stencil_mode), steps.first_mut())
    {
        step.aspects |= aspects;
    }
    let resolve = DepthStencilResolve {
        depth_mode,
        stencil_mode,
        steps,
        drawn_into,
        split,
    };
    if let Some((pick, aspects)) = resolve.picked() {
        let into = resolve
            .drawn_target()
            .ok_or_else(|| internal("a drawn resolve without its attachment"))?;
        super::depth_stencil_resolve::draws_run(device, source, &into.attachment, pick, aspects)?;
    }
    Ok(Some(resolve))
}

/// The pick of the aspects of `depth` and `stencil` mode drawn after the pass, and which: `None`
/// where neither is.
fn picked(depth: ResolveMode, stencil: ResolveMode) -> Option<(DepthPick, ImageAspectFlags)> {
    let mut drawn = None;
    for (mode, aspect) in [
        (depth, ImageAspectFlags::DEPTH),
        (stencil, ImageAspectFlags::STENCIL),
    ] {
        if let ResolveMode::Picked(pick) = mode {
            let aspects = drawn.map_or(ImageAspectFlags::empty(), |(_, a)| a);
            drawn = Some((pick, aspects | aspect));
        }
    }
    drawn
}

/// The subpass resolves of `source` into `destination` in `depth` and `stencil` mode on a device
/// with `modes`: one where the device resolves the pair in one subpass, else one per resolved
/// aspect into a scratch image (the module docs).
fn steps(
    maker: &CommandBuffer,
    source: &Target,
    destination: Destination,
    depth: ResolveMode,
    stencil: ResolveMode,
    modes: &ResolveModes,
) -> Result<Vec<ResolveStep>, Error> {
    let none = ResolveMode::None;
    let named = |mode: ResolveMode, aspect: ImageAspectFlags| {
        if mode == none {
            ImageAspectFlags::empty()
        } else {
            aspect
        }
    };
    let aspects = named(depth, ImageAspectFlags::DEPTH) | named(stencil, ImageAspectFlags::STENCIL);
    // The attachment's Vulkan format (a `Depth32Float` texture may be held combined) decides
    // whether its aspects resolve as a pair.
    let both =
        image_aspects(source.format).contains(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL);
    if modes.allows(depth, stencil, both) {
        let needed = ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        return Ok(vec![ResolveStep {
            resolve: destination.attach(maker, source, needed, true)?,
            depth,
            stencil,
            aspects,
            one_aspect: false,
        }]);
    }
    // One subpass per aspect, each into a scratch image copied into the resolve texture.
    if !both || !destination.usage.contains(ImageUsageFlags::TRANSFER_DST) {
        return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
    }
    let mut steps = Vec::new();
    for (mode, aspect) in [
        (depth, ImageAspectFlags::DEPTH),
        (stencil, ImageAspectFlags::STENCIL),
    ] {
        if mode == none {
            continue;
        }
        let pair = |other| {
            if aspect == ImageAspectFlags::DEPTH {
                (mode, other)
            } else {
                (other, mode)
            }
        };
        let (depth, stencil) = [pair(none), pair(mode)]
            .into_iter()
            .find(|&(d, s)| modes.allows(d, s, true))
            .ok_or(Error::NotImplemented(RuntimeGap::DepthStencilResolve))?;
        steps.push(ResolveStep {
            resolve: Resolve {
                attachment: scratch(maker, source, ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, 1)?,
                copy_to: Some(destination.target.clone()),
                shader: None,
                clear: None,
            },
            depth,
            stencil,
            aspects: aspect,
            one_aspect: true,
        });
    }
    Ok(steps)
}

/// The steps of a split resolve of `source`: for each aspect of `aspects` resolved (its mode not
/// `None`), a subpass resolving it in that mode (the other aspect `NONE` where the device allows,
/// else the same mode) into a scratch image of `source`, whose aspect alone is copied into the
/// aspect's destination after the pass. Refused `NotImplemented(DepthStencilResolve)` where the
/// device has no such pair of modes or a destination's image takes no copy.
fn split_steps(
    maker: &CommandBuffer,
    source: &Target,
    aspects: [(ResolveMode, ImageAspectFlags, Option<Destination>); 2],
    modes: &ResolveModes,
) -> Result<Vec<ResolveStep>, Error> {
    let none = ResolveMode::None;
    let both =
        image_aspects(source.format).contains(ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL);
    let mut steps = Vec::new();
    for (mode, aspect, destination) in aspects {
        if mode == none {
            continue;
        }
        let destination =
            destination.ok_or_else(|| internal("a resolved aspect without its destination"))?;
        if !destination.usage.contains(ImageUsageFlags::TRANSFER_DST)
            || super::render_pass::not_2d(&destination.target)
        {
            return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
        }
        let pair = |other| {
            if aspect == ImageAspectFlags::DEPTH {
                (mode, other)
            } else {
                (other, mode)
            }
        };
        let (depth, stencil) = [pair(none), pair(mode)]
            .into_iter()
            .find(|&(d, s)| modes.allows(d, s, both))
            .ok_or(Error::NotImplemented(RuntimeGap::DepthStencilResolve))?;
        steps.push(ResolveStep {
            resolve: Resolve {
                attachment: scratch(maker, source, ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, 1)?,
                copy_to: Some(destination.target),
                shader: None,
                clear: None,
            },
            depth,
            stencil,
            aspects: aspect,
            one_aspect: true,
        });
    }
    Ok(steps)
}

/// A resolve texture's level and slice, with its image's usage.
#[derive(Clone)]
struct Destination {
    target: Target,
    usage: ImageUsageFlags,
}

impl Destination {
    /// The resolve into the destination: the pass's attachment where its image can be one with
    /// `needed` usage, else a scratch image whose render area is copied there after the pass. A
    /// render pass's depth/stencil resolve (`depth_stencil`) on a device that writes its resolve
    /// attachment image's first layer and level alone ([`first_subresource_only`]) takes the
    /// scratch image where the destination begins at another.
    fn attach(
        self,
        maker: &CommandBuffer,
        source: &Target,
        needed: ImageUsageFlags,
        depth_stencil: bool,
    ) -> Result<Resolve, Error> {
        let device = maker.device();
        // A layered pass's resolve attachment has every layer of the pass: a texture whose slices
        // hold them all is one, else the pass resolves into a scratch image of its layers whose
        // layers that fit are copied. A level of another extent than the attachment's is
        // resolved into through a scratch image too, whose overlap with it is copied.
        let misplaced =
            depth_stencil && first_subresource_only(device) && !at_first_subresource(&self.target);
        // Nor is a linear image the device renders only where every color attachment is linear
        // (`linear_attachment.rs`): the pass's are multisampled, never linear.
        let bound = !depth_stencil && linear_attachment::bound(device, &self.target)?;
        if bound {
            let format = self.target.texture.pixel_format();
            device.degrade(Degradation::LinearAttachmentCopies, || {
                (
                    format!("a {format:?} texture's linear image, the resolve of a render pass"),
                    "the device renders a linear image of the format only where every color \
                     attachment of the render pass instance is linear \
                     (VK_NV_linear_color_attachment), and the pass's are multisampled"
                        .to_owned(),
                )
            });
        }
        if self.usage.contains(needed)
            && self.target.layers == source.layers
            && self.target.extent == source.extent
            && !misplaced
            && !bound
        {
            return Ok(Resolve {
                attachment: self.target,
                copy_to: None,
                shader: None,
                clear: None,
            });
        }
        // The scratch's copy into a 3D texture's planes, or into a 1D texture, is not built
        // (`render_pass::not_2d`).
        if !self.usage.contains(ImageUsageFlags::TRANSFER_DST)
            || super::render_pass::not_2d(&self.target)
        {
            return Err(Error::NotImplemented(RuntimeGap::Render));
        }
        Ok(Resolve {
            attachment: scratch(maker, source, needed, 1)?,
            copy_to: Some(self.target),
            shader: None,
            clear: None,
        })
    }
}

/// Refuse the resolve of a single-sampled `source` (measured: Metal's validation stops it), but
/// for a color attachment's, a copy (`color_copy`).
fn single_sampled(call: &'static str, which: Attachment, source: &Target) -> Result<(), Error> {
    if source.samples < 2 {
        return Err(problem(
            call,
            RenderPassProblem::ResolveSourceSamples(which),
        ));
    }
    Ok(())
}

/// The resolve of single-sampled color attachment `source` (`which`) as `wanted` states it: the
/// resolve texture's level and slice that the attachment's texels are copied into after the pass,
/// over the render area and the pass's layers that fit (`ResolveCopies`), as plain Metal resolves
/// one sample (Metal probe case `ref_single_sampled_source` on an M4, which its validation stops:
/// the resolve texture holds the draw); `None` where it does not resolve. Refused as `color`
/// refuses a resolve but for the sample count, and `NotImplemented(Render)` where the copy is not
/// built: a memoryless attachment, or an attachment or destination whose image takes no copy or is
/// not 2D.
pub(super) fn color_copy(
    call: &'static str,
    device: &Device,
    which: Attachment,
    source: &Target,
    wanted: &Wanted<'_>,
) -> Result<Option<Target>, Error> {
    if !resolves(call, which, wanted)? {
        return Ok(None);
    }
    let texture = wanted
        .texture
        .ok_or_else(|| internal("a resolve without its texture"))?;
    if matches!(
        sampled_type(source.texture.pixel_format()),
        Some(SampledType::Uint | SampledType::Sint | SampledType::Uint64)
    ) {
        return Err(problem(call, RenderPassProblem::ResolveFormat(which)));
    }
    let destination = destination(call, device, which, source, texture, wanted)?
        .ok_or_else(|| internal("a single-sampled resolve left untouched"))?;
    // A level of another extent, measured of a multisampled attachment alone (`color`), stays
    // refused.
    if destination.target.extent != source.extent {
        return Err(problem(call, RenderPassProblem::ResolveSize(which)));
    }
    let copied = source.texture.storage_mode() != StorageMode::Memoryless
        && source
            .texture
            .image_plan()?
            .usage
            .contains(ImageUsageFlags::TRANSFER_SRC)
        && !super::render_pass::not_2d(source);
    if !copied
        || !destination.usage.contains(ImageUsageFlags::TRANSFER_DST)
        || super::render_pass::not_2d(&destination.target)
    {
        return Err(Error::NotImplemented(RuntimeGap::Render));
    }
    Ok(Some(destination.target))
}

/// The resolve destination of `source` in `texture` at `wanted`'s level, slice and depth plane,
/// checked as Metal checks it (but for `source`'s sample count, `single_sampled`); `None` for a
/// multisampled color resolve texture of another sample count than a multisampled `source`'s,
/// which plain Metal leaves untouched (the module docs).
fn destination(
    call: &'static str,
    device: &Device,
    which: Attachment,
    source: &Target,
    texture: &Texture,
    wanted: &Wanted<'_>,
) -> Result<Option<Destination>, Error> {
    if !texture.device().same(device) {
        return Err(invalid(call, Problem::ForeignObject));
    }
    if texture.storage_mode() == StorageMode::Memoryless {
        return Err(problem(call, RenderPassProblem::MemorylessResolve(which)));
    }
    let facts = Facts::of(texture.record());
    // A multisampled color resolve texture of another sample count is left untouched; one of
    // the attachment's count (plain Metal copies the samples, not built) and a depth or stencil
    // one (a layout pattern) are refused.
    let untouched = facts.samples > 1
        && source.samples > 1
        && facts.samples != source.samples
        && matches!(which, Attachment::Color(_));
    if facts.samples > 1 && !untouched {
        return Err(problem(call, RenderPassProblem::ResolveSamples(which)));
    }
    // A multisampled color attachment resolving into its format's sRGB twin keeps its own
    // encoding (the module docs): it resolves in the attachment's format.
    let pixel_format = source.texture.pixel_format();
    let twin = source.samples > 1 && srgb_twin(pixel_format) == Some(facts.pixel_format);
    if facts.pixel_format != pixel_format && !twin {
        // The other channel order of the 8-bit four-channel family: measured, not built.
        if source.samples > 1 && other_channel_order(pixel_format, facts.pixel_format) {
            return Err(Error::NotImplemented(RuntimeGap::Render));
        }
        return Err(problem(call, RenderPassProblem::ResolveFormat(which)));
    }
    match facts.texture_type {
        TextureType::Type2DMultisample | TextureType::Type2DMultisampleArray if untouched => {}
        TextureType::Type2D
        | TextureType::Type2DArray
        | TextureType::Cube
        | TextureType::CubeArray => {}
        // A texture buffer is resolved into through its attachment twin, as it renders
        // (`Texture::attachment_twin`; Metal resolves into one plain, the probe's `tb_resolve`).
        TextureType::TextureBuffer => {}
        // A 3D one is resolved into from its depth plane (the probe's `t3d_resolve_plane3`),
        // through a 2D view of its planes, and a 1D one through a 1D view (`t1d_resolve`: Metal
        // resolves plain, its validation stops the pass).
        TextureType::Type3D | TextureType::Type1D | TextureType::Type1DArray => {}
        _ => return Err(Error::NotImplemented(RuntimeGap::Render)),
    }
    let volume = facts.texture_type == TextureType::Type3D;
    let planes = |level: u64| (facts.extent[2] >> level.min(31)).max(1);
    // An index the texture's type does not have is ignored.
    let (slice, depth_plane) =
        super::render_pass::used_indices(&facts, wanted.slice, wanted.depth_plane);
    let past = if volume {
        wanted.level >= u64::from(facts.levels)
            || slice != 0
            || depth_plane >= u64::from(planes(wanted.level))
    } else {
        wanted.level >= u64::from(facts.levels)
            || slice >= u64::from(facts.slices)
            || depth_plane != 0
    };
    if past {
        return Err(problem(call, RenderPassProblem::ResolveSubresource(which)));
    }
    // The layer the resolve begins at, and the layers there are from it: a 3D texture's planes.
    let (first, available) = if volume {
        (depth_plane as u32, planes(wanted.level))
    } else {
        (
            facts.base_layer + slice as u32,
            facts.base_layer + facts.slices,
        )
    };
    let level = wanted.level as u32;
    let extent = [
        (facts.extent[0] >> level).max(1),
        (facts.extent[1] >> level).max(1),
    ];
    // A level of another extent resolves the overlap (`color` refuses a color one reaching past
    // the attachment without a clear); one of another sample count is left untouched where
    // its extent is the attachment's (measured), and refused elsewhere (unmeasured).
    if untouched {
        return match extent == source.extent {
            true => Ok(None),
            false => Err(problem(call, RenderPassProblem::ResolveSize(which))),
        };
    }
    // A view of another format is resolved into in that format, as it renders
    // (`render_pass::attachment_format`), and the sRGB twin of the attachment's format in the
    // attachment's, the twin among its image's view formats. A 2D texture over a buffer is resolved
    // into through its aliasing image or its shadow, as it renders (`render/shadows.rs`), a texture
    // buffer through its attachment twin, and a rotated texture through its twin of the pass's
    // frame.
    let texture = &texture.pass_texture()?;
    let plan = texture.image_plan()?;
    let (format, usage) =
        match super::render_pass::attachment_format(which, plan, pixel_format, device) {
            Ok(format) => (format, plan.usage),
            // A view in a format the device does not render resolves through a scratch of the
            // source's format (the same Metal format), copied in bytes (`Destination::attach`).
            Err(Error::Unsupported(_)) => (
                source.format,
                plan.usage.difference(
                    ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                ),
            ),
            Err(error) => return Err(error),
        };
    if format != source.format {
        return Err(Error::NotImplemented(RuntimeGap::Render));
    }
    // Layer `i` of a layered pass resolves into slice `resolveSlice + i`, and a layer past the
    // texture's slices is not resolved (a Metal probe, an M3 Max and an M4 alike): the resolve
    // writes the layers that fit, at least the first. A depth or stencil attachment resolves its
    // first layer alone, into slice `resolveSlice`; the later layers are not resolved (a probe, an
    // M3 Max and an M4 alike).
    let layers = match which {
        Attachment::Color(_) => source.layers,
        Attachment::Depth | Attachment::Stencil => 1,
    };
    let fitting = (available - first).min(layers);
    let resolved = Target {
        texture: texture.clone(),
        level: facts.base_level + level,
        layer: first,
        layers: fitting,
        format,
        samples: 1,
        extent,
    };
    Ok(Some(Destination {
        target: resolved,
        usage,
    }))
}

/// Whether `a` and `b` are `RGBA8Unorm` and `BGRA8Unorm`, either of them its sRGB twin: one
/// resolves into the other in the destination's channel order and the attachment's encoding
/// (the module docs).
fn other_channel_order(a: PixelFormat, b: PixelFormat) -> bool {
    use PixelFormat as P;
    matches!(
        (block_family(a), block_family(b)),
        (P::RGBA8Unorm, P::BGRA8Unorm) | (P::BGRA8Unorm, P::RGBA8Unorm)
    )
}

/// An image of `source`'s format, extent and layers, of `samples`, that the pass resolves into,
/// then copies from (one sample), or that a copy of `source` is resolved from (its samples,
/// [`ResolveSource`], and the exact color resolve's copy, `color_resolve.rs`).
pub(super) fn scratch(
    maker: &CommandBuffer,
    source: &Target,
    usage: ImageUsageFlags,
    samples: u32,
) -> Result<Target, Error> {
    let format = source.format;
    let [width, height] = source.extent;
    let layers = source.layers;
    let pixel_format = source.texture.pixel_format();
    let descriptor = TextureDescriptor {
        texture_type: match (samples > 1, layers > 1) {
            (false, false) => TextureType::Type2D,
            (false, true) => TextureType::Type2DArray,
            (true, false) => TextureType::Type2DMultisample,
            (true, true) => TextureType::Type2DMultisampleArray,
        },
        pixel_format,
        width: width.into(),
        height: height.into(),
        array_length: layers.into(),
        sample_count: samples.into(),
        usage: TextureUsage::RENDER_TARGET,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let aspects = aspects(pixel_format).unwrap_or(ImageAspectFlags::COLOR);
    let plan = ImagePlan {
        pixel_format,
        flags: ImageCreateFlags::empty(),
        image_type: ImageType::TYPE_2D,
        format,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples: sample_count(samples)
            .ok_or_else(|| internal("an attachment of a sample count Vulkan lacks"))?,
        usage: usage | ImageUsageFlags::TRANSFER_SRC,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(format, aspects),
        unavailable: TextureUsage::SHADER_READ
            | TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    let texture = Device::new_planned_texture(maker, &descriptor, plan)?;
    Ok(Target {
        texture,
        level: 0,
        layer: 0,
        layers,
        format,
        samples,
        extent: source.extent,
    })
}

/// Whether `device`'s render pass resolves a depth/stencil attachment from its image's first
/// layer and level, and into its resolve attachment image's, whatever layer and level their
/// views begin at (`RuntimeDriverDefect::DepthStencilResolveFirstSubresource`).
pub(super) fn first_subresource_only(device: &Device) -> bool {
    device
        .properties()
        .driver_defects
        .contains(&RuntimeDriverDefect::DepthStencilResolveFirstSubresource)
}

/// Whether `target` begins at its image's first layer and level.
fn at_first_subresource(target: &Target) -> bool {
    target.layer == 0 && target.level == 0
}

/// The copy a pass's depth/stencil resolves read on a device whose render pass resolves from the
/// attachment image's first layer and level ([`first_subresource_only`]), where the pass's
/// depth/stencil attachment begins at another: the pass's layers of the attachment, every aspect
/// of its format, copied after the pass's instances into a scratch image of its format, samples,
/// extent and layers, from the first (`vkCmdCopyImage` of one format and sample count). The pass
/// then resolves in an instance of no draws whose depth/stencil attachment is the scratch image,
/// as do its resolves after the pass (`render.rs`).
pub(super) struct ResolveSource {
    /// The scratch image: the depth/stencil attachment of the instances that resolve.
    pub(super) target: Target,
    from: vk::Image,
    to: vk::Image,
    region: vk::ImageCopy,
    /// The copy's read of the attachment and write of the scratch image.
    pub(super) uses: Vec<Use<ResourceKey>>,
}

impl ResolveSource {
    /// The copy `layout`'s depth/stencil resolves read on `device`: `None` where the device
    /// resolves where the attachment lies, the attachment begins at its image's first layer and
    /// level, or no render pass resolves it (the draws of `depth_stencil_resolve.rs` read it
    /// through views). Refused `NotImplemented(DepthStencilResolve)` where the attachment's image
    /// takes no copy (a memoryless texture's).
    pub(super) fn plan(
        maker: &CommandBuffer,
        layout: &PassLayout,
    ) -> Result<Option<ResolveSource>, Error> {
        let device = maker.device();
        let Some(ds) = &layout.depth_stencil else {
            return Ok(None);
        };
        let source = &ds.target;
        let resolved = ds.resolve.as_ref().is_some_and(|r| !r.steps.is_empty());
        if !resolved || !first_subresource_only(device) || at_first_subresource(source) {
            return Ok(None);
        }
        let copied = source.texture.storage_mode() != StorageMode::Memoryless
            && source
                .texture
                .image_plan()?
                .usage
                .contains(ImageUsageFlags::TRANSFER_SRC);
        if !copied {
            return Err(Error::NotImplemented(RuntimeGap::DepthStencilResolve));
        }
        let usage = ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::TRANSFER_DST;
        let target = scratch(maker, source, usage, source.samples)?;
        let aspect_mask = image_aspects(source.format);
        let layers = |t: &Target| vk::ImageSubresourceLayers {
            aspect_mask: aspect_mask.into(),
            mip_level: t.level,
            base_array_layer: t.layer,
            layer_count: t.layers,
        };
        let [width, height] = source.extent;
        let region = vk::ImageCopy {
            src_subresource: layers(source),
            src_offset: vk::Offset3D::default(),
            dst_subresource: layers(&target),
            dst_offset: vk::Offset3D::default(),
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let at = |t: &Target| t.level..t.level + 1;
        let on = |t: &Target| t.layer..t.layer + t.layers;
        let uses = vec![
            source
                .texture
                .image_use(at(source), on(source), read, false)?,
            target
                .texture
                .image_use(at(&target), on(&target), write, true)?,
        ];
        Ok(Some(ResolveSource {
            from: source.texture.raw_image()?,
            to: target.texture.raw_image()?,
            region,
            uses,
            target,
        }))
    }

    /// `layout` with the scratch image as its depth/stencil attachment: the layout of the
    /// instances that resolve.
    pub(super) fn resolving(&self, layout: &PassLayout) -> PassLayout {
        let mut resolving = layout.clone();
        if let Some(ds) = resolving.depth_stencil.as_mut() {
            ds.target = self.target.clone();
        }
        resolving
    }

    /// Record the copy, after a barrier from everything before.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it; both
    /// images are live until it completes (the encoder retains the attachment, and this the
    /// scratch image), and rest in `GENERAL`.
    pub(super) unsafe fn record(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let general = vk::ImageLayout::GENERAL;
        // SAFETY: the caller's contract; the region is the attachment's level and the pass's
        // layers of both images, which have one format, sample count and extent.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            raw.cmd_copy_image(
                command_buffer,
                self.from,
                general,
                self.to,
                general,
                &[self.region],
            );
        }
    }
}

/// The copies of a pass's scratch resolves into their textures, over the render area within the
/// texture's level (a smaller level takes the overlap) and the texture's layers (those of a layered
/// pass that fit in its slices), each after the clear its resolve names of those layers.
///
/// A copy of every aspect its format has is one image copy. A copy of one aspect of a
/// depth/stencil format with two (a resolve of one aspect at a time, or one aspect resolved into a
/// scratch image) goes out of the scratch image into a staging buffer and from it into the
/// texture: a buffer copy's bytes are the aspect's alone by definition, while lavapipe's image
/// copy of `D32_SFLOAT_S8_UINT`'s stencil aspect also wrote the depth aspect
/// (`runtime_lane::resolves`, `ds_unresolved_sample0`, at 1ae4ebbc).
pub(super) struct ResolveCopies {
    /// `(texture, clear color bits, its level and layers)` of each clear before the copies.
    clears: Vec<(vk::Image, [u32; 4], vk::ImageSubresourceRange)>,
    /// `(scratch, texture, region)` of each image copy.
    copies: Vec<(vk::Image, vk::Image, vk::ImageCopy)>,
    /// The aspect copies through a staging buffer.
    staged: Option<Staged>,
    uses: Vec<Use<ResourceKey>>,
}

/// One-aspect copies through a staging buffer: each into the buffer, then each out of it.
struct Staged {
    staging: Buffer,
    into_buffer: Vec<(vk::Image, vk::BufferImageCopy)>,
    out_of_buffer: Vec<(vk::Image, vk::BufferImageCopy)>,
}

impl ResolveCopies {
    /// The copies of `resolves` (each with the aspects it resolved) over the render area
    /// `extent`; `None` when no resolve went through a scratch image.
    pub(super) fn plan(
        device: &Device,
        resolves: &[(&Resolve, ImageAspectFlags)],
        extent: [u32; 2],
    ) -> Result<Option<ResolveCopies>, Error> {
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let levels = |t: &Target| t.level..t.level + 1;
        // A copy takes the texture's layers (those of a layered pass that fit in its slices)
        // from the scratch image's first.
        let on = |t: &Target, count: u32| t.layer..t.layer + count;
        let layers =
            |t: &Target, aspect: ImageAspectFlags, count: u32| vk::ImageSubresourceLayers {
                aspect_mask: aspect.into(),
                mip_level: t.level,
                base_array_layer: t.layer,
                layer_count: count,
            };
        let mut clears = Vec::new();
        let mut copies = Vec::new();
        let mut uses = Vec::new();
        // `(scratch, texture, aspect, offset, layers)` of each staged copy.
        let mut aspect_copies = Vec::new();
        let mut length = 0u64;
        for &(resolve, aspect) in resolves {
            let Some(to) = &resolve.copy_to else {
                continue;
            };
            let from = &resolve.attachment;
            let count = to.layers;
            let [width, height] = [extent[0].min(to.extent[0]), extent[1].min(to.extent[1])];
            uses.push(
                from.texture
                    .image_use(levels(from), on(from, count), read, false)?,
            );
            uses.push(
                to.texture
                    .image_use(levels(to), on(to, count), write, true)?,
            );
            if let Some(bits) = resolve.clear {
                let range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: to.level,
                    level_count: 1,
                    base_array_layer: to.layer,
                    layer_count: count,
                };
                clears.push((to.texture.raw_image()?, bits, range));
            }
            if aspect == image_aspects(from.format) {
                let region = vk::ImageCopy {
                    src_subresource: layers(from, aspect, count),
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: layers(to, aspect, count),
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                };
                copies.push((from.texture.raw_image()?, to.texture.raw_image()?, region));
                continue;
            }
            let bytes = copy_units(from.format, from.texture.pixel_format())
                .into_iter()
                .find(|unit| unit.0 == aspect)
                .map(|unit| u64::from(unit.1))
                .ok_or_else(|| internal("a resolve copy of an aspect its format lacks"))?;
            // A depth/stencil copy's buffer offset is a multiple of 4; its layers lie one after
            // another.
            let offset = length.next_multiple_of(4);
            let texels = u64::from(width) * u64::from(height);
            length = offset + bytes * texels * u64::from(count);
            aspect_copies.push((from, to, aspect, offset, count, [width, height]));
        }
        let staged = if aspect_copies.is_empty() {
            None
        } else {
            let staging = device.own_buffer(length, ResourceOptions::private())?;
            uses.push(writes(&staging, 0..length));
            uses.push(reads(&staging, 0..length));
            let (mut into_buffer, mut out_of_buffer) = (Vec::new(), Vec::new());
            for (from, to, aspect, offset, count, [width, height]) in aspect_copies {
                let region = |target: &Target| vk::BufferImageCopy {
                    buffer_offset: staging.at(offset).1,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: layers(target, aspect, count),
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                };
                into_buffer.push((from.texture.raw_image()?, region(from)));
                out_of_buffer.push((to.texture.raw_image()?, region(to)));
            }
            Some(Staged {
                staging,
                into_buffer,
                out_of_buffer,
            })
        };
        Ok(
            (!copies.is_empty() || staged.is_some()).then_some(ResolveCopies {
                clears,
                copies,
                staged,
                uses,
            }),
        )
    }

    /// Record the copies in `core`'s command buffer after the pass, between barriers from
    /// everything before and to everything after, and hold the staging buffer until the
    /// command buffer completes.
    pub(super) fn record(self, core: &mut EncoderCore) {
        let ResolveCopies {
            clears,
            copies,
            staged,
            uses,
        } = self;
        core.barrier(Barrier::everything());
        core.command(TRANSFER, &uses, |device, raw| {
            let general = vk::ImageLayout::GENERAL;
            // SAFETY: the encoder's command buffer is recording outside a render pass; the images
            // and the staging buffer are live until it completes (the encoder retains the
            // textures and the scratch images, and this the buffer); every image rests in
            // `GENERAL`, and each region lies in its image and in the buffer (the scratch image is
            // the attachment's extent and has the pass's layers, the region the render area
            // within the resolve level, which has the copied layers; the buffer as planned); a
            // cleared image has transfer-destination usage, as its copy needs.
            unsafe {
                for (image, bits, range) in &clears {
                    let color = vk::ClearColorValue { uint32: *bits };
                    device.cmd_clear_color_image(raw, *image, general, &color, &[*range]);
                }
                if !clears.is_empty() {
                    Barrier::everything().record(device, raw);
                }
                for (from, to, region) in &copies {
                    device.cmd_copy_image(raw, *from, general, *to, general, &[*region]);
                }
                if let Some(staged) = &staged {
                    let (buffer, _) = staged.staging.at(0);
                    for (image, region) in &staged.into_buffer {
                        device.cmd_copy_image_to_buffer(raw, *image, general, buffer, &[*region]);
                    }
                    Barrier::everything().record(device, raw);
                    for (image, region) in &staged.out_of_buffer {
                        device.cmd_copy_buffer_to_image(raw, buffer, *image, general, &[*region]);
                    }
                }
            }
        });
        core.barrier(Barrier::everything());
        if let Some(staged) = staged {
            core.retain(staged.staging);
        }
    }
}
