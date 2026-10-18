//! A render target's 32-bit float depth that holds a value Vulkan's depth test compares otherwise
//! than Metal's, tested as Metal tests it, over its magnitude stand-in, and stored as Metal stores
//! it after a pass with depth writes, without its sign.
//!
//! **Metal's answer** (probe `depthkeep`, the M3 Max under macOS 27.0.1, 2026-10-15): a
//! `Depth32Float` texel keeps every `float`'s bits, NaN included, through a blit in and out,
//! `replaceRegion`, a kernel read, a nearest sample, and a pass's load and store without depth
//! writes; `sample_compare` is the IEEE compare of the reference with the texel, neither clamped
//! (720 of 720 per texture class); and the depth test compares the fragment's depth with the stored
//! value's magnitude, as the integers of their low 31 bits: the sign is dropped and a NaN orders
//! above +inf (270 of 270 outcomes: 5 compare functions, fragments at 0, 0.5 and 1, 18 stored
//! values). A pass that loads and stores the depth and sets a depth/stencil state with depth writes
//! in it, whether or not a draw follows or a fragment passes, stores every texel of the attachment
//! without its sign, `w & 0x7fffffff` (a NaN's payload and a denormal kept, -0 as +0), where no
//! fragment wrote it; a pass without one keeps every bit (our probe `depthstore`, the M3 Max under
//! macOS 27.0.1: single-sample `Depth32Float` and `Depth32Float_Stencil8`, 4x4 and 64x64, `Shared`
//! and `Private`, read by `getBytes`, a blit and a kernel, 42 cases; a 4x multisampled
//! `Depth32Float` a blit from a buffer wrote, per sample). Vulkan's depth test compares the stored
//! value as it is, and on most drivers a stored denormal as zero
//! (`DriverFact::DepthTestKeepsDenormals`); its pass keeps every bit.
//!
//! - **Which textures** ([`Texture::magnitude_form`]): a render target whose image of its own
//!   holds a 32-bit float depth (`D32_SFLOAT` or `D32_SFLOAT_S8_UINT`) with transfer usage, not
//!   memoryless and not sparse, on a device that keeps a value outside `[0, 1]` in a depth image
//!   (`VK_EXT_depth_range_unrestricted` enabled, or
//!   `DriverFact::DepthImageKeepsValuesOutsideUnitRange`). Its image keeps every value a write
//!   stores, so every read, sample, compare and copy out of it is Metal's; its depth test and the
//!   stores of a pass with depth writes may not be. A single-sample `Depth32Float` with depth
//!   attachment usage, on a device whose depth test is not Metal's already, takes the magnitude
//!   stand-in ([`MagnitudeForm::StandIn`]): Metal's test and Metal's stores. Every other such
//!   target takes the strip ([`MagnitudeForm::Strip`]): Metal's stores, and on a device whose
//!   depth test compares magnitudes as Metal's (`DriverFact::DepthTestComparesMagnitudes`:
//!   MoltenVK, whose test and pass are the GPU's own Metal ones) Metal's test too; elsewhere its
//!   test compares the value as it is, announced at the write (`signed-depth-test`). A
//!   multisampled one takes it where the device shades at sample rate and either copies its depth
//!   to and from colour (maintenance8, without
//!   `RuntimeDriverDefect::MultisampledDepthStencilColourCopies`) or writes a 32-bit float
//!   `FragDepth` (`depth_copy::drawn`). A device that may clamp such a value announces
//!   `restricted-depth-upload`.
//! - **Switched on** ([`Texture::switch_on_signed_depth`]) by a write that may store a value whose
//!   stand-in is not itself, before its data lands: every blit into the depth from a buffer or
//!   another texture, but from a texture known to hold none ([`Texture::may_hold_signed_depth`]), a
//!   `replace_region` whose bytes hold one, a pass that clears it to a denormal where the
//!   device's test compares one as zero, and a pass that loads a new texture's pattern, a NaN with
//!   the sign set, before a write covered it ([`Texture::switch_on_fresh_pattern`], as the pass is
//!   planned: `fresh_unwritten.rs`). A flag on the image, set once and never cleared, counted
//!   once (`WorkCounts::signed_depth_switch_ons`). A pass encoded before the switch in a command
//!   buffer not yet committed renders the depth in place: where another command buffer than the
//!   writer's was uncommitted at the switch, `signed-depth-test` is announced.
//! - **The pass.** A pass whose depth attachment is such a texture and loads it (`Load`) renders
//!   over its magnitude stand-in, the image the texture keeps for the queue
//!   (`attachment_stand_in::kept_backing`, `Cause::Magnitude`), decided as the encoder begins
//!   (`render_pass.rs`). Before the pass, the depth's words over the render area are copied into a
//!   scratch buffer, one dispatch of `bind::depth_magnitude_module` writes each word's stand-in
//!   (its magnitude, a NaN's +inf, and where the device's test compares a denormal as zero, a
//!   denormal's the smallest normal), and those are copied into the stand-in, which the pass loads,
//!   tests and writes: a fragment's depth is never negative or NaN, so the device's test gives
//!   Metal's answer. After the pass, where it stores the depth and may have written it (a
//!   depth/stencil state with depth writes set in it, or an indirect command buffer's commands that
//!   set their own), the stand-in is copied out, merged into the depth's words (a word the pass
//!   left as it loaded it is the depth's without its sign, as Metal stores it; any other is the
//!   pass's), and those are copied back.
//!   Copies keep every bit on such a device but a denormal where its GPU copies store one as a zero
//!   (no `DriverFact::DepthImageKeepsDenormals`, which `depth-denormals` names), and the kernel's
//!   are integer operations, so no driver's fragment depth rounding enters (NVIDIA's stores a NaN
//!   as -inf). Counted per pass (`WorkCounts::magnitude_stand_in_passes`, `magnitude_write_backs`).
//!   A pass that clears its depth or leaves it to the pass (`DontCare`) reads nothing of it and
//!   renders it in place.
//! - **The strip** ([`SignStrip`]). After a pass that loads and stores a switched-on depth in
//!   place, or through another image copied back into it (separate depth and stencil textures'
//!   scratch image, an attachment stand-in), and sets a state with depth writes in it, where the
//!   device's pass leaves the words it did not write signed, the depth's words over the render area
//!   are stored without their signs, Metal's words, as no fragment wrote them, and a fragment's
//!   depth is never negative. A device whose pass is Metal's own strips them as Metal's does
//!   wherever a draw that records a command ran under such a state, which hands the device's pass
//!   that state; a pass that sets one but draws under none (no draw, draws of no vertices or no
//!   instances, which record nothing, or draws under a state replaced by one without depth writes)
//!   leaves them signed (measured through MoltenVK on the M3 Max: an indirect draw whose arguments
//!   draw no instances hands it the state too), so only such a pass is stripped there; on any other
//!   device, every such pass. A single-sample depth's words are copied out into a scratch buffer,
//!   one dispatch of the kernel's strip stores each without its sign, and they are copied back. A
//!   multisampled depth's samples, which Vulkan copies into no buffer, are copied into an
//!   `R32_UINT` colour image of their bits by maintenance8's `vkCmdCopyImage`, written without the
//!   sign bit into another by a sample-rate bits draw of each layer
//!   (`DepthStencilCopy::MagnitudeBits`), and copied back: exact. Where those copies are missing or
//!   run wrong, they are copied into a sampled image of the depth's format and written back by a
//!   sample-rate `FragDepth` draw of each layer (`DepthStencilCopy::DepthMagnitude`), exact where
//!   the driver's fragment depth stores every value unchanged, and elsewhere the announced
//!   near-equivalent `drawn-depth-copy` (what MoltenVK's and NVIDIA's are measured to change, or
//!   Vulkan leaves to a driver not measured; on those drivers a blit into a multisampled depth
//!   goes by the same draw). A combined depth/stencil format's
//!   stencil is not touched. Counted per pass
//!   (`WorkCounts::sign_strips`); a strip the device does not make is announced
//!   (`signed-depth-test`).
//! - **What stays announced.** `signed-depth-test`, where a pass's merge may keep a value the pass
//!   wrote over (a draw that writes exactly +inf, a stored NaN's stand-in, over it); where a pass
//!   loads such a depth through another image (separate depth and stencil textures' scratch image,
//!   a stand-in for custom sample positions), whose test compares the value as it is; at a switch
//!   with another command buffer uncommitted; and for a target that takes the strip on a device
//!   whose test is not Metal's (multisampled, or `Depth32Float_Stencil8`), whose test compares the
//!   value as it is. `depth-denormals`, where a denormal is tested over the smallest normal (a
//!   fragment at exactly 2^-126 passes `Equal` and fails `Greater` against it, where Metal's
//!   answers otherwise). Unmeasured and left as the device's: a multisampled depth's resolve in a
//!   pass with depth writes (`StoreAndMultisampleResolve`), which resolves the samples as they are
//!   before the strip.
//! - **The cost** where no texture switched on: one flag read as a pass's encoder begins and as
//!   it ends, and a boolean as a depth/stencil state is set and at each draw. No barrier, copy,
//!   command buffer or descriptor.

use super::{Storage, Texture};
use crate::blit::TRANSFER;
use crate::buffer::Buffer;
use crate::command_buffer::CommandBuffer;
use crate::degradation::Degradation;
use crate::depth_copy::{self, DepthDraw, Drawn};
use crate::depth_stencil_scratch::colour_copies_refused;
use crate::encoder::EncoderCore;
use crate::error::{internal, Error};
use crate::image_plan::{sample_count, ImagePlan};
use crate::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use crate::render_pass::Target;
use crate::resource::ResourceKey;
use crate::sample_copy::bits_image;
use crate::stencil_copy::{self, DrawArea};
use crate::texture_descriptor::TextureDescriptor;
use crate::tracker::{Access, Barrier, Use};
use crate::values::{ResourceOptions, StorageMode};
use crate::Device;
use ash::vk;
use metal2vulkan::bind::DepthMagnitudeModule;
use metal2vulkan::device::{DriverFact, VkFeature};
use metal2vulkan::error::Feature;
use metal2vulkan::metal::format::FormatMapping;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{
    Format, ImageAspectFlags, ImageCreateFlags, ImageType, ImageUsageFlags, SampleCountFlags,
};
use std::sync::atomic::Ordering;

/// What the offsets of the bindings are rounded to: a multiple of every
/// `minStorageBufferOffsetAlignment`.
const ALIGNMENT: u64 = 256;

/// Whether the depth test of `device` compares a stored denormal otherwise than Metal's (no
/// `DriverFact::DepthTestKeepsDenormals`): a denormal's stand-in is then the smallest normal.
pub(crate) fn test_flushes_denormals(device: &Device) -> bool {
    !device
        .properties()
        .driver_facts
        .contains(&DriverFact::DepthTestKeepsDenormals)
}

/// Whether `device`'s depth test and its pass's stores are Metal's own
/// (`DriverFact::DepthTestComparesMagnitudes`: MoltenVK).
fn tests_magnitudes(device: &Device) -> bool {
    let profile = device.compile_device().profile();
    device
        .shared()
        .textures
        .facts(profile)
        .depth_images
        .test_magnitudes
}

/// How a render target's 32-bit float depth that may hold a value whose stand-in is not itself
/// keeps Metal's depth test and Metal's stores (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MagnitudeForm {
    /// A pass that loads it renders over its magnitude stand-in, merged back after it where it
    /// sets a state with depth writes.
    StandIn,
    /// A pass renders it in place, and after one that sets a state with depth writes, whose
    /// stores the device's pass leaves signed, its words are stored without their signs
    /// ([`SignStrip`]): after every such pass, or, where the device's depth test and pass are
    /// Metal's own (`DriverFact::DepthTestComparesMagnitudes`), after one that draws under none.
    Strip,
}

/// How a multisampled depth of `format` and image `usage` is stripped of its signs on `device`
/// (the module docs), `None` where it is not: through maintenance8's depth/colour copies where
/// they move its depth, else by draws where the device's fragment depth stores a 32-bit float
/// depth (`depth_copy::drawn`); each at sample rate.
fn sample_strip(device: &Device, format: Format, usage: ImageUsageFlags) -> Option<SampleStrip> {
    if !device
        .enabled()
        .features
        .contains(&VkFeature::SampleRateShading)
    {
        return None;
    }
    let feature = Feature::SeparateMultisampledDepthStencil;
    if colour_copies_refused(device, feature, ImageAspectFlags::DEPTH).is_none() {
        return Some(SampleStrip::Colour);
    }
    (usage.contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        && depth_copy::drawn(device, format).is_some())
    .then_some(SampleStrip::Draws)
}

/// How a multisampled depth's strip moves its samples ([`sample_strip`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleStrip {
    /// Into an `R32_UINT` colour image of its bits by maintenance8's copy, without the sign bit
    /// by a bits draw into another, and back.
    Colour,
    /// Into a sampled image of its format by a copy, and back without the sign by a `FragDepth`
    /// draw.
    Draws,
}

impl Texture {
    /// How the texture's depth keeps Metal's answers once a write may have stored a value whose
    /// stand-in is not itself (the module docs' textures): `None` for any other texture.
    pub(crate) fn magnitude_form(&self) -> Option<MagnitudeForm> {
        let image = self.image().ok()?;
        if !matches!(self.0.storage, Storage::Image(_)) {
            return None;
        }
        let plan = image.plan();
        let transfer = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        let device = &image.device;
        let profile = device.compile_device().profile();
        let facts = device.shared().textures.facts(profile);
        let held = matches!(plan.format, Format::D32_SFLOAT | Format::D32_SFLOAT_S8_UINT)
            && plan.usage.contains(transfer)
            && image.twin.is_none()
            && self.usage().contains(TextureUsage::RENDER_TARGET)
            && self.storage_mode() != StorageMode::Memoryless
            && !self.is_sparse()
            && !facts.depth_clamps();
        if !held {
            return None;
        }
        let single = plan.samples == SampleCountFlags::from_raw(1);
        let stands_in = single
            && plan.format == Format::D32_SFLOAT
            && plan
                .usage
                .contains(ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            && !facts.depth_images.test_magnitudes;
        match stands_in {
            true => Some(MagnitudeForm::StandIn),
            false => (single || sample_strip(device, plan.format, plan.usage).is_some())
                .then_some(MagnitudeForm::Strip),
        }
    }

    /// Whether a pass renders the texture's depth over its magnitude stand-in once a write may
    /// have stored a value the device's depth test compares otherwise than Metal's (the module
    /// docs' textures).
    pub(crate) fn takes_magnitude_stand_in(&self) -> bool {
        self.magnitude_form() == Some(MagnitudeForm::StandIn)
    }

    /// Whether a write switched the texture's depth on (the module docs). One atomic read.
    fn switched_on(&self) -> bool {
        self.image()
            .is_ok_and(|image| image.signed_depth.load(Ordering::Acquire))
    }

    /// Whether a pass that loads the texture's depth renders it over its magnitude stand-in: a
    /// write switched it on (the module docs), and it takes the stand-in. One atomic read where
    /// none did.
    pub(crate) fn renders_over_magnitude(&self) -> bool {
        self.switched_on() && self.takes_magnitude_stand_in()
    }

    /// Whether a pass that loads and stores the texture's depth, not over its magnitude stand-in,
    /// and sets a depth/stencil state with depth writes leaves its words to be stored without
    /// their signs after it ([`SignStrip`]): a write switched it on, it has a [`MagnitudeForm`],
    /// and the device's pass keeps the signs, which on a device whose pass is Metal's own is one
    /// where no draw ran under such a state (`drew`). One atomic read where none switched on.
    pub(crate) fn strips_signs(&self, drew: bool) -> bool {
        self.switched_on()
            && self.magnitude_form().is_some()
            && !(drew && tests_magnitudes(self.device()))
    }

    /// Whether the texture's depth may hold a value whose stand-in is not itself, which a copy
    /// from it carries: all but a texture with a form that was never switched on, every write of
    /// which that may store one would have switched it on.
    pub(crate) fn may_hold_signed_depth(&self) -> bool {
        self.magnitude_form().is_none() || self.switched_on()
    }

    /// Switch the texture's depth on for a pass, being planned, that loads a new texture's
    /// pattern, a NaN with the sign set, before a write covered it (`fresh_unwritten.rs`): as
    /// [`Self::switch_on_signed_depth`], but no pass encoded before can have tested the pattern in
    /// place, as the first that loads it switches it on, so no other command buffer is weighed.
    /// Whether the device's depth test then answers as Metal's (else the caller announces).
    pub(crate) fn switch_on_fresh_pattern(&self) -> bool {
        let Some(form) = self.magnitude_form() else {
            return false;
        };
        let Ok(image) = self.image() else {
            return false;
        };
        let device = &image.device;
        if !image.signed_depth.swap(true, Ordering::AcqRel) {
            device.shared().counts.signed_depth_switch_on();
        }
        form == MagnitudeForm::StandIn || tests_magnitudes(device)
    }

    /// Switch the texture's depth on for `call`'s write, which may store a value whose stand-in
    /// is not itself (the module docs), `encoding` where a command buffer of the program's records
    /// the write: whether the device's depth test then answers as Metal's, the texture taking
    /// the magnitude stand-in or the device's test being Metal's own (else the caller announces).
    /// From then on a pass that loads it renders over its magnitude stand-in, or its words are
    /// stripped after a pass with depth writes whose stores the device's pass leaves signed
    /// ([`MagnitudeForm::Strip`]). At the switch, a command buffer uncommitted besides the
    /// writer's may hold a pass encoded before it, which does neither: `signed-depth-test`.
    pub(crate) fn switch_on_signed_depth(&self, call: &'static str, encoding: bool) -> bool {
        let Some(form) = self.magnitude_form() else {
            return false;
        };
        let Ok(image) = self.image() else {
            return false;
        };
        let device = &image.device;
        let native = tests_magnitudes(device);
        let tested = form == MagnitudeForm::StandIn || native;
        if image.signed_depth.swap(true, Ordering::AcqRel) {
            return tested;
        }
        let shared = device.shared();
        shared.counts.signed_depth_switch_on();
        let others = shared
            .uncommitted
            .count()
            .saturating_sub(usize::from(encoding));
        if others > 0 {
            device.degrade(Degradation::SignedDepthTest, || {
                let format = self.pixel_format();
                let why = match (form, native) {
                    (MagnitudeForm::StandIn, _) => format!(
                        "{call} may store a negative or NaN depth in a render target's depth \
                         image, whose passes render over its magnitude stand-in from now on; \
                         {others} other command buffer(s) not yet committed may hold a pass \
                         encoded before, which tests the stored value as it is, where Metal's \
                         test compares its magnitude"
                    ),
                    (MagnitudeForm::Strip, true) => format!(
                        "{call} may store a negative or NaN depth in a render target's depth \
                         image, whose words a pass with depth writes that draws under none stores \
                         without their signs from now on, as Metal's; {others} other command \
                         buffer(s) not yet committed may hold such a pass encoded before, which \
                         keeps the signs"
                    ),
                    (MagnitudeForm::Strip, false) => format!(
                        "{call} may store a negative or NaN depth in a render target's depth \
                         image, whose words a pass with depth writes stores without their signs \
                         from now on, as Metal's; {others} other command buffer(s) not yet \
                         committed may hold such a pass encoded before, which keeps the signs"
                    ),
                };
                (format!("a {format:?} texture"), why)
            });
        }
        tested
    }
}

/// Announce `signed-depth-test` for a pass that loads `own`, a depth switched on to render over
/// its magnitude stand-in, through another image (`through`): separate depth and stencil textures'
/// scratch image, or a stand-in for custom sample positions, which the pass loads as it is.
pub(crate) fn degrade_through(device: &Device, own: &Target, through: &str) {
    device.degrade(Degradation::SignedDepthTest, || {
        let format = own.texture.pixel_format();
        let why = format!(
            "a pass loads a depth that may hold a negative or NaN value through {through}, whose \
             depth test compares the stored value as it is, where Metal's compares its magnitude"
        );
        (format!("a {format:?} texture"), why)
    });
}

/// Announce `signed-depth-test` for a pass that renders `own`, a depth, over its magnitude
/// stand-in and merges what it changed back: a draw that wrote exactly a stored NaN's stand-in,
/// +inf, over it leaves the NaN without its sign, where Metal stores the draw's.
pub(crate) fn degrade_write_back(device: &Device, own: &Target) {
    device.degrade(Degradation::SignedDepthTest, || {
        let format = own.texture.pixel_format();
        let why = "a pass that loads a depth holding a negative or NaN value tests it over its \
                   magnitude, as Metal's test does, and stores the values its draws changed and the \
                   others without their signs, as Metal stores them: a draw that writes exactly \
                   +inf over a stored NaN leaves the NaN, where Metal stores the draw's"
            .to_owned();
        (format!("a {format:?} texture"), why)
    });
}

/// One dispatch of the kernel: its workgroups and push bytes for each direction.
struct Window {
    groups: u32,
    count: u32,
}

/// The copies and dispatches around a pass that renders a depth over its magnitude stand-in (the
/// module docs): the stand-in written before it, and, where `back`, the pass's words merged back
/// after it.
pub(crate) struct MagnitudeCopies {
    /// The depth's own image, and the stand-in the pass renders.
    depth: vk::Image,
    stand_in: vk::Image,
    /// The render area's copy of the depth's words into the scratch's exact part, and of the
    /// stand-in's from its stand-in part.
    depth_region: vk::BufferImageCopy,
    stand_in_region: vk::BufferImageCopy,
    scratch: vk::Buffer,
    _held: Buffer,
    kernel: OwnedKernel,
    /// One set per window, binding its words of the exact part and of the stand-in part.
    sets: OwnedSets,
    windows: Vec<Window>,
    /// A denormal's stand-in is the smallest normal.
    flush: bool,
    /// The pass's words are merged back after it.
    back: bool,
    /// The copy in's uses: a read of the depth, a write of the stand-in; the merge's: a read of
    /// the stand-in, a write of the depth.
    in_uses: Vec<Use<ResourceKey>>,
    back_uses: Vec<Use<ResourceKey>>,
}

/// One window's binding ranges: a buffer, an offset and bytes for each of the kernel's bindings.
type Bound = Vec<(vk::Buffer, u64, u64)>;

/// The kernel's windows over `words` words, and each one's two binding ranges, which `ranges`
/// gives for the window's first word and its bytes: within `maxStorageBufferRange` and
/// `maxComputeWorkGroupCount[0]` workgroups, and a whole number of alignments, so the next
/// window's bindings start at a multiple of it. Refused (internal) where no window holds one.
fn windows(
    device: &Device,
    words: u64,
    ranges: impl Fn(u64, u64) -> Bound,
) -> Result<(Vec<Window>, Vec<Bound>), Error> {
    let limits = &device.properties().description.limits;
    let local = u64::from(DepthMagnitudeModule::LOCAL_SIZE);
    let by_range = u64::from(limits.max_storage_buffer_range) / 4;
    let by_groups = u64::from(limits.max_compute_work_group_count[0]) * local;
    let most = by_range.min(by_groups) / (ALIGNMENT / 4) * (ALIGNMENT / 4);
    if most == 0 {
        return Err(internal("a magnitude stand-in no window holds"));
    }
    let mut windows = Vec::new();
    let mut bound = Vec::new();
    let mut start = 0;
    while start < words {
        let taken = most.min(words - start);
        bound.push(ranges(start, taken * 4));
        windows.push(Window {
            groups: u32::try_from(taken.div_ceil(local))
                .map_err(|_| internal("a magnitude window past u32 workgroups"))?,
            count: u32::try_from(taken)
                .map_err(|_| internal("a magnitude window past u32 words"))?,
        });
        start += taken;
    }
    Ok((windows, bound))
}

impl MagnitudeCopies {
    /// The copies of the pass rendering `own`, a depth, over `stand_in`, its magnitude stand-in,
    /// over the render area `[width, height]` of every layer of the pass's, on `device`: merged
    /// back where `back`. Refused (internal) where the area holds more words than a `u32` counts,
    /// and as the scratch buffer and the kernel are.
    pub(crate) fn plan(
        device: &Device,
        stand_in: &Target,
        own: &Target,
        [width, height]: [u32; 2],
        back: bool,
    ) -> Result<MagnitudeCopies, Error> {
        let layers = stand_in.layers.max(1);
        let words = u64::from(width) * u64::from(height) * u64::from(layers);
        let count =
            u32::try_from(words).map_err(|_| internal("a magnitude stand-in past u32 words"))?;
        if count == 0 {
            return Err(internal("a magnitude stand-in of no words"));
        }
        // The exact part, then the stand-in part from the next multiple of the alignment.
        let part = (words * 4).next_multiple_of(ALIGNMENT);
        let held = device.scratch_buffer(2 * part)?;
        let (scratch, base) = held.at(0);
        let (windows, ranges) = windows(device, words, |start, bytes| {
            vec![
                (scratch, base + start * 4, bytes),
                (scratch, base + part + start * 4, bytes),
            ]
        })?;
        let kernel = device.owned_kernel(OwnedModule::DepthMagnitude)?;
        let sets = OwnedSets::write(device, &kernel, &ranges)?;
        let region = |target: &Target, offset: u64| vk::BufferImageCopy {
            buffer_offset: offset,
            buffer_row_length: width,
            buffer_image_height: height,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                mip_level: target.level,
                base_array_layer: target.layer,
                layer_count: layers,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let level = |target: &Target| target.level..target.level + 1;
        let on = |target: &Target| target.layer..target.layer + layers;
        let depth = &own.texture;
        let standing = &stand_in.texture;
        let in_uses = vec![
            depth.image_use(level(own), on(own), read, false)?,
            standing.image_use(level(stand_in), on(stand_in), write, true)?,
        ];
        let back_uses = match back {
            true => vec![
                standing.image_use(level(stand_in), on(stand_in), read, false)?,
                depth.image_use(level(own), on(own), write, true)?,
            ],
            false => Vec::new(),
        };
        Ok(MagnitudeCopies {
            depth: depth.raw_image()?,
            stand_in: standing.raw_image()?,
            depth_region: region(own, base),
            stand_in_region: region(stand_in, base + part),
            scratch,
            _held: held,
            kernel,
            sets,
            windows,
            flush: test_flushes_denormals(device),
            back,
            in_uses,
            back_uses,
        })
    }

    /// Record the stand-in's writing in `core`'s command buffer, before the pass, between
    /// barriers from everything before and to everything after: the depth's words copied out,
    /// their stand-ins dispatched, and copied into the stand-in.
    pub(crate) fn record_in(&self, core: &mut EncoderCore) {
        let stages = vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER;
        core.barrier(Barrier::everything());
        core.command(stages, &self.in_uses, |device, raw| {
            // SAFETY: the encoder's command buffer is recording outside a render pass; the
            // images rest in `GENERAL` with transfer usage (`Texture::takes_magnitude_stand_in`,
            // and the stand-in made for copies); the regions lie in both images (the render area
            // of the pass's layers) and in the scratch's parts; the sets bind windows of those
            // parts, and the kernel's push range holds the shape; the encoder retains the copies
            // (their scratch and sets) and the textures until the command buffer completes.
            unsafe {
                device.cmd_copy_image_to_buffer(
                    raw,
                    self.depth,
                    vk::ImageLayout::GENERAL,
                    self.scratch,
                    &[self.depth_region],
                );
                Barrier::everything().record(device, raw);
                self.dispatch(device, raw, false);
                Barrier::everything().record(device, raw);
                device.cmd_copy_buffer_to_image(
                    raw,
                    self.scratch,
                    self.stand_in,
                    vk::ImageLayout::GENERAL,
                    &[self.stand_in_region],
                );
            }
        });
        core.barrier(Barrier::everything());
    }

    /// Record the merge back in `core`'s command buffer, after the pass, where it merges, between
    /// barriers from everything before and to everything after: the stand-in's words copied out,
    /// merged into the depth's kept from before the pass, and copied back.
    pub(crate) fn record_back(&self, core: &mut EncoderCore) {
        if !self.back {
            return;
        }
        let stages = vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER;
        core.barrier(Barrier::everything());
        core.command(stages, &self.back_uses, |device, raw| {
            // SAFETY: as in `record_in`; the exact part holds the depth's words from before the
            // pass, which wrote no other part of the scratch.
            unsafe {
                device.cmd_copy_image_to_buffer(
                    raw,
                    self.stand_in,
                    vk::ImageLayout::GENERAL,
                    self.scratch,
                    &[self.stand_in_region],
                );
                Barrier::everything().record(device, raw);
                self.dispatch(device, raw, true);
                Barrier::everything().record(device, raw);
                device.cmd_copy_buffer_to_image(
                    raw,
                    self.scratch,
                    self.depth,
                    vk::ImageLayout::GENERAL,
                    &[self.depth_region],
                );
            }
        });
        core.barrier(Barrier::everything());
    }

    /// The kernel's dispatches over every window, a merge where `merge`.
    ///
    /// # Safety
    ///
    /// `raw` is recording outside a render pass; the kernel, the sets and the scratch live until
    /// it completes.
    unsafe fn dispatch(&self, device: &ash::Device, raw: vk::CommandBuffer, merge: bool) {
        // SAFETY: the caller's contract; each set binds its window's words of both parts.
        unsafe {
            device.cmd_bind_pipeline(raw, vk::PipelineBindPoint::COMPUTE, self.kernel.pipeline);
            for (set, window) in self.sets.sets.iter().zip(&self.windows) {
                device.cmd_bind_descriptor_sets(
                    raw,
                    vk::PipelineBindPoint::COMPUTE,
                    self.kernel.layout,
                    0,
                    std::slice::from_ref(set),
                    &[],
                );
                let shape = DepthMagnitudeModule::shape(window.count, merge, self.flush);
                device.cmd_push_constants(
                    raw,
                    self.kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &shape,
                );
                device.cmd_dispatch(raw, window.groups, 1, 1);
            }
        }
    }
}

/// The strip after a pass that renders a switched-on depth in place, or through another image
/// copied back into it, and sets a depth/stencil state with depth writes whose stores the
/// device's pass leaves signed (the module docs): the depth's words over the render area stored
/// without their signs, as Metal stores them after such a pass where no fragment wrote.
pub(crate) struct SignStrip {
    route: Route,
    /// The depth's read and write by the strip, and its images' and draws' accesses.
    uses: Vec<Use<ResourceKey>>,
}

/// How a [`SignStrip`] moves the depth's words.
enum Route {
    /// A single-sample depth: copied out into a scratch buffer, stored without their signs by one
    /// dispatch of the kernel's strip per window, and copied back.
    Words {
        depth: vk::Image,
        region: vk::BufferImageCopy,
        scratch: vk::Buffer,
        _held: Buffer,
        kernel: OwnedKernel,
        /// One set per window, binding its words to both of the kernel's bindings (the strip
        /// reads no stand-in).
        sets: OwnedSets,
        windows: Vec<Window>,
    },
    /// A multisampled depth through maintenance8's depth/colour copies: its samples' bits copied
    /// into `bits`, a bits draw of each layer writing them without the sign bit into `stripped`,
    /// and `stripped` copied back into the depth. Exact.
    Colour {
        depth: vk::Image,
        /// The two colour images' handles, and the images, held.
        bits: vk::Image,
        stripped: vk::Image,
        _held: [Texture; 2],
        out: vk::ImageCopy,
        back: vk::ImageCopy,
        draws: Vec<DepthDraw>,
    },
    /// A multisampled depth by draws: copied into `scratch`, a sampled image of its format, and
    /// written back without its signs by a `FragDepth` draw of each layer, exact where the
    /// device's fragment depth stores every value unchanged (`depth_copy::drawn`).
    Draws {
        depth: vk::Image,
        /// The scratch's handle, and the scratch, held.
        scratch: vk::Image,
        _held: Texture,
        copy: vk::ImageCopy,
        draws: Vec<DepthDraw>,
    },
}

impl SignStrip {
    /// The strip of `own`, a depth, over the render area `[width, height]` of every layer of the
    /// pass's, made with `maker`'s device; where `own` is multisampled, the draws' changes, where
    /// the device's fragment depth is measured to make some, are announced
    /// (`drawn-depth-copy`). Refused (internal) where the area holds more words than a `u32`
    /// counts, and as the scratch buffer, the images, the kernel and the draws are.
    pub(crate) fn plan(
        maker: &CommandBuffer,
        own: &Target,
        extent: [u32; 2],
    ) -> Result<SignStrip, Error> {
        let device = maker.device();
        let [width, height] = extent;
        let layers = own.layers.max(1);
        let levels = own.level..own.level + 1;
        let on = own.layer..own.layer + layers;
        let depth = own.texture.raw_image()?;
        let read = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_READ);
        let write = Access::new(TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        let mut uses = vec![
            own.texture
                .image_use(levels.clone(), on.clone(), read, false)?,
            own.texture.image_use(levels, on, write, true)?,
        ];
        if own.samples <= 1 {
            let words = u64::from(width) * u64::from(height) * u64::from(layers);
            if words == 0 || u32::try_from(words).is_err() {
                return Err(internal("a sign strip of no words or past u32 words"));
            }
            let held = device.scratch_buffer((words * 4).next_multiple_of(ALIGNMENT))?;
            let (scratch, base) = held.at(0);
            let (windows, ranges) = windows(device, words, |start, bytes| {
                let at = (scratch, base + start * 4, bytes);
                vec![at, at]
            })?;
            let kernel = device.owned_kernel(OwnedModule::DepthMagnitude)?;
            let sets = OwnedSets::write(device, &kernel, &ranges)?;
            let region = vk::BufferImageCopy {
                buffer_offset: base,
                buffer_row_length: width,
                buffer_image_height: height,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    mip_level: own.level,
                    base_array_layer: own.layer,
                    layer_count: layers,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            };
            let route = Route::Words {
                depth,
                region,
                scratch,
                _held: held,
                kernel,
                sets,
                windows,
            };
            return Ok(SignStrip { route, uses });
        }
        let usage = own.texture.image_plan()?.usage;
        let sample_strip = sample_strip(device, own.format, usage)
            .ok_or_else(|| internal("a multisampled depth's sign strip the device does not run"))?;
        let area = DrawArea::whole(extent);
        let feature = Feature::SampleRateShading;
        let subresource = |target: &Target, aspect: ImageAspectFlags| vk::ImageSubresourceLayers {
            aspect_mask: aspect.into(),
            mip_level: target.level,
            base_array_layer: target.layer,
            layer_count: layers,
        };
        let copy = |from: &Target, from_aspect, to: &Target, to_aspect| vk::ImageCopy {
            src_subresource: subresource(from, from_aspect),
            src_offset: vk::Offset3D::default(),
            dst_subresource: subresource(to, to_aspect),
            dst_offset: vk::Offset3D::default(),
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        let whole = |texture: &Texture, format: Format| Target {
            texture: texture.clone(),
            level: 0,
            layer: 0,
            layers,
            format,
            samples: own.samples,
            extent,
        };
        let layer = |target: &Target, i: u32| Target {
            layer: target.layer + i,
            layers: 1,
            ..target.clone()
        };
        let (depth_aspect, colour) = (ImageAspectFlags::DEPTH, ImageAspectFlags::COLOR);
        let route = match sample_strip {
            SampleStrip::Colour => {
                let format = (Format::R32_UINT, PixelFormat::R32Uint);
                let bits = bits_image(maker, format, own.samples, extent, layers)?;
                let stripped = bits_image(maker, format, own.samples, extent, layers)?;
                let (from, into) = (whole(&bits, format.0), whole(&stripped, format.0));
                let mut draws = Vec::new();
                for i in 0..layers {
                    let (source, destination) = (layer(&from, i), layer(&into, i));
                    uses.push(stencil_copy::sampled_use(&source)?);
                    draws.push(DepthDraw::magnitude_bits(
                        device,
                        &source,
                        &destination,
                        area,
                        &feature,
                    )?);
                }
                let output = Access::new(
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                );
                uses.push(bits.image_use(0..1, 0..layers, write, true)?);
                uses.push(stripped.image_use(0..1, 0..layers, output, true)?);
                uses.push(stripped.image_use(0..1, 0..layers, read, false)?);
                Route::Colour {
                    depth,
                    out: copy(own, depth_aspect, &from, colour),
                    back: copy(&into, colour, own, depth_aspect),
                    bits: bits.raw_image()?,
                    stripped: stripped.raw_image()?,
                    _held: [bits, stripped],
                    draws,
                }
            }
            SampleStrip::Draws => {
                let scratch = sampled_depth(maker, own, extent, layers)?;
                let from = whole(&scratch, own.format);
                let mut draws = Vec::new();
                for i in 0..layers {
                    let (source, destination) = (layer(&from, i), layer(own, i));
                    uses.push(stencil_copy::sampled_use(&source)?);
                    uses.push(stencil_copy::attachment_use(&destination)?);
                    draws.push(DepthDraw::magnitude(
                        device,
                        &source,
                        &destination,
                        area,
                        &feature,
                    )?);
                }
                uses.push(scratch.image_use(0..1, 0..layers, write, true)?);
                if let Some(Drawn::Changes(changes)) = depth_copy::drawn(device, own.format) {
                    let format = own.texture.pixel_format();
                    depth_copy::announce_changes(device, changes, || {
                        format!(
                            "a multisampled {format:?} depth stored without its signs after a \
                             pass with depth writes"
                        )
                    });
                }
                Route::Draws {
                    depth,
                    copy: copy(own, depth_aspect, &from, depth_aspect),
                    scratch: scratch.raw_image()?,
                    _held: scratch,
                    draws,
                }
            }
        };
        Ok(SignStrip { route, uses })
    }

    /// Record the strip in `core`'s command buffer, after the pass, between barriers from
    /// everything before and to everything after, each step after a barrier from the one before.
    pub(crate) fn record(&self, core: &mut EncoderCore) {
        let stages = match &self.route {
            Route::Words { .. } => {
                vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER
            }
            Route::Colour { .. } | Route::Draws { .. } => {
                TRANSFER
                    | stencil_copy::DRAW_STAGES
                    | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
            }
        };
        core.barrier(Barrier::everything());
        core.command(stages, &self.uses, |device, raw| {
            let general = vk::ImageLayout::GENERAL;
            match &self.route {
                Route::Words {
                    depth,
                    region,
                    scratch,
                    kernel,
                    sets,
                    windows,
                    ..
                } => {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // the image rests in `GENERAL` with transfer usage
                    // (`Texture::magnitude_form`); the region lies in it (the render area of the
                    // pass's layers) and in the scratch; the sets bind windows of the scratch to
                    // both bindings, which the strip writes through the first alone; the
                    // kernel's push range holds the shape; the encoder retains the strip (its
                    // scratch and sets) and the texture until the command buffer completes.
                    unsafe {
                        device.cmd_copy_image_to_buffer(raw, *depth, general, *scratch, &[*region]);
                        Barrier::everything().record(device, raw);
                        device.cmd_bind_pipeline(
                            raw,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.pipeline,
                        );
                        for (set, window) in sets.sets.iter().zip(windows) {
                            device.cmd_bind_descriptor_sets(
                                raw,
                                vk::PipelineBindPoint::COMPUTE,
                                kernel.layout,
                                0,
                                std::slice::from_ref(set),
                                &[],
                            );
                            let shape = DepthMagnitudeModule::strip(window.count);
                            device.cmd_push_constants(
                                raw,
                                kernel.layout,
                                vk::ShaderStageFlags::COMPUTE,
                                0,
                                &shape,
                            );
                            device.cmd_dispatch(raw, window.groups, 1, 1);
                        }
                        Barrier::everything().record(device, raw);
                        device.cmd_copy_buffer_to_image(raw, *scratch, *depth, general, &[*region]);
                    }
                }
                Route::Colour {
                    depth,
                    bits,
                    stripped,
                    out,
                    back,
                    draws,
                    ..
                } => {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // every image rests in `GENERAL`; the copies are between a depth aspect and
                    // `R32_UINT` colour images of its samples, which maintenance8 lets
                    // `vkCmdCopyImage` copy (`sample_strip`), over the render area of the pass's
                    // layers, in each image; the draws read the bits and write the stripped
                    // image, one layer each; the encoder retains the strip (its images and the
                    // draws' objects) and the texture until the command buffer completes.
                    unsafe {
                        device.cmd_copy_image(raw, *depth, general, *bits, general, &[*out]);
                        for draw in draws {
                            Barrier::everything().record(device, raw);
                            draw.record(device, raw);
                        }
                        Barrier::everything().record(device, raw);
                        device.cmd_copy_image(raw, *stripped, general, *depth, general, &[*back]);
                    }
                }
                Route::Draws {
                    depth,
                    scratch,
                    copy,
                    draws,
                    ..
                } => {
                    // SAFETY: as above; the copy is between two images of the depth's format and
                    // samples, its depth aspect alone; the draws read the scratch's and write the
                    // depth's, one layer each.
                    unsafe {
                        device.cmd_copy_image(raw, *depth, general, *scratch, general, &[*copy]);
                        for draw in draws {
                            Barrier::everything().record(device, raw);
                            draw.record(device, raw);
                        }
                    }
                }
            }
        });
        core.barrier(Barrier::everything());
    }
}

/// A multisampled image of `own`'s depth format and samples, `extent` and `layers`, sampled,
/// which a multisampled depth's strip by draws copies the depth into and draws back from: the
/// encoder's own. Refused where the device does not make it.
fn sampled_depth(
    maker: &CommandBuffer,
    own: &Target,
    [width, height]: [u32; 2],
    layers: u32,
) -> Result<Texture, Error> {
    let samples = sample_count(own.samples)
        .ok_or_else(|| internal("a multisampled depth of a sample count Vulkan lacks"))?;
    let pixel_format = own.texture.pixel_format();
    let descriptor = TextureDescriptor {
        texture_type: if layers > 1 {
            TextureType::Type2DMultisampleArray
        } else {
            TextureType::Type2DMultisample
        },
        pixel_format,
        width: width.into(),
        height: height.into(),
        array_length: layers.into(),
        sample_count: own.samples.into(),
        usage: TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
        resource_options: ResourceOptions::private(),
        ..TextureDescriptor::default()
    };
    let plan = ImagePlan {
        pixel_format,
        flags: ImageCreateFlags::empty(),
        image_type: ImageType::TYPE_2D,
        format: own.format,
        extent: [width, height, 1],
        levels: 1,
        layers,
        samples,
        usage: ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | ImageUsageFlags::SAMPLED
            | ImageUsageFlags::TRANSFER_DST,
        view_formats: Vec::new(),
        mapping: FormatMapping::direct(
            own.format,
            ImageAspectFlags::DEPTH | ImageAspectFlags::STENCIL,
        ),
        unavailable: TextureUsage::SHADER_WRITE
            | TextureUsage::SHADER_ATOMIC
            | TextureUsage::PIXEL_FORMAT_VIEW,
    };
    Device::new_planned_texture(maker, &descriptor, plan)
}

/// Announce `signed-depth-test` for a pass after which `own`, a depth a write switched on, keeps
/// the signs of the words it did not write, where Metal stores them without: the device makes
/// no strip for it (`error`).
pub(crate) fn degrade_strip(device: &Device, own: &Target, error: &Error) {
    device.degrade(Degradation::SignedDepthTest, || {
        let format = own.texture.pixel_format();
        let why = format!(
            "a pass with depth writes leaves a depth that may hold a negative or NaN value with \
             its signs, where Metal stores it without them: the device makes no strip for it \
             ({error})"
        );
        (format!("a {format:?} texture"), why)
    });
}
