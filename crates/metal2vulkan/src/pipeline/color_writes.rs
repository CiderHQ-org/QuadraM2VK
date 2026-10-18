//! How a fragment function's float color outputs reach the attachments whose formats convert
//! them (`LoweringInputs::color_writes`).
//!
//! Metal stores a float written to a normalized attachment as the exact product with the
//! format's maximum code rounded to nearest, to an 8-bit sRGB one through its twelve-bit encode,
//! and to a half-float one rounded toward zero (`metal::format::write_conversion`, probes on an
//! Apple M3 Max and an M4). Vulkan's drivers round the `f32` product with ties to even, encode sRGB
//! by their own approximations, and round a half to nearest, so near every tie they store another
//! code (`runtime_lane::color_writes`: ANV, RADV, NVIDIA and lavapipe); only MoltenVK on Apple's
//! GPUs stores Metal's (`VkProperty::MetalColorWrites`). Everywhere else the fragment function
//! rounds each such output by Metal's rule itself and writes the `f32` nearest the code's exact
//! value, which every measured driver stores as that code. On MoltenVK the map keeps the half-float
//! channels alone, which round nothing and record the device's NaN encoding of a `half` output.
//!
//! Under `TextureWriteRoundingMode::NearestEven` (Apple's `textureWriteRoundingMode` 2) Metal
//! rounds a render state's half-float channels to nearest even ([`HALF_NEAREST`]), which no device
//! does for a conversion it stores after a blend or as Metal's default: the function rounds each
//! such channel itself on every profile, MoltenVK's included, where the channel stores the output
//! as written.
//!
//! The rounding is exact only where nothing after the shader changes the value: an attachment
//! that blends with factors other than one and zero keeps the device's conversion
//! ([`ChannelWrite::Device`], recorded as `PrecisionNote::DeviceColorConversion`), and so does
//! the alpha of attachment 0 under alpha-to-coverage, which the coverage reads as computed. A
//! format whose conversion Metal was not measured with keeps the device's too, and so does a
//! 32-bit float attachment that blends so, whose stored value is the device's blend arithmetic.
//!
//! `Device::compile_render_pipeline` derives the map for a `Function` fragment stage from every
//! color attachment of the pipeline's descriptor ([`color_writes`]); the fragment function
//! records the entries of the attachments it writes (`StageIo::color_writes`), and link derives
//! them again from the pipeline it links into ([`check`]): a fragment compiled for another map,
//! a standalone one compiled without the descriptor included, is `NeedsInput(ColorWrites)`.

use std::collections::{BTreeMap, BTreeSet};

use crate::device::requirements::VkProperty;
use crate::device::DeviceProfile;
use crate::error::{Error, RuntimeInput};
use crate::library::FunctionArtifact;
use crate::metal;
use crate::metal::format::{color_output_components, write_conversion, WriteConversion};
use crate::metal::TextureWriteRoundingMode;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::graphics::ColorAttachmentState;
use crate::pipeline::lower_inputs::{
    ChannelWrite, ExtendedRangeColor, FactorLane, HalfOutputLane, RotatedBlendForm,
};
use crate::pipeline::rotated_blend::{rotated_blend, rotated_form, RotatedBlend};
use crate::pipeline::stage_io::{IoRole, SelectedWrite};

/// The color writes of a fragment function writing the attachments `written` (dual-source index
/// 0) of a pipeline whose color attachments are `colors` (alpha-to-coverage on where
/// `alpha_to_coverage`, its half-float channels rounding by `rounding`) on profile `p` (the
/// module docs). Where the device stores Metal's codes itself (`VkProperty::MetalColorWrites`)
/// the map keeps only the half-float channels ([`metal_stored`]), whose `half` NaN the
/// function's `float` output does not carry.
///
/// Elsewhere an attachment each of whose channels takes a selectable rule ([`code`]: none, the
/// device's conversion, an 8-bit unorm channel's, a half-float channel's), one of them a rounding
/// (an 8-bit unorm or a half-float channel's), takes [`ChannelWrite::Selected`] on every channel
/// ([`selects`]): the function computes each such rule and the draw selects one by the attachment's
/// rule word ([`rule_words`]), so the module is the same for every such format and blend.
/// [`exact_writes`] is the map before that.
pub(crate) fn color_writes(
    written: impl IntoIterator<Item = u32>,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> BTreeMap<u32, [Option<ChannelWrite>; 4]> {
    let mut out = exact_writes(written, colors, rounding, p);
    if !metal_color_writes(p) {
        for rules in out.values_mut() {
            if selects(rules) {
                *rules = [Some(ChannelWrite::Selected); 4];
            }
        }
    }
    out
}

/// Whether profile `p`'s device stores Metal's codes itself (`VkProperty::MetalColorWrites`).
fn metal_color_writes(p: &DeviceProfile) -> bool {
    p.fields()
        .enabled
        .properties
        .contains(&VkProperty::MetalColorWrites)
}

/// The color writes [`color_writes`] derives before an attachment takes a selected rule: each
/// channel's own rule.
pub(crate) fn exact_writes(
    written: impl IntoIterator<Item = u32>,
    colors: &[Option<ColorAttachmentDescriptor>],
    (alpha_to_coverage, rounding): (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> BTreeMap<u32, [Option<ChannelWrite>; 4]> {
    let metal = metal_color_writes(p);
    let mut out = BTreeMap::new();
    for attachment in written {
        let Some(c) = colors.get(attachment as usize).copied().flatten() else {
            continue;
        };
        // A word attachment's output is the word the function encodes (`word_colors`).
        if word_color(attachment, &c, alpha_to_coverage, p).is_some() {
            continue;
        }
        let coverage_alpha = attachment == 0 && alpha_to_coverage;
        let write = match metal {
            true => metal_stored(&c),
            false => attachment_write(&c, coverage_alpha),
        };
        let write = match rounding {
            TextureWriteRoundingMode::NearestEven => {
                write.map(|w| nearest_even(w, &c, coverage_alpha))
            }
            _ => write,
        };
        if let Some(write) = write {
            out.insert(attachment, write);
        }
    }
    out
}

/// A rule word's code of a channel that takes no rule: the output as written.
pub(crate) const NO_RULE_CODE: u32 = 0;

/// A rule word's code of a channel the device converts ([`ChannelWrite::Device`]): the output as
/// written, recorded as `PrecisionNote::DeviceColorConversion`.
pub(crate) const DEVICE_CODE: u32 = 1;

/// A rule word's code of an 8-bit unorm channel ([`ChannelWrite::Unorm`] of 8 bits).
pub(crate) const UNORM8_CODE: u32 = 2;

/// A rule word's code of a half-float channel ([`HALF`]).
pub(crate) const HALF_CODE: u32 = 3;

/// The bits of each channel's code in an attachment's rule word: channel `i`'s code is
/// `(word >> (RULE_BITS * i)) & RULE_MASK`.
pub(crate) const RULE_BITS: u32 = 4;

/// One channel's code in a rule word, shifted down.
pub(crate) const RULE_MASK: u32 = (1 << RULE_BITS) - 1;

/// The code of a channel that takes `rule` in an attachment's rule word, or `None` where `rule`
/// is no selectable one.
fn code(rule: Option<ChannelWrite>) -> Option<u32> {
    match rule {
        None => Some(NO_RULE_CODE),
        Some(ChannelWrite::Device) => Some(DEVICE_CODE),
        Some(ChannelWrite::Unorm { bits: 8 }) => Some(UNORM8_CODE),
        Some(rule) if rule == HALF => Some(HALF_CODE),
        Some(_) => None,
    }
}

/// The rule word of an attachment whose channels take `rules` ([`code`] of channel `i` at bits
/// `RULE_BITS * i`), or `None` where a channel takes a rule no code selects.
pub(crate) fn rule_word(rules: &[Option<ChannelWrite>; 4]) -> Option<u32> {
    rules
        .iter()
        .zip(0u32..)
        .try_fold(NO_RULE_CODE, |word, (&rule, i)| {
            Some(word | code(rule)? << (RULE_BITS * i))
        })
}

/// Whether an attachment whose channels take `rules` selects its rules at the draw
/// ([`color_writes`]): each channel's rule has a code ([`rule_word`]), and one of them rounds
/// (`UNORM8_CODE` or `HALF_CODE`). One that only keeps the device's conversion rounds nothing,
/// and keeps its module free of the rounding.
fn selects(rules: &[Option<ChannelWrite>; 4]) -> bool {
    let rounds = |&r: &Option<ChannelWrite>| matches!(code(r), Some(UNORM8_CODE | HALF_CODE));
    rule_word(rules).is_some() && rules.iter().any(rounds)
}

/// The codes the lanes `sel` writes take under rule word `word`.
fn lane_codes(sel: &SelectedWrite, word: u32) -> impl Iterator<Item = u32> {
    (0..u32::from(sel.lanes.min(4))).map(move |i| (word >> (RULE_BITS * i)) & RULE_MASK)
}

/// Whether the write `sel` of an attachment of rule word `word` rounds a lane to Metal's code in
/// the shader, as an attachment the function lists in `StageIo::rounded_colors` does.
pub(crate) fn selected_rounds(sel: &SelectedWrite, word: u32) -> bool {
    lane_codes(sel, word).any(|c| sel.rounds & (1 << c) != 0)
}

/// The precision notes the write `sel` of an attachment of rule word `word` keeps: the device's
/// conversion of a channel it converts (`PrecisionNote::DeviceColorConversion`), and its encoding
/// of a NaN where a lane's code stores one so (`PrecisionNote::DeviceNanEncoding`).
pub(crate) fn selected_notes(sel: &SelectedWrite, word: u32) -> BTreeSet<PrecisionNote> {
    let mut notes = BTreeSet::new();
    for c in lane_codes(sel, word) {
        if c == DEVICE_CODE {
            notes.insert(PrecisionNote::DeviceColorConversion);
        }
        if sel.nans & (1 << c) != 0 {
            notes.insert(PrecisionNote::DeviceNanEncoding);
        }
    }
    notes
}

/// The precision notes and the attachments it rounds in the shader of fragment function `a` in
/// a render pipeline whose rule words are `rules` ([`rule_words`]) and whose color attachments
/// are `colors` (`GraphicsState::color_attachments`): the function's own
/// (`FunctionArtifact::precision`, `StageIo::rounded_colors`), for each attachment whose
/// rules it selects (`StageIo::selected_colors`) those its rule word selects
/// ([`selected_notes`], [`selected_rounds`]), `PrecisionNote::HalfOutputPrecision` where an
/// attachment of `StageIo::half_blend_colors` blends ([`blends_output`]), and
/// `PrecisionNote::HalfOutputFastMathBlend` where one of `StageIo::half_fast_colors` stores some
/// channel ([`stores_output`]). Each sorted, each entry once. `None` where such an attachment has
/// no rule word.
pub(crate) fn pipeline_writes(
    a: &FunctionArtifact,
    rules: &[(u32, u32)],
    colors: &[Option<ColorAttachmentState>],
) -> Option<(Vec<PrecisionNote>, Vec<u32>)> {
    let mut notes = a.precision.clone();
    let mut rounded = a.interface.io.rounded_colors.clone();
    fn insert<T: Ord>(v: &mut Vec<T>, x: T) {
        if let Err(at) = v.binary_search(&x) {
            v.insert(at, x);
        }
    }
    let attachment = |c: &u32| {
        usize::try_from(*c)
            .ok()
            .and_then(|i| colors.get(i))
            .and_then(Option::as_ref)
    };
    let blended = |c: &u32| attachment(c).is_some_and(blends_output);
    if a.interface.io.half_blend_colors.iter().any(blended) {
        insert(&mut notes, PrecisionNote::HalfOutputPrecision);
    }
    // A `float` output of a module without fast math into a 16-bit float attachment's blend of one
    // and zero, which the pipeline blends ([`keeps_zero`]); its other blends record the device's
    // conversion already.
    let one_zero = |c: &u32| attachment(c).is_some_and(|s| blends_output(s) && one_zero_half(s));
    if a.interface.io.float_blend_colors.iter().any(one_zero) {
        insert(&mut notes, PrecisionNote::HalfOutputPrecision);
    }
    let stored = |c: &u32| attachment(c).is_some_and(stores_output);
    if a.interface.io.half_fast_colors.iter().any(stored) {
        insert(&mut notes, PrecisionNote::HalfOutputFastMathBlend);
    }
    for (&color, sel) in &a.interface.io.selected_colors {
        let &(_, word) = rules.iter().find(|(c, _)| *c == color)?;
        if selected_rounds(sel, word) {
            insert(&mut rounded, color);
        }
        for note in selected_notes(sel, word) {
            insert(&mut notes, note);
        }
    }
    Some((notes, rounded))
}

/// The rule word of each attachment the fragment stage of interface `io` selects its rules for
/// (`StageIo::selected_colors`) in a render pipeline whose color attachments are `colors`
/// (alpha-to-coverage on where `alpha_to_coverage`, rounding by `rounding`) on `p`: its channels'
/// codes ([`rule_word`] of [`exact_writes`]), 0 where the pipeline converts nothing there. Sorted
/// by attachment. `NeedsInput(ColorWrites)` where a channel takes a rule no code selects, which
/// [`check`] refuses alike.
pub(crate) fn rule_words(
    io: &crate::pipeline::stage_io::StageIo,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> Result<Vec<(u32, u32)>, Error> {
    let exact = exact_writes(io.selected_colors.keys().copied(), colors, rounding, p);
    io.selected_colors
        .keys()
        .map(|&a| {
            let word = match exact.get(&a) {
                Some(rules) => rule_word(rules),
                None => Some(NO_RULE_CODE),
            };
            word.map(|w| (a, w))
                .ok_or(Error::NeedsInput(RuntimeInput::ColorWrites))
        })
        .collect()
}

/// A half-float channel's rule ([`ChannelWrite::FloatTowardZero`] of ten fraction bits, signed).
pub(crate) const HALF: ChannelWrite = ChannelWrite::FloatTowardZero {
    mantissa: 10,
    signed: true,
};

/// A half-float channel's rule under `TextureWriteRoundingMode::NearestEven`
/// ([`ChannelWrite::FloatNearestEven`] of ten fraction bits, signed).
pub(crate) const HALF_NEAREST: ChannelWrite = ChannelWrite::FloatNearestEven {
    mantissa: 10,
    signed: true,
};

/// Whether `rule` is a half-float channel's ([`HALF`] or [`HALF_NEAREST`]).
pub(crate) fn half_float(rule: ChannelWrite) -> bool {
    rule == HALF || rule == HALF_NEAREST
}

/// A half-float channel's rule under `TextureWriteRoundingMode::NearestEven` after a blend of
/// factors other than one and zero ([`ChannelWrite::BlendedNearestEven`]).
pub(crate) const HALF_BLENDED: ChannelWrite = ChannelWrite::BlendedNearestEven {
    mantissa: 10,
    signed: true,
};

/// The channels `rules` of attachment `c` (alpha-to-coverage reading its alpha where
/// `coverage_alpha`) under `TextureWriteRoundingMode::NearestEven`: each channel the format
/// rounds as a half ([`HALF`], whatever the default map made of it) rounded to nearest even
/// ([`HALF_NEAREST`]) where it stores the output as written (no blend, or one of factors one and
/// zero), Metal's rounding of the blend's result ([`HALF_BLENDED`], which the device's blend
/// makes) after a blend of other factors, and the device's ([`ChannelWrite::Device`]) for the
/// alpha alpha-to-coverage reads.
fn nearest_even(
    mut rules: [Option<ChannelWrite>; 4],
    c: &ColorAttachmentDescriptor,
    coverage_alpha: bool,
) -> [Option<ChannelWrite>; 4] {
    let Some(WriteConversion::Rules(format)) = write_conversion(c.pixel_format) else {
        return rules;
    };
    let replaced = !c.blending_enabled || replaces(c);
    for (i, (rule, own)) in rules.iter_mut().zip(format).enumerate() {
        if own != Some(HALF) || rule.is_none() {
            continue;
        }
        *rule = Some(match (coverage_alpha && i == 3, replaced) {
            (true, _) => ChannelWrite::Device,
            (false, true) => HALF_NEAREST,
            (false, false) => HALF_BLENDED,
        });
    }
    rules
}

/// The channels of attachment `c` on a device that stores a float output as Metal does
/// (`VkProperty::MetalColorWrites`): its half-float channels, whatever its blend, and nothing
/// where it has none or writes none. The function rounds nothing there; a `half` output it writes
/// to such a channel is held widened to `float`, and the device stores that `float`'s NaN as Metal
/// stores a `float` NaN, not as the `half`'s own bits, so the function records
/// `PrecisionNote::DeviceNanEncoding` for it.
fn metal_stored(c: &ColorAttachmentDescriptor) -> Option<[Option<ChannelWrite>; 4]> {
    if c.write_mask.is_empty() {
        return None;
    }
    let WriteConversion::Rules(rules) = write_conversion(c.pixel_format)? else {
        return None;
    };
    let halves = rules.map(|r| r.filter(|&r| r == HALF));
    halves.iter().any(Option::is_some).then_some(halves)
}

/// Check the fragment stage `fragment` of a render pipeline whose color attachments are
/// `colors` (alpha-to-coverage on where `alpha_to_coverage`, rounding by `rounding`) on `p`: the
/// color writes it compiled under are the pipeline's for the attachments it writes, selected or
/// each channel's own ([`writes_hold`]), and so are its extended-range attachments' writes
/// ([`wanted_extended_range`]), and every color it stores nowhere (`StageIo::unattached_colors`)
/// has no attachment and is not color 0 under alpha to coverage, and the second outputs it writes
/// for rotated attachments are the pipeline's ([`wanted_rotated_factors`]), and so are the forms of
/// their blends ([`wanted_rotated_blends`]), and the attachments holding Metal's alpha in red it
/// writes or reads are the pipeline's ([`wanted_alpha_in_red`]), and the pipeline takes the `half` outputs it computes
/// at `float` precision as it compiled them ([`half_output_lanes_hold`]), and its texture writes
/// that Metal's mode 2 stores otherwise compiled under the pipeline's `textureWriteRoundingMode`
/// (`StageIo::texture_write_rounding`), else `NeedsInput(ColorWrites)`.
pub(crate) fn check(
    fragment: Option<&FunctionArtifact>,
    colors: &[Option<ColorAttachmentDescriptor>],
    (alpha_to_coverage, rounding): (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> Result<(), Error> {
    let Some(fragment) = fragment else {
        return Ok(());
    };
    let io = &fragment.interface.io;
    let unattached = io.unattached_colors.iter().all(|&c| {
        colors.get(c as usize).copied().flatten().is_none() && !(alpha_to_coverage && c == 0)
    });
    if writes_hold(fragment, colors, (alpha_to_coverage, rounding), p)
        && wanted_extended_range(fragment, colors, rounding) == io.extended_range_colors
        && wanted_word_colors(fragment, colors, alpha_to_coverage, p) == io.word_colors
        && wanted_srgb_fetches(fragment, colors) == io.srgb_fetches
        && wanted_rotated(fragment, colors, p) == io.rotated_colors
        && wanted_alpha_in_red(fragment, colors, p) == io.alpha_in_red_colors
        && wanted_rotated_factors(fragment, colors, p) == io.rotated_factors
        && wanted_rotated_blends(fragment, colors, p) == io.rotated_blends
        && half_output_lanes_hold(fragment, colors)
        && io.texture_write_rounding.is_none_or(|m| m == rounding)
        && unattached
    {
        Ok(())
    } else {
        Err(Error::NeedsInput(RuntimeInput::ColorWrites))
    }
}

/// Whether a fetched color attachment of `pf` is read through Metal's decode of its bytes
/// (`LoweringInputs::srgb_fetches`): `RGBA8Unorm_sRGB` and `BGRA8Unorm_sRGB`, whose fetch our
/// probes measured. The one- and two-channel sRGB formats' are not.
pub fn srgb_fetched(pf: metal::PixelFormat) -> bool {
    matches!(
        pf,
        metal::PixelFormat::RGBA8UnormSRGB | metal::PixelFormat::BGRA8UnormSRGB
    )
}

/// The input attachments `fragment` reads whose pixel format in `colors` takes Metal's decode
/// ([`srgb_fetched`]): the set the function must have compiled under
/// (`StageIo::srgb_fetches`).
pub(crate) fn wanted_srgb_fetches(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
) -> BTreeSet<u32> {
    fragment
        .interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            crate::pipeline::sym::ResourceKind::InputAttachment(color) => Some(color),
            _ => None,
        })
        .filter(|&c| {
            colors
                .get(c as usize)
                .copied()
                .flatten()
                .is_some_and(|a| srgb_fetched(a.pixel_format))
        })
        .collect()
}

/// The extended-range color attachments of a render pipeline whose color attachments are
/// `colors` and whose `textureWriteRoundingMode` is `rounding`, by color index
/// (`LoweringInputs::extended_range_colors`): every attachment of one of Apple's extended-range
/// formats, blended where the pipeline enables blending there, a bfloat one rounding by the mode
/// (`ExtendedRangeColor::nearest_even`).
pub(crate) fn extended_range_colors(
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: TextureWriteRoundingMode,
) -> BTreeMap<u32, ExtendedRangeColor> {
    colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let c = c.as_ref()?;
            let form = metal::format::ExtendedRange::of(c.pixel_format)?;
            let write = ExtendedRangeColor::new(form, c.blending_enabled).with_rounding(rounding);
            Some((i as u32, write))
        })
        .collect()
}

/// The word view through which profile `p`'s device renders color attachment `index` of a
/// render pipeline, described by `c`, that covers by alpha where `alpha_to_coverage`
/// (`LoweringInputs::word_colors`): `metal::format::WordAttachment::of` the attachment's format,
/// where the pipeline writes it unblended (or blended with factors one and zero, added, which
/// Metal stores alike), under a write mask of all of the format's channels or none, or of some of
/// them where the function merges the word into the texel over its input attachment
/// ([`merged_word`]), and it is not attachment 0 under alpha-to-coverage (whose coverage reads the
/// output's alpha as computed). `None` for every other attachment: one the function stores in the
/// interlocked storage form ([`stored_word`]), or one that takes its format's own attachment, as
/// the device has it or not.
pub(crate) fn word_color(
    index: u32,
    c: &ColorAttachmentDescriptor,
    alpha_to_coverage: bool,
    p: &DeviceProfile,
) -> Option<metal::format::WordAttachment> {
    let word = metal::format::WordAttachment::of(c.pixel_format, p)?;
    let kept = c.write_mask & word_channels(word);
    let whole = kept.is_empty() || kept == word_channels(word);
    let merged = merges_over_input(c, word, p);
    let unblended = !c.blending_enabled || replaces(c);
    (unblended && (whole || merged) && !(index == 0 && alpha_to_coverage)).then_some(word)
}

/// The channels of `word`'s fields: Metal's red, green, blue and alpha, in its field order.
fn word_channels(word: metal::format::WordAttachment) -> metal::ColorWriteMask {
    use metal::ColorWriteMask as M;
    [M::RED, M::GREEN, M::BLUE, M::ALPHA]
        .into_iter()
        .zip(word.fields())
        .filter(|(_, f)| f.is_some())
        .fold(M::NONE, |m, (bit, _)| m | bit)
}

/// Whether the fragment function merges its word of attachment `c` (a `word` attachment) into the
/// texel over the attachment's input attachment: a write mask of some of the word's channels, on a
/// device that orders framebuffer fetch through rasterization-order input attachments
/// (`lower::fragment`'s `word_input`), for a format whose mask the function compiles as the
/// descriptor sets it (`lower::fragment::fetch::kept_at_draw` false: the 8-bit sRGB words, where a
/// packed 16-bit or `RGB9E5Float` word's mask stands for its class).
fn merges_over_input(
    c: &ColorAttachmentDescriptor,
    word: metal::format::WordAttachment,
    p: &DeviceProfile,
) -> bool {
    let channels = word_channels(word);
    let kept = c.write_mask & channels;
    let partial = !kept.is_empty() && kept != channels;
    partial
        && !crate::lower::fragment::fetch::kept_at_draw(c.pixel_format)
        && crate::lower::fragment::inputs_order(p)
}

/// The channels the fragment function keeps from the texel of color attachment `index` of a
/// render pipeline, described by `c`, that covers by alpha where `alpha_to_coverage`, where it
/// merges its word into the texel over the attachment's input attachment ([`word_color`],
/// `lower::fragment`'s `word_input`): those of the word's channels the write mask leaves out.
/// `None` for every other attachment.
pub(crate) fn merged_word(
    index: u32,
    c: &ColorAttachmentDescriptor,
    alpha_to_coverage: bool,
    p: &DeviceProfile,
) -> Option<metal::ColorWriteMask> {
    let word = word_color(index, c, alpha_to_coverage, p)?;
    merges_over_input(c, word, p).then(|| word_channels(word).difference(c.write_mask))
}

/// The word view of color attachment `c` of a render pipeline on profile `p` whose writes the
/// fragment function stores in the interlocked storage form (`lower::fragment::fetch`), reading
/// the word first inside the interlock: a packed 16-bit word attachment
/// (`metal::format::WordAttachment::of`) written under a blend other than one and zero, added,
/// whose factors read no second source (Metal's packed 16-bit blend, which the function computes
/// over the word, `extended_range::Blend::packed16`), or a word attachment other than
/// `RGB9E5Float`'s written unblended under a write mask of some of the format's channels (the
/// function merges the fields of the channels the mask keeps), on a device that orders fetch
/// through the pixel interlock alone.
/// `None` for every other attachment, and for a write mask of none of the format's channels.
pub(crate) fn stored_word(
    c: &ColorAttachmentDescriptor,
    p: &DeviceProfile,
) -> Option<metal::format::WordAttachment> {
    use metal::{BlendFactor as F, ColorWriteMask as M};
    let word = metal::format::WordAttachment::of(c.pixel_format, p)?;
    let channels = [M::RED, M::GREEN, M::BLUE, M::ALPHA]
        .into_iter()
        .zip(word.fields())
        .filter(|(_, f)| f.is_some())
        .fold(M::NONE, |m, (bit, _)| m | bit);
    let kept = c.write_mask & channels;
    let partial = !kept.is_empty() && kept != channels;
    let unblended = !c.blending_enabled || replaces(c);
    let second = [
        c.source_rgb_blend_factor,
        c.destination_rgb_blend_factor,
        c.source_alpha_blend_factor,
        c.destination_alpha_blend_factor,
    ]
    .iter()
    .any(|f| {
        matches!(
            f,
            F::Source1Color | F::OneMinusSource1Color | F::Source1Alpha | F::OneMinusSource1Alpha
        )
    });
    // Metal's blend of a packed 16-bit word is built; an 8-bit sRGB word's is not, nor is an
    // `RGB9E5Float` word's blend or partial mask (its mantissas share the exponent).
    let packed = matches!(word, metal::format::WordAttachment::Packed16 { .. });
    let fielded = word != metal::format::WordAttachment::Rgb9e5;
    let stored = match unblended {
        true => partial && fielded,
        false => packed && !kept.is_empty() && !second,
    };
    (stored && crate::lower::fragment::interlock_orders(p)).then_some(word)
}

/// The word attachments of a render pipeline whose color attachments are `colors` (covering by
/// alpha where `alpha_to_coverage`) on `p`, by color index ([`word_color`]).
pub(crate) fn word_colors(
    colors: &[Option<ColorAttachmentDescriptor>],
    alpha_to_coverage: bool,
    p: &DeviceProfile,
) -> BTreeMap<u32, metal::format::WordAttachment> {
    colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let word = word_color(i as u32, c.as_ref()?, alpha_to_coverage, p)?;
            Some((i as u32, word))
        })
        .collect()
}

/// The word attachments the fragment stage `fragment` writes through output variables of
/// dual-source index 0 in a render pipeline whose color attachments are `colors` (covering by
/// alpha where `alpha_to_coverage`) on `p`: [`word_colors`]'s, for those attachments.
pub(crate) fn wanted_word_colors(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    alpha_to_coverage: bool,
    p: &DeviceProfile,
) -> BTreeMap<u32, metal::format::WordAttachment> {
    let written: BTreeSet<u32> = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        })
        .collect();
    let mut out = word_colors(colors, alpha_to_coverage, p);
    out.retain(|a, _| written.contains(a));
    out
}

/// The color attachments of a render pipeline whose color attachments are `colors` held with
/// their channels rotated on `p`, by color index (`LoweringInputs::rotated_colors`): every
/// attachment of a format `metal::format::rotated_channels` names (`ARGB8Unorm`), and every one
/// whose image holds Metal's alpha in red (`metal::format::alpha_in_red`: `A8Unorm` without
/// `A8_UNORM`), which the rotation's red is.
pub(crate) fn rotated_colors(
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    colors
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            c.is_some_and(|c| {
                metal::format::rotated_channels(c.pixel_format)
                    || metal::format::alpha_in_red(c.pixel_format, p)
            })
        })
        .map(|(i, _)| i as u32)
        .collect()
}

/// The rotated attachments the fragment stage `fragment` writes in a render pipeline whose
/// color attachments are `colors`: [`rotated_colors`]'s, for the attachments it writes through
/// output variables of either dual-source index.
pub(crate) fn wanted_rotated(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    let written: BTreeSet<u32> = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color { attachment, .. } => Some(attachment),
            _ => None,
        })
        .collect();
    let mut out = rotated_colors(colors, p);
    out.retain(|a| written.contains(a));
    out
}

/// The color attachments of a render pipeline whose color attachments are `colors` whose image
/// holds Metal's alpha in red on `p` (`metal::format::alpha_in_red`: `A8Unorm` without
/// `A8_UNORM`), by color index (`LoweringInputs::alpha_in_red_colors`).
pub(crate) fn alpha_in_red_colors(
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    colors
        .iter()
        .enumerate()
        .filter(|(_, c)| c.is_some_and(|c| metal::format::alpha_in_red(c.pixel_format, p)))
        .map(|(i, _)| i as u32)
        .collect()
}

/// The attachments holding Metal's alpha in red that the fragment stage `fragment` writes or reads
/// in a render pipeline whose color attachments are `colors` on `p`: [`alpha_in_red_colors`]'s,
/// for the attachments it writes through output variables of either dual-source index, reads
/// through an input attachment, or reads in the interlocked storage form
/// (`RuntimeFact::InterlockedFramebufferFetch`), the set it must have compiled under
/// (`StageIo::alpha_in_red_colors`).
pub(crate) fn wanted_alpha_in_red(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeSet<u32> {
    let written = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color { attachment, .. } => Some(attachment),
            _ => None,
        });
    let inputs = fragment
        .interface
        .descriptors
        .iter()
        .filter_map(|d| match d.slot.kind {
            crate::pipeline::sym::ResourceKind::InputAttachment(color) => Some(color),
            _ => None,
        });
    let interlocked = fragment.facts.iter().filter_map(|f| match *f {
        RuntimeFact::InterlockedFramebufferFetch { color, .. } => Some(color),
        _ => None,
    });
    let used: BTreeSet<u32> = written.chain(inputs).chain(interlocked).collect();
    let mut out = alpha_in_red_colors(colors, p);
    out.retain(|a| used.contains(a));
    out
}

/// The channels of the color attachments `colors` that take a fragment function's `half` output
/// at `float` precision where Metal computes its last operation so, by color index
/// (`LoweringInputs::half_output_lanes`): each attachment of a 32-bit float format whose every
/// stored channel either does not blend ([`HalfOutputLane::Float32`]) or blends with factors one
/// and one under `Add` ([`HalfOutputLane::Float32Added`]); a channel the format lacks or the write
/// mask drops is `None`.
pub(crate) fn half_output_lanes(
    colors: &[Option<ColorAttachmentDescriptor>],
) -> BTreeMap<u32, [Option<HalfOutputLane>; 4]> {
    use metal::{BlendFactor as F, BlendOperation as O, ColorWriteMask as M, PixelFormat as P};
    let mut out = BTreeMap::new();
    for (i, c) in colors.iter().enumerate() {
        let Some(c) = c else { continue };
        let channels = match c.pixel_format {
            P::R32Float => 1,
            P::RG32Float => 2,
            P::RGBA32Float => 4,
            _ => continue,
        };
        let mut lanes = [None; 4];
        let mut every = true;
        for (k, (lane, bit)) in lanes
            .iter_mut()
            .zip([M::RED, M::GREEN, M::BLUE, M::ALPHA])
            .enumerate()
            .take(channels)
        {
            if !c.write_mask.contains(bit) {
                continue;
            }
            let (s, d, op) = match k {
                3 => (
                    c.source_alpha_blend_factor,
                    c.destination_alpha_blend_factor,
                    c.alpha_blend_operation,
                ),
                _ => (
                    c.source_rgb_blend_factor,
                    c.destination_rgb_blend_factor,
                    c.rgb_blend_operation,
                ),
            };
            *lane = match (c.blending_enabled, s, d, op) {
                (false, ..) => Some(HalfOutputLane::Float32),
                (true, F::One, F::One, O::Add) => Some(HalfOutputLane::Float32Added),
                _ => {
                    every = false;
                    None
                }
            };
        }
        if every && lanes.iter().any(Option::is_some) {
            out.insert(i as u32, lanes);
        }
    }
    out
}

/// Whether a render pipeline whose color attachments are `colors` takes the `half` outputs the
/// fragment stage `fragment` computes at `float` precision as it compiled them
/// (`StageIo::half_output_lanes`): each attachment the function recorded has the same entry of
/// [`half_output_lanes`]. An attachment it recorded none for, every one where it compiled without
/// the map, takes the `half` the function rounds, and the function records
/// `PrecisionNote::HalfOutputPrecision` for it (under fast math it lists the attachment,
/// `StageIo::half_fast_colors`, for `HalfOutputFastMathBlend`): a missing map loses Metal's
/// `float`, never the pipeline.
pub(crate) fn half_output_lanes_hold(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
) -> bool {
    let recorded = &fragment.interface.io.half_output_lanes;
    if recorded.is_empty() {
        return true;
    }
    let lanes = half_output_lanes(colors);
    recorded.iter().all(|(c, l)| lanes.get(c) == Some(l))
}

/// The second outputs of the attachments of `colors` held with their channels rotated, on `p`
/// (`LoweringInputs::rotated_factors`): every such attachment whose blend Vulkan's factors read
/// from a second output the rotation needs (`rotated_blend::RotatedBlend::Synthesized`). An
/// attachment whose blend no Vulkan state gives has none; link refuses its pipeline.
pub(crate) fn rotated_factors(
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeMap<u32, [FactorLane; 4]> {
    colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let c = c.as_ref()?;
            if !metal::format::rotated_channels(c.pixel_format) {
                return None;
            }
            match rotated_blend(c, p) {
                Ok(Some(RotatedBlend::Synthesized(_, lanes))) => Some((i as u32, lanes)),
                _ => None,
            }
        })
        .collect()
}

/// The second outputs the fragment stage `fragment` writes for rotated attachments in a render
/// pipeline whose color attachments are `colors` on `p`: [`rotated_factors`]'s, for the
/// attachments it writes through a first (dual-source index 0) output.
pub(crate) fn wanted_rotated_factors(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeMap<u32, [FactorLane; 4]> {
    let written: BTreeSet<u32> = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        })
        .collect();
    let mut out = rotated_factors(colors, p);
    out.retain(|a, _| written.contains(a));
    out
}

/// The forms of the blends of the attachments of `colors` held with their channels rotated, on
/// `p` (`LoweringInputs::rotated_blends`): every such attachment whose blend the fragment
/// function shapes or announces (`rotated_blend::rotated_form`).
pub(crate) fn rotated_blends(
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeMap<u32, RotatedBlendForm> {
    colors
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            let c = c.as_ref()?;
            if !metal::format::rotated_channels(c.pixel_format) {
                return None;
            }
            Some((i as u32, rotated_form(c, p)?))
        })
        .collect()
}

/// The forms of the blends of rotated attachments the fragment stage `fragment` compiles under in
/// a render pipeline whose color attachments are `colors` on `p`: [`rotated_blends`]'s, for the
/// attachments it writes through a first (dual-source index 0) output.
pub(crate) fn wanted_rotated_blends(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    p: &DeviceProfile,
) -> BTreeMap<u32, RotatedBlendForm> {
    let written: BTreeSet<u32> = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        })
        .collect();
    let mut out = rotated_blends(colors, p);
    out.retain(|a, _| written.contains(a));
    out
}

/// The extended-range writes and reads the fragment stage `fragment` compiles under in a render
/// pipeline whose color attachments are `colors` and whose `textureWriteRoundingMode` is
/// `rounding`: [`extended_range_colors`]'s, for the attachments it writes through output
/// variables of dual-source index 0 or reads through input attachments.
pub(crate) fn wanted_extended_range(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: TextureWriteRoundingMode,
) -> BTreeMap<u32, ExtendedRangeColor> {
    let interface = &fragment.interface;
    let written: std::collections::BTreeSet<u32> = interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        })
        .chain(
            interface
                .descriptors
                .iter()
                .filter_map(|d| match d.slot.kind {
                    crate::pipeline::sym::ResourceKind::InputAttachment(color) => Some(color),
                    _ => None,
                }),
        )
        .collect();
    let mut out = extended_range_colors(colors, rounding);
    out.retain(|a, _| written.contains(a));
    out
}

/// The color writes the fragment stage `fragment` compiles under in a render pipeline whose
/// color attachments are `colors` (alpha-to-coverage on where `alpha_to_coverage`, rounding by
/// `rounding`) on `p`: the pipeline's, for the attachments it writes through output variables of
/// dual-source index 0 or in the storage form ([`written`]).
pub(crate) fn wanted(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> BTreeMap<u32, [Option<ChannelWrite>; 4]> {
    color_writes(written(fragment), colors, rounding, p)
}

/// The color attachments the fragment stage `fragment` writes through output variables of
/// dual-source index 0 or in the storage form.
fn written(fragment: &FunctionArtifact) -> impl Iterator<Item = u32> + '_ {
    let outputs = fragment
        .interface
        .io
        .outputs
        .iter()
        .filter_map(|v| match v.role {
            IoRole::Color {
                attachment,
                index: 0,
            } => Some(attachment),
            _ => None,
        });
    let storage = fragment.facts.iter().filter_map(|f| match *f {
        RuntimeFact::InterlockedFramebufferFetch {
            color,
            writes: true,
            ..
        } => Some(color),
        _ => None,
    });
    outputs.chain(storage)
}

/// Whether the color writes the fragment stage `fragment` compiled under
/// (`StageIo::color_writes`) are the pipeline's: each attachment's entry is the one the pipeline
/// gives it ([`wanted`]), or the entry of its channels' own rules ([`exact_writes`]), which a
/// function compiled under them (a caller's inputs, the map) takes, and the function has an entry
/// for no other attachment.
fn writes_hold(
    fragment: &FunctionArtifact,
    colors: &[Option<ColorAttachmentDescriptor>],
    rounding: (bool, TextureWriteRoundingMode),
    p: &DeviceProfile,
) -> bool {
    let selected = wanted(fragment, colors, rounding, p);
    let exact = exact_writes(written(fragment), colors, rounding, p);
    let compiled = &fragment.interface.io.color_writes;
    let mut attachments = selected.keys().chain(exact.keys()).chain(compiled.keys());
    attachments.all(|a| {
        let got = compiled.get(a);
        got == selected.get(a) || got == exact.get(a)
    })
}

/// How attachment `c` takes each channel of a float output; `coverage_alpha` where
/// alpha-to-coverage reads its alpha. `None` where it stores the value as written or nothing at
/// all.
fn attachment_write(
    c: &ColorAttachmentDescriptor,
    coverage_alpha: bool,
) -> Option<[Option<ChannelWrite>; 4]> {
    if c.write_mask.is_empty() {
        return None;
    }
    let device = |channels: u8| {
        let mut out = [None; 4];
        for channel in out.iter_mut().take(usize::from(channels)) {
            *channel = Some(ChannelWrite::Device);
        }
        out
    };
    let replaced = !c.blending_enabled || replaces(c);
    let mut rules = match write_conversion(c.pixel_format)? {
        // An extended-range attachment's texels are the function's own encoding.
        WriteConversion::Exact | WriteConversion::Encoded => return None,
        // A 32-bit float channel stores the output as written; a blend of it is the device's.
        WriteConversion::Float if replaced => return None,
        WriteConversion::Float | WriteConversion::Unmeasured => {
            return color_output_components(c.pixel_format).map(device);
        }
        WriteConversion::Rules(rules) => rules,
    };
    for rule in rules.iter_mut().flatten() {
        match rule {
            _ if !replaced => *rule = ChannelWrite::Device,
            ChannelWrite::Snorm { blended, .. } => *blended = c.blending_enabled,
            _ => {}
        }
    }
    if coverage_alpha && rules[3].is_some() {
        rules[3] = Some(ChannelWrite::Device);
    }
    Some(rules)
}

/// Whether the pipeline's attachment `c` blends the fragment function's output: its blend is
/// enabled and it writes some channel. Where it does not, a `half` output of a module without
/// fast math stores Metal's value (`StageIo::half_blend_colors`). Factors one and zero, added,
/// blend too: lavapipe (Mesa 26.2.4) stores the source through the destination there, one
/// `float` unit in the last place at the destination's magnitude off it in up to 11% of a
/// 32-bit float attachment's channels.
pub(crate) fn blends_output(c: &ColorAttachmentState) -> bool {
    c.blend.is_some() && !c.write_mask.is_empty()
}

/// Whether the pipeline's attachment `c` stores some channel of the fragment function's output,
/// blended or not: its write mask is not empty. Where it stores none, a `half` output of a module
/// with fast math leaves nothing Metal's compiler could fold otherwise
/// (`StageIo::half_fast_colors`).
pub(crate) fn stores_output(c: &ColorAttachmentState) -> bool {
    !c.write_mask.is_empty()
}

/// Whether a 16-bit float attachment `c` at `color` blends the output of the fragment stage of
/// interface `io` by one and zero though [`replaces`] holds of it: an output of a module without
/// fast math (`StageIo::half_blend_colors`, `StageIo::float_blend_colors`), whose blend there
/// Metal computes keeping the destination's product by zero (a NaN of an infinite or NaN texel,
/// `+0` of a `-0` output over a texel of `+0` or above), which no write of the
/// output alone stores. The pipeline blends it and announces `PrecisionNote::HalfOutputPrecision`;
/// the storage form computes Metal's (`lower::fragment::fetch`).
pub(crate) fn keeps_zero(
    io: &crate::pipeline::stage_io::StageIo,
    color: u32,
    c: &ColorAttachmentDescriptor,
) -> bool {
    crate::lower::fragment::half_float_format(c.pixel_format)
        && (io.half_blend_colors.contains(&color) || io.float_blend_colors.contains(&color))
}

/// Whether the linked attachment `s` is a 16-bit float one blended by one and zero, added, for
/// color and alpha ([`keeps_zero`]'s).
fn one_zero_half(s: &ColorAttachmentState) -> bool {
    use crate::vk::{BlendFactor as F, BlendOp as O, Format};
    let half = matches!(
        s.format,
        Format::R16_SFLOAT | Format::R16G16_SFLOAT | Format::R16G16B16A16_SFLOAT
    );
    half && s.blend.as_ref().is_some_and(|b| {
        (b.src_color, b.dst_color, b.color_op) == (F::ONE, F::ZERO, O::ADD)
            && (b.src_alpha, b.dst_alpha, b.alpha_op) == (F::ONE, F::ZERO, O::ADD)
    })
}

/// Whether `c`'s blend stores the source as it is: factors one and zero, added, for color and
/// alpha.
pub(crate) fn replaces(c: &ColorAttachmentDescriptor) -> bool {
    use metal::{BlendFactor as F, BlendOperation as O};
    (
        c.source_rgb_blend_factor,
        c.destination_rgb_blend_factor,
        c.rgb_blend_operation,
    ) == (F::One, F::Zero, O::Add)
        && (
            c.source_alpha_blend_factor,
            c.destination_alpha_blend_factor,
            c.alpha_blend_operation,
        ) == (F::One, F::Zero, O::Add)
}

#[cfg(test)]
mod tests;
