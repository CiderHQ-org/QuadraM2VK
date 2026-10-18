//! A fragment function's color outputs rounded to the codes Metal stores: which lanes of each
//! output round, from `LoweringInputs::color_writes`, and their emission (`arith::color_write`).
//!
//! The function records the map it compiled under for each attachment it writes
//! (`StageIo::color_writes`, which link checks against the pipeline's), and rounds each lane
//! whose channel has a rule:
//!
//! - a channel the device converts ([`ChannelWrite::Device`]: a blend other than one and zero,
//!   alpha-to-coverage's alpha, or a format Metal was not measured with) is written as computed
//!   and the module records `PrecisionNote::DeviceColorConversion`; one whose blend Metal rounds
//!   to nearest even ([`ChannelWrite::BlendedNearestEven`], a half-float channel under
//!   `TextureWriteRoundingMode::NearestEven`) is too, and records
//!   `PrecisionNote::BlendedWriteRounding`;
//! - a `half` output (held widened) to a half-float channel is already a half: it is not
//!   rounded again (to an 11- or 10-bit float channel it is), and to a packed 16-bit format's
//!   channel ([`ChannelWrite::HalfUnorm`]) it rounds as a unorm channel's;
//! - a `bfloat` output never arrives: Metal's compiler rejects one, and the stage's outputs refuse
//!   it before (`legal::refuse_bfloat_color`);
//! - an integer output to a channel with a rule is Metal's pipeline error, which link refuses
//!   (`ColorAttachment`): nothing rounds it;
//! - a NaN written to a half-float channel is stored as the device's conversion encodes it
//!   (`PrecisionNote::DeviceNanEncoding`) unless the device is measured to store Metal's: Metal
//!   stores a `float` NaN as `0x7e00` and a `half` NaN's own bits. A `float` output's NaN (the
//!   rounding writes `0x7fc00000`) is Metal's where the profile takes
//!   `DriverDeviation::MetalFloatColorNans` or `MetalColorWrites`, a `half` output's (held
//!   widened, its payload in the significand's top bits) where it takes `MetalHalfColorNans`,
//!   and either where the function writes it through a 16-bit output (below); elsewhere the
//!   module records the note (`runtime_lane::color_writes`: ANV and RADV store `0xfe00`, from a
//!   16-bit output too, lavapipe quiets or canonicalizes a `half` NaN, MoltenVK canonicalizes it
//!   from a 32-bit output).
//!
//! On a profile that stores Metal's codes itself (`VkProperty::MetalColorWrites`) the map holds
//! only half-float channels (`pipeline::color_writes`): nothing rounds, and only the NaN note is
//! recorded, but for a channel that rounds to nearest even (`color_writes::HALF_NEAREST`, a
//! pipeline's `TextureWriteRoundingMode::NearestEven`), which no device's conversion does: the
//! function rounds it on every profile, and the device's toward-zero conversion of the rounded half
//! is exact.
//!
//! Where the device is measured to convert one kind of channel as Metal does, an output variable's
//! lane to such a channel is written as computed, its NaN chosen where the device's store of it
//! would not be Metal's ([`color_write::Lane::NanAs`]), and the module requires the deviation's
//! property:
//! - a `float` output to a half-float or unsigned small float channel
//!   (`DriverDeviation::MetalSmallFloatColorWrites`: RADV on the Radeon 780M, ANV on the Iris Xe
//!   of RPL-P, NVIDIA's on the RTX 3090), its NaN written as the rounding's `0x7fc00000`, which
//!   every such device stores as Metal's NaN of an unsigned channel and NVIDIA's as `0x7e00`
//!   (RADV and ANV store `0xfe00` for every NaN of a half-float channel, noted as above, so
//!   nothing is chosen there);
//! - an output to a 16-bit unorm or snorm channel (`DriverDeviation::MetalNorm16ColorWrites`:
//!   RADV on the Radeon 780M), a blended snorm channel's NaN written as -1.0.
//!
//! An attachment no lane of which rounds is not listed as rounded: it keeps its Vulkan blend of
//! one and zero, through which the facts were measured too, and announces no shader conversion.
//! A write in the storage form, or through a 16-bit output, rounds as before: the facts are of
//! the device's conversion of an output variable. A storage write the shader blends into a 16-bit
//! float attachment gives every NaN as `0x7fc00000`, which the storage image's store converts:
//! Metal's `0x7e00` where the profile takes `DriverDeviation::MetalFloatStorageNans` (the module
//! requires its property), else the device's NaN, noted `PrecisionNote::DeviceNanEncoding` (ANV
//! stores `0xfe00`).
//!
//! Where the device does not store the output's NaN as Metal does, but is measured to store a
//! 16-bit output's bits unchanged (`DriverDeviation::HalfColorOutputBits`: MoltenVK 1.4.1 on the
//! M3 Max and the M4), its conversion to 16 bits is Metal's `half(float)`
//! (`DriverDeviation::MetalHalfConversion`), and the profile declares 16-bit outputs
//! (`storageInputOutput16`, with `shaderInt16` and `shaderFloat16`), an output to an attachment
//! whose every channel written is half-float is written as the bits of the half Metal stores,
//! through a 16-bit output variable ([`half_output`]): a `float` output rounded toward zero, a
//! `half` output as it is, each converted exactly to 16 bits by that conversion, a NaN as its own
//! payload's code by integer operations (`arith::color_write::half_bits`), so no conversion of the
//! device's chooses the NaN, the note is not recorded, and the module requires
//! `VkProperty::HalfColorOutputBits` and `VkProperty::MetalHalfConversion`. An attachment a lane of
//! which rounds is listed as rounded (`rounded_colors`), and one whose output is written so but
//! rounds nothing (a `half` output's) as written as a half's bits (`half_bit_colors`), its own
//! notice; either takes no Vulkan blend of one and zero. On a device not measured so the output
//! stays 32-bit: ANV and RADV store every NaN of a 16-bit output as `0xfe00` too, so the route
//! would change nothing there.

use crate::air::abi::{AbiInt, OutputRole};
use crate::device::requirements::VkProperty;
use crate::error::DriverDeviation;
use crate::lower::arith::color_write;
use crate::lower::arith::srgb;
use crate::lower::arith::{FloatFmt, Form};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::color_writes;
use crate::pipeline::lower_inputs::ChannelWrite;
use crate::pipeline::stage_io::SelectedWrite;
use crate::pipeline::sym::{PushField, Stage};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use std::collections::BTreeSet;

/// Plan the rounding of a fragment function's output of form `f` to color attachment `color`
/// (dual-source index 0), through an output variable or in the storage form (see the module
/// docs).
pub(crate) fn plan(cx: &mut SelectCx<'_>, color: u32, f: Form) {
    if cx.stage != Stage::Fragment {
        return;
    }
    let Some(rules) = cx.inputs.color_writes.get(&color).copied() else {
        return;
    };
    cx.families
        .stage_io
        .interface
        .color_writes
        .insert(color, rules);
    let metal = cx
        .profile
        .takes_driver_deviation(DriverDeviation::MetalColorWrites);
    let metal_nans = metal_nans(cx, f);
    let half = half_output(cx, color, f);
    let storage = crate::lower::fragment::writes_through_storage(cx, color);
    // A storage write the shader blends into a 16-bit float attachment stores the blend's `half`
    // in every channel, which the device converts exactly, but for a NaN: no conversion of an
    // output variable's bears on it, and its NaN is the storage image's store's ([`storage_nan`]).
    // One the shader blends into an 8-bit unorm attachment stores the code Metal's unorm rule
    // gives the blend's result, which the shader rounds itself (`fetch::unorm8_blended`).
    let shader_blended = crate::lower::fragment::blends_in_storage(cx, color);
    let sixteen_blended = crate::lower::fragment::blends_sixteen_in_storage(cx, color);
    let mut lanes = [None; 4];
    let mut device = false;
    let mut blended = false;
    let mut half_nans = false;
    let mut relied = BTreeSet::new();
    let mut selected = None;
    for (slot, rule) in lanes.iter_mut().zip(rules).take(usize::from(f.lanes)) {
        let Some(rule) = rule else { continue };
        half_nans |= !shader_blended && color_writes::half_float(rule) && !metal_nans && !half;
        if metal && rule == color_writes::HALF {
            continue;
        }
        *slot = match (rule, f.float_fmt()) {
            (_, None | Some(FloatFmt::F64 | FloatFmt::BFloat)) => None,
            // The draw selects the channel's rule: each one's lane, and what it does.
            (ChannelWrite::Selected, Some(fmt)) => {
                let (lane, write) = selected_lane(
                    cx,
                    (fmt, f.lanes),
                    (metal_nans, half || storage),
                    &mut relied,
                );
                selected = Some(write);
                Some(lane)
            }
            (
                ChannelWrite::FloatTowardZero {
                    mantissa: 10..,
                    signed: true,
                }
                | ChannelWrite::FloatNearestEven {
                    mantissa: 10..,
                    signed: true,
                },
                Some(FloatFmt::Half),
            ) => None,
            (ChannelWrite::BlendedNearestEven { .. }, _) => {
                blended |= !shader_blended;
                None
            }
            (ChannelWrite::HalfUnorm { bits }, Some(fmt @ FloatFmt::Half)) => lane(
                cx,
                ChannelWrite::Unorm { bits },
                fmt,
                (metal_nans, half || storage),
                &mut relied,
            ),
            (ChannelWrite::Device, _) => {
                device |= !shader_blended;
                None
            }
            (rule, Some(fmt @ (FloatFmt::F32 | FloatFmt::Half))) => {
                lane(cx, rule, fmt, (metal_nans, half || storage), &mut relied)
            }
        };
    }
    // The draw's rule word, and what each of its codes does, which link reads.
    if let Some(write) = selected {
        cx.push.insert(PushField::ColorWriteRule(color));
        cx.families
            .stage_io
            .interface
            .selected_colors
            .insert(color, write);
    }
    if device {
        cx.precision.insert(PrecisionNote::DeviceColorConversion);
    }
    if blended {
        cx.precision.insert(PrecisionNote::BlendedWriteRounding);
    }
    if sixteen_blended {
        half_nans |= !storage_nan(cx, &mut relied);
    }
    if half_nans {
        cx.precision.insert(PrecisionNote::DeviceNanEncoding);
    }
    if half {
        cx.families.stage_io.halves.insert(color);
        relied.extend([
            VkProperty::HalfColorOutputBits,
            VkProperty::MetalHalfConversion,
        ]);
    }
    cx.families.stage_io.color_properties.extend(relied);
    let rounds = lanes
        .iter()
        .any(|l| matches!(l, Some(color_write::Lane::Rounded(_))));
    // The color table, once: its buffer counts in the stage's limits.
    if lanes.contains(&Some(color_write::Lane::Rounded(ChannelWrite::Srgb)))
        && cx.families.arith.color_table.is_none()
    {
        let class = srgb::ColorTable::of(cx.profile);
        cx.families.arith.color_table = Some(class);
        let d = &mut cx.descriptors;
        match class {
            srgb::ColorTable::Uniform => d.uniform_buffers = d.uniform_buffers.saturating_add(1),
            srgb::ColorTable::Storage => d.storage_buffers = d.storage_buffers.saturating_add(1),
        }
    }
    let sp = &mut cx.families.stage_io;
    if lanes.iter().any(Option::is_some) {
        sp.rounded.insert(color, lanes);
    }
    let listed = match (rounds, half) {
        (true, _) => &mut sp.interface.rounded_colors,
        (false, true) => &mut sp.interface.half_bit_colors,
        (false, false) => return,
    };
    if let Err(at) = listed.binary_search(&color) {
        listed.insert(at, color);
    }
}

/// Whether a storage write the shader blends into a 16-bit float attachment stores its NaN as
/// Metal does: the blend gives every NaN as `0x7fc00000` (`fetch::half_blended`; Metal stores
/// every NaN of that blend as `0x7e00`), which the storage image's store converts to Metal's
/// `0x7e00` where the profile takes `DriverDeviation::MetalFloatStorageNans`, whose property the
/// module then relies on (`relied`). Elsewhere the NaN stored is the device's (ANV stores every NaN
/// as `0xfe00`), which the module notes (`PrecisionNote::DeviceNanEncoding`): no value written to a
/// 16-bit float storage image stores another there.
fn storage_nan(cx: &SelectCx<'_>, relied: &mut BTreeSet<VkProperty>) -> bool {
    let metal = cx
        .profile
        .takes_driver_deviation(DriverDeviation::MetalFloatStorageNans);
    if metal {
        relied.insert(VkProperty::MetalFloatStorageNans);
    }
    metal
}

/// The `f32` NaN the rounding writes for a small float's NaN (`0x7fc00000`), whose store each
/// driver of [`DriverDeviation::MetalSmallFloatColorWrites`] is measured by.
const QUIET_NAN: u32 = 0x7fc0_0000;

/// How a fragment function writes a channel of rule `rule` (a rule the function applies, not
/// the device's) from an output of `fmt` (`F32` or `Half`) (`metal_nans`: the device stores the
/// output's NaN to a half-float channel as Metal does; `own`: the function writes the output
/// otherwise than as a value of its output variable the device converts, as a half's bits
/// through a 16-bit output, [`half_output`], or as a texel of a storage image): rounded by
/// Metal's rule, unless the device is measured to convert the channel's output as Metal does, whose
/// property the module then relies on (`relied`):
///
/// - a `float` output to a half-float or unsigned small float channel where the profile takes
///   `DriverDeviation::MetalSmallFloatColorWrites`: as computed, a NaN as the rounding's
///   `0x7fc00000` (an unsigned channel's always, a half-float channel's where `metal_nans`;
///   elsewhere the device stores its own NaN there, noted `PrecisionNote::DeviceNanEncoding`);
/// - an output to a 16-bit unorm or snorm channel where the profile takes
///   `DriverDeviation::MetalNorm16ColorWrites`: as computed, a blended snorm channel's NaN as
///   -1.0, as Metal stores it.
fn lane(
    cx: &SelectCx<'_>,
    rule: ChannelWrite,
    fmt: FloatFmt,
    (metal_nans, own): (bool, bool),
    relied: &mut BTreeSet<VkProperty>,
) -> Option<color_write::Lane> {
    use color_write::Lane;
    let takes = |d| !own && cx.profile.takes_driver_deviation(d);
    match rule {
        ChannelWrite::FloatTowardZero { signed, .. }
            if fmt == FloatFmt::F32 && takes(DriverDeviation::MetalSmallFloatColorWrites) =>
        {
            relied.insert(VkProperty::MetalSmallFloatColorWrites);
            (!signed || metal_nans).then_some(Lane::NanAs(QUIET_NAN))
        }
        ChannelWrite::Unorm { bits: 16 } | ChannelWrite::Snorm { bits: 16, .. }
            if takes(DriverDeviation::MetalNorm16ColorWrites) =>
        {
            relied.insert(VkProperty::MetalNorm16ColorWrites);
            matches!(rule, ChannelWrite::Snorm { blended: true, .. })
                .then_some(Lane::NanAs((-1.0f32).to_bits()))
        }
        rule => Some(Lane::Rounded(rule)),
    }
}

/// How a fragment function writes a lane of an output of `fmt` (`F32` or `Half`; `lanes` lanes)
/// whose channel's rule the draw selects (`ChannelWrite::Selected`), and what the attachment's
/// write does under each code of its rule word (`metal_nans` and `own` as [`lane`]'s, which
/// gives a half-float channel's lane, and records the properties it relies on in `relied`): a
/// lane of an 8-bit unorm channel rounds; of a half-float channel, it rounds where a `float`
/// output's lane does, a `half` output being a half already, and stores a NaN as the device
/// encodes it where the device does not store the output's NaN as Metal does (a lane the device
/// converts, or of no rule, as computed).
fn selected_lane(
    cx: &SelectCx<'_>,
    (fmt, lanes): (FloatFmt, u8),
    (metal_nans, own): (bool, bool),
    relied: &mut BTreeSet<VkProperty>,
) -> (color_write::Lane, SelectedWrite) {
    use color_write::{HalfLane, Lane};
    let half = match fmt {
        FloatFmt::Half => HalfLane::AsComputed,
        _ => match lane(cx, color_writes::HALF, fmt, (metal_nans, own), relied) {
            Some(Lane::Rounded(_)) => HalfLane::Rounded,
            Some(Lane::NanAs(bits)) => HalfLane::NanAs(bits),
            Some(Lane::Selected { .. }) | None => HalfLane::AsComputed,
        },
    };
    let bit = |code: u32| 1u8 << code;
    let half_rounds = match half {
        HalfLane::Rounded => bit(color_writes::HALF_CODE),
        HalfLane::AsComputed | HalfLane::NanAs(_) => 0,
    };
    let write = SelectedWrite {
        lanes,
        rounds: bit(color_writes::UNORM8_CODE) | half_rounds,
        nans: match metal_nans {
            true => 0,
            false => bit(color_writes::HALF_CODE),
        },
    };
    (Lane::Selected { half }, write)
}

/// Whether the device stores the NaN of a fragment function's output of form `f` to a
/// half-float channel as Metal does (the module docs): a `float` output's where the profile
/// takes `DriverDeviation::MetalFloatColorNans` or `MetalColorWrites`, a `half` output's where
/// it takes `MetalHalfColorNans`. Another output stores no NaN of its own there.
fn metal_nans(cx: &SelectCx<'_>, f: Form) -> bool {
    let takes = |d| cx.profile.takes_driver_deviation(d);
    match f.float_fmt() {
        Some(FloatFmt::F32) => {
            takes(DriverDeviation::MetalColorWrites) || takes(DriverDeviation::MetalFloatColorNans)
        }
        Some(FloatFmt::Half) => takes(DriverDeviation::MetalHalfColorNans),
        _ => true,
    }
}

/// Whether a fragment function writes its output of form `f` (dual-source index 0) to color
/// attachment `color` as the bits of the half Metal stores, through a 16-bit output variable
/// (the module docs): the output is `float` or `half` and the device does not store its NaN as
/// Metal does, but stores a 16-bit output's bits unchanged
/// (`DriverDeviation::HalfColorOutputBits`); its conversion to 16 bits is Metal's `half(float)`
/// (`narrow::QuantizeForm::Converted`: `DriverDeviation::MetalHalfConversion` and `shaderFloat16`,
/// not native half rounding's conversion); every channel of the attachment it writes is half-float
/// (`color_writes::HALF`), none the device's; the profile declares 16-bit outputs
/// (`storageInputOutput16`) and the `Int16` arithmetic that builds a NaN's code (a channel that
/// rounds to nearest even, `color_writes::HALF_NEAREST`, is half-float too); and the
/// function writes the attachment through an output variable, with no dual-source second
/// output beside it.
pub(super) fn half_output(cx: &SelectCx<'_>, color: u32, f: Form) -> bool {
    let Some(rules) = cx.inputs.color_writes.get(&color) else {
        return false;
    };
    let leaf = cx.profile.fields().leaf;
    let written = || rules.iter().take(usize::from(f.lanes));
    let dual = cx.entry.ep.outputs.iter().any(|o| {
        matches!(
            o.role,
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(a),
                index: 1,
                ..
            } if a == color
        )
    });
    cx.stage == Stage::Fragment
        && matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half))
        && !metal_nans(cx, f)
        && cx
            .profile
            .takes_driver_deviation(DriverDeviation::HalfColorOutputBits)
        && crate::lower::arith::narrow::quantize_form(cx.profile)
            == crate::spirv::builder::QuantizeForm::Converted
        // The measured conversion, not native half rounding's.
        && cx
            .profile
            .takes_driver_deviation(DriverDeviation::MetalHalfConversion)
        && leaf.input_output_16bit
        && leaf.int16
        && written().any(Option::is_some)
        && written().flatten().all(|&r| color_writes::half_float(r))
        && !dual
        && !crate::lower::fragment::writes_through_storage(cx, color)
}

/// The words of the planned writes, each emitted once per return.
pub(super) fn words(rounded: impl IntoIterator<Item = [Option<color_write::Lane>; 4]>) -> u64 {
    rounded
        .into_iter()
        .map(|r| color_write::write_words(&r))
        .sum()
}

/// Output `x` (form `f`) to color attachment `color`, written as the plan writes each lane:
/// rounded, or its NaN chosen, or by the rule the attachment's rule word selects
/// (`PushField::ColorWriteRule`).
pub(crate) fn round(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    f: Form,
    x: Value,
) -> BResult<Value> {
    let Some(lanes) = cx.plan.families.stage_io.rounded.get(&color).copied() else {
        return Ok(x);
    };
    let table = cx.decl.families.arith.color_table;
    let selects = lanes
        .iter()
        .any(|l| matches!(l, Some(color_write::Lane::Selected { .. })));
    let word = match selects {
        true => Some(cx.push_field(fb, PushField::ColorWriteRule(color))?),
        false => None,
    };
    color_write::write(fb, f, x, &lanes, (table, word))
}
