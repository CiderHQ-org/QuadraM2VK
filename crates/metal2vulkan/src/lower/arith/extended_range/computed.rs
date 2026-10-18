//! The blends of a `float4` output computed by whole-vector arithmetic over loads and constants
//! (`lower::arith::output_precision`'s class) whose bytes Metal draws as a loaded output's, under
//! fast math, into a linear extended-range form.
//!
//! Metal's compiler folds the blend into the function, so how it rounds a computed source follows
//! the computation; for a whole-vector product, sum, difference, quotient or fused multiply-add
//! over vector loads and constants it rounds the source as it rounds a loaded one in every lane
//! class but those of [`COLOR_DIFFERS`] and [`ALPHA_DIFFERS`] (plain Metal on an Apple M4 under
//! macOS 27.0.1: the factor sweeps' `float4` configurations of the source times a loaded vector of
//! ones, byte for byte against our probes' plain and loaded sources; every color and alpha
//! key of `Add`, `Subtract` and `ReverseSubtract` but the second source's, and 343 of `Min` and
//! `Max`, whose factors Metal ignores). A key of either table differs from the loaded source's on
//! some configuration (a third of the color keys whose source factor is the source alpha read the
//! source unrounded beside it, among others), so a blend whose color or alpha lane class takes one
//! stays announced.

use crate::metal::format::ExtendedRange;
use crate::metal::{BlendFactor as F, BlendOperation as O};
use crate::pipeline::lower_inputs::BlendEquation;

/// The color lanes' keys (source factor, destination factor, operation) whose computed source
/// Metal blends otherwise than its loaded one somewhere (the module docs).
pub(crate) const COLOR_DIFFERS: [(F, F, O); 62] = [
    (F::One, F::SourceAlpha, O::Add),
    (F::One, F::SourceAlpha, O::ReverseSubtract),
    (F::SourceColor, F::SourceAlpha, O::Add),
    (F::SourceColor, F::OneMinusSourceAlpha, O::Add),
    (F::SourceColor, F::DestinationColor, O::Add),
    (F::SourceColor, F::DestinationAlpha, O::Add),
    (F::SourceColor, F::OneMinusDestinationAlpha, O::Add),
    (F::OneMinusSourceColor, F::OneMinusDestinationColor, O::Add),
    (F::OneMinusSourceColor, F::SourceAlphaSaturated, O::Add),
    (F::OneMinusSourceColor, F::BlendAlpha, O::Add),
    (F::SourceAlpha, F::Zero, O::Add),
    (F::SourceAlpha, F::Zero, O::Subtract),
    (F::SourceAlpha, F::One, O::Add),
    (F::SourceAlpha, F::One, O::Subtract),
    (F::SourceAlpha, F::One, O::ReverseSubtract),
    (F::SourceAlpha, F::OneMinusSourceColor, O::Subtract),
    (F::SourceAlpha, F::OneMinusSourceColor, O::ReverseSubtract),
    (F::SourceAlpha, F::OneMinusSourceAlpha, O::Add),
    (F::SourceAlpha, F::OneMinusSourceAlpha, O::Subtract),
    (F::SourceAlpha, F::OneMinusSourceAlpha, O::ReverseSubtract),
    (F::SourceAlpha, F::DestinationColor, O::Add),
    (F::SourceAlpha, F::DestinationColor, O::Subtract),
    (F::SourceAlpha, F::DestinationColor, O::ReverseSubtract),
    (F::SourceAlpha, F::OneMinusDestinationColor, O::Add),
    (F::SourceAlpha, F::OneMinusDestinationColor, O::Subtract),
    (
        F::SourceAlpha,
        F::OneMinusDestinationColor,
        O::ReverseSubtract,
    ),
    (F::SourceAlpha, F::DestinationAlpha, O::Add),
    (F::SourceAlpha, F::DestinationAlpha, O::Subtract),
    (F::SourceAlpha, F::DestinationAlpha, O::ReverseSubtract),
    (F::SourceAlpha, F::OneMinusDestinationAlpha, O::Add),
    (F::SourceAlpha, F::OneMinusDestinationAlpha, O::Subtract),
    (
        F::SourceAlpha,
        F::OneMinusDestinationAlpha,
        O::ReverseSubtract,
    ),
    (F::SourceAlpha, F::SourceAlphaSaturated, O::Add),
    (F::SourceAlpha, F::SourceAlphaSaturated, O::Subtract),
    (F::SourceAlpha, F::SourceAlphaSaturated, O::ReverseSubtract),
    (F::SourceAlpha, F::BlendColor, O::Add),
    (F::SourceAlpha, F::BlendColor, O::Subtract),
    (F::SourceAlpha, F::BlendColor, O::ReverseSubtract),
    (F::SourceAlpha, F::OneMinusBlendColor, O::Add),
    (F::SourceAlpha, F::OneMinusBlendColor, O::Subtract),
    (F::SourceAlpha, F::OneMinusBlendColor, O::ReverseSubtract),
    (F::SourceAlpha, F::BlendAlpha, O::Add),
    (F::SourceAlpha, F::BlendAlpha, O::Subtract),
    (F::SourceAlpha, F::BlendAlpha, O::ReverseSubtract),
    (F::SourceAlpha, F::OneMinusBlendAlpha, O::Add),
    (F::SourceAlpha, F::OneMinusBlendAlpha, O::Subtract),
    (F::SourceAlpha, F::OneMinusBlendAlpha, O::ReverseSubtract),
    (F::OneMinusSourceAlpha, F::SourceAlpha, O::Add),
    (F::OneMinusSourceAlpha, F::DestinationColor, O::Add),
    (F::OneMinusSourceAlpha, F::DestinationAlpha, O::Add),
    (F::OneMinusSourceAlpha, F::OneMinusDestinationAlpha, O::Add),
    (F::DestinationColor, F::SourceColor, O::Add),
    (F::DestinationColor, F::SourceAlpha, O::Add),
    (F::DestinationColor, F::SourceAlpha, O::ReverseSubtract),
    (F::DestinationColor, F::OneMinusDestinationColor, O::Add),
    (
        F::DestinationColor,
        F::OneMinusDestinationColor,
        O::Subtract,
    ),
    (F::OneMinusDestinationColor, F::SourceAlpha, O::Add),
    (F::DestinationAlpha, F::Zero, O::ReverseSubtract),
    (F::DestinationAlpha, F::SourceAlpha, O::Add),
    (F::DestinationAlpha, F::OneMinusSourceAlpha, O::Add),
    (F::DestinationAlpha, F::DestinationColor, O::Add),
    (F::DestinationAlpha, F::OneMinusDestinationAlpha, O::Add),
];

/// Alpha's keys likewise.
pub(crate) const ALPHA_DIFFERS: [(F, F, O); 12] = [
    (F::SourceColor, F::DestinationColor, O::Add),
    (F::SourceColor, F::DestinationAlpha, O::Add),
    (F::OneMinusSourceColor, F::OneMinusDestinationColor, O::Add),
    (F::OneMinusSourceColor, F::OneMinusDestinationAlpha, O::Add),
    (F::SourceAlpha, F::DestinationColor, O::Add),
    (F::SourceAlpha, F::DestinationAlpha, O::Add),
    (F::OneMinusSourceAlpha, F::OneMinusDestinationColor, O::Add),
    (F::OneMinusSourceAlpha, F::OneMinusDestinationAlpha, O::Add),
    (
        F::DestinationColor,
        F::OneMinusDestinationColor,
        O::Subtract,
    ),
    (
        F::DestinationColor,
        F::OneMinusDestinationAlpha,
        O::Subtract,
    ),
    (
        F::DestinationAlpha,
        F::OneMinusDestinationColor,
        O::Subtract,
    ),
    (
        F::DestinationAlpha,
        F::OneMinusDestinationAlpha,
        O::Subtract,
    ),
];

/// Whether Metal blends `eq` into an attachment of `form` from a `float4` output of the computed
/// class as from a loaded one (the module docs): a linear form, and neither lane class's key in
/// [`COLOR_DIFFERS`] or [`ALPHA_DIFFERS`] (alpha's but where the form holds no alpha).
pub(crate) fn holds_as_loaded(eq: &BlendEquation, form: ExtendedRange) -> bool {
    let color = (eq.source_rgb, eq.destination_rgb, eq.rgb_operation);
    let alpha = (eq.source_alpha, eq.destination_alpha, eq.alpha_operation);
    let has_alpha = !matches!(form, ExtendedRange::Bgr10 { .. });
    matches!(
        form,
        ExtendedRange::Bgra10 { srgb: false }
            | ExtendedRange::Bgr10 { srgb: false }
            | ExtendedRange::Planes { srgb: false }
    ) && !COLOR_DIFFERS.contains(&color)
        && !(has_alpha && ALPHA_DIFFERS.contains(&alpha))
}
