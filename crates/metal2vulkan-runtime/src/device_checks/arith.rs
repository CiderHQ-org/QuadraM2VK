//! The checks of two arithmetic deviations Metal's rule decides (`DriverDeviation`'s
//! `MetalCanonicalNans` and `MetalFloatDenormals`): each a kernel of the device's own instructions
//! over operand rows, against Metal's results by its rule
//! (`metal2vulkan::bind::canonical_nans_check`, `float_denormals_check`), in each math mode the
//! device has (`MathMode::of`). Metal's answer in every mode is Metal's; another in any mode is the
//! other.

use super::color::combined;
use super::converter::kernel_answer;
use super::kit::Kit;
use super::Target;
use metal2vulkan::bind::{canonical_nans_check, float_denormals_check, CheckKernel, MathMode};
use metal2vulkan::device::{CheckAnswer, Device};
use metal2vulkan::error::Error;

/// `check`'s answer on `target` in every math mode its device has.
pub(super) fn in_every_mode(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    check: fn(&Device, MathMode) -> Result<CheckKernel, Error>,
) -> CheckAnswer {
    let answers = MathMode::of(target.compile)
        .into_iter()
        .map(|mode| kernel_answer(kit, check(target.compile, mode)))
        .collect();
    combined(answers)
}

/// The two checks' answers on `target`, in their facts' order.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![
        in_every_mode(kit, target, canonical_nans_check),
        in_every_mode(kit, target, float_denormals_check),
    ]
}
