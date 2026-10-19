//! The check of `DriverDeviation::MetalHalfQuantize`: the device's `OpQuantizeToF16` in
//! translation's form, over every rounding edge of the conversion to `half`, against Metal's
//! conversion (`metal2vulkan::bind::half_quantize_check`). Any other word is the other answer: a
//! rounding that is not Metal's.

use super::kit::Kit;
use super::Target;
use metal2vulkan::device::CheckAnswer;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![answer(kit, target)]
}

/// The check's one answer.
fn answer(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let kernel = match metal2vulkan::bind::half_quantize_check(target.compile) {
        Ok(kernel) => kernel,
        Err(error) => {
            return CheckAnswer::Unanswered(format!("its kernel does not build: {error}"))
        }
    };
    match kit.kernel(&kernel) {
        Ok(words) => match kernel.difference(&words) {
            None => CheckAnswer::Metals,
            Some(how) => CheckAnswer::NotMetals(how),
        },
        Err(error) => CheckAnswer::Unanswered(error.0),
    }
}
