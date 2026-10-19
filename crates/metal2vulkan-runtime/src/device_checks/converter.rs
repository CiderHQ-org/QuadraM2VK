//! The checks of three shader-converter defects translation works around (`DriverWorkaround`'s
//! `CompositeInsertsWrittenInPlace`, `LoadsForwardedIntoLoops` and
//! `LocalizedLoadsForwardedPastAtomicsAndCalls`): each a kernel of the instructions translation
//! writes where the profile does not take the workaround
//! (`metal2vulkan::bind::composite_inserts_check`, `loads_forwarded_check`,
//! `localized_loads_check`). Metal's answer is what the instructions mean; any other word is the
//! defect, which the workaround avoids.

use super::kit::Kit;
use super::Target;
use metal2vulkan::bind::{
    composite_inserts_check, loads_forwarded_check, localized_loads_check, CheckKernel,
};
use metal2vulkan::device::CheckAnswer;
use metal2vulkan::error::Error;

/// The answer of `kernel` on the kit's device: Metal's where every word it reads back is, the
/// other where one is not, and none where it does not build or run.
pub(super) fn kernel_answer(kit: &mut Kit<'_>, kernel: Result<CheckKernel, Error>) -> CheckAnswer {
    let kernel = match kernel {
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

/// The three checks' answers on `target`, in their facts' order.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    let device = target.compile;
    vec![
        kernel_answer(kit, composite_inserts_check(device)),
        kernel_answer(kit, loads_forwarded_check(device)),
        kernel_answer(kit, localized_loads_check(device)),
    ]
}
