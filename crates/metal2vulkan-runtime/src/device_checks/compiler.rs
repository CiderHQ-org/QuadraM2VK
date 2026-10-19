//! The checks of two shader-compiler defects translation works around (`DriverWorkaround`'s
//! `NarrowMaskAfterSignedClamp` and `HalfConversionPairFolded`): each a kernel of the instructions
//! translation writes where the profile does not take the workaround
//! (`metal2vulkan::bind::narrow_mask_check`, `half_round_trip_check`) over inputs whose answers the
//! defect changes, the round trip in each math mode the device has (`MathMode::of`). The round
//! trip does not run on a device without `shaderFloat16` or the 16-bit float controls, where no
//! module rounds in that form.

use super::arith::in_every_mode;
use super::converter::kernel_answer;
use super::kit::Kit;
use super::Target;
use metal2vulkan::bind::{half_round_trip_check, half_round_trip_unsupported, narrow_mask_check};
use metal2vulkan::device::CheckAnswer;

/// The two checks' answers on `target`, in their facts' order.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    let device = target.compile;
    let round_trip = match half_round_trip_unsupported(device) {
        Some(why) => CheckAnswer::NotRun(why.to_owned()),
        None => in_every_mode(kit, target, half_round_trip_check),
    };
    vec![kernel_answer(kit, narrow_mask_check(device)), round_trip]
}
