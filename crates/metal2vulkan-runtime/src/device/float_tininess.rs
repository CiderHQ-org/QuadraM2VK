//! `float-tininess-after-rounding`: where the device's own `float` multiply and fused multiply-add
//! answer for Metal's. Metal flushes a `float` product or fused multiply-add whose exact value lies
//! below `2^-126` to the zero of its sign, deciding tininess before rounding; every Vulkan device
//! measured off MoltenVK (RADV, ANV, NVIDIA and lavapipe, whether they flush denormals or keep
//! them) rounds before it decides, so an exact value within half a unit below `2^-126` answers
//! `2^-126`.
//! Under `FloatMathsPolicy::Exact` the translation computes Metal's answer (the multiply's check of
//! its operands' bits, the fused multiply-add's routine); under native device maths it takes the
//! device's, which every pipeline of a `float` multiply may meet, so the device announces it once,
//! as it is made, rather than each pipeline.

use crate::degradation::Degradation;
use metal2vulkan::device::FloatMathsPolicy;
use metal2vulkan::error::DriverDeviation;

/// Whether `compile` takes the device's own `float` multiply and fused multiply-add where they
/// round a tiny result before flushing it: its policy takes native device maths
/// (`FloatMathsPolicy::Native`) and its float functions are not Metal's own
/// (`DriverDeviation::MetalFloatFunctions`, MoltenVK).
pub(crate) fn rounds_before_flushing(compile: &metal2vulkan::device::Device) -> bool {
    compile.policy().float_maths == FloatMathsPolicy::Native
        && !compile
            .profile()
            .takes_driver_deviation(DriverDeviation::MetalFloatFunctions)
}

impl super::Device {
    /// Announce `float-tininess-after-rounding` once where the device takes the device's own
    /// `float` multiply and fused multiply-add ([`rounds_before_flushing`]).
    pub(super) fn degrade_float_tininess(&self) {
        if !rounds_before_flushing(self.compile_device()) {
            return;
        }
        self.degrade(Degradation::FloatTininessAfterRounding, || {
            (
                format!("the device {}", self.name()),
                "its policy takes native device maths (`FloatMathsPolicy::Native`), and its \
                 `float` multiply and fused multiply-add round before they flush"
                    .to_owned(),
            )
        });
    }
}
