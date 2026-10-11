//! Why MoltenVK refused a pipeline, read from the report it sends with the refusal.
//!
//! MoltenVK creates a pipeline in three steps: SPIRV-Cross converts each module to MSL, Metal's
//! compiler builds the MSL into a library (its front end), specializes the entry point, and then
//! builds the pipeline state (its GPU back end, which runs in Metal's compiler service, a separate
//! process). A step that fails makes `vkCreate*Pipelines` return an error and sends an error
//! message through the debug messenger, which the acceptance sweep keeps with the row
//! (`corpus-pipeline`'s detail column). MoltenVK 1.4.1 words it as
//!
//! - `SPIR-V to MSL conversion error: ...` when SPIRV-Cross refuses the module;
//! - `<step> compile failed (Error code N):\n<description>.` when a Metal step fails, `<step>` one
//!   of `Shader library`, `Function specialization`, `Compute pipeline`, or `Render pipeline`, and
//!   `<description>` Metal's error, or MoltenVK's own `Timeout after T milliseconds. Likely
//!   internal Metal compiler error` when the step outlived `MVK_CONFIG_METAL_COMPILE_TIMEOUT`.
//!
//! When Metal's compiler service dies during a step, Metal's description names the lost
//! connection (`XPC_ERROR_CONNECTION_INTERRUPTED`): the compiler crashed, as opposed to rejecting
//! the shader with a diagnostic or outliving the timeout. [`MetalCompileFailure::classify`] reads
//! those words. It is a label for the sweep's report, never a verdict: every refusal fails the
//! sweep whatever its class.

/// The step of MoltenVK's pipeline creation that failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetalCompileStep {
    /// SPIRV-Cross's conversion of the SPIR-V module to MSL.
    Conversion,
    /// Metal's front end building the MSL into a library.
    Library,
    /// Specializing the entry point with its function constants.
    Specialization,
    /// Building the compute or render pipeline state (Metal's GPU back end).
    Pipeline,
}

/// How the step failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetalFailureKind {
    /// The tool rejected its input with a diagnostic.
    Error,
    /// Metal's compiler service lost its connection: the compiler process died.
    ServiceCrash,
    /// The step outlived MoltenVK's compile timeout (`MVK_CONFIG_METAL_COMPILE_TIMEOUT`).
    Timeout,
}

/// One refusal's class: the step that failed and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MetalCompileFailure {
    pub step: MetalCompileStep,
    pub kind: MetalFailureKind,
}

impl MetalCompileFailure {
    /// The class of the refusal whose driver report is `detail`, or `None` when it names no
    /// MoltenVK compile step (another driver, or a refusal of another object).
    pub fn classify(detail: &str) -> Option<Self> {
        const CONVERSION: &str = "SPIR-V to MSL conversion error";
        const STEPS: [(&str, MetalCompileStep); 4] = [
            ("Shader library compile failed", MetalCompileStep::Library),
            (
                "Function specialization compile failed",
                MetalCompileStep::Specialization,
            ),
            (
                "Compute pipeline compile failed",
                MetalCompileStep::Pipeline,
            ),
            ("Render pipeline compile failed", MetalCompileStep::Pipeline),
        ];
        let kind = if detail.contains("XPC_ERROR_CONNECTION_INTERRUPTED") {
            MetalFailureKind::ServiceCrash
        } else if detail.contains("Likely internal Metal compiler error") {
            MetalFailureKind::Timeout
        } else {
            MetalFailureKind::Error
        };
        let step = if detail.contains(CONVERSION) {
            MetalCompileStep::Conversion
        } else {
            // The first step the report names: Metal stops at the step that failed.
            STEPS
                .iter()
                .filter_map(|(words, step)| detail.find(words).map(|at| (at, *step)))
                .min()
                .map(|(_, step)| step)?
        };
        Some(MetalCompileFailure { step, kind })
    }

    /// The class's report key, `<step>:<kind>`.
    pub fn key(&self) -> String {
        let step = match self.step {
            MetalCompileStep::Conversion => "conversion",
            MetalCompileStep::Library => "library",
            MetalCompileStep::Specialization => "specialization",
            MetalCompileStep::Pipeline => "pipeline",
        };
        let kind = match self.kind {
            MetalFailureKind::Error => "error",
            MetalFailureKind::ServiceCrash => "service_crash",
            MetalFailureKind::Timeout => "timeout",
        };
        format!("{step}:{kind}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(detail: &str) -> Option<String> {
        MetalCompileFailure::classify(detail).map(|class| class.key())
    }

    /// The reports MoltenVK 1.4.1 sends, as the acceptance sweep keeps them (the validation
    /// layer's messenger prefixes the level and message id).
    #[test]
    fn moltenvk_reports_classify_by_step_and_kind() {
        // Metal's compiler service died building the pipeline state (the trigger).
        assert_eq!(
            class(
                "error mvk-error: VK_ERROR_INITIALIZATION_FAILED: Compute pipeline compile failed \
                 (Error code 2):\nCompilation failed due to an interrupted connection: \
                 XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries.."
            )
            .as_deref(),
            Some("pipeline:service_crash")
        );
        // A back-end diagnostic without a crash.
        assert_eq!(
            class(
                "error mvk-error: VK_ERROR_INITIALIZATION_FAILED: Render pipeline compile failed \
                 (Error code 3):\nFragment input(s) `user(locn0)` mismatching vertex shader output"
            )
            .as_deref(),
            Some("pipeline:error")
        );
        // The front end rejected the MSL, and the pipeline step then reported it too: the first
        // step named is the one that failed.
        assert_eq!(
            class(
                "error mvk-error: VK_ERROR_INITIALIZATION_FAILED: Shader library compile failed \
                 (Error code 3):\nprogram_source:12:5: error: use of undeclared identifier | \
                 error mvk-error: VK_ERROR_INITIALIZATION_FAILED: Compute pipeline compile failed"
            )
            .as_deref(),
            Some("library:error")
        );
        assert_eq!(
            class(
                "error mvk-error: VK_ERROR_INITIALIZATION_FAILED: Function specialization compile \
                 failed (Error code 5): function constant mismatch"
            )
            .as_deref(),
            Some("specialization:error")
        );
        assert_eq!(
            class(
                "error mvk-error: VK_ERROR_INITIALIZATION_FAILED: SPIR-V to MSL conversion \
                 error: Unsupported capability"
            )
            .as_deref(),
            Some("conversion:error")
        );
        // MoltenVK's own timeout, in its `"%s compile failed (Error code %li):\n%s."` format.
        assert_eq!(
            class(
                "error mvk-error: VK_ERROR_INITIALIZATION_FAILED: Compute pipeline compile failed \
                 (Error code 1):\nTimeout after 5000.000 milliseconds. Likely internal Metal \
                 compiler error."
            )
            .as_deref(),
            Some("pipeline:timeout")
        );
        assert_eq!(
            class(
                "error mvk-error: VK_ERROR_INITIALIZATION_FAILED: Shader library compile failed \
                 (Error code 1):\nTimeout after 120000.000 milliseconds. Likely internal Metal \
                 compiler error."
            )
            .as_deref(),
            Some("library:timeout")
        );
        // Not a MoltenVK compile report: no class.
        assert_eq!(class("create pipeline: ERROR_OUT_OF_HOST_MEMORY"), None);
        assert_eq!(
            class("error VUID-vkCreateComputePipelines-x: bad layout"),
            None
        );
    }
}
