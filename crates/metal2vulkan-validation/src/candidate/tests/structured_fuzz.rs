//! Generated STRUCTURED control flow (the dev-tools crate's `fuzz::structured`): nested loops,
//! two-armed conditionals, multi-way switches, multi-level breaks and continues, early returns, and
//! a reducible cross-arm edge, each compared against the program interpreted in Rust.

use super::gpu_fuzz::engine;
use metal2vulkan_devtools::fuzz::{structured, GATE_SEEDS};

/// Cyclic on a GPU too: the engine's modules run as compiled.
#[test]
fn generated_structured_control_flow_computes_what_the_program_says_on_the_device() {
    structured::check(&mut engine(), GATE_SEEDS).assert_clean("structured (engine, device)", 12);
}
