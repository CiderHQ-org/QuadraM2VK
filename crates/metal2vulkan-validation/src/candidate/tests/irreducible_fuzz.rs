//! Generated IRREDUCIBLE control flow (the dev-tools crate's `fuzz::irreducible`): an automaton
//! whose single cycle is entered at every one of its blocks, compared against the automaton stepped
//! in Rust.

use super::gpu_fuzz::engine;
use metal2vulkan_devtools::fuzz::{irreducible, GATE_SEEDS};

/// Cyclic on a GPU too: the engine's modules run as compiled.
#[test]
fn generated_irreducible_cycles_step_the_same_automaton_on_the_device() {
    irreducible::check(&mut engine(), GATE_SEEDS).assert_clean("irreducible (engine, device)", 12);
}
