//! The same computation written across helpers and written flat (the dev-tools crate's
//! `fuzz::inline`), both compared against the computation done in Rust.

use super::gpu_fuzz::engine;
use metal2vulkan_devtools::fuzz::{inline, GATE_SEEDS};

/// Cyclic on a GPU too: the engine's modules run as compiled.
#[test]
fn a_helper_computes_what_the_same_body_written_flat_computes_on_the_device() {
    inline::check(&mut engine(), GATE_SEEDS).assert_clean("inline (engine, device)", 12);
}
