//! Threadgroup memory, barriers, and threadgroup atomics (the dev-tools crate's `fuzz::workgroup`),
//! compared against the same rounds computed in Rust.

use super::gpu_fuzz::engine;
use metal2vulkan_devtools::fuzz::{workgroup, GATE_SEEDS};

#[test]
fn a_threadgroup_slot_holds_what_its_own_thread_wrote_on_the_device() {
    workgroup::check(&mut engine(), GATE_SEEDS).assert_clean("workgroup (engine, device)", 12);
}
