//! Pointers chosen at run time (the dev-tools crate's `fuzz::pointer_select`): chains of `select`s,
//! `phi`s, and trailing GEPs between buffers, loaded through and finally stored through, compared
//! word for word over the whole output buffer against the chain resolved in Rust. On the engine
//! some programs choose between pointers parked in a private aggregate (pointer cells) and, on a
//! device with device addresses, between pointers loaded from a pointer table (dynamic roots).

use super::gpu_fuzz::engine;
use metal2vulkan_devtools::fuzz::{pointer_select, GATE_SEEDS};

#[test]
fn a_selected_pointer_reads_and_writes_where_the_choice_says_on_the_device() {
    pointer_select::check(&mut engine(), GATE_SEEDS)
        .assert_clean("pointer select (engine, device)", 12);
}
