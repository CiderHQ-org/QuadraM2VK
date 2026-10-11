//! Two derivations of one address over generated aggregate layouts, a whole-buffer store check, and
//! the sub-word store race (the dev-tools crate's `fuzz::access`), compared byte for byte against
//! the layout rules computed in Rust.

use super::gpu_fuzz::engine;
use metal2vulkan_devtools::fuzz::{access, GATE_SEEDS};

#[test]
fn a_typed_access_chain_and_a_raw_byte_offset_name_the_same_word_on_the_device() {
    access::check_load(&mut engine(), GATE_SEEDS).assert_clean("access load (engine, device)", 12);
}

#[test]
fn a_stored_leaf_lands_on_its_own_bytes_and_no_others_on_the_device() {
    access::check_store(&mut engine(), GATE_SEEDS)
        .assert_clean("access store (engine, device)", 12);
}

#[test]
fn one_threads_subword_store_does_not_erase_another_threads_on_the_device() {
    access::check_race(&mut engine()).assert_clean("access race (engine, device)", 12);
}
