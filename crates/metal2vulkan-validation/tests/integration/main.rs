//! The validation crate's integration tests, one test binary for all of them.
//!
//! Each module below was its own test binary. Every binary of this crate's tests links the whole
//! translator, the runtime, and validation, so one binary instead of sixteen saves fifteen links
//! (and their codegen and disk) on every build of the suite. nextest still runs each test in its
//! own process. A test that installs its own global allocator to measure a compile alone
//! (`ledger_call_payloads.rs`, the `ledger_coverage_<k>.rs` shards) keeps a binary of its own:
//! a binary has one global allocator, and those tests need no other test thread allocating while
//! they measure.

mod audit_worker;
mod authored_compile_census;
mod bins;
mod committed_public_cases_check;
mod committed_shards_keep_the_store_layout;
mod compile_worker;
mod device_worker;
mod drm_shim;
mod engine_linked_cases;
mod fast_trig_within_apples_bound;
mod fixtures_assemble_with_the_pinned_llvm;
mod linux_gate_policy;
mod metal_coverage;
mod pipeline_selection;
mod private_corpus_root;
mod public_entry_rows;
mod published_crates;
mod translation_l3_stage;
