//! Shard 1 of the ledger's coverage of the owned fixtures' compiles (`common/ledger_coverage.rs`).
//!
//! This binary installs the worker's counting allocator and runs one test, so no other test
//! thread allocates while a compile is measured and the allocator peak is exactly the compile's.

use metal2vulkan_validation::worker::alloc::CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[path = "common/ledger_coverage.rs"]
mod ledger_coverage;

#[test]
fn ledger_covers_the_allocator_peak_of_every_owned_fixture_compile() {
    ledger_coverage::ledger_covers_the_allocator_peak_of_every_owned_fixture_compile(1);
}
