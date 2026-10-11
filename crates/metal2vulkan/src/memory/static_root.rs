//! The walking skeleton's entry into the memory model, kept as the stable path the skeleton's
//! callers import.
//!
//! Its analysis was replaced by the provenance lattice ([`super::plan::analyze_memory`],
//! re-exported here with [`MemoryPlan`]) and its word view and leaf store by the descriptor views
//! ([`super::views`]), leaf accesses ([`super::leaf`]), and containment encodings
//! ([`super::contain`]). What remains is the stage of an entry.

#[cfg(test)]
pub use super::plan::analyze_memory;
pub use super::plan::MemoryPlan;

use crate::air::Entry;
use crate::pipeline::sym::Stage;

/// The pipeline stage of an AIR entry point.
pub fn stage_of(e: &Entry) -> Stage {
    match e.ep.stage {
        crate::air::abi::Stage::Kernel { .. } => Stage::Kernel,
        crate::air::abi::Stage::Vertex { .. } => Stage::Vertex,
        crate::air::abi::Stage::Fragment { .. } => Stage::Fragment,
        crate::air::abi::Stage::Object { .. } => Stage::Object,
        crate::air::abi::Stage::Mesh { .. } => Stage::Mesh,
    }
}
