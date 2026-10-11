//! The single generic graph-algorithm module: RPO, dominators, loop forest, SCC.
//!
//! Every graph algorithm of the new core lives here and runs over the [`Graph`] trait: the AIR IR
//! (`air::ir`), the structurizer (`structurize`), and the SPIR-V verifier (`spirv::verify`) all
//! use these implementations and keep no private copy.
//!
//! - [`Cfg`]: a frozen, compressed (CSR) copy of a [`Graph`] with predecessors, one iterative DFS
//!   (preorder, DFS-tree intervals), and reverse postorder. Built once, reused by every analysis.
//! - [`DomTree`]: Semi-NCA dominators; dominance queries are O(1) through dominator-tree
//!   preorder intervals. [`PostDomTree`]: the same algorithm on the reverse graph with a virtual
//!   exit.
//! - [`LoopForest`]: the cycle nesting forest (reducible natural loops and irreducible
//!   multi-entry cycles), each cycle headed by its DFS-first block with its entries listed.
//!   Irreducible nesting deeper than `Limit::IrreducibleDepth` is refused.
//! - [`Sccs`]: iterative Tarjan strongly connected components over any [`Graph`].
//!
//! # Budget
//!
//! Every algorithm takes the per-attempt [`Meter`]. Work units are counted per visited node, per
//! scanned edge, and per step of any inner walk (so the count is proportional to the real work
//! and deterministic for a given graph), and polled in batches. Every input-scaled container is
//! charged to the byte ledger before it is reserved. Temporary buffers are released from the
//! ledger when the algorithm returns; the returned structure reports what it keeps charged through
//! its `ledger_bytes()` so its owner can [`Meter::release`] it when dropping the structure.
//!
//! # Recursion
//!
//! No algorithm recurses on an input-scaled depth: all traversals use explicit stacks sized by
//! the node count, so a 10⁶-block chain runs on a small thread stack.
//!
//! # Node contract
//!
//! Nodes are dense indices `0..node_count` ([`NodeId`]). Query methods on the analysis structures
//! take nodes of the graph they were computed from; a node outside `0..node_count` is a caller
//! bug and panics. Building from a [`Graph`] that breaks its own contract (a successor out of
//! range, an entry out of range, successors that change between calls) returns
//! `Error::Internal(Invariant)`; it never panics.

use std::fmt::Debug;
use std::hash::Hash;

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind, Limit};

mod cfg;
mod dom;
mod loops;
mod scc;

pub use cfg::Cfg;
pub use dom::{DomTree, PostDomTree};
pub use loops::{LoopForest, LoopId};
pub use scc::Sccs;

/// A dense node index type (e.g. a block id). `from_node_index(i).node_index() == i`.
pub trait NodeId: Copy + Eq + Ord + Hash + Debug {
    /// The node with dense index `index`.
    fn from_node_index(index: u32) -> Self;
    /// The dense index of this node.
    fn node_index(self) -> u32;
}

impl NodeId for u32 {
    fn from_node_index(index: u32) -> u32 {
        index
    }

    fn node_index(self) -> u32 {
        self
    }
}

/// A directed graph with a distinguished entry over dense nodes `0..node_count`.
///
/// Contract: `node_count() >= 1`, `entry()` and every successor are `< node_count()`, and
/// `successors(n)` yields the same sequence every time it is called. Duplicate successors (one
/// edge per `CondBr(c, t, t)` arm, say) and self-loops are allowed; every algorithm treats the
/// edge list as a multiset. The successor order is significant only where a result is defined by
/// the DFS (reverse postorder, the header of an irreducible cycle): it is always the order given
/// here, so results are deterministic.
pub trait Graph {
    /// The node type.
    type Node: NodeId;

    /// The number of nodes; nodes are `0..node_count()`.
    fn node_count(&self) -> usize;

    /// The entry node.
    fn entry(&self) -> Self::Node;

    /// The successors of `node`, in branch order, with duplicates.
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> + '_;
}

/// "No node / no loop" in the dense `u32` side tables.
const NONE: u32 = u32::MAX;

/// A broken [`Graph`] contract or an internal inconsistency.
fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// A `usize` count or index as a node-table `u32`, refusing sizes the `u32` tables cannot hold.
fn to_u32(value: usize, what: &str) -> Result<u32, Error> {
    match u32::try_from(value) {
        Ok(v) if v != NONE => Ok(v),
        _ => Err(invariant(format!(
            "{what} ({value}) exceeds the u32 graph index space"
        ))),
    }
}

/// Work-unit batching: counts deterministic work locally and ticks the meter every
/// [`Work::BATCH`] units, so the per-edge count does not pay a clock read per edge.
struct Work {
    pending: u32,
}

impl Work {
    const BATCH: u32 = 1024;

    fn new() -> Work {
        Work { pending: 0 }
    }

    /// Count `units` of work, polling the meter once a batch is full.
    fn step(&mut self, meter: &mut Meter, units: u32) -> Result<(), Error> {
        self.pending = self.pending.saturating_add(units);
        if self.pending >= Self::BATCH {
            let units = self.pending;
            self.pending = 0;
            meter.tick(units)?;
        }
        Ok(())
    }

    /// Tick the remainder, so the total counted is exact.
    fn finish(self, meter: &mut Meter) -> Result<(), Error> {
        if self.pending == 0 {
            Ok(())
        } else {
            meter.tick(self.pending)
        }
    }
}

/// Byte-ledger bookkeeping for the vectors one structure (or one algorithm's scratch) allocates.
struct Ledger {
    bytes: usize,
}

impl Ledger {
    fn new() -> Ledger {
        Ledger { bytes: 0 }
    }

    /// An empty vector with room for `capacity` elements, charged before reserving.
    fn with_capacity<T>(&mut self, meter: &mut Meter, capacity: usize) -> Result<Vec<T>, Error> {
        let mut v = Vec::new();
        self.grow(meter, &mut v, capacity)?;
        Ok(v)
    }

    /// Charge `additional` elements of `v` to the meter and to this ledger, then reserve them.
    /// The charge is recorded here before the allocation is attempted, so a failed reservation
    /// is still released with the rest of this ledger.
    fn grow<T>(
        &mut self,
        meter: &mut Meter,
        v: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), Error> {
        let bytes = additional
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(Error::LimitExceeded(Limit::LedgerBytes))?;
        meter.charge(bytes)?;
        self.bytes = self.bytes.saturating_add(bytes);
        v.try_reserve(additional).map_err(|e| {
            Error::Internal(InternalError::new(
                InternalKind::AllocationFailed,
                format!("reserving {bytes} ledger-admitted bytes: {e}"),
            ))
        })
    }

    /// A vector of `len` copies of `value`, charged before reserving.
    fn filled<T: Clone>(
        &mut self,
        meter: &mut Meter,
        len: usize,
        value: T,
    ) -> Result<Vec<T>, Error> {
        let mut v = self.with_capacity(meter, len)?;
        v.resize(len, value);
        Ok(v)
    }

    /// Push `value`, growing `v` by doubling (at least 16) and charging each growth before it is
    /// reserved. For containers whose size is not bounded by the node count up front.
    fn push<T>(&mut self, meter: &mut Meter, v: &mut Vec<T>, value: T) -> Result<(), Error> {
        if v.len() == v.capacity() {
            let additional = v.capacity().max(16);
            self.grow(meter, v, additional)?;
        }
        v.push(value);
        Ok(())
    }

    /// Return everything this ledger charged (scratch buffers going out of scope).
    fn release(self, meter: &mut Meter) {
        meter.release(self.bytes);
    }
}

/// Run one algorithm with its two ledgers: `keep` for the vectors of the returned structure and
/// `scratch` for temporaries. Scratch is released on every path. On success the kept bytes stay
/// charged (the body records them in the structure's `ledger_bytes`, and its owner releases them
/// when dropping it); on error they are released too, so a refused or failed computation leaves
/// the meter's ledger exactly as it found it.
fn metered<T>(
    meter: &mut Meter,
    body: impl FnOnce(&mut Meter, &mut Ledger, &mut Ledger) -> Result<T, Error>,
) -> Result<T, Error> {
    let mut keep = Ledger::new();
    let mut scratch = Ledger::new();
    let result = body(meter, &mut keep, &mut scratch);
    scratch.release(meter);
    if result.is_err() {
        keep.release(meter);
    }
    result
}

/// A slice index from a `u32` table entry.
fn ix(v: u32) -> usize {
    v as usize
}

#[cfg(test)]
mod tests;
