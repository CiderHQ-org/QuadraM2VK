//! P5: loop normalization.
//!
//! Every loop `L` of the prepared graph (single-entry after P1) is given control's
//! canonical shape, which the tree states directly:
//!
//! 1. a fresh empty header `Hs` entered from outside only (the `Stmt::Loop` itself; the old header
//!    `h` becomes the first block of the body);
//! 2. one continue block `C → Hs`: every back edge `b → h` becomes `Continue`;
//! 3. one fresh merge `E` holding the exit dispatch: every exit edge `u → T_j` becomes
//!    `SetExit(L, j); Break` (the `SetExit` is omitted when `L` has one exit target), and `E`
//!    dispatches on the loop's exit selector to `T_1..T_k` (nothing when `k = 0`, a plain edge when
//!    `k = 1`). The dispatch edges are ordinary edges of the parent region: an exit target outside
//!    the parent too becomes the parent's exit, and the parent's header its `continue`, so
//!    multi-level `break`/`continue` become one dispatch per crossed loop (the exit ladder).
//!
//! In the parent region `L` collapses to one node whose successors are its exit targets. P6 builds
//! the regions; this module owns what normalization decides: the exit selectors (allocated
//! innermost loop first) and the classification of every edge target seen from a region
//! (`Normalizer::classify`).

use super::cfg::{at, at_mut, invariant, to_u32, Node, SelVar};
use super::loops::{Kept, LoopIx, Loops, NONE};
use crate::budget::Meter;
use crate::error::Error;

/// Where an edge of a region goes after normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Target {
    /// A node of the region: an own block, or the header of a loop directly nested in it (the
    /// collapsed loop node).
    Local(Node),
    /// The region's loop header: `Continue`.
    Continue,
    /// Exit target `j` (RPO order) of the region's loop: `SetExit(L, j); Break`.
    Exit(u32),
}

/// P5's decisions, and the per-region edge classifier P6 uses.
#[derive(Debug)]
pub(super) struct Normalizer {
    /// Loop -> its exit selector (`NONE` when it has fewer than two exit targets).
    exit_sel: Vec<u32>,
    /// Node -> its index among the current region loop's exit targets (valid only for them).
    exit_slot: Vec<u32>,
    region: Option<LoopIx>,
    selectors: u32,
    kept: Kept,
}

impl Normalizer {
    /// Allocate an exit selector for every loop with two or more exit targets, innermost loop
    /// first, numbered from `first_selector`. On error nothing stays charged.
    pub(super) fn new(
        loops: &Loops,
        nodes: usize,
        first_selector: u32,
        meter: &mut Meter,
    ) -> Result<Normalizer, Error> {
        let mut kept = Kept::default();
        let tables = (|| {
            let mut exit_sel = kept.filled(meter, loops.len(), NONE)?;
            let exit_slot = kept.filled(meter, nodes, NONE)?;
            let mut next = first_selector;
            for l in loops.innermost_first() {
                meter.tick(1)?;
                if loops.exits(l).len() >= 2 {
                    *at_mut(&mut exit_sel, l.index())? = next;
                    next = next
                        .checked_add(1)
                        .filter(|&n| n != NONE)
                        .ok_or_else(|| invariant("selector count exceeds u32"))?;
                }
            }
            Ok((exit_sel, exit_slot, next))
        })();
        match tables {
            Ok((exit_sel, exit_slot, selectors)) => Ok(Normalizer {
                exit_sel,
                exit_slot,
                region: None,
                selectors,
                kept,
            }),
            Err(e) => {
                meter.release(kept.bytes());
                Err(e)
            }
        }
    }

    /// The number of selector variables after P5 (P1's included).
    pub(super) fn selectors(&self) -> u32 {
        self.selectors
    }

    /// The exit selector of `l`, when it has two or more exit targets.
    pub(super) fn exit_sel(&self, l: LoopIx) -> Option<SelVar> {
        self.exit_sel
            .get(l.index())
            .copied()
            .filter(|&s| s != NONE)
            .map(SelVar::new)
    }

    /// Classify edges from the region of `region` (the function body for `None`) from now on.
    pub(super) fn enter(
        &mut self,
        loops: &Loops,
        region: Option<LoopIx>,
        meter: &mut Meter,
    ) -> Result<(), Error> {
        self.region = region;
        if let Some(r) = region {
            for (j, t) in loops.exits(r).enumerate() {
                meter.tick(1)?;
                *at_mut(&mut self.exit_slot, t.index())? = to_u32(j)?;
            }
        }
        Ok(())
    }

    /// Where an edge of the current region to `t` goes. An edge into a nested loop other than
    /// at the header of a loop directly nested in the region breaks the single-entry invariant
    /// P1 establishes: `Error::Internal(Invariant)`.
    pub(super) fn classify(&self, loops: &Loops, t: Node) -> Result<Target, Error> {
        if let Some(r) = self.region {
            if loops.header(r) == Some(t) {
                return Ok(Target::Continue);
            }
            if !loops.contains(r, t) {
                return Ok(Target::Exit(*at(&self.exit_slot, t.index())?));
            }
        }
        let inner = loops.innermost(t);
        if inner == self.region {
            return Ok(Target::Local(t));
        }
        match loops.loop_of_header(t) {
            Some(c) if loops.parent(c) == self.region => Ok(Target::Local(t)),
            _ => Err(invariant(format!(
                "edge into {t:?} enters loop {inner:?} below region {:?} not at a child header",
                self.region
            ))),
        }
    }

    /// Return the ledger bytes this normalizer holds.
    pub(super) fn release(self, meter: &mut Meter) {
        meter.release(self.kept.bytes());
    }
}
