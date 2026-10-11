//! Ideal tangles: the reference side of convergence equivalence.
//!
//! Each lane runs the source CFG alone under the testkit scheme ([`Hash`]), seeded per lane. An
//! execution of original block `B` is keyed by `B` and the iteration number of every loop
//! enclosing `B`; the ideal tangle of an execution is the set of lanes with an execution of the
//! same key. "Loop" means the loop nest after P0–P2 (the natural loops): that nest is where the
//! structurizer draws iteration boundaries, and C5 states that each such boundary is a `Loop` node
//! of the output.
//!
//! Irreducible inputs have no reference: P1's dispatch loops have no source-level iteration to
//! compare with, and a tangled irreducible region is refused (`IrreducibleConvergent`).
//!
//! Both SIMT models compare against this: the tree-level one in `structurize::emit` tests and the
//! SPIR-V interpreter's maximal reconvergence in `spirv::interp` tests.

use std::collections::{BTreeMap, BTreeSet};

use super::hash::Hash;
use super::interp::{End, Run};
use super::view::{TestCfg, TestKind};
use crate::air::ir::{BlockId, ValueId};
use crate::structurize::cfg::Content;
use crate::structurize::loops::LoopIx;
use crate::structurize::Prepared;

/// Per original block, the lane sets of its executions, sorted (so multisets compare as
/// vectors).
pub type Tangles = BTreeMap<BlockId, Vec<BTreeSet<usize>>>;

/// Sort each block's lane sets.
pub fn sorted(mut t: Tangles) -> Tangles {
    for sets in t.values_mut() {
        sets.sort();
    }
    t
}

/// Every lane's run alone and the ideal tangles of all their executions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ideal {
    /// Lane `i`'s run of the source, seeded with `seeds[i]`.
    pub runs: Vec<Run>,
    pub tangles: Tangles,
}

/// Whether P1 left `p` unchanged (every node is an original block): the graph was reducible, so
/// its executions have source-level loop iterations to key by.
pub fn has_source_iterations(p: &Prepared) -> bool {
    p.cfg
        .blocks()
        .all(|(_, b)| matches!(b.content, Content::Code(_)))
}

/// The ideal tangles of `g` (prepared as `p`) for one lane per seed, each executing at most
/// `events` original blocks; `None` when `g` has no source-level iterations
/// ([`has_source_iterations`]).
pub fn ideal(g: &TestCfg, p: &Prepared, seeds: &[u64], events: usize) -> Option<Ideal> {
    if !has_source_iterations(p) {
        return None;
    }
    let mut keys: BTreeMap<(BlockId, Vec<u32>), BTreeSet<usize>> = BTreeMap::new();
    let runs = seeds
        .iter()
        .enumerate()
        .map(|(lane, &seed)| run_keyed(g, p, seed, events, lane, &mut keys))
        .collect();
    let mut tangles: Tangles = BTreeMap::new();
    for ((b, _), lanes) in keys {
        tangles.entry(b).or_default().push(lanes);
    }
    Some(Ideal {
        runs,
        tangles: sorted(tangles),
    })
}

/// The loops of the prepared graph containing original block `b`, outermost first.
fn chain(p: &Prepared, b: BlockId) -> Vec<LoopIx> {
    let n = p.cfg.node_of(b).expect("reachable block");
    let mut out = Vec::new();
    let mut l = p.loops.innermost(n);
    while let Some(x) = l {
        out.push(x);
        l = p.loops.parent(x);
    }
    out.reverse();
    out
}

/// Run lane `lane` alone on the source CFG, adding its executions' keys to `keys`.
fn run_keyed(
    g: &TestCfg,
    p: &Prepared,
    seed: u64,
    events: usize,
    lane: usize,
    keys: &mut BTreeMap<(BlockId, Vec<u32>), BTreeSet<usize>>,
) -> Run {
    let mut h = Hash::seeded(seed);
    let mut trace = Vec::new();
    let mut stack: Vec<(LoopIx, u32)> = Vec::new();
    let mut b = BlockId::new(0);
    let end = loop {
        if trace.len() == events {
            break End::Cutoff;
        }
        trace.push(b);
        h.exec(b);
        let n = p.cfg.node_of(b).expect("reachable block");
        let loops = chain(p, b);
        while stack.last().is_some_and(|(l, _)| !p.loops.contains(*l, n)) {
            stack.pop();
        }
        assert!(
            stack.iter().zip(&loops).all(|((a, _), b)| a == b),
            "loop stack is not a prefix of the nest"
        );
        let depth = stack.len();
        if let Some((l, count)) = stack.last_mut() {
            if p.loops.header(*l) == Some(n) && loops.len() == depth {
                *count += 1;
            }
        }
        for l in &loops[stack.len()..] {
            assert_eq!(
                p.loops.header(*l),
                Some(n),
                "{l:?} entered below its header"
            );
            stack.push((*l, 1));
        }
        let key = (b, stack.iter().map(|(_, c)| *c).collect::<Vec<_>>());
        assert!(
            keys.entry(key).or_default().insert(lane),
            "a lane repeated a key"
        );
        let block = &g.blocks[b.raw() as usize];
        let v = ValueId::new(b.raw());
        b = match &block.kind {
            TestKind::Br => block.succ[0],
            TestKind::CondBr => block.succ[if h.cond(v) { 0 } else { 1 }],
            TestKind::Switch(literals) => {
                let sel = h.selector(v);
                match literals.iter().position(|&l| l == sel) {
                    Some(i) => block.succ[i + 1],
                    None => block.succ[0],
                }
            }
            TestKind::Return => break End::Returned(None),
            TestKind::ReturnValue => break End::Returned(Some(h.value(v))),
            TestKind::Unreachable => break End::Unreachable,
        };
    };
    Run { trace, end }
}
