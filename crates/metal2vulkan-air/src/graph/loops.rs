//! [`LoopForest`]: the cycle nesting forest of reducible and irreducible loops.

use std::marker::PhantomData;

use super::{ix, metered, to_u32, Cfg, Ledger, NodeId, Work, NONE};
use crate::budget::Meter;
use crate::error::{Error, Limit};

/// A loop (cycle) of a [`LoopForest`]. Ids are dense, in a preorder walk of the forest: a parent
/// precedes its children, and a loop's descendants are the ids `l+1 .. end(l)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopId(u32);

impl LoopId {
    /// The dense index, for side tables.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// The loop nesting forest of the reachable part of a [`Cfg`] (the "cycle" forest of LLVM's
/// `GenericCycleInfo`, computed by the same DFS-based algorithm).
///
/// - The outermost loops are the reachable strongly connected components that contain a cycle
///   (two or more blocks, or one block with a self-loop).
/// - Every loop has a **header**: its block of least DFS preorder. Its **entries** are the header
///   plus every block with a predecessor outside the loop. A loop with one entry is reducible, and
///   is then exactly the natural loop of its header (the header dominates it; its back edges are
///   the edges into the header from inside).
/// - The loops nested in `L` are the outermost loops of `L`'s blocks with the header removed.
///
/// A graph is reducible iff every loop is ([`LoopForest::is_reducible_graph`]). Irreducible
/// loops keep all of their entries, so the structurizer can dispatch them.
///
/// Discovery processes header candidates in reverse DFS preorder and collapses already found
/// loops through a union-find over loops. A block is an entry of a chain of nested loops that
/// are all irreducible (except the one it heads), so the entry lists total O(n·(d + 2)) for an
/// irreducible nesting depth `d`. `d` is capped at [`Limit::IrreducibleDepth`] (the structurizer
/// refuses deeper nests), and the cap is checked as each loop is completed, innermost first, so the
/// work is O((n + e)·α) for reducible graphs and O((n + e)·(d + 2)) in general, and a deeper nest
/// is refused as `LimitExceeded(IrreducibleDepth)` before its cost grows.
#[derive(Clone, Debug)]
pub struct LoopForest<N> {
    /// LoopId -> header node.
    header: Vec<u32>,
    /// LoopId -> parent LoopId (`NONE` for outermost loops).
    parent: Vec<u32>,
    /// LoopId -> one past its last descendant LoopId.
    end: Vec<u32>,
    /// Entries CSR by LoopId (header first, then by DFS preorder).
    entry_start: Vec<u32>,
    entries: Vec<u32>,
    /// LoopId -> its blocks (nested loops included): `blocks[block_start..block_end]`.
    block_start: Vec<u32>,
    block_end: Vec<u32>,
    blocks: Vec<u32>,
    /// Node -> innermost LoopId containing it (`NONE` when in no loop).
    innermost: Vec<u32>,
    irreducible: usize,
    ledger_bytes: usize,
    node: PhantomData<fn() -> N>,
}

/// Union-find `find` with path halving over loop creation ids.
fn find(uf: &mut [u32], mut x: u32, work: &mut Work, meter: &mut Meter) -> Result<u32, Error> {
    while uf[ix(x)] != x {
        let grand = uf[ix(uf[ix(x)])];
        uf[ix(x)] = grand;
        x = grand;
        work.step(meter, 1)?;
    }
    Ok(x)
}

/// Mutable discovery state for one header candidate.
struct Discovery<'a> {
    pre_of: &'a [u32],
    /// The candidate's DFS subtree interval.
    lo: u32,
    hi: u32,
    worklist: Vec<u32>,
    entries: Vec<u32>,
}

impl<N: NodeId> LoopForest<N> {
    /// The loop forest of `cfg`.
    ///
    /// On error nothing stays charged to `meter`'s ledger.
    pub fn compute(cfg: &Cfg<N>, meter: &mut Meter) -> Result<LoopForest<N>, Error> {
        metered(meter, |meter, keep, scratch| {
            Self::compute_in(cfg, meter, keep, scratch)
        })
    }

    fn compute_in(
        cfg: &Cfg<N>,
        meter: &mut Meter,
        keep: &mut Ledger,
        scratch: &mut Ledger,
    ) -> Result<LoopForest<N>, Error> {
        let mut work = Work::new();

        let count = cfg.node_count();
        let pre_of = cfg.pre_raw();
        let order = cfg.order_raw();
        let subtree_end = cfg.subtree_end_raw();
        let r = order.len();

        // Discovery, over creation ids (created in reverse preorder of their headers).
        let mut innermost_c = scratch.filled(meter, count, NONE)?;
        let mut uf = scratch.with_capacity::<u32>(meter, r)?;
        let mut c_parent = scratch.with_capacity::<u32>(meter, r)?;
        let mut c_header = scratch.with_capacity::<u32>(meter, r)?;
        let mut c_entries = scratch.with_capacity::<(u32, u32)>(meter, r)?;
        // Creation id -> the most irreducible loops on one nesting chain ending at it.
        let mut c_irreducible_height = scratch.with_capacity::<u32>(meter, r)?;
        let mut d = Discovery {
            pre_of,
            lo: 0,
            hi: 0,
            worklist: scratch.with_capacity::<u32>(meter, r)?,
            entries: scratch.with_capacity::<u32>(meter, r)?,
        };

        for i in (0..r).rev() {
            let h = order[i];
            d.lo = to_u32(i, "preorder number")?;
            d.hi = subtree_end[i];
            d.worklist.clear();
            for &p in cfg.pred_raw(h) {
                work.step(meter, 1)?;
                if d.in_subtree(p) {
                    scratch.push(meter, &mut d.worklist, p)?;
                }
            }
            if d.worklist.is_empty() {
                continue;
            }
            let c = to_u32(c_header.len(), "loop count")?;
            c_header.push(h);
            uf.push(c);
            c_parent.push(NONE);
            let first_entry = to_u32(d.entries.len(), "entry count")?;
            scratch.push(meter, &mut d.entries, h)?;
            innermost_c[ix(h)] = c;
            let mut nested_height = 0u32;

            while let Some(b) = d.worklist.pop() {
                work.step(meter, 1)?;
                if b == h {
                    continue;
                }
                let inner = innermost_c[ix(b)];
                if inner == NONE {
                    innermost_c[ix(b)] = c;
                    d.process(cfg, b, scratch, &mut work, meter)?;
                    continue;
                }
                let top = find(&mut uf, inner, &mut work, meter)?;
                if top != c {
                    // `top` is an outermost loop found earlier: nest it in `c` and continue from
                    // its entries (its other blocks have predecessors only inside it).
                    uf[ix(top)] = c;
                    c_parent[ix(top)] = c;
                    nested_height = nested_height.max(c_irreducible_height[ix(top)]);
                    let (from, to) = c_entries[ix(top)];
                    for at in from..to {
                        let e = d.entries[ix(at)];
                        d.process(cfg, e, scratch, &mut work, meter)?;
                    }
                }
            }
            let last_entry = to_u32(d.entries.len(), "entry count")?;
            c_entries.push((first_entry, last_entry));
            let height = nested_height + u32::from(last_entry - first_entry > 1);
            meter.cap(Limit::IrreducibleDepth, u64::from(height))?;
            c_irreducible_height.push(height);
        }
        let m = c_header.len();

        // Loop-tree children (ascending header preorder = descending creation id) and the
        // direct blocks of every loop (ascending preorder).
        let mut child_start = scratch.filled(meter, m + 1, 0u32)?;
        for &p in &c_parent {
            if p != NONE {
                child_start[ix(p) + 1] += 1;
            }
        }
        for c in 0..m {
            child_start[c + 1] += child_start[c];
        }
        let mut cursor = scratch.with_capacity::<u32>(meter, m)?;
        cursor.extend_from_slice(&child_start[..m]);
        let mut children = scratch.filled(meter, m, 0u32)?;
        let mut roots = scratch.with_capacity::<u32>(meter, m)?;
        for c in (0..m).rev() {
            let p = c_parent[c];
            let c = to_u32(c, "loop count")?;
            if p == NONE {
                roots.push(c);
            } else {
                let at = &mut cursor[ix(p)];
                children[ix(*at)] = c;
                *at += 1;
            }
        }
        let mut direct_start = scratch.filled(meter, m + 1, 0u32)?;
        for &v in order {
            let c = innermost_c[ix(v)];
            if c != NONE {
                direct_start[ix(c) + 1] += 1;
            }
        }
        for c in 0..m {
            direct_start[c + 1] += direct_start[c];
        }
        cursor.clear();
        cursor.extend_from_slice(&direct_start[..m]);
        let mut direct = scratch.filled(meter, ix(direct_start[m]), 0u32)?;
        for &v in order {
            let c = innermost_c[ix(v)];
            if c != NONE {
                let at = &mut cursor[ix(c)];
                direct[ix(*at)] = v;
                *at += 1;
            }
            work.step(meter, 1)?;
        }

        // Preorder walk of the loop tree: final ids, intervals, block ranges, entries.
        let mut new_id = scratch.filled(meter, m, NONE)?;
        let mut header = keep.with_capacity::<u32>(meter, m)?;
        let mut parent = keep.with_capacity::<u32>(meter, m)?;
        let mut end = keep.filled(meter, m, 0u32)?;
        let mut entry_start = keep.with_capacity::<u32>(meter, m + 1)?;
        let mut entries = keep.with_capacity::<u32>(meter, d.entries.len())?;
        let mut block_start = keep.with_capacity::<u32>(meter, m)?;
        let mut block_end = keep.filled(meter, m, 0u32)?;
        let mut blocks = keep.with_capacity::<u32>(meter, direct.len())?;
        let mut stack = scratch.with_capacity::<(u32, u32)>(meter, m)?;
        let mut irreducible = 0usize;
        entry_start.push(0);

        for &root in &roots {
            stack.push((root, NONE));
            while let Some(top) = stack.last_mut() {
                let (c, at) = *top;
                if at == NONE {
                    // Enter `c`.
                    top.1 = child_start[ix(c)];
                    let id = to_u32(header.len(), "loop count")?;
                    new_id[ix(c)] = id;
                    let p = c_parent[ix(c)];
                    let p_id = if p == NONE { NONE } else { new_id[ix(p)] };
                    header.push(c_header[ix(c)]);
                    parent.push(p_id);
                    block_start.push(to_u32(blocks.len(), "block count")?);
                    blocks.extend_from_slice(
                        &direct[ix(direct_start[ix(c)])..ix(direct_start[ix(c) + 1])],
                    );
                    let (from, to) = c_entries[ix(c)];
                    let own = &d.entries[ix(from)..ix(to)];
                    let first = entries.len();
                    entries.extend_from_slice(own);
                    if let Some(rest) = entries.get_mut(first + 1..) {
                        rest.sort_unstable_by_key(|&e| pre_of[ix(e)]);
                    }
                    if own.len() > 1 {
                        irreducible += 1;
                    }
                    entry_start.push(to_u32(entries.len(), "entry count")?);
                    work.step(meter, 1 + to - from)?;
                } else if at < child_start[ix(c) + 1] {
                    top.1 += 1;
                    stack.push((children[ix(at)], NONE));
                } else {
                    stack.pop();
                    let id = new_id[ix(c)];
                    end[ix(id)] = to_u32(header.len(), "loop count")?;
                    block_end[ix(id)] = to_u32(blocks.len(), "block count")?;
                }
            }
        }
        let mut innermost = keep.filled(meter, count, NONE)?;
        for (v, &c) in innermost_c.iter().enumerate() {
            if c != NONE {
                innermost[v] = new_id[ix(c)];
            }
        }

        drop((
            uf,
            c_parent,
            c_header,
            c_entries,
            c_irreducible_height,
            d,
            innermost_c,
        ));
        drop((
            child_start,
            cursor,
            children,
            roots,
            direct_start,
            direct,
            new_id,
            stack,
        ));
        work.finish(meter)?;
        Ok(LoopForest {
            header,
            parent,
            end,
            entry_start,
            entries,
            block_start,
            block_end,
            blocks,
            innermost,
            irreducible,
            ledger_bytes: keep.bytes,
            node: PhantomData,
        })
    }

    /// The number of loops.
    pub fn len(&self) -> usize {
        self.header.len()
    }

    /// Whether the graph has no loop.
    pub fn is_empty(&self) -> bool {
        self.header.is_empty()
    }

    /// All loops, outer before inner (forest preorder).
    pub fn loops(&self) -> impl DoubleEndedIterator<Item = LoopId> + ExactSizeIterator {
        (0..self.header.len() as u32).map(LoopId)
    }

    /// All loops, inner before outer (reverse forest preorder): every loop comes after all of
    /// its descendants.
    pub fn innermost_first(&self) -> impl Iterator<Item = LoopId> {
        self.loops().rev()
    }

    /// The outermost loops, in ascending header DFS preorder.
    #[cfg(test)]
    pub fn top_level(&self) -> impl Iterator<Item = LoopId> + '_ {
        self.siblings(0, self.header.len() as u32)
    }

    /// The loops directly nested in `l`, in ascending header DFS preorder.
    #[cfg(test)]
    pub fn children(&self, l: LoopId) -> impl Iterator<Item = LoopId> + '_ {
        self.siblings(l.0 + 1, self.end[l.index()])
    }

    #[cfg(test)]
    fn siblings(&self, first: u32, end: u32) -> impl Iterator<Item = LoopId> + '_ {
        std::iter::successors(Some(first).filter(|&c| c < end), move |&c| {
            Some(self.end[ix(c)]).filter(|&next| next < end)
        })
        .map(LoopId)
    }

    /// The header of `l`: its block of least DFS preorder (the unique entry when reducible).
    pub fn header(&self, l: LoopId) -> N {
        N::from_node_index(self.header[l.index()])
    }

    /// The loop `l` is directly nested in.
    pub fn parent(&self, l: LoopId) -> Option<LoopId> {
        Some(self.parent[l.index()])
            .filter(|&p| p != NONE)
            .map(LoopId)
    }

    /// Whether `a` is `b` or contains it.
    pub fn loop_contains(&self, a: LoopId, b: LoopId) -> bool {
        a <= b && b.0 < self.end[a.index()]
    }

    /// The entries of `l`: the header first, then the other blocks with a predecessor outside
    /// `l`, in ascending DFS preorder.
    pub fn entries(&self, l: LoopId) -> impl ExactSizeIterator<Item = N> + '_ {
        let range = ix(self.entry_start[l.index()])..ix(self.entry_start[l.index() + 1]);
        self.entries[range].iter().map(|&e| N::from_node_index(e))
    }

    /// Whether `l` has a single entry (its header).
    pub fn is_reducible(&self, l: LoopId) -> bool {
        self.entry_start[l.index() + 1] - self.entry_start[l.index()] == 1
    }

    /// Whether every loop is reducible, i.e. the reachable graph is reducible.
    pub fn is_reducible_graph(&self) -> bool {
        self.irreducible == 0
    }

    /// The number of irreducible (multi-entry) loops.
    #[cfg(test)]
    pub fn irreducible_count(&self) -> usize {
        self.irreducible
    }

    /// Every block of `l`, nested loops included: first the blocks whose innermost loop is `l`
    /// (ascending DFS preorder), then those of each child in order.
    pub fn blocks(&self, l: LoopId) -> impl ExactSizeIterator<Item = N> + '_ {
        let range = ix(self.block_start[l.index()])..ix(self.block_end[l.index()]);
        self.blocks[range].iter().map(|&v| N::from_node_index(v))
    }

    /// The innermost loop containing `node`, if any.
    pub fn innermost(&self, node: N) -> Option<LoopId> {
        Some(self.innermost[ix(node.node_index())])
            .filter(|&l| l != NONE)
            .map(LoopId)
    }

    /// Whether `node` is a block of `l` (possibly through a nested loop). O(1).
    pub fn contains(&self, l: LoopId, node: N) -> bool {
        self.innermost(node)
            .is_some_and(|m| self.loop_contains(l, m))
    }

    /// The loop `node` is the header of, if any.
    pub fn loop_of_header(&self, node: N) -> Option<LoopId> {
        self.innermost(node)
            .filter(|&l| self.header[l.index()] == node.node_index())
    }

    /// Bytes this structure keeps charged to the ledger it was built under.
    pub fn ledger_bytes(&self) -> usize {
        self.ledger_bytes
    }
}

impl Discovery<'_> {
    fn in_subtree(&self, v: u32) -> bool {
        let p = self.pre_of[ix(v)];
        p != NONE && self.lo <= p && p < self.hi
    }

    /// Queue the predecessors of loop block `b` that lie in the candidate's DFS subtree (they
    /// reach the header through `b`), and record `b` as an entry if any reachable predecessor
    /// lies outside it.
    fn process<N: NodeId>(
        &mut self,
        cfg: &Cfg<N>,
        b: u32,
        scratch: &mut Ledger,
        work: &mut Work,
        meter: &mut Meter,
    ) -> Result<(), Error> {
        let mut is_entry = false;
        for &p in cfg.pred_raw(b) {
            work.step(meter, 1)?;
            if self.in_subtree(p) {
                scratch.push(meter, &mut self.worklist, p)?;
            } else if self.pre_of[ix(p)] != NONE {
                is_entry = true;
            }
        }
        if is_entry {
            scratch.push(meter, &mut self.entries, b)?;
        }
        Ok(())
    }
}
