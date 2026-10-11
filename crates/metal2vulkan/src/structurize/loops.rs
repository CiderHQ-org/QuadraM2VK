//! P2: the loop nest of the reducible working graph, and the final [`Loops`] that
//! the later passes read.
//!
//! [`LoopNest`] is the natural-loop forest from `graph::LoopForest` (every loop reducible after
//! P1, so each is the natural loop of its header) with the `Cfg` it was computed from: the
//! innermost loop of a block and the parent of a loop. Membership is the chain of parents from a
//! block's innermost loop. No block outside a natural loop joins it: a loop-exit tail runs after
//! its loop, as AGX runs it (which retires the P3). `LoopNest::finish` freezes the nest into
//! [`Loops`], with preorder loop ids, contiguous block ranges, O(1) membership, and each loop's
//! exit targets.

use super::cfg::StructCfg;
use super::cfg::{at, at_mut, invariant, to_u32, try_reserve, with_scratch, Node, Scratch};
use crate::budget::Meter;
use crate::error::Error;
use crate::graph::{Cfg, LoopForest};

pub(super) const NONE: u32 = u32::MAX;

/// A loop of a [`Loops`]. Ids are dense in a preorder walk of the nest: a parent precedes its
/// children, and a loop's descendants are the ids `l + 1 .. end(l)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoopIx(u32);

impl LoopIx {
    /// The dense index, for side tables.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Vectors charged to the ledger as a group, released together.
#[derive(Clone, Debug, Default)]
pub(super) struct Kept {
    bytes: usize,
}

impl Kept {
    pub(super) fn filled<T: Clone>(
        &mut self,
        meter: &mut Meter,
        len: usize,
        value: T,
    ) -> Result<Vec<T>, Error> {
        let mut v = self.with_capacity(meter, len)?;
        v.resize(len, value);
        Ok(v)
    }

    pub(super) fn with_capacity<T>(
        &mut self,
        meter: &mut Meter,
        capacity: usize,
    ) -> Result<Vec<T>, Error> {
        let bytes = capacity.saturating_mul(std::mem::size_of::<T>());
        meter.charge(bytes)?;
        self.bytes = self.bytes.saturating_add(bytes);
        let mut v = Vec::new();
        try_reserve(&mut v, capacity, bytes)?;
        Ok(v)
    }

    /// Push, growing by doubling (at least 16) and charging each growth first.
    pub(super) fn push<T>(&mut self, meter: &mut Meter, v: &mut Vec<T>, x: T) -> Result<(), Error> {
        if v.len() == v.capacity() {
            let additional = v.capacity().max(16);
            let bytes = additional.saturating_mul(std::mem::size_of::<T>());
            meter.charge(bytes)?;
            self.bytes = self.bytes.saturating_add(bytes);
            try_reserve(v, additional, bytes)?;
        }
        v.push(x);
        Ok(())
    }

    /// Charge `bytes` of heap owned by values kept with the group.
    pub(super) fn charge(&mut self, meter: &mut Meter, bytes: usize) -> Result<(), Error> {
        meter.charge(bytes)?;
        self.bytes = self.bytes.saturating_add(bytes);
        Ok(())
    }

    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}

/// P2's loop nest.
#[derive(Debug)]
pub struct LoopNest {
    cfg: Cfg<Node>,
    /// Loop -> header (loops in `graph::LoopForest` preorder).
    header: Vec<Node>,
    /// Loop -> parent loop (`NONE` for outermost).
    parent: Vec<u32>,
    /// Node -> innermost loop (`NONE` outside every loop).
    innermost: Vec<u32>,
    kept: Kept,
}

/// The tables of a [`LoopNest`], as P2 computes them.
struct NestTables {
    header: Vec<Node>,
    parent: Vec<u32>,
    innermost: Vec<u32>,
}

impl LoopNest {
    /// P2 over a reducible `sc`. On error nothing stays charged to `meter`'s ledger.
    pub fn compute(sc: &StructCfg, meter: &mut Meter) -> Result<LoopNest, Error> {
        let cfg = Cfg::build(sc, meter)?;
        let mut kept = Kept::default();
        let tables = LoopForest::compute(&cfg, meter)
            .map_err(Error::from)
            .and_then(|forest| {
                let tables = Self::tables(&cfg, &forest, &mut kept, meter);
                meter.release(forest.ledger_bytes());
                tables
            });
        match tables {
            Ok(t) => Ok(LoopNest {
                cfg,
                header: t.header,
                parent: t.parent,
                innermost: t.innermost,
                kept,
            }),
            Err(e) => {
                meter.release(kept.bytes());
                meter.release(cfg.ledger_bytes());
                Err(e)
            }
        }
    }

    fn tables(
        cfg: &Cfg<Node>,
        forest: &LoopForest<Node>,
        kept: &mut Kept,
        meter: &mut Meter,
    ) -> Result<NestTables, Error> {
        if !forest.is_reducible_graph() {
            return Err(invariant("P2 requires a reducible graph (run P1 first)"));
        }
        let m = forest.len();
        let n = cfg.node_count();
        let mut header = kept.with_capacity(meter, m)?;
        let mut parent = kept.with_capacity(meter, m)?;
        for l in forest.loops() {
            header.push(forest.header(l));
            parent.push(forest.parent(l).map_or(NONE, |p| p.index() as u32));
        }
        let mut innermost = kept.filled(meter, n, NONE)?;
        for v in cfg.preorder() {
            meter.tick(1)?;
            if let Some(l) = forest.innermost(v) {
                *at_mut(&mut innermost, v.index())? = to_u32(l.index())?;
            }
        }
        Ok(NestTables {
            header,
            parent,
            innermost,
        })
    }

    /// The number of loops.
    pub(crate) fn len(&self) -> usize {
        self.header.len()
    }

    /// Return every ledger byte this nest holds (its tables and `Cfg`).
    fn release(self, meter: &mut Meter) {
        meter.release(self.kept.bytes());
        meter.release(self.cfg.ledger_bytes());
    }

    /// Freeze into [`Loops`]. The nest is released on every path.
    pub(super) fn finish(self, sc: &StructCfg, meter: &mut Meter) -> Result<Loops, Error> {
        let mut kept = Kept::default();
        let loops = with_scratch(meter, |meter, scratch| {
            Loops::build(&self, sc, &mut kept, scratch, meter)
        });
        self.release(meter);
        match loops {
            Ok(mut loops) => {
                loops.ledger_bytes = kept.bytes();
                Ok(loops)
            }
            Err(e) => {
                meter.release(kept.bytes());
                Err(e)
            }
        }
    }
}

/// The loops of the prepared graph: its natural loops.
///
/// Every loop is single-entry: its header dominates its blocks, and every edge into the loop from
/// outside targets the header. A loop's blocks are its natural loop, the blocks of the loops
/// nested in it included.
#[derive(Clone, Debug)]
pub struct Loops {
    header: Vec<Node>,
    parent: Vec<u32>,
    end: Vec<u32>,
    /// Node -> innermost loop (`NONE` outside every loop).
    innermost: Vec<u32>,
    /// Loop -> its blocks: `blocks[block_start..block_end]` (own blocks in RPO, then each child's).
    block_start: Vec<u32>,
    block_end: Vec<u32>,
    blocks: Vec<Node>,
    /// Loop -> exit targets in RPO: `exits[exit_start[l]..exit_start[l + 1]]`.
    exit_start: Vec<u32>,
    exits: Vec<Node>,
    ledger_bytes: usize,
}

impl Loops {
    fn build(
        nest: &LoopNest,
        sc: &StructCfg,
        kept: &mut Kept,
        scratch: &mut Scratch,
        meter: &mut Meter,
    ) -> Result<Loops, Error> {
        let m = nest.len();
        let n = nest.innermost.len();
        let rpo = |v: Node| nest.cfg.rpo_number(v).unwrap_or(u32::MAX);

        // Children of every loop (and the roots), by ascending header RPO.
        let mut order: Vec<u32> = scratch.with_capacity(meter, m)?;
        order.extend(0..to_u32(m)?);
        order.sort_by_key(|&l| {
            at(&nest.header, l as usize)
                .map(|&h| rpo(h))
                .unwrap_or(u32::MAX)
        });
        let mut child_count = scratch.filled(meter, m + 1, 0u32)?;
        for &l in &order {
            let p = *at(&nest.parent, l as usize)?;
            let slot = if p == NONE { m } else { p as usize };
            *at_mut(&mut child_count, slot)? += 1;
        }
        // CSR: slot s's children at children[start[s]..start[s + 1]], roots last (slot m).
        let mut start = scratch.filled(meter, m + 2, 0u32)?;
        for s in 0..=m {
            *at_mut(&mut start, s + 1)? = *at(&start, s)? + *at(&child_count, s)?;
        }
        let mut cursor = scratch.with_capacity(meter, m + 1)?;
        cursor.extend_from_slice(start.get(..=m).unwrap_or(&[]));
        let mut children = scratch.filled(meter, m, 0u32)?;
        for &l in &order {
            let p = *at(&nest.parent, l as usize)?;
            let slot = if p == NONE { m } else { p as usize };
            let c = at_mut(&mut cursor, slot)?;
            *at_mut(&mut children, *c as usize)? = l;
            *c += 1;
        }

        // Preorder walk: new ids, end.
        let mut new_id = scratch.filled(meter, m, NONE)?;
        let mut header = kept.with_capacity(meter, m)?;
        let mut parent = kept.with_capacity(meter, m)?;
        let mut end = kept.filled(meter, m, 0u32)?;
        let mut stack: Vec<(u32, u32)> = scratch.with_capacity(meter, m + 1)?;
        stack.push((m as u32, *at(&start, m)?));
        while let Some(top) = stack.last_mut() {
            let (s, at_child) = *top;
            let limit = *at(&start, s as usize + 1)?;
            if at_child < limit {
                top.1 += 1;
                let c = *at(&children, at_child as usize)?;
                let id = to_u32(header.len())?;
                *at_mut(&mut new_id, c as usize)? = id;
                header.push(*at(&nest.header, c as usize)?);
                let p = *at(&nest.parent, c as usize)?;
                let p_id = if p == NONE {
                    NONE
                } else {
                    *at(&new_id, p as usize)?
                };
                parent.push(p_id);
                stack.push((c, *at(&start, c as usize)?));
            } else {
                stack.pop();
                if (s as usize) < m {
                    let id = *at(&new_id, s as usize)?;
                    *at_mut(&mut end, id as usize)? = to_u32(header.len())?;
                }
            }
            meter.tick(1)?;
        }
        if header.len() != m {
            return Err(invariant("loop nest has a parent cycle"));
        }

        // Innermost loop per node (new ids) and blocks grouped by loop in preorder.
        let mut innermost = kept.filled(meter, n, NONE)?;
        let mut direct_count = scratch.filled(meter, m + 1, 0u32)?;
        for v in 0..n {
            let l = *at(&nest.innermost, v)?;
            if l != NONE {
                let id = *at(&new_id, l as usize)?;
                *at_mut(&mut innermost, v)? = id;
                *at_mut(&mut direct_count, id as usize + 1)? += 1;
            }
        }
        for l in 0..m {
            *at_mut(&mut direct_count, l + 1)? += *at(&direct_count, l)?;
        }
        let total = *at(&direct_count, m)? as usize;
        let mut blocks = kept.filled(meter, total, Node::new(0))?;
        let mut fill = scratch.with_capacity(meter, m)?;
        fill.extend_from_slice(direct_count.get(..m).unwrap_or(&[]));
        for v in nest.cfg.rpo() {
            let l = *at(&innermost, v.index())?;
            if l != NONE {
                let f = at_mut(&mut fill, l as usize)?;
                *at_mut(&mut blocks, *f as usize)? = v;
                *f += 1;
            }
            meter.tick(1)?;
        }
        let mut block_start = kept.with_capacity(meter, m)?;
        let mut block_end = kept.with_capacity(meter, m)?;
        for l in 0..m {
            block_start.push(*at(&direct_count, l)?);
            let last = *at(&end, l)? as usize;
            block_end.push(*at(&direct_count, last)?);
        }

        let mut loops = Loops {
            header,
            parent,
            end,
            innermost,
            block_start,
            block_end,
            blocks,
            exit_start: Vec::new(),
            exits: Vec::new(),
            ledger_bytes: 0,
        };

        // Exit targets per loop, in RPO.
        let mut exit_start = kept.with_capacity(meter, m + 1)?;
        exit_start.push(0u32);
        let mut exits: Vec<Node> = Vec::new();
        let mut mark = scratch.filled(meter, n, NONE)?;
        let mut these: Vec<Node> = Vec::new();
        for l in 0..m {
            let li = LoopIx(to_u32(l)?);
            these.clear();
            for b in loops.blocks(li) {
                for s in sc.block(b)?.term.targets() {
                    meter.tick(1)?;
                    if !loops.contains(li, s) && *at(&mark, s.index())? != l as u32 {
                        *at_mut(&mut mark, s.index())? = l as u32;
                        scratch.push(meter, &mut these, s)?;
                    }
                }
            }
            these.sort_by_key(|&s| rpo(s));
            for &s in &these {
                kept.push(meter, &mut exits, s)?;
            }
            exit_start.push(to_u32(exits.len())?);
        }
        loops.exit_start = exit_start;
        loops.exits = exits;
        Ok(loops)
    }

    /// The number of loops.
    pub(crate) fn len(&self) -> usize {
        self.header.len()
    }

    /// All loops, outer before inner.
    pub fn loops(&self) -> impl DoubleEndedIterator<Item = LoopIx> + ExactSizeIterator {
        (0..self.header.len() as u32).map(LoopIx)
    }

    /// All loops, inner before outer: every loop comes after all of its descendants.
    pub fn innermost_first(&self) -> impl Iterator<Item = LoopIx> {
        self.loops().rev()
    }

    /// The header of `l`.
    pub fn header(&self, l: LoopIx) -> Option<Node> {
        self.header.get(l.index()).copied()
    }

    /// The loop `l` is directly nested in.
    pub fn parent(&self, l: LoopIx) -> Option<LoopIx> {
        self.parent
            .get(l.index())
            .copied()
            .filter(|&p| p != NONE)
            .map(LoopIx)
    }

    /// Whether `a` is `b` or contains it.
    pub fn loop_contains(&self, a: LoopIx, b: LoopIx) -> bool {
        a <= b && self.end.get(a.index()).is_some_and(|&e| b.0 < e)
    }

    /// The innermost loop containing `node`.
    pub fn innermost(&self, node: Node) -> Option<LoopIx> {
        self.innermost
            .get(node.index())
            .copied()
            .filter(|&l| l != NONE)
            .map(LoopIx)
    }

    /// Whether `node` is a block of `l` (possibly through a nested loop). O(1).
    pub fn contains(&self, l: LoopIx, node: Node) -> bool {
        self.innermost(node)
            .is_some_and(|m| self.loop_contains(l, m))
    }

    /// The loop `node` heads, if any.
    pub fn loop_of_header(&self, node: Node) -> Option<LoopIx> {
        self.innermost(node)
            .filter(|&l| self.header(l) == Some(node))
    }

    /// Every block of `l`: first those whose innermost loop is `l` (in RPO), then each child's.
    pub fn blocks(&self, l: LoopIx) -> impl ExactSizeIterator<Item = Node> + '_ {
        let from = self.block_start.get(l.index()).copied().unwrap_or(0) as usize;
        let to = self.block_end.get(l.index()).copied().unwrap_or(0) as usize;
        self.blocks.get(from..to).unwrap_or(&[]).iter().copied()
    }

    /// The blocks whose innermost loop is `l` (its header first), in RPO.
    pub fn own_blocks(&self, l: LoopIx) -> impl Iterator<Item = Node> + '_ {
        self.blocks(l)
            .take_while(move |&b| self.innermost(b) == Some(l))
    }

    /// The loops directly nested in `l`, by ascending header RPO.
    pub fn children(&self, l: LoopIx) -> impl Iterator<Item = LoopIx> + '_ {
        let end = self.end.get(l.index()).copied().unwrap_or(0);
        self.siblings_from(l.0.saturating_add(1), end)
    }

    /// The outermost loops, by ascending header RPO.
    pub fn top_level(&self) -> impl Iterator<Item = LoopIx> + '_ {
        self.siblings_from(0, self.header.len() as u32)
    }

    /// Loops `first`, `end(first)`, `end(end(first))`, … below `limit`: consecutive siblings in
    /// the preorder numbering.
    fn siblings_from(&self, first: u32, limit: u32) -> impl Iterator<Item = LoopIx> + '_ {
        std::iter::successors(Some(first), move |&c| self.end.get(c as usize).copied())
            .take_while(move |&c| c < limit)
            .map(LoopIx)
    }

    /// The exit targets of `l` (blocks outside `l` with a predecessor in it), in RPO.
    pub fn exits(&self, l: LoopIx) -> impl ExactSizeIterator<Item = Node> + '_ {
        let from = self.exit_start.get(l.index()).copied().unwrap_or(0) as usize;
        let to = self.exit_start.get(l.index() + 1).copied().unwrap_or(0) as usize;
        self.exits.get(from..to).unwrap_or(&[]).iter().copied()
    }

    /// Bytes this structure keeps charged to the ledger it was built under.
    pub fn ledger_bytes(&self) -> usize {
        self.ledger_bytes
    }
}
