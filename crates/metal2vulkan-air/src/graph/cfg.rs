//! [`Cfg`]: the frozen CSR form of a [`Graph`], with predecessors, DFS numbering, and RPO.

use std::marker::PhantomData;

use super::{invariant, ix, metered, to_u32, Graph, Ledger, NodeId, Work, NONE};
use crate::budget::Meter;
use crate::error::Error;

/// A frozen control-flow graph: successors and predecessors in compressed (CSR) arrays, plus the
/// facts of one iterative depth-first search from the entry.
///
/// The DFS visits successors in [`Graph::successors`] order. It yields:
///
/// - the **preorder** numbering of reachable nodes (`0` is the entry) with the DFS-tree parent
///   and the subtree interval of every node (`b` is a DFS descendant of `a` iff
///   `pre(a) <= pre(b) < end(a)`), used by the dominator and loop algorithms;
/// - the **reverse postorder** of reachable nodes.
///
/// Predecessors of a node are listed by ascending source node, each source repeated once per
/// parallel edge. Unreachable nodes keep their edges (they are part of the graph), but have no
/// DFS or RPO number.
#[derive(Clone, Debug)]
pub struct Cfg<N> {
    entry: u32,
    succ_start: Vec<u32>,
    succ: Vec<u32>,
    pred_start: Vec<u32>,
    pred: Vec<u32>,
    /// Node -> DFS preorder number, `NONE` when unreachable.
    pre: Vec<u32>,
    /// Preorder number -> node.
    order: Vec<u32>,
    /// Preorder number -> preorder number of the DFS-tree parent (`NONE` for the entry).
    dfs_parent: Vec<u32>,
    /// Preorder number -> one past the last preorder number of its DFS subtree.
    subtree_end: Vec<u32>,
    /// Reachable nodes in reverse postorder.
    rpo: Vec<u32>,
    /// Node -> index in `rpo`, `NONE` when unreachable.
    rpo_number: Vec<u32>,
    ledger_bytes: usize,
    node: PhantomData<fn() -> N>,
}

impl<N: NodeId> Cfg<N> {
    /// Freeze `graph` and run the DFS. O(n + e) work and memory.
    /// On error nothing stays charged to `meter`'s ledger.
    pub fn build<G: Graph<Node = N>>(graph: &G, meter: &mut Meter) -> Result<Cfg<N>, Error> {
        metered(meter, |meter, keep, scratch| {
            Self::build_in(graph, meter, keep, scratch)
        })
    }

    fn build_in<G: Graph<Node = N>>(
        graph: &G,
        meter: &mut Meter,
        keep: &mut Ledger,
        scratch: &mut Ledger,
    ) -> Result<Cfg<N>, Error> {
        let mut work = Work::new();

        let count = graph.node_count();
        if count == 0 {
            return Err(invariant("graph has no nodes"));
        }
        let n = to_u32(count, "node count")?;
        let entry = graph.entry().node_index();
        if entry >= n {
            return Err(invariant(format!(
                "entry {entry} is not below the node count {n}"
            )));
        }

        // Successor CSR: count, then fill, refusing a graph whose successors change.
        let mut succ_start = keep.filled(meter, count + 1, 0u32)?;
        let mut total: u32 = 0;
        for v in 0..n {
            let mut degree: u32 = 0;
            for s in graph.successors(N::from_node_index(v)) {
                let s = s.node_index();
                if s >= n {
                    return Err(invariant(format!(
                        "successor {s} of node {v} is not below the node count {n}"
                    )));
                }
                degree = degree
                    .checked_add(1)
                    .ok_or_else(|| invariant("out-degree exceeds u32"))?;
                work.step(meter, 1)?;
            }
            total = total
                .checked_add(degree)
                .filter(|&t| t != NONE)
                .ok_or_else(|| invariant("edge count exceeds the u32 graph index space"))?;
            succ_start[ix(v) + 1] = total;
            work.step(meter, 1)?;
        }
        let edges = ix(total);
        let mut succ = keep.with_capacity::<u32>(meter, edges)?;
        for v in 0..n {
            let end = ix(succ_start[ix(v) + 1]);
            for s in graph.successors(N::from_node_index(v)) {
                if succ.len() >= end {
                    return Err(invariant(format!(
                        "successors of node {v} changed between calls"
                    )));
                }
                succ.push(s.node_index());
                work.step(meter, 1)?;
            }
            if succ.len() != end {
                return Err(invariant(format!(
                    "successors of node {v} changed between calls"
                )));
            }
        }

        // Predecessor CSR by counting sort over sources (ascending source order).
        let mut pred_start = keep.filled(meter, count + 1, 0u32)?;
        for &s in &succ {
            pred_start[ix(s) + 1] += 1;
        }
        for v in 0..count {
            pred_start[v + 1] += pred_start[v];
        }
        let mut cursor = scratch.filled(meter, count, 0u32)?;
        cursor.copy_from_slice(&pred_start[..count]);
        let mut pred = keep.filled(meter, edges, 0u32)?;
        for v in 0..n {
            for &s in &succ[ix(succ_start[ix(v)])..ix(succ_start[ix(v) + 1])] {
                let at = &mut cursor[ix(s)];
                pred[ix(*at)] = v;
                *at += 1;
                work.step(meter, 1)?;
            }
        }

        // Iterative DFS: preorder, DFS parents, subtree intervals, postorder.
        let mut pre = keep.filled(meter, count, NONE)?;
        let mut order = keep.with_capacity::<u32>(meter, count)?;
        let mut dfs_parent = keep.with_capacity::<u32>(meter, count)?;
        let mut subtree_end = keep.filled(meter, count, 0u32)?;
        let mut rpo = keep.with_capacity::<u32>(meter, count)?;
        let mut rpo_number = keep.filled(meter, count, NONE)?;
        let mut stack = scratch.with_capacity::<(u32, u32)>(meter, count)?;

        pre[ix(entry)] = 0;
        order.push(entry);
        dfs_parent.push(NONE);
        stack.push((entry, succ_start[ix(entry)]));
        while let Some(top) = stack.last_mut() {
            let (v, at) = *top;
            if at < succ_start[ix(v) + 1] {
                top.1 += 1;
                let w = succ[ix(at)];
                if pre[ix(w)] == NONE {
                    let number = to_u32(order.len(), "preorder number")?;
                    pre[ix(w)] = number;
                    order.push(w);
                    dfs_parent.push(pre[ix(v)]);
                    stack.push((w, succ_start[ix(w)]));
                }
            } else {
                stack.pop();
                subtree_end[ix(pre[ix(v)])] = to_u32(order.len(), "preorder number")?;
                rpo.push(v);
            }
            work.step(meter, 1)?;
        }
        rpo.reverse();
        for (i, &v) in rpo.iter().enumerate() {
            rpo_number[ix(v)] = to_u32(i, "RPO number")?;
        }
        subtree_end.truncate(order.len());

        drop(stack);
        drop(cursor);
        work.finish(meter)?;
        Ok(Cfg {
            entry,
            succ_start,
            succ,
            pred_start,
            pred,
            pre,
            order,
            dfs_parent,
            subtree_end,
            rpo,
            rpo_number,
            ledger_bytes: keep.bytes,
            node: PhantomData,
        })
    }

    /// The entry node.
    pub fn entry(&self) -> N {
        N::from_node_index(self.entry)
    }

    /// The number of nodes, reachable or not.
    pub fn node_count(&self) -> usize {
        self.pre.len()
    }

    /// The number of edges, parallel edges counted separately.
    #[cfg(test)]
    pub fn edge_count(&self) -> usize {
        self.succ.len()
    }

    /// The number of nodes reachable from the entry.
    pub fn reachable_count(&self) -> usize {
        self.order.len()
    }

    /// The successors of `node`, in branch order, with duplicates.
    pub fn successors(&self, node: N) -> impl ExactSizeIterator<Item = N> + '_ {
        self.succ_raw(node.node_index())
            .iter()
            .map(|&s| N::from_node_index(s))
    }

    /// The predecessors of `node` (reachable or not), by ascending source, once per edge.
    pub fn predecessors(&self, node: N) -> impl ExactSizeIterator<Item = N> + '_ {
        self.pred_raw(node.node_index())
            .iter()
            .map(|&p| N::from_node_index(p))
    }

    /// Whether `node` is reachable from the entry.
    pub fn is_reachable(&self, node: N) -> bool {
        self.pre[ix(node.node_index())] != NONE
    }

    /// Reachable nodes in reverse postorder (the entry first).
    pub fn rpo(&self) -> impl ExactSizeIterator<Item = N> + '_ {
        self.rpo.iter().map(|&v| N::from_node_index(v))
    }

    /// The position of `node` in [`Cfg::rpo`], or `None` when it is unreachable.
    pub fn rpo_number(&self, node: N) -> Option<u32> {
        Some(self.rpo_number[ix(node.node_index())]).filter(|&r| r != NONE)
    }

    /// Reachable nodes in DFS preorder (the entry first).
    pub fn preorder(&self) -> impl ExactSizeIterator<Item = N> + '_ {
        self.order.iter().map(|&v| N::from_node_index(v))
    }

    /// The DFS preorder number of `node`, or `None` when it is unreachable.
    #[cfg(test)]
    pub fn preorder_number(&self, node: N) -> Option<u32> {
        Some(self.pre[ix(node.node_index())]).filter(|&p| p != NONE)
    }

    /// The parent of `node` in the DFS tree (`None` for the entry and unreachable nodes).
    #[cfg(test)]
    pub fn dfs_parent(&self, node: N) -> Option<N> {
        let p = self.pre[ix(node.node_index())];
        if p == NONE {
            return None;
        }
        let parent = self.dfs_parent[ix(p)];
        (parent != NONE).then(|| N::from_node_index(self.order[ix(parent)]))
    }

    /// Whether `ancestor` is `node` or a proper ancestor of it in the DFS tree. False when either
    /// is unreachable.
    pub fn is_dfs_ancestor(&self, ancestor: N, node: N) -> bool {
        let a = self.pre[ix(ancestor.node_index())];
        let b = self.pre[ix(node.node_index())];
        a != NONE && b != NONE && a <= b && b < self.subtree_end[ix(a)]
    }

    /// Whether the edge `from -> to` is retreating: both reachable and `to` a DFS ancestor of
    /// `from` (self-loops included). In a reducible graph these are exactly the back edges.
    pub fn is_retreating(&self, from: N, to: N) -> bool {
        self.is_dfs_ancestor(to, from)
    }

    /// Bytes this structure keeps charged to the ledger it was built under.
    pub fn ledger_bytes(&self) -> usize {
        self.ledger_bytes
    }

    // Raw accessors for the other algorithms of this module.

    pub(super) fn succ_raw(&self, v: u32) -> &[u32] {
        &self.succ[ix(self.succ_start[ix(v)])..ix(self.succ_start[ix(v) + 1])]
    }

    pub(super) fn pred_raw(&self, v: u32) -> &[u32] {
        &self.pred[ix(self.pred_start[ix(v)])..ix(self.pred_start[ix(v) + 1])]
    }

    pub(super) fn entry_raw(&self) -> u32 {
        self.entry
    }

    /// Node -> preorder number (`NONE` if unreachable).
    pub(super) fn pre_raw(&self) -> &[u32] {
        &self.pre
    }

    /// Preorder number -> node.
    pub(super) fn order_raw(&self) -> &[u32] {
        &self.order
    }

    /// Preorder number -> parent preorder number (`NONE` for the entry).
    pub(super) fn dfs_parent_raw(&self) -> &[u32] {
        &self.dfs_parent
    }

    /// Preorder number -> end of its DFS subtree interval.
    pub(super) fn subtree_end_raw(&self) -> &[u32] {
        &self.subtree_end
    }
}

impl<N: NodeId> Graph for Cfg<N> {
    type Node = N;

    fn node_count(&self) -> usize {
        Cfg::node_count(self)
    }

    fn entry(&self) -> N {
        Cfg::entry(self)
    }

    fn successors(&self, node: N) -> impl Iterator<Item = N> + '_ {
        Cfg::successors(self, node)
    }
}
