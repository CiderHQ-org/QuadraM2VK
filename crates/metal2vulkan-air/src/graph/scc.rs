//! [`Sccs`]: strongly connected components by iterative Tarjan.

use std::marker::PhantomData;

use super::{invariant, ix, metered, to_u32, Graph, Ledger, NodeId, Work, NONE};
use crate::budget::Meter;
use crate::error::Error;

/// The strongly connected components of a whole graph (every node, reachable from the entry or
/// not), computed by Tarjan's algorithm with an explicit stack.
///
/// Components are numbered `0..len()` in **reverse topological order** of the condensation: for
/// every edge `u -> v` with `u` and `v` in different components, `u`'s component number is greater
/// than `v`'s.
/// Roots are tried in ascending node order and successors in [`Graph::successors`] order, so the
/// numbering and the member order are deterministic.
#[derive(Clone, Debug)]
pub struct Sccs<N> {
    member_start: Vec<u32>,
    members: Vec<u32>,
    cyclic: Vec<bool>,
    ledger_bytes: usize,
    node: PhantomData<fn() -> N>,
}

impl<N: NodeId> Sccs<N> {
    /// The components of `graph`. O(n + e) work and memory.
    /// On error nothing stays charged to `meter`'s ledger.
    pub fn compute<G: Graph<Node = N>>(graph: &G, meter: &mut Meter) -> Result<Sccs<N>, Error> {
        metered(meter, |meter, keep, scratch| {
            Self::compute_in(graph, meter, keep, scratch)
        })
    }

    fn compute_in<G: Graph<Node = N>>(
        graph: &G,
        meter: &mut Meter,
        keep: &mut Ledger,
        scratch: &mut Ledger,
    ) -> Result<Sccs<N>, Error> {
        let mut work = Work::new();

        let count = graph.node_count();
        let n = to_u32(count, "node count")?;

        let mut index = scratch.filled(meter, count, NONE)?;
        let mut low = scratch.filled(meter, count, 0u32)?;
        let mut on_stack = scratch.filled(meter, count, false)?;
        let mut self_loop = scratch.filled(meter, count, false)?;
        let mut stack = scratch.with_capacity::<u32>(meter, count)?;
        let mut frames = scratch.with_capacity(meter, count)?;

        let mut member_start = keep.with_capacity::<u32>(meter, count + 1)?;
        let mut members = keep.with_capacity::<u32>(meter, count)?;
        let mut cyclic = keep.with_capacity::<bool>(meter, count)?;
        member_start.push(0);

        let mut next_index: u32 = 0;
        for root in 0..n {
            if index[ix(root)] != NONE {
                continue;
            }
            index[ix(root)] = next_index;
            low[ix(root)] = next_index;
            next_index += 1;
            stack.push(root);
            on_stack[ix(root)] = true;
            frames.push((root, graph.successors(N::from_node_index(root))));
            work.step(meter, 1)?;

            while let Some((v, successors)) = frames.last_mut() {
                let v = *v;
                if let Some(w) = successors.next() {
                    work.step(meter, 1)?;
                    let w = w.node_index();
                    if w >= n {
                        return Err(invariant(format!(
                            "successor {w} of node {v} is not below the node count {n}"
                        )));
                    }
                    if w == v {
                        self_loop[ix(v)] = true;
                    }
                    if index[ix(w)] == NONE {
                        index[ix(w)] = next_index;
                        low[ix(w)] = next_index;
                        next_index += 1;
                        stack.push(w);
                        on_stack[ix(w)] = true;
                        frames.push((w, graph.successors(N::from_node_index(w))));
                    } else if on_stack[ix(w)] {
                        low[ix(v)] = low[ix(v)].min(index[ix(w)]);
                    }
                    continue;
                }
                frames.pop();
                if low[ix(v)] == index[ix(v)] {
                    let first = members.len();
                    while let Some(u) = stack.pop() {
                        on_stack[ix(u)] = false;
                        members.push(u);
                        work.step(meter, 1)?;
                        if u == v {
                            break;
                        }
                    }
                    cyclic.push(members.len() - first > 1 || self_loop[ix(v)]);
                    member_start.push(to_u32(members.len(), "member count")?);
                }
                if let Some((parent, _)) = frames.last() {
                    let parent = ix(*parent);
                    low[parent] = low[parent].min(low[ix(v)]);
                }
            }
        }

        drop(frames);
        work.finish(meter)?;
        Ok(Sccs {
            member_start,
            members,
            cyclic,
            ledger_bytes: keep.bytes,
            node: PhantomData,
        })
    }

    /// The number of components.
    pub fn len(&self) -> usize {
        self.cyclic.len()
    }

    /// Whether there is no component (the graph has no node).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The members of component `c`, in the order Tarjan's stack released them.
    pub fn members(&self, c: u32) -> impl ExactSizeIterator<Item = N> + '_ {
        let range = ix(self.member_start[ix(c)])..ix(self.member_start[ix(c) + 1]);
        self.members[range].iter().map(|&v| N::from_node_index(v))
    }

    /// Whether component `c` contains a cycle: two or more members, or one with a self-loop.
    pub fn is_cyclic(&self, c: u32) -> bool {
        self.cyclic[ix(c)]
    }

    /// Bytes this structure keeps charged to the ledger it was built under.
    pub fn ledger_bytes(&self) -> usize {
        self.ledger_bytes
    }
}
