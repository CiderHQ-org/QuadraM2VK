//! The control-flow graph of a function body as a [`Graph`].
//!
//! A [`Body`] is its own graph: blocks are the nodes, block 0 is the entry, and the successors of
//! a block are its terminator's targets in operand order ([`Body::successors`]), duplicates kept
//! (`br c, %t, %t` and a switch with two cases to one block are two parallel edges each). The
//! frozen [`Cfg`] and the dominator tree are the single implementations in `graph/`; this module
//! only adapts the IR to them.
//!
//! [`Cfg`]: crate::graph::Cfg

use super::id::BlockId;
use super::module::Body;
use crate::graph::{Graph, NodeId};

impl NodeId for BlockId {
    fn from_node_index(index: u32) -> BlockId {
        BlockId::new(index)
    }

    fn node_index(self) -> u32 {
        self.raw()
    }
}

/// The graph contract holds for a body that [`super::Module::new`] accepted: it has at least one
/// block and every branch target is in range. On an unverified body `graph::Cfg::build` reports
/// the broken contract as an internal error; it never panics.
impl Graph for Body {
    type Node = BlockId;

    fn node_count(&self) -> usize {
        self.blocks.len()
    }

    fn entry(&self) -> BlockId {
        BlockId::new(0)
    }

    fn successors(&self, node: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        Body::successors(self, node)
    }
}

#[cfg(test)]
mod tests {
    use super::super::build::ModuleBuilder;
    use super::super::*;
    use crate::budget::{Budget, Meter};
    use crate::graph::{Cfg, DomTree, Graph};

    fn b(n: u32) -> BlockId {
        BlockId::new(n)
    }

    /// `entry: switch a1 [default bb2, 1 -> bb1, 2 -> bb1]`, `bb1: br a0, bb3, bb3`,
    /// `bb2: br bb3`, `bb3: ret`, `bb4 (unreachable): br bb3`.
    fn diamond_body() -> Body {
        let mut m = ModuleBuilder::new();
        let void = m.void();
        let i1 = m.int(1);
        let i32 = m.int(32);
        let fty = m.fn_ty(void, &[i1, i32]);
        let mut f = m.define("k", fty);
        let (b1, b2, b3, b4) = (f.block(), f.block(), f.block(), f.block());
        f.push_void(InstKind::Switch {
            v: f.arg(1),
            default: b2,
            cases: Box::new([(1, b1), (2, b1)]),
            hints: None,
        });
        f.switch_to(b1);
        f.cond_br(f.arg(0), b3, b3);
        f.switch_to(b2);
        f.br(b3);
        f.switch_to(b3);
        f.ret(None);
        f.switch_to(b4);
        f.br(b3);
        let k = f.finish();
        let module = m.finish().unwrap();
        module.func(k).unwrap().body.clone().unwrap()
    }

    #[test]
    fn a_body_is_its_graph_with_parallel_edges_kept() {
        let body = diamond_body();
        assert_eq!(Graph::node_count(&body), 5);
        assert_eq!(Graph::entry(&body), b(0));
        for block in body.blocks.ids() {
            assert_eq!(
                Graph::successors(&body, block).collect::<Vec<_>>(),
                body.terminator(block)
                    .unwrap()
                    .kind
                    .successors()
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(
            Graph::successors(&body, b(0)).collect::<Vec<_>>(),
            [b(2), b(1), b(1)]
        );
        assert_eq!(body.successors(b(9)).next(), None, "out of range: no edges");

        let mut meter = Meter::new(&Budget::default());
        let cfg = Cfg::build(&body, &mut meter).unwrap();
        assert_eq!(cfg.entry(), b(0));
        assert_eq!(cfg.edge_count(), 7);
        assert_eq!(cfg.predecessors(b(1)).collect::<Vec<_>>(), [b(0), b(0)]);
        assert_eq!(
            cfg.predecessors(b(3)).collect::<Vec<_>>(),
            [b(1), b(1), b(2), b(4)]
        );
        assert!(!cfg.is_reachable(b(4)));
        assert_eq!(cfg.reachable_count(), 4);

        let dom = DomTree::compute(&cfg, &mut meter).unwrap();
        assert_eq!(dom.idom(b(1)), Some(b(0)));
        assert_eq!(dom.idom(b(3)), Some(b(0)), "bb3 joins bb1 and bb2");
        assert!(!dom.is_reachable(b(4)));
        assert!(!dom.dominates(b(4), b(3)) && !dom.dominates(b(0), b(4)));
    }

    #[test]
    fn an_unverified_body_breaks_the_contract_without_panicking() {
        let mut meter = Meter::new(&Budget::default());
        assert!(Cfg::build(&Body::default(), &mut meter).is_err());
        let mut body = diamond_body();
        if let Some(InstKind::Br { dest, .. }) =
            body.insts.values_mut().last().map(|inst| &mut inst.kind)
        {
            *dest = b(40);
        }
        assert!(Cfg::build(&body, &mut meter).is_err());
    }
}
