//! [`TestCfg`]: a small AIR-shaped control-flow graph implementing [`CfgView`].

use crate::air::ir::{BlockId, ValueId};
use crate::error::{DiagName, Location};
use crate::graph::Graph;
use crate::spirv::ids::DemoteTy;
use crate::structurize::{CfgView, TermShape};

/// A block's terminator kind; its targets are the block's `succ`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TestKind {
    /// One successor.
    Br,
    /// Two successors (true, false); the condition is value `ValueId(block)`.
    CondBr,
    /// `succ[0]` is the default, `succ[i + 1]` is taken for `literals[i]`; the selector is value
    /// `ValueId(block)`. Literals are below [`SELECTOR_RANGE`] so the default is taken too.
    Switch(Vec<u64>),
    /// `ret void`.
    Return,
    /// `ret v` with `v = ValueId(block)`.
    ReturnValue,
    Unreachable,
}

/// Switch selectors take values in `0..SELECTOR_RANGE`.
pub const SELECTOR_RANGE: u64 = 8;

/// One block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestBlock {
    pub kind: TestKind,
    pub succ: Vec<BlockId>,
    pub tangled: bool,
}

/// A test CFG; the entry is `bb0`. `cross` and `phis` are what the view reports to P4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestCfg {
    pub blocks: Vec<TestBlock>,
    pub cross: Vec<(ValueId, DemoteTy)>,
    pub phis: Vec<(ValueId, DemoteTy)>,
}

impl TestCfg {
    /// A CFG from successor lists: none returns a value, one branches, two is a conditional
    /// branch, more is a switch whose first successor is the default and whose cases take
    /// literals `0, 1, ...`.
    pub fn from_successors(succ: &[Vec<u32>]) -> TestCfg {
        let blocks = succ
            .iter()
            .map(|s| {
                let kind = match s.len() {
                    0 => TestKind::ReturnValue,
                    1 => TestKind::Br,
                    2 => TestKind::CondBr,
                    n => TestKind::Switch((0..n as u64 - 1).collect()),
                };
                TestBlock {
                    kind,
                    succ: s.iter().map(|&b| BlockId::new(b)).collect(),
                    tangled: false,
                }
            })
            .collect();
        TestCfg {
            blocks,
            cross: Vec::new(),
            phis: Vec::new(),
        }
    }

    /// Mark `blocks` as holding a tangled instruction.
    pub fn with_tangled(mut self, blocks: &[u32]) -> TestCfg {
        for &b in blocks {
            self.blocks[b as usize].tangled = true;
        }
        self
    }

    /// The successor lists (as `u32`).
    pub fn successor_lists(&self) -> Vec<Vec<u32>> {
        self.blocks
            .iter()
            .map(|b| b.succ.iter().map(|s| s.raw()).collect())
            .collect()
    }

    /// The number of edges.
    pub fn edge_count(&self) -> usize {
        self.blocks.iter().map(|b| b.succ.len()).sum()
    }
}

impl Graph for TestCfg {
    type Node = BlockId;

    fn node_count(&self) -> usize {
        self.blocks.len()
    }

    fn entry(&self) -> BlockId {
        BlockId::new(0)
    }

    fn successors(&self, node: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.blocks
            .get(node.raw() as usize)
            .map_or(&[][..], |b| &b.succ[..])
            .iter()
            .copied()
    }
}

impl CfgView for TestCfg {
    fn terminator(&self, b: BlockId) -> TermShape {
        let v = ValueId::new(b.raw());
        match &self.blocks[b.raw() as usize].kind {
            TestKind::Br => TermShape::Br,
            TestKind::CondBr => TermShape::CondBr(v),
            TestKind::Switch(literals) => TermShape::Switch(v, literals.clone().into()),
            TestKind::Return => TermShape::Return(None),
            TestKind::ReturnValue => TermShape::Return(Some(v)),
            TestKind::Unreachable => TermShape::Unreachable,
        }
    }

    fn has_tangled(&self, b: BlockId) -> bool {
        self.blocks[b.raw() as usize].tangled
    }

    fn cross_block_values(&self) -> &[(ValueId, DemoteTy)] {
        &self.cross
    }

    fn phis(&self) -> &[(ValueId, DemoteTy)] {
        &self.phis
    }

    fn location(&self, b: BlockId) -> Location {
        Location::function(DiagName::new("testkit"), Some(b.raw()), None)
    }
}
