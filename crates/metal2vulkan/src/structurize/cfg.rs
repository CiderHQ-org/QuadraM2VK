//! [`StructCfg`]: the structurizer's working control-flow graph.
//!
//! P0 copies the reachable blocks of a [`super::CfgView`] into it; later passes add synthetic
//! blocks (dispatch headers, selector stores) and retarget edges, but never duplicate an original
//! block: every original block appears in exactly one [`Content::Code`] node (the
//! "no duplication" invariant). Nodes are dense ([`Node`]); the graph implements [`Graph`] so the
//! single `graph` module analyses it.
//!
//! Synthetic control flow is expressed with structurizer-owned selector variables ([`SelVar`]):
//! a [`Content::SetSel`] block stores a constant, and a [`Term::Dispatch`] branches on it. They
//! run no original code.

use crate::air::ir::{BlockId, ValueId};
use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::graph::{Graph, NodeId};

/// A node of a [`StructCfg`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node(u32);

impl Node {
    /// The node with this dense index.
    pub const fn new(index: u32) -> Node {
        Node(index)
    }

    /// The dense index, for side tables.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl NodeId for Node {
    fn from_node_index(index: u32) -> Node {
        Node(index)
    }

    fn node_index(self) -> u32 {
        self.0
    }
}

/// A structurizer-owned `u32` selector variable (irreducible entry selector, later loop exit
/// selector). Lowered to a `Function` variable; never an SSA value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SelVar(u32);

impl SelVar {
    /// The selector with this dense index (allocation goes through the structurizer passes).
    pub(super) const fn new(index: u32) -> SelVar {
        SelVar(index)
    }

    /// The dense index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// What a node executes before its terminator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Content {
    /// The original block's instructions (no terminator).
    Code(BlockId),
    /// `sel = value`.
    SetSel { sel: SelVar, value: u32 },
    /// Nothing.
    Empty,
}

/// A node's terminator. Successor order is [`Term::targets`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Term {
    Br(Node),
    /// `if cond { t } else { f }`, `t != f`.
    CondBr {
        cond: ValueId,
        t: Node,
        f: Node,
    },
    /// An original switch with its literals grouped by target: every case target is distinct and
    /// differs from `default`, and every case has at least one literal.
    Switch {
        sel: ValueId,
        default: Node,
        cases: Box<[(Box<[u64]>, Node)]>,
    },
    /// A synthetic dispatch: `sel == i` branches to `targets[i]`. Only values the structurizer
    /// stored are ever read.
    Dispatch {
        sel: SelVar,
        targets: Box<[Node]>,
    },
    Return(Option<ValueId>),
    Unreachable,
}

impl Term {
    /// The successors in branch order: `Br` its target; `CondBr` true then false; `Switch` the
    /// default then each case; `Dispatch` its targets in selector order.
    pub fn targets(&self) -> impl Iterator<Item = Node> + '_ {
        let (first, second, rest): (Option<Node>, Option<Node>, &[Node]) = match self {
            Term::Br(t) => (Some(*t), None, &[]),
            Term::CondBr { t, f, .. } => (Some(*t), Some(*f), &[]),
            Term::Switch { default, .. } => (Some(*default), None, &[]),
            Term::Dispatch { targets, .. } => (None, None, targets),
            Term::Return(_) | Term::Unreachable => (None, None, &[]),
        };
        let cases: &[(Box<[u64]>, Node)] = match self {
            Term::Switch { cases, .. } => cases,
            _ => &[],
        };
        first
            .into_iter()
            .chain(second)
            .chain(rest.iter().copied())
            .chain(cases.iter().map(|(_, n)| *n))
    }

    /// Replace every successor equal to `from` with `to`.
    pub fn retarget(&mut self, from: Node, to: Node) {
        let swap = |n: &mut Node| {
            if *n == from {
                *n = to;
            }
        };
        match self {
            Term::Br(t) => swap(t),
            Term::CondBr { t, f, .. } => {
                swap(t);
                swap(f);
            }
            Term::Switch { default, cases, .. } => {
                swap(default);
                cases.iter_mut().for_each(|(_, n)| swap(n));
            }
            Term::Dispatch { targets, .. } => targets.iter_mut().for_each(swap),
            Term::Return(_) | Term::Unreachable => {}
        }
    }

    /// Heap bytes this terminator owns (charged to the ledger with its block).
    fn heap_bytes(&self) -> usize {
        match self {
            Term::Switch { cases, .. } => cases.iter().fold(
                std::mem::size_of_val::<[(Box<[u64]>, Node)]>(cases),
                |sum, (lits, _)| sum.saturating_add(std::mem::size_of_val::<[u64]>(lits)),
            ),
            Term::Dispatch { targets, .. } => std::mem::size_of_val::<[Node]>(targets),
            _ => 0,
        }
    }
}

/// One node: content, then terminator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub content: Content,
    pub term: Term,
}

/// The working graph. Node [`StructCfg::entry`] is the function entry.
#[derive(Clone, Debug)]
pub struct StructCfg {
    blocks: Vec<Block>,
    entry: Node,
    /// Original block -> node (`None` when pruned).
    original: Vec<Option<Node>>,
    selectors: u32,
    ledger_bytes: usize,
}

impl StructCfg {
    /// An empty graph for `originals` original blocks, with room for `capacity` nodes. The entry
    /// is set by the first [`StructCfg::push`].
    pub(super) fn with_capacity(
        meter: &mut Meter,
        originals: usize,
        capacity: usize,
    ) -> Result<StructCfg, Error> {
        let mut sc = StructCfg {
            blocks: Vec::new(),
            entry: Node(0),
            original: Vec::new(),
            selectors: 0,
            ledger_bytes: 0,
        };
        let grown = sc.reserve_blocks(meter, capacity).and_then(|()| {
            let bytes = originals.saturating_mul(std::mem::size_of::<Option<Node>>());
            sc.charge(meter, bytes)?;
            try_reserve(&mut sc.original, originals, bytes)?;
            sc.original.resize(originals, None);
            Ok(())
        });
        match grown {
            Ok(()) => Ok(sc),
            Err(e) => {
                meter.release(sc.ledger_bytes);
                Err(e)
            }
        }
    }

    fn charge(&mut self, meter: &mut Meter, bytes: usize) -> Result<(), Error> {
        meter.charge(bytes)?;
        self.ledger_bytes = self.ledger_bytes.saturating_add(bytes);
        Ok(())
    }

    fn reserve_blocks(&mut self, meter: &mut Meter, additional: usize) -> Result<(), Error> {
        let bytes = additional.saturating_mul(std::mem::size_of::<Block>());
        self.charge(meter, bytes)?;
        try_reserve(&mut self.blocks, additional, bytes)
    }

    /// Append a node, charging its storage first. The first node pushed is the entry.
    pub fn push(&mut self, meter: &mut Meter, block: Block) -> Result<Node, Error> {
        let node = Node(to_u32(self.blocks.len())?);
        if self.blocks.len() == self.blocks.capacity() {
            self.reserve_blocks(meter, self.blocks.capacity().max(16))?;
        }
        self.charge(meter, block.term.heap_bytes())?;
        if let Content::Code(b) = block.content {
            let slot = self
                .original
                .get_mut(b.raw() as usize)
                .ok_or_else(|| invariant(format!("original block {b:?} out of range")))?;
            if slot.is_some() {
                return Err(invariant(format!("original block {b:?} placed twice")));
            }
            *slot = Some(node);
        }
        self.blocks.push(block);
        Ok(node)
    }

    /// A fresh selector variable.
    pub fn new_selector(&mut self) -> Result<SelVar, Error> {
        let sel = SelVar(self.selectors);
        self.selectors = self
            .selectors
            .checked_add(1)
            .ok_or_else(|| invariant("selector count exceeds u32"))?;
        Ok(sel)
    }

    /// Replace every edge `n -> from` with `n -> to` (no allocation).
    pub fn retarget(&mut self, n: Node, from: Node, to: Node) -> Result<(), Error> {
        self.block_mut(n)?.term.retarget(from, to);
        Ok(())
    }

    /// The entry node.
    pub fn entry(&self) -> Node {
        self.entry
    }

    /// The number of nodes.
    pub(crate) fn len(&self) -> usize {
        self.blocks.len()
    }

    /// The number of edges, parallel edges counted.
    pub fn edge_count(&self) -> usize {
        self.blocks.iter().map(|b| b.term.targets().count()).sum()
    }

    /// Every node with its block.
    pub fn blocks(&self) -> impl ExactSizeIterator<Item = (Node, &Block)> + '_ {
        self.blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (Node(i as u32), b))
    }

    /// Node `n`.
    pub fn block(&self, n: Node) -> Result<&Block, Error> {
        self.blocks
            .get(n.index())
            .ok_or_else(|| invariant(format!("node {n:?} out of range")))
    }

    fn block_mut(&mut self, n: Node) -> Result<&mut Block, Error> {
        self.blocks
            .get_mut(n.index())
            .ok_or_else(|| invariant(format!("node {n:?} out of range")))
    }

    /// The node holding original block `b`, if `b` was reachable.
    #[cfg(test)]
    pub fn node_of(&self, b: BlockId) -> Option<Node> {
        self.original.get(b.raw() as usize).copied().flatten()
    }

    /// The number of selector variables allocated.
    pub fn selector_count(&self) -> usize {
        self.selectors as usize
    }

    /// Bytes this graph keeps charged to the ledger it was built under.
    pub fn ledger_bytes(&self) -> usize {
        self.ledger_bytes
    }
}

impl Graph for StructCfg {
    type Node = Node;

    fn node_count(&self) -> usize {
        self.blocks.len()
    }

    fn entry(&self) -> Node {
        self.entry
    }

    fn successors(&self, node: Node) -> impl Iterator<Item = Node> + '_ {
        self.blocks
            .get(node.index())
            .into_iter()
            .flat_map(|b| b.term.targets())
    }
}

/// An internal invariant failure.
pub(super) fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// A count or index as `u32`, refusing sizes beyond the index space.
pub(super) fn to_u32(value: usize) -> Result<u32, Error> {
    u32::try_from(value)
        .ok()
        .filter(|&v| v != u32::MAX)
        .ok_or_else(|| invariant(format!("{value} exceeds the u32 index space")))
}

/// Reserve `additional` elements whose `bytes` are already charged.
pub(super) fn try_reserve<T>(v: &mut Vec<T>, additional: usize, bytes: usize) -> Result<(), Error> {
    v.try_reserve(additional).map_err(|e| {
        Error::Internal(InternalError::new(
            InternalKind::AllocationFailed,
            format!("reserving {bytes} ledger-admitted bytes: {e}"),
        ))
    })
}

/// A ledger for the side tables one pass allocates: every vector is charged before it is
/// reserved; [`Scratch::release`] returns the total.
pub(super) struct Scratch {
    bytes: usize,
}

impl Scratch {
    pub(super) fn new() -> Scratch {
        Scratch { bytes: 0 }
    }

    /// A vector of `len` copies of `value`, charged first.
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

    /// An empty vector with room for `capacity` elements, charged first.
    pub(super) fn with_capacity<T>(
        &mut self,
        meter: &mut Meter,
        capacity: usize,
    ) -> Result<Vec<T>, Error> {
        let mut v = Vec::new();
        self.grow(meter, &mut v, capacity)?;
        Ok(v)
    }

    /// Make `v` hold `len` copies of `value`, reusing its buffer and charging any growth first.
    pub(super) fn refill<T: Clone>(
        &mut self,
        meter: &mut Meter,
        v: &mut Vec<T>,
        len: usize,
        value: T,
    ) -> Result<(), Error> {
        v.clear();
        if v.capacity() < len {
            self.grow(meter, v, len)?;
        }
        v.resize(len, value);
        Ok(())
    }

    /// Push, growing by doubling (at least 16) and charging each growth first.
    pub(super) fn push<T>(&mut self, meter: &mut Meter, v: &mut Vec<T>, x: T) -> Result<(), Error> {
        if v.len() == v.capacity() {
            self.grow(meter, v, v.capacity().max(16))?;
        }
        v.push(x);
        Ok(())
    }

    fn grow<T>(
        &mut self,
        meter: &mut Meter,
        v: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), Error> {
        let bytes = additional.saturating_mul(std::mem::size_of::<T>());
        meter.charge(bytes)?;
        self.bytes = self.bytes.saturating_add(bytes);
        try_reserve(v, additional, bytes)
    }

    /// Return everything charged.
    pub(super) fn release(self, meter: &mut Meter) {
        meter.release(self.bytes);
    }
}

/// Run `body` with a scratch ledger that is released on every path.
pub(super) fn with_scratch<T>(
    meter: &mut Meter,
    body: impl FnOnce(&mut Meter, &mut Scratch) -> Result<T, Error>,
) -> Result<T, Error> {
    let mut scratch = Scratch::new();
    let result = body(meter, &mut scratch);
    scratch.release(meter);
    result
}

/// `v[i]` for a side table indexed by a validated node or loop index.
pub(super) fn at<T>(v: &[T], i: usize) -> Result<&T, Error> {
    v.get(i)
        .ok_or_else(|| invariant(format!("side-table index {i} out of range")))
}

/// `&mut v[i]` for a side table indexed by a validated node or loop index.
pub(super) fn at_mut<T>(v: &mut [T], i: usize) -> Result<&mut T, Error> {
    v.get_mut(i)
        .ok_or_else(|| invariant(format!("side-table index {i} out of range")))
}
