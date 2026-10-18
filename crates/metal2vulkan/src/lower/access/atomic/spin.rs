//! Spin reads: the relaxed atomic loads that an exit of a loop holding them depends on, which
//! acquire their memory ([`super::AtomicAccess::spin`] and the texture family's `TexAtomic::spin`).
//!
//! Such a load is a spin's: the loop runs again until another invocation's store arrives at the
//! word it reads. Metal's relaxed load sees the store (Apple's decoupled-lookback scans spin on
//! one, and a 1,024-threadgroup chain of them completes on the M3 Max and the M4). On Mesa's RADV
//! and ANV, with the robust buffer access the device states, a relaxed one never saw a store that
//! arrived after its first read: a 64-threadgroup chain stopped at the first group whose first
//! load missed its predecessor's word, in every run, and completed with acquire (Mesa 26.2.4; with
//! the robustness features withheld it completed either way). Acquire is
//! an order Metal's relaxed one admits, so every answer stays one Metal can give, on every device;
//! the rule reads the function's structure only.
//!
//! **The dependence.** For each natural loop of the function's CFG that holds a relaxed atomic
//! load, the backward slice of its exits within its blocks: the seeds are the terminators of the
//! loop's blocks with a successor outside it, and an instruction of the loop in the slice brings
//! in
//! - every operand it reads (a phi's incoming values included), and, for a phi, the terminators
//!   of its incoming blocks in the loop, which choose the value it takes;
//! - the terminators its block is control dependent on in one iteration of the loop: on the
//!   loop's iteration graph (its blocks, with each back edge to its header and each exit edge
//!   cut to one sink), a branch with a successor the block post-dominates, which the block does
//!   not strictly post-dominate (Ferrante, Ottenstein and Warren). An iteration that may end
//!   either way at a branch makes the exit depend on it, which the function's own post-dominators,
//!   blind to a path that never leaves the loop, would not show;
//! - for a memory read (a load, an atomic, or a call), every write of the loop (a store, an
//!   atomic, or a call), with no alias analysis: a word the loop writes may be the one read.
//!
//! Values defined outside the loop end the walk. A relaxed atomic load in the slice is a spin
//! read; a relaxed load the loop repeats for any other use stays relaxed. The compare-and-swap
//! loops the lowering builds (`cas_loop`, `narrow`) exit on their read of the word by
//! construction, so that read acquires too (`Op::reload`).

use std::collections::BTreeSet;

use crate::air::intrinsics::{AirIntrinsic, AtomicOp as AirOp, MemoryOrder as AirOrder, TexOp};
use crate::air::ir::{BlockId, Body, Callee, Inst, InstId, InstKind, Ordering, Value};
use crate::budget::{tree_bytes, Meter};
use crate::error::Error;
use crate::graph::{Cfg, Graph, LoopForest, LoopId, PostDomTree};
use crate::lower::select::SelectCx;

/// "Not in the loop" in the per-block side table.
const OUT: u32 = u32::MAX;

/// Whether `inst` is a relaxed atomic load: an LLVM `load atomic` that is `unordered` or
/// `monotonic`, a relaxed AIR memory atomic load, or a texture atomic load (MSL's texture atomics
/// are relaxed; the texture family refuses any other order).
fn relaxed_load(inst: &Inst) -> bool {
    match &inst.kind {
        InstKind::Load {
            atomic: Some(a), ..
        } => matches!(a.ordering, Ordering::Unordered | Ordering::Monotonic),
        InstKind::Call(call) => match &call.callee {
            Callee::Air {
                op: AirIntrinsic::Atomic(a),
                ..
            } => a.op == AirOp::Load && a.order == AirOrder::Relaxed,
            Callee::Air {
                op: AirIntrinsic::Texture(t),
                ..
            } => matches!(t.op, TexOp::Atomic(AirOp::Load)),
            _ => false,
        },
        _ => false,
    }
}

/// Whether `inst` reads memory (a load, an atomic, or a call).
fn reads_memory(inst: &Inst) -> bool {
    matches!(
        inst.kind,
        InstKind::Load { .. }
            | InstKind::AtomicRmw { .. }
            | InstKind::CmpXchg { .. }
            | InstKind::Call(_)
    )
}

/// Whether `inst` writes memory (a store, an atomic, or a call).
fn writes_memory(inst: &Inst) -> bool {
    matches!(
        inst.kind,
        InstKind::Store { .. }
            | InstKind::AtomicRmw { .. }
            | InstKind::CmpXchg { .. }
            | InstKind::Call(_)
    )
}

/// Plan the function's spin reads into the access plan, before any instruction is selected: no
/// loop analysis runs for a function without a relaxed atomic load.
pub fn plan(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    let body = cx.body;
    let mut any = false;
    for inst in body.insts.values() {
        m.tick(1)?;
        if relaxed_load(inst) {
            any = true;
            break;
        }
    }
    if !any {
        return Ok(());
    }
    let loops = LoopForest::compute(&cx.entry.cfg, m)?;
    let r = slices(body, &loops, m);
    m.release(loops.ledger_bytes());
    let spins = r?;
    m.charge(tree_bytes(spins.len(), std::mem::size_of::<InstId>()))?;
    cx.families.access.spins = spins;
    Ok(())
}

/// Whether instruction `at` is a spin read ([`plan`] computed them).
pub fn is_spin(cx: &SelectCx<'_>, at: InstId) -> bool {
    cx.families.access.spins.contains(&at)
}

/// The relaxed atomic loads in an exit slice of a loop holding them. Every byte the walk charges
/// is released before returning, on either path.
fn slices(
    body: &Body,
    loops: &LoopForest<BlockId>,
    m: &mut Meter,
) -> Result<BTreeSet<InstId>, Error> {
    if loops.is_empty() {
        return Ok(BTreeSet::new());
    }
    let (n, nb) = (body.insts.len(), body.blocks.len());
    // `block_of` and `seen` per instruction; `ran` and `local` per block.
    let fixed = n
        .saturating_mul(2)
        .saturating_add(nb.saturating_mul(2))
        .saturating_mul(4);
    m.charge(fixed)?;
    let mut s = Slicer {
        body,
        loops,
        block_of: vec![0; n],
        seen: vec![0; n],
        ran: vec![0; nb],
        local: vec![OUT; nb],
        work: Vec::new(),
        operands: Vec::new(),
        controls: Vec::new(),
        out: BTreeSet::new(),
        charged: fixed,
    };
    let r = s.run(m);
    m.release(s.charged);
    r.map(|()| s.out)
}

struct Slicer<'a> {
    body: &'a Body,
    loops: &'a LoopForest<BlockId>,
    /// The block of each instruction, by raw id.
    block_of: Vec<u32>,
    /// The loop stamp (its index plus one) of each instruction already in the loop's slice.
    seen: Vec<u32>,
    /// The loop stamp of each block whose control dependences the slice already holds.
    ran: Vec<u32>,
    /// Each block's node in the current loop's iteration graph ([`Iteration`]), or [`OUT`].
    local: Vec<u32>,
    work: Vec<InstId>,
    /// The instructions one instruction of the slice brings in.
    operands: Vec<InstId>,
    /// The loop's control dependences, `(block, branching block)` by raw id, sorted.
    controls: Vec<(u32, u32)>,
    /// The spin reads found; charged by the caller once it keeps them.
    out: BTreeSet<InstId>,
    /// The bytes charged to the ledger for the walk.
    charged: usize,
}

/// Charge `m` for the capacity `v` gained since it held `before`, adding it to `charged`.
fn grown<T>(v: &Vec<T>, before: usize, charged: &mut usize, m: &mut Meter) -> Result<(), Error> {
    let bytes = v
        .capacity()
        .saturating_sub(before)
        .saturating_mul(std::mem::size_of::<T>());
    *charged = charged.saturating_add(bytes);
    m.charge(bytes)?;
    Ok(())
}

impl Slicer<'_> {
    fn run(&mut self, m: &mut Meter) -> Result<(), Error> {
        for (b, block) in self.body.blocks.iter() {
            for id in block.inst_ids() {
                m.tick(1)?;
                if let Some(s) = self.block_of.get_mut(id.raw() as usize) {
                    *s = b.raw();
                }
            }
        }
        for (k, l) in self.loops.loops().enumerate() {
            let stamp = u32::try_from(k).unwrap_or(u32::MAX - 1) + 1;
            let mut holds = false;
            for b in self.loops.blocks(l) {
                m.tick(1)?;
                holds |= self.body.block_insts(b).any(|(_, i)| relaxed_load(i));
            }
            if holds {
                self.slice(l, stamp, m)?;
            }
        }
        Ok(())
    }

    /// The terminator of block `b`.
    fn terminator(&self, b: BlockId) -> Option<InstId> {
        self.body.blocks.get(b).and_then(|block| block.last())
    }

    /// Push `id` onto the walk.
    fn push(&mut self, id: InstId, m: &mut Meter) -> Result<(), Error> {
        let before = self.work.capacity();
        self.work.push(id);
        grown(&self.work, before, &mut self.charged, m)
    }

    /// The slice of loop `l`'s exits, stamped `stamp`.
    fn slice(&mut self, l: LoopId, stamp: u32, m: &mut Meter) -> Result<(), Error> {
        let (loops, body) = (self.loops, self.body);
        self.controls_of(l, m)?;
        let mut memory = false;
        // The seeds: the terminators of the loop's blocks with a successor outside it.
        for b in loops.blocks(l) {
            m.tick(1)?;
            if body.successors(b).any(|s| !loops.contains(l, s)) {
                if let Some(t) = self.terminator(b) {
                    self.push(t, m)?;
                }
            }
        }
        while let Some(id) = self.work.pop() {
            m.tick(1)?;
            let Some(&b) = self.block_of.get(id.raw() as usize) else {
                continue;
            };
            if !loops.contains(l, BlockId::new(b)) {
                continue;
            }
            match self.seen.get_mut(id.raw() as usize) {
                Some(s) if *s != stamp => *s = stamp,
                _ => continue,
            }
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            if relaxed_load(inst) {
                self.out.insert(id);
            }
            // Its operands, and a phi's choosing terminators.
            let before = self.operands.capacity();
            self.operands.clear();
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    self.operands.push(d);
                }
            });
            if let InstKind::Phi { incoming, .. } = &inst.kind {
                for &(_, from) in incoming.iter() {
                    if loops.contains(l, from) {
                        if let Some(t) = body.blocks.get(from).and_then(|k| k.last()) {
                            self.operands.push(t);
                        }
                    }
                }
            }
            // The terminators its block is control dependent on, once per block.
            if self.ran.get(b as usize).is_some_and(|&r| r != stamp) {
                if let Some(r) = self.ran.get_mut(b as usize) {
                    *r = stamp;
                }
                let mut i = self.controls.partition_point(|&(x, _)| x < b);
                while let Some(&(x, a)) = self.controls.get(i) {
                    if x != b {
                        break;
                    }
                    if let Some(t) = body.blocks.get(BlockId::new(a)).and_then(|k| k.last()) {
                        self.operands.push(t);
                    }
                    i += 1;
                }
            }
            grown(&self.operands, before, &mut self.charged, m)?;
            let before = self.work.capacity();
            self.work.extend_from_slice(&self.operands);
            grown(&self.work, before, &mut self.charged, m)?;
            // The loop's writes, once per loop.
            if !memory && reads_memory(inst) {
                memory = true;
                for w in loops.blocks(l) {
                    for (wid, winst) in body.block_insts(w) {
                        m.tick(1)?;
                        if writes_memory(winst) {
                            self.push(wid, m)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The control dependences of loop `l`'s blocks on its iteration graph, into `controls`: for
    /// each branch and each of its successors, the blocks on the post-dominator path from the
    /// successor up to (not including) the branching block's immediate post-dominator.
    fn controls_of(&mut self, l: LoopId, m: &mut Meter) -> Result<(), Error> {
        let (loops, body) = (self.loops, self.body);
        let before = self.controls.capacity();
        self.controls.clear();
        let mut nodes: Vec<BlockId> = Vec::new();
        for b in loops.blocks(l) {
            m.tick(1)?;
            if let Some(x) = self.local.get_mut(b.raw() as usize) {
                *x = u32::try_from(nodes.len()).unwrap_or(OUT);
            }
            nodes.push(b);
        }
        let held = nodes
            .capacity()
            .saturating_mul(std::mem::size_of::<BlockId>());
        m.charge(held)?;
        let r = (|| -> Result<(), Error> {
            let g = Iteration {
                body,
                header: loops.header(l),
                nodes: &nodes,
                local: &self.local,
            };
            let cfg = Cfg::build(&g, m)?;
            let pdom = PostDomTree::compute(&cfg, m);
            m.release(cfg.ledger_bytes());
            let pdom = pdom?;
            let sink = g.sink();
            for (ai, &a) in nodes.iter().enumerate() {
                let ai = u32::try_from(ai).unwrap_or(OUT);
                let mut succ: Vec<u32> = g.successors(ai).collect();
                succ.sort_unstable();
                succ.dedup();
                if succ.len() < 2 {
                    continue;
                }
                let stop = pdom.ipdom(ai);
                for s in succ {
                    let mut x = Some(s);
                    while let Some(y) = x {
                        m.tick(1)?;
                        if Some(y) == stop || y == sink {
                            break;
                        }
                        if let Some(yb) = nodes.get(y as usize) {
                            self.controls.push((yb.raw(), a.raw()));
                        }
                        // A block that reaches no end of the iteration has no post-dominator.
                        x = pdom.ipdom(y).filter(|&p| p != y);
                    }
                }
            }
            m.release(pdom.ledger_bytes());
            Ok(())
        })();
        for b in &nodes {
            if let Some(x) = self.local.get_mut(b.raw() as usize) {
                *x = OUT;
            }
        }
        m.release(held);
        r?;
        self.controls.sort_unstable();
        self.controls.dedup();
        grown(&self.controls, before, &mut self.charged, m)
    }
}

/// One iteration of a loop as a graph: node `i` is the loop's block `nodes[i]`, and the sink
/// (node `nodes.len()`) ends the iteration. A block's edges stay, except an edge back to the
/// header or out of the loop, which goes to the sink.
struct Iteration<'a> {
    body: &'a Body,
    header: BlockId,
    nodes: &'a [BlockId],
    /// Each block's node, or [`OUT`] for a block outside the loop.
    local: &'a [u32],
}

impl Iteration<'_> {
    fn sink(&self) -> u32 {
        u32::try_from(self.nodes.len()).unwrap_or(OUT)
    }
}

impl Graph for Iteration<'_> {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.nodes.len() + 1
    }

    fn entry(&self) -> u32 {
        self.local
            .get(self.header.raw() as usize)
            .copied()
            .unwrap_or(0)
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let block = self.nodes.get(node as usize).copied();
        let sink = self.sink();
        block
            .into_iter()
            .flat_map(move |b| self.body.successors(b))
            .map(move |s| match self.local.get(s.raw() as usize) {
                Some(&x) if x != OUT && s != self.header => x,
                _ => sink,
            })
    }
}
