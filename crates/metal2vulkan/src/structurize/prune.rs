//! P0: copy the reachable part of a [`CfgView`] into a [`StructCfg`].
//!
//! - Blocks unreachable from the entry are dropped.
//! - `CondBr(c, t, t)` becomes `Br(t)`.
//! - Switch literals are grouped by target, in first-appearance order; cases that target the
//!   default are dropped (they are the default), and a switch left without cases becomes
//!   `Br(default)`.
//!
//! Nodes are numbered entry first, then the other reachable blocks by ascending `BlockId`, so the
//! result is a deterministic function of the view. The view's contract (terminator shape agrees
//! with the successors, distinct switch literals) is checked here; a broken contract is
//! `Error::Internal(Invariant)`.

use super::cfg::{at, at_mut, invariant, to_u32, with_scratch, Block, Content, Node, Scratch};
use super::cfg::{StructCfg, Term};
use super::{CfgView, TermShape};
use crate::air::ir::BlockId;
use crate::budget::Meter;
use crate::error::Error;
use crate::graph::Cfg;

const NONE: u32 = u32::MAX;

/// P0. On error nothing stays charged to `meter`'s ledger.
pub fn prune(view: &impl CfgView, meter: &mut Meter) -> Result<StructCfg, Error> {
    let cfg = Cfg::build(view, meter)?;
    let result = with_scratch(meter, |meter, scratch| build(view, &cfg, meter, scratch));
    meter.release(cfg.ledger_bytes());
    result
}

fn build(
    view: &impl CfgView,
    cfg: &Cfg<BlockId>,
    meter: &mut Meter,
    scratch: &mut Scratch,
) -> Result<StructCfg, Error> {
    let count = view.node_count();
    let entry = view.entry();

    // Original block -> node, entry first.
    let mut map = scratch.filled(meter, count, NONE)?;
    let mut order = scratch.with_capacity::<BlockId>(meter, cfg.reachable_count())?;
    order.push(entry);
    *at_mut(&mut map, entry.raw() as usize)? = 0;
    for raw in 0..to_u32(count)? {
        let b = BlockId::new(raw);
        if b != entry && cfg.is_reachable(b) {
            *at_mut(&mut map, raw as usize)? = to_u32(order.len())?;
            order.push(b);
        }
    }

    let mut work = Grouping {
        succ: Vec::new(),
        group_of: scratch.filled(meter, order.len(), NONE)?,
        group_len: Vec::new(),
        literals: Vec::new(),
        flat: Vec::new(),
    };
    // Nothing may fail between creating `sc` and the block below, which releases it on error.
    let mut sc = StructCfg::with_capacity(meter, count, order.len())?;
    let built = (|| {
        for &b in &order {
            meter.tick(1)?;
            let term = work.term(view, b, &map, meter, scratch)?;
            sc.push(
                meter,
                Block {
                    content: Content::Code(b),
                    term,
                },
            )?;
        }
        Ok(())
    })();
    match built {
        Ok(()) => Ok(sc),
        Err(e) => {
            meter.release(sc.ledger_bytes());
            Err(e)
        }
    }
}

/// Reusable buffers for translating one terminator.
struct Grouping {
    succ: Vec<Node>,
    /// Node -> group index while one switch is grouped (`NONE` otherwise).
    group_of: Vec<u32>,
    group_len: Vec<(Node, u32)>,
    literals: Vec<u64>,
    /// One switch's literals laid out group by group.
    flat: Vec<u64>,
}

impl Grouping {
    fn term(
        &mut self,
        view: &impl CfgView,
        b: BlockId,
        map: &[u32],
        meter: &mut Meter,
        scratch: &mut Scratch,
    ) -> Result<Term, Error> {
        self.succ.clear();
        for s in view.successors(b) {
            let node = *at(map, s.raw() as usize)?;
            if node == NONE {
                return Err(invariant(format!(
                    "successor {s:?} of reachable {b:?} is unreachable"
                )));
            }
            scratch.push(meter, &mut self.succ, Node::new(node))?;
        }
        meter.tick(to_u32(self.succ.len())?.saturating_add(1))?;
        let arity = |n: usize| -> Result<(), Error> {
            if self.succ.len() == n {
                Ok(())
            } else {
                Err(invariant(format!(
                    "{b:?}: terminator shape expects {n} successors, the graph has {}",
                    self.succ.len()
                )))
            }
        };
        match view.terminator(b) {
            TermShape::Br => {
                arity(1)?;
                Ok(Term::Br(*at(&self.succ, 0)?))
            }
            TermShape::CondBr(cond) => {
                arity(2)?;
                let (t, f) = (*at(&self.succ, 0)?, *at(&self.succ, 1)?);
                Ok(if t == f {
                    Term::Br(t)
                } else {
                    Term::CondBr { cond, t, f }
                })
            }
            TermShape::Switch(sel, literals) => {
                arity(literals.len().saturating_add(1))?;
                self.switch(b, sel, &literals, meter, scratch)
            }
            TermShape::Return(v) => {
                arity(0)?;
                Ok(Term::Return(v))
            }
            TermShape::Unreachable => {
                arity(0)?;
                Ok(Term::Unreachable)
            }
        }
    }

    fn switch(
        &mut self,
        b: BlockId,
        sel: crate::air::ir::ValueId,
        literals: &[u64],
        meter: &mut Meter,
        scratch: &mut Scratch,
    ) -> Result<Term, Error> {
        // Distinct literals (the view's contract).
        self.literals.clear();
        for &l in literals {
            scratch.push(meter, &mut self.literals, l)?;
        }
        self.literals.sort_unstable();
        if self.literals.windows(2).any(|w| w.first() == w.get(1)) {
            return Err(invariant(format!("{b:?}: duplicate switch literal")));
        }

        // Group by target in first-appearance order, skipping cases to the default.
        let default = *at(&self.succ, 0)?;
        self.group_len.clear();
        for &t in self.succ.get(1..).unwrap_or(&[]) {
            if t == default {
                continue;
            }
            let g = at_mut(&mut self.group_of, t.index())?;
            if *g == NONE {
                *g = to_u32(self.group_len.len())?;
                scratch.push(meter, &mut self.group_len, (t, 0))?;
            }
            let g = *g as usize;
            let len = &mut at_mut(&mut self.group_len, g)?.1;
            *len = len.saturating_add(1);
        }
        let cases = self.fill_groups(literals, meter, scratch);
        for &(t, _) in &self.group_len {
            *at_mut(&mut self.group_of, t.index())? = NONE;
        }
        let cases = cases?;
        if cases.is_empty() {
            return Ok(Term::Br(default));
        }
        Ok(Term::Switch {
            sel,
            default,
            cases,
        })
    }

    /// The literals of each group (`group_of` still set), in one pass over the cases.
    ///
    /// The working buffers (`flat`, reused across switches, and `cases`) are charged to `scratch`.
    /// The per-case literal boxes are the output: their total is checked against the ledger
    /// before they are allocated, and the block's [`StructCfg::push`] charges the finished
    /// terminator for real.
    fn fill_groups(
        &mut self,
        literals: &[u64],
        meter: &mut Meter,
        scratch: &mut Scratch,
    ) -> Result<Box<[(Box<[u64]>, Node)]>, Error> {
        let boxed = literals.len().saturating_mul(std::mem::size_of::<u64>());
        meter.charge(boxed)?;
        meter.release(boxed);

        // Group g's literals go to flat[start(g) ..]; `group_len.1` becomes the fill cursor.
        let mut start = 0u32;
        for (_, len) in self.group_len.iter_mut() {
            let n = *len;
            *len = start;
            start = start.saturating_add(n);
        }
        self.flat.clear();
        for _ in 0..start {
            scratch.push(meter, &mut self.flat, 0)?;
        }
        for (&l, &s) in literals.iter().zip(self.succ.get(1..).unwrap_or(&[])) {
            let g = *at(&self.group_of, s.index())?;
            if g == NONE {
                continue; // a case to the default
            }
            let cursor = &mut at_mut(&mut self.group_len, g as usize)?.1;
            *at_mut(&mut self.flat, *cursor as usize)? = l;
            *cursor = cursor.saturating_add(1);
        }
        meter.tick(to_u32(literals.len())?)?;
        // Each cursor now ends where the next group starts.
        let mut cases = scratch.with_capacity(meter, self.group_len.len())?;
        let mut from = 0usize;
        for &(t, end) in &self.group_len {
            let lits = self
                .flat
                .get(from..end as usize)
                .ok_or_else(|| invariant("switch literal grouping out of range"))?;
            cases.push((Box::<[u64]>::from(lits), t));
            from = end as usize;
        }
        Ok(cases.into_boxed_slice())
    }
}
