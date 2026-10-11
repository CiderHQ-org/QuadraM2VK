//! P9: dominance-checked single-store promotion, IR→IR on the emitted module.
//!
//! The dominator tree is computed on the *output* CFG of the function (P5–P8 add guard and
//! dispatch paths, so the input dominance does not carry over). A `Function` variable is promoted
//! when:
//!
//! - its pointer is used only as the pointer operand of `OpLoad`s and of exactly one `OpStore`,
//!   none volatile (no access chain, call argument, phi, copy, or stored pointer lets it escape);
//! - the store is reachable and dominates every load: a load in the store's block comes after the
//!   store, a load elsewhere is in a block the store's block strictly dominates, and no load is in
//!   an unreachable block;
//! - its type is one `OpCopyObject` takes (plain data, not runtime-sized).
//!
//! Then every path from the entry to a load passes the store, and the stored value's definition
//! does not run again between the store's latest execution and the load: the definition dominates
//! the store, so a path re-running it after the store would continue to the load without passing
//! the store again, and prefixing it with a path to the definition that avoids the store (one
//! exists, or the store would dominate its own operand's definition) would reach the load
//! avoiding the store. The load therefore reads exactly the stored SSA value. It becomes
//! `OpCopyObject` of that value (its result stays, so no use moves), the store goes, and the
//! variable leaves the function
//! ([`TypedModule::apply_promotions`]). The value's definition dominates the store, which
//! dominates the load, so SSA dominance holds by construction; the finalize verifier re-checks it.
//!
//! Work is linear in the function's instructions plus one dominator computation, metered on the
//! per-attempt [`Meter`] with every table charged to its ledger.

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::graph::{Cfg, DomTree, Graph};
use crate::spirv::ids::{BlockId, FuncId, Value, VarId};
use crate::spirv::ir::{Inst, InstAt, Promotion, Shape, Terminator, TypedModule};

/// What P9 did to a module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PromoteReport {
    /// Variables promoted and removed.
    pub promoted: u32,
    /// Variables kept (escaping, several or no stores, or a load the store does not dominate).
    pub kept: u32,
}

fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// A function's blocks as a graph over layout positions (the entry is position 0).
struct FnGraph {
    succ: Vec<Vec<u32>>,
}

impl Graph for FnGraph {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.succ.len()
    }

    fn entry(&self) -> u32 {
        0
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.succ
            .get(node as usize)
            .map_or(&[][..], |s| &s[..])
            .iter()
            .copied()
    }
}

fn targets(t: &Terminator) -> Vec<BlockId> {
    match t {
        Terminator::Branch(b) => vec![*b],
        Terminator::CondBranch { t, f, .. } => vec![*t, *f],
        Terminator::Switch { default, cases, .. } => std::iter::once(*default)
            .chain(cases.iter().map(|(_, b)| *b))
            .collect(),
        _ => Vec::new(),
    }
}

/// What the scan saw of one variable.
#[derive(Clone, Debug, Default)]
struct Uses {
    escaped: bool,
    stores: Vec<(u32, usize)>,
    loads: Vec<(u32, usize)>,
}

/// Run P9 on every function of `m`, then apply all promotions at once. On error `m` is unchanged.
pub fn promote_single_store(
    m: &mut TypedModule,
    meter: &mut Meter,
) -> Result<PromoteReport, Error> {
    let mut promotions = Vec::new();
    let mut vars = 0usize;
    // Each value's variable slot in the function being planned (`u32::MAX`: not one of its
    // variables), one table for every function: the scan looks up every operand it reads.
    let mut table = 0usize;
    charge::<u32>(meter, &mut table, m.values().len())?;
    let mut slots = vec![u32::MAX; m.values().len()];
    for fi in 0..m.functions().len() {
        let func = FuncId::new(u32::try_from(fi).map_err(|_| invariant("function count"))?);
        vars = vars.saturating_add(m.functions().get(fi).map_or(0, |f| f.vars.len()));
        let mut ledger = 0usize;
        let r = plan(m, func, &mut slots, meter, &mut ledger);
        meter.release(ledger);
        match r {
            Ok(r) => promotions.extend(r),
            Err(e) => {
                meter.release(table);
                return Err(e);
            }
        }
    }
    meter.release(table);
    let report = PromoteReport {
        promoted: u32::try_from(promotions.len()).unwrap_or(u32::MAX),
        kept: u32::try_from(vars.saturating_sub(promotions.len())).unwrap_or(u32::MAX),
    };
    if !promotions.is_empty() {
        m.apply_promotions(&promotions)?;
    }
    Ok(report)
}

/// Charge `n` elements of `T` to the ledger.
fn charge<T>(meter: &mut Meter, ledger: &mut usize, n: usize) -> Result<(), Error> {
    let bytes = n.saturating_mul(std::mem::size_of::<T>());
    meter.charge(bytes)?;
    *ledger = ledger.saturating_add(bytes);
    Ok(())
}

/// Decide the promotions of `func` (read-only), with `slots` (every entry `u32::MAX`, and so
/// again on return) for its variables' slots.
fn plan(
    m: &TypedModule,
    func: FuncId,
    slots: &mut [u32],
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Vec<Promotion>, Error> {
    let f = m
        .functions()
        .get(func.index())
        .ok_or_else(|| invariant(format!("{func:?} out of range")))?;
    if f.vars.is_empty() {
        return Ok(Vec::new());
    }
    let blocks = m.blocks();

    // Layout position of each block of the function (sorted by block id for lookup).
    charge::<(BlockId, u32)>(meter, ledger, f.blocks.len())?;
    let mut pos: Vec<(BlockId, u32)> = f
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| u32::try_from(i).map(|i| (*b, i)))
        .collect::<Result<_, _>>()
        .map_err(|_| invariant("block count exceeds u32"))?;
    pos.sort_unstable_by_key(|(b, _)| *b);
    let pos_of = |b: BlockId| -> Result<u32, Error> {
        pos.binary_search_by_key(&b, |(k, _)| *k)
            .ok()
            .and_then(|i| pos.get(i))
            .map(|(_, p)| *p)
            .ok_or_else(|| invariant(format!("{b:?} is not a block of {func:?}")))
    };
    let block = |b: BlockId| {
        blocks
            .get(b.index())
            .ok_or_else(|| invariant(format!("{b:?} out of range")))
    };

    // The output CFG and its dominators.
    let mut succ = Vec::with_capacity(f.blocks.len());
    charge::<Vec<u32>>(meter, ledger, f.blocks.len())?;
    for b in &f.blocks {
        meter.tick(1)?;
        let ts = targets(&block(*b)?.term);
        charge::<u32>(meter, ledger, ts.len())?;
        succ.push(ts.into_iter().map(pos_of).collect::<Result<Vec<_>, _>>()?);
    }
    let g = FnGraph { succ };
    let cfg = Cfg::build(&g, meter)?;
    let dom = DomTree::compute(&cfg, meter).map_err(Error::from);
    let result = dom.and_then(|dom| {
        let r = decide(m, func, &f.blocks, &dom, slots, meter, ledger);
        meter.release(dom.ledger_bytes());
        r
    });
    meter.release(cfg.ledger_bytes());
    result
}

/// [`plan`]'s decision, `slots` holding each of `func`'s variables' slot while it runs.
fn decide(
    m: &TypedModule,
    func: FuncId,
    order: &[BlockId],
    dom: &DomTree<u32>,
    slots: &mut [u32],
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Vec<Promotion>, Error> {
    let f = m
        .functions()
        .get(func.index())
        .ok_or_else(|| invariant(format!("{func:?} out of range")))?;
    // Pointer value -> variable slot.
    let mut ptrs = Vec::with_capacity(f.vars.len());
    charge::<Value>(meter, ledger, f.vars.len())?;
    for v in &f.vars {
        let lv = m
            .vars()
            .get(v.index())
            .ok_or_else(|| invariant(format!("{v:?} out of range")))?;
        ptrs.push(lv.value);
    }
    for (slot, v) in ptrs.iter().enumerate() {
        if let Some(s) = slots.get_mut(v.index()) {
            *s = u32::try_from(slot).unwrap_or(u32::MAX);
        }
    }
    let r = decide_slots(m, func, order, dom, slots, meter, ledger);
    for v in &ptrs {
        if let Some(s) = slots.get_mut(v.index()) {
            *s = u32::MAX;
        }
    }
    r
}

/// [`decide`] with `slots` set.
fn decide_slots(
    m: &TypedModule,
    func: FuncId,
    order: &[BlockId],
    dom: &DomTree<u32>,
    slots: &[u32],
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Vec<Promotion>, Error> {
    let f = m
        .functions()
        .get(func.index())
        .ok_or_else(|| invariant(format!("{func:?} out of range")))?;
    let slot_of = |v: Value| {
        slots
            .get(v.index())
            .copied()
            .filter(|s| *s != u32::MAX)
            .map(|s| s as usize)
    };
    charge::<Uses>(meter, ledger, f.vars.len())?;
    let mut uses: Vec<Uses> = vec![Uses::default(); f.vars.len()];
    let escape = |uses: &mut Vec<Uses>, v: Value| {
        if let Some(u) = slot_of(v).and_then(|s| uses.get_mut(s)) {
            u.escaped = true;
        }
    };

    for (p, b) in order.iter().enumerate() {
        let p = u32::try_from(p).map_err(|_| invariant("block count exceeds u32"))?;
        let blk = m
            .blocks()
            .get(b.index())
            .ok_or_else(|| invariant(format!("{b:?} out of range")))?;
        meter.tick(
            u32::try_from(blk.insts.len())
                .unwrap_or(u32::MAX)
                .saturating_add(1),
        )?;
        for phi in &blk.phis {
            for (v, _) in &phi.incoming {
                escape(&mut uses, *v);
            }
        }
        for (i, n) in blk.insts.iter().enumerate() {
            match &n.inst {
                Inst::Load { ptr, access } => match slot_of(*ptr).and_then(|s| uses.get_mut(s)) {
                    Some(u) if !access.volatile => {
                        charge::<(u32, usize)>(meter, ledger, 1)?;
                        u.loads.push((p, i));
                    }
                    Some(u) => u.escaped = true,
                    None => {}
                },
                Inst::Store { ptr, value, access } => {
                    escape(&mut uses, *value);
                    match slot_of(*ptr).and_then(|s| uses.get_mut(s)) {
                        Some(u) if !access.volatile => {
                            charge::<(u32, usize)>(meter, ledger, 1)?;
                            u.stores.push((p, i));
                        }
                        Some(u) => u.escaped = true,
                        None => {}
                    }
                }
                other => other.for_each_value(|v| escape(&mut uses, v)),
            }
        }
        blk.term.for_each_value(|v| escape(&mut uses, v));
    }

    let mut out = Vec::new();
    for (slot, var) in f.vars.iter().enumerate() {
        meter.tick(1)?;
        let Some(u) = uses.get(slot) else { continue };
        if !copyable(m, *var)? {
            continue;
        }
        if let Some(pr) = promotable(u, dom, order, func, *var) {
            charge::<Promotion>(meter, ledger, 1)?;
            out.push(pr);
        }
    }
    Ok(out)
}

/// Whether `var`'s type is one `OpCopyObject` takes (the form a promoted load becomes).
fn copyable(m: &TypedModule, var: VarId) -> Result<bool, Error> {
    let lv = m
        .vars()
        .get(var.index())
        .ok_or_else(|| invariant(format!("{var:?} out of range")))?;
    let f = m.types().facts(lv.pointee)?;
    Ok(matches!(f.shape, Shape::Plain | Shape::Implicit | Shape::Explicit) && !f.runtime_sized)
}

/// The promotion of `var` when its uses allow one.
fn promotable(
    u: &Uses,
    dom: &DomTree<u32>,
    order: &[BlockId],
    func: FuncId,
    var: VarId,
) -> Option<Promotion> {
    let [(sp, si)] = u.stores[..] else {
        return None;
    };
    if u.escaped || !dom.is_reachable(sp) {
        return None;
    }
    let dominated = |&(lp, li): &(u32, usize)| {
        dom.is_reachable(lp)
            && if lp == sp {
                si < li
            } else {
                dom.strictly_dominates(sp, lp)
            }
    };
    if !u.loads.iter().all(dominated) {
        return None;
    }
    let at = |(p, index): (u32, usize)| {
        order.get(p as usize).map(|block| InstAt {
            block: *block,
            index,
        })
    };
    Some(Promotion {
        func,
        var,
        store: at((sp, si))?,
        loads: u.loads.iter().map(|l| at(*l)).collect::<Option<Vec<_>>>()?,
    })
}
