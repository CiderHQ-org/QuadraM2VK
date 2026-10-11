//! Sparse conditional constant propagation.
//!
//! One standard algorithm (Wegman–Zadeck) per function: an SSA worklist and a flow worklist over
//! executable edges, a lattice per instruction, and folding by [`Folder`]. Two AIR-specific
//! lattice facts are added:
//!
//! - `air.is_function_constant_defined(@g)` folds to whether `@g`'s constant was supplied, and a
//!   load from a bound (supplied) function constant folds to its value, like any load from a
//!   constant global with a known initializer.
//! - A value computed from an **unsupplied** function constant is [`Lat::Fc`]: the constant is
//!   undefined, and the taint sticks through every pure operation. Pure uses of it may be deleted
//!   when their result is unused (they become `undef`), but any other use surviving SCCP (a branch
//!   on it, a store of it, a call argument, a return value) is
//!   `Error::NeedsInput(FunctionConstant{index})`.
//!
//! Afterwards unreachable blocks are deleted, branches on constants become unconditional, PHIs
//! keep one entry per remaining edge, and instructions with a constant value are replaced by it.
//! A reachable branch on a literal `undef`/`poison` is refused (`CanonError::UndefCondition`).
//!
//! A PHI's value is met incrementally: each PHI keeps the meet of its entries' values over the
//! executable edges so far, and an entry's value is met into it once when its edge becomes
//! executable and once each time that value changes, so a visit reads the PHI's value instead of
//! meeting all its entries again. The lattice values only move down and the meet is
//! commutative, associative, and idempotent, so the kept meet is, at every visit, the meet a full
//! pass over the executable entries computes: the worklists see the same values in the same
//! order, and the fixed point, the folds, and the dead blocks are the same. Every meet, the
//! index, and both worklists are metered.
//!
//! [`entry_block_refusal`] decides the one refusal the entry's own entry block settles before the
//! entry is inlined, so a module that needs a caller's value there is refused without building its
//! inlined body.

use std::collections::{HashMap, HashSet};

use crate::budget::{vec_bytes, Meter};
use crate::error::{AbiError, CanonError, DiagName, InputError, InputErrorKind, Location};
use crate::error::{Error, MissingInput};
use crate::intrinsics::{AirIntrinsic, FunctionConstant};
use crate::ir::{Arena, BlockId, Body, Callee, Const, ConstId, ConstTable, Constant, Global};
use crate::ir::{ArgIdx, GlobalId, Idx, InstId, InstKind, Value};

use super::fold::Folder;
use super::rewrite::{apply, filled, invariant, scoped, BodyEdit};

/// A function constant as SCCP sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FcState {
    pub index: u32,
    pub supplied: bool,
    /// A declared function constant (its `air.fc_initializer` global), as opposed to a global a
    /// static initializer derived from an unsupplied one, whose reads are tainted the same way but
    /// which `air.is_function_constant_defined` may not name.
    pub decl: bool,
}

/// The SCCP lattice. Ordered `Top < Const < Bottom < Fc`: an unsupplied-constant taint is the
/// strongest fact, because every use of it must be accounted for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lat {
    Top,
    Const(ConstId),
    Bottom,
    Fc(u32),
}

#[cfg(test)]
thread_local! {
    /// Tests only: the meets this thread has made, whoever made them (the boundedness test counts
    /// the real work with it, metered or not).
    static MEETS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn meet(a: Lat, b: Lat) -> Lat {
    #[cfg(test)]
    MEETS.with(|n| n.set(n.get().saturating_add(1)));
    match (a, b) {
        (Lat::Top, x) | (x, Lat::Top) => x,
        (Lat::Fc(i), Lat::Fc(j)) => Lat::Fc(i.min(j)),
        (Lat::Fc(i), _) | (_, Lat::Fc(i)) => Lat::Fc(i),
        (Lat::Bottom, _) | (_, Lat::Bottom) => Lat::Bottom,
        (Lat::Const(x), Lat::Const(y)) if x == y => Lat::Const(x),
        (Lat::Const(_), Lat::Const(_)) => Lat::Bottom,
    }
}

/// What SCCP needs from the module around the function.
pub(super) struct Context<'a> {
    pub folder: Folder<'a>,
    pub globals: &'a Arena<GlobalId, Global>,
    pub fcs: &'a HashMap<GlobalId, FcState>,
    pub name: DiagName,
}

/// Run SCCP over `body` and rebuild it. The solver's scope ends with its edit, so its tables are
/// back in the ledger before the new body is built: the rebuild holds only the body it reads,
/// the edit, and the body it builds (with the solver held too, a 1.9M-instruction body that folds
/// nothing passed the ledger here).
pub(super) fn run(body: &Body, cx: &mut Context<'_>, m: &mut Meter) -> Result<Body, Error> {
    let out = scoped(m, |m, edit_held| {
        let edit = scoped(m, |m, held| {
            let mut s = Solver::new(body, cx, m, held)?;
            s.solve(cx, m, held)?;
            s.check(cx)?;
            s.edit(cx, m, edit_held)
        })?;
        apply(body, &edit, &|c| c, m)
    });
    // Every SCCP a unit test runs is also solved by meeting each PHI's entries in full at every
    // visit, as before, and must agree.
    #[cfg(test)]
    tests::agrees_with_the_full_meet(body, cx, &out);
    out
}

/// What SCCP's fixpoint says of an instruction before the body is devirtualized: the function
/// constants bound so far settle it, or make it one constant, or leave it to run time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fact {
    /// It is computed from an unsupplied constant, whose first use that is not pure SCCP
    /// refuses.
    Settled,
    /// It is this constant on every run that reaches it.
    Const(ConstId),
    /// It varies from run to run, or SCCP has nothing to say of it.
    Varies,
}

/// Each instruction's [`Fact`] in a body.
pub(super) struct Facts {
    of: Vec<Fact>,
}

impl Facts {
    /// The fact of `v`: an argument or a constant varies, as far as this is concerned.
    pub(super) fn of(&self, v: Value) -> Fact {
        match v {
            Value::Inst(i) => self.of.get(i.index()).copied().unwrap_or(Fact::Varies),
            Value::Arg(_) | Value::Const(_) => Fact::Varies,
        }
    }
}

/// Each instruction's [`Fact`] in `body` as it stands: SCCP's fixpoint, without its rewrite. The
/// facts are charged to `held`, the fixpoint only while it runs.
///
/// The fixpoint treats the result of each call the body still makes as varying, so a call that
/// is inlined later can only refine the values its result reaches. A value it reaches computed
/// from an unsupplied constant is settled. A value it does not reach varies: a block no edge
/// reaches yet (a recursive function's copy, before its machine is assembled) or one only a branch
/// it cannot decide leads to can run once the body is refined. A constant it finds is a fact only
/// where its check passes: a branch on an unsupplied constant's value or on `undef` runs neither
/// way in the fixpoint, so where the check would refuse one, a refinement that unties it can run a
/// block the fixpoint found not running, and a constant met there can vary.
pub(super) fn facts(
    body: &Body,
    cx: &mut Context<'_>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Facts, Error> {
    let mut of = filled(m, held, body.insts.len(), Fact::Varies)?;
    scoped(m, |m, scratch| {
        let mut s = Solver::new(body, cx, m, scratch)?;
        s.solve(cx, m, scratch)?;
        let checked = s.check(cx).is_ok();
        for (i, slot) in of.iter_mut().enumerate() {
            m.tick(1)?;
            let live = s
                .block_of
                .get(i)
                .and_then(|b| s.exec.get(b.index()))
                .copied()
                .unwrap_or(false);
            *slot = match (live, s.val.get(i).copied().unwrap_or(Lat::Bottom)) {
                (true, Lat::Fc(_)) => Fact::Settled,
                (true, Lat::Const(k)) if checked => Fact::Const(k),
                _ => Fact::Varies,
            };
        }
        Ok(())
    })?;
    Ok(Facts { of })
}

/// Make room for `extra` more items on a worklist, charging the capacity the growth adds to the
/// ledger and to `held`.
fn room<T>(list: &mut Vec<T>, extra: usize, m: &mut Meter, held: &mut usize) -> Result<(), Error> {
    let before = vec_bytes(list);
    m.grow(list, extra)?;
    *held = held.saturating_add(vec_bytes(list).saturating_sub(before));
    Ok(())
}

/// Whether `kind` computes a value from its operands alone (no memory, no side effects): its
/// uses of an unsupplied constant only taint its result.
pub(super) fn is_pure(kind: &InstKind) -> bool {
    match kind {
        InstKind::BinOp { .. }
        | InstKind::FNeg { .. }
        | InstKind::Cast { .. }
        | InstKind::ICmp { .. }
        | InstKind::FCmp { .. }
        | InstKind::Select { .. }
        | InstKind::Freeze(_)
        | InstKind::ExtractElement { .. }
        | InstKind::InsertElement { .. }
        | InstKind::ShuffleVector { .. }
        | InstKind::ExtractValue { .. }
        | InstKind::InsertValue { .. }
        | InstKind::Gep { .. }
        | InstKind::Phi { .. } => true,
        InstKind::Load {
            volatile, atomic, ..
        } => !volatile && atomic.is_none(),
        _ => false,
    }
}

/// The function-constant global a call to `air.is_function_constant_defined` names.
fn is_defined_arg(
    kind: &InstKind,
    consts: &ConstTable,
    fcs: &HashMap<GlobalId, FcState>,
) -> Option<Option<GlobalId>> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    let Callee::Air {
        op: AirIntrinsic::FunctionConstant(FunctionConstant::IsDefined),
        ..
    } = call.callee
    else {
        return None;
    };
    let g = match call.args.first() {
        Some(Value::Const(c)) => match consts.get(*c).map(|k| &k.value) {
            Some(Const::GlobalAddr(g)) if fcs.get(g).is_some_and(|f| f.decl) => Some(*g),
            _ => None,
        },
        _ => None,
    };
    Some(g)
}

/// The unsupplied function constant a constant with `value` depends on: its global's own when it
/// is a global's address, otherwise the lowest among its operands' (`operand` gives an
/// operand's). The one rule for [`Solver`] and [`entry_block_refusal`].
fn const_taint(
    value: &Const,
    fcs: &HashMap<GlobalId, FcState>,
    operand: impl Fn(ConstId) -> Option<u32>,
) -> Option<u32> {
    match value {
        Const::GlobalAddr(g) => fcs.get(g).and_then(|f| (!f.supplied).then_some(f.index)),
        v => v.children().iter().filter_map(|&ch| operand(ch)).min(),
    }
}

/// Whether `kind` is `air.normalize_function_constant_predicate`, which SCCP folds: an undefined
/// operand is a use it accounts for, not one that needs the caller's value.
fn is_normalize(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::Call(call) if matches!(
            call.callee,
            Callee::Air {
                op: AirIntrinsic::FunctionConstant(FunctionConstant::NormalizePredicate(_)),
                ..
            }
        )
    )
}

/// What the lattice value of a branch condition decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    /// Not known yet (`Top`), or tainted by an unsupplied constant (`check` refuses it): no
    /// edge is executable yet.
    Wait,
    /// A known integer: exactly one successor.
    Taken(u64),
    /// A literal `undef`/`poison`: no edge; `check` refuses it if the branch is reachable.
    Undef,
    /// Overdefined, or a constant SCCP cannot read as an integer (a constant expression such as
    /// `ptrtoint (ptr @g to i64)`): every successor.
    All,
}

/// The lattice value of `v` given the instructions' values `val` and the constants' `taint`.
fn lat_of(val: &[Lat], taint: &[Option<u32>], v: Value) -> Lat {
    match v {
        Value::Arg(_) => Lat::Bottom,
        Value::Inst(i) => val.get(i.index()).copied().unwrap_or(Lat::Bottom),
        Value::Const(c) => match taint.get(c.index()).copied().flatten() {
            Some(fc) => Lat::Fc(fc),
            None => Lat::Const(c),
        },
    }
}

/// The PHIs' entries, indexed for the incremental meet: by the edge each names and by the
/// instruction whose value it carries.
///
/// A block's PHIs lead it (a verified body's layout), so PHI `i` of block `b` has slot
/// `first[b] + (i - start(b))`. Each slot keeps the meet of its entries' values over the
/// executable edges so far. An entry is met into its PHI when its edge becomes executable
/// ([`Phis::open`]) and again each time its value changes on an executable edge
/// ([`Phis::changed`]), one work unit an entry each time.
struct Phis {
    /// Per block, then one past the last: the slot of the block's first PHI.
    first: Vec<u32>,
    /// Per PHI slot: the meet of its entries' values over the executable edges.
    acc: Vec<Lat>,
    /// The index of each edge `(pred, block)` some PHI entry names, in first-named order.
    edge_of: HashMap<(BlockId, BlockId), u32>,
    /// Per edge index: whether the edge is executable.
    open: Vec<bool>,
    /// Per edge index, then one past the last: its first entry in `by_edge`.
    edge_start: Vec<u32>,
    /// Every entry as (PHI slot, value), grouped by edge.
    by_edge: Vec<(u32, Value)>,
    /// Per instruction, then one past the last: the first entry carrying its value in `by_def`.
    def_start: Vec<u32>,
    /// Every entry whose value is an instruction as (PHI slot, edge index), grouped by that
    /// instruction.
    by_def: Vec<(u32, u32)>,
}

/// The half-open range `start[k]..start[k + 1]` of a CSR table.
fn span(start: &[u32], k: usize) -> std::ops::Range<usize> {
    let lo = start.get(k).copied().unwrap_or(0) as usize;
    let hi = start.get(k + 1).copied().unwrap_or(0) as usize;
    lo..hi.max(lo)
}

/// Work units for `n` items, saturating.
fn units(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

impl Phis {
    /// Index the PHIs of `body`, whose blocks' PHI slots `first` gives. A work unit per PHI and
    /// per entry for each of the two passes; every table is charged to `held`.
    fn new(body: &Body, first: Vec<u32>, m: &mut Meter, held: &mut usize) -> Result<Phis, Error> {
        let n = body.insts.len();
        let phis = first.last().copied().unwrap_or(0) as usize;
        // Each PHI's entries, in slot order, with its block.
        let each = || {
            body.blocks.iter().flat_map(move |(b, block)| {
                block
                    .inst_ids()
                    .map_while(move |i| match body.insts.get(i).map(|x| &x.kind) {
                        Some(InstKind::Phi { incoming, .. }) => Some((b, &**incoming)),
                        _ => None,
                    })
            })
        };
        let total: usize = each().map(|(_, incoming)| incoming.len()).sum();
        let mut edge_of: HashMap<(BlockId, BlockId), u32> = HashMap::new();
        if total == 0 {
            // No entry to index: a function without PHIs holds no table but its slots.
            return Ok(Phis {
                first,
                acc: filled(m, held, phis, Lat::Top)?,
                edge_of,
                open: Vec::new(),
                edge_start: Vec::new(),
                by_edge: Vec::new(),
                def_start: Vec::new(),
                by_def: Vec::new(),
            });
        }
        // Counts first, then (inclusive prefix sums) each group's end, then (filling downwards)
        // its start.
        let mut edge_start = filled(m, held, total + 1, 0u32)?;
        let mut def_start = filled(m, held, n + 1, 0u32)?;
        let mut edges = 0u32;
        let mut defs = 0usize;
        for (b, incoming) in each() {
            m.tick(units(incoming.len() + 1))?;
            for &(v, pred) in incoming {
                let e = match edge_of.get(&(pred, b)) {
                    Some(&e) => e,
                    None => {
                        let bytes = std::mem::size_of::<((BlockId, BlockId), u32)>() * 2;
                        m.charge(bytes)?;
                        *held = held.saturating_add(bytes);
                        edge_of.insert((pred, b), edges);
                        edges += 1;
                        edges - 1
                    }
                };
                if let Some(c) = edge_start.get_mut(e as usize) {
                    *c += 1;
                }
                if let Value::Inst(d) = v {
                    if let Some(c) = def_start.get_mut(d.index()).filter(|_| d.index() < n) {
                        *c += 1;
                        defs += 1;
                    }
                }
            }
        }
        let edges = edges as usize;
        for k in 1..edges {
            let prev = edge_start.get(k - 1).copied().unwrap_or(0);
            if let Some(c) = edge_start.get_mut(k) {
                *c += prev;
            }
        }
        for k in 1..n {
            let prev = def_start.get(k - 1).copied().unwrap_or(0);
            if let Some(c) = def_start.get_mut(k) {
                *c += prev;
            }
        }
        let mut by_edge = filled(m, held, total, (0u32, Value::Arg(ArgIdx::new(0))))?;
        let mut by_def = filled(m, held, defs, (0u32, 0u32))?;
        for ((b, incoming), slot) in each().zip(0u32..) {
            m.tick(units(incoming.len() + 1))?;
            for &(v, pred) in incoming {
                let e = edge_of
                    .get(&(pred, b))
                    .copied()
                    .ok_or_else(|| invariant("a PHI edge left unindexed"))?;
                if let Some(at) = edge_start.get_mut(e as usize) {
                    *at -= 1;
                    if let Some(x) = by_edge.get_mut(*at as usize) {
                        *x = (slot, v);
                    }
                }
                if let Value::Inst(d) = v {
                    if let Some(at) = def_start.get_mut(d.index()).filter(|_| d.index() < n) {
                        *at -= 1;
                        if let Some(x) = by_def.get_mut(*at as usize) {
                            *x = (slot, e);
                        }
                    }
                }
            }
        }
        if let Some(end) = edge_start.get_mut(edges) {
            *end = u32::try_from(total).map_err(|_| invariant("PHI entries past u32"))?;
        }
        if let Some(end) = def_start.get_mut(n) {
            *end = u32::try_from(defs).map_err(|_| invariant("PHI entries past u32"))?;
        }
        Ok(Phis {
            first,
            acc: filled(m, held, phis, Lat::Top)?,
            edge_of,
            open: filled(m, held, edges, false)?,
            edge_start,
            by_edge,
            def_start,
            by_def,
        })
    }

    /// The slot of PHI `i`, which leads block `b` of `body`.
    fn slot(&self, body: &Body, b: BlockId, i: InstId) -> Option<usize> {
        let start = body.blocks.get(b)?.start;
        let k = i.index().checked_sub(start.index())?;
        let lo = *self.first.get(b.index())? as usize;
        let hi = *self.first.get(b.index() + 1)? as usize;
        (lo + k < hi).then_some(lo + k)
    }

    /// Edge `(from, to)` became executable: meet each entry naming it into its PHI.
    fn open(
        &mut self,
        from: BlockId,
        to: BlockId,
        val: &[Lat],
        taint: &[Option<u32>],
        m: &mut Meter,
    ) -> Result<(), Error> {
        let Some(&e) = self.edge_of.get(&(from, to)) else {
            return Ok(());
        };
        if let Some(o) = self.open.get_mut(e as usize) {
            *o = true;
        }
        let entries = self
            .by_edge
            .get(span(&self.edge_start, e as usize))
            .unwrap_or(&[]);
        m.tick(units(entries.len()))?;
        for &(s, v) in entries {
            if let Some(a) = self.acc.get_mut(s as usize) {
                *a = meet(*a, lat_of(val, taint, v));
            }
        }
        Ok(())
    }

    /// Instruction `d`'s value changed to `now`: meet it into each PHI entry carrying it on an
    /// executable edge.
    fn changed(&mut self, d: InstId, now: Lat, m: &mut Meter) -> Result<(), Error> {
        let entries = self
            .by_def
            .get(span(&self.def_start, d.index()))
            .unwrap_or(&[]);
        if entries.is_empty() {
            return Ok(());
        }
        m.tick(units(entries.len()))?;
        for &(s, e) in entries {
            if self.open.get(e as usize).copied().unwrap_or(false) {
                if let Some(a) = self.acc.get_mut(s as usize) {
                    *a = meet(*a, now);
                }
            }
        }
        Ok(())
    }
}

struct Solver<'b> {
    body: &'b Body,
    block_of: Vec<BlockId>,
    /// Users of each instruction (CSR: `user_start[i]..user_start[i+1]` in `users`).
    user_start: Vec<u32>,
    users: Vec<InstId>,
    /// Per constant present when SCCP started: the unsupplied function constant it depends on.
    taint: Vec<Option<u32>>,
    val: Vec<Lat>,
    exec: Vec<bool>,
    edges: HashSet<(BlockId, BlockId)>,
    phis: Phis,
    /// The two worklists, their capacity charged as they grow.
    flow: Vec<(BlockId, BlockId)>,
    ssa: Vec<InstId>,
    /// Tests only: meet every executable entry of a PHI at each visit, as SCCP once did.
    #[cfg(test)]
    full_meet: bool,
    /// Tests only: at each visit of a PHI, check its slot against the full meet.
    #[cfg(test)]
    check_slots: bool,
}

impl<'b> Solver<'b> {
    fn new(
        body: &'b Body,
        cx: &Context<'_>,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Solver<'b>, Error> {
        let n = body.insts.len();
        let mut block_of = filled(m, held, n, BlockId::new(0))?;
        // Per block: the slot of its first PHI; a PHI after a block's first other instruction is
        // not a verified body's.
        let mut first = filled(m, held, body.blocks.len() + 1, 0u32)?;
        let mut phis = 0u32;
        for (b, block) in body.blocks.iter() {
            if let Some(slot) = first.get_mut(b.index()) {
                *slot = phis;
            }
            let mut leading = true;
            for i in block.inst_ids() {
                if let Some(slot) = block_of.get_mut(i.index()) {
                    *slot = b;
                }
                let phi = matches!(
                    body.insts.get(i).map(|x| &x.kind),
                    Some(InstKind::Phi { .. })
                );
                if phi && !leading {
                    return Err(invariant("a PHI after its block's first other instruction"));
                }
                leading &= phi;
                phis = phis.saturating_add(u32::from(phi));
            }
        }
        if let Some(slot) = first.last_mut() {
            *slot = phis;
        }
        let mut user_start = filled(m, held, n + 1, 0u32)?;
        let mut total = 0usize;
        for inst in body.insts.values() {
            m.tick(1)?;
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    if let Some(c) = user_start.get_mut(d.index() + 1) {
                        *c += 1;
                        total += 1;
                    }
                }
            });
        }
        for i in 1..=n {
            let prev = user_start.get(i - 1).copied().unwrap_or(0);
            if let Some(c) = user_start.get_mut(i) {
                *c += prev;
            }
        }
        let mut users = filled(m, held, total, InstId::new(0))?;
        let mut next = user_start.clone();
        *held = held.saturating_add(next.len() * std::mem::size_of::<u32>());
        m.charge(next.len() * std::mem::size_of::<u32>())?;
        for (i, inst) in body.insts.iter() {
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    if let Some(at) = next.get_mut(d.index()) {
                        if let Some(slot) = users.get_mut(*at as usize) {
                            *slot = i;
                        }
                        *at += 1;
                    }
                }
            });
        }
        let consts = &cx.folder.consts;
        let mut taint = filled(m, held, consts.len(), None)?;
        for (c, k) in consts.iter() {
            m.tick(1)?;
            let t = const_taint(&k.value, cx.fcs, |ch| {
                taint.get(ch.index()).copied().flatten()
            });
            if let Some(slot) = taint.get_mut(c.index()) {
                *slot = t;
            }
        }
        Ok(Solver {
            body,
            block_of,
            user_start,
            users,
            taint,
            val: filled(m, held, n, Lat::Top)?,
            exec: filled(m, held, body.blocks.len(), false)?,
            edges: HashSet::new(),
            phis: Phis::new(body, first, m, held)?,
            flow: Vec::new(),
            ssa: Vec::new(),
            #[cfg(test)]
            full_meet: false,
            #[cfg(test)]
            check_slots: false,
        })
    }

    fn lat(&self, v: Value) -> Lat {
        lat_of(&self.val, &self.taint, v)
    }

    /// What branching on `v` decides.
    fn decide(&self, v: Value, cx: &Context<'_>) -> Decision {
        match self.lat(v) {
            Lat::Top | Lat::Fc(_) => Decision::Wait,
            Lat::Bottom => Decision::All,
            Lat::Const(k) => match cx.folder.int(k) {
                Some(x) => Decision::Taken(x),
                None => match cx.folder.consts.get(k).map(|c| &c.value) {
                    Some(Const::Undef | Const::Poison) => Decision::Undef,
                    _ => Decision::All,
                },
            },
        }
    }

    fn solve(
        &mut self,
        cx: &mut Context<'_>,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        self.exec_block(BlockId::new(0), m, held)?;
        loop {
            if let Some((from, to)) = self.flow.pop() {
                m.tick(1)?;
                if !self.edges.insert((from, to)) {
                    continue;
                }
                let bytes = std::mem::size_of::<(BlockId, BlockId)>() * 2;
                m.charge(bytes)?;
                *held = held.saturating_add(bytes);
                // The PHIs of `to` meet the values the edge brings before any of them is visited.
                self.phis.open(from, to, &self.val, &self.taint, m)?;
                if self.exec.get(to.index()).copied().unwrap_or(true) {
                    // Newly executable edge into a live block: its PHIs see one more input.
                    let phis = span(&self.phis.first, to.index());
                    if let Some(block) = self.body.blocks.get(to) {
                        room(&mut self.ssa, phis.len(), m, held)?;
                        let start = block.start.raw();
                        for k in 0..units(phis.len()) {
                            self.ssa.push(InstId::new(start.saturating_add(k)));
                        }
                    }
                } else {
                    self.exec_block(to, m, held)?;
                }
                continue;
            }
            if let Some(i) = self.ssa.pop() {
                m.tick(1)?;
                let live = self
                    .block_of
                    .get(i.index())
                    .and_then(|b| self.exec.get(b.index()))
                    .copied()
                    .unwrap_or(false);
                if live {
                    self.visit(i, cx, m, held)?;
                }
                continue;
            }
            return Ok(());
        }
    }

    fn exec_block(&mut self, b: BlockId, m: &mut Meter, held: &mut usize) -> Result<(), Error> {
        if let Some(e) = self.exec.get_mut(b.index()) {
            *e = true;
        }
        let block = self
            .body
            .blocks
            .get(b)
            .ok_or_else(|| invariant("block out of range"))?;
        room(&mut self.ssa, block.len(), m, held)?;
        self.ssa.extend(block.inst_ids().rev());
        Ok(())
    }

    /// Push a CFG edge onto the flow worklist.
    fn flow_to(
        &mut self,
        edge: (BlockId, BlockId),
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        room(&mut self.flow, 1, m, held)?;
        self.flow.push(edge);
        Ok(())
    }

    /// Lower instruction `i`'s value to its meet with `new`; when that changes it, its users are
    /// visited again and the PHI entries carrying it meet the new value.
    fn set(&mut self, i: InstId, new: Lat, m: &mut Meter, held: &mut usize) -> Result<(), Error> {
        let Some(slot) = self.val.get_mut(i.index()) else {
            return Ok(());
        };
        let merged = meet(*slot, new);
        if merged == *slot {
            return Ok(());
        }
        *slot = merged;
        if let Some(us) = self.users.get(span(&self.user_start, i.index())) {
            room(&mut self.ssa, us.len(), m, held)?;
            self.ssa.extend_from_slice(us);
        }
        self.phis.changed(i, merged, m)
    }

    /// The value of PHI `i` of block `b`: the meet its slot keeps.
    fn phi(&self, i: InstId, b: BlockId) -> Result<Lat, Error> {
        let slot = self
            .phis
            .slot(self.body, b, i)
            .ok_or_else(|| invariant("a PHI without a slot"))?;
        let value = self
            .phis
            .acc
            .get(slot)
            .copied()
            .ok_or_else(|| invariant("PHI slot out of range"))?;
        #[cfg(test)]
        self.check_slot(i, b, value);
        Ok(value)
    }

    /// The lattice value of a pure operation over `ops`: tainted if any operand is, unknown while
    /// any is unknown, `fold` over constants, otherwise overdefined.
    fn pure(
        &self,
        ops: &[Value],
        cx: &mut Context<'_>,
        m: &mut Meter,
        fold: impl FnOnce(&mut Folder<'_>, &[ConstId], &mut Meter) -> Result<Option<ConstId>, Error>,
    ) -> Result<Lat, Error> {
        let mut consts = Vec::with_capacity(ops.len());
        let mut any_top = false;
        let mut any_bottom = false;
        let mut fc: Option<u32> = None;
        for &v in ops {
            match self.lat(v) {
                Lat::Fc(i) => fc = Some(fc.map_or(i, |j| j.min(i))),
                Lat::Top => any_top = true,
                Lat::Bottom => any_bottom = true,
                Lat::Const(c) => consts.push(c),
            }
        }
        Ok(if let Some(i) = fc {
            Lat::Fc(i)
        } else if any_top {
            Lat::Top
        } else if any_bottom {
            Lat::Bottom
        } else {
            fold(&mut cx.folder, &consts, m)?.map_or(Lat::Bottom, Lat::Const)
        })
    }

    fn visit(
        &mut self,
        i: InstId,
        cx: &mut Context<'_>,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        let body = self.body;
        let inst = body
            .insts
            .get(i)
            .ok_or_else(|| invariant("instruction out of range"))?;
        let b = self
            .block_of
            .get(i.index())
            .copied()
            .unwrap_or(BlockId::new(0));
        let ty = inst.ty;
        let never = |_: &mut Folder<'_>, _: &[ConstId], _: &mut Meter| Ok(None);
        let new = match &inst.kind {
            InstKind::BinOp { op, a, b, .. } => {
                let op = *op;
                self.pure(&[*a, *b], cx, m, |f, c, m| match c {
                    [x, y] => f.binop(op, ty, *x, *y, m),
                    _ => Ok(None),
                })?
            }
            InstKind::ICmp { pred, a, b } => {
                let pred = *pred;
                self.pure(&[*a, *b], cx, m, |f, c, m| match c {
                    [x, y] => f.icmp(pred, ty, *x, *y, m),
                    _ => Ok(None),
                })?
            }
            InstKind::FCmp { pred, a, b, .. } => {
                let pred = *pred;
                self.pure(&[*a, *b], cx, m, |f, c, m| match c {
                    [x, y] => f.fcmp(pred, ty, *x, *y, m),
                    _ => Ok(None),
                })?
            }
            InstKind::Cast { op, v } => {
                let op = *op;
                self.pure(&[*v], cx, m, |f, c, m| match c {
                    [x] => f.cast(op, *x, ty, m),
                    _ => Ok(None),
                })?
            }
            InstKind::Freeze(v) => self.pure(&[*v], cx, m, |f, c, m| match c {
                [x] => f.freeze(*x, m),
                _ => Ok(None),
            })?,
            InstKind::ExtractElement { vec, idx } => {
                self.pure(&[*vec, *idx], cx, m, |f, c, m| match c {
                    [v, x] => f.extract_element(*v, *x, m),
                    _ => Ok(None),
                })?
            }
            InstKind::InsertElement { vec, elt, idx } => {
                self.pure(&[*vec, *elt, *idx], cx, m, |f, c, m| match c {
                    [v, e, x] => f.insert_element(ty, *v, *e, *x, m),
                    _ => Ok(None),
                })?
            }
            InstKind::ExtractValue { agg, indices } => {
                self.pure(&[*agg], cx, m, |f, c, m| match c {
                    [a] => f.extract_value(*a, indices, m),
                    _ => Ok(None),
                })?
            }
            InstKind::InsertValue { agg, val, indices } => {
                self.pure(&[*agg, *val], cx, m, |f, c, m| match c {
                    [a, v] => f.insert_value(*a, *v, indices, m),
                    _ => Ok(None),
                })?
            }
            InstKind::FNeg { v, .. } => self.pure(&[*v], cx, m, never)?,
            InstKind::ShuffleVector { a, b, mask } => {
                self.pure(&[*a, *b], cx, m, |f, c, m| match c {
                    [x, y] => f.shuffle(ty, *x, *y, mask, m),
                    _ => Ok(None),
                })?
            }
            InstKind::Gep { base, indices, .. } => {
                let mut ops = Vec::with_capacity(indices.len() + 1);
                ops.push(*base);
                ops.extend_from_slice(indices);
                self.pure(&ops, cx, m, never)?
            }
            InstKind::Select { c, t, f, .. } => match self.lat(*c) {
                Lat::Top => Lat::Top,
                Lat::Fc(i) => Lat::Fc(i),
                Lat::Const(k) => match cx.folder.int(k) {
                    Some(1) => self.lat(*t),
                    Some(0) => self.lat(*f),
                    _ => meet(self.lat(*t), self.lat(*f)),
                },
                Lat::Bottom => meet(self.lat(*t), self.lat(*f)),
            },
            #[cfg(test)]
            InstKind::Phi { incoming, .. } if self.full_meet => self.full_phi(incoming, b, m)?,
            InstKind::Phi { .. } => self.phi(i, b)?,
            InstKind::Load {
                ptr,
                volatile: false,
                atomic: None,
                ..
            } => match self.lat(*ptr) {
                Lat::Top => Lat::Top,
                Lat::Fc(i) => Lat::Fc(i),
                Lat::Bottom => Lat::Bottom,
                Lat::Const(k) => match cx.folder.consts.get(k).map(|c| &c.value) {
                    Some(Const::GlobalAddr(g)) => match cx.globals.get(*g) {
                        Some(g)
                            if g.constant
                                && !g.externally_initialized
                                && g.value_ty == ty
                                && g.init.is_some() =>
                        {
                            g.init.map_or(Lat::Bottom, Lat::Const)
                        }
                        _ => Lat::Bottom,
                    },
                    _ => Lat::Bottom,
                },
            },
            InstKind::Call(call) if is_normalize(&inst.kind) => match call.args.first() {
                Some(&x) => match self.lat(x) {
                    Lat::Top => Lat::Top,
                    Lat::Bottom => Lat::Bottom,
                    Lat::Fc(_) => cx
                        .folder
                        .normalize_predicate(ty, None, m)?
                        .map_or(Lat::Bottom, Lat::Const),
                    Lat::Const(k) => cx
                        .folder
                        .normalize_predicate(ty, Some(k), m)?
                        .map_or(Lat::Bottom, Lat::Const),
                },
                None => Lat::Bottom,
            },
            kind @ InstKind::Call(_) => match is_defined_arg(kind, cx.folder.consts, cx.fcs) {
                Some(Some(g)) => {
                    let supplied = cx.fcs.get(&g).is_some_and(|f| f.supplied);
                    let c = cx.folder.consts.intern(
                        Constant {
                            ty,
                            value: Const::Int(u64::from(supplied)),
                        },
                        m,
                    )?;
                    Lat::Const(c)
                }
                _ => Lat::Bottom,
            },
            InstKind::Br { dest, .. } => return self.flow_to((b, *dest), m, held),
            InstKind::CondBr { c, t, f, .. } => {
                match self.decide(*c, cx) {
                    Decision::Taken(x) => {
                        self.flow_to((b, if x == 0 { *f } else { *t }), m, held)?
                    }
                    Decision::All => {
                        self.flow_to((b, *t), m, held)?;
                        self.flow_to((b, *f), m, held)?;
                    }
                    Decision::Wait | Decision::Undef => {}
                }
                return Ok(());
            }
            InstKind::Switch {
                v, default, cases, ..
            } => {
                match self.decide(*v, cx) {
                    Decision::Taken(x) => {
                        let dest = cases
                            .iter()
                            .find(|&&(case, _)| case == x)
                            .map_or(*default, |&(_, d)| d);
                        self.flow_to((b, dest), m, held)?;
                    }
                    Decision::All => {
                        room(&mut self.flow, cases.len() + 1, m, held)?;
                        self.flow.push((b, *default));
                        self.flow.extend(cases.iter().map(|&(_, d)| (b, d)));
                    }
                    Decision::Wait | Decision::Undef => {}
                }
                return Ok(());
            }
            _ => Lat::Bottom,
        };
        self.set(i, new, m, held)
    }

    /// After the fixpoint: every reachable non-pure use of an unsupplied constant needs the
    /// caller's value, and a reachable branch on a literal `undef`/`poison` is refused.
    fn check(&self, cx: &Context<'_>) -> Result<(), Error> {
        let at = |b: BlockId, i: InstId| {
            Location::function(cx.name.clone(), Some(b.raw()), Some(i.raw()))
        };
        for b in self.body.blocks.ids() {
            if !self.exec.get(b.index()).copied().unwrap_or(false) {
                continue;
            }
            for (i, inst) in self.body.block_insts(b) {
                if let Some(g) = is_defined_arg(&inst.kind, cx.folder.consts, cx.fcs) {
                    if g.is_none() {
                        return Err(Error::InvalidInput(InputError::new(
                            InputErrorKind::Abi(AbiError::BadValue),
                            at(b, i),
                        )));
                    }
                    continue;
                }
                if is_pure(&inst.kind) || is_normalize(&inst.kind) {
                    continue;
                }
                let cond = match &inst.kind {
                    InstKind::CondBr { c, .. } => Some(*c),
                    InstKind::Switch { v, .. } => Some(*v),
                    _ => None,
                };
                if let Some(c) = cond {
                    if self.decide(c, cx) == Decision::Undef {
                        return Err(Error::InvalidInput(InputError::new(
                            InputErrorKind::Canon(CanonError::UndefCondition),
                            at(b, i),
                        )));
                    }
                }
                let mut needs = None;
                inst.kind.for_each_value(|v| {
                    if let Lat::Fc(index) = self.lat(v) {
                        needs.get_or_insert(index);
                    }
                });
                if let Some(index) = needs {
                    return Err(Error::NeedsInput(MissingInput::FunctionConstant { index }));
                }
            }
        }
        Ok(())
    }

    fn edit(
        &self,
        cx: &mut Context<'_>,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<BodyEdit, Error> {
        let mut edit = BodyEdit::keep_all(self.body, m, held)?;
        for (slot, &e) in edit.keep.iter_mut().zip(&self.exec) {
            *slot = e;
        }
        for (b, block) in self.body.blocks.iter() {
            if !self.exec.get(b.index()).copied().unwrap_or(false) {
                continue;
            }
            for i in block.inst_ids() {
                m.tick(1)?;
                let inst = self
                    .body
                    .insts
                    .get(i)
                    .ok_or_else(|| invariant("instruction out of range"))?;
                let lat = self.val.get(i.index()).copied().unwrap_or(Lat::Bottom);
                let deletable = is_pure(&inst.kind)
                    || is_normalize(&inst.kind)
                    || is_defined_arg(&inst.kind, cx.folder.consts, cx.fcs).is_some();
                match (&inst.kind, lat) {
                    (_, Lat::Const(c)) if deletable => edit.replace(i, Value::Const(c)),
                    (_, Lat::Fc(_)) if deletable => {
                        let undef = cx.folder.consts.intern(
                            Constant {
                                ty: inst.ty,
                                value: Const::Undef,
                            },
                            m,
                        )?;
                        edit.replace(i, Value::Const(undef));
                    }
                    (InstKind::Select { c, t, f, .. }, _) => {
                        if let Lat::Const(k) = self.lat(*c) {
                            match cx.folder.int(k) {
                                Some(1) => edit.replace(i, *t),
                                Some(0) => edit.replace(i, *f),
                                _ => {}
                            }
                        }
                    }
                    (InstKind::CondBr { c, t, f, hints }, _) => {
                        if let Decision::Taken(x) = self.decide(*c, cx) {
                            let dest = if x == 0 { *f } else { *t };
                            edit.rewrite.insert(
                                i,
                                InstKind::Br {
                                    dest,
                                    hints: *hints,
                                },
                            );
                        }
                    }
                    (
                        InstKind::Switch {
                            v,
                            default,
                            cases,
                            hints,
                        },
                        _,
                    ) => {
                        if let Decision::Taken(x) = self.decide(*v, cx) {
                            let dest = cases
                                .iter()
                                .find(|&&(case, _)| case == x)
                                .map_or(*default, |&(_, d)| d);
                            edit.rewrite.insert(
                                i,
                                InstKind::Br {
                                    dest,
                                    hints: *hints,
                                },
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(edit)
    }
}

/// What the entry block alone decides about the lattice value of one of its values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Taint {
    /// Not [`Lat::Fc`].
    Clean,
    /// [`Lat::Fc`] of this constant.
    Fc(u32),
    /// Not decided without folding (a select whose arms differ).
    Unknown,
}

impl Taint {
    /// A pure operation's taint from its operands' ([`Solver::pure`]): the lowest constant among
    /// the tainted operands, which an undecided operand could lower.
    fn of_pure(ops: impl Iterator<Item = Taint>) -> Taint {
        let mut out = Taint::Clean;
        for t in ops {
            out = match (out, t) {
                (Taint::Unknown, _) | (_, Taint::Unknown) => return Taint::Unknown,
                (Taint::Fc(i), Taint::Fc(j)) => Taint::Fc(i.min(j)),
                (Taint::Fc(i), Taint::Clean) | (Taint::Clean, Taint::Fc(i)) => Taint::Fc(i),
                (Taint::Clean, Taint::Clean) => Taint::Clean,
            };
        }
        out
    }

    /// A select's ([`Solver::visit`]): its condition's taint, else its arms' when they agree
    /// (the value a folded condition picks, or their meet).
    fn of_select(c: Taint, t: Taint, f: Taint) -> Taint {
        match c {
            Taint::Clean if t == f => t,
            Taint::Fc(_) => c,
            Taint::Clean | Taint::Unknown => Taint::Unknown,
        }
    }
}

/// The unsupplied function constant whose value SCCP over the inlined entry is certain to refuse
/// first, when the entry's own entry block decides it: `Some(index)` exactly when [`Solver::check`]
/// would return `NeedsInput(FunctionConstant{index})` at an instruction of `body`'s entry block
/// before its first call that `inlined` says the inliner replaces.
///
/// Why the entry's own body decides it. The entry block has no predecessors and always executes.
/// The inliner keeps its instructions up to that first inlined call, in order, as the start of
/// the inlined body's first block, which `check` scans first; erasure only deletes no-op pointer
/// casts, which are pure, and keeps every constant's taint. Those instructions' operands are
/// arguments, constants, and each other, so their lattice values depend on no callee and on no
/// other block. Their taint needs no folding (the constant table is only read): a pure result is
/// tainted by its operands, a select by its condition or else by both arms alike, a normalized
/// predicate never. The caller runs this once the inliner's own refusals are decided; what
/// remains before `check` (building the inlined body, erasure, solving) refuses only budget and
/// invariant failures.
///
/// Any instruction whose outcome in `check` the block does not decide ends the scan with `None`,
/// and SCCP decides: a `BadValue` or `undef`-condition refusal, an operand whose taint is not
/// decided, or the first inlined call.
pub(super) fn entry_block_refusal(
    body: &Body,
    consts: &ConstTable,
    fcs: &HashMap<GlobalId, FcState>,
    inlined: &dyn Fn(&InstKind) -> bool,
    m: &mut Meter,
) -> Result<Option<u32>, Error> {
    scoped(m, |m, held| {
        let block = body
            .blocks
            .get(BlockId::new(0))
            .copied()
            .ok_or_else(|| invariant("a body without blocks"))?;
        // Per instruction of the block, in order: its taint once visited.
        let mut seen = filled(m, held, block.len(), Taint::Unknown)?;
        let mut memo = HashMap::new();
        let mut ops = Vec::new();
        let mut taints = Vec::new();
        for (at, i) in block.inst_ids().enumerate() {
            m.tick(1)?;
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            if inlined(&inst.kind) {
                return Ok(None);
            }
            ops.clear();
            inst.kind.for_each_value(|v| ops.push(v));
            taints.clear();
            for &v in &ops {
                taints.push(match v {
                    Value::Arg(_) => Taint::Clean,
                    Value::Inst(j) => j
                        .index()
                        .checked_sub(block.start.index())
                        .and_then(|k| seen.get(k).copied())
                        .unwrap_or(Taint::Unknown),
                    Value::Const(c) => match taint_of(c, consts, fcs, &mut memo, m, held)? {
                        Some(index) => Taint::Fc(index),
                        None => Taint::Clean,
                    },
                });
            }
            let taint = match is_defined_arg(&inst.kind, consts, fcs) {
                // `check` refuses the argument as `BadValue` first.
                Some(None) => return Ok(None),
                Some(Some(_)) => Taint::Clean,
                None => match (&inst.kind, taints.as_slice()) {
                    (InstKind::Select { .. }, &[c, t, f]) => Taint::of_select(c, t, f),
                    (InstKind::Phi { .. }, _) => Taint::Unknown,
                    (kind, _) if is_normalize(kind) => Taint::Clean,
                    (kind, _) if is_pure(kind) => Taint::of_pure(taints.iter().copied()),
                    // Any other use: `check` refuses its first tainted operand. A tainted branch
                    // condition waits, so it is not an `undef` condition first.
                    _ => match taints.iter().find(|&&t| t != Taint::Clean) {
                        Some(&Taint::Fc(index)) => return Ok(Some(index)),
                        Some(_) => return Ok(None),
                        None => Taint::Clean,
                    },
                },
            };
            if let Some(slot) = seen.get_mut(at) {
                *slot = taint;
            }
        }
        Ok(None)
    })
}

/// [`const_taint`] of constant `c`, its operands first. An explicit stack, since constants nest
/// deeply; `memo` keeps each constant's once computed.
fn taint_of(
    c: ConstId,
    consts: &ConstTable,
    fcs: &HashMap<GlobalId, FcState>,
    memo: &mut HashMap<ConstId, Option<u32>>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Option<u32>, Error> {
    let mut stack = vec![c];
    while let Some(&k) = stack.last() {
        if memo.contains_key(&k) {
            stack.pop();
            continue;
        }
        m.tick(1)?;
        let value = &consts
            .get(k)
            .ok_or_else(|| invariant("constant out of range"))?
            .value;
        let pending = stack.len();
        for &ch in value.children() {
            if !memo.contains_key(&ch) {
                let bytes = std::mem::size_of::<ConstId>();
                m.charge(bytes)?;
                *held = held.saturating_add(bytes);
                stack.push(ch);
            }
        }
        if stack.len() == pending {
            let t = const_taint(value, fcs, |ch| memo.get(&ch).copied().flatten());
            let bytes = std::mem::size_of::<(ConstId, Option<u32>)>() * 2;
            m.charge(bytes)?;
            *held = held.saturating_add(bytes);
            memo.insert(k, t);
            stack.pop();
        }
    }
    Ok(memo.get(&c).copied().flatten())
}

#[cfg(test)]
#[path = "sccp_tests.rs"]
mod tests;
