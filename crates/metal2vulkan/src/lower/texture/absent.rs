//! A nil texture's arm: the code the default presence specialization never runs.
//!
//! `is_null` of a static texture slot is the negation of the slot's `ResourcePresence`
//! specialization constant ([`TextureStrategy::IsNull`]), whose default is "present"
//! ([`super::declare`]): the arm a shader keeps for a nil texture is code the default
//! specialization never runs, and the driver can fold it away when it folds the constant at
//! pipeline creation. That arm is still walked where a kernel's emulated simdgroups are predicated:
//! a texture call's result is divergent there (`simd::uniform`), so a nil arm holding an emulated
//! group operation or a barrier is walked by every invocation with its `live` flag off, and its
//! synchronizations run at the default presence too. Either way its words are words only the arm
//! adds. The old engine folded `is_null` to `false` and dropped that arm, which is wrong for a nil
//! texture, a binding Metal allows; the engine keeps both arms, so its module is larger by that
//! code. [`absent_arm`] names the code structurally, for the strategy census alone (`Plan::census`
//! counts its words under [`ABSENT_ARM`]); nothing emitted reads it.
//!
//! **The arm.** With every presence constant at its default, each `is_null` of a static slot is
//! `false`, and so is every `i1` the pure boolean operations of [`decide`] compute from such
//! results and constants. A block belongs to the arm when the body reaches it but the entry does
//! not reach it along the edges those decided conditions select: the default (all-present)
//! specialization never runs it. That is each nil texture's arm, and more only by dead code: a
//! condition built from presences without depending on them (`and %n, false`, `xor %n, %n`, an
//! `icmp ne` of two `is_null`s of one slot) is decided too, so a block behind it that no binding
//! runs is counted as well. Nothing else is decided, so a condition read through a phi, a heap
//! handle's `is_null` (a value of the argument buffer, not a constant), and a multi-candidate
//! `is_null` switch (whose null arm is the null handle, not a presence) keep both of their edges:
//! the arm never holds code the default specialization runs, and it may omit some it does not.

use crate::air::ir::{
    BinOp, BlockId, Body, Const, InstId, InstKind, IntPredicate, Type, TypeId, Value, ValueId,
    ValueIndex,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::strategy::Strategy;

use super::TextureStrategy;

/// The census name of the code the default presence specialization never runs, a nil texture's
/// arm.
pub const ABSENT_ARM: &str = "texture.is_null.absent_arm";

/// A condition's value under the default presence specialization, when that value depends on a
/// presence; `None` when it is not decided, or decided without one.
type Decided = Option<bool>;

/// Whether `s` reads a static slot's presence constant (`is_null` of one slot).
fn reads_presence(s: &Option<Strategy>) -> bool {
    matches!(s, Some(Strategy::Texture(TextureStrategy::IsNull { .. })))
}

/// The instructions with a strategy (`strategy`, by value id) that the default presence
/// specialization never runs, sorted by value id: those in blocks the body reaches but that
/// specialization does not (module docs). Empty, without a walk, when no strategy reads a presence. The result
/// stays charged to `m`; the plan keeps it.
pub(crate) fn absent_arm(
    e: &Entry,
    body: &Body,
    strategy: &[Option<Strategy>],
    values: ValueIndex,
    m: &mut Meter,
) -> Result<Vec<ValueId>, Error> {
    if !strategy.iter().any(reads_presence) {
        return Ok(Vec::new());
    }
    let blocks = body.blocks.len();
    // The decided conditions (one byte each), the blocks the default reaches, and the walk's
    // stack (each block is pushed at most once).
    let bytes = body
        .insts
        .len()
        .saturating_add(blocks)
        .saturating_add(blocks.saturating_mul(std::mem::size_of::<BlockId>()));
    m.charge(bytes)?;
    let arm = walk(e, body, strategy, values, m);
    m.release(bytes);
    arm
}

fn walk(
    e: &Entry,
    body: &Body,
    strategy: &[Option<Strategy>],
    values: ValueIndex,
    m: &mut Meter,
) -> Result<Vec<ValueId>, Error> {
    let strategy_of = |id: InstId| {
        values
            .inst(id)
            .and_then(|v| strategy.get(v.index()).map(|s| (v, s)))
    };
    // Every operand is defined in a block that dominates its use, so a dominator-first order (the
    // reverse post-order) decides each condition after its operands; phis decide nothing.
    let mut decided: Vec<Decided> = vec![None; body.insts.len()];
    for b in e.cfg.rpo() {
        for (id, inst) in body.block_insts(b) {
            m.tick(1)?;
            let d = match strategy_of(id) {
                Some((_, s)) if reads_presence(s) => Some(false),
                _ => decide(e, &inst.kind, inst.ty, &decided),
            };
            if let Some(slot) = decided.get_mut(id.raw() as usize) {
                *slot = d;
            }
        }
    }
    let cond = |v: Value| match v {
        Value::Inst(i) => decided.get(i.raw() as usize).copied().flatten(),
        _ => None,
    };
    // The blocks the default specialization reaches: a decided branch takes the edge its
    // condition selects, every other terminator all of its edges.
    let mut live = vec![false; body.blocks.len()];
    let entry = e.cfg.entry();
    let mut stack = vec![entry];
    if let Some(l) = live.get_mut(entry.raw() as usize) {
        *l = true;
    }
    while let Some(b) = stack.pop() {
        m.tick(1)?;
        let Some(term) = body.terminator(b) else {
            continue;
        };
        let mut reach = |to: BlockId| {
            if let Some(l) = live.get_mut(to.raw() as usize) {
                if !*l {
                    *l = true;
                    stack.push(to);
                }
            }
        };
        match &term.kind {
            InstKind::CondBr { c, t, f, .. } => match cond(*c) {
                Some(true) => reach(*t),
                Some(false) => reach(*f),
                None => {
                    reach(*t);
                    reach(*f);
                }
            },
            // Case literals are zero-extended bit patterns of the selector.
            InstKind::Switch {
                v, default, cases, ..
            } => match cond(*v) {
                Some(bit) => reach(
                    cases
                        .iter()
                        .find(|(literal, _)| *literal == u64::from(bit))
                        .map_or(*default, |(_, to)| *to),
                ),
                None => term.kind.successors().for_each(&mut reach),
            },
            kind => kind.successors().for_each(&mut reach),
        }
    }
    // The arm: the strategies of the reachable blocks the default does not reach.
    let in_arm = |b: BlockId| !live.get(b.raw() as usize).copied().unwrap_or(true);
    let mut count = 0usize;
    for b in e.cfg.rpo().filter(|b| in_arm(*b)) {
        for (id, _) in body.block_insts(b) {
            m.tick(1)?;
            if strategy_of(id).is_some_and(|(_, s)| s.is_some()) {
                count += 1;
            }
        }
    }
    let mut arm = Vec::new();
    m.reserve(&mut arm, count)?;
    for b in e.cfg.rpo().filter(|b| in_arm(*b)) {
        for (id, _) in body.block_insts(b) {
            if let Some((v, Some(_))) = strategy_of(id) {
                arm.push(v);
            }
        }
    }
    arm.sort_unstable();
    Ok(arm)
}

/// The value of a pure boolean instruction under the default presence specialization, when it
/// depends on a decided condition (`decided`, by instruction): `and`, `or`, `xor`, `icmp eq`/`ne`
/// of `i1`s, `select`, and `freeze`, each folded as the device computes it from operands that are
/// decided conditions or `i1` constants (an `and` with a decided `false`, or an `or` with a
/// decided `true`, needs no other operand). Anything else is undecided.
fn decide(e: &Entry, kind: &InstKind, ty: TypeId, decided: &[Decided]) -> Decided {
    let is_bool = |t: TypeId| matches!(e.module.types().get(t), Some(Type::Int(1)));
    if !is_bool(ty) {
        return None;
    }
    // An operand's value, and whether a presence decides it.
    let known = |v: Value| -> Option<(bool, bool)> {
        match v {
            Value::Inst(i) => decided
                .get(i.raw() as usize)
                .copied()
                .flatten()
                .map(|b| (b, true)),
            Value::Const(c) => {
                let k = e.module.consts().get(c).filter(|k| is_bool(k.ty))?;
                match k.value {
                    Const::Int(b) => Some((b & 1 == 1, false)),
                    Const::Null => Some((false, false)),
                    _ => None,
                }
            }
            Value::Arg(_) => None,
        }
    };
    // Both operands known, and at least one decided by a presence.
    let both = |a: Value, b: Value, f: fn(bool, bool) -> bool| match (known(a), known(b)) {
        (Some((x, px)), Some((y, py))) if px || py => Some(f(x, y)),
        _ => None,
    };
    let decided_as =
        |a: Value, b: Value, x: bool| [a, b].iter().any(|v| known(*v) == Some((x, true)));
    match kind {
        InstKind::BinOp {
            op: BinOp::And,
            a,
            b,
            ..
        } => {
            if decided_as(*a, *b, false) {
                Some(false)
            } else {
                both(*a, *b, |x, y| x & y)
            }
        }
        InstKind::BinOp {
            op: BinOp::Or,
            a,
            b,
            ..
        } => {
            if decided_as(*a, *b, true) {
                Some(true)
            } else {
                both(*a, *b, |x, y| x | y)
            }
        }
        InstKind::BinOp {
            op: BinOp::Xor,
            a,
            b,
            ..
        } => both(*a, *b, |x, y| x ^ y),
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a,
            b,
        } => both(*a, *b, |x, y| x == y),
        InstKind::ICmp {
            pred: IntPredicate::Ne,
            a,
            b,
        } => both(*a, *b, |x, y| x != y),
        // A known condition selects one operand, whose value the result is when that operand is
        // known; with an unknown condition, two known operands decide it only when they agree.
        InstKind::Select { c, t, f, .. } => match known(*c) {
            Some((x, pc)) => known(if x { *t } else { *f })
                .filter(|(_, p)| pc || *p)
                .map(|(y, _)| y),
            None => match (known(*t), known(*f)) {
                (Some((x, px)), Some((y, py))) if x == y && (px || py) => Some(x),
                _ => None,
            },
        },
        InstKind::Freeze(v) => known(*v).filter(|(_, p)| *p).map(|(x, _)| x),
        _ => None,
    }
}
