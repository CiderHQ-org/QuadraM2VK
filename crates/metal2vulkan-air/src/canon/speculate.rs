//! A branch's short arms, which Metal's backend speculates.
//!
//! Metal's backend runs a branch's arms of plain computation in every lane that reaches the
//! branch, and picks each result the arms join with by the branch's condition (if-conversion).
//! A group operation in such an arm then runs in every lane of the branch's block: Metal's
//! `quad_broadcast_first` in `if (t % 3 != 0) r = quad_broadcast_first(10 t + 1);` answers with
//! the quad's lane 0 whether it takes the arm or not, where the AIR's branch would give the quad's
//! first lane that takes it.
//!
//! **What Metal does** (the Apple M3 Max, macOS 27.0.1, runtime-compiled MSL 64-thread kernels
//! whose arms the condition `t % 3 != 0` takes, read in their AIR (`xcrun metal -S -emit-llvm`)):
//! - **The operations.** An arm holding a quad shuffle, broadcast, vote (`all`, `any`,
//!   `ballot`, `is_first`, `active_threads_mask`) or `quad_broadcast_first`, a simdgroup shuffle
//!   (not `simd_broadcast`) or `simd_is_first` is speculated: each answers over the lanes of the
//!   branch's block. One holding a quad reduction or scan, `simd_broadcast`,
//!   `simd_broadcast_first`, a simdgroup reduction, scan or vote other than `is_first` is not, and
//!   the arm's other operations then keep the arm's lanes too (all 52 measured, each beside a
//!   `quad_broadcast_first`).
//! - **The rest of the arm.** Integer and float arithmetic (a division too), conversions,
//!   `freeze`, `select`, `sqrt`, and a load through a `constant` reference parameter are
//!   speculated; a device, threadgroup or texture read, an atomic, a store, a loop, or a second
//!   exit are not. A store that repeats the branch's block's last store to the same pointer is
//!   speculated as a store of the chosen value (`out[t] = 7777; if (c) out[t] = ...;`); a store
//!   to another pointer is not.
//! - **The size.** One arm (a triangle): speculated up to 25 instructions (a sunk store and the
//!   branch not counted), not from 26, for arms of 32-bit integer arithmetic (chains with and
//!   without multiplications, divisions, one or two results), comparisons and selects of values,
//!   `freeze`s and up to eight quad broadcasts. An arm holding another kind is not known to follow
//!   that budget: one of quad votes, 16-bit masks, casts and selects of constants is speculated at
//!   21 instructions and not at 25 (`r = quad_is_first() | quad_active_threads_mask() << 1 |
//!   quad_all(...) << 5 | ...;`, with its `%` replaced by a mask too). Two arms (a diamond):
//!   speculated up to 25 in both together, not from 26; and where the arm the condition's true
//!   edge takes holds at most 3 instructions and no call, whenever the other arm holds at most 25
//!   (up to 28 in both measured). An arm of at most 3 instructions with a call on the false edge
//!   is not exempt (1 + 25 and 3 + 25 not speculated).
//! - **Nesting.** A short branch inside an arm is speculated first and counts with the arm it
//!   joins (`if (c) { if (d) x = ...; r = op(x); }`); a branch inside a loop is speculated alike.
//!
//! [`expand`] speculates each outermost branch Metal speculates whose arms hold a group operation
//! that answers differently there (the operations above): the branch becomes a jump into its arm
//! (a diamond's arms run one after the other), each value the arms join with becomes a `select`
//! on the condition, and a sunk store stores the chosen value where the arms join. A branch whose
//! arms hold no such operation is left as it is, as is one whose arms hold an instruction no
//! measurement covers (a vector or aggregate operation, an address computation, a math function
//! other than `sqrt`, a call outside the families above, a load other than a `constant`
//! reference's), that join with more than two values or with a pointer, that divide integers by a
//! value (speculated, a division by zero where the condition guarded it is undefined in SPIR-V),
//! that hold 22 to 25 instructions with one of a kind the budget was not measured with
//! ([`OFF_EDGE`]), or of one of the two diamond shapes no probe covers (a small arm without a call
//! on the false edge, or with one on the true edge, past 25 together): those keep the AIR's lanes,
//! where Metal may speculate them. An entry holding such a branch whose arms hold a group
//! operation of the speculated families records it ([`super::Entry::unmeasured_speculation`]), and
//! its pipeline the precision note of an unmeasured speculation.

use std::collections::BTreeMap;

use crate::budget::Meter;
use crate::error::Error;
use crate::graph::Cfg;
use crate::intrinsics::{AirIntrinsic, GroupOp, GroupScope, Math, MathFn, TexOp};
use crate::ir::{AddrSpace, BinOp, BlockId, Body, Callee, FastMath, Idx, InstId, InstKind};
use crate::ir::{ConstTable, TypeId, TypeTable, Value};
use crate::ir::{Param, Type};

use super::rewrite::{apply, scoped, BodyEdit};

/// The most instructions Metal's backend speculates in a branch's arms, measured with arms of
/// 32-bit integer arithmetic, comparisons and selects of values, `freeze`s and quad broadcasts.
const BUDGET: usize = 25;

/// The most instructions Metal was measured to speculate in arms holding another kind (a vote, a
/// cast, a 16-bit operation, a select of a constant): 21 speculated, 25 of the same kinds not. Such
/// an arm of 22 to [`BUDGET`] instructions is unknown.
const OFF_EDGE: usize = 21;

/// The most instructions of a diamond's arm on the condition's true edge, without a call, that
/// Metal speculates beside a whole budget in the other arm.
const SMALL: usize = 3;

/// The instructions before a branch searched for the store a sunk store repeats: LLVM's
/// SimplifyCFG window (the measured store lies three back).
const LOOKBACK: usize = 9;

/// The deepest nesting of branches inside an arm the walk follows.
const DEPTH: usize = 8;

/// The most blocks one arm's walk takes.
const ARM_BLOCKS: usize = 64;

/// What Metal does with a branch's arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Speculated: the arms run in every lane of the branch's block.
    Speculated,
    /// Kept: a measured reason keeps the branch.
    Kept,
    /// No measurement covers the arms.
    Unknown,
}

/// What an arm holds.
#[derive(Clone, Copy, Debug, Default)]
struct Content {
    /// The instructions Metal counts.
    cost: usize,
    /// A group operation that answers over the branch's block's lanes when speculated.
    group: bool,
    /// Something Metal never speculates.
    blocker: bool,
    /// Something no measurement covers.
    unmeasured: bool,
    /// A call.
    calls: bool,
    /// An instruction of a kind the budget was not measured with ([`OFF_EDGE`]).
    off_edge: bool,
}

impl Content {
    fn add(&mut self, o: Content) {
        self.cost = self.cost.saturating_add(o.cost);
        self.group |= o.group;
        self.blocker |= o.blocker;
        self.unmeasured |= o.unmeasured;
        self.calls |= o.calls;
        self.off_edge |= o.off_edge;
    }
}

/// One arm: the blocks from `first` to `last`, entered from the branch's block alone.
#[derive(Clone, Debug)]
struct Arm {
    first: BlockId,
    last: BlockId,
    /// The arm the condition's true edge takes.
    on_true: bool,
    blocks: Vec<BlockId>,
    content: Content,
    /// The branches nested in the arm.
    inner: Vec<Fold>,
}

/// A branch whose arms rejoin at `join`: one arm (a triangle) or two (a diamond).
#[derive(Clone, Debug)]
struct Fold {
    head: BlockId,
    cond: Value,
    arms: Vec<Arm>,
    join: BlockId,
    /// A sunk store: the branch's block's store and the arm's store to the same pointer.
    sunk: Option<(InstId, InstId)>,
    content: Content,
    decision: Decision,
}

/// What the walk reads.
struct Cx<'a> {
    body: &'a Body,
    cfg: &'a Cfg<BlockId>,
    types: &'a TypeTable,
    consts: &'a ConstTable,
    /// The entry's parameter types and attributes.
    param_tys: &'a [TypeId],
    params: &'a [Param],
}

impl Cx<'_> {
    fn preds(&self, b: BlockId) -> Vec<BlockId> {
        self.cfg.predecessors(b).collect()
    }

    fn value_ty(&self, v: Value) -> Option<TypeId> {
        value_ty(self.body, self.consts, self.param_tys, v)
    }

    fn is_pointer(&self, v: Value) -> bool {
        self.value_ty(v)
            .and_then(|t| self.types.get(t))
            .is_some_and(|t| matches!(t, Type::Ptr { .. }))
    }

    fn space(&self, v: Value) -> Option<AddrSpace> {
        match self.types.get(self.value_ty(v)?)? {
            Type::Ptr { space } => Some(*space),
            _ => None,
        }
    }

    /// Whether `v` is a 32-bit integer.
    fn word(&self, v: Value) -> bool {
        self.value_ty(v)
            .and_then(|t| self.types.get(t))
            .is_some_and(|t| *t == Type::Int(32))
    }

    /// Whether an instruction of kind `kind` is of a kind the budget was measured with: 32-bit
    /// integer arithmetic, a comparison of 32-bit integers, a select of two values that are not
    /// constants, a `freeze`, or a broadcast or shuffle of a group (`BUDGET`).
    fn on_edge(&self, kind: &InstKind) -> bool {
        match kind {
            InstKind::BinOp { a, b, .. } => self.word(*a) && self.word(*b),
            InstKind::ICmp { a, b, .. } => self.word(*a) && self.word(*b),
            InstKind::Select { t, f, .. } => {
                !matches!(t, Value::Const(_)) && !matches!(f, Value::Const(_)) && self.word(*t)
            }
            InstKind::Freeze(_) => true,
            InstKind::Call(c) => match &c.callee {
                Callee::Air {
                    op: AirIntrinsic::Group(g),
                    ..
                } => !matches!(
                    g.op,
                    GroupOp::IsFirst
                        | GroupOp::All
                        | GroupOp::Any
                        | GroupOp::Ballot
                        | GroupOp::ActiveThreadsMask
                ),
                _ => false,
            },
            _ => false,
        }
    }

    /// The content of instruction `kind` in an arm (a store is the caller's).
    fn class(&self, kind: &InstKind) -> Content {
        let c = self.cost_class(kind);
        Content {
            off_edge: !self.on_edge(kind),
            ..c
        }
    }

    /// [`Cx::class`] but whether its kind is on the budget's edge.
    fn cost_class(&self, kind: &InstKind) -> Content {
        let plain = Content {
            cost: 1,
            ..Content::default()
        };
        let blocker = Content {
            blocker: true,
            ..plain
        };
        let unmeasured = Content {
            unmeasured: true,
            ..plain
        };
        match kind {
            // An integer division by a value the arm's condition may guard: speculated, it would
            // divide by zero in a lane that takes no arm, which SPIR-V leaves undefined, where
            // Metal's division answers (a division was measured speculated).
            InstKind::BinOp {
                op: BinOp::UDiv | BinOp::SDiv | BinOp::URem | BinOp::SRem,
                b,
                ..
            } if !matches!(b, Value::Const(_)) => unmeasured,
            InstKind::BinOp { .. }
            | InstKind::FNeg { .. }
            | InstKind::Cast { .. }
            | InstKind::ICmp { .. }
            | InstKind::FCmp { .. }
            | InstKind::Select { .. }
            | InstKind::Freeze(_) => plain,
            InstKind::Load {
                ptr,
                volatile: false,
                atomic: None,
                ..
            } => match (*ptr, self.space(*ptr)) {
                (Value::Arg(n), Some(AddrSpace::Constant))
                    if self
                        .params
                        .get(n.index())
                        .is_some_and(|p| p.attrs.dereferenceable.is_some()) =>
                {
                    plain
                }
                (_, Some(AddrSpace::Device | AddrSpace::Threadgroup)) => blocker,
                _ => unmeasured,
            },
            InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::AtomicRmw { .. }
            | InstKind::CmpXchg { .. }
            | InstKind::Fence(_) => blocker,
            InstKind::Call(c) => {
                let call = Content {
                    calls: true,
                    ..plain
                };
                match &c.callee {
                    Callee::Air { op, .. } => match op {
                        AirIntrinsic::Group(g) => match speculated(g.scope, g.op) {
                            true => Content {
                                group: true,
                                ..call
                            },
                            false => Content {
                                blocker: true,
                                ..call
                            },
                        },
                        // The measured calls: `sqrt` and a conversion.
                        AirIntrinsic::Math(Math {
                            f: MathFn::Sqrt, ..
                        })
                        | AirIntrinsic::Convert(_)
                            if !c.args.iter().any(|&a| self.is_pointer(a)) =>
                        {
                            call
                        }
                        AirIntrinsic::Atomic(_) => Content {
                            blocker: true,
                            ..call
                        },
                        AirIntrinsic::Texture(t) if t.op == TexOp::Read => Content {
                            blocker: true,
                            ..call
                        },
                        _ => Content {
                            unmeasured: true,
                            ..call
                        },
                    },
                    _ => Content {
                        unmeasured: true,
                        ..call
                    },
                }
            }
            _ => unmeasured,
        }
    }

    /// The content of block `b`'s instructions but its terminator, its PHIs counted as the
    /// selects they become where `joined` (a nested branch's join).
    fn block(&self, b: BlockId, joined: bool) -> Content {
        let mut c = Content::default();
        for (_, i) in self.body.block_insts(b) {
            match &i.kind {
                k if k.is_terminator() => {}
                InstKind::Phi { .. } if joined => c.cost = c.cost.saturating_add(1),
                InstKind::Phi { .. } => c.unmeasured = true,
                k => c.add(self.class(k)),
            }
        }
        c
    }

    /// Walk an arm from `first`, entered from `from` alone: the blocks it runs through, nested
    /// branches folded, to the first block it reaches that more than its last block enters.
    /// `None` where it is no arm (a second entry, a loop, a return or switch).
    fn walk(
        &self,
        first: BlockId,
        from: BlockId,
        depth: usize,
        m: &mut Meter,
    ) -> Result<Option<(Arm, BlockId)>, Error> {
        if self.preds(first) != [from] {
            return Ok(None);
        }
        let mut arm = Arm {
            first,
            last: first,
            on_true: false,
            blocks: Vec::new(),
            content: Content::default(),
            inner: Vec::new(),
        };
        let mut b = first;
        let mut joined = false;
        loop {
            m.tick(1)?;
            if arm.blocks.len() >= ARM_BLOCKS {
                return Ok(None);
            }
            arm.blocks.push(b);
            arm.content.add(self.block(b, joined));
            arm.last = b;
            match self.body.terminator(b).map(|t| &t.kind) {
                Some(InstKind::Br { dest, .. }) => {
                    if self.preds(*dest) == [b] && *dest != first {
                        b = *dest;
                        joined = false;
                        continue;
                    }
                    return Ok(Some((arm, *dest)));
                }
                Some(InstKind::CondBr { .. }) if depth < DEPTH => {
                    let Some(inner) = self.fold(b, depth + 1, m)? else {
                        return Ok(None);
                    };
                    for a in &inner.arms {
                        arm.blocks.extend(a.blocks.iter().copied());
                        arm.content.add(a.content);
                    }
                    let join = inner.join;
                    if arm.blocks.contains(&join) {
                        return Ok(None);
                    }
                    arm.inner.push(inner);
                    b = join;
                    joined = true;
                }
                _ => return Ok(None),
            }
        }
    }

    /// The branch ending block `head` as Metal's backend folds it, when its arms rejoin.
    fn fold(&self, head: BlockId, depth: usize, m: &mut Meter) -> Result<Option<Fold>, Error> {
        let Some(InstKind::CondBr { c, t, f, .. }) = self.body.terminator(head).map(|i| &i.kind)
        else {
            return Ok(None);
        };
        let (c, t, f) = (*c, *t, *f);
        if t == f || t == head || f == head {
            return Ok(None);
        }
        let wt = self.walk(t, head, depth, m)?;
        let wf = self.walk(f, head, depth, m)?;
        let joins = |j: BlockId, a: BlockId, b: BlockId| {
            let mut p = self.preds(j);
            p.sort();
            let mut want = vec![a, b];
            want.sort();
            p == want
        };
        let (arms, join) = match (wt, wf) {
            (Some((mut a, exit)), _) if exit == f && joins(f, head, a.last) => {
                a.on_true = true;
                (vec![a], f)
            }
            (_, Some((a, exit))) if exit == t && joins(t, head, a.last) => (vec![a], t),
            (Some((mut a, ea)), Some((b, eb))) if ea == eb && joins(ea, a.last, b.last) => {
                a.on_true = true;
                (vec![a, b], ea)
            }
            _ => return Ok(None),
        };
        let mut content = Content::default();
        let mut sunk = None;
        for a in &arms {
            content.add(a.content);
        }
        // The values the arms join with: one or two measured, each a `select` the lowering takes
        // (a pointer's is not one).
        let joined: Vec<TypeId> = self
            .body
            .block_insts(join)
            .filter(|(_, i)| matches!(i.kind, InstKind::Phi { .. }))
            .map(|(_, i)| i.ty)
            .collect();
        content.unmeasured |= joined.len() > 2
            || joined
                .iter()
                .any(|&t| matches!(self.types.get(t), Some(Type::Ptr { .. })));
        // A store in the arms: speculated only as one arm's single store repeating the branch's
        // block's last store to its pointer.
        let stores: Vec<(InstId, BlockId)> = arms
            .iter()
            .flat_map(|a| a.blocks.iter().copied())
            .flat_map(|b| {
                self.body
                    .block_insts(b)
                    .filter(|(_, i)| matches!(i.kind, InstKind::Store { .. }))
                    .map(move |(id, _)| (id, b))
            })
            .collect();
        match (&arms[..], &stores[..]) {
            (_, []) => {}
            ([a], [(s, b)]) if a.blocks.len() == 1 && *b == a.first => {
                match self.repeated(head, *s) {
                    // Not counted, and no blocker. The branch's block's store moves to where the
                    // arms join: a read after it in that block would read past it (unmeasured).
                    Some(h) => {
                        sunk = Some((h, *s));
                        content.blocker = self.blocked_but(a.first, *s);
                        content.cost = content.cost.saturating_sub(1);
                        content.unmeasured |= self.read_after(head, h);
                    }
                    None => content.blocker = true,
                }
            }
            _ => content.blocker = true,
        }
        let decision = decide(&arms, content);
        Ok(Some(Fold {
            head,
            cond: c,
            arms,
            join,
            sunk,
            content,
            decision,
        }))
    }

    /// Whether block `b` holds a blocker other than its store `store`.
    fn blocked_but(&self, b: BlockId, store: InstId) -> bool {
        self.body
            .block_insts(b)
            .filter(|&(id, i)| id != store && !i.kind.is_terminator())
            .any(|(_, i)| self.class(&i.kind).blocker)
    }

    /// Whether an instruction of block `head` after `store` but its terminator reads memory.
    fn read_after(&self, head: BlockId, store: InstId) -> bool {
        self.body
            .block_insts(head)
            .skip_while(|&(id, _)| id != store)
            .skip(1)
            .filter(|(_, i)| !i.kind.is_terminator())
            .any(|(_, i)| match &i.kind {
                InstKind::Load { .. } => true,
                InstKind::Call(_) => self.class(&i.kind).unmeasured || self.class(&i.kind).blocker,
                _ => false,
            })
    }

    /// The store of block `head` that arm store `store` repeats: the last instruction before the
    /// branch that writes memory, within [`LOOKBACK`], a plain store of the same type to the same
    /// pointer.
    fn repeated(&self, head: BlockId, store: InstId) -> Option<InstId> {
        let InstKind::Store {
            val,
            ptr,
            volatile: false,
            atomic: None,
            ..
        } = self.body.insts.get(store)?.kind
        else {
            return None;
        };
        let ty = self.value_ty(val);
        let insts: Vec<(InstId, &InstKind)> = self
            .body
            .block_insts(head)
            .map(|(id, i)| (id, &i.kind))
            .collect();
        for &(id, kind) in insts.iter().rev().take(LOOKBACK) {
            match kind {
                InstKind::Store {
                    val: v,
                    ptr: p,
                    volatile: false,
                    atomic: None,
                    ..
                } => return (*p == ptr && self.value_ty(*v) == ty).then_some(id),
                k if writes(k) => return None,
                _ => {}
            }
        }
        None
    }
}

/// The type of value `v` of `body`, of the entry of parameter types `param_tys`.
fn value_ty(body: &Body, consts: &ConstTable, param_tys: &[TypeId], v: Value) -> Option<TypeId> {
    match v {
        Value::Inst(i) => body.insts.get(i).map(|n| n.ty),
        Value::Arg(n) => param_tys.get(n.index()).copied(),
        Value::Const(k) => consts.ty(k),
    }
}

/// Whether `kind` may write memory.
fn writes(kind: &InstKind) -> bool {
    match kind {
        InstKind::Store { .. }
        | InstKind::AtomicRmw { .. }
        | InstKind::CmpXchg { .. }
        | InstKind::Fence(_) => true,
        InstKind::Call(c) => match &c.callee {
            Callee::Air { op, .. } => match op {
                AirIntrinsic::Math(_) | AirIntrinsic::Convert(_) | AirIntrinsic::Group(_) => false,
                AirIntrinsic::Texture(t) => !matches!(
                    t.op,
                    TexOp::Read
                        | TexOp::Sample { .. }
                        | TexOp::SampleGrad { .. }
                        | TexOp::Gather { .. }
                ),
                _ => true,
            },
            _ => true,
        },
        _ => false,
    }
}

/// Whether Metal's backend speculates an arm holding group operation `op` of `scope` (the
/// module docs).
fn speculated(scope: GroupScope, op: GroupOp) -> bool {
    use GroupOp as G;
    match op {
        G::Shuffle
        | G::ShuffleUp
        | G::ShuffleDown
        | G::ShuffleXor
        | G::ShuffleRotateDown
        | G::ShuffleRotateUp
        | G::ShuffleAndFillUp
        | G::ShuffleAndFillDown
        | G::IsFirst => true,
        G::Broadcast | G::BroadcastFirst | G::All | G::Any | G::Ballot | G::ActiveThreadsMask => {
            scope == GroupScope::Quad
        }
        G::Sum
        | G::Min
        | G::Max
        | G::And
        | G::Or
        | G::Xor
        | G::Product
        | G::PrefixExclusiveSum
        | G::PrefixInclusiveSum
        | G::PrefixExclusiveProduct
        | G::PrefixInclusiveProduct => false,
    }
}

/// What Metal does with arms `arms` of content `content` (the module docs).
fn decide(arms: &[Arm], content: Content) -> Decision {
    let inner = arms.iter().flat_map(|a| a.inner.iter());
    let mut unknown = false;
    for i in inner {
        match i.decision {
            Decision::Speculated => {}
            Decision::Kept => return Decision::Kept,
            Decision::Unknown => unknown = true,
        }
    }
    if content.blocker {
        return Decision::Kept;
    }
    if unknown || content.unmeasured {
        return Decision::Unknown;
    }
    let within = |n: usize| n <= BUDGET;
    // Within the budget, an arm of a kind it was not measured with is known speculated only up to
    // `OFF_EDGE`.
    let measured = match !content.off_edge || content.cost <= OFF_EDGE {
        true => Decision::Speculated,
        false => Decision::Unknown,
    };
    match arms {
        [_] => match within(content.cost) {
            true => measured,
            false => Decision::Kept,
        },
        [a, b] => {
            let (t, f) = match a.on_true {
                true => (a.content, b.content),
                false => (b.content, a.content),
            };
            let small = |c: Content| c.cost <= SMALL;
            if within(content.cost) || (small(t) && !t.calls && within(f.cost)) {
                measured
            } else if (small(f) && !f.calls && within(t.cost))
                || (small(t) && t.calls && within(f.cost))
            {
                Decision::Unknown
            } else {
                Decision::Kept
            }
        }
        _ => Decision::Unknown,
    }
}

/// Record in `edit` the speculation of `fold` and of the branches nested in its arms.
fn flatten(
    body: &Body,
    fold: &Fold,
    edit: &mut BodyEdit,
    value_ty: &dyn Fn(Value) -> Option<TypeId>,
) {
    let c = fold.cond;
    let term = |b: BlockId| body.blocks.get(b).and_then(|k| k.last());
    let Some(head_term) = term(fold.head) else {
        return;
    };
    match &fold.arms[..] {
        [a] => {
            edit.rewrite.insert(
                head_term,
                InstKind::Br {
                    dest: a.first,
                    hints: None,
                },
            );
        }
        [a, b] => {
            edit.rewrite.insert(
                head_term,
                InstKind::Br {
                    dest: a.first,
                    hints: None,
                },
            );
            if let Some(last) = term(a.last) {
                edit.rewrite.insert(
                    last,
                    InstKind::Br {
                        dest: b.first,
                        hints: None,
                    },
                );
            }
        }
        _ => return,
    }
    // Each value the arms join with: the true edge's value where the condition holds.
    for (id, i) in body.block_insts(fold.join) {
        let InstKind::Phi { incoming, fmf } = &i.kind else {
            continue;
        };
        let from = |b: BlockId| incoming.iter().find(|&&(_, p)| p == b).map(|&(v, _)| v);
        let (on_true, on_false) = match &fold.arms[..] {
            [a] => match a.on_true {
                true => (from(a.last), from(fold.head)),
                false => (from(fold.head), from(a.last)),
            },
            [a, b] => match a.on_true {
                true => (from(a.last), from(b.last)),
                false => (from(b.last), from(a.last)),
            },
            _ => (None, None),
        };
        if let (Some(t), Some(f)) = (on_true, on_false) {
            edit.rewrite
                .insert(id, InstKind::Select { c, t, f, fmf: *fmf });
        }
    }
    // A sunk store: the arm's store becomes the chosen value, which the branch's block's store,
    // moved to where the arms join, stores.
    if let (Some((h, s)), [a]) = (fold.sunk, &fold.arms[..]) {
        if let (
            Some(InstKind::Store {
                val: v0,
                ptr,
                align,
                ..
            }),
            Some(InstKind::Store { val: v1, .. }),
        ) = (
            body.insts.get(h).map(|i| &i.kind),
            body.insts.get(s).map(|i| &i.kind),
        ) {
            let (t, f) = match a.on_true {
                true => (*v1, *v0),
                false => (*v0, *v1),
            };
            if let Some(ty) = value_ty(*v0) {
                edit.rewrite.insert(
                    s,
                    InstKind::Select {
                        c,
                        t,
                        f,
                        fmf: FastMath::NONE,
                    },
                );
                edit.retype.insert(s, ty);
                edit.rewrite.insert(
                    h,
                    InstKind::Store {
                        val: Value::Inst(s),
                        ptr: *ptr,
                        align: *align,
                        volatile: false,
                        atomic: None,
                    },
                );
                edit.moved.insert(h, fold.join);
            }
        }
    }
    for a in &fold.arms {
        for inner in &a.inner {
            flatten(body, inner, edit, value_ty);
        }
    }
}

/// The branches of `cx`'s body as Metal's backend folds them, by the block ending in each.
fn folds(cx: &Cx<'_>, m: &mut Meter) -> Result<BTreeMap<BlockId, Fold>, Error> {
    let mut out = BTreeMap::new();
    for (b, _) in cx.body.blocks.iter() {
        m.tick(1)?;
        if let Some(fold) = cx.fold(b, 0, m)? {
            out.insert(b, fold);
        }
    }
    Ok(out)
}

/// What Metal's backend does with each branch of `body` whose arms rejoin, by the block ending
/// in it.
#[cfg(test)]
pub(super) fn decisions(
    body: &Body,
    (types, consts): (&TypeTable, &ConstTable),
    (param_tys, params): (&[TypeId], &[Param]),
    m: &mut Meter,
) -> Result<BTreeMap<BlockId, Decision>, Error> {
    let found = analysed(body, (types, consts), (param_tys, params), m)?;
    Ok(found.into_iter().map(|(b, f)| (b, f.decision)).collect())
}

/// The branches of `body` as Metal's backend folds them ([`folds`]), its CFG built and returned.
fn analysed(
    body: &Body,
    (types, consts): (&TypeTable, &ConstTable),
    (param_tys, params): (&[TypeId], &[Param]),
    m: &mut Meter,
) -> Result<BTreeMap<BlockId, Fold>, Error> {
    let cfg = Cfg::build(body, m)?;
    let found = {
        let cx = Cx {
            body,
            cfg: &cfg,
            types,
            consts,
            param_tys,
            params,
        };
        folds(&cx, m)
    };
    m.release(cfg.ledger_bytes());
    found
}

/// Speculate each outermost branch of `body` that Metal's backend speculates and whose arms hold
/// a group operation that answers over the branch's block's lanes there (the module docs), in the
/// entry of parameter types `param_tys` and parameters `params`. Also whether a branch whose arms
/// hold such an operation is one no measurement decides ([`Decision::Unknown`]): it keeps the AIR's
/// lanes, where Metal's backend may speculate it.
pub(super) fn expand(
    body: Body,
    (types, consts): (&TypeTable, &ConstTable),
    (param_tys, params): (&[TypeId], &[Param]),
    m: &mut Meter,
) -> Result<(Body, bool), Error> {
    let found = analysed(&body, (types, consts), (param_tys, params), m)?;
    let unmeasured = found
        .values()
        .any(|f| f.decision == Decision::Unknown && f.content.group);
    let speculated: Vec<&Fold> = found
        .values()
        .filter(|f| f.decision == Decision::Speculated && f.content.group)
        .collect();
    // The outermost: no other speculated branch's arms hold its block.
    let outermost: Vec<&Fold> = speculated
        .iter()
        .copied()
        .filter(|f| {
            !speculated
                .iter()
                .any(|o| o.head != f.head && o.arms.iter().any(|a| a.blocks.contains(&f.head)))
        })
        .collect();
    if outermost.is_empty() {
        return Ok((body, unmeasured));
    }
    let value_ty = |v: Value| value_ty(&body, consts, param_tys, v);
    let next = scoped(m, |m, held| {
        let mut edit = BodyEdit::keep_all(&body, m, held)?;
        for f in &outermost {
            flatten(&body, f, &mut edit, &value_ty);
        }
        apply(&body, &edit, &|c| c, m)
    })?;
    body.release(m);
    Ok((next, unmeasured))
}
