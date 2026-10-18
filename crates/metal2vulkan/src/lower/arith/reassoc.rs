//! Metal's regrouping of a fast-math sum.
//!
//! **What Metal does.** Under fast math Metal's compiler regroups a tree of floating-point
//! additions and subtractions as LLVM's reassociation does, after it splits vectors into lanes,
//! and its instruction combiner then folds the negations the regrouping leaves. A tree is a sum
//! (its root) and the sums and differences it alone uses, in its block, each carrying `reassoc`
//! and `nsz`; its leaves are the other values it reads. A subtraction is an addition of the
//! negated operand, the negation pushed into a subtracted sum's own operands. The leaves are
//! ordered by rank and added left to right, the two of lowest rank first. A value's rank is LLVM's:
//! a parameter's is 3, 4, ... in order; each instruction that may touch memory or must stay in
//! place (a load, a store, a call that is not `readnone`, a phi; [`Ranks::unmovable`]) has one of
//! its own, increasing in program order from a base per block; any other instruction ranks one
//! above its highest operand, a negation none above it. Ranks are taken lane by lane: a lane
//! selection, insertion or extraction adds nothing, and a lane of an instruction's or a
//! parameter's vector, or a member of a call's result, adds one. Leaves of one rank keep the order
//! the tree lists them in (operands first to last, a subtree after the operands beside it). The
//! combiner then rewrites the chain from its innermost addition: `-a + -b` is `-(a + b)` where
//! neither negation has another use, `-a + b` is `b - a`, `a + -b` is `a - b`, and `(p - q) - z`
//! is `p - (q + z)`. A leaf negated in this tree and in one Metal regroups after it shares its
//! negation, which keeps the first rule from it.
//!
//! **Measured** on an Apple M4 (macOS 27.0.1) and the M3 Max (macOS 26.7), byte for byte alike,
//! with our own probes: 448 lanes of `half` and `float` kernel sums (orders, signs, vectors), 168
//! fragment lanes of sample and select sums, and a gradient filter's two sums of six of nine
//! samples in twelve sample orders, each sum as written (the M3 package of the same metallibs). The
//! rank above fits 708 of the 712 lanes; the four others are one sample order whose two leaves tie
//! in rank where Metal orders them as their latest memory reads. The owned fixture `sum_regroup`,
//! as the frontend compiles it, gives Metal's bits on every pixel for its four pinned sums.
//!
//! **The rewrite.** [`apply`] rewrites each sum whose grouping is pinned before selection, so the
//! lowering reads Metal's grouping as plain AIR. Pinned: the rank order is the order of the
//! leaves' latest memory reads (the last ranked instruction each depends on, ties as the rank's),
//! with and without LLVM's operand canonicalization (it puts the operand of lower rank first,
//! which the probes cannot tell apart). Every 582 pinned probe lane is exact. The rewritten
//! additions carry every flag but `reassoc`: they are Metal's grouping, which the device must not
//! regroup again, so the lowering decorates them `NoContraction` and notes no
//! `PrecisionNote::FastMathRewrites` for them. A sum of two leaves that joins no other sum, lane
//! by lane, is Metal's grouping as written ([`Sums::pair`]) and is settled alike.
//!
//! **Not rewritten** (kept as written, with the note): a sum that is not pinned; a vector sum of
//! three leaves or more; one whose operations do not carry every fast-math flag (the probes
//! measured `fast`); one that reaches another block; one with a constant, repeated, negated or
//! multiplied leaf (LLVM folds constants and repeats, factors products, and Metal fuses them); a
//! subtraction inside a subtracted sum; a leaf whose rank reads an instruction the probes did not
//! place (a division, a call outside the AIR and LLVM intrinsics); a leaf negated in a sum that is
//! not modeled or in an earlier sum not rewritten.

use crate::air::ir::{
    BinOp as AirBin, Block, BlockId, Body, Call, Callee, Const, FastMath, FuncId, Idx, Inst,
    InstId, InstKind, IntFlags, MemoryEffects, ModRef, Module, Type, TypeId, Value,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::lower::users::Users;
use std::collections::{BTreeMap, BTreeSet};

/// Every fast-math flag (`fast`), which each operation of a sum carries where the probes measured
/// it. LLVM's reassociation needs `reassoc` and `nsz` of them.
const FAST: FastMath = FastMath::REASSOC
    .union(FastMath::NNAN)
    .union(FastMath::NINF)
    .union(FastMath::NSZ)
    .union(FastMath::ARCP)
    .union(FastMath::CONTRACT)
    .union(FastMath::AFN);

/// The flags LLVM's reassociation reads: an operation with both may join a tree.
const ASSOCIATIVE: FastMath = FastMath::REASSOC.union(FastMath::NSZ);

/// The most leaves a rewritten sum may have.
const MOST_LEAVES: usize = 64;

/// The deepest operand chain a rank follows; a leaf whose rank reads deeper is not modeled.
const DEPTH: u32 = 2048;

fn invariant(what: &'static str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, what))
}

/// Each value's rank and latest memory read, lane by lane (module docs).
struct Ranks<'a> {
    module: &'a Module,
    func: FuncId,
    body: &'a Body,
    /// The rank of each instruction that has one of its own (module docs), by instruction
    /// number; 0 for one without (every own rank is above `3 << 16`).
    own: Vec<u32>,
    /// `(rank, latest read)` of a value's lane (`None`: the scalar, or the vector whole), or
    /// `None` where the rank reads what the probes did not place.
    memo: BTreeMap<(Value, Option<u32>), Option<(u32, u32)>>,
}

impl<'a> Ranks<'a> {
    /// The ranks of the instructions of `body` that have one of their own, block by block in
    /// reverse post-order (LLVM's `BuildRankMap`).
    fn new(module: &'a Module, func: FuncId, body: &'a Body, m: &mut Meter) -> Result<Self, Error> {
        let params = module.func(func).map_or(0, |f| f.params.len());
        let mut rank = 2u32.saturating_add(u32::try_from(params).unwrap_or(u32::MAX));
        let mut this = Ranks {
            module,
            func,
            body,
            own: Vec::new(),
            memo: BTreeMap::new(),
        };
        let mut own = vec![0u32; body.insts.len()];
        for block in reverse_post_order(body, m)? {
            rank = rank.saturating_add(1);
            let mut at = rank
                .checked_shl(16)
                .ok_or_else(|| invariant("a block's rank"))?;
            let Some(b) = body.blocks.get(block) else {
                continue;
            };
            for id in b.inst_ids() {
                m.tick(1)?;
                let Some(inst) = body.insts.get(id) else {
                    continue;
                };
                if this.unmovable(&inst.kind) {
                    at = at.saturating_add(1);
                    if let Some(slot) = own.get_mut(id.index()) {
                        *slot = at;
                    }
                }
            }
        }
        this.own = own;
        Ok(this)
    }

    /// Whether `kind` has a rank of its own (LLVM's `mayHaveNonDefUseDependency`): it may read or
    /// write memory, or may not be executed speculatively. A call is ranked by its operands only
    /// where it is an AIR or LLVM intrinsic that touches no memory and is not convergent.
    fn unmovable(&self, kind: &InstKind) -> bool {
        match kind {
            InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::Alloca { .. }
            | InstKind::AtomicRmw { .. }
            | InstKind::CmpXchg { .. }
            | InstKind::Fence(_)
            | InstKind::Phi { .. }
            | InstKind::Ret(_)
            | InstKind::Br { .. }
            | InstKind::CondBr { .. }
            | InstKind::Switch { .. }
            | InstKind::Unreachable => true,
            InstKind::Call(call) => !self.movable_call(call),
            InstKind::BinOp {
                op: AirBin::UDiv | AirBin::SDiv | AirBin::URem | AirBin::SRem,
                b,
                ..
            } => !self.safe_divisor(*b),
            _ => false,
        }
    }

    /// Whether `call` is an AIR or LLVM intrinsic that touches no memory and is not convergent.
    fn movable_call(&self, call: &Call) -> bool {
        let decl = match call.callee {
            Callee::Air { decl, .. } | Callee::Llvm { decl, .. } => decl,
            _ => return false,
        };
        let none = Some(MemoryEffects::uniform(ModRef::NoModRef));
        let site = call.attrs.as_ref().map(|a| &a.func);
        let Some(f) = self.module.func(decl) else {
            return false;
        };
        let convergent = f.attrs.convergent || site.is_some_and(|a| a.convergent);
        !convergent && (f.attrs.memory == none || site.is_some_and(|a| a.memory == none))
    }

    /// Whether an integer division by `b` is safe to execute anywhere: `b` a constant that is
    /// neither 0 nor all ones.
    fn safe_divisor(&self, b: Value) -> bool {
        let Value::Const(c) = b else {
            return false;
        };
        match self.module.consts().get(c).map(|c| &c.value) {
            Some(Const::Int(v)) => *v != 0 && self.int_bits(b).is_some_and(|w| !all_ones(*v, w)),
            _ => false,
        }
    }

    fn int_bits(&self, v: Value) -> Option<u32> {
        match self
            .module
            .types()
            .get(self.module.value_ty(self.func, v)?)?
        {
            Type::Int(w) => Some(u32::from(*w)),
            _ => None,
        }
    }

    /// The lane count of `v`'s vector type, if it is a vector.
    fn lanes(&self, v: Value) -> Option<u32> {
        match self
            .module
            .types()
            .get(self.module.value_ty(self.func, v)?)?
        {
            Type::Vector { len, .. } => Some(*len),
            _ => None,
        }
    }

    /// The integer constant `v`, if it is one.
    fn const_index(&self, v: Value) -> Option<u32> {
        let Value::Const(c) = v else {
            return None;
        };
        match self.module.consts().get(c).map(|c| &c.value) {
            Some(Const::Int(i)) => u32::try_from(*i).ok(),
            _ => None,
        }
    }

    /// `(rank, latest read)` of lane `lane` of `v` (`None`: a scalar, or a vector whole), or
    /// `None` where it reads an instruction the probes did not place.
    fn of(
        &mut self,
        v: Value,
        lane: Option<u32>,
        depth: u32,
        m: &mut Meter,
    ) -> Result<Option<(u32, u32)>, Error> {
        if let Some(&r) = self.memo.get(&(v, lane)) {
            return Ok(r);
        }
        m.tick(1)?;
        let extra = u32::from(lane.is_some());
        let r = match v {
            Value::Const(_) => Some((0, 0)),
            Value::Arg(a) => Some((a.raw().saturating_add(3).saturating_add(extra), 0)),
            Value::Inst(id) => match self.own.get(id.index()).copied().filter(|r| *r != 0) {
                Some(own) => Some((own.saturating_add(extra), own)),
                None if depth == 0 => None,
                None => self.computed(id, lane, depth - 1, m)?,
            },
        };
        self.memo.insert((v, lane), r);
        Ok(r)
    }

    /// [`Ranks::of`] an instruction without a rank of its own.
    fn computed(
        &mut self,
        id: InstId,
        lane: Option<u32>,
        depth: u32,
        m: &mut Meter,
    ) -> Result<Option<(u32, u32)>, Error> {
        let Some(inst) = self.body.insts.get(id) else {
            return Ok(None);
        };
        let up =
            |r: Option<(u32, u32)>, by: u32| r.map(|(rank, read)| (rank.saturating_add(by), read));
        Ok(match &inst.kind {
            InstKind::InsertElement { vec, elt, idx } => {
                let Some(k) = self.const_index(*idx) else {
                    return Ok(None);
                };
                match lane {
                    Some(l) if l != k => self.of(*vec, lane, depth, m)?,
                    _ => self.of(*elt, None, depth, m)?,
                }
            }
            InstKind::ShuffleVector { a, b, mask } => {
                let Some(n) = self.lanes(*a) else {
                    return Ok(None);
                };
                match mask.get(lane.unwrap_or(0) as usize).copied() {
                    Some(Some(j)) if j < n => self.of(*a, Some(j), depth, m)?,
                    Some(Some(j)) => self.of(*b, Some(j - n), depth, m)?,
                    Some(None) => Some((0, 0)),
                    None => None,
                }
            }
            InstKind::ExtractElement { vec, idx } => match self.const_index(*idx) {
                Some(k) => self.of(*vec, Some(k), depth, m)?,
                None => None,
            },
            InstKind::ExtractValue { agg, .. } => up(
                self.of(*agg, None, depth, m)?,
                1u32.saturating_add(u32::from(lane.is_some())),
            ),
            InstKind::Call(call) => {
                // A movable intrinsic (`Ranks::unmovable`): its vector operands' every lane.
                let mut best = (0u32, 0u32);
                for &arg in call.args.iter() {
                    let r = match self.lanes(arg) {
                        Some(n) => {
                            let mut lanes = (0u32, 0u32);
                            for l in 0..n {
                                let Some(x) = self.of(arg, Some(l), depth, m)? else {
                                    return Ok(None);
                                };
                                lanes = (lanes.0.max(x.0), lanes.1.max(x.1));
                            }
                            lanes
                        }
                        None => match self.of(arg, None, depth, m)? {
                            Some(x) => x,
                            None => return Ok(None),
                        },
                    };
                    best = (best.0.max(r.0), best.1.max(r.1));
                }
                // Its result's lane is one more extraction.
                up(Some(best), if lane.is_some() { 2 } else { 1 })
            }
            // Divisions are movable in LLVM today and were not in older releases; the probes
            // placed none.
            InstKind::BinOp {
                op:
                    AirBin::FDiv
                    | AirBin::FRem
                    | AirBin::UDiv
                    | AirBin::SDiv
                    | AirBin::URem
                    | AirBin::SRem,
                ..
            } => None,
            InstKind::BinOp { .. }
            | InstKind::FNeg { .. }
            | InstKind::Cast { .. }
            | InstKind::ICmp { .. }
            | InstKind::FCmp { .. }
            | InstKind::Select { .. }
            | InstKind::Freeze(_)
            | InstKind::Gep { .. } => {
                let vector = self.lanes(Value::Inst(id)).is_some();
                let mut operands = Vec::new();
                inst.kind.for_each_value(|v| operands.push(v));
                let mut best = (0u32, 0u32);
                for o in operands {
                    let r = match (self.lanes(o), vector) {
                        (Some(_), true) => self.of(o, Some(lane.unwrap_or(0)), depth, m)?,
                        (None, _) => self.of(o, None, depth, m)?,
                        // A scalar of a vector's lanes (a bitcast): not split lane by lane.
                        (Some(_), false) => None,
                    };
                    let Some(r) = r else {
                        return Ok(None);
                    };
                    best = (best.0.max(r.0), best.1.max(r.1));
                }
                let by = u32::from(!matches!(inst.kind, InstKind::FNeg { .. }));
                up(Some(best), by)
            }
            _ => None,
        })
    }
}

/// Whether `v` is all ones in `width` bits.
fn all_ones(v: u64, width: u32) -> bool {
    match width {
        0 => false,
        w if w >= 64 => v == u64::MAX,
        w => v & ((1u64 << w) - 1) == (1u64 << w) - 1,
    }
}

/// The blocks of `body` reachable from the entry, in reverse post-order (successors in operand
/// order, as LLVM's traversal visits them).
fn reverse_post_order(body: &Body, m: &mut Meter) -> Result<Vec<BlockId>, Error> {
    let n = body.blocks.len();
    let mut seen = vec![false; n];
    let mut post = Vec::new();
    let mut stack: Vec<(BlockId, Vec<BlockId>)> = Vec::new();
    let entry = BlockId::new(0);
    if n == 0 {
        return Ok(post);
    }
    if let Some(first) = seen.first_mut() {
        *first = true;
    }
    stack.push((
        entry,
        body.successors(entry)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect(),
    ));
    while let Some((block, rest)) = stack.last_mut() {
        m.tick(1)?;
        match rest.pop() {
            Some(next) => {
                let fresh = match seen.get_mut(next.raw() as usize) {
                    Some(s) if !*s => {
                        *s = true;
                        true
                    }
                    _ => false,
                };
                if fresh {
                    let succ: Vec<BlockId> = body
                        .successors(next)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    stack.push((next, succ));
                }
            }
            None => {
                post.push(*block);
                stack.pop();
            }
        }
    }
    post.reverse();
    Ok(post)
}

/// A leaf of a linearized sum: the value read, negated where `neg`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Leaf {
    v: Value,
    neg: bool,
}

/// A linearized sum (subtractions broken up, negations pushed into subtracted sums).
#[derive(Clone, Debug)]
enum Lin {
    Add(Box<Lin>, Box<Lin>),
    Leaf(Leaf),
}

/// The regrouped sum: the expression the combiner leaves.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Expr {
    Leaf(Value),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
}

/// The facts of the function a sum is read in.
struct Sums<'a> {
    module: &'a Module,
    body: &'a Body,
    /// Each instruction's users, once per operand.
    users: Users,
    /// Each instruction's block, by instruction number.
    block: Vec<Option<BlockId>>,
}

/// What a sum's root holds, as the rewrite reads it.
enum Found {
    /// No sum LLVM reassociates is rooted here (or one with nothing to regroup).
    Nothing,
    /// A sum the rewrite models.
    Modeled(Sum),
    /// A sum LLVM reassociates that the rewrite does not model.
    Unknown,
}

/// A sum as the rewrite reads it.
struct Sum {
    root: InstId,
    /// Its additions and subtractions (the root's tree).
    nodes: BTreeSet<InstId>,
    /// The linearized sum (canonicalized operands or not).
    lin: Lin,
    lin_canonical: Lin,
}

impl<'a> Sums<'a> {
    fn new(module: &'a Module, body: &'a Body, m: &mut Meter) -> Result<Self, Error> {
        let mut block = vec![None; body.insts.len()];
        for (b, blk) in body.blocks.iter() {
            for id in blk.inst_ids() {
                if let Some(slot) = block.get_mut(id.index()) {
                    *slot = Some(b);
                }
            }
        }
        // The work of a walk of every instruction, as the users were counted one by one.
        m.tick(u32::try_from(body.insts.len()).unwrap_or(u32::MAX))?;
        let users = Users::of(body);
        Ok(Sums {
            module,
            body,
            users,
            block,
        })
    }

    fn uses(&self, id: InstId) -> usize {
        self.users.get(id).len()
    }

    /// The block of instruction `id`.
    fn block_of(&self, id: InstId) -> Option<BlockId> {
        self.block.get(id.index()).copied().flatten()
    }

    fn kind(&self, v: Value) -> Option<&'a InstKind> {
        match v {
            Value::Inst(id) => self.body.insts.get(id).map(|i| &i.kind),
            _ => None,
        }
    }

    /// Whether `v` is an addition or subtraction (of any type) carrying `reassoc` and `nsz`.
    fn associative(&self, v: Value) -> bool {
        matches!(
            self.kind(v),
            Some(InstKind::BinOp {
                op: AirBin::FAdd | AirBin::FSub,
                fmf,
                ..
            }) if fmf.contains(ASSOCIATIVE)
        )
    }

    /// Whether `v` is a lane selection, insertion or extraction that reaches such an addition
    /// through at most `depth` of them: after the vectors are split into lanes, the addition's
    /// lane may join the sum that reads `v`.
    fn lane_reaches_sum(&self, v: Value, depth: u32) -> bool {
        let sources = match self.kind(v) {
            Some(InstKind::ExtractElement { vec, .. }) => [Some(*vec), None],
            Some(InstKind::InsertElement { vec, elt, .. }) => [Some(*vec), Some(*elt)],
            Some(InstKind::ShuffleVector { a, b, .. }) => [Some(*a), Some(*b)],
            _ => return false,
        };
        sources
            .into_iter()
            .flatten()
            .any(|s| self.associative(s) || (depth > 0 && self.lane_reaches_sum(s, depth - 1)))
    }

    /// Whether a lane selection, insertion or extraction reading `id` reaches such an addition
    /// through at most `depth` of them: `id`'s lanes may join the sum it reads.
    fn feeds_sum(&self, id: InstId, depth: u32) -> bool {
        self.users
            .get(id)
            .iter()
            .any(|&u| match self.body.insts.get(u).map(|i| &i.kind) {
                Some(
                    InstKind::ExtractElement { .. }
                    | InstKind::InsertElement { .. }
                    | InstKind::ShuffleVector { .. },
                ) => {
                    self.users
                        .get(u)
                        .iter()
                        .any(|&w| self.associative(Value::Inst(w)))
                        || (depth > 0 && self.feeds_sum(u, depth - 1))
                }
                _ => false,
            })
    }

    /// The users of `id` that read it whole (no lane selection, insertion or extraction).
    fn whole_readers(&self, id: InstId) -> usize {
        self.users
            .get(id)
            .iter()
            .filter(|&&u| {
                !matches!(
                    self.body.insts.get(u).map(|i| &i.kind),
                    Some(
                        InstKind::ExtractElement { .. }
                            | InstKind::InsertElement { .. }
                            | InstKind::ShuffleVector { .. }
                    )
                )
            })
            .count()
    }

    /// A sum of two leaves (`half` or `float`, scalar or vector) carrying every fast-math flag,
    /// as written: Metal's grouping, which LLVM's reassociation leaves alone (no operand of it
    /// joins it and it joins no other sum, lane by lane), and the combiner too (neither leaf a
    /// constant, a negation, a product or a sum of the other). `None` for any other.
    fn pair(&self, root: InstId) -> Option<Expr> {
        let inst = self.body.insts.get(root)?;
        let InstKind::BinOp {
            op: op @ (AirBin::FAdd | AirBin::FSub),
            fmf,
            a,
            b,
            ..
        } = &inst.kind
        else {
            return None;
        };
        let elem = match self.module.types().get(inst.ty)? {
            Type::Vector { elem, .. } => self.module.types().get(*elem)?,
            t => t,
        };
        if !matches!(elem, Type::Half | Type::Float) || !fmf.contains(FAST) || a == b {
            return None;
        }
        // Each lane of an operand that is a sum is read twice or more: a leaf.
        let leaf = |v: Value, other: Value| -> bool {
            if self.odd_leaf(v) || self.lane_reaches_sum(v, 4) {
                return false;
            }
            match (self.kind(v), v) {
                (
                    Some(InstKind::BinOp {
                        op: AirBin::FAdd | AirBin::FSub,
                        a: x,
                        b: y,
                        ..
                    }),
                    Value::Inst(id),
                ) => *x != other && *y != other && self.whole_readers(id) >= 2,
                _ => true,
            }
        };
        if !leaf(*a, *b) || !leaf(*b, *a) {
            return None;
        }
        // It joins no sum: every lane of it is read twice or more, or no sum reads it, and no
        // lane of it reaches one.
        let joins = self
            .users
            .get(root)
            .iter()
            .any(|&u| self.associative(Value::Inst(u)));
        if (joins && self.whole_readers(root) < 2) || self.feeds_sum(root, 4) {
            return None;
        }
        let (x, y) = (Box::new(Expr::Leaf(*a)), Box::new(Expr::Leaf(*b)));
        Some(if *op == AirBin::FAdd {
            Expr::Add(x, y)
        } else {
            Expr::Sub(x, y)
        })
    }

    /// `(op, fmf, a, b)` of an addition or subtraction `v` of a `half` or `float` scalar.
    fn add(&self, v: Value) -> Option<(AirBin, FastMath, Value, Value)> {
        let Value::Inst(id) = v else {
            return None;
        };
        let inst = self.body.insts.get(id)?;
        let InstKind::BinOp {
            op: op @ (AirBin::FAdd | AirBin::FSub),
            fmf,
            a,
            b,
            ..
        } = &inst.kind
        else {
            return None;
        };
        matches!(
            self.module.types().get(inst.ty),
            Some(Type::Half | Type::Float)
        )
        .then_some((*op, *fmf, *a, *b))
    }

    /// Whether `v` is an addition or subtraction LLVM's reassociation would take into the tree of
    /// its one user (`isReassociableOp`): one use, both flags. `None` where it would and the
    /// rewrite does not model it (not every flag, or another block than `block`).
    fn joins(&self, v: Value, block: BlockId) -> Option<bool> {
        let Some((_, fmf, _, _)) = self.add(v) else {
            return Some(false);
        };
        let Value::Inst(id) = v else {
            return Some(false);
        };
        if self.uses(id) != 1 || !fmf.contains(ASSOCIATIVE) {
            return Some(false);
        }
        (fmf.contains(FAST) && self.block_of(id) == Some(block)).then_some(true)
    }

    /// Whether the subtraction `id` (`a - b`) is broken up (LLVM's `ShouldBreakUpSubtract`):
    /// an operand joins a tree, or its one user is a sum that joins one.
    fn broken_up(&self, id: InstId, a: Value, b: Value) -> bool {
        let joinable = |v: Value| {
            let Value::Inst(i) = v else { return false };
            self.add(v)
                .is_some_and(|(_, fmf, _, _)| self.uses(i) == 1 && fmf.contains(ASSOCIATIVE))
        };
        joinable(a)
            || joinable(b)
            || (self.uses(id) == 1
                && self
                    .users
                    .get(id)
                    .first()
                    .is_some_and(|&u| joinable(Value::Inst(u))))
    }

    /// The sum rooted at `root` (module docs).
    fn sum(&self, root: InstId, ranks: &mut Ranks<'_>, m: &mut Meter) -> Result<Found, Error> {
        let Some((op, fmf, a, b)) = self.add(Value::Inst(root)) else {
            return Ok(Found::Nothing);
        };
        if !fmf.contains(ASSOCIATIVE) {
            return Ok(Found::Nothing);
        }
        let Some(block) = self.block_of(root) else {
            return Ok(Found::Nothing);
        };
        // The root's one user would take it into its own tree.
        if self.uses(root) == 1 {
            let user = self.users.get(root).first().copied();
            if let Some(u) = user {
                if let Some((_, ufmf, _, _)) = self.add(Value::Inst(u)) {
                    if ufmf.contains(ASSOCIATIVE) {
                        return Ok(Found::Nothing);
                    }
                }
            }
        }
        if op == AirBin::FSub && !self.broken_up(root, a, b) {
            // A difference of two leaves: nothing to regroup.
            return Ok(Found::Nothing);
        }
        if !fmf.contains(FAST) || self.feeds_sum(root, 4) {
            return Ok(Found::Unknown);
        }
        let mut nodes = BTreeSet::new();
        nodes.insert(root);
        let walk = |canonical| Walk { block, canonical };
        let Some(lin) = self.linear((op, a, b), false, walk(false), &mut nodes, ranks, m)? else {
            return Ok(Found::Unknown);
        };
        let mut canonical_nodes = BTreeSet::new();
        canonical_nodes.insert(root);
        let Some(lin_canonical) = self.linear(
            (op, a, b),
            false,
            walk(true),
            &mut canonical_nodes,
            ranks,
            m,
        )?
        else {
            return Ok(Found::Unknown);
        };
        Ok(Found::Modeled(Sum {
            root,
            nodes,
            lin,
            lin_canonical,
        }))
    }

    /// The linearized operation `op` (`a op b`) negated where `neg`, as `walk` walks it, its
    /// joined operations added to `nodes`. `None` where the rewrite does not model it.
    fn linear(
        &self,
        (op, a, b): (AirBin, Value, Value),
        neg: bool,
        walk: Walk,
        nodes: &mut BTreeSet<InstId>,
        ranks: &mut Ranks<'_>,
        m: &mut Meter,
    ) -> Result<Option<Lin>, Error> {
        let Walk { block, canonical } = walk;
        m.tick(1)?;
        if nodes.len() > MOST_LEAVES {
            return Ok(None);
        }
        let (a, b) = if canonical && op == AirBin::FAdd {
            match (self.tree_rank(a, ranks, m)?, self.tree_rank(b, ranks, m)?) {
                (Some(ra), Some(rb)) if rb < ra => (b, a),
                (Some(_), Some(_)) => (a, b),
                _ => return Ok(None),
            }
        } else {
            (a, b)
        };
        // A subtraction inside a subtracted sum: a double negation the rewrite does not model.
        if op == AirBin::FSub && neg {
            return Ok(None);
        }
        let side = |this: &Self,
                    v: Value,
                    neg: bool,
                    nodes: &mut BTreeSet<InstId>,
                    ranks: &mut Ranks<'_>,
                    m: &mut Meter|
         -> Result<Option<Lin>, Error> {
            match this.joins(v, block) {
                None => Ok(None),
                Some(true) => {
                    let (Some((op, _, x, y)), Value::Inst(id)) = (this.add(v), v) else {
                        return Ok(None);
                    };
                    if op == AirBin::FSub && !this.broken_up(id, x, y) {
                        return Ok(None);
                    }
                    nodes.insert(id);
                    this.linear((op, x, y), neg, walk, nodes, ranks, m)
                }
                Some(false) => Ok(Some(Lin::Leaf(Leaf { v, neg }))),
            }
        };
        let left = side(self, a, neg, nodes, ranks, m)?;
        let right = side(self, b, neg != (op == AirBin::FSub), nodes, ranks, m)?;
        Ok(match (left, right) {
            (Some(l), Some(r)) => Some(Lin::Add(Box::new(l), Box::new(r))),
            _ => None,
        })
    }

    /// The rank LLVM gives `v` as an operand of a sum it canonicalizes: a joined operation's is
    /// one above its operands', a leaf's its own.
    fn tree_rank(
        &self,
        v: Value,
        ranks: &mut Ranks<'_>,
        m: &mut Meter,
    ) -> Result<Option<u32>, Error> {
        if let (Some((_, fmf, a, b)), Value::Inst(id)) = (self.add(v), v) {
            if self.uses(id) == 1 && fmf.contains(ASSOCIATIVE) {
                let (Some(ra), Some(rb)) =
                    (self.tree_rank(a, ranks, m)?, self.tree_rank(b, ranks, m)?)
                else {
                    return Ok(None);
                };
                return Ok(Some(ra.max(rb).saturating_add(1)));
            }
        }
        Ok(ranks.of(v, None, DEPTH, m)?.map(|(r, _)| r))
    }

    /// Whether `v` may not be a leaf the rewrite models: a constant, a negation, or a product.
    fn odd_leaf(&self, v: Value) -> bool {
        match v {
            Value::Const(_) => true,
            Value::Arg(_) => false,
            Value::Inst(id) => match self.body.insts.get(id).map(|i| &i.kind) {
                Some(InstKind::FNeg { .. }) => true,
                Some(InstKind::BinOp {
                    op: AirBin::FMul, ..
                }) => true,
                Some(InstKind::BinOp {
                    op: AirBin::FSub,
                    a: Value::Const(c),
                    ..
                }) => matches!(
                    self.module.consts().get(*c).map(|c| &c.value),
                    Some(Const::Float(_))
                ),
                _ => false,
            },
        }
    }

    /// Every value the sum rooted at `root` (one LLVM reassociates) reads: the operands of its
    /// operations, through the sums and differences it alone uses.
    fn reads(&self, root: InstId, m: &mut Meter) -> Result<BTreeSet<Value>, Error> {
        let mut read = BTreeSet::new();
        let mut stack = vec![Value::Inst(root)];
        while let Some(v) = stack.pop() {
            m.tick(1)?;
            if let (Some((_, _, a, b)), Value::Inst(i)) = (self.add(v), v) {
                if i == root || self.uses(i) == 1 {
                    stack.extend([a, b]);
                    continue;
                }
            }
            read.insert(v);
        }
        Ok(read)
    }

    /// Whether `v` already has a negation in the function, which LLVM's negation would reuse.
    fn negated_elsewhere(&self, v: Value) -> bool {
        let Value::Inst(id) = v else {
            return false;
        };
        self.users.get(id).iter().any(|&u| {
            matches!(
                self.body.insts.get(u).map(|i| &i.kind),
                Some(InstKind::FNeg { .. })
            )
        })
    }
}

/// How a linearization walks a sum ([`Sums::linear`]): within the root's `block`, each addition's
/// operands canonicalized where `canonical` (the one of lower rank first, LLVM's
/// `canonicalizeOperands`).
#[derive(Clone, Copy)]
struct Walk {
    block: BlockId,
    canonical: bool,
}

/// The leaves of `lin` in LLVM's order (`LinearizeExprTree`: a stack of sums, each sum's
/// operands first to last, a leaf listed when met).
fn leaf_order(lin: &Lin) -> Vec<Leaf> {
    let mut out = Vec::new();
    let mut stack = vec![lin];
    while let Some(Lin::Add(a, b)) = stack.pop() {
        for side in [a.as_ref(), b.as_ref()] {
            match side {
                Lin::Leaf(l) => out.push(*l),
                add @ Lin::Add(..) => stack.push(add),
            }
        }
    }
    if let Lin::Leaf(l) = lin {
        out.push(*l);
    }
    out
}

/// `leaves` sorted by `key` descending, ties in their order (a stable sort).
fn descending(leaves: &[Leaf], key: &BTreeMap<Value, u32>) -> Vec<Leaf> {
    let mut v: Vec<(usize, Leaf)> = leaves.iter().copied().enumerate().collect();
    v.sort_by(|(i, x), (j, y)| {
        let (kx, ky) = (
            key.get(&x.v).copied().unwrap_or(0),
            key.get(&y.v).copied().unwrap_or(0),
        );
        ky.cmp(&kx).then(i.cmp(j))
    });
    v.into_iter().map(|(_, l)| l).collect()
}

/// The combiner's result over the left-linear chain of `ops` (highest rank first): from the
/// innermost addition `ops[n-2] + ops[n-1]`, each addition folded by the module docs' rules, a
/// negation of a leaf of `shared` having another use.
fn fold(ops: &[Leaf], shared: &BTreeSet<Value>) -> Option<Expr> {
    let leaf = |l: &Leaf| {
        let e = Expr::Leaf(l.v);
        if l.neg {
            Expr::Neg(Box::new(e))
        } else {
            e
        }
    };
    let one_use = |e: &Expr| match e {
        Expr::Neg(x) => !matches!(x.as_ref(), Expr::Leaf(v) if shared.contains(v)),
        _ => false,
    };
    let fsub = |x: Expr, z: Expr| match x {
        Expr::Sub(p, q) => Expr::Sub(p, Box::new(Expr::Add(q, Box::new(z)))),
        x => Expr::Sub(Box::new(x), Box::new(z)),
    };
    let fadd = |l: Expr, r: Expr| match (l, r) {
        (Expr::Neg(a), Expr::Neg(b))
            if one_use(&Expr::Neg(a.clone())) && one_use(&Expr::Neg(b.clone())) =>
        {
            Expr::Neg(Box::new(Expr::Add(a, b)))
        }
        (Expr::Neg(a), r) => fsub(r, *a),
        (l, Expr::Neg(b)) => fsub(l, *b),
        (l, r) => Expr::Add(Box::new(l), Box::new(r)),
    };
    let (rest, [x, y]) = ops.split_at_checked(ops.len().checked_sub(2)?)? else {
        return None;
    };
    let mut acc = fadd(leaf(x), leaf(y));
    for l in rest.iter().rev() {
        acc = fadd(acc, leaf(l));
    }
    Some(acc)
}

/// The regrouped expression and the operations of each sum of the entry the rewrite rewrites,
/// by root.
type Planned = BTreeMap<InstId, (Expr, BTreeSet<InstId>)>;

/// A sum LLVM reassociates, in its visiting order: the sum and its leaves where the rewrite models
/// it, and the values it negates (or every value its operations read, which it may negate).
type Visited = (Option<(Sum, Vec<Leaf>)>, BTreeSet<Value>);

/// What the rewrite decides for each sum of the entry (module docs), visiting them in LLVM's
/// order.
fn plan(module: &Module, func: FuncId, body: &Body, m: &mut Meter) -> Result<Planned, Error> {
    let mut ranks = Ranks::new(module, func, body, m)?;
    let sums = Sums::new(module, body, m)?;
    let mut seen: Vec<Visited> = Vec::new();
    let mut pairs: Planned = BTreeMap::new();
    for block in reverse_post_order(body, m)? {
        let Some(b) = body.blocks.get(block) else {
            continue;
        };
        for id in b.inst_ids() {
            m.tick(1)?;
            if let Some(e) = sums.pair(id) {
                pairs.insert(id, (e, BTreeSet::from([id])));
                continue;
            }
            match sums.sum(id, &mut ranks, m)? {
                Found::Nothing => {}
                Found::Modeled(sum) => {
                    let leaves = leaf_order(&sum.lin);
                    let negated = leaves.iter().filter(|l| l.neg).map(|l| l.v).collect();
                    seen.push((Some((sum, leaves)), negated));
                }
                Found::Unknown => seen.push((None, sums.reads(id, m)?)),
            }
        }
    }
    let negated: Vec<(bool, &BTreeSet<Value>)> = seen
        .iter()
        .map(|(s, values)| (s.is_some(), values))
        .collect();
    let mut out = pairs;
    let mut results: Vec<Option<Expr>> = Vec::new();
    for (at, (s, _)) in seen.iter().enumerate() {
        let decided = match s {
            Some((sum, leaves)) => {
                let order = Order {
                    at,
                    negated: &negated,
                    earlier: &results,
                };
                let e = decide(&sums, &mut ranks, (sum, leaves), &order, m)?;
                if let Some(e) = &e {
                    out.insert(sum.root, (e.clone(), sum.nodes.clone()));
                }
                e
            }
            None => None,
        };
        results.push(decided);
    }
    Ok(out)
}

/// Where a sum lies among the function's sums, in LLVM's visiting order ([`decide`]).
struct Order<'v> {
    /// The sum's index.
    at: usize,
    /// Whether each sum is modeled, and the values it negates (or every value it reads).
    negated: &'v [(bool, &'v BTreeSet<Value>)],
    /// Each earlier sum's result (`None`: not rewritten).
    earlier: &'v [Option<Expr>],
}

/// Metal's expression for `sum` (whose leaves are `leaves`, at `order` among the function's
/// sums), if it is pinned (module docs).
fn decide(
    sums: &Sums<'_>,
    ranks: &mut Ranks<'_>,
    (sum, leaves): (&Sum, &[Leaf]),
    order: &Order<'_>,
    m: &mut Meter,
) -> Result<Option<Expr>, Error> {
    if leaves.len() < 3 || leaves.len() > MOST_LEAVES {
        return Ok(None);
    }
    let distinct: BTreeSet<Value> = leaves.iter().map(|l| l.v).collect();
    if distinct.len() != leaves.len()
        || leaves
            .iter()
            .any(|l| sums.odd_leaf(l.v) || sums.lane_reaches_sum(l.v, 4))
    {
        return Ok(None);
    }
    // A leaf that is a sum of another leaf (`(x + y) - y`): the combiner may fold it.
    for l in leaves {
        if let Some(InstKind::BinOp {
            op: AirBin::FAdd | AirBin::FSub,
            a,
            b,
            ..
        }) = sums.kind(l.v)
        {
            if distinct.contains(a) || distinct.contains(b) {
                return Ok(None);
            }
        }
    }
    let mut rank = BTreeMap::new();
    let mut read = BTreeMap::new();
    for l in leaves {
        let Some((r, latest)) = ranks.of(l.v, None, DEPTH, m)? else {
            return Ok(None);
        };
        rank.insert(l.v, r);
        read.insert(l.v, latest);
    }
    // Pinned: the rank's order is the latest reads' order, canonicalized or not.
    let ops = descending(leaves, &rank);
    if descending(leaves, &read) != ops {
        return Ok(None);
    }
    let canonical = leaf_order(&sum.lin_canonical);
    if descending(&canonical, &rank) != ops {
        return Ok(None);
    }
    // The negations this sum shares: a leaf negated in a later sum too. A leaf negated in a sum
    // not modeled, or in an earlier one not rewritten, is not modeled.
    let mut shared = BTreeSet::new();
    for l in leaves.iter().filter(|l| l.neg) {
        if sums.negated_elsewhere(l.v) {
            return Ok(None);
        }
        for (j, &(modeled, values)) in order.negated.iter().enumerate() {
            if j == order.at || !values.contains(&l.v) {
                continue;
            }
            if !modeled {
                return Ok(None);
            }
            if j > order.at {
                shared.insert(l.v);
            } else if order.earlier.get(j).is_none_or(Option::is_none) {
                return Ok(None);
            }
        }
    }
    Ok(fold(&ops, &shared))
}

/// The entry with each sum whose grouping is pinned rewritten to Metal's (module docs). An entry
/// with none is returned unchanged.
pub fn apply(mut e: Entry, m: &mut Meter) -> Result<Entry, Error> {
    let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
        return Ok(e);
    };
    let planned = plan(&e.module, e.func, body, m)?;
    if planned.is_empty() {
        return Ok(e);
    }
    let rewritten = rebuild(body, &planned, m)?;
    let mut parts = e.module.into_parts();
    let func = parts
        .funcs
        .get_mut(e.func)
        .ok_or_else(|| invariant("the entry out of range"))?;
    if let Some(old) = func.body.replace(rewritten) {
        old.release(m);
    }
    e.module = Module::new(parts, m)?;
    Ok(e)
}

/// The flags of a rewritten operation: the root's, but `reassoc`.
fn settled(fmf: FastMath) -> FastMath {
    FastMath::ALL
        .iter()
        .filter(|(f, _)| *f != FastMath::REASSOC && fmf.contains(*f))
        .fold(FastMath::NONE, |s, (f, _)| s.union(*f))
}

/// The operations `e` takes, innermost first.
fn count(e: &Expr) -> u32 {
    match e {
        Expr::Leaf(_) => 0,
        Expr::Neg(x) => 1 + count(x),
        Expr::Add(a, b) | Expr::Sub(a, b) => 1 + count(a) + count(b),
    }
}

/// `body` with each planned sum's operations deleted and its root replaced by the operations of
/// its expression, placed where the root was.
fn rebuild(
    body: &Body,
    planned: &BTreeMap<InstId, (Expr, BTreeSet<InstId>)>,
    m: &mut Meter,
) -> Result<Body, Error> {
    // Each instruction a planned sum deletes, and each planned root, by instruction number.
    let mut deleted = vec![false; body.insts.len()];
    let mut root = vec![false; body.insts.len()];
    for (r, (_, nodes)) in planned {
        for n in nodes.iter().filter(|n| *n != r) {
            if let Some(d) = deleted.get_mut(n.index()) {
                *d = true;
            }
        }
        if let Some(x) = root.get_mut(r.index()) {
            *x = true;
        }
    }
    let deleted = |id: InstId| deleted.get(id.index()).copied().unwrap_or(false);
    let planned_at = |id: InstId| {
        if root.get(id.index()).copied().unwrap_or(false) {
            planned.get(&id)
        } else {
            None
        }
    };
    let mut map: Vec<Option<InstId>> = Vec::new();
    m.reserve(&mut map, body.insts.len())?;
    let mut next = 0u32;
    for (id, _) in body.insts.iter() {
        m.tick(1)?;
        if deleted(id) {
            map.push(None);
            continue;
        }
        if let Some((e, _)) = planned_at(id) {
            let ops = count(e);
            if ops == 0 {
                return Err(invariant("a regrouped sum without an operation"));
            }
            next = next
                .checked_add(ops - 1)
                .ok_or_else(|| invariant("instruction ids"))?;
        }
        map.push(Some(InstId::new(next)));
        next = next
            .checked_add(1)
            .ok_or_else(|| invariant("instruction ids"))?;
    }
    let remap = |v: Value| -> Result<Value, Error> {
        match v {
            Value::Inst(i) => match map.get(i.index()) {
                Some(Some(n)) => Ok(Value::Inst(*n)),
                _ => Err(invariant("a use of a deleted instruction")),
            },
            other => Ok(other),
        }
    };
    let mut out = Body::with_capacity(next as usize, body.blocks.len(), m)?;
    for (_, block) in body.blocks.iter() {
        m.tick(1)?;
        let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        for id in block.inst_ids() {
            if deleted(id) {
                continue;
            }
            let inst = body
                .insts
                .get(id)
                .ok_or_else(|| invariant("instruction out of range"))?;
            if let Some((e, _)) = planned_at(id) {
                let InstKind::BinOp { flags, fmf, .. } = &inst.kind else {
                    return Err(invariant("a regrouped sum's root is not an addition"));
                };
                let fmf = settled(*fmf);
                let last = emit(e, inst.ty, *flags, fmf, &remap, &mut out, m)?;
                if map.get(id.index()) != Some(&Some(last)) {
                    return Err(invariant("a regrouped sum's root out of place"));
                }
                continue;
            }
            let mut kind = inst.kind.clone();
            let mut failed = None;
            kind.for_each_value_mut(|v| match remap(*v) {
                Ok(n) => *v = n,
                Err(e) => {
                    failed.get_or_insert(e);
                }
            });
            if let Some(e) = failed {
                return Err(e);
            }
            out.push_inst(Inst { ty: inst.ty, kind }, m)?;
        }
        let end = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        out.blocks.push(
            Block {
                start,
                end,
                name: block.name,
            },
            m,
        )?;
    }
    m.release(crate::budget::vec_bytes(&map));
    Ok(out)
}

/// Push the operations of `e` (type `ty`, flags `flags` and `fmf`) into `out`, innermost first,
/// and return the instruction holding its value.
fn emit(
    e: &Expr,
    ty: TypeId,
    flags: IntFlags,
    fmf: FastMath,
    remap: &dyn Fn(Value) -> Result<Value, Error>,
    out: &mut Body,
    m: &mut Meter,
) -> Result<InstId, Error> {
    let value = |x: &Expr, out: &mut Body, m: &mut Meter| -> Result<Value, Error> {
        match x {
            Expr::Leaf(v) => remap(*v),
            other => Ok(Value::Inst(emit(other, ty, flags, fmf, remap, out, m)?)),
        }
    };
    let kind = match e {
        Expr::Leaf(_) => return Err(invariant("a leaf emitted as an operation")),
        Expr::Neg(x) => InstKind::FNeg {
            v: value(x, out, m)?,
            fmf,
        },
        Expr::Add(a, b) | Expr::Sub(a, b) => {
            let a = value(a, out, m)?;
            let b = value(b, out, m)?;
            InstKind::BinOp {
                op: if matches!(e, Expr::Add(..)) {
                    AirBin::FAdd
                } else {
                    AirBin::FSub
                },
                flags,
                fmf,
                a,
                b,
            }
        }
    };
    Ok(out.push_inst(Inst { ty, kind }, m)?)
}
