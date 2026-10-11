//! Counted loops: a loop whose header visits per entry a constant bounds, read from its structure
//! alone.
//!
//! A loop is counted when it is reducible with one latch, and the header's or the latch's
//! conditional branch leaves the loop on an integer comparison of a constant with its induction
//! variable: a header phi whose value from outside the loop is one integer constant `init` and
//! whose value from the latch is that phi plus or minus a constant (the comparison may read that
//! sum instead of the phi). Every back edge comes from the latch, so the phi's value at the
//! header's n-th visit follows from `init` and the step whatever path the body took, and the
//! branch, which every iteration passes (the header's, or the sole latch's), sees it. Running that
//! sequence through the comparison, in the phi's width with its wrap, gives the first visit at
//! which the branch exits: the loop's bound. A sum whose `nuw` or `nsw` flag the sequence breaks
//! is poison, so such a loop is not counted, nor is one that has not exited within the limit the
//! caller gives.
//!
//! Exits elsewhere in the loop only end it sooner. Nothing here reads a function constant, a
//! buffer, or a name.

use std::collections::{BTreeMap, BTreeSet};

use metal2vulkan_air::graph::{LoopForest, LoopId};
use metal2vulkan_air::ir::{
    BinOp, BlockId, Body, Const, InstId, InstKind, IntFlags, IntPredicate, Module, Type, Value,
};

/// The most header visits a counted loop may take: a loop that has not exited after this many is
/// not counted.
pub const COUNTED_LOOP_LIMIT: u64 = 1 << 16;

/// One loop of one function, as the analysis reads it.
struct LoopAt<'a> {
    module: &'a Module,
    body: &'a Body,
    forest: &'a LoopForest<BlockId>,
    l: LoopId,
    header: BlockId,
    latch: BlockId,
    /// The header's predecessors.
    into: &'a BTreeSet<BlockId>,
}

/// The header visits per entry of loop `l` of a function with body `body`, loop forest `forest`
/// and header predecessors `preds`, when the loop is counted and exits within `limit` visits
/// (see the module documentation); `None` otherwise.
pub(super) fn header_visits(
    module: &Module,
    body: &Body,
    forest: &LoopForest<BlockId>,
    preds: &BTreeMap<BlockId, BTreeSet<BlockId>>,
    l: LoopId,
    limit: u64,
) -> Option<u64> {
    if !forest.is_reducible(l) {
        return None;
    }
    let header = forest.header(l);
    let into = preds.get(&header)?;
    let mut latches = into.iter().filter(|&&p| forest.contains(l, p));
    let latch = *latches.next()?;
    if latches.next().is_some() {
        return None;
    }
    let at = LoopAt {
        module,
        body,
        forest,
        l,
        header,
        latch,
        into,
    };
    let mut tested = vec![header];
    if latch != header {
        tested.push(latch);
    }
    tested
        .into_iter()
        .filter_map(|block| at.exit_test(block))
        .filter_map(|test| test.visits(limit))
        .min()
}

/// How one iteration moves the induction variable.
#[derive(Clone, Copy)]
enum Step {
    /// `add phi, k`.
    Add(u64),
    /// `sub phi, k`.
    Sub(u64),
}

/// The exit test of a loop: its induction variable and the branch that leaves on a comparison.
struct ExitTest {
    /// The induction variable's width in bits (1 to 64).
    width: u32,
    /// Its value at the first header visit.
    init: u64,
    step: Step,
    /// The step's wrap flags.
    flags: IntFlags,
    /// Whether the comparison reads the step's result rather than the phi.
    reads_sum: bool,
    /// The comparison with the induction variable on the left.
    pred: IntPredicate,
    /// The constant on its right.
    bound: u64,
    /// Whether the branch leaves the loop when the comparison holds.
    exits_when: bool,
}

impl ExitTest {
    /// The first header visit at which the branch exits, counted from 1, within `limit`.
    fn visits(&self, limit: u64) -> Option<u64> {
        let m = mask(self.width);
        let mut v = self.init & m;
        for n in 1..=limit {
            let (next, wrapped_unsigned, wrapped_signed) = match self.step {
                Step::Add(k) => add(v, k & m, self.width),
                Step::Sub(k) => sub(v, k & m, self.width),
            };
            if (self.flags.contains(IntFlags::NUW) && wrapped_unsigned)
                || (self.flags.contains(IntFlags::NSW) && wrapped_signed)
            {
                return None;
            }
            let x = if self.reads_sum { next } else { v };
            if compare(self.pred, x, self.bound & m, self.width) == self.exits_when {
                return Some(n);
            }
            v = next;
        }
        None
    }
}

impl LoopAt<'_> {
    /// The exit test at `block` (the header or the sole latch), when it is one.
    fn exit_test(&self, block: BlockId) -> Option<ExitTest> {
        let body = self.body;
        let InstKind::CondBr { c, t, f, .. } = &body.terminator(block)?.kind else {
            return None;
        };
        let exits_when = match (
            self.forest.contains(self.l, *t),
            self.forest.contains(self.l, *f),
        ) {
            (false, true) => true,
            (true, false) => false,
            _ => return None,
        };
        let Value::Inst(cmp) = c else { return None };
        let InstKind::ICmp { pred, a, b } = &body.insts.get(*cmp)?.kind else {
            return None;
        };
        let (x, pred, bound) = match (self.int_const(*a), self.int_const(*b)) {
            (None, Some(k)) => (*a, *pred, k),
            (Some(k), None) => (*b, swapped(*pred), k),
            _ => return None,
        };
        let Value::Inst(x) = x else { return None };
        // The header phi, and whether the comparison reads it or the step's result.
        let (phi, reads_sum) = if self.is_header_phi(x) {
            (x, false)
        } else {
            match &body.insts.get(x)?.kind {
                InstKind::BinOp { a, b, .. } => {
                    let p = [a, b].into_iter().find_map(|v| match v {
                        Value::Inst(p) if self.is_header_phi(*p) => Some(*p),
                        _ => None,
                    })?;
                    (p, true)
                }
                _ => return None,
            }
        };
        let InstKind::Phi { incoming, .. } = &body.insts.get(phi)?.kind else {
            return None;
        };
        let width = match self.module.types().get(body.insts.get(phi)?.ty)? {
            Type::Int(w) if (1..=64).contains(w) => u32::from(*w),
            _ => return None,
        };
        // Every incoming edge is one of the header's; the latch's carries the step's result,
        // every other one the same constant.
        let mut init = None;
        let mut next = None;
        for (value, from) in incoming.iter() {
            if !self.into.contains(from) {
                return None;
            }
            if *from == self.latch {
                let Value::Inst(s) = value else { return None };
                if next.replace(*s).is_some_and(|prev| prev != *s) {
                    return None;
                }
            } else {
                let k = self.int_const(*value)?;
                if init.replace(k).is_some_and(|prev| prev != k) {
                    return None;
                }
            }
        }
        let (init, next) = (init?, next?);
        if reads_sum && next != x {
            return None;
        }
        let InstKind::BinOp {
            op, flags, a, b, ..
        } = &body.insts.get(next)?.kind
        else {
            return None;
        };
        let of_phi = |v: &Value| matches!(v, Value::Inst(p) if *p == phi);
        let step = match op {
            BinOp::Add if of_phi(a) => Step::Add(self.int_const(*b)?),
            BinOp::Add if of_phi(b) => Step::Add(self.int_const(*a)?),
            BinOp::Sub if of_phi(a) => Step::Sub(self.int_const(*b)?),
            _ => return None,
        };
        Some(ExitTest {
            width,
            init,
            step,
            flags: *flags,
            reads_sum,
            pred,
            bound,
            exits_when,
        })
    }

    /// Whether instruction `i` is a phi of the loop's header.
    fn is_header_phi(&self, i: InstId) -> bool {
        matches!(
            self.body.insts.get(i).map(|inst| &inst.kind),
            Some(InstKind::Phi { .. })
        ) && self.body.block_insts(self.header).any(|(id, _)| id == i)
    }

    /// The integer constant `v` is (zero-extended), when it is one.
    fn int_const(&self, v: Value) -> Option<u64> {
        let Value::Const(c) = v else { return None };
        let constant = self.module.consts().get(c)?;
        if !matches!(self.module.types().get(constant.ty)?, Type::Int(_)) {
            return None;
        }
        match constant.value {
            Const::Int(k) => Some(k),
            Const::Null => Some(0),
            _ => None,
        }
    }
}

/// `pred` with its operands exchanged.
fn swapped(pred: IntPredicate) -> IntPredicate {
    match pred {
        IntPredicate::Eq => IntPredicate::Eq,
        IntPredicate::Ne => IntPredicate::Ne,
        IntPredicate::Ugt => IntPredicate::Ult,
        IntPredicate::Uge => IntPredicate::Ule,
        IntPredicate::Ult => IntPredicate::Ugt,
        IntPredicate::Ule => IntPredicate::Uge,
        IntPredicate::Sgt => IntPredicate::Slt,
        IntPredicate::Sge => IntPredicate::Sle,
        IntPredicate::Slt => IntPredicate::Sgt,
        IntPredicate::Sle => IntPredicate::Sge,
    }
}

/// The low `width` bits set.
fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// `v` sign-extended from `width` bits.
fn signed(v: u64, width: u32) -> i64 {
    let shift = 64 - width;
    ((v << shift) as i64) >> shift
}

/// `a + b` in `width` bits, and whether it wrapped as unsigned and as signed values.
fn add(a: u64, b: u64, width: u32) -> (u64, bool, bool) {
    let sum = a.wrapping_add(b) & mask(width);
    let unsigned = u128::from(a) + u128::from(b) > u128::from(mask(width));
    let wide = i128::from(signed(a, width)) + i128::from(signed(b, width));
    (sum, unsigned, wide != i128::from(signed(sum, width)))
}

/// `a - b` in `width` bits, and whether it wrapped as unsigned and as signed values.
fn sub(a: u64, b: u64, width: u32) -> (u64, bool, bool) {
    let difference = a.wrapping_sub(b) & mask(width);
    let wide = i128::from(signed(a, width)) - i128::from(signed(b, width));
    (
        difference,
        a < b,
        wide != i128::from(signed(difference, width)),
    )
}

/// `pred(a, b)` on `width`-bit values.
fn compare(pred: IntPredicate, a: u64, b: u64, width: u32) -> bool {
    let (sa, sb) = (signed(a, width), signed(b, width));
    match pred {
        IntPredicate::Eq => a == b,
        IntPredicate::Ne => a != b,
        IntPredicate::Ugt => a > b,
        IntPredicate::Uge => a >= b,
        IntPredicate::Ult => a < b,
        IntPredicate::Ule => a <= b,
        IntPredicate::Sgt => sa > sb,
        IntPredicate::Sge => sa >= sb,
        IntPredicate::Slt => sa < sb,
        IntPredicate::Sle => sa <= sb,
    }
}
