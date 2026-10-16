//! Contraction: the multiply-add pairs Metal fuses into one rounding.
//!
//! LLVM lets a compiler fuse a multiply into the addition or subtraction that uses it when both
//! carry the `contract` fast-math flag, and Metal's compiler does so by a rule the probe cases
//! measured on every lane (e761c0aa under fast math, c3cc7397 with `#pragma clang fp` flags only,
//! `float` and `half` alike): it fuses an addition with its multiply operand when
//!
//! - both carry `contract` (with `reassoc` or not; a pair either of which lacks it is two
//!   roundings, in a fast-math module too),
//! - they are in the same block (a product reaching the addition through a `select` or a `phi`
//!   is two roundings), and
//! - every use of the product is such an addition that fuses it: the product disappears. A
//!   product with any other use (a store, a `select`, a `phi`) is computed once and rounded,
//!   and no addition fuses it; one that two additions use is fused into both.
//!
//! An addition of two such products fuses its first operand's; `c - a * b` is
//! `fma(-a, b, c)` (or `fma(-b, a, c)`, the same value, where `a` is a sum and `b` is not: see
//! [`Fusion::a`]), `a * b - c` is `fma(a, b, -c)`, and the nested `(a * b + c) * d + e` is
//! `fma(fma(a, b, c), d, e)`, which the rule gives pair by pair. A `half` product widened to
//! `float` by an `fpext` whose only use is the addition is fused into the `float` addition,
//! `fma(float(a), float(b), c)`, a product that is exact in `float`. Vectors of two to four lanes
//! fuse lane-wise as scalars do.
//!
//! A fused pair is one correctly rounded fused multiply-add ([`super::fma`]) by the profile's
//! route; the multiply (and its `fpext`) has no value of its own. An addition the rule leaves
//! unfused whose operand is a `contract` multiply is `NoContraction`, so that no device fuses what
//! Metal keeps apart. Where the rule cannot be read from the measured shapes, the pair stays two
//! roundings and the plan notes `PrecisionNote::FastMathRewrites`: a product in another block than
//! its addition, a negation between them, an addition of a product to itself, a product left
//! unfused although every use of it is an addition (one of which chose another product), an
//! addition whose chosen product is kept while its other operand is a product, and the forms the
//! probe did not cover (`bfloat`, `double`, and chunked vectors).
//!
//! A tree of additions carrying `reassoc` as well is not fused pair by pair: Metal regroups it
//! (`chain`), and a fused addition there may be a chain of fused multiply-adds ([`Fusion::inner`]).

mod chain;

use super::fma::Fma;
use super::legal::{self, FloatFmt, Form};
use super::math::FloatFn;
use super::nan;
use super::{denormal, ArithStrategy};
use crate::air::intrinsics::{AirIntrinsic, LlvmFloatOp, LlvmIntrinsic, Math, MathFn};
use crate::air::ir::{
    BinOp as AirBin, Callee, CastOp, FastMath, Inst, InstId, InstKind, TypeId, Value, ValueId,
};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::select::SelectCx;
use crate::pipeline::artifact::PrecisionNote;
use std::collections::{BTreeMap, BTreeSet};

/// Which operand of a fused pair is negated: `c - a * b` negates the product, `a * b - c` the
/// addend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Negate {
    Neither,
    Product,
    Addend,
}

/// An addition fused with its product: `fma(a, b, c)`, with [`Negate`] applied, or, in a chain
/// (`chain`), `fma(a, b, fma(a1, b1, ... fma(an, bn, c)))`, its [`Fusion::inner`] links.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fusion {
    /// The folded multiply, or the folded `fpext` of a `half` multiply.
    pub product: InstId,
    /// The multiply's own flags.
    pub product_fmf: FastMath,
    /// The factors: the multiply's operands, in its order, except that `c - a * b` with `a` a
    /// sum (an addition, a subtraction, or a fused multiply-add) and `b` not one lists `b`
    /// first, so that the negated factor is not a sum. The value is the same, since a product's
    /// negation is exact on either factor; a compiler that ignores zeros' signs moves a
    /// negation into a sum, `-(x + y)` becoming `-x - y`, which makes the sum's `+0` a `+0`
    /// where the negation gives `-0`. MoltenVK's Metal compile of a module without
    /// `SignedZeroInfNanPreserve` is such a compiler: it answered `+0` for
    /// `fma(-fma(+0, 1, -0), 0.5, -0)`, where Metal answers `-0` (A1 case f9a12252, words 65
    /// and 81).
    pub a: Value,
    pub b: Value,
    /// The addend: the addition's other operand, or the innermost link's addend in a chain.
    pub c: Value,
    /// [`Negate::Product`] negates `a`; [`Negate::Addend`] negates `c`, the innermost addend.
    pub negate: Negate,
    /// A chain's fused multiply-adds inside the outer one, outermost first: each one's value is
    /// the addend of the one before, and the last one's addend is `c`. Empty for a pair.
    pub inner: Vec<Link>,
}

/// A fused multiply-add inside a chain ([`Fusion::inner`]): `fma(a, b, z)`, `a` negated where
/// `negated`, `z` the next link's value or the chain's addend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Link {
    /// The folded multiply.
    pub product: InstId,
    /// The multiply's own flags.
    pub product_fmf: FastMath,
    /// The factors, ordered as [`Fusion::a`]'s are.
    pub a: Value,
    pub b: Value,
    pub negated: bool,
}

/// A link of [`ArithStrategy::Fused`]: [`Link`] in value IDs, with the flushes of its factors
/// (bits 0 and 1) and of its value the module makes itself ([`denormal::Flush`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedLink {
    pub a: ValueId,
    pub b: ValueId,
    pub negate: bool,
    pub flush: denormal::Flush,
}

/// The contraction plan of a function.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Each fused addition with its product.
    pub fused: BTreeMap<InstId, Fusion>,
    /// The multiplies (and `fpext`s) fused into every use: no value of their own.
    pub folded: BTreeSet<InstId>,
    /// The additions the rule keeps apart from a `contract` multiply operand, and those a chain
    /// keeps as additions: `NoContraction`.
    pub kept: BTreeSet<InstId>,
    /// The additions of the trees Metal regroups (`chain`): what they regroup is Metal's
    /// measured rule, so their operands that are the tree's own are not noted.
    pub members: BTreeSet<InstId>,
}

impl Plan {
    /// Call `f` with each value instruction `at` reads where it stands under the plan, and say
    /// whether the plan decides that: a folded multiply (or its `fpext`) reads nothing, since
    /// it has no value of its own; a fused addition reads the two factors and the addend, which
    /// its `fma` computes from, and each link's factors. `false` for any other instruction, which reads its operands
    /// ([`crate::lower::select::FamilyPlans::reads`]).
    pub fn reads(&self, at: InstId, mut f: impl FnMut(Value)) -> bool {
        if self.folded.contains(&at) {
            return true;
        }
        match self.fused.get(&at) {
            Some(fusion) => {
                f(fusion.a);
                f(fusion.b);
                f(fusion.c);
                for link in &fusion.inner {
                    f(link.a);
                    f(link.b);
                }
                true
            }
            None => false,
        }
    }
}

/// A `contract` multiply `v` names, as an addition's operand would read it.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Product {
    /// The multiply, or the `fpext` widening it.
    at: InstId,
    /// The multiply itself.
    mul: InstId,
    fmf: FastMath,
    a: Value,
    b: Value,
}

/// How an addition operand reaches a `contract` multiply.
#[derive(Clone, Copy)]
enum Reach {
    /// The operand is the product (or its widening) in the addition's block: fusable.
    Direct(Product),
    /// Through a negation, from another block, or a widening the rule does not cover: the
    /// product's fusion there is not measured.
    Unmeasured,
}

/// Plan the function's contractions into `cx.families.arith.contract`, metering each walk.
pub fn plan(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    let body = cx.body;
    let n = body.insts.len();
    let mut block_of = vec![u32::MAX; n];
    let mut pos_of = vec![u32::MAX; n];
    for (b, block) in body.blocks.iter() {
        for (i, id) in block.inst_ids().enumerate() {
            m.tick(1)?;
            if let Some(slot) = block_of.get_mut(id.raw() as usize) {
                *slot = b.raw();
            }
            if let Some(slot) = pos_of.get_mut(id.raw() as usize) {
                *slot = u32::try_from(i).unwrap_or(u32::MAX);
            }
        }
    }
    // Every use of each instruction's result, and whether one is not a float addition.
    let mut uses = vec![0u32; n];
    let mut other = vec![false; n];
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        let addition = matches!(
            inst.kind,
            InstKind::BinOp {
                op: AirBin::FAdd | AirBin::FSub,
                ..
            }
        );
        inst.kind.for_each_value(|v| {
            if let Value::Inst(d) = v {
                let i = d.raw() as usize;
                if let Some(u) = uses.get_mut(i) {
                    *u = u.saturating_add(1);
                }
                if !addition {
                    if let Some(o) = other.get_mut(i) {
                        *o = true;
                    }
                }
            }
        });
    }
    let block = |i: InstId| block_of.get(i.raw() as usize).copied();

    // The trees Metal regroups, each planned whole; a tree the model does not cover takes the pair
    // rule below.
    let mut out = Plan::default();
    let facts = chain::Facts {
        block: &block_of,
        pos: &pos_of,
        uses: &uses,
        other: &other,
    };
    let mut leaves = Vec::new();
    for root in chain::roots(cx, &facts, m)? {
        let Some(tree) = chain::plan(cx, &facts, root, m)? else {
            continue;
        };
        let fmf = match body.insts.get(root).map(|i| &i.kind) {
            Some(InstKind::BinOp { fmf, .. }) => *fmf,
            _ => FastMath::NONE,
        };
        leaves.extend(tree.leaves.into_iter().map(|v| (fmf, v)));
        if tree.unmeasured {
            cx.precision.insert(PrecisionNote::FastMathRewrites);
        }
        out.members.extend(tree.members);
        out.fused.extend(tree.fused);
        out.kept.extend(tree.kept);
        out.folded.extend(tree.folded);
    }
    // A tree may still be regrouped with the operation defining a value it reads.
    for (fmf, v) in leaves {
        m.tick(1)?;
        super::note_rewrites(cx, None, (AirBin::FAdd, fmf), v, v);
    }

    // Each `contract` addition's product operands, and the one it chooses: its first direct one,
    // with the other operand as the addend and whether the product is the first operand.
    struct Addition {
        at: InstId,
        op: AirBin,
        twice: bool,
        reach: [Option<Reach>; 2],
        choice: Option<(Product, Value, bool)>,
        fusable: bool,
    }
    let mut additions = Vec::new();
    let mut chosen = vec![0u32; n];
    for (at, inst) in body.insts.iter() {
        m.tick(1)?;
        if out.members.contains(&at) {
            continue;
        }
        let InstKind::BinOp {
            op: op @ (AirBin::FAdd | AirBin::FSub),
            fmf,
            a,
            b,
            ..
        } = &inst.kind
        else {
            continue;
        };
        if !fmf.contains(FastMath::CONTRACT) {
            continue;
        }
        let fusable = fusable_form(cx, inst.ty).is_some();
        let reach = [*a, *b].map(|v| reach(cx, v, inst.ty, block(at), &uses, &block));
        if reach.iter().all(Option::is_none) {
            continue;
        }
        let choice = match reach {
            _ if a == b || !fusable => None,
            [Some(Reach::Direct(p)), _] => Some((p, *b, true)),
            [_, Some(Reach::Direct(p))] => Some((p, *a, false)),
            _ => None,
        };
        if let Some((p, _, _)) = choice {
            if let Some(c) = chosen.get_mut(p.at.raw() as usize) {
                *c = c.saturating_add(1);
            }
        }
        additions.push(Addition {
            at,
            op: *op,
            twice: a == b,
            reach,
            choice,
            fusable,
        });
    }

    // A product is fused when every use of it chose it.
    let fused_product = |p: InstId| {
        let i = p.raw() as usize;
        matches!((chosen.get(i), uses.get(i)), (Some(c), Some(u)) if *c > 0 && c == u)
    };
    let mut unmeasured = false;
    for add in &additions {
        m.tick(1)?;
        let picked = add.choice.filter(|(p, _, _)| fused_product(p.at));
        if let Some((p, c, first)) = picked {
            let negate = match (add.op, first) {
                (AirBin::FSub, true) => Negate::Addend,
                (AirBin::FSub, false) => Negate::Product,
                _ => Negate::Neither,
            };
            let (a, b) = match negate {
                Negate::Product if sum(cx, p.a) && !sum(cx, p.b) => (p.b, p.a),
                _ => (p.a, p.b),
            };
            out.fused.insert(
                add.at,
                Fusion {
                    product: p.at,
                    product_fmf: p.fmf,
                    a,
                    b,
                    c,
                    negate,
                    inner: Vec::new(),
                },
            );
            out.folded.insert(p.at);
            out.folded.insert(p.mul);
            continue;
        }
        out.kept.insert(add.at);
        // Kept apart: measured only where each product operand is direct and has a use that is
        // not an addition, which Metal computes and rounds (a store, a `select`, a `phi`).
        let measured = add.fusable
            && !add.twice
            && add.reach.iter().flatten().all(|r| match r {
                Reach::Direct(p) => other.get(p.at.raw() as usize).copied().unwrap_or(false),
                Reach::Unmeasured => false,
            });
        if !measured {
            unmeasured = true;
        }
    }
    if unmeasured {
        cx.precision.insert(PrecisionNote::FastMathRewrites);
    }
    cx.families.arith.contract = out;
    Ok(())
}

/// The strategy of instruction `at` when the plan fused or folded it: a fused addition is selected
/// as a written `fma` is, with both operations' fast-math flags.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
) -> Result<Option<ArithStrategy>, Error> {
    if cx.families.arith.contract.folded.contains(&at) {
        // The product may still be regrouped with its factors' operations.
        if let InstKind::BinOp {
            op: op @ AirBin::FMul,
            fmf,
            a,
            b,
            ..
        } = &inst.kind
        {
            super::note_rewrites(cx, Some(at), (*op, *fmf), *a, *b);
        }
        return Ok(Some(ArithStrategy::Folded));
    }
    let Some(fusion) = cx.families.arith.contract.fused.get(&at).cloned() else {
        return Ok(None);
    };
    let InstKind::BinOp { op, fmf, .. } = &inst.kind else {
        return Err(cx.invariant("a fused addition that is not a binary operator"));
    };
    // The fused pair is Metal's; the addition may still be regrouped with its addend's operation,
    // unless it is a tree's that Metal regroups by the measured rule (`chain`, whose plan notes the
    // values the tree reads).
    if !cx.families.arith.contract.members.contains(&at) {
        super::note_rewrites(cx, Some(at), (*op, *fmf), fusion.c, fusion.c);
    }
    let f = super::result_form(cx, inst)?;
    // A `half` output's last operation Metal computes at `float` precision: one `float` fused
    // multiply-add of the widened `half`s.
    let f = match cx.families.arith.unrounded.contains(&at) {
        true => f.with(legal::Held::Float(FloatFmt::F32)),
        false => f,
    };
    let fma = Fma::on(cx.profile, f)
        .ok_or_else(|| cx.invariant("a fused addition of a form without a fused multiply-add"))?;
    let mut reads = vec![fusion.a, fusion.b, fusion.c];
    reads.extend(fusion.inner.iter().flat_map(|l| [l.a, l.b]));
    super::check_operands(cx, &reads)?;
    cx.lowered_repr(Value::Inst(at), inst.ty)?;
    // `OpFmaKHR` is the device's at denormals and NaNs; the routines are Metal's. Its
    // denormals are Metal's where the device reads them so, else the module flushes them. Metal
    // answers a fused pair's NaN as a written `fma`'s (the probe `e98b103f`'s `a * b + c` lanes
    // under fast math, `float` and `half`): `OpFmaKHR` answers the device's `float` NaN and a
    // `half` result's rounding the device's `half` NaN on every route, so Metal's is selected.
    let khr = fma.route.device_float();
    // A chain's links flush their factors and values as the outer pair does; `OpFmaKHR` checked
    // ([`Fma::checked`]) answers no denormal, so only its factors and addend are read.
    let flush_of = |cx: &mut SelectCx<'_>, operands: &[Value]| {
        if khr {
            let flush = super::fused_flush(cx, f, operands);
            denormal::Flush {
                result: flush.result && !fma.checked(),
                ..flush
            }
        } else {
            denormal::Flush::NONE
        }
    };
    let flush = flush_of(cx, &[fusion.a, fusion.b, fusion.c]);
    let nan = match f.float_fmt() {
        Some(FloatFmt::Half) => nan::function_answer(FloatFn::Fma, false, false, FloatFmt::Half),
        _ => khr.then_some(nan::NanAnswer::Canonical),
    };
    super::fp_env(cx, *fmf, f);
    let factors = super::form(cx, fusion.a)?;
    super::fp_env(cx, fusion.product_fmf, factors);
    for link in &fusion.inner {
        super::fp_env(cx, link.product_fmf, factors);
    }
    cx.families.arith.take_fma(fma);
    let id = |cx: &SelectCx<'_>, v: Value| {
        cx.values
            .id(v)
            .ok_or_else(|| cx.invariant("a fused operand out of the value range"))
    };
    let mut inner = Vec::with_capacity(fusion.inner.len());
    for link in &fusion.inner {
        inner.push(FusedLink {
            a: id(cx, link.a)?,
            b: id(cx, link.b)?,
            negate: link.negated,
            flush: flush_of(cx, &[link.a, link.b]),
        });
    }
    let s = ArithStrategy::Fused {
        fma,
        a: id(cx, fusion.a)?,
        b: id(cx, fusion.b)?,
        c: id(cx, fusion.c)?,
        negate: fusion.negate,
        flush,
        inner: inner.into_boxed_slice(),
    };
    Ok(Some(super::with_nan(cx, at, s, nan, f)))
}

/// Whether `v` is a sum: the result of a float addition or subtraction (fused or not) or of a
/// fused multiply-add call (`llvm.fma`, `llvm.fmuladd`, `air.fma`), a value whose negation a
/// compiler that ignores zeros' signs may move into its operands ([`Fusion::a`]).
fn sum(cx: &SelectCx<'_>, v: Value) -> bool {
    let Value::Inst(id) = v else { return false };
    let Some(inst) = cx.body.insts.get(id) else {
        return false;
    };
    match &inst.kind {
        InstKind::BinOp {
            op: AirBin::FAdd | AirBin::FSub,
            ..
        } => true,
        InstKind::Call(call) => matches!(
            call.callee,
            Callee::Llvm {
                op: LlvmIntrinsic::Float {
                    op: LlvmFloatOp::Fma | LlvmFloatOp::FMulAdd,
                    ..
                },
                ..
            } | Callee::Air {
                op: AirIntrinsic::Math(Math { f: MathFn::Fma, .. }),
                ..
            }
        ),
        _ => false,
    }
}

/// The form of a fusable addition of AIR type `ty`: `float` or `half` lanes, two to four in a
/// vector (the forms the probe measured and [`Fma`] lowers).
fn fusable_form(cx: &SelectCx<'_>, ty: TypeId) -> Option<Form> {
    if legal::chunked(cx.entry.module.types(), ty).is_some() {
        return None;
    }
    let f = legal::form_on(cx.entry, cx.profile, ty).ok()?;
    match f.float_fmt()? {
        FloatFmt::F32 | FloatFmt::Half => Fma::on(cx.profile, f).map(|_| f),
        FloatFmt::BFloat | FloatFmt::F64 => None,
    }
}

/// How operand `v` of an addition of type `ty` in block `home` reaches a `contract` multiply, if
/// it does.
fn reach(
    cx: &SelectCx<'_>,
    v: Value,
    ty: TypeId,
    home: Option<u32>,
    uses: &[u32],
    block: &dyn Fn(InstId) -> Option<u32>,
) -> Option<Reach> {
    let body = cx.body;
    let mut v = v;
    let mut negated = false;
    loop {
        let Value::Inst(id) = v else { return None };
        let inst = body.insts.get(id)?;
        match &inst.kind {
            InstKind::FNeg { v: x, .. } => {
                negated = true;
                v = *x;
            }
            InstKind::BinOp {
                op: AirBin::FMul,
                fmf,
                a,
                b,
                ..
            } if fmf.contains(FastMath::CONTRACT) => {
                let direct = !negated && block(id) == home && inst.ty == ty;
                return Some(if direct {
                    Reach::Direct(Product {
                        at: id,
                        mul: id,
                        fmf: *fmf,
                        a: *a,
                        b: *b,
                    })
                } else {
                    Reach::Unmeasured
                });
            }
            InstKind::Cast {
                op: CastOp::FPExt,
                v: Value::Inst(q),
            } => {
                let mul = body.insts.get(*q)?;
                let InstKind::BinOp {
                    op: AirBin::FMul,
                    fmf,
                    a,
                    b,
                    ..
                } = &mul.kind
                else {
                    return None;
                };
                if !fmf.contains(FastMath::CONTRACT) {
                    return None;
                }
                // A `half` product widened to `float` in the addition's block, the widening the
                // product's only use (a probe case e761c0aa, word 63).
                let widened = fusable_form(cx, mul.ty)
                    .is_some_and(|f| f.float_fmt() == Some(FloatFmt::Half))
                    && fusable_form(cx, inst.ty)
                        .is_some_and(|f| f.float_fmt() == Some(FloatFmt::F32));
                let direct = !negated
                    && widened
                    && inst.ty == ty
                    && block(id) == home
                    && block(*q) == home
                    && uses.get(q.raw() as usize) == Some(&1);
                return Some(if direct {
                    Reach::Direct(Product {
                        at: id,
                        mul: *q,
                        fmf: *fmf,
                        a: *a,
                        b: *b,
                    })
                } else {
                    Reach::Unmeasured
                });
            }
            _ => return None,
        }
    }
}
