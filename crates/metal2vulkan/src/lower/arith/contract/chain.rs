//! Chains: how Metal fuses a tree of `contract` and `reassoc` additions over `contract` products.
//!
//! An addition whose operand is another such addition, used by it alone, in its block and of its
//! type, forms a tree with it; the tree's leaves are its products (each used by it alone) and the
//! other values it reads. Metal does not fuse such a tree pair by pair, as [`super::plan`]'s rule
//! for one addition does: it regroups it as LLVM's DAG combiner forms fused multiply-adds where
//! reassociation is allowed, with one rule of its own. The model below gives Metal's word on every
//! thread of every scalar sum and every lane of the vector sum of the owned probe
//! `engine/unm_contract_chain` (`float` and `half`, the Apple M4 and M3 Max):
//!
//! - an addition one of whose operands is a product fuses it, the first operand's where both are,
//!   with the other operand as the addend; a subtraction negates the product or the addend as the
//!   pair rule does;
//! - an addition of a fused multiply-add chain and a value `z` moves `z` into the chain's innermost
//!   addend where that addend is a product: `fma(a, b, c * d) + z` is `fma(a, b, fma(c, d, z))`,
//!   and a subtraction moves `-z`; `z - fma(a, b, c * d)` is `fma(-a, b, fma(-c, d, z))`;
//! - Metal's own: an addition `(p + q) + z` of a sum of two products and an addend that reads
//!   memory ([`regroups_into`]) is `fma(q, fma(p, z))`, `p`'s fused first, and a subtraction
//!   `z - (p + q)` of such a sum is `fma(-q, fma(-p, z))`. An addend computed without reading
//!   memory is not regrouped: `(p + q) + z` is `fma(p, fma(q, z))`.
//!
//! The probe measured the regrouping on loads, vectors of loads, and `abs`, `min`, `fma` and
//! the conversions between `float` and `half` of loads, and its absence on a conversion of an
//! integer the thread computes. Where a tree's regrouping turns on an addend of another kind,
//! the model takes the rule as stated and the plan notes `PrecisionNote::FastMathRewrites`.
//!
//! The model visits the tree's additions from the last (an addition it makes before any of the
//! tree's own), applies the first rule that holds, and starts again until none does. A rule at an
//! addition reads only what is below it, so after a rewrite only the additions above it are
//! visited again ([`Model::touch`]); the visit order is the same as a fresh start's. The result is
//! placed back on the tree's own additions ([`Assigned`]): each fused chain on the addition whose
//! value it is, each addition the result keeps as an addition on itself, and every other addition
//! of the tree folded into a chain. Where the result cannot be placed so, or the tree holds what
//! the probe did not measure (a product another addition also uses, one reached through a
//! negation, from another block or through a widening), the tree takes the pair rule.

use super::{Fusion, Link, Negate, Product, Reach};
use crate::air::intrinsics::{AirIntrinsic, Math, MathFn, NumKind};
use crate::air::ir::{BinOp as AirBin, Callee, CastOp, FastMath, InstId, InstKind, Value};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::select::SelectCx;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// No node: the root's parent.
const NONE: usize = usize::MAX;

/// The most additions one tree may hold; a larger tree takes the pair rule. It bounds the plan's
/// walks: the model's work is about linear in the tree, and its placement and the walks above a
/// rewrite are linear in the tree's depth.
const MOST: usize = 4096;

/// A node of a tree's model.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// `x + y`, or `x - y` where `sub`; `at` is the tree's addition it stands for, if it is one.
    Add {
        x: usize,
        y: usize,
        sub: bool,
        at: Option<InstId>,
    },
    /// A product of the tree.
    Mul { p: Product },
    /// `fma(a, b, c)` of product `p`'s factors, `a` negated where `neg`.
    Fma { p: Product, neg: bool, c: usize },
    /// A value the tree reads.
    Leaf { v: Value },
    /// `-x`.
    Neg { x: usize },
}

/// What the plan does with each of a tree's additions.
pub(super) struct Assigned {
    /// The tree's additions.
    pub members: Vec<InstId>,
    /// The additions fused, each with its chain.
    pub fused: BTreeMap<InstId, Fusion>,
    /// The additions kept as additions.
    pub kept: Vec<InstId>,
    /// The additions and products with no value of their own.
    pub folded: Vec<InstId>,
    /// The values the tree reads other than its products.
    pub leaves: Vec<Value>,
    /// Whether the regrouping turned on an addend of a kind the probe did not measure.
    pub unmeasured: bool,
}

/// The facts of the function [`super::plan`] gathered, by instruction: its block, its position
/// in it, its result's uses, and whether one of them is not a float addition.
pub(super) struct Facts<'a> {
    pub block: &'a [u32],
    pub pos: &'a [u32],
    pub uses: &'a [u32],
    pub other: &'a [bool],
}

impl Facts<'_> {
    fn block(&self, i: InstId) -> Option<u32> {
        self.block.get(i.raw() as usize).copied()
    }

    fn pos(&self, i: InstId) -> u32 {
        self.pos.get(i.raw() as usize).copied().unwrap_or(u32::MAX)
    }

    fn uses(&self, i: InstId) -> u32 {
        self.uses.get(i.raw() as usize).copied().unwrap_or(0)
    }

    fn other(&self, i: InstId) -> bool {
        self.other.get(i.raw() as usize).copied().unwrap_or(true)
    }
}

/// Whether `at` is an addition a tree may hold: a `float` or `half` addition or subtraction of a
/// fusable form carrying `contract` and `reassoc`, of two distinct operands.
fn member(cx: &SelectCx<'_>, at: InstId) -> bool {
    let Some(inst) = cx.body.insts.get(at) else {
        return false;
    };
    let InstKind::BinOp {
        op: AirBin::FAdd | AirBin::FSub,
        fmf,
        a,
        b,
        ..
    } = &inst.kind
    else {
        return false;
    };
    fmf.contains(FastMath::CONTRACT.union(FastMath::REASSOC))
        && a != b
        && super::fusable_form(cx, inst.ty).is_some()
}

/// The addition operand `v` of the tree's addition `parent` names, if it is one of the tree's: a
/// member it alone uses, in its block and of its type.
fn inner(cx: &SelectCx<'_>, facts: &Facts<'_>, parent: InstId, v: Value) -> Option<InstId> {
    let Value::Inst(id) = v else { return None };
    let (Some(p), Some(i)) = (cx.body.insts.get(parent), cx.body.insts.get(id)) else {
        return None;
    };
    (member(cx, id)
        && facts.uses(id) == 1
        && facts.block(id) == facts.block(parent)
        && i.ty == p.ty)
        .then_some(id)
}

/// The operands of binary operator `at`.
fn operands(cx: &SelectCx<'_>, at: InstId) -> Option<(AirBin, Value, Value)> {
    match &cx.body.insts.get(at)?.kind {
        InstKind::BinOp { op, a, b, .. } => Some((*op, *a, *b)),
        _ => None,
    }
}

/// The additions that head a tree of at least two: members with a member operand, which no
/// member uses as its operand.
pub(super) fn roots(
    cx: &SelectCx<'_>,
    facts: &Facts<'_>,
    m: &mut Meter,
) -> Result<Vec<InstId>, Error> {
    let mut held = BTreeSet::new();
    let mut heads = Vec::new();
    for (at, _) in cx.body.insts.iter() {
        m.tick(1)?;
        if !member(cx, at) {
            continue;
        }
        let Some((_, a, b)) = operands(cx, at) else {
            continue;
        };
        let mut any = false;
        for v in [a, b] {
            if let Some(id) = inner(cx, facts, at, v) {
                held.insert(id);
                any = true;
            }
        }
        if any {
            heads.push(at);
        }
    }
    Ok(heads.into_iter().filter(|at| !held.contains(at)).collect())
}

/// Regroup the tree `root` heads as Metal does, or `None` where the model does not cover it or
/// the tree has no product.
pub(super) fn plan(
    cx: &SelectCx<'_>,
    facts: &Facts<'_>,
    root: InstId,
    m: &mut Meter,
) -> Result<Option<Assigned>, Error> {
    // The tree's additions, operands before their users.
    let mut members = Vec::new();
    let mut stack = vec![root];
    while let Some(at) = stack.pop() {
        m.tick(1)?;
        if members.len() >= MOST {
            return Ok(None);
        }
        members.push(at);
        let Some((_, a, b)) = operands(cx, at) else {
            return Ok(None);
        };
        stack.extend([a, b].into_iter().filter_map(|v| inner(cx, facts, at, v)));
    }
    members.sort_by_key(|&at| facts.pos(at));
    let set: BTreeSet<InstId> = members.iter().copied().collect();

    let mut model = Model::default();
    let mut node_of = BTreeMap::new();
    let mut leaves = Vec::new();
    let mut products = 0usize;
    for &at in &members {
        m.tick(1)?;
        let (Some((op, a, b)), Some(inst)) = (operands(cx, at), cx.body.insts.get(at)) else {
            return Ok(None);
        };
        let mut xy = [NONE; 2];
        for (slot, v) in xy.iter_mut().zip([a, b]) {
            if let Some(id) = inner(cx, facts, at, v) {
                // An operand addition precedes its user, so its node is made.
                let Some(n) = node_of.get(&id) else {
                    return Ok(None);
                };
                *slot = *n;
                continue;
            }
            let block = |i: InstId| facts.block(i);
            *slot = match super::reach(cx, v, inst.ty, facts.block(at), facts.uses, &block) {
                Some(Reach::Direct(p)) if p.at == p.mul && facts.uses(p.at) == 1 => {
                    products += 1;
                    model.make(Kind::Mul { p }, None)
                }
                // A product Metal computes and rounds, since it has a use that is not an
                // addition (a store, a `select`, a `phi`): a value the tree reads.
                Some(Reach::Direct(p)) if p.at == p.mul && facts.other(p.at) => {
                    leaves.push(v);
                    model.make(Kind::Leaf { v }, None)
                }
                Some(_) => return Ok(None),
                None => {
                    leaves.push(v);
                    model.make(Kind::Leaf { v }, None)
                }
            };
        }
        if xy.contains(&NONE) {
            return Ok(None);
        }
        let n = model.make(
            Kind::Add {
                x: xy[0],
                y: xy[1],
                sub: op == AirBin::FSub,
                at: Some(at),
            },
            Some(u64::from(facts.pos(at))),
        );
        node_of.insert(at, n);
    }
    let Some(&top) = node_of.get(&root) else {
        return Ok(None);
    };
    if products == 0 {
        return Ok(None);
    }
    model.root = top;
    model.next = u64::from(u32::MAX);

    let regroup = |v: Value| regroups_into(cx, v);
    if !model.settle(&regroup, members.len(), &mut || Ok(m.tick(1)?))? {
        return Ok(None);
    }

    let mut out = Assigned {
        members: members.clone(),
        fused: BTreeMap::new(),
        kept: Vec::new(),
        folded: Vec::new(),
        leaves,
        unmeasured: model.unmeasured,
    };
    if !model.place(cx, facts, &set, root, &mut out, m)? {
        return Ok(None);
    }
    for at in members {
        if !out.fused.contains_key(&at) && !out.kept.contains(&at) {
            out.folded.push(at);
        }
    }
    Ok(Some(out))
}

/// A node of a tree's model.
struct Node {
    kind: Kind,
    /// The visiting order: the AIR position of the tree's own additions, later for those the model
    /// makes.
    order: u64,
    parent: usize,
    /// Toward the outermost fused multiply-add of the chain a fused multiply-add is in
    /// ([`Model::top`]).
    up: usize,
    /// Replaced, or folded into what replaced it.
    dead: bool,
    queued: bool,
}

/// A tree's model: its nodes, the additions to visit, and the order the next node the model makes
/// takes.
#[derive(Default)]
struct Model {
    nodes: Vec<Node>,
    queue: BinaryHeap<(u64, usize)>,
    root: usize,
    next: u64,
    /// Whether a rule turned on an addend of a kind the probe did not measure.
    unmeasured: bool,
}

impl Model {
    /// Apply the rules until none holds, each addition visited from the last; whether the model
    /// settled within two rewrites per addition of the tree's `additions` (each rewrite turns an
    /// addition into a fused multiply-add or, by Metal's own rule, into an addition the next
    /// rewrite fuses). `tick` meters each visit.
    fn settle(
        &mut self,
        regroup: &dyn Fn(Value) -> Regroup,
        additions: usize,
        tick: &mut dyn FnMut() -> Result<(), Error>,
    ) -> Result<bool, Error> {
        let mut rewrites = 0usize;
        while let Some((_, n)) = self.queue.pop() {
            tick()?;
            if let Some(node) = self.nodes.get_mut(n) {
                node.queued = false;
            }
            if self.dead(n) || !self.visit(regroup, n) {
                continue;
            }
            rewrites += 1;
            if rewrites > 2 * additions + 2 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// A node of kind `kind`, the parent of the nodes it reads; an addition is queued.
    fn make(&mut self, kind: Kind, order: Option<u64>) -> usize {
        let n = self.nodes.len();
        let order = order.unwrap_or_else(|| {
            self.next += 1;
            self.next
        });
        self.nodes.push(Node {
            kind,
            order,
            parent: NONE,
            up: n,
            dead: false,
            queued: false,
        });
        for c in children(kind) {
            self.set_parent(c, n);
        }
        if matches!(kind, Kind::Add { .. }) {
            self.enqueue(n);
        }
        n
    }

    fn kind(&self, n: usize) -> Option<Kind> {
        self.nodes.get(n).map(|x| x.kind)
    }

    fn parent(&self, n: usize) -> usize {
        self.nodes.get(n).map_or(NONE, |x| x.parent)
    }

    fn set_parent(&mut self, n: usize, p: usize) {
        if let Some(x) = self.nodes.get_mut(n) {
            x.parent = p;
        }
    }

    fn up(&self, n: usize) -> usize {
        self.nodes.get(n).map_or(n, |x| x.up)
    }

    fn dead(&self, n: usize) -> bool {
        self.nodes.get(n).is_none_or(|x| x.dead)
    }

    fn kill(&mut self, n: usize) {
        if let Some(x) = self.nodes.get_mut(n) {
            x.dead = true;
        }
    }

    fn enqueue(&mut self, n: usize) {
        if let Some(x) = self.nodes.get_mut(n) {
            if !x.queued && !x.dead {
                x.queued = true;
                self.queue.push((x.order, n));
            }
        }
    }

    fn is_fma(&self, n: usize) -> bool {
        matches!(self.kind(n), Some(Kind::Fma { .. }))
    }

    fn is_add(&self, n: usize) -> bool {
        matches!(self.kind(n), Some(Kind::Add { .. }))
    }

    /// Put `new` where `old` is read, and queue the additions whose rules read what changed.
    fn replace(&mut self, old: usize, new: usize) {
        let p = self.parent(old);
        self.kill(old);
        self.set_parent(new, p);
        match self.nodes.get_mut(p).map(|x| &mut x.kind) {
            None => self.root = new,
            Some(kind) => {
                let swap = |c: &mut usize| {
                    if *c == old {
                        *c = new;
                    }
                };
                match kind {
                    Kind::Add { x, y, .. } => {
                        swap(x);
                        swap(y);
                    }
                    Kind::Fma { c, .. } => swap(c),
                    Kind::Neg { x } => swap(x),
                    Kind::Mul { .. } | Kind::Leaf { .. } => {}
                }
            }
        }
        self.touch(new);
    }

    /// The outermost fused multiply-add of the chain `f` is in: up through the fused
    /// multiply-adds whose addend it is.
    fn top(&mut self, f: usize) -> usize {
        let mut t = f;
        // Each step goes strictly up the tree, so the walk ends within the tree's depth.
        for _ in 0..self.nodes.len() {
            let (u, p) = (self.up(t), self.parent(t));
            if u != t {
                t = u;
            } else if self.is_fma(p) {
                t = p;
            } else {
                break;
            }
        }
        // A chain only grows, at either end, so every node on the way keeps `t` above it.
        let mut s = f;
        for _ in 0..self.nodes.len() {
            if s == t {
                break;
            }
            let u = self.up(s);
            let next = if u != s { u } else { self.parent(s) };
            if let Some(x) = self.nodes.get_mut(s) {
                x.up = t;
            }
            s = next;
        }
        t
    }

    /// Queue the additions whose rules read node `m`, which changed: a rule reads its operands,
    /// an operand addition's operands, and the innermost addend of an operand chain.
    fn touch(&mut self, m: usize) {
        let p = self.parent(m);
        let s = if self.is_fma(p) { self.top(p) } else { m };
        let a = self.parent(s);
        if self.is_add(a) {
            self.enqueue(a);
            let g = self.parent(a);
            if self.is_add(g) {
                self.enqueue(g);
            }
        }
    }

    /// The fused multiply-add of chain `f` whose addend is a product, if the chain ends so.
    fn bottom(&self, f: usize) -> Option<usize> {
        let mut t = f;
        while let Some(Kind::Fma { c, .. }) = self.kind(t) {
            match self.kind(c) {
                Some(Kind::Mul { .. }) => return Some(t),
                Some(Kind::Fma { .. }) => t = c,
                _ => return None,
            }
        }
        None
    }

    /// Fuse the product addend of chain node `t` with `e`.
    fn sink(&mut self, t: usize, e: usize) {
        let Some(Kind::Fma { c, .. }) = self.kind(t) else {
            return;
        };
        let Some(Kind::Mul { p }) = self.kind(c) else {
            return;
        };
        let f = self.make(
            Kind::Fma {
                p,
                neg: false,
                c: e,
            },
            None,
        );
        self.kill(c);
        if let Some(Kind::Fma { c: slot, .. }) = self.nodes.get_mut(t).map(|x| &mut x.kind) {
            *slot = f;
        }
        self.set_parent(f, t);
    }

    /// Apply the first rule that holds at addition `n`; whether one did.
    fn visit(&mut self, regroup: &dyn Fn(Value) -> Regroup, n: usize) -> bool {
        let Some(Kind::Add { x, y, sub, .. }) = self.kind(n) else {
            return false;
        };
        if !sub {
            // Metal's own: `(p + q) + z` is `fma(q, fma(p, z))`.
            if let Some(Kind::Add {
                x: px,
                y: py,
                sub: false,
                ..
            }) = self.kind(x)
            {
                if let (Some(Kind::Mul { p }), Some(Kind::Mul { .. })) =
                    (self.kind(px), self.kind(py))
                {
                    let r = self.regroups(regroup, y);
                    self.unmeasured |= !r.measured;
                    if r.regroups {
                        self.kill(x);
                        self.kill(px);
                        let f = self.make(
                            Kind::Fma {
                                p,
                                neg: false,
                                c: y,
                            },
                            None,
                        );
                        let a = self.make(
                            Kind::Add {
                                x: f,
                                y: py,
                                sub: false,
                                at: None,
                            },
                            None,
                        );
                        self.replace(n, a);
                        return true;
                    }
                }
            }
            if let Some(Kind::Mul { p }) = self.kind(x) {
                self.kill(x);
                let f = self.make(
                    Kind::Fma {
                        p,
                        neg: false,
                        c: y,
                    },
                    None,
                );
                self.replace(n, f);
                return true;
            }
            if let Some(Kind::Mul { p }) = self.kind(y) {
                self.kill(y);
                let f = self.make(
                    Kind::Fma {
                        p,
                        neg: false,
                        c: x,
                    },
                    None,
                );
                self.replace(n, f);
                return true;
            }
            if self.is_fma(x) {
                if let Some(t) = self.bottom(x) {
                    self.sink(t, y);
                    self.replace(n, x);
                    return true;
                }
            } else if self.is_fma(y) {
                if let Some(t) = self.bottom(y) {
                    self.sink(t, x);
                    self.replace(n, y);
                    return true;
                }
            }
            return false;
        }
        // Metal's own: `z - (p + q)` is `fma(-q, fma(-p, z))`.
        if let Some(Kind::Add {
            x: px,
            y: py,
            sub: false,
            ..
        }) = self.kind(y)
        {
            if let (Some(Kind::Mul { p }), Some(Kind::Mul { .. })) = (self.kind(px), self.kind(py))
            {
                let r = self.regroups(regroup, x);
                // Measured on a loaded addend alone.
                self.unmeasured |= !r.load;
                if r.regroups {
                    self.kill(y);
                    self.kill(px);
                    let f = self.make(Kind::Fma { p, neg: true, c: x }, None);
                    let s = self.make(
                        Kind::Add {
                            x: f,
                            y: py,
                            sub: true,
                            at: None,
                        },
                        None,
                    );
                    self.replace(n, s);
                    return true;
                }
            }
        }
        if let Some(Kind::Mul { p }) = self.kind(x) {
            self.kill(x);
            let negated = self.make(Kind::Neg { x: y }, None);
            let f = self.make(
                Kind::Fma {
                    p,
                    neg: false,
                    c: negated,
                },
                None,
            );
            self.replace(n, f);
            return true;
        }
        if let Some(Kind::Mul { p }) = self.kind(y) {
            self.kill(y);
            let f = self.make(Kind::Fma { p, neg: true, c: x }, None);
            self.replace(n, f);
            return true;
        }
        if self.is_fma(x) {
            if let Some(t) = self.bottom(x) {
                let negated = self.make(Kind::Neg { x: y }, None);
                self.sink(t, negated);
                self.replace(n, x);
                return true;
            }
        }
        // `z - fma(a, b, c * d)` is `fma(-a, b, fma(-c, d, z))`.
        if let Some(Kind::Fma { p, neg, c }) = self.kind(y) {
            if let Some(Kind::Mul { p: q }) = self.kind(c) {
                self.kill(y);
                self.kill(c);
                let inner = self.make(
                    Kind::Fma {
                        p: q,
                        neg: true,
                        c: x,
                    },
                    None,
                );
                let f = self.make(
                    Kind::Fma {
                        p,
                        neg: !neg,
                        c: inner,
                    },
                    None,
                );
                self.replace(n, f);
                return true;
            }
        }
        false
    }

    /// Whether node `z` is an addend Metal regroups into the first of two products before the
    /// second ([`regroups_into`]).
    fn regroups(&self, regroup: &dyn Fn(Value) -> Regroup, z: usize) -> Regroup {
        match self.kind(z) {
            Some(Kind::Leaf { v }) => regroup(v),
            // A product the tree fuses (the probe's `a * b + c * d + e * f`).
            Some(Kind::Mul { .. }) => Regroup {
                regroups: false,
                measured: true,
                load: false,
            },
            _ => Regroup {
                regroups: false,
                measured: false,
                load: false,
            },
        }
    }

    /// Place the root's value on addition `root`, and what it reads on the tree's other
    /// additions: a chain on the addition whose value it is, an addition on itself, whose
    /// operands are its AIR operands or placed in turn. Whether every node could be.
    fn place(
        &self,
        cx: &SelectCx<'_>,
        facts: &Facts<'_>,
        members: &BTreeSet<InstId>,
        root: InstId,
        out: &mut Assigned,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        let mut work = vec![(self.root, root)];
        while let Some((n, slot)) = work.pop() {
            m.tick(1)?;
            if out.fused.contains_key(&slot) || out.kept.contains(&slot) {
                return Ok(false);
            }
            match self.kind(n) {
                Some(Kind::Fma { .. }) => {
                    let Some(fusion) = self.chain(cx, facts, n, slot, &mut work, out) else {
                        return Ok(false);
                    };
                    out.fused.insert(slot, fusion);
                }
                Some(Kind::Add {
                    x,
                    y,
                    sub,
                    at: Some(at),
                }) if at == slot => {
                    let Some((op, a, b)) = operands(cx, at) else {
                        return Ok(false);
                    };
                    if (op == AirBin::FSub) != sub {
                        return Ok(false);
                    }
                    for (node, v) in [(x, a), (y, b)] {
                        match v {
                            Value::Inst(id) if members.contains(&id) => work.push((node, id)),
                            _ => {
                                let same = match self.kind(node) {
                                    Some(Kind::Leaf { v: w }) => w == v,
                                    Some(Kind::Mul { p }) => Value::Inst(p.at) == v,
                                    _ => false,
                                };
                                if !same {
                                    return Ok(false);
                                }
                            }
                        }
                    }
                    out.kept.push(at);
                }
                _ => return Ok(false),
            }
        }
        Ok(true)
    }

    /// The fusion of chain `n` placed on addition `slot`, queueing a kept addition its innermost
    /// addend is on `work`; `None` where the chain cannot be placed there.
    fn chain(
        &self,
        cx: &SelectCx<'_>,
        facts: &Facts<'_>,
        n: usize,
        slot: InstId,
        work: &mut Vec<(usize, InstId)>,
        out: &mut Assigned,
    ) -> Option<Fusion> {
        let mut links = Vec::new();
        let mut t = n;
        while let Some(Kind::Fma { p, neg, c }) = self.kind(t) {
            links.push((p, neg));
            t = c;
        }
        let (base, negated) = match self.kind(t) {
            Some(Kind::Neg { x }) => (x, true),
            _ => (t, false),
        };
        let c = match self.kind(base)? {
            Kind::Leaf { v } => v,
            Kind::Mul { p } => Value::Inst(p.at),
            Kind::Add { at: Some(at), .. } => {
                work.push((base, at));
                Value::Inst(at)
            }
            _ => return None,
        };
        let (&(outer, outer_neg), rest) = links.split_first()?;
        let negate = match (outer_neg, negated) {
            (false, false) => Negate::Neither,
            (true, false) => Negate::Product,
            (false, true) => Negate::Addend,
            (true, true) => return None,
        };
        // A negated factor is not a sum where the other is not ([`Fusion::a`]).
        let factors = |p: Product, neg: bool| {
            if neg && super::sum(cx, p.a) && !super::sum(cx, p.b) {
                (p.b, p.a)
            } else {
                (p.a, p.b)
            }
        };
        let inner: Vec<Link> = rest
            .iter()
            .map(|&(p, neg)| {
                let (a, b) = factors(p, neg);
                Link {
                    product: p.at,
                    product_fmf: p.fmf,
                    a,
                    b,
                    negated: neg,
                }
            })
            .collect();
        let (a, b) = factors(outer, negate == Negate::Product);
        // Every value the chain reads is defined before `slot` (an operand of the tree's
        // additions under it, so this holds by construction; checked all the same).
        let before = |v: Value| match v {
            Value::Inst(i) => facts.block(i) != facts.block(slot) || facts.pos(i) < facts.pos(slot),
            _ => true,
        };
        let reads_before =
            before(c) && before(a) && before(b) && inner.iter().all(|l| before(l.a) && before(l.b));
        if !reads_before {
            return None;
        }
        out.folded.extend(links.iter().map(|(p, _)| p.at));
        Some(Fusion {
            product: outer.at,
            product_fmf: outer.fmf,
            a,
            b,
            c,
            negate,
            inner,
        })
    }
}

/// The nodes `kind` reads.
fn children(kind: Kind) -> impl Iterator<Item = usize> {
    let (a, b) = match kind {
        Kind::Add { x, y, .. } => (Some(x), Some(y)),
        Kind::Fma { c, .. } => (Some(c), None),
        Kind::Neg { x } => (Some(x), None),
        Kind::Mul { .. } | Kind::Leaf { .. } => (None, None),
    };
    a.into_iter().chain(b)
}

/// Whether Metal regroups an addend into the first of two products before the second, and
/// whether the probe measured an addend of its kind.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Regroup {
    regroups: bool,
    measured: bool,
    /// Whether the addend is a load, the one addend a subtraction's regrouping was measured on.
    load: bool,
}

/// How Metal treats addend `z` of `(p + q) + z`: it regroups one that reads memory into `p`'s
/// fused multiply-add first, `fma(q, fma(p, z))`. Measured: a load, a vector of loads, and `abs`,
/// `min`, `fma`, or a conversion between `float` and `half` of loads, each regrouped; a
/// conversion of an integer computed without reading memory (the thread's position, masked and
/// offset), not regrouped. Any other addend is taken by whether it reads memory, unmeasured.
fn regroups_into(cx: &SelectCx<'_>, z: Value) -> Regroup {
    if loaded(cx, z, 0) {
        return Regroup {
            regroups: true,
            measured: true,
            load: load(cx, z),
        };
    }
    if computed_integer(cx, z) {
        return Regroup {
            regroups: false,
            measured: true,
            load: false,
        };
    }
    let mut budget = WALK;
    Regroup {
        regroups: reads_memory(cx, z, &mut budget),
        measured: false,
        load: false,
    }
}

/// The most instructions a walk of an addend's operands visits.
const WALK: u32 = 64;

fn load(cx: &SelectCx<'_>, v: Value) -> bool {
    let Value::Inst(id) = v else { return false };
    matches!(
        cx.body.insts.get(id).map(|i| &i.kind),
        Some(InstKind::Load { .. })
    )
}

/// Whether `v` is of a loaded kind the probe measured: a load; a vector of loads built lane by
/// lane; `abs`, `min` or `fma` of loads; a conversion between `float` and `half` of a load.
fn loaded(cx: &SelectCx<'_>, v: Value, depth: u32) -> bool {
    let Value::Inst(id) = v else { return false };
    let Some(inst) = cx.body.insts.get(id) else {
        return false;
    };
    match &inst.kind {
        InstKind::Load { .. } => true,
        InstKind::InsertElement { vec, elt, .. } => {
            load(cx, *elt)
                && depth < 16
                && (matches!(vec, Value::Const(_)) || loaded(cx, *vec, depth + 1))
        }
        InstKind::Cast {
            op: CastOp::FPExt | CastOp::FPTrunc,
            v,
        } => load(cx, *v),
        InstKind::Call(call) => {
            matches!(
                call.callee,
                Callee::Air {
                    op: AirIntrinsic::Math(Math {
                        f: MathFn::Fabs | MathFn::FMin | MathFn::Fma,
                        ..
                    }),
                    ..
                }
            ) && call.args.iter().all(|a| load(cx, *a))
        }
        _ => false,
    }
}

/// Whether `v` converts to a float an integer computed from the function's arguments and
/// constants by integer arithmetic alone, the kind the probe measured (`float((i & 15) - 7)`).
fn computed_integer(cx: &SelectCx<'_>, v: Value) -> bool {
    let Value::Inst(id) = v else { return false };
    let Some(inst) = cx.body.insts.get(id) else {
        return false;
    };
    let x = match &inst.kind {
        InstKind::Cast {
            op: CastOp::SIToFP | CastOp::UIToFP,
            v,
        } => *v,
        InstKind::Call(call) => match (&call.callee, call.args.first()) {
            (
                Callee::Air {
                    op: AirIntrinsic::Convert(c),
                    ..
                },
                Some(x),
            ) if c.dst_kind == NumKind::Float
                && matches!(c.src_kind, NumKind::Signed | NumKind::Unsigned)
                && c.src_fp8.is_none()
                && call.args.len() == 1 =>
            {
                *x
            }
            _ => return false,
        },
        _ => return false,
    };
    let mut stack = vec![x];
    let mut budget = WALK;
    while let Some(v) = stack.pop() {
        let Some(left) = budget.checked_sub(1) else {
            return false;
        };
        budget = left;
        let Value::Inst(id) = v else { continue };
        let Some(inst) = cx.body.insts.get(id) else {
            return false;
        };
        match &inst.kind {
            InstKind::BinOp { op, a, b, .. } if !op.is_float() => stack.extend([*a, *b]),
            InstKind::Cast {
                op: CastOp::Trunc | CastOp::ZExt | CastOp::SExt,
                v,
            } => stack.push(*v),
            _ => return false,
        }
    }
    true
}

/// Whether `v` reads memory: a load or a call other than a math function or a conversion
/// among its operands, within `budget` instructions (none past it).
fn reads_memory(cx: &SelectCx<'_>, v: Value, budget: &mut u32) -> bool {
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        let Some(left) = budget.checked_sub(1) else {
            return false;
        };
        *budget = left;
        let Value::Inst(id) = v else { continue };
        let Some(inst) = cx.body.insts.get(id) else {
            continue;
        };
        match &inst.kind {
            InstKind::Load { .. } => return true,
            InstKind::Call(call)
                if !matches!(
                    call.callee,
                    Callee::Air {
                        op: AirIntrinsic::Math(_) | AirIntrinsic::Convert(_),
                        ..
                    } | Callee::Llvm { .. }
                ) =>
            {
                return true
            }
            kind => kind.for_each_value(|o| stack.push(o)),
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::ir::ArgIdx;

    /// A product of instruction `i` (its factors arbitrary).
    fn product(i: u32) -> Product {
        Product {
            at: InstId::new(i),
            mul: InstId::new(i),
            fmf: FastMath::CONTRACT,
            a: Value::Arg(ArgIdx::new(0)),
            b: Value::Arg(ArgIdx::new(1)),
        }
    }

    /// Leaves `Arg(k)`: those of even `k` regroup.
    fn regroup(v: Value) -> Regroup {
        let even = matches!(v, Value::Arg(a) if a.raw() % 2 == 0);
        Regroup {
            regroups: even,
            measured: true,
            load: even,
        }
    }

    /// A pseudo-random tree of `adds` additions over products and leaves, its additions in the
    /// order of their operands.
    fn tree(seed: u64, adds: usize) -> Model {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut model = Model::default();
        let mut open: Vec<usize> = Vec::new();
        let mut ids = 0u32;
        for i in 0..adds {
            let mut operand = |model: &mut Model, open: &mut Vec<usize>, r: u64| {
                if !open.is_empty() && r.is_multiple_of(3) {
                    let k = (r / 3) as usize % open.len();
                    open.swap_remove(k)
                } else if r % 3 == 1 {
                    ids += 1;
                    model.make(Kind::Mul { p: product(ids) }, None)
                } else {
                    ids += 1;
                    model.make(
                        Kind::Leaf {
                            v: Value::Arg(ArgIdx::new(ids)),
                        },
                        None,
                    )
                }
            };
            let a = operand(&mut model, &mut open, next());
            let b = operand(&mut model, &mut open, next());
            let n = model.make(
                Kind::Add {
                    x: a,
                    y: b,
                    sub: next().is_multiple_of(4),
                    at: Some(InstId::new(1000 + i as u32)),
                },
                Some(i as u64),
            );
            open.push(n);
        }
        // Join what is left into one tree.
        let mut i = adds;
        while open.len() > 1 {
            let (a, b) = (open.remove(0), open.remove(0));
            let n = model.make(
                Kind::Add {
                    x: a,
                    y: b,
                    sub: false,
                    at: Some(InstId::new(1000 + i as u32)),
                },
                Some(i as u64),
            );
            open.push(n);
            i += 1;
        }
        model.root = open.first().copied().unwrap_or(0);
        model.next = u64::from(u32::MAX);
        model
    }

    /// The model's value, as text.
    fn show(model: &Model, n: usize) -> String {
        match model.kind(n) {
            Some(Kind::Add { x, y, sub, .. }) => format!(
                "({} {} {})",
                show(model, x),
                if sub { "-" } else { "+" },
                show(model, y)
            ),
            Some(Kind::Mul { p }) => format!("m{}", p.at.raw()),
            Some(Kind::Fma { p, neg, c }) => format!(
                "fma({}m{}, {})",
                if neg { "-" } else { "" },
                p.at.raw(),
                show(model, c)
            ),
            Some(Kind::Leaf { v }) => format!("{v:?}"),
            Some(Kind::Neg { x }) => format!("-{}", show(model, x)),
            None => "?".to_owned(),
        }
    }

    /// The additions the root reaches.
    fn additions(model: &Model) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![model.root];
        while let Some(n) = stack.pop() {
            if let Some(kind) = model.kind(n) {
                if matches!(kind, Kind::Add { .. }) {
                    out.push(n);
                }
                stack.extend(children(kind));
            }
        }
        out
    }

    /// The rule as stated: visit every addition from the last, apply the first rule that holds,
    /// and start again.
    fn settle_from_the_start(model: &mut Model) {
        loop {
            let mut adds = additions(model);
            adds.sort_by_key(|&n| std::cmp::Reverse(model.nodes[n].order));
            if !adds.into_iter().any(|n| model.visit(&regroup, n)) {
                return;
            }
        }
    }

    /// Visiting again only the additions above a rewrite settles every tree as a fresh start after
    /// each rewrite does.
    #[test]
    fn revisiting_above_a_rewrite_settles_as_starting_again() {
        let mut changed = 0;
        for seed in 1..3000u64 {
            let adds = 1 + (seed % 9) as usize;
            let mut fast = tree(seed, adds);
            let mut slow = tree(seed, adds);
            let before = show(&fast, fast.root);
            assert!(fast.settle(&regroup, 64, &mut || Ok(())).expect("settle"));
            settle_from_the_start(&mut slow);
            let (a, b) = (show(&fast, fast.root), show(&slow, slow.root));
            assert_eq!(a, b, "seed {seed}: {before}");
            changed += usize::from(a != before);
        }
        assert!(changed > 1000, "{changed} trees rewritten");
    }
}
