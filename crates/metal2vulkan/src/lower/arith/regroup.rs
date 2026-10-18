//! Metal's pairing of a multiply chain's lane-uniform factors.
//!
//! **What Metal does.** Under fast math Metal's compiler reassociates a chain of floating-point
//! multiplies after it splits vectors into lanes, as LLVM's reassociation does: of a chain's
//! factors it multiplies first the two that occur together in the most of the function's chains.
//! A vector chain `(u * v)[lanes] * w` (the inner product's lanes read through a lane selection
//! or whole) whose factors `u` and `w` are lane-uniform (a splat of a scalar or of one lane, or a
//! splat constant) and whose `v` differs per lane is one chain per lane, each holding the pair
//! `u, w`, so Metal computes `v[lanes] * (u * w)`, the uniform product rounded once, and fuses
//! that product into an addition as the pair rule does ([`super::contract`]). The probes measured
//! it on an Apple M4 (macOS 27.0.1) with every lane of every case Metal's: whatever order the
//! factors are loaded in and the operands written, whether or not the inner product's other
//! lanes are used, in `half` and `float`, over 2 to 4 lanes. A chain whose inner lanes the outer
//! multiply reads have another use keeps its grouping (the product is a leaf there), and so does
//! one already paired, `v * (u * w)`.
//!
//! **The rewrite.** [`apply`] rewrites each such chain of the entry before selection, so the
//! lowering reads Metal's grouping as plain AIR: the outer multiply becomes `v' * (u' * w)`,
//! where `v'` and `u'` are `v` and `u` through the chain's lane selection (none where the outer
//! multiply reads the inner product whole), and the lane selection, and the inner product where
//! nothing else reads it, are deleted. The rewritten chain is already paired, which
//! [`paired`] recognizes, so the lowering notes no `PrecisionNote::FastMathRewrites` for it.
//!
//! **Not modeled.** Where the third factor differs per lane too, no pair repeats and Metal orders
//! the factors by their rank (their load order, measured but not modeled); where all three are
//! uniform, every pair repeats alike. Those chains, longer ones, chains through a negation, and
//! chains whose multiplies do not carry every fast-math flag (the probes measured `fast`) keep
//! their grouping and note `PrecisionNote::FastMathRewrites` (`arith::note_rewrites`).

use crate::air::ir::{
    BinOp as AirBin, Block, BlockId, Body, Const, FastMath, FuncId, Idx, Inst, InstId, InstKind,
    Module, Type, Value,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use std::collections::{BTreeMap, BTreeSet};

/// The flags each multiply of a chain carries where the probes measured it: every fast-math flag
/// (`fast`). LLVM's reassociation of a floating-point product needs `reassoc` and `nsz` of them.
const FAST: FastMath = FastMath::REASSOC
    .union(FastMath::NNAN)
    .union(FastMath::NINF)
    .union(FastMath::NSZ)
    .union(FastMath::ARCP)
    .union(FastMath::CONTRACT)
    .union(FastMath::AFN);

fn invariant(what: &'static str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, what))
}

/// The function a chain is read in: its module, for types and constants, and its body.
#[derive(Clone, Copy)]
pub(crate) struct Ir<'a> {
    pub module: &'a Module,
    pub func: FuncId,
    pub body: &'a Body,
}

impl<'a> Ir<'a> {
    fn inst(&self, v: Value) -> Option<(InstId, &'a InstKind)> {
        match v {
            Value::Inst(id) => self.body.insts.get(id).map(|i| (id, &i.kind)),
            _ => None,
        }
    }

    /// The lanes of `v`'s vector type, with its element: `half` or `float` alone.
    fn float_vector(&self, v: Value) -> Option<(u32, Type)> {
        let types = self.module.types();
        let ty = self.module.value_ty(self.func, v)?;
        match types.get(ty)? {
            Type::Vector { elem, len } => match types.get(*elem)? {
                t @ (Type::Half | Type::Float) => Some((*len, t.clone())),
                _ => None,
            },
            _ => None,
        }
    }

    /// The number of lanes of `v`'s vector type.
    fn lanes(&self, v: Value) -> Option<u32> {
        let types = self.module.types();
        match types.get(self.module.value_ty(self.func, v)?)? {
            Type::Vector { len, .. } => Some(*len),
            _ => None,
        }
    }

    /// The index of the block holding `id`: blocks hold consecutive instructions, in order.
    fn block_of(&self, id: InstId) -> Option<u32> {
        let blocks = &self.body.blocks;
        let n = u32::try_from(blocks.len()).ok()?;
        let (mut lo, mut hi) = (0u32, n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let b = blocks.get(BlockId::new(mid))?;
            if id.raw() < b.start.raw() {
                hi = mid;
            } else if id.raw() >= b.end.raw() {
                lo = mid + 1;
            } else {
                return Some(mid);
            }
        }
        None
    }

    /// Whether `v` is a multiply's or a negation's value, or reaches one through lane
    /// selections, insertions or extractions: a factor a chain would extend through, which the
    /// probes did not measure.
    fn reaches_product(&self, v: Value, depth: u32) -> bool {
        let Some((_, kind)) = self.inst(v) else {
            return false;
        };
        match kind {
            InstKind::BinOp {
                op: AirBin::FMul, ..
            }
            | InstKind::FNeg { .. } => true,
            _ if depth == 0 => false,
            InstKind::ShuffleVector { a, b, .. } => {
                self.reaches_product(*a, depth - 1) || self.reaches_product(*b, depth - 1)
            }
            InstKind::InsertElement { vec, elt, .. } => {
                self.reaches_product(*vec, depth - 1) || self.reaches_product(*elt, depth - 1)
            }
            InstKind::ExtractElement { vec, .. } => self.reaches_product(*vec, depth - 1),
            _ => false,
        }
    }

    /// Whether every lane of `lanes` of `v` is one scalar (a splat): a constant vector whose
    /// lanes there are one floating-point constant, or a lane selection that reads one lane of
    /// one source there, or reads lanes of one source that are one scalar there (a selection of
    /// a splat), through at most `depth` selections.
    fn uniform_within(&self, v: Value, lanes: &[u32], depth: u32) -> bool {
        match v {
            Value::Const(c) => {
                let consts = self.module.consts();
                let Some(Const::Aggregate(elems)) = consts.get(c).map(|c| &c.value) else {
                    return false;
                };
                let mut picked = lanes.iter().map(|&l| elems.get(l as usize).copied());
                let Some(Some(first)) = picked.next() else {
                    return false;
                };
                matches!(consts.get(first).map(|c| &c.value), Some(Const::Float(_)))
                    && picked.all(|e| e == Some(first))
            }
            Value::Inst(_) => {
                let Some((_, InstKind::ShuffleVector { a, b, mask })) = self.inst(v) else {
                    return false;
                };
                let Some(n) = self.lanes(*a) else {
                    return false;
                };
                // Each lane read, as (whether it is of `b`, its lane there); none may be poison.
                let read: Option<Vec<(bool, u32)>> = lanes
                    .iter()
                    .map(|&l| {
                        mask.get(l as usize)
                            .copied()
                            .flatten()
                            .map(|j| (j >= n, j % n))
                    })
                    .collect();
                let Some(read) = read else {
                    return false;
                };
                let Some(&(of_b, first)) = read.first() else {
                    return false;
                };
                if read.iter().any(|&(o, _)| o != of_b) {
                    return false;
                }
                let source = if of_b { *b } else { *a };
                // The lane read is of a source that is defined there.
                if let Value::Const(c) = source {
                    if matches!(
                        self.module.consts().get(c).map(|c| &c.value),
                        Some(Const::Undef | Const::Poison)
                    ) {
                        return false;
                    }
                }
                if read.iter().all(|&(_, j)| j == first) {
                    return true;
                }
                let inner: Vec<u32> = read.iter().map(|&(_, j)| j).collect();
                depth > 0 && self.uniform_within(source, &inner, depth - 1)
            }
            Value::Arg(_) => false,
        }
    }

    /// [`Ir::uniform_within`] through at most four selections.
    fn uniform(&self, v: Value, lanes: &[u32]) -> bool {
        self.uniform_within(v, lanes, 4)
    }

    /// The chain whose outer multiply is `outer` (module docs), if it is one Metal regroups,
    /// before the use test ([`Ir::users`]).
    fn chain(&self, outer: InstId) -> Option<Chain> {
        let InstKind::BinOp {
            op: AirBin::FMul,
            fmf,
            a,
            b,
            ..
        } = &self.body.insts.get(outer)?.kind
        else {
            return None;
        };
        let (width, elem) = self.float_vector(Value::Inst(outer))?;
        if !fmf.contains(FAST) || width < 2 || a == b {
            return None;
        }
        let block = self.block_of(outer)?;
        [(*a, *b), (*b, *a)].into_iter().find_map(|(reach, other)| {
            let (sel, inner, lanes) = match self.inst(reach)? {
                (sel, InstKind::ShuffleVector { a, b, mask }) => {
                    let (inner, _) = self.inst(*a)?;
                    let n = self.lanes(*a)?;
                    let poison = matches!(b, Value::Const(_));
                    let lanes: Option<Vec<u32>> =
                        mask.iter().map(|j| j.filter(|&j| j < n)).collect();
                    (poison && self.block_of(sel)? == block).then_some(())?;
                    (Some(sel), inner, lanes?)
                }
                (inner, _) => (None, inner, (0..width).collect()),
            };
            let InstKind::BinOp {
                op: AirBin::FMul,
                fmf: inner_fmf,
                a: p,
                b: q,
                ..
            } = &self.body.insts.get(inner)?.kind
            else {
                return None;
            };
            let (_, inner_elem) = self.float_vector(Value::Inst(inner))?;
            if !inner_fmf.contains(FAST) || inner_elem != elem || self.block_of(inner)? != block {
                return None;
            }
            let all: Vec<u32> = (0..width).collect();
            if !self.uniform(other, &all)
                || [*p, *q, other].iter().any(|&v| self.reaches_product(v, 3))
            {
                return None;
            }
            let (per_lane, uniform) = match (self.uniform(*p, &lanes), self.uniform(*q, &lanes)) {
                (false, true) => (*p, *q),
                (true, false) => (*q, *p),
                _ => return None,
            };
            Some(Chain {
                outer,
                sel,
                inner,
                lanes,
                per_lane,
                uniform,
                other,
            })
        })
    }

    /// Each instruction of `of` with the instructions that read it, in order, once per operand.
    fn users(
        &self,
        of: &BTreeSet<InstId>,
        m: &mut Meter,
    ) -> Result<BTreeMap<InstId, Vec<InstId>>, Error> {
        let mut users: BTreeMap<InstId, Vec<InstId>> = BTreeMap::new();
        let mut marked = vec![false; self.body.insts.len()];
        for d in of {
            if let Some(x) = marked.get_mut(d.index()) {
                *x = true;
            }
        }
        for (at, inst) in self.body.insts.iter() {
            m.tick(1)?;
            let mut read = Vec::new();
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    if marked.get(d.index()).copied().unwrap_or(false) {
                        read.push(d);
                    }
                }
            });
            for d in read {
                users.entry(d).or_default().push(at);
            }
        }
        Ok(users)
    }

    /// Whether the inner product's lanes the chain reads have no other use, and no multiply reads
    /// the outer one (it is its chain's last), given each instruction's `users`.
    fn alone(&self, c: &Chain, users: &BTreeMap<InstId, Vec<InstId>>) -> bool {
        let none = Vec::new();
        let of = |i: InstId| users.get(&i).unwrap_or(&none);
        if of(c.outer).iter().any(|&u| {
            matches!(
                self.body.insts.get(u).map(|i| &i.kind),
                Some(InstKind::BinOp {
                    op: AirBin::FMul,
                    ..
                })
            )
        }) {
            return false;
        }
        let reader = c.sel.unwrap_or(c.outer);
        if let Some(sel) = c.sel {
            if of(sel).as_slice() != [c.outer] {
                return false;
            }
        }
        let n = self.lanes(Value::Inst(c.inner)).unwrap_or(0);
        let read: BTreeSet<u32> = c.lanes.iter().copied().collect();
        let mut readers = 0;
        of(c.inner).iter().all(|&u| {
            if u == reader {
                readers += 1;
                return readers == 1;
            }
            match self.body.insts.get(u).map(|i| &i.kind) {
                Some(InstKind::ShuffleVector { a, b, mask }) => mask.iter().flatten().all(|&j| {
                    let lane = match (*a == Value::Inst(c.inner), *b == Value::Inst(c.inner)) {
                        (true, _) if j < n => Some(j),
                        (_, true) if j >= n => Some(j - n),
                        _ => None,
                    };
                    lane.is_none_or(|l| !read.contains(&l))
                }),
                Some(InstKind::ExtractElement {
                    idx: Value::Const(k),
                    ..
                }) => match self.module.consts().get(*k).map(|c| &c.value) {
                    Some(Const::Int(l)) => u32::try_from(*l).is_ok_and(|l| !read.contains(&l)),
                    _ => false,
                },
                _ => false,
            }
        })
    }
}

/// A chain Metal regroups (module docs): `outer = sel(inner) * other` (either order), `inner =
/// per_lane * uniform` (either order).
#[derive(Clone, Debug)]
struct Chain {
    outer: InstId,
    /// The lane selection the outer multiply reads the inner product through, if any.
    sel: Option<InstId>,
    inner: InstId,
    /// The inner product's lane each lane of the outer multiply reads.
    lanes: Vec<u32>,
    per_lane: Value,
    uniform: Value,
    other: Value,
}

/// Whether the multiply `at` is a chain Metal keeps as it is because its uniform factors are
/// already paired (module docs): `v * w` (either order), `v` differing per lane,
/// with every fast-math flag, of 2 lanes or more, and `w`, in its block, lane-uniform:
/// the product of two lane-uniform vectors, or a splat of the product of two scalars, each
/// product carrying every fast-math flag and no factor reaching another product. A multiply that
/// reads `at` extends the chain, and its own note says so (`arith::note_rewrites`).
pub(crate) fn paired(ir: Ir<'_>, at: InstId) -> bool {
    let Some(InstKind::BinOp {
        op: AirBin::FMul,
        fmf,
        a,
        b,
        ..
    }) = ir.body.insts.get(at).map(|i| &i.kind)
    else {
        return false;
    };
    let Some((width, _)) = ir.float_vector(Value::Inst(at)) else {
        return false;
    };
    if !fmf.contains(FAST) || width < 2 {
        return false;
    }
    let all: Vec<u32> = (0..width).collect();
    let product = |v: Value| match ir.inst(v) {
        Some((
            _,
            InstKind::BinOp {
                op: AirBin::FMul,
                fmf,
                a,
                b,
                ..
            },
        )) => fmf.contains(FAST).then_some((*a, *b)),
        _ => None,
    };
    let pair = |w: Value| {
        if let Some((u1, u2)) = product(w) {
            return ir.uniform(u1, &all)
                && ir.uniform(u2, &all)
                && !ir.reaches_product(u1, 3)
                && !ir.reaches_product(u2, 3);
        }
        // A splat of a scalar product: `shufflevector (insertelement _, s1 * s2, k), _, <k ...>`.
        let Some((_, InstKind::ShuffleVector { a: src, mask, .. })) = ir.inst(w) else {
            return false;
        };
        let Some((_, InstKind::InsertElement { elt, idx, .. })) = ir.inst(*src) else {
            return false;
        };
        let lane = match idx {
            Value::Const(k) => match ir.module.consts().get(*k).map(|c| &c.value) {
                Some(Const::Int(l)) => u32::try_from(*l).ok(),
                _ => None,
            },
            _ => None,
        };
        ir.uniform(w, &all)
            && mask.first().copied().flatten() == lane
            && product(*elt)
                .is_some_and(|(s1, s2)| !ir.reaches_product(s1, 3) && !ir.reaches_product(s2, 3))
    };
    let Some(block) = ir.block_of(at) else {
        return false;
    };
    let near = |w: Value| {
        ir.inst(w)
            .is_some_and(|(id, _)| ir.block_of(id) == Some(block))
    };
    [(*a, *b), (*b, *a)]
        .into_iter()
        .any(|(v, w)| !ir.uniform(v, &all) && !ir.reaches_product(v, 3) && near(w) && pair(w))
}

/// The entry with each chain Metal regroups rewritten to Metal's grouping (module docs). An
/// entry with none is returned unchanged.
pub fn apply(mut e: Entry, m: &mut Meter) -> Result<Entry, Error> {
    let Some(body) = e.module.func(e.func).and_then(|f| f.body.as_ref()) else {
        return Ok(e);
    };
    let ir = Ir {
        module: &e.module,
        func: e.func,
        body,
    };
    let mut chains = Vec::new();
    for (at, _) in body.insts.iter() {
        m.tick(1)?;
        if let Some(c) = ir.chain(at) {
            chains.push(c);
        }
    }
    if chains.is_empty() {
        return Ok(e);
    }
    let of: BTreeSet<InstId> = chains
        .iter()
        .flat_map(|c| [Some(c.outer), c.sel, Some(c.inner)])
        .flatten()
        .collect();
    let users = ir.users(&of, m)?;
    chains.retain(|c| ir.alone(c, &users));
    if chains.is_empty() {
        return Ok(e);
    }
    // The lane selections die with their chains; an inner product dies where every reader is a
    // chain's (its selection, or its outer multiply reading it whole).
    let readers: BTreeSet<InstId> = chains.iter().map(|c| c.sel.unwrap_or(c.outer)).collect();
    let mut deleted: BTreeSet<InstId> = chains.iter().filter_map(|c| c.sel).collect();
    for c in &chains {
        if users
            .get(&c.inner)
            .is_some_and(|us| us.iter().all(|u| readers.contains(u)))
        {
            deleted.insert(c.inner);
        }
    }
    let rewritten = rebuild(body, &chains, &deleted, m)?;
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

/// `body` with each chain's outer multiply rewritten, preceded by the instructions its grouping
/// adds, and the `deleted` instructions dropped.
fn rebuild(
    body: &Body,
    chains: &[Chain],
    deleted: &BTreeSet<InstId>,
    m: &mut Meter,
) -> Result<Body, Error> {
    let at: BTreeMap<InstId, &Chain> = chains.iter().map(|c| (c.outer, c)).collect();
    // Each old instruction's id in the output (`None`: deleted), and the instructions each
    // chain adds before its outer multiply: its two lane selections where it has one, and the
    // uniform product.
    let added = |c: &Chain| if c.sel.is_some() { 3 } else { 1 };
    let mut map: Vec<Option<InstId>> = Vec::new();
    m.reserve(&mut map, body.insts.len())?;
    let mut next = 0u32;
    for (id, _) in body.insts.iter() {
        m.tick(1)?;
        if deleted.contains(&id) {
            map.push(None);
            continue;
        }
        if let Some(c) = at.get(&id) {
            next = next
                .checked_add(added(c))
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
            if deleted.contains(&id) {
                continue;
            }
            let inst = body
                .insts
                .get(id)
                .ok_or_else(|| invariant("instruction out of range"))?;
            let kind = match at.get(&id) {
                Some(c) => regrouped(body, c, inst, &remap, &mut out, m)?,
                None => {
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
                    kind
                }
            };
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

/// Push the instructions chain `c`'s grouping adds before its outer multiply `outer` into
/// `out`, and return the outer multiply's new operation: `v' * (u' * w)`.
fn regrouped(
    body: &Body,
    c: &Chain,
    outer: &Inst,
    remap: &dyn Fn(Value) -> Result<Value, Error>,
    out: &mut Body,
    m: &mut Meter,
) -> Result<InstKind, Error> {
    let InstKind::BinOp {
        op: AirBin::FMul,
        flags,
        fmf,
        ..
    } = &outer.kind
    else {
        return Err(invariant(
            "a regrouped chain's outer operation is not a multiply",
        ));
    };
    let (per_lane, uniform) = match c.sel {
        Some(sel) => {
            let sel = body
                .insts
                .get(sel)
                .ok_or_else(|| invariant("a lane selection out of range"))?;
            let InstKind::ShuffleVector { b, mask, .. } = &sel.kind else {
                return Err(invariant("a chain's lane selection is not a shuffle"));
            };
            let mut select = |v: Value| -> Result<Value, Error> {
                let kind = InstKind::ShuffleVector {
                    a: remap(v)?,
                    b: *b,
                    mask: mask.clone(),
                };
                Ok(Value::Inst(out.push_inst(Inst { ty: sel.ty, kind }, m)?))
            };
            (select(c.per_lane)?, select(c.uniform)?)
        }
        None => (remap(c.per_lane)?, remap(c.uniform)?),
    };
    let pair = InstKind::BinOp {
        op: AirBin::FMul,
        flags: *flags,
        fmf: *fmf,
        a: uniform,
        b: remap(c.other)?,
    };
    let pair = Value::Inst(out.push_inst(
        Inst {
            ty: outer.ty,
            kind: pair,
        },
        m,
    )?);
    Ok(InstKind::BinOp {
        op: AirBin::FMul,
        flags: *flags,
        fmf: *fmf,
        a: per_lane,
        b: pair,
    })
}
