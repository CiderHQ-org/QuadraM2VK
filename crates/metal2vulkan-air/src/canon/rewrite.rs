//! Rebuilding a function body after a canon pass decided what survives.
//!
//! Every canon pass reads a verified body, records its decisions in a [`BodyEdit`], and builds a
//! new body with [`apply`]: deleted instructions have their uses replaced (transitively) or have
//! none that survives, dropped blocks disappear with the branch edges into them, and every PHI
//! keeps exactly one entry per remaining predecessor edge. The old body is only borrowed; the
//! result is a fresh arena with dense block and instruction ids in the old layout order, ready for
//! [`Module::new`] to verify.
//!
//! [`Module::new`]: crate::ir::Module::new

use std::collections::{BTreeMap, HashMap};

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::ir::{Block, BlockId, Body, ConstId, FuncId, Idx, Inst, InstId, InstKind, ModuleParts};
use crate::ir::{Param, Type, TypeId, Value};

/// What a pass keeps, deletes, and rewrites in one body.
#[derive(Debug)]
pub(super) struct BodyEdit {
    /// Per instruction: `Some(v)` deletes it and replaces its uses by `v`, which may itself name a
    /// deleted instruction (resolved transitively).
    pub replace: Vec<Option<Value>>,
    /// Instructions replaced by a new operation (a branch whose condition folded).
    pub rewrite: BTreeMap<InstId, InstKind>,
    /// Per block: whether it survives.
    pub keep: Vec<bool>,
    /// Per instruction: deleted with no replacement, because nothing that survives uses it (a
    /// surviving use is an invariant failure).
    pub delete: Vec<bool>,
    /// Instructions moved to another block, each placed after that block's PHIs in the old layout
    /// order. The pass guarantees that every operand still dominates the new place and that the
    /// new place dominates every use.
    pub moved: BTreeMap<InstId, BlockId>,
    /// Instructions whose rewritten operation (`rewrite`) has another type than the old one.
    pub retype: BTreeMap<InstId, TypeId>,
}

impl BodyEdit {
    /// An edit of `body` that keeps everything, with its tables charged to `held`.
    pub fn keep_all(body: &Body, m: &mut Meter, held: &mut usize) -> Result<BodyEdit, Error> {
        Ok(BodyEdit {
            replace: filled(m, held, body.insts.len(), None)?,
            rewrite: BTreeMap::new(),
            keep: filled(m, held, body.blocks.len(), true)?,
            delete: filled(m, held, body.insts.len(), false)?,
            moved: BTreeMap::new(),
            retype: BTreeMap::new(),
        })
    }

    /// Delete `i`, replacing its uses by `v`.
    pub fn replace(&mut self, i: InstId, v: Value) {
        if let Some(slot) = self.replace.get_mut(i.index()) {
            *slot = Some(v);
        }
    }

    /// Delete `i`, which nothing that survives uses.
    pub fn delete(&mut self, i: InstId) {
        if let Some(slot) = self.delete.get_mut(i.index()) {
            *slot = true;
        }
    }
}

/// Append parameters of types `tys` to entry `func`, returning the first one's index: its function
/// type gains them after its own, and its parameter list as many plain parameters. A pass that
/// adds a built-in the entry reads (`matmul2d`'s lanes; a global built-in) states their bindings in
/// the entry point.
pub(super) fn append_params(
    parts: &mut ModuleParts,
    func: FuncId,
    tys: &[TypeId],
    m: &mut Meter,
) -> Result<u32, Error> {
    let f = parts
        .funcs
        .get(func)
        .ok_or_else(|| invariant("entry out of range"))?;
    let Some(Type::Function {
        ret,
        params,
        vararg,
    }) = parts.types.get(f.ty)
    else {
        return Err(invariant("the entry's type is not a function"));
    };
    let (ret, vararg) = (*ret, *vararg);
    let mut ps = params.to_vec();
    let n = u32::try_from(ps.len()).map_err(|_| invariant("entry parameters"))?;
    ps.extend_from_slice(tys);
    let ty = parts.types.intern(
        Type::Function {
            ret,
            params: ps.into(),
            vararg,
        },
        m,
    )?;
    let f = parts
        .funcs
        .get_mut(func)
        .ok_or_else(|| invariant("entry out of range"))?;
    f.ty = ty;
    let mut params = f.params.to_vec();
    m.charge(tys.len() * std::mem::size_of::<Param>())?;
    params.extend(tys.iter().map(|_| Param::default()));
    f.params = params.into_boxed_slice();
    Ok(n)
}

/// An internal-invariant error of a canon pass.
pub(super) fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// A vector of `n` copies of `v`, charged to the ledger and recorded in `held` so the pass can
/// return it when the vector is dropped.
pub(super) fn filled<T: Clone>(
    m: &mut Meter,
    held: &mut usize,
    n: usize,
    v: T,
) -> Result<Vec<T>, Error> {
    let mut out = Vec::new();
    m.reserve(&mut out, n)?;
    *held = held.saturating_add(n.saturating_mul(std::mem::size_of::<T>()));
    out.resize(n, v);
    Ok(out)
}

/// Run `f` with a counter of temporary ledger bytes, and return them to the ledger afterwards,
/// whether `f` succeeded or not.
pub(super) fn scoped<R>(
    m: &mut Meter,
    f: impl FnOnce(&mut Meter, &mut usize) -> Result<R, Error>,
) -> Result<R, Error> {
    let mut held = 0usize;
    let r = f(m, &mut held);
    m.release(held);
    r
}

/// Build the body `edit` describes. `cmap` renames constant operands (identity for most passes).
pub(super) fn apply(
    body: &Body,
    edit: &BodyEdit,
    cmap: &dyn Fn(ConstId) -> ConstId,
    m: &mut Meter,
) -> Result<Body, Error> {
    scoped(m, |m, held| apply_in(body, edit, cmap, m, held))
}

fn apply_in(
    body: &Body,
    edit: &BodyEdit,
    cmap: &dyn Fn(ConstId) -> ConstId,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Body, Error> {
    if edit.replace.len() != body.insts.len()
        || edit.delete.len() != body.insts.len()
        || edit.keep.len() != body.blocks.len()
    {
        return Err(invariant("body edit does not match its body"));
    }
    // New block and instruction ids, dense in the old layout order.
    let mut new_block: Vec<Option<BlockId>> = filled(m, held, body.blocks.len(), None)?;
    let mut new_inst: Vec<Option<InstId>> = filled(m, held, body.insts.len(), None)?;
    let (mut nb, mut ni) = (0u32, 0u32);
    for (b, block) in body.blocks.iter() {
        m.tick(1)?;
        if !edit.keep.get(b.index()).copied().unwrap_or(false) {
            continue;
        }
        if let Some(slot) = new_block.get_mut(b.index()) {
            *slot = Some(BlockId::new(nb));
            nb += 1;
        }
        for i in placed(body, block, b, edit) {
            let deleted = edit.delete.get(i.index()).copied().unwrap_or(true);
            if !deleted && edit.replace.get(i.index()).is_some_and(Option::is_none) {
                if let Some(slot) = new_inst.get_mut(i.index()) {
                    *slot = Some(InstId::new(ni));
                    ni += 1;
                }
            }
        }
    }
    if new_block.first().copied().flatten() != Some(BlockId::new(0)) {
        return Err(invariant("a body edit dropped the entry block"));
    }

    let limit = body.insts.len();
    let map_value = |v: Value| -> Result<Value, Error> {
        let mut v = v;
        for _ in 0..=limit {
            match v {
                Value::Inst(i) => match edit.replace.get(i.index()).copied().flatten() {
                    Some(next) => v = next,
                    None => {
                        let id =
                            new_inst.get(i.index()).copied().flatten().ok_or_else(|| {
                                invariant("a use of a deleted instruction survives")
                            })?;
                        return Ok(Value::Inst(id));
                    }
                },
                Value::Const(c) => return Ok(Value::Const(cmap(c))),
                Value::Arg(_) => return Ok(v),
            }
        }
        Err(invariant("instruction replacements form a cycle"))
    };
    let map_block = |b: BlockId| -> Option<BlockId> { new_block.get(b.index()).copied().flatten() };

    // The new body's size is known: room for exactly its survivors, so it holds no arena slack
    // beside the old body.
    let mut out = Body::with_capacity(ni as usize, nb as usize, m)?;
    for (b, block) in body.blocks.iter() {
        if map_block(b).is_none() {
            continue;
        }
        let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        for i in placed(body, block, b, edit) {
            m.tick(1)?;
            if new_inst.get(i.index()).copied().flatten().is_none() {
                continue;
            }
            let old = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            let mut kind = match edit.rewrite.get(&i) {
                Some(k) => k.clone(),
                None => old.kind.clone(),
            };
            // Blocks first: a PHI entry from a dropped block goes with its block, and its value
            // (which may be defined only in that block) is never mapped.
            map_targets(&mut kind, &map_block)?;
            let mut failed = None;
            kind.for_each_value_mut(|v| match map_value(*v) {
                Ok(n) => *v = n,
                Err(e) => {
                    failed.get_or_insert(e);
                }
            });
            if let Some(e) = failed {
                return Err(e);
            }
            let ty = edit.retype.get(&i).copied().unwrap_or(old.ty);
            out.push_inst(Inst { ty, kind }, m)?;
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
    fix_phis(&mut out, m, held)?;
    Ok(out)
}

/// The instructions block `b` (`block`) holds after `edit`'s moves, in order: its PHIs, the
/// instructions moved to it, then the rest of its own that stay.
fn placed<'a>(
    body: &'a Body,
    block: &'a Block,
    b: BlockId,
    edit: &'a BodyEdit,
) -> impl Iterator<Item = InstId> + 'a {
    let is_phi = move |i: &InstId| {
        body.insts
            .get(*i)
            .is_some_and(|n| matches!(n.kind, InstKind::Phi { .. }))
    };
    let stays = move |i: &InstId| !edit.moved.contains_key(i);
    let phis = block.inst_ids().filter(is_phi).filter(stays);
    let arrivals = edit
        .moved
        .iter()
        .filter(move |(_, &to)| to == b)
        .map(|(&i, _)| i);
    let rest = block.inst_ids().filter(move |i| !is_phi(i)).filter(stays);
    phis.chain(arrivals).chain(rest)
}

/// Rename the block operands of a terminator or PHI. PHI entries from dropped blocks are removed
/// here; a branch to a dropped block is an invariant failure (the pass must have rewritten it).
fn map_targets(
    kind: &mut InstKind,
    map_block: &dyn Fn(BlockId) -> Option<BlockId>,
) -> Result<(), Error> {
    let target = |b: BlockId| map_block(b).ok_or_else(|| invariant("a branch to a dropped block"));
    match kind {
        InstKind::Br { dest, .. } => *dest = target(*dest)?,
        InstKind::CondBr { t, f, .. } => {
            *t = target(*t)?;
            *f = target(*f)?;
        }
        InstKind::Switch { default, cases, .. } => {
            *default = target(*default)?;
            for (_, b) in cases.iter_mut() {
                *b = target(*b)?;
            }
        }
        InstKind::Phi { incoming, .. } => {
            let kept: Vec<(Value, BlockId)> = incoming
                .iter()
                .filter_map(|&(v, b)| Some((v, map_block(b)?)))
                .collect();
            *incoming = kept.into_boxed_slice();
        }
        _ => {}
    }
    Ok(())
}

/// Keep, for each PHI, one entry per remaining predecessor edge: an edge that a folded branch
/// removed takes its entry with it, and parallel edges keep as many entries as remain.
fn fix_phis(body: &mut Body, m: &mut Meter, held: &mut usize) -> Result<(), Error> {
    let mut edges: HashMap<(BlockId, BlockId), u32> = HashMap::new();
    let mut charged = 0usize;
    for b in body.blocks.ids() {
        for s in body.successors(b) {
            m.tick(1)?;
            let e = edges.entry((b, s)).or_insert(0);
            if *e == 0 {
                let bytes = std::mem::size_of::<((BlockId, BlockId), u32)>() * 2;
                m.charge(bytes)?;
                charged += bytes;
            }
            *e += 1;
        }
    }
    *held = held.saturating_add(charged);
    let mut remaining: HashMap<BlockId, u32> = HashMap::new();
    for index in 0..body.blocks.len() {
        let Some((b, block)) =
            BlockId::from_index(index).and_then(|b| Some((b, *body.blocks.get(b)?)))
        else {
            continue;
        };
        for i in block.inst_ids() {
            let Some(inst) = body.insts.get_mut(i) else {
                continue;
            };
            let InstKind::Phi { incoming, .. } = &mut inst.kind else {
                break;
            };
            m.tick(1)?;
            remaining.clear();
            let kept: Vec<(Value, BlockId)> = incoming
                .iter()
                .copied()
                .filter(|&(_, p)| {
                    let left = remaining
                        .entry(p)
                        .or_insert_with(|| edges.get(&(p, b)).copied().unwrap_or(0));
                    if *left == 0 {
                        false
                    } else {
                        *left -= 1;
                        true
                    }
                })
                .collect();
            *incoming = kept.into_boxed_slice();
        }
    }
    Ok(())
}
