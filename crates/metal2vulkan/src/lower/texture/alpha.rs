//! Whether a texture read's alpha is observed.
//!
//! Metal's out-of-bounds read is the zero texel with the pixel format's missing alpha read as 1,
//! so where the shader selects that texel itself (a read it checks, or a level-view read past
//! the bound level count on any profile), its alpha depends on the bound texture's format, which
//! the binder specializes (`SpecKey::OutOfBoundsAlpha`). A read whose fourth lane nothing observes
//! needs no such constant: its out-of-bounds texel is zero in every lane anything reads.
//!
//! A read returns `{<4 x T>, i8}`. Its texel is taken by `extractvalue 0`, and the whole result
//! or the texel may be copied first by a `phi`, a `select` (its two values), or a `freeze`, as
//! [`super::status`] follows them; an `insertelement` into a texel at a constant lane below 3
//! carries the texel's alpha on, and one at lane 3 replaces it (`float4(t.read(c).xyz, 1)`). A
//! value carrying a read's texel observes its alpha when it has a use other than such a copy or
//! insertion, an `extractelement` of a constant lane below 3, or a `shufflevector` whose mask
//! takes no lane 3 of it; an observed carrier observes what it was derived from.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::intrinsics::{AirIntrinsic, TexOp};
use crate::air::ir::{Body, Callee, InstId, InstKind, Value};
use crate::lower::select::SelectCx;
use crate::lower::users::Users;

use super::const_int;

/// The lane of the texel's alpha.
const ALPHA: u64 = 3;

/// What a carrier carries: a read's whole result, or its texel vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carried {
    Result,
    Texel,
}

/// Whether `kind` is a texture read with a four-lane texel.
fn four_lane_read(kind: &InstKind) -> bool {
    let InstKind::Call(c) = kind else {
        return false;
    };
    matches!(
        &c.callee,
        Callee::Air { op: AirIntrinsic::Texture(t), .. }
            if t.op == TexOp::Read && t.texel.is_some_and(|x| x.ty.lanes == 4)
    )
}

/// The operands a copy passes through whole: a `phi`'s incoming values, a `select`'s two
/// values, and a `freeze`'s operand.
fn copied(kind: &InstKind) -> Option<Vec<Value>> {
    Some(match kind {
        InstKind::Phi { incoming, .. } => incoming.iter().map(|(v, _)| *v).collect(),
        InstKind::Select { t, f, .. } => vec![*t, *f],
        InstKind::Freeze(v) => vec![*v],
        _ => return None,
    })
}

/// The four-lane texture reads of `body` whose texel's alpha is observed.
pub(super) fn observed(cx: &SelectCx<'_>, body: &Body) -> BTreeSet<InstId> {
    // Every instruction that uses each instruction.
    let users = Users::of(body);
    // The carriers, from the reads through `extractvalue 0` and copies, each with the carriers
    // it was derived from.
    let reads: Vec<InstId> = body
        .insts
        .iter()
        .filter(|(_, i)| four_lane_read(&i.kind))
        .map(|(at, _)| at)
        .collect();
    let mut carried: BTreeMap<InstId, Carried> =
        reads.iter().map(|r| (*r, Carried::Result)).collect();
    let mut sources: BTreeMap<InstId, Vec<InstId>> = BTreeMap::new();
    let mut work = reads.clone();
    while let Some(c) = work.pop() {
        let kind = carried.get(&c).copied().unwrap_or(Carried::Texel);
        for &u in users.get(c) {
            let Some(inst) = body.insts.get(u) else {
                continue;
            };
            let derived = match &inst.kind {
                InstKind::ExtractValue { agg, indices }
                    if kind == Carried::Result
                        && *agg == Value::Inst(c)
                        && indices.as_ref() == [0] =>
                {
                    Some(Carried::Texel)
                }
                k if copied(k).is_some_and(|p| p.contains(&Value::Inst(c))) => Some(kind),
                // An insertion keeps the texel's alpha unless it writes lane 3.
                InstKind::InsertElement { vec, idx, .. }
                    if kind == Carried::Texel && *vec == Value::Inst(c) =>
                {
                    match const_int(cx, *idx) {
                        Some(k) if k != ALPHA => Some(Carried::Texel),
                        _ => None,
                    }
                }
                _ => None,
            };
            let Some(d) = derived else { continue };
            sources.entry(u).or_default().push(c);
            if carried.insert(u, d).is_none() {
                work.push(u);
            }
        }
    }
    // The carriers with an observing use.
    let mut seen: BTreeSet<InstId> = BTreeSet::new();
    for inst in body.insts.values() {
        let passes = copied(&inst.kind);
        inst.kind.for_each_value(|v| {
            let Value::Inst(d) = v else { return };
            let Some(&kind) = carried.get(&d) else { return };
            if passes.as_ref().is_some_and(|p| p.contains(&v)) {
                return;
            }
            let quiet = match (&inst.kind, kind) {
                // The texel, or the status word (its residency, never the texel's alpha).
                (InstKind::ExtractValue { indices, .. }, Carried::Result) => {
                    matches!(indices.as_ref(), [0] | [1])
                }
                (InstKind::ExtractElement { vec, idx }, Carried::Texel) => {
                    *vec == v && const_int(cx, *idx).is_some_and(|k| k < ALPHA)
                }
                // At a constant lane the insertion is a carrier (checked itself) or drops the
                // alpha.
                (InstKind::InsertElement { vec, elt, idx }, Carried::Texel) => {
                    *vec == v && *elt != v && const_int(cx, *idx).is_some()
                }
                (InstKind::ShuffleVector { a, b, mask }, Carried::Texel) => {
                    // Both operands have the texel's four lanes; `b`'s follow `a`'s.
                    let mut alpha = Vec::new();
                    if *a == v {
                        alpha.push(ALPHA);
                    }
                    if *b == v {
                        alpha.push(ALPHA + 4);
                    }
                    !mask
                        .iter()
                        .flatten()
                        .any(|m| alpha.contains(&u64::from(*m)))
                }
                _ => false,
            };
            if !quiet {
                seen.insert(d);
            }
        });
    }
    // An observed carrier observes what it was derived from.
    let mut work: Vec<InstId> = seen.iter().copied().collect();
    while let Some(o) = work.pop() {
        for &s in sources.get(&o).into_iter().flatten() {
            if seen.insert(s) {
                work.push(s);
            }
        }
    }
    reads.into_iter().filter(|r| seen.contains(r)).collect()
}
