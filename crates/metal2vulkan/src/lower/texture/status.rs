//! The status word of a texel result.
//!
//! `read`, `sample`, and `gather` return `{T, i8}`. The status word is Metal's residency status:
//! `sparse_read`, `sparse_sample`, and `sparse_gather` compile to the same intrinsics as `read`,
//! `sample`, and `gather`, and `sparse_color::resident()` is `(status & 1) == 0` (bit 0 set: the
//! access touched a non-resident region). No MSL function reads another bit: `metal_stdlib`'s
//! `sparse_color` keeps only `from_resident_pages : 1`, built from bit 0, so the other bits are
//! not established, and a function whose effects depend on one of them is refused.
//!
//! The whole `{T, i8}` value may be copied before its parts are taken: a `phi` merging reads from
//! different blocks, a `select`, or a `freeze`. Such a copy carries the status without observing
//! it. A carrier (the call or such a copy) used other than by `extractvalue 0` (the texel),
//! `extractvalue 1` (the word), or another copy exposes the whole word, which is refused.
//!
//! **Demanded bits**. Which bits of a word the function's effects depend on is a backward
//! demanded-bits analysis over the word's forward slice: `and` with a constant keeps the constant's
//! bits, `or`, `xor`, `phi`, `freeze`, and a `select`'s two values pass the bits demanded of
//! their result, shifts by a constant move them, `zext`, `sext`, and `trunc` re-width them, and
//! `add`, `sub`, and `mul` demand every bit at or below the highest bit demanded of their result
//! (carries only rise). Every other use (a store, a call, a comparison, a branch, a conversion,
//! a vector insertion, ...) demands every bit of the value it uses. Front-end spellings of
//! `resident()` (`and 1` then `xor 1` or `icmp eq 0`, `trunc` to `i1`, and the optimizer's
//! `shl k` / `and (1 << k)` / `or` packing of several words) demand bit 0 alone.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::intrinsics::{AirIntrinsic, TexOp};
use crate::air::ir::{BinOp, Body, Callee, CastOp, InstId, InstKind, Type, Value};
use crate::lower::select::SelectCx;
use crate::lower::users::Users;

use super::const_int;

/// How a texel call's status word is observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusUse {
    /// The function's effects depend on bit 0 of the word (`!resident()`) and on no other bit.
    Residency,
    /// An effect depends on another bit of the word, or on the whole `{T, i8}` value: not
    /// established.
    Other,
}

/// Whether `kind` is a texture call returning a texel with a status word.
fn texel_call(kind: &InstKind) -> bool {
    let InstKind::Call(c) = kind else {
        return false;
    };
    matches!(
        &c.callee,
        Callee::Air { op: AirIntrinsic::Texture(t), .. } if matches!(
            t.op,
            TexOp::Read | TexOp::Sample { .. } | TexOp::SampleGrad { .. } | TexOp::Gather { .. }
        )
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

/// The low `w` bits.
fn mask(w: u32) -> u64 {
    if w >= 64 {
        u64::MAX
    } else {
        (1u64 << w) - 1
    }
}

/// Every bit at or below the highest bit of `d` (0 for none).
fn at_or_below(d: u64) -> u64 {
    match d {
        0 => 0,
        _ => mask(64 - d.leading_zeros()),
    }
}

/// The integer width of instruction `i`'s scalar result, if it is one.
fn width(cx: &SelectCx<'_>, body: &Body, i: InstId) -> Option<u32> {
    let ty = body.insts.get(i)?.ty;
    match cx.entry.module.types().get(ty)? {
        Type::Int(w) => Some(u32::from(*w)),
        _ => None,
    }
}

/// The bits of operand `v` (of width `wv`) that user `u` (of width `wu`, when an integer
/// scalar) demands, given the bits `d` demanded of `u`'s result; `None` where `u` passes no
/// bits through (it demands every bit of `v`).
fn transfer(
    cx: &SelectCx<'_>,
    kind: &InstKind,
    v: Value,
    wv: u32,
    wu: Option<u32>,
    d: u64,
) -> Option<u64> {
    let constant = |x: Value| const_int(cx, x);
    match kind {
        InstKind::Phi { .. } | InstKind::Freeze(_) => Some(d),
        InstKind::Select { t, f, .. } if *t == v || *f == v => Some(d),
        InstKind::BinOp { op, a, b, .. } => {
            wu?;
            let other = if *a == v { *b } else { *a };
            match op {
                BinOp::And => Some(match constant(other) {
                    Some(c) => d & c,
                    None => d,
                }),
                BinOp::Or | BinOp::Xor => Some(d),
                BinOp::Add | BinOp::Sub | BinOp::Mul => Some(at_or_below(d)),
                BinOp::Shl | BinOp::LShr | BinOp::AShr if *a == v && *b != v => {
                    let k = constant(*b).filter(|k| *k < u64::from(wv))?;
                    let k = u32::try_from(k).ok()?;
                    Some(match op {
                        BinOp::Shl => d >> k,
                        BinOp::LShr => (d << k) & mask(wv),
                        _ => {
                            let shifted = (d << k) & mask(wv);
                            // The top `k` bits of the result copy the sign bit.
                            let top = d & !(mask(wv) >> k) & mask(wv);
                            if top != 0 {
                                shifted | 1u64 << (wv - 1)
                            } else {
                                shifted
                            }
                        }
                    })
                }
                _ => None,
            }
        }
        InstKind::Cast { op, .. } => {
            let wu = wu?;
            match op {
                CastOp::ZExt => Some(d & mask(wv)),
                CastOp::SExt => {
                    let sign = if d & !mask(wv) != 0 {
                        1u64 << (wv - 1)
                    } else {
                        0
                    };
                    Some((d & mask(wv)) | sign)
                }
                CastOp::Trunc => Some(d & mask(wu)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The texel calls of `body` whose status word is observed, by how.
pub(super) fn observed(cx: &SelectCx<'_>, body: &Body) -> BTreeMap<InstId, StatusUse> {
    // Every instruction that uses each instruction.
    let users = Users::of(body);
    // The carriers: the texel calls and every whole copy of a carrier, each with the calls it
    // carries.
    let calls: Vec<InstId> = body
        .insts
        .iter()
        .filter(|(_, i)| texel_call(&i.kind))
        .map(|(at, _)| at)
        .collect();
    let mut carries: BTreeMap<InstId, BTreeSet<InstId>> =
        calls.iter().map(|c| (*c, BTreeSet::from([*c]))).collect();
    let mut work = calls.clone();
    while let Some(c) = work.pop() {
        let from = carries.get(&c).cloned().unwrap_or_default();
        for &u in users.get(c) {
            let passes = body
                .insts
                .get(u)
                .and_then(|i| copied(&i.kind))
                .is_some_and(|p| p.contains(&Value::Inst(c)));
            if !passes {
                continue;
            }
            let set = carries.entry(u).or_default();
            let before = set.len();
            set.extend(from.iter().copied());
            if set.len() != before {
                work.push(u);
            }
        }
    }
    // Each carrier's uses: copies and texel extractions pass; a word extraction starts a slice;
    // anything else exposes the whole value.
    let mut out: BTreeMap<InstId, StatusUse> = BTreeMap::new();
    let mut words: Vec<(InstId, InstId)> = Vec::new();
    for (&c, from) in &carries {
        for &u in users.get(c) {
            let Some(inst) = body.insts.get(u) else {
                continue;
            };
            if copied(&inst.kind).is_some_and(|p| p.contains(&Value::Inst(c))) {
                continue;
            }
            match &inst.kind {
                InstKind::ExtractValue { agg, indices } if *agg == Value::Inst(c) => {
                    match indices.as_ref() {
                        [0] => continue,
                        [1] => words.push((u, c)),
                        _ => {
                            for call in from {
                                out.insert(*call, StatusUse::Other);
                            }
                        }
                    }
                }
                _ => {
                    for call in from {
                        out.insert(*call, StatusUse::Other);
                    }
                }
            }
        }
    }
    // Demanded bits over the words' forward slices: a value joins the slice through a use that
    // passes bits; every other use demands all of its bits.
    let mut slice: BTreeSet<InstId> = words.iter().map(|(w, _)| *w).collect();
    let mut work: Vec<InstId> = slice.iter().copied().collect();
    while let Some(v) = work.pop() {
        let Some(wv) = width(cx, body, v) else {
            continue;
        };
        for &u in users.get(v) {
            let Some(inst) = body.insts.get(u) else {
                continue;
            };
            let wu = width(cx, body, u);
            if transfer(cx, &inst.kind, Value::Inst(v), wv, wu, 1).is_some()
                && wu.is_some()
                && slice.insert(u)
            {
                work.push(u);
            }
        }
    }
    let mut demanded: BTreeMap<InstId, u64> = BTreeMap::new();
    let mut work: Vec<InstId> = slice.iter().copied().collect();
    while let Some(v) = work.pop() {
        let wv = width(cx, body, v).unwrap_or(64);
        let mut d = 0u64;
        for &u in users.get(v) {
            let Some(inst) = body.insts.get(u) else {
                continue;
            };
            let passed = if slice.contains(&u) {
                let du = demanded.get(&u).copied().unwrap_or(0);
                transfer(cx, &inst.kind, Value::Inst(v), wv, width(cx, body, u), du)
            } else {
                None
            };
            d |= passed.unwrap_or(u64::MAX) & mask(wv);
        }
        if demanded.get(&v).copied().unwrap_or(0) != d {
            demanded.insert(v, d);
            // The operands in the slice depend on this value's demand.
            if let Some(inst) = body.insts.get(v) {
                inst.kind.for_each_value(|x| {
                    if let Value::Inst(o) = x {
                        if slice.contains(&o) {
                            work.push(o);
                        }
                    }
                });
            }
        }
    }
    for (w, c) in words {
        let d = demanded.get(&w).copied().unwrap_or(0) & mask(8);
        let used = match d {
            0 => continue,
            1 => StatusUse::Residency,
            _ => StatusUse::Other,
        };
        for call in carries.get(&c).into_iter().flatten() {
            let e = out.entry(*call).or_insert(used);
            if used == StatusUse::Other {
                *e = StatusUse::Other;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{at_or_below, mask};

    #[test]
    fn masks_cover_their_widths() {
        assert_eq!(mask(1), 1);
        assert_eq!(mask(8), 0xFF);
        assert_eq!(mask(64), u64::MAX);
        assert_eq!(at_or_below(0), 0);
        assert_eq!(at_or_below(1), 1);
        assert_eq!(at_or_below(0x10), 0x1F);
        assert_eq!(at_or_below(1 << 63), u64::MAX);
    }
}
