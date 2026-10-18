//! The AIR 2.7 simdgroup-matrix block load and store in their AIR 2.8 form.
//!
//! `simdgroup_load(m, p, E, origin, transpose)` and `simdgroup_store(m, p, E, origin, transpose)`
//! compile for macOS 15 and earlier (AIR 2.7: `xcrun metal` 32023.921, `-std=metal3.0 -target
//! air64-apple-macos15.0`) to `air.simdgroup_matrix_8x8_{load,store}` with `i64 E`, `<2 x i64>
//! origin` and `i1 transpose` after the pointer, and for macOS 26 (AIR 2.8) to the same names with
//! three `<2 x i64>`: an extent, a stride and an origin. The same MSL compiled both ways gives the
//! mapping: untransposed, the extent `(E, 8)`, the stride `(1, E)` and the
//! origin as it is; transposed, the extent `(8, E)`, the stride `(E, 1)` and the origin's lanes
//! swapped. `expand` rewrites each AIR 2.7 call whose transpose flag SCCP made a constant into
//! the AIR 2.8 call, building the vectors as the AIR 2.8 front end does (`insertelement` of `E`
//! into a constant vector, a `shufflevector` swapping the origin's lanes; a constant where `E` or
//! the origin is one), and retypes its declaration, so the lowering, memory provenance and the
//! lockstep facts read one form. A declaration with a call whose flag is no constant keeps every
//! call in the AIR 2.7 form, which lowering refuses.

use std::collections::{BTreeMap, BTreeSet};

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{AirIntrinsic, SimdMatrix};
use crate::ir::{Block, Body, Call, Callee, Const, ConstId, Constant, FuncId, Idx, Inst, InstId};
use crate::ir::{InstKind, ModuleParts, Type, TypeId, Value};

use super::rewrite::invariant;

/// An AIR 2.7 block operation's call: its declaration, its operation, and the call.
fn legacy(kind: &InstKind) -> Option<(FuncId, SimdMatrix, &Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    match call.callee {
        Callee::Air {
            decl,
            op: AirIntrinsic::SimdMatrix(op),
        } if op.modern().is_some() => Some((decl, op, call)),
        _ => None,
    }
}

/// The call's transpose flag (its last operand), when a constant.
fn flag(parts: &ModuleParts, call: &Call) -> Option<bool> {
    let Value::Const(c) = *call.args.last()? else {
        return None;
    };
    match parts.consts.get(c)?.value {
        Const::Int(v) => Some(v & 1 == 1),
        _ => None,
    }
}

/// One AIR 2.8 descriptor operand: a value at hand, or an instruction to emit before the call.
#[derive(Clone, Debug)]
enum Desc {
    Ready(Value),
    Emit(InstKind),
}

/// A rewritten call's AIR 2.8 operation and descriptors (extent, stride, origin).
#[derive(Clone, Debug)]
struct Plan {
    op: SimdMatrix,
    descriptors: [Desc; 3],
}

impl Plan {
    fn emitted(&self) -> u32 {
        self.descriptors
            .iter()
            .filter(|d| matches!(d, Desc::Emit(_)))
            .count() as u32
    }
}

/// The constants the plans build from.
struct Vocab {
    v2: TypeId,
    poison: ConstId,
}

impl Vocab {
    fn int(&self, parts: &mut ModuleParts, x: u64, m: &mut Meter) -> Result<ConstId, Error> {
        let i64t = parts.types.intern(Type::Int(64), m)?;
        parts.consts.intern(
            Constant {
                ty: i64t,
                value: Const::Int(x),
            },
            m,
        )
    }

    /// The `<2 x i64>` constant of lanes `a` and `b`.
    fn pair(
        &self,
        parts: &mut ModuleParts,
        a: ConstId,
        b: ConstId,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let c = parts.consts.intern(
            Constant {
                ty: self.v2,
                value: Const::Aggregate(Box::new([a, b])),
            },
            m,
        )?;
        Ok(Value::Const(c))
    }

    /// `(e, k)` (`at` 0) or `(k, e)` (`at` 1), `e` the elements per row: a constant where `e` is
    /// an integer constant, else `insertelement` of `e` into `<poison, k>` or `<k, poison>`.
    fn with(
        &self,
        parts: &mut ModuleParts,
        e: Value,
        k: u64,
        at: u64,
        m: &mut Meter,
    ) -> Result<Desc, Error> {
        let kc = self.int(parts, k, m)?;
        let known = match e {
            Value::Const(c) => {
                matches!(parts.consts.get(c).map(|k| &k.value), Some(Const::Int(_))).then_some(c)
            }
            _ => None,
        };
        if let Some(ec) = known {
            let (a, b) = if at == 0 { (ec, kc) } else { (kc, ec) };
            return Ok(Desc::Ready(self.pair(parts, a, b, m)?));
        }
        let (a, b) = if at == 0 {
            (self.poison, kc)
        } else {
            (kc, self.poison)
        };
        let vec = self.pair(parts, a, b, m)?;
        let idx = Value::Const(self.int(parts, at, m)?);
        Ok(Desc::Emit(InstKind::InsertElement { vec, elt: e, idx }))
    }

    /// The origin `o` with its lanes swapped: a constant where `o` is one whose lanes are known,
    /// else a `shufflevector`.
    fn swapped(&self, parts: &mut ModuleParts, o: Value, m: &mut Meter) -> Result<Desc, Error> {
        if let Value::Const(c) = o {
            match parts.consts.get(c).map(|k| k.value.clone()) {
                Some(Const::Aggregate(l)) => {
                    if let [a, b] = *l {
                        return Ok(Desc::Ready(self.pair(parts, b, a, m)?));
                    }
                }
                Some(Const::Null | Const::Undef | Const::Poison) => return Ok(Desc::Ready(o)),
                _ => {}
            }
        }
        let poison = parts.consts.intern(
            Constant {
                ty: self.v2,
                value: Const::Poison,
            },
            m,
        )?;
        Ok(Desc::Emit(InstKind::ShuffleVector {
            a: o,
            b: Value::Const(poison),
            mask: Box::new([Some(1), Some(0)]),
        }))
    }
}

/// Rewrite every AIR 2.7 block load and store in the solved `body` whose declaration's every call
/// has a constant transpose flag into the AIR 2.8 form, and retype those declarations (see the
/// module). `body` is returned unchanged when there is none.
pub(super) fn expand(body: Body, parts: &mut ModuleParts, m: &mut Meter) -> Result<Body, Error> {
    // The declarations to rewrite: every call of each has a constant flag.
    let mut kept: BTreeSet<FuncId> = BTreeSet::new();
    let mut refused: BTreeSet<FuncId> = BTreeSet::new();
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        if let Some((decl, _, call)) = legacy(&inst.kind) {
            if flag(parts, call).is_some() {
                kept.insert(decl);
            } else {
                refused.insert(decl);
            }
        }
    }
    kept.retain(|d| !refused.contains(d));
    if kept.is_empty() {
        return Ok(body);
    }
    let i64t = parts.types.intern(Type::Int(64), m)?;
    let v2 = parts.types.intern(Type::Vector { elem: i64t, len: 2 }, m)?;
    let poison = parts.consts.intern(
        Constant {
            ty: i64t,
            value: Const::Poison,
        },
        m,
    )?;
    let v = Vocab { v2, poison };
    // Each rewritten call's plan, its descriptors built from its own operands.
    let mut plans: BTreeMap<InstId, Plan> = BTreeMap::new();
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some((decl, op, call)) = legacy(&inst.kind) else {
            continue;
        };
        if !kept.contains(&decl) {
            continue;
        }
        let modern = op
            .modern()
            .ok_or_else(|| invariant("a legacy op without a modern form"))?;
        let transposed = flag(parts, call).ok_or_else(|| invariant("a kept flag that varies"))?;
        let first = usize::from(matches!(op, SimdMatrix::Store8x8Legacy { .. }));
        let (Some(&e), Some(&o)) = (call.args.get(first + 1), call.args.get(first + 2)) else {
            return Err(invariant("a legacy block operation without its operands"));
        };
        let descriptors = if transposed {
            [
                v.with(parts, e, 8, 1, m)?,
                v.with(parts, e, 1, 0, m)?,
                v.swapped(parts, o, m)?,
            ]
        } else {
            [
                v.with(parts, e, 8, 0, m)?,
                v.with(parts, e, 1, 1, m)?,
                Desc::Ready(o),
            ]
        };
        plans.insert(
            i,
            Plan {
                op: modern,
                descriptors,
            },
        );
    }
    // Each old instruction's id in the output: a rewritten call's comes after its descriptors.
    let mut result = Vec::new();
    m.reserve(&mut result, body.insts.len())?;
    let mut next = 0u32;
    for (i, _) in body.insts.iter() {
        let n = 1 + plans.get(&i).map_or(0, Plan::emitted);
        next = next
            .checked_add(n)
            .ok_or_else(|| invariant("instruction ids"))?;
        result.push(InstId::new(next - 1));
    }
    let map = |val: Value| -> Result<Value, Error> {
        match val {
            Value::Inst(i) => result
                .get(i.index())
                .map(|&n| Value::Inst(n))
                .ok_or_else(|| invariant("an operand out of range")),
            other => Ok(other),
        }
    };
    let remap = |kind: &mut InstKind| -> Result<(), Error> {
        let mut failed = None;
        kind.for_each_value_mut(|val| match map(*val) {
            Ok(n) => *val = n,
            Err(e) => {
                failed.get_or_insert(e);
            }
        });
        failed.map_or(Ok(()), Err)
    };
    let mut out = Body::default();
    for (_, block) in body.blocks.iter() {
        m.tick(1)?;
        let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        for i in block.inst_ids() {
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            let mut kind = inst.kind.clone();
            remap(&mut kind)?;
            if let (Some(plan), InstKind::Call(call)) = (plans.get(&i), &mut kind) {
                let mut operands = Vec::with_capacity(3);
                for d in &plan.descriptors {
                    operands.push(match d {
                        Desc::Ready(v) => map(*v)?,
                        Desc::Emit(k) => {
                            let mut k = k.clone();
                            remap(&mut k)?;
                            Value::Inst(out.push_inst(Inst { ty: v2, kind: k }, m)?)
                        }
                    });
                }
                let first = usize::from(matches!(plan.op, SimdMatrix::Store8x8 { .. }));
                let mut args: Vec<Value> = call.args.iter().take(first + 1).copied().collect();
                args.extend(operands);
                call.args = args.into();
                if let Callee::Air { op, .. } = &mut call.callee {
                    *op = AirIntrinsic::SimdMatrix(plan.op);
                }
            }
            out.push_inst(Inst { ty: inst.ty, kind }, m)?;
            if result.get(i.index()).map(|r| r.index() + 1) != Some(out.insts.len()) {
                return Err(invariant(
                    "a rewritten call's length differs from its measure",
                ));
            }
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
    m.release(result.len().saturating_mul(std::mem::size_of::<InstId>()));
    body.release(m);
    // Each rewritten declaration takes the AIR 2.8 type: its pointer (and stored matrix), then
    // three `<2 x i64>`.
    for decl in kept {
        let f = parts
            .funcs
            .get(decl)
            .ok_or_else(|| invariant("a declaration out of range"))?;
        let Some(Type::Function {
            ret,
            params,
            vararg,
        }) = parts.types.get(f.ty)
        else {
            return Err(invariant("a declaration's type is not a function"));
        };
        let (ret, vararg) = (*ret, *vararg);
        let mut ps: Vec<TypeId> = params
            .iter()
            .take(params.len().saturating_sub(3))
            .copied()
            .collect();
        ps.extend([v2, v2, v2]);
        let ty = parts.types.intern(
            Type::Function {
                ret,
                params: ps.into(),
                vararg,
            },
            m,
        )?;
        if let Some(f) = parts.funcs.get_mut(decl) {
            f.ty = ty;
        }
    }
    Ok(out)
}
