//! Pointer erasure: no-op pointer casts disappear.
//!
//! Decode already erased pointee types, so `bitcast i32 addrspace(1)* %p to i8 addrspace(1)*`
//! from typed-pointer bitcode is a `bitcast` whose operand and result are the same
//! `ptr addrspace(1)`. Opaque-pointer bitcode of the same program has no such cast. Removing every
//! `bitcast` whose operand already has the result type (instructions and constant expressions,
//! including those nested in initializers and other expressions) makes both spellings the same IR,
//! so nothing downstream can depend on the pointee types the typed spelling carried.
//!
//! The same holds for a constant `getelementptr` whose every index is zero: it is its base
//! address, and LLVM folds it to the base when it reads typed-pointer bitcode as opaque pointers
//! (`getelementptr ([256 x float], @t, 0, 0)`, the typed frontend's spelling of an array's first
//! element as an intrinsic operand, is `@t`). It erases to its base when the two have one type.

use crate::budget::Meter;
use crate::error::Error;
use crate::ir::{Body, Const, ConstExpr, ConstId, ConstTable, Constant, Global, Idx};
use crate::ir::{CastOp, InstKind, Value};

use super::rewrite::{apply, invariant, scoped, BodyEdit};

/// For every constant of `consts`, the constant it erases to, interning rebuilt expressions
/// whose operands changed. Indexed by `ConstId`; constants interned later are their own erasure.
pub(super) fn erase_consts(consts: &mut ConstTable, m: &mut Meter) -> Result<Vec<ConstId>, Error> {
    let n = consts.len();
    let mut map = Vec::new();
    m.reserve(&mut map, n)?;
    for index in 0..n {
        m.tick(1)?;
        let id = ConstId::from_index(index).ok_or_else(|| invariant("constant id space"))?;
        let c = consts
            .get(id)
            .cloned()
            .ok_or_else(|| invariant("constant out of range"))?;
        let mapped = |k: ConstId| map.get(k.index()).copied().unwrap_or(k);
        let erased = match &c.value {
            Const::Expr(ConstExpr::Cast {
                op: CastOp::Bitcast,
                ops: [v],
            }) if consts.ty(mapped(*v)) == Some(c.ty) => mapped(*v),
            Const::Expr(ConstExpr::Gep { ops, .. })
                if ops
                    .first()
                    .is_some_and(|&b| consts.ty(mapped(b)) == Some(c.ty))
                    && ops.iter().skip(1).all(|&i| is_zero(consts, mapped(i))) =>
            {
                ops.first().map_or(id, |&b| mapped(b))
            }
            value if value.children().iter().any(|&k| mapped(k) != k) => {
                let value = match value {
                    Const::Aggregate(elems) => {
                        Const::Aggregate(elems.iter().map(|&e| mapped(e)).collect())
                    }
                    Const::Expr(ConstExpr::Cast { op, ops: [v] }) => Const::Expr(ConstExpr::Cast {
                        op: *op,
                        ops: [mapped(*v)],
                    }),
                    Const::Expr(ConstExpr::BinOp {
                        op,
                        flags,
                        ops: [a, b],
                    }) => Const::Expr(ConstExpr::BinOp {
                        op: *op,
                        flags: *flags,
                        ops: [mapped(*a), mapped(*b)],
                    }),
                    Const::Expr(ConstExpr::Gep {
                        src_ty,
                        inbounds,
                        ops,
                    }) => Const::Expr(ConstExpr::Gep {
                        src_ty: *src_ty,
                        inbounds: *inbounds,
                        ops: ops.iter().map(|&o| mapped(o)).collect(),
                    }),
                    other => other.clone(),
                };
                consts.intern(Constant { ty: c.ty, value }, m)?
            }
            _ => id,
        };
        map.push(erased);
    }
    Ok(map)
}

/// Whether constant `k` is an integer zero (a `getelementptr` index).
fn is_zero(consts: &ConstTable, k: ConstId) -> bool {
    matches!(
        consts.get(k).map(|c| &c.value),
        Some(Const::Int(0) | Const::Null)
    )
}

/// Rename the initializers of `globals` through the erasure map.
pub(super) fn erase_globals<'g>(globals: impl Iterator<Item = &'g mut Global>, map: &[ConstId]) {
    for g in globals {
        if let Some(init) = g.init {
            g.init = Some(map.get(init.index()).copied().unwrap_or(init));
        }
    }
}

/// Erase the no-op pointer casts of `body` and rename its constant operands through `map`.
/// `value_ty` types an operand of the body. `None` when there is nothing to erase or rename: the
/// body is already erased (every defined function is erased before inlining, so an inlined body
/// built from them is), and is not rebuilt (that copy of an inlined body was a translation's ledger
/// peak).
pub(super) fn erase_body(
    body: &Body,
    map: &[ConstId],
    consts: &ConstTable,
    value_ty: &dyn Fn(Value) -> Option<crate::ir::TypeId>,
    m: &mut Meter,
) -> Result<Option<Body>, Error> {
    let cmap = |c: ConstId| map.get(c.index()).copied().unwrap_or(c);
    let erased = |inst: &crate::ir::Inst| match inst.kind {
        InstKind::Cast {
            op: CastOp::Bitcast,
            v,
        } => {
            let src = match v {
                Value::Const(c) => consts.ty(cmap(c)),
                other => value_ty(other),
            };
            src == Some(inst.ty)
        }
        _ => false,
    };
    let mut changes = false;
    for inst in body.insts.values() {
        m.tick(1)?;
        inst.kind.for_each_value(|v| {
            changes |= matches!(v, Value::Const(c) if cmap(c) != c);
        });
        changes |= erased(inst);
        if changes {
            break;
        }
    }
    if !changes {
        return Ok(None);
    }
    scoped(m, |m, held| {
        let mut edit = BodyEdit::keep_all(body, m, held)?;
        for (i, inst) in body.insts.iter() {
            m.tick(1)?;
            if let InstKind::Cast { v, .. } = inst.kind {
                if erased(inst) {
                    edit.replace(i, v);
                }
            }
        }
        apply(body, &edit, &cmap, m).map(Some)
    })
}
