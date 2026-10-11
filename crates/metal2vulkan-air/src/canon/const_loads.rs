//! Pointers loaded from fields of constant globals.
//!
//! A constant global whose initializer is known once function constants are bound and the static
//! initializers have run may hold pointers: a program-scope `constexpr sampler` taken by value is
//! a struct holding the address of its sampler state, and a function that picks one of two such
//! samplers by a function constant (`select i1 %c, ptr @a, ptr @b`) reads the picked one's field
//! through a `getelementptr`. SCCP folds the select, and a load of a whole global, but not a load
//! of a field. The memory plan models no pointer read back from memory and no global outside the
//! constant and threadgroup spaces, so such a module was refused, under the values that made the
//! select one global.
//!
//! This pass reads such a load as the initializer's constant at the field: a non-volatile,
//! non-atomic load of a pointer whose address is a field of a constant global at constant indices
//! (a `getelementptr` instruction or constant expression over the global's own value type, its
//! first index 0), where the field's type is the load's. A copy LLVM made through an integer (a
//! load of the field as an integer of the pointer's size, then `inttoptr` to the field's type)
//! reads the same constant at its `inttoptr`. The global is never written (it is `constant`, and
//! initialized by no one else), so every run reads that constant: the address of the sampler
//! state, which the sampler family reads as the sampler it names. Only pointers are read this
//! way: a load of any other type from a constant global is lowered from the global's form (its
//! typed private copy, or the constant blob), and reading it here would change those modules
//! without changing what they compute. SCCP then runs again over a body this pass changed, so
//! what the constants decide (an aggregate of them, a phi of them, a member of either) folds too.

use crate::budget::Meter;
use crate::error::Error;
use crate::ir::{Body, CastOp, Const, ConstExpr, ConstId, Constant, GlobalId};
use crate::ir::{InstId, InstKind, ModuleParts, Type, TypeId, Value};

use super::rewrite::{apply, scoped, BodyEdit};

/// Read every load of a pointer from a field of a constant global in `body`, and every
/// `inttoptr` of such a field's integer copy, as the field's constant (module docs), and say
/// whether there was one. Returns `body` itself when there is none.
pub(super) fn fold(
    body: Body,
    parts: &mut ModuleParts,
    m: &mut Meter,
) -> Result<(Body, bool), Error> {
    let mut found = Vec::new();
    let r = find(&body, parts, &mut found, m).and_then(|()| {
        if found.is_empty() {
            return Ok(None);
        }
        scoped(m, |m, held| {
            let mut edit = BodyEdit::keep_all(&body, m, held)?;
            for &(id, c) in &found {
                edit.replace(id, Value::Const(c));
            }
            apply(&body, &edit, &|c| c, m)
        })
        .map(Some)
    });
    m.release(found.capacity() * std::mem::size_of::<(InstId, ConstId)>());
    match r? {
        Some(rebuilt) => {
            body.release(m);
            Ok((rebuilt, true))
        }
        None => Ok((body, false)),
    }
}

/// Each instruction of `body` that reads a pointer field of a constant global (module docs), with
/// the field's constant.
fn find(
    body: &Body,
    parts: &mut ModuleParts,
    found: &mut Vec<(InstId, ConstId)>,
    m: &mut Meter,
) -> Result<(), Error> {
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some(space) = parts.types.ptr_space(inst.ty) else {
            continue;
        };
        if !matches!(parts.types.get(inst.ty), Some(Type::Ptr { .. })) {
            continue;
        }
        let ptr = match inst.kind {
            InstKind::Load {
                ptr,
                volatile: false,
                atomic: None,
                ..
            } => ptr,
            // `inttoptr` of an integer load of the pointer's size: a copy through an integer.
            InstKind::Cast {
                op: CastOp::IntToPtr,
                v: Value::Inst(l),
            } => match body.insts.get(l).map(|l| (&l.kind, l.ty)) {
                Some((
                    &InstKind::Load {
                        ptr,
                        volatile: false,
                        atomic: None,
                        ..
                    },
                    ty,
                )) if parts.types.int_bits(ty).map(u32::from)
                    == Some(parts.data_layout.pointer(space.raw()).size_bits)
                    && !matches!(parts.types.get(ty), Some(Type::Vector { .. })) =>
                {
                    ptr
                }
                _ => continue,
            },
            _ => continue,
        };
        let Some((g, path)) = field_address(body, parts, ptr) else {
            continue;
        };
        if let Some(c) = field_value(parts, g, &path, inst.ty, m)? {
            m.reserve(found, 1)?;
            found.push((id, c));
        }
    }
    Ok(())
}

/// The constant global `v` addresses a field of, and the member indices of that field below the
/// global (empty for the global itself): `@g`, a `getelementptr` over `@g`'s value type at
/// constant indices whose first is 0 (an instruction of `body` or a constant expression), through
/// no-op pointer casts. `None` for any other address.
fn field_address(body: &Body, parts: &ModuleParts, v: Value) -> Option<(GlobalId, Vec<u64>)> {
    let consts = &parts.consts;
    let (src_ty, base, indices): (TypeId, Value, Vec<Value>) = match v {
        Value::Inst(i) => match &body.insts.get(i)?.kind {
            InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            } => (*src_ty, *base, indices.to_vec()),
            _ => return None,
        },
        Value::Const(c) => match &consts.get(c)?.value {
            Const::GlobalAddr(g) => return Some((*g, Vec::new())),
            Const::Expr(ConstExpr::Cast {
                op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                ops: [k],
            }) => return field_address(body, parts, Value::Const(*k)),
            Const::Expr(ConstExpr::Gep { src_ty, ops, .. }) => {
                let (base, rest) = ops.split_first()?;
                (
                    *src_ty,
                    Value::Const(*base),
                    rest.iter().map(|&k| Value::Const(k)).collect(),
                )
            }
            _ => return None,
        },
        Value::Arg(_) => return None,
    };
    let (g, outer) = field_address(body, parts, base)?;
    if !outer.is_empty() || parts.globals.get(g)?.value_ty != src_ty {
        return None;
    }
    let mut path = Vec::with_capacity(indices.len());
    for x in indices {
        let Value::Const(k) = x else {
            return None;
        };
        let bits = parts.types.int_bits(consts.ty(k)?)?;
        let x = match consts.get(k)?.value {
            Const::Int(x) => x,
            Const::Null => 0,
            _ => return None,
        };
        // A negative index (its sign bit set) addresses no field.
        if (x >> (u32::from(bits.clamp(1, 64)) - 1)) & 1 != 0 {
            return None;
        }
        path.push(x);
    }
    match path.split_first() {
        Some((0, rest)) => Some((g, rest.to_vec())),
        _ => None,
    }
}

/// The constant at member indices `path` of constant global `g`'s initializer, when the member's
/// type is `ty`, or when its first leaf at byte 0 (its first member, recursively) is of type `ty`:
/// an access at an aggregate's own address reads its first member's bytes. `None` for a global
/// that may be written or is initialized from outside (a function constant left unsupplied), an
/// index past its aggregate, a part the initializer leaves `undef` or `poison`, and a member of
/// another type.
fn field_value(
    parts: &mut ModuleParts,
    g: GlobalId,
    path: &[u64],
    ty: TypeId,
    m: &mut Meter,
) -> Result<Option<ConstId>, Error> {
    let Some(global) = parts.globals.get(g) else {
        return Ok(None);
    };
    let (true, false, Some(mut c)) = (global.constant, global.externally_initialized, global.init)
    else {
        return Ok(None);
    };
    let mut t = global.value_ty;
    let mut rest = path.iter();
    loop {
        m.tick(1)?;
        // The next member: the path's, then the first member while the type is not `ty`.
        let i = match rest.next() {
            Some(&i) => i,
            None if t == ty => break,
            None => 0,
        };
        let next = match parts.types.get(t) {
            Some(Type::Struct { fields, .. }) => {
                usize::try_from(i).ok().and_then(|i| fields.get(i).copied())
            }
            Some(Type::Array { elem, len }) if i < *len => Some(*elem),
            _ => None,
        };
        let Some(next) = next else {
            return Ok(None);
        };
        c = match parts.consts.get(c).map(|k| &k.value) {
            Some(Const::Aggregate(elems)) => {
                match usize::try_from(i).ok().and_then(|i| elems.get(i)) {
                    Some(&e) => e,
                    None => return Ok(None),
                }
            }
            Some(Const::Null) => parts.consts.intern(
                Constant {
                    ty: next,
                    value: Const::Null,
                },
                m,
            )?,
            _ => return Ok(None),
        };
        t = next;
    }
    Ok(match parts.consts.get(c).map(|k| &k.value) {
        Some(Const::Undef | Const::Poison) | None => None,
        Some(_) => Some(c),
    })
}
