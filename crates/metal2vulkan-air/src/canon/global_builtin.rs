//! Global built-in variables as built-in parameters.
//!
//! Metal 3.1 lets a kernel declare a built-in as a program-scope variable (`uint3 tptg
//! [[threads_per_threadgroup]];`), which any function may read. `xcrun metal` 32023.921 compiles
//! each read of one to an `air.get_*` call ([`GlobalBuiltin`]): a vector built-in one call per
//! lane read, its operand the lane (`tptg.y` is `air.get_local_size.i32(i32 1)`), a scalar one a
//! call without operands. Each reads the value the entry's parameter of that built-in would hold,
//! so after SCCP (whose constants give every lane) `expand` adds to the entry one built-in
//! parameter per built-in and element width read this way, a `<3 x T>` for a vector built-in and
//! a `T` for a scalar one, as `matmul2d` adds its lanes, and makes each call a read of it: for a
//! vector built-in, `extractelement` of the lane its operand names, which must be a constant below
//! 3; for a scalar one, the parameter itself. Lowering then derives the parameter as it derives
//! every built-in parameter of its stage. A call whose lane is not such a constant stays a call,
//! and lowering refuses it. So does a fragment function's read of a built-in Metal refuses there
//! ([`GlobalVar::metal_refuses_in_fragment`]): the same built-in as the fragment's parameter runs
//! on Metal, so the read must not become one.

use crate::abi::{ArgTypeInfo, ParamBinding, ParamRole, Stage};
use crate::budget::Meter;
use crate::error::{DiagName, Error};
use crate::intrinsics::{AirIntrinsic, Elem, GlobalBuiltin, GlobalVar};
use crate::ir::{ArgIdx, Body, Call, Callee, Const, FuncId, InstKind, ModuleParts, Type};
use crate::ir::{TypeId, Value};

use super::rewrite::{append_params, apply, scoped, BodyEdit};

/// The global built-in `kind` reads, with the call, when it is such a read.
fn read(kind: &InstKind) -> Option<(GlobalBuiltin, &Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    match call.callee {
        Callee::Air {
            op: AirIntrinsic::GlobalBuiltin(g),
            ..
        } => Some((g, call)),
        _ => None,
    }
}

/// The lane a vector read's call reads: its operand, a constant below 3.
fn lane(parts: &ModuleParts, call: &Call) -> Option<Value> {
    let [v @ Value::Const(c)] = call.args.as_ref() else {
        return None;
    };
    match parts.consts.get(*c).map(|k| &k.value) {
        Some(Const::Int(n)) if *n < 3 => Some(*v),
        _ => None,
    }
}

/// Whether Metal refuses `var` read as a global in an entry of `stage`.
fn refused(var: GlobalVar, stage: &Stage) -> bool {
    matches!(stage, Stage::Fragment { .. }) && var.metal_refuses_in_fragment()
}

/// Make each read of a global built-in in the `stage` entry `func`'s solved `body` a read of a
/// built-in parameter added to the entry (see the module), returning the new body and the added
/// parameters' bindings (none, and `body` unchanged, when nothing reads a global built-in at a
/// lane it can name).
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    stage: &Stage,
    m: &mut Meter,
) -> Result<(Body, Vec<ParamBinding>), Error> {
    // The parameters to add, in the order the body first reads them: one per built-in and width.
    let mut read_as: Vec<GlobalBuiltin> = Vec::new();
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some((g, call)) = read(&inst.kind) else {
            continue;
        };
        if (g.var.is_vector() && lane(parts, call).is_none()) || refused(g.var, stage) {
            continue;
        }
        if !read_as.contains(&g) {
            read_as.push(g);
        }
    }
    if read_as.is_empty() {
        return Ok((body, Vec::new()));
    }
    let mut tys: Vec<TypeId> = Vec::new();
    for g in &read_as {
        let bits = match g.ty {
            Elem::I16 => 16,
            _ => 32,
        };
        let elem = parts.types.intern(Type::Int(bits), m)?;
        tys.push(if g.var.is_vector() {
            parts.types.intern(Type::Vector { elem, len: 3 }, m)?
        } else {
            elem
        });
    }
    let first = append_params(parts, func, &tys, m)?;
    let param = |g: &GlobalBuiltin| -> Option<u32> {
        let at = read_as.iter().position(|r| r == g)?;
        first.checked_add(u32::try_from(at).ok()?)
    };
    let out = scoped(m, |m, held| {
        let mut edit = BodyEdit::keep_all(&body, m, held)?;
        for (i, inst) in body.insts.iter() {
            m.tick(1)?;
            let Some((g, call)) = read(&inst.kind) else {
                continue;
            };
            let Some(p) = param(&g) else {
                continue;
            };
            let arg = Value::Arg(ArgIdx::new(p));
            if g.var.is_vector() {
                if let Some(idx) = lane(parts, call) {
                    edit.rewrite
                        .insert(i, InstKind::ExtractElement { vec: arg, idx });
                }
            } else {
                edit.replace(i, arg);
            }
        }
        apply(&body, &edit, &|c| c, m)
    })?;
    body.release(m);
    let bindings = read_as
        .iter()
        .zip(first..)
        .map(|(g, param)| ParamBinding {
            param,
            role: ParamRole::Builtin(g.var.builtin()),
            unused: false,
            predicate: None,
            ty: ArgTypeInfo::default(),
            name: Some(DiagName::new("global built-in")),
        })
        .collect();
    Ok((out, bindings))
}
