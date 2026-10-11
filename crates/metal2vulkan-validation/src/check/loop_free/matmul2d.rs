//! The `matmul2d` ABI's calls as bounded leaves of the call graph.
//!
//! A call to a `__tensorops_impl_matmul2d_op_*` declaration (`metal2vulkan_air::tensor_ops`)
//! resolves to a body Metal supplies at pipeline creation, so no executed module defines it. It
//! is a leaf, as an `air.*` intrinsic is, exactly when its work is fixed by a constant descriptor:
//! the descriptor is `matmul2d`'s template argument, so the Metal front end passes it as constants
//! (each cooperative-tensor helper, by value) or as a stack copy of a constant global (`run`, by
//! reference), and a static inner dimension `k` leaves nothing to read from the operands' extents.
//! The body then runs a fixed number of steps of `m · n · k` multiply-adds (a helper, of the
//! descriptor's tile) and calls no guest code. Any other shape keeps the call unresolved:
//!
//! - a family name the contract does not model, or a call whose argument count is not the form's;
//! - a by-value descriptor whose fields are not all constants, or that does not decode
//!   (`tensor_ops::Descriptor::from_fields`);
//! - a `run` whose descriptor is not an `alloca` that, through pointer casts, nothing writes but
//!   `llvm.memcpy` of a whole descriptor from a constant global with a decodable initializer, and
//!   nothing reads but `run` calls, with such a copy earlier in the call's own block;
//! - a dynamic `k` (`dynamic_extent`), whose work an operand's runtime extent sets.

use std::collections::BTreeSet;

use metal2vulkan_air::intrinsics::LlvmIntrinsic;
use metal2vulkan_air::ir::{
    Body, Call, Callee, CastOp, Const, ConstExpr, ConstId, Function, InstId, InstKind, Module,
    Section, Value,
};
use metal2vulkan_air::tensor_ops::{Descriptor, Inner, MatMul2dFn, DESCRIPTOR_BYTES};

/// Whether the call `at` (`call`, to the declaration `decl`) is a `matmul2d` ABI call whose
/// constant descriptor bounds its work (module documentation).
pub(super) fn bounded_leaf(
    module: &Module,
    body: &Body,
    at: InstId,
    call: &Call,
    decl: &Function,
) -> bool {
    let symbols = module.symbols();
    let section = match decl.section {
        Some(Section::Other(sym)) => symbols.get(sym),
        _ => None,
    };
    let Some(form) =
        MatMul2dFn::of_declaration(symbols.get(decl.name).unwrap_or_default(), section)
    else {
        return false;
    };
    if form.arity() != Some(call.args.len()) {
        return false;
    }
    let descriptor = if form.descriptor_by_value() {
        call.args
            .get(1..8)
            .and_then(|fields| by_value(module, fields))
    } else {
        call.args
            .first()
            .and_then(|&desc| by_reference(module, body, at, desc))
    };
    descriptor.is_some_and(|d| matches!(d.k, Inner::Static(_)))
}

/// The descriptor of seven constant field arguments.
fn by_value(module: &Module, fields: &[Value]) -> Option<Descriptor> {
    let mut values = [0u64; 7];
    for (slot, &field) in values.iter_mut().zip(fields) {
        *slot = int(module, field)?;
    }
    Descriptor::from_fields(values)
}

/// The integer constant `v` is, zero-extended.
fn int(module: &Module, v: Value) -> Option<u64> {
    match v {
        Value::Const(c) => match module.consts().get(c)?.value {
            Const::Int(v) => Some(v),
            _ => None,
        },
        _ => None,
    }
}

/// The constant descriptor a `run` call at `at` reads through `desc` (module documentation).
fn by_reference(module: &Module, body: &Body, at: InstId, desc: Value) -> Option<Descriptor> {
    let Value::Inst(root) = strip_casts(body, desc) else {
        return None;
    };
    if !matches!(body.insts.get(root)?.kind, InstKind::Alloca { .. }) {
        return None;
    }
    // The object and every pointer cast of it.
    let mut object = BTreeSet::from([root]);
    for (id, inst) in body.insts.iter() {
        if let InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v: Value::Inst(src),
        } = inst.kind
        {
            if object.contains(&src) {
                object.insert(id);
            }
        }
    }
    let member = |v: &Value| matches!(v, Value::Inst(i) if object.contains(i));
    let mut descriptor = None;
    let mut copied_before = false;
    for block in body.blocks.ids() {
        let mut copied_here = false;
        for (id, inst) in body.block_insts(block) {
            let mut operands = Vec::new();
            inst.kind.for_each_value(|v| operands.push(v));
            if !operands.iter().any(member) {
                continue;
            }
            match &inst.kind {
                InstKind::Cast { .. } if object.contains(&id) => {}
                InstKind::Call(c) if is_memcpy(c) => {
                    let (Some(dst), Some(&src), Some(&len)) =
                        (c.args.first(), c.args.get(1), c.args.get(2))
                    else {
                        return None;
                    };
                    // Only the destination may be the object, and the copy is a whole descriptor.
                    if !member(dst)
                        || member(&src)
                        || int(module, len) != Some(DESCRIPTOR_BYTES as u64)
                    {
                        return None;
                    }
                    let copy = constant_descriptor(module, strip_casts(body, src))?;
                    if !matches!(copy.k, Inner::Static(_)) {
                        return None;
                    }
                    descriptor.get_or_insert(copy);
                    copied_here = true;
                }
                // A `run` reads the descriptor through its first argument only.
                InstKind::Call(c) if is_run(module, c) => {
                    if c.args.iter().skip(1).any(member) {
                        return None;
                    }
                    if id == at {
                        copied_before = copied_here;
                    }
                }
                _ => return None,
            }
        }
    }
    copied_before.then_some(descriptor).flatten()
}

/// `v` without its pointer casts (instructions and constant expressions).
fn strip_casts(body: &Body, mut v: Value) -> Value {
    loop {
        match v {
            Value::Inst(i) => match body.insts.get(i).map(|inst| &inst.kind) {
                Some(InstKind::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    v: src,
                }) => v = *src,
                _ => return v,
            },
            _ => return v,
        }
    }
}

/// The descriptor a constant global holds, when `v` is its address.
fn constant_descriptor(module: &Module, v: Value) -> Option<Descriptor> {
    let Value::Const(mut c) = v else {
        return None;
    };
    let global = loop {
        match &module.consts().get(c)?.value {
            Const::GlobalAddr(g) => break module.global(*g)?,
            Const::Expr(ConstExpr::Cast {
                op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                ops: [src],
            }) => c = *src,
            _ => return None,
        }
    };
    if !global.constant {
        return None;
    }
    let fields = match &module.consts().get(global.init?)?.value {
        Const::Aggregate(fields) => fields,
        _ => return None,
    };
    let fields: &[ConstId; 7] = fields.as_ref().try_into().ok()?;
    let mut values = [0u64; 7];
    for (slot, &field) in values.iter_mut().zip(fields) {
        *slot = int(module, Value::Const(field))?;
    }
    Descriptor::from_fields(values)
}

fn is_memcpy(call: &Call) -> bool {
    matches!(
        call.callee,
        Callee::Llvm {
            op: LlvmIntrinsic::Memcpy {
                volatile: false,
                ..
            },
            ..
        }
    )
}

/// Whether `call` calls a `run` form of the `matmul2d` ABI.
fn is_run(module: &Module, call: &Call) -> bool {
    let Callee::Direct(f) = call.callee else {
        return false;
    };
    let Some(decl) = module.func(f).filter(|d| d.body.is_none()) else {
        return false;
    };
    let symbols = module.symbols();
    let section = match decl.section {
        Some(Section::Other(sym)) => symbols.get(sym),
        _ => None,
    };
    matches!(
        MatMul2dFn::of_declaration(symbols.get(decl.name).unwrap_or_default(), section),
        Some(MatMul2dFn::Run(_))
    )
}
