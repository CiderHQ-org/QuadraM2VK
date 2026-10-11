//! What a call through a pointer may reach in one module. The loop-freedom checker
//! ([`crate::check::loop_free`]) and the oracle's fuel transform ([`crate::oracle_fuel`]) read a
//! module's call targets from here, so that they agree on every call:
//!
//! - [`address_taken`]: every function whose address an instruction operand or a global
//!   initializer takes. Metadata references (entry lists, ABI nodes) are not address uses.
//! - [`closed_calls`]: the indirect calls whose callee is *closed*, and the functions whose
//!   address *escapes*. A callee is closed when it is a function address, or a tree of `select`s
//!   whose leaves are function addresses and null, where
//!   - every leaf names a function the module defines, of exactly the call's function type, and
//!   - every select of the tree is used only as an arm of another select of a callee tree, as an
//!     operand of an `icmp eq` / `icmp ne` against null, or as the callee of an indirect call.
//!
//!   Such a call runs one of its leaves' functions or none (a null callee is undefined behaviour,
//!   which the code guards against with the null test), and nothing else. A function whose
//!   address is used anywhere but as a leaf of closed callees only (stored, passed, returned,
//!   merged by a phi, cast, compared other than with null, in a global initializer, inside
//!   another constant, or a leaf of a callee that is not closed) escapes: any call whose targets
//!   are unknown may reach it.
//!
//! A tree is never followed through memory, an argument, a phi, a cast, or a call's result, and
//! an operand it cannot follow makes the callee open, so the rule only ever narrows a call to
//! targets the module's own instructions name.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use metal2vulkan_air::ir::{
    Body, Callee, Const, ConstId, FuncId, InstId, InstKind, IntPredicate, Module, Value,
};

/// The closed indirect calls of one module and the functions whose address escapes.
#[derive(Debug, Default)]
pub struct ClosedCalls {
    /// Per closed indirect call, keyed by its function and the call: the functions it may run.
    pub calls: BTreeMap<(FuncId, InstId), BTreeSet<FuncId>>,
    /// Every function whose address some use outside a closed callee takes.
    pub escaped: BTreeSet<FuncId>,
}

impl ClosedCalls {
    /// The targets of the call `call` in `f`, when its callee is closed.
    pub fn targets(&self, f: FuncId, call: InstId) -> Option<&BTreeSet<FuncId>> {
        self.calls.get(&(f, call))
    }
}

/// Every function whose address an instruction operand or a global initializer of `module`
/// takes. Metadata references are not address uses.
pub fn address_taken(module: &Module) -> BTreeSet<FuncId> {
    let mut taken = BTreeSet::new();
    let mut seen = HashSet::new();
    for function in module.parts().funcs.values() {
        let Some(body) = &function.body else {
            continue;
        };
        for inst in body.insts.values() {
            inst.kind.for_each_value(|value| {
                if let Value::Const(id) = value {
                    const_funcs(module, id, &mut taken, &mut seen);
                }
            });
        }
    }
    for global in module.parts().globals.values() {
        if let Some(init) = global.init {
            const_funcs(module, init, &mut taken, &mut seen);
        }
    }
    taken
}

/// The functions whose address constant `id` contains, transitively.
pub fn const_funcs(
    module: &Module,
    id: ConstId,
    funcs: &mut BTreeSet<FuncId>,
    seen: &mut HashSet<ConstId>,
) {
    let mut stack = vec![id];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(constant) = module.consts().get(id) else {
            continue;
        };
        if let Const::FuncAddr(func) = constant.value {
            funcs.insert(func);
        }
        stack.extend(constant.value.children().iter().copied());
    }
}

/// The closed indirect calls of `module` and the functions whose address escapes; see the
/// module documentation.
pub fn closed_calls(module: &Module) -> ClosedCalls {
    let mut out = ClosedCalls::default();
    let mut seen = HashSet::new();
    for global in module.parts().globals.values() {
        if let Some(init) = global.init {
            const_funcs(module, init, &mut out.escaped, &mut seen);
        }
    }
    for (f, function) in module.parts().funcs.iter() {
        if let Some(body) = &function.body {
            closed_in(module, f, body, &mut out, &mut seen);
        }
    }
    out
}

/// The function `f`'s part of [`closed_calls`].
fn closed_in(
    module: &Module,
    f: FuncId,
    body: &Body,
    out: &mut ClosedCalls,
    seen: &mut HashSet<ConstId>,
) {
    let func_addr = |value: Value| -> Option<FuncId> {
        let Value::Const(id) = value else {
            return None;
        };
        match module.consts().get(id).map(|constant| &constant.value) {
            Some(Const::FuncAddr(g)) => Some(*g),
            _ => None,
        }
    };
    let is_null = |value: Value| -> bool {
        let Value::Const(id) = value else {
            return false;
        };
        matches!(
            module.consts().get(id).map(|constant| &constant.value),
            Some(Const::Null)
        )
    };

    // Each indirect call's callee tree: its selects, and its leaf functions (`None` when a leaf
    // is anything but null or a function the module defines with the call's type).
    let mut trees = Vec::new();
    let mut members = BTreeSet::new();
    for (at, inst) in body.insts.iter() {
        let InstKind::Call(call) = &inst.kind else {
            continue;
        };
        let Callee::Indirect { callee, fn_ty } = call.callee else {
            continue;
        };
        let mut selects = BTreeSet::new();
        let mut leaves = Some(BTreeSet::new());
        let mut stack = vec![callee];
        while let Some(value) = stack.pop() {
            if is_null(value) {
                continue;
            }
            if let Some(g) = func_addr(value) {
                let defined = module
                    .func(g)
                    .is_some_and(|target| target.body.is_some() && target.ty == fn_ty);
                if !defined {
                    leaves = None;
                } else if let Some(leaves) = &mut leaves {
                    leaves.insert(g);
                }
                continue;
            }
            match value {
                Value::Inst(i) => match body.insts.get(i).map(|inst| &inst.kind) {
                    Some(InstKind::Select { t, f: other, .. }) => {
                        if selects.insert(i) {
                            stack.push(*t);
                            stack.push(*other);
                        }
                    }
                    _ => leaves = None,
                },
                Value::Arg(_) | Value::Const(_) => leaves = None,
            }
        }
        members.extend(selects.iter().copied());
        trees.push((at, selects, leaves));
    }

    // A select of a tree escapes when anything uses it but another tree's select (as an arm), a
    // null test, or a call (as its callee).
    let mut escaping = BTreeSet::new();
    let member = |value: Value| match value {
        Value::Inst(i) if members.contains(&i) => Some(i),
        _ => None,
    };
    for (at, inst) in body.insts.iter() {
        match &inst.kind {
            InstKind::Select { c, t, f: other, .. } => {
                escaping.extend(member(*c));
                if !members.contains(&at) {
                    escaping.extend(member(*t));
                    escaping.extend(member(*other));
                }
            }
            InstKind::ICmp { pred, a, b } => {
                let test = matches!(pred, IntPredicate::Eq | IntPredicate::Ne);
                if !(test && is_null(*b)) {
                    escaping.extend(member(*a));
                }
                if !(test && is_null(*a)) {
                    escaping.extend(member(*b));
                }
            }
            InstKind::Call(call) => {
                for &arg in call.args.iter() {
                    escaping.extend(member(arg));
                }
            }
            kind => kind.for_each_value(|value| escaping.extend(member(value))),
        }
    }

    // A call is closed when its tree is; a select only in closed trees passes its function
    // addresses on to closed calls only.
    let mut closed_selects = BTreeSet::new();
    let mut open_selects = BTreeSet::new();
    for (at, selects, leaves) in trees {
        match leaves {
            Some(leaves) if selects.is_disjoint(&escaping) => {
                closed_selects.extend(selects);
                out.calls.insert((f, at), leaves);
            }
            _ => open_selects.extend(selects),
        }
    }

    // Every address use outside a closed callee escapes.
    for (at, inst) in body.insts.iter() {
        let mut escape = |value: Value| {
            if let Value::Const(id) = value {
                const_funcs(module, id, &mut out.escaped, seen);
            }
        };
        match &inst.kind {
            InstKind::Select { c, t, f: other, .. }
                if closed_selects.contains(&at) && !open_selects.contains(&at) =>
            {
                escape(*c);
                for arm in [*t, *other] {
                    if func_addr(arm).is_none() {
                        escape(arm);
                    }
                }
            }
            InstKind::Call(call) => {
                if let Callee::Indirect { callee, .. } = call.callee {
                    if !(out.calls.contains_key(&(f, at)) && func_addr(callee).is_some()) {
                        escape(callee);
                    }
                }
                for &arg in call.args.iter() {
                    escape(arg);
                }
            }
            kind => kind.for_each_value(&mut escape),
        }
    }
}

#[cfg(test)]
mod tests;
