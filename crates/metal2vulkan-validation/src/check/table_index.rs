//! Constant indices past a device-buffer table.
//!
//! A device-buffer array (`device T* buffers[N] [[buffer(k)]]`) is an entry parameter whose memory
//! is the table of `N` buffer addresses Metal binds at slots `k..k+N`: element `i` is the address
//! at byte `i * 8`. The case binds exactly the `N` its function constants give the table
//! (`conformance`), so code that reads element `i >= N` reads past what Metal binds, and a write
//! through the address it finds there lands wherever that address points. A case whose code
//! reaches such an element with a constant index is malformed input, whether or not its inputs
//! drive execution there: Metal promises nothing about it, and the write becomes real as soon as
//! the inputs reach it (three gather cases ran out of fuel on it).
//!
//! The scan follows each table pointer, as a may-analysis, over every function the executed
//! modules define that the entry reaches:
//!
//! - through values: casts, `freeze`, phis, selects, and GEPs (a constant GEP keeps the byte
//!   offset, any other makes it unknown);
//! - through thread-local memory: a store of a followed pointer into an `alloca`'d object at a
//!   constant offset is read back by a load from that object at that offset (a store or load at an
//!   unknown offset reaches every offset), and `llvm.memcpy`/`llvm.memmove` of a constant length
//!   from one such object to another copies what the source range holds;
//! - through calls: into a function the same module defines, into every executed definition of
//!   a declaration's name (a visible-function stub's by the symbol its
//!   `!air.visible_function_references` entry names), and, for an indirect call, into every
//!   function a linked module defines; a returned pointer flows back to the call.
//!
//! Every load or store through a table pointer at a known byte offset is a constant-index access.
//! One at or past the table's end is the refusal. An access at an unknown offset (a loop over the
//! table) is not judged: its index is data. A linked module that does not decode is not scanned;
//! the loop classifier already refuses the case for it.

use super::interface::{gep_offset, AirInterface};
use super::loop_free::{DecodedModule, ModuleRole};
use metal2vulkan::Budget;
use metal2vulkan_air::abi::AirAbi;
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::intrinsics::LlvmIntrinsic;
use metal2vulkan_air::ir::{
    ArgIdx, Callee, Const, FuncId, Idx, InstId, InstKind, Module, Type, TypeLayouts, Value,
};
use std::collections::{BTreeSet, HashMap};

/// The bytes of one table element: a device address.
const ELEMENT_BYTES: u64 = 8;

/// Constant offsets one object keeps before its offsets collapse to unknown, so the analysis has
/// finite height.
const MAX_OFFSETS: usize = 64;

/// A thing a followed pointer points into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Object {
    /// The memory of the entry's table parameter `param`.
    Table(u32),
    /// A thread-local object: the `alloca` `inst` of function `func` in executed module `module`.
    Local {
        module: usize,
        func: FuncId,
        inst: InstId,
    },
}

/// A followed pointer: its object and its byte offset in it (`None`: unknown).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Pointer {
    object: Object,
    offset: Option<u64>,
}

type Pointers = BTreeSet<Pointer>;

/// One executed module with what the scan reads of it.
struct Scanned<'m> {
    module: &'m Module,
    layouts: TypeLayouts,
    /// Each visible-function stub's linked symbol.
    stubs: HashMap<FuncId, Vec<u8>>,
}

/// Every constant-index access past its table that the code `case_entry` runs makes: for each
/// table parameter of `interface` whose length the case's constants fix, the elements read or
/// written at or past it. `modules` are the executed modules, the entry's first.
pub fn past_the_end(
    case_entry: &str,
    modules: &[(ModuleRole, &DecodedModule)],
    interface: &AirInterface,
) -> Vec<String> {
    let tables = interface
        .params
        .iter()
        .filter(|param| param.table)
        .filter_map(|param| {
            let (slot, count) = param.slots()?;
            // A count of 0 is a table whose length the pipeline chooses.
            (count != 0).then_some((param.binding.param, slot, count))
        })
        .collect::<Vec<_>>();
    if tables.is_empty() {
        return Vec::new();
    }
    let accesses = table_accesses(
        case_entry,
        modules,
        &tables
            .iter()
            .map(|(param, ..)| *param)
            .collect::<BTreeSet<_>>(),
    );
    judge(&tables, &accesses)
}

/// The refusal of each `(parameter, slot, count)` table `accesses` reaches at or past its end.
fn judge(tables: &[(u32, u32, u32)], accesses: &BTreeSet<(u32, u64)>) -> Vec<String> {
    let mut errors = Vec::new();
    for &(param, slot, count) in tables {
        let past = accesses
            .iter()
            .filter(|(table, offset)| {
                *table == param && *offset >= u64::from(count) * ELEMENT_BYTES
            })
            .map(|(_, offset)| offset / ELEMENT_BYTES)
            .collect::<BTreeSet<_>>();
        if let Some(last) = past.last() {
            errors.push(format!(
                "device-buffer array {slot} binds {count} element{}, but the code the case runs \
                 reads element{} {} of it with a constant index; bind {} elements",
                if count == 1 { "" } else { "s" },
                if past.len() == 1 { "" } else { "s" },
                past.iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
                last + 1
            ));
        }
    }
    errors
}

/// The `(table parameter, byte offset)` of every load or store through a pointer into one of the
/// entry's `tables` (parameter indices) at a known offset, over the code `entry` reaches in
/// `modules`.
fn table_accesses(
    entry: &str,
    modules: &[(ModuleRole, &DecodedModule)],
    tables: &BTreeSet<u32>,
) -> BTreeSet<(u32, u64)> {
    let mut meter = Meter::new(&Budget::default());
    let scanned = modules
        .iter()
        .filter_map(|(_, decoded)| {
            let module = decoded.module();
            let layouts =
                TypeLayouts::new(module.data_layout(), module.types(), &mut meter).ok()?;
            let stubs = decoded
                .abi()
                .ok()
                .into_iter()
                .flat_map(AirAbi::visible_function_stubs)
                .map(|(stub, symbol)| (stub, symbol.0.to_vec()))
                .collect();
            Some(Scanned {
                module,
                layouts,
                stubs,
            })
        })
        .collect::<Vec<_>>();
    if scanned.len() != modules.len() {
        return BTreeSet::new();
    }
    let Some(entry_func) = defined(scanned[0].module, entry.as_bytes()) else {
        return BTreeSet::new();
    };
    let linked = modules
        .iter()
        .enumerate()
        .filter(|(_, (role, _))| *role != ModuleRole::Entry)
        .flat_map(|(position, _)| {
            let module = scanned[position].module;
            module
                .parts()
                .funcs
                .iter()
                .filter(|(_, function)| function.body.is_some())
                .map(move |(func, _)| (position, func))
        })
        .collect::<Vec<_>>();
    let mut scan = Scan {
        scanned: &scanned,
        linked,
        values: HashMap::new(),
        memory: HashMap::new(),
        returns: HashMap::new(),
        accesses: BTreeSet::new(),
        reached: BTreeSet::from([(0, entry_func)]),
    };
    for &param in tables {
        if let Some(arg) = ArgIdx::from_index(param as usize) {
            scan.values.insert(
                (0, entry_func, Value::Arg(arg)),
                BTreeSet::from([Pointer {
                    object: Object::Table(param),
                    offset: Some(0),
                }]),
            );
        }
    }
    // Every function is revisited until nothing grows; every set has finite height.
    loop {
        let mut changed = false;
        for (module, func) in scan.reached.clone() {
            changed |= scan.propagate(module, func);
        }
        if !changed {
            return scan.accesses;
        }
    }
}

/// The function `module` defines under `name`.
fn defined(module: &Module, name: &[u8]) -> Option<FuncId> {
    module
        .parts()
        .funcs
        .iter()
        .find(|(_, function)| {
            function.body.is_some() && module.symbols().get(function.name) == Some(name)
        })
        .map(|(func, _)| func)
}

struct Scan<'s, 'm> {
    scanned: &'s [Scanned<'m>],
    /// Every function a linked module defines: an indirect call's possible targets.
    linked: Vec<(usize, FuncId)>,
    values: HashMap<(usize, FuncId, Value), Pointers>,
    /// What each thread-local object holds at each offset (`None`: stored at an unknown offset).
    memory: HashMap<(Object, Option<u64>), Pointers>,
    returns: HashMap<(usize, FuncId), Pointers>,
    accesses: BTreeSet<(u32, u64)>,
    reached: BTreeSet<(usize, FuncId)>,
}

impl Scan<'_, '_> {
    fn get(&self, module: usize, func: FuncId, value: Value) -> Pointers {
        self.values
            .get(&(module, func, value))
            .cloned()
            .unwrap_or_default()
    }

    /// Add `add` to `set`, collapsing an object's offsets to unknown past [`MAX_OFFSETS`] or once
    /// one is unknown; returns whether `set` grew.
    fn join(set: &mut Pointers, add: Pointers) -> bool {
        let before = set.len();
        let had_unknown = set
            .iter()
            .filter(|pointer| pointer.offset.is_none())
            .count();
        set.extend(add);
        let objects = set
            .iter()
            .map(|pointer| pointer.object)
            .collect::<BTreeSet<_>>();
        for object in objects {
            let unknown = Pointer {
                object,
                offset: None,
            };
            let count = set
                .iter()
                .filter(|pointer| pointer.object == object)
                .count();
            if set.contains(&unknown) || count > MAX_OFFSETS {
                set.retain(|pointer| pointer.object != object);
                set.insert(unknown);
            }
        }
        set.len() != before
            || set
                .iter()
                .filter(|pointer| pointer.offset.is_none())
                .count()
                != had_unknown
    }

    fn join_value(&mut self, module: usize, func: FuncId, value: Value, add: Pointers) -> bool {
        if add.is_empty() {
            return false;
        }
        Self::join(self.values.entry((module, func, value)).or_default(), add)
    }

    /// What a load through `pointers` reads back from thread-local memory.
    fn loaded(&self, pointers: &Pointers) -> Pointers {
        let mut out = Pointers::new();
        for pointer in pointers {
            if !matches!(pointer.object, Object::Local { .. }) {
                continue;
            }
            for ((object, offset), held) in &self.memory {
                if *object == pointer.object
                    && (offset.is_none() || pointer.offset.is_none() || *offset == pointer.offset)
                {
                    out.extend(held.iter().copied());
                }
            }
        }
        out
    }

    /// Record the table accesses of a load or store through `pointers`.
    fn access(&mut self, pointers: &Pointers) {
        for pointer in pointers {
            if let (Object::Table(param), Some(offset)) = (pointer.object, pointer.offset) {
                self.accesses.insert((param, offset));
            }
        }
    }

    /// Bind a call's arguments to `target`'s parameters; returns whether anything grew.
    fn bind(&mut self, args: &[Pointers], target: (usize, FuncId)) -> bool {
        let mut changed = self.reached.insert(target);
        for (position, add) in args.iter().enumerate() {
            if let Some(param) = ArgIdx::from_index(position) {
                changed |= self.join_value(target.0, target.1, Value::Arg(param), add.clone());
            }
        }
        changed
    }

    /// The executed definitions a direct call to `callee` in `module` reaches.
    fn direct_targets(&self, module: usize, callee: FuncId) -> Vec<(usize, FuncId)> {
        let scanned = &self.scanned[module];
        if scanned
            .module
            .func(callee)
            .is_some_and(|function| function.body.is_some())
        {
            return vec![(module, callee)];
        }
        let name = match scanned.stubs.get(&callee) {
            Some(symbol) => symbol.as_slice(),
            None => scanned
                .module
                .func(callee)
                .and_then(|function| scanned.module.symbols().get(function.name))
                .unwrap_or_default(),
        };
        self.scanned
            .iter()
            .enumerate()
            .filter(|(position, _)| *position != module)
            .filter_map(|(position, other)| Some((position, defined(other.module, name)?)))
            .collect()
    }

    /// One pass over `func` of executed module `module`; returns whether anything grew.
    fn propagate(&mut self, module: usize, func: FuncId) -> bool {
        let all = self.scanned;
        let scanned = &all[module];
        let ir = scanned.module;
        let Some(body) = ir.func(func).and_then(|function| function.body.as_ref()) else {
            return false;
        };
        let mut changed = false;
        for (block, _) in body.blocks.iter() {
            for (id, inst) in body.block_insts(block) {
                let add = match &inst.kind {
                    InstKind::Alloca { .. } => BTreeSet::from([Pointer {
                        object: Object::Local {
                            module,
                            func,
                            inst: id,
                        },
                        offset: Some(0),
                    }]),
                    InstKind::Cast { v, .. } | InstKind::Freeze(v) => self.get(module, func, *v),
                    InstKind::Select { t, f, .. } => {
                        let mut both = self.get(module, func, *t);
                        both.extend(self.get(module, func, *f));
                        both
                    }
                    InstKind::Phi { incoming, .. } => incoming
                        .iter()
                        .flat_map(|(value, _)| self.get(module, func, *value))
                        .collect(),
                    InstKind::Gep {
                        src_ty,
                        base,
                        indices,
                        ..
                    } => {
                        let delta = gep_offset(ir, &scanned.layouts, *src_ty, indices);
                        self.get(module, func, *base)
                            .into_iter()
                            .map(|pointer| Pointer {
                                object: pointer.object,
                                offset: pointer.offset.zip(delta).map(|(a, b)| a.wrapping_add(b)),
                            })
                            .collect()
                    }
                    InstKind::Load { ptr, .. } => {
                        let through = self.get(module, func, *ptr);
                        self.access(&through);
                        if matches!(ir.types().get(inst.ty), Some(Type::Ptr { .. })) {
                            self.loaded(&through)
                        } else {
                            Pointers::new()
                        }
                    }
                    InstKind::Store { val, ptr, .. } => {
                        let through = self.get(module, func, *ptr);
                        self.access(&through);
                        let stored = self.get(module, func, *val);
                        if !stored.is_empty() {
                            for pointer in &through {
                                if matches!(pointer.object, Object::Local { .. }) {
                                    changed |= Self::join(
                                        self.memory
                                            .entry((pointer.object, pointer.offset))
                                            .or_default(),
                                        stored.clone(),
                                    );
                                }
                            }
                        }
                        Pointers::new()
                    }
                    InstKind::Ret(Some(value)) => {
                        let returned = self.get(module, func, *value);
                        if !returned.is_empty() {
                            changed |= Self::join(
                                self.returns.entry((module, func)).or_default(),
                                returned,
                            );
                        }
                        Pointers::new()
                    }
                    InstKind::Call(call) => {
                        let args = call
                            .args
                            .iter()
                            .map(|arg| self.get(module, func, *arg))
                            .collect::<Vec<_>>();
                        let targets = match &call.callee {
                            Callee::Direct(callee) => self.direct_targets(module, *callee),
                            Callee::Indirect { .. } => self.linked.clone(),
                            Callee::Llvm {
                                op: LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. },
                                ..
                            } => {
                                changed |= self.copy(ir, &args, call.args.get(2).copied());
                                Vec::new()
                            }
                            _ => Vec::new(),
                        };
                        let mut returned = Pointers::new();
                        for target in targets {
                            changed |= self.bind(&args, target);
                            returned
                                .extend(self.returns.get(&target).into_iter().flatten().copied());
                        }
                        returned
                    }
                    _ => Pointers::new(),
                };
                changed |= self.join_value(module, func, Value::Inst(id), add);
            }
        }
        changed
    }

    /// `llvm.memcpy`/`llvm.memmove(dst, src, len)`: what the source's thread-local objects hold in
    /// the copied range, at the matching destination offsets (every offset when a position or the
    /// length is unknown); returns whether memory grew.
    fn copy(&mut self, ir: &Module, args: &[Pointers], len: Option<Value>) -> bool {
        let (Some(dst), Some(src)) = (args.first(), args.get(1)) else {
            return false;
        };
        let length = len.and_then(|value| match value {
            Value::Const(id) => match ir.consts().get(id)?.value {
                Const::Int(raw) => Some(raw),
                _ => None,
            },
            _ => None,
        });
        let mut writes = Vec::new();
        for from in src {
            for ((object, offset), held) in &self.memory {
                if *object != from.object {
                    continue;
                }
                let relative = match (from.offset, *offset, length) {
                    (Some(start), Some(at), Some(length)) if at >= start && at - start < length => {
                        Some(Some(at - start))
                    }
                    (Some(_), Some(_), Some(_)) => None,
                    _ => Some(None),
                };
                let Some(relative) = relative else {
                    continue;
                };
                for to in dst {
                    if matches!(to.object, Object::Local { .. }) {
                        let at = to.offset.zip(relative).map(|(a, b)| a.wrapping_add(b));
                        writes.push(((to.object, at), held.clone()));
                    }
                }
            }
        }
        let mut changed = false;
        for (key, held) in writes {
            changed |= Self::join(self.memory.entry(key).or_default(), held);
        }
        changed
    }
}

#[cfg(test)]
mod tests;
