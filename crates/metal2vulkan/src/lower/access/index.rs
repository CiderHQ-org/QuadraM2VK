//! Narrow GEP indices.
//!
//! LLVM's `getelementptr` takes pointer-width (`i64`) indices, so Metal code indexes a buffer
//! with `zext`/`sext i32 %x to i64` followed by the GEP. A static-root offset is a `u32`: the GEP
//! only needs each index modulo 2³², and the low word of a zero or sign extension of a value of at
//! most 32 bits is that value zero- or sign-extended to 32 bits. The 64-bit extension is therefore
//! dead arithmetic that would need `shaderInt64`.
//!
//! The rule is structural: an **index extension** is a scalar `zext`/`sext` instruction whose
//! operand is a non-constant integer of 1 to 32 bits and whose result is an integer of 33 to 64
//! bits, and whose every use is an index operand of a `getelementptr`, in any block, or an
//! operand of an index word. The access family consumes such a GEP index as the extension's
//! operand ([`narrowed`]), with the [`IndexForm`] of the extension, and the arith family gives
//! the extension an empty strategy (`ArithStrategy::IndexExtension`), so no 64-bit value is ever
//! formed.
//!
//! **Index words**. The low word of an `add`, `sub`, `mul`, `and`, `or`, `xor`, or `shl` by a
//! constant below 32 depends only on the low words of its operands. An index word is such a scalar
//! operation with an integer result of 33 to 64 bits whose every operand is an index extension,
//! another index word, or an integer constant, and whose every use is an index operand of a
//! `getelementptr` whose base is not a wide pointer (a `u32`-offset pointer: a static root, an
//! arena, a table, a typed alloca) or an operand of another index word. It is held as its low word,
//! a `u32` (`ArithStrategy::IndexWord`), and a GEP reads it as a zero-extended word; a wide
//! pointer's offset is 64-bit, so an index of one needs the whole value and keeps its 64-bit form.
//! Both sets are the largest fixed point of these rules ([`index_values`]). Any other `i64` index
//! (a 64-bit load, a shift right, arithmetic with any other use) is a 64-bit value of the arith
//! family and still needs `shaderInt64`.
//!
//! **Across blocks**. An index extension has no value of its own: a use of it reads its operand,
//! where the use stands. `select` computes the block-crossing values and `lower::emit` each block's
//! imports through the same two hooks, [`read_value`] (a use of an extension is a use of its
//! operand) and [`reads_nothing`] (the extension itself emits nothing, so its own operand use is no
//! use), so an extension defined before a loop and used after it demotes its narrow operand, never
//! a 64-bit value. The operand dominates the extension, which dominates each of its uses, so the
//! operand is available wherever the extension was.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::ir::{
    BinOp as AirBin, Body, CastOp, Const, FuncId, Inst, InstId, InstKind, Module,
};
use crate::air::ir::{Type, Value, ValueId};
use crate::budget::{tree_bytes, Meter};
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::k32;
use crate::memory::plan::MemoryPlan;
use crate::memory::provenance::Repr;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, ScalarTy, Value as SValue};
use crate::spirv::ir::BinOp;

use super::emit::index_u32;
use super::{AccessPlan, IndexForm};

/// The narrow index values of one function: its index extensions and index words.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndexValues {
    /// Index extensions: no value of their own.
    pub extensions: BTreeSet<InstId>,
    /// Index words: held as their low word, a `u32`.
    pub words: BTreeSet<InstId>,
}

/// What an index-value candidate is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Node {
    Extension,
    Word,
}

/// One use of a candidate.
#[derive(Clone, Copy)]
enum Use {
    /// An index operand of a `getelementptr` whose base is (`wide`) or is not a wide pointer.
    Index { wide: bool },
    /// An operand of index-word candidate.
    Word(InstId),
    /// Any other use.
    Other,
}

/// Whether `op` with shift amount `b` computes the low word of its result from the low words of
/// its operands alone (the module docs' word operators).
fn word_op(module: &Module, op: AirBin, b: Value) -> bool {
    match op {
        AirBin::Add | AirBin::Sub | AirBin::Mul | AirBin::And | AirBin::Or | AirBin::Xor => true,
        AirBin::Shl => const_word(module, b).is_some_and(|k| k < 32),
        _ => false,
    }
}

/// The low word of integer constant `v`.
fn const_word(module: &Module, v: Value) -> Option<u32> {
    let Value::Const(c) = v else { return None };
    match &module.consts().get(c)?.value {
        Const::Int(u) => Some(*u as u32),
        Const::Null => Some(0),
        _ => None,
    }
}

/// The index extensions and index words of `body` (the module docs' rules), whose GEPs' bases
/// have the representations of `memory`. The candidates are found in one pass over the body and
/// their uses in a second; a worklist then removes every candidate with a disqualifying use or
/// operand, each at most once, so the work is linear in the body.
///
/// **Metered**. Each pass ticks `m` once per instruction and the worklist once per item, so the
/// scan charges at most `2·n + c + 2·u` work units for `n` instructions, `c` candidates, and `u`
/// instruction operands (every worklist item is a candidate at the start or a use or operand of one
/// removed candidate). Its tables are charged before they are built, from the first pass's counts,
/// and released on return; the two result sets are the plan's.
pub fn index_values(
    module: &Module,
    func: FuncId,
    body: &Body,
    memory: &MemoryPlan,
    m: &mut Meter,
) -> Result<IndexValues, Error> {
    let mut node: BTreeMap<InstId, Node> = BTreeMap::new();
    let mut charged = 0usize;
    let result = candidates(module, func, body, &mut node, &mut charged, m).and_then(|operands| {
        if node.is_empty() {
            Ok(IndexValues::default())
        } else {
            fixed_point(body, memory, &node, operands, m)
        }
    });
    m.release(charged);
    result
}

/// [`index_values`]' first pass: fill `node` with the candidates, charging each entry to `m`
/// before it is inserted (adding to `charged`, which the caller releases). Returns the number of
/// instruction operands in the body, which bounds the uses the second pass records.
fn candidates(
    module: &Module,
    func: FuncId,
    body: &Body,
    node: &mut BTreeMap<InstId, Node>,
    charged: &mut usize,
    m: &mut Meter,
) -> Result<usize, Error> {
    let types = module.types();
    let int_of = |v: Value| -> Option<u8> {
        match module.value_ty(func, v).and_then(|t| types.get(t)) {
            Some(Type::Int(w)) => Some(*w),
            _ => None,
        }
    };
    let entry = tree_bytes(1, std::mem::size_of::<(InstId, Node)>());
    let mut operands = 0usize;
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        inst.kind
            .for_each_value(|v| operands += usize::from(matches!(v, Value::Inst(_))));
        if !matches!(types.get(inst.ty), Some(Type::Int(33..=64))) {
            continue;
        }
        let candidate = match &inst.kind {
            InstKind::Cast {
                op: CastOp::ZExt | CastOp::SExt,
                v,
            } if !matches!(v, Value::Const(_)) && matches!(int_of(*v), Some(1..=32)) => {
                Some(Node::Extension)
            }
            InstKind::BinOp { op, a, b, .. } => {
                let operand = |v: &Value| match v {
                    Value::Const(_) => const_word(module, *v).is_some(),
                    Value::Inst(_) => true,
                    Value::Arg(_) => false,
                };
                (word_op(module, *op, *b) && operand(a) && operand(b)).then_some(Node::Word)
            }
            _ => None,
        };
        if let Some(kind) = candidate {
            m.charge(entry)?;
            *charged += entry;
            node.insert(id, kind);
        }
    }
    Ok(operands)
}

/// [`index_values`]' uses pass and worklist over the candidates `node`, for a body whose
/// instructions have `operands_total` instruction operands.
fn fixed_point(
    body: &Body,
    memory: &MemoryPlan,
    node: &BTreeMap<InstId, Node>,
    operands_total: usize,
    m: &mut Meter,
) -> Result<IndexValues, Error> {
    let c = node.len();
    // The uses table (at most one entry per candidate and one `Use` per operand), the live set,
    // and the worklist (the candidates, then at most two items per operand); a vector grown by
    // pushes holds up to twice what it stores.
    let id = std::mem::size_of::<InstId>();
    let items = c.saturating_add(operands_total.saturating_mul(2));
    let scratch = tree_bytes(c, std::mem::size_of::<(InstId, Vec<Use>)>())
        .saturating_add(operands_total.saturating_mul(2 * std::mem::size_of::<Use>()))
        .saturating_add(tree_bytes(c, id))
        .saturating_add(items.saturating_mul(2 * id));
    m.charge(scratch)?;
    let result = fixed_point_charged(body, memory, node, m);
    m.release(scratch);
    result
}

fn fixed_point_charged(
    body: &Body,
    memory: &MemoryPlan,
    node: &BTreeMap<InstId, Node>,
    m: &mut Meter,
) -> Result<IndexValues, Error> {
    let mut uses: BTreeMap<InstId, Vec<Use>> = BTreeMap::new();
    let record = |uses: &mut BTreeMap<InstId, Vec<Use>>, v: Value, u: Use| {
        if let Value::Inst(d) = v {
            if node.contains_key(&d) {
                uses.entry(d).or_default().push(u);
            }
        }
    };
    for (_, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            m.tick(1)?;
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            match &inst.kind {
                InstKind::Gep { base, indices, .. } => {
                    record(&mut uses, *base, Use::Other);
                    let wide = matches!(memory.repr(*base), Some(Repr::BufPtr { wide: true, .. }));
                    for i in indices.iter() {
                        record(&mut uses, *i, Use::Index { wide });
                    }
                }
                InstKind::BinOp { a, b, .. } if node.get(&id) == Some(&Node::Word) => {
                    record(&mut uses, *a, Use::Word(id));
                    record(&mut uses, *b, Use::Word(id));
                }
                kind => kind.for_each_value(|v| record(&mut uses, v, Use::Other)),
            }
        }
    }
    let operands = |id: InstId| -> Vec<InstId> {
        match body.insts.get(id).map(|i| &i.kind) {
            Some(InstKind::BinOp { a, b, .. }) => [*a, *b]
                .into_iter()
                .filter_map(|v| match v {
                    Value::Inst(d) => Some(d),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    };
    let mut alive: BTreeSet<InstId> = node.keys().copied().collect();
    let mut work: Vec<InstId> = alive.iter().copied().collect();
    while let Some(n) = work.pop() {
        m.tick(1)?;
        if !alive.contains(&n) {
            continue;
        }
        let kind = node.get(&n).copied();
        let own = uses.get(&n).map(Vec::as_slice).unwrap_or_default();
        // A word's low word is the index only modulo 2³²: a wide pointer's 64-bit offset needs
        // the whole value. An extension there is still extended by its form.
        let uses_ok = own.iter().all(|u| match u {
            Use::Other => false,
            Use::Word(w) => alive.contains(w),
            Use::Index { wide } => !(*wide && kind == Some(Node::Word)),
        });
        let ops = if kind == Some(Node::Word) {
            operands(n)
        } else {
            Vec::new()
        };
        let ops_ok = ops.iter().all(|d| alive.contains(d));
        if uses_ok && ops_ok {
            continue;
        }
        alive.remove(&n);
        work.extend(own.iter().filter_map(|u| match u {
            Use::Word(w) => Some(*w),
            _ => None,
        }));
        work.extend(ops.iter().filter(|d| node.contains_key(d)));
    }
    let mut out = IndexValues::default();
    for id in alive {
        match node.get(&id) {
            Some(Node::Extension) => out.extensions.insert(id),
            Some(Node::Word) => out.words.insert(id),
            None => false,
        };
    }
    Ok(out)
}

/// Compute the function's index extensions and index words, metered. `select` calls this once per
/// function before selecting any instruction, so every query below reads the planned sets, and so
/// does [`read_value`] for the block-crossing values.
pub fn plan_extensions(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    let values = index_values(&cx.entry.module, cx.entry.func, cx.body, cx.memory, m)?;
    let access = &mut cx.families.access;
    access.index_extensions = values.extensions;
    access.index_words = values.words;
    Ok(())
}

/// The value a use of `v` reads: the narrow operand of a planned index extension, which has no
/// value of its own; `v` itself otherwise.
pub fn read_value(body: &Body, plan: &AccessPlan, v: Value) -> Value {
    let Value::Inst(d) = v else { return v };
    if !reads_nothing(plan, d) {
        return v;
    }
    match body.insts.get(d).map(|i| &i.kind) {
        Some(InstKind::Cast { v: src, .. }) => *src,
        _ => v,
    }
}

/// Whether instruction `at` is a planned index extension or a `ptrtoint` whose bits go only into
/// forwarding cells (`AccessPlan::cell_bits`): it emits nothing, so it reads its operand nowhere,
/// and a use of it reads that operand ([`read_value`]).
pub fn reads_nothing(plan: &AccessPlan, at: InstId) -> bool {
    plan.index_extensions.contains(&at) || plan.cell_bits.contains(&at)
}

/// Whether instruction `at` is an index extension ([`plan_extensions`] computed them).
pub fn is_index_extension(cx: &SelectCx<'_>, at: InstId) -> bool {
    cx.families.access.index_extensions.contains(&at)
}

/// Whether instruction `at` is an index word (planned with the index extensions).
pub fn is_index_word(cx: &SelectCx<'_>, at: InstId) -> bool {
    cx.families.access.index_words.contains(&at)
}

/// The lowered representation of `v` when it is a planned index word: its low word, a `u32`.
/// `select` asks this for the block-crossing values after [`plan_extensions`].
pub fn word_repr(plan: &AccessPlan, v: Value) -> Option<DemoteTy> {
    let Value::Inst(d) = v else { return None };
    plan.index_words
        .contains(&d)
        .then_some(DemoteTy::Scalar(ScalarTy::Int { bits: 32 }))
}

/// The narrow operand and index form of GEP index `v` when `v` is an index extension: the GEP
/// reads the extension's operand, whose low word, extended as the form says, is `v mod 2³²`. An
/// index word is read as itself, held as its low word.
pub fn narrowed(cx: &mut SelectCx<'_>, v: Value) -> Option<(Value, IndexForm)> {
    let Value::Inst(d) = v else { return None };
    if is_index_word(cx, d) {
        return Some((v, IndexForm::WordZero));
    }
    if !is_index_extension(cx, d) {
        return None;
    }
    let InstKind::Cast { op, v: src } = &cx.body.insts.get(d)?.kind else {
        return None;
    };
    let bits = match cx
        .value_ty(*src)
        .and_then(|t| cx.entry.module.types().get(t))
    {
        Some(Type::Int(w)) => *w,
        _ => return None,
    };
    let form = match (op, bits) {
        (CastOp::SExt, 1) => IndexForm::Bool,
        (CastOp::ZExt, 1) => IndexForm::BoolZero,
        (CastOp::SExt, 2..=31) => IndexForm::Narrow { bits },
        // Held zero-extended in a `u32`: the container is the zero extension.
        (CastOp::ZExt, 2..=32) => IndexForm::WordZero,
        (CastOp::SExt, 32) => IndexForm::Word,
        _ => return None,
    };
    Some((*src, form))
}

/// One operand of an index word: a value read as its index word by `form`, or a constant's low
/// word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordOperand {
    Value { v: ValueId, form: IndexForm },
    Const(u32),
}

/// An index word: `op` on the low words of `a` and `b`, which is the low word of the 64-bit result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexWord {
    pub op: BinOp,
    pub a: WordOperand,
    pub b: WordOperand,
}

/// The index word strategy of instruction `inst` (a planned index word).
pub fn select_word(cx: &mut SelectCx<'_>, inst: &Inst) -> Result<IndexWord, Error> {
    let InstKind::BinOp { op, a, b, .. } = &inst.kind else {
        return Err(cx.invariant("an index word that is no binary operator"));
    };
    let op = match op {
        AirBin::Add => BinOp::IAdd,
        AirBin::Sub => BinOp::ISub,
        AirBin::Mul => BinOp::IMul,
        AirBin::Shl => BinOp::ShiftLeftLogical,
        AirBin::And => BinOp::BitwiseAnd,
        AirBin::Or => BinOp::BitwiseOr,
        AirBin::Xor => BinOp::BitwiseXor,
        _ => return Err(cx.invariant("an index word with a non-word operator")),
    };
    let a = word_operand(cx, *a)?;
    let b = word_operand(cx, *b)?;
    Ok(IndexWord { op, a, b })
}

fn word_operand(cx: &mut SelectCx<'_>, v: Value) -> Result<WordOperand, Error> {
    if let Some(k) = const_word(&cx.entry.module, v) {
        return Ok(WordOperand::Const(k));
    }
    let (read, form) = narrowed(cx, v)
        .ok_or_else(|| cx.invariant("an index word operand that is no index value"))?;
    let id = cx
        .values
        .id(read)
        .ok_or_else(|| cx.invariant("an index word operand out of the value range"))?;
    Ok(WordOperand::Value { v: id, form })
}

/// Emit index word `w`: its value, a `u32`.
pub fn emit_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    w: &IndexWord,
) -> BResult<SValue> {
    let a = read_word(cx, fb, w.a)?;
    let b = read_word(cx, fb, w.b)?;
    fb.binary(w.op, a, b)
}

fn read_word(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, o: WordOperand) -> BResult<SValue> {
    match o {
        WordOperand::Const(k) => k32(fb, k),
        WordOperand::Value { v, form } => {
            let x = cx.operand(fb, v)?;
            index_u32(fb, x, form)
        }
    }
}
