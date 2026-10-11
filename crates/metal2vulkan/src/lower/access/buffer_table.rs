//! Buffer tables.
//!
//! A buffer table is a buffer or argument-buffer role passed as an `array_ref` of `count`
//! buffers (`memory::provenance::buffer_table`): its LLVM parameter is a thread-space pointer to
//! the element pointers, and element `k` is the buffer bound to Metal slot `first + k`. The
//! memory plan models the table as read only, one whole element pointer at a time, and gives each
//! accessed element its own slot (`Root::BufferElement`, `MemoryPlan::slot_of_root`).
//!
//! **Table pointers** (`Repr::BufferTable`): the parameter and every GEP, cast, `freeze`, or
//! `select` of it are `u32` byte offsets into the table, demoted as `DemoteTy::ArenaOffset`. The
//! parameter is offset 0; a GEP adds its datalayout offsets with the buffer family's index rule
//! (`access::gep_offset`).
//!
//! **Element loads.** A load of an element pointer through a table pointer yields the static fat
//! pointer `uvec2(2 + first + k, 0)` of element `k` (the slot's root number). The candidate
//! elements are the load's roots in the memory plan (a constant offset names one element, a range
//! the elements it overlaps within the table, an unknown offset every element; a read wholly
//! outside the table is null):
//!
//! - no element: the null pointer `uvec2(0, 0)`;
//! - one element: its constant fat pointer (every other offset reads outside the table, which
//!   Metal leaves undefined);
//! - several (a contiguous range `lo..=hi`): the element index `offset / size` is **clamped** into
//!   `lo..=hi` (unsigned, so a negative offset clamps to `hi`), which keeps an out-of-range
//!   dynamic index on an element the function binds: containment, not a new behavior of a
//!   defined read. The division is exact because the load's `align` is at least the element
//!   pointer's size (LLVM makes a misaligned load undefined, and the table, an array of
//!   pointers, starts at a pointer-aligned address); a load with a smaller alignment would read
//!   parts of two pointers and is not implemented.
//!
//! A table pointer parked in a forwarding cell (`access::cell`) is its `u32` offset there. Every
//! other use of a table (stores, atomics, a table pointer that escapes or merges with another root)
//! is refused by the memory analysis already.

use crate::air::abi::{AbiInt, ParamBinding};
use crate::air::ir::{CastOp, Inst, InstId, InstKind, Value, ValueId};
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::k32;
use crate::memory::provenance::{self, Repr};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, Value as SValue};
use crate::spirv::ir::{BinOp, Type as SType};

use super::emit::index_u32;
use super::root_sets::TableSet;
use super::{is_ptr_ty, operand, refuse, slot_root, GepTerms, NULL_ROOT};

/// The element pointer a table load yields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Element {
    /// A constant root: the null pointer (`NULL_ROOT`) or one element's slot root.
    Root(u32),
    /// The clamped element of a dynamic offset.
    Dynamic {
        /// The table pointer.
        ptr: ValueId,
        /// The element pointer's size in bytes.
        size: u32,
        /// The root number of element `lo`, and the element range the load may read.
        root_lo: u32,
        lo: u32,
        hi: u32,
    },
}

/// The buffer-table strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TableStrategy {
    /// `getelementptr` on a table offset: `base + konst + Σ index · stride` (wrapping `u32`).
    Gep {
        base: ValueId,
        konst: u32,
        terms: GepTerms,
    },
    /// `bitcast`, `addrspacecast`, or `freeze` of a table pointer.
    Copy { of: ValueId },
    /// `select c, t, f` of two pointers into one table.
    Select { c: ValueId, t: ValueId, f: ValueId },
    /// A load of an element pointer.
    Load(Element),
}

impl TableStrategy {
    /// The census name (`access.buffer_table.<op>`).
    pub fn name(&self) -> &'static str {
        match self {
            TableStrategy::Gep { .. } => "access.buffer_table.gep",
            TableStrategy::Copy { .. } => "access.buffer_table.pointer_copy",
            TableStrategy::Select { .. } => "access.buffer_table.pointer_select",
            TableStrategy::Load(Element::Root(_)) => "access.buffer_table.load",
            TableStrategy::Load(Element::Dynamic { .. }) => "access.buffer_table.load.clamped",
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        match self {
            TableStrategy::Gep { terms, .. } => {
                12u32.saturating_add(16u32.saturating_mul(terms.len() as u32))
            }
            TableStrategy::Copy { .. } => 2,
            TableStrategy::Select { .. } => 6,
            TableStrategy::Load(Element::Root(_)) => 8,
            TableStrategy::Load(Element::Dynamic { .. }) => 48,
        }
    }
}

/// The buffer-table parameter pointer value `v` points into, if it is a table pointer.
fn table_of(cx: &SelectCx<'_>, v: Value) -> Option<u32> {
    match cx.memory.repr(v)? {
        Repr::BufferTable { param } => Some(param),
        _ => None,
    }
}

/// The table pointer operand `v` of a pointer into table `param`.
fn table_operand(cx: &SelectCx<'_>, v: Value, param: u32) -> Result<ValueId, Error> {
    if table_of(cx, v) != Some(param) {
        return Err(refuse());
    }
    operand(cx, v)
}

/// The lowered representation of a table pointer (a `u32` offset), `None` for any other value.
pub fn lowered_pointer(cx: &SelectCx<'_>, v: Value) -> Option<DemoteTy> {
    table_of(cx, v).map(|_| DemoteTy::ArenaOffset)
}

/// Whether parameter `b` is a buffer table.
pub fn is_table_param(e: &crate::air::Entry, b: &ParamBinding) -> bool {
    provenance::buffer_table_of(e, b).is_some()
}

/// The value of a buffer-table parameter: offset 0 into its table.
pub fn param_value(fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    k32(fb, 0)
}

/// The element pointer that load `at` through table pointer `ptr` into table `param` yields.
fn element(
    cx: &SelectCx<'_>,
    at: InstId,
    ptr: Value,
    param: u32,
    align: Option<crate::air::ir::Align>,
) -> Result<Element, Error> {
    let t = cx.memory.abi.buffer_table(param).ok_or_else(refuse)?;
    let AbiInt::Const(first) = t.first else {
        return Err(refuse());
    };
    let Some(Repr::BufPtr { roots, wide: false }) = cx.memory.repr(Value::Inst(at)) else {
        return Err(refuse());
    };
    // The elements the set names, read once for the function (`root_sets`). A read wholly outside
    // the table is the null pointer alone.
    let set = cx
        .root_sets
        .get(roots)
        .ok_or_else(|| cx.invariant("a table load's root set was not read"))?;
    let (lo, hi, count) = match set.table {
        TableSet::Null => return Ok(Element::Root(NULL_ROOT)),
        TableSet::Elements {
            param: p,
            lo,
            hi,
            count,
        } if p == param => (lo, hi, count),
        _ => return Err(cx.invariant("a table load with a root outside its table")),
    };
    let root = |k: u32| first.checked_add(k).map(slot_root).ok_or_else(refuse);
    match count {
        1 => Ok(Element::Root(root(lo)?)),
        _ => {
            // The set is sorted (`RootSets::intern`), so `lo` is its first element and `hi` its
            // last.
            let span = hi
                .checked_sub(lo)
                .ok_or_else(|| cx.invariant("a table load's elements out of order"))?;
            if u64::from(span) + 1 != count as u64 {
                return Err(cx.invariant("a table load over non-contiguous elements"));
            }
            let ty = cx
                .body
                .insts
                .get(at)
                .map(|i| i.ty)
                .ok_or_else(|| cx.invariant("a table load without a type"))?;
            let size = cx
                .memory
                .layouts
                .store_size(ty)
                .and_then(|s| u32::try_from(s).ok())
                .filter(|s| *s > 0)
                .ok_or_else(refuse)?;
            // The division by the element size is exact only for an aligned load.
            if align.map_or(1, |a| a.bytes()) < u64::from(size) {
                return Err(refuse());
            }
            Ok(Element::Dynamic {
                ptr: operand(cx, ptr)?,
                size,
                root_lo: root(lo)?,
                lo,
                hi,
            })
        }
    }
}

/// Select the buffer-table strategy of `inst`, or `None` when it involves no table pointer.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
) -> Result<Option<TableStrategy>, Error> {
    let here = Value::Inst(at);
    Ok(Some(match &inst.kind {
        InstKind::Gep {
            src_ty,
            base,
            indices,
            ..
        } => {
            let Some(param) = table_of(cx, here) else {
                return Ok(None);
            };
            let base = table_operand(cx, *base, param)?;
            let (konst, terms) = super::gep_offset(cx, *src_ty, indices)?;
            TableStrategy::Gep { base, konst, terms }
        }
        InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v,
        }
        | InstKind::Freeze(v)
            if is_ptr_ty(cx, inst.ty) =>
        {
            let Some(param) = table_of(cx, here) else {
                return Ok(None);
            };
            TableStrategy::Copy {
                of: table_operand(cx, *v, param)?,
            }
        }
        InstKind::Select { c, t, f, .. } if is_ptr_ty(cx, inst.ty) => {
            let Some(param) = table_of(cx, here) else {
                return Ok(None);
            };
            TableStrategy::Select {
                c: operand(cx, *c)?,
                t: table_operand(cx, *t, param)?,
                f: table_operand(cx, *f, param)?,
            }
        }
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } => {
            let Some(param) = table_of(cx, *ptr) else {
                return Ok(None);
            };
            TableStrategy::Load(element(cx, at, *ptr, param, *align)?)
        }
        _ => {
            // Any other instruction on a table pointer has no model.
            let mut touches = false;
            inst.kind
                .for_each_value(|v| touches |= table_of(cx, v).is_some());
            if touches || table_of(cx, here).is_some() {
                return Err(refuse());
            }
            return Ok(None);
        }
    }))
}

/// Emit table strategy `s`, defining value `vid`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &TableStrategy,
) -> BResult<()> {
    let r = match s {
        TableStrategy::Gep { base, konst, terms } => {
            let mut off = cx.operand(fb, *base)?;
            if *konst != 0 {
                let c = k32(fb, *konst)?;
                off = fb.binary(BinOp::IAdd, off, c)?;
            }
            for (v, stride, form) in terms {
                let x = cx.operand(fb, *v)?;
                let mut x = index_u32(fb, x, *form)?;
                if *stride != 1 {
                    let s = k32(fb, *stride)?;
                    x = fb.binary(BinOp::IMul, x, s)?;
                }
                off = fb.binary(BinOp::IAdd, off, x)?;
            }
            off
        }
        TableStrategy::Copy { of } => cx.operand(fb, *of)?,
        TableStrategy::Select { c, t, f } => {
            let c = cx.operand(fb, *c)?;
            let t = cx.operand(fb, *t)?;
            let f = cx.operand(fb, *f)?;
            fb.select(c, t, f)?
        }
        TableStrategy::Load(e) => {
            let root = match e {
                Element::Root(r) => k32(fb, *r)?,
                Element::Dynamic {
                    ptr,
                    size,
                    root_lo,
                    lo,
                    hi,
                } => {
                    let off = cx.operand(fb, *ptr)?;
                    let size = k32(fb, *size)?;
                    let k = fb.binary(BinOp::UDiv, off, size)?;
                    let (lo, hi) = (k32(fb, *lo)?, k32(fb, *hi)?);
                    let below = fb.binary(BinOp::ULessThan, k, lo)?;
                    let k = fb.select(below, lo, k)?;
                    let above = fb.binary(BinOp::ULessThan, hi, k)?;
                    let k = fb.select(above, hi, k)?;
                    // Element `k` is root `root_lo + (k - lo)`.
                    let delta = fb.binary(BinOp::ISub, k, lo)?;
                    let base = k32(fb, *root_lo)?;
                    fb.binary(BinOp::IAdd, base, delta)?
                }
            };
            let zero = k32(fb, 0)?;
            let u = fb.module().ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let v2 = fb.module().ty(SType::Vector { elem: u, len: 2 })?;
            fb.composite_construct(v2, &[root, zero])?
        }
    };
    cx.define(vid, r);
    Ok(())
}
