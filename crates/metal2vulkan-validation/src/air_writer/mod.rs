//! A validation-only AIR text writer: decode-level IR to typed-pointer LLVM text that
//! `xcrun metal-as` assembles.
//!
//! The Metal oracle runs cyclic cases only through fuel-instrumented AIR; the
//! instrumentation rewrites decoded IR, and this writer turns the result back into text for
//! `metal-as`/`metallib`. It is never on a product path, and it is not the diagnostic printer.
//!
//! # Contract
//!
//! [`write()`] spells a verified [`Module`], the typed-pointer pointees of its functions'
//! parameters and return values ([`Pointees`], from [`Decoded::pointees`]), and the enum
//! attributes the IR does not carry ([`EnumAttrTable`], from [`Decoded::enum_attrs`] of a decode
//! that kept them, [`decode_as_written`]). The IR keeps only
//! each pointer's address space, so every other pointee is inferred from how values are used
//! (`terms`, `infer`): a load's pointer points at the loaded type, a GEP's base at its source
//! element type, an argument is its callee's parameter type, and so on. For a module decoded from
//! typed-pointer bitcode the original typing satisfies every one of these equations, so the
//! writer finds a typing and the text reassembles into a module that decodes to the same IR:
//! decode → [`write()`] → `metal-as` → decode is the identity on everything the IR keeps
//! (see the round-trip tests). A pointee no use constrains is spelled `i8`.
//!
//! Every enum attribute reaches the writer (the ones the IR interprets in its named fields, the
//! rest in the [`EnumAttrTable`] a decode keeps beside it on request), with each global's and
//! function's visibility and `unnamed_addr`, and the writer spells them all back. They do not change what a program computes, but Metal's compiler relies
//! on them, so the module the oracle hands it must carry them as written: it deletes an unused call
//! to a `readnone` function only when the call is `nounwind` and the callee `willreturn`, and a
//! dead call to a `"deferred-static-alloca-size"` function that survives makes pipeline creation
//! fail. An enum kind newer than the LLVM dialect `metal-as` parses is refused
//! ([`Unspellable::AttributeKind`]), not dropped.
//!
//! What the IR does not keep is not written: integer, type, and constant-range attributes outside
//! the named fields (`uwtable(sync)`, `range(...)`, ...), dropped attachments (`!tbaa`,
//! `!alias.scope`, ...), value names, and markers (`llvm.lifetime.*`, `llvm.assume`). None of them
//! changes what a program computes; that Metal's compiler relies on none of them is not
//! established.
//!
//! Every construct the typed-pointer text cannot spell is a typed [`WriteError`], never a guess:
//! an opaque-pointer module whose uses admit no typed-pointer typing
//! ([`WriteError::PointeeConflict`]), memory effects outside the pre-`memory(...)` spellings,
//! debug metadata, and unnamed globals or opaque structs ([`Unspellable`]).

mod assemble;
mod infer;
mod terms;
#[cfg(test)]
pub(crate) mod tests;
mod text;

use std::fmt;

use metal2vulkan_air::bitcode::{decode_with, DecodeOptions, Decoded, EnumAttrTable, PointeeTable};
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::ir::{FuncId, Idx, Module, TypeId};

pub use assemble::{metal_as, AssembleError};

/// A construct the typed-pointer text cannot spell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unspellable {
    /// Memory effects with no pre-`memory(...)` spelling (e.g. `memory(argmem: read, other:
    /// write)`).
    MemoryEffects,
    /// A reachable debug-info record: the IR keeps no contents for it.
    DebugMetadata,
    /// A global without a name.
    UnnamedGlobal,
    /// An identified struct without a name.
    UnnamedOpaqueStruct,
    /// A pointee that would have to contain itself.
    InfinitePointee,
    /// `byval`/`sret` on a return value.
    ReturnTypeAttribute,
    /// An enum attribute kind with no spelling in the LLVM dialect `metal-as` parses (one newer
    /// than its LLVM, such as LLVM 23's `noipa`). Dropping it would hand Metal a different module.
    AttributeKind(u32),
}

/// Why a module could not be written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WriteError {
    /// The module has a construct the typed-pointer text cannot spell.
    Unsupported(Unspellable),
    /// No typed-pointer typing satisfies the module's uses (only possible for input that was
    /// not typed-pointer bitcode, such as an opaque-pointer module using one pointer at two
    /// pointee types).
    PointeeConflict { site: String, detail: String },
    /// An id the verified module should not contain.
    Malformed(String),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::Unsupported(u) => write!(f, "unsupported by the AIR writer: {u:?}"),
            WriteError::PointeeConflict { site, detail } => {
                write!(f, "no typed-pointer typing at {site}: {detail}")
            }
            WriteError::Malformed(what) => write!(f, "malformed module: {what}"),
        }
    }
}

impl std::error::Error for WriteError {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FnPointees {
    ret: Option<TypeId>,
    params: Vec<Option<TypeId>>,
}

/// The typed-pointer pointee of each function's parameters and return value, as erased IR
/// types: from the decoder's [`PointeeTable`], editable for a rewritten module (a parameter an
/// instrumentation adds, say). `None` where the pointee is unknown (opaque-pointer input) or the
/// value is not a pointer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pointees {
    funcs: Vec<FnPointees>,
}

impl Pointees {
    /// The pointees `table` records for every function of `module`.
    pub fn from_table(table: &PointeeTable, module: &Module) -> Pointees {
        let funcs = module
            .parts()
            .funcs
            .ids()
            .map(|f| FnPointees {
                ret: table.ret(f),
                params: table.params(f).map(<[_]>::to_vec).unwrap_or_default(),
            })
            .collect();
        Pointees { funcs }
    }

    /// The pointee of parameter `index` of `func`.
    pub fn param(&self, func: FuncId, index: usize) -> Option<TypeId> {
        self.funcs
            .get(func.index())?
            .params
            .get(index)
            .copied()
            .flatten()
    }

    /// The pointee of `func`'s return value.
    pub fn ret(&self, func: FuncId) -> Option<TypeId> {
        self.funcs.get(func.index())?.ret
    }

    fn slot(&mut self, func: FuncId) -> &mut FnPointees {
        if self.funcs.len() <= func.index() {
            self.funcs.resize(func.index() + 1, FnPointees::default());
        }
        &mut self.funcs[func.index()]
    }

    /// Record the pointee of parameter `index` of `func`.
    pub fn set_param(&mut self, func: FuncId, index: usize, pointee: Option<TypeId>) {
        let slot = self.slot(func);
        if slot.params.len() <= index {
            slot.params.resize(index + 1, None);
        }
        slot.params[index] = pointee;
    }

    /// Record the pointee of `func`'s return value.
    pub fn set_ret(&mut self, func: FuncId, pointee: Option<TypeId>) {
        self.slot(func).ret = pointee;
    }
}

/// Decode `bitcode` for this writer: keeping the enum attributes the IR does not carry, which the
/// module as written includes ([`DecodeOptions::enum_attrs`]).
pub fn decode_as_written(
    bitcode: &[u8],
    m: &mut Meter,
) -> Result<Decoded, metal2vulkan::error::Error> {
    let opts = DecodeOptions {
        enum_attrs: true,
        ..DecodeOptions::default()
    };
    decode_with(bitcode, opts, m).map_err(metal2vulkan::error::Error::from)
}

/// The typed-pointer text of `module` with the function pointees `pointees` and the enum
/// attributes `enums`.
pub fn write(
    module: &Module,
    pointees: &Pointees,
    enums: &EnumAttrTable,
) -> Result<String, WriteError> {
    let mut terms = terms::Terms::new(module.types(), module.symbols());
    let typing = infer::infer(module, pointees, &mut terms)?;
    let next_md = u32::try_from(module.parts().metadata.len())
        .map_err(|_| WriteError::Malformed("metadata count".into()))?;
    let mut p = text::Printer {
        m: module,
        enums,
        t: &mut terms,
        typing: &typing,
        out: String::new(),
        extra_md: Vec::new(),
        next_md,
    };
    p.module()?;
    Ok(p.out)
}

/// The typed-pointer text of a decoded module (its own pointees and enum attributes; a module
/// decoded for translation has none of the latter, see [`decode_as_written`]).
pub fn write_decoded(d: &Decoded) -> Result<String, WriteError> {
    write(
        &d.module,
        &Pointees::from_table(&d.pointees, &d.module),
        &d.enum_attrs,
    )
}
