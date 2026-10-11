//! The attributes the IR keeps. The decoder recognizes every attribute kind in LLVM's closed table.
//! The ones that can matter downstream are decoded into named fields; every other enum attribute
//! means nothing to the translation, and the IR does not carry it (a decode asked for them records
//! them beside the module, for the validation AIR writer that must hand them back to Metal's
//! compiler: `bitcode::EnumAttrTable`). Integer, type, and constant-range attributes outside the
//! named fields are advisory and dropped at decode.

use super::id::{Sym, TypeId};
use super::inst::{no_heap, Align};

/// Function attributes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FnAttrs {
    /// `convergent`: a call must not be made control-dependent on more values than it already is.
    pub convergent: bool,
    /// `memory(...)`, or the older `readnone`/`readonly`/`writeonly` spelled as memory effects.
    pub memory: Option<MemoryEffects>,
    /// String attributes as `(key, value)`; `value` is empty for key-only attributes.
    pub strings: Box<[(Sym, Sym)]>,
}

impl FnAttrs {
    /// The heap bytes of the string attributes. Every field is named, so a new one must be counted
    /// or shown to be held inline (`no_heap`).
    pub fn heap_bytes(&self) -> usize {
        let FnAttrs {
            convergent,
            memory,
            strings,
        } = self;
        no_heap(convergent)
            .saturating_add(no_heap(memory))
            .saturating_add(std::mem::size_of_val::<[(Sym, Sym)]>(strings))
    }
}

/// Parameter and return-value attributes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ParamAttrs {
    /// `align(N)`: the alignment the pointer is known to have.
    pub align: Option<Align>,
    /// `dereferenceable(N)`: the number of bytes known dereferenceable through the pointer.
    pub dereferenceable: Option<u64>,
    /// `dereferenceable_or_null(N)`: as `dereferenceable`, unless the pointer is null.
    pub dereferenceable_or_null: Option<u64>,
    /// `noalias`: memory reached through the pointer is not reached through any pointer not
    /// derived from it.
    pub noalias: bool,
    /// `readnone` (`NoModRef`), `readonly` (`Ref`), or `writeonly` (`Mod`).
    pub access: Option<ModRef>,
    /// `byval(T)`.
    pub byval: Option<TypeId>,
    /// `sret(T)`.
    pub sret: Option<TypeId>,
    /// `nocapture` or `captures(none)`.
    pub nocapture: bool,
    /// String attributes as `(key, value)`; `value` is empty for key-only attributes.
    pub strings: Box<[(Sym, Sym)]>,
}

impl ParamAttrs {
    /// The heap bytes of the string attributes. Every field is named, so a new one must be counted
    /// or shown to be held inline (`no_heap`).
    pub fn heap_bytes(&self) -> usize {
        let ParamAttrs {
            align,
            dereferenceable,
            dereferenceable_or_null,
            noalias,
            access,
            byval,
            sret,
            nocapture,
            strings,
        } = self;
        [
            no_heap(align),
            no_heap(dereferenceable),
            no_heap(dereferenceable_or_null),
            no_heap(noalias),
            no_heap(access),
            no_heap(byval),
            no_heap(sret),
            no_heap(nocapture),
        ]
        .into_iter()
        .fold(
            std::mem::size_of_val::<[(Sym, Sym)]>(strings),
            usize::saturating_add,
        )
    }

    /// The types these attributes mention.
    pub fn types(&self) -> impl Iterator<Item = TypeId> + '_ {
        self.byval.iter().chain(self.sret.iter()).copied()
    }

    /// The symbols these attributes mention.
    pub fn syms(&self) -> impl Iterator<Item = Sym> + '_ {
        self.strings.iter().flat_map(|&(k, v)| [k, v])
    }
}

/// Call-site attributes.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct CallAttrs {
    /// The function attributes at the call site.
    pub func: FnAttrs,
    /// The return-value attributes.
    pub ret: ParamAttrs,
    /// One entry per argument position that carries attributes; missing trailing positions have
    /// none.
    pub params: Box<[ParamAttrs]>,
}

impl CallAttrs {
    /// The heap bytes the attributes hold beyond their own box: the string attributes and the
    /// per-argument list with each entry's strings. Every field is named, so a new one must be
    /// counted.
    pub fn heap_bytes(&self) -> usize {
        let CallAttrs { func, ret, params } = self;
        params.iter().fold(
            func.heap_bytes()
                .saturating_add(ret.heap_bytes())
                .saturating_add(std::mem::size_of_val::<[ParamAttrs]>(params)),
            |sum, p| sum.saturating_add(p.heap_bytes()),
        )
    }
}

/// Which accesses a location kind may see: LLVM's `ModRefInfo`, whose variant names the IR
/// printouts carry (`memory(Mod, NoModRef, NoModRef)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModRef {
    /// Neither reads nor writes.
    NoModRef,
    /// May read.
    Ref,
    /// May write.
    Mod,
    /// May read and write.
    ModRef,
}

/// `memory(...)` effects per location kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryEffects {
    /// Memory reached through pointer arguments (`argmem`).
    pub argmem: ModRef,
    /// Memory the module cannot otherwise reach (`inaccessiblemem`).
    pub inaccessible: ModRef,
    /// Every other location kind, unioned.
    pub other: ModRef,
}

impl MemoryEffects {
    /// The same effect on every location kind (`readnone`, `readonly`, `writeonly`).
    pub const fn uniform(m: ModRef) -> MemoryEffects {
        MemoryEffects {
            argmem: m,
            inaccessible: m,
            other: m,
        }
    }
}
