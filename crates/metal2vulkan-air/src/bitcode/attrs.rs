//! `PARAMATTR_GROUP` and `PARAMATTR` blocks: attribute groups and the lists that reference them.
//!
//! Every attribute kind code is checked against LLVM's closed table (`AttributeKindCodes`, 1 to
//! 109 as of LLVM 23.1); an unknown code is an error, because a new kind can carry semantics. Kinds
//! that change the calling ABI and that AIR never uses (`inalloca`, `preallocated`, `byref`,
//! `nest`, the `swift*` kinds) are refused. The kinds the IR interprets are decoded into the named
//! fields of [`FnAttrs`] and [`ParamAttrs`]; every other kind in enum form is read into an
//! [`EnumAttrs`] set per position, which a decode asked for them records by function and call site
//! ([`SiteEnums`]) so a writer can hand the module back as written, and which is otherwise left
//! behind; every other integer, type, or constant-range attribute is advisory and dropped.
//! `denormal_fpenv` (LLVM 23) is dropped after its value is checked: the IR carries no
//! floating-point environment, whose denormal handling is the device's, like the
//! `"denormal-fp-math"` string attribute it replaces.
//!
//! Type attributes name bitcode types, so a list is resolved against the type table (and, for the
//! older untyped `byval`/`sret` spelling, against the pointee of the parameter's typed pointer)
//! when it is attached to a function or call site.

use std::collections::HashMap;

use super::codes::*;
use super::enum_attrs::{EnumAttrs, SiteEnums};
use super::stream::{chars, Stream};
use super::types::{grow, BitcodeTypes};
use crate::budget::Meter;
use crate::error::BitcodeError;
use crate::error::{Error, Feature};
use crate::ir::{Align, FnAttrs, MemoryEffects, ModRef, ParamAttrs, Sym, Symbols, TypeId};

/// `paramidx` of a group that applies to the function itself.
const FUNCTION_INDEX: u64 = 0xFFFF_FFFF;
/// `paramidx` of a group that applies to the return value.
const RETURN_INDEX: u64 = 0;
/// The largest attribute kind code LLVM defines (`ATTR_KIND_NOIPA`, LLVM 23.1).
const MAX_KIND: u64 = 109;
// Every known kind fits an [`EnumAttrs`].
const _: () = assert!(MAX_KIND <= EnumAttrs::MAX_KIND as u64);

// Attribute kinds the decoder interprets.
const KIND_ALIGNMENT: u64 = 1;
const KIND_BY_VAL: u64 = 3;
const KIND_NO_ALIAS: u64 = 9;
const KIND_NO_CAPTURE: u64 = 11;
const KIND_READ_NONE: u64 = 20;
const KIND_READ_ONLY: u64 = 21;
const KIND_STRUCT_RET: u64 = 29;
const KIND_DEREFERENCEABLE: u64 = 41;
const KIND_DEREFERENCEABLE_OR_NULL: u64 = 42;
const KIND_CONVERGENT: u64 = 43;
const KIND_ARGMEMONLY: u64 = 45;
const KIND_INACCESSIBLEMEM_ONLY: u64 = 49;
const KIND_INACCESSIBLEMEM_OR_ARGMEMONLY: u64 = 50;
const KIND_WRITEONLY: u64 = 52;
const KIND_MEMORY: u64 = 86;
const KIND_CAPTURES: u64 = 102;
const KIND_DENORMAL_FPENV: u64 = 106;
/// NEST, IN_ALLOCA, SWIFT_SELF, SWIFT_ERROR, PREALLOCATED, BYREF, SWIFT_ASYNC.
const KIND_REFUSED: &[u64] = &[8, 38, 46, 47, 65, 69, 75];

/// A type named by a type attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TyRef {
    /// A bitcode type index.
    Explicit(u64),
    /// The older spelling without a type: the pointee of the parameter's typed pointer.
    Pointee,
}

/// Where a function-level memory restriction applies (the pre-`memory(...)` spellings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemLoc {
    Arg,
    Inaccessible,
    InaccessibleOrArg,
}

/// The attributes of one group (one position), before type resolution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Decoded {
    align: Option<Align>,
    dereferenceable: Option<u64>,
    dereferenceable_or_null: Option<u64>,
    noalias: bool,
    nocapture: bool,
    convergent: bool,
    access: Option<ModRef>,
    loc: Option<MemLoc>,
    memory: Option<MemoryEffects>,
    byval: Option<TyRef>,
    sret: Option<TyRef>,
    strings: Vec<(Sym, Sym)>,
    enums: EnumAttrs,
}

impl Decoded {
    /// Merge `other` (a later group for the same position) into `self`.
    fn merge(&mut self, other: &Decoded, m: &mut Meter) -> Result<(), Error> {
        self.align = other.align.or(self.align);
        self.dereferenceable = other.dereferenceable.or(self.dereferenceable);
        self.dereferenceable_or_null = other
            .dereferenceable_or_null
            .or(self.dereferenceable_or_null);
        self.noalias |= other.noalias;
        self.nocapture |= other.nocapture;
        self.convergent |= other.convergent;
        self.access = match (self.access, other.access) {
            (Some(a), Some(b)) => Some(intersect(a, b)),
            (a, b) => a.or(b),
        };
        self.loc = other.loc.or(self.loc);
        self.memory = other.memory.or(self.memory);
        self.byval = other.byval.or(self.byval);
        self.sret = other.sret.or(self.sret);
        self.enums = self.enums.union(other.enums);
        m.reserve(&mut self.strings, other.strings.len())?;
        self.strings.extend_from_slice(&other.strings);
        Ok(())
    }

    fn fn_attrs(&self, m: &mut Meter) -> Result<FnAttrs, Error> {
        let memory = self.memory.or_else(|| {
            if self.access.is_none() && self.loc.is_none() {
                return None;
            }
            let mr = self.access.unwrap_or(ModRef::ModRef);
            Some(match self.loc {
                None => MemoryEffects::uniform(mr),
                Some(MemLoc::Arg) => MemoryEffects {
                    argmem: mr,
                    inaccessible: ModRef::NoModRef,
                    other: ModRef::NoModRef,
                },
                Some(MemLoc::Inaccessible) => MemoryEffects {
                    argmem: ModRef::NoModRef,
                    inaccessible: mr,
                    other: ModRef::NoModRef,
                },
                Some(MemLoc::InaccessibleOrArg) => MemoryEffects {
                    argmem: mr,
                    inaccessible: mr,
                    other: ModRef::NoModRef,
                },
            })
        });
        Ok(FnAttrs {
            convergent: self.convergent,
            memory,
            strings: boxed(&self.strings, m)?,
        })
    }

    fn param_attrs(
        &self,
        types: &BitcodeTypes,
        pointee: Option<u32>,
        m: &mut Meter,
    ) -> Option<Result<ParamAttrs, Error>> {
        let resolve = |r: Option<TyRef>| -> Option<Option<TypeId>> {
            match r {
                None => Some(None),
                Some(TyRef::Explicit(i)) => types.ty(i).map(Some),
                Some(TyRef::Pointee) => pointee.and_then(|p| types.ty(u64::from(p))).map(Some),
            }
        };
        let byval = resolve(self.byval)?;
        let sret = resolve(self.sret)?;
        Some(boxed(&self.strings, m).map(|strings| ParamAttrs {
            align: self.align,
            dereferenceable: self.dereferenceable,
            dereferenceable_or_null: self.dereferenceable_or_null,
            noalias: self.noalias,
            access: self.access,
            byval,
            sret,
            nocapture: self.nocapture,
            strings,
        }))
    }
}

/// One `PARAMATTR` list: groups by position (function, return value, parameters).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AttrList {
    /// `(paramidx, attributes)` in increasing position order, one entry per position.
    positions: Vec<(u64, Decoded)>,
}

/// An attribute list resolved for one function or call site.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedAttrs {
    pub func: FnAttrs,
    pub ret: ParamAttrs,
    /// One entry per parameter position up to the last with attributes.
    pub params: Vec<ParamAttrs>,
    /// The uninterpreted enum kinds by position, when the resolution was asked for them.
    pub enums: SiteEnums,
}

impl AttrList {
    /// Resolve against the type table for a function or call site with `arity` parameters.
    /// `pointee(i)` is the bitcode pointee index of parameter `i`'s typed pointer, used by the
    /// untyped `byval`/`sret` spelling. With `keep_enums`, the uninterpreted enum kinds are
    /// collected by position (`ResolvedAttrs::enums`; empty otherwise). `None` when a position
    /// is outside the signature or a type attribute cannot be resolved.
    pub fn resolve(
        &self,
        types: &BitcodeTypes,
        arity: usize,
        pointee: impl Fn(usize) -> Option<u32>,
        keep_enums: bool,
        m: &mut Meter,
    ) -> Result<Option<ResolvedAttrs>, Error> {
        let mut out = ResolvedAttrs::default();
        for (idx, d) in &self.positions {
            match *idx {
                FUNCTION_INDEX => {
                    out.func = d.fn_attrs(m)?;
                    if keep_enums {
                        out.enums.func = d.enums;
                    }
                }
                RETURN_INDEX => match d.param_attrs(types, None, m) {
                    Some(r) => {
                        out.ret = r?;
                        if keep_enums {
                            out.enums.ret = d.enums;
                        }
                    }
                    None => return Ok(None),
                },
                i => {
                    let p = match usize::try_from(i - 1) {
                        Ok(p) if p < arity => p,
                        _ => return Ok(None),
                    };
                    let Some(attrs) = d.param_attrs(types, pointee(p), m) else {
                        return Ok(None);
                    };
                    let attrs = attrs?;
                    if out.params.len() <= p {
                        let more = p + 1 - out.params.len();
                        m.reserve(&mut out.params, more)?;
                        out.params.resize(p + 1, ParamAttrs::default());
                    }
                    if let Some(slot) = out.params.get_mut(p) {
                        *slot = attrs;
                    }
                    if keep_enums && !d.enums.is_empty() {
                        let enums = &mut out.enums.params;
                        if enums.len() <= p {
                            let more = p + 1 - enums.len();
                            m.reserve(enums, more)?;
                            enums.resize(p + 1, EnumAttrs::default());
                        }
                        if let Some(slot) = enums.get_mut(p) {
                            *slot = d.enums;
                        }
                    }
                }
            }
        }
        Ok(Some(out))
    }
}

/// The attribute groups of a module, by group id, and which attribute blocks were read.
///
/// LLVM's reader keys its groups by a 32-bit id and resolves a list's ids as it reads the list,
/// inserting every id it looks up. So a group id or position above `u32::MAX` (which LLVM would
/// truncate onto another), a group block after a list block, and a second block of either kind
/// (which LLVM refuses once the first had entries) are malformed here. A list then resolves its ids
/// against every group the module defines, and an id none defines names no attributes, as LLVM
/// reads it.
#[derive(Debug, Default)]
pub(crate) struct Groups {
    by_id: HashMap<u32, (u64, Decoded)>,
    /// A `PARAMATTR_GROUP` block was read.
    groups_read: bool,
    /// A `PARAMATTR` block was read.
    lists_read: bool,
}

/// A 32-bit group id or position, as LLVM keys and applies them; a wider one is malformed.
fn u32_op(s: &Stream<'_>, block: u32, code: u32, value: u64) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| s.malformed(block, code))
}

/// Decode a `PARAMATTR_GROUP` block that was just entered: the module's only one, before any
/// `PARAMATTR` block. An entry is `[id, position, attribute...]` with at least one attribute
/// operand: LLVM's reader refuses a shorter one ("Invalid grp record"), so it is malformed here
/// rather than read as an empty group.
pub(crate) fn decode_groups(
    s: &mut Stream<'_>,
    groups: &mut Groups,
    symbols: &mut Symbols,
    m: &mut Meter,
) -> Result<(), Error> {
    const B: u32 = PARAMATTR_GROUP_BLOCK;
    if groups.groups_read || groups.lists_read {
        return Err(s.malformed(B, 0));
    }
    groups.groups_read = true;
    s.records(m, B, |s, code, _, m| {
        if code != PARAMATTR_GRP_ENTRY {
            return Err(s.err(BitcodeError::UnknownRecord { block: B, code }));
        }
        if s.ops().len() < 3 {
            return Err(s.malformed(B, code));
        }
        let id = u32_op(s, B, code, s.op(B, code, 0)?)?;
        let idx = u64::from(u32_op(s, B, code, s.op(B, code, 1)?)?);
        let decoded = decode_group(s, symbols, m)?;
        m.charge(std::mem::size_of::<(u32, (u64, Decoded))>().saturating_mul(2))?;
        if groups.by_id.insert(id, (idx, decoded)).is_some() {
            return Err(s.malformed(B, code));
        }
        Ok(())
    })
}

/// Decode a `PARAMATTR` block that was just entered, appending one list per record: the module's
/// only one.
pub(crate) fn decode_lists(
    s: &mut Stream<'_>,
    groups: &mut Groups,
    lists: &mut Vec<AttrList>,
    m: &mut Meter,
) -> Result<(), Error> {
    const B: u32 = PARAMATTR_BLOCK;
    if groups.lists_read {
        return Err(s.malformed(B, 0));
    }
    groups.lists_read = true;
    let groups = &*groups;
    s.records(m, B, |s, code, _, m| {
        if code == PARAMATTR_ENTRY_OLD {
            return Err(s.unsupported(Feature::BitcodeRecord { block: B, code }));
        }
        if code != PARAMATTR_ENTRY {
            return Err(s.err(BitcodeError::UnknownRecord { block: B, code }));
        }
        // Resolve every group id, then stable-sort by position so the groups of one position form
        // one run in record order, and merge each run. Sorting keeps the record O(n log n) in its
        // operand count however many distinct positions it names.
        //
        // A group id no `PARAMATTR_GROUP` entry defined names no attributes: LLVM's reader looks
        // each id up in its group map with `operator[]`, which yields an empty attribute set, so
        // that is what the module means to every LLVM-based consumer. The group block, if any, was
        // read before this one ([`Groups`]).
        let ops = s.ops();
        let mut refs: Vec<(u64, &Decoded)> = Vec::new();
        m.reserve(&mut refs, ops.len())?;
        for &gid in ops {
            m.tick(1)?;
            let gid = u32_op(s, B, code, gid)?;
            if let Some((idx, d)) = groups.by_id.get(&gid) {
                refs.push((*idx, d));
            }
        }
        m.tick(u32::try_from(refs.len()).unwrap_or(u32::MAX))?;
        refs.sort_by_key(|&(i, _)| i);
        let mut positions: Vec<(u64, Decoded)> = Vec::new();
        for (idx, d) in refs {
            m.tick(1)?;
            match positions.last_mut() {
                Some((last, existing)) if *last == idx => existing.merge(d, m)?,
                _ => {
                    grow(m, &mut positions)?;
                    m.charge(std::mem::size_of::<(Sym, Sym)>().saturating_mul(d.strings.len()))?;
                    positions.push((idx, d.clone()));
                }
            }
        }
        grow(m, lists)?;
        lists.push(AttrList { positions });
        Ok(())
    })
}

/// Decode the attributes of a `PARAMATTR_GRP_CODE_ENTRY` after its id and position.
fn decode_group(s: &Stream<'_>, symbols: &mut Symbols, m: &mut Meter) -> Result<Decoded, Error> {
    const B: u32 = PARAMATTR_GROUP_BLOCK;
    const C: u32 = PARAMATTR_GRP_ENTRY;
    let ops = s.ops();
    let mut d = Decoded::default();
    let mut i = 2;
    let next = |i: &mut usize| -> Result<u64, Error> {
        let v = ops.get(*i).copied().ok_or_else(|| s.malformed(B, C))?;
        *i += 1;
        Ok(v)
    };
    while i < ops.len() {
        m.tick(1)?;
        let form = next(&mut i)?;
        match form {
            // Enum attribute: [kind].
            0 => {
                let kind = known(s, next(&mut i)?)?;
                match kind {
                    KIND_BY_VAL => d.byval = Some(TyRef::Pointee),
                    KIND_STRUCT_RET => d.sret = Some(TyRef::Pointee),
                    KIND_NO_ALIAS => d.noalias = true,
                    KIND_NO_CAPTURE => d.nocapture = true,
                    KIND_CONVERGENT => d.convergent = true,
                    KIND_READ_NONE => d.access = Some(meet(d.access, ModRef::NoModRef)),
                    KIND_READ_ONLY => d.access = Some(meet(d.access, ModRef::Ref)),
                    KIND_WRITEONLY => d.access = Some(meet(d.access, ModRef::Mod)),
                    KIND_ARGMEMONLY => d.loc = Some(MemLoc::Arg),
                    KIND_INACCESSIBLEMEM_ONLY => d.loc = Some(MemLoc::Inaccessible),
                    KIND_INACCESSIBLEMEM_OR_ARGMEMONLY => d.loc = Some(MemLoc::InaccessibleOrArg),
                    // `known` bounds every kind by `MAX_KIND`, below the set's largest.
                    _ => {
                        let kind = u32::try_from(kind).map_err(|_| s.malformed(B, C))?;
                        if !d.enums.insert(kind) {
                            return Err(s.malformed(B, C));
                        }
                    }
                }
            }
            // Integer attribute: [kind, value].
            1 => {
                let kind = known(s, next(&mut i)?)?;
                let value = next(&mut i)?;
                match kind {
                    KIND_ALIGNMENT => {
                        d.align = Some(Align::from_bytes(value).ok_or_else(|| s.malformed(B, C))?)
                    }
                    KIND_DEREFERENCEABLE => d.dereferenceable = Some(value),
                    KIND_DEREFERENCEABLE_OR_NULL => d.dereferenceable_or_null = Some(value),
                    KIND_MEMORY => d.memory = Some(memory_effects(s, value)?),
                    KIND_CAPTURES => d.nocapture = value == 0,
                    KIND_DENORMAL_FPENV => denormal_fpenv(s, value)?,
                    _ => {}
                }
            }
            // String attribute: key NUL, or key NUL value NUL.
            3 | 4 => {
                let key = cstr(s, ops, &mut i, m)?;
                let value = if form == 4 {
                    cstr(s, ops, &mut i, m)?
                } else {
                    Vec::new()
                };
                let pair = (symbols.intern(&key, m)?, symbols.intern(&value, m)?);
                grow(m, &mut d.strings)?;
                d.strings.push(pair);
            }
            // Type attribute: [kind] or [kind, type].
            5 | 6 => {
                let kind = known(s, next(&mut i)?)?;
                let ty = if form == 6 {
                    TyRef::Explicit(next(&mut i)?)
                } else {
                    TyRef::Pointee
                };
                match kind {
                    KIND_BY_VAL => d.byval = Some(ty),
                    KIND_STRUCT_RET => d.sret = Some(ty),
                    _ => {}
                }
            }
            // Constant range: [kind, bitwidth, range]; advisory.
            7 => {
                known(s, next(&mut i)?)?;
                let width = next(&mut i)?;
                skip_range(s, ops, &mut i, width)?;
            }
            // Constant range list: [kind, count, bitwidth, count x range]; advisory.
            8 => {
                known(s, next(&mut i)?)?;
                let count = next(&mut i)?;
                let width = next(&mut i)?;
                if count > ops.len() as u64 {
                    return Err(s.malformed(B, C));
                }
                for _ in 0..count {
                    skip_range(s, ops, &mut i, width)?;
                }
            }
            _ => return Err(s.err(BitcodeError::UnknownAttribute)),
        }
    }
    Ok(d)
}

/// Check `kind` against the closed table and refuse the ABI-changing kinds.
fn known(s: &Stream<'_>, kind: u64) -> Result<u64, Error> {
    if kind == 0 || kind > MAX_KIND {
        return Err(s.err(BitcodeError::UnknownAttribute));
    }
    if KIND_REFUSED.contains(&kind) {
        let kind = u32::try_from(kind).unwrap_or(u32::MAX);
        return Err(s.unsupported(Feature::Attribute { kind }));
    }
    Ok(kind)
}

/// The NUL-terminated string starting at `ops[*i]`, as bytes.
fn cstr(s: &Stream<'_>, ops: &[u64], i: &mut usize, m: &mut Meter) -> Result<Vec<u8>, Error> {
    let bad = || s.malformed(PARAMATTR_GROUP_BLOCK, PARAMATTR_GRP_ENTRY);
    let rest = ops.get(*i..).unwrap_or(&[]);
    let len = rest.iter().position(|&c| c == 0).ok_or_else(bad)?;
    let bytes = chars(rest.get(..len).unwrap_or(&[]), m)?.ok_or_else(bad)?;
    *i += len + 1;
    Ok(bytes)
}

/// Skip one encoded constant range of `width` bits.
fn skip_range(s: &Stream<'_>, ops: &[u64], i: &mut usize, width: u64) -> Result<(), Error> {
    let words = if width > 64 {
        let packed = *ops
            .get(*i)
            .ok_or_else(|| s.malformed(PARAMATTR_GROUP_BLOCK, PARAMATTR_GRP_ENTRY))?;
        1 + (packed & 0xFFFF_FFFF) + (packed >> 32)
    } else {
        2
    };
    let end = (*i as u64).saturating_add(words);
    if end > ops.len() as u64 {
        return Err(s.malformed(PARAMATTR_GROUP_BLOCK, PARAMATTR_GRP_ENTRY));
    }
    *i = end as usize;
    Ok(())
}

/// Decode a `memory(...)` value. Version 0 (bits 56..64) has three locations (argmem,
/// inaccessible, other); version 1 splits errno and target memory out of "other", which the IR
/// folds back in. Version 2 (LLVM 23) has version 1's layout: it only marks the target locations
/// as written by the producer, where LLVM's reader copies the inaccessible location into them for
/// an older version under an AArch64 triple (never an AIR one).
fn memory_effects(s: &Stream<'_>, value: u64) -> Result<MemoryEffects, Error> {
    let loc = |n: u32| modref((value >> (2 * n)) & 3);
    let version = value >> 56;
    let payload = value & 0x00FF_FFFF_FFFF_FFFF;
    let (used_bits, other) = match version {
        0 => (6, loc(2)),
        1 | 2 => (12, union(union(loc(2), loc(3)), union(loc(4), loc(5)))),
        _ => return Err(s.err(BitcodeError::UnknownAttribute)),
    };
    if payload >> used_bits != 0 {
        return Err(s.err(BitcodeError::UnknownAttribute));
    }
    Ok(MemoryEffects {
        argmem: loc(0),
        inaccessible: loc(1),
        other,
    })
}

/// Check a `denormal_fpenv` value (LLVM 23): the default mode in bits 0..4 and the `float`
/// override in bits 4..8, each an output and an input kind of two bits, every one of which LLVM
/// defines (`ieee`, `preservesign`, `positivezero`, `dynamic`). The value is dropped (module
/// docs).
fn denormal_fpenv(s: &Stream<'_>, value: u64) -> Result<(), Error> {
    if value >> 8 != 0 {
        return Err(s.err(BitcodeError::UnknownAttribute));
    }
    Ok(())
}

fn modref(bits: u64) -> ModRef {
    match bits & 3 {
        0 => ModRef::NoModRef,
        1 => ModRef::Ref,
        2 => ModRef::Mod,
        _ => ModRef::ModRef,
    }
}

fn bits(m: ModRef) -> u64 {
    match m {
        ModRef::NoModRef => 0,
        ModRef::Ref => 1,
        ModRef::Mod => 2,
        ModRef::ModRef => 3,
    }
}

fn union(a: ModRef, b: ModRef) -> ModRef {
    modref(bits(a) | bits(b))
}

fn intersect(a: ModRef, b: ModRef) -> ModRef {
    modref(bits(a) & bits(b))
}

/// Combine a new access restriction with an earlier one on the same position.
fn meet(prev: Option<ModRef>, new: ModRef) -> ModRef {
    prev.map_or(new, |p| intersect(p, new))
}

fn boxed(v: &[(Sym, Sym)], m: &mut Meter) -> Result<Box<[(Sym, Sym)]>, Error> {
    let mut out = Vec::new();
    m.reserve(&mut out, v.len())?;
    out.extend_from_slice(v);
    Ok(out.into_boxed_slice())
}
