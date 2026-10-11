//! The entry's parameter bindings and argument-buffer layouts, indexed once per function.
//!
//! Memory analysis and selection ask, per root and per access, which binding a parameter has,
//! whether it is a handle array or a buffer table, and which argument-buffer member holds a byte
//! offset. Each answer was a scan: of the entry's bindings for the parameter, and of a layout's
//! members for the offset. [`AbiIndex::of`] reads every binding and every layout member once,
//! before provenance (one work unit per binding and per member, two per member boundary), its
//! tables charged to the ledger and retained with the memory plan. A parameter's binding is the
//! first in the entry's order that names it, and the member of a layout at an offset is the first,
//! in member order, whose bytes hold the offset: what the scans found.
//!
//! A layout's bytes are cut at every member's start and end; each piece holds the first member
//! that covers it, so finding the member at an offset is a binary search over the pieces, exact
//! for layouts whose members overlap as for those whose members are disjoint.

use std::collections::BTreeSet;

use crate::air::abi::{Member, ParamBinding, StructInfo, StructRef};
use crate::air::ir::Type;
use crate::air::Entry;
use crate::budget::{tree_bytes, Meter, LAYOUT_NESTING};
use crate::error::{Error, InternalError, InternalKind};

use super::provenance::{buffer_table_of, handle_array_of, BufferTable, HandleClass};

/// No binding, or no member.
const NONE: u32 = u32::MAX;

/// The entry's bindings by parameter and its layouts' members by offset.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AbiIndex {
    /// By LLVM parameter: its binding and what it is.
    params: Vec<ParamFact>,
    /// By layout `s`, `pieces[starts[s]..starts[s + 1]]` are the layout's pieces.
    starts: Vec<u32>,
    /// Each layout's bytes cut at its members' starts and ends, in order: each piece's first byte
    /// and the first member covering it (`NONE` for a gap or the bytes past the last member).
    pieces: Vec<(u64, u32)>,
}

/// What one parameter is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ParamFact {
    /// Its binding's position in the entry's bindings (`NONE` for none).
    binding: u32,
    /// Its handle class when it is a handle array ([`handle_array_of`]).
    handle_array: Option<HandleClass>,
    /// Its table when it is a buffer table ([`buffer_table_of`]).
    buffer_table: Option<BufferTable>,
    /// Its handle class when it is a texture or sampler a function constant removed that the body
    /// still reads (`Entry::absent_handles`): it has no binding.
    absent: Option<HandleClass>,
}

impl ParamFact {
    const NONE: ParamFact = ParamFact {
        binding: NONE,
        handle_array: None,
        buffer_table: None,
        absent: None,
    };
}

impl AbiIndex {
    /// Index the bindings and layouts of `e`.
    pub fn of(e: &Entry, m: &mut Meter) -> Result<AbiIndex, Error> {
        let count = match e
            .module
            .func(e.func)
            .and_then(|f| e.module.types().get(f.ty))
        {
            Some(Type::Function { params, .. }) => params.len(),
            _ => 0,
        };
        let mut params = Vec::new();
        m.reserve(&mut params, count)?;
        params.resize(count, ParamFact::NONE);
        for (i, b) in e.ep.params.iter().enumerate() {
            m.tick(1)?;
            let Some(fact) = params.get_mut(b.param as usize) else {
                continue;
            };
            if fact.binding != NONE {
                continue;
            }
            *fact = ParamFact {
                binding: u32::try_from(i).map_err(|_| invariant("more bindings than u32"))?,
                handle_array: handle_array_of(e, b),
                buffer_table: buffer_table_of(e, b),
                absent: None,
            };
        }
        // A removed texture or sampler the body still reads binds nothing, but its handles, a
        // table's elements included, are followed as a present one's are.
        for b in &e.absent_handles {
            m.tick(1)?;
            let Some(fact) = params.get_mut(b.param as usize) else {
                continue;
            };
            if fact.binding != NONE || fact.absent.is_some() {
                continue;
            }
            fact.handle_array = handle_array_of(e, b);
            fact.absent = b.role.handle().map(HandleClass::of_abi);
        }
        Ok(AbiIndex {
            params,
            ..AbiIndex::layouts(&e.structs, m)?
        })
    }

    /// Index the layouts `structs` alone (no parameter has a binding).
    pub(crate) fn layouts(structs: &[StructInfo], m: &mut Meter) -> Result<AbiIndex, Error> {
        let mut starts = Vec::new();
        m.reserve(&mut starts, structs.len().saturating_add(1))?;
        starts.push(0);
        let members = structs
            .iter()
            .map(|s| s.members.len())
            .fold(0usize, usize::saturating_add);
        let mut pieces = Vec::new();
        m.reserve(&mut pieces, members.saturating_mul(2))?;
        for info in structs {
            cut(&info.members, &mut pieces, m)?;
            starts.push(u32::try_from(pieces.len()).map_err(|_| invariant("pieces past u32"))?);
        }
        Ok(AbiIndex {
            params: Vec::new(),
            starts,
            pieces,
        })
    }

    /// The ledger bytes the index holds, which the memory plan retains.
    pub fn retained_bytes(&self) -> usize {
        self.params.capacity() * std::mem::size_of::<ParamFact>()
            + self.starts.capacity() * std::mem::size_of::<u32>()
            + self.pieces.capacity() * std::mem::size_of::<(u64, u32)>()
    }

    /// The binding of parameter `p` in `e` (the entry the index was built from).
    pub fn binding<'e>(&self, e: &'e Entry, p: u32) -> Option<&'e ParamBinding> {
        let fact = self.params.get(p as usize)?;
        e.ep.params.get(fact.binding as usize)
    }

    /// The handle class of parameter `p` when it is an array of resource handles
    /// ([`handle_array_of`]).
    pub fn handle_array(&self, p: u32) -> Option<HandleClass> {
        self.params.get(p as usize)?.handle_array
    }

    /// The buffer table parameter `p` is, if it is one ([`buffer_table_of`]).
    pub fn buffer_table(&self, p: u32) -> Option<BufferTable> {
        self.params.get(p as usize)?.buffer_table
    }

    /// The handle class of parameter `p` when it is a texture or sampler (or a table of them) a
    /// function constant removed that the body still reads (`Entry::absent_handles`): every handle
    /// it holds is the absent handle (`memory::handles::Candidate::Absent`).
    pub fn absent(&self, p: u32) -> Option<HandleClass> {
        self.params.get(p as usize)?.absent
    }

    /// The index among layout `s`'s members of the first whose bytes hold offset `off`.
    pub fn member_index(&self, s: StructRef, off: u64) -> Option<usize> {
        let lo = *self.starts.get(s.0 as usize)? as usize;
        let hi = *self.starts.get((s.0 as usize).checked_add(1)?)? as usize;
        let pieces = self.pieces.get(lo..hi)?;
        let k = pieces.partition_point(|(at, _)| *at <= off);
        let (_, member) = *pieces.get(k.checked_sub(1)?)?;
        (member != NONE).then_some(member as usize)
    }

    /// The argument-buffer member at byte offset `off` of layout `s` of `structs` (the entry's
    /// layouts), following nested struct members and member arrays (at most `LAYOUT_NESTING`
    /// levels); an offset inside a member must start one of its elements
    /// (`provenance::member_element_size`).
    pub fn member_at<'s>(
        &self,
        structs: &'s [StructInfo],
        s: StructRef,
        off: u64,
    ) -> Option<&'s Member> {
        self.member_place_at(structs, s, off).map(|(_, mem)| mem)
    }

    /// [`AbiIndex::member_at`], with the member's place in `structs`: its layout's index and its
    /// index among that layout's members.
    pub fn member_place_at<'s>(
        &self,
        structs: &'s [StructInfo],
        mut s: StructRef,
        mut off: u64,
    ) -> Option<((usize, usize), &'s Member)> {
        use crate::air::abi::MemberKind;
        for _ in 0..LAYOUT_NESTING {
            let info = structs.get(s.0 as usize)?;
            let index = self.member_index(s, off)?;
            let mem = info.members.get(index)?;
            let within = off - u64::from(mem.offset);
            let within = if mem.size > 0 {
                within % u64::from(mem.size)
            } else {
                within
            };
            match (mem.layout, &mem.kind) {
                (Some(inner), MemberKind::Plain | MemberKind::IndirectStruct { .. }) => {
                    s = inner;
                    off = within;
                }
                _ => {
                    let element = u64::from(super::provenance::member_element_size(mem));
                    return (within == 0 || (element > 0 && within.is_multiple_of(element)))
                        .then_some(((s.0 as usize, index), mem));
                }
            }
        }
        None
    }
}

/// The bytes of member `mem`: `[offset, offset + size · max(array length, 1))`.
fn span(mem: &Member) -> (u64, u64) {
    let n = u64::from(mem.array_len.max(1));
    let start = u64::from(mem.offset);
    let len = u64::from(mem.size).saturating_mul(n);
    (start, start.saturating_add(len))
}

/// Cut one layout's bytes at its members' starts and ends into `out` (reserved for two pieces per
/// member): each piece with the first member that covers it.
fn cut(members: &[Member], out: &mut Vec<(u64, u32)>, m: &mut Meter) -> Result<(), Error> {
    let mut by_start: Vec<(u64, u64, u32)> = Vec::new();
    let mut by_end: Vec<(u64, u64, u32)> = Vec::new();
    // The bytes charged for the two orders and, at most one entry per member, the active set.
    let mut charged = 0usize;
    let result = (|| {
        let order = members
            .len()
            .saturating_mul(std::mem::size_of::<(u64, u64, u32)>());
        m.reserve(&mut by_start, members.len())?;
        charged = charged.saturating_add(order);
        m.reserve(&mut by_end, members.len())?;
        charged = charged.saturating_add(order);
        let active_bytes = tree_bytes(members.len(), std::mem::size_of::<u32>());
        m.charge(active_bytes)?;
        charged = charged.saturating_add(active_bytes);
        for (i, mem) in members.iter().enumerate() {
            m.tick(1)?;
            let (start, end) = span(mem);
            if end > start {
                let i = u32::try_from(i).map_err(|_| invariant("more members than u32"))?;
                by_start.push((start, end, i));
                by_end.push((end, start, i));
            }
        }
        by_start.sort_unstable();
        by_end.sort_unstable();
        // The members covering the bytes from the current boundary on.
        let mut active: BTreeSet<u32> = BTreeSet::new();
        let (mut a, mut b) = (0usize, 0usize);
        while let Some(&(at, _, _)) = by_end.get(b) {
            let at = by_start.get(a).map_or(at, |s| s.0.min(at));
            while let Some(&(_, _, i)) = by_end.get(b).filter(|x| x.0 == at) {
                m.tick(1)?;
                active.remove(&i);
                b += 1;
            }
            while let Some(&(_, _, i)) = by_start.get(a).filter(|x| x.0 == at) {
                m.tick(1)?;
                active.insert(i);
                a += 1;
            }
            if out.len() == out.capacity() {
                return Err(invariant("a layout's pieces past their reservation"));
            }
            out.push((at, active.first().copied().unwrap_or(NONE)));
        }
        Ok(())
    })();
    m.release(charged);
    result
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

#[cfg(test)]
mod tests;
