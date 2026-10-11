//! A texture or sampler table copied into thread memory, as the handle copies it is.
//!
//! A texture or sampler table argument (`array<texture2d<T>, N>`, `array<sampler, N>`) is a
//! thread pointer to its `N` handles, each a pointer of the handle's address space (a texture's
//! `ptr addrspace(1)`, a sampler's `ptr addrspace(2)`, `intrinsics::HandleKind::space`). Metal's
//! compiler copies such a table, as any array member of a struct, into a stack object with
//! `llvm.memcpy`, and the stack object's handles are loaded back. Lowering follows a handle stored
//! into a stack slot and loaded back, but not through a byte copy, whose bytes are data: such a
//! module was refused `Lowering(Access)`.
//!
//! [`expand`] makes each such copy the handle copies it is ([`super::pieces`]): for each handle it
//! covers, a load of the handle's type from the table at the handle's offset and a store of it
//! into the destination at the same distance from the copy's start, in the copy's place, in
//! order. A copy is rewritten when it is a non-volatile `llvm.memcpy` or `llvm.memmove` between
//! thread pointers whose source is the table argument itself or an address derived from it by
//! `getelementptr`s at constant, non-negative indices and bitcasts, and whose length is a
//! constant: the copied bytes start at a handle, are whole handles, and end within the table (its
//! location's count, once the function constants resolved it). Both forms move the same bytes (a
//! source and destination that overlap cannot, the table being an argument the function only
//! reads), so every load and store afterwards reads and writes what it did. Any other copy stays
//! as it was.

use crate::abi::{AbiInt, ParamBinding, ParamRole};
use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::LlvmIntrinsic;
use crate::ir::{AddrSpace, ArgIdx, Body, Call, Callee, FuncId};
use crate::ir::{InstId, InstKind, Layout, ModuleParts, Type, TypeId, TypeLayouts, Value};

use super::pieces::{int, offset_from, replace, Piece, Replacement};
use super::rewrite::invariant;

/// The bytes of one handle (the AIR datalayout's `p:64` in every address space).
const HANDLE_BYTES: u64 = 8;

/// Make each copy of a texture or sampler table into thread memory in the body of entry `func`
/// the handle copies it is (module docs). `tables` are the entry's parameters, those the function
/// constants removed included.
pub(super) fn expand<'a>(
    parts: &mut ModuleParts,
    func: FuncId,
    tables: impl Iterator<Item = &'a ParamBinding>,
    m: &mut Meter,
) -> Result<(), Error> {
    let f = parts
        .funcs
        .get(func)
        .ok_or_else(|| invariant("entry out of range"))?;
    let Some(Type::Function { params, .. }) = parts.types.get(f.ty) else {
        return Err(invariant("the entry has no function type"));
    };
    // Each table: its parameter, its handle's address space, and its handles.
    let mut found: Vec<(ArgIdx, AddrSpace, u64)> = Vec::new();
    for b in tables {
        m.tick(1)?;
        let (space, location) = match &b.role {
            ParamRole::Texture(t) => (AddrSpace::Device, t.location),
            ParamRole::Sampler { location } => (AddrSpace::Constant, *location),
            _ => continue,
        };
        let thread = params
            .get(b.param as usize)
            .and_then(|&t| parts.types.ptr_space(t))
            == Some(AddrSpace::Thread);
        if let (true, AbiInt::Const(count)) = (thread, location.count) {
            m.reserve(&mut found, 1)?;
            found.push((ArgIdx::new(b.param), space, u64::from(count)));
        }
    }
    if found.is_empty() {
        return Ok(());
    }
    let mut handles = Vec::new();
    m.reserve(&mut handles, found.len())?;
    for &(_, space, _) in &found {
        handles.push(parts.types.intern(Type::Ptr { space }, m)?);
    }
    let layouts = TypeLayouts::new(&parts.data_layout, &parts.types, m)?;
    let copies = {
        let body = parts
            .funcs
            .get(func)
            .and_then(|f| f.body.as_ref())
            .ok_or_else(|| invariant("the entry has no body"))?;
        copies(body, parts, &layouts, &found, &handles, m)
    };
    m.release(
        parts.types.len() * std::mem::size_of::<Option<Layout>>()
            + found.capacity() * std::mem::size_of::<(ArgIdx, AddrSpace, u64)>()
            + handles.capacity() * std::mem::size_of::<TypeId>(),
    );
    let copies = copies?;
    let r = replace(parts, func, &copies, m);
    m.release(
        copies.capacity() * std::mem::size_of::<(InstId, Replacement)>()
            + copies.iter().map(|(_, r)| r.bytes()).sum::<usize>(),
    );
    r
}

/// The copies of `body` to rewrite (module docs), in body order, of the `tables` (each its
/// parameter, its handle's address space, and its handles), `handles` their handle types.
fn copies(
    body: &Body,
    parts: &ModuleParts,
    layouts: &TypeLayouts,
    tables: &[(ArgIdx, AddrSpace, u64)],
    handles: &[TypeId],
    m: &mut Meter,
) -> Result<Vec<(InstId, Replacement)>, Error> {
    let mut out = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Call(call) = &inst.kind else {
            continue;
        };
        let Some((src, dst, n, ty)) = table_copy(body, parts, layouts, call, tables, handles)
        else {
            continue;
        };
        let mut pieces = Vec::new();
        m.reserve(&mut pieces, usize::try_from(n).unwrap_or(usize::MAX))?;
        for k in 0..n {
            pieces.push(Piece {
                at: k * HANDLE_BYTES,
                ty,
            });
        }
        m.reserve(&mut out, 1)?;
        out.push((id, Replacement { src, dst, pieces }));
    }
    Ok(out)
}

/// The source, destination, handles and handle type of `call`, when it copies whole handles of
/// one of `tables` into thread memory.
fn table_copy(
    body: &Body,
    parts: &ModuleParts,
    layouts: &TypeLayouts,
    call: &Call,
    tables: &[(ArgIdx, AddrSpace, u64)],
    handles: &[TypeId],
) -> Option<(Value, Value, u64, TypeId)> {
    let Callee::Llvm {
        op:
            LlvmIntrinsic::Memcpy {
                dst: AddrSpace::Thread,
                src: AddrSpace::Thread,
                ..
            }
            | LlvmIntrinsic::Memmove {
                dst: AddrSpace::Thread,
                src: AddrSpace::Thread,
                ..
            },
        ..
    } = call.callee
    else {
        return None;
    };
    let [dst, src, len, volatile] = *call.args else {
        return None;
    };
    if int(parts, volatile)? != 0 {
        return None;
    }
    let len = int(parts, len)?;
    let (base, from) = offset_from(body, parts, layouts, src)?;
    let Value::Arg(table) = base else {
        return None;
    };
    let k = tables.iter().position(|(a, _, _)| *a == table)?;
    let (_, _, count) = *tables.get(k)?;
    let end = from.checked_add(len)?;
    let whole = from % HANDLE_BYTES == 0 && len % HANDLE_BYTES == 0 && len > 0;
    (whole && end <= count.checked_mul(HANDLE_BYTES)?).then_some((
        src,
        dst,
        len / HANDLE_BYTES,
        *handles.get(k)?,
    ))
}
