//! Byte layout of typed data, and typed loads and stores.
//!
//! Explicitly laid-out types use their declared offsets and strides (`Offset`, `ArrayStride`,
//! `MatrixStride`); implicit-layout types (`Function`, `Private`, `Workgroup` data) use a packed
//! scalar-aligned layout private to the interpreter, which logical addressing cannot observe.
//! Loads and stores go scalar by scalar, so struct padding is never read and a partially written
//! `Workgroup` aggregate traps only on the bytes that matter.

use crate::spirv::ids::TypeId;
use crate::spirv::ir::{StorageClass, Type};

use super::memory::Mem;
use super::value::{Ptr, Root, Tys, Val};
use super::{malformed, IResult, TrapKind, Unmodelled};

/// The byte size of a scalar (`bool` takes one byte) or physical pointer.
fn scalar_bytes(t: &Type) -> Option<u64> {
    match t {
        Type::Bool => Some(1),
        Type::Int { width, .. } | Type::Float { width } => Some(u64::from(*width) / 8),
        Type::Pointer {
            class: StorageClass::PhysicalStorageBuffer,
            ..
        } => Some(8),
        _ => None,
    }
}

fn round_up(v: u64, a: u64) -> IResult<u64> {
    let a = a.max(1);
    v.checked_add(a - 1)
        .map(|x| x / a * a)
        .ok_or_else(|| malformed("layout overflow"))
}

/// Alignment in the implicit layout.
fn align(t: &Tys<'_>, ty: TypeId) -> IResult<u64> {
    let tt = t.get(ty)?;
    if let Some(b) = scalar_bytes(tt) {
        return Ok(b);
    }
    match tt {
        Type::Vector { elem, .. } | Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => {
            align(t, *elem)
        }
        Type::Matrix { col, .. } => align(t, *col),
        Type::Struct { members, .. } => {
            let mut a = 1;
            for m in members.iter() {
                a = a.max(align(t, *m)?);
            }
            Ok(a)
        }
        _ => Err(Unmodelled::Type(ty).err()),
    }
}

/// The element stride of an array type.
fn stride(t: &Tys<'_>, elem: TypeId, explicit: Option<u32>, ms: Option<u32>) -> IResult<u64> {
    match explicit {
        Some(s) => Ok(u64::from(s)),
        None => round_up(size(t, elem, ms)?, align(t, elem)?),
    }
}

/// The byte size of `ty` (a runtime array counts as empty). `ms` is the matrix stride when `ty` is
/// a matrix or array of matrices in an explicit layout.
pub(super) fn size(t: &Tys<'_>, ty: TypeId, ms: Option<u32>) -> IResult<u64> {
    let tt = t.get(ty)?;
    if let Some(b) = scalar_bytes(tt) {
        return Ok(b);
    }
    let overflow = || malformed("layout overflow");
    match tt {
        Type::Vector { elem, len } => Ok(u64::from(*len) * size(t, *elem, None)?),
        Type::Matrix { col, cols } => {
            let c = size(t, *col, None)?;
            let s = ms.map_or(c, u64::from);
            Ok(u64::from(*cols) * s)
        }
        Type::Array {
            elem,
            len,
            stride: s,
        } => t
            .array_len(*len)?
            .checked_mul(stride(t, *elem, *s, ms)?)
            .ok_or_else(overflow),
        Type::RuntimeArray { .. } => Ok(0),
        Type::Struct { members, layout } => {
            let mut end = 0u64;
            let mut off = 0u64;
            for (i, m) in members.iter().enumerate() {
                let (o, mms) = match layout {
                    Some(l) => {
                        let ml = l
                            .members
                            .get(i)
                            .ok_or_else(|| malformed("struct layout member count"))?;
                        (u64::from(ml.offset), ml.matrix_stride)
                    }
                    None => (round_up(off, align(t, *m)?)?, None),
                };
                off = o.checked_add(size(t, *m, mms)?).ok_or_else(overflow)?;
                end = end.max(off);
            }
            match layout {
                Some(_) => Ok(end),
                None => round_up(end, align(t, ty)?),
            }
        }
        _ => Err(Unmodelled::Type(ty).err()),
    }
}

/// Part `i` of `ty`: its type, byte offset inside `ty`, and matrix stride. An index outside a
/// vector, matrix, or sized array traps; one past a runtime array's end is left to the access's
/// bounds check.
pub(super) fn child(
    t: &Tys<'_>,
    ty: TypeId,
    ms: Option<u32>,
    i: i64,
) -> IResult<(TypeId, u64, Option<u32>)> {
    let tt = t.get(ty)?;
    let range = |len: u64| -> IResult<u64> {
        match u64::try_from(i) {
            Ok(k) if k < len => Ok(k),
            _ => Err(TrapKind::IndexOutOfRange { index: i, len }.err()),
        }
    };
    let far = || TrapKind::AddressOverflow.err();
    match tt {
        Type::Vector { elem, len } => {
            let k = range(u64::from(*len))?;
            Ok((*elem, k * size(t, *elem, None)?, None))
        }
        Type::Matrix { col, cols } => {
            let k = range(u64::from(*cols))?;
            let s = match ms {
                Some(s) => u64::from(s),
                None => size(t, *col, None)?,
            };
            Ok((*col, k * s, None))
        }
        Type::Array {
            elem,
            len,
            stride: s,
        } => {
            let k = range(t.array_len(*len)?)?;
            let off = k.checked_mul(stride(t, *elem, *s, ms)?).ok_or_else(far)?;
            Ok((*elem, off, ms))
        }
        Type::RuntimeArray { elem, stride: s } => {
            // Indices are signed; a runtime array has no element before the first.
            let k = u64::try_from(i).map_err(|_| TrapKind::NegativeIndex { index: i }.err())?;
            let off = k.checked_mul(stride(t, *elem, *s, ms)?).ok_or_else(far)?;
            Ok((*elem, off, ms))
        }
        Type::Struct { members, layout } => {
            let k = range(members.len() as u64)? as usize;
            let m = *members
                .get(k)
                .ok_or_else(|| malformed("struct member out of range"))?;
            match layout {
                Some(l) => {
                    let ml = l
                        .members
                        .get(k)
                        .ok_or_else(|| malformed("struct layout member count"))?;
                    Ok((m, u64::from(ml.offset), ml.matrix_stride))
                }
                None => {
                    let mut off = 0u64;
                    for (j, mt) in members.iter().enumerate() {
                        off = round_up(off, align(t, *mt)?)?;
                        if j == k {
                            break;
                        }
                        off += size(t, *mt, None)?;
                    }
                    Ok((m, off, None))
                }
            }
        }
        _ => Err(malformed(format!("type {ty:?} cannot be indexed"))),
    }
}

/// Offset `p` by `d` bytes; an offset past `u64::MAX` traps instead of wrapping.
pub(super) fn offset(p: Ptr, d: u64) -> IResult<Ptr> {
    let off = p
        .off
        .checked_add(d)
        .ok_or_else(|| TrapKind::AddressOverflow.err())?;
    Ok(Ptr { off, ..p })
}

/// Load a `ty` from `p`.
pub(super) fn load(t: &Tys<'_>, mem: &mut Mem<'_>, p: Ptr, ty: TypeId) -> IResult<Val> {
    let tt = t.get(ty)?;
    if let Some(b) = scalar_bytes(tt) {
        let mut buf = [0u8; 8];
        let slot = buf
            .get_mut(..b as usize)
            .ok_or_else(|| malformed("scalar wider than 64 bits"))?;
        mem.read(p.root, p.off, slot)?;
        let bits = u64::from_le_bytes(buf);
        return Ok(match tt {
            Type::Bool => Val::S(u64::from(bits != 0)),
            Type::Pointer { .. } => Val::P(Ptr {
                root: Root::Physical,
                off: bits,
                ms: None,
            }),
            _ => Val::S(bits),
        });
    }
    let n = match tt {
        Type::Vector { .. } | Type::Matrix { .. } | Type::Array { .. } | Type::Struct { .. } => {
            t.part_count(ty)?
        }
        _ => return Err(Unmodelled::Type(ty).err()),
    };
    let mut parts = Vec::with_capacity(n.min(1 << 16) as usize);
    for i in 0..n {
        let (ct, co, cms) = child(t, ty, p.ms, i as i64)?;
        parts.push(load(
            t,
            mem,
            Ptr {
                ms: cms,
                ..offset(p, co)?
            },
            ct,
        )?);
    }
    Ok(Val::C(parts.into()))
}

/// Store `v` (a `ty`) to `p`.
pub(super) fn store(t: &Tys<'_>, mem: &mut Mem<'_>, p: Ptr, ty: TypeId, v: &Val) -> IResult<()> {
    let tt = t.get(ty)?;
    if let Some(b) = scalar_bytes(tt) {
        let bits = match v {
            Val::S(x) => *x,
            Val::P(q) if q.root == Root::Physical => q.off,
            _ => return Err(malformed("storing a non-scalar as a scalar")),
        };
        let bytes = bits.to_le_bytes();
        let slot = bytes
            .get(..b as usize)
            .ok_or_else(|| malformed("scalar wider than 64 bits"))?;
        return mem.write(p.root, p.off, slot);
    }
    let parts = v.parts()?;
    for (i, part) in parts.iter().enumerate() {
        let (ct, co, cms) = child(t, ty, p.ms, i as i64)?;
        store(
            t,
            mem,
            Ptr {
                ms: cms,
                ..offset(p, co)?
            },
            ct,
            part,
        )?;
    }
    Ok(())
}

/// The bytes of `v` (a `ty`) in the implicit layout, for initializing a fresh allocation.
pub(super) fn encode(t: &Tys<'_>, ty: TypeId, v: &Val) -> IResult<Vec<u8>> {
    let n = size(t, ty, None)?;
    let n = usize::try_from(n).map_err(|_| Unmodelled::Type(ty).err())?;
    let mut lane = vec![super::memory::Region::new(
        super::MemoryKind::Private,
        vec![0; n],
        true,
    )];
    let mut sh = super::memory::Shared {
        wg: Vec::new(),
        bufs: Vec::new(),
        push: super::memory::Region::new(super::MemoryKind::PushConstant, Vec::new(), false),

        images: Vec::new(),
    };
    let mut mem = Mem {
        lane: &mut lane,
        sh: &mut sh,
    };
    store(
        t,
        &mut mem,
        Ptr {
            root: Root::Lane(0),
            off: 0,
            ms: None,
        },
        ty,
        v,
    )?;
    lane.pop()
        .map(|r| r.bytes)
        .ok_or_else(|| malformed("encode lost its region"))
}
