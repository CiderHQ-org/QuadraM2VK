//! Tensor objects: the `air.*_private_tensor` intrinsics over a `tensor_inline`'s object.
//!
//! A `tensor<…, tensor_inline>` is an object in thread memory whose bytes the program never reads
//! itself: Metal's tensor header (`metal_tensor`) sizes it with `air.get_descriptor_size_tensor`,
//! fills it with `air.init_strided_private_tensor` (a data pointer, and `rank` extents and strides
//! of the index type), copies it with `air.slice_private_tensor_private_tensor` (no offsets and no
//! extents: the copy constructor), and reads an extent with `air.get_extent_private_tensor`;
//! `matmul2d` reads the object it is given. Its layout is therefore the translation's, and this
//! one is:
//!
//! | Bytes | Field |
//! |---|---|
//! | 0..8 | the data pointer (a pointer cell) |
//! | 8 + 4d | extent `d` (`i32`) |
//! | 8 + 4·rank + 4d | stride `d`, in elements (`i32`) |
//!
//! Only the `i32` index type (`.i32`, size 4) is modeled. A slice with offsets or extents is left
//! a call, which lowering refuses (`Unsupported{Tensors}`): what Metal reads past a slice's
//! extents is not established.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::Tensor;
use crate::ir::{AddrSpace, BinOp, CastOp, Const, ConstTable, Type, Value};

use super::super::out::Out;
use super::super::rewrite::invariant;

/// The largest rank modeled.
pub const MAX_RANK: u64 = 4;

/// The index type's size in bytes (`.i32`).
pub const INDEX_BYTES: u64 = 4;

/// The object's alignment.
pub const ALIGN: u64 = 8;

/// The bytes of an object of `rank`.
pub fn object_bytes(rank: u64) -> u64 {
    8 + 2 * INDEX_BYTES * rank
}

/// The byte offset of extent `d`.
pub fn extent_at(d: u64) -> u64 {
    8 + INDEX_BYTES * d
}

/// The byte offset of stride `d` in an object of `rank`.
pub fn stride_at(rank: u64, d: u64) -> u64 {
    8 + INDEX_BYTES * (rank + d)
}

/// The modeled form of a call of tensor operation `t` with `args`: its rank, read by `int`, when
/// the rank is at most [`MAX_RANK`], an initialization's index arrays are not null, and a slice
/// has neither offsets nor extents (the copy constructor). `None` for another form, which stays a
/// call.
pub fn form(
    t: Tensor,
    args: &[Value],
    int: &dyn Fn(Value) -> Option<u64>,
    null: &dyn Fn(Value) -> bool,
) -> Option<u64> {
    let rank = |v: Value| int(v).filter(|&r| r <= MAX_RANK);
    match (t, args) {
        (Tensor::DescriptorSize, &[r, bytes]) => {
            let r = rank(r)?;
            (int(bytes)? == INDEX_BYTES).then_some(r)
        }
        (Tensor::Extent, &[_, r, _]) => rank(r),
        (Tensor::InitStrided { .. }, &[_, r, _, extents, strides, _]) => {
            let r = rank(r)?;
            (r == 0 || !(null(extents) || null(strides))).then_some(r)
        }
        (Tensor::Slice, &[_, _, r, offsets, extents]) => {
            let r = rank(r)?;
            (null(offsets) && null(extents)).then_some(r)
        }
        _ => None,
    }
}

/// Whether `v` is the null pointer constant.
pub fn is_null(consts: &ConstTable, v: Value) -> bool {
    matches!(v, Value::Const(c) if matches!(consts.get(c).map(|k| &k.value), Some(Const::Null)))
}

/// `air.init_strided_private_tensor.i32.<space>(obj, rank, data, extents, strides, kind)` of
/// [`form`] `rank`: the data pointer, then each extent and stride copied from the two index
/// arrays.
pub fn init(out: &mut Out<'_>, args: &[Value], rank: u64, m: &mut Meter) -> Result<(), Error> {
    let [obj, _, data, extents, strides, _kind] = *args else {
        return Err(invariant("a tensor initialization of another form"));
    };
    out.store(data, obj, ALIGN, m)?;
    let i32t = out.int_ty(32, m)?;
    for d in 0..rank {
        let from = out.byte_gep_k(AddrSpace::Thread, extents, INDEX_BYTES * d, m)?;
        let e = out.load(i32t, from, INDEX_BYTES, m)?;
        let to = out.byte_gep_k(AddrSpace::Thread, obj, extent_at(d), m)?;
        out.store(e, to, INDEX_BYTES, m)?;
        let from = out.byte_gep_k(AddrSpace::Thread, strides, INDEX_BYTES * d, m)?;
        let s = out.load(i32t, from, INDEX_BYTES, m)?;
        let to = out.byte_gep_k(AddrSpace::Thread, obj, stride_at(rank, d), m)?;
        out.store(s, to, INDEX_BYTES, m)?;
    }
    Ok(())
}

/// `air.get_extent_private_tensor.i32(obj, rank, dim)`: extent `dim` (an extent past the rank
/// reads another field; Metal's header asks for one below it).
pub fn extent(out: &mut Out<'_>, args: &[Value], m: &mut Meter) -> Result<Value, Error> {
    let [obj, _, dim] = *args else {
        return Err(invariant("a tensor extent of another form"));
    };
    let i32t = out.int_ty(32, m)?;
    let i64t = out.int_ty(64, m)?;
    let d = out.cast(CastOp::ZExt, dim, i64t, m)?;
    let four = out.k64(INDEX_BYTES, m)?;
    let off = out.bin(i64t, BinOp::Mul, d, four, m)?;
    let first = out.k64(extent_at(0), m)?;
    let off = out.bin(i64t, BinOp::Add, off, first, m)?;
    let at = out.byte_gep(AddrSpace::Thread, obj, off, m)?;
    out.load(i32t, at, INDEX_BYTES, m)
}

/// `air.slice_private_tensor_private_tensor.s.i32(dst, src, rank, null, null)` of [`form`]
/// `rank`, the copy constructor: the object's bytes copied (`llvm.memcpy`, which moves the data
/// pointer's cell).
pub fn copy(out: &mut Out<'_>, args: &[Value], rank: u64, m: &mut Meter) -> Result<(), Error> {
    let [dst, src, ..] = *args else {
        return Err(invariant("a tensor copy of another form"));
    };
    let ptr = out.ptr_ty(AddrSpace::Thread, m)?;
    let i64t = out.int_ty(64, m)?;
    let i1t = out.int_ty(1, m)?;
    let void = out.ty(Type::Void, m)?;
    let len = out.k64(object_bytes(rank), m)?;
    let no = out.int(1, 0, m)?;
    out.intrinsic(
        "llvm.memcpy.p0.p0.i64",
        void,
        &[ptr, ptr, i64t, i1t],
        &[dst, src, len, no],
        m,
    )?;
    Ok(())
}
