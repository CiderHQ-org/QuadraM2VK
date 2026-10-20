//! Cooperative imageblock slice writes: which slice writes every invocation of the workgroup
//! shares.
//!
//! Metal writes a slice (`air.write_imageblock_slice_to_texture_2d`) as one block store, and a
//! kernel calls it the way Apple's samples do: every thread stores its cells, a threadgroup
//! barrier, then one thread writes the slice,
//!
//! ```text
//! threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
//! if (tid.x == 0 && tid.y == 0) out.write(block.slice(...), at, level);
//! ```
//!
//! Emulated, the slice is a loop over its cells (`super::slice`), and in that one invocation it is
//! serial: a 16 x 32 slice is 512 iterations of a cell load and a texture write in one thread,
//! which a threadgroup of the drag's blur kernel spent 500 us on where Metal takes 8 us. Where
//! the shape below holds, every invocation writes a strided share of the cells instead (the loop
//! starts at its `LocalInvocationIndex` and steps by the workgroup's size), then a workgroup
//! barrier, and the guard's branch is taken by every invocation ([`TexturePlan::cooperative`]).
//!
//! The shape, all read off the IR's structure:
//!
//! - the call is in a kernel, not a tile function, with an explicit imageblock of its own (not one
//!   aliasing a tile's);
//! - its block `A` has one predecessor `P`, which ends in `br c, A, M`, and `A` ends in `br M`;
//! - `P` is workgroup uniform (`simd::uniform`, under the barrier contract) and holds a threadgroup
//!   barrier ordering threadgroup or imageblock memory, after which it computes nothing with an
//!   effect before its branch: every invocation reaches the branch after every cell store;
//! - `c` holds for the invocation at local id 0: it is computed from the local id builtins
//!   (`thread_position_in_threadgroup`, `thread_index_in_threadgroup`) and constants alone, by
//!   integer arithmetic, lane extraction, comparisons and `all`/`any` reductions, and holds when
//!   they are 0; and `c` has no use but the branch;
//! - `A` computes nothing with an effect but its slice writes (one or more), every value it
//!   computes is workgroup uniform (so are the writes' operands) but its pointers into the
//!   imageblock, of which a slice reads only the field's constant offset, and `M` has no phi with
//!   an incoming edge from `A`. Every slice write of `A` is then shared alike.
//!
//! Then every invocation that takes `A` writes the same slice (uniform operands), and invocation 0
//! takes it, so the slice is written exactly as the one write Metal makes: the same cells to the
//! same texels, read after the barrier. Taking `A` in every invocation computes only uniform
//! values without effect. The barrier after the shared loop orders every share before any
//! invocation goes on to store a cell again. A slice write any condition fails keeps the serial
//! loop.

use crate::air::abi::{Builtin, ImageblockKind, ParamRole, Stage};
use crate::air::intrinsics::{
    AirIntrinsic, BoolReduce, Imageblock, Sync, TexOp, TexRole, Texture, ThreadScope,
};
use crate::air::ir::{
    AddrSpace, BinOp, BlockId, Body, Callee, CastOp, Const, InstId, InstKind, IntPredicate, Type,
    Value,
};
use crate::budget::Meter;
use crate::error::Error;
use crate::graph::LoopForest;
use crate::lower::select::SelectCx;
use crate::lower::simd::uniform::{Among, Uniformity};

/// The guard condition of slice write `call` when every invocation may share the write (module
/// docs), else `None`.
pub(super) fn guard(
    cx: &SelectCx<'_>,
    call: InstId,
    m: &mut Meter,
) -> Result<Option<InstId>, Error> {
    let e = cx.entry;
    let body = cx.body;
    if !matches!(e.ep.stage, Stage::Kernel { .. })
        || crate::lower::access::tile::is_tile(cx)
        || !explicit_imageblock(cx)
    {
        return Ok(None);
    }
    let Some(a) = block_of(body, call) else {
        return Ok(None);
    };
    let mut preds = e.cfg.predecessors(a);
    let (Some(p), None) = (preds.next(), preds.next()) else {
        return Ok(None);
    };
    let Some(InstKind::CondBr { c, t, f, .. }) = body.terminator(p).map(|i| &i.kind) else {
        return Ok(None);
    };
    let merge = *f;
    let Some(InstKind::Br { dest, .. }) = body.terminator(a).map(|i| &i.kind) else {
        return Ok(None);
    };
    if *t != a || *dest != merge || merge == a {
        return Ok(None);
    }
    let Value::Inst(cond) = *c else {
        return Ok(None);
    };
    if !at_origin_true(cx, *c, m)? || !only_branch_uses(body, cond, p, m)? {
        return Ok(None);
    }
    if !after_barrier(body, p, m)? || !merge_reads_none_of(body, merge, a, m)? {
        return Ok(None);
    }
    let loops = LoopForest::compute(&e.cfg, m)?;
    let u = Uniformity::under_barriers(e, body, Among::Workgroup, &loops, m)?;
    let held = match u.is_uniform(p) {
        true => region_holds(cx, &u, a, call, m),
        false => Ok(false),
    };
    m.release(u.ledger_bytes());
    m.release(loops.ledger_bytes());
    Ok(held?.then_some(cond))
}

/// The kernel has an explicit imageblock of its own, not one aliasing a tile's.
fn explicit_imageblock(cx: &SelectCx<'_>) -> bool {
    cx.entry.ep.params.iter().any(|b| match &b.role {
        ParamRole::Imageblock(arg) => arg.kind == ImageblockKind::Explicit && !arg.alias_implicit,
        _ => false,
    })
}

/// The block holding instruction `id` (blocks tile the instruction arena in order).
fn block_of(body: &Body, id: InstId) -> Option<BlockId> {
    body.blocks
        .iter()
        .find(|(_, k)| k.start.raw() <= id.raw() && id.raw() < k.end.raw())
        .map(|(b, _)| b)
}

/// Whether `cond`'s only user is block `p`'s terminator.
fn only_branch_uses(body: &Body, cond: InstId, p: BlockId, m: &mut Meter) -> Result<bool, Error> {
    let Some(term) = body.blocks.get(p).and_then(|k| k.last()) else {
        return Ok(false);
    };
    let mut other = false;
    for (at, inst) in body.insts.iter() {
        m.tick(1)?;
        if at != term {
            inst.kind
                .for_each_value(|v| other |= v == Value::Inst(cond));
        }
    }
    Ok(!other)
}

/// Whether an instruction may run in invocations the original did not run it in without an
/// effect: arithmetic, comparisons, casts, selections, vector and aggregate lanes, address
/// arithmetic, a load of `constant` memory, and the imageblock's data pointer.
fn effect_free(cx: &SelectCx<'_>, kind: &InstKind) -> bool {
    match kind {
        InstKind::BinOp { .. }
        | InstKind::FNeg { .. }
        | InstKind::Cast { .. }
        | InstKind::ICmp { .. }
        | InstKind::FCmp { .. }
        | InstKind::Select { .. }
        | InstKind::Freeze(_)
        | InstKind::ExtractElement { .. }
        | InstKind::InsertElement { .. }
        | InstKind::ShuffleVector { .. }
        | InstKind::ExtractValue { .. }
        | InstKind::InsertValue { .. }
        | InstKind::Gep { .. } => true,
        InstKind::Load {
            ptr,
            volatile: false,
            atomic: None,
            ..
        } => {
            let e = cx.entry;
            e.module
                .value_ty(e.func, *ptr)
                .and_then(|t| e.module.types().get(t))
                .is_some_and(|t| {
                    matches!(
                        t,
                        Type::Ptr {
                            space: AddrSpace::Constant
                        }
                    )
                })
        }
        InstKind::Call(call) => matches!(
            call.callee,
            Callee::Air {
                op: AirIntrinsic::Imageblock(Imageblock::Data),
                ..
            }
        ),
        _ => false,
    }
}

/// Whether block `p` holds a threadgroup barrier ordering threadgroup or imageblock memory after
/// which it computes nothing with an effect before its terminator.
fn after_barrier(body: &Body, p: BlockId, m: &mut Meter) -> Result<bool, Error> {
    let Some(k) = body.blocks.get(p) else {
        return Ok(false);
    };
    // Walk back from the terminator to the last barrier.
    for id in k.inst_ids().rev().skip(1) {
        m.tick(1)?;
        let Some(inst) = body.insts.get(id) else {
            return Ok(false);
        };
        if let InstKind::Call(call) = &inst.kind {
            if let Callee::Air {
                op:
                    AirIntrinsic::Sync(Sync::Barrier {
                        flags,
                        scope: ThreadScope::Threadgroup,
                        ..
                    }),
                ..
            } = &call.callee
            {
                return Ok(flags.threadgroup || flags.threadgroup_imageblock);
            }
        }
        if !pure_kind(&inst.kind) {
            return Ok(false);
        }
    }
    Ok(false)
}

/// Arithmetic and lane operations alone (no memory access, no call).
fn pure_kind(kind: &InstKind) -> bool {
    matches!(
        kind,
        InstKind::BinOp { .. }
            | InstKind::FNeg { .. }
            | InstKind::Cast { .. }
            | InstKind::ICmp { .. }
            | InstKind::FCmp { .. }
            | InstKind::Select { .. }
            | InstKind::Freeze(_)
            | InstKind::ExtractElement { .. }
            | InstKind::InsertElement { .. }
            | InstKind::ShuffleVector { .. }
            | InstKind::ExtractValue { .. }
            | InstKind::InsertValue { .. }
    )
}

/// Whether no phi of block `merge` has an incoming edge from block `a`.
fn merge_reads_none_of(
    body: &Body,
    merge: BlockId,
    a: BlockId,
    m: &mut Meter,
) -> Result<bool, Error> {
    let Some(k) = body.blocks.get(merge) else {
        return Ok(false);
    };
    for id in k.inst_ids() {
        m.tick(1)?;
        match body.insts.get(id).map(|i| &i.kind) {
            Some(InstKind::Phi { incoming, .. }) => {
                if incoming.iter().any(|(_, from)| *from == a) {
                    return Ok(false);
                }
            }
            Some(_) => return Ok(true),
            None => return Ok(false),
        }
    }
    Ok(true)
}

/// Whether block `a` computes only workgroup-uniform values without effect besides its slice
/// writes (`call` among them), whose operands are uniform, and its terminator.
fn region_holds(
    cx: &SelectCx<'_>,
    u: &Uniformity,
    a: BlockId,
    call: InstId,
    m: &mut Meter,
) -> Result<bool, Error> {
    if cx
        .body
        .insts
        .get(call)
        .and_then(|i| slice_write(&i.kind))
        .is_none()
    {
        return Ok(false);
    }
    let e = cx.entry;
    let body = cx.body;
    let Some(k) = body.blocks.get(a) else {
        return Ok(false);
    };
    let term = k.last();
    for id in k.inst_ids() {
        m.tick(1)?;
        let Some(inst) = body.insts.get(id) else {
            return Ok(false);
        };
        if Some(id) == term {
            continue;
        }
        if let Some((c, t)) = slice_write(&inst.kind) {
            // The size operand is read only where the explicit-size flag is set
            // (`super::slice::select`).
            let role = |r| t.layout().index(r).and_then(|i| c.args.get(i)).copied();
            let implicit = role(TexRole::Opaque(2)).is_some_and(|f| is_false(cx, f));
            let ignored = role(TexRole::Opaque(4)).filter(|_| implicit);
            if !c.args.iter().all(|v| {
                Some(*v) == ignored || imageblock_pointer(cx, *v) || u.is_uniform_value(e, *v)
            }) {
                return Ok(false);
            }
            continue;
        }
        let v = Value::Inst(id);
        if !effect_free(cx, &inst.kind) || !(imageblock_pointer(cx, v) || u.is_uniform_value(e, v))
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The call of `kind` and its intrinsic when it is a slice write.
fn slice_write(kind: &InstKind) -> Option<(&crate::air::ir::Call, &Texture)> {
    match kind {
        InstKind::Call(c) => match &c.callee {
            Callee::Air {
                op: AirIntrinsic::Texture(t),
                ..
            } if t.op == TexOp::WriteImageblockSlice => Some((c, t)),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `v` is the constant `false`.
fn is_false(cx: &SelectCx<'_>, v: Value) -> bool {
    let Value::Const(k) = v else {
        return false;
    };
    matches!(
        cx.entry.module.consts().get(k).map(|c| &c.value),
        Some(Const::Int(0) | Const::Null)
    )
}

/// Whether `v` is a pointer into the imageblock. A slice write reads only its field's offset,
/// which its selection pins to one constant whatever cell the pointer names
/// (`super::slice::select`), so such a pointer need not be uniform; the pointer is a
/// per-invocation call's result to the uniformity analysis.
fn imageblock_pointer(cx: &SelectCx<'_>, v: Value) -> bool {
    let e = cx.entry;
    e.module
        .value_ty(e.func, v)
        .and_then(|t| e.module.types().get(t))
        .is_some_and(|t| {
            matches!(
                t,
                Type::Ptr {
                    space: AddrSpace::ThreadgroupImageblock
                }
            )
        })
}

/// The deepest expression [`at_origin`] evaluates.
const DEPTH: u32 = 24;

/// Whether `c` is true at local id 0 (module docs): computed from the local id builtins and
/// constants alone.
fn at_origin_true(cx: &SelectCx<'_>, c: Value, m: &mut Meter) -> Result<bool, Error> {
    Ok(at_origin(cx, c, DEPTH, m)?.is_some_and(|lanes| lanes == [1]))
}

/// The lanes of integer value `v` at local id 0, each within its type's width, where `v` is
/// computed from the local id builtins and constants alone; else `None`.
fn at_origin(
    cx: &SelectCx<'_>,
    v: Value,
    depth: u32,
    m: &mut Meter,
) -> Result<Option<Vec<u64>>, Error> {
    m.tick(1)?;
    let e = cx.entry;
    let Some(depth) = depth.checked_sub(1) else {
        return Ok(None);
    };
    let Some((bits, lanes)) = int_shape(cx, v) else {
        return Ok(None);
    };
    let mask = |x: u64| {
        if bits >= 64 {
            x
        } else {
            x & ((1u64 << bits) - 1)
        }
    };
    let out = match v {
        Value::Arg(arg) => {
            let local = e.ep.params.iter().any(|b| {
                b.param == arg.raw()
                    && matches!(
                        b.role,
                        ParamRole::Builtin(
                            Builtin::ThreadPositionInThreadgroup
                                | Builtin::ThreadIndexInThreadgroup
                        )
                    )
            });
            local.then(|| vec![0; lanes])
        }
        Value::Const(k) => match e.module.consts().get(k).map(|c| &c.value) {
            Some(Const::Int(x)) if lanes == 1 => Some(vec![mask(*x)]),
            Some(Const::Null) => Some(vec![0; lanes]),
            Some(Const::Aggregate(parts)) if parts.len() == lanes => {
                let mut out = Vec::with_capacity(lanes);
                for p in parts.iter() {
                    match e.module.consts().get(*p).map(|c| &c.value) {
                        Some(Const::Int(x)) => out.push(mask(*x)),
                        Some(Const::Null) => out.push(0),
                        _ => return Ok(None),
                    }
                }
                Some(out)
            }
            _ => None,
        },
        Value::Inst(id) => {
            let Some(inst) = cx.body.insts.get(id) else {
                return Ok(None);
            };
            match &inst.kind {
                InstKind::BinOp { op, a, b, .. } => {
                    let (Some(x), Some(y)) =
                        (at_origin(cx, *a, depth, m)?, at_origin(cx, *b, depth, m)?)
                    else {
                        return Ok(None);
                    };
                    let f = |x: u64, y: u64| -> Option<u64> {
                        Some(match op {
                            BinOp::Add => x.wrapping_add(y),
                            BinOp::Sub => x.wrapping_sub(y),
                            BinOp::Mul => x.wrapping_mul(y),
                            BinOp::Or => x | y,
                            BinOp::And => x & y,
                            BinOp::Xor => x ^ y,
                            BinOp::Shl if y < u64::from(bits) => x << y,
                            BinOp::LShr if y < u64::from(bits) => x >> y,
                            _ => return None,
                        })
                    };
                    lanewise(&x, &y, f).map(|r| r.into_iter().map(mask).collect())
                }
                InstKind::ICmp { pred, a, b } => {
                    let (Some(x), Some(y)) =
                        (at_origin(cx, *a, depth, m)?, at_origin(cx, *b, depth, m)?)
                    else {
                        return Ok(None);
                    };
                    let f = |x: u64, y: u64| -> Option<u64> {
                        match pred {
                            IntPredicate::Eq => Some(u64::from(x == y)),
                            IntPredicate::Ne => Some(u64::from(x != y)),
                            _ => None,
                        }
                    };
                    lanewise(&x, &y, f)
                }
                InstKind::Cast {
                    op: CastOp::ZExt | CastOp::Trunc,
                    v: from,
                } => at_origin(cx, *from, depth, m)?.map(|x| x.into_iter().map(mask).collect()),
                InstKind::Call(call) => match (&call.callee, &call.args[..]) {
                    (
                        Callee::Air {
                            op: AirIntrinsic::BoolReduce(BoolReduce { all, .. }),
                            ..
                        },
                        [x],
                    ) => at_origin(cx, *x, depth, m)?.map(|lanes| {
                        let set = |l: &u64| *l != 0;
                        let r = match all {
                            true => lanes.iter().all(set),
                            false => lanes.iter().any(set),
                        };
                        vec![u64::from(r)]
                    }),
                    _ => None,
                },
                InstKind::ExtractElement { vec, idx } => {
                    let (Some(x), Some(i)) = (
                        at_origin(cx, *vec, depth, m)?,
                        at_origin(cx, *idx, depth, m)?,
                    ) else {
                        return Ok(None);
                    };
                    match i[..] {
                        [i] => usize::try_from(i)
                            .ok()
                            .and_then(|i| x.get(i).copied())
                            .map(|x| vec![x]),
                        _ => None,
                    }
                }
                _ => None,
            }
        }
    };
    Ok(out.filter(|o| o.len() == lanes))
}

/// `f` of `x` and `y` lane by lane, lanes of equal count.
fn lanewise(x: &[u64], y: &[u64], f: impl Fn(u64, u64) -> Option<u64>) -> Option<Vec<u64>> {
    if x.len() != y.len() {
        return None;
    }
    x.iter().zip(y).map(|(a, b)| f(*a, *b)).collect()
}

/// The integer width and lane count of value `v` (an `i1` is 1 bit), or `None` for any other type.
fn int_shape(cx: &SelectCx<'_>, v: Value) -> Option<(u8, usize)> {
    let e = cx.entry;
    let types = e.module.types();
    let t = types.get(e.module.value_ty(e.func, v)?)?;
    match *t {
        Type::Int(bits) => Some((bits, 1)),
        Type::Vector { elem, len } => match types.get(elem)? {
            Type::Int(bits) => Some((*bits, usize::try_from(len).ok()?)),
            _ => None,
        },
        _ => None,
    }
}
