//! The capture assembly and settle kernels: [`assembly_typed_module`] and
//! [`settle_typed_module`], which [`capture_assembly_module`](super::capture_assembly_module) and
//! [`capture_settle_module`](super::capture_settle_module) serialize and the CPU interpreter
//! runs, by the module docs' rules, step for step as the tests' reference assembly states them
//! (`tests.rs`).
//!
//! **No division and no float.** Each kernel copies `f32`s as their bits and computes in `u32`.
//! The records, entries, primitives and mask words a binding holds come by shifts (each record's
//! words are a power of two), an invocation's entry by a binary search, its place in the entry by
//! a subtraction (an entry is one instance), and a triangle's parity by a scan of at most
//! [`SCAN_LIMIT`](A::SCAN_LIMIT) records: nothing divides by a value it reads, which the tile
//! reach kernel's docs explain the price of.

use crate::bind::status::{StatusBits, StatusWord};
use crate::bind::tile_reach::{layout as reach, TileReachModule as R, AREA_LIMIT};
use crate::metal::PrimitiveType;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, AtomicOp, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, Scope,
    Semantics, StorageClass, Type, TypedModule, UnOp,
};

use super::{layout, CaptureAssemblyModule as A, CaptureSettleModule as S, NOTHING, UNKNOWN};

/// The binding symbols of the assembly's records, draws and primitives in its typed module.
pub(super) const RECORDS: ResourceSlot = slot(0);
pub(super) const DRAWS: ResourceSlot = slot(1);
pub(super) const PRIMITIVES: ResourceSlot = slot(2);
/// The binding symbols of the settle's pass, tile mask and `ExecutionStatus` words in its typed
/// module.
pub(super) const PASS: ResourceSlot = slot(0);
pub(super) const TILES: ResourceSlot = slot(1);
pub(super) const STATUS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
    index: 0,
};

const fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// The raw topologies (`MTLPrimitiveType`).
const POINTS: u32 = PrimitiveType::Point.as_raw();
const LINES: u32 = PrimitiveType::Line.as_raw();
const LINE_STRIP: u32 = PrimitiveType::LineStrip.as_raw();
const TRIANGLES: u32 = PrimitiveType::Triangle.as_raw();
const TRIANGLE_STRIP: u32 = PrimitiveType::TriangleStrip.as_raw();

/// The shift that counts a binding's records of `words` words, a power of two.
const fn shift(words: u32) -> u32 {
    assert!(words.is_power_of_two());
    words.trailing_zeros()
}
const RECORD_SHIFT: u32 = shift(A::RECORD_WORDS);
const DRAW_SHIFT: u32 = shift(A::DRAW_WORDS);
const PRIMITIVE_SHIFT: u32 = shift(R::PRIMITIVE_WORDS);

fn k32(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

fn op(f: &mut FunctionBuilder<'_>, o: BinOp, a: Value, b: Value) -> BResult<Value> {
    f.binary(o, a, b)
}

/// `a o v` for a `u32` constant `v`.
fn opk(f: &mut FunctionBuilder<'_>, o: BinOp, a: Value, v: u32) -> BResult<Value> {
    let c = k32(f, v)?;
    f.binary(o, a, c)
}

fn sel(f: &mut FunctionBuilder<'_>, c: Value, a: Value, b: Value) -> BResult<Value> {
    f.select(c, a, b)
}

fn and(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    f.binary(BinOp::LogicalAnd, a, b)
}

fn or(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    f.binary(BinOp::LogicalOr, a, b)
}

fn not(f: &mut FunctionBuilder<'_>, a: Value) -> BResult<Value> {
    f.unary(UnOp::LogicalNot, a)
}

/// The smaller of two `u32`s.
fn umin(f: &mut FunctionBuilder<'_>, a: Value, b: Value) -> BResult<Value> {
    let less = op(f, BinOp::ULessThan, a, b)?;
    sel(f, less, a, b)
}

/// Whether a `u32` is not zero.
fn nonzero(f: &mut FunctionBuilder<'_>, a: Value) -> BResult<Value> {
    opk(f, BinOp::INotEqual, a, 0)
}

fn load(f: &mut FunctionBuilder<'_>, p: Value) -> BResult<Value> {
    f.load(p, MemAccess::default())
}

fn store(f: &mut FunctionBuilder<'_>, p: Value, v: Value) -> BResult<()> {
    f.store(p, v, MemAccess::default())
}

/// A `Function` `u32` holding `v`, as its pointer.
fn local(f: &mut FunctionBuilder<'_>, u32_: TypeId, v: Value) -> BResult<Value> {
    let var = f.local_var(u32_, None)?;
    let p = f.var_ptr(var)?;
    store(f, p, v)?;
    Ok(p)
}

/// Word `i` (a `u32`) of a buffer of words, as a pointer.
fn word_ptr(f: &mut FunctionBuilder<'_>, buffer: Value, i: Value) -> BResult<Value> {
    let zero = k32(f, 0)?;
    f.access_chain(buffer, &[zero, i])
}

/// Word `i` (a `u32`) of a buffer of words.
fn word(f: &mut FunctionBuilder<'_>, buffer: Value, i: Value) -> BResult<Value> {
    let p = word_ptr(f, buffer, i)?;
    load(f, p)
}

/// Word `base + at` of a buffer of words.
fn field(f: &mut FunctionBuilder<'_>, buffer: Value, base: Value, at: u32) -> BResult<Value> {
    let i = opk(f, BinOp::IAdd, base, at)?;
    word(f, buffer, i)
}

/// Run `body` where `c` holds.
fn when<'a>(
    f: &mut FunctionBuilder<'_>,
    c: Value,
    body: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
) -> BResult<()> {
    let arm: ArmFn<'_, ()> = Box::new(move |f, _| {
        body(f)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), c, arm, None)?;
    Ok(())
}

/// Run `then` where `c` holds and `other` where it does not.
fn either<'a>(
    f: &mut FunctionBuilder<'_>,
    c: Value,
    then: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
    other: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
) -> BResult<()> {
    let yes: ArmFn<'_, ()> = Box::new(move |f, _| {
        then(f)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    let no: ArmFn<'_, ()> = Box::new(move |f, _| {
        other(f)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), c, yes, Some(no))?;
    Ok(())
}

/// Loop: each pass, leave where `stop` (computed first, at the top) holds, else run `body`.
fn repeat<'a>(
    f: &mut FunctionBuilder<'_>,
    stop: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<Value> + 'a,
    body: impl FnOnce(&mut FunctionBuilder<'_>) -> BResult<()> + 'a,
) -> BResult<()> {
    let walk: LoopFn<'_, ()> = Box::new(move |f, _, lc| {
        let done = stop(f)?;
        let leave: ArmFn<'_, ()> = Box::new(|f, _| {
            lc.break_(f)?;
            Ok(ArmEnd::Exits)
        });
        f.if_(&mut (), done, leave, None)?;
        body(f)
    });
    f.loop_(&mut (), LoopSpec::default(), walk)
}

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

/// The `u32` type and the `uvec3` of an invocation's id.
fn types(b: &mut ModuleBuilder) -> BResult<(TypeId, TypeId)> {
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    Ok((u32_, uvec3))
}

/// Make `id` the `GLCompute` entry point of local size (`x`, 1, 1).
fn entry(b: &mut ModuleBuilder, id: FuncId, x: u32) -> BResult<()> {
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x, y: 1, z: 1 }],
    )?;
    Ok(())
}

/// The assembly as a typed module, before serialization: the module
/// [`capture_assembly_module`](super::capture_assembly_module) serializes, which the CPU
/// interpreter runs. Its bindings are the symbols `RECORDS`, `DRAWS` and `PRIMITIVES`.
pub(super) fn assembly_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let (u32_, uvec3) = types(&mut b)?;
    let records = words_buffer(&mut b, RECORDS, u32_)?;
    let draws = words_buffer(&mut b, DRAWS, u32_)?;
    let prims = words_buffer(&mut b, PRIMITIVES, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let records = f.module().global_value(records)?;
    let draws = f.module().global_value(draws)?;
    let prims = f.module().global_value(prims)?;
    let invocation = f.module().global_value(invocation)?;
    assembly_body(&mut f, u32_, [records, draws, prims], invocation)?;
    f.ret(None)?;
    let id = f.finish()?;
    entry(&mut b, id, A::LOCAL_SIZE)?;
    b.finalize()
}

/// The assembly's buffers, and the whole records its records' binding holds.
#[derive(Clone, Copy)]
struct G {
    u32_: TypeId,
    records: Value,
    draws: Value,
    prims: Value,
    held: Value,
}

/// An entry's words (the module docs' table).
#[derive(Clone, Copy)]
struct Entry {
    first: Value,
    primitives: Value,
    first_record: Value,
    records: Value,
    topology: Value,
    restarts: Value,
    culls: Value,
    state: Value,
}

/// The entry's body: invocation `i` below the header's total and the primitives the binding
/// holds writes primitive `i`, from its entry where the header names one, else unknown.
fn assembly_body(
    f: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    [records, draws, prims]: [Value; 3],
    invocation: Value,
) -> BResult<()> {
    let gid = load(f, invocation)?;
    let i = f.composite_extract(gid, &[0])?;
    let len = f.array_length(records, 0)?;
    let g = G {
        u32_,
        records,
        draws,
        prims,
        held: opk(f, BinOp::ShiftRightLogical, len, RECORD_SHIFT)?,
    };
    let len = f.array_length(prims, 0)?;
    let out = opk(f, BinOp::ShiftRightLogical, len, PRIMITIVE_SHIFT)?;
    let words = f.array_length(draws, 0)?;
    let header = opk(f, BinOp::UGreaterThanEqual, words, A::HEADER_WORDS)?;
    when(f, header, move |f| {
        let at = k32(f, layout::TOTAL)?;
        let total = word(f, g.draws, at)?;
        let at = k32(f, layout::ENTRIES)?;
        let named = word(f, g.draws, at)?;
        let after = opk(f, BinOp::ISub, words, A::HEADER_WORDS)?;
        let entries = opk(f, BinOp::ShiftRightLogical, after, DRAW_SHIFT)?;
        let count = umin(f, named, entries)?;
        let below = op(f, BinOp::ULessThan, i, total)?;
        let fits = op(f, BinOp::ULessThan, i, out)?;
        let go = and(f, below, fits)?;
        when(f, go, move |f| {
            let some = opk(f, BinOp::UGreaterThan, count, 0)?;
            either(
                f,
                some,
                move |f| {
                    let e = search(f, g, i, count)?;
                    let e = read_entry(f, g, e)?;
                    primitive(f, g, i, e)
                },
                move |f| {
                    let zero = k32(f, 0)?;
                    placeholder(f, g, i, zero, UNKNOWN)
                },
            )
        })
    })
}

/// The index of the last of the `count` (at least one) entries whose first primitive is at most
/// `i`, by a binary search (the entries' first primitives do not decrease): entry 0 where none
/// is.
fn search(f: &mut FunctionBuilder<'_>, g: G, i: Value, count: Value) -> BResult<Value> {
    let zero = k32(f, 0)?;
    let lo = local(f, g.u32_, zero)?;
    let hi = local(f, g.u32_, count)?;
    repeat(
        f,
        move |f| {
            let l = load(f, lo)?;
            let h = load(f, hi)?;
            let span = op(f, BinOp::ISub, h, l)?;
            opk(f, BinOp::ULessThanEqual, span, 1)
        },
        move |f| {
            let l = load(f, lo)?;
            let h = load(f, hi)?;
            // Below 2^29 entries each: the sum does not wrap.
            let mid = op(f, BinOp::IAdd, l, h)?;
            let mid = opk(f, BinOp::ShiftRightLogical, mid, 1)?;
            let base = entry_base(f, mid)?;
            let first = field(f, g.draws, base, layout::FIRST_PRIMITIVE)?;
            let up = op(f, BinOp::ULessThanEqual, first, i)?;
            let l = sel(f, up, mid, l)?;
            let h = sel(f, up, h, mid)?;
            store(f, lo, l)?;
            store(f, hi, h)
        },
    )?;
    load(f, lo)
}

/// The first word of entry `e`.
fn entry_base(f: &mut FunctionBuilder<'_>, e: Value) -> BResult<Value> {
    let at = opk(f, BinOp::ShiftLeftLogical, e, DRAW_SHIFT)?;
    opk(f, BinOp::IAdd, at, A::HEADER_WORDS)
}

/// Entry `e`'s words.
fn read_entry(f: &mut FunctionBuilder<'_>, g: G, e: Value) -> BResult<Entry> {
    let base = entry_base(f, e)?;
    let mut w = |at: u32| field(f, g.draws, base, at);
    Ok(Entry {
        first: w(layout::FIRST_PRIMITIVE)?,
        primitives: w(layout::PRIMITIVES)?,
        first_record: w(layout::FIRST_RECORD)?,
        records: w(layout::RECORDS)?,
        topology: w(layout::TOPOLOGY)?,
        restarts: w(layout::RESTARTS)?,
        culls: w(layout::CULLS)?,
        state: w(layout::STATE)?,
    })
}

/// Words `at` of record `v` (a `u32`) of the entry's instance, and whether it is there: `v`
/// below the entry's records and its slot below the binding's records. A record not there reads
/// as zeros, and nothing is read.
fn fetch<const N: usize>(
    f: &mut FunctionBuilder<'_>,
    g: G,
    e: &Entry,
    v: Value,
    at: [u32; N],
) -> BResult<([Value; N], Value)> {
    let within = op(f, BinOp::ULessThan, v, e.records)?;
    let slot = op(f, BinOp::IAdd, e.first_record, v)?;
    let held = op(f, BinOp::ULessThan, slot, g.held)?;
    let there = and(f, within, held)?;
    let zero = k32(f, 0)?;
    let mut vars = [zero; N];
    for p in &mut vars {
        *p = local(f, g.u32_, zero)?;
    }
    when(f, there, move |f| {
        let base = opk(f, BinOp::ShiftLeftLogical, slot, RECORD_SHIFT)?;
        for (p, at) in vars.into_iter().zip(at) {
            let w = field(f, g.records, base, at)?;
            store(f, p, w)?;
        }
        Ok(())
    })?;
    let mut words = [zero; N];
    for (w, p) in words.iter_mut().zip(vars) {
        *w = load(f, p)?;
    }
    Ok((words, there))
}

/// A vertex the primitive reads: its position's four words, its point size, whether it
/// restarts (a `bool`), and whether it is there.
#[derive(Clone, Copy)]
struct Vertex {
    position: [Value; 4],
    size: Value,
    restart: Value,
    there: Value,
}

/// Record `v` of the entry's instance as a vertex.
fn vertex(f: &mut FunctionBuilder<'_>, g: G, e: &Entry, v: Value) -> BResult<Vertex> {
    let p = layout::POSITION;
    let at = [p, p + 1, p + 2, p + 3, layout::POINT_SIZE, layout::RESTART];
    let (w, there) = fetch(f, g, e, v, at)?;
    Ok(Vertex {
        position: [w[0], w[1], w[2], w[3]],
        size: w[4],
        restart: nonzero(f, w[5])?,
        there,
    })
}

/// Write primitive `i` of entry `e` (the module docs): its records, a restart's placeholder, or
/// the unknown one.
fn primitive(f: &mut FunctionBuilder<'_>, g: G, i: Value, e: Entry) -> BResult<()> {
    let j = op(f, BinOp::ISub, i, e.first)?;
    let in_entry = op(f, BinOp::ULessThan, j, e.primitives)?;
    let named = opk(f, BinOp::ULessThanEqual, e.topology, TRIANGLE_STRIP)?;
    let is = |f: &mut FunctionBuilder<'_>, t: u32| opk(f, BinOp::IEqual, e.topology, t);
    let lines = is(f, LINES)?;
    let triangles = is(f, TRIANGLES)?;
    let line_strip = is(f, LINE_STRIP)?;
    let triangle_strip = is(f, TRIANGLE_STRIP)?;
    let point = is(f, POINTS)?;
    // The primitive's first record: `j` times its records in a list, `j` in a strip or points.
    let (k1, k2, k3) = (k32(f, 1)?, k32(f, 2)?, k32(f, 3)?);
    let step = sel(f, triangles, k3, k1)?;
    let step = sel(f, lines, k2, step)?;
    let a = op(f, BinOp::IMul, j, step)?;
    let b = opk(f, BinOp::IAdd, a, 1)?;
    let c = opk(f, BinOp::IAdd, a, 2)?;
    let two = not(f, point)?;
    let three = or(f, triangles, triangle_strip)?;
    let va = vertex(f, g, &e, a)?;
    let vb = vertex(f, g, &e, b)?;
    let vc = vertex(f, g, &e, c)?;
    // Each vertex its kind reads is there.
    let no_b = not(f, two)?;
    let has_b = or(f, no_b, vb.there)?;
    let no_c = not(f, three)?;
    let has_c = or(f, no_c, vc.there)?;
    let there = and(f, va.there, has_b)?;
    let there = and(f, there, has_c)?;
    let ok = and(f, in_entry, named)?;
    let ok = and(f, ok, there)?;
    // A restarting strip's primitive one of whose records is flagged reaches nothing.
    let strip = or(f, line_strip, triangle_strip)?;
    let restarts = nonzero(f, e.restarts)?;
    let restarting = and(f, strip, restarts)?;
    let touched = or(f, va.restart, vb.restart)?;
    let c_flag = and(f, three, vc.restart)?;
    let touched = or(f, touched, c_flag)?;
    let nothing = and(f, restarting, touched)?;
    // A triangle strip's parity: from vertex 0, or, where it restarts and culls, from its piece's
    // first record, unknown past the scan.
    let parity = opk(f, BinOp::BitwiseAnd, j, 1)?;
    let parity = local(f, g.u32_, parity)?;
    let zero = k32(f, 0)?;
    let lost = local(f, g.u32_, zero)?;
    let culls = nonzero(f, e.culls)?;
    let scan = and(f, triangle_strip, restarting)?;
    let scan = and(f, scan, culls)?;
    let kept = not(f, nothing)?;
    let scan = and(f, scan, kept)?;
    let scan = and(f, scan, ok)?;
    when(f, scan, move |f| {
        let (odd, known) = piece_parity(f, g, &e, j)?;
        store(f, parity, odd)?;
        let (k0, k1) = (k32(f, 0)?, k32(f, 1)?);
        let unknown = sel(f, known, k0, k1)?;
        store(f, lost, unknown)
    })?;
    let odd = load(f, parity)?;
    let odd = nonzero(f, odd)?;
    let lost = load(f, lost)?;
    let lost = nonzero(f, lost)?;
    let fine = not(f, lost)?;
    let fine = and(f, ok, fine)?;
    either(
        f,
        fine,
        move |f| {
            either(
                f,
                nothing,
                move |f| placeholder(f, g, i, e.state, NOTHING),
                move |f| {
                    // Vulkan's strip order: an odd triangle takes its records `j, j + 2, j + 1`.
                    let swap = and(f, triangle_strip, odd)?;
                    let zero = k32(f, 0)?;
                    let mut second = [zero; 4];
                    let mut third = [zero; 4];
                    let pairs = vb.position.into_iter().zip(vc.position);
                    for ((s, t), (b, c)) in second.iter_mut().zip(&mut third).zip(pairs) {
                        let to_second = sel(f, swap, c, b)?;
                        *s = sel(f, two, to_second, zero)?;
                        let to_third = sel(f, swap, b, c)?;
                        *t = sel(f, three, to_third, zero)?;
                    }
                    let (kp, kl, kt) = (k32(f, R::POINT)?, k32(f, R::LINE)?, k32(f, R::TRIANGLE)?);
                    let kind = sel(f, three, kt, kl)?;
                    let kind = sel(f, point, kp, kind)?;
                    let vertices = [va.position, second, third];
                    emit(f, g, i, [kind, e.state, va.size], vertices)
                },
            )
        },
        move |f| placeholder(f, g, i, e.state, UNKNOWN),
    )
}

/// Primitive `j`'s parity in its piece (1 where odd) and whether it is known (a `bool`): the
/// last flagged record within [`SCAN_LIMIT`](A::SCAN_LIMIT) records before `j` starts its piece
/// after it; with none, the piece starts at vertex 0, known where the scan reached it.
fn piece_parity(f: &mut FunctionBuilder<'_>, g: G, e: &Entry, j: Value) -> BResult<(Value, Value)> {
    let bound = k32(f, A::SCAN_LIMIT)?;
    let limit = umin(f, j, bound)?;
    let (zero, one) = (k32(f, 0)?, k32(f, 1)?);
    let back = local(f, g.u32_, one)?;
    let found = local(f, g.u32_, zero)?;
    let start = local(f, g.u32_, zero)?;
    let e = *e;
    repeat(
        f,
        move |f| {
            let d = load(f, back)?;
            let past = op(f, BinOp::UGreaterThan, d, limit)?;
            let got = load(f, found)?;
            let got = nonzero(f, got)?;
            or(f, past, got)
        },
        move |f| {
            let d = load(f, back)?;
            let q = op(f, BinOp::ISub, j, d)?;
            let ([flag], _) = fetch(f, g, &e, q, [layout::RESTART])?;
            let flagged = nonzero(f, flag)?;
            when(f, flagged, move |f| {
                let one = k32(f, 1)?;
                store(f, found, one)?;
                let after = opk(f, BinOp::IAdd, q, 1)?;
                store(f, start, after)
            })?;
            let next = opk(f, BinOp::IAdd, d, 1)?;
            store(f, back, next)
        },
    )?;
    let got = load(f, found)?;
    let got = nonzero(f, got)?;
    let near = op(f, BinOp::ULessThanEqual, j, bound)?;
    let known = or(f, got, near)?;
    let first = load(f, start)?;
    let into = op(f, BinOp::ISub, j, first)?;
    let odd = opk(f, BinOp::BitwiseAnd, into, 1)?;
    Ok((odd, known))
}

/// Write primitive `i` as a triangle of three `vertex`es (`NOTHING` or `UNKNOWN`) of size 0,
/// under `state`.
fn placeholder(
    f: &mut FunctionBuilder<'_>,
    g: G,
    i: Value,
    state: Value,
    vertex: [u32; 4],
) -> BResult<()> {
    let kind = k32(f, R::TRIANGLE)?;
    let size = k32(f, 0)?;
    let mut v = [size; 4];
    for (w, bits) in v.iter_mut().zip(vertex) {
        *w = k32(f, bits)?;
    }
    emit(f, g, i, [kind, state, size], [v; 3])
}

/// Write primitive `i`'s record: its kind, state and size, layer 0, and its vertices.
fn emit(
    f: &mut FunctionBuilder<'_>,
    g: G,
    i: Value,
    [kind, state, size]: [Value; 3],
    vertices: [[Value; 4]; 3],
) -> BResult<()> {
    let base = opk(f, BinOp::ShiftLeftLogical, i, PRIMITIVE_SHIFT)?;
    let zero = k32(f, 0)?;
    let mut words = vec![
        (reach::KIND, kind),
        (reach::STATE, state),
        (reach::LAYER, zero),
        (reach::SIZE, size),
    ];
    for (v, vertex) in vertices.iter().enumerate() {
        for (c, &w) in vertex.iter().enumerate() {
            words.push((reach::VERTICES + 4 * v as u32 + c as u32, w));
        }
    }
    for (at, w) in words {
        let to = opk(f, BinOp::IAdd, base, at)?;
        let p = word_ptr(f, g.prims, to)?;
        store(f, p, w)?;
    }
    Ok(())
}

/// The settle as a typed module, before serialization: the module
/// [`capture_settle_module`](super::capture_settle_module) serializes, which the CPU interpreter
/// runs. Its bindings are the symbols `PASS`, `TILES` and `STATUS`.
pub(super) fn settle_typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let (u32_, uvec3) = types(&mut b)?;
    let pass = words_buffer(&mut b, PASS, u32_)?;
    let tiles = words_buffer(&mut b, TILES, u32_)?;
    let status = words_buffer(&mut b, STATUS, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let pass = f.module().global_value(pass)?;
    let tiles = f.module().global_value(tiles)?;
    let status = f.module().global_value(status)?;
    let invocation = f.module().global_value(invocation)?;
    settle_body(&mut f, [pass, tiles, status], invocation)?;
    f.ret(None)?;
    let id = f.finish()?;
    entry(&mut b, id, S::LOCAL_SIZE)?;
    b.finalize()
}

/// The settle's body: where the header counted a primitive near-far or unmeasured, invocation
/// `i` below the pass's mask words and the binding's words ORs the bits of the pass's tiles in
/// word `i` into it, and invocation 0 ORs `CAPTURED_REACH_UNMEASURED` into the status word.
fn settle_body(
    f: &mut FunctionBuilder<'_>,
    [pass, tiles, status]: [Value; 3],
    invocation: Value,
) -> BResult<()> {
    let gid = load(f, invocation)?;
    let i = f.composite_extract(gid, &[0])?;
    let len = f.array_length(pass, 0)?;
    let header = opk(f, BinOp::UGreaterThanEqual, len, R::HEADER_WORDS)?;
    let held = f.array_length(tiles, 0)?;
    when(f, header, move |f| {
        let at = |f: &mut FunctionBuilder<'_>, w: u32| {
            let w = k32(f, w)?;
            word(f, pass, w)
        };
        let near_far = at(f, R::NEAR_FAR_WORD)?;
        let unmeasured = at(f, R::UNMEASURED_WORD)?;
        let near_far = nonzero(f, near_far)?;
        let unmeasured = nonzero(f, unmeasured)?;
        let fell = or(f, near_far, unmeasured)?;
        let first = opk(f, BinOp::IEqual, i, 0)?;
        let report = and(f, fell, first)?;
        when(f, report, move |f| {
            let at = k32(f, StatusWord::Status.index())?;
            let p = word_ptr(f, status, at)?;
            let bit = k32(f, StatusBits::CAPTURED_REACH_UNMEASURED.0)?;
            f.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
            Ok(())
        })?;
        // The pass's tiles (`TileReachModule::mask_words`): its render area binned at most
        // `AREA_LIMIT` a side, in tiles of 32 by 32 pixels, 32 by 16 at 4 samples.
        let samples = at(f, reach::SAMPLES)?;
        let four = opk(f, BinOp::IEqual, samples, 4)?;
        let (k4, k5) = (k32(f, 4)?, k32(f, 5)?);
        let log = sel(f, four, k4, k5)?;
        let limit = k32(f, AREA_LIMIT)?;
        let mut sides = [log; 2];
        for (side, (w, log)) in sides
            .iter_mut()
            .zip([(reach::AREA_W, k5), (reach::AREA_H, log)])
        {
            let extent = at(f, w)?;
            let extent = umin(f, extent, limit)?;
            let one = k32(f, 1)?;
            let size = op(f, BinOp::ShiftLeftLogical, one, log)?;
            let up = op(f, BinOp::IAdd, extent, size)?;
            let up = opk(f, BinOp::ISub, up, 1)?;
            *side = op(f, BinOp::ShiftRightLogical, up, log)?;
        }
        let layers = at(f, reach::LAYERS)?;
        let count = op(f, BinOp::IMul, sides[0], sides[1])?;
        let count = op(f, BinOp::IMul, count, layers)?;
        let words = opk(f, BinOp::IAdd, count, 31)?;
        let words = opk(f, BinOp::ShiftRightLogical, words, 5)?;
        let inside = op(f, BinOp::ULessThan, i, words)?;
        let bound = op(f, BinOp::ULessThan, i, held)?;
        let go = and(f, inside, bound)?;
        let go = and(f, go, fell)?;
        when(f, go, move |f| {
            // The tiles from word `i`'s first on: all 32 bits, or the low `rest`.
            let first = opk(f, BinOp::ShiftLeftLogical, i, 5)?;
            let rest = op(f, BinOp::ISub, count, first)?;
            let full = opk(f, BinOp::UGreaterThanEqual, rest, 32)?;
            let zero = k32(f, 0)?;
            let by = sel(f, full, zero, rest)?;
            let one = k32(f, 1)?;
            let part = op(f, BinOp::ShiftLeftLogical, one, by)?;
            let part = opk(f, BinOp::ISub, part, 1)?;
            let all = k32(f, u32::MAX)?;
            let bits = sel(f, full, all, part)?;
            let p = word_ptr(f, tiles, i)?;
            let old = load(f, p)?;
            let new = op(f, BinOp::BitwiseOr, old, bits)?;
            store(f, p, new)
        })
    })
}
