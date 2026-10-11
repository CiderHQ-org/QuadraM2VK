//! Values SPIRV-Cross would forward into their one reader without the dependencies of their
//! operands are materialized: an `OpQuantizeToF16`, an `OpImageQuerySizeLod` of a level that is not
//! a constant, an `OpUMulExtended` or `OpSMulExtended` with a factor that is not a constant, an
//! `InterpolateAtSample` or `InterpolateAtOffset` whose sample or offset is not a constant, or an
//! `OpFunctionCall` with an argument that is not a constant, read once, is written under its own id
//! and followed by the `OpCopyObject` its reader reads. One read twice (SPIRV-Cross declares it
//! where it stands), one of constants only, and one never read are written as before. MoltenVK's
//! MSL then evaluates each where the module does, before a later write overwrites what it reads.

use super::families::tys;
use super::families_image::{image_ty, load_global, texture, Img, SAMPLED_2D};
use super::{assert_spirv_valid, compute, cval, elem_ptr, finish_compute, gid_x, key, ssbo};
use crate::pipeline::testing::{moltenvk_msl, AssembleError};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::ids::{FuncId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, Constant, ExecModel, GlslOp, ImageOperands, ImageQuery, ImageUsage, IoDecor, IoDir,
    MemAccess, TexelMemory, Type, TypedModule, UnOp,
};
use crate::spirv_binary::{Instruction, Operand};
use crate::spv::{GlslStd450Op, ImageFormat, Op};

fn parse(words: &[u32]) -> Vec<Instruction> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    crate::spirv_binary::parse_bytes(&bytes)
        .expect("parse")
        .instructions
}

fn ids(i: &Instruction) -> Vec<u32> {
    i.operands
        .iter()
        .filter_map(|o| match o {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect()
}

/// `vec2 neg(vec2 v) { return -v; }`.
fn negate(b: &mut ModuleBuilder, vec2: TypeId) -> FuncId {
    let mut f = b.function(vec2).expect("callee");
    let v = f.param(vec2).expect("v");
    let r = f.unary(UnOp::FNegate, v).expect("negate");
    f.ret(Some(r)).expect("ret");
    f.finish().expect("finish")
}

/// What an instruction is: its opcode, and for an `OpExtInst` its `GLSL.std.450` instruction.
#[derive(Debug, PartialEq)]
enum Kind {
    Op(Op),
    Glsl(u32),
}

fn kind(i: &Instruction) -> Kind {
    match (i.opcode, i.operands.get(1)) {
        (Op::ExtInst, Some(Operand::LiteralExtInstInteger(n))) => Kind::Glsl(*n),
        (op, _) => Kind::Op(op),
    }
}

/// What each `OpCopyObject` copies, checking that the copy directly follows it, in order.
fn copied(insts: &[Instruction]) -> Vec<Kind> {
    let mut copied = Vec::new();
    for (at, i) in insts.iter().enumerate() {
        if i.opcode != Op::CopyObject {
            continue;
        }
        let before = &insts[at - 1];
        assert_eq!(
            before.result_id,
            Some(ids(i)[0]),
            "the copy follows its operand"
        );
        copied.push(kind(before));
    }
    copied
}

/// Nothing but its copy reads a copied instruction's own id.
fn assert_only_copies_read_own_ids(insts: &[Instruction]) {
    let own = |id: u32| {
        insts
            .iter()
            .position(|d| d.result_id == Some(id))
            .is_some_and(|d| insts[d + 1].opcode == Op::CopyObject)
    };
    for i in insts.iter().filter(|i| i.opcode != Op::CopyObject) {
        for id in ids(i) {
            assert!(
                !own(id),
                "{:?} reads a materialized instruction's own id",
                i.opcode
            );
        }
    }
}

#[test]
fn values_forwarded_without_dependencies_are_copied_when_read_once() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let vectors = ssbo(&mut b, 0, t.vec2, 8);
    let ints = ssbo(&mut b, 1, t.u32_, 4);
    let tex_ty = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let tex = texture(&mut b, 2, tex_ty);
    let neg = negate(&mut b, t.vec2);
    let half = b
        .constant(Constant::Scalar {
            ty: t.f32_,
            bits: crate::spirv::ir::ConstBits(0x3f00_0000),
        })
        .expect("0.5");
    let halves = b
        .constant(Constant::Composite {
            ty: t.vec2,
            parts: [half, half].into(),
        })
        .expect("(0.5, 0.5)");
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let at = elem_ptr(f, vectors, i);
        let x = f.load(at, MemAccess::default()).expect("load");
        let k = f.module().const_value(halves).expect("constant");
        let store = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, v| {
            let dst = elem_ptr(f, vectors, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        };
        // Read once: copied.
        let q = f.unary(UnOp::QuantizeToF16, x).expect("quantize");
        store(f, q);
        // Read once through a copy (written as its operand): copied.
        let q = f.unary(UnOp::QuantizeToF16, x).expect("quantize");
        let q = f.unary(UnOp::CopyObject, q).expect("copy");
        store(f, q);
        // Read twice, by two instructions: SPIRV-Cross declares it where it stands.
        let q = f.unary(UnOp::QuantizeToF16, x).expect("quantize");
        store(f, q);
        let s = f.binary(BinOp::FAdd, q, x).expect("add");
        store(f, s);
        // Of a constant, which nothing overwrites.
        let q = f.unary(UnOp::QuantizeToF16, k).expect("quantize");
        store(f, q);
        // Never read.
        f.unary(UnOp::QuantizeToF16, x).expect("quantize");
        // A size at a loaded level, read once: copied; at a constant level: not.
        let img = load_global(f, tex);
        let at = elem_ptr(f, ints, i);
        let level = f.load(at, MemAccess::default()).expect("level");
        let zero = cval(f, t.u32_, 0);
        for level in [level, zero] {
            let size = f.query(ImageQuery::SizeLod(level), img).expect("size lod");
            let w = f.composite_extract(size, &[0]).expect("width");
            let dst = elem_ptr(f, ints, i);
            f.store(dst, w, MemAccess::default()).expect("store");
        }
        // A wide product of a loaded factor, read once: copied; read twice (both halves), or of
        // constants only: not. Either signedness.
        let at = elem_ptr(f, ints, i);
        let u = f.load(at, MemAccess::default()).expect("factor");
        let seven = cval(f, t.u32_, 7);
        for (op, a, halves) in [
            (BinOp::UMulExtended, u, 1),
            (BinOp::SMulExtended, u, 1),
            (BinOp::UMulExtended, u, 2),
            (BinOp::UMulExtended, seven, 1),
        ] {
            let p = f.binary(op, a, seven).expect("wide product");
            for half in 0..halves {
                let h = f.composite_extract(p, &[1 - half]).expect("half");
                let dst = elem_ptr(f, ints, i);
                f.store(dst, h, MemAccess::default()).expect("store");
            }
        }
        // A call of a loaded value, read once: copied; of constants only: not.
        for arg in [x, k] {
            let r = f.call(neg, &[arg]).expect("call").expect("value");
            store(f, r);
        }
    });
    let words = assert_spirv_valid(finish_compute(b, main));
    let insts = parse(&words);
    assert_eq!(
        copied(&insts),
        [
            Kind::Op(Op::QuantizeToF16),
            Kind::Op(Op::QuantizeToF16),
            Kind::Op(Op::ImageQuerySizeLod),
            Kind::Op(Op::UMulExtended),
            Kind::Op(Op::SMulExtended),
            Kind::Op(Op::FunctionCall),
        ],
        "exactly the values read once whose operands a write may overwrite"
    );
    // The quantization read twice is read directly, by a store and by the add.
    assert_only_copies_read_own_ids(&insts);
    let add = insts.iter().find(|i| i.opcode == Op::FAdd).expect("add");
    let q = ids(add)[0];
    assert!(insts
        .iter()
        .any(|d| d.result_id == Some(q) && d.opcode == Op::QuantizeToF16));
    let max = insts.iter().filter_map(|i| i.result_id).max().unwrap_or(0);
    assert_eq!(words[3], max + 1, "the id bound stays tight");
}

/// The lines of the entry point's body in `msl` (a kernel or a fragment function) that contain
/// `what` without declaring a temporary (`<type> _<id> = ...;`, where SPIRV-Cross declares a
/// value it does not forward).
fn undeclared(msl: &str, what: &str) -> Vec<String> {
    let (_, body) = ["kernel void", "\nfragment "]
        .iter()
        .find_map(|entry| msl.split_once(entry))
        .expect("an entry point");
    body.lines()
        .map(str::trim)
        .filter(|line| line.contains(what))
        .filter(|line| {
            let declared = line.split_once(" = ").is_some_and(|(lhs, _)| {
                let mut words = lhs.split_whitespace();
                match (words.next(), words.next(), words.next()) {
                    (Some(_ty), Some(name), None) => name
                        .strip_prefix('_')
                        .is_some_and(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit())),
                    _ => false,
                }
            });
            !declared
        })
        .map(str::to_owned)
        .collect()
}

#[test]
fn the_scan_finds_values_written_into_a_later_statement() {
    let body = "kernel void main0()\n{\n    float2 _24 = spvQuantizeToF16(_22);\n    \
                _22.x = _22.x + 4.0;\n    _2._m0[1u] = spvQuantizeToF16(_22);\n    \
                int2 _25 = int2(_3.get_width(_22), _3.get_height(_22));\n}\n";
    assert_eq!(
        undeclared(body, "spvQuantizeToF16("),
        ["_2._m0[1u] = spvQuantizeToF16(_22);"]
    );
    assert!(undeclared(body, "get_width(").is_empty());
}

/// The MSL `spirv-cross` writes for `words` with MoltenVK's options, or `None` when it is not
/// installed (the test skips visibly).
fn msl_of(words: &[u32]) -> Option<String> {
    match moltenvk_msl(words) {
        Ok(msl) => Some(msl),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("spirv-cross: {why}"),
    }
}

/// `*p = v` for a function variable `p` of `ty` holding `v`, then `*p` loaded again: a value
/// SPIRV-Cross reads from the variable, so a later store to it invalidates the value.
fn through_variable(f: &mut FunctionBuilder<'_>, ty: TypeId, v: Value) -> (Value, Value) {
    let var = f.local_var(ty, None).expect("var");
    let p = f.var_ptr(var).expect("var pointer");
    f.store(p, v, MemAccess::default()).expect("store");
    let v = f.load(p, MemAccess::default()).expect("load");
    (p, v)
}

/// The shapes MoltenVK 1.4.1's SPIRV-Cross (`adec7acb`) miscompiled in a kernel previously, through
/// the MSL `spirv-cross` writes with MoltenVK's options: a quantized value and a call result read
/// after SPIRV-Cross performs a later `OpCompositeInsert` in place on their operand, and a size and
/// a wide product read after a store to the variable their level or factor was loaded from. Each is
/// now declared where the module evaluates it, before the write. `spirv-cross` is external; without
/// it the test skips visibly.
#[test]
fn moltenvk_msl_evaluates_them_before_a_later_write() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let vectors = ssbo(&mut b, 0, t.vec2, 8);
    let ints = ssbo(&mut b, 1, t.u32_, 4);
    let tex_ty = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let tex = texture(&mut b, 2, tex_ty);
    let neg = negate(&mut b, t.vec2);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let at = elem_ptr(f, vectors, i);
        let x = f.load(at, MemAccess::default()).expect("load");
        let q = f.unary(UnOp::QuantizeToF16, x).expect("quantize");
        // The argument is an expression of `x`, not `x` itself: SPIRV-Cross ties a call whose
        // argument is a load to the loaded variable, and the stores below write that buffer.
        let doubled = f.binary(BinOp::FAdd, x, x).expect("x + x");
        let r = f.call(neg, &[doubled]).expect("call").expect("value");
        // `x.x += 4`, which SPIRV-Cross performs in place on `x`: nothing reads `x` after it.
        let four = cval(f, t.f32_, 0x4080_0000);
        let e = f.composite_extract(x, &[0]).expect("x.x");
        let e = f.binary(BinOp::FAdd, e, four).expect("add");
        let stepped = f.composite_insert(e, x, &[0]).expect("insert");
        for v in [stepped, q, r] {
            let dst = elem_ptr(f, vectors, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        }
        // A level and a factor loaded from variables, the variables stored, then the size's
        // width and the product's high half stored.
        let at = elem_ptr(f, ints, i);
        let level = f.load(at, MemAccess::default()).expect("level");
        let (pl, level) = through_variable(f, t.u32_, level);
        let img = load_global(f, tex);
        let size = f.query(ImageQuery::SizeLod(level), img).expect("size lod");
        let at = elem_ptr(f, ints, i);
        let factor = f.load(at, MemAccess::default()).expect("factor");
        let (pf, factor) = through_variable(f, t.u32_, factor);
        let wide = f
            .binary(BinOp::UMulExtended, factor, factor)
            .expect("wide product");
        let two = cval(f, t.u32_, 2);
        for p in [pl, pf] {
            f.store(p, two, MemAccess::default()).expect("overwrite");
        }
        let w = f.composite_extract(size, &[0]).expect("width");
        let hi = f.composite_extract(wide, &[1]).expect("high half");
        for v in [w, hi] {
            let dst = elem_ptr(f, ints, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        }
        for p in [pl, pf] {
            let after = f.load(p, MemAccess::default()).expect("reload");
            let dst = elem_ptr(f, ints, i);
            f.store(dst, after, MemAccess::default()).expect("store");
        }
    });
    let words = assert_spirv_valid(finish_compute(b, main));
    let Some(msl) = msl_of(&words) else {
        return;
    };
    // The callee is the one function before the kernel: `<type> <name>(...)`.
    let (helpers, _) = msl.split_once("kernel void").expect("a kernel");
    let callee = helpers
        .lines()
        .filter_map(|l| l.strip_prefix("float2 "))
        .find_map(|l| l.split_once('(').map(|(name, _)| name.to_owned()))
        .unwrap_or_else(|| panic!("the callee:\n{msl}"));
    for what in [
        "spvQuantizeToF16(",
        "get_width(",
        "spvMulExtended<",
        &format!("{callee}("),
    ] {
        let lines = undeclared(&msl, what);
        assert!(lines.is_empty(), "{what} written into {lines:#?}\n{msl}");
    }
}

/// A fragment entry over the input `uv` and the output `target`, with storage buffers of `i32`
/// samples (binding 0) and `vec2` offsets (binding 1): `body` gets the input pointer and the
/// two buffers and returns the value written to `target`.
fn interpolating(
    body: impl FnOnce(&mut FunctionBuilder<'_>, &super::families::Tys, Value, [Value; 2]) -> Value,
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let samples = ssbo(&mut b, 0, t.i32_, 4);
    let offsets = ssbo(&mut b, 1, t.vec2, 8);
    let input = b
        .io(key("uv"), IoDir::Input, t.vec4, IoDecor::default())
        .expect("input");
    let output = b
        .io(key("target"), IoDir::Output, t.vec4, IoDecor::default())
        .expect("output");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("main");
    let i = cval(&mut f, t.u32_, 0);
    let p = f.module().global_value(input).expect("uv");
    let s = elem_ptr(&mut f, samples, i);
    let o = elem_ptr(&mut f, offsets, i);
    let v = body(&mut f, &t, p, [s, o]);
    let out = f.module().global_value(output).expect("target");
    f.store(out, v, MemAccess::default()).expect("store");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    b.finalize().expect("finalize")
}

#[test]
fn interpolations_at_a_sample_or_offset_that_is_not_a_constant_are_copied_when_read_once() {
    let m = interpolating(|f, t, p, [sp, op]| {
        let s = f.load(sp, MemAccess::default()).expect("sample");
        let o = f.load(op, MemAccess::default()).expect("offset");
        let k = cval(f, t.i32_, 0);
        let zero = f
            .module()
            .constant(Constant::Scalar {
                ty: t.f32_,
                bits: crate::spirv::ir::ConstBits(0),
            })
            .expect("0");
        let center = f
            .module()
            .constant(Constant::Composite {
                ty: t.vec2,
                parts: [zero, zero].into(),
            })
            .expect("(0, 0)");
        let c = f.module().const_value(center).expect("center");
        let at = |f: &mut FunctionBuilder<'_>, op, x| f.glsl(op, &[p, x]).expect("interpolate");
        // Read once: copied at a loaded sample or offset; not at a constant one.
        let mut sum = at(f, GlslOp::InterpolateAtSample, s);
        for (op, x) in [
            (GlslOp::InterpolateAtOffset, o),
            (GlslOp::InterpolateAtSample, k),
            (GlslOp::InterpolateAtOffset, c),
        ] {
            let v = at(f, op, x);
            sum = f.binary(BinOp::FAdd, sum, v).expect("add");
        }
        // Read twice, by two instructions: SPIRV-Cross declares it where it stands.
        let twice = at(f, GlslOp::InterpolateAtSample, s);
        let sum = f.binary(BinOp::FAdd, sum, twice).expect("add");
        f.binary(BinOp::FAdd, sum, twice).expect("add")
    });
    let insts = parse(&assert_spirv_valid(m));
    assert_eq!(
        copied(&insts),
        [
            Kind::Glsl(GlslStd450Op::InterpolateAtSample as u32),
            Kind::Glsl(GlslStd450Op::InterpolateAtOffset as u32),
        ],
        "exactly the interpolations read once at a sample or offset a write may overwrite"
    );
    assert_only_copies_read_own_ids(&insts);
}

/// The fragment shapes MoltenVK 1.4.1's SPIRV-Cross (`adec7acb`) miscompiled without the forwarding
/// forms: an interpolation read after a store to the variable its sample or offset was loaded from.
/// Each is now declared where the module evaluates it, before the store.
#[test]
fn moltenvk_msl_interpolates_before_a_later_write() {
    let m = interpolating(|f, t, p, [sp, op]| {
        let s = f.load(sp, MemAccess::default()).expect("sample");
        let (vs, s) = through_variable(f, t.i32_, s);
        let o = f.load(op, MemAccess::default()).expect("offset");
        let (vo, o) = through_variable(f, t.vec2, o);
        let a = f
            .glsl(GlslOp::InterpolateAtSample, &[p, s])
            .expect("at sample");
        let b = f
            .glsl(GlslOp::InterpolateAtOffset, &[p, o])
            .expect("at offset");
        let three = cval(f, t.i32_, 3);
        f.store(vs, three, MemAccess::default())
            .expect("overwrite sample");
        let quarter = cval(f, t.f32_, 0x3e80_0000);
        let quarters = f
            .composite_construct(t.vec2, &[quarter, quarter])
            .expect("(0.25, 0.25)");
        f.store(vo, quarters, MemAccess::default())
            .expect("overwrite offset");
        for (v, dst) in [(vs, sp), (vo, op)] {
            let after = f.load(v, MemAccess::default()).expect("reload");
            f.store(dst, after, MemAccess::default()).expect("store");
        }
        f.binary(BinOp::FAdd, a, b).expect("add")
    });
    let words = assert_spirv_valid(m);
    let Some(msl) = msl_of(&words) else {
        return;
    };
    for what in ["interpolate_at_sample(", "interpolate_at_offset("] {
        let lines = undeclared(&msl, what);
        assert!(lines.is_empty(), "{what} written into {lines:#?}\n{msl}");
    }
}

/// Whether each `OpQuantizeToF16`, in order, is followed by its copy.
fn quantizations_copied(insts: &[Instruction]) -> Vec<bool> {
    insts
        .iter()
        .enumerate()
        .filter(|(_, i)| i.opcode == Op::QuantizeToF16)
        .map(|(at, _)| {
            insts
                .get(at + 1)
                .is_some_and(|n| n.opcode == Op::CopyObject)
        })
        .collect()
}

/// The read count is a lower bound of SPIRV-Cross's reads, not the number of times the module
/// names the value: one instruction reads a value once however often it names it, a shuffle reads
/// an operand only when a selected component comes from it, a copy is its operand, and the scalar
/// extracts of one vector together are one read (SPIRV-Cross merges adjacent ones in a
/// constructor into one swizzle), made only if one of them is read (SPIRV-Cross reads the vector
/// where the extract is read).
#[test]
fn reads_count_what_spirv_cross_reads_not_what_the_module_names() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let vectors = ssbo(&mut b, 0, t.vec4, 16);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let at = elem_ptr(f, vectors, i);
        let x = f.load(at, MemAccess::default()).expect("load");
        let store = |f: &mut FunctionBuilder<'_>, v| {
            let dst = elem_ptr(f, vectors, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        };
        let quantize =
            |f: &mut FunctionBuilder<'_>, v| f.unary(UnOp::QuantizeToF16, v).expect("quantize");
        // A swizzle names it twice and reads it once: copied.
        let q = quantize(f, x);
        let s = f.vector_shuffle(q, q, &[2, 1, 0, 3]).expect("swizzle");
        store(f, s);
        // The second operand of a shuffle whose components all come from the first is not read;
        // the store is its one read: copied.
        let q = quantize(f, x);
        let s = f.vector_shuffle(x, q, &[0, 1, 2, 3]).expect("shuffle");
        store(f, s);
        store(f, q);
        // Nor is the first operand of a shuffle whose components all come from the second: copied.
        let q = quantize(f, x);
        let s = f.vector_shuffle(q, x, &[4, 5, 6, 7]).expect("shuffle");
        store(f, s);
        store(f, q);
        // A splat names it four times and reads it once: copied.
        let xx = f.composite_extract(x, &[0]).expect("x.x");
        let q = quantize(f, xx);
        let s = f.composite_construct(t.vec4, &[q, q, q, q]).expect("splat");
        store(f, s);
        // The same splat through a copy, which is its operand: copied.
        let q = quantize(f, xx);
        let c = f.unary(UnOp::CopyObject, q).expect("copy");
        let s = f.composite_construct(t.vec4, &[q, c, q, c]).expect("splat");
        store(f, s);
        // Two scalar extracts of it in one constructor, one swizzle in SPIRV-Cross: copied.
        let q = quantize(f, x);
        let qx = f.composite_extract(q, &[0]).expect("q.x");
        let qy = f.composite_extract(q, &[1]).expect("q.y");
        let zero = cval(f, t.f32_, 0);
        let s = f
            .composite_construct(t.vec4, &[qx, qy, zero, zero])
            .expect("constructor");
        store(f, s);
        // One instruction naming it twice: copied (SPIRV-Cross may read it twice; a copy too many
        // is the safe side).
        let q = quantize(f, x);
        let s = f.binary(BinOp::FAdd, q, q).expect("double");
        store(f, s);
        // A scalar extract of it that nothing reads never reads it; the store is its one read:
        // copied.
        let q = quantize(f, x);
        f.composite_extract(q, &[0]).expect("unread q.x");
        store(f, q);
        // Read by two instructions: not copied.
        let q = quantize(f, x);
        store(f, q);
        let s = f.binary(BinOp::FAdd, q, x).expect("add");
        store(f, s);
        // A scalar extract of it that is read, and a store: not copied.
        let q = quantize(f, x);
        let qx = f.composite_extract(q, &[0]).expect("q.x");
        let s = f
            .composite_construct(t.vec4, &[qx, zero, zero, zero])
            .expect("constructor");
        store(f, s);
        store(f, q);
        // Read by a shuffle that takes components from it, and by a store: not copied.
        let q = quantize(f, x);
        let s = f.vector_shuffle(x, q, &[0, 5, 2, 7]).expect("shuffle");
        store(f, s);
        store(f, q);
    });
    let insts = parse(&assert_spirv_valid(finish_compute(b, main)));
    assert_eq!(
        quantizations_copied(&insts),
        [true, true, true, true, true, true, true, true, false, false, false]
    );
    assert_only_copies_read_own_ids(&insts);
}

/// An image or buffer chosen from a descriptor array by an index that is not a constant is
/// written by SPIRV-Cross with that index, and no image query, sample, fetch, or array length
/// records the index's dependencies: each such result read once is copied, a fetch through
/// `OpImage` of a sampled image included. Through a constant index, a single image, or a single
/// buffer, or read twice, it is not.
#[test]
fn image_and_buffer_reads_through_an_index_that_is_not_a_constant_are_copied_when_read_once() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let ints = ssbo(&mut b, 0, t.u32_, 4);
    let vectors = ssbo(&mut b, 1, t.vec4, 16);
    let ivec2 = super::vec(&mut b, t.i32_, 2);
    let image = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let four = b
        .constant(Constant::Scalar {
            ty: t.u32_,
            bits: crate::spirv::ir::ConstBits(4),
        })
        .expect("4");
    let images = b
        .ty(Type::Array {
            elem: image,
            len: four,
            stride: None,
        })
        .expect("images");
    let textures = b
        .resource(
            super::slot(crate::pipeline::sym::ResourceKind::Texture, 2),
            crate::spirv::ir::StorageClass::UniformConstant,
            images,
        )
        .expect("texture array");
    let single = texture(&mut b, 3, image);
    let sampler = super::families_image::sampler(&mut b, 0);
    let rta = b
        .ty(Type::RuntimeArray {
            elem: t.u32_,
            stride: Some(4),
        })
        .expect("words");
    let layout = crate::spirv::ir::natural_layout(
        b.types(),
        crate::spirv::ir::LayoutRule::Std430,
        true,
        &[rta],
    )
    .expect("layout");
    let block = b
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    let blocks = b
        .ty(Type::Array {
            elem: block,
            len: four,
            stride: None,
        })
        .expect("blocks");
    let buffers = b
        .resource(
            super::slot(crate::pipeline::sym::ResourceKind::Buffer, 4),
            crate::spirv::ir::StorageClass::StorageBuffer,
            blocks,
        )
        .expect("buffer array");
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let at = elem_ptr(f, ints, i);
        let h = f.load(at, MemAccess::default()).expect("index");
        let two = cval(f, t.u32_, 2);
        let element = |f: &mut FunctionBuilder<'_>, index| {
            let base = f.module().global_value(textures).expect("textures");
            let p = f.access_chain(base, &[index]).expect("element");
            f.load(p, MemAccess::default()).expect("image")
        };
        let int = |f: &mut FunctionBuilder<'_>, v| {
            let dst = elem_ptr(f, ints, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        };
        let vector = |f: &mut FunctionBuilder<'_>, v| {
            let dst = elem_ptr(f, vectors, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        };
        let zero = cval(f, t.i32_, 0);
        let origin = f.composite_construct(ivec2, &[zero, zero]).expect("(0, 0)");
        let level = ImageOperands {
            lod: Some(zero),
            ..ImageOperands::default()
        };
        // Through the loaded index, read once: copied.
        let img = element(f, h);
        let texel = f.fetch(img, origin, &level).expect("fetch");
        vector(f, texel);
        let size = f.query(ImageQuery::SizeLod(zero), img).expect("size");
        let w = f.composite_extract(size, &[0]).expect("width");
        int(f, w);
        let levels = f.query(ImageQuery::Levels, img).expect("levels");
        int(f, levels);
        let s = load_global(f, sampler);
        let si = f.sampled_image(img, s).expect("sampled image");
        let half = cval(f, t.f32_, 0x3f00_0000);
        let middle = f
            .composite_construct(t.vec2, &[half, half])
            .expect("middle");
        let fzero = cval(f, t.f32_, 0);
        let explicit = ImageOperands {
            lod: Some(fzero),
            ..ImageOperands::default()
        };
        let sampled = f
            .sample(si, middle, None, false, &explicit)
            .expect("sample");
        vector(f, sampled);
        let through = f.image(si).expect("image of the sampled image");
        let texel = f.fetch(through, origin, &level).expect("fetch");
        vector(f, texel);
        let base = f.module().global_value(buffers).expect("buffers");
        let p = f.access_chain(base, &[h]).expect("buffer");
        let length = f.array_length(p, 0).expect("length");
        int(f, length);
        // Through a constant index, a single image, or a single buffer: not copied.
        let img2 = element(f, two);
        let texel = f.fetch(img2, origin, &level).expect("fetch");
        vector(f, texel);
        let lone = load_global(f, single);
        let texel = f.fetch(lone, origin, &level).expect("fetch");
        vector(f, texel);
        let p = f.module().global_value(ints).expect("ints");
        let length = f.array_length(p, 0).expect("length");
        int(f, length);
        // Through the loaded index, read twice: not copied.
        let texel = f.fetch(img, origin, &level).expect("fetch");
        vector(f, texel);
        vector(f, texel);
    });
    let insts = parse(&assert_spirv_valid(finish_compute(b, main)));
    assert_eq!(
        copied(&insts),
        [
            Kind::Op(Op::ImageFetch),
            Kind::Op(Op::ImageQuerySizeLod),
            Kind::Op(Op::ImageQueryLevels),
            Kind::Op(Op::ImageSampleExplicitLod),
            Kind::Op(Op::ImageFetch),
            Kind::Op(Op::ArrayLength),
        ],
        "exactly the reads through the loaded index, read once"
    );
    assert_only_copies_read_own_ids(&insts);
}

/// The rest of the image arms: a gather, a depth gather, and a storage read through an image
/// chosen by an index that is not a constant, and a sample whose sampler (not its image) is
/// chosen that way. Each, read once, is copied. The same sample through a constant index is not.
#[test]
fn gathers_storage_reads_and_dynamic_samplers_are_copied_when_read_once() {
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let ints = ssbo(&mut b, 0, t.u32_, 4);
    let vectors = ssbo(&mut b, 1, t.vec4, 16);
    let ivec2 = super::vec(&mut b, t.i32_, 2);
    let length = |b: &mut ModuleBuilder, n: u64| {
        b.constant(Constant::Scalar {
            ty: t.u32_,
            bits: crate::spirv::ir::ConstBits(n),
        })
        .expect("length")
    };
    let four = length(&mut b, 4);
    let array_of = |b: &mut ModuleBuilder, elem: TypeId, len| {
        b.ty(Type::Array {
            elem,
            len,
            stride: None,
        })
        .expect("array")
    };
    let resource = |b: &mut ModuleBuilder, kind, index, ty| {
        b.resource(
            super::slot(kind, index),
            crate::spirv::ir::StorageClass::UniformConstant,
            ty,
        )
        .expect("resource")
    };
    use crate::pipeline::sym::ResourceKind::{Sampler, Texture};
    let image = image_ty(&mut b, t.f32_, SAMPLED_2D);
    let depth = image_ty(
        &mut b,
        t.f32_,
        Img {
            depth: true,
            ..SAMPLED_2D
        },
    );
    let storage = image_ty(
        &mut b,
        t.f32_,
        Img {
            usage: ImageUsage::Storage,
            format: ImageFormat::Rgba32f,
            ..SAMPLED_2D
        },
    );
    let ty = array_of(&mut b, image, four);
    let textures = resource(&mut b, Texture, 2, ty);
    let ty = array_of(&mut b, depth, four);
    let depths = resource(&mut b, Texture, 3, ty);
    let ty = array_of(&mut b, storage, four);
    let storages = resource(&mut b, Texture, 4, ty);
    let single = texture(&mut b, 6, image);
    let sampler_ty = b.ty(Type::Sampler).expect("sampler type");
    let ty = array_of(&mut b, sampler_ty, four);
    let samplers = resource(&mut b, Sampler, 1, ty);
    let smp = super::families_image::sampler(&mut b, 0);
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let at = elem_ptr(f, ints, i);
        let h = f.load(at, MemAccess::default()).expect("index");
        let two = cval(f, t.u32_, 2);
        let element = |f: &mut FunctionBuilder<'_>, g, index| {
            let base = f.module().global_value(g).expect("array");
            let p = f.access_chain(base, &[index]).expect("element");
            f.load(p, MemAccess::default()).expect("element value")
        };
        let vector = |f: &mut FunctionBuilder<'_>, v| {
            let dst = elem_ptr(f, vectors, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        };
        let zero = cval(f, t.i32_, 0);
        let origin = f.composite_construct(ivec2, &[zero, zero]).expect("(0, 0)");
        let half = cval(f, t.f32_, 0x3f00_0000);
        let middle = f
            .composite_construct(t.vec2, &[half, half])
            .expect("middle");
        let fzero = cval(f, t.f32_, 0);
        let explicit = ImageOperands {
            lod: Some(fzero),
            ..ImageOperands::default()
        };
        let none = ImageOperands::default();
        let s = load_global(f, smp);
        let lone = load_global(f, single);
        // Through the loaded index, read once: copied.
        let chosen = element(f, samplers, h);
        let si = f.sampled_image(lone, chosen).expect("sampled image");
        let texel = f
            .sample(si, middle, None, false, &explicit)
            .expect("sample");
        vector(f, texel);
        let img = element(f, textures, h);
        let si = f.sampled_image(img, s).expect("sampled image");
        let texel = f.gather(si, middle, 0, &none).expect("gather");
        vector(f, texel);
        let d = element(f, depths, h);
        let si = f.sampled_image(d, s).expect("sampled depth image");
        let texel = f
            .dref_gather(si, middle, fzero, &none)
            .expect("depth gather");
        vector(f, texel);
        let st = element(f, storages, h);
        let texel = f
            .read(st, origin, &none, TexelMemory::PRIVATE)
            .expect("storage read");
        vector(f, texel);
        // Through a constant index: not copied.
        let fixed = element(f, samplers, two);
        let si = f.sampled_image(lone, fixed).expect("sampled image");
        let texel = f
            .sample(si, middle, None, false, &explicit)
            .expect("sample");
        vector(f, texel);
    });
    let insts = parse(&assert_spirv_valid(finish_compute(b, main)));
    assert_eq!(
        copied(&insts),
        [
            Kind::Op(Op::ImageSampleExplicitLod),
            Kind::Op(Op::ImageGather),
            Kind::Op(Op::ImageDrefGather),
            Kind::Op(Op::ImageRead),
        ],
        "exactly the reads through the loaded index, read once"
    );
    assert_only_copies_read_own_ids(&insts);
}

/// SPIRV-Cross writes a switch with no case as a block and never reads its selector, so a call
/// result that selects such a switch and is stored once is read once: copied. Selecting a switch
/// with a case, it is read twice: not copied.
#[test]
fn a_switch_with_no_case_does_not_read_its_selector() {
    use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
    let empty = || -> ArmFn<'static, ()> { Box::new(|_, _| Ok(ArmEnd::FallThrough(Vec::new()))) };
    let mut b = ModuleBuilder::new();
    let t = tys(&mut b);
    let ints = ssbo(&mut b, 0, t.u32_, 4);
    let twice = {
        let mut f = b.function(t.u32_).expect("callee");
        let v = f.param(t.u32_).expect("v");
        let r = f.binary(BinOp::IAdd, v, v).expect("v + v");
        f.ret(Some(r)).expect("ret");
        f.finish().expect("finish")
    };
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let at = elem_ptr(f, ints, i);
        let x = f.load(at, MemAccess::default()).expect("load");
        let store = |f: &mut FunctionBuilder<'_>, v| {
            let dst = elem_ptr(f, ints, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        };
        let n = f.call(twice, &[x]).expect("call").expect("value");
        f.switch_(&mut (), n, Vec::new(), empty()).expect("switch");
        store(f, n);
        let m = f.call(twice, &[x]).expect("call").expect("value");
        f.switch_(&mut (), m, vec![(vec![1], empty())], empty())
            .expect("switch");
        store(f, m);
    });
    let insts = parse(&assert_spirv_valid(finish_compute(b, main)));
    assert_eq!(copied(&insts), [Kind::Op(Op::FunctionCall)]);
    assert_only_copies_read_own_ids(&insts);
}

/// [`crate::spirv::testing::forwarding::quantized_after_a_step`] through the MSL `spirv-cross`
/// writes with MoltenVK's options: every quantization is declared before the step.
#[test]
fn moltenvk_msl_quantizes_before_a_step_read_through_a_swizzle_a_splat_or_extracts() {
    let kernel = crate::spirv::testing::forwarding::quantized_after_a_step().expect("kernel");
    let Some(msl) = msl_of(&assert_spirv_valid(kernel.module)) else {
        return;
    };
    let lines = undeclared(&msl, "spvQuantizeToF16(");
    assert!(lines.is_empty(), "written into {lines:#?}\n{msl}");
}

/// [`crate::spirv::testing::forwarding::images_after_their_index_steps`] through the MSL
/// `spirv-cross` writes with MoltenVK's options: every fetch, size, level count, sample, and
/// buffer length is declared before the index steps.
#[test]
fn moltenvk_msl_reads_images_and_buffers_before_their_index_steps() {
    let kernel =
        crate::spirv::testing::forwarding::images_after_their_index_steps().expect("kernel");
    let Some(msl) = msl_of(&assert_spirv_valid(kernel.module)) else {
        return;
    };
    for what in [
        ".read(",
        "get_width(",
        "get_num_mip_levels(",
        ".sample(",
        "BufferSize[",
    ] {
        let lines = undeclared(&msl, what);
        assert!(lines.is_empty(), "{what} written into {lines:#?}\n{msl}");
    }
}
