//! Tangled values that SPIRV-Cross could evaluate lazily are materialized: a subgroup operation or
//! derivative that reaches `OpSelect`'s `t`/`f` or `OpLogicalAnd`/`OpLogicalOr`'s second operand is
//! written under its own id and followed by the `OpCopyObject` its consumers read; one used only
//! eagerly is written as before.

use super::{assert_spirv_valid, cval, elem_ptr, finish_compute, gid_x, int, key, ssbo, vec};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ir::{
    BinOp, DerivOp, ExecModel, IoDecor, IoDir, MemAccess, ShuffleOp, Type, UnOp,
};
use crate::spirv_binary::{Instruction, Operand};
use crate::spv::Op;

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

/// Whether the instruction defining `id` is one of `ops`.
fn defined_by(insts: &[Instruction], id: u32, ops: &[Op]) -> bool {
    insts
        .iter()
        .any(|i| i.result_id == Some(id) && ops.contains(&i.opcode))
}

/// Each `OpCopyObject` directly follows the instruction that defines its operand, and that
/// instruction is one of `tangled`. Returns the copies' result ids.
fn copies_follow_their_tangled_operand(insts: &[Instruction], tangled: &[Op]) -> Vec<u32> {
    let mut copies = Vec::new();
    for (at, i) in insts.iter().enumerate() {
        if i.opcode != Op::CopyObject {
            continue;
        }
        let operand = ids(i)[0];
        let before = &insts[at - 1];
        assert_eq!(
            before.result_id,
            Some(operand),
            "the copy follows its operand"
        );
        assert!(
            tangled.contains(&before.opcode),
            "a copy of {:?}",
            before.opcode
        );
        copies.push(i.result_id.expect("copy result"));
    }
    copies
}

#[test]
fn subgroup_values_reaching_lazy_operands_are_copied() {
    let mut b = ModuleBuilder::new();
    let u32_ = int(&mut b, 32, false);
    let input = ssbo(&mut b, 0, u32_, 4);
    let output = ssbo(&mut b, 1, u32_, 4);
    let main = super::compute(&mut b, |f| {
        let i = gid_x(f);
        let x = elem_ptr(f, input, i);
        let x = f.load(x, MemAccess::default()).expect("load");
        let six = cval(f, u32_, 6);
        let k = cval(f, u32_, 0xABCD_ABCD);
        let zero = cval(f, u32_, 0);
        let cond = f.binary(BinOp::ULessThan, i, six).expect("cond");
        let pred = f.binary(BinOp::INotEqual, x, zero).expect("pred");
        let store = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, v| {
            let dst = elem_ptr(f, output, i);
            f.store(dst, v, MemAccess::default()).expect("store");
        };
        // Lazy: a select arm, a select arm through an add, `&&`'s second operand, and a select
        // arm through a copy (P9's promoted load, written as its operand).
        let s = f
            .group_shuffle(ShuffleOp::Shuffle, x, six)
            .expect("shuffle");
        let r = f.select(cond, s, k).expect("select");
        store(f, r);
        let s = f.group_shuffle(ShuffleOp::Xor, x, six).expect("xor");
        let s = f.binary(BinOp::IAdd, s, six).expect("add");
        let r = f.select(cond, k, s).expect("select");
        store(f, r);
        let s = f.group_vote(true, pred).expect("all");
        let l = f.binary(BinOp::LogicalAnd, cond, s).expect("and");
        let r = f.select(l, six, zero).expect("select");
        store(f, r);
        let s = f
            .group_shuffle(ShuffleOp::Shuffle, x, six)
            .expect("shuffle");
        let s = f.unary(UnOp::CopyObject, s).expect("copy");
        let r = f.select(cond, s, zero).expect("select");
        store(f, r);
        // Eager: an add that is stored, a select's condition, `||`'s first operand.
        let s = f.group_broadcast_first(x).expect("broadcast first");
        let s = f.binary(BinOp::IAdd, s, x).expect("add");
        store(f, s);
        let s = f.group_vote(false, pred).expect("any");
        let r = f.select(s, x, k).expect("select");
        store(f, r);
        let s = f.group_vote(true, pred).expect("all");
        let l = f.binary(BinOp::LogicalOr, s, cond).expect("or");
        let r = f.select(l, six, zero).expect("select");
        store(f, r);
    });
    let words = assert_spirv_valid(finish_compute(b, main));
    let insts = parse(&words);
    let copies = copies_follow_their_tangled_operand(
        &insts,
        &[
            Op::GroupNonUniformShuffle,
            Op::GroupNonUniformShuffleXor,
            Op::GroupNonUniformAll,
        ],
    );
    assert_eq!(copies.len(), 4, "the four lazily used subgroup values");
    // The lazy consumers read the copies; nothing else reads a copied instruction's own id.
    let group = [
        Op::GroupNonUniformShuffle,
        Op::GroupNonUniformShuffleXor,
        Op::GroupNonUniformAll,
        Op::GroupNonUniformAny,
        Op::GroupNonUniformBroadcastFirst,
    ];
    for i in &insts {
        match i.opcode {
            Op::Select => {
                let [_, t, f] = ids(i)[..] else {
                    panic!("select operands")
                };
                for arm in [t, f] {
                    assert!(
                        !defined_by(&insts, arm, &group),
                        "a subgroup value in an arm"
                    );
                }
            }
            Op::LogicalAnd | Op::LogicalOr => {
                let second = *ids(i).last().expect("operand");
                assert!(
                    !defined_by(&insts, second, &group),
                    "a subgroup value in `b`"
                );
            }
            Op::CopyObject => {}
            _ if !group.contains(&i.opcode) => {
                for id in ids(i) {
                    let own = insts
                        .iter()
                        .position(|d| d.result_id == Some(id))
                        .is_some_and(|d| {
                            insts
                                .get(d + 1)
                                .is_some_and(|n| n.opcode == Op::CopyObject && ids(n)[0] == id)
                        });
                    assert!(
                        !own,
                        "{:?} reads a materialized instruction's own id",
                        i.opcode
                    );
                }
            }
            _ => {}
        }
    }
    // The eager ones are read directly: the stored add, the select's condition, `||`'s `a`.
    let reads_group = |op: Op, at: usize| {
        insts.iter().filter(|i| i.opcode == op).any(|i| {
            ids(i)
                .get(at)
                .is_some_and(|id| defined_by(&insts, *id, &group))
        })
    };
    assert!(
        reads_group(Op::IAdd, 0),
        "the stored add reads the broadcast"
    );
    assert!(
        reads_group(Op::Select, 0),
        "a select's condition reads the vote"
    );
    assert!(
        reads_group(Op::LogicalOr, 0),
        "`||` reads the vote as its first operand"
    );
    let max = insts.iter().filter_map(|i| i.result_id).max().unwrap_or(0);
    assert_eq!(words[3], max + 1, "the id bound stays tight");
}

#[test]
fn derivatives_in_select_arms_are_copied() {
    let mut b = ModuleBuilder::new();
    let void = b.ty(Type::Void).expect("void");
    let f32_ = b.ty(Type::Float { width: 32 }).expect("float");
    let vec4 = vec(&mut b, f32_, 4);
    let input = b
        .io(key("uv"), IoDir::Input, vec4, IoDecor::default())
        .expect("input");
    let out = b
        .io(key("color"), IoDir::Output, f32_, IoDecor::default())
        .expect("output");
    let mut f = b.function(void).expect("main");
    let p = f.module().global_value(input).expect("input value");
    let uv = f.load(p, MemAccess::default()).expect("load");
    let x = f.composite_extract(uv, &[0]).expect("x");
    let zero = cval(&mut f, f32_, 0);
    let cond = f.binary(BinOp::FOrdLessThan, x, zero).expect("cond");
    let o = f.module().global_value(out).expect("output value");
    // Lazy: `cond ? dfdx(x) : 0`. Eager: `dfdy(x)` stored.
    let d = f.derivative(DerivOp::DPdx, x).expect("dpdx");
    let r = f.select(cond, d, zero).expect("select");
    f.store(o, r, MemAccess::default()).expect("store");
    let d = f.derivative(DerivOp::DPdy, x).expect("dpdy");
    f.store(o, d, MemAccess::default()).expect("store");
    f.ret(None).expect("ret");
    let main = f.finish().expect("finish");
    b.entry(ExecModel::Fragment, main, vec![]).expect("entry");
    let words = assert_spirv_valid(b.finalize().expect("finalize"));
    let insts = parse(&words);
    let copies = copies_follow_their_tangled_operand(&insts, &[Op::DPdx]);
    assert_eq!(copies.len(), 1, "only the lazily used derivative");
    let select = insts
        .iter()
        .find(|i| i.opcode == Op::Select)
        .expect("select");
    assert_eq!(ids(select)[1], copies[0], "the arm reads the copy");
}
