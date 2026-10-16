//! The branches Metal's backend speculates, over bodies built with the typed test builder: the
//! shapes measured on the M3 Max (macOS 27.0.1), and what `expand` makes of them.

use super::speculate::{decisions, expand, Decision};
use crate::budget::{Budget, Meter};
use crate::graph::{Cfg, DomTree};
use crate::intrinsics::AirIntrinsic;
use crate::ir::build::{FunctionBuilder, ModuleBuilder};
use crate::ir::{
    AddrSpace, BinOp, BlockId, Body, Call, Callee, ConstTable, FastMath, InstId, InstKind,
    IntPredicate, Param, TailKind, Type, TypeId, TypeTable, Value,
};

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

/// The types a body uses.
struct Types {
    void: TypeId,
    i1: TypeId,
    i32t: TypeId,
}

/// A body of `fn(i32 t, ptr addrspace(1) out)` that `build` writes, with its type table and
/// parameter types.
struct Built {
    body: Body,
    types: TypeTable,
    consts: ConstTable,
    param_tys: Vec<TypeId>,
}

impl Built {
    fn decisions(&self) -> Vec<Decision> {
        let mut m = meter();
        let params = vec![Param::default(); self.param_tys.len()];
        decisions(
            &self.body,
            (&self.types, &self.consts),
            (&self.param_tys, &params),
            &mut m,
        )
        .unwrap()
        .into_values()
        .collect()
    }

    fn expanded(self) -> Body {
        self.expanded_noting().0
    }

    /// The expanded body and whether `expand` reports a branch of a group operation no
    /// measurement decides.
    fn expanded_noting(self) -> (Body, bool) {
        let mut m = meter();
        let params = vec![Param::default(); self.param_tys.len()];
        expand(
            self.body,
            (&self.types, &self.consts),
            (&self.param_tys, &params),
            &mut m,
        )
        .unwrap()
    }
}

fn built(build: impl FnOnce(&mut FunctionBuilder<'_>, &Types)) -> Built {
    let mut m = ModuleBuilder::new();
    let types = Types {
        void: m.void(),
        i1: m.int(1),
        i32t: m.int(32),
    };
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(types.void, &[types.i32t, dev]);
    let mut f = m.define("f", fty);
    build(&mut f, &types);
    let id = f.finish();
    let parts = m.finish_parts().expect("body");
    let body = parts
        .funcs
        .get(id)
        .and_then(|func| func.body.clone())
        .expect("a defined body");
    let fty = parts.funcs.get(id).map(|f| f.ty).expect("the function");
    let Some(Type::Function { params, .. }) = parts.types.get(fty) else {
        panic!("a function type");
    };
    let param_tys = params.to_vec();
    Built {
        body,
        types: parts.types,
        consts: parts.consts,
        param_tys,
    }
}

/// A call of the `air.*` intrinsic `name` of type `ret(params)`.
fn air_call(
    f: &mut FunctionBuilder<'_>,
    name: &str,
    ret: TypeId,
    params: &[TypeId],
    args: &[Value],
) -> Value {
    let ty = f.module().fn_ty(ret, params);
    let decl = f.module().declare(name, ty);
    let op = AirIntrinsic::parse(name).unwrap_or_else(|| panic!("{name}"));
    f.push(
        ret,
        InstKind::Call(Box::new(Call {
            callee: Callee::Air { decl, op },
            args: args.into(),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        })),
    )
}

fn first(f: &mut FunctionBuilder<'_>, t: &Types, x: Value) -> Value {
    air_call(f, "air.quad_broadcast_first.u.i32", t.i32t, &[t.i32t], &[x])
}

fn quad_sum(f: &mut FunctionBuilder<'_>, t: &Types, x: Value) -> Value {
    air_call(f, "air.quad_sum.u.i32", t.i32t, &[t.i32t], &[x])
}

/// `t % 3 == 0`.
fn skips(f: &mut FunctionBuilder<'_>, t: &Types) -> Value {
    let three = f.module().const_int(t.i32t, 3);
    let zero = f.module().const_int(t.i32t, 0);
    let r = f.bin(BinOp::URem, f.arg(0), three);
    f.push(
        t.i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: r,
            b: zero,
        },
    )
}

/// `10 t + 1` then `n` more instructions (alternately `^ t` and `+ t`): `n + 2` in all.
fn chain(f: &mut FunctionBuilder<'_>, t: &Types, n: usize) -> Value {
    let ten = f.module().const_int(t.i32t, 10);
    let one = f.module().const_int(t.i32t, 1);
    let x = f.bin(BinOp::Mul, f.arg(0), ten);
    let mut x = f.bin(BinOp::Or, x, one);
    for k in 0..n {
        let op = match k % 2 {
            0 => BinOp::Xor,
            _ => BinOp::Add,
        };
        x = f.bin(op, x, f.arg(0));
    }
    x
}

fn phi(f: &mut FunctionBuilder<'_>, t: &Types, incoming: &[(Value, BlockId)]) -> Value {
    f.push(
        t.i32t,
        InstKind::Phi {
            incoming: incoming.into(),
            fmf: FastMath::NONE,
        },
    )
}

/// `out[t] = v`.
fn store_out(f: &mut FunctionBuilder<'_>, t: &Types, v: Value) {
    let p = gep_out(f, t);
    f.push_void(InstKind::Store {
        val: v,
        ptr: p,
        align: None,
        volatile: false,
        atomic: None,
    });
}

fn gep_out(f: &mut FunctionBuilder<'_>, t: &Types) -> Value {
    let dev = f.module().ptr(AddrSpace::Device);
    f.push(
        dev,
        InstKind::Gep {
            src_ty: t.i32t,
            base: f.arg(1),
            indices: Box::new([f.arg(0)]),
            inbounds: true,
        },
    )
}

/// `r = 7777; if (t % 3 != 0) r = arm(); out[t] = r;`, the arm a block of its own.
fn triangle(arm: impl FnOnce(&mut FunctionBuilder<'_>, &Types) -> Value) -> Built {
    built(|f, t| {
        let entry = BlockId::new(0);
        let a = f.block();
        let join = f.block();
        let c = skips(f, t);
        f.cond_br(c, join, a);
        f.switch_to(a);
        let v = arm(f, t);
        f.br(join);
        f.switch_to(join);
        let k = f.module().const_int(t.i32t, 7777);
        let r = phi(f, t, &[(v, a), (k, entry)]);
        store_out(f, t, r);
        f.ret(None);
    })
}

/// `if (t % 3 != 0) r = arm_false(); else r = arm_true(); out[t] = r;`
fn diamond(
    on_true: impl FnOnce(&mut FunctionBuilder<'_>, &Types) -> Value,
    on_false: impl FnOnce(&mut FunctionBuilder<'_>, &Types) -> Value,
) -> Built {
    built(|f, t| {
        let a = f.block();
        let b = f.block();
        let join = f.block();
        let c = skips(f, t);
        f.cond_br(c, a, b);
        f.switch_to(a);
        let va = on_true(f, t);
        f.br(join);
        f.switch_to(b);
        let vb = on_false(f, t);
        f.br(join);
        f.switch_to(join);
        let r = phi(f, t, &[(va, a), (vb, b)]);
        store_out(f, t, r);
        f.ret(None);
    })
}

/// The kinds of `body`'s terminators that branch on a condition.
fn conditional_branches(body: &Body) -> usize {
    body.blocks
        .iter()
        .filter(|&(b, _)| {
            matches!(
                body.terminator(b).map(|i| &i.kind),
                Some(InstKind::CondBr { .. })
            )
        })
        .count()
}

/// The number of `body`'s PHIs and selects.
fn phis_and_selects(body: &Body) -> (usize, usize) {
    let mut out = (0, 0);
    for (_, i) in body.insts.iter() {
        match i.kind {
            InstKind::Phi { .. } => out.0 += 1,
            InstKind::Select { .. } => out.1 += 1,
            _ => {}
        }
    }
    out
}

/// Every operand is defined where its use runs: in a block that dominates the use's (a PHI's
/// entry: its incoming block), earlier in the same block.
fn assert_defined_at_uses(body: &Body) {
    let mut m = meter();
    let cfg = Cfg::build(body, &mut m).unwrap();
    let dom = DomTree::compute(&cfg, &mut m).unwrap();
    let block_of = |id: InstId| {
        body.blocks
            .iter()
            .find(|(b, _)| body.block_insts(*b).any(|(i, _)| i == id))
            .map(|(b, _)| b)
            .expect("an instruction in a block")
    };
    for (b, _) in body.blocks.iter() {
        for (id, inst) in body.block_insts(b) {
            let check = |v: Value, at: BlockId, same_block: bool| {
                if let Value::Inst(d) = v {
                    let db = block_of(d);
                    assert!(dom.dominates(db, at), "{d:?} does not dominate {id:?}");
                    if same_block && db == at {
                        assert!(d.raw() < id.raw(), "{d:?} follows its use {id:?}");
                    }
                }
            };
            match &inst.kind {
                InstKind::Phi { incoming, .. } => {
                    for &(v, from) in incoming {
                        check(v, from, false);
                    }
                }
                kind => kind.for_each_value(|v| check(v, b, true)),
            }
        }
    }
}

/// The authored kernel's shape: `quad_broadcast_first` of `10 t + 1` in an arm that joins
/// through a PHI. Metal runs the arm in every lane of the branch's block: the branch becomes a
/// jump into the arm, and the PHI a select on the condition, the arm's value where it is false.
#[test]
fn a_quad_broadcast_first_in_a_short_arm_runs_in_every_lane_of_the_branch() {
    let b = triangle(|f, t| {
        let x = chain(f, t, 0);
        first(f, t, x)
    });
    assert_eq!(b.decisions(), [Decision::Speculated]);
    let body = b.expanded();
    assert_eq!(conditional_branches(&body), 0);
    assert_eq!(phis_and_selects(&body), (0, 1));
    let select = body
        .insts
        .iter()
        .find_map(|(_, i)| match i.kind {
            InstKind::Select { t, f, .. } => Some((t, f)),
            _ => None,
        })
        .expect("a select");
    assert!(matches!(select.0, Value::Const(_)), "{select:?}");
    assert!(matches!(select.1, Value::Inst(_)), "{select:?}");
    assert_defined_at_uses(&body);
}

/// A quad reduction keeps its arm (and the arm's other operations keep its lanes too): the body
/// is left as it is.
#[test]
fn a_quad_reduction_keeps_its_arm() {
    let b = triangle(|f, t| {
        let x = chain(f, t, 0);
        let y = first(f, t, x);
        let s = quad_sum(f, t, x);
        f.bin(BinOp::Add, y, s)
    });
    assert_eq!(b.decisions(), [Decision::Kept]);
    let body = b.expanded();
    assert_eq!(conditional_branches(&body), 1);
    assert_eq!(phis_and_selects(&body), (1, 0));
}

/// A device read keeps its arm.
#[test]
fn a_device_read_keeps_its_arm() {
    let b = triangle(|f, t| {
        let p = gep_out(f, t);
        let x = f.push(
            t.i32t,
            InstKind::Load {
                ptr: p,
                align: None,
                volatile: false,
                atomic: None,
                range: None,
            },
        );
        first(f, t, x)
    });
    assert_eq!(b.decisions(), [Decision::Kept]);
    assert_eq!(conditional_branches(&b.expanded()), 1);
}

/// One arm: up to 25 instructions speculated, 26 not.
#[test]
fn one_arm_speculates_up_to_twenty_five_instructions() {
    for (n, decision) in [(25, Decision::Speculated), (26, Decision::Kept)] {
        // `chain` makes `k + 2`, and the call one more.
        let b = triangle(|f, t| {
            let x = chain(f, t, n - 3);
            first(f, t, x)
        });
        assert_eq!(b.decisions(), [decision], "{n}");
        let branches = b.expanded();
        assert_eq!(
            conditional_branches(&branches),
            usize::from(decision == Decision::Kept),
            "{n}"
        );
    }
}

/// An arm holding a kind the budget was not measured with (here a quad vote and a select of a
/// constant) is speculated up to 21 instructions, as the M3 Max speculates such an arm of 21 (a
/// divergent arm of quad votes, 16-bit masks and casts); from 22 to 25 it is unknown and left (the
/// M3 Max keeps such an arm of 25, `a3_quad_votes_divergent`'s), and past 25 it is kept. An arm of
/// the budget's own kinds is speculated up to 25.
#[test]
fn an_arm_of_other_kinds_is_known_speculated_up_to_twenty_one() {
    // `chain` makes `k + 2`; the comparison, the vote and the select three more.
    let voting = |n: usize| {
        triangle(move |f, t| {
            let x = chain(f, t, n - 5);
            let nine = f.module().const_int(t.i32t, 9);
            let c = f.push(
                t.i1,
                InstKind::ICmp {
                    pred: IntPredicate::Ne,
                    a: x,
                    b: nine,
                },
            );
            let v = air_call(f, "air.quad_all", t.i1, &[t.i1], &[c]);
            let seven = f.module().const_int(t.i32t, 7);
            f.push(
                t.i32t,
                InstKind::Select {
                    c: v,
                    t: x,
                    f: seven,
                    fmf: FastMath::NONE,
                },
            )
        })
    };
    for (n, decision) in [
        (21, Decision::Speculated),
        (22, Decision::Unknown),
        (25, Decision::Unknown),
        (26, Decision::Kept),
    ] {
        let b = voting(n);
        assert_eq!(b.decisions(), [decision], "{n}");
        assert_eq!(
            conditional_branches(&b.expanded()),
            usize::from(decision != Decision::Speculated),
            "{n}"
        );
    }
}

/// Two arms: up to 25 together; beside an arm on the condition's true edge of at most 3
/// instructions and no call, up to 25 in the other; an arm of at most 3 with a call on the false
/// edge is not exempt; the two shapes no probe covers are unknown and left.
#[test]
fn two_arms_count_together_but_a_small_true_arm() {
    let plain = |n: usize| {
        move |f: &mut FunctionBuilder<'_>, t: &Types| {
            let seven = f.module().const_int(t.i32t, 7000);
            let mut y = f.bin(BinOp::Add, f.arg(0), seven);
            for k in 1..n {
                let op = match k % 2 {
                    0 => BinOp::Add,
                    _ => BinOp::Xor,
                };
                y = f.bin(op, y, f.arg(0));
            }
            y
        }
    };
    let op = |n: usize| {
        move |f: &mut FunctionBuilder<'_>, t: &Types| {
            let x = chain(f, t, n - 3);
            first(f, t, x)
        }
    };
    let only = |n: usize| {
        move |f: &mut FunctionBuilder<'_>, t: &Types| {
            let _ = n;
            first(f, t, f.arg(0))
        }
    };
    // (the true edge's arm, the false edge's arm): measured.
    assert_eq!(
        diamond(plain(10), op(13)).decisions(),
        [Decision::Speculated]
    );
    assert_eq!(
        diamond(plain(16), op(9)).decisions(),
        [Decision::Speculated]
    );
    assert_eq!(diamond(plain(13), op(13)).decisions(), [Decision::Kept]);
    assert_eq!(diamond(plain(4), op(22)).decisions(), [Decision::Kept]);
    assert_eq!(
        diamond(plain(3), op(25)).decisions(),
        [Decision::Speculated]
    );
    assert_eq!(
        diamond(plain(1), op(25)).decisions(),
        [Decision::Speculated]
    );
    assert_eq!(diamond(plain(3), op(26)).decisions(), [Decision::Kept]);
    assert_eq!(diamond(plain(25), only(1)).decisions(), [Decision::Kept]);
    assert_eq!(
        diamond(plain(24), only(1)).decisions(),
        [Decision::Speculated]
    );
    // Unmeasured: a small arm without a call on the false edge, past 25 together.
    let unknown = diamond(op(25), plain(3));
    assert_eq!(unknown.decisions(), [Decision::Unknown]);
    assert_eq!(conditional_branches(&unknown.expanded()), 1);
    // Speculated: the arms run one after the other, the true edge's first.
    let body = diamond(plain(3), op(25)).expanded();
    assert_eq!(conditional_branches(&body), 0);
    assert_eq!(phis_and_selects(&body), (0, 1));
    assert_defined_at_uses(&body);
}

/// `out[t] = 7777; if (t % 3 != 0) out[t] = quad_broadcast_first(10 t + 1);`: the arm's store
/// repeats the branch's block's last store to its pointer. Metal speculates it: the arm's store
/// becomes the chosen value, which the branch's block's store, moved to where the arms join,
/// stores.
#[test]
fn a_store_repeating_the_branchs_store_stores_the_chosen_value() {
    let b = built(|f, t| {
        let a = f.block();
        let join = f.block();
        let p = gep_out(f, t);
        let k = f.module().const_int(t.i32t, 7777);
        f.push_void(InstKind::Store {
            val: k,
            ptr: p,
            align: None,
            volatile: false,
            atomic: None,
        });
        let c = skips(f, t);
        f.cond_br(c, join, a);
        f.switch_to(a);
        let x = chain(f, t, 0);
        let v = first(f, t, x);
        f.push_void(InstKind::Store {
            val: v,
            ptr: p,
            align: None,
            volatile: false,
            atomic: None,
        });
        f.br(join);
        f.switch_to(join);
        f.ret(None);
    });
    assert_eq!(b.decisions(), [Decision::Speculated]);
    let body = b.expanded();
    assert_eq!(conditional_branches(&body), 0);
    let stores: Vec<(BlockId, Value)> = body
        .blocks
        .iter()
        .flat_map(|(blk, _)| {
            body.block_insts(blk)
                .filter_map(move |(_, i)| match i.kind {
                    InstKind::Store { val, .. } => Some((blk, val)),
                    _ => None,
                })
        })
        .collect();
    assert_eq!(stores.len(), 1, "{stores:?}");
    let Value::Inst(chosen) = stores[0].1 else {
        panic!("{stores:?}");
    };
    let chosen = body.insts.get(chosen).expect("the chosen value");
    let InstKind::Select {
        f: Value::Inst(arm),
        ..
    } = chosen.kind
    else {
        panic!("{chosen:?}");
    };
    assert_eq!(body.insts.get(arm).map(|i| i.ty), Some(chosen.ty));
    assert_defined_at_uses(&body);
}

/// A store to another pointer keeps its arm.
#[test]
fn a_store_to_another_pointer_keeps_its_arm() {
    let b = built(|f, t| {
        let a = f.block();
        let join = f.block();
        let c = skips(f, t);
        f.cond_br(c, join, a);
        f.switch_to(a);
        let x = chain(f, t, 0);
        let v = first(f, t, x);
        store_out(f, t, v);
        f.br(join);
        f.switch_to(join);
        f.ret(None);
    });
    assert_eq!(b.decisions(), [Decision::Kept]);
}

/// A short branch inside an arm folds first and counts with the arm it joins: both become jumps.
#[test]
fn a_short_branch_inside_an_arm_folds_with_it() {
    let b = built(|f, t| {
        let entry = BlockId::new(0);
        let outer = f.block();
        let inner = f.block();
        let rejoin = f.block();
        let join = f.block();
        let c = skips(f, t);
        f.cond_br(c, join, outer);
        f.switch_to(outer);
        let three = f.module().const_int(t.i32t, 3);
        let zero = f.module().const_int(t.i32t, 0);
        let q = f.bin(BinOp::And, f.arg(0), three);
        let d = f.push(
            t.i1,
            InstKind::ICmp {
                pred: IntPredicate::Eq,
                a: q,
                b: zero,
            },
        );
        f.cond_br(d, rejoin, inner);
        f.switch_to(inner);
        let w = f.bin(BinOp::Mul, q, q);
        f.br(rejoin);
        f.switch_to(rejoin);
        let s = phi(f, t, &[(zero, outer), (w, inner)]);
        let x = chain(f, t, 0);
        let x = f.bin(BinOp::Add, x, s);
        let v = first(f, t, x);
        f.br(join);
        f.switch_to(join);
        let k = f.module().const_int(t.i32t, 7777);
        let r = phi(f, t, &[(v, rejoin), (k, entry)]);
        store_out(f, t, r);
        f.ret(None);
    });
    let mut d = b.decisions();
    d.sort_by_key(|d| *d as u8);
    assert_eq!(d, [Decision::Speculated, Decision::Speculated]);
    let body = b.expanded();
    assert_eq!(conditional_branches(&body), 0);
    assert_eq!(phis_and_selects(&body), (0, 2));
    assert_defined_at_uses(&body);
}

/// An arm without a group operation that answers differently is left as it is, speculated or
/// not: only the operations Metal's speculation changes move.
#[test]
fn an_arm_without_such_an_operation_is_left() {
    let b = triangle(|f, t| chain(f, t, 3));
    assert_eq!(b.decisions(), [Decision::Speculated]);
    assert_eq!(conditional_branches(&b.expanded()), 1);
}

/// An address computation in the arm: no measurement covers its count, so the arm is left.
#[test]
fn an_unmeasured_instruction_leaves_the_arm() {
    let b = triangle(|f, t| {
        gep_out(f, t);
        let x = chain(f, t, 0);
        first(f, t, x)
    });
    assert_eq!(b.decisions(), [Decision::Unknown]);
    assert_eq!(conditional_branches(&b.expanded()), 1);
}

/// An integer division by a variable in the arm: speculated, it would divide by zero where the
/// condition guarded it, which SPIR-V leaves undefined, so the arm is left.
#[test]
fn a_division_by_a_variable_leaves_the_arm() {
    let b = triangle(|f, t| {
        let x = chain(f, t, 0);
        let q = f.bin(BinOp::UDiv, x, f.arg(0));
        first(f, t, q)
    });
    assert_eq!(b.decisions(), [Decision::Unknown]);
    assert_eq!(conditional_branches(&b.expanded()), 1);
}

/// `expand` reports a branch no measurement decides exactly where its arms hold a group operation
/// that answers over the branch's block's lanes when speculated (the entry's
/// `unmeasured_speculation`, its pipeline's `PrecisionNote::UnmeasuredGroupSpeculation`): an
/// unmeasured instruction, an integer division by a variable, or an unmeasured diamond beside
/// `quad_broadcast_first`; not that operation in an arm Metal was measured to speculate or keep,
/// nor an unmeasured arm without it.
#[test]
fn only_an_unmeasured_branch_of_a_group_operation_is_reported() {
    let unmeasured_first = triangle(|f, t| {
        gep_out(f, t);
        let x = chain(f, t, 0);
        first(f, t, x)
    });
    let divided_first = triangle(|f, t| {
        let x = chain(f, t, 0);
        let q = f.bin(BinOp::UDiv, x, f.arg(0));
        first(f, t, q)
    });
    // A small arm without a call on the false edge, past 25 together: no probe covers it.
    let diamond_first = diamond(
        |f, t| {
            let x = chain(f, t, 22);
            first(f, t, x)
        },
        |f, t| chain(f, t, 1),
    );
    for (what, b) in [
        ("an unmeasured instruction", unmeasured_first),
        ("a division by a variable", divided_first),
        ("an unmeasured diamond", diamond_first),
    ] {
        assert_eq!(b.decisions(), [Decision::Unknown], "{what}");
        let (body, reported) = b.expanded_noting();
        assert!(reported, "{what}: not reported");
        assert_eq!(conditional_branches(&body), 1, "{what}: the branch kept");
    }
    let speculated = triangle(|f, t| {
        let x = chain(f, t, 0);
        first(f, t, x)
    });
    let kept = triangle(|f, t| {
        let x = chain(f, t, 0);
        quad_sum(f, t, x)
    });
    let unmeasured_plain = triangle(|f, t| {
        gep_out(f, t);
        chain(f, t, 0)
    });
    for (what, b, decision) in [
        ("a measured arm", speculated, Decision::Speculated),
        ("a kept arm", kept, Decision::Kept),
        ("no group operation", unmeasured_plain, Decision::Unknown),
    ] {
        assert_eq!(b.decisions(), [decision], "{what}");
        assert!(!b.expanded_noting().1, "{what}: reported");
    }
}
