//! Where a simdgroup count before a discard is placed, over bodies built with the typed test
//! builder: the measured shapes, and the checks a move must pass.

use super::discard_sink::{expand, placement, Facts, Placement};
use crate::budget::{Budget, Meter};
use crate::graph::{Cfg, DomTree};
use crate::intrinsics::AirIntrinsic;
use crate::ir::build::{FunctionBuilder, ModuleBuilder};
use crate::ir::{
    AddrSpace, Align, BinOp, BlockId, Body, Call, Callee, FastMath, InstId, InstKind, IntPredicate,
    TailKind, TypeId, Value,
};

fn meter() -> Meter {
    Meter::new(&Budget::default())
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

/// The types a body uses.
struct Types {
    void: TypeId,
    i1: TypeId,
    i32t: TypeId,
}

/// The body of a function `fn(i32, ptr addrspace(1))` that `build` writes.
fn body(build: impl FnOnce(&mut FunctionBuilder<'_>, &Types)) -> Body {
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
    let mut parts = m.finish_parts().expect("body");
    parts
        .funcs
        .get_mut(id)
        .and_then(|func| func.body.take())
        .expect("a defined body")
}

/// `simd_sum(x)`.
fn sum(f: &mut FunctionBuilder<'_>, t: &Types, x: Value) -> Value {
    air_call(f, "air.simd_sum.s.i32", t.i32t, &[t.i32t], &[x])
}

/// `discard_fragment()`.
fn discard(f: &mut FunctionBuilder<'_>, t: &Types) {
    air_call(f, "air.discard_fragment", t.void, &[], &[]);
}

/// `a == b`, as an `i1`.
fn eq(f: &mut FunctionBuilder<'_>, t: &Types, a: Value, b: Value) -> Value {
    f.push(
        t.i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a,
            b,
        },
    )
}

/// `if (c) discard_fragment();` from the current block, then `join` current.
fn guarded(f: &mut FunctionBuilder<'_>, t: &Types, c: Value) -> BlockId {
    let s = f.block();
    let join = f.block();
    f.cond_br(c, s, join);
    f.switch_to(s);
    discard(f, t);
    f.br(join);
    f.switch_to(join);
    join
}

/// The block and id of the body's first call of `name`.
fn find(body: &Body, name: &str) -> (BlockId, InstId) {
    let op = AirIntrinsic::parse(name).expect("an intrinsic");
    body.blocks
        .iter()
        .find_map(|(b, _)| {
            body.block_insts(b).find_map(|(id, i)| match &i.kind {
                InstKind::Call(c) if matches!(c.callee, Callee::Air { op: o, .. } if o == op) => {
                    Some((b, id))
                }
                _ => None,
            })
        })
        .unwrap_or_else(|| panic!("no call of {name}"))
}

/// The placement of the body's first `simd_sum`.
fn placed(body: &Body) -> Placement {
    let mut m = meter();
    let facts = Facts::of(body, &mut m).unwrap();
    let (b, op) = find(body, "air.simd_sum.s.i32");
    let placed = placement(&facts, body, op, b);
    facts.release(&mut m);
    placed
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
                    assert!(
                        dom.dominates(db, at),
                        "{d:?} does not dominate its use {id:?}"
                    );
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

/// The measured shape: a count, then `if (c) discard_fragment();` whose condition reads
/// something else, then a use. Metal counts only the lanes that survive the discard, and the
/// count moves to the start of the block where the discard's paths rejoin, ahead of its use.
#[test]
fn a_count_ahead_of_a_guarded_discard_moves_past_it() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        let s = sum(f, t, one);
        let c = eq(f, t, f.arg(0), zero);
        guarded(f, t, c);
        f.bin(BinOp::Add, s, one);
        f.ret(None);
    });
    let (home, _) = find(&b, "air.simd_sum.s.i32");
    let Placement::Moved(to) = placed(&b) else {
        panic!("{:?}", placed(&b));
    };
    assert_ne!(to, home);
    let mut m = meter();
    let moved = expand(b, &mut m).unwrap();
    let (at, op) = find(&moved, "air.simd_sum.s.i32");
    let (discarded, _) = find(&moved, "air.discard_fragment");
    assert_ne!(at, home);
    assert!(moved.successors(discarded).any(|s| s == at));
    // The count leads its block, and its use follows it there.
    assert_eq!(moved.block_insts(at).next().map(|(id, _)| id), Some(op));
    assert_defined_at_uses(&moved);
    assert_eq!(placed(&moved), Placement::NoDiscard);
}

/// The discard's condition reads the count: Metal counts the dropped quads (the shape measured
/// `counted_dependent`), and the count stays.
#[test]
fn a_count_the_condition_reads_is_counted() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let s = sum(f, t, one);
        let c = eq(f, t, s, f.arg(0));
        guarded(f, t, c);
        f.ret(None);
    });
    assert_eq!(placed(&b), Placement::Counted);
    let (home, op) = find(&b, "air.simd_sum.s.i32");
    let kept = expand(b, &mut meter()).unwrap();
    assert_eq!(find(&kept, "air.simd_sum.s.i32"), (home, op));
}

/// A store between the count and the discard (measured `counted_store`, which stores the count,
/// and `counted_atomic`, an atomic that reads nothing of it): Metal counts the dropped quads.
#[test]
fn a_side_effect_between_keeps_the_count() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        sum(f, t, one);
        f.push_void(InstKind::Store {
            val: one,
            ptr: f.arg(1),
            align: Align::from_bytes(4),
            volatile: false,
            atomic: None,
        });
        let c = eq(f, t, f.arg(0), zero);
        guarded(f, t, c);
        f.ret(None);
    });
    assert_eq!(placed(&b), Placement::Counted);
}

/// The guarded discard nested in a further branch: Metal counted the dropped quads in one shape
/// measured and moved the sum in another (`hoisted_nested` of the owned fixture), so neither is
/// known.
#[test]
fn a_nested_discard_is_unknown() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        sum(f, t, one);
        let outer = eq(f, t, f.arg(0), one);
        let inner_block = f.block();
        let join = f.block();
        f.cond_br(outer, inner_block, join);
        f.switch_to(inner_block);
        let c = eq(f, t, f.arg(0), zero);
        guarded(f, t, c);
        f.br(join);
        f.switch_to(join);
        f.ret(None);
    });
    assert_eq!(placed(&b), Placement::Unknown);
}

/// Pure arithmetic reads the count ahead of the discard (the shape `counted_call_use` takes once
/// its call is inlined): no measurement says where Metal places it.
#[test]
fn a_use_between_leaves_the_count_unknown() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        let s = sum(f, t, one);
        f.bin(BinOp::Mul, s, one);
        let c = eq(f, t, f.arg(0), zero);
        guarded(f, t, c);
        f.ret(None);
    });
    assert_eq!(placed(&b), Placement::Unknown);
}

/// The entry block of a body [`body`] builds.
const ENTRY: BlockId = BlockId::new(0);

/// A loop wholly between the count and the discard (measured `hoisted_loop`) is passed through:
/// each lane runs the count and reaches the join once.
#[test]
fn a_loop_between_is_passed_through() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        sum(f, t, one);
        let header = f.block();
        let latch = f.block();
        let exit = f.block();
        f.br(header);
        f.switch_to(header);
        let i = f.push(
            t.i32t,
            InstKind::Phi {
                incoming: vec![(zero, ENTRY), (one, latch)].into(),
                fmf: FastMath::NONE,
            },
        );
        let done = eq(f, t, i, f.arg(0));
        f.cond_br(done, exit, latch);
        f.switch_to(latch);
        f.br(header);
        f.switch_to(exit);
        let c = eq(f, t, f.arg(0), zero);
        guarded(f, t, c);
        f.ret(None);
    });
    assert!(
        matches!(placed(&b), Placement::Moved(_)),
        "{:?}",
        placed(&b)
    );
    let moved = expand(b, &mut meter()).unwrap();
    assert_defined_at_uses(&moved);
    assert_eq!(placed(&moved), Placement::NoDiscard);
}

/// A count inside a loop, ahead of a discard in the same iteration, moves within the iteration:
/// to the join ahead of the back edge, which runs once each time the count did.
#[test]
fn a_count_in_a_loop_moves_within_its_iteration() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        let header = f.block();
        let s = f.block();
        let join = f.block();
        let exit = f.block();
        f.br(header);
        f.switch_to(header);
        let phi = f.push(
            t.i32t,
            InstKind::Phi {
                incoming: vec![(zero, ENTRY), (one, join)].into(),
                fmf: FastMath::NONE,
            },
        );
        sum(f, t, one);
        let c = eq(f, t, phi, zero);
        f.cond_br(c, s, join);
        f.switch_to(s);
        discard(f, t);
        f.br(join);
        f.switch_to(join);
        let again = eq(f, t, f.arg(0), one);
        f.cond_br(again, header, exit);
        f.switch_to(exit);
        f.ret(None);
    });
    let (home, _) = find(&b, "air.simd_sum.s.i32");
    match placed(&b) {
        Placement::Moved(to) => assert_ne!(to, home),
        other => panic!("{other:?}"),
    }
    let moved = expand(b, &mut meter()).unwrap();
    assert_defined_at_uses(&moved);
}

/// A count ahead of a loop whose body holds the discard: the count's block and the discard's
/// join do not run equally often, so nothing moves.
#[test]
fn a_discard_in_a_later_loop_is_unknown() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        sum(f, t, one);
        let header = f.block();
        let exit = f.block();
        f.br(header);
        f.switch_to(header);
        let c = eq(f, t, f.arg(0), zero);
        let join = guarded(f, t, c);
        let again = eq(f, t, f.arg(0), one);
        f.cond_br(again, header, exit);
        f.switch_to(exit);
        f.ret(None);
        let _ = join;
    });
    assert_eq!(placed(&b), Placement::Unknown);
}

/// The other reductions are unmeasured: a `simd_max` ahead of the discard stays, as `Unknown`.
#[test]
fn the_other_reductions_are_unmeasured() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        air_call(f, "air.simd_max.s.i32", t.i32t, &[t.i32t], &[one]);
        let c = eq(f, t, f.arg(0), zero);
        guarded(f, t, c);
        f.ret(None);
    });
    let mut m = meter();
    let facts = Facts::of(&b, &mut m).unwrap();
    let (home, op) = find(&b, "air.simd_max.s.i32");
    assert_eq!(placement(&facts, &b, op, home), Placement::Unknown);
    facts.release(&mut m);
    let kept = expand(b, &mut m).unwrap();
    assert_eq!(find(&kept, "air.simd_max.s.i32"), (home, op));
}

/// An unconditional discard after the count is no measured shape.
#[test]
fn an_unconditional_discard_is_unknown() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        sum(f, t, one);
        discard(f, t);
        f.ret(None);
    });
    assert_eq!(placed(&b), Placement::Unknown);
}

/// The count moves past a later discard only: one with no discard after it is `NoDiscard`.
#[test]
fn a_count_with_no_discard_after_it_stays() {
    let b = body(|f, t| {
        let one = f.module().const_int(t.i32t, 1);
        let zero = f.module().const_int(t.i32t, 0);
        let c = eq(f, t, f.arg(0), zero);
        guarded(f, t, c);
        sum(f, t, one);
        f.ret(None);
    });
    assert_eq!(placed(&b), Placement::NoDiscard);
}
