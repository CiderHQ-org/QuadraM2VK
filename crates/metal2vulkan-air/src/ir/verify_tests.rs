//! Verifier tests: well-formed IR built through the test builder is accepted, and each
//! construction invariant rejects a hand-built violation with its typed error.

use super::build::{FunctionBuilder, ModuleBuilder};
use super::*;
use crate::budget::{Budget, Meter};
use crate::error::Error;
use crate::error::{InputErrorKind, VerifyError};

fn verify_kind(r: Result<Module, Error>) -> VerifyError {
    match r {
        Err(Error::InvalidInput(e)) => match e.kind {
            InputErrorKind::Verify(v) => v,
            other => panic!("unexpected {other:?}"),
        },
        other => panic!("expected a verify error, got {other:?}"),
    }
}

#[derive(Clone, Copy)]
struct Tys {
    void: TypeId,
    i1: TypeId,
    i32: TypeId,
    f32: TypeId,
    dev: TypeId,
}

fn tys(m: &mut ModuleBuilder) -> Tys {
    Tys {
        void: m.void(),
        i1: m.int(1),
        i32: m.int(32),
        f32: m.float(),
        dev: m.ptr(AddrSpace::Device),
    }
}

/// A module with one kernel `void k(ptr addrspace(1) %a0, i32 %a1)` whose body `body` builds.
fn kernel(body: impl FnOnce(&mut FunctionBuilder<'_>, Tys)) -> Result<Module, Error> {
    let mut m = ModuleBuilder::new();
    let t = tys(&mut m);
    let fty = m.fn_ty(t.void, &[t.dev, t.i32]);
    let mut f = m.define("k", fty);
    body(&mut f, t);
    f.finish();
    m.finish()
}

fn load(f: &mut FunctionBuilder<'_>, ty: TypeId, ptr: Value) -> Value {
    f.push(
        ty,
        InstKind::Load {
            ptr,
            align: Align::from_bytes(4),
            volatile: false,
            atomic: None,
            range: None,
        },
    )
}

fn store(f: &mut FunctionBuilder<'_>, val: Value, ptr: Value) {
    f.push_void(InstKind::Store {
        val,
        ptr,
        align: None,
        volatile: false,
        atomic: None,
    });
}

fn icmp(f: &mut FunctionBuilder<'_>, t: Tys, a: Value, b: Value) -> Value {
    f.push(
        t.i1,
        InstKind::ICmp {
            pred: IntPredicate::Ult,
            a,
            b,
        },
    )
}

#[test]
fn a_well_formed_loop_kernel_verifies() {
    let mut m = ModuleBuilder::new();
    let t = tys(&mut m);
    let helper_ty = m.fn_ty(t.i32, &[t.i32]);
    let helper = m.declare("helper", helper_ty);
    let arr = m.array(t.f32, 4);
    let s = m.struct_ty(&[t.i32, arr]);
    let pair = m.struct_ty(&[t.i32, t.i1]);
    let fty = m.fn_ty(t.void, &[t.dev, t.i32]);
    let zero = m.const_int(t.i32, 0);
    let one = m.const_int(t.i32, 1);
    let mut f = m.define("k", fty);
    let (entry, header, body, exit) = (BlockId::new(0), f.block(), f.block(), f.block());
    f.br(header); // i0

    // Layout-order ids: header = [i1 phi, i2 icmp, i3 br], body starts at i4.
    f.switch_to(header);
    let next = Value::Inst(InstId::new(4));
    let i = f.push(
        t.i32,
        InstKind::Phi {
            incoming: Box::new([(zero, entry), (next, body)]),
            fmf: FastMath::NONE,
        },
    );
    let c = icmp(&mut f, t, i, arg(1));
    f.cond_br(c, body, exit);

    f.switch_to(body);
    assert_eq!(f.bin(BinOp::Add, i, one), next);
    let field = f.push(
        t.dev,
        InstKind::Gep {
            src_ty: s,
            base: f.arg(0),
            indices: Box::new([zero, one, i]),
            inbounds: true,
        },
    );
    let x = load(&mut f, t.f32, field);
    let y = f.push(
        t.f32,
        InstKind::BinOp {
            op: BinOp::FMul,
            flags: IntFlags::NONE,
            fmf: FastMath::CONTRACT,
            a: x,
            b: x,
        },
    );
    store(&mut f, y, field);
    let r = f.push(
        pair,
        InstKind::CmpXchg {
            ptr: f.arg(0),
            cmp: zero,
            new: one,
            success: Atomic {
                ordering: Ordering::SeqCst,
                scope: SyncScope::System,
            },
            failure: Ordering::Monotonic,
            weak: false,
            volatile: false,
            align: None,
        },
    );
    f.push(
        t.i32,
        InstKind::ExtractValue {
            agg: r,
            indices: Box::new([0]),
        },
    );
    f.push(
        t.i32,
        InstKind::Call(Box::new(Call {
            callee: Callee::Direct(helper),
            args: Box::new([next]),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        })),
    );
    f.br(header);
    f.switch_to(exit);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let body = module.func(k).unwrap().body.as_ref().unwrap();
    assert_eq!(body.blocks.len(), 4);
    assert_eq!(body.insts.len(), 14);
    assert_eq!(
        body.successors(header).collect::<Vec<_>>(),
        [body_id(2), exit]
    );
}

fn body_id(n: u32) -> BlockId {
    BlockId::new(n)
}

#[test]
fn builder_lays_blocks_out_contiguously_and_renumbers() {
    let mut m = ModuleBuilder::new();
    let t = tys(&mut m);
    let fty = m.fn_ty(t.i32, &[t.i32]);
    let one = m.const_int(t.i32, 1);
    let mut f = m.define("k", fty);
    let (entry, next) = (BlockId::new(0), f.block());
    f.switch_to(next);
    let placeholder_ret = f.push(t.void, InstKind::Unreachable);
    let _ = placeholder_ret;
    f.switch_to(entry);
    let a = f.bin(BinOp::Add, f.arg(0), one);
    let b = f.bin(BinOp::Mul, a, a);
    f.br(next);
    assert_eq!(
        (a, b),
        (Value::Inst(InstId::new(1)), Value::Inst(InstId::new(2)))
    );
    let id = f.id();
    f.finish();
    let module = m.finish().unwrap();
    let body = module.func(id).unwrap().body.as_ref().unwrap();
    let blocks: Vec<(u32, u32)> = body
        .blocks
        .values()
        .map(|b| (b.start.raw(), b.end.raw()))
        .collect();
    assert_eq!(blocks, [(0, 3), (3, 4)]);
    // The entry's multiply now uses the add at its final position 0.
    match &body.insts.get(InstId::new(1)).unwrap().kind {
        InstKind::BinOp { a, b, .. } => {
            assert_eq!(
                (*a, *b),
                (Value::Inst(InstId::new(0)), Value::Inst(InstId::new(0)))
            );
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(module.value_ty(id, Value::Arg(ArgIdx::new(0))), Some(t.i32));
    assert_eq!(
        module.value_ty(id, Value::Inst(InstId::new(0))),
        Some(t.i32)
    );
}

/// A loop kernel whose phi refers to a value in a later block, built in layout order.
fn loop_kernel(
    phi: impl FnOnce(Value, BlockId, Value, BlockId) -> Box<[(Value, BlockId)]>,
) -> Result<Module, Error> {
    kernel(|f, t| {
        let zero = f.module().const_int(t.i32, 0);
        let one = f.module().const_int(t.i32, 1);
        let (entry, header, body, exit) = (BlockId::new(0), f.block(), f.block(), f.block());
        f.br(header); // 0
        f.switch_to(header);
        // Ids in layout order: header = [1 phi, 2 icmp, 3 condbr], body = [4 add, 5 br].
        let next = Value::Inst(InstId::new(4));
        let i = f.push(
            t.i32,
            InstKind::Phi {
                incoming: phi(zero, entry, next, body),
                fmf: FastMath::NONE,
            },
        );
        let c = icmp(f, t, i, arg(1));
        f.cond_br(c, body, exit);
        f.switch_to(body);
        let n = f.bin(BinOp::Add, i, one);
        assert_eq!(n, next);
        f.br(header);
        f.switch_to(exit);
        f.ret(None);
    })
}

#[test]
fn loops_with_phis_verify_and_phi_edges_must_match_predecessors() {
    let module =
        loop_kernel(|zero, entry, next, body| Box::new([(zero, entry), (next, body)])).unwrap();
    assert_eq!(module.parts().funcs.len(), 1);
    let missing = loop_kernel(|zero, entry, _, _| Box::new([(zero, entry)]));
    assert_eq!(verify_kind(missing), VerifyError::PhiPredecessors);
    let extra = loop_kernel(|zero, entry, next, body| {
        Box::new([(zero, entry), (next, body), (zero, BlockId::new(3))])
    });
    assert_eq!(verify_kind(extra), VerifyError::PhiPredecessors);
    let dangling =
        loop_kernel(|zero, entry, next, _| Box::new([(zero, entry), (next, BlockId::new(40))]));
    assert_eq!(verify_kind(dangling), VerifyError::BadBlockRef);
    let wrong_type = loop_kernel(|_, entry, next, body| {
        Box::new([(Value::Arg(ArgIdx::new(0)), entry), (next, body)])
    });
    assert_eq!(verify_kind(wrong_type), VerifyError::OperandType);
}

#[test]
fn duplicate_edges_need_duplicate_consistent_phi_entries() {
    let build = |second: bool| {
        kernel(|f, t| {
            let zero = f.module().const_int(t.i32, 0);
            let one = f.module().const_int(t.i32, 1);
            let join = f.block();
            f.push_void(InstKind::Switch {
                v: f.arg(1),
                default: join,
                cases: Box::new([(1, join)]),
                hints: None,
            });
            f.switch_to(join);
            let v = if second { one } else { zero };
            f.push(
                t.i32,
                InstKind::Phi {
                    incoming: Box::new([(zero, BlockId::new(0)), (v, BlockId::new(0))]),
                    fmf: FastMath::NONE,
                },
            );
            f.ret(None);
        })
    };
    assert!(build(false).is_ok());
    assert_eq!(verify_kind(build(true)), VerifyError::PhiPredecessors);
}

#[test]
fn block_structure_is_enforced() {
    let no_term = kernel(|f, t| {
        let one = f.module().const_int(t.i32, 1);
        f.bin(BinOp::Add, one, one);
    });
    assert_eq!(verify_kind(no_term), VerifyError::MissingTerminator);

    let early_term = kernel(|f, _| {
        f.ret(None);
        f.ret(None);
    });
    assert_eq!(verify_kind(early_term), VerifyError::TerminatorNotLast);

    let empty_block = kernel(|f, _| {
        f.block();
        f.ret(None);
    });
    assert_eq!(verify_kind(empty_block), VerifyError::MissingTerminator);

    let late_phi = kernel(|f, t| {
        let one = f.module().const_int(t.i32, 1);
        f.bin(BinOp::Add, one, one);
        f.push(
            t.i32,
            InstKind::Phi {
                incoming: Box::new([]),
                fmf: FastMath::NONE,
            },
        );
        f.ret(None);
    });
    assert_eq!(verify_kind(late_phi), VerifyError::PhiPredecessors);

    let entry_pred = kernel(|f, _| f.br(BlockId::new(0)));
    assert_eq!(
        verify_kind(entry_pred),
        VerifyError::EntryBlockHasPredecessors
    );

    let bad_target = kernel(|f, _| f.br(BlockId::new(7)));
    assert_eq!(verify_kind(bad_target), VerifyError::BadBlockRef);
}

#[test]
fn operand_references_are_checked() {
    let bad_arg = kernel(|f, t| {
        let one = f.module().const_int(t.i32, 1);
        f.bin(BinOp::Add, Value::Arg(ArgIdx::new(5)), one);
        f.ret(None);
    });
    assert_eq!(verify_kind(bad_arg), VerifyError::BadValueRef);

    let bad_const = kernel(|f, t| {
        f.bin(BinOp::Add, f.arg(1), Value::Const(ConstId::new(999)));
        let _ = t;
        f.ret(None);
    });
    assert_eq!(verify_kind(bad_const), VerifyError::BadValueRef);

    let void_use = kernel(|f, _| {
        f.push_void(InstKind::Fence(Atomic {
            ordering: Ordering::SeqCst,
            scope: SyncScope::System,
        }));
        f.push_void(InstKind::Ret(Some(Value::Inst(InstId::new(0)))));
    });
    assert_eq!(verify_kind(void_use), VerifyError::BadValueRef);

    let use_before_def = kernel(|f, t| {
        let later = Value::Inst(InstId::new(1));
        f.push(t.i32, add(later, later));
        f.bin(BinOp::Add, f.arg(1), f.arg(1));
        f.ret(None);
    });
    assert_eq!(verify_kind(use_before_def), VerifyError::UseNotDominated);

    let self_use = kernel(|f, t| {
        let me = Value::Inst(InstId::new(0));
        f.push(t.i32, add(me, me));
        f.ret(None);
    });
    assert_eq!(verify_kind(self_use), VerifyError::UseNotDominated);
}

#[test]
fn instruction_typing_is_enforced() {
    type Case = (&'static str, fn(&mut FunctionBuilder<'_>, Tys), VerifyError);
    let cases: [Case; 16] = [
        (
            "int add of float",
            |f, t| {
                let x = f.module().constant(t.f32, Const::Float(0));
                f.bin(BinOp::Add, Value::Const(x), Value::Const(x));
            },
            VerifyError::OperandType,
        ),
        (
            "mixed operand types",
            |f, t| {
                let x = f.module().constant(t.f32, Const::Float(0));
                f.bin(BinOp::FAdd, Value::Const(x), f.arg(1));
            },
            VerifyError::OperandType,
        ),
        (
            "nsw on udiv",
            |f, _| {
                f.push(
                    f.value_ty(f.arg(1)).unwrap(),
                    InstKind::BinOp {
                        op: BinOp::UDiv,
                        flags: IntFlags::NSW,
                        fmf: FastMath::NONE,
                        a: f.arg(1),
                        b: f.arg(1),
                    },
                );
            },
            VerifyError::OperandType,
        ),
        (
            "icmp to i32",
            |f, t| {
                f.push(
                    t.i32,
                    InstKind::ICmp {
                        pred: IntPredicate::Eq,
                        a: f.arg(1),
                        b: f.arg(1),
                    },
                );
            },
            VerifyError::ResultType,
        ),
        (
            "widening trunc",
            |f, _| {
                let i64t = f.module().int(64);
                f.push(
                    i64t,
                    InstKind::Cast {
                        op: CastOp::Trunc,
                        v: f.arg(1),
                    },
                );
            },
            VerifyError::OperandType,
        ),
        (
            "bitcast of different size",
            |f, _| {
                let h = f.module().ty(Type::Half);
                f.push(
                    h,
                    InstKind::Cast {
                        op: CastOp::Bitcast,
                        v: f.arg(1),
                    },
                );
            },
            VerifyError::OperandType,
        ),
        (
            "same-space addrspacecast",
            |f, t| {
                f.push(
                    t.dev,
                    InstKind::Cast {
                        op: CastOp::AddrSpaceCast,
                        v: f.arg(0),
                    },
                );
            },
            VerifyError::OperandType,
        ),
        (
            "store through an integer",
            |f, _| store(f, arg(1), arg(1)),
            VerifyError::OperandType,
        ),
        (
            "release load",
            |f, t| {
                f.push(
                    t.i32,
                    InstKind::Load {
                        ptr: f.arg(0),
                        align: None,
                        volatile: false,
                        atomic: Some(Atomic {
                            ordering: Ordering::Release,
                            scope: SyncScope::System,
                        }),
                        range: None,
                    },
                );
            },
            VerifyError::OperandType,
        ),
        (
            "gep with a variable struct index",
            |f, t| {
                let zero = f.module().const_int(t.i32, 0);
                let s = f.module().struct_ty(&[t.i32, t.f32]);
                f.push(
                    t.dev,
                    InstKind::Gep {
                        src_ty: s,
                        base: f.arg(0),
                        indices: Box::new([zero, f.arg(1)]),
                        inbounds: false,
                    },
                );
            },
            VerifyError::OperandType,
        ),
        (
            "gep changing address space",
            |f, t| {
                let tg = f.module().ptr(AddrSpace::Threadgroup);
                let zero = f.module().const_int(t.i32, 0);
                f.push(
                    tg,
                    InstKind::Gep {
                        src_ty: t.i32,
                        base: f.arg(0),
                        indices: Box::new([zero]),
                        inbounds: false,
                    },
                );
            },
            VerifyError::OperandType,
        ),
        (
            "cmpxchg with a scalar result",
            |f, t| {
                f.push(
                    t.i32,
                    InstKind::CmpXchg {
                        ptr: f.arg(0),
                        cmp: f.arg(1),
                        new: f.arg(1),
                        success: Atomic {
                            ordering: Ordering::SeqCst,
                            scope: SyncScope::System,
                        },
                        failure: Ordering::SeqCst,
                        weak: false,
                        volatile: false,
                        align: None,
                    },
                );
            },
            VerifyError::ResultType,
        ),
        (
            "branch on i32",
            |f, _| {
                let b = f.block();
                f.cond_br(f.arg(1), b, b);
                f.switch_to(b);
            },
            VerifyError::OperandType,
        ),
        (
            "duplicate switch case",
            |f, _| {
                let b = f.block();
                f.push_void(InstKind::Switch {
                    v: f.arg(1),
                    default: b,
                    cases: Box::new([(3, b), (3, b)]),
                    hints: None,
                });
                f.switch_to(b);
            },
            VerifyError::OperandType,
        ),
        (
            "extractvalue out of range",
            |f, t| {
                let pair = f.module().struct_ty(&[t.i32, t.i32]);
                let u = f.module().constant(pair, Const::Undef);
                f.push(
                    t.i32,
                    InstKind::ExtractValue {
                        agg: Value::Const(u),
                        indices: Box::new([2]),
                    },
                );
            },
            VerifyError::OperandType,
        ),
        (
            "shuffle mask out of range",
            |f, t| {
                let v2 = f.module().vector(t.i32, 2);
                let u = f.module().constant(v2, Const::Undef);
                f.push(
                    v2,
                    InstKind::ShuffleVector {
                        a: Value::Const(u),
                        b: Value::Const(u),
                        mask: Box::new([Some(0), Some(4)]),
                    },
                );
            },
            VerifyError::OperandType,
        ),
    ];
    for (name, build, want) in cases {
        let r = kernel(|f, t| {
            build(f, t);
            f.ret(None);
        });
        assert_eq!(verify_kind(r), want, "{name}");
    }
}

#[test]
fn returns_and_calls_match_signatures() {
    let wrong_ret = kernel(|f, _| f.ret(Some(f.arg(1))));
    assert_eq!(verify_kind(wrong_ret), VerifyError::OperandType);

    let call = |args: &[fn(&FunctionBuilder<'_>) -> Value], ret_i32: bool, with_body: bool| {
        let mut m = ModuleBuilder::new();
        let t = tys(&mut m);
        let callee_ty = m.fn_ty(t.i32, &[t.i32]);
        let callee = if with_body {
            let mut g = m.define("callee", callee_ty);
            g.ret(Some(g.arg(0)));
            g.finish()
        } else {
            m.declare("air.unknown", callee_ty)
        };
        let fty = m.fn_ty(t.void, &[t.dev, t.i32]);
        let mut f = m.define("k", fty);
        let args: Box<[Value]> = args.iter().map(|a| a(&f)).collect();
        f.push(
            if ret_i32 { t.i32 } else { t.f32 },
            InstKind::Call(Box::new(Call {
                callee: Callee::UnknownIntrinsic(callee),
                args,
                fmf: FastMath::NONE,
                tail: TailKind::None,
                attrs: None,
                range: None,
            })),
        );
        f.ret(None);
        f.finish();
        m.finish()
    };
    assert!(call(&[|f| f.arg(1)], true, false).is_ok());
    assert_eq!(
        verify_kind(call(&[|f| f.arg(0)], true, false)),
        VerifyError::OperandType
    );
    assert_eq!(
        verify_kind(call(&[], true, false)),
        VerifyError::OperandType
    );
    assert_eq!(
        verify_kind(call(&[|f| f.arg(1)], false, false)),
        VerifyError::ResultType
    );
    assert_eq!(
        verify_kind(call(&[|f| f.arg(1)], true, true)),
        VerifyError::OperandType,
        "an intrinsic callee must be a declaration"
    );
}

#[test]
fn constants_globals_functions_and_metadata_are_checked() {
    let mut m = ModuleBuilder::new();
    let i8t = m.int(8);
    m.constant(i8t, Const::Int(300));
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedConstant);

    let mut m = ModuleBuilder::new();
    let i8t = m.int(8);
    let a = m.array(i8t, 3);
    let one = m.constant(i8t, Const::Int(1));
    m.constant(a, Const::Aggregate(Box::new([one, one])));
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedConstant);

    let mut m = ModuleBuilder::new();
    let i8t = m.int(8);
    let g = m.simple_global("g", i8t, AddrSpace::Threadgroup, None);
    let wrong_space = m.ptr(AddrSpace::Device);
    m.constant(wrong_space, Const::GlobalAddr(g));
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedConstant);

    let mut m = ModuleBuilder::new();
    let i8t = m.int(8);
    let i32t = m.int(32);
    let init = m.constant(i32t, Const::Int(0));
    m.simple_global("g", i8t, AddrSpace::Constant, Some(init));
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedGlobal);

    let mut m = ModuleBuilder::new();
    let i8t = m.int(8);
    let g = m.simple_global("g", i8t, AddrSpace::Constant, None);
    let init = m.constant(i8t, Const::Int(0));
    let ok_addr = m.global_addr(g);
    let _ = ok_addr;
    let mut parts = m.finish_parts().unwrap();
    parts.globals.get_mut(g).unwrap().linkage = Linkage::Internal;
    assert_eq!(
        verify_kind(Module::new(parts.clone(), &mut meter())),
        VerifyError::MalformedGlobal,
        "an internal declaration"
    );
    parts.globals.get_mut(g).unwrap().init = Some(init);
    assert!(Module::new(parts, &mut meter()).is_ok());

    let mut m = ModuleBuilder::new();
    let void = m.void();
    let fty = m.fn_ty(void, &[]);
    let f = m.declare("f", fty);
    m.func_mut(f).unwrap().linkage = Linkage::Internal;
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedFunction);

    let mut m = ModuleBuilder::new();
    let void = m.void();
    m.declare("f", void);
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedFunction);

    let mut m = ModuleBuilder::new();
    let void = m.void();
    let fty = m.fn_ty(void, &[]);
    let f = m.declare("f", fty);
    m.func_mut(f).unwrap().params = vec![Param::default()].into();
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedFunction);

    let mut m = ModuleBuilder::new();
    let s = m.md_string("air.kernel");
    let node = m.md_node(&[Some(s), None]);
    m.named_md("air.kernel", &[node]);
    assert!(m.finish().is_ok());

    let mut m = ModuleBuilder::new();
    m.md_node(&[Some(MdId::new(5))]);
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedMetadata);

    let mut m = ModuleBuilder::new();
    m.named_md("x", &[MdId::new(0)]);
    assert_eq!(verify_kind(m.finish()), VerifyError::MalformedMetadata);
}

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

#[test]
fn errors_locate_function_block_and_instruction() {
    let r = kernel(|f, t| {
        let b = f.block();
        f.br(b);
        f.switch_to(b);
        let x = f.module().constant(t.f32, Const::Float(0));
        f.bin(BinOp::Add, Value::Const(x), Value::Const(x));
        f.ret(None);
    });
    let Err(Error::InvalidInput(e)) = r else {
        panic!("{r:?}");
    };
    assert_eq!(
        e.to_string(),
        "Verify(OperandType) at function k, block 1, instruction 1"
    );
}

#[test]
fn verification_is_metered() {
    let mut m = ModuleBuilder::new();
    let t = tys(&mut m);
    let fty = m.fn_ty(t.void, &[t.dev, t.i32]);
    let mut f = m.define("k", fty);
    for _ in 0..100 {
        f.bin(BinOp::Add, f.arg(1), f.arg(1));
    }
    f.ret(None);
    f.finish();
    let parts = m.finish_parts().unwrap();
    let mut tight = Meter::with_caps(&Budget::default(), 50, u64::MAX);
    assert_eq!(
        Module::new(parts.clone(), &mut tight),
        Err(Error::LimitExceeded(crate::error::Limit::WorkUnits))
    );
    let mut m = meter();
    assert!(Module::new(parts, &mut m).is_ok());
    assert!(m.work() >= 101);
}

fn arg(i: u32) -> Value {
    Value::Arg(ArgIdx::new(i))
}

fn add(a: Value, b: Value) -> InstKind {
    InstKind::BinOp {
        op: BinOp::Add,
        flags: IntFlags::NONE,
        fmf: FastMath::NONE,
        a,
        b,
    }
}

fn phi_of(f: &mut FunctionBuilder<'_>, ty: TypeId, incoming: &[(Value, BlockId)]) -> Value {
    f.push(
        ty,
        InstKind::Phi {
            incoming: incoming.into(),
            fmf: FastMath::NONE,
        },
    )
}

/// `entry: c = icmp a1, a1; x = add a1, a1; br c, then, else`, `then: y = add x, x; br join`,
/// `else: e = add x, 1; br join`, `join: <join_body(x, y, e)>; ret`.
fn diamond(
    join_body: impl FnOnce(&mut FunctionBuilder<'_>, Tys, [Value; 3], [BlockId; 2]),
) -> Result<Module, Error> {
    kernel(|f, t| {
        let a1 = f.arg(1);
        let one = f.module().const_int(t.i32, 1);
        let (then, els, join) = (f.block(), f.block(), f.block());
        let c = icmp(f, t, a1, a1);
        let x = f.bin(BinOp::Add, a1, a1);
        f.cond_br(c, then, els);
        f.switch_to(then);
        let y = f.bin(BinOp::Add, x, x);
        f.br(join);
        f.switch_to(els);
        let e = f.bin(BinOp::Add, x, one);
        f.br(join);
        f.switch_to(join);
        join_body(f, t, [x, y, e], [then, els]);
        f.ret(None);
    })
}

#[test]
fn cross_block_uses_must_be_dominated() {
    // A value of the dominating entry block is usable in the join; one of an arm is not.
    assert!(diamond(|f, _, [x, _, _], _| {
        f.bin(BinOp::Add, x, x);
    })
    .is_ok());
    let arm_value = diamond(|f, _, [x, y, _], _| {
        f.bin(BinOp::Add, x, y);
    });
    assert_eq!(verify_kind(arm_value), VerifyError::UseNotDominated);

    // Terminator operands are ordinary uses.
    let arm_condition = kernel(|f, t| {
        let a1 = f.arg(1);
        let (then, exit, done) = (f.block(), f.block(), f.block());
        let c = icmp(f, t, a1, a1);
        f.cond_br(c, then, exit);
        f.switch_to(then);
        let d = icmp(f, t, a1, a1);
        f.br(exit);
        f.switch_to(exit);
        f.cond_br(d, done, done);
        f.switch_to(done);
        f.ret(None);
    });
    assert_eq!(verify_kind(arm_condition), VerifyError::UseNotDominated);

    // Dominance, not layout order: the defining block is laid out after the using block, so the
    // definition's id is larger than the use's.
    let late_layout = kernel(|f, _| {
        let a1 = f.arg(1);
        let (use_block, def_block) = (f.block(), f.block());
        f.br(def_block);
        f.switch_to(def_block);
        let x = f.bin(BinOp::Add, a1, a1);
        f.br(use_block);
        f.switch_to(use_block);
        f.bin(BinOp::Add, x, x);
        f.ret(None);
    });
    let module = late_layout.unwrap();
    let body = module.parts().funcs.values().next();
    let body = body.and_then(|k| k.body.as_ref()).unwrap();
    let use_inst = body.insts.get(InstId::new(1)).unwrap();
    assert_eq!(
        use_inst.kind,
        add(Value::Inst(InstId::new(3)), Value::Inst(InstId::new(3)))
    );

    // A loop-body value is not available in the header, although the back edge reaches it.
    let body_in_header = kernel(|f, t| {
        let a1 = f.arg(1);
        let (header, latch, exit) = (f.block(), f.block(), f.block());
        f.br(header);
        f.switch_to(latch);
        let n = f.bin(BinOp::Add, a1, a1);
        f.br(header);
        f.switch_to(header);
        let c = icmp(f, t, n, a1);
        f.cond_br(c, latch, exit);
        f.switch_to(exit);
        f.ret(None);
    });
    assert_eq!(verify_kind(body_in_header), VerifyError::UseNotDominated);
}

/// `entry: br header`, `header: p = phi [<from entry>, entry], [<from header>, header];
/// q = add p, 1; c = icmp q, a1; br c, header, exit`, `exit: ret`. `incoming` receives `p`, `q`.
fn self_loop(incoming: fn(Value, Value, Value) -> [Value; 2]) -> Result<Module, Error> {
    kernel(move |f, t| {
        let a1 = f.arg(1);
        let zero = f.module().const_int(t.i32, 0);
        let one = f.module().const_int(t.i32, 1);
        let (header, exit) = (f.block(), f.block());
        f.br(header); // i0
        f.switch_to(header);
        // Push order is layout order here, so the builder keeps these ids.
        let (p, q) = (Value::Inst(InstId::new(1)), Value::Inst(InstId::new(2)));
        let [from_entry, from_header] = incoming(p, q, zero);
        let phi = phi_of(
            f,
            t.i32,
            &[(from_entry, BlockId::new(0)), (from_header, header)],
        );
        assert_eq!(phi, p);
        assert_eq!(f.bin(BinOp::Add, p, one), q);
        let c = icmp(f, t, q, a1);
        f.cond_br(c, header, exit);
        f.switch_to(exit);
        f.ret(None);
    })
}

#[test]
fn phi_values_must_dominate_the_end_of_their_incoming_block() {
    // Each arm's value flows in on its own edge; the entry's value on either.
    assert!(diamond(|f, t, [x, y, e], [then, els]| {
        phi_of(f, t.i32, &[(y, then), (e, els)]);
        phi_of(f, t.i32, &[(x, then), (x, els)]);
    })
    .is_ok());
    // A value of the other arm does not reach this edge.
    let crossed = diamond(|f, t, [_, y, e], [then, els]| {
        phi_of(f, t.i32, &[(e, then), (y, els)]);
    });
    assert_eq!(verify_kind(crossed), VerifyError::UseNotDominated);

    // On a self-loop the back edge carries a later value of the PHI's own block, or the PHI.
    assert!(self_loop(|_, q, zero| [zero, q]).is_ok());
    assert!(self_loop(|p, _, zero| [zero, p]).is_ok());
    // The header's values do not reach the edge from the entry, which the header does not
    // dominate.
    assert_eq!(
        verify_kind(self_loop(|_, q, zero| [q, zero])),
        VerifyError::UseNotDominated
    );
    assert_eq!(
        verify_kind(self_loop(|p, _, zero| [p, zero])),
        VerifyError::UseNotDominated
    );
}

#[test]
fn unreachable_uses_are_exempt_and_unreachable_definitions_dominate_nothing() {
    // `entry: <entry_body>`, `dead: x = add x, x; y = add z, z; ret` (unreachable),
    // `other_dead: z = add a1, a1; ret` (unreachable). `x` uses itself and `y` uses a value of a
    // block that does not dominate it: both are exempt, as in LLVM.
    let build = |entry_body: fn(&mut FunctionBuilder<'_>, Tys, [Value; 2], BlockId)| {
        kernel(move |f, t| {
            let a1 = f.arg(1);
            let (join, dead, other_dead) = (f.block(), f.block(), f.block());
            f.switch_to(other_dead);
            let z = f.bin(BinOp::Add, a1, a1);
            f.ret(None);
            f.switch_to(dead);
            let x = f.push(
                t.i32,
                add(Value::Inst(InstId::new(2)), Value::Inst(InstId::new(2))),
            );
            assert_eq!(x, Value::Inst(InstId::new(2)));
            f.bin(BinOp::Add, z, z);
            f.br(join);
            f.switch_to(BlockId::new(0));
            entry_body(f, t, [x, z], dead);
            f.switch_to(join);
            f.ret(None);
        })
    };
    // The entry branches to `join`; `join` has the unreachable `dead` as a second predecessor.
    let ok = build(|f, t, [x, _], dead| {
        f.br(BlockId::new(1));
        f.switch_to(BlockId::new(1));
        // The edge from the unreachable block may carry anything.
        let a1 = f.arg(1);
        phi_of(f, t.i32, &[(a1, BlockId::new(0)), (x, dead)]);
    });
    assert!(ok.is_ok(), "{ok:?}");

    // A definition in an unreachable block is not available in a reachable one.
    let from_dead = build(|f, _, [_, z], _| {
        f.bin(BinOp::Add, z, z);
        f.br(BlockId::new(1));
        f.switch_to(BlockId::new(1));
        let a1 = f.arg(1);
        f.bin(BinOp::Add, a1, a1);
    });
    assert_eq!(verify_kind(from_dead), VerifyError::UseNotDominated);
    let phi_from_dead = build(|f, t, [_, z], dead| {
        f.br(BlockId::new(1));
        f.switch_to(BlockId::new(1));
        phi_of(f, t.i32, &[(z, BlockId::new(0)), (z, dead)]);
    });
    assert_eq!(verify_kind(phi_from_dead), VerifyError::UseNotDominated);
}

#[test]
fn dominance_is_charged_to_the_ledger_and_returned() {
    let parts = diamond(|f, t, [_, y, e], [then, els]| {
        phi_of(f, t.i32, &[(y, then), (e, els)]);
    })
    .unwrap()
    .into_parts();
    let mut m = meter();
    assert!(Module::new(parts.clone(), &mut m).is_ok());
    assert_eq!(m.ledger(), 0, "the CFG and dominator tree are released");
    let peak = m.ledger_peak();
    assert!(peak > 0);
    // Refused before reserving when the ledger cannot hold the analysis.
    let mut tight = Meter::with_caps(&Budget::default(), u64::MAX, peak - 1);
    assert_eq!(
        Module::new(parts, &mut tight),
        Err(Error::LimitExceeded(crate::error::Limit::LedgerBytes))
    );
}
