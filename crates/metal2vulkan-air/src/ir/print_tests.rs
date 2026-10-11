//! Printer tests: a golden snapshot and determinism.

use super::build::ModuleBuilder;
use super::print::print;
use super::*;

fn sample() -> Module {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let f = m.float();
    let f3 = m.vector(f, 3);
    let dev = m.ptr(AddrSpace::Device);
    let arr = m.array(f3, 2);
    let s = m.struct_ty(&[i32t, arr]);
    let zero = m.constant(f3, Const::Null);
    let init = m.constant(arr, Const::Aggregate(Box::new([zero, zero])));
    let g = m.simple_global("table\"x", arr, AddrSpace::Constant, Some(init));
    let decl_ty = m.fn_ty(i32t, &[i32t]);
    let decl = m.declare("air.helper", decl_ty);
    let fty = m.fn_ty(void, &[dev, i32t]);
    let one = m.const_int(i32t, 1);
    let mut k = m.define("kernel", fty);
    let exit = k.block();
    let sum = k.bin(BinOp::Add, k.arg(1), one);
    let gaddr = k.module().global_addr(g);
    let p = k.push(
        dev,
        InstKind::Gep {
            src_ty: s,
            base: k.arg(0),
            indices: Box::new([one, one]),
            inbounds: true,
        },
    );
    let _ = gaddr;
    let r = k.push(
        i32t,
        InstKind::Call(Box::new(Call {
            callee: Callee::UnknownIntrinsic(decl),
            args: Box::new([sum]),
            fmf: FastMath::NONE,
            tail: TailKind::Tail,
            attrs: None,
            range: None,
        })),
    );
    k.push_void(InstKind::Store {
        val: r,
        ptr: p,
        align: Align::from_bytes(4),
        volatile: false,
        atomic: None,
    });
    k.br(exit);
    k.switch_to(exit);
    k.ret(None);
    k.finish();
    let name = m.md_string("air.kernel");
    let node = m.md_node(&[Some(name), None]);
    m.named_md("air.kernel", &[node]);
    m.finish().unwrap()
}

#[test]
fn printer_output_is_pinned() {
    let text = print(&sample());
    let body = text.split_once('\n').map_or("", |(_, rest)| rest);
    let expected = r#"%t5 = type [2 x <3 x float>]
%t6 = type { i32, %t5 }
%t7 = type i32 (i32)
%t8 = type void (ptr addrspace(1), i32)
c1 = %t5 [zeroinitializer, zeroinitializer]
@g0 "table\22x" = internal addrspace(2) global %t5 c1

declare %t7 @f0 "air.helper"

define %t8 @f1 "kernel" {
bb0:
  %i0 = add i32 %a1, i32 1
  %i1 = getelementptr inbounds %t6, %a0, i32 1, i32 1
  %i2 = tail call i32 @f0 [unknown](%i0)
  store %i2, %i1, align 4
  br bb1
bb1:
  ret void
}
!md0 = !"air.kernel"
!md1 = !{!md0, null}
!"air.kernel" = !{!md1}
"#;
    assert_eq!(body, expected, "full output:\n{text}");
    assert!(text.starts_with("; datalayout DataLayout {"));
}

/// One instruction prints as its line of the full printout, for void and non-void results.
#[test]
fn an_instruction_prints_as_its_line_of_the_module() {
    let module = sample();
    let text = print(&module);
    let body = module
        .parts()
        .funcs
        .iter()
        .find_map(|(_, f)| f.body.as_ref())
        .unwrap();
    let mut printed = 0;
    for (_, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            let line = super::print::print_inst(&module, id, body.insts.get(id).unwrap());
            assert!(
                text.contains(&format!("\n  {line}\n")),
                "{line:?} in:\n{text}"
            );
            printed += 1;
        }
    }
    assert_eq!(printed, body.insts.len());
    let store = super::print::print_inst(
        &module,
        InstId::new(3),
        body.insts.get(InstId::new(3)).unwrap(),
    );
    assert!(store.starts_with("store "), "{store}");
}

#[test]
fn printing_is_deterministic() {
    let a = print(&sample());
    assert_eq!(a, print(&sample()));
    assert_eq!(a, print(&sample().clone()));
}

/// Every opcode, in a module that verifies, prints with its mnemonic.
#[test]
fn every_opcode_prints() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let f = m.float();
    let v2 = m.vector(i32t, 2);
    let pair = m.struct_ty(&[i32t, i1]);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let helper_ty = m.fn_ty(void, &[]);
    let helper = m.declare("helper", helper_ty);
    let fty = m.fn_ty(void, &[dev, i32t]);
    let one = m.const_int(i32t, 1);
    let fone = Value::Const(m.constant(f, Const::Float(0x3f80_0000)));
    let uv = Value::Const(m.constant(v2, Const::Undef));
    let up = Value::Const(m.constant(pair, Const::Undef));
    let t = Value::Const(m.constant(i1, Const::Int(1)));
    let mut k = m.define("k", fty);
    let (a0, a1) = (k.arg(0), k.arg(1));
    let (b1, b2, b3, b4) = (k.block(), k.block(), k.block(), k.block());
    let sys = Atomic {
        ordering: Ordering::SeqCst,
        scope: SyncScope::System,
    };
    let none = FastMath::NONE;
    let sum = k.bin(BinOp::Add, a1, one);
    let nsz = FastMath::NSZ;
    k.push(f, InstKind::FNeg { v: fone, fmf: nsz });
    let zext = CastOp::ZExt;
    k.push(i64t, InstKind::Cast { op: zext, v: sum });
    let (eq, olt) = (IntPredicate::Eq, FloatPredicate::Olt);
    let c = k.push(
        i1,
        InstKind::ICmp {
            pred: eq,
            a: sum,
            b: one,
        },
    );
    let (a, b) = (fone, fone);
    k.push(
        i1,
        InstKind::FCmp {
            pred: olt,
            fmf: none,
            a,
            b,
        },
    );
    k.push(
        i32t,
        InstKind::Select {
            c,
            t: sum,
            f: one,
            fmf: none,
        },
    );
    k.push(i32t, InstKind::Freeze(sum));
    k.push(i32t, InstKind::ExtractElement { vec: uv, idx: one });
    k.push(
        v2,
        InstKind::InsertElement {
            vec: uv,
            elt: sum,
            idx: one,
        },
    );
    let mask = Box::new([Some(3), None]);
    k.push(v2, InstKind::ShuffleVector { a: uv, b: uv, mask });
    let indices = Box::new([0]);
    k.push(i32t, InstKind::ExtractValue { agg: up, indices });
    let indices = Box::new([0]);
    k.push(
        pair,
        InstKind::InsertValue {
            agg: up,
            val: sum,
            indices,
        },
    );
    let align = Align::from_bytes(4);
    k.push(
        thread,
        InstKind::Alloca {
            ty: i32t,
            count: one,
            align,
        },
    );
    let (atomic, range) = (Some(sys), None);
    let load = InstKind::Load {
        ptr: a0,
        align: None,
        volatile: true,
        atomic,
        range,
    };
    let x = k.push(i32t, load);
    k.push_void(InstKind::Store {
        val: x,
        ptr: a0,
        align: None,
        volatile: false,
        atomic: None,
    });
    let indices = Box::new([one]);
    k.push(
        dev,
        InstKind::Gep {
            src_ty: i32t,
            base: a0,
            indices,
            inbounds: false,
        },
    );
    let op = AtomicRmwOp::UMax;
    let rmw = InstKind::AtomicRmw {
        op,
        ptr: a0,
        val: one,
        atomic: sys,
        volatile: false,
        align,
    };
    k.push(i32t, rmw);
    k.push(
        pair,
        InstKind::CmpXchg {
            ptr: a0,
            cmp: one,
            new: sum,
            success: sys,
            failure: Ordering::Acquire,
            weak: true,
            volatile: false,
            align: None,
        },
    );
    k.push_void(InstKind::Fence(Atomic {
        ordering: Ordering::Release,
        scope: SyncScope::SingleThread,
    }));
    k.push_void(InstKind::Call(Box::new(Call {
        callee: Callee::Direct(helper),
        args: Box::new([]),
        fmf: none,
        tail: TailKind::None,
        attrs: None,
        range: None,
    })));
    let hints = Some(LoopHints {
        unroll: Unroll::Count(4),
        must_progress: true,
    });
    let cases = Box::new([(7, b2)]);
    k.push_void(InstKind::Switch {
        v: a1,
        default: b1,
        cases,
        hints,
    });
    k.switch_to(b1);
    let incoming = Box::new([(sum, BlockId::new(0))]);
    k.push(
        i32t,
        InstKind::Phi {
            incoming,
            fmf: none,
        },
    );
    k.cond_br(t, b3, b4);
    k.switch_to(b2);
    k.push_void(InstKind::Unreachable);
    k.switch_to(b3);
    k.br(b4);
    k.switch_to(b4);
    k.ret(None);
    k.finish();
    let text = print(&m.finish().unwrap());
    for op in Opcode::ALL {
        let token = match op {
            Opcode::BinOp => "= add i32",
            Opcode::FNeg => "fneg nsz float",
            Opcode::Cast => "zext %i0 to i64",
            Opcode::ICmp => "icmp eq",
            Opcode::FCmp => "fcmp olt",
            Opcode::Select => "select i32",
            Opcode::Freeze => "freeze i32",
            Opcode::ExtractElement => "extractelement",
            Opcode::InsertElement => "insertelement",
            Opcode::ShuffleVector => "shufflevector <2 x i32>",
            Opcode::ExtractValue => "extractvalue i32",
            Opcode::InsertValue => "insertvalue",
            Opcode::Alloca => "alloca i32, i32 1, align 4",
            Opcode::Load => "load atomic volatile i32, %a0 seq_cst",
            Opcode::Store => "store %i13, %a0",
            Opcode::Gep => "getelementptr i32, %a0",
            Opcode::AtomicRmw => "atomicrmw umax",
            Opcode::CmpXchg => "cmpxchg weak",
            Opcode::Fence => "fence syncscope(\"singlethread\") release",
            Opcode::Call => "call void @f0()",
            Opcode::Phi => "phi i32 [%i0, bb0]",
            Opcode::Ret => "ret void",
            Opcode::Br => "br bb4",
            Opcode::CondBr => "br i1 1, bb3, bb4",
            Opcode::Switch => "switch %a1, bb1 [ 7: bb2 ], !llvm.loop",
            Opcode::Unreachable => "unreachable",
        };
        assert!(text.contains(token), "{op:?}: {token:?} missing in\n{text}");
    }
}
