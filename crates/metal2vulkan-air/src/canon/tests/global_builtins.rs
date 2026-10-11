//! Global built-in variables become built-in parameters: each `air.get_*` read of a program-scope
//! built-in is a read of a parameter appended to the entry, one per built-in and width, a vector
//! built-in's lane by `extractelement`.

use super::*;
use crate::abi::{Builtin, Stage};
use crate::ir::{InstKind, Type, TypeId};

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
    call(f, ret, Callee::Air { decl, op }, args)
}

/// `kernel k(device i32* out [[buffer(0)]])` storing the `i32` sum `body` computes to `out`.
fn entry(body: impl FnOnce(&mut FunctionBuilder<'_>) -> Value) -> Entry {
    entry_in(false, body)
}

/// [`entry`] as a kernel, or (`fragment`) as a fragment function storing to the same buffer.
fn entry_in(fragment: bool, body: impl FnOnce(&mut FunctionBuilder<'_>) -> Value) -> Entry {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = body(&mut f);
    store(&mut f, v, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let mut ep = kernel(k, vec![buffer(0, 0)]);
    if fragment {
        ep.stage = Stage::Fragment {
            early_fragment_tests: false,
        };
    }
    let abi = abi(vec![ep], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter()).unwrap()
}

/// The entry's parameter types, printed.
fn param_types(e: &Entry) -> Vec<String> {
    let f = e.module.func(e.func).unwrap();
    let Some(Type::Function { params, .. }) = e.module.types().get(f.ty) else {
        panic!("the entry's type is not a function");
    };
    params
        .iter()
        .map(|t| format!("{:?}", e.module.types().get(*t).unwrap()))
        .collect()
}

/// The global built-in reads left as calls.
fn calls_left(e: &Entry) -> usize {
    let body = e.module.func(e.func).unwrap().body.as_ref().unwrap();
    body.insts
        .iter()
        .filter(|(_, i)| {
            matches!(&i.kind, InstKind::Call(c) if matches!(c.callee,
                Callee::Air { op: AirIntrinsic::GlobalBuiltin(_), .. }))
        })
        .count()
}

/// `threads_per_threadgroup` read at lanes 1 and 2 as `uint3` is one `<3 x i32>` parameter,
/// `thread_position_in_grid`'s `x` as `ushort2` a `<3 x i16>` one, and
/// `thread_index_in_threadgroup` an `i32`, appended after the entry's own in the order the body
/// first reads them; every call becomes a lane of its parameter or the parameter itself.
#[test]
fn global_builtin_reads_become_builtin_parameters() {
    let e = entry(|f| {
        let i32t = f.module().int(32);
        let i16t = f.module().int(16);
        let (one, two) = (f.module().const_int(i32t, 1), f.module().const_int(i32t, 2));
        let zero16 = f.module().const_int(i16t, 0);
        let y = air_call(f, "air.get_local_size.i32", i32t, &[i32t], &[one]);
        let z = air_call(f, "air.get_local_size.i32", i32t, &[i32t], &[two]);
        let x = air_call(f, "air.get_global_id.i16", i16t, &[i16t], &[zero16]);
        let x = f.push(
            i32t,
            InstKind::Cast {
                op: CastOp::ZExt,
                v: x,
            },
        );
        let l = air_call(f, "air.get_local_linear_id.i32", i32t, &[], &[]);
        let yz = f.bin(BinOp::Add, y, z);
        let xl = f.bin(BinOp::Add, x, l);
        f.bin(BinOp::Add, yz, xl)
    });
    let roles: Vec<_> =
        e.ep.params
            .iter()
            .map(|p| (p.param, p.role.clone()))
            .collect();
    assert_eq!(
        roles[1..],
        [
            (1, ParamRole::Builtin(Builtin::ThreadsPerThreadgroup)),
            (2, ParamRole::Builtin(Builtin::ThreadPositionInGrid)),
            (3, ParamRole::Builtin(Builtin::ThreadIndexInThreadgroup)),
        ],
        "{roles:?}"
    );
    let tys = param_types(&e);
    assert_eq!(tys.len(), 4, "{tys:?}");
    assert!(
        tys[1].contains("Vector") && tys[1].contains("len: 3"),
        "{tys:?}"
    );
    assert!(
        tys[2].contains("Vector") && tys[2].contains("len: 3"),
        "{tys:?}"
    );
    assert_eq!(tys[3], "Int(32)", "{tys:?}");
    assert_eq!(calls_left(&e), 0);
    let text = body_text(&e);
    assert_eq!(text.matches("extractelement").count(), 3, "{text}");
    assert!(!text.contains("air.get_"), "{text}");
}

/// A read at a lane that is no constant below 3 stays a call (lowering refuses it) and adds no
/// parameter; a module without global built-ins keeps its entry's parameters.
#[test]
fn a_global_builtin_read_at_an_unknown_lane_stays_a_call() {
    let e = entry(|f| {
        let i32t = f.module().int(32);
        let lane = load(f, i32t, arg(0));
        let three = f.module().const_int(i32t, 3);
        let a = air_call(f, "air.get_local_size.i32", i32t, &[i32t], &[lane]);
        let b = air_call(f, "air.get_num_groups.i32", i32t, &[i32t], &[three]);
        f.bin(BinOp::Add, a, b)
    });
    assert_eq!(e.ep.params.len(), 1, "{:?}", e.ep.params);
    assert_eq!(param_types(&e).len(), 1);
    assert_eq!(calls_left(&e), 2);
    let plain = entry(|f| {
        let i32t = f.module().int(32);
        load(f, i32t, arg(0))
    });
    assert_eq!(plain.ep.params.len(), 1);
    assert_eq!(param_types(&plain).len(), 1);
}

/// In a fragment function, a global `thread_index_in_quadgroup` read stays a call (Metal refuses it
/// there; lowering refuses it as Metal does) and adds no parameter, while a global
/// `thread_index_in_simdgroup` read becomes its parameter as in a kernel; in a kernel the quad lane
/// becomes its parameter too.
#[test]
fn a_fragment_s_global_quad_lane_read_stays_a_call() {
    let body = |f: &mut FunctionBuilder<'_>| {
        let i32t = f.module().int(32);
        let quad = air_call(f, "air.get_thread_index_in_quadgroup.i32", i32t, &[], &[]);
        let simd = air_call(f, "air.get_thread_index_in_simdgroup.i32", i32t, &[], &[]);
        f.bin(BinOp::Add, quad, simd)
    };
    let fragment = entry_in(true, body);
    let roles: Vec<_> = fragment.ep.params.iter().map(|p| p.role.clone()).collect();
    assert_eq!(
        roles[1..],
        [ParamRole::Builtin(Builtin::ThreadIndexInSimdgroup)],
        "{roles:?}"
    );
    assert_eq!(calls_left(&fragment), 1);
    assert!(
        body_text(&fragment).contains("air.get_thread_index_in_quadgroup"),
        "{}",
        body_text(&fragment)
    );
    let kernel = entry(body);
    let roles: Vec<_> = kernel.ep.params.iter().map(|p| p.role.clone()).collect();
    assert_eq!(
        roles[1..],
        [
            ParamRole::Builtin(Builtin::ThreadIndexInQuadgroup),
            ParamRole::Builtin(Builtin::ThreadIndexInSimdgroup),
        ],
        "{roles:?}"
    );
    assert_eq!(calls_left(&kernel), 0);
}
