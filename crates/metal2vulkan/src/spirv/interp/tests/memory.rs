//! Memory: explicit layouts, push constants, physical pointers and their traps, descriptor and
//! access-chain bounds with wrapped indices.

use super::*;
use crate::pipeline::sym::PushField;

fn aligned(a: u32) -> MemAccess {
    MemAccess {
        aligned: Some(a),
        ..MemAccess::default()
    }
}

/// A `Block { u64 data[]; }` buffer at slot `index`.
fn u64_buffer(b: &mut ModuleBuilder, index: u32) -> GlobalId {
    let u64_ = b
        .ty(Type::Int {
            width: 64,
            signed: false,
        })
        .expect("u64");
    let rta = b
        .ty(Type::RuntimeArray {
            elem: u64_,
            stride: Some(8),
        })
        .expect("rta");
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta]).expect("layout");
    let block = b
        .ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })
        .expect("block");
    b.resource(slot(index), StorageClass::StorageBuffer, block)
        .expect("resource")
}

const BASE: u64 = 0x1_0000_0000;

/// `n` invocations: `p = (u32*)(addr[0] + 4 * li + skew); *p += 10`.
fn physical_kernel(n: u32, skew: u64) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let _ = buffer(&mut b, 0);
    let addrs = u64_buffer(&mut b, 1);
    let u32_ = u32_ty(&mut b);
    let u64_ = b
        .ty(Type::Int {
            width: 64,
            signed: false,
        })
        .expect("u64");
    let ptr = b
        .ty(Type::Pointer {
            class: StorageClass::PhysicalStorageBuffer,
            pointee: u32_,
        })
        .expect("ptr");
    let skew_c = b
        .constant(Constant::Scalar {
            ty: u64_,
            bits: ConstBits(skew),
        })
        .expect("skew");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let li = lid(&mut f);
    let zero = c32(&mut f, 0);
    let a = load_at(&mut f, addrs, zero).expect("addr");
    let four = c32(&mut f, 4);
    let off = f.binary(BinOp::IMul, li, four).expect("mul");
    let off = f.convert(ConvOp::UConvert, off, u64_).expect("widen");
    let a = f.binary(BinOp::IAdd, a, off).expect("add");
    let sk = f.module().const_value(skew_c).expect("skew");
    let a = f.binary(BinOp::IAdd, a, sk).expect("add");
    let p = f.convert(ConvOp::ConvertUToPtr, a, ptr).expect("to ptr");
    let v = f.load(p, aligned(4)).expect("load");
    let ten = c32(&mut f, 10);
    let v = f.binary(BinOp::IAdd, v, ten).expect("add");
    f.store(p, v, aligned(4)).expect("store");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: n, y: 1, z: 1 }],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

fn physical_resources() -> Resources {
    Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: Some(BASE),
                bytes: [1u32, 2, 3, 4]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: BASE.to_le_bytes().to_vec(),
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    }
}

#[test]
fn physical_pointers_reach_addressed_buffers() {
    let m = physical_kernel(4, 0);
    let out = run_both(&m, Config::default(), &physical_resources());
    assert_eq!(words(&out.buffers[0]), vec![11, 12, 13, 14]);
}

#[test]
fn physical_access_past_the_buffer_traps() {
    let m = physical_kernel(5, 0);
    match run(&m, &Config::default(), physical_resources()) {
        Err(InterpError::Trap(t)) => {
            assert_eq!(
                t.kind,
                TrapKind::OutOfBounds {
                    memory: MemoryKind::Physical,
                    offset: BASE + 16,
                    len: 4,
                    size: 0
                }
            );
            assert_eq!(t.invocation, Some([4, 0, 0]));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn misaligned_physical_access_traps() {
    let m = physical_kernel(1, 2);
    assert_eq!(
        trap_of(run(&m, &Config::default(), physical_resources())),
        TrapKind::Misaligned {
            address: BASE + 2,
            align: 4
        }
    );
}

#[test]
fn descriptor_indices_never_wrap() {
    // out[li - 1]: invocation 0 indexes 0xFFFF_FFFF, which is out of bounds, not element 0 of a
    // wrapped address space.
    let m = kernel([2, 1, 1], 1, |f, b| {
        let li = lid(f);
        let one = c32(f, 1);
        let i = f.binary(BinOp::ISub, li, one)?;
        store_at(f, b[0], i, li)
    });
    // Access-chain indices are signed: 0xFFFF_FFFF as a u32 index is -1.
    match run(&m, &Config::default(), buffers(&[4])) {
        Err(InterpError::Trap(t)) => {
            assert_eq!(t.kind, TrapKind::NegativeIndex { index: -1 });
            assert_eq!(t.invocation, Some([0, 0, 0]));
        }
        other => panic!("{other:?}"),
    }
    // A 64-bit index past the end is reported against the buffer.
    let m = kernel([1, 1, 1], 1, |f, b| {
        let u64_ = f.module().ty(Type::Int {
            width: 64,
            signed: false,
        })?;
        let c = f.module().constant(Constant::Scalar {
            ty: u64_,
            bits: ConstBits(0x4000_0000),
        })?;
        let i = f.module().const_value(c)?;
        let zero = c32(f, 0);
        store_at(f, b[0], i, zero)
    });
    assert_eq!(
        trap_of(run(&m, &Config::default(), buffers(&[4]))),
        TrapKind::OutOfBounds {
            memory: MemoryKind::Buffer,
            offset: 0x1_0000_0000,
            len: 4,
            size: 16
        }
    );
}

#[test]
fn sized_array_indices_are_checked() {
    for (idx, signed, want) in [(4u32, false, 4i64), (u32::MAX, true, -1)] {
        let m = kernel([1, 1, 1], 1, |f, b| {
            let u32_ = u32_ty(f.module());
            let len = f.u32_const(4)?;
            let arr = f.module().ty(Type::Array {
                elem: u32_,
                len,
                stride: None,
            })?;
            let var = f.local_var(arr, None)?;
            let base = f.var_ptr(var)?;
            // The index comes from memory so the builder cannot range-check it.
            let zero = c32(f, 0);
            let x = load_at(f, b[0], zero)?;
            let k = c32(f, idx);
            let i = f.binary(BinOp::IAdd, x, k)?;
            let i = if signed {
                let i32_ = f.module().ty(Type::Int {
                    width: 32,
                    signed: true,
                })?;
                f.convert(ConvOp::Bitcast, i, i32_)?
            } else {
                i
            };
            let p = f.access_chain(base, &[i])?;
            f.store(p, zero, MemAccess::default())
        });
        assert_eq!(
            trap_of(run(&m, &Config::default(), buffers(&[1]))),
            TrapKind::IndexOutOfRange {
                index: want,
                len: 4
            }
        );
    }
}

#[test]
fn explicit_layouts_and_push_constants() {
    let mut b = ModuleBuilder::new();
    let out = buffer(&mut b, 0);
    let f32_ = b.ty(Type::Float { width: 32 }).expect("f32");
    let u32_ = u32_ty(&mut b);
    let vec2 = b.ty(Type::Vector { elem: f32_, len: 2 }).expect("vec2");
    let mat2 = b.ty(Type::Matrix { col: vec2, cols: 2 }).expect("mat2");
    // Block { f32 s; mat2 m; } under std140: m at 16 with MatrixStride 16.
    let layout =
        natural_layout(b.types(), LayoutRule::Std140, true, &[f32_, mat2]).expect("layout");
    assert_eq!(layout.members[1].offset, 16);
    assert_eq!(layout.members[1].matrix_stride, Some(16));
    let block = b
        .ty(Type::Struct {
            members: [f32_, mat2].into(),
            layout: Some(layout),
        })
        .expect("block");
    let ubo = b
        .resource(slot(1), StorageClass::Uniform, block)
        .expect("ubo");
    b.push_member(PushField::DispatchRegion, u32_, 0)
        .expect("push");
    let push = b.push_global().expect("push block");
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    let base = f.module().global_value(ubo).expect("ubo");
    let one = c32(&mut f, 1);
    let zero = c32(&mut f, 0);
    // m[1][0] through the access chain, m[1][1] through a whole-matrix load.
    let p = f.access_chain(base, &[one, one, zero]).expect("chain");
    let a = f.load(p, MemAccess::default()).expect("load");
    let pm = f.access_chain(base, &[one]).expect("chain");
    let m = f.load(pm, MemAccess::default()).expect("load");
    let bb = f.composite_extract(m, &[1, 1]).expect("extract");
    let pp = f.module().global_value(push).expect("push");
    let pq = f.access_chain(pp, &[zero]).expect("chain");
    let c = f.load(pq, MemAccess::default()).expect("load");
    let a = f.convert(ConvOp::Bitcast, a, u32_).expect("cast");
    let bb = f.convert(ConvOp::Bitcast, bb, u32_).expect("cast");
    store_at(&mut f, out, zero, a).expect("store");
    store_at(&mut f, out, one, bb).expect("store");
    let two = c32(&mut f, 2);
    store_at(&mut f, out, two, c).expect("store");
    f.ret(None).expect("ret");
    let id = f.finish().expect("finish");
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .expect("entry");
    let m = b.finalize().expect("finalize");
    let mut ubo_bytes = vec![0u8; 48];
    for (off, v) in [(32usize, 5.5f32), (36, -7.25)] {
        ubo_bytes[off..off + 4].copy_from_slice(&v.to_bits().to_le_bytes());
    }
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes: vec![0; 12],
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: ubo_bytes,
            },
        ],
        push: 0xdead_beefu32.to_le_bytes().to_vec(),
        spec: Default::default(),
    };
    let out = run_both(&m, Config::default(), &res);
    assert_eq!(
        words(&out.buffers[0]),
        vec![5.5f32.to_bits(), (-7.25f32).to_bits(), 0xdead_beef]
    );
}

#[test]
fn an_unbound_buffer_is_a_config_error() {
    let m = kernel([1, 1, 1], 1, |f, b| {
        let zero = c32(f, 0);
        store_at(f, b[0], zero, zero)
    });
    assert!(matches!(
        run(&m, &Config::default(), Resources::default()),
        Err(InterpError::Config(_))
    ));
}

/// A run encodes a `Private` global's initializer once and gives each invocation its own copy
/// (`Prog::private_init`): an invocation that changes its copy leaves every later invocation's,
/// in its subgroup and in the next workgroup, at the initializer.
#[test]
fn every_invocation_starts_its_private_global_at_the_initializer() {
    let m = kernel([4, 1, 1], 1, |f, b| {
        let u32_ = u32_ty(f.module());
        let seven = f.u32_const(7)?;
        let g = f
            .module()
            .plain_global(StorageClass::Private, u32_, Some(seven))?;
        let p = f.module().global_value(g)?;
        let i = gid(f);
        let first = f.load(p, MemAccess::default())?;
        let one = c32(f, 1);
        let bump = f.binary(BinOp::IAdd, i, one)?;
        let changed = f.binary(BinOp::IAdd, first, bump)?;
        f.store(p, changed, MemAccess::default())?;
        let after = f.load(p, MemAccess::default())?;
        let two = c32(f, 2);
        let at = f.binary(BinOp::IMul, i, two)?;
        store_at(f, b[0], at, first)?;
        let next = f.binary(BinOp::IAdd, at, one)?;
        store_at(f, b[0], next, after)
    });
    let out = run_both(
        &m,
        cfg(2, SubgroupSize::W32, Order::Forward),
        &buffers(&[16]),
    );
    let expected: Vec<u32> = (0..8).flat_map(|i| [7, 8 + i]).collect();
    assert_eq!(words(&out.buffers[0]), expected);
}
