//! L0 tests: buffer tables (`array_ref<device T*>` parameters) with one slot per accessed element,
//! `ptrtoint` as no class seed, and the shared `insertvalue`-chain resolution.

use super::*;

/// `[[buffer(first)]]` passed as a table of `count` device pointers (a thread-space parameter).
fn table(first: u32, count: u32) -> ParamRole {
    ParamRole::Buffer(BufferArg {
        location: Slot {
            index: AbiInt::Const(first),
            count: AbiInt::Const(count),
        },
        access: Access::ReadWrite,
        space: AddrSpace::Device,
        buffer_size: None,
        layout: None,
        raster_order_group: None,
    })
}

/// A load of the device pointer at element `k` of the table `t` (a GEP over pointers).
fn element(fb: &mut FunctionBuilder<'_>, t: Value, k: Value) -> Value {
    let ptr = fb.module().ptr(AddrSpace::Device);
    let at = gep(fb, ptr, t, &[k]);
    fb.push(
        ptr,
        InstKind::Load {
            ptr: at,
            align: align(8),
            volatile: false,
            atomic: None,
            range: None,
        },
    )
}

fn arg(i: u32) -> Value {
    Value::Arg(crate::air::ir::ArgIdx::new(i))
}

fn slot_view(p: &MemoryPlan) -> Vec<(u32, Option<u32>, AccessSet, Option<Window>)> {
    p.slots
        .iter()
        .map(|s| (s.slot.index, s.element, s.access, s.window))
        .collect()
}

/// Constant element indices: each accessed element is its own slot `first + k` with its own
/// window, the element pointer is a static fat pointer over that element, and the table pointer
/// is a `BufferTable` offset. The table has no slot of its own.
#[test]
fn a_buffer_table_has_one_slot_per_accessed_element() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Thread, table(2, 3));
    let (mut e1, mut at, mut e2) = (arg(0), arg(0), arg(0));
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let (one, two, three) = (
            fb.module().const_int(i64t, 1),
            fb.module().const_int(i64t, 2),
            fb.module().const_int(i64t, 3),
        );
        e1 = element(fb, fb.arg(0), one);
        at = gep(fb, i32t, e1, &[three]);
        let v = fb.module().const_int(i32t, 7);
        store(fb, v, at);
        e2 = element(fb, fb.arg(0), two);
        let _ = load(fb, i32t, e2);
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(
        slot_view(&plan),
        vec![
            (
                3,
                Some(1),
                AccessSet::WRITE,
                Some(Window::Bounded { start: 12, len: 4 })
            ),
            (
                4,
                Some(2),
                AccessSet::READ,
                Some(Window::Bounded { start: 0, len: 4 })
            ),
        ]
    );
    assert_eq!(plan.repr(arg(0)), Some(Repr::BufferTable { param: 0 }));
    assert_eq!(
        repr_roots(&plan, e1),
        vec![Root::BufferElement {
            param: 0,
            element: 1
        }]
    );
    assert!(matches!(
        plan.repr(e1),
        Some(Repr::BufPtr { wide: false, .. })
    ));
    assert_eq!(plan.static_slot(at), Some((0, 12)));
    assert_eq!(plan.static_slot(e2), Some((1, 0)));
    assert_eq!(plan.slot_of_param(0), None);
    assert_eq!(
        plan.slot_of_root(Root::BufferElement {
            param: 0,
            element: 2
        }),
        Some(1)
    );
    assert!(!plan.needs.dynamic_roots);
}

/// A dynamic element index may read any element: the pointer's candidates are every element,
/// each accessed element gets a slot with the access's window, and nothing is a dynamic root,
/// so Profile F (no device addresses) accepts it.
#[test]
fn a_dynamic_element_index_reaches_every_element_and_stays_static() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Thread, table(0, 3));
    k.param(AddrSpace::Device, buffer(5, AddrSpace::Device, None));
    let mut q = arg(0);
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let i = load(fb, i64t, fb.arg(1));
        q = element(fb, fb.arg(0), i);
        let v = fb.module().const_int(i32t, 1);
        store(fb, v, q);
        fb.ret(None);
    });
    for p in [addressed(), floor()] {
        let plan = plan(&e, &p).expect("analyze");
        let want: Vec<Root> = (0..3)
            .map(|element| Root::BufferElement { param: 0, element })
            .collect();
        assert_eq!(repr_roots(&plan, q), want);
        let w = Some(Window::Bounded { start: 0, len: 4 });
        assert_eq!(
            slot_view(&plan),
            vec![
                (0, Some(0), AccessSet::WRITE, w),
                (1, Some(1), AccessSet::WRITE, w),
                (2, Some(2), AccessSet::WRITE, w),
                (
                    5,
                    None,
                    AccessSet::READ,
                    Some(Window::Bounded { start: 0, len: 8 })
                ),
            ]
        );
        assert!(!plan.needs.dynamic_roots);
    }
}

/// A constant index outside the table reads no element (undefined in Metal): the null
/// pointer, and no slot.
#[test]
fn a_constant_index_outside_the_table_is_the_null_pointer() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Thread, table(0, 2));
    let mut q = arg(0);
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let i64t = fb.module().int(64);
        let five = fb.module().const_int(i64t, 5);
        q = element(fb, fb.arg(0), five);
        let _ = load(fb, i32t, q);
        fb.ret(None);
    });
    let plan = ok(&e);
    assert_eq!(repr_roots(&plan, q), vec![Root::Null]);
    assert!(plan.slots.is_empty());
}

/// Every access to a table other than a load of one whole element pointer has no model: a
/// store into it, a load of its bytes as data, a load at an offset inside an element, and the
/// table pointer escaping (stored as a value) or merged with another root.
#[test]
fn other_uses_of_a_table_are_not_implemented() {
    let refused = |body: &dyn Fn(&mut FunctionBuilder<'_>)| {
        let mut k = Kernel::new();
        k.param(AddrSpace::Thread, table(0, 2));
        let e = k.build(|fb| {
            body(fb);
            fb.ret(None);
        });
        refused_feature(plan(&e, &addressed()))
    };
    let access = Feature::Lowering(LowerFamily::Access);
    // A store of a pointer into the table.
    assert_eq!(
        refused(&|fb| {
            let ptr = fb.module().ptr(AddrSpace::Device);
            let null = fb.module().constant(ptr, Const::Null);
            store(fb, Value::Const(null), fb.arg(0));
        }),
        access
    );
    // A load of the table's bytes as an integer.
    assert_eq!(
        refused(&|fb| {
            let i64t = fb.module().int(64);
            let _ = load(fb, i64t, fb.arg(0));
        }),
        access
    );
    // A pointer load at byte 4, inside element 0.
    assert_eq!(
        refused(&|fb| {
            let i8t = fb.module().int(8);
            let i64t = fb.module().int(64);
            let ptr = fb.module().ptr(AddrSpace::Device);
            let four = fb.module().const_int(i64t, 4);
            let at = gep(fb, i8t, fb.arg(0), &[four]);
            let _ = load(fb, ptr, at);
        }),
        access
    );
    // The table pointer stored into a private slot whose bytes are also read as an integer (an
    // open cell, the pointer escapes).
    assert_eq!(
        refused(&|fb| {
            let tptr = fb.module().ptr(AddrSpace::Thread);
            let i32t = fb.module().int(32);
            let i64t = fb.module().int(64);
            let one = fb.module().const_int(i32t, 1);
            let cell = fb.push(
                tptr,
                InstKind::Alloca {
                    ty: tptr,
                    count: one,
                    align: align(8),
                },
            );
            store(fb, fb.arg(0), cell);
            let _ = load(fb, i64t, cell);
        }),
        access
    );
}

/// An argument-buffer table's elements are argument buffers: a load at a member whose
/// `air.struct_type_info` kind is a texture seeds a (dynamic) texture handle.
#[test]
fn an_argument_buffer_table_element_seeds_its_members() {
    let mut k = Kernel::new();
    k.structs.push(StructInfo {
        members: vec![Member {
            offset: 0,
            size: 8,
            array_len: 0,
            layout: None,
            kind: MemberKind::Indirect(IndirectArg {
                ordinal: 0,
                role: texture(0),
                ty: ArgTypeInfo::default(),
                name: None,
            }),
            render_target: None,
            raster_order_group: None,
            type_name: DiagName::new("texture2d<float>"),
            name: DiagName::new("t"),
            key: crate::air::abi::UserKey(b"t".as_slice().into()),
        }],
    });
    k.param(
        AddrSpace::Thread,
        ParamRole::ArgumentBuffer(BufferArg {
            location: Slot {
                index: AbiInt::Const(4),
                count: AbiInt::Const(2),
            },
            access: Access::Read,
            space: AddrSpace::Constant,
            buffer_size: None,
            layout: Some(StructRef(0)),
            raster_order_group: None,
        }),
    );
    let mut t = arg(0);
    let e = k.build(|fb| {
        let i64t = fb.module().int(64);
        let cptr = fb.module().ptr(AddrSpace::Constant);
        let dptr = fb.module().ptr(AddrSpace::Device);
        let one = fb.module().const_int(i64t, 1);
        let at = gep(fb, cptr, fb.arg(0), &[one]);
        let ab = fb.push(
            cptr,
            InstKind::Load {
                ptr: at,
                align: align(8),
                volatile: false,
                atomic: None,
                range: None,
            },
        );
        t = fb.push(
            dptr,
            InstKind::Load {
                ptr: ab,
                align: align(8),
                volatile: false,
                atomic: None,
                range: None,
            },
        );
        let _ = fb.push(dptr, InstKind::Freeze(t));
        fb.ret(None);
    });
    let plan = ok(&e);
    assert!(matches!(
        plan.repr(t),
        Some(Repr::Handle {
            class: HandleClass::Texture,
            ..
        })
    ));
    assert_eq!(plan.needs.handle_table, Some(vec![HandleClass::Texture]));
    assert_eq!(
        slot_view(&plan),
        vec![(
            5,
            Some(1),
            AccessSet::READ,
            Some(Window::Bounded { start: 0, len: 8 })
        )]
    );
    assert!(plan.slots.first().is_some_and(|s| s.argument_buffer));
}

/// A threadgroup role passed as a table (an array of regions) has no arena model.
#[test]
fn a_threadgroup_table_is_not_implemented() {
    let mut k = Kernel::new();
    k.param(
        AddrSpace::Thread,
        ParamRole::Threadgroup(BufferArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(2),
            },
            access: Access::ReadWrite,
            space: AddrSpace::Threadgroup,
            buffer_size: None,
            layout: None,
            raster_order_group: None,
        }),
    );
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let _ = load(fb, i32t, fb.arg(0));
        fb.ret(None);
    });
    assert_eq!(
        refused_feature(plan(&e, &addressed())),
        Feature::Lowering(LowerFamily::Access)
    );
}

/// `ptrtoint` of a handle is its resource identity, not a byte use (the frontend copies a
/// struct holding a texture through an integer): the texture stays a handle. A GEP on it still
/// conflicts (`a_handle_used_as_bytes_or_merged_with_bytes_is_refused`).
#[test]
fn ptrtoint_of_a_handle_keeps_it_a_handle() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, texture(0));
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    let e = k.build(|fb| {
        let i64t = fb.module().int(64);
        let bits = fb.push(
            i64t,
            InstKind::Cast {
                op: CastOp::PtrToInt,
                v: fb.arg(0),
            },
        );
        store(fb, bits, fb.arg(1));
        fb.ret(None);
    });
    let plan = ok(&e);
    assert!(matches!(
        plan.repr(arg(0)),
        Some(Repr::Handle {
            class: HandleClass::Texture,
            ..
        })
    ));
    assert_eq!(repr_roots(&plan, arg(0)), vec![Root::Param(0)]);
}

// ---- extractvalue resolution (shared insertvalue chains) ----

/// A struct of `n` buffer pointers built by an `insertvalue` chain over `n` parameters, then
/// every field extracted and stored through: the extracts return the fields' pointers.
fn chain_entry(n: u32) -> (Entry, Vec<Value>) {
    let mut k = Kernel::new();
    for i in 0..n {
        k.param(AddrSpace::Device, buffer(i, AddrSpace::Device, None));
    }
    let mut out = Vec::new();
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let fields: Vec<TypeId> = (0..n).map(|_| ptr).collect();
        let st = fb.module().struct_ty(&fields);
        let mut agg = Value::Const(fb.module().constant(st, Const::Undef));
        for i in 0..n {
            agg = fb.push(
                st,
                InstKind::InsertValue {
                    agg,
                    val: fb.arg(i),
                    indices: [i].into(),
                },
            );
        }
        let v = fb.module().const_int(i32t, 1);
        for i in 0..n {
            let q = fb.push(
                ptr,
                InstKind::ExtractValue {
                    agg,
                    indices: [i].into(),
                },
            );
            store(fb, v, q);
            out.push(q);
        }
        fb.ret(None);
    });
    (e, out)
}

/// Every extract of one chain resolves to its own field, and the chain is read once: the work
/// grows linearly with the chain (doubling it at most about doubles the work; reading the chain
/// per extract would quadruple it).
#[test]
fn extracts_of_one_chain_share_one_visit() {
    let work = |n: u32| {
        let (e, qs) = chain_entry(n);
        let mut m = meter();
        let plan = analyze_memory(&e, &addressed(), &mut m).expect("analyze");
        for (i, q) in qs.iter().enumerate() {
            assert_eq!(repr_roots(&plan, *q), vec![Root::Param(i as u32)]);
        }
        m.work()
    };
    let (w64, w128) = (work(64), work(128));
    assert!(w128 < w64 * 5 / 2, "work {w64} -> {w128} is not linear");
}

/// The first insertion from the top decides: a later insertion at the same index hides an
/// earlier one, and a nearer insertion at a sibling path (`[0, 1]` for the member `[0, 0]`)
/// does not hide the farther insertion of the whole enclosing aggregate (`[0]`).
#[test]
fn the_nearest_insertion_supplies_the_member() {
    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    k.param(AddrSpace::Device, buffer(1, AddrSpace::Device, None));
    let mut q = arg(0);
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let st = fb.module().struct_ty(&[ptr, ptr]);
        let undef = Value::Const(fb.module().constant(st, Const::Undef));
        let a = fb.push(
            st,
            InstKind::InsertValue {
                agg: undef,
                val: fb.arg(0),
                indices: [0].into(),
            },
        );
        let b = fb.push(
            st,
            InstKind::InsertValue {
                agg: a,
                val: fb.arg(1),
                indices: [0].into(),
            },
        );
        q = fb.push(
            ptr,
            InstKind::ExtractValue {
                agg: b,
                indices: [0].into(),
            },
        );
        let v = fb.module().const_int(i32t, 1);
        store(fb, v, q);
        fb.ret(None);
    });
    assert_eq!(repr_roots(&ok(&e), q), vec![Root::Param(1)]);

    let mut k = Kernel::new();
    k.param(AddrSpace::Device, buffer(0, AddrSpace::Device, None));
    let mut q = arg(0);
    let e = k.build(|fb| {
        let i32t = fb.module().int(32);
        let ptr = fb.module().ptr(AddrSpace::Device);
        let inner = fb.module().struct_ty(&[ptr, ptr]);
        let st = fb.module().struct_ty(&[inner]);
        let whole = fb.module().constant(inner, Const::Undef);
        let undef = Value::Const(fb.module().constant(st, Const::Undef));
        let a = fb.push(
            st,
            InstKind::InsertValue {
                agg: undef,
                val: Value::Const(whole),
                indices: [0].into(),
            },
        );
        let b = fb.push(
            st,
            InstKind::InsertValue {
                agg: a,
                val: fb.arg(0),
                indices: [0, 1].into(),
            },
        );
        let inner_v = fb.push(
            inner,
            InstKind::ExtractValue {
                agg: b,
                indices: [0].into(),
            },
        );
        q = fb.push(
            ptr,
            InstKind::ExtractValue {
                agg: inner_v,
                indices: [0].into(),
            },
        );
        let v = fb.module().const_int(i32t, 1);
        store(fb, v, q);
        fb.ret(None);
    });
    // `extractvalue (extractvalue b, 0), 0` is the path [0, 0] of `b`: [0, 1] does not
    // overlap it, and [0] (below) supplies it as `undef` (null).
    let plan = ok(&e);
    assert_eq!(repr_roots(&plan, q), vec![Root::Null]);
    assert_eq!(window_of(&plan, 0), None);
}
