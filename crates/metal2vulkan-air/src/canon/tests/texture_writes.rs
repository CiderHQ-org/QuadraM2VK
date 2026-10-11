//! Whether an entry may write a texture: where its metadata names a texture it takes with write
//! access, and where its body calls a texture operation that stores a texel, whatever handle the
//! operation's texture operand is, or makes a call the walk cannot see into.

use super::command_buffers::{handle_from_integer, loaded_handle, module, void_call};
use super::*;
use crate::abi::TextureArg;
use crate::intrinsics::AtomicOp;

/// A texture at slot 0 declared with `access`, as parameter 0.
fn texture(access: Access) -> ParamBinding {
    ParamBinding {
        role: ParamRole::Texture(TextureArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(1),
            },
            access,
            raster_order_group: None,
        }),
        ..buffer(0, 0)
    }
}

/// `air.write_texture_2d.v4f32(tex, (0, 0), (0, 0, 0, 0), 0, 2)`: a write of texel (0, 0).
fn write(f: &mut FunctionBuilder<'_>, tex: Value) {
    let name = "air.write_texture_2d.v4f32";
    let void = f.module().void();
    let (i32t, f32t) = (f.module().int(32), f.module().float());
    let (v2i, v4f) = (f.module().vector(i32t, 2), f.module().vector(f32t, 4));
    let dev = f.module().ptr(AddrSpace::Device);
    let ty = f.module().fn_ty(void, &[dev, v2i, v4f, i32t, i32t]);
    let decl = f.module().declare(name, ty);
    let op = AirIntrinsic::parse(name).unwrap();
    assert!(matches!(op, AirIntrinsic::Texture(t) if t.op == TexOp::Write));
    let coord = Value::Const(f.module().constant(v2i, Const::Null));
    let texel = Value::Const(f.module().constant(v4f, Const::Null));
    let (zero, two) = (f.module().const_int(i32t, 0), f.module().const_int(i32t, 2));
    void_call(f, Callee::Air { decl, op }, &[tex, coord, texel, zero, two]);
}

/// A texture the metadata names with write access counts whatever the body does with it; one
/// declared for reading or sampling does not, nor does a kernel that takes none.
#[test]
fn a_writable_texture_the_metadata_names_counts_untouched() {
    for (first, may) in [
        (texture(Access::Write), true),
        (texture(Access::ReadWrite), true),
        (texture(Access::Read), false),
        (texture(Access::Sample), false),
        (buffer(0, 0), false),
    ] {
        let what = format!("{:?}", first.role);
        let (module, ep) = module(vec![first, buffer(1, 1)], |f| {
            let i32t = f.module().int(32);
            let one = f.module().const_int(i32t, 1);
            store(f, one, arg(1));
        });
        let abi = abi(vec![ep], Vec::new());
        let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
        assert_eq!(e.may_write_textures(), may, "{what}");
        assert_eq!(e.ep.takes_writable_textures(&e.structs), may, "{what}");
    }
}

/// A write counts whatever handle it writes through: a texture the metadata declares read-only,
/// a buffer's own address, a handle loaded from a member no layout describes, a handle made
/// from an integer (as a resource ID read from a buffer is). The same handles stored rather than
/// written through write no texture.
#[test]
fn a_write_through_any_handle_counts() {
    type Shape = fn(&mut FunctionBuilder<'_>) -> Value;
    let own: Shape = |_| arg(0);
    let cases: [(&str, ParamBinding, Shape); 4] = [
        ("a read-only texture", texture(Access::Read), own),
        ("the buffer's own address", buffer(0, 0), own),
        ("a handle loaded from a member", buffer(0, 0), loaded_handle),
        (
            "a handle made from an integer",
            buffer(0, 0),
            handle_from_integer,
        ),
    ];
    for (what, first, shape) in cases {
        let entry = |stores: bool| {
            let (module, ep) = module(vec![first.clone(), buffer(1, 1)], |f| {
                let tex = shape(f);
                if stores {
                    store(f, tex, arg(1));
                } else {
                    write(f, tex);
                }
            });
            let abi = abi(vec![ep], Vec::new());
            specialize(module, &abi, &request(&[]), &mut meter()).unwrap()
        };
        let writes = entry(false);
        assert!(
            !writes.ep.takes_writable_textures(&writes.structs),
            "{what}"
        );
        assert!(writes.may_write_textures(), "{what}");
        assert!(!entry(true).may_write_textures(), "{what}");
    }
}

/// Each texture operation that stores a texel counts, an atomic load and every read, sample,
/// query and fence do not; a call the walk cannot see into counts, a defined function's body is
/// walked as the entry's is. The walk is run over the module as built.
#[test]
fn each_texture_operation_that_stores_a_texel_counts() {
    let ops: [(&str, Option<TexOp>, bool); 13] = [
        ("air.write_texture_2d.v4f32", Some(TexOp::Write), true),
        (
            "air.write_texture_buffer_1d.v4f32",
            Some(TexOp::Write),
            true,
        ),
        (
            "air.write_imageblock_slice_to_texture_2d.v4f32",
            Some(TexOp::WriteImageblockSlice),
            true,
        ),
        (
            "air.atomic_fetch_add_explicit_texture_2d.u.v4i32",
            Some(TexOp::Atomic(AtomicOp::Add)),
            true,
        ),
        (
            "air.atomic_exchange_explicit_texture_2d.u.v4i32",
            Some(TexOp::Atomic(AtomicOp::Xchg)),
            true,
        ),
        (
            "air.atomic_store_explicit_texture_2d.u.v4i32",
            Some(TexOp::Atomic(AtomicOp::Store)),
            true,
        ),
        (
            "air.atomic_compare_exchange_weak_explicit_texture_2d.u.v4i32",
            Some(TexOp::Atomic(AtomicOp::CmpXchgWeak)),
            true,
        ),
        (
            "air.atomic_max_explicit_texture_2d.u.v4i64",
            Some(TexOp::Atomic(AtomicOp::Max)),
            true,
        ),
        (
            "air.atomic_load_explicit_texture_2d.u.v4i32",
            Some(TexOp::Atomic(AtomicOp::Load)),
            false,
        ),
        ("air.read_texture_2d.v4f32", Some(TexOp::Read), false),
        ("air.fence_texture_2d", Some(TexOp::Fence), false),
        ("air.get_width_texture_2d", None, false),
        ("air.wg.barrier", None, false),
    ];
    for (name, op, writes) in ops {
        let parsed = AirIntrinsic::parse(name).unwrap_or_else(|| panic!("{name}"));
        if let Some(op) = op {
            assert!(
                matches!(parsed, AirIntrinsic::Texture(t) if t.op == op),
                "{name}: {parsed:?}"
            );
        }
        let (module, ep) = module(vec![buffer(0, 0), buffer(1, 1)], |f| {
            let void = f.module().void();
            let ty = f.module().fn_ty(void, &[]);
            let decl = f.module().declare(name, ty);
            void_call(f, Callee::Air { decl, op: parsed }, &[]);
        });
        assert_eq!(calls(&module, &ep, writes_texture), writes, "{name}");
    }
    type Body = fn(&mut FunctionBuilder<'_>);
    let bodies: [(&str, Body, bool); 4] = [
        (
            "an indirect call",
            |f| {
                let void = f.module().void();
                let fn_ty = f.module().fn_ty(void, &[]);
                let callee = loaded_handle(f);
                void_call(f, Callee::Indirect { callee, fn_ty }, &[]);
            },
            true,
        ),
        (
            "a declaration",
            |f| {
                let void = f.module().void();
                let ty = f.module().fn_ty(void, &[]);
                let decl = f.module().declare("elsewhere", ty);
                void_call(f, Callee::Direct(decl), &[]);
            },
            true,
        ),
        (
            "a defined function that writes",
            |f| {
                let void = f.module().void();
                let dev = f.module().ptr(AddrSpace::Device);
                let ty = f.module().fn_ty(void, &[dev]);
                let mut g = f.module().define("helper", ty);
                write(&mut g, arg(0));
                g.ret(None);
                let helper = g.finish();
                void_call(f, Callee::Direct(helper), &[arg(0)]);
            },
            true,
        ),
        (
            "a defined function that does not",
            |f| {
                let void = f.module().void();
                let ty = f.module().fn_ty(void, &[]);
                let mut g = f.module().define("helper", ty);
                g.ret(None);
                let helper = g.finish();
                void_call(f, Callee::Direct(helper), &[]);
            },
            false,
        ),
    ];
    for (what, body, writes) in bodies {
        let (module, ep) = module(vec![buffer(0, 0), buffer(1, 1)], body);
        assert_eq!(calls(&module, &ep, writes_texture), writes, "{what}");
    }
}
