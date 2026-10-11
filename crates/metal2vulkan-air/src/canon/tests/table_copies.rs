//! A texture table copied into thread memory by `llvm.memcpy`: a copy of whole handles of the table
//! becomes the handle copies it is, a load of each from the table and a store of it at the same
//! distance from the copy's start; any other copy stays.

use super::*;
use crate::abi::TextureArg;
use crate::intrinsics::{Elem, LlvmIntrinsic};

/// What [`table_copy`]'s kernel copies from its three-texture table `ts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Copied {
    /// All 24 bytes, from `ts` itself.
    Whole,
    /// The last two handles, from `&ts[1]`.
    Tail,
    /// 12 bytes: one handle and half of the next.
    Partial,
    /// Two handles from `&ts[2]`: past the table's end.
    Past,
}

/// `kernel k(array<texture2d<float>, 3> ts [[texture(0)]], device uint* out)` copying from `ts`
/// as `what` says into a private `texture2d<float> arr[3]` at its start, then storing
/// `arr[1].get_width()` to `out`.
fn table_copy(what: Copied) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let (i1, i32t, i64t) = (m.int(1), m.int(32), m.int(64));
    let (dev, thread) = (m.ptr(AddrSpace::Device), m.ptr(AddrSpace::Thread));
    let table = m.array(dev, 3);
    let memcpy_ty = m.fn_ty(void, &[thread, thread, i64t, i1]);
    let memcpy = m.declare("llvm.memcpy.p0.p0.i64", memcpy_ty);
    let width_ty = m.fn_ty(i32t, &[dev, i32t]);
    let width_name = "air.get_width_texture_2d";
    let width = m.declare(width_name, width_ty);
    let width_op = AirIntrinsic::parse(width_name).expect("a width query parses");
    let fty = m.fn_ty(void, &[thread, dev]);
    let k = |m: &mut ModuleBuilder, ty: TypeId, v: u64| m.const_int(ty, v);
    let single = k(&mut m, i32t, 1);
    let (zero, one, two) = (k(&mut m, i64t, 0), k(&mut m, i64t, 1), k(&mut m, i64t, 2));
    let (len, lod, no) = (
        k(
            &mut m,
            i64t,
            match what {
                Copied::Whole => 24,
                Copied::Tail | Copied::Past => 16,
                Copied::Partial => 12,
            },
        ),
        k(&mut m, i32t, 0),
        m.const_int(i1, 0),
    );
    let mut f = m.define("k", fty);
    let gep = |f: &mut FunctionBuilder<'_>, base: Value, at: Value| {
        f.push(
            thread,
            InstKind::Gep {
                src_ty: table,
                base,
                indices: Box::new([zero, at]),
                inbounds: true,
            },
        )
    };
    let arr = f.push(
        thread,
        InstKind::Alloca {
            ty: table,
            count: single,
            align: None,
        },
    );
    let src = match what {
        Copied::Whole | Copied::Partial => arg(0),
        Copied::Tail => gep(&mut f, arg(0), one),
        Copied::Past => gep(&mut f, arg(0), two),
    };
    let op = LlvmIntrinsic::Memcpy {
        dst: AddrSpace::Thread,
        src: AddrSpace::Thread,
        len: Elem::I64,
        volatile: false,
    };
    call(
        &mut f,
        void,
        Callee::Llvm { decl: memcpy, op },
        &[arr, src, len, no],
    );
    let at = gep(&mut f, arr, one);
    let h = load(&mut f, dev, at);
    let w = call(
        &mut f,
        i32t,
        Callee::Air {
            decl: width,
            op: width_op,
        },
        &[h, lod],
    );
    store(&mut f, w, arg(1));
    f.ret(None);
    let kernel_fn = f.finish();
    let module = m.finish().unwrap();
    let ts = ParamBinding {
        role: ParamRole::Texture(TextureArg {
            location: Slot {
                index: AbiInt::Const(0),
                count: AbiInt::Const(3),
            },
            access: Access::Read,
            raster_order_group: None,
        }),
        name: Some(DiagName::new("ts")),
        ..buffer(0, 0)
    };
    let abi = abi(vec![kernel(kernel_fn, vec![ts, buffer(1, 0)])], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// A copy of whole handles of a texture table into thread memory is copied handle by handle: no
/// `memcpy` stays, and the handle the kernel queries is loaded from the table. What nothing reads
/// (the other handles' stores) goes.
#[test]
fn a_texture_table_copied_whole_handles_is_copied_handle_by_handle() {
    for what in [Copied::Whole, Copied::Tail] {
        let e = table_copy(what).unwrap();
        let text = body_text(&e);
        assert!(!text.contains("Memcpy"), "{what:?}: {text}");
        // The handle loaded from the table, then loaded back from `arr`.
        let loads = text.matches("load ptr addrspace(1)").count();
        assert_eq!(loads, 2, "{what:?}: {text}");
        assert!(
            text.contains("air.get_width_texture_2d"),
            "{what:?}: {text}"
        );
    }
}

/// A copy of part of a handle, or past the table's end, is no copy of handles: it stays as it was.
#[test]
fn a_texture_table_copy_of_partial_or_missing_handles_stays() {
    for what in [Copied::Partial, Copied::Past] {
        let e = table_copy(what).unwrap();
        let text = body_text(&e);
        assert!(text.contains("Memcpy"), "{what:?}: {text}");
    }
}
