//! Loads SPIRV-Cross forwards past a write it does not see. Where the profile takes
//! `DriverWorkaround::LocalizedLoadsForwardedPastAtomicsAndCalls` (MoltenVK 1.4.1, whose
//! SPIRV-Cross declares `Workgroup` and `Private` variables in the entry point, so an atomic on
//! one, or a call that writes one, does not stop a load from it being forwarded past the write),
//! `serialize_for` materializes each load from such a variable in a function holding such a
//! write. A load in a function holding none is written as before, and every other profile writes
//! the module as `serialize` does. Through the MSL `spirv-cross` writes with MoltenVK's options,
//! the store then reads the values loaded before the atomic and the call, where without the
//! materialization it reads the variables after them.

use super::loop_reads::{copied, msl_of};
use super::{compute, cval, elem_ptr, finish_compute, gid_x, int, scalar, ssbo};
use crate::device::profile::DriverWorkaround;
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ir::{
    AtomicOp, BinOp, MemAccess, Scope, Semantics, StorageClass, Type, TypedModule,
};
use crate::spirv::serialize::{relocate, serialize};
use crate::spirv::testing::{assert_valid, committed_profiles, is_moltenvk_capture, link_dense};
use crate::spv::Op;

/// The `Private` variable's initial value, which names it in the MSL (`uint _N = 24301u;`).
const SEED: u64 = 0x5eed;

/// `a = w` (a `Workgroup` word), then `w += 10` atomically; `c = p` (a `Private` word, initially
/// [`SEED`]), then a call to `bump`, which stores 15 to `p`; a call to `record`, which stores `w`
/// to `out[1]` (a `Workgroup` load in a function with no atomic and no call); then `out[gid] =
/// a + c`, which must read the values loaded before the atomic and the call.
fn kernel() -> TypedModule {
    let mut b = ModuleBuilder::new();
    // Only these types: every committed profile must accept the module.
    let u32_ = int(&mut b, 32, false);
    let out = ssbo(&mut b, 0, u32_, 4);
    let w = b
        .plain_global(StorageClass::Workgroup, u32_, None)
        .expect("w");
    let seed = scalar(&mut b, u32_, SEED);
    let p = b
        .plain_global(StorageClass::Private, u32_, Some(seed))
        .expect("p");
    let void = b.ty(Type::Void).expect("void");
    let bump = {
        let mut f = b.routine(void).expect("bump");
        let pp = f.module().global_value(p).expect("p");
        let fifteen = cval(&mut f, u32_, 15);
        f.store(pp, fifteen, MemAccess::default()).expect("p = 15");
        f.ret(None).expect("return");
        f.finish().expect("bump")
    };
    let record = {
        let mut f = b.routine(void).expect("record");
        let wp = f.module().global_value(w).expect("w");
        let v = f.load(wp, MemAccess::default()).expect("w");
        let one = cval(&mut f, u32_, 1);
        let at = elem_ptr(&mut f, out, one);
        f.store(at, v, MemAccess::default()).expect("out[1] = w");
        f.ret(None).expect("return");
        f.finish().expect("record")
    };
    let main = compute(&mut b, |f| {
        let i = gid_x(f);
        let wp = f.module().global_value(w).expect("w");
        let pp = f.module().global_value(p).expect("p");
        let a = f.load(wp, MemAccess::default()).expect("a = w");
        let ten = cval(f, u32_, 10);
        f.atomic(
            AtomicOp::IAdd,
            wp,
            Scope::Workgroup,
            Semantics::RELAXED,
            ten,
        )
        .expect("w += 10");
        let c = f.load(pp, MemAccess::default()).expect("c = p");
        f.call(bump, &[]).expect("bump()");
        f.call(record, &[]).expect("record()");
        let s = f.binary(BinOp::IAdd, a, c).expect("a + c");
        let at = elem_ptr(f, out, i);
        f.store(at, s, MemAccess::default())
            .expect("out[gid] = a + c");
    });
    finish_compute(b, main)
}

/// Only the MoltenVK captures take the workaround; there the entry's loads of `w` (its function
/// holds an atomic on `w`) and of `p` (it calls `bump`, which writes `p`) are materialized, each
/// written under its own id and copied, and `record`'s load of `w` (no atomic, no call) is not.
/// Every other profile writes no copy. Every form passes L3.
#[test]
fn a_load_past_an_atomic_or_a_call_that_writes_its_memory_is_materialized_on_moltenvk() {
    let profiles = committed_profiles().expect("committed profiles");
    assert!(profiles.iter().any(|(stem, _)| is_moltenvk_capture(stem)));
    for (stem, profile) in &profiles {
        let takes = profile
            .fields()
            .workarounds
            .contains(&DriverWorkaround::LocalizedLoadsForwardedPastAtomicsAndCalls);
        assert_eq!(takes, is_moltenvk_capture(stem), "{stem}");
        let words = assert_valid(kernel(), profile);
        let expected = if takes {
            vec![Op::Load, Op::Load]
        } else {
            Vec::new()
        };
        assert_eq!(copied(&words), expected, "{stem}");
    }
}

/// The identifiers the right-hand side of `msl`'s last store to a buffer element in its kernel
/// reads, and the names of the kernel's threadgroup word and of its private word (declared with
/// [`SEED`]).
fn store_reads(msl: &str) -> (Vec<String>, String, String) {
    let (_, kernel) = msl.split_once("kernel void").expect("a kernel");
    let lines: Vec<&str> = kernel.lines().map(str::trim).collect();
    let name_of = |prefix: &str, suffix: &str| {
        lines
            .iter()
            .find_map(|l| l.strip_prefix(prefix).and_then(|r| r.strip_suffix(suffix)))
            .unwrap_or_else(|| panic!("`{prefix}…{suffix}` in the kernel:\n{msl}"))
            .to_owned()
    };
    let threadgroup = name_of("threadgroup uint ", ";");
    let private = name_of("uint ", &format!(" = {SEED}u;"));
    let rhs = lines
        .iter()
        .rev()
        .filter_map(|l| l.split_once(" = "))
        .find(|(lhs, _)| lhs.contains("._m0["))
        .map(|(_, rhs)| rhs)
        .unwrap_or_else(|| panic!("a store to a buffer element:\n{msl}"));
    let reads = rhs
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect();
    (reads, threadgroup, private)
}

/// Through the MSL `spirv-cross` writes with MoltenVK's options: without the materialization the
/// store reads the threadgroup word after the atomic and the private word after the call (both
/// forwarded past them); with it, the store reads neither variable. `spirv-cross` is external;
/// without it the test skips visibly.
#[test]
fn moltenvk_msl_reads_no_localized_variable_past_an_atomic_or_a_call() {
    let profiles = committed_profiles().expect("committed profiles");
    let (_, moltenvk) = profiles
        .iter()
        .find(|(stem, _)| is_moltenvk_capture(stem))
        .expect("a MoltenVK capture");
    let materialized = assert_valid(kernel(), moltenvk);
    let plain = serialize(kernel()).expect("serialize");
    let plain = relocate(&plain, &link_dense(&plain)).expect("relocate");
    let (Some(materialized), Some(plain)) = (msl_of(&materialized), msl_of(&plain)) else {
        return;
    };
    let (reads, threadgroup, private) = store_reads(&materialized);
    assert!(
        !reads.contains(&threadgroup) && !reads.contains(&private),
        "{reads:?} reads {threadgroup} or {private}\n{materialized}"
    );
    let (reads, threadgroup, private) = store_reads(&plain);
    assert!(
        reads.contains(&threadgroup) && reads.contains(&private),
        "spirv-cross no longer forwards these loads past the atomic and the call: re-measure \
         whether MoltenVK needs `LocalizedLoadsForwardedPastAtomicsAndCalls` ({reads:?})\n{plain}"
    );
}
