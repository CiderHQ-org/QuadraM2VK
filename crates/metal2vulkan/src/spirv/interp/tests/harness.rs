//! The OOB-trap harness: the uncontained access traps exactly outside the normative bounds, the
//! normative guard contains every probe, and a wrapping `off + n <= size` guard is caught.

use super::super::harness::{
    check_containment, checked_access, expected, in_bounds, int_const, probe_module, probes,
    raw_access, run_probe, AccessArgs, OffsetWidth, ProbeOutcome, ACCESS_SIZES,
};
use super::*;

#[test]
fn the_normative_check_does_not_wrap() {
    assert!(in_bounds(16, 12, 4));
    assert!(!in_bounds(16, 13, 4));
    assert!(!in_bounds(2, 0, 4));
    assert!(in_bounds(0, 0, 0));
    // `off + n` would wrap to 3 in 32 bits and pass a naive check.
    assert!(!in_bounds(16, 0xFFFF_FFFF, 4));
    assert!(!in_bounds(u64::MAX, u64::MAX, 1));
    assert!(in_bounds(u64::MAX, u64::MAX - 1, 1));
    let ps = probes(16);
    assert!(ps
        .iter()
        .any(|p| p.offset == 0xFFFF_FFFF && p.width == OffsetWidth::U32));
    assert!(ps
        .iter()
        .any(|p| p.offset == u64::MAX && p.width == OffsetWidth::U64));
    assert!(ps
        .iter()
        .all(|p| p.width == OffsetWidth::U64 || p.offset <= u64::from(u32::MAX)));
}

#[test]
fn raw_accesses_trap_exactly_out_of_bounds() {
    let size = 20;
    let all = probes(size);
    for width in [OffsetWidth::U32, OffsetWidth::U64] {
        for n in ACCESS_SIZES {
            let m = probe_module(n, width, &raw_access).expect("probe module");
            for p in all.iter().filter(|p| p.n == n && p.width == width) {
                let got = run_probe(&m, size, p).expect("run");
                if in_bounds(size, p.offset, u64::from(n)) {
                    assert_eq!(got, ProbeOutcome::Loaded(expected(size, p)), "{p:?}");
                } else {
                    assert!(
                        matches!(
                            got,
                            ProbeOutcome::Trapped(
                                TrapKind::OutOfBounds { .. } | TrapKind::AddressOverflow
                            )
                        ),
                        "{p:?}: {got:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_normative_guard_contains_every_probe() {
    for size in [0, 1, 5, 16, 64] {
        check_containment(&checked_access, size).expect("contained");
    }
}

/// `off + n <= size` computed in the offset's width: wraps near the top of the range.
fn naive_access(fb: &mut FunctionBuilder<'_>, a: &AccessArgs) -> BResult<Value> {
    let n = int_const(fb, a, u64::from(a.n))?;
    let end = fb.binary(BinOp::IAdd, a.offset, n)?;
    let inb = fb.binary(BinOp::ULessThanEqual, end, a.size)?;
    let null = fb.module().constant(Constant::Null { ty: a.elem })?;
    let var = fb.local_var(a.elem, Some(null))?;
    let result = fb.var_ptr(var)?;
    let args = *a;
    let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let v = raw_access(fb, &args)?;
        fb.store(result, v, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), inb, then, None)?;
    fb.load(result, MemAccess::default())
}

#[test]
fn a_wrapping_guard_is_caught() {
    let err = check_containment(&naive_access, 16).expect_err("the naive guard wraps");
    // Every failure is an escaped access at a wrapped offset, and 32-bit offsets fail.
    assert!(err.contains("width: U32"), "{err}");
    for line in err.lines() {
        assert!(line.contains("Trapped"), "{line}");
    }
}
