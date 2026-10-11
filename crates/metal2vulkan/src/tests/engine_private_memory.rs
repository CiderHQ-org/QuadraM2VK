//! A kernel's private memory against the device's Function storage.
//!
//! Owned synthetic kernels: `words` keeps a private array of `N` 32-bit words, written and read
//! at indices only the thread id decides (so it stays an array in memory); `lookup` reads a
//! constant table at such an index (held per thread as a `Private` variable); `fixed` keeps an
//! array of `N` words it reaches only at constant indices; `plain` keeps no private memory.
//!
//! - On a measured device (the RTX 5080: 32 KiB a thread) an array of exactly 32 KiB compiles
//!   with no note, and 64 bytes more is `Unsupported{PrivateMemory}` needing that much.
//! - On a device not measured (`amd-rdna3-radv`) the floor, 16 KiB, holds: an array of 16 KiB
//!   compiles and records `PrecisionNote::FunctionStorageAssumed`, as does the constant table;
//!   64 bytes more is refused needing it; a kernel without private memory records nothing.
//! - An array reached only at constant indices counts nothing: three times the RTX 5080's 32 KiB
//!   compiles there, and records no note on a device not measured.
//!
//! `llvm-as` assembles the kernels; a missing tool skips visibly.

use crate::device::VkLimit;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::testing::{compile, link_and_validate, profile, AssembleError};

/// The kernels, `words` with a private array of `n` words.
fn module(n: u32) -> Option<Vec<u8>> {
    let text = format!(
        r#"; Owned synthetic kernels (WP-UNI): private memory of {n} words, a constant table, none.
source_filename = "private_memory.metal"

@table = internal addrspace(2) constant [8 x i32] [i32 3, i32 1, i32 4, i32 1, i32 5, i32 9, i32 2, i32 6], align 4

define void @words(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = alloca [{n} x i32], align 4
  %w = urem i32 %gid, {n}
  %p = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %w
  store i32 %gid, ptr %p, align 4
  %r = mul i32 %gid, 7
  %i = urem i32 %r, {n}
  %q = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %i
  %x = load i32, ptr %q, align 4
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
}}

define void @lookup(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %i = and i32 %gid, 7
  %q = getelementptr inbounds [8 x i32], ptr addrspace(2) @table, i32 0, i32 %i
  %x = load i32, ptr addrspace(2) %q, align 4
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
}}

define void @fixed(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = alloca [{n} x i32], align 4
  %p0 = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 0
  store i32 %gid, ptr %p0, align 4
  %pl = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 {last}
  %g3 = mul i32 %gid, 3
  store i32 %g3, ptr %pl, align 4
  %x = load i32, ptr %p0, align 4
  %y = load i32, ptr %pl, align 4
  %s = add i32 %x, %y
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %s, ptr addrspace(1) %o, align 4
  ret void
}}

define void @plain(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %gid, ptr addrspace(1) %o, align 4
  ret void
}}

!air.kernel = !{{!0, !5, !6, !7}}
!0 = !{{ptr @words, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
!5 = !{{ptr @lookup, !1, !2}}
!6 = !{{ptr @plain, !1, !2}}
!7 = !{{ptr @fixed, !1, !2}}
"#,
        last = n - 1
    );
    match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernels: {why}"),
    }
}

/// `Unsupported{PrivateMemory}` needing `required` bytes a thread, of which the device holds
/// `available`.
fn needs(required: u64, available: u64) -> Result<(), Error> {
    Err(Error::Unsupported(Unsupported::needs_any_of(
        Feature::PrivateMemory,
        vec![vec![Capability::Limit {
            limit: VkLimit::FunctionStorageBytesPerThread,
            required,
            available,
        }]],
    )))
}

/// Kernel `entry` of [`module`]`(n)` on `stem`: its precision notes, or what refused it.
fn notes(stem: &str, n: u32, entry: &str) -> Option<Result<Vec<PrecisionNote>, Error>> {
    let bitcode = module(n)?;
    let p = profile(stem, Default::default()).expect("profile");
    Some(compile(&bitcode, entry, &p).map(|a| {
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry} {n}: L3: {e}"));
        a.precision.clone()
    }))
}

/// On the RTX 5080's measured 32 KiB a thread, 8192 words fit with no note; 8208 do not.
#[test]
fn private_memory_past_a_measured_device_is_refused() {
    let Some(fits) = notes("nvidia-rtx5080-laptop", 8192, "words") else {
        return;
    };
    let fits = fits.unwrap_or_else(|e| panic!("8192 words: {e}"));
    assert!(!fits.contains(&PrecisionNote::FunctionStorageAssumed));
    let past = notes("nvidia-rtx5080-laptop", 8208, "words").expect("assembled");
    assert_eq!(past.map(|_| ()), needs(8208 * 4, 32 * 1024));
}

/// On a device not measured the floor holds: 4096 words, and the constant table, fit and record
/// that they rest on it; 4112 words do not fit; a kernel without private memory records nothing.
#[test]
fn private_memory_on_a_device_not_measured_rests_on_the_floor() {
    let stem = "amd-rdna3-radv";
    let Some(fits) = notes(stem, 4096, "words") else {
        return;
    };
    let assumed = PrecisionNote::FunctionStorageAssumed;
    let fits = fits.unwrap_or_else(|e| panic!("4096 words: {e}"));
    assert!(fits.contains(&assumed), "{fits:?}");
    let table = notes(stem, 4096, "lookup").expect("assembled");
    let table = table.unwrap_or_else(|e| panic!("lookup: {e}"));
    assert!(table.contains(&assumed), "{table:?}");
    let plain = notes(stem, 4096, "plain").expect("assembled");
    let plain = plain.unwrap_or_else(|e| panic!("plain: {e}"));
    assert!(!plain.contains(&assumed), "{plain:?}");
    let past = notes(stem, 4112, "words").expect("assembled");
    assert_eq!(past.map(|_| ()), needs(4112 * 4, 16 * 1024));
}

/// An array reached only at constant indices counts nothing: a driver holds its parts in registers.
/// Three times the RTX 5080's measured 32 KiB compiles there, and on a device not measured it
/// records no note; the same array indexed at run time is refused.
#[test]
fn an_array_reached_only_at_constant_indices_counts_nothing() {
    let n = 3 * 8192;
    let Some(fits) = notes("nvidia-rtx5080-laptop", n, "fixed") else {
        return;
    };
    let fits = fits.unwrap_or_else(|e| panic!("{n} words at constant indices: {e}"));
    assert!(
        !fits.contains(&PrecisionNote::FunctionStorageAssumed),
        "{fits:?}"
    );
    let floor = notes("amd-rdna3-radv", n, "fixed").expect("assembled");
    let floor = floor.unwrap_or_else(|e| panic!("on the floor: {e}"));
    assert!(
        !floor.contains(&PrecisionNote::FunctionStorageAssumed),
        "{floor:?}"
    );
    let indexed = notes("nvidia-rtx5080-laptop", n, "words").expect("assembled");
    assert_eq!(indexed.map(|_| ()), needs(u64::from(n) * 4, 32 * 1024));
}
