//! `widening`: a load of `half`s takes the device's widening of its bits where the profile's
//! widening is Metal's (`DriverDeviation::MetalHalfArithmetic`) and no use of the value reads a
//! NaN's bits, and keeps the integer decode elsewhere.

use crate::device::requirements::VkProperty;
use crate::device::DevicePolicy;
use crate::lower::arith::narrow::widens_bits;
use crate::pipeline::testing::{
    compile, llvm_assemble, meter, profile, typed_module, AssembleError,
};
use metal2vulkan_devtools::llvm_assemble_text;

/// Kernels that load the `half` of their thread from a device buffer (`%y`, a load of `half`s) and
/// use it: in a sum (`k_sum`), through a select into a product (`k_select_sum`), widened to `float`
/// (`k_ext`), compared (`k_cmp`), stored as it is (`k_move`), through a select stored
/// (`k_select_move`), and negated (`k_neg`); and two that read it back through threadgroup words
/// (an `i32` array read and written as `half`s, so held as words), into a sum (`k_words_sum`) and
/// stored as it is (`k_words_move`).
const KERNELS: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@w = internal addrspace(3) global [128 x i32] undef, align 4

define void @k_sum(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %y = load half, ptr addrspace(1) %ps, align 2
  %z = fadd half %y, 0xH3C00
  %po = getelementptr inbounds half, ptr addrspace(1) %out, i32 %g
  store half %z, ptr addrspace(1) %po, align 2
  ret void
}

define void @k_select_sum(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %y = load half, ptr addrspace(1) %ps, align 2
  %f = load i32, ptr addrspace(1) %flags, align 4
  %c = icmp eq i32 %f, 0
  %x = select i1 %c, half %y, half 0xH3C00
  %z = fmul half %x, 0xH3800
  %po = getelementptr inbounds half, ptr addrspace(1) %out, i32 %g
  store half %z, ptr addrspace(1) %po, align 2
  ret void
}

define void @k_ext(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %y = load half, ptr addrspace(1) %ps, align 2
  %e = fpext half %y to float
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %e, ptr addrspace(1) %po, align 4
  ret void
}

define void @k_cmp(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %y = load half, ptr addrspace(1) %ps, align 2
  %c = fcmp olt half %y, 0xH3C00
  %v = zext i1 %c to i32
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %g
  store i32 %v, ptr addrspace(1) %po, align 4
  ret void
}

define void @k_move(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %y = load half, ptr addrspace(1) %ps, align 2
  %po = getelementptr inbounds half, ptr addrspace(1) %out, i32 %g
  store half %y, ptr addrspace(1) %po, align 2
  ret void
}

define void @k_select_move(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %y = load half, ptr addrspace(1) %ps, align 2
  %f = load i32, ptr addrspace(1) %flags, align 4
  %c = icmp eq i32 %f, 0
  %x = select i1 %c, half %y, half 0xH3C00
  %po = getelementptr inbounds half, ptr addrspace(1) %out, i32 %g
  store half %x, ptr addrspace(1) %po, align 2
  ret void
}

define void @k_neg(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %y = load half, ptr addrspace(1) %ps, align 2
  %n = fneg half %y
  %po = getelementptr inbounds half, ptr addrspace(1) %out, i32 %g
  store half %n, ptr addrspace(1) %po, align 2
  ret void
}

define void @k_words_sum(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %a = load half, ptr addrspace(1) %ps, align 2
  %t2 = shl i32 %t, 1
  %pt = getelementptr inbounds i8, ptr addrspace(3) @w, i32 %t2
  store half %a, ptr addrspace(3) %pt, align 2
  tail call void @air.wg.barrier(i32 2, i32 1)
  %r = sub i32 255, %t
  %r2 = shl i32 %r, 1
  %pr = getelementptr inbounds i8, ptr addrspace(3) @w, i32 %r2
  %y = load half, ptr addrspace(3) %pr, align 2
  %z = fadd half %y, 0xH3C00
  %po = getelementptr inbounds half, ptr addrspace(1) %out, i32 %g
  store half %z, ptr addrspace(1) %po, align 2
  ret void
}

define void @k_words_move(ptr addrspace(1) %src, ptr addrspace(1) %out, ptr addrspace(1) %flags, i32 %g, i32 %t) {
  %ps = getelementptr inbounds half, ptr addrspace(1) %src, i32 %g
  %a = load half, ptr addrspace(1) %ps, align 2
  %t2 = shl i32 %t, 1
  %pt = getelementptr inbounds i8, ptr addrspace(3) @w, i32 %t2
  store half %a, ptr addrspace(3) %pt, align 2
  tail call void @air.wg.barrier(i32 2, i32 1)
  %r = sub i32 255, %t
  %r2 = shl i32 %r, 1
  %pr = getelementptr inbounds i8, ptr addrspace(3) @w, i32 %r2
  %y = load half, ptr addrspace(3) %pr, align 2
  %po = getelementptr inbounds half, ptr addrspace(1) %out, i32 %g
  store half %y, ptr addrspace(1) %po, align 2
  ret void
}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!10, !11, !12, !13, !14, !15, !16, !17, !18}
!10 = !{ptr @k_sum, !1, !2}
!11 = !{ptr @k_select_sum, !1, !2}
!12 = !{ptr @k_ext, !1, !2}
!13 = !{ptr @k_cmp, !1, !2}
!14 = !{ptr @k_move, !1, !2}
!15 = !{ptr @k_select_move, !1, !2}
!16 = !{ptr @k_neg, !1, !2}
!17 = !{ptr @k_words_sum, !1, !2}
!18 = !{ptr @k_words_move, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"flags"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!8 = !{i32 4, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
"#;

/// The kernels, and whether the load `%y` of each takes the device's widening where the profile's
/// widening is Metal's.
const CASES: [(&str, bool); 9] = [
    ("k_sum", true),
    ("k_select_sum", true),
    ("k_ext", true),
    ("k_cmp", true),
    ("k_move", false),
    ("k_select_move", false),
    ("k_neg", false),
    ("k_words_sum", true),
    ("k_words_move", false),
];

/// On MoltenVK (the M3 Max and the M4 profiles, which take `MetalHalfArithmetic`), the load `%y`
/// of each kernel whose uses read no NaN's bits (a sum or product, through a select or not, a
/// widening to `float`, a compare), from a device buffer or threadgroup words, is the device's
/// widening of its bits, and the module requires the fact; a load stored as it is, through a
/// select or negated, keeps the integer decode, as the device load stored to threadgroup words
/// does. On the other profiles no load takes it by this rule.
#[test]
fn a_half_load_whose_nans_no_use_reads_is_the_devices_widening_where_it_is_metals() {
    let bc = match llvm_assemble_text(KERNELS) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernels: {why}"),
    };
    for stem in [
        "moltenvk-m3",
        "moltenvk-m4",
        "lavapipe",
        "nvidia-rtx5080-laptop",
    ] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let metal = stem.starts_with("moltenvk");
        for (entry, widened) in CASES {
            let what = format!("{stem} {entry}");
            let (plan, module) =
                typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let expect = metal && widened;
            assert_eq!(
                plan.families.arith.widened.len(),
                usize::from(expect),
                "{what}: {:?}",
                plan.families.arith.widened
            );
            // Elsewhere a profile's own rules may widen every `half` (the float controls).
            if metal {
                let mut m = meter();
                let widens = widens_bits(&module, &mut m).expect("widens_bits");
                assert_eq!(widens, expect, "{what}: the module's widening");
            }
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            if expect {
                assert!(
                    a.requirements
                        .properties
                        .contains(&VkProperty::MetalHalfArithmetic),
                    "{what}: {:?}",
                    a.requirements.properties
                );
            }
        }
    }
}

/// The owned fixture of the device lane
/// `driver_deviations::half_loads_the_device_widens_answer_as_metal` (`hwd_half_reads`, as Metal's
/// compiler builds it): on MoltenVK its five reads used only in a sum, a widening, a compare or a
/// product take the device's widening (the mirror's pattern from threadgroup words, the pattern
/// `g ^ 1` from the buffer, and the constants `-0` and `1` of the sum and compare and the one `1`
/// of the product's branch); the read moved through a select to memory, the `1` the other branch
/// both moves and multiplies, and the pattern stored to threadgroup memory keep the integer decode;
/// on lavapipe none takes it. The lane runs the fixture on the device.
#[test]
fn the_lane_fixture_widens_its_reads_used_in_arithmetic_alone() {
    let ll = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/hwd_half_reads.ll");
    let bc = match llvm_assemble(&ll) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected hwd_half_reads: {why}"),
    };
    for (stem, widened) in [("moltenvk-m3", 5), ("moltenvk-m4", 5), ("lavapipe", 0)] {
        let p = profile(stem, DevicePolicy::default()).expect("profile");
        let (plan, module) =
            typed_module(&bc, "hwd_half_reads", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(plan.families.arith.widened.len(), widened, "{stem}");
        if widened > 0 {
            let mut m = meter();
            assert!(widens_bits(&module, &mut m).expect("widens_bits"), "{stem}");
        }
    }
}
