//! A settled sum no device can combine with a neighbour is left as written (`lower::arith::reassoc`,
//! its module docs' **Alone**), so the lowering leaves it undecorated; every other settled sum keeps
//! `NoContraction`.
//!
//! One module holds the forms, each a kernel (or a fragment) of two-leaf `fast` sums:
//!
//! - left as written: a sum of a buffer's lanes stored to device memory (`stored`) or to
//!   threadgroup memory (`shared`), a difference converted to an integer (`converted`), and sums
//!   of a buffer's lanes and a constant buffer's that are texture coordinates, whole, lane by lane
//!   through insertions, and with a negated lane (`(d.x, -d.y) + uv`), with sums of the samples
//!   stored (`sampled`); a fragment's coordinates `uv + d` and `(d.x, -d.y) + uv` (`bg`);
//! - settled: each twin of the first, its sum given one more use that a device may combine with it,
//!   a comparison (`stored_compared`, and a lane of `sampled_compared`'s vector coordinate) or a
//!   product (`converted_multiplied`); a sum stored to thread memory and read back
//!   (`stored_private`), one whose operand is read back from thread memory (`from_private`), a sum
//!   of `(v.x, -v.x)` and `v`, whose operands read one value and a negation of it
//!   (`negated_beside_its_source`), and the fragment's sum of its two samples, which it returns
//!   (`bg`).
//!
//! `llvm-as` assembles the module; without it the tests skip visibly.

use std::collections::BTreeMap;

use crate::air::ir::{BinOp as AirBin, FastMath, InstKind, Type};
use crate::device::profile::DeviceProfile;
use crate::error::Error;
use crate::library::FunctionConstantValues;
use crate::lower::select::Plan;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, meter, specialized_entry, typed_module,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::committed_profiles;

const MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@tg = internal addrspace(3) global [64 x float] undef, align 4

define void @stored(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fadd fast float %a, %b
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %s, ptr addrspace(1) %po, align 4
  ret void
}

define void @stored_compared(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fadd fast float %a, %b
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %s, ptr addrspace(1) %po, align 4
  %lt = fcmp fast olt float %s, 0.000000e+00
  %z = zext i1 %lt to i32
  %pz = getelementptr inbounds i32, ptr addrspace(1) %aux, i32 %g
  store i32 %z, ptr addrspace(1) %pz, align 4
  ret void
}

define void @converted(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fsub fast float %a, %b
  %i = fptosi float %s to i32
  %pz = getelementptr inbounds i32, ptr addrspace(1) %aux, i32 %g
  store i32 %i, ptr addrspace(1) %pz, align 4
  ret void
}

define void @converted_multiplied(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fsub fast float %a, %b
  %i = fptosi float %s to i32
  %pz = getelementptr inbounds i32, ptr addrspace(1) %aux, i32 %g
  store i32 %i, ptr addrspace(1) %pz, align 4
  %p = fmul fast float %s, %a
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %p, ptr addrspace(1) %po, align 4
  ret void
}

define void @stored_private(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %m = alloca [2 x float], align 4
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fadd fast float %a, %b
  %j = and i32 %g, 1
  %pm = getelementptr inbounds [2 x float], ptr %m, i32 0, i32 %j
  store float %s, ptr %pm, align 4
  %r = load float, ptr %pm, align 4
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

define void @from_private(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %m = alloca [2 x float], align 4
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %j = and i32 %g, 1
  %pm = getelementptr inbounds [2 x float], ptr %m, i32 0, i32 %j
  store float %a, ptr %pm, align 4
  %r = load float, ptr %pm, align 4
  %s = fadd fast float %r, %b
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %s, ptr addrspace(1) %po, align 4
  ret void
}

define void @shared(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g, i32 %t) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %b = extractelement <2 x float> %v, i64 1
  %s = fadd fast float %a, %b
  %pt = getelementptr inbounds [64 x float], ptr addrspace(3) @tg, i32 0, i32 %t
  store float %s, ptr addrspace(3) %pt, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %u = sub i32 63, %t
  %pu = getelementptr inbounds [64 x float], ptr addrspace(3) @tg, i32 0, i32 %u
  %r = load float, ptr addrspace(3) %pu, align 4
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

define void @sampled(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, ptr addrspace(2) %k, ptr addrspace(1) %tex, ptr addrspace(2) %smp, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %uv = load <2 x float>, ptr addrspace(1) %pi, align 8
  %d = load <2 x float>, ptr addrspace(2) %k, align 8
  %c = fadd fast <2 x float> %uv, %d
  %a = extractelement <2 x float> %uv, i64 0
  %b = extractelement <2 x float> %d, i64 1
  %x = fadd fast float %a, %b
  %y = fsub fast float %a, %b
  %c0 = insertelement <2 x float> poison, float %x, i64 0
  %c1 = insertelement <2 x float> %c0, float %y, i64 1
  %r0 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v0 = extractvalue { <4 x float>, i8 } %r0, 0
  %r1 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c1, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v1 = extractvalue { <4 x float>, i8 } %r1, 0
  %dx = extractelement <2 x float> %d, i64 0
  %dy = extractelement <2 x float> %d, i64 1
  %ny = fneg fast float %dy
  %n0 = insertelement <2 x float> poison, float %dx, i64 0
  %dn = insertelement <2 x float> %n0, float %ny, i64 1
  %c2 = fadd fast <2 x float> %dn, %uv
  %r2 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c2, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v2 = extractvalue { <4 x float>, i8 } %r2, 0
  %w = fadd fast <4 x float> %v0, %v1
  %w2 = fadd fast <4 x float> %v2, %v1
  %g2 = shl i32 %g, 1
  %po = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g2
  store <4 x float> %w, ptr addrspace(1) %po, align 16
  %g21 = or i32 %g2, 1
  %po2 = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g21
  store <4 x float> %w2, ptr addrspace(1) %po2, align 16
  ret void
}

define void @sampled_compared(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, ptr addrspace(2) %k, ptr addrspace(1) %tex, ptr addrspace(2) %smp, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %uv = load <2 x float>, ptr addrspace(1) %pi, align 8
  %d = load <2 x float>, ptr addrspace(2) %k, align 8
  %c = fadd fast <2 x float> %uv, %d
  %a = extractelement <2 x float> %uv, i64 0
  %b = extractelement <2 x float> %d, i64 1
  %x = fadd fast float %a, %b
  %y = fsub fast float %a, %b
  %c0 = insertelement <2 x float> poison, float %x, i64 0
  %c1 = insertelement <2 x float> %c0, float %y, i64 1
  %r0 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v0 = extractvalue { <4 x float>, i8 } %r0, 0
  %r1 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c1, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v1 = extractvalue { <4 x float>, i8 } %r1, 0
  %dx = extractelement <2 x float> %d, i64 0
  %dy = extractelement <2 x float> %d, i64 1
  %ny = fneg fast float %dy
  %n0 = insertelement <2 x float> poison, float %dx, i64 0
  %dn = insertelement <2 x float> %n0, float %ny, i64 1
  %c2 = fadd fast <2 x float> %dn, %uv
  %r2 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c2, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v2 = extractvalue { <4 x float>, i8 } %r2, 0
  %w = fadd fast <4 x float> %v0, %v1
  %w2 = fadd fast <4 x float> %v2, %v1
  %g2 = shl i32 %g, 1
  %po = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g2
  store <4 x float> %w, ptr addrspace(1) %po, align 16
  %g21 = or i32 %g2, 1
  %po2 = getelementptr inbounds <4 x float>, ptr addrspace(1) %out, i32 %g21
  store <4 x float> %w2, ptr addrspace(1) %po2, align 16
  %cx = extractelement <2 x float> %c, i64 0
  %lt = fcmp fast olt float %cx, 5.000000e-01
  %z = zext i1 %lt to i32
  %pz = getelementptr inbounds i32, ptr addrspace(1) %aux, i32 %g
  store i32 %z, ptr addrspace(1) %pz, align 4
  ret void
}

define <4 x float> @bg(<2 x float> %uv, ptr addrspace(2) %k, ptr addrspace(1) %tex, ptr addrspace(2) %smp) {
  %d = load <2 x float>, ptr addrspace(2) %k, align 8
  %c0 = fadd fast <2 x float> %uv, %d
  %dx = extractelement <2 x float> %d, i64 0
  %dy = extractelement <2 x float> %d, i64 1
  %ny = fneg fast float %dy
  %n0 = insertelement <2 x float> poison, float %dx, i64 0
  %dn = insertelement <2 x float> %n0, float %ny, i64 1
  %c1 = fadd fast <2 x float> %dn, %uv
  %r0 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c0, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v0 = extractvalue { <4 x float>, i8 } %r0, 0
  %r1 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %smp, <2 x float> %c1, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %v1 = extractvalue { <4 x float>, i8 } %r1, 0
  %w = fadd fast <4 x float> %v0, %v1
  ret <4 x float> %w
}

define void @negated_beside_its_source(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %aux, i32 %g) {
  %pi = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <2 x float>, ptr addrspace(1) %pi, align 8
  %a = extractelement <2 x float> %v, i64 0
  %na = fneg fast float %a
  %m = insertelement <2 x float> %v, float %na, i64 1
  %s = fadd fast <2 x float> %m, %v
  %g2 = shl i32 %g, 1
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g2
  store <2 x float> %s, ptr addrspace(1) %po, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!20, !21, !22, !23, !24, !25, !26, !27, !28, !29}
!air.fragment = !{!40}
!air.version = !{!1}
!air.language_version = !{!2}

!1 = !{i32 2, i32 8, i32 0}
!2 = !{!"Metal", i32 4, i32 0, i32 0}
!3 = !{}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"in"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!6 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"aux"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!8 = !{i32 4, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!9 = !{!4, !5, !6, !7}
!10 = !{!4, !5, !6, !7, !8}
!11 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"k"}
!12 = !{i32 4, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!13 = !{i32 5, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
!14 = !{i32 6, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!15 = !{!4, !16, !6, !11, !12, !13, !14}
!16 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!20 = !{ptr @stored, !3, !9}
!21 = !{ptr @stored_compared, !3, !9}
!22 = !{ptr @converted, !3, !9}
!23 = !{ptr @converted_multiplied, !3, !9}
!24 = !{ptr @stored_private, !3, !9}
!25 = !{ptr @from_private, !3, !9}
!26 = !{ptr @shared, !3, !10}
!27 = !{ptr @sampled, !3, !15}
!28 = !{ptr @sampled_compared, !3, !15}
!29 = !{ptr @negated_beside_its_source, !3, !9}
!40 = !{ptr @bg, !41, !43}
!41 = !{!42}
!42 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!43 = !{!44, !45, !46, !47}
!44 = !{i32 0, !"air.fragment_input", !"generated(2uvDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!45 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"k"}
!46 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!47 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"smp"}
"#;

/// Each form: its entry, and its two-leaf `float` sums left as written and settled.
const FORMS: [(&str, usize, usize); 11] = [
    ("stored", 1, 0),
    ("stored_compared", 0, 1),
    ("converted", 1, 0),
    ("converted_multiplied", 0, 1),
    ("stored_private", 0, 1),
    ("from_private", 0, 1),
    ("shared", 1, 0),
    ("sampled", 6, 0),
    ("sampled_compared", 5, 1),
    ("bg", 2, 1),
    ("negated_beside_its_source", 0, 1),
];

/// Each form left as written and its settled twin, whose one more use of a sum is one a device may
/// combine it with.
const TWINS: [(&str, &str); 3] = [
    ("stored", "stored_compared"),
    ("converted", "converted_multiplied"),
    ("sampled", "sampled_compared"),
];

fn assemble() -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("m2v-settled-sums-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, MODULE).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    }
}

/// The `float` additions and subtractions of `entry` after the rewrite on `p`: those that keep
/// `reassoc` (left as written) and those without it (settled).
fn sums(bc: &[u8], entry: &str, p: &DeviceProfile) -> (usize, usize) {
    let e = specialized_entry(
        bc,
        entry,
        p,
        &FunctionConstantValues::defaults(),
        &[],
        &mut meter(),
    )
    .unwrap_or_else(|err| panic!("{entry}: {err}"));
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("a body");
    let types = e.module.types();
    let (mut alone, mut settled) = (0, 0);
    for (_, inst) in body.insts.iter() {
        let InstKind::BinOp {
            op: AirBin::FAdd | AirBin::FSub,
            fmf,
            ..
        } = &inst.kind
        else {
            continue;
        };
        let elem = match types.get(inst.ty) {
            Some(Type::Vector { elem, .. }) => types.get(*elem),
            t => t,
        };
        assert!(matches!(elem, Some(Type::Float)), "{entry}: a float sum");
        if fmf.contains(FastMath::REASSOC) {
            alone += 1;
        } else {
            settled += 1;
        }
    }
    (alone, settled)
}

/// After the rewrite each form's sums are left as written or settled as the module docs say, on
/// every committed profile.
#[test]
fn a_sum_no_device_can_combine_is_left_as_written() {
    let Some(bc) = assemble() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for (entry, alone, settled) in FORMS {
            assert_eq!(
                sums(&bc, entry, &p),
                (alone, settled),
                "{stem} {entry}: (left as written, settled)"
            );
        }
    }
}

/// The ids of every `OpFAdd` and `OpFSub` result of `words` decorated `NoContraction`.
fn decorated_sums(words: &[u32]) -> usize {
    let (mut sums, mut decorated) = (Vec::new(), Vec::new());
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Decorate as u32
            && words[i + 2] == crate::spv::Decoration::NoContraction as u32
        {
            decorated.push(words[i + 1]);
        }
        if opcode == crate::spv::Op::FAdd as u32 || opcode == crate::spv::Op::FSub as u32 {
            sums.push(words[i + 2]);
        }
        i += len;
    }
    sums.iter().filter(|s| decorated.contains(s)).count()
}

/// In SPIR-V, on every committed profile, each settled twin decorates exactly one addition or
/// subtraction more than its form left as written, and the forms whose only sum is left as
/// written (`stored`, `converted`, `shared`) decorate none; every module validates.
#[test]
fn only_a_settled_sum_is_no_contraction() {
    let Some(bc) = assemble() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        let decorated = |entry: &str| -> Option<usize> {
            let a = match compile(&bc, entry, &p) {
                Ok(a) => a,
                Err(Error::Unsupported(u)) => {
                    eprintln!("{stem} {entry}: refused: {u}");
                    return None;
                }
                Err(e) => panic!("{stem} {entry}: {e}"),
            };
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
            Some(decorated_sums(&words))
        };
        for entry in ["stored", "converted", "shared"] {
            if let Some(n) = decorated(entry) {
                assert_eq!(n, 0, "{stem} {entry}: decorated sums");
            }
        }
        for (alone, twin) in TWINS {
            if let (Some(a), Some(t)) = (decorated(alone), decorated(twin)) {
                assert_eq!(t, a + 1, "{stem} {alone} {twin}: decorated sums");
            }
        }
    }
}

/// The pairs the kernels read: the special pairs below, then pairs from a fixed generator, each
/// operand normal and the result in range for `converted`'s integer.
fn pairs() -> Vec<[f32; 2]> {
    let ulp = 2f32.powi(-23);
    let mut out = vec![
        // Ties: each rounds to even.
        [1.0, ulp / 2.0],
        [1.0 + ulp, ulp / 2.0],
        [1.0, 3.0 * ulp / 2.0],
        [-1.0, -ulp / 2.0],
        // Zeros: `-0 + -0` is `-0`, `x + -x` is `+0`.
        [-0.0, -0.0],
        [0.0, -0.0],
        [1.5, -1.5],
        // Near an integer, where the difference's rounding moves its truncation.
        [16_777_216.0, -0.5],
        [4_194_304.5, 0.25],
        [-1_048_576.0 - 0.25, 0.125],
        [1.0e30, 1.0e30],
        [1.0e-37, 1.0e-37],
    ];
    let mut x: u32 = 0x9E37_79B9;
    while out.len() < 4096 {
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        // Normal operands of magnitude below 2^24, with 23 random fraction bits.
        let mut normal = || {
            let w = next();
            let exponent = 127 - 20 + (w >> 24) % 44;
            f32::from_bits((w & 0x8000_0000) | (exponent << 23) | (next() & 0x7F_FFFF))
        };
        out.push([normal(), normal()]);
    }
    out
}

/// The words of `entry`'s `out` and `aux` buffers on the interpreter, for `p`, over [`pairs`].
fn interpret(
    plan_module: &(Plan, crate::spirv::ir::TypedModule),
    n: usize,
) -> (Vec<u32>, Vec<u32>) {
    let (plan, module) = plan_module;
    let slot = |index: u32| ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    };
    let buffer = |index: u32, bytes: Vec<u8>| Buffer {
        slot: Some(slot(index)),
        address: None,
        bytes,
    };
    let input: Vec<u8> = pairs()
        .iter()
        .flat_map(|[a, b]| a.to_le_bytes().into_iter().chain(b.to_le_bytes()))
        .collect();
    let mut spec = BTreeMap::new();
    spec.insert(SpecKey::LocalSize(Axis::X), 64);
    spec.insert(SpecKey::LocalSize(Axis::Y), 1);
    spec.insert(SpecKey::LocalSize(Axis::Z), 1);
    spec.insert(SpecKey::FlatLocalSize, 64);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let resources = Resources {
        buffers: vec![
            buffer(0, input),
            buffer(1, vec![0xEE; 4 * n]),
            buffer(2, vec![0xEE; 4 * n]),
        ],
        push,
        spec,
    };
    let cfg = Config {
        workgroups: [u32::try_from(n / 64).expect("groups"), 1, 1],
        order: Order::Forward,
        step_limit: 1 << 28,
        ..Config::default()
    };
    let o = run(module, &cfg, resources).expect("run");
    let words = |b: &[u8]| -> Vec<u32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    };
    (words(&o.buffers[1]), words(&o.buffers[2]))
}

/// On the interpreter the forms and their settled twins give the same words, on every committed
/// profile: `stored`'s sums and `converted`'s integers, over 4096 pairs. The interpreter rounds
/// each operation on its own whether or not it is decorated, so this holds the module's own
/// semantics, not a device's (the runtime lane `settled_sums` runs the forms on the device): and
/// each sum is the correctly rounded one, each integer the truncated difference.
#[test]
fn a_sum_left_as_written_gives_the_settled_twin_s_words() {
    let Some(bc) = assemble() else {
        return;
    };
    let pairs = pairs();
    let n = pairs.len();
    for (stem, p) in committed_profiles().expect("profiles") {
        let run_of = |entry: &str| {
            let m = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            interpret(&m, n)
        };
        let (sums, _) = run_of("stored");
        let (twin_sums, _) = run_of("stored_compared");
        assert_eq!(sums, twin_sums, "{stem}: stored");
        let (_, ints) = run_of("converted");
        let (_, twin_ints) = run_of("converted_multiplied");
        assert_eq!(ints, twin_ints, "{stem}: converted");
        for (g, [a, b]) in pairs.iter().enumerate() {
            let s = a + b;
            if s.is_normal() || s == 0.0 {
                assert_eq!(sums[g], s.to_bits(), "{stem}: {a:e} + {b:e}");
            }
            let d = a - b;
            if d.is_finite() && d.abs() < 2f32.powi(31) {
                assert_eq!(ints[g] as i32, d as i32, "{stem}: {a:e} - {b:e}");
            }
        }
    }
}
