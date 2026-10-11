; Generated from kernel_quad_vote.metal by
;   xcrun metal -std=metal3.0 -S -emit-llvm, retripled to spirv-unknown-vulkan1.2, opt -S.
; No hand edits.
;
; The shape this reaches: `air.quad_all` and `air.quad_any`. SPIR-V's subgroup vote instructions
; cover the WHOLE implementation subgroup, which holds several Metal quads, so they are not a
; semantic match; `passes::air_calls::integer_simd::lower_quad_vote` instead runs the same XOR
; butterfly as quad_sum -- shuffle-xor by 1 to combine adjacent lanes, then by 2 to combine the two
; pairs -- over the predicate as 0/1, with BitwiseAnd for `all` and BitwiseOr for `any`. Eleven
; corpus sources call these two intrinsics and no authored case reached the lowering.
source_filename = "q.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_quad_vote(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, i32 %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %4
  %6 = load i32, ptr addrspace(1) %5, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %7 = icmp ne i32 %6, 0
  %8 = tail call i1 @air.quad_all(i1 %7) #2
  %9 = zext i1 %8 to i32
  %10 = shl i32 %2, 1
  %11 = zext i32 %10 to i64
  %12 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %11
  store i32 %9, ptr addrspace(1) %12, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %13 = tail call i1 @air.quad_any(i1 %7) #2
  %14 = zext i1 %13 to i32
  %15 = or i32 %10, 1
  %16 = zext i32 %15 to i64
  %17 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %16
  store i32 %14, ptr addrspace(1) %17, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare i1 @air.quad_all(i1) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i1 @air.quad_any(i1) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_quad_vote, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 3, i32 0, i32 0}
!21 = !{!"/private/tmp/qv/q.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_quad_vote)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
