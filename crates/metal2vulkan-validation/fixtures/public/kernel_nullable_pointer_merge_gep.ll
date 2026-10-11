; ModuleID = 'kernel_nullable_pointer_merge_gep.ll'
source_filename = "kernel_nullable_pointer_merge_gep.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @gep_through_a_nullable_pointer_merge(ptr addrspace(1) readonly "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(2) readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %2, i32 %3) local_unnamed_addr #0 {
  %5 = load i32, ptr addrspace(2) %2, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %6 = icmp eq i32 %5, 0
  %7 = and i32 %3, 3
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %8
  %10 = select i1 %6, ptr addrspace(1) null, ptr addrspace(1) %9
  %11 = zext i32 %3 to i64
  %12 = getelementptr inbounds float, ptr addrspace(1) %10, i64 %11
  %13 = load float, ptr addrspace(1) %12, align 4, !tbaa !33, !alias.scope !35, !noalias !36
  %14 = icmp eq ptr addrspace(1) %10, null
  %15 = select fast i1 %14, float -1.000000e+00, float %13
  %16 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %11
  store float %15, ptr addrspace(1) %16, align 4, !tbaa !33, !alias.scope !37, !noalias !38
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!16, !17, !18}
!llvm.ident = !{!19}
!air.version = !{!20}
!air.language_version = !{!21}
!air.source_file_name = !{!22}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @gep_through_a_nullable_pointer_merge, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"bias"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"has_bias"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 3, i32 0, i32 0}
!22 = !{!"kernel_nullable_pointer_merge_gep.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(2)"}
!29 = distinct !{!29, !"air-alias-scopes(gep_through_a_nullable_pointer_merge)"}
!30 = !{!31, !32}
!31 = distinct !{!31, !29, !"air-alias-scope-arg(0)"}
!32 = distinct !{!32, !29, !"air-alias-scope-arg(1)"}
!33 = !{!34, !34, i64 0}
!34 = !{!"float", !25, i64 0}
!35 = !{!31}
!36 = !{!32, !28}
!37 = !{!32}
!38 = !{!31, !28}
