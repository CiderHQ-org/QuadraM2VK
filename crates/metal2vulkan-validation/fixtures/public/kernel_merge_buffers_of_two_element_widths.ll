; ModuleID = '/tmp/selload/f.fix.ll'
source_filename = "validation/fixtures/public/kernel_merge_buffers_of_two_element_widths.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @merge_buffers_of_two_element_widths(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(2) readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %3, i32 %4) local_unnamed_addr #0 {
  %6 = and i32 %4, 1
  %7 = load i32, ptr addrspace(2) %3, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %8 = icmp eq i32 %6, %7
  %9 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %10 = select i1 %8, ptr addrspace(1) %9, ptr addrspace(1) %1
  %11 = zext i32 %4 to i64
  %12 = getelementptr inbounds i32, ptr addrspace(1) %10, i64 %11
  %13 = load i32, ptr addrspace(1) %12, align 4, !tbaa !24, !alias.scope !35, !noalias !36
  %14 = shl i32 %4, 1
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %15
  store i32 %13, ptr addrspace(1) %16, align 4, !tbaa !24, !alias.scope !37, !noalias !38
  %17 = xor i32 %4, 7
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds i32, ptr addrspace(1) %10, i64 %18
  %20 = load i32, ptr addrspace(1) %19, align 4, !tbaa !24, !alias.scope !35, !noalias !36
  %21 = or i32 %14, 1
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %22
  store i32 %20, ptr addrspace(1) %23, align 4, !tbaa !24, !alias.scope !37, !noalias !38
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @merge_buffers_of_two_element_widths, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"bytes"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"words"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"pick"}
!16 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 3, i32 0, i32 0}
!23 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_merge_buffers_of_two_element_widths.metal"}
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(3)"}
!30 = distinct !{!30, !"air-alias-scopes(merge_buffers_of_two_element_widths)"}
!31 = !{!32, !33, !34}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(0)"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(1)"}
!34 = distinct !{!34, !30, !"air-alias-scope-arg(2)"}
!35 = !{!32, !33}
!36 = !{!34, !29}
!37 = !{!34}
!38 = !{!32, !33, !29}
