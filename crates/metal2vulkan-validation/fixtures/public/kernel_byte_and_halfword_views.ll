; ModuleID = 'kernel_byte_and_halfword_views.metal'
source_filename = "kernel_byte_and_halfword_views.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn
define void @kernel_byte_and_halfword_views(i32 addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %0, i32 addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = bitcast i32 addrspace(1)* %0 to i8 addrspace(1)*
  %5 = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 1
  %6 = bitcast i32 addrspace(1)* %5 to i16 addrspace(1)*
  %7 = zext i32 %2 to i64
  %8 = getelementptr inbounds i8, i8 addrspace(1)* %4, i64 %7
  %9 = load i8, i8 addrspace(1)* %8, align 1, !tbaa !22, !alias.scope !25, !noalias !28
  %10 = zext i8 %9 to i32
  %11 = getelementptr inbounds i16, i16 addrspace(1)* %6, i64 %7
  %12 = load i16, i16 addrspace(1)* %11, align 2, !tbaa !30, !alias.scope !25, !noalias !28
  %13 = zext i16 %12 to i32
  %14 = shl nuw i32 %13, 16
  %15 = or i32 %14, %10
  %16 = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 %7
  store i32 %15, i32 addrspace(1)* %16, align 4, !tbaa !32, !alias.scope !28, !noalias !25
  ret void
}

attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

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
!9 = !{void (i32 addrspace(1)*, i32 addrspace(1)*, i32)* @kernel_byte_and_halfword_views, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"o"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"kernel_byte_and_halfword_views.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_byte_and_halfword_views)"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"air-alias-scope-arg(1)"}
!30 = !{!31, !31, i64 0}
!31 = !{!"short", !23, i64 0}
!32 = !{!33, !33, i64 0}
!33 = !{!"int", !23, i64 0}
