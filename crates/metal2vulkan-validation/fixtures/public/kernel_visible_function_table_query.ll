; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'vftq.bc'
source_filename = "validation/fixtures/public/kernel_visible_function_table_query.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nounwind willreturn
define void @kernel_visible_function_table_query(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) %1, ptr addrspace(1) %2) local_unnamed_addr #0 {
  %4 = tail call i1 @air.is_null_visible_function_table(ptr addrspace(1) readonly captures(none) %1) #2, !alias.scope !22, !noalias !25
  %5 = select i1 %4, i32 7, i32 3
  store i32 %5, ptr addrspace(1) %0, align 4, !tbaa !28, !alias.scope !32, !noalias !33
  %6 = tail call i32 @air.get_size_visible_function_table(ptr addrspace(1) readonly captures(none) %1) #2, !alias.scope !22, !noalias !25
  %7 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %6, ptr addrspace(1) %7, align 4, !tbaa !28, !alias.scope !32, !noalias !33
  %8 = icmp eq i32 %6, 0
  %9 = select i1 %8, i32 11, i32 5
  %10 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %9, ptr addrspace(1) %10, align 4, !tbaa !28, !alias.scope !32, !noalias !33
  %11 = tail call i1 @air.is_null_visible_function_table(ptr addrspace(1) readonly captures(none) %2) #2, !alias.scope !34, !noalias !35
  %12 = select i1 %11, i32 17, i32 13
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %12, ptr addrspace(1) %13, align 4, !tbaa !28, !alias.scope !32, !noalias !33
  %14 = tail call i32 @air.get_size_visible_function_table(ptr addrspace(1) readonly captures(none) %2) #2, !alias.scope !34, !noalias !35
  %15 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %14, ptr addrspace(1) %15, align 4, !tbaa !28, !alias.scope !32, !noalias !33
  %16 = add i32 %6, %14
  %17 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %16, ptr addrspace(1) %17, align 4, !tbaa !28, !alias.scope !32, !noalias !33
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i1 @air.is_null_visible_function_table(ptr addrspace(1) readonly captures(none)) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare i32 @air.get_size_visible_function_table(ptr addrspace(1) readonly captures(none)) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind willreturn memory(argmem: read) }

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
!9 = !{ptr @kernel_visible_function_table_query, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.visible_function_table", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table", !"air.arg_name", !"few"}
!14 = !{i32 2, !"air.visible_function_table", !"air.location_index", i32 2, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table", !"air.arg_name", !"many"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_visible_function_table_query.metal"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"air-alias-scope-arg(1)"}
!24 = distinct !{!24, !"air-alias-scopes(kernel_visible_function_table_query)"}
!25 = !{!26, !27}
!26 = distinct !{!26, !24, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !24, !"air-alias-scope-arg(2)"}
!28 = !{!29, !29, i64 0}
!29 = !{!"int", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C++ TBAA"}
!32 = !{!26}
!33 = !{!23, !27}
!34 = !{!27}
!35 = !{!26, !23}
