; Owned synthetic fixture: `llvm-dis` of `xcrun -sdk macosx metal -c -x metal -` fed the sibling
; .metal on stdin. Not derived from a third-party metallib.
; ModuleID = '<stdin>'
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Commands = type { %"struct.metal::command_buffer" }
%"struct.metal::command_buffer" = type { ptr addrspace(1) }

; Function Attrs: mustprogress nounwind willreturn
define void @kernel_icb_encode_compute_commands(ptr addrspace(1) noundef readonly align 8 captures(none) dereferenceable(8) "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = getelementptr inbounds %struct.Commands, ptr addrspace(1) %0, i64 0, i32 0, i32 0
  %6 = load ptr addrspace(1), ptr addrspace(1) %5, align 8, !tbaa.struct !24, !alias.scope !29, !noalias !32
  %7 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  tail call void @air.set_kernel_buffer_compute_command.p1i8(ptr addrspace(1) captures(none) %6, i32 %3, ptr addrspace(1) %7, i64 -1, i32 0) #2
  tail call void @air.set_threadgroup_memory_length_compute_command(ptr addrspace(1) captures(none) %6, i32 %3, i32 16, i32 0) #2
  %8 = add i32 %3, 1
  %9 = insertelement <3 x i32> <i32 poison, i32 1, i32 1>, i32 %8, i64 0
  tail call void @air.concurrent_dispatch_threadgroups_compute_command(ptr addrspace(1) captures(none) %6, i32 %3, <3 x i32> %9, <3 x i32> <i32 32, i32 1, i32 1>) #2
  tail call void @air.set_barrier_compute_command(ptr addrspace(1) captures(none) %6, i32 %3) #2
  %10 = zext i32 %3 to i64
  %11 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %10
  %12 = load i32, ptr addrspace(1) %11, align 4, !tbaa !35, !alias.scope !37, !noalias !38
  %13 = mul i32 %12, 3
  %14 = add i32 %13, 7
  %15 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %10
  store i32 %14, ptr addrspace(1) %15, align 4, !tbaa !35, !alias.scope !39, !noalias !40
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare void @air.set_threadgroup_memory_length_compute_command(ptr addrspace(1) captures(none), i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn
declare void @air.concurrent_dispatch_threadgroups_compute_command(ptr addrspace(1) captures(none), i32, <3 x i32>, <3 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn
declare void @air.set_barrier_compute_command(ptr addrspace(1) captures(none), i32) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn
declare void @air.set_kernel_buffer_compute_command.p1i8(ptr addrspace(1) captures(none), i32, ptr addrspace(1), i64, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nounwind willreturn }
attributes #2 = { nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_icb_encode_compute_commands, !10, !11}
!10 = !{}
!11 = !{!12, !15, !16, !17}
!12 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !13, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Commands", !"air.arg_name", !"encoder"}
!13 = !{i32 0, i32 8, i32 0, !"command_buffer", !"commands", !"air.indirect_argument", !14}
!14 = !{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"commands"}
!15 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"data"}
!16 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
!24 = !{i64 0, i64 8, !25}
!25 = !{!26, !26, i64 0}
!26 = !{!"__metal_command_buffer_t", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C++ TBAA"}
!29 = !{!30}
!30 = distinct !{!30, !31, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !"air-alias-scopes(kernel_icb_encode_compute_commands)"}
!32 = !{!33, !34}
!33 = distinct !{!33, !31, !"air-alias-scope-arg(1)"}
!34 = distinct !{!34, !31, !"air-alias-scope-arg(2)"}
!35 = !{!36, !36, i64 0}
!36 = !{!"int", !27, i64 0}
!37 = !{!33}
!38 = !{!30, !34}
!39 = !{!34}
!40 = !{!30, !33}
