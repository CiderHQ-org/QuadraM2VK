; Owned synthetic fixture: `llvm-dis` of `xcrun -sdk macosx metal -c -x metal -` fed the sibling
; .metal on stdin. Not derived from a third-party metallib.
; ModuleID = '<stdin>'
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Commands = type <{ %"struct.metal::command_buffer", i32, [4 x i8] }>
%"struct.metal::command_buffer" = type { ptr addrspace(1) }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_icb_declared_handle_is_inert(ptr addrspace(1) noundef readonly align 8 captures(none) dereferenceable(12) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = getelementptr inbounds %struct.Commands, ptr addrspace(1) %0, i64 0, i32 1
  %5 = load i32, ptr addrspace(1) %4, align 8, !tbaa !24, !alias.scope !31, !noalias !34
  %6 = mul i32 %5, %2
  %7 = add i32 %6, 1
  %8 = zext i32 %2 to i64
  %9 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %8
  store i32 %7, ptr addrspace(1) %9, align 4, !tbaa !36, !alias.scope !34, !noalias !31
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

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
!9 = !{ptr @kernel_icb_declared_handle_is_inert, !10, !11}
!10 = !{}
!11 = !{!12, !16, !17}
!12 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !13, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Commands", !"air.arg_name", !"encoder"}
!13 = !{i32 0, i32 8, i32 0, !"command_buffer", !"commands", !"air.indirect_argument", !14, i32 8, i32 4, i32 0, !"uint", !"scale", !"air.indirect_argument", !15}
!14 = !{i32 0, !"air.command_buffer", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"command_buffer", !"air.arg_name", !"commands"}
!15 = !{i32 1, !"air.indirect_constant", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"scale"}
!16 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
!24 = !{!25, !30, i64 8}
!25 = !{!"_ZTS8Commands", !26, i64 0, !30, i64 8}
!26 = !{!"_ZTSN5metal14command_bufferE", !27, i64 0}
!27 = !{!"__metal_command_buffer_t", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !{!"int", !28, i64 0}
!31 = !{!32}
!32 = distinct !{!32, !33, !"air-alias-scope-arg(0)"}
!33 = distinct !{!33, !"air-alias-scopes(kernel_icb_declared_handle_is_inert)"}
!34 = !{!35}
!35 = distinct !{!35, !33, !"air-alias-scope-arg(1)"}
!36 = !{!30, !30, i64 0}
