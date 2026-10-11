; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -x metal
; -S -emit-llvm -` (stdin) and round-tripped through LLVM 21's llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_rate_map_argument_buffer.bc'
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.MapArguments = type { ptr addrspace(2), ptr addrspace(2), ptr addrspace(2) }

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: write)
define void @kernel_rate_map_argument_buffer(ptr addrspace(2) noundef readonly align 8 captures(none) dereferenceable(24) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, <2 x i32> noundef %2) local_unnamed_addr #0 {
  %4 = extractelement <2 x i32> %2, i64 1
  %5 = icmp eq i32 %4, 0
  %6 = getelementptr inbounds %struct.MapArguments, ptr addrspace(2) %0, i64 0, i32 1
  %7 = getelementptr inbounds %struct.MapArguments, ptr addrspace(2) %0, i64 0, i32 2
  %8 = select i1 %5, ptr addrspace(2) %6, ptr addrspace(2) %7
  %9 = bitcast ptr addrspace(2) %8 to ptr addrspace(2)
  %10 = load ptr addrspace(2), ptr addrspace(2) %9, align 8, !tbaa !25, !alias.scope !29, !noalias !32
  %11 = getelementptr inbounds %struct.MapArguments, ptr addrspace(2) %0, i64 0, i32 0
  %12 = load ptr addrspace(2), ptr addrspace(2) %11, align 8, !tbaa !34, !alias.scope !29, !noalias !32
  %13 = extractelement <2 x i32> %2, i64 0
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds <2 x float>, ptr addrspace(2) %12, i64 %14
  %16 = load <2 x float>, ptr addrspace(2) %15, align 8, !tbaa !36
  %17 = tail call fast <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float> %16, ptr addrspace(2) readonly captures(none) %10, i32 0) #2
  %18 = shl i32 %4, 3
  %19 = add i32 %18, %13
  %20 = bitcast <2 x float> %17 to <2 x i32>
  %21 = extractelement <2 x i32> %20, i64 0
  %22 = shl i32 %19, 1
  %23 = zext i32 %22 to i64
  %24 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %23
  store i32 %21, ptr addrspace(1) %24, align 4, !tbaa !37, !alias.scope !32, !noalias !29
  %25 = extractelement <2 x i32> %20, i64 1
  %26 = or i32 %22, 1
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %27
  store i32 %25, ptr addrspace(1) %28, align 4, !tbaa !37, !alias.scope !32, !noalias !29
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare <2 x float> @air.map_physical_to_screen_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2) readonly captures(none), i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!19, !20, !21}
!llvm.ident = !{!22}
!air.version = !{!23}
!air.language_version = !{!24}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_rate_map_argument_buffer, !10, !11}
!10 = !{}
!11 = !{!12, !17, !18}
!12 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !13, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"MapArguments", !"air.arg_name", !"arguments"}
!13 = !{i32 0, i32 8, i32 0, !"float2", !"coordinates", !"air.indirect_argument", !14, i32 8, i32 8, i32 0, !"metal::rasterization_rate_map_data", !"left", !"air.indirect_argument", !15, i32 16, i32 8, i32 0, !"metal::rasterization_rate_map_data", !"right", !"air.indirect_argument", !16}
!14 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"coordinates"}
!15 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"left"}
!16 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"right"}
!17 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!18 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}
!19 = !{!"air.compile.denorms_disable"}
!20 = !{!"air.compile.fast_math_enable"}
!21 = !{!"air.compile.framebuffer_fetch_enable"}
!22 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 4, i32 0, i32 0}
!25 = !{!26, !26, i64 0}
!26 = !{!"any pointer", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C++ TBAA"}
!29 = !{!30}
!30 = distinct !{!30, !31, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !"air-alias-scopes(kernel_rate_map_argument_buffer)"}
!32 = !{!33}
!33 = distinct !{!33, !31, !"air-alias-scope-arg(1)"}
!34 = !{!35, !26, i64 0}
!35 = !{!"_ZTS12MapArguments", !26, i64 0, !26, i64 8, !26, i64 16}
!36 = !{!27, !27, i64 0}
!37 = !{!38, !38, i64 0}
!38 = !{!"int", !27, i64 0}
