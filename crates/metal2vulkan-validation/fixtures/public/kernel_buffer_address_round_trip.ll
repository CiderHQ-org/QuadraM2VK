; ModuleID = 'kernel_buffer_address_round_trip.air'
source_filename = "validation/fixtures/public/kernel_buffer_address_round_trip.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Params = type { i32 }
%struct.Row = type { i16, i16, i16 }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define void @buffer_address_round_trip(ptr addrspace(2) noundef "air-buffer-no-alias" %0, ptr addrspace(2) noundef readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = ptrtoint ptr addrspace(2) %0 to i64
  %6 = getelementptr inbounds %struct.Params, ptr addrspace(2) %1, i64 0, i32 0
  %7 = load i32, ptr addrspace(2) %6, align 4, !tbaa !25, !alias.scope !30, !noalias !33
  %8 = zext i32 %7 to i64
  %9 = add i64 %8, %5
  %10 = inttoptr i64 %9 to ptr addrspace(2)
  %11 = zext i32 %3 to i64
  %12 = getelementptr inbounds %struct.Row, ptr addrspace(2) %10, i64 %11, i32 0
  %13 = load i16, ptr addrspace(2) %12, align 2, !tbaa.struct !36
  %14 = getelementptr inbounds %struct.Row, ptr addrspace(2) %10, i64 %11, i32 1
  %15 = load i16, ptr addrspace(2) %14, align 2, !tbaa.struct !39
  %16 = getelementptr inbounds %struct.Row, ptr addrspace(2) %10, i64 %11, i32 2
  %17 = load i16, ptr addrspace(2) %16, align 2, !tbaa.struct !40
  %18 = zext i16 %13 to i32
  %19 = zext i16 %15 to i32
  %20 = shl nuw nsw i32 %19, 8
  %21 = or i32 %20, %18
  %22 = zext i16 %17 to i32
  %23 = shl nuw i32 %22, 16
  %24 = or i32 %21, %23
  %25 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %11
  store i32 %24, ptr addrspace(1) %25, align 4, !tbaa !41, !alias.scope !42, !noalias !43
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @buffer_address_round_trip, !10, !11}
!10 = !{}
!11 = !{!12, !14, !16, !17}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !13, !"air.arg_type_size", i32 6, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"Row", !"air.arg_name", !"base"}
!13 = !{i32 0, i32 2, i32 0, !"ushort", !"a", i32 2, i32 2, i32 0, !"ushort", !"b", i32 4, i32 2, i32 0, !"ushort", !"c"}
!14 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !15, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"p"}
!15 = !{i32 0, i32 4, i32 0, !"uint", !"byte_offset"}
!16 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
!24 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_buffer_address_round_trip.metal"}
!25 = !{!26, !27, i64 0}
!26 = !{!"_ZTS6Params", !27, i64 0}
!27 = !{!"int", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"air-alias-scope-arg(1)"}
!32 = distinct !{!32, !"air-alias-scopes(buffer_address_round_trip)"}
!33 = !{!34, !35}
!34 = distinct !{!34, !32, !"air-alias-scope-arg(0)"}
!35 = distinct !{!35, !32, !"air-alias-scope-arg(2)"}
!36 = !{i64 0, i64 2, !37, i64 2, i64 2, !37, i64 4, i64 2, !37}
!37 = !{!38, !38, i64 0}
!38 = !{!"short", !28, i64 0}
!39 = !{i64 0, i64 2, !37, i64 2, i64 2, !37}
!40 = !{i64 0, i64 2, !37}
!41 = !{!27, !27, i64 0}
!42 = !{!35}
!43 = !{!34, !31}
