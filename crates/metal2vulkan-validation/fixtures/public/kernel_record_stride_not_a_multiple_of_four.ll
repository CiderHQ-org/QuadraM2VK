; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = '/tmp/rs.bc'
source_filename = "validation/fixtures/public/kernel_record_stride_not_a_multiple_of_four.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.Rec = type { [5 x i16], [7 x i8] }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write)
define void @record_stride_not_a_multiple_of_four(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %4
  store i32 18002, ptr addrspace(1) %5, align 4, !tbaa !23, !alias.scope !27, !noalias !30
  %6 = trunc i32 %2 to i16
  %7 = mul i16 %6, 7
  %8 = add i16 %7, 1
  %9 = getelementptr inbounds %struct.Rec, ptr addrspace(1) %0, i64 %4, i32 0, i64 3
  store i16 %8, ptr addrspace(1) %9, align 2, !tbaa !32, !alias.scope !30, !noalias !27
  %10 = trunc i32 %2 to i8
  %11 = xor i8 %10, -91
  %12 = getelementptr inbounds %struct.Rec, ptr addrspace(1) %0, i64 %4, i32 1, i64 1
  store i8 %11, ptr addrspace(1) %12, align 1, !tbaa !34, !alias.scope !30, !noalias !27
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

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
!9 = !{ptr @record_stride_not_a_multiple_of_four, !10, !11}
!10 = !{}
!11 = !{!12, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !13, !"air.arg_type_size", i32 18, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"Rec", !"air.arg_name", !"recs"}
!13 = !{i32 0, i32 2, i32 5, !"short", !"a", i32 10, i32 1, i32 7, !"uchar", !"b"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"sizes"}
!15 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_record_stride_not_a_multiple_of_four.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!28}
!28 = distinct !{!28, !29, !"air-alias-scope-arg(1)"}
!29 = distinct !{!29, !"air-alias-scopes(record_stride_not_a_multiple_of_four)"}
!30 = !{!31}
!31 = distinct !{!31, !29, !"air-alias-scope-arg(0)"}
!32 = !{!33, !33, i64 0}
!33 = !{!"short", !25, i64 0}
!34 = !{!25, !25, i64 0}
