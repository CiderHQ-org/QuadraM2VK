; ModuleID = 'kernel_source_order_product_is_not_regroupable'
source_filename = "kernel_source_order_product_is_not_regroupable.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; The Metal half. It exists because `newLibraryWithSource` compiles a `.metal` with the DEFAULT
; compile options, and Metal's default math mode is fast -- which would let the oracle itself
; regroup the product this fixture is about. Assembled by `xcrun metal-as`, the empty flag runs
; below preserve the source multiplication grouping in the native AIR.
define void @kernel_source_order_product_is_not_regroupable(float addrspace(1)* nocapture "air-buffer-no-alias" %0, float addrspace(1)* nocapture readonly "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = load float, float addrspace(1)* %1, align 4
  %4 = getelementptr inbounds float, float addrspace(1)* %1, i64 1
  %5 = load float, float addrspace(1)* %4, align 4
  %6 = getelementptr inbounds float, float addrspace(1)* %1, i64 2
  %7 = load float, float addrspace(1)* %6, align 4
  %8 = fmul float %3, %5
  %9 = fmul float %8, %7
  store float %9, float addrspace(1)* %0, align 4
  ret void
}

attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!air.kernel = !{!9}
!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.version = !{!20}
!air.language_version = !{!21}
!air.source_file_name = !{!22}
!air.compile_options = !{!16}
!llvm.ident = !{!19}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (float addrspace(1)*, float addrspace(1)*)* @kernel_source_order_product_is_not_regroupable, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!16 = !{!"air.compile.denorms_disable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 3, i32 0, i32 0}
!22 = !{!"kernel_source_order_product_is_not_regroupable.metal"}
