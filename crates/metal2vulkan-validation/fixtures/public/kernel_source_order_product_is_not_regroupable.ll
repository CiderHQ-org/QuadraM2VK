; ModuleID = 'f.tri.ll'
source_filename = "f.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Both multiplies carry an EMPTY flag run: no `fast`, no `contract`, no `reassoc`. Metal groups
; them exactly as written, and the emitted module has to say so.
define void @kernel_source_order_product_is_not_regroupable(ptr addrspace(1) captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = load float, ptr addrspace(1) %1, align 4, !tbaa !21, !alias.scope !24
  %4 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  %5 = load float, ptr addrspace(1) %4, align 4, !tbaa !21, !alias.scope !24
  %6 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  %7 = load float, ptr addrspace(1) %6, align 4, !tbaa !21, !alias.scope !24
  %8 = fmul float %3, %5
  %9 = fmul float %8, %7
  store float %9, ptr addrspace(1) %0, align 4, !tbaa !21, !alias.scope !24
  ret void
}

attributes #0 = { convergent mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) }

!air.kernel = !{!0}
!llvm.module.flags = !{!30, !31}
!air.version = !{!32}
!air.language_version = !{!33}
!air.compile_options = !{!34}

!0 = !{ptr @kernel_source_order_product_is_not_regroupable, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!21 = !{!22, !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C++ TBAA"}
!24 = !{!25}
!25 = distinct !{!25, !26, !"air-alias-scope-arg(0)"}
!26 = distinct !{!26, !"air-alias-scopes(kernel_source_order_product_is_not_regroupable)"}
!30 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
!31 = !{i32 1, !"wchar_size", i32 4}
!32 = !{i32 2, i32 8, i32 0}
!33 = !{!"Metal", i32 4, i32 0, i32 0}
!34 = !{!"air.compile.denorms_disable"}
