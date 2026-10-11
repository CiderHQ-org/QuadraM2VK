; ModuleID = 'validation/fixtures/public/kernel_fc_selected_view_of_one_slot.metal'
source_filename = "validation/fixtures/public/kernel_fc_selected_view_of_one_slot.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@_ZL15kWeightsAreHalf = internal unnamed_addr addrspace(2) global i8 undef, align 1
@_Z15kWeightsAreHalf.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@__metal_implicit_fc_pred_1 = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_k.metal, ptr null }]
@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(2) @__metal_implicit_fc_pred_0 to ptr), ptr addrspacecast (ptr addrspace(2) @__metal_implicit_fc_pred_1 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
define void @read_the_fc_selected_view_of_one_slot(ptr addrspace(1) nocapture noundef readonly "air-buffer-no-alias" %0, ptr addrspace(1) nocapture noundef readonly "air-buffer-no-alias" %1, ptr addrspace(1) nocapture noundef writeonly "air-buffer-no-alias" %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = load i8, ptr addrspace(2) @_ZL15kWeightsAreHalf, align 1, !tbaa !26, !range !30
  %6 = icmp eq i8 %5, 0
  %7 = zext i32 %3 to i64
  br i1 %6, label %12, label %8

8:                                                ; preds = %4
  %9 = getelementptr inbounds half, ptr addrspace(1) %0, i64 %7
  %10 = load half, ptr addrspace(1) %9, align 2, !tbaa !31, !alias.scope !33, !noalias !36
  %11 = fpext half %10 to float
  br label %15

12:                                               ; preds = %4
  %13 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %7
  %14 = load float, ptr addrspace(1) %13, align 4, !tbaa !39, !alias.scope !41, !noalias !42
  br label %15

15:                                               ; preds = %12, %8
  %16 = phi fast float [ %14, %12 ], [ %11, %8 ]
  %17 = getelementptr inbounds float, ptr addrspace(1) %2, i64 %7
  store float %16, ptr addrspace(1) %17, align 4, !tbaa !39, !alias.scope !43, !noalias !44
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i8 @air.normalize_function_constant_predicate.i8(i8) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn
define internal void @_GLOBAL__sub_I_k.metal() #2 section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @_Z15kWeightsAreHalf.MTL_FC_INIT_0_b, align 1, !tbaa !26, !range !30
  store i8 %1, ptr addrspace(2) @_ZL15kWeightsAreHalf, align 1, !tbaa !26
  %2 = xor i8 %1, 1
  %3 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1) #3
  store i8 %3, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  %4 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %2) #3
  store i8 %4, ptr addrspace(2) @__metal_implicit_fc_pred_1, align 1
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #2 = { mustprogress nofree nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #3 = { nounwind readnone willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!18, !19, !20}
!air.function_constants = !{!21}
!llvm.ident = !{!22}
!air.version = !{!23}
!air.language_version = !{!24}
!air.source_file_name = !{!25}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @read_the_fc_selected_view_of_one_slot, !10, !11}
!10 = !{}
!11 = !{!12, !14, !16, !17}
!12 = !{i32 0, !"air.function_constant", !13, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"wh"}
!13 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"kWeightsAreHalf"}
!14 = !{i32 1, !"air.function_constant", !15, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"wf"}
!15 = !{ptr addrspace(2) @__metal_implicit_fc_pred_1, !"bool", !"kWeightsAreFloat"}
!16 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!17 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{ptr addrspace(2) @_Z15kWeightsAreHalf.MTL_FC_INIT_0_b, !"bool", !"kWeightsAreHalf", i32 0}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 4, i32 0, i32 0}
!25 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_fc_selected_view_of_one_slot.metal"}
!26 = !{!27, !27, i64 0}
!27 = !{!"bool", !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
!30 = !{i8 0, i8 2}
!31 = !{!32, !32, i64 0}
!32 = !{!"half", !28, i64 0}
!33 = !{!34}
!34 = distinct !{!34, !35, !"air-alias-scope-arg(0)"}
!35 = distinct !{!35, !"air-alias-scopes(read_the_fc_selected_view_of_one_slot)"}
!36 = !{!37, !38}
!37 = distinct !{!37, !35, !"air-alias-scope-arg(1)"}
!38 = distinct !{!38, !35, !"air-alias-scope-arg(2)"}
!39 = !{!40, !40, i64 0}
!40 = !{!"float", !28, i64 0}
!41 = !{!37}
!42 = !{!34, !38}
!43 = !{!38}
!44 = !{!34, !37}
