; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. The DEFAULT compile is
; wanted here: it emits `air.fast_acosh`/`asinh`/`atanh`/`tan`, which are the uncovered symbols.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_inverse_hyperbolics_and_a_gated_bias.bc'
source_filename = "validation/fixtures/public/kernel_inverse_hyperbolics_and_a_gated_bias.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@_ZL8has_bias = internal unnamed_addr addrspace(2) global i16 undef, align 2
@_Z8has_bias.MTL_FC_INIT_0_s = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i16 0, align 2
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_k.metal, ptr null }]
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(2) @__metal_implicit_fc_pred_0 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree nosync nounwind willreturn
define void @kernel_inverse_hyperbolics_and_a_gated_bias(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %3) local_unnamed_addr #0 {
  %5 = load float, ptr addrspace(1) %0, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %6 = tail call fast float @air.fast_acosh.f32(float %5) #3
  %7 = bitcast ptr addrspace(1) %3 to ptr addrspace(1)
  store float %6, ptr addrspace(1) %7, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %8 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  %9 = load float, ptr addrspace(1) %8, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %10 = tail call fast float @air.fast_acosh.f32(float %9) #3
  %11 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 1
  %12 = bitcast ptr addrspace(1) %11 to ptr addrspace(1)
  store float %10, ptr addrspace(1) %12, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %13 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  %14 = load float, ptr addrspace(1) %13, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %15 = tail call fast float @air.fast_asinh.f32(float %14) #3
  %16 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 2
  %17 = bitcast ptr addrspace(1) %16 to ptr addrspace(1)
  store float %15, ptr addrspace(1) %17, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  %19 = load float, ptr addrspace(1) %18, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %20 = tail call fast float @air.fast_asinh.f32(float %19) #3
  %21 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 3
  %22 = bitcast ptr addrspace(1) %21 to ptr addrspace(1)
  store float %20, ptr addrspace(1) %22, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %23 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  %24 = load float, ptr addrspace(1) %23, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %25 = tail call fast float @air.fast_asinh.f32(float %24) #3
  %26 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 4
  %27 = bitcast ptr addrspace(1) %26 to ptr addrspace(1)
  store float %25, ptr addrspace(1) %27, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %28 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  %29 = load float, ptr addrspace(1) %28, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %30 = tail call fast float @air.fast_atanh.f32(float %29) #3
  %31 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 5
  %32 = bitcast ptr addrspace(1) %31 to ptr addrspace(1)
  store float %30, ptr addrspace(1) %32, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %33 = getelementptr inbounds float, ptr addrspace(1) %0, i64 6
  %34 = load float, ptr addrspace(1) %33, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %35 = tail call fast float @air.fast_atanh.f32(float %34) #3
  %36 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 6
  %37 = bitcast ptr addrspace(1) %36 to ptr addrspace(1)
  store float %35, ptr addrspace(1) %37, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %38 = getelementptr inbounds float, ptr addrspace(1) %0, i64 7
  %39 = load float, ptr addrspace(1) %38, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %40 = tail call fast float @air.fast_atanh.f32(float %39) #3
  %41 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 7
  %42 = bitcast ptr addrspace(1) %41 to ptr addrspace(1)
  store float %40, ptr addrspace(1) %42, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %43 = getelementptr inbounds float, ptr addrspace(1) %0, i64 8
  %44 = load float, ptr addrspace(1) %43, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %45 = tail call fast float @air.fast_atanh.f32(float %44) #3
  %46 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 8
  %47 = bitcast ptr addrspace(1) %46 to ptr addrspace(1)
  store float %45, ptr addrspace(1) %47, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %48 = load <3 x float>, ptr addrspace(1) %1, align 16, !tbaa !40, !alias.scope !41, !noalias !42
  %49 = tail call fast <3 x float> @air.fast_tan.v3f32(<3 x float> %48) #3
  %50 = bitcast <3 x float> %49 to <3 x i32>
  %51 = extractelement <3 x i32> %50, i64 0
  %52 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 9
  store i32 %51, ptr addrspace(1) %52, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %53 = extractelement <3 x i32> %50, i64 1
  %54 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 10
  store i32 %53, ptr addrspace(1) %54, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %55 = extractelement <3 x i32> %50, i64 2
  %56 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 11
  store i32 %55, ptr addrspace(1) %56, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  %57 = getelementptr inbounds float, ptr addrspace(1) %0, i64 9
  %58 = load float, ptr addrspace(1) %57, align 4, !tbaa !25, !alias.scope !29, !noalias !32
  %59 = load i16, ptr addrspace(2) @_ZL8has_bias, align 2, !tbaa !43
  %60 = icmp eq i16 %59, 0
  br i1 %60, label %64, label %61

61:                                               ; preds = %4
  %62 = load float, ptr addrspace(1) %2, align 4, !tbaa !25, !alias.scope !45, !noalias !46
  %63 = fadd fast float %62, %58
  br label %64

64:                                               ; preds = %61, %4
  %65 = phi float [ %63, %61 ], [ %58, %4 ]
  %66 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 12
  %67 = bitcast ptr addrspace(1) %66 to ptr addrspace(1)
  store float %65, ptr addrspace(1) %67, align 4, !tbaa !36, !alias.scope !38, !noalias !39
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.normalize_function_constant_predicate.i16(i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_acosh.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_asinh.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_atanh.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_tan.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn
define internal void @_GLOBAL__sub_I_k.metal() #2 section "air.static_init" {
  %1 = load i16, ptr addrspace(2) @_Z8has_bias.MTL_FC_INIT_0_s, align 2, !tbaa !43
  store i16 %1, ptr addrspace(2) @_ZL8has_bias, align 2, !tbaa !43
  %2 = tail call i16 @air.normalize_function_constant_predicate.i16(i16 %1) #3
  store i16 %2, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 2
  ret void
}

attributes #0 = { mustprogress nofree nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { mustprogress nofree nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #3 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!air.function_constants = !{!20}
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
!9 = !{ptr @kernel_inverse_hyperbolics_and_a_gated_bias, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fin"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float3", !"air.arg_name", !"vin"}
!14 = !{i32 2, !"air.function_constant", !15, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"bias"}
!15 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"short", !"has_bias"}
!16 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{ptr addrspace(2) @_Z8has_bias.MTL_FC_INIT_0_s, !"short", !"has_bias", i32 0}
!21 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!22 = !{i32 2, i32 8, i32 0}
!23 = !{!"Metal", i32 4, i32 0, i32 0}
!24 = !{!"/private/tmp/ft/k.metal"}
!25 = !{!26, !26, i64 0}
!26 = !{!"float", !27, i64 0}
!27 = !{!"omnipotent char", !28, i64 0}
!28 = !{!"Simple C++ TBAA"}
!29 = !{!30}
!30 = distinct !{!30, !31, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !"air-alias-scopes(kernel_inverse_hyperbolics_and_a_gated_bias)"}
!32 = !{!33, !34, !35}
!33 = distinct !{!33, !31, !"air-alias-scope-arg(1)"}
!34 = distinct !{!34, !31, !"air-alias-scope-arg(2)"}
!35 = distinct !{!35, !31, !"air-alias-scope-arg(3)"}
!36 = !{!37, !37, i64 0}
!37 = !{!"int", !27, i64 0}
!38 = !{!35}
!39 = !{!30, !33, !34}
!40 = !{!27, !27, i64 0}
!41 = !{!33}
!42 = !{!30, !34, !35}
!43 = !{!44, !44, i64 0}
!44 = !{!"short", !27, i64 0}
!45 = !{!34}
!46 = !{!30, !33, !35}
