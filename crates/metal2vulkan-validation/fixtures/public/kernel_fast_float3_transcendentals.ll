; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'ft.bc'
source_filename = "validation/fixtures/public/kernel_fast_float3_transcendentals.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_fast_float3_transcendentals(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = load float, ptr addrspace(1) %1, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %4 = insertelement <3 x float> undef, float %3, i64 0
  %5 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  %6 = load float, ptr addrspace(1) %5, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %7 = insertelement <3 x float> %4, float %6, i64 1
  %8 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  %9 = load float, ptr addrspace(1) %8, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %10 = insertelement <3 x float> %7, float %9, i64 2
  %11 = getelementptr inbounds float, ptr addrspace(1) %1, i64 3
  %12 = load float, ptr addrspace(1) %11, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %13 = insertelement <3 x float> undef, float %12, i64 0
  %14 = getelementptr inbounds float, ptr addrspace(1) %1, i64 4
  %15 = load float, ptr addrspace(1) %14, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %16 = insertelement <3 x float> %13, float %15, i64 1
  %17 = getelementptr inbounds float, ptr addrspace(1) %1, i64 5
  %18 = load float, ptr addrspace(1) %17, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %19 = insertelement <3 x float> %16, float %18, i64 2
  %20 = getelementptr inbounds float, ptr addrspace(1) %1, i64 6
  %21 = load float, ptr addrspace(1) %20, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %22 = insertelement <3 x float> undef, float %21, i64 0
  %23 = getelementptr inbounds float, ptr addrspace(1) %1, i64 7
  %24 = load float, ptr addrspace(1) %23, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %25 = insertelement <3 x float> %22, float %24, i64 1
  %26 = getelementptr inbounds float, ptr addrspace(1) %1, i64 8
  %27 = load float, ptr addrspace(1) %26, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %28 = insertelement <3 x float> %25, float %27, i64 2
  %29 = getelementptr inbounds float, ptr addrspace(1) %1, i64 9
  %30 = load float, ptr addrspace(1) %29, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %31 = insertelement <3 x float> undef, float %30, i64 0
  %32 = getelementptr inbounds float, ptr addrspace(1) %1, i64 10
  %33 = load float, ptr addrspace(1) %32, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %34 = insertelement <3 x float> %31, float %33, i64 1
  %35 = getelementptr inbounds float, ptr addrspace(1) %1, i64 11
  %36 = load float, ptr addrspace(1) %35, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %37 = insertelement <3 x float> %34, float %36, i64 2
  %38 = getelementptr inbounds float, ptr addrspace(1) %1, i64 12
  %39 = load float, ptr addrspace(1) %38, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %40 = insertelement <3 x float> undef, float %39, i64 0
  %41 = getelementptr inbounds float, ptr addrspace(1) %1, i64 13
  %42 = load float, ptr addrspace(1) %41, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %43 = insertelement <3 x float> %40, float %42, i64 1
  %44 = getelementptr inbounds float, ptr addrspace(1) %1, i64 14
  %45 = load float, ptr addrspace(1) %44, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %46 = insertelement <3 x float> %43, float %45, i64 2
  %47 = tail call fast <3 x float> @air.fast_log2.v3f32(<3 x float> %10) #2
  %48 = tail call fast <3 x float> @air.fast_log.v3f32(<3 x float> %19) #2
  %49 = tail call fast <3 x float> @air.fast_exp2.v3f32(<3 x float> %28) #2
  %50 = tail call fast <3 x float> @air.fast_rsqrt.v3f32(<3 x float> %37) #2
  %51 = tail call fast <3 x float> @air.fast_sin.v3f32(<3 x float> %46) #2
  %52 = extractelement <3 x float> %47, i64 0
  store float %52, ptr addrspace(1) %0, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %53 = extractelement <3 x float> %47, i64 1
  %54 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  store float %53, ptr addrspace(1) %54, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %55 = extractelement <3 x float> %47, i64 2
  %56 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  store float %55, ptr addrspace(1) %56, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %57 = extractelement <3 x float> %48, i64 0
  %58 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  store float %57, ptr addrspace(1) %58, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %59 = extractelement <3 x float> %48, i64 1
  %60 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  store float %59, ptr addrspace(1) %60, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %61 = extractelement <3 x float> %48, i64 2
  %62 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  store float %61, ptr addrspace(1) %62, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %63 = extractelement <3 x float> %49, i64 0
  %64 = getelementptr inbounds float, ptr addrspace(1) %0, i64 6
  store float %63, ptr addrspace(1) %64, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %65 = extractelement <3 x float> %49, i64 1
  %66 = getelementptr inbounds float, ptr addrspace(1) %0, i64 7
  store float %65, ptr addrspace(1) %66, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %67 = extractelement <3 x float> %49, i64 2
  %68 = getelementptr inbounds float, ptr addrspace(1) %0, i64 8
  store float %67, ptr addrspace(1) %68, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %69 = extractelement <3 x float> %50, i64 0
  %70 = getelementptr inbounds float, ptr addrspace(1) %0, i64 9
  store float %69, ptr addrspace(1) %70, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %71 = extractelement <3 x float> %50, i64 1
  %72 = getelementptr inbounds float, ptr addrspace(1) %0, i64 10
  store float %71, ptr addrspace(1) %72, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %73 = extractelement <3 x float> %50, i64 2
  %74 = getelementptr inbounds float, ptr addrspace(1) %0, i64 11
  store float %73, ptr addrspace(1) %74, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %75 = extractelement <3 x float> %51, i64 0
  %76 = getelementptr inbounds float, ptr addrspace(1) %0, i64 12
  store float %75, ptr addrspace(1) %76, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %77 = extractelement <3 x float> %51, i64 1
  %78 = getelementptr inbounds float, ptr addrspace(1) %0, i64 13
  store float %77, ptr addrspace(1) %78, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %79 = extractelement <3 x float> %51, i64 2
  %80 = getelementptr inbounds float, ptr addrspace(1) %0, i64 14
  store float %79, ptr addrspace(1) %80, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_log2.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_log.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_exp2.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_rsqrt.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.fast_sin.v3f32(<3 x float>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!14, !15, !16}
!llvm.ident = !{!17}
!air.version = !{!18}
!air.language_version = !{!19}
!air.source_file_name = !{!20}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_fast_float3_transcendentals, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_fast_float3_transcendentals.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"float", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(1)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_fast_float3_transcendentals)"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"air-alias-scope-arg(0)"}
