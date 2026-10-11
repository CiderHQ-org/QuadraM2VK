; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'ffq.bc'
source_filename = "validation/fixtures/public/kernel_fast_fmod_large_quotient.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_fast_fmod_large_quotient(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = load float, ptr addrspace(1) %1, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %4 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  %5 = load float, ptr addrspace(1) %4, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %6 = tail call fast float @air.fast_fmod.f32(float %3, float %5) #2
  %7 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  store float %6, ptr addrspace(1) %7, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %8 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  %9 = load float, ptr addrspace(1) %8, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %10 = getelementptr inbounds float, ptr addrspace(1) %1, i64 3
  %11 = load float, ptr addrspace(1) %10, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %12 = tail call fast float @air.fast_fmod.f32(float %9, float %11) #2
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  %14 = bitcast ptr addrspace(1) %13 to ptr addrspace(1)
  store float %12, ptr addrspace(1) %14, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %15 = getelementptr inbounds float, ptr addrspace(1) %1, i64 4
  %16 = load float, ptr addrspace(1) %15, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %17 = getelementptr inbounds float, ptr addrspace(1) %1, i64 5
  %18 = load float, ptr addrspace(1) %17, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %19 = tail call fast float @air.fast_fmod.f32(float %16, float %18) #2
  %20 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  %21 = bitcast ptr addrspace(1) %20 to ptr addrspace(1)
  store float %19, ptr addrspace(1) %21, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %22 = getelementptr inbounds float, ptr addrspace(1) %1, i64 6
  %23 = load float, ptr addrspace(1) %22, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %24 = getelementptr inbounds float, ptr addrspace(1) %1, i64 7
  %25 = load float, ptr addrspace(1) %24, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %26 = tail call fast float @air.fast_fmod.f32(float %23, float %25) #2
  %27 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  %28 = bitcast ptr addrspace(1) %27 to ptr addrspace(1)
  store float %26, ptr addrspace(1) %28, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %29 = getelementptr inbounds float, ptr addrspace(1) %1, i64 8
  %30 = load float, ptr addrspace(1) %29, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %31 = getelementptr inbounds float, ptr addrspace(1) %1, i64 9
  %32 = load float, ptr addrspace(1) %31, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %33 = tail call fast float @air.fast_fmod.f32(float %30, float %32) #2
  %34 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  %35 = bitcast ptr addrspace(1) %34 to ptr addrspace(1)
  store float %33, ptr addrspace(1) %35, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %36 = getelementptr inbounds float, ptr addrspace(1) %1, i64 10
  %37 = load float, ptr addrspace(1) %36, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %38 = getelementptr inbounds float, ptr addrspace(1) %1, i64 11
  %39 = load float, ptr addrspace(1) %38, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %40 = tail call fast float @air.fast_fmod.f32(float %37, float %39) #2
  %41 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  %42 = bitcast ptr addrspace(1) %41 to ptr addrspace(1)
  store float %40, ptr addrspace(1) %42, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %43 = getelementptr inbounds float, ptr addrspace(1) %1, i64 12
  %44 = load float, ptr addrspace(1) %43, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %45 = getelementptr inbounds float, ptr addrspace(1) %1, i64 13
  %46 = load float, ptr addrspace(1) %45, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %47 = tail call fast float @air.fast_fmod.f32(float %44, float %46) #2
  %48 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  %49 = bitcast ptr addrspace(1) %48 to ptr addrspace(1)
  store float %47, ptr addrspace(1) %49, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %50 = getelementptr inbounds float, ptr addrspace(1) %1, i64 14
  %51 = load float, ptr addrspace(1) %50, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %52 = getelementptr inbounds float, ptr addrspace(1) %1, i64 15
  %53 = load float, ptr addrspace(1) %52, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %54 = tail call fast float @air.fast_fmod.f32(float %51, float %53) #2
  %55 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  %56 = bitcast ptr addrspace(1) %55 to ptr addrspace(1)
  store float %54, ptr addrspace(1) %56, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %57 = getelementptr inbounds float, ptr addrspace(1) %1, i64 16
  %58 = load float, ptr addrspace(1) %57, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %59 = insertelement <2 x float> undef, float %58, i64 0
  %60 = getelementptr inbounds float, ptr addrspace(1) %1, i64 17
  %61 = load float, ptr addrspace(1) %60, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %62 = insertelement <2 x float> %59, float %61, i64 1
  %63 = getelementptr inbounds float, ptr addrspace(1) %1, i64 18
  %64 = load float, ptr addrspace(1) %63, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %65 = insertelement <2 x float> undef, float %64, i64 0
  %66 = getelementptr inbounds float, ptr addrspace(1) %1, i64 19
  %67 = load float, ptr addrspace(1) %66, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %68 = insertelement <2 x float> %65, float %67, i64 1
  %69 = tail call fast <2 x float> @air.fast_fmod.v2f32(<2 x float> %62, <2 x float> %68) #2
  %70 = getelementptr inbounds float, ptr addrspace(1) %1, i64 20
  %71 = load float, ptr addrspace(1) %70, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %72 = insertelement <2 x float> undef, float %71, i64 0
  %73 = getelementptr inbounds float, ptr addrspace(1) %1, i64 21
  %74 = load float, ptr addrspace(1) %73, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %75 = insertelement <2 x float> %72, float %74, i64 1
  %76 = getelementptr inbounds float, ptr addrspace(1) %1, i64 22
  %77 = load float, ptr addrspace(1) %76, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %78 = insertelement <2 x float> undef, float %77, i64 0
  %79 = getelementptr inbounds float, ptr addrspace(1) %1, i64 23
  %80 = load float, ptr addrspace(1) %79, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %81 = insertelement <2 x float> %78, float %80, i64 1
  %82 = tail call fast <2 x float> @air.fast_fmod.v2f32(<2 x float> %75, <2 x float> %81) #2
  %83 = bitcast <2 x float> %69 to <2 x i32>
  %84 = extractelement <2 x i32> %83, i64 0
  %85 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  store i32 %84, ptr addrspace(1) %85, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %86 = extractelement <2 x i32> %83, i64 1
  %87 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 9
  store i32 %86, ptr addrspace(1) %87, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %88 = bitcast <2 x float> %82 to <2 x i32>
  %89 = extractelement <2 x i32> %88, i64 0
  %90 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 10
  store i32 %89, ptr addrspace(1) %90, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %91 = extractelement <2 x i32> %88, i64 1
  %92 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 11
  store i32 %91, ptr addrspace(1) %92, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_fmod.f32(float, float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_fmod.v2f32(<2 x float>, <2 x float>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_fast_fmod_large_quotient, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_fast_fmod_large_quotient.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"float", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(1)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_fast_fmod_large_quotient)"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"air-alias-scope-arg(0)"}
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !23, i64 0}
