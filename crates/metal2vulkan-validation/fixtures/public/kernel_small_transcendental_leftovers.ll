; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. The DEFAULT compile
; is wanted for `sinh`/`cosh`, which emits `air.fast_*`; `round` is spelled `precise::round`
; because the default compile emits `air.fast_round.f32`, which is already covered.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_small_transcendental_leftovers.bc'
source_filename = "validation/fixtures/public/kernel_small_transcendental_leftovers.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_small_transcendental_leftovers(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %2) local_unnamed_addr #0 {
  %4 = load float, ptr addrspace(1) %0, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %5 = tail call fast float @air.fast_sinh.f32(float %4) #2
  %6 = bitcast ptr addrspace(1) %2 to ptr addrspace(1)
  store float %5, ptr addrspace(1) %6, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %7 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  %8 = load float, ptr addrspace(1) %7, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %9 = tail call fast float @air.fast_sinh.f32(float %8) #2
  %10 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1
  %11 = bitcast ptr addrspace(1) %10 to ptr addrspace(1)
  store float %9, ptr addrspace(1) %11, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %12 = tail call fast float @air.fast_cosh.f32(float %4) #2
  %13 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 2
  %14 = bitcast ptr addrspace(1) %13 to ptr addrspace(1)
  store float %12, ptr addrspace(1) %14, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %15 = tail call fast float @air.fast_cosh.f32(float %8) #2
  %16 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 3
  %17 = bitcast ptr addrspace(1) %16 to ptr addrspace(1)
  store float %15, ptr addrspace(1) %17, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  %19 = load float, ptr addrspace(1) %18, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %20 = tail call fast float @air.round.f32(float %19) #2
  %21 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 4
  %22 = bitcast ptr addrspace(1) %21 to ptr addrspace(1)
  store float %20, ptr addrspace(1) %22, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %23 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  %24 = load float, ptr addrspace(1) %23, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %25 = tail call fast float @air.round.f32(float %24) #2
  %26 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 5
  %27 = bitcast ptr addrspace(1) %26 to ptr addrspace(1)
  store float %25, ptr addrspace(1) %27, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %28 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  %29 = load float, ptr addrspace(1) %28, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %30 = tail call fast float @air.round.f32(float %29) #2
  %31 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 6
  %32 = bitcast ptr addrspace(1) %31 to ptr addrspace(1)
  store float %30, ptr addrspace(1) %32, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %33 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  %34 = load float, ptr addrspace(1) %33, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %35 = tail call fast float @air.round.f32(float %34) #2
  %36 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 7
  %37 = bitcast ptr addrspace(1) %36 to ptr addrspace(1)
  store float %35, ptr addrspace(1) %37, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %38 = load half, ptr addrspace(1) %1, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %39 = insertelement <2 x half> undef, half %38, i64 0
  %40 = getelementptr inbounds half, ptr addrspace(1) %1, i64 1
  %41 = load half, ptr addrspace(1) %40, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %42 = insertelement <2 x half> %39, half %41, i64 1
  %43 = tail call fast <2 x half> @air.sqrt.v2f16(<2 x half> %42) #2
  %44 = getelementptr inbounds half, ptr addrspace(1) %1, i64 2
  %45 = load half, ptr addrspace(1) %44, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %46 = insertelement <2 x half> undef, half %45, i64 0
  %47 = getelementptr inbounds half, ptr addrspace(1) %1, i64 3
  %48 = load half, ptr addrspace(1) %47, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %49 = insertelement <2 x half> %46, half %48, i64 1
  %50 = tail call fast <2 x half> @air.sqrt.v2f16(<2 x half> %49) #2
  %51 = bitcast <2 x half> %43 to <2 x i16>
  %52 = extractelement <2 x i16> %51, i64 0
  %53 = zext i16 %52 to i32
  %54 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 8
  store i32 %53, ptr addrspace(1) %54, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %55 = extractelement <2 x i16> %51, i64 1
  %56 = zext i16 %55 to i32
  %57 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 9
  store i32 %56, ptr addrspace(1) %57, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %58 = bitcast <2 x half> %50 to <2 x i16>
  %59 = extractelement <2 x i16> %58, i64 0
  %60 = zext i16 %59 to i32
  %61 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 10
  store i32 %60, ptr addrspace(1) %61, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %62 = extractelement <2 x i16> %58, i64 1
  %63 = zext i16 %62 to i32
  %64 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 11
  store i32 %63, ptr addrspace(1) %64, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %65 = getelementptr inbounds half, ptr addrspace(1) %1, i64 4
  %66 = load half, ptr addrspace(1) %65, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %67 = insertelement <2 x half> undef, half %66, i64 0
  %68 = getelementptr inbounds half, ptr addrspace(1) %1, i64 5
  %69 = load half, ptr addrspace(1) %68, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %70 = insertelement <2 x half> %67, half %69, i64 1
  %71 = tail call fast <2 x half> @air.cos.v2f16(<2 x half> %70) #2
  %72 = bitcast <2 x half> %71 to <2 x i16>
  %73 = extractelement <2 x i16> %72, i64 0
  %74 = zext i16 %73 to i32
  %75 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 12
  store i32 %74, ptr addrspace(1) %75, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %76 = extractelement <2 x i16> %72, i64 1
  %77 = zext i16 %76 to i32
  %78 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 13
  store i32 %77, ptr addrspace(1) %78, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %79 = getelementptr inbounds half, ptr addrspace(1) %1, i64 6
  %80 = load half, ptr addrspace(1) %79, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %81 = tail call fast half @air.sinpi.f16(half %80) #2
  %82 = bitcast half %81 to i16
  %83 = zext i16 %82 to i32
  %84 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 14
  store i32 %83, ptr addrspace(1) %84, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %85 = getelementptr inbounds half, ptr addrspace(1) %1, i64 7
  %86 = load half, ptr addrspace(1) %85, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %87 = tail call fast half @air.sinpi.f16(half %86) #2
  %88 = bitcast half %87 to i16
  %89 = zext i16 %88 to i32
  %90 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 15
  store i32 %89, ptr addrspace(1) %90, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %91 = getelementptr inbounds half, ptr addrspace(1) %1, i64 8
  %92 = load half, ptr addrspace(1) %91, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %93 = tail call fast half @air.sinpi.f16(half %92) #2
  %94 = bitcast half %93 to i16
  %95 = zext i16 %94 to i32
  %96 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 16
  store i32 %95, ptr addrspace(1) %96, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %97 = getelementptr inbounds half, ptr addrspace(1) %1, i64 9
  %98 = load half, ptr addrspace(1) %97, align 2, !tbaa !36, !alias.scope !38, !noalias !39
  %99 = tail call fast half @air.sinpi.f16(half %98) #2
  %100 = bitcast half %99 to i16
  %101 = zext i16 %100 to i32
  %102 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 17
  store i32 %101, ptr addrspace(1) %102, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_sinh.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_cosh.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.round.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x half> @air.sqrt.v2f16(<2 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x half> @air.cos.v2f16(<2 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.sinpi.f16(half) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="32" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_small_transcendental_leftovers, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fin"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_small_transcendental_leftovers.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_small_transcendental_leftovers)"}
!29 = !{!30, !31}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = distinct !{!31, !28, !"air-alias-scope-arg(2)"}
!32 = !{!33, !33, i64 0}
!33 = !{!"int", !24, i64 0}
!34 = !{!31}
!35 = !{!27, !30}
!36 = !{!37, !37, i64 0}
!37 = !{!"half", !24, i64 0}
!38 = !{!30}
!39 = !{!27, !31}
