; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. The DEFAULT compile
; is essential here: it is what makes every call `air.fast_*`, which is the whole point of the case.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_fast_variant_reach.bc'
source_filename = "validation/fixtures/public/kernel_fast_variant_reach.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_fast_variant_reach(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = load float, ptr addrspace(1) %0, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %4 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  %5 = load float, ptr addrspace(1) %4, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %6 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  %7 = load float, ptr addrspace(1) %6, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %8 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  %9 = load float, ptr addrspace(1) %8, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %10 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  %11 = load float, ptr addrspace(1) %10, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %12 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  %13 = load float, ptr addrspace(1) %12, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %14 = getelementptr inbounds float, ptr addrspace(1) %0, i64 6
  %15 = load float, ptr addrspace(1) %14, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %16 = getelementptr inbounds float, ptr addrspace(1) %0, i64 7
  %17 = load float, ptr addrspace(1) %16, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 8
  %19 = load float, ptr addrspace(1) %18, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %20 = getelementptr inbounds float, ptr addrspace(1) %0, i64 9
  %21 = load float, ptr addrspace(1) %20, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %22 = getelementptr inbounds float, ptr addrspace(1) %0, i64 10
  %23 = load float, ptr addrspace(1) %22, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %24 = getelementptr inbounds float, ptr addrspace(1) %0, i64 11
  %25 = load float, ptr addrspace(1) %24, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %26 = tail call fast float @air.fast_exp.f32(float %3) #2
  %27 = bitcast ptr addrspace(1) %1 to ptr addrspace(1)
  store float %26, ptr addrspace(1) %27, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %28 = tail call fast float @air.fast_exp.f32(float %5) #2
  %29 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1
  %30 = bitcast ptr addrspace(1) %29 to ptr addrspace(1)
  store float %28, ptr addrspace(1) %30, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %31 = tail call fast float @air.fast_powr.f32(float %7, float %11) #2
  %32 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 2
  %33 = bitcast ptr addrspace(1) %32 to ptr addrspace(1)
  store float %31, ptr addrspace(1) %33, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %34 = tail call fast float @air.fast_powr.f32(float %9, float %13) #2
  %35 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 3
  %36 = bitcast ptr addrspace(1) %35 to ptr addrspace(1)
  store float %34, ptr addrspace(1) %36, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %37 = tail call fast float @air.fast_log.f32(float %15) #2
  %38 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 4
  %39 = bitcast ptr addrspace(1) %38 to ptr addrspace(1)
  store float %37, ptr addrspace(1) %39, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %40 = tail call fast float @air.fast_log.f32(float %17) #2
  %41 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 5
  %42 = bitcast ptr addrspace(1) %41 to ptr addrspace(1)
  store float %40, ptr addrspace(1) %42, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %43 = tail call fast float @air.fast_rsqrt.f32(float %19) #2
  %44 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 6
  %45 = bitcast ptr addrspace(1) %44 to ptr addrspace(1)
  store float %43, ptr addrspace(1) %45, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %46 = tail call fast float @air.fast_rsqrt.f32(float %21) #2
  %47 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 7
  %48 = bitcast ptr addrspace(1) %47 to ptr addrspace(1)
  store float %46, ptr addrspace(1) %48, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %49 = tail call fast float @air.fast_sqrt.f32(float %23) #2
  %50 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 8
  %51 = bitcast ptr addrspace(1) %50 to ptr addrspace(1)
  store float %49, ptr addrspace(1) %51, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  %52 = tail call fast float @air.fast_sqrt.f32(float %25) #2
  %53 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 9
  %54 = bitcast ptr addrspace(1) %53 to ptr addrspace(1)
  store float %52, ptr addrspace(1) %54, align 4, !tbaa !30, !alias.scope !28, !noalias !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_exp.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_powr.f32(float, float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_log.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_rsqrt.f32(float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_sqrt.f32(float) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_fast_variant_reach, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_fast_variant_reach.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"float", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_fast_variant_reach)"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"air-alias-scope-arg(1)"}
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !23, i64 0}
