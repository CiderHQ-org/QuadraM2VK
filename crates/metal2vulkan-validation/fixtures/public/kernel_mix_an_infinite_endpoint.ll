; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal
; -fno-fast-math -S -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; `-fno-fast-math` makes the frontend emit the plain `air.mix.f32` rather than `air.fast_mix.f32`.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_mix_an_infinite_endpoint.bc'
source_filename = "validation/fixtures/public/kernel_mix_an_infinite_endpoint.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_mix_an_infinite_endpoint(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = load float, ptr addrspace(1) %0, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %4 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  %5 = load float, ptr addrspace(1) %4, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %6 = getelementptr inbounds float, ptr addrspace(1) %0, i64 12
  %7 = load float, ptr addrspace(1) %6, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %8 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  %9 = load float, ptr addrspace(1) %8, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %10 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  %11 = load float, ptr addrspace(1) %10, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %12 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  %13 = load float, ptr addrspace(1) %12, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %14 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  %15 = load float, ptr addrspace(1) %14, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %16 = getelementptr inbounds float, ptr addrspace(1) %0, i64 6
  %17 = load float, ptr addrspace(1) %16, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 7
  %19 = load float, ptr addrspace(1) %18, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %20 = getelementptr inbounds float, ptr addrspace(1) %0, i64 8
  %21 = load float, ptr addrspace(1) %20, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %22 = getelementptr inbounds float, ptr addrspace(1) %0, i64 9
  %23 = load float, ptr addrspace(1) %22, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %24 = getelementptr inbounds float, ptr addrspace(1) %0, i64 10
  %25 = load float, ptr addrspace(1) %24, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %26 = getelementptr inbounds float, ptr addrspace(1) %0, i64 11
  %27 = load float, ptr addrspace(1) %26, align 4, !tbaa !21, !alias.scope !25, !noalias !28
  %28 = tail call float @air.mix.f32(float %3, float %9, float %13) #2
  %29 = bitcast float %28 to i32
  %30 = and i32 %29, 2147483647
  %31 = icmp ugt i32 %30, 2139095040
  %32 = select i1 %31, float 1.000000e+00, float 0.000000e+00
  store float %32, ptr addrspace(1) %1, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %33 = tail call float @air.mix.f32(float %9, float %3, float %11) #2
  %34 = bitcast float %33 to i32
  %35 = and i32 %34, 2147483647
  %36 = icmp ugt i32 %35, 2139095040
  %37 = select i1 %36, float 1.000000e+00, float 0.000000e+00
  %38 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  store float %37, ptr addrspace(1) %38, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %39 = tail call float @air.mix.f32(float %5, float %9, float %13) #2
  %40 = bitcast float %39 to i32
  %41 = and i32 %40, 2147483647
  %42 = icmp ugt i32 %41, 2139095040
  %43 = select i1 %42, float 1.000000e+00, float 0.000000e+00
  %44 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  store float %43, ptr addrspace(1) %44, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %45 = tail call float @air.mix.f32(float %9, float %5, float %11) #2
  %46 = bitcast float %45 to i32
  %47 = and i32 %46, 2147483647
  %48 = icmp ugt i32 %47, 2139095040
  %49 = select i1 %48, float 1.000000e+00, float 0.000000e+00
  %50 = getelementptr inbounds float, ptr addrspace(1) %1, i64 3
  store float %49, ptr addrspace(1) %50, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %51 = tail call float @air.mix.f32(float %3, float %7, float %13) #2
  %52 = bitcast float %51 to i32
  %53 = and i32 %52, 2147483647
  %54 = icmp ugt i32 %53, 2139095040
  %55 = select i1 %54, float 1.000000e+00, float 0.000000e+00
  %56 = getelementptr inbounds float, ptr addrspace(1) %1, i64 4
  store float %55, ptr addrspace(1) %56, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %57 = tail call float @air.mix.f32(float %17, float %15, float %13) #2
  %58 = getelementptr inbounds float, ptr addrspace(1) %1, i64 5
  store float %57, ptr addrspace(1) %58, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %59 = tail call float @air.mix.f32(float %19, float %21, float %13) #2
  %60 = getelementptr inbounds float, ptr addrspace(1) %1, i64 6
  store float %59, ptr addrspace(1) %60, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %61 = tail call float @air.mix.f32(float %19, float %21, float %11) #2
  %62 = getelementptr inbounds float, ptr addrspace(1) %1, i64 7
  store float %61, ptr addrspace(1) %62, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  %63 = tail call float @air.mix.f32(float %25, float %27, float %23) #2
  %64 = getelementptr inbounds float, ptr addrspace(1) %1, i64 8
  store float %63, ptr addrspace(1) %64, align 4, !tbaa !21, !alias.scope !28, !noalias !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.mix.f32(float, float, float) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
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
!9 = !{ptr @kernel_mix_an_infinite_endpoint, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_disable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/private/tmp/mixcase/kernel_mix_an_infinite_endpoint.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"float", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_mix_an_infinite_endpoint)"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"air-alias-scope-arg(1)"}
