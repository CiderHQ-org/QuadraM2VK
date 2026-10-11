; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'h.bc'
source_filename = "validation/fixtures/public/kernel_half_transcendentals.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_half_transcendentals(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = load half, ptr addrspace(1) %1, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %5 = insertelement <4 x half> undef, half %4, i64 0
  %6 = getelementptr inbounds half, ptr addrspace(1) %1, i64 1
  %7 = load half, ptr addrspace(1) %6, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %8 = insertelement <4 x half> %5, half %7, i64 1
  %9 = getelementptr inbounds half, ptr addrspace(1) %1, i64 2
  %10 = load half, ptr addrspace(1) %9, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %11 = insertelement <4 x half> %8, half %10, i64 2
  %12 = getelementptr inbounds half, ptr addrspace(1) %1, i64 3
  %13 = load half, ptr addrspace(1) %12, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %14 = insertelement <4 x half> %11, half %13, i64 3
  %15 = getelementptr inbounds half, ptr addrspace(1) %1, i64 4
  %16 = load half, ptr addrspace(1) %15, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %17 = insertelement <4 x half> undef, half %16, i64 0
  %18 = getelementptr inbounds half, ptr addrspace(1) %1, i64 5
  %19 = load half, ptr addrspace(1) %18, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %20 = insertelement <4 x half> %17, half %19, i64 1
  %21 = getelementptr inbounds half, ptr addrspace(1) %1, i64 6
  %22 = load half, ptr addrspace(1) %21, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %23 = insertelement <4 x half> %20, half %22, i64 2
  %24 = getelementptr inbounds half, ptr addrspace(1) %1, i64 7
  %25 = load half, ptr addrspace(1) %24, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %26 = insertelement <4 x half> %23, half %25, i64 3
  %27 = getelementptr inbounds half, ptr addrspace(1) %1, i64 8
  %28 = load half, ptr addrspace(1) %27, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %29 = insertelement <4 x half> undef, half %28, i64 0
  %30 = getelementptr inbounds half, ptr addrspace(1) %1, i64 9
  %31 = load half, ptr addrspace(1) %30, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %32 = insertelement <4 x half> %29, half %31, i64 1
  %33 = getelementptr inbounds half, ptr addrspace(1) %1, i64 10
  %34 = load half, ptr addrspace(1) %33, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %35 = insertelement <4 x half> %32, half %34, i64 2
  %36 = getelementptr inbounds half, ptr addrspace(1) %1, i64 11
  %37 = load half, ptr addrspace(1) %36, align 2, !tbaa !22, !alias.scope !26, !noalias !29
  %38 = insertelement <4 x half> %35, half %37, i64 3
  %39 = tail call fast <4 x half> @air.log2.v4f16(<4 x half> %14) #2
  %40 = tail call fast <4 x half> @air.log.v4f16(<4 x half> %26) #2
  %41 = tail call fast <4 x half> @air.exp.v4f16(<4 x half> %38) #2
  %42 = extractelement <4 x half> %39, i64 0
  store half %42, ptr addrspace(1) %0, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %43 = extractelement <4 x half> %39, i64 1
  %44 = getelementptr inbounds half, ptr addrspace(1) %0, i64 1
  store half %43, ptr addrspace(1) %44, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %45 = extractelement <4 x half> %39, i64 2
  %46 = getelementptr inbounds half, ptr addrspace(1) %0, i64 2
  store half %45, ptr addrspace(1) %46, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %47 = extractelement <4 x half> %39, i64 3
  %48 = getelementptr inbounds half, ptr addrspace(1) %0, i64 3
  store half %47, ptr addrspace(1) %48, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %49 = extractelement <4 x half> %40, i64 0
  %50 = getelementptr inbounds half, ptr addrspace(1) %0, i64 4
  store half %49, ptr addrspace(1) %50, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %51 = extractelement <4 x half> %40, i64 1
  %52 = getelementptr inbounds half, ptr addrspace(1) %0, i64 5
  store half %51, ptr addrspace(1) %52, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %53 = extractelement <4 x half> %40, i64 2
  %54 = getelementptr inbounds half, ptr addrspace(1) %0, i64 6
  store half %53, ptr addrspace(1) %54, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %55 = extractelement <4 x half> %40, i64 3
  %56 = getelementptr inbounds half, ptr addrspace(1) %0, i64 7
  store half %55, ptr addrspace(1) %56, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %57 = extractelement <4 x half> %41, i64 0
  %58 = getelementptr inbounds half, ptr addrspace(1) %0, i64 8
  store half %57, ptr addrspace(1) %58, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %59 = extractelement <4 x half> %41, i64 1
  %60 = getelementptr inbounds half, ptr addrspace(1) %0, i64 9
  store half %59, ptr addrspace(1) %60, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %61 = extractelement <4 x half> %41, i64 2
  %62 = getelementptr inbounds half, ptr addrspace(1) %0, i64 10
  store half %61, ptr addrspace(1) %62, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  %63 = extractelement <4 x half> %41, i64 3
  %64 = getelementptr inbounds half, ptr addrspace(1) %0, i64 11
  store half %63, ptr addrspace(1) %64, align 2, !tbaa !22, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.log2.v4f16(<4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.log.v4f16(<4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.exp.v4f16(<4 x half>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="64" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_half_transcendentals, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"in"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid", !"air.arg_unused"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/privatevalidation/fixtures/public/kernel_half_transcendentals.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"half", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_half_transcendentals)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
