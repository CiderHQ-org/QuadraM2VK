; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'lx.bc'
source_filename = "validation/fixtures/public/kernel_fast_ldexp_range.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_fast_ldexp_range(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2) local_unnamed_addr #0 {
  %4 = load float, ptr addrspace(1) %1, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %5 = load i32, ptr addrspace(1) %2, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %6 = tail call fast float @air.fast_ldexp.f32(float %4, i32 %5) #2
  store float %6, ptr addrspace(1) %0, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %7 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  %8 = load float, ptr addrspace(1) %7, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %9 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1
  %10 = load i32, ptr addrspace(1) %9, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %11 = tail call fast float @air.fast_ldexp.f32(float %8, i32 %10) #2
  %12 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  store float %11, ptr addrspace(1) %12, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %13 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  %14 = load float, ptr addrspace(1) %13, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %15 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 2
  %16 = load i32, ptr addrspace(1) %15, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %17 = tail call fast float @air.fast_ldexp.f32(float %14, i32 %16) #2
  %18 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2
  store float %17, ptr addrspace(1) %18, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %19 = getelementptr inbounds float, ptr addrspace(1) %1, i64 3
  %20 = load float, ptr addrspace(1) %19, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %21 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 3
  %22 = load i32, ptr addrspace(1) %21, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %23 = tail call fast float @air.fast_ldexp.f32(float %20, i32 %22) #2
  %24 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3
  store float %23, ptr addrspace(1) %24, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %25 = getelementptr inbounds float, ptr addrspace(1) %1, i64 4
  %26 = load float, ptr addrspace(1) %25, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %27 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 4
  %28 = load i32, ptr addrspace(1) %27, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %29 = tail call fast float @air.fast_ldexp.f32(float %26, i32 %28) #2
  %30 = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  store float %29, ptr addrspace(1) %30, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %31 = getelementptr inbounds float, ptr addrspace(1) %1, i64 5
  %32 = load float, ptr addrspace(1) %31, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %33 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 5
  %34 = load i32, ptr addrspace(1) %33, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %35 = tail call fast float @air.fast_ldexp.f32(float %32, i32 %34) #2
  %36 = getelementptr inbounds float, ptr addrspace(1) %0, i64 5
  store float %35, ptr addrspace(1) %36, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %37 = getelementptr inbounds float, ptr addrspace(1) %1, i64 6
  %38 = load float, ptr addrspace(1) %37, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %39 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 6
  %40 = load i32, ptr addrspace(1) %39, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %41 = tail call fast float @air.fast_ldexp.f32(float %38, i32 %40) #2
  %42 = getelementptr inbounds float, ptr addrspace(1) %0, i64 6
  store float %41, ptr addrspace(1) %42, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %43 = getelementptr inbounds float, ptr addrspace(1) %1, i64 7
  %44 = load float, ptr addrspace(1) %43, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %45 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 7
  %46 = load i32, ptr addrspace(1) %45, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %47 = tail call fast float @air.fast_ldexp.f32(float %44, i32 %46) #2
  %48 = getelementptr inbounds float, ptr addrspace(1) %0, i64 7
  store float %47, ptr addrspace(1) %48, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_ldexp.f32(float, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_fast_ldexp_range, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"iin"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_fast_ldexp_range.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_fast_ldexp_range)"}
!29 = !{!30, !31}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !28, !"air-alias-scope-arg(2)"}
!32 = !{!33, !33, i64 0}
!33 = !{!"int", !24, i64 0}
!34 = !{!31}
!35 = !{!30, !27}
!36 = !{!30}
!37 = !{!27, !31}
