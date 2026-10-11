; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'sr.bc'
source_filename = "validation/fixtures/public/kernel_srgb_pack_roundtrip.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_srgb_pack_roundtrip(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2) local_unnamed_addr #0 {
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
  %15 = tail call i32 @air.pack.unorm4x8.srgb.v4f16(<4 x half> %14) #2
  %16 = load i32, ptr addrspace(1) %2, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %17 = tail call fast <4 x half> @air.unpack.unorm4x8.srgb.v4f16(i32 %16) #2
  %18 = getelementptr inbounds i32, ptr addrspace(1) %2, i64 1
  %19 = load i32, ptr addrspace(1) %18, align 4, !tbaa !32, !alias.scope !34, !noalias !35
  %20 = tail call fast <4 x float> @air.unpack.unorm4x8.srgb.v4f32(i32 %19) #2
  store i32 %15, ptr addrspace(1) %0, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  %21 = bitcast <4 x half> %17 to <4 x i16>
  %22 = extractelement <4 x i16> %21, i64 0
  %23 = zext i16 %22 to i32
  %24 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %23, ptr addrspace(1) %24, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  %25 = extractelement <4 x i16> %21, i64 1
  %26 = zext i16 %25 to i32
  %27 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %26, ptr addrspace(1) %27, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  %28 = extractelement <4 x i16> %21, i64 2
  %29 = zext i16 %28 to i32
  %30 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %29, ptr addrspace(1) %30, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  %31 = extractelement <4 x i16> %21, i64 3
  %32 = zext i16 %31 to i32
  %33 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %32, ptr addrspace(1) %33, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  %34 = bitcast <4 x float> %20 to <4 x i32>
  %35 = extractelement <4 x i32> %34, i64 0
  %36 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %35, ptr addrspace(1) %36, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  %37 = extractelement <4 x i32> %34, i64 1
  %38 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  store i32 %37, ptr addrspace(1) %38, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  %39 = extractelement <4 x i32> %34, i64 2
  %40 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  store i32 %39, ptr addrspace(1) %40, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  %41 = extractelement <4 x i32> %34, i64 3
  %42 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  store i32 %41, ptr addrspace(1) %42, align 4, !tbaa !32, !alias.scope !36, !noalias !37
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.pack.unorm4x8.srgb.v4f16(<4 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x half> @air.unpack.unorm4x8.srgb.v4f16(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x float> @air.unpack.unorm4x8.srgb.v4f32(i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_srgb_pack_roundtrip, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"hin"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"uin"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_srgb_pack_roundtrip.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"half", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_srgb_pack_roundtrip)"}
!29 = !{!30, !31}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !28, !"air-alias-scope-arg(2)"}
!32 = !{!33, !33, i64 0}
!33 = !{!"int", !24, i64 0}
!34 = !{!31}
!35 = !{!30, !27}
!36 = !{!30}
!37 = !{!27, !31}
