; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'sa.bc'
source_filename = "validation/fixtures/public/kernel_srgb_unpack_all_bytes.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_srgb_unpack_all_bytes(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %4
  %6 = load i32, ptr addrspace(1) %5, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %7 = tail call fast <4 x float> @air.unpack.unorm4x8.srgb.v4f32(i32 %6) #2
  %8 = extractelement <4 x float> %7, i64 0
  %9 = shl i32 %2, 2
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %10
  store float %8, ptr addrspace(1) %11, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %12 = extractelement <4 x float> %7, i64 1
  %13 = or i32 %9, 1
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %14
  store float %12, ptr addrspace(1) %15, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %16 = extractelement <4 x float> %7, i64 2
  %17 = or i32 %9, 2
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %18
  store float %16, ptr addrspace(1) %19, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  %20 = extractelement <4 x float> %7, i64 3
  %21 = or i32 %9, 3
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %22
  store float %20, ptr addrspace(1) %23, align 4, !tbaa !31, !alias.scope !29, !noalias !26
  ret void
}

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
!9 = !{ptr @kernel_srgb_unpack_all_bytes, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_srgb_unpack_all_bytes.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_srgb_unpack_all_bytes)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
!31 = !{!32, !32, i64 0}
!32 = !{!"float", !24, i64 0}
