; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. The 7 x 12 edge grid
; for `lower_metal_pow`, the hand-rolled negative-base guard 984 corpus sources reach and none of
; them exercises.
; Not derived from a third-party metallib.
; ModuleID = '/tmp/powf/g.bc'
source_filename = "validation/fixtures/public/kernel_pow_signed_zero_and_infinite_base.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_pow_signed_zero_and_infinite_base(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds <2 x float>, ptr addrspace(1) %0, i64 %4
  %6 = load <2 x float>, ptr addrspace(1) %5, align 8, !alias.scope !22, !noalias !25
  %7 = extractelement <2 x float> %6, i64 0
  %8 = extractelement <2 x float> %6, i64 1
  %9 = tail call fast float @air.pow.f32(float %7, float %8) #2
  %10 = tail call fast float @air.fast_pow.f32(float %7, float %8) #2
  %11 = fptrunc float %7 to half
  %12 = fptrunc float %8 to half
  %13 = tail call fast half @air.pow.f16(half %11, half %12) #2
  %14 = fpext half %13 to float
  %15 = insertelement <4 x float> <float poison, float poison, float poison, float 0.000000e+00>, float %9, i64 0
  %16 = insertelement <4 x float> %15, float %10, i64 1
  %17 = insertelement <4 x float> %16, float %14, i64 2
  %18 = getelementptr inbounds <4 x float>, ptr addrspace(1) %1, i64 %4
  store <4 x float> %17, ptr addrspace(1) %18, align 16, !tbaa !27, !alias.scope !25, !noalias !22
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.pow.f32(float, float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.fast_pow.f32(float, float) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.pow.f16(half, half) local_unnamed_addr #1

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
!9 = !{ptr @kernel_pow_signed_zero_and_infinite_base, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_pow_signed_zero_and_infinite_base.metal"}
!22 = !{!23}
!23 = distinct !{!23, !24, !"air-alias-scope-arg(0)"}
!24 = distinct !{!24, !"air-alias-scopes(kernel_pow_signed_zero_and_infinite_base)"}
!25 = !{!26}
!26 = distinct !{!26, !24, !"air-alias-scope-arg(1)"}
!27 = !{!28, !28, i64 0}
!28 = !{!"omnipotent char", !29, i64 0}
!29 = !{!"Simple C++ TBAA"}
