; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. The one shape that can
; reach `air.calculate_unclamped_lod_texture_2d`: it needs fragment implicit derivatives, and all
; three corpus sources that call it are fragment shaders.
; Not derived from a third-party metallib.
; ModuleID = 'fragment_unclamped_lod_at_four_scales.bc'
source_filename = "validation/fixtures/public/fragment_unclamped_lod_at_four_scales.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601036873, i64 0], align 8

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
define <4 x float> @fragment_unclamped_lod_at_four_scales(<4 x float> noundef %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) %4) local_unnamed_addr #0 {
  %6 = shufflevector <4 x float> %0, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %7 = fmul fast <2 x float> %6, splat (float 6.250000e-02)
  %8 = tail call fast float @air.calculate_unclamped_lod_texture_2d(ptr addrspace(1) readonly captures(none) %1, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> %7, i32 0) #2
  %9 = insertelement <4 x float> undef, float %8, i64 0
  %10 = tail call fast float @air.calculate_unclamped_lod_texture_2d(ptr addrspace(1) readonly captures(none) %2, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> %7, i32 0) #2
  %11 = insertelement <4 x float> %9, float %10, i64 1
  %12 = tail call fast float @air.calculate_unclamped_lod_texture_2d(ptr addrspace(1) readonly captures(none) %3, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> %7, i32 0) #2
  %13 = insertelement <4 x float> %11, float %12, i64 2
  %14 = tail call fast float @air.calculate_unclamped_lod_texture_2d(ptr addrspace(1) readonly captures(none) %4, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> %7, i32 0) #2
  %15 = insertelement <4 x float> %13, float %14, i64 3
  ret <4 x float> %15
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare float @air.calculate_unclamped_lod_texture_2d(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i32) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree nounwind willreturn memory(read) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { convergent nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.fragment = !{!9}
!air.compile_options = !{!18, !19, !20}
!air.sampler_states = !{!21}
!llvm.ident = !{!22}
!air.version = !{!23}
!air.language_version = !{!24}
!air.source_file_name = !{!25}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @fragment_unclamped_lod_at_four_scales, !10, !12}
!10 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!12 = !{!13, !14, !15, !16, !17}
!13 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!14 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t16"}
!15 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t64"}
!16 = !{i32 3, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t04"}
!17 = !{i32 4, !"air.texture", !"air.location_index", i32 3, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"t32"}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!22 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!23 = !{i32 2, i32 8, i32 0}
!24 = !{!"Metal", i32 4, i32 0, i32 0}
!25 = !{!"/private/tmp/lod/k.metal"}
