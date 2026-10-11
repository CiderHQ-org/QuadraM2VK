; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = '/tmp/amp/f.bc'
source_filename = "/tmp/amp/vertex_amplification_id_layer.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
define <{ <4 x float>, i32, <4 x float> }> @vertex_amplification_id_layer(i32 noundef %0, i16 noundef %1) local_unnamed_addr #0 {
  %3 = shl i32 %0, 1
  %4 = and i32 %3, 2
  %5 = tail call fast float @air.convert.f.f32.u.i32(i32 %4) #2
  %6 = insertelement <2 x float> undef, float %5, i64 0
  %7 = and i32 %0, 2
  %8 = tail call fast float @air.convert.f.f32.u.i32(i32 %7) #2
  %9 = insertelement <2 x float> %6, float %8, i64 1
  %10 = fmul fast <2 x float> %9, splat (float 2.000000e+00)
  %11 = fadd fast <2 x float> %10, splat (float -1.000000e+00)
  %12 = shufflevector <2 x float> %11, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %13 = shufflevector <4 x float> %12, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %14 = zext i16 %1 to i32
  %15 = tail call fast float @air.convert.f.f32.u.i16(i16 %1) #2
  %16 = fadd fast float %15, 1.250000e+00
  %17 = insertelement <4 x float> <float poison, float 2.500000e+00, float 3.750000e+00, float 5.000000e+00>, float %16, i64 0
  %18 = insertvalue <{ <4 x float>, i32, <4 x float> }> undef, <4 x float> %13, 0
  %19 = insertvalue <{ <4 x float>, i32, <4 x float> }> %18, i32 %14, 1
  %20 = insertvalue <{ <4 x float>, i32, <4 x float> }> %19, <4 x float> %17, 2
  ret <{ <4 x float>, i32, <4 x float> }> %20
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i16(i16) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(none) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.vertex = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @vertex_amplification_id_layer, !10, !14}
!10 = !{!11, !12, !13}
!11 = !{!"air.position", !"air.shared", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!12 = !{!"air.render_target_array_index", !"air.arg_type_name", !"uint", !"air.arg_name", !"layer"}
!13 = !{!"air.vertex_output", !"user(locn0)", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!14 = !{!15, !16}
!15 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!16 = !{i32 1, !"air.amplification_id", !"air.arg_type_name", !"ushort", !"air.arg_name", !"amp"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 4, i32 0, i32 0}
!23 = !{!"/private/tmp/amp/vertex_amplification_id_layer.metal"}
