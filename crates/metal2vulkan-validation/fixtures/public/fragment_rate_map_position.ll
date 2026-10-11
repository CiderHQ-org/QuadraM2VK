; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -x metal
; -S -emit-llvm -` (stdin) and round-tripped through LLVM 21's llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'fragment_rate_map_position.bc'
source_filename = "-"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
define <4 x float> @fragment_rate_map_position(<4 x float> %0, <2 x float> %1, ptr addrspace(2) noundef readonly align 8 captures(none) dereferenceable(8) "air-buffer-no-alias" %2) local_unnamed_addr #0 {
  %4 = extractelement <2 x float> %1, i64 1
  %5 = fsub fast float 1.000000e+00, %4
  %6 = insertelement <2 x float> %1, float %5, i64 1
  %7 = load <2 x float>, ptr addrspace(2) %2, align 8, !tbaa !22, !alias.scope !25
  %8 = fmul fast <2 x float> %7, %6
  %9 = tail call fast <2 x float> @air.fast_floor.v2f32(<2 x float> %8) #2
  %10 = shufflevector <2 x float> %9, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %11 = shufflevector <4 x float> %0, <4 x float> %10, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  ret <4 x float> %11
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x float> @air.fast_floor.v2f32(<2 x float>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(none) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.fragment = !{!9}
!air.compile_options = !{!16, !17, !18}
!llvm.ident = !{!19}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @fragment_rate_map_position, !10, !12}
!10 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!12 = !{!13, !14, !15}
!13 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!14 = !{i32 1, !"air.fragment_input", !"generated(2uvDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!15 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"screen"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!23, !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(2)"}
!27 = distinct !{!27, !"air-alias-scopes(fragment_rate_map_position)"}
