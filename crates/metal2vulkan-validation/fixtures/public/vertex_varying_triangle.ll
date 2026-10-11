; Owned synthetic fixture for vertex-input and varying-observer A/B.
; Not derived from a third-party metallib.
source_filename = "vertex_varying_triangle.metal"

define <{ <4 x float>, <2 x float> }> @vertex_varying_triangle(<2 x float> %position, <2 x float> %uv) {
  %extended = shufflevector <2 x float> %position, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %clip = shufflevector <4 x float> %extended, <4 x float> <float poison, float poison, float 0.000000e+00, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %with_position = insertvalue <{ <4 x float>, <2 x float> }> undef, <4 x float> %clip, 0
  %output = insertvalue <{ <4 x float>, <2 x float> }> %with_position, <2 x float> %uv, 1
  ret <{ <4 x float>, <2 x float> }> %output
}

!air.vertex = !{!0}
!0 = !{ptr @vertex_varying_triangle, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(2uvDv2_f)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"position"}
!6 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
