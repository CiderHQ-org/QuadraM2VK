; Owned synthetic fixture for generated fragment-varying A/B.
; Not derived from a third-party metallib.
source_filename = "fragment_varying_color.metal"

define <4 x float> @fragment_varying_color(<4 x float> %position, <2 x float> %uv) {
  %extended = shufflevector <2 x float> %uv, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %color = shufflevector <4 x float> %extended, <4 x float> <float poison, float poison, float 5.000000e-01, float 1.000000e+00>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  ret <4 x float> %color
}

!air.fragment = !{!0}
!0 = !{ptr @fragment_varying_color, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position", !"air.arg_unused"}
!5 = !{i32 1, !"air.fragment_input", !"generated(2uvDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
