; Owned synthetic fixture for fragment render-pipeline A/B.
; Not derived from a third-party metallib.
source_filename = "fragment_constant_color.metal"

define <4 x float> @fragment_constant_color() {
  ret <4 x float> <float 2.500000e-01, float 5.000000e-01, float 7.500000e-01, float 1.000000e+00>
}

!air.fragment = !{!0}
!0 = !{ptr @fragment_constant_color, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{}
