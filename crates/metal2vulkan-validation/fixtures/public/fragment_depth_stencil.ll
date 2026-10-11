target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ float, i32 }> @fragment_depth_stencil() {
  %depth = insertvalue <{ float, i32 }> undef, float 5.000000e-01, 0
  %stencil = insertvalue <{ float, i32 }> %depth, i32 7, 1
  ret <{ float, i32 }> %stencil
}

!air.fragment = !{!0}
!0 = !{ptr @fragment_depth_stencil, !1, !2}
!1 = !{!3, !4}
!2 = !{}
!3 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float"}
!4 = !{!"air.stencil", !"air.arg_type_name", !"uint"}
