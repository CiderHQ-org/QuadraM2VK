target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

@vector_value.MTL_FC_INIT_0_Dv4_j = internal addrspace(2) externally_initialized constant <4 x i32> undef, section "air.fc_initializer", align 16

define void @kernel_vector_function_constant(ptr addrspace(1) %output) {
  %value = load <4 x i32>, ptr addrspace(2) @vector_value.MTL_FC_INIT_0_Dv4_j, align 16
  %x = extractelement <4 x i32> %value, i64 0
  %y = extractelement <4 x i32> %value, i64 1
  %z = extractelement <4 x i32> %value, i64 2
  %w = extractelement <4 x i32> %value, i64 3
  %xy = add i32 %x, %y
  %xyz = add i32 %xy, %z
  %sum = add i32 %xyz, %w
  store i32 %sum, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!4}
!0 = !{ptr @kernel_vector_function_constant, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{ptr addrspace(2) @vector_value.MTL_FC_INIT_0_Dv4_j, !"uint4", !"vector_value", i32 0}
