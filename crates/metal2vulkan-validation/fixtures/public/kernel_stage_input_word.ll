; Owned synthetic fixture for the authored kernel stage-input execution contract.
; Not derived from a third-party metallib.
source_filename = "kernel_stage_input_word.metal"

define void @stage_input_word(<3 x i32> %input, ptr addrspace(1) %output) {
  %value = extractelement <3 x i32> %input, i64 0
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @stage_input_word, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.stage_in", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"uint3", !"air.arg_name", !"input"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
