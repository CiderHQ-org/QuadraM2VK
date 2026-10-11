; Owned synthetic fixture for literal primitive acceleration-structure binding.
; Not derived from a third-party metallib.
source_filename = "kernel_primitive_as_word.metal"

define void @primitive_as_word(ptr addrspace(1) %acceleration_structure, ptr addrspace(1) %output) {
  store i32 42, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @primitive_as_word, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.primitive_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<>", !"air.arg_name", !"acceleration_structure"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
