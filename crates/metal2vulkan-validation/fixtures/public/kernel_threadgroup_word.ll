; Owned synthetic fixture for the authored threadgroup-memory execution contract.
; Not derived from a third-party metallib.
source_filename = "kernel_threadgroup_word.metal"

define void @threadgroup_word(ptr addrspace(3) %scratch, ptr addrspace(1) %output) {
  store i32 42, ptr addrspace(3) %scratch, align 4
  %value = load i32, ptr addrspace(3) %scratch, align 4
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @threadgroup_word, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_name", !"uint", !"air.arg_name", !"scratch"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
