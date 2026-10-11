; Owned synthetic fixture for public drift / A/B samples.
; Not derived from a third-party metallib.
source_filename = "kernel_invert_word.metal"

define void @invert_word(ptr addrspace(1) %value) {
  %old = load i32, ptr addrspace(1) %value, align 4
  %new = xor i32 %old, -1
  store i32 %new, ptr addrspace(1) %value, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @invert_word, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"value"}
