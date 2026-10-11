; Owned synthetic fixture for public drift / A/B samples.
; Not derived from a third-party metallib.
source_filename = "kernel_store_const.metal"

define void @store_const(ptr addrspace(1) %out) {
  store i32 42, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @store_const, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
