; ModuleID = 'kernel_compare_two_element_pointers.ll'
source_filename = "validation/fixtures/public/kernel_compare_two_element_pointers.metal"
target triple = "spirv-unknown-vulkan1.2"

define void @compare_two_element_pointers(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %buf, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out, i32 %gid) {
entry:
  %i = zext i32 %gid to i64
  %half = lshr i32 %gid, 1
  %even = shl i32 %half, 1
  %j = zext i32 %even to i64
  %p = getelementptr inbounds float, ptr addrspace(1) %buf, i64 %i
  %q = getelementptr inbounds float, ptr addrspace(1) %buf, i64 %j
  %r = getelementptr inbounds float, ptr addrspace(1) %buf, i64 3
  %pv = load float, ptr addrspace(1) %p, align 4
  %rv = load float, ptr addrspace(1) %r, align 4
  %eq = icmp eq ptr addrspace(1) %p, %q
  %v = select i1 %eq, float 1.000000e+02, float 0.000000e+00
  %sum = fadd float %v, %pv
  %o = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %sum, ptr addrspace(1) %o, align 4
  %ne = icmp ne ptr addrspace(1) %p, %r
  %v2 = select i1 %ne, float 1.000000e+03, float 0.000000e+00
  %sum2 = fadd float %v2, %rv
  %k = add nuw nsw i32 %gid, 8
  %ki = zext i32 %k to i64
  %o2 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %ki
  store float %sum2, ptr addrspace(1) %o2, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @compare_two_element_pointers, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
