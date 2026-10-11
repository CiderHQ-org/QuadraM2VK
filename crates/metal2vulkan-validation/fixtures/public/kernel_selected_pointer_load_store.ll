; ModuleID = 'kernel_selected_pointer_load_store.ll'
source_filename = "validation/fixtures/public/kernel_selected_pointer_load_store.metal"
target triple = "spirv-unknown-vulkan1.2"

define void @selected_pointer_load_store(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %a, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %b, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out, i32 %gid) {
entry:
  %i = zext i32 %gid to i64
  %lo = icmp ult i32 %gid, 4
  %pa = getelementptr inbounds float, ptr addrspace(1) %a, i64 %i
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %i
  %p = select i1 %lo, ptr addrspace(1) %pa, ptr addrspace(1) %pb
  %v = load float, ptr addrspace(1) %p, align 4
  %k = add nuw nsw i32 %gid, 8
  %ki = zext i32 %k to i64
  %o1 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  %o2 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %ki
  %op = select i1 %lo, ptr addrspace(1) %o1, ptr addrspace(1) %o2
  store float %v, ptr addrspace(1) %op, align 4
  %base = select i1 %lo, ptr addrspace(1) %b, ptr addrspace(1) %a
  %e = getelementptr inbounds float, ptr addrspace(1) %base, i64 %i
  %v2 = load float, ptr addrspace(1) %e, align 4
  %m = add nuw nsw i32 %gid, 16
  %mi = zext i32 %m to i64
  %o3 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %mi
  store float %v2, ptr addrspace(1) %o3, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @selected_pointer_load_store, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
