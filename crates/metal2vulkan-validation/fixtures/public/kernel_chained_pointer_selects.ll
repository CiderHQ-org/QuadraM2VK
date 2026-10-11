; ModuleID = 'kernel_chained_pointer_selects.ll'
source_filename = "validation/fixtures/public/kernel_chained_pointer_selects.metal"
target triple = "spirv-unknown-vulkan1.2"

define void @chained_pointer_selects(ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %a, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %b, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %c, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out, i32 %gid) {
entry:
  %odd = and i32 %gid, 1
  %isodd = icmp ne i32 %odd, 0
  %p = select i1 %isodd, ptr addrspace(1) %a, ptr addrspace(1) %b
  %lo = icmp ult i32 %gid, 4
  %q = select i1 %lo, ptr addrspace(1) %p, ptr addrspace(1) %c
  %i = zext i32 %gid to i64
  %e0 = getelementptr inbounds i32, ptr addrspace(1) %q, i64 %i
  %v0 = load i32, ptr addrspace(1) %e0, align 4
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %v0, ptr addrspace(1) %o0, align 4
  %e1 = getelementptr inbounds i32, ptr addrspace(1) %q, i64 1
  %v1 = load i32, ptr addrspace(1) %e1, align 4
  %k8 = add nuw nsw i32 %gid, 8
  %k8i = zext i32 %k8 to i64
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %k8i
  store i32 %v1, ptr addrspace(1) %o1, align 4
  %w = add nuw nsw i32 %gid, 500
  %slot = getelementptr inbounds i32, ptr addrspace(1) %q, i64 %k8i
  store i32 %w, ptr addrspace(1) %slot, align 4
  %ra = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %k8i
  %va = load i32, ptr addrspace(1) %ra, align 4
  %k16 = add nuw nsw i32 %gid, 16
  %k16i = zext i32 %k16 to i64
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %k16i
  store i32 %va, ptr addrspace(1) %o2, align 4
  %rb = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %k8i
  %vb = load i32, ptr addrspace(1) %rb, align 4
  %k24 = add nuw nsw i32 %gid, 24
  %k24i = zext i32 %k24 to i64
  %o3 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %k24i
  store i32 %vb, ptr addrspace(1) %o3, align 4
  %rc = getelementptr inbounds i32, ptr addrspace(1) %c, i64 %k8i
  %vc = load i32, ptr addrspace(1) %rc, align 4
  %k32 = add nuw nsw i32 %gid, 32
  %k32i = zext i32 %k32 to i64
  %o4 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %k32i
  store i32 %vc, ptr addrspace(1) %o4, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @chained_pointer_selects, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"c"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!7 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
