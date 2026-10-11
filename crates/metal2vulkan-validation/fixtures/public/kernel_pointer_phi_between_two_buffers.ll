; ModuleID = 'kernel_pointer_phi_between_two_buffers.ll'
source_filename = "validation/fixtures/public/kernel_pointer_phi_between_two_buffers.metal"
target triple = "spirv-unknown-vulkan1.2"

define void @pointer_phi_between_two_buffers(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %a, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %b, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out, i32 %gid) {
entry:
  %lo = icmp ult i32 %gid, 4
  br i1 %lo, label %take_a, label %take_b

take_a:
  br label %merge

take_b:
  br label %merge

merge:
  %p = phi ptr addrspace(1) [ %a, %take_a ], [ %b, %take_b ]
  %q = phi ptr addrspace(1) [ %b, %take_a ], [ %a, %take_b ]
  %i = zext i32 %gid to i64
  %e0 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %v0 = load i32, ptr addrspace(1) %e0, align 4
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %v0, ptr addrspace(1) %o0, align 4
  %f0 = getelementptr inbounds i32, ptr addrspace(1) %q, i64 %i
  %w0 = load i32, ptr addrspace(1) %f0, align 4
  %k = add nuw nsw i32 %gid, 8
  %ki = zext i32 %k to i64
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %ki
  store i32 %w0, ptr addrspace(1) %o1, align 4
  %rev = sub nuw nsw i32 7, %gid
  %ri = zext i32 %rev to i64
  %e1 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %ri
  %v1 = load i32, ptr addrspace(1) %e1, align 4
  %m = add nuw nsw i32 %gid, 16
  %mi = zext i32 %m to i64
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %mi
  store i32 %v1, ptr addrspace(1) %o2, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @pointer_phi_between_two_buffers, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
