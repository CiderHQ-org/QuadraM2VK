; ModuleID = 'kernel_merged_pointer_at_a_constant_index.ll'
source_filename = "validation/fixtures/public/kernel_merged_pointer_at_a_constant_index.metal"
target triple = "spirv-unknown-vulkan1.2"

define void @merged_pointer_at_a_constant_index(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %a, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %b, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out, i32 %gid) {
entry:
  %lo = icmp ult i32 %gid, 4
  br i1 %lo, label %take_a, label %take_b

take_a:
  br label %merge

take_b:
  br label %merge

merge:
  %p = phi ptr addrspace(1) [ %a, %take_a ], [ %b, %take_b ]
  %i = zext i32 %gid to i64
  %e1 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1
  %v1 = load i32, ptr addrspace(1) %e1, align 4
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %v1, ptr addrspace(1) %o0, align 4
  %e3 = getelementptr inbounds i32, ptr addrspace(1) %p, i64 3
  %v3 = load i32, ptr addrspace(1) %e3, align 4
  %k = add nuw nsw i32 %gid, 8
  %ki = zext i32 %k to i64
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %ki
  store i32 %v3, ptr addrspace(1) %o1, align 4
  %ed = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %vd = load i32, ptr addrspace(1) %ed, align 4
  %m = add nuw nsw i32 %gid, 16
  %mi = zext i32 %m to i64
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %mi
  store i32 %vd, ptr addrspace(1) %o2, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @merged_pointer_at_a_constant_index, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
