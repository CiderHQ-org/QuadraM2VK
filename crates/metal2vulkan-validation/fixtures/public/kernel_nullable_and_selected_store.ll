; ModuleID = 'kernel_nullable_and_selected_store.ll'
source_filename = "validation/fixtures/public/kernel_nullable_and_selected_store.metal"
target triple = "spirv-unknown-vulkan1.2"

define void @nullable_and_selected_store(ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %c, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out, i32 %gid) {
entry:
  %i = zext i32 %gid to i64
  %lo = icmp ult i32 %gid, 4
  %pa = getelementptr inbounds float, ptr addrspace(1) %c, i64 %i
  %maybe = select i1 %lo, ptr addrspace(1) %pa, ptr addrspace(1) null
  %isnull = icmp eq ptr addrspace(1) %maybe, null
  %nv = select i1 %isnull, float 1.000000e+00, float 0.000000e+00
  %o0 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %i
  store float %nv, ptr addrspace(1) %o0, align 4
  %wbase = select i1 %lo, ptr addrspace(1) %c, ptr addrspace(1) %out
  %k = add nuw nsw i32 %gid, 8
  %ki = zext i32 %k to i64
  %we = getelementptr inbounds float, ptr addrspace(1) %wbase, i64 %ki
  %mv = uitofp i32 %k to float
  store float %mv, ptr addrspace(1) %we, align 4
  %ce = getelementptr inbounds float, ptr addrspace(1) %c, i64 %ki
  %cv = load float, ptr addrspace(1) %ce, align 4
  %m = add nuw nsw i32 %gid, 16
  %mi = zext i32 %m to i64
  %o2 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %mi
  store float %cv, ptr addrspace(1) %o2, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @nullable_and_selected_store, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
