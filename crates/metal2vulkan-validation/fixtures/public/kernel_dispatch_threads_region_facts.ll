target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

; Every fact `KernelDispatchPlan` carries into a region, written per thread of a 10x3 grid.
; `thread_index_in_threadgroup` flattens with the region's own local size, so it is the one that
; moves when a threadgroup is short; the grid-wide facts must not move at all.
define void @kernel_dispatch_threads_region_facts(ptr addrspace(1) %output, <3 x i32> %gid, i32 %flat, <3 x i32> %threads_per_grid, <3 x i32> %groups_per_grid, <3 x i32> %group_position) {
entry:
  %x = extractelement <3 x i32> %gid, i64 0
  %y = extractelement <3 x i32> %gid, i64 1
  %row = mul i32 %y, 10
  %index = add i32 %row, %x
  %base = mul i32 %index, 4
  %base64 = zext i32 %base to i64

  %tpg_x = extractelement <3 x i32> %threads_per_grid, i64 0
  %tpg_y = extractelement <3 x i32> %threads_per_grid, i64 1
  %tpg_x10 = mul i32 %tpg_x, 10
  %tpg = add i32 %tpg_x10, %tpg_y

  %gpg_x = extractelement <3 x i32> %groups_per_grid, i64 0
  %gpg_y = extractelement <3 x i32> %groups_per_grid, i64 1
  %gpg_x10 = mul i32 %gpg_x, 10
  %gpg = add i32 %gpg_x10, %gpg_y

  %gp_x = extractelement <3 x i32> %group_position, i64 0
  %gp_y = extractelement <3 x i32> %group_position, i64 1
  %gp_x10 = mul i32 %gp_x, 10
  %gp = add i32 %gp_x10, %gp_y

  %slot0 = getelementptr i32, ptr addrspace(1) %output, i64 %base64
  store i32 %flat, ptr addrspace(1) %slot0, align 4
  %i1 = add i64 %base64, 1
  %slot1 = getelementptr i32, ptr addrspace(1) %output, i64 %i1
  store i32 %tpg, ptr addrspace(1) %slot1, align 4
  %i2 = add i64 %base64, 2
  %slot2 = getelementptr i32, ptr addrspace(1) %output, i64 %i2
  store i32 %gpg, ptr addrspace(1) %slot2, align 4
  %i3 = add i64 %base64, 3
  %slot3 = getelementptr i32, ptr addrspace(1) %output, i64 %i3
  store i32 %gp, ptr addrspace(1) %slot3, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel_dispatch_threads_region_facts, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint"}
!6 = !{i32 3, !"air.threads_per_grid", !"air.arg_type_name", !"uint3"}
!7 = !{i32 4, !"air.threadgroups_per_grid", !"air.arg_type_name", !"uint3"}
!8 = !{i32 5, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3"}
