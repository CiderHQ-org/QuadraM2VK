target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

; The simdgroup facts inside a threadgroup that a partial dispatch made short. Metal keeps
; `threads_per_simdgroup` at the hardware width and counts `simdgroups_per_threadgroup` from the
; group that actually runs, so both are constant here while the group shrinks from 8 threads to 2.
define void @kernel_dispatch_threads_short_group_simd(ptr addrspace(1) %output, <3 x i32> %gid, i32 %lane, i32 %groups, i32 %width, i32 %group_index) {
entry:
  %x = extractelement <3 x i32> %gid, i64 0
  %y = extractelement <3 x i32> %gid, i64 1
  %row = mul i32 %y, 10
  %index = add i32 %row, %x
  %base = mul i32 %index, 3
  %base64 = zext i32 %base to i64

  %groups100 = mul i32 %groups, 100
  %packed = add i32 %groups100, %width

  %slot0 = getelementptr i32, ptr addrspace(1) %output, i64 %base64
  store i32 %lane, ptr addrspace(1) %slot0, align 4
  %i1 = add i64 %base64, 1
  %slot1 = getelementptr i32, ptr addrspace(1) %output, i64 %i1
  store i32 %packed, ptr addrspace(1) %slot1, align 4
  %i2 = add i64 %base64, 2
  %slot2 = getelementptr i32, ptr addrspace(1) %output, i64 %i2
  store i32 %group_index, ptr addrspace(1) %slot2, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel_dispatch_threads_short_group_simd, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3"}
!5 = !{i32 2, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint"}
!6 = !{i32 3, !"air.simdgroups_per_threadgroup", !"air.arg_type_name", !"uint"}
!7 = !{i32 4, !"air.threads_per_simdgroup", !"air.arg_type_name", !"uint"}
!8 = !{i32 5, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint"}
