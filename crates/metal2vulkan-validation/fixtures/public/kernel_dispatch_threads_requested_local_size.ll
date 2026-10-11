target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

; Writes both threadgroup-size facts for every thread of a 10x3 grid. Under `dispatchThreads:` a
; tail region's threadgroup really is shorter, so `air.threads_per_threadgroup` varies by region
; while `air.dispatch_threads_per_threadgroup` stays at the size the dispatch asked for.
define void @kernel_dispatch_threads_requested_local_size(ptr addrspace(1) %output, <3 x i32> %gid, <3 x i32> %local_size, <3 x i32> %requested_size) {
entry:
  %x = extractelement <3 x i32> %gid, i64 0
  %y = extractelement <3 x i32> %gid, i64 1
  %row = mul i32 %y, 10
  %index = add i32 %row, %x
  %pair = mul i32 %index, 2
  %pair64 = zext i32 %pair to i64
  %local_x = extractelement <3 x i32> %local_size, i64 0
  %local_y = extractelement <3 x i32> %local_size, i64 1
  %local_y_scaled = mul i32 %local_y, 100
  %local = add i32 %local_y_scaled, %local_x
  %requested_x = extractelement <3 x i32> %requested_size, i64 0
  %requested_y = extractelement <3 x i32> %requested_size, i64 1
  %requested_y_scaled = mul i32 %requested_y, 100
  %requested = add i32 %requested_y_scaled, %requested_x
  %local_slot = getelementptr i32, ptr addrspace(1) %output, i64 %pair64
  store i32 %local, ptr addrspace(1) %local_slot, align 4
  %requested_index = add i64 %pair64, 1
  %requested_slot = getelementptr i32, ptr addrspace(1) %output, i64 %requested_index
  store i32 %requested, ptr addrspace(1) %requested_slot, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel_dispatch_threads_requested_local_size, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3"}
!5 = !{i32 2, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3"}
!6 = !{i32 3, !"air.dispatch_threads_per_threadgroup", !"air.arg_type_name", !"uint3"}
