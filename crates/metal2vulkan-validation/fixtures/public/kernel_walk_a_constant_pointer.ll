; Owned synthetic fixture, hand-written: the Metal front end strength-reduces the outer
; pointer step to `src + 4`, so no source text produces the nested pointer phi the corpus
; modules carry. The sibling .metal is a different spelling of the same eight-element sum;
; Metal runs that, and this is what the translator reads.
; Not derived from a third-party metallib.
target triple = "spirv-unknown-vulkan1.2"

define void @kernel_walk_a_constant_pointer(ptr addrspace(2) %src, ptr addrspace(1) %dst, i32 %gid) {
entry:
  br label %outer

outer:
  %oi = phi i32 [ 0, %entry ], [ %oi_next, %latch ]
  %p_outer = phi ptr addrspace(2) [ %src, %entry ], [ %p_step, %latch ]
  %acc_outer = phi float [ 0.000000e+00, %entry ], [ %acc_inner, %latch ]
  br label %inner

inner:
  %ii = phi i32 [ 0, %outer ], [ %ii_next, %inner ]
  %p_inner = phi ptr addrspace(2) [ %p_outer, %outer ], [ %p_step, %inner ]
  %acc_in = phi float [ %acc_outer, %outer ], [ %acc_next, %inner ]
  %v = load float, ptr addrspace(2) %p_inner, align 4
  %acc_next = fadd float %acc_in, %v
  %p_step = getelementptr inbounds float, ptr addrspace(2) %p_inner, i64 1
  %ii_next = add i32 %ii, 1
  %done_in = icmp sge i32 %ii_next, 4
  br i1 %done_in, label %latch, label %inner

latch:
  %acc_inner = phi float [ %acc_next, %inner ]
  %oi_next = add i32 %oi, 1
  %done_out = icmp sge i32 %oi_next, 2
  br i1 %done_out, label %exit, label %outer

exit:
  store float %acc_inner, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @kernel_walk_a_constant_pointer, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
