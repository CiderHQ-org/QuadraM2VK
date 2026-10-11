; Owned synthetic fixture for authored irreducible control flow whose exit chooses between three
; DIFFERENT buffers. Not derived from a third-party metallib.
;
; Hand-written, because no source text makes the Metal front end emit an irreducible CFG. %a, %b and
; %c form one cycle and %entry's switch enters it at ALL THREE, so no header dominates it and SPIR-V
; structured control flow cannot spell it. `reloop_nest` declines and the state-machine construction
; in src/native/relooper.rs takes it.
;
; %exit's `%pf` is a POINTER phi whose three incomings are chains off three DIFFERENT bindings, so
; the address cannot be reduced to one base plus a selected index -- which is what happens in
; kernel_irreducible_three_way.ll, and why that fixture does not reach this code. Here the relooper
; must take its `remat_phi` path: load the phi's tag and fold `select(tag == i, arm_i, ...)` back
; into a pointer. Confirmed in the disassembly as two `OpSelect %_ptr_PhysicalStorageBuffer_uint`.
;
; The sibling .metal is the same automaton with one entry -- state in {0,1,2}, step
; `v = v*3 | v+7 | v^255`, `i += 1`, exit at `i >= n` storing v through `out0|out1|out2` chosen by
; state, else advance the state cyclically -- and Metal runs THAT. The two spellings differ only in
; control-flow shape.
source_filename = "validation/fixtures/public/kernel_irreducible_three_buffers.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @irreducible_three_buffers(ptr addrspace(1) %out0, ptr addrspace(1) %out1, ptr addrspace(1) %out2, ptr addrspace(1) %input, i32 %tid) {
entry:
  %slot = zext i32 %tid to i64
  %in_ptr = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %slot
  %n = load i32, ptr addrspace(1) %in_ptr, align 4
  %sel = urem i32 %n, 3
  switch i32 %sel, label %a [
    i32 1, label %b
    i32 2, label %c
  ]

a:
  %i_a = phi i32 [ 0, %entry ], [ %i_c_out, %c ]
  %v_a = phi i32 [ 10, %entry ], [ %v_c_out, %c ]
  %v_a_out = mul i32 %v_a, 3
  %i_a_out = add i32 %i_a, 1
  %p_a = getelementptr inbounds i32, ptr addrspace(1) %out0, i64 %slot
  %d_a = icmp uge i32 %i_a_out, %n
  br i1 %d_a, label %exit, label %b

b:
  %i_b = phi i32 [ 0, %entry ], [ %i_a_out, %a ]
  %v_b = phi i32 [ 20, %entry ], [ %v_a_out, %a ]
  %v_b_out = add i32 %v_b, 7
  %i_b_out = add i32 %i_b, 1
  %p_b = getelementptr inbounds i32, ptr addrspace(1) %out1, i64 %slot
  %d_b = icmp uge i32 %i_b_out, %n
  br i1 %d_b, label %exit, label %c

c:
  %i_c = phi i32 [ 0, %entry ], [ %i_b_out, %b ]
  %v_c = phi i32 [ 30, %entry ], [ %v_b_out, %b ]
  %v_c_out = xor i32 %v_c, 255
  %i_c_out = add i32 %i_c, 1
  %p_c = getelementptr inbounds i32, ptr addrspace(1) %out2, i64 %slot
  %d_c = icmp uge i32 %i_c_out, %n
  br i1 %d_c, label %exit, label %a

exit:
  %pf = phi ptr addrspace(1) [ %p_a, %a ], [ %p_b, %b ], [ %p_c, %c ]
  %vf = phi i32 [ %v_a_out, %a ], [ %v_b_out, %b ], [ %v_c_out, %c ]
  store i32 %vf, ptr addrspace(1) %pf, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @irreducible_three_buffers, !1, !2}
!1 = !{}
!2 = !{!3, !6, !7, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out0"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out1"}
!7 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out2"}
!4 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!5 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
