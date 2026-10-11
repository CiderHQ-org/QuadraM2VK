; Owned synthetic fixture for authored irreducible-control-flow execution.
; Not derived from a third-party metallib.
;
; Hand-written, because no source text makes the Metal front end emit an irreducible CFG. Blocks
; %a and %b form a cycle that %entry enters at BOTH of them, so no loop header dominates the cycle
; and SPIR-V's structured control flow cannot spell it. That is what routes this module to the
; state-machine construction in src/native/relooper.rs, whose SSA spill/restore and phi-edge stores
; no other authored case reaches.
;
; The sibling .metal is the same automaton with one entry -- state (i, acc, in_b), step
; `acc = in_b ? acc+5 : acc*3; i += 1; exit when i >= n; else flip in_b` -- and Metal runs THAT.
; The two spellings differ only in control-flow shape; every arithmetic operation is identical.
source_filename = "validation/fixtures/public/kernel_irreducible_ping_pong.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @irreducible_ping_pong(ptr addrspace(1) %output, ptr addrspace(1) %input, i32 %tid) {
entry:
  %slot = zext i32 %tid to i64
  %in_ptr = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %slot
  %n = load i32, ptr addrspace(1) %in_ptr, align 4
  %odd = and i32 %n, 1
  %start_in_b = icmp eq i32 %odd, 1
  br i1 %start_in_b, label %b, label %a

; multiply arm: reached from %entry when n is even, and from %b thereafter
a:
  %a_i = phi i32 [ 0, %entry ], [ %b_next_i, %b ]
  %a_acc = phi i32 [ 1, %entry ], [ %b_acc_out, %b ]
  %a_acc_out = mul i32 %a_acc, 3
  %a_next_i = add i32 %a_i, 1
  %a_done = icmp uge i32 %a_next_i, %n
  br i1 %a_done, label %exit, label %b

; add arm: reached from %entry when n is odd, and from %a thereafter
b:
  %b_i = phi i32 [ 0, %entry ], [ %a_next_i, %a ]
  %b_acc = phi i32 [ 2, %entry ], [ %a_acc_out, %a ]
  %b_acc_out = add i32 %b_acc, 5
  %b_next_i = add i32 %b_i, 1
  %b_done = icmp uge i32 %b_next_i, %n
  br i1 %b_done, label %exit, label %a

exit:
  %result = phi i32 [ %a_acc_out, %a ], [ %b_acc_out, %b ]
  %out_ptr = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %slot
  store i32 %result, ptr addrspace(1) %out_ptr, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @irreducible_ping_pong, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"input"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
