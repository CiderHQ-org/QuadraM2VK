; Owned synthetic fixture for authored bounded-loop execution.
; Not derived from a third-party metallib.
source_filename = "kernel_bounded_loop_word.metal"

define void @bounded_loop_word(ptr addrspace(1) %output, ptr addrspace(1) %count) {
entry:
  %limit = load i32, ptr addrspace(1) %count, align 4
  br label %loop

loop:
  %index = phi i32 [ 0, %entry ], [ %next_index, %body ]
  %sum = phi i32 [ 0, %entry ], [ %next_sum, %body ]
  %continue = icmp ult i32 %index, %limit
  br i1 %continue, label %body, label %exit

body:
  %term = add i32 %index, 1
  %next_sum = add i32 %sum, %term
  %next_index = add i32 %index, 1
  br label %loop

exit:
  store i32 %sum, ptr addrspace(1) %output, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @bounded_loop_word, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"output"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"count"}
