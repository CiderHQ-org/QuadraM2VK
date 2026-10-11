; ModuleID = 'kernel_cloned_continuation_uniform_read.ll'
; Owned synthetic fixture, hand-written because the CFG is the point: a nested selection whose
; in-arm continuation the enclosing arm also reaches. Its metadata is in Metal's ABI grammar; a
; nested layout is named by `!"air.struct_type_info", !n` before its member, as the Metal front end
; writes it (compare new_engine/kernel_cloned_continuation_uniform_read.ll).
source_filename = "validation/fixtures/public/kernel_cloned_continuation_uniform_read.metal"
target triple = "spirv-unknown-vulkan1.2"

%struct.Coef = type { float, float, float, float }
%struct.Params = type { %struct.Coef, %struct.Coef }

define void @cloned_continuation_uniform_read(ptr addrspace(2) %params, ptr addrspace(1) %out, i32 %gid) {
entry:
  %c0 = icmp ult i32 %gid, 4
  br i1 %c0, label %outerT, label %outerF

outerT:
  %c1 = icmp ult i32 %gid, 2
  br i1 %c1, label %innerA, label %innerB

innerA:
  br label %cont

innerB:
  br label %merge

outerF:
  br label %cont

cont:
  %m = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %r = call fastcc float @coef(ptr addrspace(2) %m, float 5.000000e-01)
  br label %merge

merge:
  %v = phi float [ %r, %cont ], [ 7.000000e+00, %innerB ]
  %slot = zext i32 %gid to i64
  %op = getelementptr inbounds float, ptr addrspace(1) %out, i64 %slot
  store float %v, ptr addrspace(1) %op, align 4
  ret void
}

define internal fastcc float @coef(ptr addrspace(2) %c, float %x) {
head:
  %p2 = getelementptr inbounds %struct.Coef, ptr addrspace(2) %c, i64 0, i32 2
  %v2 = load float, ptr addrspace(2) %p2, align 4
  %s = fadd float %v2, %x
  ret float %s
}

!air.kernel = !{!0}
!0 = !{ptr @cloned_continuation_uniform_read, !1, !2}
!1 = !{}
!2 = !{!3, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 16, i32 0, !"Coef", !"a", !"air.struct_type_info", !6, i32 16, i32 16, i32 0, !"Coef", !"b"}
!5 = !{i32 0, i32 4, i32 0, !"float", !"c0", i32 4, i32 4, i32 0, !"float", !"c1", i32 8, i32 4, i32 0, !"float", !"c2", i32 12, i32 4, i32 0, !"float", !"c3"}
!6 = !{i32 0, i32 4, i32 0, !"float", !"c0", i32 4, i32 4, i32 0, !"float", !"c1", i32 8, i32 4, i32 0, !"float", !"c2", i32 12, i32 4, i32 0, !"float", !"c3"}
!7 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!8 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
