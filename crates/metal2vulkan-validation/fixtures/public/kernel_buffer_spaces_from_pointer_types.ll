; Owned synthetic fixture: buffer arguments whose nodes state no `air.address_space`.
; Not derived from a third-party metallib.
;
; An older Metal compiler (AIR 2.2, as macOS 13's system libraries carry it) writes a buffer
; argument's node without the `air.address_space` key: the parameter's pointer type is the only
; carrier of the space. The decoder takes it from there, since a node that states one must state
; that one. `buffer_spaces_kernel` reads a constant struct and writes a device buffer whose node
; names only `void` (no size); `buffer_spaces_vertex` reads its position from a device buffer by
; vertex id and adds a constant offset.
source_filename = "kernel_buffer_spaces_from_pointer_types.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx13.0.0"

%struct.Params = type { i32, i32 }

define void @buffer_spaces_kernel(ptr addrspace(1) %out, ptr addrspace(2) %params, i32 %gid) {
  %scale = load i32, ptr addrspace(2) %params, align 4
  %bias.p = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %bias = load i32, ptr addrspace(2) %bias.p, align 4
  %m = mul i32 %gid, %scale
  %v = add i32 %m, %bias
  %idx = zext i32 %gid to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}

define <{ <4 x float> }> @buffer_spaces_vertex(ptr addrspace(1) %vertices, i32 %vid, ptr addrspace(2) %offset) {
  %idx = zext i32 %vid to i64
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %vertices, i64 %idx
  %pos = load <4 x float>, ptr addrspace(1) %p, align 16
  %o = load <4 x float>, ptr addrspace(2) %offset, align 16
  %s = fadd <4 x float> %pos, %o
  %r = insertvalue <{ <4 x float> }> undef, <4 x float> %s, 0
  ret <{ <4 x float> }> %r
}

!air.kernel = !{!0}
!air.vertex = !{!7}
!air.version = !{!14}
!air.language_version = !{!15}

!0 = !{ptr @buffer_spaces_kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"void", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.struct_type_info", !5, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"scale", i32 4, i32 4, i32 0, !"uint", !"bias"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!7 = !{ptr @buffer_spaces_vertex, !8, !10}
!8 = !{!9}
!9 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!10 = !{!11, !12, !13}
!11 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"vertices"}
!12 = !{i32 1, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!13 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"offset"}
!14 = !{i32 2, i32 2, i32 0}
!15 = !{!"Metal", i32 2, i32 2, i32 0}
