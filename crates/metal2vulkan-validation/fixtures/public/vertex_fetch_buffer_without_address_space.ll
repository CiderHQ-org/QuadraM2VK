; Owned synthetic fixture: a vertex-fetch entry whose vertex buffer's node states no
; `air.address_space`. Not derived from a third-party metallib.
;
; Metal's runtime can compile a vertex function with a vertex descriptor into an entry that
; fetches the stage-in attributes from the bound vertex buffers and calls the function. That
; entry's metadata names each such buffer `void` and states no address space, in current AIR
; (2.7) as in the AIR 2.2 of older compilers: the parameter's pointer type carries the space, and
; the decoder takes it from there. Each entry fetches two `float2` attributes from a 16-byte
; vertex at its vertex id and calls `fetch_vertex`, which scales the first by a constant `float4`;
; `fetch_from_device` reads a device buffer at parameter 1, `fetch_from_constant` a constant one
; at parameter 2.
source_filename = "vertex_fetch_buffer_without_address_space.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64-apple-macosx26.0.0"

define internal <{ <4 x float>, <2 x float> }> @fetch_vertex(<2 x float> %pos, <2 x float> %tex, ptr addrspace(2) %scale) {
  %s = load <4 x float>, ptr addrspace(2) %scale, align 16
  %p4 = shufflevector <2 x float> %pos, <2 x float> <float 0.0, float 1.0>, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %out = fmul <4 x float> %p4, %s
  %r0 = insertvalue <{ <4 x float>, <2 x float> }> undef, <4 x float> %out, 0
  %r1 = insertvalue <{ <4 x float>, <2 x float> }> %r0, <2 x float> %tex, 1
  ret <{ <4 x float>, <2 x float> }> %r1
}

define <{ <4 x float>, <2 x float> }> @fetch_from_device(ptr addrspace(2) %scale, ptr addrspace(1) %vertices, i32 zeroext %vid) {
  %i = zext i32 %vid to i64
  %pp = getelementptr <{ [16 x i8] }>, ptr addrspace(1) %vertices, i64 %i, i32 0, i64 0
  %pos = load <2 x float>, ptr addrspace(1) %pp, align 4
  %tp = getelementptr <{ [16 x i8] }>, ptr addrspace(1) %vertices, i64 %i, i32 0, i64 8
  %tex = load <2 x float>, ptr addrspace(1) %tp, align 4
  %r = call <{ <4 x float>, <2 x float> }> @fetch_vertex(<2 x float> %pos, <2 x float> %tex, ptr addrspace(2) %scale)
  ret <{ <4 x float>, <2 x float> }> %r
}

define <{ <4 x float>, <2 x float> }> @fetch_from_constant(ptr addrspace(2) %scale, i32 zeroext %vid, ptr addrspace(2) %vertices) {
  %i = zext i32 %vid to i64
  %pp = getelementptr <{ [16 x i8] }>, ptr addrspace(2) %vertices, i64 %i, i32 0, i64 0
  %pos = load <2 x float>, ptr addrspace(2) %pp, align 4
  %tp = getelementptr <{ [16 x i8] }>, ptr addrspace(2) %vertices, i64 %i, i32 0, i64 8
  %tex = load <2 x float>, ptr addrspace(2) %tp, align 4
  %r = call <{ <4 x float>, <2 x float> }> @fetch_vertex(<2 x float> %pos, <2 x float> %tex, ptr addrspace(2) %scale)
  ret <{ <4 x float>, <2 x float> }> %r
}

!air.vertex = !{!0, !8}
!air.version = !{!12}
!air.language_version = !{!13}

!0 = !{ptr @fetch_from_device, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!3 = !{!"air.vertex_output", !"user(locn0)", !"air.arg_type_name", !"float2", !"air.arg_name", !"tex"}
!4 = !{!5, !6, !7}
!5 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"scale"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"void", !"air.arg_name", !"vertices"}
!7 = !{i32 2, !"air.vertex_id", !"air.arg_type_name", !"uint"}
!8 = !{ptr @fetch_from_constant, !1, !9}
!9 = !{!5, !10, !11}
!10 = !{i32 1, !"air.vertex_id", !"air.arg_type_name", !"uint"}
!11 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"void", !"air.arg_name", !"vertices"}
!12 = !{i32 2, i32 7, i32 0}
!13 = !{!"Metal", i32 3, i32 2, i32 0}
