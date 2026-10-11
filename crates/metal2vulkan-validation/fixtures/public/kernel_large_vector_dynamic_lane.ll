; Hand-written. The Metal side of this fixture is a DIFFERENT SPELLING of the same computation:
; `kernel_large_vector_dynamic_lane.metal` writes and reads a `thread uint v[8]`, while this module
; keeps the eight lanes in SSA and uses `insertelement`/`extractelement` with a dynamic index. Both
; start from the same eight constants, write `tid + 1000` at the same position and read the same two
; positions; only the storage spelling differs. See the case rationale for the derivation.
;
; The shape this reaches: a dynamic lane index into an EIGHT-lane vector. Vulkan's
; `OpVectorExtractDynamic`/`OpVectorInsertDynamic` only cover two-, three- and four-component
; vectors, so `emitter::body::core_inst::emit_large_vector_dynamic_extract` and
; `emit_large_vector_dynamic_insert` emulate the lane with a chain of `OpIEqual`/`OpSelect` over
; every lane. 93 corpus sources contain a dynamic `extractelement` on an `<8 x ...>`; no authored
; case reached either function.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

define void @kernel_large_vector_dynamic_lane(ptr addrspace(1) %0, i32 %1) local_unnamed_addr #0 {
entry:
  %ins = and i32 %1, 7
  %scaled = mul i32 %1, 5
  %ext = and i32 %scaled, 7
  %val = add i32 %1, 1000
  %w = insertelement <8 x i32> <i32 1, i32 11, i32 21, i32 31, i32 41, i32 51, i32 61, i32 71>, i32 %val, i32 %ins
  %a = extractelement <8 x i32> %w, i32 %ext
  %b = extractelement <8 x i32> %w, i32 %ins
  %o0 = shl i32 %1, 1
  %i0 = zext i32 %o0 to i64
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %i0
  store i32 %a, ptr addrspace(1) %p0, align 4
  %o1 = or i32 %o0, 1
  %i1 = zext i32 %o1 to i64
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %i1
  store i32 %b, ptr addrspace(1) %p1, align 4
  ret void
}

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "no-builtins" }

!air.kernel = !{!0}
!air.version = !{!10}
!air.language_version = !{!11}

!0 = !{ptr @kernel_large_vector_dynamic_lane, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!10 = !{i32 2, i32 8, i32 0}
!11 = !{!"Metal", i32 3, i32 0, i32 0}
