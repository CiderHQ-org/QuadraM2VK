; ModuleID = 'kernel_subword_reinterpret_roundtrip.ll'
source_filename = "validation/fixtures/public/kernel_subword_reinterpret_roundtrip.metal"
target triple = "spirv-unknown-vulkan1.2"

define void @subword_reinterpret_roundtrip(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %in, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out, i32 %gid) {
entry:
  %i = zext i32 %gid to i64
  %pin = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %i
  %v = load <2 x i16>, ptr addrspace(1) %pin, align 4
  %lo = extractelement <2 x i16> %v, i64 0
  %hi = extractelement <2 x i16> %v, i64 1
  %lo32 = zext i16 %lo to i32
  %hi32 = zext i16 %hi to i32
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %lo32, ptr addrspace(1) %o0, align 4
  %k = add nuw nsw i32 %gid, 8
  %ki = zext i32 %k to i64
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %ki
  store i32 %hi32, ptr addrspace(1) %o1, align 4
  %w = shufflevector <2 x i16> %v, <2 x i16> poison, <2 x i32> <i32 1, i32 0>
  %m = add nuw nsw i32 %gid, 16
  %mi = zext i32 %m to i64
  %o2 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %mi
  store <2 x i16> %w, ptr addrspace(1) %o2, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @subword_reinterpret_roundtrip, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
