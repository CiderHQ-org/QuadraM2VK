; ModuleID = 'kernel_zero_a_struct_prefix.ll'
source_filename = "validation/fixtures/public/kernel_zero_a_struct_prefix.metal"
target triple = "spirv-unknown-vulkan1.2"

%struct.Box = type { <2 x float>, float, float }

define void @zero_a_struct_prefix(ptr addrspace(2) noundef readonly align 8 dereferenceable(16) "air-buffer-no-alias" %src, ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %out) {
entry:
  %a = alloca %struct.Box, align 8
  %b = alloca %struct.Box, align 8
  call void @llvm.memcpy.p0.p2.i64(ptr noundef nonnull align 8 dereferenceable(16) %a, ptr addrspace(2) noundef align 8 dereferenceable(16) %src, i64 16, i1 false)
  call void @llvm.memcpy.p0.p2.i64(ptr noundef nonnull align 8 dereferenceable(16) %b, ptr addrspace(2) noundef align 8 dereferenceable(16) %src, i64 16, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(4) %a, i8 0, i64 4, i1 false)
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(12) %b, i8 0, i64 12, i1 false)
  %av = getelementptr inbounds %struct.Box, ptr %a, i64 0, i32 0
  %avv = load <2 x float>, ptr %av, align 8
  %ax = extractelement <2 x float> %avv, i64 0
  %ay = extractelement <2 x float> %avv, i64 1
  %af1p = getelementptr inbounds %struct.Box, ptr %a, i64 0, i32 1
  %af1 = load float, ptr %af1p, align 4
  %af2p = getelementptr inbounds %struct.Box, ptr %a, i64 0, i32 2
  %af2 = load float, ptr %af2p, align 4
  %bv = getelementptr inbounds %struct.Box, ptr %b, i64 0, i32 0
  %bvv = load <2 x float>, ptr %bv, align 8
  %bx = extractelement <2 x float> %bvv, i64 0
  %by = extractelement <2 x float> %bvv, i64 1
  %bf1p = getelementptr inbounds %struct.Box, ptr %b, i64 0, i32 1
  %bf1 = load float, ptr %bf1p, align 4
  %bf2p = getelementptr inbounds %struct.Box, ptr %b, i64 0, i32 2
  %bf2 = load float, ptr %bf2p, align 4
  %o0 = getelementptr inbounds float, ptr addrspace(1) %out, i64 0
  store float %ax, ptr addrspace(1) %o0, align 4
  %o1 = getelementptr inbounds float, ptr addrspace(1) %out, i64 1
  store float %ay, ptr addrspace(1) %o1, align 4
  %o2 = getelementptr inbounds float, ptr addrspace(1) %out, i64 2
  store float %af1, ptr addrspace(1) %o2, align 4
  %o3 = getelementptr inbounds float, ptr addrspace(1) %out, i64 3
  store float %af2, ptr addrspace(1) %o3, align 4
  %o4 = getelementptr inbounds float, ptr addrspace(1) %out, i64 4
  store float %bx, ptr addrspace(1) %o4, align 4
  %o5 = getelementptr inbounds float, ptr addrspace(1) %out, i64 5
  store float %by, ptr addrspace(1) %o5, align 4
  %o6 = getelementptr inbounds float, ptr addrspace(1) %out, i64 6
  store float %bf1, ptr addrspace(1) %o6, align 4
  %o7 = getelementptr inbounds float, ptr addrspace(1) %out, i64 7
  store float %bf2, ptr addrspace(1) %o7, align 4
  %o8 = getelementptr inbounds float, ptr addrspace(1) %out, i64 8
  call void @llvm.memset.p1.i64(ptr addrspace(1) noundef align 4 dereferenceable(16) %o8, i8 0, i64 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p2.i64(ptr noalias writeonly captures(none), ptr addrspace(2) noalias readonly captures(none), i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg)
declare void @llvm.memset.p1.i64(ptr addrspace(1) writeonly captures(none), i8, i64, i1 immarg)

!air.kernel = !{!0}
!0 = !{ptr @zero_a_struct_prefix, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Box", !"air.arg_name", !"src"}
!4 = !{i32 0, i32 8, i32 0, !"float2", !"v", i32 8, i32 4, i32 0, !"float", !"f1", i32 12, i32 4, i32 0, !"float", !"f2"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
