; ModuleID = 'kernel_swap_corners_through_a_local.ll'
source_filename = "validation/fixtures/public/kernel_swap_corners_through_a_local.metal"
target triple = "spirv-unknown-vulkan1.2"

%struct.Corner = type { <2 x float>, float }

define void @swap_corners_through_a_local(ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %items) {
entry:
  %tmp = alloca %struct.Corner, align 8
  %p0 = getelementptr inbounds %struct.Corner, ptr addrspace(1) %items, i64 0
  %p1 = getelementptr inbounds %struct.Corner, ptr addrspace(1) %items, i64 1
  call void @llvm.memcpy.p0.p1.i64(ptr noundef nonnull align 8 dereferenceable(16) %tmp, ptr addrspace(1) noundef align 8 dereferenceable(16) %p0, i64 16, i1 false)
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noundef align 8 dereferenceable(16) %p0, ptr addrspace(1) noundef align 8 dereferenceable(16) %p1, i64 16, i1 false)
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noundef align 8 dereferenceable(16) %p1, ptr noundef nonnull align 8 dereferenceable(16) %tmp, i64 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p1.i64(ptr noalias writeonly captures(none), ptr addrspace(1) noalias readonly captures(none), i64, i1 immarg)
declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noalias writeonly captures(none), ptr addrspace(1) noalias readonly captures(none), i64, i1 immarg)
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

!air.kernel = !{!0}
!0 = !{ptr @swap_corners_through_a_local, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Corner", !"air.arg_name", !"items"}
!4 = !{i32 0, i32 8, i32 0, !"float2", !"v", i32 8, i32 4, i32 0, !"float", !"f"}
