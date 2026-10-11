; ModuleID = 'kernel_copy_a_float_run.ll'
source_filename = "validation/fixtures/public/kernel_copy_a_float_run.metal"
target triple = "spirv-unknown-vulkan1.2"

%struct.Blob = type { i32, [3 x float] }

define void @copy_a_float_run(ptr addrspace(1) noundef captures(none) "air-buffer-no-alias" %blobs) {
entry:
  %tmp = alloca [3 x float], align 4
  %d = getelementptr inbounds [3 x float], ptr %tmp, i64 0, i64 0
  %s0 = getelementptr inbounds %struct.Blob, ptr addrspace(1) %blobs, i64 0, i32 1, i64 0
  call void @llvm.memcpy.p0.p1.i64(ptr noundef nonnull align 4 dereferenceable(12) %d, ptr addrspace(1) noundef align 4 dereferenceable(12) %s0, i64 12, i1 false)
  %s1 = getelementptr inbounds %struct.Blob, ptr addrspace(1) %blobs, i64 1, i32 1, i64 0
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noundef align 4 dereferenceable(12) %s1, ptr noundef nonnull align 4 dereferenceable(12) %d, i64 12, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p1.i64(ptr noalias writeonly captures(none), ptr addrspace(1) noalias readonly captures(none), i64, i1 immarg)
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

!air.kernel = !{!0}
!0 = !{ptr @copy_a_float_run, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Blob", !"air.arg_name", !"blobs"}
