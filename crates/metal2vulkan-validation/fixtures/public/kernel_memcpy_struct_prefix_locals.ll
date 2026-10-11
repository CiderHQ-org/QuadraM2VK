; ModuleID = 'kernel_memcpy_struct_prefix_locals.ll'
source_filename = "validation/fixtures/public/kernel_memcpy_struct_prefix_locals.metal"
target triple = "spirv-unknown-vulkan1.2"

%struct.Pair = type { float, i32 }
%struct.Row = type { [4 x float], float }
%struct.Cfg = type { %struct.Pair, float }
%struct.Params = type { %struct.Row, %struct.Cfg }

define void @memcpy_struct_prefix_locals(ptr addrspace(2) %params, ptr addrspace(1) %out, i32 %gid) {
entry:
  %rows = alloca [4 x float], align 4
  %pair = alloca %struct.Pair, align 4
  %prow = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  call void @llvm.memcpy.p0.p2.i64(ptr align 4 %rows, ptr addrspace(2) align 4 %prow, i64 16, i1 false)
  %pcfg = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  call void @llvm.memcpy.p0.p2.i64(ptr align 4 %pair, ptr addrspace(2) align 4 %pcfg, i64 8, i1 false)
  %isrow = icmp ult i32 %gid, 4
  br i1 %isrow, label %rowpath, label %cfgpath

rowpath:
  %idx = zext i32 %gid to i64
  %rp = getelementptr inbounds [4 x float], ptr %rows, i64 0, i64 %idx
  %rv = load float, ptr %rp, align 4
  br label %merge

cfgpath:
  %isf = icmp eq i32 %gid, 4
  %pf = getelementptr inbounds %struct.Pair, ptr %pair, i64 0, i32 0
  %fv = load float, ptr %pf, align 4
  %pi = getelementptr inbounds %struct.Pair, ptr %pair, i64 0, i32 1
  %iv = load i32, ptr %pi, align 4
  %ivf = sitofp i32 %iv to float
  %cv = select i1 %isf, float %fv, float %ivf
  br label %merge

merge:
  %v = phi float [ %rv, %rowpath ], [ %cv, %cfgpath ]
  %slot = zext i32 %gid to i64
  %op = getelementptr inbounds float, ptr addrspace(1) %out, i64 %slot
  store float %v, ptr addrspace(1) %op, align 4
  ret void
}

declare void @llvm.memcpy.p0.p2.i64(ptr nocapture writeonly, ptr addrspace(2) nocapture readonly, i64, i1 immarg)

!air.kernel = !{!0}
!0 = !{ptr @memcpy_struct_prefix_locals, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
