; ModuleID = 'kernel_agx_emask_edge_lanes'
source_filename = "kernel_agx_emask_edge_lanes.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: argmemonly mustprogress nofree norecurse nosync nounwind willreturn
define void @agx_emask_edge_lanes(i32 addrspace(1)* nocapture readonly "air-buffer-no-alias" %0, i32 addrspace(1)* nocapture "air-buffer-no-alias" %1, i32 addrspace(2)* nocapture readonly align 4 dereferenceable(4) "air-buffer-no-alias" %2, i32 %3) local_unnamed_addr #0 {
  %count = load i32, i32 addrspace(2)* %2, align 4
  %base = sub i32 %3, 2
  %mask = tail call zeroext i16 @llvm.agx3.edgecheck(i32 %base, i32 0, i32 %count)
  %idxa = shl i32 %3, 2
  %idxa64 = zext i32 %idxa to i64
  %pa = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %idxa64
  %va = tail call <4 x i32> @llvm.agx3.load.with.emask.global.v4i32(i32 addrspace(1)* %pa, i16 zeroext %mask, i16 zeroext 15, i16 zeroext 4)
  %qa = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 %idxa64
  tail call void @llvm.agx3.store.with.emask.global.v4i32(i32 addrspace(1)* %qa, <4 x i32> %va, i16 zeroext 15, i16 zeroext 15, i16 zeroext 4)
  %idxb = add i32 %idxa, 32
  %idxb64 = zext i32 %idxb to i64
  %pb = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %idxb64
  %vb = tail call <4 x i32> @llvm.agx3.load.with.emask.global.v4i32(i32 addrspace(1)* %pb, i16 zeroext 15, i16 zeroext 15, i16 zeroext 4)
  %qb = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 %idxb64
  tail call void @llvm.agx3.store.with.emask.global.v4i32(i32 addrspace(1)* %qb, <4 x i32> %vb, i16 zeroext %mask, i16 zeroext 15, i16 zeroext 4)
  ret void
}

declare i16 @llvm.agx3.edgecheck(i32, i32, i32)
declare <4 x i32> @llvm.agx3.load.with.emask.global.v4i32(i32 addrspace(1)*, i16, i16, i16)
declare void @llvm.agx3.store.with.emask.global.v4i32(i32 addrspace(1)*, <4 x i32>, i16, i16, i16)

attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!16, !17, !18}
!llvm.ident = !{!19}
!air.version = !{!20}
!air.language_version = !{!21}
!air.source_file_name = !{!22}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(2)*, i32)* @agx_emask_edge_lanes, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"count"}
!15 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 3, i32 0, i32 0}
!22 = !{!"kernel_agx_emask_edge_lanes.metal"}
