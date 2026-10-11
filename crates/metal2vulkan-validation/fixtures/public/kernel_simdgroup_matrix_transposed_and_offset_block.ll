; ModuleID = '/tmp/sgmt.bc'
source_filename = "validation/fixtures/public/kernel_simdgroup_matrix_transposed_and_offset_block.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@_ZZ51kernel_simdgroup_matrix_transposed_and_offset_blockPU9MTLdeviceKfPU9MTLdevicefjE2ta = internal addrspace(3) global [256 x float] undef, align 4

; Function Attrs: convergent mustprogress nounwind
define void @kernel_simdgroup_matrix_transposed_and_offset_block(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = icmp ult i32 %2, 256
  br i1 %4, label %16, label %5

5:                                                ; preds = %16, %3
  tail call void @air.wg.barrier(i32 2, i32 1) #4
  %6 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) readonly captures(none) @_ZZ51kernel_simdgroup_matrix_transposed_and_offset_blockPU9MTLdeviceKfPU9MTLdevicefjE2ta, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer) #5
  %7 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) readonly captures(none) @_ZZ51kernel_simdgroup_matrix_transposed_and_offset_blockPU9MTLdeviceKfPU9MTLdevicefjE2ta, <2 x i64> <i64 8, i64 16>, <2 x i64> <i64 16, i64 1>, <2 x i64> zeroinitializer) #5
  %8 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) readonly captures(none) @_ZZ51kernel_simdgroup_matrix_transposed_and_offset_blockPU9MTLdeviceKfPU9MTLdevicefjE2ta, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 3, i64 2>) #5
  %9 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) readonly captures(none) @_ZZ51kernel_simdgroup_matrix_transposed_and_offset_blockPU9MTLdeviceKfPU9MTLdevicefjE2ta, <2 x i64> <i64 8, i64 16>, <2 x i64> <i64 16, i64 1>, <2 x i64> <i64 2, i64 3>) #5
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %6, ptr addrspace(1) writeonly captures(none) %1, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  %10 = getelementptr inbounds float, ptr addrspace(1) %1, i64 64
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %7, ptr addrspace(1) writeonly captures(none) %10, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  %11 = getelementptr inbounds float, ptr addrspace(1) %1, i64 128
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %8, ptr addrspace(1) writeonly captures(none) %11, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  %12 = getelementptr inbounds float, ptr addrspace(1) %1, i64 192
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %9, ptr addrspace(1) writeonly captures(none) %12, <2 x i64> splat (i64 8), <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer) #6
  %13 = getelementptr inbounds float, ptr addrspace(1) %1, i64 256
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %6, ptr addrspace(1) writeonly captures(none) %13, <2 x i64> <i64 8, i64 16>, <2 x i64> <i64 16, i64 1>, <2 x i64> zeroinitializer) #6
  %14 = getelementptr inbounds float, ptr addrspace(1) %1, i64 432
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %6, ptr addrspace(1) writeonly captures(none) %14, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> <i64 3, i64 2>) #6
  %15 = getelementptr inbounds float, ptr addrspace(1) %1, i64 608
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %6, ptr addrspace(1) writeonly captures(none) %15, <2 x i64> <i64 8, i64 16>, <2 x i64> <i64 16, i64 1>, <2 x i64> <i64 2, i64 3>) #6
  ret void

16:                                               ; preds = %16, %3
  %17 = phi i32 [ %22, %16 ], [ %2, %3 ]
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %18
  %20 = load float, ptr addrspace(1) %19, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %21 = getelementptr inbounds [256 x float], ptr addrspace(3) @_ZZ51kernel_simdgroup_matrix_transposed_and_offset_blockPU9MTLdeviceKfPU9MTLdevicefjE2ta, i64 0, i64 %18
  store float %20, ptr addrspace(3) %21, align 4, !tbaa !22
  %22 = add nuw nsw i32 %17, 32
  %23 = icmp ult i32 %17, 224
  br i1 %23, label %16, label %5, !llvm.loop !31
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) readonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn memory(write)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float>, ptr addrspace(1) writeonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #3

attributes #0 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="2048" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(read) }
attributes #3 = { convergent mustprogress nounwind willreturn memory(write) }
attributes #4 = { convergent nounwind willreturn }
attributes #5 = { convergent nounwind willreturn memory(read) }
attributes #6 = { convergent nounwind willreturn memory(write) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_simdgroup_matrix_transposed_and_offset_block, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fa"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 3, i32 1, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simdgroup_matrix_transposed_and_offset_block.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_simdgroup_matrix_transposed_and_offset_block)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = distinct !{!31, !32}
!32 = !{!"llvm.loop.mustprogress"}
