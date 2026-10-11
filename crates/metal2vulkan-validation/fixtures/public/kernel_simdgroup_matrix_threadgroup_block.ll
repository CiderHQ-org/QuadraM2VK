; ModuleID = '/tmp/tgm.bc'
source_filename = "validation/fixtures/public/kernel_simdgroup_matrix_threadgroup_block.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2ta = internal addrspace(3) global [128 x float] undef, align 4
@_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2tb = internal addrspace(3) global [128 x float] undef, align 4
@_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2th = internal addrspace(3) global [128 x half] undef, align 2
@_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2ts = internal addrspace(3) global [128 x float] undef, align 4
@_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE3ths = internal addrspace(3) global [128 x half] undef, align 2

; Function Attrs: convergent nounwind
define void @kernel_simdgroup_matrix_threadgroup_block(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %3, i32 noundef %4) local_unnamed_addr #0 {
  %6 = icmp ult i32 %4, 128
  br i1 %6, label %14, label %7

7:                                                ; preds = %14, %5
  tail call void @air.wg.barrier(i32 2, i32 1) #4
  %8 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) readonly captures(none) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2ta, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer) #5
  %9 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) readonly captures(none) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2tb, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer) #5
  %10 = tail call fast <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p3f16(ptr addrspace(3) readonly captures(none) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2th, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer) #5
  %11 = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %8, <64 x float> %9, <64 x float> %8) #4
  %12 = tail call fast <64 x half> @air.simdgroup_matrix_8x8_init_diag.v64f16.f16(half 0xH4000) #4
  %13 = tail call fast <64 x half> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f16.v64f16.v64f16(<64 x half> %10, <64 x half> %12, <64 x half> %10) #4
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float> %11, ptr addrspace(3) writeonly captures(none) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2ts, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer) #6
  tail call void @air.simdgroup_matrix_8x8_store.v64f16.p3f16(<64 x half> %13, ptr addrspace(3) writeonly captures(none) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE3ths, <2 x i64> <i64 16, i64 8>, <2 x i64> <i64 1, i64 16>, <2 x i64> zeroinitializer) #6
  tail call void @air.wg.barrier(i32 2, i32 1) #4
  br i1 %6, label %31, label %30

14:                                               ; preds = %14, %5
  %15 = phi i32 [ %28, %14 ], [ %4, %5 ]
  %16 = zext i32 %15 to i64
  %17 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %16
  %18 = load float, ptr addrspace(1) %17, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %19 = getelementptr inbounds [128 x float], ptr addrspace(3) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2ta, i64 0, i64 %16
  store float %18, ptr addrspace(3) %19, align 4, !tbaa !24
  %20 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %16
  %21 = load float, ptr addrspace(1) %20, align 4, !tbaa !24, !alias.scope !35, !noalias !36
  %22 = getelementptr inbounds [128 x float], ptr addrspace(3) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2tb, i64 0, i64 %16
  store float %21, ptr addrspace(3) %22, align 4, !tbaa !24
  %23 = getelementptr inbounds half, ptr addrspace(1) %2, i64 %16
  %24 = load half, ptr addrspace(1) %23, align 2, !tbaa !37, !alias.scope !39, !noalias !40
  %25 = getelementptr inbounds [128 x half], ptr addrspace(3) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2th, i64 0, i64 %16
  store half %24, ptr addrspace(3) %25, align 2, !tbaa !37
  %26 = getelementptr inbounds [128 x float], ptr addrspace(3) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2ts, i64 0, i64 %16
  store float 0.000000e+00, ptr addrspace(3) %26, align 4, !tbaa !24
  %27 = getelementptr inbounds [128 x half], ptr addrspace(3) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE3ths, i64 0, i64 %16
  store half 0xH0000, ptr addrspace(3) %27, align 2, !tbaa !37
  %28 = add nuw nsw i32 %15, 32
  %29 = icmp ult i32 %15, 96
  br i1 %29, label %14, label %7, !llvm.loop !41

30:                                               ; preds = %31, %7
  ret void

31:                                               ; preds = %31, %7
  %32 = phi i32 [ %45, %31 ], [ %4, %7 ]
  %33 = zext i32 %32 to i64
  %34 = getelementptr inbounds [128 x float], ptr addrspace(3) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE2ts, i64 0, i64 %33
  %35 = bitcast ptr addrspace(3) %34 to ptr addrspace(3)
  %36 = load i32, ptr addrspace(3) %35, align 4, !tbaa !24
  %37 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 %33
  store i32 %36, ptr addrspace(1) %37, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %38 = getelementptr inbounds [128 x half], ptr addrspace(3) @_ZZ41kernel_simdgroup_matrix_threadgroup_blockPU9MTLdeviceKfS0_PU9MTLdeviceKDhPU9MTLdevicejjE3ths, i64 0, i64 %33
  %39 = bitcast ptr addrspace(3) %38 to ptr addrspace(3)
  %40 = load i16, ptr addrspace(3) %39, align 2, !tbaa !37
  %41 = zext i16 %40 to i32
  %42 = add nuw nsw i32 %32, 128
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 %43
  store i32 %41, ptr addrspace(1) %44, align 4, !tbaa !43, !alias.scope !45, !noalias !46
  %45 = add nuw nsw i32 %32, 32
  %46 = icmp ult i32 %32, 96
  br i1 %46, label %31, label %30, !llvm.loop !47
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(ptr addrspace(3) readonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(read)
declare <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p3f16(ptr addrspace(3) readonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x half> @air.simdgroup_matrix_8x8_init_diag.v64f16.f16(half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <64 x half> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f16.v64f16.v64f16.v64f16(<64 x half>, <64 x half>, <64 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn memory(write)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32(<64 x float>, ptr addrspace(3) writeonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #3

; Function Attrs: convergent mustprogress nounwind willreturn memory(write)
declare void @air.simdgroup_matrix_8x8_store.v64f16.p3f16(<64 x half>, ptr addrspace(3) writeonly captures(none), <2 x i64>, <2 x i64>, <2 x i64>) local_unnamed_addr #3

attributes #0 = { convergent nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="2048" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(read) }
attributes #3 = { convergent mustprogress nounwind willreturn memory(write) }
attributes #4 = { convergent nounwind willreturn }
attributes #5 = { convergent nounwind willreturn memory(read) }
attributes #6 = { convergent nounwind willreturn memory(write) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_simdgroup_matrix_threadgroup_block, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fa"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"fb"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"ha"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!16 = !{i32 4, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!21 = !{i32 2, i32 8, i32 0}
!22 = !{!"Metal", i32 3, i32 1, i32 0}
!23 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simdgroup_matrix_threadgroup_block.metal"}
!24 = !{!25, !25, i64 0}
!25 = !{!"float", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(0)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_simdgroup_matrix_threadgroup_block)"}
!31 = !{!32, !33, !34}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(1)"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(2)"}
!34 = distinct !{!34, !30, !"air-alias-scope-arg(3)"}
!35 = !{!32}
!36 = !{!29, !33, !34}
!37 = !{!38, !38, i64 0}
!38 = !{!"half", !26, i64 0}
!39 = !{!33}
!40 = !{!29, !32, !34}
!41 = distinct !{!41, !42}
!42 = !{!"llvm.loop.mustprogress"}
!43 = !{!44, !44, i64 0}
!44 = !{!"int", !26, i64 0}
!45 = !{!34}
!46 = !{!29, !32, !33}
!47 = distinct !{!47, !42}
