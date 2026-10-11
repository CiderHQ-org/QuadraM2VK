; ModuleID = 'm.fix.ll'
source_filename = "m.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

%struct.Box = type { [3 x float], [3 x float] }

@_ZZ37kernel_threadgroup_float_bbox_atomicsPU9MTLdeviceKfPU9MTLdevicefjE2tg = internal addrspace(3) global %struct.Box zeroinitializer, align 4

; Function Attrs: convergent mustprogress nounwind
define void @kernel_threadgroup_float_bbox_atomics(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, i32 %2) local_unnamed_addr #0 {
  %4 = icmp eq i32 %2, 0
  br i1 %4, label %5, label %12

5:                                                ; preds = %5, %3
  %6 = phi i32 [ %10, %5 ], [ 0, %3 ]
  %7 = zext i32 %6 to i64
  %8 = getelementptr inbounds %struct.Box, ptr addrspace(3) @_ZZ37kernel_threadgroup_float_bbox_atomicsPU9MTLdeviceKfPU9MTLdevicefjE2tg, i64 0, i32 0, i64 %7
  store float 0x47EFFFFFE0000000, ptr addrspace(3) %8, align 4, !tbaa !22
  %9 = getelementptr inbounds %struct.Box, ptr addrspace(3) @_ZZ37kernel_threadgroup_float_bbox_atomicsPU9MTLdeviceKfPU9MTLdevicefjE2tg, i64 0, i32 1, i64 %7
  store float -0.000000e+00, ptr addrspace(3) %9, align 4, !tbaa !22
  %10 = add nuw nsw i32 %6, 1
  %11 = icmp eq i32 %10, 3
  br i1 %11, label %12, label %5, !llvm.loop !26

12:                                               ; preds = %5, %3
  tail call void @air.wg.barrier(i32 2, i32 1) #3
  %13 = mul i32 %2, 3
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %14
  %16 = load float, ptr addrspace(1) %15, align 4, !tbaa !22, !alias.scope !28, !noalias !31
  %17 = insertelement <3 x float> undef, float %16, i64 0
  %18 = add i32 %13, 1
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %19
  %21 = load float, ptr addrspace(1) %20, align 4, !tbaa !22, !alias.scope !28, !noalias !31
  %22 = insertelement <3 x float> %17, float %21, i64 1
  %23 = add i32 %13, 2
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds float, ptr addrspace(1) %0, i64 %24
  %26 = load float, ptr addrspace(1) %25, align 4, !tbaa !22, !alias.scope !28, !noalias !31
  %27 = insertelement <3 x float> %22, float %26, i64 2
  %28 = bitcast <3 x float> %27 to <3 x i32>
  br label %30

29:                                               ; preds = %30
  tail call void @air.wg.barrier(i32 2, i32 1) #3
  br i1 %4, label %45, label %68

30:                                               ; preds = %30, %12
  %31 = phi i32 [ 0, %12 ], [ %43, %30 ]
  %32 = extractelement <3 x i32> %28, i32 %31
  %33 = icmp slt i32 %32, 0
  %34 = xor i32 %32, 2147483647
  %35 = select i1 %33, i32 %34, i32 %32
  %36 = zext i32 %31 to i64
  %37 = getelementptr inbounds %struct.Box, ptr addrspace(3) @_ZZ37kernel_threadgroup_float_bbox_atomicsPU9MTLdeviceKfPU9MTLdevicefjE2tg, i64 0, i32 0, i64 %36
  %38 = bitcast ptr addrspace(3) %37 to ptr addrspace(3)
  %39 = tail call i32 @air.atomic.local.min.s.i32(ptr addrspace(3) captures(none) %38, i32 %35, i32 0, i32 1, i1 true) #4
  %40 = getelementptr inbounds %struct.Box, ptr addrspace(3) @_ZZ37kernel_threadgroup_float_bbox_atomicsPU9MTLdeviceKfPU9MTLdevicefjE2tg, i64 0, i32 1, i64 %36
  %41 = bitcast ptr addrspace(3) %40 to ptr addrspace(3)
  %42 = tail call i32 @air.atomic.local.max.s.i32(ptr addrspace(3) captures(none) %41, i32 %35, i32 0, i32 1, i1 true) #4
  %43 = add nuw nsw i32 %31, 1
  %44 = icmp eq i32 %43, 3
  br i1 %44, label %29, label %30, !llvm.loop !33

45:                                               ; preds = %45, %29
  %46 = phi i32 [ %66, %45 ], [ 0, %29 ]
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds %struct.Box, ptr addrspace(3) @_ZZ37kernel_threadgroup_float_bbox_atomicsPU9MTLdeviceKfPU9MTLdevicefjE2tg, i64 0, i32 0, i64 %47
  %49 = bitcast ptr addrspace(3) %48 to ptr addrspace(3)
  %50 = load i32, ptr addrspace(3) %49, align 4, !tbaa !22
  %51 = icmp slt i32 %50, 0
  %52 = xor i32 %50, 2147483647
  %53 = select i1 %51, i32 %52, i32 %50
  %54 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %47
  %55 = bitcast ptr addrspace(1) %54 to ptr addrspace(1)
  store i32 %53, ptr addrspace(1) %55, align 4, !tbaa !22, !alias.scope !31, !noalias !28
  %56 = getelementptr inbounds %struct.Box, ptr addrspace(3) @_ZZ37kernel_threadgroup_float_bbox_atomicsPU9MTLdeviceKfPU9MTLdevicefjE2tg, i64 0, i32 1, i64 %47
  %57 = bitcast ptr addrspace(3) %56 to ptr addrspace(3)
  %58 = load i32, ptr addrspace(3) %57, align 4, !tbaa !22
  %59 = icmp slt i32 %58, 0
  %60 = xor i32 %58, 2147483647
  %61 = select i1 %59, i32 %60, i32 %58
  %62 = add nuw nsw i32 %46, 3
  %63 = zext i32 %62 to i64
  %64 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %63
  %65 = bitcast ptr addrspace(1) %64 to ptr addrspace(1)
  store i32 %61, ptr addrspace(1) %65, align 4, !tbaa !22, !alias.scope !31, !noalias !28
  %66 = add nuw nsw i32 %46, 1
  %67 = icmp eq i32 %66, 3
  br i1 %67, label %68, label %45, !llvm.loop !35

68:                                               ; preds = %45, %29
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.local.min.s.i32(ptr addrspace(3) captures(none), i32, i32, i32, i1) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.local.max.s.i32(ptr addrspace(3) captures(none), i32, i32, i32, i1) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { mustprogress nounwind willreturn }
attributes #3 = { convergent nounwind willreturn }
attributes #4 = { nounwind willreturn }

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
!9 = !{ptr @kernel_threadgroup_float_bbox_atomics, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"points"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 3, i32 0, i32 0}
!21 = !{!"/private/tmp/wgatom/m.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = distinct !{!26, !27}
!27 = !{!"llvm.loop.mustprogress"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(0)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_threadgroup_float_bbox_atomics)"}
!31 = !{!32}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(1)"}
!33 = distinct !{!33, !27, !34}
!34 = !{!"llvm.loop.unroll.disable"}
!35 = distinct !{!35, !27}
