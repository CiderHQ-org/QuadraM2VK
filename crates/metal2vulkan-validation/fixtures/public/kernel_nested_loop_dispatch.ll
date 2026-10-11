; ModuleID = 'kernel_nested_loop_dispatch.ll'
source_filename = "kernel_nested_loop_dispatch.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite)
define void @nested_loop_dispatch(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, i32 %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %4
  %6 = load i32, ptr addrspace(1) %5, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  br label %7

7:                                                ; preds = %50, %3
  %8 = phi i32 [ 0, %3 ], [ %52, %50 ]
  %9 = phi i32 [ 0, %3 ], [ %43, %50 ]
  %10 = phi i32 [ 0, %3 ], [ %13, %50 ]
  %11 = mul nuw nsw i32 %10, 7
  %12 = add i32 %11, %6
  %13 = add nuw nsw i32 %10, 1
  br label %14

14:                                               ; preds = %45, %7
  %15 = phi i32 [ %8, %7 ], [ %44, %45 ]
  %16 = phi i32 [ 0, %7 ], [ %46, %45 ]
  %17 = phi i32 [ %9, %7 ], [ %43, %45 ]
  %18 = mul nuw nsw i32 %16, 3
  %19 = add i32 %12, %18
  %20 = and i32 %19, 7
  %21 = icmp eq i32 %20, 5
  br i1 %21, label %41, label %22

22:                                               ; preds = %14
  %23 = and i32 %19, 3
  switch i32 %23, label %31 [
    i32 0, label %24
    i32 1, label %26
    i32 2, label %28
  ]

24:                                               ; preds = %22
  %25 = add i32 %19, %15
  br label %33

26:                                               ; preds = %22
  %27 = xor i32 %19, %15
  br label %33

28:                                               ; preds = %22
  %29 = mul i32 %15, 3
  %30 = add i32 %29, 1
  br label %33

31:                                               ; preds = %22
  %32 = sub i32 %15, %19
  br label %33

33:                                               ; preds = %31, %28, %26, %24
  %34 = phi i32 [ %32, %31 ], [ %30, %28 ], [ %27, %26 ], [ %25, %24 ]
  %35 = add i32 %17, 1
  %36 = and i32 %19, 15
  %37 = icmp eq i32 %36, 9
  %38 = add i32 %34, 2
  %39 = select i1 %37, i32 7, i32 0
  %40 = select i1 %37, i32 %34, i32 %38
  br label %41

41:                                               ; preds = %33, %14
  %42 = phi i32 [ 5, %14 ], [ %39, %33 ]
  %43 = phi i32 [ %17, %14 ], [ %35, %33 ]
  %44 = phi i32 [ %15, %14 ], [ %40, %33 ]
  switch i32 %42, label %48 [
    i32 0, label %45
    i32 7, label %45
  ]

45:                                               ; preds = %41, %41
  %46 = add nuw nsw i32 %16, 1
  %47 = icmp eq i32 %46, 5
  br i1 %47, label %48, label %14, !llvm.loop !31

48:                                               ; preds = %45, %41
  %49 = icmp ugt i32 %43, 16
  br i1 %49, label %54, label %50

50:                                               ; preds = %48
  %51 = mul i32 %44, 5
  %52 = add i32 %51, %10
  %53 = icmp eq i32 %13, 6
  br i1 %53, label %54, label %7, !llvm.loop !33

54:                                               ; preds = %50, %48
  %55 = phi i32 [ 6, %50 ], [ %13, %48 ]
  %56 = phi i32 [ %52, %50 ], [ %44, %48 ]
  %57 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %4
  store i32 %56, ptr addrspace(1) %57, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %58 = add i32 %2, 8
  %59 = zext i32 %58 to i64
  %60 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %59
  store i32 %43, ptr addrspace(1) %60, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %61 = add i32 %2, 16
  %62 = zext i32 %61 to i64
  %63 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %62
  store i32 %55, ptr addrspace(1) %63, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

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
!9 = !{ptr @nested_loop_dispatch, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 3, i32 0, i32 0}
!21 = !{!"/private/tmp/mirror/cfg/kernel_nested_loop_dispatch.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(nested_loop_dispatch)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = distinct !{!31, !32}
!32 = !{!"llvm.loop.mustprogress"}
!33 = distinct !{!33, !32}
