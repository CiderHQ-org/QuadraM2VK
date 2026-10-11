; ModuleID = 'kernel_three_loop_shapes.ll'
source_filename = "kernel_three_loop_shapes.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite)
define void @three_loop_shapes(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, i32 %2) local_unnamed_addr #0 {
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
  %16 = phi i32 [ %9, %7 ], [ %43, %45 ]
  %17 = phi i32 [ 0, %7 ], [ %46, %45 ]
  %18 = mul nuw nsw i32 %17, 3
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
  %35 = add i32 %16, 1
  %36 = and i32 %19, 15
  %37 = icmp eq i32 %36, 9
  %38 = add i32 %34, 2
  %39 = select i1 %37, i32 7, i32 0
  %40 = select i1 %37, i32 %34, i32 %38
  br label %41

41:                                               ; preds = %33, %14
  %42 = phi i32 [ 5, %14 ], [ %39, %33 ]
  %43 = phi i32 [ %16, %14 ], [ %35, %33 ]
  %44 = phi i32 [ %15, %14 ], [ %40, %33 ]
  switch i32 %42, label %48 [
    i32 0, label %45
    i32 7, label %45
  ]

45:                                               ; preds = %41, %41
  %46 = add nuw nsw i32 %17, 1
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
  %57 = or i32 %6, 1
  br label %58

58:                                               ; preds = %70, %54
  %59 = phi i32 [ %57, %54 ], [ %72, %70 ]
  %60 = phi i32 [ 0, %54 ], [ %61, %70 ]
  %61 = add nuw nsw i32 %60, 1
  %62 = and i32 %59, 3
  %63 = icmp eq i32 %62, 3
  br i1 %63, label %64, label %66

64:                                               ; preds = %58
  %65 = xor i32 %59, %56
  br label %75

66:                                               ; preds = %58
  %67 = icmp eq i32 %60, 9
  br i1 %67, label %68, label %70

68:                                               ; preds = %66
  %69 = add i32 %56, 1000
  br label %75

70:                                               ; preds = %66
  %71 = mul i32 %59, 3
  %72 = add i32 %71, 1
  %73 = and i32 %72, 255
  %74 = icmp eq i32 %73, 0
  br i1 %74, label %75, label %58, !llvm.loop !34

75:                                               ; preds = %70, %68, %64
  %76 = phi i32 [ %61, %64 ], [ 10, %68 ], [ %61, %70 ]
  %77 = phi i32 [ %65, %64 ], [ %69, %68 ], [ %56, %70 ]
  %78 = mul i32 %6, 3
  br label %94

79:                                               ; preds = %116
  %80 = add i32 %77, %76
  %81 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %4
  store i32 %80, ptr addrspace(1) %81, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %82 = add i32 %2, 8
  %83 = zext i32 %82 to i64
  %84 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %83
  store i32 %43, ptr addrspace(1) %84, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %85 = add i32 %2, 16
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %86
  store i32 %55, ptr addrspace(1) %87, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %88 = add i32 %2, 24
  %89 = zext i32 %88 to i64
  %90 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %89
  store i32 %76, ptr addrspace(1) %90, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %91 = add i32 %2, 32
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %92
  store i32 %117, ptr addrspace(1) %93, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  ret void

94:                                               ; preds = %116, %75
  %95 = phi i32 [ 0, %75 ], [ %119, %116 ]
  %96 = phi i32 [ 0, %75 ], [ %117, %116 ]
  %97 = add i32 %95, %78
  %98 = urem i32 %97, 5
  switch i32 %98, label %108 [
    i32 0, label %99
    i32 2, label %99
    i32 1, label %101
    i32 3, label %105
  ]

99:                                               ; preds = %94, %94
  %100 = add i32 %96, %97
  br label %112

101:                                              ; preds = %94
  %102 = xor i32 %96, %97
  %103 = and i32 %97, 1
  %104 = icmp eq i32 %103, 0
  br i1 %104, label %112, label %116

105:                                              ; preds = %94
  %106 = mul i32 %96, 3
  %107 = add i32 %106, 1
  br label %112

108:                                              ; preds = %94
  %109 = icmp ugt i32 %96, 4000
  %110 = select i1 %109, i32 7, i32 %95
  %111 = add i32 %96, 5
  br label %112

112:                                              ; preds = %108, %105, %101, %99
  %113 = phi i32 [ %111, %108 ], [ %107, %105 ], [ %102, %101 ], [ %100, %99 ]
  %114 = phi i32 [ %110, %108 ], [ %95, %105 ], [ %95, %101 ], [ %95, %99 ]
  %115 = add i32 %113, 1
  br label %116

116:                                              ; preds = %112, %101
  %117 = phi i32 [ %115, %112 ], [ %102, %101 ]
  %118 = phi i32 [ %114, %112 ], [ %95, %101 ]
  %119 = add nuw nsw i32 %118, 1
  %120 = icmp ult i32 %118, 6
  br i1 %120, label %94, label %79, !llvm.loop !35
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
!9 = !{ptr @three_loop_shapes, !10, !11}
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
!21 = !{!"/private/tmp/mirror/cfg/kernel_three_loop_shapes.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(three_loop_shapes)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
!31 = distinct !{!31, !32}
!32 = !{!"llvm.loop.mustprogress"}
!33 = distinct !{!33, !32}
!34 = distinct !{!34, !32}
!35 = distinct !{!35, !32}
