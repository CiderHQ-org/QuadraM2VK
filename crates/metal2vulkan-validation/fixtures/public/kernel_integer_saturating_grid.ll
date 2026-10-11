; ModuleID = 'k2.bc'
source_filename = "k.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_integer_saturating_grid(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, i32 %2) local_unnamed_addr #0 {
  %4 = urem i32 %2, 12
  %5 = zext i32 %4 to i64
  %6 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %5
  %7 = load i32, ptr addrspace(1) %6, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %8 = add i32 %2, 1
  %9 = urem i32 %8, 12
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %10
  %12 = load i32, ptr addrspace(1) %11, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %13 = add i32 %2, 2
  %14 = urem i32 %13, 12
  %15 = zext i32 %14 to i64
  %16 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %15
  %17 = load i32, ptr addrspace(1) %16, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %18 = mul i32 %2, 15
  %19 = tail call i32 @air.mul_hi.s.i32(i32 %7, i32 %12) #2
  %20 = zext i32 %18 to i64
  %21 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %20
  store i32 %19, ptr addrspace(1) %21, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %22 = tail call i32 @air.mad_hi.s.i32(i32 %7, i32 %12, i32 %17) #2
  %23 = add i32 %18, 1
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %24
  store i32 %22, ptr addrspace(1) %25, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %26 = tail call i32 @air.mad_hi.u.i32(i32 %7, i32 %12, i32 %17) #2
  %27 = add i32 %18, 2
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %28
  store i32 %26, ptr addrspace(1) %29, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %30 = tail call i32 @air.mad_sat.u.i32(i32 %7, i32 %12, i32 %17) #2
  %31 = add i32 %18, 3
  %32 = zext i32 %31 to i64
  %33 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %32
  store i32 %30, ptr addrspace(1) %33, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %34 = tail call i32 @air.hadd.s.i32(i32 %7, i32 %12) #2
  %35 = add i32 %18, 4
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %36
  store i32 %34, ptr addrspace(1) %37, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %38 = tail call i32 @air.hadd.u.i32(i32 %7, i32 %12) #2
  %39 = add i32 %18, 5
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %40
  store i32 %38, ptr addrspace(1) %41, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %42 = tail call i32 @air.rhadd.s.i32(i32 %7, i32 %12) #2
  %43 = add i32 %18, 6
  %44 = zext i32 %43 to i64
  %45 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %44
  store i32 %42, ptr addrspace(1) %45, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %46 = tail call i32 @air.rhadd.u.i32(i32 %7, i32 %12) #2
  %47 = add i32 %18, 7
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %48
  store i32 %46, ptr addrspace(1) %49, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %50 = tail call i32 @air.add_sat.s.i32(i32 %7, i32 %12) #2
  %51 = add i32 %18, 8
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %52
  store i32 %50, ptr addrspace(1) %53, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %54 = tail call i32 @air.sub_sat.s.i32(i32 %7, i32 %12) #2
  %55 = add i32 %18, 9
  %56 = zext i32 %55 to i64
  %57 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %56
  store i32 %54, ptr addrspace(1) %57, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %58 = trunc i32 %7 to i16
  %59 = trunc i32 %12 to i16
  %60 = tail call i16 @air.add_sat.s.i16(i16 %58, i16 %59) #2
  %61 = zext i16 %60 to i32
  %62 = add i32 %18, 10
  %63 = zext i32 %62 to i64
  %64 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %63
  store i32 %61, ptr addrspace(1) %64, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %65 = tail call i16 @air.hadd.s.i16(i16 %58, i16 %59) #2
  %66 = zext i16 %65 to i32
  %67 = add i32 %18, 11
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %68
  store i32 %66, ptr addrspace(1) %69, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %70 = tail call i16 @air.rhadd.s.i16(i16 %58, i16 %59) #2
  %71 = zext i16 %70 to i32
  %72 = add i32 %18, 12
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %73
  store i32 %71, ptr addrspace(1) %74, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %75 = trunc i32 %7 to i8
  %76 = insertelement <3 x i8> undef, i8 %75, i64 0
  %77 = trunc i32 %12 to i8
  %78 = insertelement <3 x i8> %76, i8 %77, i64 1
  %79 = trunc i32 %17 to i8
  %80 = insertelement <3 x i8> %78, i8 %79, i64 2
  %81 = insertelement <3 x i8> undef, i8 %77, i64 0
  %82 = insertelement <3 x i8> %81, i8 %79, i64 1
  %83 = insertelement <3 x i8> %82, i8 %75, i64 2
  %84 = tail call <3 x i8> @air.sub_sat.s.v3i8(<3 x i8> %80, <3 x i8> %83) #2
  %85 = extractelement <3 x i8> %84, i64 0
  %86 = zext i8 %85 to i32
  %87 = extractelement <3 x i8> %84, i64 1
  %88 = zext i8 %87 to i32
  %89 = shl nuw nsw i32 %88, 8
  %90 = or i32 %89, %86
  %91 = extractelement <3 x i8> %84, i64 2
  %92 = zext i8 %91 to i32
  %93 = shl nuw nsw i32 %92, 16
  %94 = or i32 %90, %93
  %95 = add i32 %18, 13
  %96 = zext i32 %95 to i64
  %97 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %96
  store i32 %94, ptr addrspace(1) %97, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %98 = tail call <3 x i8> @air.rhadd.s.v3i8(<3 x i8> %80, <3 x i8> %83) #2
  %99 = extractelement <3 x i8> %98, i64 0
  %100 = zext i8 %99 to i32
  %101 = extractelement <3 x i8> %98, i64 1
  %102 = zext i8 %101 to i32
  %103 = shl nuw nsw i32 %102, 8
  %104 = or i32 %103, %100
  %105 = extractelement <3 x i8> %98, i64 2
  %106 = zext i8 %105 to i32
  %107 = shl nuw nsw i32 %106, 16
  %108 = or i32 %104, %107
  %109 = add i32 %18, 14
  %110 = zext i32 %109 to i64
  %111 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %110
  store i32 %108, ptr addrspace(1) %111, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.mul_hi.s.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.mad_hi.s.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.mad_hi.u.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.mad_sat.u.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.hadd.s.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.hadd.u.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.rhadd.s.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.rhadd.u.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.add_sat.s.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.sub_sat.s.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.add_sat.s.i16(i16, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.hadd.s.i16(i16, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.rhadd.s.i16(i16, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i8> @air.sub_sat.s.v3i8(<3 x i8>, <3 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i8> @air.rhadd.s.v3i8(<3 x i8>, <3 x i8>) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="24" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_integer_saturating_grid, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 3, i32 0, i32 0}
!21 = !{!"/private/tmp/isg/k.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(0)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_integer_saturating_grid)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(1)"}
