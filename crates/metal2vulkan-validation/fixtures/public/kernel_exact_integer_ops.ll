; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'f.bc'
source_filename = "validation/fixtures/public/kernel_exact_integer_ops.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_exact_integer_ops(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = load i32, ptr addrspace(1) %1, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %5 = insertelement <4 x i32> undef, i32 %4, i64 0
  %6 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1
  %7 = load i32, ptr addrspace(1) %6, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %8 = insertelement <4 x i32> %5, i32 %7, i64 1
  %9 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 2
  %10 = load i32, ptr addrspace(1) %9, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %11 = insertelement <4 x i32> %8, i32 %10, i64 2
  %12 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 3
  %13 = load i32, ptr addrspace(1) %12, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %14 = insertelement <4 x i32> %11, i32 %13, i64 3
  %15 = tail call i32 @air.reverse_bits.i32(i32 %4) #2
  store i32 %15, ptr addrspace(1) %0, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %16 = tail call i32 @air.rotate.i32(i32 %4, i32 %7) #2
  %17 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %16, ptr addrspace(1) %17, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %18 = tail call <4 x i32> @air.popcount.v4i32(<4 x i32> %14) #2
  %19 = extractelement <4 x i32> %18, i64 0
  %20 = extractelement <4 x i32> %18, i64 1
  %21 = shl i32 %20, 8
  %22 = add i32 %21, %19
  %23 = extractelement <4 x i32> %18, i64 2
  %24 = shl i32 %23, 16
  %25 = add i32 %22, %24
  %26 = extractelement <4 x i32> %18, i64 3
  %27 = shl i32 %26, 24
  %28 = add i32 %25, %27
  %29 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %28, ptr addrspace(1) %29, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %30 = insertelement <2 x i32> undef, i32 %4, i64 0
  %31 = insertelement <2 x i32> %30, i32 %7, i64 1
  %32 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 4
  %33 = load i32, ptr addrspace(1) %32, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %34 = insertelement <2 x i32> undef, i32 %33, i64 0
  %35 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 5
  %36 = load i32, ptr addrspace(1) %35, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %37 = insertelement <2 x i32> %34, i32 %36, i64 1
  %38 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 6
  %39 = load i32, ptr addrspace(1) %38, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %40 = insertelement <2 x i32> undef, i32 %39, i64 0
  %41 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 7
  %42 = load i32, ptr addrspace(1) %41, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %43 = insertelement <2 x i32> %40, i32 %42, i64 1
  %44 = tail call <2 x i32> @air.clamp.u.v2i32(<2 x i32> %31, <2 x i32> %37, <2 x i32> %43) #2
  %45 = extractelement <2 x i32> %44, i64 0
  %46 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %45, ptr addrspace(1) %46, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %47 = extractelement <2 x i32> %44, i64 1
  %48 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %47, ptr addrspace(1) %48, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %49 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 8
  %50 = load i32, ptr addrspace(1) %49, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %51 = trunc i32 %50 to i16
  %52 = tail call fast half @air.convert.f.f16.s.i16(i16 %51) #2
  %53 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 9
  %54 = load i32, ptr addrspace(1) %53, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %55 = trunc i32 %54 to i16
  %56 = tail call fast half @air.convert.f.f16.s.i16(i16 %55) #2
  %57 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 10
  %58 = load i32, ptr addrspace(1) %57, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %59 = trunc i32 %58 to i16
  %60 = tail call fast half @air.convert.f.f16.s.i16(i16 %59) #2
  %61 = tail call fast half @air.fmedian3.f16(half %52, half %56, half %60) #2
  %62 = tail call i32 @air.convert.s.i32.f.f16(half %61) #2
  %63 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %62, ptr addrspace(1) %63, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %64 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 11
  %65 = load i32, ptr addrspace(1) %64, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %66 = trunc i32 %65 to i16
  %67 = tail call i16 @air.ctz.i16(i16 %66, i1 false) #2
  %68 = sext i16 %67 to i32
  %69 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  store i32 %68, ptr addrspace(1) %69, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  %70 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 12
  %71 = load i32, ptr addrspace(1) %70, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %72 = trunc i32 %71 to i16
  %73 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 13
  %74 = load i32, ptr addrspace(1) %73, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %75 = trunc i32 %74 to i16
  %76 = tail call i16 @air.max.s.i16(i16 %72, i16 %75) #2
  %77 = sext i16 %76 to i32
  %78 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  store i32 %77, ptr addrspace(1) %78, align 4, !tbaa !22, !alias.scope !29, !noalias !26
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.s.i16(i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.convert.s.i32.f.f16(half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.reverse_bits.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.rotate.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.popcount.v4i32(<4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i32> @air.clamp.u.v2i32(<2 x i32>, <2 x i32>, <2 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.fmedian3.f16(half, half, half) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.ctz.i16(i16, i1) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.max.s.i16(i16, i16) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_exact_integer_ops, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"in"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid", !"air.arg_unused"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/privatevalidation/fixtures/public/kernel_exact_integer_ops.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_exact_integer_ops)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
