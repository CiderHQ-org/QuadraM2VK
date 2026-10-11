; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'bif.bc'
source_filename = "validation/fixtures/public/kernel_bitfield_and_wide_counts.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_bitfield_and_wide_counts(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2) local_unnamed_addr #0 {
  %4 = load i32, ptr addrspace(1) %1, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %5 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 1
  %6 = load i32, ptr addrspace(1) %5, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %7 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 2
  %8 = load i32, ptr addrspace(1) %7, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %9 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 3
  %10 = load i32, ptr addrspace(1) %9, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %11 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 4
  %12 = load i32, ptr addrspace(1) %11, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %13 = tail call i32 @air.insert_bits.u.i32(i32 %4, i32 %6, i32 %8, i32 %10) #2
  %14 = tail call i32 @air.insert_bits.s.i32(i32 %12, i32 %6, i32 %8, i32 %10) #2
  %15 = load i64, ptr addrspace(1) %2, align 8, !tbaa !32, !alias.scope !34, !noalias !35
  %16 = getelementptr inbounds i64, ptr addrspace(1) %2, i64 1
  %17 = load i64, ptr addrspace(1) %16, align 8, !tbaa !32, !alias.scope !34, !noalias !35
  %18 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 5
  %19 = load i32, ptr addrspace(1) %18, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %20 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 6
  %21 = load i32, ptr addrspace(1) %20, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %22 = tail call i64 @air.insert_bits.u.i64(i64 %15, i64 %17, i32 %19, i32 %21) #2
  %23 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 7
  %24 = load i32, ptr addrspace(1) %23, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %25 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 8
  %26 = load i32, ptr addrspace(1) %25, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %27 = tail call i32 @air.extract_bits.s.i32(i32 %12, i32 %24, i32 %26) #2
  %28 = getelementptr inbounds i64, ptr addrspace(1) %2, i64 2
  %29 = load i64, ptr addrspace(1) %28, align 8, !tbaa !32, !alias.scope !34, !noalias !35
  %30 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 9
  %31 = load i32, ptr addrspace(1) %30, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %32 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 10
  %33 = load i32, ptr addrspace(1) %32, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %34 = tail call i64 @air.extract_bits.u.i64(i64 %29, i32 %31, i32 %33) #2
  %35 = getelementptr inbounds i64, ptr addrspace(1) %2, i64 3
  %36 = load i64, ptr addrspace(1) %35, align 8, !tbaa !32, !alias.scope !34, !noalias !35
  %37 = insertelement <2 x i64> undef, i64 %36, i64 0
  %38 = getelementptr inbounds i64, ptr addrspace(1) %2, i64 4
  %39 = load i64, ptr addrspace(1) %38, align 8, !tbaa !32, !alias.scope !34, !noalias !35
  %40 = insertelement <2 x i64> %37, i64 %39, i64 1
  %41 = tail call <2 x i64> @air.popcount.v2i64(<2 x i64> %40) #2
  store i32 %13, ptr addrspace(1) %0, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %42 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %14, ptr addrspace(1) %42, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %43 = trunc i64 %22 to i32
  %44 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %43, ptr addrspace(1) %44, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %45 = lshr i64 %22, 32
  %46 = trunc i64 %45 to i32
  %47 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %46, ptr addrspace(1) %47, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %48 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %27, ptr addrspace(1) %48, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %49 = trunc i64 %34 to i32
  %50 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %49, ptr addrspace(1) %50, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %51 = lshr i64 %34, 32
  %52 = trunc i64 %51 to i32
  %53 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  store i32 %52, ptr addrspace(1) %53, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %54 = bitcast <2 x i64> %41 to <4 x i32>
  %55 = extractelement <4 x i32> %54, i64 0
  %56 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  store i32 %55, ptr addrspace(1) %56, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %57 = extractelement <4 x i32> %54, i64 2
  %58 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  store i32 %57, ptr addrspace(1) %58, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %59 = getelementptr inbounds i64, ptr addrspace(1) %2, i64 5
  %60 = load i64, ptr addrspace(1) %59, align 8, !tbaa !32, !alias.scope !34, !noalias !35
  %61 = tail call i64 @air.clz.i64(i64 %60, i1 false) #2
  %62 = trunc i64 %61 to i32
  %63 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 9
  store i32 %62, ptr addrspace(1) %63, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %64 = tail call i64 @air.ctz.i64(i64 %60, i1 false) #2
  %65 = trunc i64 %64 to i32
  %66 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 10
  store i32 %65, ptr addrspace(1) %66, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %67 = getelementptr inbounds i64, ptr addrspace(1) %2, i64 6
  %68 = load i64, ptr addrspace(1) %67, align 8, !tbaa !32, !alias.scope !34, !noalias !35
  %69 = tail call i64 @air.clz.i64(i64 %68, i1 false) #2
  %70 = trunc i64 %69 to i32
  %71 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 11
  store i32 %70, ptr addrspace(1) %71, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  %72 = tail call i64 @air.ctz.i64(i64 %68, i1 false) #2
  %73 = trunc i64 %72 to i32
  %74 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 12
  store i32 %73, ptr addrspace(1) %74, align 4, !tbaa !22, !alias.scope !36, !noalias !37
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.insert_bits.u.i32(i32, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.insert_bits.s.i32(i32, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i64 @air.insert_bits.u.i64(i64, i64, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.extract_bits.s.i32(i32, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i64 @air.extract_bits.u.i64(i64, i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i64> @air.popcount.v2i64(<2 x i64>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i64 @air.clz.i64(i64, i1) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i64 @air.ctz.i64(i64, i1) local_unnamed_addr #1

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
!9 = !{ptr @kernel_bitfield_and_wide_counts, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"u"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"w"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_bitfield_and_wide_counts.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"int", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_bitfield_and_wide_counts)"}
!29 = !{!30, !31}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !28, !"air-alias-scope-arg(2)"}
!32 = !{!33, !33, i64 0}
!33 = !{!"long", !24, i64 0}
!34 = !{!31}
!35 = !{!30, !27}
!36 = !{!30}
!37 = !{!27, !31}
