; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'saf.bc'
source_filename = "validation/fixtures/public/kernel_saturating_and_absdiff.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_saturating_and_absdiff(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3) local_unnamed_addr #0 {
  %5 = load i8, ptr addrspace(1) %1, align 1, !tbaa !23, !alias.scope !26, !noalias !29
  %6 = insertelement <3 x i8> undef, i8 %5, i64 0
  %7 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 1
  %8 = load i8, ptr addrspace(1) %7, align 1, !tbaa !23, !alias.scope !26, !noalias !29
  %9 = insertelement <3 x i8> %6, i8 %8, i64 1
  %10 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 2
  %11 = load i8, ptr addrspace(1) %10, align 1, !tbaa !23, !alias.scope !26, !noalias !29
  %12 = insertelement <3 x i8> %9, i8 %11, i64 2
  %13 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 3
  %14 = load i8, ptr addrspace(1) %13, align 1, !tbaa !23, !alias.scope !26, !noalias !29
  %15 = insertelement <3 x i8> undef, i8 %14, i64 0
  %16 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 4
  %17 = load i8, ptr addrspace(1) %16, align 1, !tbaa !23, !alias.scope !26, !noalias !29
  %18 = insertelement <3 x i8> %15, i8 %17, i64 1
  %19 = getelementptr inbounds i8, ptr addrspace(1) %1, i64 5
  %20 = load i8, ptr addrspace(1) %19, align 1, !tbaa !23, !alias.scope !26, !noalias !29
  %21 = insertelement <3 x i8> %18, i8 %20, i64 2
  %22 = tail call <3 x i8> @air.add_sat.u.v3i8(<3 x i8> %12, <3 x i8> %21) #2
  %23 = tail call <3 x i8> @air.sub_sat.u.v3i8(<3 x i8> %12, <3 x i8> %21) #2
  %24 = load i16, ptr addrspace(1) %2, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %25 = insertelement <4 x i16> undef, i16 %24, i64 0
  %26 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 1
  %27 = load i16, ptr addrspace(1) %26, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %28 = insertelement <4 x i16> %25, i16 %27, i64 1
  %29 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 2
  %30 = load i16, ptr addrspace(1) %29, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %31 = insertelement <4 x i16> %28, i16 %30, i64 2
  %32 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 3
  %33 = load i16, ptr addrspace(1) %32, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %34 = insertelement <4 x i16> %31, i16 %33, i64 3
  %35 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 4
  %36 = load i16, ptr addrspace(1) %35, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %37 = insertelement <4 x i16> undef, i16 %36, i64 0
  %38 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 5
  %39 = load i16, ptr addrspace(1) %38, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %40 = insertelement <4 x i16> %37, i16 %39, i64 1
  %41 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 6
  %42 = load i16, ptr addrspace(1) %41, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %43 = insertelement <4 x i16> %40, i16 %42, i64 2
  %44 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 7
  %45 = load i16, ptr addrspace(1) %44, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %46 = insertelement <4 x i16> %43, i16 %45, i64 3
  %47 = tail call <4 x i16> @air.abs_diff.u.v4i16(<4 x i16> %34, <4 x i16> %46) #2
  %48 = insertelement <2 x i16> undef, i16 %24, i64 0
  %49 = insertelement <2 x i16> %48, i16 %27, i64 1
  %50 = insertelement <2 x i16> undef, i16 %36, i64 0
  %51 = insertelement <2 x i16> %50, i16 %39, i64 1
  %52 = tail call <2 x i16> @air.abs_diff.u.v2i16(<2 x i16> %49, <2 x i16> %51) #2
  %53 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 8
  %54 = load i16, ptr addrspace(1) %53, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %55 = getelementptr inbounds i16, ptr addrspace(1) %2, i64 9
  %56 = load i16, ptr addrspace(1) %55, align 2, !tbaa !33, !alias.scope !35, !noalias !36
  %57 = tail call i16 @air.rhadd.u.i16(i16 %54, i16 %56) #2
  %58 = load i32, ptr addrspace(1) %3, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %59 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 1
  %60 = load i32, ptr addrspace(1) %59, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %61 = tail call i32 @air.abs_diff.u.i32(i32 %58, i32 %60) #2
  %62 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 2
  %63 = load i32, ptr addrspace(1) %62, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %64 = insertelement <4 x i32> undef, i32 %63, i64 0
  %65 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 3
  %66 = load i32, ptr addrspace(1) %65, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %67 = insertelement <4 x i32> %64, i32 %66, i64 1
  %68 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 4
  %69 = load i32, ptr addrspace(1) %68, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %70 = insertelement <4 x i32> %67, i32 %69, i64 2
  %71 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 5
  %72 = load i32, ptr addrspace(1) %71, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %73 = insertelement <4 x i32> %70, i32 %72, i64 3
  %74 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 6
  %75 = load i32, ptr addrspace(1) %74, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %76 = insertelement <4 x i32> undef, i32 %75, i64 0
  %77 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 7
  %78 = load i32, ptr addrspace(1) %77, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %79 = insertelement <4 x i32> %76, i32 %78, i64 1
  %80 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 8
  %81 = load i32, ptr addrspace(1) %80, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %82 = insertelement <4 x i32> %79, i32 %81, i64 2
  %83 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 9
  %84 = load i32, ptr addrspace(1) %83, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %85 = insertelement <4 x i32> %82, i32 %84, i64 3
  %86 = tail call <4 x i32> @air.abs_diff.s.v4i32(<4 x i32> %73, <4 x i32> %85) #2
  %87 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 10
  %88 = load i32, ptr addrspace(1) %87, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %89 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 11
  %90 = load i32, ptr addrspace(1) %89, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %91 = getelementptr inbounds i32, ptr addrspace(1) %3, i64 12
  %92 = load i32, ptr addrspace(1) %91, align 4, !tbaa !37, !alias.scope !39, !noalias !40
  %93 = tail call i32 @air.mad_sat.s.i32(i32 %88, i32 %90, i32 %92) #2
  %94 = extractelement <3 x i8> %22, i64 0
  %95 = zext i8 %94 to i32
  store i32 %95, ptr addrspace(1) %0, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %96 = extractelement <3 x i8> %22, i64 1
  %97 = zext i8 %96 to i32
  %98 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 1
  store i32 %97, ptr addrspace(1) %98, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %99 = extractelement <3 x i8> %22, i64 2
  %100 = zext i8 %99 to i32
  %101 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 2
  store i32 %100, ptr addrspace(1) %101, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %102 = extractelement <3 x i8> %23, i64 0
  %103 = zext i8 %102 to i32
  %104 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 3
  store i32 %103, ptr addrspace(1) %104, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %105 = extractelement <3 x i8> %23, i64 1
  %106 = zext i8 %105 to i32
  %107 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 4
  store i32 %106, ptr addrspace(1) %107, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %108 = extractelement <3 x i8> %23, i64 2
  %109 = zext i8 %108 to i32
  %110 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 5
  store i32 %109, ptr addrspace(1) %110, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %111 = extractelement <4 x i16> %47, i64 0
  %112 = zext i16 %111 to i32
  %113 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 6
  store i32 %112, ptr addrspace(1) %113, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %114 = extractelement <4 x i16> %47, i64 1
  %115 = zext i16 %114 to i32
  %116 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 7
  store i32 %115, ptr addrspace(1) %116, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %117 = extractelement <4 x i16> %47, i64 2
  %118 = zext i16 %117 to i32
  %119 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 8
  store i32 %118, ptr addrspace(1) %119, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %120 = extractelement <4 x i16> %47, i64 3
  %121 = zext i16 %120 to i32
  %122 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 9
  store i32 %121, ptr addrspace(1) %122, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %123 = extractelement <2 x i16> %52, i64 0
  %124 = zext i16 %123 to i32
  %125 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 10
  store i32 %124, ptr addrspace(1) %125, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %126 = extractelement <2 x i16> %52, i64 1
  %127 = zext i16 %126 to i32
  %128 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 11
  store i32 %127, ptr addrspace(1) %128, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %129 = zext i16 %57 to i32
  %130 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 12
  store i32 %129, ptr addrspace(1) %130, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %131 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 13
  store i32 %61, ptr addrspace(1) %131, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %132 = extractelement <4 x i32> %86, i64 0
  %133 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 14
  store i32 %132, ptr addrspace(1) %133, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %134 = extractelement <4 x i32> %86, i64 1
  %135 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 15
  store i32 %134, ptr addrspace(1) %135, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %136 = extractelement <4 x i32> %86, i64 2
  %137 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 16
  store i32 %136, ptr addrspace(1) %137, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %138 = extractelement <4 x i32> %86, i64 3
  %139 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 17
  store i32 %138, ptr addrspace(1) %139, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  %140 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 18
  store i32 %93, ptr addrspace(1) %140, align 4, !tbaa !37, !alias.scope !41, !noalias !42
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i8> @air.add_sat.u.v3i8(<3 x i8>, <3 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x i8> @air.sub_sat.u.v3i8(<3 x i8>, <3 x i8>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i16> @air.abs_diff.u.v4i16(<4 x i16>, <4 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <2 x i16> @air.abs_diff.u.v2i16(<2 x i16>, <2 x i16>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.rhadd.u.i16(i16, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.abs_diff.u.i32(i32, i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <4 x i32> @air.abs_diff.s.v4i32(<4 x i32>, <4 x i32>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.mad_sat.s.i32(i32, i32, i32) local_unnamed_addr #1

attributes #0 = { mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_saturating_and_absdiff, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"b8"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort", !"air.arg_name", !"b16"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b32"}
!16 = !{!"air.compile.denorms_disable"}
!17 = !{!"air.compile.fast_math_enable"}
!18 = !{!"air.compile.framebuffer_fetch_enable"}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_saturating_and_absdiff.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_saturating_and_absdiff)"}
!29 = !{!30, !31, !32}
!30 = distinct !{!30, !28, !"air-alias-scope-arg(0)"}
!31 = distinct !{!31, !28, !"air-alias-scope-arg(2)"}
!32 = distinct !{!32, !28, !"air-alias-scope-arg(3)"}
!33 = !{!34, !34, i64 0}
!34 = !{!"short", !24, i64 0}
!35 = !{!31}
!36 = !{!30, !27, !32}
!37 = !{!38, !38, i64 0}
!38 = !{!"int", !24, i64 0}
!39 = !{!32}
!40 = !{!30, !27, !31}
!41 = !{!30}
!42 = !{!27, !31, !32}
