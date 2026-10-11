; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers. Sixteen `air.simd_*`
; type variants that the corpus reaches once each and that no authored case covered.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_simd_variant_leftovers.bc'
source_filename = "validation/fixtures/public/kernel_simd_variant_leftovers.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_simd_variant_leftovers(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %3, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %4, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %5, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %6, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %7, i32 noundef %8) local_unnamed_addr #0 {
  %10 = zext i32 %8 to i64
  %11 = getelementptr inbounds <4 x float>, ptr addrspace(1) %0, i64 %10
  %12 = load <4 x float>, ptr addrspace(1) %11, align 16, !tbaa !28, !alias.scope !31, !noalias !34
  %13 = getelementptr inbounds <4 x float>, ptr addrspace(1) %1, i64 %10
  %14 = load <4 x float>, ptr addrspace(1) %13, align 16, !tbaa !28, !alias.scope !42, !noalias !43
  %15 = getelementptr inbounds <4 x float>, ptr addrspace(1) %2, i64 %10
  %16 = load <4 x float>, ptr addrspace(1) %15, align 16, !tbaa !28, !alias.scope !44, !noalias !45
  %17 = getelementptr inbounds <4 x half>, ptr addrspace(1) %3, i64 %10
  %18 = load <4 x half>, ptr addrspace(1) %17, align 8, !tbaa !28, !alias.scope !46, !noalias !47
  %19 = getelementptr inbounds <4 x half>, ptr addrspace(1) %4, i64 %10
  %20 = load <4 x half>, ptr addrspace(1) %19, align 8, !tbaa !28, !alias.scope !48, !noalias !49
  %21 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %5, i64 %10
  %22 = load <4 x i32>, ptr addrspace(1) %21, align 16, !tbaa !28, !alias.scope !50, !noalias !51
  %23 = getelementptr inbounds <4 x i32>, ptr addrspace(1) %6, i64 %10
  %24 = load <4 x i32>, ptr addrspace(1) %23, align 16, !tbaa !28, !alias.scope !52, !noalias !53
  %25 = shufflevector <4 x float> %16, <4 x float> poison, <2 x i32> <i32 0, i32 1>
  %26 = shufflevector <4 x i32> %22, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
  %27 = freeze <4 x i32> %22
  %28 = extractelement <4 x i32> %27, i64 2
  %29 = trunc i32 %28 to i16
  %30 = extractelement <4 x i32> %24, i64 0
  %31 = trunc i32 %30 to i16
  %32 = insertelement <2 x i16> undef, i16 %31, i64 0
  %33 = extractelement <4 x i32> %24, i64 1
  %34 = trunc i32 %33 to i16
  %35 = insertelement <2 x i16> %32, i16 %34, i64 1
  %36 = freeze <4 x i32> %24
  %37 = extractelement <4 x i32> %36, i64 2
  %38 = trunc i32 %37 to i8
  %39 = mul i32 %8, 33
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds i32, ptr addrspace(1) %7, i64 %40
  %42 = tail call fast <2 x float> @air.simd_sum.v2f32(<2 x float> %25) #3
  %43 = bitcast <2 x float> %42 to <2 x i32>
  %44 = extractelement <2 x i32> %43, i64 0
  store i32 %44, ptr addrspace(1) %41, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %45 = extractelement <2 x i32> %43, i64 1
  %46 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 1
  store i32 %45, ptr addrspace(1) %46, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %47 = tail call <2 x i32> @air.simd_sum.u.v2i32(<2 x i32> %26) #3
  %48 = extractelement <2 x i32> %47, i64 0
  %49 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 2
  store i32 %48, ptr addrspace(1) %49, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %50 = extractelement <2 x i32> %47, i64 1
  %51 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 3
  store i32 %50, ptr addrspace(1) %51, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %52 = freeze <4 x float> %12
  %53 = tail call fast <4 x float> @air.simd_shuffle_xor.v4f32(<4 x float> %52, i16 8) #3
  %54 = bitcast <4 x float> %53 to <4 x i32>
  %55 = extractelement <4 x i32> %54, i64 0
  %56 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 4
  store i32 %55, ptr addrspace(1) %56, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %57 = extractelement <4 x i32> %54, i64 1
  %58 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 5
  store i32 %57, ptr addrspace(1) %58, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %59 = extractelement <4 x i32> %54, i64 2
  %60 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 6
  store i32 %59, ptr addrspace(1) %60, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %61 = extractelement <4 x i32> %54, i64 3
  %62 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 7
  store i32 %61, ptr addrspace(1) %62, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %63 = tail call i16 @air.simd_shuffle_xor.u.i16(i16 %29, i16 4) #3
  %64 = zext i16 %63 to i32
  %65 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 8
  store i32 %64, ptr addrspace(1) %65, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %66 = freeze <2 x i16> %35
  %67 = tail call <2 x i16> @air.simd_shuffle_xor.s.v2i16(<2 x i16> %66, i16 16) #3
  %68 = extractelement <2 x i16> %67, i64 0
  %69 = zext i16 %68 to i32
  %70 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 9
  store i32 %69, ptr addrspace(1) %70, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %71 = extractelement <2 x i16> %67, i64 1
  %72 = zext i16 %71 to i32
  %73 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 10
  store i32 %72, ptr addrspace(1) %73, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %74 = freeze <4 x half> %18
  %75 = tail call fast <4 x half> @air.simd_shuffle_up.v4f16(<4 x half> %74, i16 3) #3
  %76 = tail call i8 @air.simd_shuffle_up.u.i8(i8 %38, i16 2) #3
  %77 = tail call i16 @air.simd_shuffle_up.u.i16(i16 %29, i16 5) #3
  %78 = icmp ugt i32 %8, 2
  br i1 %78, label %79, label %89

79:                                               ; preds = %9
  %80 = bitcast <4 x half> %75 to <4 x i16>
  %81 = extractelement <4 x i16> %80, i64 0
  %82 = zext i16 %81 to i32
  %83 = extractelement <4 x i16> %80, i64 1
  %84 = zext i16 %83 to i32
  %85 = extractelement <4 x i16> %80, i64 2
  %86 = zext i16 %85 to i32
  %87 = extractelement <4 x i16> %80, i64 3
  %88 = zext i16 %87 to i32
  br label %89

89:                                               ; preds = %79, %9
  %90 = phi i32 [ %82, %79 ], [ -1412584499, %9 ]
  %91 = phi i32 [ %84, %79 ], [ -1412584499, %9 ]
  %92 = phi i32 [ %86, %79 ], [ -1412584499, %9 ]
  %93 = phi i32 [ %88, %79 ], [ -1412584499, %9 ]
  %94 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 11
  store i32 %90, ptr addrspace(1) %94, align 4
  %95 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 12
  store i32 %91, ptr addrspace(1) %95, align 4
  %96 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 13
  store i32 %92, ptr addrspace(1) %96, align 4
  %97 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 14
  store i32 %93, ptr addrspace(1) %97, align 4
  %98 = extractelement <4 x i32> %27, i64 3
  %99 = trunc i32 %98 to i16
  %100 = icmp ugt i32 %8, 1
  %101 = zext i8 %76 to i32
  %102 = select i1 %100, i32 %101, i32 -1412584499
  %103 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 15
  store i32 %102, ptr addrspace(1) %103, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %104 = icmp ugt i32 %8, 4
  %105 = zext i16 %77 to i32
  %106 = select i1 %104, i32 %105, i32 -1412584499
  %107 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 16
  store i32 %106, ptr addrspace(1) %107, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %108 = tail call i16 @air.get_simdgroup_size.i16() #4
  %109 = tail call fast <4 x float> @air.simd_shuffle_and_fill_down.v4f32(<4 x float> %52, <4 x float> %14, i16 6, i16 %108) #3
  %110 = bitcast <4 x float> %109 to <4 x i32>
  %111 = extractelement <4 x i32> %110, i64 0
  %112 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 17
  store i32 %111, ptr addrspace(1) %112, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %113 = extractelement <4 x i32> %110, i64 1
  %114 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 18
  store i32 %113, ptr addrspace(1) %114, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %115 = extractelement <4 x i32> %110, i64 2
  %116 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 19
  store i32 %115, ptr addrspace(1) %116, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %117 = extractelement <4 x i32> %110, i64 3
  %118 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 20
  store i32 %117, ptr addrspace(1) %118, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %119 = shufflevector <4 x half> %74, <4 x half> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %120 = shufflevector <4 x half> %20, <4 x half> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %121 = tail call fast <3 x half> @air.simd_shuffle_and_fill_down.v3f16(<3 x half> %119, <3 x half> %120, i16 9, i16 %108) #3
  %122 = bitcast <3 x half> %121 to <3 x i16>
  %123 = extractelement <3 x i16> %122, i64 0
  %124 = zext i16 %123 to i32
  %125 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 21
  store i32 %124, ptr addrspace(1) %125, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %126 = extractelement <3 x i16> %122, i64 1
  %127 = zext i16 %126 to i32
  %128 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 22
  store i32 %127, ptr addrspace(1) %128, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %129 = extractelement <3 x i16> %122, i64 2
  %130 = zext i16 %129 to i32
  %131 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 23
  store i32 %130, ptr addrspace(1) %131, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %132 = extractelement <4 x float> %16, i64 2
  %133 = tail call fast float @air.simd_prefix_inclusive_sum.f32(float %132) #3
  %134 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 24
  %135 = bitcast ptr addrspace(1) %134 to ptr addrspace(1)
  store float %133, ptr addrspace(1) %135, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %136 = tail call i16 @air.simd_prefix_exclusive_sum.u.i16(i16 %99) #3
  %137 = zext i16 %136 to i32
  %138 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 25
  store i32 %137, ptr addrspace(1) %138, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %139 = tail call <2 x i16> @air.simd_min.s.v2i16(<2 x i16> %66) #3
  %140 = tail call <2 x i16> @air.simd_max.s.v2i16(<2 x i16> %66) #3
  %141 = extractelement <2 x i16> %139, i64 0
  %142 = zext i16 %141 to i32
  %143 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 26
  store i32 %142, ptr addrspace(1) %143, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %144 = extractelement <2 x i16> %139, i64 1
  %145 = zext i16 %144 to i32
  %146 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 27
  store i32 %145, ptr addrspace(1) %146, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %147 = extractelement <2 x i16> %140, i64 0
  %148 = zext i16 %147 to i32
  %149 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 28
  store i32 %148, ptr addrspace(1) %149, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %150 = extractelement <2 x i16> %140, i64 1
  %151 = zext i16 %150 to i32
  %152 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 29
  store i32 %151, ptr addrspace(1) %152, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %153 = freeze <4 x float> %16
  %154 = extractelement <4 x float> %153, i64 3
  %155 = tail call fast float @air.simd_broadcast_first.f32(float %154) #3
  %156 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 30
  %157 = bitcast ptr addrspace(1) %156 to ptr addrspace(1)
  store float %155, ptr addrspace(1) %157, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %158 = extractelement <4 x i32> %36, i64 3
  %159 = icmp ne i32 %158, 0
  %160 = tail call i1 @air.simd_all(i1 %159) #3
  %161 = zext i1 %160 to i32
  %162 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 31
  store i32 %161, ptr addrspace(1) %162, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  %163 = icmp sgt i32 %158, -1
  %164 = tail call i1 @air.simd_all(i1 %163) #3
  %165 = zext i1 %164 to i32
  %166 = getelementptr inbounds i32, ptr addrspace(1) %41, i64 32
  store i32 %165, ptr addrspace(1) %166, align 4, !tbaa !54, !alias.scope !56, !noalias !57
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare i1 @air.simd_all(i1) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x float> @air.simd_sum.v2f32(<2 x float>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x i32> @air.simd_sum.u.v2i32(<2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x float> @air.simd_shuffle_xor.v4f32(<4 x float>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i16 @air.simd_shuffle_xor.u.i16(i16, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x i16> @air.simd_shuffle_xor.s.v2i16(<2 x i16>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x half> @air.simd_shuffle_up.v4f16(<4 x half>, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i8 @air.simd_shuffle_up.u.i8(i8, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i16 @air.simd_shuffle_up.u.i16(i16, i16) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.get_simdgroup_size.i16() local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x float> @air.simd_shuffle_and_fill_down.v4f32(<4 x float>, <4 x float>, i16, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x half> @air.simd_shuffle_and_fill_down.v3f16(<3 x half>, <3 x half>, i16, i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_prefix_inclusive_sum.f32(float) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i16 @air.simd_prefix_exclusive_sum.u.i16(i16) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x i16> @air.simd_min.s.v2i16(<2 x i16>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x i16> @air.simd_max.s.v2i16(<2 x i16>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_broadcast_first.f32(float) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent nounwind willreturn }
attributes #4 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!21, !22, !23}
!llvm.ident = !{!24}
!air.version = !{!25}
!air.language_version = !{!26}
!air.source_file_name = !{!27}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_simd_variant_leftovers, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19, !20}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"fa"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"fb"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"fc"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"ha"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"hb"}
!17 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"uint4", !"air.arg_name", !"ua"}
!18 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4", !"air.arg_name", !"ia"}
!19 = !{i32 7, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!20 = !{i32 8, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!21 = !{!"air.compile.denorms_disable"}
!22 = !{!"air.compile.fast_math_enable"}
!23 = !{!"air.compile.framebuffer_fetch_enable"}
!24 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!25 = !{i32 2, i32 8, i32 0}
!26 = !{!"Metal", i32 4, i32 0, i32 0}
!27 = !{!"/private/tmp/simdv/k.metal"}
!28 = !{!29, !29, i64 0}
!29 = !{!"omnipotent char", !30, i64 0}
!30 = !{!"Simple C++ TBAA"}
!31 = !{!32}
!32 = distinct !{!32, !33, !"air-alias-scope-arg(0)"}
!33 = distinct !{!33, !"air-alias-scopes(kernel_simd_variant_leftovers)"}
!34 = !{!35, !36, !37, !38, !39, !40, !41}
!35 = distinct !{!35, !33, !"air-alias-scope-arg(1)"}
!36 = distinct !{!36, !33, !"air-alias-scope-arg(2)"}
!37 = distinct !{!37, !33, !"air-alias-scope-arg(3)"}
!38 = distinct !{!38, !33, !"air-alias-scope-arg(4)"}
!39 = distinct !{!39, !33, !"air-alias-scope-arg(5)"}
!40 = distinct !{!40, !33, !"air-alias-scope-arg(6)"}
!41 = distinct !{!41, !33, !"air-alias-scope-arg(7)"}
!42 = !{!35}
!43 = !{!32, !36, !37, !38, !39, !40, !41}
!44 = !{!36}
!45 = !{!32, !35, !37, !38, !39, !40, !41}
!46 = !{!37}
!47 = !{!32, !35, !36, !38, !39, !40, !41}
!48 = !{!38}
!49 = !{!32, !35, !36, !37, !39, !40, !41}
!50 = !{!39}
!51 = !{!32, !35, !36, !37, !38, !40, !41}
!52 = !{!40}
!53 = !{!32, !35, !36, !37, !38, !39, !41}
!54 = !{!55, !55, i64 0}
!55 = !{!"int", !29, i64 0}
!56 = !{!41}
!57 = !{!32, !35, !36, !37, !38, !39, !40}
