; ModuleID = '/tmp/b3/f.bc'
source_filename = "validation/fixtures/public/kernel_simd_and_quad_shuffles.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_simd_and_quad_shuffles(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call fast half @air.convert.f.f16.u.i32(i32 %1) #3
  %4 = insertelement <4 x half> undef, half %3, i64 0
  %5 = fadd fast half %3, 0xH3C00
  %6 = insertelement <4 x half> %4, half %5, i64 1
  %7 = fadd fast half %3, 0xH4000
  %8 = insertelement <4 x half> %6, half %7, i64 2
  %9 = fadd fast half %3, 0xH4200
  %10 = insertelement <4 x half> %8, half %9, i64 3
  %11 = insertelement <2 x half> undef, half %3, i64 0
  %12 = fadd fast half %3, 0xH4C00
  %13 = insertelement <2 x half> %11, half %12, i64 1
  %14 = insertelement <3 x half> undef, half %3, i64 0
  %15 = fmul fast half %3, 0xH4000
  %16 = insertelement <3 x half> %14, half %15, i64 1
  %17 = fmul fast half %3, 0xH4400
  %18 = insertelement <3 x half> %16, half %17, i64 2
  %19 = trunc i32 %1 to i16
  %20 = insertelement <4 x i16> poison, i16 %19, i64 0
  %21 = add i16 %19, 100
  %22 = insertelement <4 x i16> %20, i16 %21, i64 1
  %23 = add i16 %19, 200
  %24 = insertelement <4 x i16> %22, i16 %23, i64 2
  %25 = add i16 %19, 300
  %26 = insertelement <4 x i16> %24, i16 %25, i64 3
  %27 = add i32 %1, 1
  %28 = trunc i32 %27 to i16
  %29 = insertelement <2 x i16> undef, i16 %28, i64 0
  %30 = mul i16 %28, 7
  %31 = insertelement <2 x i16> %29, i16 %30, i64 1
  %32 = tail call fast float @air.convert.f.f32.u.i32(i32 %1) #3
  %33 = fmul fast float %32, 2.000000e+00
  %34 = insertelement <4 x float> <float poison, float poison, float 1.000000e+00, float 2.000000e+00>, float %32, i64 0
  %35 = insertelement <4 x float> %34, float %33, i64 1
  %36 = freeze <4 x half> %10
  %37 = tail call fast <4 x half> @air.simd_shuffle_xor.v4f16(<4 x half> %36, i16 1) #4
  %38 = freeze <2 x half> %13
  %39 = tail call fast <2 x half> @air.simd_shuffle_xor.v2f16(<2 x half> %38, i16 2) #4
  %40 = freeze <3 x half> %18
  %41 = tail call fast <3 x half> @air.simd_shuffle_up.v3f16(<3 x half> %40, i16 1) #4
  %42 = tail call <4 x i16> @air.simd_shuffle_down.u.v4i16(<4 x i16> %26, i16 1) #4
  %43 = tail call i16 @air.get_simdgroup_size.i16() #3
  %44 = tail call fast <2 x half> @air.simd_shuffle_and_fill_down.v2f16(<2 x half> %38, <2 x half> splat (half 0xH5630), i16 1, i16 %43) #4
  %45 = tail call fast <4 x float> @air.simd_prefix_exclusive_sum.v4f32(<4 x float> %35) #4
  %46 = tail call <2 x i16> @air.simd_xor.u.v2i16(<2 x i16> %31) #4
  %47 = freeze half %3
  %48 = tail call fast half @air.quad_shuffle_xor.f16(half %47, i16 1) #4
  %49 = tail call fast <2 x half> @air.quad_shuffle_xor.v2f16(<2 x half> %38, i16 2) #4
  %50 = tail call fast <3 x half> @air.quad_shuffle_xor.v3f16(<3 x half> %40, i16 3) #4
  %51 = mul i32 %1, 10
  %52 = bitcast <4 x half> %37 to <4 x i16>
  %53 = extractelement <4 x i16> %52, i64 3
  %54 = zext i16 %53 to i32
  %55 = zext i32 %51 to i64
  %56 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %55
  store i32 %54, ptr addrspace(1) %56, align 4, !tbaa !21, !alias.scope !25
  %57 = bitcast <2 x half> %39 to <2 x i16>
  %58 = extractelement <2 x i16> %57, i64 1
  %59 = zext i16 %58 to i32
  %60 = or i32 %51, 1
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %61
  store i32 %59, ptr addrspace(1) %62, align 4, !tbaa !21, !alias.scope !25
  %63 = icmp eq i32 %1, 0
  br i1 %63, label %71, label %64

64:                                               ; preds = %2
  %65 = bitcast <3 x half> %41 to <3 x i16>
  %66 = extractelement <3 x i16> %65, i64 2
  %67 = zext i16 %66 to i32
  %68 = add i32 %51, 2
  %69 = zext i32 %68 to i64
  %70 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %69
  store i32 %67, ptr addrspace(1) %70, align 4, !tbaa !21, !alias.scope !25
  br label %75

71:                                               ; preds = %2
  %72 = add nuw nsw i32 %51, 2
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %73
  store i32 43981, ptr addrspace(1) %74, align 4, !tbaa !21, !alias.scope !25
  br label %75

75:                                               ; preds = %71, %64
  %76 = icmp ult i32 %27, 32
  br i1 %76, label %77, label %83

77:                                               ; preds = %75
  %78 = extractelement <4 x i16> %42, i64 2
  %79 = zext i16 %78 to i32
  %80 = add i32 %51, 3
  %81 = zext i32 %80 to i64
  %82 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %81
  store i32 %79, ptr addrspace(1) %82, align 4, !tbaa !21, !alias.scope !25
  br label %87

83:                                               ; preds = %75
  %84 = add i32 %51, 3
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %85
  store i32 43981, ptr addrspace(1) %86, align 4, !tbaa !21, !alias.scope !25
  br label %87

87:                                               ; preds = %83, %77
  %88 = bitcast <2 x half> %44 to <2 x i16>
  %89 = extractelement <2 x i16> %88, i64 1
  %90 = zext i16 %89 to i32
  %91 = add i32 %51, 4
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %92
  store i32 %90, ptr addrspace(1) %93, align 4, !tbaa !21, !alias.scope !25
  %94 = extractelement <4 x float> %45, i64 1
  %95 = fadd fast float %94, 1.000000e+00
  %96 = add i32 %51, 5
  %97 = zext i32 %96 to i64
  %98 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %97
  %99 = bitcast ptr addrspace(1) %98 to ptr addrspace(1)
  store float %95, ptr addrspace(1) %99, align 4, !tbaa !21, !alias.scope !25
  %100 = extractelement <2 x i16> %46, i64 0
  %101 = zext i16 %100 to i32
  %102 = add i32 %51, 6
  %103 = zext i32 %102 to i64
  %104 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %103
  store i32 %101, ptr addrspace(1) %104, align 4, !tbaa !21, !alias.scope !25
  %105 = bitcast half %48 to i16
  %106 = zext i16 %105 to i32
  %107 = add i32 %51, 7
  %108 = zext i32 %107 to i64
  %109 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %108
  store i32 %106, ptr addrspace(1) %109, align 4, !tbaa !21, !alias.scope !25
  %110 = bitcast <2 x half> %49 to <2 x i16>
  %111 = extractelement <2 x i16> %110, i64 1
  %112 = zext i16 %111 to i32
  %113 = add i32 %51, 8
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %114
  store i32 %112, ptr addrspace(1) %115, align 4, !tbaa !21, !alias.scope !25
  %116 = bitcast <3 x half> %50 to <3 x i16>
  %117 = extractelement <3 x i16> %116, i64 2
  %118 = zext i16 %117 to i32
  %119 = add i32 %51, 9
  %120 = zext i32 %119 to i64
  %121 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %120
  store i32 %118, ptr addrspace(1) %121, align 4, !tbaa !21, !alias.scope !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x half> @air.simd_shuffle_xor.v4f16(<4 x half>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x half> @air.simd_shuffle_xor.v2f16(<2 x half>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x half> @air.simd_shuffle_up.v3f16(<3 x half>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x i16> @air.simd_shuffle_down.u.v4i16(<4 x i16>, i16) local_unnamed_addr #2

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.get_simdgroup_size.i16() local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x half> @air.simd_shuffle_and_fill_down.v2f16(<2 x half>, <2 x half>, i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x float> @air.simd_prefix_exclusive_sum.v4f32(<4 x float>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x i16> @air.simd_xor.u.v2i16(<2 x i16>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare half @air.quad_shuffle_xor.f16(half, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x half> @air.quad_shuffle_xor.v2f16(<2 x half>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x half> @air.quad_shuffle_xor.v3f16(<3 x half>, i16) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent mustprogress nounwind willreturn }
attributes #3 = { nounwind willreturn memory(none) }
attributes #4 = { convergent nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!14, !15, !16}
!llvm.ident = !{!17}
!air.version = !{!18}
!air.language_version = !{!19}
!air.source_file_name = !{!20}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_simd_and_quad_shuffles, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 3, i32 2, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simd_and_quad_shuffles.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_simd_and_quad_shuffles)"}
