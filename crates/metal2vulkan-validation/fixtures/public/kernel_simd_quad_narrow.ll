; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 's.bc'
source_filename = "validation/fixtures/public/kernel_simd_quad_narrow.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_simd_quad_narrow(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add i32 %1, 1
  %4 = tail call fast half @air.convert.f.f16.u.i32(i32 %3) #3
  %5 = insertelement <4 x half> undef, half %4, i64 0
  %6 = fadd fast half %4, 0xH3C00
  %7 = insertelement <4 x half> %5, half %6, i64 1
  %8 = fadd fast half %4, 0xH4000
  %9 = insertelement <4 x half> %7, half %8, i64 2
  %10 = fadd fast half %4, 0xH4200
  %11 = insertelement <4 x half> %9, half %10, i64 3
  %12 = freeze <4 x half> %11
  %13 = tail call fast <4 x half> @air.simd_shuffle_down.v4f16(<4 x half> %12, i16 5) #4
  %14 = trunc i32 %3 to i8
  %15 = mul i8 %14, 3
  %16 = add i8 %15, -40
  %17 = trunc i32 %3 to i16
  %18 = mul i16 %17, 300
  %19 = add i16 %18, -5000
  %20 = tail call i8 @air.simd_shuffle_down.u.i8(i8 %15, i16 3) #4
  %21 = tail call i8 @air.simd_shuffle_down.s.i8(i8 %16, i16 3) #4
  %22 = tail call i16 @air.simd_shuffle_down.s.i16(i16 %19, i16 6) #4
  %23 = tail call fast float @air.convert.f.f32.u.i32(i32 %3) #3
  %24 = insertelement <3 x float> undef, float %23, i64 0
  %25 = fmul fast float %23, 2.000000e+00
  %26 = insertelement <3 x float> %24, float %25, i64 1
  %27 = fmul fast float %23, 4.000000e+00
  %28 = insertelement <3 x float> %26, float %27, i64 2
  %29 = insertelement <2 x float> undef, float %23, i64 0
  %30 = fmul fast float %23, 8.000000e+00
  %31 = insertelement <2 x float> %29, float %30, i64 1
  %32 = tail call fast <3 x float> @air.quad_sum.v3f32(<3 x float> %28) #4
  %33 = tail call fast <2 x float> @air.quad_sum.v2f32(<2 x float> %31) #4
  %34 = tail call fast <3 x float> @air.quad_shuffle_down.v3f32(<3 x float> %28, i16 1) #4
  %35 = freeze <2 x float> %31
  %36 = tail call fast <2 x float> @air.quad_shuffle_down.v2f32(<2 x float> %35, i16 2) #4
  %37 = mul i32 %1, 17
  %38 = bitcast <4 x half> %13 to <4 x i16>
  %39 = extractelement <4 x i16> %38, i64 0
  %40 = zext i16 %39 to i32
  %41 = zext i32 %37 to i64
  %42 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %41
  store i32 %40, ptr addrspace(1) %42, align 4, !tbaa !21, !alias.scope !25
  %43 = extractelement <4 x i16> %38, i64 1
  %44 = zext i16 %43 to i32
  %45 = add i32 %37, 1
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %46
  store i32 %44, ptr addrspace(1) %47, align 4, !tbaa !21, !alias.scope !25
  %48 = extractelement <4 x i16> %38, i64 2
  %49 = zext i16 %48 to i32
  %50 = add i32 %37, 2
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %51
  store i32 %49, ptr addrspace(1) %52, align 4, !tbaa !21, !alias.scope !25
  %53 = extractelement <4 x i16> %38, i64 3
  %54 = zext i16 %53 to i32
  %55 = add i32 %37, 3
  %56 = zext i32 %55 to i64
  %57 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %56
  store i32 %54, ptr addrspace(1) %57, align 4, !tbaa !21, !alias.scope !25
  %58 = zext i8 %20 to i32
  %59 = add i32 %37, 4
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %60
  store i32 %58, ptr addrspace(1) %61, align 4, !tbaa !21, !alias.scope !25
  %62 = sext i8 %21 to i32
  %63 = add i32 %37, 5
  %64 = zext i32 %63 to i64
  %65 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %64
  store i32 %62, ptr addrspace(1) %65, align 4, !tbaa !21, !alias.scope !25
  %66 = sext i16 %22 to i32
  %67 = add i32 %37, 6
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %68
  store i32 %66, ptr addrspace(1) %69, align 4, !tbaa !21, !alias.scope !25
  %70 = bitcast <3 x float> %32 to <3 x i32>
  %71 = extractelement <3 x i32> %70, i64 0
  %72 = add i32 %37, 7
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %73
  store i32 %71, ptr addrspace(1) %74, align 4, !tbaa !21, !alias.scope !25
  %75 = extractelement <3 x i32> %70, i64 1
  %76 = add i32 %37, 8
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %77
  store i32 %75, ptr addrspace(1) %78, align 4, !tbaa !21, !alias.scope !25
  %79 = extractelement <3 x i32> %70, i64 2
  %80 = add i32 %37, 9
  %81 = zext i32 %80 to i64
  %82 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %81
  store i32 %79, ptr addrspace(1) %82, align 4, !tbaa !21, !alias.scope !25
  %83 = bitcast <2 x float> %33 to <2 x i32>
  %84 = extractelement <2 x i32> %83, i64 0
  %85 = add i32 %37, 10
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %86
  store i32 %84, ptr addrspace(1) %87, align 4, !tbaa !21, !alias.scope !25
  %88 = extractelement <2 x i32> %83, i64 1
  %89 = add i32 %37, 11
  %90 = zext i32 %89 to i64
  %91 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %90
  store i32 %88, ptr addrspace(1) %91, align 4, !tbaa !21, !alias.scope !25
  %92 = bitcast <3 x float> %34 to <3 x i32>
  %93 = extractelement <3 x i32> %92, i64 0
  %94 = add i32 %37, 12
  %95 = zext i32 %94 to i64
  %96 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %95
  store i32 %93, ptr addrspace(1) %96, align 4, !tbaa !21, !alias.scope !25
  %97 = extractelement <3 x i32> %92, i64 1
  %98 = add i32 %37, 13
  %99 = zext i32 %98 to i64
  %100 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %99
  store i32 %97, ptr addrspace(1) %100, align 4, !tbaa !21, !alias.scope !25
  %101 = extractelement <3 x i32> %92, i64 2
  %102 = add i32 %37, 14
  %103 = zext i32 %102 to i64
  %104 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %103
  store i32 %101, ptr addrspace(1) %104, align 4, !tbaa !21, !alias.scope !25
  %105 = bitcast <2 x float> %36 to <2 x i32>
  %106 = extractelement <2 x i32> %105, i64 0
  %107 = add i32 %37, 15
  %108 = zext i32 %107 to i64
  %109 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %108
  store i32 %106, ptr addrspace(1) %109, align 4, !tbaa !21, !alias.scope !25
  %110 = extractelement <2 x i32> %105, i64 1
  %111 = add i32 %37, 16
  %112 = zext i32 %111 to i64
  %113 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %112
  store i32 %110, ptr addrspace(1) %113, align 4, !tbaa !21, !alias.scope !25
  %114 = icmp ugt i32 %1, 26
  br i1 %114, label %115, label %116

115:                                              ; preds = %2
  store i32 0, ptr addrspace(1) %42, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %47, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %52, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %57, align 4, !tbaa !21, !alias.scope !25
  br label %116

116:                                              ; preds = %115, %2
  %117 = icmp ugt i32 %1, 28
  br i1 %117, label %118, label %119

118:                                              ; preds = %116
  store i32 0, ptr addrspace(1) %61, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %65, align 4, !tbaa !21, !alias.scope !25
  br label %119

119:                                              ; preds = %118, %116
  %120 = icmp ugt i32 %1, 25
  br i1 %120, label %121, label %122

121:                                              ; preds = %119
  store i32 0, ptr addrspace(1) %69, align 4, !tbaa !21, !alias.scope !25
  br label %122

122:                                              ; preds = %121, %119
  %123 = and i32 %1, 3
  %124 = icmp eq i32 %123, 3
  br i1 %124, label %125, label %126

125:                                              ; preds = %122
  store i32 0, ptr addrspace(1) %96, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %100, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %104, align 4, !tbaa !21, !alias.scope !25
  br label %126

126:                                              ; preds = %125, %122
  %127 = icmp ugt i32 %123, 1
  br i1 %127, label %128, label %129

128:                                              ; preds = %126
  store i32 0, ptr addrspace(1) %109, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %113, align 4, !tbaa !21, !alias.scope !25
  br label %129

129:                                              ; preds = %128, %126
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x half> @air.simd_shuffle_down.v4f16(<4 x half>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare i8 @air.simd_shuffle_down.u.i8(i8, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare i8 @air.simd_shuffle_down.s.i8(i8, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare i16 @air.simd_shuffle_down.s.i16(i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x float> @air.quad_sum.v3f32(<3 x float>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x float> @air.quad_sum.v2f32(<2 x float>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x float> @air.quad_shuffle_down.v3f32(<3 x float>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x float> @air.quad_shuffle_down.v2f32(<2 x float>, i16) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
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
!9 = !{ptr @kernel_simd_quad_narrow, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simd_quad_narrow.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_simd_quad_narrow)"}
