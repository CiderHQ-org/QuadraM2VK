; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 's.bc'
source_filename = "validation/fixtures/public/kernel_simd_shuffle_and_fill.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_simd_shuffle_and_fill(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add i32 %1, 1
  %4 = add i32 %1, 100
  %5 = tail call i16 @air.get_simdgroup_size.i16() #3
  %6 = tail call i32 @air.simd_shuffle_and_fill_up.u.i32(i32 %3, i32 %4, i16 3, i16 %5) #4
  %7 = tail call i32 @air.simd_shuffle_and_fill_down.u.i32(i32 %3, i32 %4, i16 5, i16 %5) #4
  %8 = insertelement <2 x i32> poison, i32 %3, i64 0
  %9 = shl i32 %3, 1
  %10 = insertelement <2 x i32> %8, i32 %9, i64 1
  %11 = tail call <2 x i32> @air.simd_shuffle_down.u.v2i32(<2 x i32> %10, i16 7) #4
  %12 = tail call fast float @air.convert.f.f32.u.i32(i32 %3) #3
  %13 = insertelement <3 x float> undef, float %12, i64 0
  %14 = fmul fast float %12, 2.000000e+00
  %15 = insertelement <3 x float> %13, float %14, i64 1
  %16 = fmul fast float %12, 4.000000e+00
  %17 = insertelement <3 x float> %15, float %16, i64 2
  %18 = tail call fast float @air.convert.f.f32.u.i32(i32 %4) #3
  %19 = insertelement <3 x float> undef, float %18, i64 0
  %20 = fmul fast float %18, 2.000000e+00
  %21 = insertelement <3 x float> %19, float %20, i64 1
  %22 = fmul fast float %18, 4.000000e+00
  %23 = insertelement <3 x float> %21, float %22, i64 2
  %24 = freeze <3 x float> %17
  %25 = tail call fast <3 x float> @air.simd_shuffle_and_fill_up.v3f32(<3 x float> %24, <3 x float> %23, i16 2, i16 %5) #4
  %26 = tail call fast <3 x float> @air.simd_shuffle_and_fill_down.v3f32(<3 x float> %24, <3 x float> %23, i16 4, i16 %5) #4
  %27 = tail call fast <3 x float> @air.simd_shuffle_up.v3f32(<3 x float> %24, i16 6) #4
  %28 = freeze float %12
  %29 = tail call fast float @air.simd_shuffle_and_fill_up.f32(float %28, float %18, i16 1, i16 %5) #4
  %30 = tail call fast float @air.simd_shuffle_and_fill_down.f32(float %28, float %18, i16 9, i16 %5) #4
  %31 = tail call fast half @air.convert.f.f16.u.i32(i32 %3) #3
  %32 = insertelement <4 x half> undef, half %31, i64 0
  %33 = fadd fast half %31, 0xH3C00
  %34 = insertelement <4 x half> %32, half %33, i64 1
  %35 = fadd fast half %31, 0xH4000
  %36 = insertelement <4 x half> %34, half %35, i64 2
  %37 = fadd fast half %31, 0xH4200
  %38 = insertelement <4 x half> %36, half %37, i64 3
  %39 = tail call fast half @air.convert.f.f16.u.i32(i32 %4) #3
  %40 = insertelement <4 x half> undef, half %39, i64 0
  %41 = fadd fast half %39, 0xH3C00
  %42 = insertelement <4 x half> %40, half %41, i64 1
  %43 = fadd fast half %39, 0xH4000
  %44 = insertelement <4 x half> %42, half %43, i64 2
  %45 = fadd fast half %39, 0xH4200
  %46 = insertelement <4 x half> %44, half %45, i64 3
  %47 = freeze <4 x half> %38
  %48 = tail call fast <4 x half> @air.simd_shuffle_and_fill_down.v4f16(<4 x half> %47, <4 x half> %46, i16 3, i16 %5) #4
  %49 = insertelement <3 x half> undef, half %31, i64 0
  %50 = fmul fast half %31, 0xH4000
  %51 = insertelement <3 x half> %49, half %50, i64 1
  %52 = fmul fast half %31, 0xH4400
  %53 = insertelement <3 x half> %51, half %52, i64 2
  %54 = freeze <3 x half> %53
  %55 = tail call fast <3 x half> @air.simd_shuffle_down.v3f16(<3 x half> %54, i16 5) #4
  %56 = mul i32 %1, 22
  %57 = zext i32 %56 to i64
  %58 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %57
  store i32 %6, ptr addrspace(1) %58, align 4, !tbaa !21, !alias.scope !25
  %59 = or i32 %56, 1
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %60
  store i32 %7, ptr addrspace(1) %61, align 4, !tbaa !21, !alias.scope !25
  %62 = extractelement <2 x i32> %11, i64 0
  %63 = add i32 %56, 2
  %64 = zext i32 %63 to i64
  %65 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %64
  store i32 %62, ptr addrspace(1) %65, align 4, !tbaa !21, !alias.scope !25
  %66 = extractelement <2 x i32> %11, i64 1
  %67 = add i32 %56, 3
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %68
  store i32 %66, ptr addrspace(1) %69, align 4, !tbaa !21, !alias.scope !25
  %70 = bitcast <3 x float> %25 to <3 x i32>
  %71 = extractelement <3 x i32> %70, i64 0
  %72 = add i32 %56, 4
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %73
  store i32 %71, ptr addrspace(1) %74, align 4, !tbaa !21, !alias.scope !25
  %75 = extractelement <3 x i32> %70, i64 1
  %76 = add i32 %56, 5
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %77
  store i32 %75, ptr addrspace(1) %78, align 4, !tbaa !21, !alias.scope !25
  %79 = extractelement <3 x i32> %70, i64 2
  %80 = add i32 %56, 6
  %81 = zext i32 %80 to i64
  %82 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %81
  store i32 %79, ptr addrspace(1) %82, align 4, !tbaa !21, !alias.scope !25
  %83 = bitcast <3 x float> %26 to <3 x i32>
  %84 = extractelement <3 x i32> %83, i64 0
  %85 = add i32 %56, 7
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %86
  store i32 %84, ptr addrspace(1) %87, align 4, !tbaa !21, !alias.scope !25
  %88 = extractelement <3 x i32> %83, i64 1
  %89 = add i32 %56, 8
  %90 = zext i32 %89 to i64
  %91 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %90
  store i32 %88, ptr addrspace(1) %91, align 4, !tbaa !21, !alias.scope !25
  %92 = extractelement <3 x i32> %83, i64 2
  %93 = add i32 %56, 9
  %94 = zext i32 %93 to i64
  %95 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %94
  store i32 %92, ptr addrspace(1) %95, align 4, !tbaa !21, !alias.scope !25
  %96 = bitcast <3 x float> %27 to <3 x i32>
  %97 = extractelement <3 x i32> %96, i64 0
  %98 = add i32 %56, 10
  %99 = zext i32 %98 to i64
  %100 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %99
  store i32 %97, ptr addrspace(1) %100, align 4, !tbaa !21, !alias.scope !25
  %101 = extractelement <3 x i32> %96, i64 1
  %102 = add i32 %56, 11
  %103 = zext i32 %102 to i64
  %104 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %103
  store i32 %101, ptr addrspace(1) %104, align 4, !tbaa !21, !alias.scope !25
  %105 = extractelement <3 x i32> %96, i64 2
  %106 = add i32 %56, 12
  %107 = zext i32 %106 to i64
  %108 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %107
  store i32 %105, ptr addrspace(1) %108, align 4, !tbaa !21, !alias.scope !25
  %109 = add i32 %56, 13
  %110 = zext i32 %109 to i64
  %111 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %110
  %112 = bitcast ptr addrspace(1) %111 to ptr addrspace(1)
  store float %29, ptr addrspace(1) %112, align 4, !tbaa !21, !alias.scope !25
  %113 = add i32 %56, 14
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %114
  %116 = bitcast ptr addrspace(1) %115 to ptr addrspace(1)
  store float %30, ptr addrspace(1) %116, align 4, !tbaa !21, !alias.scope !25
  %117 = bitcast <4 x half> %48 to <4 x i16>
  %118 = extractelement <4 x i16> %117, i64 0
  %119 = zext i16 %118 to i32
  %120 = add i32 %56, 15
  %121 = zext i32 %120 to i64
  %122 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %121
  store i32 %119, ptr addrspace(1) %122, align 4, !tbaa !21, !alias.scope !25
  %123 = extractelement <4 x i16> %117, i64 1
  %124 = zext i16 %123 to i32
  %125 = add i32 %56, 16
  %126 = zext i32 %125 to i64
  %127 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %126
  store i32 %124, ptr addrspace(1) %127, align 4, !tbaa !21, !alias.scope !25
  %128 = extractelement <4 x i16> %117, i64 2
  %129 = zext i16 %128 to i32
  %130 = add i32 %56, 17
  %131 = zext i32 %130 to i64
  %132 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %131
  store i32 %129, ptr addrspace(1) %132, align 4, !tbaa !21, !alias.scope !25
  %133 = extractelement <4 x i16> %117, i64 3
  %134 = zext i16 %133 to i32
  %135 = add i32 %56, 18
  %136 = zext i32 %135 to i64
  %137 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %136
  store i32 %134, ptr addrspace(1) %137, align 4, !tbaa !21, !alias.scope !25
  %138 = bitcast <3 x half> %55 to <3 x i16>
  %139 = extractelement <3 x i16> %138, i64 0
  %140 = zext i16 %139 to i32
  %141 = add i32 %56, 19
  %142 = zext i32 %141 to i64
  %143 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %142
  store i32 %140, ptr addrspace(1) %143, align 4, !tbaa !21, !alias.scope !25
  %144 = extractelement <3 x i16> %138, i64 1
  %145 = zext i16 %144 to i32
  %146 = add i32 %56, 20
  %147 = zext i32 %146 to i64
  %148 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %147
  store i32 %145, ptr addrspace(1) %148, align 4, !tbaa !21, !alias.scope !25
  %149 = extractelement <3 x i16> %138, i64 2
  %150 = zext i16 %149 to i32
  %151 = add i32 %56, 21
  %152 = zext i32 %151 to i64
  %153 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %152
  store i32 %150, ptr addrspace(1) %153, align 4, !tbaa !21, !alias.scope !25
  %154 = icmp ult i32 %1, 6
  br i1 %154, label %155, label %156

155:                                              ; preds = %2
  store i32 0, ptr addrspace(1) %100, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %104, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %108, align 4, !tbaa !21, !alias.scope !25
  br label %156

156:                                              ; preds = %155, %2
  %157 = icmp ugt i32 %1, 24
  br i1 %157, label %158, label %159

158:                                              ; preds = %156
  store i32 0, ptr addrspace(1) %65, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %69, align 4, !tbaa !21, !alias.scope !25
  br label %159

159:                                              ; preds = %158, %156
  %160 = icmp ugt i32 %1, 26
  br i1 %160, label %161, label %162

161:                                              ; preds = %159
  store i32 0, ptr addrspace(1) %143, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %148, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %153, align 4, !tbaa !21, !alias.scope !25
  br label %162

162:                                              ; preds = %161, %159
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i16 @air.get_simdgroup_size.i16() local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.simd_shuffle_and_fill_up.u.i32(i32, i32, i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.simd_shuffle_and_fill_down.u.i32(i32, i32, i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x i32> @air.simd_shuffle_down.u.v2i32(<2 x i32>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x float> @air.simd_shuffle_and_fill_up.v3f32(<3 x float>, <3 x float>, i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x float> @air.simd_shuffle_and_fill_down.v3f32(<3 x float>, <3 x float>, i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x float> @air.simd_shuffle_up.v3f32(<3 x float>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_shuffle_and_fill_up.f32(float, float, i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_shuffle_and_fill_down.f32(float, float, i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x half> @air.simd_shuffle_and_fill_down.v4f16(<4 x half>, <4 x half>, i16, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x half> @air.simd_shuffle_down.v3f16(<3 x half>, i16) local_unnamed_addr #2

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
!9 = !{ptr @kernel_simd_shuffle_and_fill, !10, !11}
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
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_simd_shuffle_and_fill.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_simd_shuffle_and_fill)"}
