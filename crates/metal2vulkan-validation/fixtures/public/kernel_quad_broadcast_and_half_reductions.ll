; ModuleID = '/tmp/mkfix2/q2.bc'
source_filename = "validation/fixtures/public/kernel_quad_broadcast_and_half_reductions.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_quad_broadcast_and_half_reductions(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add i32 %1, 1
  %4 = tail call fast half @air.convert.f.f16.u.i32(i32 %3) #3
  %5 = insertelement <4 x half> undef, half %4, i64 0
  %6 = fadd fast half %4, 0xH3C00
  %7 = insertelement <4 x half> %5, half %6, i64 1
  %8 = fmul fast half %4, 0xH4000
  %9 = insertelement <4 x half> %7, half %8, i64 2
  %10 = fadd fast half %4, 0xH5000
  %11 = insertelement <4 x half> %9, half %10, i64 3
  %12 = tail call fast float @air.convert.f.f32.u.i32(i32 %3) #3
  %13 = insertelement <4 x float> undef, float %12, i64 0
  %14 = fmul fast float %12, 2.000000e+00
  %15 = insertelement <4 x float> %13, float %14, i64 1
  %16 = fmul fast float %12, 4.000000e+00
  %17 = insertelement <4 x float> %15, float %16, i64 2
  %18 = fmul fast float %12, 8.000000e+00
  %19 = insertelement <4 x float> %17, float %18, i64 3
  %20 = insertelement <4 x i32> poison, i32 %3, i64 0
  %21 = mul i32 %3, 3
  %22 = insertelement <4 x i32> %20, i32 %21, i64 1
  %23 = mul i32 %3, 5
  %24 = insertelement <4 x i32> %22, i32 %23, i64 2
  %25 = mul i32 %3, 7
  %26 = insertelement <4 x i32> %24, i32 %25, i64 3
  %27 = freeze <4 x half> %11
  %28 = tail call fast <4 x half> @air.quad_shuffle_xor.v4f16(<4 x half> %27, i16 1) #4
  %29 = freeze <4 x float> %19
  %30 = tail call fast <4 x float> @air.quad_broadcast.v4f32(<4 x float> %29, i16 2) #4
  %31 = tail call fast <4 x half> @air.simd_sum.v4f16(<4 x half> %27) #4
  %32 = tail call fast <4 x half> @air.quad_sum.v4f16(<4 x half> %27) #4
  %33 = tail call <4 x i32> @air.quad_shuffle_down.u.v4i32(<4 x i32> %26, i16 1) #4
  %34 = tail call i1 @air.quad_is_first() #4
  %35 = mul i32 %1, 18
  %36 = bitcast <4 x half> %28 to <4 x i16>
  %37 = extractelement <4 x i16> %36, i64 0
  %38 = zext i16 %37 to i32
  %39 = zext i32 %35 to i64
  %40 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %39
  store i32 %38, ptr addrspace(1) %40, align 4, !tbaa !21, !alias.scope !25
  %41 = extractelement <4 x i16> %36, i64 1
  %42 = zext i16 %41 to i32
  %43 = or i32 %35, 1
  %44 = zext i32 %43 to i64
  %45 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %44
  store i32 %42, ptr addrspace(1) %45, align 4, !tbaa !21, !alias.scope !25
  %46 = extractelement <4 x i16> %36, i64 2
  %47 = zext i16 %46 to i32
  %48 = add i32 %35, 2
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %49
  store i32 %47, ptr addrspace(1) %50, align 4, !tbaa !21, !alias.scope !25
  %51 = extractelement <4 x i16> %36, i64 3
  %52 = zext i16 %51 to i32
  %53 = add i32 %35, 3
  %54 = zext i32 %53 to i64
  %55 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %54
  store i32 %52, ptr addrspace(1) %55, align 4, !tbaa !21, !alias.scope !25
  %56 = bitcast <4 x float> %30 to <4 x i32>
  %57 = extractelement <4 x i32> %56, i64 0
  %58 = add i32 %35, 4
  %59 = zext i32 %58 to i64
  %60 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %59
  store i32 %57, ptr addrspace(1) %60, align 4, !tbaa !21, !alias.scope !25
  %61 = extractelement <4 x i32> %56, i64 1
  %62 = add i32 %35, 5
  %63 = zext i32 %62 to i64
  %64 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %63
  store i32 %61, ptr addrspace(1) %64, align 4, !tbaa !21, !alias.scope !25
  %65 = extractelement <4 x i32> %56, i64 2
  %66 = add i32 %35, 6
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %67
  store i32 %65, ptr addrspace(1) %68, align 4, !tbaa !21, !alias.scope !25
  %69 = extractelement <4 x i32> %56, i64 3
  %70 = add i32 %35, 7
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %71
  store i32 %69, ptr addrspace(1) %72, align 4, !tbaa !21, !alias.scope !25
  %73 = bitcast <4 x half> %31 to <4 x i16>
  %74 = extractelement <4 x i16> %73, i64 0
  %75 = zext i16 %74 to i32
  %76 = add i32 %35, 8
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %77
  store i32 %75, ptr addrspace(1) %78, align 4, !tbaa !21, !alias.scope !25
  %79 = extractelement <4 x i16> %73, i64 1
  %80 = zext i16 %79 to i32
  %81 = add i32 %35, 9
  %82 = zext i32 %81 to i64
  %83 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %82
  store i32 %80, ptr addrspace(1) %83, align 4, !tbaa !21, !alias.scope !25
  %84 = extractelement <4 x i16> %73, i64 2
  %85 = zext i16 %84 to i32
  %86 = add i32 %35, 10
  %87 = zext i32 %86 to i64
  %88 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %87
  store i32 %85, ptr addrspace(1) %88, align 4, !tbaa !21, !alias.scope !25
  %89 = extractelement <4 x i16> %73, i64 3
  %90 = zext i16 %89 to i32
  %91 = add i32 %35, 11
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %92
  store i32 %90, ptr addrspace(1) %93, align 4, !tbaa !21, !alias.scope !25
  %94 = bitcast <4 x half> %32 to <4 x i16>
  %95 = extractelement <4 x i16> %94, i64 0
  %96 = zext i16 %95 to i32
  %97 = add i32 %35, 12
  %98 = zext i32 %97 to i64
  %99 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %98
  store i32 %96, ptr addrspace(1) %99, align 4, !tbaa !21, !alias.scope !25
  %100 = extractelement <4 x i16> %94, i64 3
  %101 = zext i16 %100 to i32
  %102 = add i32 %35, 13
  %103 = zext i32 %102 to i64
  %104 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %103
  store i32 %101, ptr addrspace(1) %104, align 4, !tbaa !21, !alias.scope !25
  %105 = extractelement <4 x i32> %33, i64 0
  %106 = add i32 %35, 14
  %107 = zext i32 %106 to i64
  %108 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %107
  store i32 %105, ptr addrspace(1) %108, align 4, !tbaa !21, !alias.scope !25
  %109 = extractelement <4 x i32> %33, i64 1
  %110 = add i32 %35, 15
  %111 = zext i32 %110 to i64
  %112 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %111
  store i32 %109, ptr addrspace(1) %112, align 4, !tbaa !21, !alias.scope !25
  %113 = extractelement <4 x i32> %33, i64 3
  %114 = add i32 %35, 16
  %115 = zext i32 %114 to i64
  %116 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %115
  store i32 %113, ptr addrspace(1) %116, align 4, !tbaa !21, !alias.scope !25
  %117 = zext i1 %34 to i32
  %118 = add i32 %35, 17
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %119
  store i32 %117, ptr addrspace(1) %120, align 4, !tbaa !21, !alias.scope !25
  %121 = and i32 %1, 3
  %122 = icmp eq i32 %121, 3
  br i1 %122, label %123, label %124

123:                                              ; preds = %2
  store i32 0, ptr addrspace(1) %108, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %112, align 4, !tbaa !21, !alias.scope !25
  store i32 0, ptr addrspace(1) %116, align 4, !tbaa !21, !alias.scope !25
  br label %124

124:                                              ; preds = %123, %2
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare i1 @air.quad_is_first() local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x half> @air.quad_shuffle_xor.v4f16(<4 x half>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x float> @air.quad_broadcast.v4f32(<4 x float>, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x half> @air.simd_sum.v4f16(<4 x half>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x half> @air.quad_sum.v4f16(<4 x half>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x i32> @air.quad_shuffle_down.u.v4i32(<4 x i32>, i16) local_unnamed_addr #2

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
!9 = !{ptr @kernel_quad_broadcast_and_half_reductions, !10, !11}
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
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_quad_broadcast_and_half_reductions.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_quad_broadcast_and_half_reductions)"}
