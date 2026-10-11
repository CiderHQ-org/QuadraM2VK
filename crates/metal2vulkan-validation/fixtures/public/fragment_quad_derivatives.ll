; ModuleID = '/tmp/da/fq.bc'
source_filename = "validation/fixtures/public/fragment_quad_derivatives.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define <4 x float> @fragment_quad_derivatives(<4 x float> noundef %0) local_unnamed_addr #0 {
  %2 = extractelement <4 x float> %0, i64 0
  %3 = fadd fast float %2, -5.000000e-01
  %4 = extractelement <4 x float> %0, i64 1
  %5 = fadd fast float %4, -5.000000e-01
  %6 = insertelement <3 x float> undef, float %3, i64 0
  %7 = fmul fast float %3, 2.000000e+00
  %8 = fmul fast float %5, 3.000000e+00
  %9 = fadd fast float %7, %8
  %10 = insertelement <3 x float> %6, float %9, i64 1
  %11 = fmul fast float %5, 7.000000e+00
  %12 = insertelement <3 x float> %10, float %11, i64 2
  %13 = tail call fast <3 x float> @air.fwidth.v3f32(<3 x float> %12) #3
  %14 = fmul fast float %3, 4.000000e+00
  %15 = insertelement <4 x float> undef, float %14, i64 0
  %16 = fmul fast float %5, 9.000000e+00
  %17 = insertelement <4 x float> %15, float %16, i64 1
  %18 = fadd fast float %3, %5
  %19 = insertelement <4 x float> %17, float %18, i64 2
  %20 = fmul fast float %3, 1.100000e+01
  %21 = fmul fast float %5, 2.000000e+00
  %22 = fsub fast float %20, %21
  %23 = insertelement <4 x float> %19, float %22, i64 3
  %24 = tail call fast <4 x float> @air.fwidth.v4f32(<4 x float> %23) #3
  %25 = fmul fast float %3, 6.000000e+00
  %26 = fptrunc float %25 to half
  %27 = insertelement <2 x half> undef, half %26, i64 0
  %28 = fmul fast float %3, 8.000000e+00
  %29 = fadd fast float %28, %21
  %30 = fptrunc float %29 to half
  %31 = insertelement <2 x half> %27, half %30, i64 1
  %32 = tail call fast <2 x half> @air.fwidth.v2f16(<2 x half> %31) #3
  %33 = fmul fast float %3, 3.000000e+00
  %34 = fptrunc float %33 to half
  %35 = insertelement <3 x half> undef, half %34, i64 0
  %36 = fmul fast float %5, 1.400000e+01
  %37 = fptrunc float %36 to half
  %38 = insertelement <3 x half> %35, half %37, i64 1
  %39 = fmul fast float %3, 1.500000e+01
  %40 = fmul fast float %5, 1.600000e+01
  %41 = fsub fast float %39, %40
  %42 = fptrunc float %41 to half
  %43 = insertelement <3 x half> %38, half %42, i64 2
  %44 = tail call fast <3 x half> @air.dfdx.v3f16(<3 x half> %43) #3
  %45 = tail call fast <3 x half> @air.dfdy.v3f16(<3 x half> %43) #3
  %46 = tail call i32 @air.convert.u.i32.f.f32(float %3) #4
  %47 = icmp ult i32 %46, 2
  br i1 %47, label %48, label %51

48:                                               ; preds = %1
  %49 = shufflevector <3 x float> %13, <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %50 = shufflevector <4 x float> %49, <4 x float> %24, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  br label %89

51:                                               ; preds = %1
  %52 = lshr i32 %46, 1
  %53 = extractelement <3 x half> %45, i64 0
  %54 = fpext half %53 to float
  %55 = insertelement <4 x float> undef, float %54, i64 0
  %56 = extractelement <3 x half> %45, i64 1
  %57 = fpext half %56 to float
  %58 = insertelement <4 x float> %55, float %57, i64 1
  %59 = extractelement <3 x half> %45, i64 2
  %60 = fpext half %59 to float
  %61 = insertelement <4 x float> %58, float %60, i64 2
  %62 = extractelement <3 x half> %44, i64 0
  %63 = fpext half %62 to float
  %64 = fadd fast float %57, %63
  %65 = insertelement <4 x float> %61, float %64, i64 3
  %66 = extractelement <2 x half> %32, i64 1
  %67 = fpext half %66 to float
  %68 = insertelement <4 x float> undef, float %67, i64 0
  %69 = insertelement <4 x float> %68, float %63, i64 1
  %70 = extractelement <3 x half> %44, i64 1
  %71 = fpext half %70 to float
  %72 = insertelement <4 x float> %69, float %71, i64 2
  %73 = extractelement <3 x half> %44, i64 2
  %74 = fpext half %73 to float
  %75 = insertelement <4 x float> %72, float %74, i64 3
  %76 = extractelement <4 x float> %24, i64 1
  %77 = insertelement <4 x float> undef, float %76, i64 0
  %78 = extractelement <4 x float> %24, i64 2
  %79 = insertelement <4 x float> %77, float %78, i64 1
  %80 = extractelement <4 x float> %24, i64 3
  %81 = insertelement <4 x float> %79, float %80, i64 2
  %82 = extractelement <2 x half> %32, i64 0
  %83 = fpext half %82 to float
  %84 = insertelement <4 x float> %81, float %83, i64 3
  %85 = icmp eq i32 %52, 1
  %86 = icmp eq i32 %52, 2
  %87 = select fast i1 %86, <4 x float> %75, <4 x float> %65
  %88 = select i1 %85, <4 x float> %84, <4 x float> %87
  br label %89

89:                                               ; preds = %51, %48
  %90 = phi fast <4 x float> [ %50, %48 ], [ %88, %51 ]
  ret <4 x float> %90
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare i32 @air.convert.u.i32.f.f32(float) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x float> @air.fwidth.v3f32(<3 x float>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x float> @air.fwidth.v4f32(<4 x float>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <2 x half> @air.fwidth.v2f16(<2 x half>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x half> @air.dfdx.v3f16(<3 x half>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare <3 x half> @air.dfdy.v3f16(<3 x half>) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #2 = { convergent mustprogress nounwind willreturn }
attributes #3 = { convergent nounwind willreturn }
attributes #4 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.fragment = !{!9}
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
!9 = !{ptr @fragment_quad_derivatives, !10, !12}
!10 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!12 = !{!13}
!13 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/fragment_quad_derivatives.metal"}
