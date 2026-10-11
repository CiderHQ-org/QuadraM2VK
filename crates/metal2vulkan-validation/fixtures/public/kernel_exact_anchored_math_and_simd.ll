; ModuleID = '/tmp/mkfix2/m3.bc'
source_filename = "validation/fixtures/public/kernel_exact_anchored_math_and_simd.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_exact_anchored_math_and_simd(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = urem i32 %1, 6
  %4 = shl nuw nsw i32 %3, 1
  %5 = and i32 %4, 14
  %6 = shl nuw nsw i32 1, %5
  %7 = tail call fast float @air.convert.f.f32.u.i32(i32 %6) #3
  %8 = fmul fast float %7, 4.000000e+00
  %9 = insertelement <3 x float> undef, float %8, i64 0
  %10 = fmul fast float %7, 1.600000e+01
  %11 = insertelement <3 x float> %9, float %10, i64 1
  %12 = fmul fast float %7, 2.500000e-01
  %13 = insertelement <3 x float> %11, float %12, i64 2
  %14 = tail call fast <3 x float> @air.rsqrt.v3f32(<3 x float> %13) #3
  %15 = urem i32 %1, 11
  %16 = shl nuw nsw i32 1, %15
  %17 = tail call fast half @air.convert.f.f16.u.i32(i32 %16) #3
  %18 = insertelement <3 x half> <half poison, half 0xH3800, half 0xH6400>, half %17, i64 0
  %19 = tail call fast <3 x half> @air.log2.v3f16(<3 x half> %18) #3
  %20 = and i32 %1, 7
  %21 = tail call fast half @air.convert.f.f16.u.i32(i32 %20) #3
  %22 = tail call fast half @air.cospi.f16(half %21) #3
  %23 = add i32 %1, 1
  %24 = tail call fast float @air.convert.f.f32.u.i32(i32 %23) #3
  %25 = and i32 %1, 3
  %26 = tail call fast float @air.convert.f.f32.u.i32(i32 %25) #3
  %27 = insertelement <4 x float> <float poison, float 1.000000e+00, float poison, float 3.000000e+00>, float %24, i64 0
  %28 = insertelement <4 x float> %27, float %26, i64 2
  %29 = tail call fast <4 x float> @air.simd_sum.v4f32(<4 x float> %28) #4
  %30 = trunc i32 %1 to i16
  %31 = mul i16 %30, 300
  %32 = add i16 %31, -4000
  %33 = tail call i16 @air.simd_shuffle_up.s.i16(i16 %32, i16 3) #4
  %34 = mul i32 %1, 9
  %35 = bitcast <3 x float> %14 to <3 x i32>
  %36 = extractelement <3 x i32> %35, i64 0
  %37 = zext i32 %34 to i64
  %38 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %37
  store i32 %36, ptr addrspace(1) %38, align 4, !tbaa !21, !alias.scope !25
  %39 = extractelement <3 x i32> %35, i64 1
  %40 = add i32 %34, 1
  %41 = zext i32 %40 to i64
  %42 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %41
  store i32 %39, ptr addrspace(1) %42, align 4, !tbaa !21, !alias.scope !25
  %43 = extractelement <3 x i32> %35, i64 2
  %44 = add i32 %34, 2
  %45 = zext i32 %44 to i64
  %46 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %45
  store i32 %43, ptr addrspace(1) %46, align 4, !tbaa !21, !alias.scope !25
  %47 = bitcast <3 x half> %19 to <3 x i16>
  %48 = extractelement <3 x i16> %47, i64 0
  %49 = zext i16 %48 to i32
  %50 = add i32 %34, 3
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %51
  store i32 %49, ptr addrspace(1) %52, align 4, !tbaa !21, !alias.scope !25
  %53 = extractelement <3 x i16> %47, i64 1
  %54 = zext i16 %53 to i32
  %55 = add i32 %34, 4
  %56 = zext i32 %55 to i64
  %57 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %56
  store i32 %54, ptr addrspace(1) %57, align 4, !tbaa !21, !alias.scope !25
  %58 = bitcast half %22 to i16
  %59 = zext i16 %58 to i32
  %60 = add i32 %34, 5
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %61
  store i32 %59, ptr addrspace(1) %62, align 4, !tbaa !21, !alias.scope !25
  %63 = bitcast <4 x float> %29 to <4 x i32>
  %64 = extractelement <4 x i32> %63, i64 0
  %65 = add i32 %34, 6
  %66 = zext i32 %65 to i64
  %67 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %66
  store i32 %64, ptr addrspace(1) %67, align 4, !tbaa !21, !alias.scope !25
  %68 = extractelement <4 x i32> %63, i64 2
  %69 = add i32 %34, 7
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %70
  store i32 %68, ptr addrspace(1) %71, align 4, !tbaa !21, !alias.scope !25
  %72 = sext i16 %33 to i32
  %73 = icmp ult i32 %1, 3
  %74 = select i1 %73, i32 0, i32 %72
  %75 = add i32 %34, 8
  %76 = zext i32 %75 to i64
  %77 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %76
  store i32 %74, ptr addrspace(1) %77, align 4, !tbaa !21, !alias.scope !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.convert.f.f16.u.i32(i32) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x float> @air.rsqrt.v3f32(<3 x float>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare <3 x half> @air.log2.v3f16(<3 x half>) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare half @air.cospi.f16(half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x float> @air.simd_sum.v4f32(<4 x float>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare i16 @air.simd_shuffle_up.s.i16(i16, i16) local_unnamed_addr #2

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
!9 = !{ptr @kernel_exact_anchored_math_and_simd, !10, !11}
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
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_exact_anchored_math_and_simd.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_exact_anchored_math_and_simd)"}
