; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'q1.bc'
source_filename = "validation/fixtures/public/kernel_quad_ops_linear.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @kernel_quad_ops_linear(ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = tail call fast float @air.convert.f.f32.u.i32(i32 %1) #3
  %4 = insertelement <4 x float> undef, float %3, i64 0
  %5 = fmul fast float %3, 2.000000e+00
  %6 = insertelement <4 x float> %4, float %5, i64 1
  %7 = fmul fast float %3, 4.000000e+00
  %8 = insertelement <4 x float> %6, float %7, i64 2
  %9 = fmul fast float %3, 8.000000e+00
  %10 = insertelement <4 x float> %8, float %9, i64 3
  %11 = tail call fast <4 x float> @air.quad_sum.v4f32(<4 x float> %10) #4
  %12 = mul nsw i32 %1, 3
  %13 = add nsw i32 %12, -40
  %14 = tail call i32 @air.quad_max.s.i32(i32 %13) #4
  %15 = tail call i32 @air.quad_min.s.i32(i32 %13) #4
  %16 = freeze float %3
  %17 = tail call fast float @air.quad_broadcast.f32(float %16, i16 2) #4
  %18 = tail call fast float @air.quad_shuffle_xor.f32(float %16, i16 3) #4
  %19 = shl i32 %1, 3
  %20 = bitcast <4 x float> %11 to <4 x i32>
  %21 = extractelement <4 x i32> %20, i64 0
  %22 = zext i32 %19 to i64
  %23 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %22
  store i32 %21, ptr addrspace(1) %23, align 4, !tbaa !21, !alias.scope !25
  %24 = extractelement <4 x i32> %20, i64 1
  %25 = or i32 %19, 1
  %26 = zext i32 %25 to i64
  %27 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %26
  store i32 %24, ptr addrspace(1) %27, align 4, !tbaa !21, !alias.scope !25
  %28 = extractelement <4 x i32> %20, i64 2
  %29 = or i32 %19, 2
  %30 = zext i32 %29 to i64
  %31 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %30
  store i32 %28, ptr addrspace(1) %31, align 4, !tbaa !21, !alias.scope !25
  %32 = extractelement <4 x i32> %20, i64 3
  %33 = or i32 %19, 3
  %34 = zext i32 %33 to i64
  %35 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %34
  store i32 %32, ptr addrspace(1) %35, align 4, !tbaa !21, !alias.scope !25
  %36 = or i32 %19, 4
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %37
  store i32 %14, ptr addrspace(1) %38, align 4, !tbaa !21, !alias.scope !25
  %39 = or i32 %19, 5
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %40
  store i32 %15, ptr addrspace(1) %41, align 4, !tbaa !21, !alias.scope !25
  %42 = or i32 %19, 6
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %43
  %45 = bitcast ptr addrspace(1) %44 to ptr addrspace(1)
  store float %17, ptr addrspace(1) %45, align 4, !tbaa !21, !alias.scope !25
  %46 = or i32 %19, 7
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %47
  %49 = bitcast ptr addrspace(1) %48 to ptr addrspace(1)
  store float %18, ptr addrspace(1) %49, align 4, !tbaa !21, !alias.scope !25
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nounwind willreturn
declare <4 x float> @air.quad_sum.v4f32(<4 x float>) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.quad_max.s.i32(i32) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare i32 @air.quad_min.s.i32(i32) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.quad_broadcast.f32(float, i16) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.quad_shuffle_xor.f32(float, i16) local_unnamed_addr #2

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
!9 = !{ptr @kernel_quad_ops_linear, !10, !11}
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
!20 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_quad_ops_linear.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"int", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !{!26}
!26 = distinct !{!26, !27, !"air-alias-scope-arg(0)"}
!27 = distinct !{!27, !"air-alias-scopes(kernel_quad_ops_linear)"}
