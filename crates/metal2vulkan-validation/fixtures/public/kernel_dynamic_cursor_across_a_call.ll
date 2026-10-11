; ModuleID = 'f.tri.ll'
source_filename = "f.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spirv-unknown-vulkan1.2"

; Function Attrs: convergent mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite)
define void @kernel_dynamic_cursor_across_a_call(ptr addrspace(1) captures(none) "air-buffer-no-alias" %0, i32 %1) local_unnamed_addr #0 {
  %3 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  %4 = load i8, ptr addrspace(1) %3, align 1, !tbaa !21, !alias.scope !24
  %5 = zext i8 %4 to i32
  %6 = getelementptr inbounds <4 x float>, ptr addrspace(1) %0, i64 0, i64 1
  %7 = bitcast ptr addrspace(1) %6 to ptr addrspace(1)
  %8 = load i8, ptr addrspace(1) %7, align 1, !tbaa !21, !alias.scope !24
  %9 = zext i8 %8 to i32
  %10 = zext i8 %4 to i64
  %11 = getelementptr inbounds <4 x float>, ptr addrspace(1) %0, i64 %10
  %12 = add i32 %9, %1
  tail call fastcc void @_ZL8store_atPU9MTLdeviceDv4_fj(ptr addrspace(1) %11, i32 %12) #3, !alias.scope !24
  %13 = tail call fast float @air.convert.f.f32.u.i32(i32 %9) #4
  %14 = tail call fast float @air.convert.f.f32.u.i32(i32 %5) #4
  %15 = insertelement <4 x float> <float poison, float poison, float 0.000000e+00, float 0.000000e+00>, float %13, i64 0
  %16 = insertelement <4 x float> %15, float %14, i64 1
  store <4 x float> %16, ptr addrspace(1) %0, align 16, !tbaa !21, !alias.scope !24
  ret void
}

; Function Attrs: mustprogress nofree noinline nosync nounwind willreturn memory(argmem: write)
define internal fastcc void @_ZL8store_atPU9MTLdeviceDv4_fj(ptr addrspace(1) writeonly captures(none) %0, i32 %1) unnamed_addr #1 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %13, label %4

4:                                                ; preds = %2
  %5 = tail call fast float @air.convert.f.f32.u.i32(i32 %1) #4
  %6 = insertelement <4 x float> undef, float %5, i64 0
  %7 = fadd fast float %5, 1.000000e+00
  %8 = insertelement <4 x float> %6, float %7, i64 1
  %9 = fadd fast float %5, 2.000000e+00
  %10 = insertelement <4 x float> %8, float %9, i64 2
  %11 = fadd fast float %5, 3.000000e+00
  %12 = insertelement <4 x float> %10, float %11, i64 3
  store <4 x float> %12, ptr addrspace(1) %0, align 16, !tbaa !21
  br label %13

13:                                               ; preds = %4, %2
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind willreturn memory(none)
declare float @air.convert.f.f32.u.i32(i32) local_unnamed_addr #2

attributes #0 = { convergent mustprogress nofree nosync nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree noinline nosync nounwind willreturn memory(argmem: write) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #2 = { mustprogress nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nobuiltin "no-builtins" }
attributes #4 = { nounwind willreturn memory(none) }

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
!9 = !{ptr @kernel_dynamic_cursor_across_a_call, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!18 = !{i32 2, i32 8, i32 0}
!19 = !{!"Metal", i32 3, i32 0, i32 0}
!20 = !{!"/private/tmp/dyncurs/f.metal"}
!21 = !{!22, !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C++ TBAA"}
!24 = !{!25}
!25 = distinct !{!25, !26, !"air-alias-scope-arg(0)"}
!26 = distinct !{!26, !"air-alias-scopes(kernel_dynamic_cursor_across_a_call)"}
