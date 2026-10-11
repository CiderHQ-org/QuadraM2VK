; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'bicwrap.bc'
source_filename = "validation/fixtures/public/kernel_bicubic_repeat_wrap.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601023122, i64 0], align 8
@_ZL2kU = internal unnamed_addr addrspace(2) constant [8 x float] [float 6.250000e-02, float 1.875000e-01, float 3.125000e-01, float 4.375000e-01, float 3.062500e+00, float 3.187500e+00, float 3.312500e+00, float 3.437500e+00], align 4

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: readwrite)
define void @kernel_bicubic_repeat_wrap(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = zext i32 %2 to i64
  %5 = getelementptr inbounds [8 x float], ptr addrspace(2) @_ZL2kU, i64 0, i64 %4
  %6 = load float, ptr addrspace(2) %5, align 4, !tbaa !23
  %7 = insertelement <2 x float> <float poison, float 4.375000e-01>, float %6, i64 0
  %8 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <2 x float> %7, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.000000e+00, float 0.000000e+00, i32 0) #2
  %9 = extractvalue { <4 x float>, i8 } %8, 0
  %10 = getelementptr inbounds <4 x float>, ptr addrspace(1) %1, i64 %4
  store <4 x float> %9, ptr addrspace(1) %10, align 16, !tbaa !27, !alias.scope !28, !noalias !31
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <2 x float>, i1, <2 x i32>, i1, float, float, i32) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree nounwind willreturn memory(argmem: readwrite) "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #2 = { convergent nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!15, !16, !17}
!air.sampler_states = !{!18}
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
!9 = !{ptr @kernel_bicubic_repeat_wrap, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!19 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"/Users/aneesiqbal/Projects/steelbrain/metal2vulkan/validation/fixtures/public/kernel_bicubic_repeat_wrap.metal"}
!23 = !{!24, !24, i64 0}
!24 = !{!"float", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C++ TBAA"}
!27 = !{!25, !25, i64 0}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(1)"}
!30 = distinct !{!30, !"air-alias-scopes(kernel_bicubic_repeat_wrap)"}
!31 = !{!32}
!32 = distinct !{!32, !30, !"air-alias-scope-textures"}
