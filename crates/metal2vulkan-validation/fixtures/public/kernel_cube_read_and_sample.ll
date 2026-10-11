; Owned synthetic fixture: compiled from the sibling .metal with `xcrun -sdk macosx metal -S
; -emit-llvm` and round-tripped through llvm-as/llvm-dis for opaque pointers.
; Not derived from a third-party metallib.
; ModuleID = 'kernel_cube_read_and_sample.bc'
source_filename = "validation/fixtures/public/kernel_cube_read_and_sample.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601017929, i64 0], align 8

; Function Attrs: convergent mustprogress nofree nounwind willreturn
define void @kernel_cube_read_and_sample(ptr addrspace(1) %0, ptr addrspace(1) noundef writeonly captures(none) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = tail call ptr addrspace(2) @air.get_read_sampler() #4
  %4 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %3, <2 x i32> zeroinitializer, i32 0, i32 0, i32 0) #5
  %5 = extractvalue { <4 x float>, i8 } %4, 0
  %6 = extractelement <4 x float> %5, i64 0
  store float %6, ptr addrspace(1) %1, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %7 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %3, <2 x i32> <i32 1, i32 0>, i32 1, i32 0, i32 0) #5
  %8 = extractvalue { <4 x float>, i8 } %7, 0
  %9 = extractelement <4 x float> %8, i64 0
  %10 = getelementptr inbounds float, ptr addrspace(1) %1, i64 1
  store float %9, ptr addrspace(1) %10, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %11 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %3, <2 x i32> <i32 0, i32 1>, i32 2, i32 0, i32 0) #5
  %12 = extractvalue { <4 x float>, i8 } %11, 0
  %13 = extractelement <4 x float> %12, i64 0
  %14 = getelementptr inbounds float, ptr addrspace(1) %1, i64 2
  store float %13, ptr addrspace(1) %14, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %15 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %3, <2 x i32> splat (i32 1), i32 3, i32 0, i32 0) #5
  %16 = extractvalue { <4 x float>, i8 } %15, 0
  %17 = extractelement <4 x float> %16, i64 0
  %18 = getelementptr inbounds float, ptr addrspace(1) %1, i64 3
  store float %17, ptr addrspace(1) %18, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %19 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %3, <2 x i32> zeroinitializer, i32 4, i32 0, i32 0) #5
  %20 = extractvalue { <4 x float>, i8 } %19, 0
  %21 = extractelement <4 x float> %20, i64 0
  %22 = getelementptr inbounds float, ptr addrspace(1) %1, i64 4
  store float %21, ptr addrspace(1) %22, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %23 = tail call { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) %3, <2 x i32> splat (i32 1), i32 5, i32 0, i32 0) #5
  %24 = extractvalue { <4 x float>, i8 } %23, 0
  %25 = extractelement <4 x float> %24, i64 0
  %26 = getelementptr inbounds float, ptr addrspace(1) %1, i64 5
  store float %25, ptr addrspace(1) %26, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %27 = tail call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #6
  %28 = extractvalue { <4 x float>, i8 } %27, 0
  %29 = extractelement <4 x float> %28, i64 0
  %30 = getelementptr inbounds float, ptr addrspace(1) %1, i64 6
  store float %29, ptr addrspace(1) %30, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  %31 = tail call { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) readonly captures(none) %0, ptr addrspace(2) readonly captures(none) @__air_sampler_state, <3 x float> <float 0.000000e+00, float 0.000000e+00, float -1.000000e+00>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0) #6
  %32 = extractvalue { <4 x float>, i8 } %31, 0
  %33 = extractelement <4 x float> %32, i64 0
  %34 = getelementptr inbounds float, ptr addrspace(1) %1, i64 7
  store float %33, ptr addrspace(1) %34, align 4, !tbaa !22, !alias.scope !26, !noalias !29
  ret void
}

; Function Attrs: mustprogress nofree nounwind willreturn memory(inaccessiblemem: read)
declare ptr addrspace(2) @air.get_read_sampler() local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.read_texture_cube.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2), <2 x i32>, i32, i32, i32) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(argmem: read)
declare { <4 x float>, i8 } @air.sample_texture_cube.v4f32(ptr addrspace(1) readonly captures(none), ptr addrspace(2) readonly captures(none), <3 x float>, i1, float, float, i32) local_unnamed_addr #3

attributes #0 = { convergent mustprogress nofree nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="128" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nounwind willreturn memory(inaccessiblemem: read) }
attributes #2 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #3 = { convergent mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { nounwind willreturn memory(inaccessiblemem: read) }
attributes #5 = { nounwind willreturn memory(argmem: read) }
attributes #6 = { convergent nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!14, !15, !16}
!air.sampler_states = !{!17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @kernel_cube_read_and_sample, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!12 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>", !"air.arg_name", !"tex"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!18 = !{!"Apple metal version 32023.883 (metalfe-32023.883)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"/private/tmp/cb/k.metal"}
!22 = !{!23, !23, i64 0}
!23 = !{!"float", !24, i64 0}
!24 = !{!"omnipotent char", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !{!27}
!27 = distinct !{!27, !28, !"air-alias-scope-arg(1)"}
!28 = distinct !{!28, !"air-alias-scopes(kernel_cube_read_and_sample)"}
!29 = !{!30}
!30 = distinct !{!30, !28, !"air-alias-scope-textures"}
